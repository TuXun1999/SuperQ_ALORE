# Copyright (c) 2023 Boston Dynamics, Inc.  All rights reserved.
#
# Downloading, reproducing, distributing or otherwise using the SDK Software
# is subject to the terms and conditions of the Boston Dynamics Software
# Development Kit License (20191101-BDSDK-SL).

import time
from threading import Condition, Lock, Thread

from constants import DOF

from bosdyn.api import robot_command_pb2
from bosdyn.client.robot_command import RobotCommandBuilder
from bosdyn.util import seconds_to_duration, set_timestamp_from_now


class LinearInterpolator:
    # LinearInterpolator is a helper class to linearly interpolate between
    # two joint positions over a fixed duration.
    def __init__(self, duration, init_pos, target_pos):
        self.duration = duration
        self.init_pos = init_pos
        self.target_pos = target_pos

    def calculate_command(self, time):
        # The final streaming update can arrive just after the requested plan
        # duration.  Clamp it so timing jitter cannot overshoot the target.
        time = min(max(time, 0.0), self.duration)
        pos = self.init_pos + (self.target_pos - self.init_pos) * time / self.duration
        vel = (self.target_pos - self.init_pos) / self.duration
        return pos, vel


class JointAPIInterface:
    # JointAPIInterface is a helper class to help to use joint control API.
    # This class contains the way of activating joint control mode, getting lightweight
    # robot state, and streaming either interpolated or direct position commands.
    def __init__(self, robot, n_dofs):
        self.robot = robot
        self.latest_state_stream_data = None
        # Timestamp the arrival of each streamed state so consumers can measure
        # how old the joint sample is when it is used.
        self.latest_state_stream_receive_time = None
        self._state_lock = Lock()
        self.should_stop = False
        self.started_streaming = False
        self.cmd_history = {}
        self._next_command_key = 1

        # ReLIC publishes a new target at 100 Hz, while Spot expects all of
        # those updates to belong to one long-lived streaming RPC.  Keep the
        # most recent target behind a condition variable so that the policy
        # loop never opens a new RPC or blocks on network latency.
        self._command_target_condition = Condition()
        self._latest_pd_target = None
        self._latest_pd_gains = None
        self._target_generation = 0
        self._command_stream_thread = None
        self._command_stream_error = None

        if n_dofs != DOF.N_DOF_LEGS and n_dofs != DOF.N_DOF:
            # DOF error
            self.robot.logger.warning("Incorrect number of DOF. Joint API will not be activated")
            self.should_stop = True
        self.n_dofs = n_dofs

    # should_stop is a flag to enable/disable sending/receiving command
    def set_should_stop(self, flag):
        with self._command_target_condition:
            self.should_stop = flag
            # Wake a command generator that is waiting for its first or next
            # ReLIC target, so the gRPC request iterator can close promptly.
            self._command_target_condition.notify_all()

    def start_joint_pure_pd_command_stream(self, command_streaming_client):
        """Start one persistent Spot joint-command RPC for ReLIC targets.

        The stream does not emit a command until ``publish_joint_pure_pd_target``
        supplies the first policy result.  This retains the old command-watchdog
        behavior: if ReLIC stops publishing, no stale pose is replayed and the
        final request expires after its 100 ms ``end_time``.
        """
        with self._command_target_condition:
            if self.should_stop:
                raise RuntimeError("Cannot start a joint command stream after it has been stopped.")
            if self._command_stream_thread is not None and self._command_stream_thread.is_alive():
                return
            if self._command_stream_error is not None:
                raise RuntimeError("The previous joint command stream failed.") from self._command_stream_error

            self._command_stream_thread = Thread(
                target=self._run_joint_pure_pd_command_stream,
                args=(command_streaming_client,),
                name="relic_joint_command_stream",
            )
            self._command_stream_thread.start()

    def join_joint_pure_pd_command_stream(self):
        """Wait for the persistent command RPC to finish after ``set_should_stop``."""
        command_stream_thread = self._command_stream_thread
        if command_stream_thread is not None:
            command_stream_thread.join()

    def _run_joint_pure_pd_command_stream(self, command_streaming_client):
        """Own the one blocking gRPC call used throughout a ReLIC rollout."""
        try:
            command_streaming_client.send_joint_control_commands(
                self.generate_joint_pure_pd_command_stream()
            )
            with self._command_target_condition:
                stopped = self.should_stop
            if not stopped:
                raise RuntimeError("Spot joint command stream ended unexpectedly.")
        except Exception as error:
            with self._command_target_condition:
                self._command_stream_error = error
                self._command_target_condition.notify_all()
            self.robot.logger.error(f"ReLIC joint command stream failed: {error}")

    def publish_joint_pure_pd_target(self, target_pose, k_q_p, k_qd_p):
        """Publish the next direct position-PD target to the persistent stream.

        Publishing is intentionally non-blocking with respect to the Spot RPC.
        If the RPC is temporarily slower than the 100 Hz ReLIC loop, the stream
        receives the latest safe target rather than an old queue of commands.
        """
        if len(target_pose) != self.n_dofs:
            raise ValueError(
                f"Expected {self.n_dofs} joint targets, got {len(target_pose)}."
            )
        if len(k_q_p) != self.n_dofs or len(k_qd_p) != self.n_dofs:
            raise ValueError(
                "Position and velocity gain vectors must match the controlled DOF count."
            )

        with self._command_target_condition:
            if self.should_stop:
                raise RuntimeError("Cannot publish a target after joint control has been stopped.")
            if self._command_stream_error is not None:
                raise RuntimeError("The ReLIC joint command stream failed.") from self._command_stream_error
            if self._command_stream_thread is None:
                raise RuntimeError("Joint command stream has not been started.")

            self._latest_pd_target = [float(value) for value in target_pose]
            self._latest_pd_gains = (
                [float(value) for value in k_q_p],
                [float(value) for value in k_qd_p],
            )
            self._target_generation += 1
            self._command_target_condition.notify_all()

    def _create_joint_pure_pd_request(self, target_pose, k_q_p, k_qd_p, include_gains):
        """Create one absolute-position PD request in Boston Dynamics joint order."""
        update_proto = robot_command_pb2.JointControlStreamRequest()
        set_timestamp_from_now(update_proto.header.request_timestamp)
        update_proto.header.client_name = "relic_pd_controller"

        if include_gains:
            update_proto.joint_command.gains.k_q_p.extend(k_q_p)
            update_proto.joint_command.gains.k_qd_p.extend(k_qd_p)

        update_proto.joint_command.position.extend(target_pose)
        update_proto.joint_command.velocity.extend([0.0] * self.n_dofs)
        update_proto.joint_command.load.extend([0.0] * self.n_dofs)

        # Match spot_rl_inference's 100 ms command validity and 5 ms
        # extrapolation allowance.  New ReLIC targets arrive at 100 Hz, but a
        # stopped policy must not make the stream replay a stale target.
        update_proto.joint_command.end_time.CopyFrom(
            self.robot.time_sync.robot_timestamp_from_local_secs(time.time() + 0.1)
        )
        update_proto.joint_command.extrapolation_duration.CopyFrom(seconds_to_duration(0.005))

        command_key = self._next_command_key
        self._next_command_key += 1
        update_proto.joint_command.user_command_key = command_key
        self.cmd_history[command_key] = time.time()
        return update_proto

    def generate_joint_pure_pd_command_stream(self):
        """Yield each newly-published ReLIC target through one persistent RPC.

        Gains appear only in the first request, as in Spot's reference joint
        streaming examples.  Waiting for a new generation instead of repeatedly
        yielding the last target preserves the 100 ms command watchdog.
        """
        last_generation = 0
        gains_sent = False
        while True:
            with self._command_target_condition:
                while self._target_generation == last_generation and not self.should_stop:
                    self._command_target_condition.wait()

                if self.should_stop:
                    return

                target_pose = self._latest_pd_target.copy()
                k_q_p, k_qd_p = self._latest_pd_gains
                last_generation = self._target_generation

            update_proto = self._create_joint_pure_pd_request(
                target_pose=target_pose,
                k_q_p=k_q_p,
                k_qd_p=k_qd_p,
                include_gains=not gains_sent,
            )
            gains_sent = True
            # The activation helper waits for this transition before it asks
            # Spot to enter full-body joint-control mode.
            self.started_streaming = True
            yield update_proto

    # activate function is to activate joint control mode
    def activate(self, command_client):
        # Wait for streaming to start
        while not self.started_streaming:
            time.sleep(0.001)

            if self.should_stop:
                return

        # Activate joint control
        self.robot.logger.info("Activating joint control")
        joint_command = RobotCommandBuilder.joint_command()

        try:
            command_client.robot_command(joint_command)
        except Exception:
            # Signal everything else to stop too
            self.set_should_stop(True)

    # handle_state_streaming function is to get lightweight robot state streaming message.
    def handle_state_streaming(self, robot_state_streaming_client):
        for state in robot_state_streaming_client.get_robot_state_stream():
            receive_time = time.time()

            with self._state_lock:
                self.latest_state_stream_data = state
                self.latest_state_stream_receive_time = time.perf_counter()

            if self.should_stop:
                return

            if (state.last_command.user_command_key != 0 and
                    state.last_command.user_command_key in self.cmd_history.keys()):
                sent_time = self.cmd_history[state.last_command.user_command_key]
                received_time = self.robot.time_sync.get_robot_time_converter(
                ).local_seconds_from_robot_timestamp(state.last_command.received_timestamp)
                latency_ms = (received_time - sent_time) * 1000
                roundtrip_ms = (receive_time - sent_time) * 1000
                # Note that the latency measurement here has the receive time converted from
                # robot time, so it's not very accurate
                self.robot.logger.info(f"Latency {latency_ms:.3f}\tRoundtrip {roundtrip_ms:.3f}")
            else:
                self.robot.logger.info(f"No key: {state.last_command.user_command_key}")

    def get_latest_state_with_receive_time(self):
        """Return one consistent streamed-state snapshot and its local arrival time."""
        while True:
            with self._state_lock:
                state = self.latest_state_stream_data
                receive_time = self.latest_state_stream_receive_time

            if state is not None:
                return state, receive_time

            time.sleep(0.001)

    # get_latest_joints_state function is to get a latest joint state
    def get_latest_pos_vel_and_load_state(self):
        """Return position, velocity, and load from one streamed-state snapshot."""
        state, _ = self.get_latest_state_with_receive_time()
        kin_pos_state = state.joint_states.position
        kin_vel_state = state.joint_states.velocity
        kin_load_state = state.joint_states.load
        return kin_pos_state, kin_vel_state, kin_load_state

    # get_latest_kinematic_state function is to get a latest kinematic state
    def get_latest_kinematic_state(self):
        """Return kinematics from one streamed-state snapshot."""
        state, _ = self.get_latest_state_with_receive_time()
        kin_state = state.kinematic_state
        return kin_state
    
    # generate_joint_pos_interp_commands functions is an example function to send the joint
    # command by interpolating commanded poses
    # cmd_poses : series of commanded joints poses
    # cmd_loads : commanded joint loads
    # duration : desired duration between the poses [s]
    # k_q_p : proportional gain for position control
    # k_qd_p : proportional gain for velocity control
    def generate_joint_pos_interp_commands(self, cmd_poses, cmd_load, duration, k_q_p, k_qd_p):
        update_proto = robot_command_pb2.JointControlStreamRequest()

        pos_cmd = [0] * self.n_dofs
        vel_cmd = [0] * self.n_dofs
        load_cmd = [0] * self.n_dofs

        count = 0

        # Joint plan parameter [s]
        dt = 1 / 333

        joints_to_control = {}

        for iter in range(len(cmd_poses) - 1):
            new_command = True
            starting_time = time.time()
            previous_dt_time = starting_time
            while time.time() < (starting_time + duration):
                count += 1

                if self.should_stop:
                    return

                # Want to send the next command no sooner than ``dt`` after
                # the previous one, but never schedule the final update beyond
                # the requested trajectory duration.  This matters for ReLIC's
                # 10 ms physical control period: 333 Hz otherwise produces a
                # fourth update at 12 ms and silently lowers the control rate.
                this_dt_time_d = min(
                    max(previous_dt_time + dt, time.time()), starting_time + duration
                )
                sleep_time = max(0, this_dt_time_d - time.time())
                time.sleep(sleep_time)

                this_dt_time_actual = time.time()
                plan_time = this_dt_time_actual - starting_time

                for joint_ind in range(self.n_dofs):
                    if new_command:
                        # Set up joint planner
                        joints_to_control[joint_ind] = LinearInterpolator(
                            duration, cmd_poses[iter][joint_ind], cmd_poses[iter + 1][joint_ind])

                    # Calculate joint command
                    pos, vel = joints_to_control[joint_ind].calculate_command(plan_time)
                    pos_cmd[joint_ind] = pos
                    vel_cmd[joint_ind] = vel
                load_cmd = cmd_load
                new_command = False

                # Construct proto
                update_proto.Clear()
                set_timestamp_from_now(update_proto.header.request_timestamp)
                update_proto.header.client_name = "streaming_example_client"

                # Fill in gains the first dt
                if count == 1:
                    update_proto.joint_command.gains.k_q_p.extend(k_q_p)
                    update_proto.joint_command.gains.k_qd_p.extend(k_qd_p)

                update_proto.joint_command.position.extend(pos_cmd)
                update_proto.joint_command.velocity.extend(vel_cmd)
                update_proto.joint_command.load.extend(load_cmd)

                update_proto.joint_command.end_time.CopyFrom(
                    self.robot.time_sync.robot_timestamp_from_local_secs(time.time() + 0.05))

                # Let it extrapolate the command a little
                update_proto.joint_command.extrapolation_duration.CopyFrom(
                    seconds_to_duration(0.005))

                # Set user key for latency tracking
                update_proto.joint_command.user_command_key = count

                # Track the actual send time
                self.cmd_history[count] = time.time()
                yield update_proto

                self.started_streaming = True
                previous_dt_time = this_dt_time_d

    def generate_joint_pure_pd_commands(self, target_pose, k_q_p, k_qd_p):
        """Yield one standalone direct pure-PD command for compatibility.

        ReLIC uses the persistent stream above.  Retain this one-request
        generator for other callers that explicitly need a short-lived stream.
        """
        if len(target_pose) != self.n_dofs:
            raise ValueError(
                f"Expected {self.n_dofs} joint targets, got {len(target_pose)}."
            )
        self.started_streaming = True
        yield self._create_joint_pure_pd_request(
            target_pose=target_pose,
            k_q_p=k_q_p,
            k_qd_p=k_qd_p,
            include_gains=True,
        )
