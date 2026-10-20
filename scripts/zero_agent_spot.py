# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Approach and grasp a chair, then pull backward using Spot MPC or ReLIC."""

import argparse
import time


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    # Accept the SDK's positional hostname as well as the existing --hostname
    # spelling, without registering two arguments with the same destination.
    parser.add_argument("robot_hostname", nargs="?", help="Robot address; alternatively use --hostname.")
    parser.add_argument("--hostname", help="Robot address (default: 192.168.80.3).")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print debug-level messages.")
    parser.add_argument(
        "--use-relic", action="store_true",
        help="Use ReLIC after grasping. On exit: release, stow, self-right, then power off.",
    )
    parser.add_argument(
        "--steps", type=int, default=100,
        help="Number of backward-command updates. ReLIC runs at 100 Hz (100 steps is about 1 s).",
    )
    return parser


def main(args_cli):
    if args_cli.steps <= 0:
        raise ValueError("--steps must be positive.")
    if args_cli.hostname and args_cli.robot_hostname and args_cli.hostname != args_cli.robot_hostname:
        raise ValueError("Specify the same robot address when using both hostname arguments.")
    args_cli.hostname = args_cli.hostname or args_cli.robot_hostname or "192.168.80.3"
    # This flag authorizes joint commands, but streaming starts only when the
    # ReLIC environment is created AFTER the grasp sequence below.
    args_cli.enable_joint_commands = args_cli.use_relic

    from robot import SPOT, SpotRLEnvPLAY, SpotReLICEnvPLAY
    import torch

    robot = SPOT(args_cli)
    lease = env = None
    power_requested = False
    try:
        lease = robot.lease_alive()
        power_requested = True
        robot.power_on_stand()

        # Keep the existing SDK/MPC approach and grasp sequence in both modes.
        robot.robot.logger.info("Moving to pregrasp location based on AprilTag detection.")
        robot.pregrasp_location_apriltag(option="world_object_service")
        robot.robot.logger.info("Reached pregrasp location.")
        robot.robot.logger.info("Moving to grasp location.")
        robot.open_gripper()
        robot.arm_joint_control([0.00, -1.64, 1.87, 0.0, 1.04, 0.0, -0.9])
        time.sleep(1)
        robot.close_gripper()  # Includes the existing 0.5-second closing wait.
        robot.robot.logger.info("Gripper closing wait completed; starting backward motion.")

        if args_cli.use_relic:
            # Same controller as zero_agent_ReLIC_spot.py, with no bridge,
            # visualizer, latency reporting, or policy/observation recording.
            # The first step latches the measured arm AND closed gripper pose.
            env = SpotReLICEnvPLAY(
                robot,
                enable_latency_reporting=False,
                enable_observation_logging=False,
                enable_policy_trace_logging=False,
            )
        else:
            env = SpotRLEnvPLAY(robot)

        # [vx, vy, yaw rate, seven arm deltas, pitch, height]. Zero arm
        # deltas hold the grasp pose; the action term handles torso commands.
        actions = torch.zeros((1, 12))
        actions[:, 0] = -0.2
        next_step_time = time.perf_counter()
        for i in range(args_cli.steps):
            with torch.inference_mode():
                env.step(actions)
            if args_cli.use_relic:
                # ReLIC needs continuous 100 Hz observations/inference/targets,
                # not a single SDK velocity request. Do not burst late updates.
                next_step_time += env.control_period_s
                remaining = next_step_time - time.perf_counter()
                if remaining > 0:
                    time.sleep(remaining)
                else:
                    next_step_time = time.perf_counter()
    finally:
        try:
            if args_cli.use_relic and power_requested:
                if env is not None:
                    # Same shutdown as the existing ReLIC deployment script,
                    # including on Ctrl-C or an inference error.
                    env.close(relic = True)
                else:
                    # Construction/approach can fail before env is assigned.
                    robot.shutdown_smooth()
            # Preserve MPC's existing behavior: leave Spot powered on.
        finally:
            if lease is not None:
                robot.lease_return()


if __name__ == "__main__":
    main(build_parser().parse_args())
