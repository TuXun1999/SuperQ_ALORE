# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run an environment with zero action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse
from pathlib import Path
import time
from robot import SPOT, SpotReLICEnvPLAY
import bosdyn
import torch
from constants import ORDERED_DOF_NAMES
from relic_pose_visualizer import ReLICPoseVisualizer
# add argparse arguments
parser = argparse.ArgumentParser(description="Zero agent for RL policy deployment on SPOT")
bosdyn.client.util.add_base_arguments(parser)
parser.add_argument(
    "--hostname", type=str, default="192.168.80.3", help="Hostname of the robot."
)
parser.add_argument(
    "--enable-joint-commands",
    action="store_true",
    help="Transmit ReLIC joint commands. Omit for an observation/policy dry run.",
)
parser.add_argument(
    "--steps", type=int, default=100000, help="Number of 100 Hz ReLIC updates to run."
)
parser.add_argument(
    "--visualize-pose",
    action="store_true",
    help="Show live current and ReLIC-target Spot poses in a local browser dashboard.",
)
parser.add_argument(
    "--visualizer-host",
    type=str,
    default="127.0.0.1",
    help="Dashboard bind address; use 0.0.0.0 to expose it on the local network.",
)
parser.add_argument(
    "--visualizer-port",
    type=int,
    default=8765,
    help="Dashboard port (use 0 to choose an available port automatically).",
)
parser.add_argument(
    "--record-policy-trace",
    action="store_true",
    help="Write each grouped 84-D ReLIC input and raw 12-D output to a text file.",
)
parser.add_argument(
    "--policy-trace-log",
    type=str,
    default="relic_policy_trace.txt",
    help="Text-file path for --record-policy-trace; overwritten at the start of each run.",
)
# parse the arguments
args_cli = parser.parse_args()

# Set to False to run the policy without collecting or printing latency data.
ENABLE_LATENCY_REPORTING = False
# Set to False to skip storing observations. The file is overwritten each run.
ENABLE_OBSERVATION_LOGGING = False
OBSERVATION_LOG_PATH = "relic_observations.pt"


def main():
    """Zero actions agent to deploy the pretrained RL policy."""
    # Connect to SPOT
    robot = SPOT(args_cli)
    robot.lease_alive()
    robot.power_on_stand()
    
    # Create environment
    env = SpotReLICEnvPLAY(
        robot,
        enable_latency_reporting=ENABLE_LATENCY_REPORTING,
        enable_observation_logging=ENABLE_OBSERVATION_LOGGING,
        observation_log_path=OBSERVATION_LOG_PATH,
        enable_policy_trace_logging=args_cli.record_policy_trace,
        policy_trace_log_path=args_cli.policy_trace_log,
    )

    try:
        if args_cli.visualize_pose:
            urdf_path = (
                Path(__file__).resolve().parents[1]
                / "source"
                / "SuperQ_ALORE"
                / "SuperQ_ALORE"
                / "assets"
                / "spot"
                / "spot_with_arm.urdf"
            )
            env.set_joint_pose_visualizer(
                ReLICPoseVisualizer(
                    urdf_path=urdf_path,
                    joint_names=ORDERED_DOF_NAMES,
                    host=args_cli.visualizer_host,
                    port=args_cli.visualizer_port,
                )
            )
        steps = 0
        max_steps = args_cli.steps
        next_step_time = time.perf_counter()
        while steps < max_steps:
            with torch.inference_mode():
                # Zero base velocity plus zero arm deltas is the ReLIC standing
                # command.  ``SpotReLICEnvPLAY`` latches the current arm pose
                # as the reference during the first update.
                actions = torch.zeros((1, 12))
                
                # Moving the spot backward slowly
                # actions[:, :3] = torch.tensor([-0.2, 0.0, 0.0], device=actions.device)
                env.step(actions)
                steps += 1

                # ``env.step`` streams a 10 ms joint trajectory.  Schedule
                # against an absolute deadline so this does not accidentally
                # add another 10 ms sleep and reduce ReLIC to 50 Hz.
                next_step_time += env.control_period_s
                remaining_time = next_step_time - time.perf_counter()
                if remaining_time > 0:
                    time.sleep(remaining_time)
                else:
                    next_step_time = time.perf_counter()
    finally:
        # Also release joint control and power down on Ctrl-C or a policy error.
        env.close()


if __name__ == "__main__":
    # run the main function
    main()
