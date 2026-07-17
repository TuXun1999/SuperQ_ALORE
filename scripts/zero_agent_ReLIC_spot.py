# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run an environment with zero action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse
import time
from robot import SPOT, SpotReLICEnvPLAY
import bosdyn
import torch
# add argparse arguments
parser = argparse.ArgumentParser(description="Zero agent for RL policy deployment on SPOT")
bosdyn.client.util.add_base_arguments(parser)
parser.add_argument(
    "--hostname", type=str, default="192.168.80.3", help="Hostname of the robot."
)

# parse the arguments
args_cli = parser.parse_args()



def main():
    """Zero actions agent to deploy the pretrained RL policy."""
    # Connect to SPOT
    robot = SPOT(args_cli)
    robot.lease_alive()
    robot.power_on_stand()
    
    # Create environment
    env = SpotReLICEnvPLAY(robot)

    steps = 0
    max_steps = 100  # Define the maximum number of steps
    while steps < max_steps:
        with torch.inference_mode():
            # compute zero actions
            actions = torch.zeros((1, 12))
            """
            actions: base velocity (3) + arm joint (7) + base pose (2: pitch, height)
            (Forced to match 12D action space of the pretrained locomotion policy)
            """

            # set the active arm joints to the reference joint positions for the active pose, 
            # so that the arm will hold the desired pose with zero actions
            # also notice that we now use relative action, so we send zero delta.
            actions[:, 3:10] = torch.zeros_like(actions[:, 3:10]) # command zero delta for the arm joints to hold the arm at the desired pose defined by the active arm joint reference

            # Only command the base to be at a suitable height & pitch 
            # (roll action not desired)
            actions[:, :3] = torch.tensor([-0.0, 0.0, 0.0]) # command a base velocity to move forward after chair reset, to avoid the disturbance from chair reset and keep the grasping pose stable

            env.step(actions)
            steps += 1


    # # close the simulator
    # env.close()


if __name__ == "__main__":
    # run the main function
    main()
