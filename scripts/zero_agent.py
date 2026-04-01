# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run an environment with zero action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Zero agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import isaaclab_tasks  # noqa: F401
import SuperQ_ALORE.tasks  # noqa: F401
import torch
from isaaclab_tasks.utils import parse_env_cfg


def main():
    """Zero actions agent with Isaac Lab environment."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment
    env.reset()
    dt = env.unwrapped.step_dt
    steps = env.unwrapped.max_episode_length

    timestep = 0
    arm_target = torch.tensor([0.0099, -1.3287, 1.7197,-0.0031, 0.9530, -0.0154, -1.1852])
    arm_default = torch.tensor([0.0, -0.9, 1.8, 0.0, -0.9, 0.0, -1.54]) # obtained from the initial joint state of the robot in the simulation (also the default joint position configured in the asset)

    chair_reset_timestep = (int)(1.8 / dt) # The chair will be reset after 2s, so we want to complete the grasping before that
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # compute zero actions
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            """
            actions: base velocity (3) + arm joint (7) + base pose (2: pitch, height)
            (Forced to match 12D action space of the pretrained locomotion policy)
            """
            # if timestep <= chair_reset_timestep: # The chair will be reset after 2s, so we want to complete the grasping before that
            #     actions[:, 3:10] = torch.lerp(arm_default, arm_target, timestep / chair_reset_timestep) # linear interpolation from default to target joint positions
            # else:
            #     actions[:, 3:10] = arm_target # directly command the target joint positions after chair reset, to bypass the disturbance from chair reset and keep the grasping pose stable
            actions[:, 3:10] = arm_target
            # Only command the base to be at a suitable height & pitch 
            # (roll action not desired)
            actions[:, :3] = torch.tensor([0.0, 0.0, 0.0], device=env.unwrapped.device) # zero base velocity
            actions[:, -2:] = torch.tensor([-0.0,  0.515], device=env.unwrapped.device) #
            env.step(actions)
            timestep += 1
            
            timestep = timestep % steps

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
