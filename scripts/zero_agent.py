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
parser.add_argument(
    "--object_id",
    type=str,
    default=None,
    help="Optional catalog object id to force for all envs (e.g. chair_1).",
)
parser.add_argument(
    "--pose_id",
    type=str,
    default=None,
    help="Optional pose id to force for all envs (e.g. back, handle). Requires --object_id.",
)

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
from SuperQ_ALORE.assets.object_catalog import (
    ARM_JOINT_NAMES_IN_ORDER,
    OBJECT_CATALOG,
    OBJECT_IDS,
    POSE_IDS_BY_OBJECT,
)
from SuperQ_ALORE.assets.spot.constants import GRASP_POSE_1_JOINT_POS
from SuperQ_ALORE.tasks.manager_based.superq_alore.mdp import event as mdp_event
from SuperQ_ALORE.tasks.manager_based.superq_alore.mdp import object_management as mdp_object_management


def _apply_catalog_selection(env, object_id: str | None, pose_id: str | None) -> None:
    """Optionally force all envs to one catalog (object, pose) selection."""
    if object_id is None and pose_id is None:
        return
    if object_id is None or pose_id is None:
        raise ValueError("Both --object_id and --pose_id must be provided together.")
    if object_id not in OBJECT_IDS:
        raise ValueError(f"Invalid --object_id='{object_id}'. Available: {list(OBJECT_IDS)}")
    if pose_id not in POSE_IDS_BY_OBJECT[object_id]:
        raise ValueError(
            f"Invalid --pose_id='{pose_id}' for object '{object_id}'. "
            f"Available: {list(POSE_IDS_BY_OBJECT[object_id])}"
        )

    # load objects and initialize the catalog
    mdp_object_management.ensure_catalog_state(env)

    # shape: (num_envs,) long tensor of the selected catalog object index for each env
    env_ids = torch.arange(env.num_envs, dtype=torch.long, device=env.device)

    # look for the indices of the specified object and pose in the catalog, and set them as the active ones for all envs
    obj_idx = OBJECT_IDS.index(object_id)
    pose_idx = POSE_IDS_BY_OBJECT[object_id].index(pose_id)
    pose_entry = OBJECT_CATALOG[obj_idx].poses[pose_idx]

    env.active_object_indices[env_ids] = obj_idx
    env.active_pose_indices[env_ids] = pose_idx

    # set the arm joint reference for all envs to the specified pose's joint positions, 
    # so that the default zero action will hold the arm at the desired pose
    ref = torch.tensor(
        [pose_entry.joint_positions[name] for name in ARM_JOINT_NAMES_IN_ORDER],
        dtype=torch.float32,
        device=env.device,
    )

    # obtain the tensor of shape (num_envs, num_arm_joints) for the active arm joint reference
    # by expanding the pose's joint positions to all envs
    env.active_arm_joint_reference[env_ids] = ref.unsqueeze(0).expand(env.num_envs, -1)

    # set the flag to indicate that the target assignment is ready, so that there will be no random target assignment in the event function 
    # and the specified catalog selection will be used for all envs
    env.target_assignment_ready[env_ids] = True

    # obtain the active pose
    origins = env.scene.env_origins[env_ids]
    pose_pos = torch.tensor(pose_entry.position, dtype=torch.float32, device=env.device).unsqueeze(0)
    pose_rot = torch.tensor(pose_entry.orientation, dtype=torch.float32, device=env.device).unsqueeze(0)
    pose_pos = pose_pos.expand(env.num_envs, -1)
    pose_rot = pose_rot.expand(env.num_envs, -1)

    # the underground pose for non-active objects: same position for all envs, with a z value underground; and identity rotation
    underground_pos = origins.clone()
    underground_pos[:, 2] = -100.0
    underground_rot = torch.zeros_like(pose_rot)
    underground_rot[:, 0] = 1.0

    # iterate through all the objects
    for idx in range(len(OBJECT_CATALOG)):

        # obtain the object entry from the scene
        obj = env.scene[f"target_object_{idx}"]
        state = obj.data.default_root_state[env_ids].clone()
        state[:, 7:] = 0.0

        # for specified active object, set it to its active pose
        if idx == obj_idx:
            state[:, 0:3] = origins + pose_pos
            state[:, 3:7] = pose_rot

        # for other non-active objects, set them to the underground pose
        else:
            state[:, 0:3] = underground_pos
            state[:, 3:7] = underground_rot
        obj.write_root_state_to_sim(state, env_ids=env_ids)

    # set the arm joints corresponding to the active object and its active pose 
    mdp_event.reset_joints_around_grasp_pose(
        env=env,
        env_ids=env_ids,
        position_range=(0.0, 0.0),
        velocity_range=(0.0, 0.0),
        joint_position_ref={
            # obtain the corresponding joint positions from the pose_entry
            name: float(pose_entry.joint_positions[name])
            for name in ARM_JOINT_NAMES_IN_ORDER
        },
    )

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
    _apply_catalog_selection(env.unwrapped, args_cli.object_id, args_cli.pose_id)

    dt = env.unwrapped.step_dt
    steps = env.unwrapped.max_episode_length

    timestep = 0
    
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # compute zero actions
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            """
            actions: base velocity (3) + arm joint (7) + base pose (2: pitch, height)
            (Forced to match 12D action space of the pretrained locomotion policy)
            """

            # set the active arm joints to the reference joint positions for the active pose, 
            # so that the arm will hold the desired pose with zero actions
            actions[:, 3:10] = env.unwrapped.active_arm_joint_reference[:, :7]

            # Only command the base to be at a suitable height & pitch 
            # (roll action not desired)
            actions[:, :3] = torch.tensor([-0.2, 0.0, 0.0], device=env.unwrapped.device) # command a base velocity to move forward after chair reset, to avoid the disturbance from chair reset and keep the grasping pose stable

            obs, _, _, _, _ = env.step(actions)

            timestep += 1
            timestep = timestep % steps

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
