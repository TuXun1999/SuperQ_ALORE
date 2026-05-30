# Copyright (c) 2024 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from __future__ import annotations

import torch
from typing import TYPE_CHECKING
import numpy as np
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import sample_uniform

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

from SuperQ_ALORE.assets.object_catalog import (
    ARM_JOINT_NAMES_IN_ORDER,
    OBJECT_CATALOG,
    OBJECT_IDS,
    POSE_IDS_BY_OBJECT,
    PoseEntry,
)
from SuperQ_ALORE.tasks.manager_based.superq_alore.mdp.scene import OBJECT_IDX_ENVS, POSE_IDX_LOCAL_ENVS

""" 
Reset the robot joints at the pre-defined initial position obtained 
from SuperQ-GRASP & Inverse Kinematics, with some small random noise added to them.
"""
# TODO: reset the robot joint at the pre-defined initial position
# TODO: reset the robot at different initial conditions based on different grasp poses
def reset_joints_around_grasp_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    position_range: tuple[float, float],
    velocity_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the robot joints in the pre-defined positions obtained from SuperQ-GRASP & Inverse Kinematics,
    with some small random noise added to them
    
    
    The ranges are clipped to fit inside the soft joint limits. The sampled values are then set into the physics
    simulation.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # get default joint state
    joint_default_pos = asset.data.default_joint_pos[env_ids]
    
    # obtain the reference joint positions for the assigned grasp pose
    obj_idx_reset = np.asarray(OBJECT_IDX_ENVS, dtype=int)[env_ids.tolist()]
    pose_idx_reset = np.asarray(POSE_IDX_LOCAL_ENVS, dtype=int)[env_ids.tolist()]

    # Construct the dict corresponding to the joint angles for each
    # joint separately across all envs. 
    # keys: joint names; 
    # values: tensor of shape (num_envs,) containing the joint angle for 
    # that joint in each env, obtained from the YAML catalog
    joint_position_ref = {}
    for joint_name in ARM_JOINT_NAMES_IN_ORDER:
        joint_position_ref[joint_name] = []
        for obj_idx, pose_idx in zip(obj_idx_reset, pose_idx_reset):
            pose_entry: PoseEntry = OBJECT_CATALOG[obj_idx].poses[pose_idx]
            joint_position_ref[joint_name].append(pose_entry.joint_positions[joint_name])
    
    for key, value in joint_position_ref.items():
        joint_position_ref[key] = torch.tensor(value, device=joint_default_pos.device)
    
    # Find the joint ids corresponding to the joint names in the reference joint position dictionary
    joint_ids, joint_names = asset.find_joints(joint_position_ref.keys())
    joint_grasp_pose_pos = joint_default_pos.clone()
    # Set the joint positions in the reference joint position dictionary to the corresponding joint ids
    for joint_name, joint_pos in joint_position_ref.items():
        joint_id = joint_ids[joint_names.index(joint_name)]
        joint_grasp_pose_pos[:, joint_id] = joint_pos
    
    # Sample a small noise around the reference joint positions
    joint_min_pos = joint_grasp_pose_pos + position_range[0]
    joint_max_pos = joint_grasp_pose_pos + position_range[1]
    
    # clip pos to range
    joint_pos_limits = asset.data.soft_joint_pos_limits[env_ids, ...]
    joint_min_pos = torch.clamp(
        joint_min_pos, min=joint_pos_limits[..., 0], max=joint_pos_limits[..., 1]
    )
    joint_max_pos = torch.clamp(
        joint_max_pos, min=joint_pos_limits[..., 0], max=joint_pos_limits[..., 1]
    )
    
    # sample these values randomly
    joint_pos = sample_uniform(
        joint_min_pos, joint_max_pos, joint_min_pos.shape, joint_min_pos.device
    )
    
    # Add some random sampling in velocity as well
    joint_vel_default = torch.zeros_like(joint_pos)
    joint_min_vel = joint_vel_default + velocity_range[0]
    joint_max_vel = joint_vel_default + velocity_range[1]
    joint_vel = sample_uniform(
        joint_min_vel, joint_max_vel, joint_min_vel.shape, joint_min_vel.device
    )

    # set into the physics simulation
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)


def reset_target_object_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
):
    # Objects are spawned as per-object subsets of environments
    env_ids_cpu = env_ids.detach().cpu().tolist()
    
    # Only handle the selected envs
    obj_idx_reset = np.asarray(OBJECT_IDX_ENVS, dtype=int)[env_ids_cpu]
    pose_idx_reset = np.asarray(POSE_IDX_LOCAL_ENVS, dtype=int)[env_ids_cpu]
    origins = env.scene.env_origins[env_ids]

    # For the objects in the selected envs, call the attribute & reset the position
    # one by one
    for obj_id in range(len(OBJECT_CATALOG)):
        selected_rows = np.where(obj_idx_reset == obj_id)[0]
        if selected_rows.size == 0:
            continue

        target_object = env.scene[f"target_object_{obj_id}"]

        # Build global->local index mapping for this object view.
        all_envs_for_obj = np.where(np.asarray(OBJECT_IDX_ENVS, dtype=int) == obj_id)[0]
        global_to_local = {int(g): int(i) for i, g in enumerate(all_envs_for_obj.tolist())}

        global_env_ids_for_obj = [env_ids_cpu[row] for row in selected_rows.tolist()]
        local_env_ids_list = [global_to_local[g] for g in global_env_ids_for_obj]
        local_env_ids = torch.tensor(local_env_ids_list, device=env.device, dtype=torch.long)

        target_object_state = target_object.data.default_root_state[local_env_ids].clone()

        for j, row in enumerate(selected_rows.tolist()):
            pose_entry: PoseEntry = OBJECT_CATALOG[obj_id].poses[int(pose_idx_reset[row])]
            offset = pose_entry.position[0:2]
            target_object_state[j, 0] = origins[row, 0] + offset[0]
            target_object_state[j, 1] = origins[row, 1] + offset[1]
            quat = pose_entry.orientation  # w, x, y, z
            target_object_state[j, 3:7] = torch.tensor(quat, device=target_object_state.device)

        target_object.write_root_state_to_sim(target_object_state, env_ids=local_env_ids)
    


def move_target_object_closer(
    env: ManagerBasedEnv,
    interval_range_s, 
    asset_name: str,
    robot_asset_name: str = "robot",
):
    # target objects to manipulate
    target_object = env.scene[asset_name]
    target_robot = env.scene[robot_asset_name]
    target_robot_state = target_robot.data.root_state_w.clone()
    target_object_state = env.scene[asset_name].data.default_root_state.clone()
    origins = env.scene.env_origins
    target_object_state[:, 0:2] = origins[:, 0:2]
    target_object.write_root_state_to_sim(target_object_state)

