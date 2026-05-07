# Copyright (c) 2024 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import sample_uniform

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

from SuperQ_ALORE.tasks.manager_based.superq_alore.mdp import object_management


def reset_target_object_from_catalog_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_name: str = "target_object",
) -> None:
    """Sample a catalog pose for each env in env_ids and write the corresponding object root state."""

    # sample a new pose for envs in env_ids and update env.active_pose_indices accordingly
    object_management.sample_target_assignments(env, env_ids)

    # obtain the sampledd pose of the object for each env in env_ids based on the sampled PoseEntry
    target_object = env.scene[asset_name]
    target_object_state = target_object.data.default_root_state[env_ids].clone()
    origins = env.scene.env_origins[env_ids]
    pose_entries = object_management.get_active_pose_entries(env, env_ids)

    pose_pos = torch.tensor(
        [pose.position for pose in pose_entries],
        dtype=target_object_state.dtype,
        device=env.device,
    )
    pose_rot = torch.tensor(
        [pose.orientation for pose in pose_entries],
        dtype=target_object_state.dtype,
        device=env.device,
    )

    # apply the initial pose for the object
    target_object_state[:, 0:3] = origins + pose_pos
    target_object_state[:, 3:7] = pose_rot
    target_object.write_root_state_to_sim(target_object_state, env_ids=env_ids)


def reset_robot_joints_from_catalog_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    position_range: tuple[float, float],
    velocity_range: tuple[float, float],
) -> None:
    """Reset robot joints from the sampled catalog pose for each env in env_ids."""
    # Guard in case event ordering places this before the object reset term.
    if not hasattr(env, "target_assignment_ready") or not torch.all(
        env.target_assignment_ready[env_ids]
    ):
        # since in object reset we first sample the target assignment and then set target_assignment_ready to True, 
        # here if we find that target_assignment_ready is not True for some envs in env_ids, 
        # then it means that we have not sampled the target assignment for these envs, which also means that we do not know which pose to use for these envs,
        #  so we cannot reset the robot joints based on the catalog pose for these envs. 
        # In this case, we just skip the reset of robot joints for these envs in this event, 
        # and wait for the next event to reset them after we have sampled the target assignment for them.
        object_management.sample_target_assignments(env, env_ids)

    pose_entries = object_management.get_active_pose_entries(env, env_ids)
    for local_i, env_id in enumerate(env_ids.tolist()):
        reset_joints_around_grasp_pose(
            env=env,
            env_ids=torch.tensor([env_id], dtype=torch.long, device=env.device),
            position_range=position_range,
            velocity_range=velocity_range,
            joint_position_ref=pose_entries[local_i].joint_positions,
        )


def reset_object_and_robot_from_catalog_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    position_range: tuple[float, float],
    velocity_range: tuple[float, float],
    asset_name: str = "target_object",
) -> None:
    """Atomically sample catalog pose, reset object pose, and reset robot joints for env_ids."""

    # 1) sample the target assignment once for this reset batch
    object_management.sample_target_assignments(env, env_ids)

    # 2) apply sampled object pose to the target object
    target_object = env.scene[asset_name]
    target_object_state = target_object.data.default_root_state[env_ids].clone()
    origins = env.scene.env_origins[env_ids]
    pose_entries = object_management.get_active_pose_entries(env, env_ids)

    pose_pos = torch.tensor(
        [pose.position for pose in pose_entries],
        dtype=target_object_state.dtype,
        device=env.device,
    )
    pose_rot = torch.tensor(
        [pose.orientation for pose in pose_entries],
        dtype=target_object_state.dtype,
        device=env.device,
    )

    target_object_state[:, 0:3] = origins + pose_pos
    target_object_state[:, 3:7] = pose_rot
    target_object.write_root_state_to_sim(target_object_state, env_ids=env_ids)

    # 3) reset robot joints using the sampled pose-specific joint references
    for local_i, env_id in enumerate(env_ids.tolist()):
        reset_joints_around_grasp_pose(
            env=env,
            env_ids=torch.tensor([env_id], dtype=torch.long, device=env.device),
            position_range=position_range,
            velocity_range=velocity_range,
            joint_position_ref=pose_entries[local_i].joint_positions,
        )


def reset_joints_around_default(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    position_range: tuple[float, float],
    velocity_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the robot joints in the interval around the default position and velocity by the given ranges.

    This function samples random values from the given ranges around the default joint positions and velocities.
    The ranges are clipped to fit inside the soft joint limits. The sampled values are then set into the physics
    simulation.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # get default joint state
    joint_min_pos = asset.data.default_joint_pos[env_ids] + position_range[0]
    joint_max_pos = asset.data.default_joint_pos[env_ids] + position_range[1]
    joint_min_vel = asset.data.default_joint_vel[env_ids] + velocity_range[0]
    joint_max_vel = asset.data.default_joint_vel[env_ids] + velocity_range[1]
    # clip pos to range
    joint_pos_limits = asset.data.soft_joint_pos_limits[env_ids, ...]
    joint_min_pos = torch.clamp(
        joint_min_pos, min=joint_pos_limits[..., 0], max=joint_pos_limits[..., 1]
    )
    joint_max_pos = torch.clamp(
        joint_max_pos, min=joint_pos_limits[..., 0], max=joint_pos_limits[..., 1]
    )
    # clip vel to range
    joint_vel_abs_limits = asset.data.soft_joint_vel_limits[env_ids]
    joint_min_vel = torch.clamp(
        joint_min_vel, min=-joint_vel_abs_limits, max=joint_vel_abs_limits
    )
    joint_max_vel = torch.clamp(
        joint_max_vel, min=-joint_vel_abs_limits, max=joint_vel_abs_limits
    )
    # sample these values randomly
    joint_pos = sample_uniform(
        joint_min_pos, joint_max_pos, joint_min_pos.shape, joint_min_pos.device
    )
    joint_vel = sample_uniform(
        joint_min_vel, joint_max_vel, joint_min_vel.shape, joint_min_vel.device
    )
    # set into the physics simulation
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
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
    joint_position_ref: dict[str, float],
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


def reset_target_object_position(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_name: str = "target_object",
    offset: tuple[float, float] = (0.0, 0.0),
):
    # target objects to manipulate
    target_object = env.scene[asset_name]
    target_object_state = env.scene[asset_name].data.default_root_state[env_ids].clone()
    origins = env.scene.env_origins[env_ids]
    target_object_state[:, 0] = origins[:, 0] + offset[0]
    target_object_state[:, 1] = origins[:, 1] + offset[1]
    target_object.write_root_state_to_sim(target_object_state, env_ids=env_ids)


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

