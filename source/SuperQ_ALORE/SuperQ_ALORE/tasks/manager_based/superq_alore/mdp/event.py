# Copyright (c) 2024 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import sample_uniform

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


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
):
    # target objects to manipulate
    target_object = env.scene[asset_name]
    target_object_state = env.scene[asset_name].data.default_root_state.clone()
    origins = env.scene.env_origins
    target_object_state[:, 0:2] = origins[:, 0:2]
    target_object.write_root_state_to_sim(target_object_state)

