"""MDP helpers for the QiVi articulated-chair task.

These helpers deliberately do not use the rigid-object catalog accessors.  The
chair root is a physical base and is therefore not a meaningful task pose once
the seat/back assembly moves.  The task tracks the ``frame`` link instead.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils

from SuperQ_ALORE.assets.object_catalog import ARM_JOINT_NAMES_IN_ORDER
from . import object_management
from .scene import (
    ARTICULATED_CHAIR_UPPER_BODY_INITIAL_POS_B,
    ARTICULATED_CHAIR_UPPER_BODY_NAME,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


def _yaw_from_quat(quat_wxyz: torch.Tensor) -> torch.Tensor:
    """Return the yaw component of a batch of wxyz quaternions."""

    w, x, y, z = quat_wxyz.unbind(dim=-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _quat_inverse(quat_wxyz: torch.Tensor) -> torch.Tensor:
    """Return a normalized quaternion inverse without assuming unit input."""

    norm_sq = torch.sum(quat_wxyz * quat_wxyz, dim=-1, keepdim=True)
    return torch.cat((quat_wxyz[..., :1], -quat_wxyz[..., 1:]), dim=-1) / norm_sq


def _tracked_body_index(env: ManagerBasedEnv, asset_name: str, body_name: str) -> int:
    """Resolve exactly one tracked body and fail with an actionable error otherwise."""

    chair = env.scene[asset_name]
    body_ids, body_names = chair.find_bodies(body_name)
    if len(body_ids) != 1:
        raise ValueError(
            f"Expected exactly one upper-body link matching '{body_name}' on '{asset_name}', "
            f"but found {list(body_names)}."
        )
    return int(body_ids[0])


def upper_body_pose_w(
    env: ManagerBasedEnv,
    asset_name: str = "target_object",
    body_name: str = ARTICULATED_CHAIR_UPPER_BODY_NAME,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return world position and orientation of the chair's upper assembly."""

    chair = env.scene[asset_name]
    body_id = _tracked_body_index(env, asset_name, body_name)
    return chair.data.body_pos_w[:, body_id], chair.data.body_quat_w[:, body_id]


def _upper_body_velocity_w(
    env: ManagerBasedEnv,
    asset_name: str,
    body_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    chair = env.scene[asset_name]
    body_id = _tracked_body_index(env, asset_name, body_name)
    return chair.data.body_lin_vel_w[:, body_id], chair.data.body_ang_vel_w[:, body_id]


def _goal_pose_w(
    env: ManagerBasedEnv,
    goal_term_name: str = "goal_pose",
) -> tuple[torch.Tensor, torch.Tensor]:
    term = env.command_manager.get_term(goal_term_name)
    goal_pos_w = term.goal_w if hasattr(term, "goal_w") else term.command[:, :3]
    if hasattr(term, "goal_quat_w"):
        return goal_pos_w, term.goal_quat_w

    goal_quat_w = torch.zeros((env.num_envs, 4), dtype=goal_pos_w.dtype, device=goal_pos_w.device)
    goal_quat_w[:, 0] = 1.0
    return goal_pos_w, goal_quat_w


def upper_body_pose_in_robot_frame_se2(
    env: ManagerBasedRLEnv,
    robot_name: str = "robot",
    asset_name: str = "target_object",
    body_name: str = ARTICULATED_CHAIR_UPPER_BODY_NAME,
) -> torch.Tensor:
    """Upper-chair-frame x/y/yaw pose in the robot base frame."""

    robot = env.scene[robot_name]
    body_pos_w, body_quat_w = upper_body_pose_w(env, asset_name, body_name)
    robot_quat_inv = _quat_inverse(robot.data.root_quat_w)
    pos_b = math_utils.quat_apply(robot_quat_inv, body_pos_w - robot.data.root_pos_w)
    quat_b = math_utils.quat_mul(robot_quat_inv, body_quat_w)
    return torch.cat((pos_b[:, :2], _yaw_from_quat(quat_b).unsqueeze(-1)), dim=-1)


def upper_body_to_goal_pos_local(
    env: ManagerBasedRLEnv,
    goal_term_name: str = "goal_pose",
    asset_name: str = "target_object",
    body_name: str = ARTICULATED_CHAIR_UPPER_BODY_NAME,
) -> torch.Tensor:
    """Goal displacement in the live upper-body frame (x/y only)."""

    goal_pos_w, _ = _goal_pose_w(env, goal_term_name)
    body_pos_w, body_quat_w = upper_body_pose_w(env, asset_name, body_name)
    return math_utils.quat_apply_inverse(body_quat_w, goal_pos_w - body_pos_w)[:, :2]


def upper_body_to_goal_rot_local(
    env: ManagerBasedRLEnv,
    goal_term_name: str = "goal_pose",
    asset_name: str = "target_object",
    body_name: str = ARTICULATED_CHAIR_UPPER_BODY_NAME,
) -> torch.Tensor:
    """Goal yaw relative to the live upper-body frame."""

    _, goal_quat_w = _goal_pose_w(env, goal_term_name)
    _, body_quat_w = upper_body_pose_w(env, asset_name, body_name)
    return _yaw_from_quat(math_utils.quat_mul(_quat_inverse(body_quat_w), goal_quat_w)).unsqueeze(-1)


def upper_body_lin_vel_in_body_frame(
    env: ManagerBasedRLEnv,
    asset_name: str = "target_object",
    body_name: str = ARTICULATED_CHAIR_UPPER_BODY_NAME,
) -> torch.Tensor:
    """Upper-body linear velocity expressed in the upper-body frame."""

    body_pos_w, body_quat_w = upper_body_pose_w(env, asset_name, body_name)
    del body_pos_w
    linear_vel_w, _ = _upper_body_velocity_w(env, asset_name, body_name)
    return math_utils.quat_apply_inverse(body_quat_w, linear_vel_w)


def upper_body_ang_vel_in_body_frame(
    env: ManagerBasedRLEnv,
    asset_name: str = "target_object",
    body_name: str = ARTICULATED_CHAIR_UPPER_BODY_NAME,
) -> torch.Tensor:
    """Upper-body angular velocity expressed in the upper-body frame."""

    _, body_quat_w = upper_body_pose_w(env, asset_name, body_name)
    _, angular_vel_w = _upper_body_velocity_w(env, asset_name, body_name)
    return math_utils.quat_apply_inverse(body_quat_w, angular_vel_w)


def upper_body_velocity(
    env: ManagerBasedRLEnv,
    asset_name: str = "target_object",
    body_name: str = ARTICULATED_CHAIR_UPPER_BODY_NAME,
) -> torch.Tensor:
    """Upper-body planar linear and yaw angular velocity for reward history."""

    linear_vel_b = upper_body_lin_vel_in_body_frame(env, asset_name, body_name)
    angular_vel_b = upper_body_ang_vel_in_body_frame(env, asset_name, body_name)
    return torch.cat((linear_vel_b[:, :2], angular_vel_b[:, 2:3]), dim=-1)


def articulated_chair_outlier_detected(
    env: ManagerBasedRLEnv,
    threshold: float = 1000.0,
    robot_name: str = "robot",
    asset_name: str = "target_object",
) -> torch.Tensor:
    """Terminate environments whose robot or any chair link has become invalid.

    This is the articulated counterpart to :func:`mdp.outlier_detected`.
    The rigid-task detector only observes the robot root, whereas a numerical
    failure can originate in any of the chair's 20 physical links or 19 joint
    states.  Checking those tensors before rewards and observations are
    computed prevents a single failed chair from propagating non-finite data
    into the PPO rollout.
    """

    def _has_outlier(values: torch.Tensor) -> torch.Tensor:
        values = values.reshape(values.shape[0], -1)
        return torch.any((torch.abs(values) > threshold) | ~torch.isfinite(values), dim=-1)

    chair = env.scene[asset_name]
    return (
        _has_outlier(env.action_manager.action)
        | _has_outlier(env.scene[robot_name].data.root_com_vel_w)
        | _has_outlier(chair.data.body_pos_w)
        | _has_outlier(chair.data.body_quat_w)
        | _has_outlier(chair.data.body_vel_w)
        | _has_outlier(chair.data.joint_pos)
        | _has_outlier(chair.data.joint_vel)
    )


def upper_body_pose_match_exp(
    env: ManagerBasedRLEnv,
    goal_term_name: str = "goal_pose",
    sigma: float = 1.0,
    asset_name: str = "target_object",
    body_name: str = ARTICULATED_CHAIR_UPPER_BODY_NAME,
) -> torch.Tensor:
    """Dense x/y/yaw pose-tracking reward for the upper chair assembly."""

    goal_pos_w, goal_quat_w = _goal_pose_w(env, goal_term_name)
    body_pos_w, body_quat_w = upper_body_pose_w(env, asset_name, body_name)
    pos_error = torch.linalg.vector_norm((goal_pos_w - body_pos_w)[:, :2], dim=-1)
    yaw_error = torch.atan2(
        torch.sin(_yaw_from_quat(goal_quat_w) - _yaw_from_quat(body_quat_w)),
        torch.cos(_yaw_from_quat(goal_quat_w) - _yaw_from_quat(body_quat_w)),
    ).abs()
    return torch.exp(-(pos_error + 0.25 * yaw_error) / max(float(sigma) ** 2, 1.0e-8))


def upper_body_velocity_toward_goal_exp(
    env: ManagerBasedRLEnv,
    goal_term_name: str = "goal_pose",
    sigma: float = 0.7071067812,
    use_unit_vel: bool = True,
    use_xy: bool = True,
    asset_name: str = "target_object",
    body_name: str = ARTICULATED_CHAIR_UPPER_BODY_NAME,
) -> torch.Tensor:
    """Reward upper-body velocity that points from its live pose toward the goal."""

    goal_pos_w, _ = _goal_pose_w(env, goal_term_name)
    body_pos_w, _ = upper_body_pose_w(env, asset_name, body_name)
    linear_vel_w, _ = _upper_body_velocity_w(env, asset_name, body_name)
    direction_w = goal_pos_w - body_pos_w
    velocity_w = linear_vel_w
    if use_xy:
        direction_w = direction_w.clone()
        velocity_w = velocity_w.clone()
        direction_w[:, 2] = 0.0
        velocity_w[:, 2] = 0.0
    direction_w = direction_w / (torch.linalg.vector_norm(direction_w, dim=-1, keepdim=True) + 1.0e-6)
    if use_unit_vel:
        velocity_w = velocity_w / (torch.linalg.vector_norm(velocity_w, dim=-1, keepdim=True) + 1.0e-6)
    return torch.exp(torch.sum(velocity_w * direction_w, dim=-1) / max(float(sigma) ** 2, 1.0e-8) - 1.0)


def upper_body_sparse_completion_reward(
    env: ManagerBasedRLEnv,
    goal_term_name: str = "goal_pose",
    dist_error: float = 0.05,
    angular_error: float = 5.0,
    success_reward: float = 1.0,
    asset_name: str = "target_object",
    body_name: str = ARTICULATED_CHAIR_UPPER_BODY_NAME,
) -> torch.Tensor:
    """Sparse completion reward based on upper-body x/y/yaw alignment."""

    goal_pos_w, goal_quat_w = _goal_pose_w(env, goal_term_name)
    body_pos_w, body_quat_w = upper_body_pose_w(env, asset_name, body_name)
    position_error = torch.linalg.vector_norm((goal_pos_w - body_pos_w)[:, :2], dim=-1)
    yaw_error_deg = torch.rad2deg(
        torch.atan2(
            torch.sin(_yaw_from_quat(goal_quat_w) - _yaw_from_quat(body_quat_w)),
            torch.cos(_yaw_from_quat(goal_quat_w) - _yaw_from_quat(body_quat_w)),
        ).abs()
    )
    success = (position_error <= float(dist_error)) & (yaw_error_deg <= float(angular_error))
    return success.to(torch.float32) * float(success_reward)


def upper_body_lin_vel_change_penalty(
    env: ManagerBasedRLEnv,
    asset_name: str = "target_object",
    body_name: str = ARTICULATED_CHAIR_UPPER_BODY_NAME,
) -> torch.Tensor:
    current = upper_body_velocity(env, asset_name, body_name)[:, :2]
    previous = env.observation_manager.compute_group("reward_calculation")["object_velocity"][:, 0, :2]
    return -torch.linalg.vector_norm(current - previous, dim=-1)


def upper_body_ang_vel_change_penalty(
    env: ManagerBasedRLEnv,
    asset_name: str = "target_object",
    body_name: str = ARTICULATED_CHAIR_UPPER_BODY_NAME,
) -> torch.Tensor:
    current = upper_body_velocity(env, asset_name, body_name)[:, 2]
    previous = env.observation_manager.compute_group("reward_calculation")["object_velocity"][:, 0, 2]
    return -(current - previous).abs()


def upper_body_flat_orientation_l2(
    env: ManagerBasedRLEnv,
    flat_threshold: float = 1.0e-3,
    asset_name: str = "target_object",
    body_name: str = ARTICULATED_CHAIR_UPPER_BODY_NAME,
) -> torch.Tensor:
    """Indicator penalty when the upper assembly is tilted away from world-up."""

    _, body_quat_w = upper_body_pose_w(env, asset_name, body_name)
    gravity_w = torch.zeros((env.num_envs, 3), dtype=body_quat_w.dtype, device=env.device)
    gravity_w[:, 2] = -1.0
    gravity_b = math_utils.quat_apply_inverse(body_quat_w, gravity_w)
    tilt_l2 = torch.sum(torch.square(gravity_b[:, :2]), dim=-1)
    return (tilt_l2 > float(flat_threshold)).to(torch.float32)


def upper_body_physical_properties(
    env: ManagerBasedRLEnv,
    asset_name: str = "target_object",
) -> torch.Tensor:
    """Return nominal friction and total articulated-chair mass for the critic."""

    chair = env.scene[asset_name]
    masses = chair.root_physx_view.get_masses()
    if masses.ndim == 3:
        masses = masses.squeeze(-1)
    total_mass = torch.sum(masses, dim=1, keepdim=True).to(env.device)
    friction = torch.full_like(total_mass, 0.5)
    return torch.cat((friction, total_mass, friction), dim=-1)


def upper_body_init_pose_in_robot_frame_se2(
    env: ManagerBasedRLEnv,
    robot_name: str = "robot",
    asset_name: str = "target_object",
) -> torch.Tensor:
    """Return the zero-joint upper-body pose relative to the catalog pre-grasp robot pose."""

    object_management.ensure_catalog_state(env)
    chair = env.scene[asset_name]
    robot_pose_local = env.active_robot_pose
    robot_pos_w = env.scene.env_origins + robot_pose_local[:, :3]
    robot_quat_inv = _quat_inverse(robot_pose_local[:, 3:])

    root_state = chair.data.default_root_state
    root_pos_w = env.scene.env_origins + root_state[:, :3]
    root_quat_w = root_state[:, 3:7]
    upper_offset_b = torch.tensor(
        ARTICULATED_CHAIR_UPPER_BODY_INITIAL_POS_B,
        device=env.device,
        dtype=root_state.dtype,
    ).expand(env.num_envs, -1)
    upper_pos_w = root_pos_w + math_utils.quat_apply(root_quat_w, upper_offset_b)
    upper_quat_w = root_quat_w

    pos_b = math_utils.quat_apply(robot_quat_inv, upper_pos_w - robot_pos_w)
    quat_b = math_utils.quat_mul(robot_quat_inv, upper_quat_w)
    return torch.cat((pos_b[:, :2], _yaw_from_quat(quat_b).unsqueeze(-1)), dim=-1)


def articulated_chair_object_idx(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The chair environment has one spawned object asset per sub-environment."""

    return torch.zeros((env.num_envs, 1), dtype=torch.long, device=env.device)


def reset_articulated_chair_and_robot_from_catalog_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    position_range: tuple[float, float],
    velocity_range: tuple[float, float],
    chair_name: str = "target_object",
) -> None:
    """Reset one QiVi articulation and the robot using catalogued pre-grasp poses.

    ``pre_grasping.yaml`` remains the source of robot placement and arm-joint
    references.  Unlike the rigid task, it is *not* used to decide which USD to
    spawn: this environment always uses the single configured chair articulation.
    """

    object_management.ensure_catalog_state(env)
    env_ids = env_ids.to(device=env.device, dtype=torch.long)
    origins = env.scene.env_origins[env_ids]

    chair = env.scene[chair_name]
    chair_state = chair.data.default_root_state[env_ids].clone()
    chair_state[:, :3] += origins
    chair_state[:, 7:13] = 0.0
    chair.write_root_state_to_sim(chair_state, env_ids=env_ids)
    chair.write_joint_state_to_sim(
        chair.data.default_joint_pos[env_ids].clone(),
        chair.data.default_joint_vel[env_ids].clone(),
        env_ids=env_ids,
    )

    robot = env.scene["robot"]
    robot_state = robot.data.default_root_state[env_ids].clone()
    robot_pose_local = env.active_robot_pose[env_ids]
    robot_state[:, :3] = origins + robot_pose_local[:, :3]
    robot_state[:, 3:7] = robot_pose_local[:, 3:]
    robot_state[:, 7:13] = 0.0
    robot.write_root_state_to_sim(robot_state, env_ids=env_ids)

    arm_joint_ids, _ = robot.find_joints(ARM_JOINT_NAMES_IN_ORDER)
    robot_joint_pos = robot.data.default_joint_pos[env_ids].clone()
    arm_joint_pos = object_management.get_active_arm_joint_reference(env, env_ids).clone()
    if position_range != (0.0, 0.0):
        arm_joint_pos += torch.empty_like(arm_joint_pos).uniform_(*position_range)
    robot_joint_pos[:, arm_joint_ids] = arm_joint_pos
    robot_joint_vel = robot.data.default_joint_vel[env_ids].clone()
    if velocity_range != (0.0, 0.0):
        robot_joint_vel[:, arm_joint_ids] = torch.empty_like(arm_joint_pos).uniform_(*velocity_range)
    else:
        robot_joint_vel[:, arm_joint_ids] = 0.0
    robot.write_joint_state_to_sim(robot_joint_pos, robot_joint_vel, env_ids=env_ids)
