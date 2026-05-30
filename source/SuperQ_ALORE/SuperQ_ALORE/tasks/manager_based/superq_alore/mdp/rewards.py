# Copyright (c) 2024 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from __future__ import annotations


import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import RewardTermCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import quat_apply, quat_mul
from SuperQ_ALORE.assets.object_catalog import OBJECT_CATALOG

"""
Group 1: Object related rewards (primary task)
"""
## (1) Command tracking rewards
# Linear velocity tracking (object velocity in robot frame)
def track_lin_vel_xy_exp(
    env: ManagerBasedRLEnv,
    command_name: str = "object_velocity",
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using abs exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    obj_lin_vel_b_list = []
    for i in range(len(OBJECT_CATALOG)):
        # Assume the objects are arranged in order by their IDs
        obj = env.scene[f"target_object_{i}"]
        obj_lin_vel_b_list.append(obj.data.root_lin_vel_b[:, :2])  # shape (N_envs_obj, 2) 
    obj_lin_vel_b = torch.cat(obj_lin_vel_b_list, dim=0)  # shape (N_envs, 2)
    # compute the error
    target = env.command_manager.get_command(command_name)[:, :2]
    lin_vel_error = torch.linalg.norm(
        (target - obj_lin_vel_b), dim=1
    )
    return torch.exp(-lin_vel_error / 0.25)

# Angular velocity tracking (object yaw velocity in robot frame)
def track_ang_vel_yaw_exp(
    env: ManagerBasedRLEnv,
    command_name: str = "object_velocity",
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) using abs exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    obj_ang_vel_b_list = []
    for i in range(len(OBJECT_CATALOG)):
        # Assume the objects are arranged in order by their IDs
        obj = env.scene[f"target_object_{i}"]
        obj_ang_vel_b_list.append(obj.data.root_ang_vel_b[:, 2].unsqueeze(1))  # shape (N_envs_obj, 1)
    obj_ang_vel_b = torch.cat(obj_ang_vel_b_list, dim=0)  # shape (N_envs, 1)
    # compute the error
    target = env.command_manager.get_command(command_name)[:, 2].unsqueeze(1)
    ang_vel_error = torch.linalg.norm(
        (target - obj_ang_vel_b), dim=1
    )
    return torch.exp(-ang_vel_error / 0.25)

## (2) System alive reward
# Alive checking
def is_alive(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward for being alive."""
    return (~env.termination_manager.terminated).float()

## (3) Smoothness rewards
# Alignment between the object & the robot
def quat_to_rot_matrix(quat: torch.Tensor) -> torch.Tensor:
    """Convert quaternion to rotation matrix."""
    # Assumes input is (w, x, y, z)
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    B = quat.size(0)
    rot = torch.zeros((B, 3, 3), device=quat.device)

    rot[:, 0, 0] = 1 - 2 * (y ** 2 + z ** 2)
    rot[:, 0, 1] = 2 * (x * y - z * w)
    rot[:, 0, 2] = 2 * (x * z + y * w)
    rot[:, 1, 0] = 2 * (x * y + z * w)
    rot[:, 1, 1] = 1 - 2 * (x ** 2 + z ** 2)
    rot[:, 1, 2] = 2 * (y * z - x * w)
    rot[:, 2, 0] = 2 * (x * z - y * w)
    rot[:, 2, 1] = 2 * (y * z + x * w)
    rot[:, 2, 2] = 1 - 2 * (x ** 2 + y ** 2)

    return rot
def _euler_from_quat(quat_angle: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternion to Euler angles (roll, pitch, yaw).
    """
    w = quat_angle[:,0]
    x = quat_angle[:,1]
    y = quat_angle[:,2]
    z = quat_angle[:,3]

    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll_x = torch.atan2(t0, t1)
    
    t2 = 2.0 * (w * y - z * x)
    t2 = torch.clip(t2, -1, 1)
    pitch_y = torch.asin(t2)
    
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw_z = torch.atan2(t3, t4)
    
    return roll_x, pitch_y, yaw_z # in radians

# (DEPRECATED) Yaw alignment doesn't seem to help in training
# def yaw_alignment_reward(
#     env: ManagerBasedRLEnv,
#     asset_name: str = "target_object",
#     robot_name: str = "robot"
#     ):
#     """Encourage the object to be aligned with the robot in the yaw direction."""
#     # Extract the assets
#     robot = env.scene[robot_name]
#     asset = env.scene[asset_name]
    
#     # Find the relative yaw different between the object & the robot
#     yaw_diff = (
#         (_euler_from_quat(asset.data.root_quat_w)[2] -         # asset.data.root_quat_w
#          # in our setting, the offset should be pi / 2
#         _euler_from_quat(robot.data.root_quat_w)[2] + torch.pi / 2.0)
#         % (2 * torch.pi) - torch.pi
#     )
#     yaw_alignment_reward = -torch.abs(yaw_diff) / torch.pi
#     return yaw_alignment_reward


# Velocity along the z-direction (to discourage lifting or digging the object)
def lin_vel_z_l2(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """Penalize linear velocity along the z-direction."""
    lin_vel_z_list = []
    for i in range(len(OBJECT_CATALOG)):
        # Assume the objects are arranged in order by their IDs
        obj = env.scene[f"target_object_{i}"]
        lin_vel_z_list.append(obj.data.root_lin_vel_b[:, 2].unsqueeze(1))  # shape (N_envs_obj, 1)
    lin_vel_z = torch.cat(lin_vel_z_list, dim=0)  # shape (N_envs, 1)
    return -torch.square(lin_vel_z).squeeze(1)

# Angular velocity along the x/y-direction (to discourage flipping the object)
def ang_vel_xy_l2(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """Penalize angular velocity along the x/y-direction."""
    ang_vel_xy_list = []
    for i in range(len(OBJECT_CATALOG)):
        # Assume the objects are arranged in order by their IDs
        obj = env.scene[f"target_object_{i}"]
        ang_vel_xy_list.append(obj.data.root_ang_vel_b[:, :2])  # shape (N_envs_obj, 2)
    ang_vel_xy = torch.cat(ang_vel_xy_list, dim=0)  # shape (N_envs, 2)
    return -torch.sum(torch.square(ang_vel_xy), dim=1)

#(DEPRECATED) Flat orientation (to discourage flipping the object)
# def flat_orientation_l2(
#     env: ManagerBasedRLEnv,
#     asset_name: str = "target_object"
# ) -> torch.Tensor:
#     """Penalize deviation from flat orientation."""
#     asset: RigidObject = env.scene[asset_name]
#     flat_orientation = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
#     return -flat_orientation

# Velocity change penalty (to encourage smooth motion)
def lin_vel_change_penalty(
    env: ManagerBasedRLEnv,
    asset_name: str = "target_object"
) -> torch.Tensor:
    """Penalize change in linear velocity."""
    current_lin_vel_list = []
    for i in range(len(OBJECT_CATALOG)):
        # Assume the objects are arranged in order by their IDs
        obj = env.scene[f"target_object_{i}"]
        current_lin_vel_list.append(obj.data.root_lin_vel_b[:, :2])  # shape (N_envs_obj, 2)
    current_lin_vel = torch.cat(current_lin_vel_list, dim=0)  # shape (N_envs, 2)
    prev_lin_vel = env.observation_manager.compute_group("reward_calculation")["object_velocity"][:, 0, :2]  # (vx, vy)
    lin_vel_change_penalty = torch.norm(current_lin_vel - prev_lin_vel, dim=-1)
    return -lin_vel_change_penalty


def ang_vel_change_penalty(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """Penalize change in angular velocity."""
    current_ang_vel_list = []
    for i in range(len(OBJECT_CATALOG)):
        # Assume the objects are arranged in order by their IDs
        obj = env.scene[f"target_object_{i}"]
        current_ang_vel_list.append(obj.data.root_ang_vel_b[:, 2].unsqueeze(1))  # shape (N_envs_obj, 1)
    current_ang_vel = torch.cat(current_ang_vel_list, dim=0)  # shape (N_envs, 1)
    prev_ang_vel = env.observation_manager.compute_group("reward_calculation")["object_velocity"][:, 0, 2].unsqueeze(1)  # yaw velocity
    ang_vel_change_penalty = torch.abs(current_ang_vel - prev_ang_vel).squeeze(1)
    return -ang_vel_change_penalty

# Distance penalty (to encourage the robot to place the object at a fixed distance away)
def quat_inverse_safe(q: torch.Tensor) -> torch.Tensor:
    norm_sq = torch.sum(q * q, dim=-1, keepdim=True)  # (..., 1)
    conj = torch.cat([q[..., :1], -q[..., 1:]], dim=-1)
    return conj / norm_sq

def distance_penalty(
    env: ManagerBasedRLEnv,
    robot_name: str = "robot",
    end_effector_link_name: str = "gripper_link",
    distance_threshold: float = 0.6
) -> torch.Tensor:
    """Penalize the distance between the object and the robot."""
    robot = env.scene[robot_name]
    
    # Find the body index for the end-effector link
    body_index = robot.body_names.index(end_effector_link_name)
    # Obtain the link transforms in robot's root frame
    ee_pos_w = robot.data.body_pos_w[:, body_index]  # shape (N_envs, 3)

    # Obtain the robot base pos & quat in world frame
    robot_base_pos = robot.data.root_pos_w  # (num_envs, 3)
    robot_quat_inv = quat_inverse_safe(robot.data.root_quat_w)  # (num_envs, 4)

    # Calculate the relative displacement
    ee_pos_relative = ee_pos_w - robot_base_pos  # (num_envs, 3)
    ee_pos_in_robot_frame = quat_apply(robot_quat_inv, ee_pos_relative)  # (num_envs, 3)
    distance_ee2base_x = ee_pos_in_robot_frame[:, 0]

    # Calculate the penalty
    distance_penalty = 1.0 / (1.0 + torch.exp(200 * torch.abs(distance_ee2base_x - distance_threshold)))
    return distance_penalty

"""Group 2: Robot state related rewards"""
## (1) Action rates
def action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize large instantaneous changes in the network action output."""
    current_action = env.action_manager.action[:, :9] 
    prev_action = env.observation_manager.compute_group("reward_calculation")["applied_actions"][:, -1, :9]  
    return -torch.linalg.norm(current_action - prev_action, dim=1)

def action_rate2_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize large instantaneous changes in the network action output."""
    current_action = env.action_manager.action[:, :9]  
    prev_action = env.observation_manager.compute_group("reward_calculation")["applied_actions"][:, -1, :9]  
    prev_prev_action = env.observation_manager.compute_group("reward_calculation")["applied_actions"][:, 0, :9] 
    return -torch.linalg.norm(
        (current_action - 2 * prev_action + prev_prev_action), dim=1
    )
## (2) Joint movements
# The torques in the joints
def joint_torques(
    env: ManagerBasedRLEnv,
    arm_joint_names: tuple[str, ...],
    robot_name: str = "robot",
) -> torch.Tensor:
    """Penalize large joint torques."""
    robot = env.scene[robot_name]
    arm_joint_ids, _ = robot.find_joints(
            arm_joint_names
        )
    joint_torques_reward = torch.sum(torch.square(robot.data.applied_torque[:, arm_joint_ids]), dim=1)
    return -joint_torques_reward

# The accelerations in the joints
def joint_accel(
    env: ManagerBasedRLEnv,
    arm_joint_names: tuple[str, ...],
    robot_name: str = "robot",
) -> torch.Tensor:
    """Penalize large joint accelerations."""
    robot = env.scene[robot_name]
    arm_joint_ids, _ = robot.find_joints(
            arm_joint_names
        )
    joint_accel_reward = torch.sum(torch.square(robot.data.joint_acc[:, arm_joint_ids]), dim=1)
    return -joint_accel_reward

# The positions of the joints w.r.t the reference joint positions
def joint_positions_wrt_reference(
    env: ManagerBasedRLEnv,
    arm_joint_names: tuple[str, ...],
    robot_name: str = "robot",
    reference_joint_positions: dict = None
) -> torch.Tensor:
    """Penalize large deviations from reference joint positions."""
    robot = env.scene[robot_name]
    arm_joint_ids, _ = robot.find_joints(
            arm_joint_names
        )
    reference_joints_positions_tensor = torch.zeros((len(arm_joint_names)), dtype=robot.data.joint_pos.dtype, device=robot.data.joint_pos.device)
    idx = 0
    for joint_name in arm_joint_names:
        if joint_name not in reference_joint_positions:
            raise ValueError(f"Reference joint positions must be provided for all arm joints. Missing: {joint_name}")
        reference_joints_positions_tensor[idx] = reference_joint_positions[joint_name]
        idx += 1
    
    joint_diff = robot.data.joint_pos[:, arm_joint_ids] - reference_joints_positions_tensor
    joint_reference_pos_reward = torch.sum(torch.abs(joint_diff), dim=1)
    return -joint_reference_pos_reward

## (3) Undesired contacts
# Penalize the undesired contacts within the robot itself
def undesired_contact_penalty(
    env: ManagerBasedRLEnv,
    undesired_contact_body_names: list[str],
    contact_sensor_name: str = "contact_sensor",
    undesired_contact_threshold: float = 1.0
) -> torch.Tensor:
    """Penalize undesired contacts within the robot itself."""
    contact_sensor = env.scene[contact_sensor_name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    undesired_contact_body_ids, _ = contact_sensor.find_bodies(undesired_contact_body_names)
    is_undesired_contact = (
            torch.max(torch.norm(net_contact_forces[:, :, undesired_contact_body_ids], dim=-1), dim=1)[0] > undesired_contact_threshold
        )
    contact = -torch.sum(is_undesired_contact, dim=1)
    return contact



# The efforts in the joints (DEPRECATED, seems to be of little weighted importance)
# def joint_efforts(
#     env: ManagerBasedRLEnv,
#     arm_joint_names: tuple[str, ...],
#     robot_name: str = "robot",
# ) -> torch.Tensor:
#     """Penalize large joint efforts (torque * accel)."""
#     robot = env.scene[robot_name]
#     arm_joint_ids, _ = robot.find_joints(
#             arm_joint_names
#         )

