# Copyright (c) 2024 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

"""Functions specific to the loco-manipulation environments."""

import torch

import isaaclab.utils.math as math_utils
from isaaclab.utils.math import quat_rotate, quat_mul
from isaaclab.envs import ManagerBasedRLEnv
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as isaac_mdp

def known_external_force_torque(
    env: ManagerBasedRLEnv,
    event_name: str,
    scale: bool,
) -> torch.Tensor:
    """Known external force torque amount."""
    # event_manager is established after first call
    if not hasattr(env, "event_manager"):
        return torch.zeros((env.num_envs, 3), device=env.device)

    else:
        event_term = env.event_manager.get_term_cfg(event_name).func
        external_forces = event_term.external_forces.view(env.num_envs, -1)
        if scale:
            external_forces = math_utils.scale_transform(
                external_forces, event_term.force_range[0], event_term.force_range[1]
            )
        return external_forces


def gait_phase(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The generated command from command term in the command manager with the given name."""
    if not hasattr(env, "reward_manager"):
        return torch.zeros((env.num_envs, 1), device=env.device)
    else:
        steps = env.reward_manager.get_term_cfg("gait").func.steps
        command_leg = env.command_manager.get_term(
            "arm_leg_joint_base_pose"
        ).command_leg
        max_length = (
            command_leg
            * env.reward_manager.get_term_cfg("gait").func.three_leg_phase_len
            + (1 - command_leg.int())
            * env.reward_manager.get_term_cfg("gait").func.four_leg_phase_len
        )
        phase = steps % max_length
        phase = math_utils.scale_transform(
            phase, torch.zeros_like(max_length), max_length
        )
        return phase.view(-1, 1)
    
def last_leg_action(
    env: ManagerBasedRLEnv,
    action_term_name: str = "high_level_action",
    clip_limit: float = 100.0,
) -> torch.Tensor:
    """Previous leg joint action vector (12)."""
    high_level_action_term = env.action_manager.get_term(action_term_name)
    
    leg_act = getattr(high_level_action_term, "_raw_actions", None)
    # input("Press any key to continue")
    if leg_act is None:
        leg_act = torch.zeros((env.num_envs, 12), device=env.device)
    

    return torch.clamp(leg_act, min=-clip_limit, max=clip_limit)


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

def get_body_orientation(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    r, p, y = _euler_from_quat(env.scene["robot"].data.root_quat_w)
    body_angles = torch.stack([r, p, y], dim=-1)
    return body_angles[:, :-1]

def quat_inverse_safe(q: torch.Tensor) -> torch.Tensor:
    norm_sq = torch.sum(q * q, dim=-1, keepdim=True)  # (..., 1)
    conj = torch.cat([q[..., :1], -q[..., 1:]], dim=-1)
    return conj / norm_sq


def ee_pose_in_robot_frame(
    env: ManagerBasedRLEnv,
    robot_name: str = "robot",
    end_effector_link_name: str = "gripper_link",
) -> torch.Tensor:
    robot = env.scene[robot_name]
    
    # Find the body index for the end-effector link
    body_names = robot.find_bodies(end_effector_link_name)[0]
    body_index = robot.body_names.index(end_effector_link_name)
    # Obtain the link transforms in robot's root frame
    ee_pos_w = robot.data.body_pos_w[:, body_index]  # shape (N_envs, 3)
    ee_quat_w = robot.data.body_quat_w[:, body_index]  # shape (N_envs, 4)

    # Obtain the robot base pos & quat in world frame
    robot_base_pos = robot.data.root_pos_w  # (num_envs, 3)
    robot_quat_inv = quat_inverse_safe(robot.data.root_quat_w)  # (num_envs, 4)

    # Calculate the relative displacement
    ee_pos_relative = ee_pos_w - robot_base_pos  # (num_envs, 3)
    ee_pos_in_robot_frame = quat_rotate(robot_quat_inv, ee_pos_relative)  # (num_envs, 3)
    ee_quat_in_robot_frame = quat_mul(robot_quat_inv, ee_quat_w)  # (num_envs, 4)

    return torch.cat([ee_pos_in_robot_frame, ee_quat_in_robot_frame], dim=-1).to(env.device)  # (num_envs, 7)

def obj_pose_in_robot_frame(
    env: ManagerBasedRLEnv,
    robot_name: str = "robot",
    object_name: str = "target_object",
) -> torch.Tensor:
    robot = env.scene[robot_name]
    obj = env.scene[object_name]

    # Obtain the object pose in world frame
    obj_pos_w = obj.data.root_pos_w  # shape (N_envs, 3)
    obj_quat_w = obj.data.root_quat_w  # shape (N_envs, 4)

    # Obtain the robot base pos & quat in world frame
    robot_base_pos = robot.data.root_pos_w  # (num_envs, 3)
    robot_quat_inv = quat_inverse_safe(robot.data.root_quat_w)  # (num_envs, 4)

    # Calculate the relative displacement
    obj_pos_relative = obj_pos_w - robot_base_pos  # (num_envs, 3)
    obj_pos_in_robot_frame = quat_rotate(robot_quat_inv, obj_pos_relative)  # (num_envs, 3)
    obj_quat_in_robot_frame = quat_mul(robot_quat_inv, obj_quat_w)  # (num_envs, 4)

    return torch.cat([obj_pos_in_robot_frame, obj_quat_in_robot_frame], dim=-1).to(env.device)  # (num_envs, 7)

def link_pose_in_robot_frame(
    env: ManagerBasedRLEnv,
    robot_name: str = "robot",
    link_names: list[str] = ["arm_link_jaw"],
) -> torch.Tensor:
    robot = env.scene[robot_name]
    
    # Find the body indices for the specified links
    body_indices = [robot.body_names.index(link_name) for link_name in link_names]
    
    # Obtain the link transforms in robot's root frame
    link_pos_w = robot.data.body_pos_w[:, body_indices, :]  # shape (N_envs, num_links, 3)
    link_quat_w = robot.data.body_quat_w[:, body_indices, :]  # shape (N_envs, num_links, 4)

    # Obtain the robot base pos & quat in world frame
    robot_base_pos = robot.data.root_pos_w  # (num_envs, 3)
    robot_quat_inv = quat_inverse_safe(robot.data.root_quat_w)  # (num_envs, 4)

    # Calculate the relative displacement for each link
    robot_base_pos_expanded = robot_base_pos.unsqueeze(1).expand(-1, len(link_names), -1)  # (num_envs, num_links, 3)
    robot_quat_inv_expanded = robot_quat_inv.unsqueeze(1).expand(-1, len(link_names), -1)  # (num_envs, num_links, 4)

    link_pos_relative = link_pos_w - robot_base_pos_expanded  # (num_envs, num_links, 3)
    link_pos_in_robot_frame = quat_rotate(robot_quat_inv_expanded, link_pos_relative)  # (num_envs, num_links, 3)
    link_quat_in_robot_frame = quat_mul(robot_quat_inv_expanded, link_quat_w)  # (num_envs, num_links, 4)
    num_envs = link_pos_w.shape[0]

    return torch.cat([link_pos_in_robot_frame, link_quat_in_robot_frame], dim=-1).reshape(num_envs, -1).to(env.device) # (num_envs, num_links * 7)


def category_encode(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    # ALORE has a constant zero category code, so we just return zeros here
    return torch.zeros((env.num_envs, 3), device=env.device)

def joint_pos_rel(
    env: ManagerBasedRLEnv,
    robot_name: str = "robot",
) -> torch.Tensor:
    robot = env.scene[robot_name]
    # Ignoring the final gripper joint
    joint_pos_rel = isaac_mdp.joint_pos_rel(env)[:, :-1]  # shape (num_envs, 18)
    return joint_pos_rel

def joint_vel(
    env: ManagerBasedRLEnv,
    robot_name: str = "robot",
) -> torch.Tensor:
    robot = env.scene[robot_name]
    # Ignoring the final gripper joint
    joint_vel_rel = isaac_mdp.joint_vel(env)[:, :-1]  # shape (num_envs, 18)
    return joint_vel_rel

def joint_pos(
    env: ManagerBasedRLEnv,
    robot_name: str = "robot",
) -> torch.Tensor:
    robot = env.scene[robot_name]
    # Ignoring the final gripper joint
    joint_pos = robot.data.joint_pos[:, :-1]  # shape (num_envs, 18)
    return joint_pos

def default_joint_pos(
    env: ManagerBasedRLEnv,
    robot_name: str = "robot",
) -> torch.Tensor:
    robot = env.scene[robot_name]
    # Ignoring the final gripper joint
    default_joint_pos = robot.data.default_joint_pos[:,:-1]  # shape (num_envs, 18)
    return default_joint_pos

def last_high_level_action(
    env: ManagerBasedRLEnv,
    clip_limit: float = 100.0,
) -> torch.Tensor:
    import isaaclab_tasks.manager_based.locomotion.velocity.mdp as isaac_mdp
    # Only the first 9 dimensions are used (x, y, omega + 6 arm joints)
    high_level_act = isaac_mdp.last_action(env)[:, :9]
    
    return torch.clamp(high_level_act, min=-clip_limit, max=clip_limit)

def ee_contact_state(
    env: ManagerBasedRLEnv,
    contact_sensor_name: str = "contact_sensor",
    gripper_links_names: list[str] = ["arm_link_fngr", "arm_link_jaw"],
) -> torch.Tensor:
    # gripper fail
    start_moving_steps = (int)(1.5 / env.step_dt)
    start_mask = env.episode_length_buf > start_moving_steps

    # Obtain the readings from the contact sensor
    contact_sensor = env.scene[contact_sensor_name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    gripper_fail = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    gripper_fngr_id, _ = contact_sensor.find_bodies(gripper_links_names[0])
    gripper_jaw_ids, _ = contact_sensor.find_bodies(gripper_links_names[1])
    if start_mask.any():  # 
        gripper_fngr_contact = torch.max(torch.norm(net_contact_forces[start_mask, :, gripper_fngr_id], dim=-1), dim=1)[0] > 1.0
        gripper_jaw_connect = torch.max(torch.norm(net_contact_forces[start_mask, :, gripper_jaw_ids], dim=-1), dim=1)[0] > 1.0
        # print("gripperMover_contact", gripperMover_contact)
        gripper_fail[start_mask] = (gripper_fngr_contact == False) & (gripper_jaw_connect == False)  # 
    
    return gripper_fail.float().unsqueeze(-1)  # shape (num_envs, 1)

def obj_lin_vel_in_robot_frame(
    env: ManagerBasedRLEnv,
    object_name: str = "target_object",
) -> torch.Tensor:
    obj_lin_vel_b = env.scene[object_name].data.root_lin_vel_b  # shape (N_envs, 3)
    return obj_lin_vel_b

def obj_ang_vel_in_robot_frame(
    env: ManagerBasedRLEnv,
    object_name: str = "target_object",
) -> torch.Tensor:
    obj_ang_vel_b = env.scene[object_name].data.root_ang_vel_b  # shape (N_envs, 3)
    return obj_ang_vel_b

def obj_physical_properties(
    env: ManagerBasedRLEnv,
    object_name: str = "target_object",
) -> torch.Tensor:
    obj = env.scene[object_name]
    # Assuming the physical properties we want are mass, friction, and restitution
    mass = torch.sum(obj.root_physx_view.get_masses(), dim=1).unsqueeze(-1)  # shape (num_envs, 1)  
    static_friction = obj.root_physx_view.get_material_properties()[:, 0, 0].unsqueeze(-1)  # shape (num_envs, 1)
    dynamic_friction = obj.root_physx_view.get_material_properties()[:, 0, 1].unsqueeze(-1)  # shape (num_envs, 1)
    return torch.cat([static_friction, mass, dynamic_friction], dim=-1).to(env.device)  # shape (num_envs, 3)