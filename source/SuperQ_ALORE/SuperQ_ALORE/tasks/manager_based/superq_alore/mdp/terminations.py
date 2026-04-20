# Copyright (c) 2024 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

"""Functions specific to the interlimb loco-manipulation environments."""

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

def outlier_detected(env: ManagerBasedRLEnv, threshold: float = 1000.0) -> torch.Tensor:
    """Terminates the environment if actions or base velocities explode."""
    
    # 1. Check for Exploding Actions
    actions = env.action_manager.action
    action_exploded = torch.any(torch.abs(actions) > threshold, dim=-1)
    action_nan = torch.any(torch.isnan(actions) | torch.isinf(actions), dim=-1)
    
    # 2. Check for Exploding Robot Base Velocities (A common symptom of physics explosions)
    root_vel = env.scene["robot"].data.root_com_vel_w
    vel_exploded = torch.any(torch.abs(root_vel) > threshold, dim=-1)
    vel_nan = torch.any(torch.isnan(root_vel) | torch.isinf(root_vel), dim=-1)

    # Combine the masks
    reset_mask = action_exploded | action_nan | vel_exploded | vel_nan
    
    return reset_mask

def illegal_ground_contact(
    env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Terminate when the contact force on the sensor exceeds the force threshold."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_forces_with_ground = contact_sensor.data.force_matrix_w[
        :, sensor_cfg.body_ids, ...
    ].squeeze()
    # check if any contact force with the ground exceeds the threshold
    return (
        torch.max(torch.norm(contact_forces_with_ground, dim=-1), dim=1)[0] > threshold
    )
    
def joint_velocity_limits(
    env: ManagerBasedRLEnv,
    max_vel: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when any of the asset's joint velocities exceeds the limit."""
    # Extract the asset from the environment
    asset = env.scene[asset_cfg.name]
    
    # Check if the absolute velocity of ANY joint exceeds the threshold
    return torch.any(torch.abs(asset.data.joint_vel) > max_vel, dim=-1)

def object_slide_off(
    env: ManagerBasedRLEnv,
    contact_sensor_name: str = "contact_forces",
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
    
    return gripper_fail # shape (num_envs, 1))