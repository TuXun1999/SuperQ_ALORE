# Copyright (c) 2024 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from pxr import PhysxSchema, UsdPhysics
import isaaclab.sim as sim_utils

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import sample_uniform
from isaaclab.utils.math import quat_apply, quat_mul
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

from SuperQ_ALORE.assets.object_catalog import OBJECT_CATALOG
from SuperQ_ALORE.tasks.manager_based.superq_alore.mdp import object_management


def configure_physx_scene_gpu_buffers(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    gpu_temp_buffer_capacity: int = 64 * 1024 * 1024,
    gpu_heap_capacity: int = 256 * 1024 * 1024,
    gpu_found_lost_pairs_capacity: int = 4_194_304,
    gpu_max_rigid_patch_count: int = 1_048_576,
) -> None:
    """Apply PhysX GPU capacities on the live PhysicsScene prim at startup."""
    del env_ids  # Unused for startup event hooks.
    stage = getattr(env.sim, "stage", None)
    if stage is None:
        return

    physics_scene_prim = None
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.Scene):
            physics_scene_prim = prim
            break
    if physics_scene_prim is None:
        return

    physx_scene_api = PhysxSchema.PhysxSceneAPI.Apply(physics_scene_prim)

    temp_attr = physx_scene_api.GetGpuTempBufferCapacityAttr()
    if not temp_attr or not temp_attr.IsValid():
        temp_attr = physx_scene_api.CreateGpuTempBufferCapacityAttr()
    temp_attr.Set(int(gpu_temp_buffer_capacity))

    heap_attr = physx_scene_api.GetGpuHeapCapacityAttr()
    if not heap_attr or not heap_attr.IsValid():
        heap_attr = physx_scene_api.CreateGpuHeapCapacityAttr()
    heap_attr.Set(int(gpu_heap_capacity))

    found_lost_attr = physx_scene_api.GetGpuFoundLostPairsCapacityAttr()
    if not found_lost_attr or not found_lost_attr.IsValid():
        found_lost_attr = physx_scene_api.CreateGpuFoundLostPairsCapacityAttr()
    found_lost_attr.Set(int(gpu_found_lost_pairs_capacity))

    patch_attr = physx_scene_api.GetGpuMaxRigidPatchCountAttr()
    if not patch_attr or not patch_attr.IsValid():
        patch_attr = physx_scene_api.CreateGpuMaxRigidPatchCountAttr()
    patch_attr.Set(int(gpu_max_rigid_patch_count))

def resample_goal_region_on_reset(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    goal_term_name: str = "goal_pose",
) -> None:
    """
    Explicitly resample goal command for reset envs.
    """

    goal_term = env.command_manager.get_term(goal_term_name)

    # Explicitly resample the goal pose for the target object
    if hasattr(goal_term, "resample_on_reset"):
        goal_term.resample_on_reset(env_ids)
        return
    if hasattr(goal_term, "_resample_command"):
        goal_term._resample_command(env_ids)
        return

    raise RuntimeError(f"Unable to resample goal term from candidates: {term_candidates}")


# (DEPRECATED) Will cause observations to fail
# def disable_inactive_object_collisions(
#     env: ManagerBasedEnv,
#     env_ids: torch.Tensor,
# ) -> None:
#     """Disable collisions for object instances that are inactive in each env.

#     This is intended to run at startup once after scene creation.
#     """

#     if hasattr(env, "_inactive_object_collisions_disabled"):
#         return

#     object_management.ensure_catalog_state(env)
#     disable_collision_cfg = sim_utils.CollisionPropertiesCfg(collision_enabled=False)

#     active_object_indices = env.active_object_indices.detach().cpu()
#     num_envs = int(env.num_envs)

#     for obj_id in range(len(OBJECT_CATALOG)):
#         target_object = env.scene[f"target_object_{obj_id}"]
#         prim_paths = list(target_object.root_physx_view.prim_paths)
#         print(len(prim_paths))
#         input("Press to continue...")
#         # root_physx_view.prim_paths is expected to be aligned with env index ordering.
#         max_envs = min(num_envs, len(prim_paths))
#         for env_idx in range(max_envs):
#             if int(active_object_indices[env_idx].item()) != obj_id:
#                 sim_utils.modify_collision_properties(prim_paths[env_idx], disable_collision_cfg)
#                 print(env_idx)
#         input("Press to continue...")
#     env._inactive_object_collisions_disabled = True

def reset_target_object_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_name: str = "target_object",
    offset: tuple[float, float] = (0.0, 0.0),
    rotation: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
):
    # target objects to manipulate
    target_object = env.scene[asset_name]
    target_object_state = env.scene[asset_name].data.default_root_state[env_ids].clone()
    origins = env.scene.env_origins[env_ids]
    target_object_state[:, 0] = origins[:, 0] + offset[0]
    target_object_state[:, 1] = origins[:, 1] + offset[1]
    target_object_state[:, 3:7] = torch.tensor(rotation, device=target_object_state.device)
    target_object.write_root_state_to_sim(target_object_state, env_ids=env_ids)


def reset_object_and_robot_from_catalog_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    position_range: tuple[float, float],
    velocity_range: tuple[float, float],
) -> None:
    """
    Reset the object pose & the grasp pose on the object automatically 
    based on the catalog configuration for each env, and add some noise 
    to the grasp pose by sampling around it with the given position and velocity ranges.
    """
    object_management.ensure_catalog_state(env)

    # Objects are spawned in all environments with per-env activation.
    env_ids_cpu = env_ids.detach().cpu().tolist()
    
    # List out the active indices for each object
    # N
    obj_idx_reset = env.active_object_indices[env_ids_cpu]
    pose_idx_reset = env.active_pose_indices[env_ids_cpu]
    # Find the env origins
    origins = env.scene.env_origins[env_ids]

    # For each object in the selected envs, find the local indices & reset the states
    for obj_id in range(len(OBJECT_CATALOG)):
        # The rows corresponding to the current object in the selected batch of envs
        selected_rows = torch.where(obj_idx_reset == obj_id)[0]
        
        # If in the selected envs, no env matched this object, skip to the next one
        if selected_rows.numel() == 0:
            continue
        
        target_object = env.scene[f"target_object_{obj_id}"]
        active_env_ids_for_obj = env_ids[selected_rows]

        # Reset the object states in the current batch of sub-envs
        target_object_state = target_object.data.default_root_state[active_env_ids_for_obj].clone()

        for j, row in enumerate(selected_rows.tolist()):
            pose_entry: object_management.PoseEntry = OBJECT_CATALOG[obj_id].poses[int(pose_idx_reset[row])]
            offset = pose_entry.position[0:2]
            target_object_state[j, 0] = origins[row, 0] + offset[0]
            target_object_state[j, 1] = origins[row, 1] + offset[1]
            quat = pose_entry.orientation  # w, x, y, z
            target_object_state[j, 3:7] = torch.tensor(quat, device=target_object_state.device)

        target_object.write_root_state_to_sim(target_object_state, env_ids=active_env_ids_for_obj)

        # Move non-active instances of this object away so each env has one active object.
        inactive_rows = torch.where(obj_idx_reset != obj_id)[0]
        if inactive_rows.numel() > 0:
            inactive_env_ids_for_obj = env_ids[inactive_rows]
            inactive_state = target_object.data.default_root_state[inactive_env_ids_for_obj].clone()
            # Keep inactive objects well outside the task workspace and high enough
            # to avoid any ground contact within normal episode horizons.
            inactive_state[:, 0] = origins[inactive_rows, 0] + 50000.0 + float(obj_id) * 20.0
            inactive_state[:, 1] = origins[inactive_rows, 1] + 50000.0
 
            inactive_state[:, 7:13] = 0.0
            target_object.write_root_state_to_sim(inactive_state, env_ids=inactive_env_ids_for_obj)
    

    # reset robot joints using the sampled pose-specific joint references
    arm_joint_ref = object_management.get_active_arm_joint_reference(env, env_ids)
    for local_i, env_id in enumerate(env_ids.tolist()):

        # set the robot joints around the grasp pose with some noise, 
        # where the reference joint position is obtained from the sampled catalog pose for this env
        reset_joints_around_grasp_pose(
            env=env,
            env_ids=torch.tensor([env_id], dtype=torch.long, device=env.device),
            position_range=position_range,
            velocity_range=velocity_range,
            joint_position_ref={
                # dictionary comprehension: {key_expression: value_expression for item in iterable}
                joint_name: float(arm_joint_ref[local_i, joint_i].item())
                for joint_i, joint_name in enumerate(object_management.ARM_JOINT_NAMES_IN_ORDER)
            },
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

# Reset the robot at different initial conditions based on different grasp poses
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


def reset_object_physical_properties(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    mass_range: tuple[float, float],
    friction_range: tuple[float, float],
    com_range: dict[str, tuple[float, float]] | None = None,
    num_buckets: int = 64
) -> None:
    """
    Reset the physical properties of the object by sampling from the given ranges.
    """
    object_management.ensure_catalog_state(env)

    # Objects are spawned in all environments with per-env activation.
    env_ids_cpu = env_ids.detach().cpu().tolist()
    
    # List out the active indices for each object
    obj_idx_reset = env.active_object_indices[env_ids_cpu]
    
    # For each object in the selected envs, find the local indices & reset the states
    for obj_id in range(len(OBJECT_CATALOG)):
        # The rows corresponding to the current object in the selected batch of envs
        selected_rows = torch.where(obj_idx_reset == obj_id)[0]
        
        # If in the selected envs, no env matched this object, skip to the next one
        if selected_rows.numel() == 0:
            continue
        
        target_object = env.scene[f"target_object_{obj_id}"]
        local_env_ids = env_ids[selected_rows].to(device="cpu", dtype=torch.long)

        # Sample mass and friction values from the given ranges
        mass_values = sample_uniform(
            torch.tensor(mass_range[0], device="cpu"),
            torch.tensor(mass_range[1], device="cpu"),
            (len(local_env_ids), target_object.num_bodies),
            device="cpu",
        )

        
        # For simplicity, we assume the static & dynamic friction coefficients are the same and sample one value for both
        static_friction_range = (friction_range[0], friction_range[1])
        dynamic_friction_range = (friction_range[0], friction_range[1])
        restitution_range = (0.0, 0.0) # No restitution for the target object
        range_list = [static_friction_range, dynamic_friction_range, restitution_range]
        ranges = torch.tensor(range_list, device="cpu")
        materials = sample_uniform(ranges[:, 0], ranges[:, 1], (num_buckets, 3), device="cpu")
        
        # Wrap up the material properties
        total_num_shapes = target_object.root_physx_view.max_shapes
        materials_idx = torch.randint(0, num_buckets, (len(local_env_ids), total_num_shapes), device="cpu")
        
        mass = target_object.root_physx_view.get_masses().clone()
        mass[local_env_ids] = mass_values
        material_properties = target_object.root_physx_view.get_material_properties().clone()
        material_properties[local_env_ids] = materials[materials_idx]

        # Randomize CoM offsets for active envs of this object.
        com_values = target_object.root_physx_view.get_coms().clone()
        if com_range is None:
            com_ranges = torch.zeros((3, 2), device="cpu")
        else:
            com_ranges = torch.tensor(
                [com_range.get(axis, (0.0, 0.0)) for axis in ["x", "y", "z"]],
                device="cpu",
            )
        com_offsets = sample_uniform(
            com_ranges[:, 0],
            com_ranges[:, 1],
            (len(local_env_ids), 3),
            device="cpu",
        )
        if com_values.ndim == 2:
            com_values[local_env_ids, :3] = com_offsets
        else:
            com_values[local_env_ids, ..., :3] = com_offsets.unsqueeze(1)

        # Set the sampled mass and friction values into the physics simulation for the current batch of sub-envs
        # NOTE: To avoid the issue of squeeze()
        target_object.root_physx_view.set_masses(mass, torch.arange(mass.shape[0]))
        
        target_object.root_physx_view.set_material_properties(material_properties, torch.arange(material_properties.shape[0]))
        target_object.root_physx_view.set_coms(com_values, torch.arange(com_values.shape[0]))
        
def reset_object_physical_properties_grasp_ranking(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    mass_range: tuple[float, float],
    friction_range: tuple[float, float],
    com_range: dict[str, tuple[float, float]] | None = None,
    num_buckets: int = 64
) -> None:
    """
    Reset the physical properties of the object by sampling from the given ranges.
    Simpler version to reset the single object in the grasp ranking
    """
    object_management.ensure_catalog_state_grasp_ranking(env)

    # Objects are spawned in all environments with per-env activation.
    env_ids_cpu = env_ids.detach().cpu().tolist()
    
    # List out the active indices for each object
    obj_idx_reset = env.active_object_indices[env_ids_cpu]
    
    # For each object in the selected envs, find the local indices & reset the states
    for obj_id in range(len(OBJECT_CATALOG)):
        # The rows corresponding to the current object in the selected batch of envs
        selected_rows = torch.where(obj_idx_reset == obj_id)[0]
        
        # If in the selected envs, no env matched this object, skip to the next one
        if selected_rows.numel() == 0:
            continue
        
        target_object = env.scene[f"target_object_{obj_id}"]
        local_env_ids = env_ids[selected_rows].to(device="cpu", dtype=torch.long)

        # Sample mass and friction values from the given ranges
        mass_values = sample_uniform(
            torch.tensor(mass_range[0], device="cpu"),
            torch.tensor(mass_range[1], device="cpu"),
            (len(local_env_ids), target_object.num_bodies),
            device="cpu",
        )

        
        # For simplicity, we assume the static & dynamic friction coefficients are the same and sample one value for both
        static_friction_range = (friction_range[0], friction_range[1])
        dynamic_friction_range = (friction_range[0], friction_range[1])
        restitution_range = (0.0, 0.0) # No restitution for the target object
        range_list = [static_friction_range, dynamic_friction_range, restitution_range]
        ranges = torch.tensor(range_list, device="cpu")
        materials = sample_uniform(ranges[:, 0], ranges[:, 1], (num_buckets, 3), device="cpu")
        
        # Wrap up the material properties
        total_num_shapes = target_object.root_physx_view.max_shapes
        materials_idx = torch.randint(0, num_buckets, (len(local_env_ids), total_num_shapes), device="cpu")
        
        mass = target_object.root_physx_view.get_masses().clone()
        mass[local_env_ids] = mass_values
        material_properties = target_object.root_physx_view.get_material_properties().clone()
        material_properties[local_env_ids] = materials[materials_idx]

        # Randomize CoM offsets for active envs of this object.
        com_values = target_object.root_physx_view.get_coms().clone()
        if com_range is None:
            com_ranges = torch.zeros((3, 2), device="cpu")
        else:
            com_ranges = torch.tensor(
                [com_range.get(axis, (0.0, 0.0)) for axis in ["x", "y", "z"]],
                device="cpu",
            )
        com_offsets = sample_uniform(
            com_ranges[:, 0],
            com_ranges[:, 1],
            (len(local_env_ids), 3),
            device="cpu",
        )
        if com_values.ndim == 2:
            com_values[local_env_ids, :3] = com_offsets
        else:
            com_values[local_env_ids, ..., :3] = com_offsets.unsqueeze(1)

        # Set the sampled mass and friction values into the physics simulation for the current batch of sub-envs
        # NOTE: To avoid the issue of squeeze()
        target_object.root_physx_view.set_masses(mass, torch.arange(mass.shape[0]))
        
        target_object.root_physx_view.set_material_properties(material_properties, torch.arange(material_properties.shape[0]))
        target_object.root_physx_view.set_coms(com_values, torch.arange(com_values.shape[0]))
        
def quat_inverse_safe(q: torch.Tensor) -> torch.Tensor:
    norm_sq = torch.sum(q * q, dim=-1, keepdim=True)  # (..., 1)
    conj = torch.cat([q[..., :1], -q[..., 1:]], dim=-1)
    return conj / norm_sq

def reset_object_robot_pose_grasp_ranking(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    object_idx: int = 0,
    pose_idx: int = 0,
) -> None:
    """
    Reset the object pose & the robot pose for grasp pose ranking purpose
    """
    # Preset the object & pose indices
    object_management.ensure_catalog_state_grasp_ranking(env, object_idx=object_idx, pose_idx=pose_idx)
    # The object is fixed at the origin
    target_object = env.scene["target_object_0"]
    target_object.write_root_state_to_sim(
        torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
        device=target_object.data.default_root_state.device).repeat(len(env_ids), 1),
        env_ids=env_ids
    )
    # Hard-coded initial robot pose in world frame during training
    robot_init_pos = (-1.0, 0.0, 0.515)
    robot_init_quat = (1.0, 0.0, 0.0, 0.0)
    
    num_envs = len(env_ids)
    robot_base_pos_init = torch.tensor(robot_init_pos, device=env.device).unsqueeze(0).repeat(num_envs, 1)
    robot_base_quat_init = torch.tensor(robot_init_quat, device=env.device).unsqueeze(0).repeat(num_envs, 1)
    # Initialized object initial pose in world frame at training
    obj_init_pose = OBJECT_CATALOG[object_idx].poses[pose_idx]
    obj_pos_w_init = torch.tensor(obj_init_pose.position, device=env.device).unsqueeze(0).repeat(num_envs, 1)
    obj_quat_w_init = torch.tensor(obj_init_pose.orientation, device=env.device).unsqueeze(0).repeat(num_envs, 1)
    
    obj_quat_inv = quat_inverse_safe(obj_quat_w_init)

    
    # Compute the robot pose in object frame (same as world frame)
    robot_pos_relative = robot_base_pos_init - obj_pos_w_init
    robot_pos_in_obj_frame = quat_apply(obj_quat_inv, robot_pos_relative)
    robot_quat_in_obj_frame = quat_mul(obj_quat_inv, robot_base_quat_init)
    
    # Set the robot pose in world frame (same as object frame)
    robot = env.scene["robot"]
    robot.write_root_state_to_sim(
        torch.cat([robot_pos_in_obj_frame, robot_quat_in_obj_frame, torch.zeros(num_envs, 6, device=env.device)], dim=-1),
        env_ids=env_ids
    )
    
    # Set the robot arm joints
    arm_joint_ref = obj_init_pose.joint_positions
    reset_joints_around_grasp_pose(
        env,
        env_ids,
        position_range = (-0.0, 0.0),
        velocity_range = (-0.0, 0.0),
        joint_position_ref = {
                # dictionary comprehension: {key_expression: value_expression for item in iterable}
                joint_name: arm_joint_ref[joint_name]
                for joint_name in object_management.ARM_JOINT_NAMES_IN_ORDER
            },
        asset_cfg = SceneEntityCfg("robot"),
    )