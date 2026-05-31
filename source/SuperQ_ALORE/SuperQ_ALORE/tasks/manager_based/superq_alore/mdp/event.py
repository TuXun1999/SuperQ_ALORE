# Copyright (c) 2024 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from pxr import PhysxSchema, UsdPhysics

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import sample_uniform

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

from SuperQ_ALORE.assets.object_catalog import OBJECT_CATALOG
from SuperQ_ALORE.tasks.manager_based.superq_alore.mdp import object_management


def configure_physx_scene_gpu_buffers(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    gpu_temp_buffer_capacity: int = 64 * 1024 * 1024,
    gpu_heap_capacity: int = 256 * 1024 * 1024,
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

    # Objects are spawned as per-object subsets of environments
    env_ids_cpu = env_ids.detach().cpu().tolist()
    
    # List out the active indices for each object
    obj_idx_reset = env.active_object_indices[env_ids_cpu]
    pose_idx_reset = env.active_pose_indices[env_ids_cpu]
    global_to_local_mapping = env.global_to_local_mapping
    
    # Find the env origins
    origins = env.scene.env_origins[env_ids]

    # For each object in the selected envs, find the local indices & reset the states
    for obj_id in range(len(OBJECT_CATALOG)):
        # The rows corresponding to the current object in the selected batch of envs
        selected_rows = torch.where(obj_idx_reset == obj_id)[0]
        
        # If in the selected envs, no env matched this object, skip to the next one
        if selected_rows.size == 0:
            continue
        
        # The env indices/indices of the selected object in the global pool of envs
        target_object = env.scene[f"target_object_{obj_id}"]

        global_to_local = global_to_local_mapping[f"target_object_{obj_id}"]

        # Within the selected envs, find the global env indices of the object
        global_env_ids_for_obj = [env_ids_cpu[row] for row in selected_rows.tolist()]

            
        # Find the local env indices of the object in the current batch of env_ids
        local_env_ids_list = [global_to_local[g] for g in global_env_ids_for_obj]
        local_env_ids = torch.tensor(local_env_ids_list, device=env.device, dtype=torch.long)

        # Reset the object states in the current batch of sub-envs
        target_object_state = target_object.data.default_root_state[local_env_ids].clone()

        for j, row in enumerate(selected_rows.tolist()):
            pose_entry: object_management.PoseEntry = OBJECT_CATALOG[obj_id].poses[int(pose_idx_reset[row])]
            offset = pose_entry.position[0:2]
            target_object_state[j, 0] = origins[row, 0] + offset[0]
            target_object_state[j, 1] = origins[row, 1] + offset[1]
            quat = pose_entry.orientation  # w, x, y, z
            target_object_state[j, 3:7] = torch.tensor(quat, device=target_object_state.device)

        target_object.write_root_state_to_sim(target_object_state, env_ids=local_env_ids)
    

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





