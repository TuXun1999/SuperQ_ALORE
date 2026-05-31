from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from SuperQ_ALORE.assets.object_catalog import (
    ARM_JOINT_NAMES_IN_ORDER,
    OBJECT_CATALOG,
    OBJECT_IDS,
    POSE_IDS_BY_OBJECT,
    PoseEntry,
)
from SuperQ_ALORE.tasks.manager_based.superq_alore.mdp.scene import \
    OBJECT_IDX_ENVS, POSE_IDX_LOCAL_ENVS, GRASP_POSE_JOINT_POSITIONS
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

# TODO: this function is maybe useless, remove it if possible to prevent any confusion
# but maybe this function is useful if we want to have deterministic initialisation of the target assignment in some cases (e.g. for testing)
def _resolve_pose_filter(env: ManagerBasedEnv, object_id: str) -> tuple[str, ...]:
    """Return the allowed pose ids for object_id given env.cfg filters."""

    # first check if there is a per-object filter in the env cfg, which is more specific and has higher priority than the global filter
    per_object = getattr(env.cfg, "target_pose_ids_by_object", None) or {}
    if object_id in per_object:

        # if there is a per-object filter, return it directly (after converting to tuple if it's a list)
        return tuple(per_object[object_id])
    
    # if there is no per-object filter, check the global filter, which applies to all objects but has lower priority than the per-object filter
    global_filter = getattr(env.cfg, "target_pose_ids", None)
    if global_filter is None:

        # if there is no global filter either, return all poses for this object from the catalog
        return POSE_IDS_BY_OBJECT[object_id]
    return tuple(global_filter)


def ensure_catalog_state(env: ManagerBasedEnv) -> None:
    """Initialise all per-env catalog tensors on env."""

    # only do the work once, even if ensure_catalog_state is called multiple times
    # but at the first time, since _catalog_ready is not set, this "return" will be skipped
    if hasattr(env, "_catalog_ready"):
        return

    # obtain the number of objects from the object_catalog.py that processes the YAML file.
    num_objects = len(OBJECT_CATALOG)

    # active_object_indices is the idx of the assigned object in each sub-env
    env.active_object_indices = torch.tensor(OBJECT_IDX_ENVS, dtype=torch.long, device=env.device)

    # initialize active pose indices, which is the idx of the assigned pose within the assigned object in each sub-env
    env.active_pose_indices = torch.tensor(POSE_IDX_LOCAL_ENVS, dtype=torch.long, device=env.device)

    # Arm joint targets [num_envs, 7], matching ARM_JOINT_NAMES_IN_ORDER
    env.active_arm_joint_reference = torch.tensor(GRASP_POSE_JOINT_POSITIONS, dtype=torch.float32, device=env.device)

    # Build global->local index mapping for this object view.
    env.global_to_local_mapping = {}
    for obj_id in range(num_objects):
        all_envs_for_obj = torch.where(env.active_object_indices == obj_id)[0]
        
        # For example, there are two objects and 20 envs
        # object 1 is at 11th sub-env, 
        # then its global index is 11, but its local index is 1 (the first 10 envs belong to object 0)
        global_to_local = {int(g): int(i) for i, g in enumerate(all_envs_for_obj.tolist())}
        env.global_to_local_mapping[f"target_object_{obj_id}"] = global_to_local
    
    # next time when the function is called, the first "if" condition will be true 
    # and the function will return immediately
    env._catalog_ready = True


def get_active_pose_entries(
    env: ManagerBasedEnv, env_ids: torch.Tensor
) -> list[PoseEntry]:
    """Return the PoseEntry currently assigned to each env in env_ids."""
    ensure_catalog_state(env)
    entries: list[PoseEntry] = []

    # iterate through each env id in the input tensor
    for env_id in env_ids.tolist():
        obj_idx = int(env.active_object_indices[env_id].item())

        # get the pose object corresponding to the env_id 
        # based on the active_object_indices and active_pose_indices tensors
        pose_idx = int(env.active_pose_indices[env_id].item())

        # a list of pose records, one per env
        entries.append(OBJECT_CATALOG[obj_idx].poses[pose_idx])
    return entries


def get_active_pose_position_orientation_tensors(
    env: ManagerBasedEnv, env_ids: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """This function converts the catalog selection state into sim-ready tensors. But it is not real-time pose."""

    # call the idepmpotent catalog function to ensure all the necessary tensors are initialized
    ensure_catalog_state(env)

    if env_ids.numel() == 0:
        raise ValueError("env_ids must not be empty.")

    # obtain the fixed object indices for the input env_ids, 
    # which will be used to lookup the assigned pose for each env
    fixed_obj_indices = env.active_object_indices[env_ids]
    pos = torch.empty((env_ids.numel(), 3), dtype=torch.float32, device=env.device)
    rot = torch.empty((env_ids.numel(), 4), dtype=torch.float32, device=env.device)

    # iterate through each unique object type present in this batch of env_ids, 
    # and update the position and orientation tensors for the envs with the same object together (to save some computation)
    for object_index in torch.unique(fixed_obj_indices).tolist():

        # mask: [num_envs_in_batch], True for envs with this object, False otherwise
        mask = fixed_obj_indices == object_index

        # sub_env_ids: [num_envs_with_this_object], the env ids that have the current object
        sub_env_ids = env_ids[mask]

        # chosen_pose_indices: [num_envs_with_this_object], the sampled pose index for each env with the current object
        chosen_pose_indices = env.active_pose_indices[sub_env_ids]

        # obtain the pose entries for the current object, 
        # which will be used to lookup the position and orientation 
        # corresponding to the sampled pose index for each env with the current object
        poses = OBJECT_CATALOG[int(object_index)].poses

        # only updates the rows in the pos tensor where the mask is True
        pos[mask] = torch.tensor(
            # iterates through the tensor of pose IDs sampled for this specifc object batch
            [poses[int(pose_idx.item())].position for pose_idx in chosen_pose_indices],
            dtype=torch.float32,
            device=env.device,
        )
        rot[mask] = torch.tensor(
            [poses[int(pose_idx.item())].orientation for pose_idx in chosen_pose_indices],
            dtype=torch.float32,
            device=env.device,
        )

    return pos, rot


def get_active_arm_joint_reference(
    env: ManagerBasedEnv, env_ids: torch.Tensor
) -> torch.Tensor:
    """Return batched active arm joint references [len(env_ids), num_arm_joints]."""
    ensure_catalog_state(env)
    return env.active_arm_joint_reference[env_ids]


def get_active_object_state_attr(
    env: ManagerBasedEnv, attr_name: str
) -> torch.Tensor:
    """Return per-env the requested data attribute from each env's active target object.
    We can use this function to obtain the real-time state of the active object.

    Stacks ``env.scene[f'target_object_{i}'].data.<attr_name>`` for all catalog
    objects, assuming the objects are initialized in the same order as the catalog
    Works for any scalar/vector attribute stored in ``RigidObjectData`` (e.g.
    ``root_pos_w``, ``root_quat_w``, ``root_lin_vel_b``, ``projected_gravity_b``, etc.).
    """
    ensure_catalog_state(env)
    n_catalog = len(OBJECT_CATALOG)

    # Collect the requested attribute from all objects into a list of tensors,
    # NOTE: The envs are already initialized in the order of i, so just stack them directly
    tensors = [
        getattr(env.scene[f"target_object_{i}"].data, attr_name)
        for i in range(n_catalog)
    ]
    # stacked: [num_envs, attr_dim]
    stacked = torch.cat(tensors, dim=0)

    return stacked    # shape: [num_envs, attr_dim]    


def get_active_object_physx_masses(env: ManagerBasedEnv) -> torch.Tensor:
    """Return the scalar total mass for each env's active target object.

    Shape: ``[num_envs, 1]``.
    """
    ensure_catalog_state(env)
    n_catalog = len(OBJECT_CATALOG)
    tensors = [
        torch.sum(env.scene[f"target_object_{i}"].root_physx_view.get_masses(), dim=1).unsqueeze(-1)
        for i in range(n_catalog)
    ]
    stacked = torch.cat(tensors, dim=0)  # [num_envs, 1]
    
    return stacked  # [num_envs, 1]


def get_active_object_physx_material_properties(env: ManagerBasedEnv) -> torch.Tensor:
    """Return the PhysX material properties (static friction, dynamic friction, restitution)
    for the first shape of each env's active target object.

    Shape: ``[num_envs, 3]``.
    """
    ensure_catalog_state(env)
    n_catalog = len(OBJECT_CATALOG)
    tensors = [
        env.scene[f"target_object_{i}"].root_physx_view.get_material_properties()[:, 0, :]
        for i in range(n_catalog)
    ]  # each: [num_envs, 3]
    stacked = torch.cat(tensors, dim=0)  # [num_envs, 3]
    return stacked  # [num_envs, 3]

