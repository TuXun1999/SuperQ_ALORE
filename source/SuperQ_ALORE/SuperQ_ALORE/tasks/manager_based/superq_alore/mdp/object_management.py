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
    """Lazy-initialise all per-env catalog tensors on env (idempotent)."""

    # only do the work once, even if ensure_catalog_state is called multiple times
    # but at the first time, since _catalog_ready is not set, this "return" will be skipped
    if hasattr(env, "_catalog_ready"):
        return

    # obtatn the number of objects from the object_catalog.py that processes the YAML file.
    num_objects = len(OBJECT_CATALOG)

    # active_object_indices is sampled randomly at each reset by sample_target_assignments.
    # Initialise to zero (meaningless until the first reset call).
    env.active_object_indices = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    # initialize a dict for the env to store the allowed pose indices for each object
    env._allowed_pose_indices = {}
    for oid in OBJECT_IDS:  # iterate through all objects

        # get all possible poses for this object
        all_poses = POSE_IDS_BY_OBJECT[oid]

        # TODO: trivial filtering based on env cfg, which we do not use at all for now
        # but we may use this for deterministic testing in the future.
        # Espeically, if no filter is specified, all poses for this object will be returned.
        requested = _resolve_pose_filter(env, oid)

        # a list of allowed poses for the object
        indices = [i for i, pid in enumerate(all_poses) if pid in requested]
        if not indices:
            raise ValueError(
                f"No valid poses for '{oid}'. Requested {requested}, available {all_poses}."
            )
        
        # store allowed pose indices as a GPU tensor for fast sampling later
        env._allowed_pose_indices[oid] = torch.tensor(
            indices, dtype=torch.long, device=env.device
        )

    # initialize active pose indices, meaningless until sample_target_assignments is called
    env.active_pose_indices = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    # a flag per env indicating whether a valid pose has been sampled yet (i.e. whether active_pose_indices is meaningful)
    #  used to coordinate with the training loop (e.g. observation, rewards)
    # TODO: figure out the connection between this and `observation.py` and `reward.py`
    env.target_assignment_ready = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    # Arm joint targets [num_envs, 7], matching ARM_JOINT_NAMES_IN_ORDER
    env.active_arm_joint_reference = torch.zeros(
        (env.num_envs, len(ARM_JOINT_NAMES_IN_ORDER)),
        dtype=torch.float32, device=env.device,
    )

    # next time when the function is called, the first "if" condition will be true 
    # and the function will return immediately
    env._catalog_ready = True


def sample_target_assignments(env: ManagerBasedEnv, env_ids: torch.Tensor) -> None:
    """Randomly sample a new (object, pose) assignment for each env in env_ids.

    For each reset env, this function samples an object index uniformly from the
    catalog, then samples a valid pose for that object and updates all derived
    per-env state (pose index, arm joint reference, readiness flag).
    """

    # call the idepmpotent catalog function to ensure all the necessary tensors are initialized
    ensure_catalog_state(env)

    # null check
    if env_ids.numel() == 0: # total number of elements in the tensor.
        return

    num_objects = len(OBJECT_CATALOG)

    # assign objects to envs fairly and randomly, without any object initialization bias.
    batch_size = env_ids.numel()

    # batch size larger than # objects, we can cover all objects at least once, 
    # so we repeat the object indices as needed and then do a random permutation to assign objects to envs randomly
    if batch_size >= num_objects:

        # up-ceil division to get the number of repeats needed to cover the batch size,
        repeats = (batch_size + num_objects - 1) // num_objects

        # create a base tensor of shape [num_objects * repeats] that contains repeated object indices [0, 1, ..., num_objects-1], 
        # and then take the first batch_size elements after a random permutation to get the final assigned object index for each env 
        # in the batch
        base = torch.arange(num_objects, device=env.device).repeat(repeats)[:batch_size]
        random_obj_indices = base[torch.randperm(batch_size, device=env.device)]
    else:
        # directly use subset of objects
        random_obj_indices = torch.randperm(num_objects, device=env.device)[:batch_size]
    
    # advanced indexing to assign the sampled object indices to the envs in env_ids
    env.active_object_indices[env_ids] = random_obj_indices

    # iterate through each unique object type present in this reset batch
    for object_index in torch.unique(random_obj_indices).tolist():

        # convert the numerical object index back to the object id string
        # based on the sturcuture of OBJECT_CATALOG in object_catalog.py and how we built the 
        # fixed_object_indices tensor in ensure_catalog_state
        object_id = OBJECT_IDS[int(object_index)]

        # get tensor of allowed pose indices for this object
        pose_pool = env._allowed_pose_indices[object_id]

        # boolean mask selecting envs in the batch that share this object
        # mask shape: [num_envs_in_batch], True for envs with this object, False otherwise
        mask = random_obj_indices == object_index

        # sub_env_ids is the list of env ids that have the current object
        # i.e. the envs that we need to update in this iteration
        # Boolean indexing to get the env ids in env_ids that have the current object
        sub_env_ids = env_ids[mask]

        # randonmly sample a pose index from the allowed pool
        # shape: (sub_env_ids.numel(),) i.e. one pose index per env in this batch that has the current object
        # each element is sampled uniformly from [0, len(pose_pool) - 1]
        pslot = torch.randint(len(pose_pool), (sub_env_ids.numel(),), device=env.device) 

        # obtain the sampled pose from the sampled index
        chosen_pose_indices = pose_pool[pslot]

        # stores chose pose index for each env in this iteration to the env.active_pose_indices tensor
        # so that we update the pose index for the envs with the current object
        env.active_pose_indices[sub_env_ids] = chosen_pose_indices

        # obtain the PoseEntry for each env that share the same object
        poses = OBJECT_CATALOG[int(object_index)].poses

        # iterate through each env with the current object
        for local_i, env_id in enumerate(sub_env_ids.tolist()):

            # looks up the randomly sampled pose for the current envrionment 
            pose = poses[int(chosen_pose_indices[local_i].item())]

            # update the pre-grasping joint angles for the current env based on the sampled pose's
            # joint configuration corresponding the sampled pose.
            env.active_arm_joint_reference[env_id] = torch.tensor(
                [pose.joint_positions[j] for j in ARM_JOINT_NAMES_IN_ORDER],
                dtype=torch.float32, device=env.device,
            )

    env.target_assignment_ready[env_ids] = True


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
    objects, then selects the row corresponding to each env's ``active_object_indices``.
    Works for any scalar/vector attribute stored in ``RigidObjectData`` (e.g.
    ``root_pos_w``, ``root_quat_w``, ``root_lin_vel_b``, ``projected_gravity_b``, etc.).
    """
    ensure_catalog_state(env)
    n_catalog = len(OBJECT_CATALOG)

    # collect the requested attribute from all objects into a list of tensors,
    # tensor: a python list consists of N_catalog tensors, [num_envs, attr_dim] for each tensor.
    tensors = [
        getattr(env.scene[f"target_object_{i}"].data, attr_name)
        for i in range(n_catalog)
    ]
    # stacked: [N_catalog, num_envs, attr_dim]
    stacked = torch.stack(tensors, dim=0)

    # corresponding object index for each env in env_ids, shape: [num_envs]
    idx = env.active_object_indices         
    arange = torch.arange(env.num_envs, device=env.device)

    # the same as: stacked[idx, arange, :]
    return stacked[idx, arange]    # shape: [num_envs, attr_dim]    


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
    stacked = torch.stack(tensors, dim=0)  # [N_catalog, num_envs, 1]
    idx = env.active_object_indices.to(stacked.device)
    env_ids = torch.arange(env.num_envs, device=stacked.device)
    return stacked[idx, env_ids]  # [num_envs, 1]


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
    stacked = torch.stack(tensors, dim=0)  # [N_catalog, num_envs, 3]
    idx = env.active_object_indices.to(stacked.device)
    env_ids = torch.arange(env.num_envs, device=stacked.device)
    return stacked[idx, env_ids]  # [num_envs, 3]

