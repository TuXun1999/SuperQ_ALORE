from __future__ import annotations

from typing import TYPE_CHECKING, Callable

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
    per_object = getattr(env.cfg, "target_pose_ids_by_object", None) or {}
    if object_id in per_object:
        return tuple(per_object[object_id])
    global_filter = getattr(env.cfg, "target_pose_ids", None)
    if global_filter is None:
        return POSE_IDS_BY_OBJECT[object_id]
    return tuple(global_filter)


def ensure_catalog_state(env: ManagerBasedEnv) -> None:
    """Lazy-initialise all per-env catalog tensors on env (idempotent)."""

    # TODO: figure out why here we can modify the attribute? what if env does not have that?

    # only do the work once, even if ensure_catalog_state is called multiple times
    if hasattr(env, "_catalog_ready"):
        return

    num_objects = len(OBJECT_CATALOG)

    # MultiAssetSpawnerCfg(random_choice=False) assigns env i → assets_cfg[i % N].
    # We build assets_cfg in OBJECT_CATALOG order, so env i has OBJECT_CATALOG[i % N].
    # kind of like hash table
    env._fixed_object_indices = (
        torch.arange(env.num_envs, dtype=torch.long, device=env.device) % num_objects
    )

    # active_object_indices never changes during training
    # so the same object will always be assigned to the same env
    # TODO: think about this: should we also fix the pose for the object for each env?
    env.active_object_indices = env._fixed_object_indices.clone()

    env._allowed_pose_indices: dict[str, torch.Tensor] = {}
    for oid in OBJECT_IDS:  # iterate through all objects

        # get all possible poses for this object
        all_poses = POSE_IDS_BY_OBJECT[oid]

        # TODO: trivial filtering based on env cfg, which we do not use at all for now
        # but we may use this for deterministic testing in the future
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

    env._catalog_ready = True


def sample_target_assignments(env: ManagerBasedEnv, env_ids: torch.Tensor) -> None:
    """Randomly sample a new pose for each env in env_ids.

    The chair model is fixed for the lifetime of training (set by MultiAssetSpawnerCfg
    at spawn).  Only the pose (arm joints + chair position/orientation) is re-sampled
    each episode, always from the poses that belong to that env's fixed chair.
    """

    # call the idepmpotent catalog function to ensure all the necessary tensors are initialized
    ensure_catalog_state(env)

    # null check
    if env_ids.numel() == 0: # total number of elements in the tensor.
        return

    # read the fixed object corresponding to the env_ids to be updated
    fixed_obj_indices = env.active_object_indices[env_ids]

    # TODO: why we use torch.unique here? why not just iterate direclty?
    # iterate through each unique object type present in this reset batch
    for object_index in torch.unique(fixed_obj_indices).tolist():

        # convert the numerical object index back to the object id string
        # based on the sturcuture of OBJECT_CATALOG in object_catalog.py and how we built the 
        # fixed_object_indices tensor in ensure_catalog_state
        object_id = OBJECT_IDS[int(object_index)]

        # get tensor of allowed pose indices for this object
        pose_pool = env._allowed_pose_indices[object_id]

        # boolean mask selecting envs in the batch that share this object
        # mask shape: [num_envs_in_batch], True for envs with this object, False otherwise
        mask = fixed_obj_indices == object_index

        # sub_env_ids is the list of env ids that have the current object
        # i.e. the envs that we need to update in this iteration
        # TODO: what is the grammar here for using mask as slicing for env_ids?
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
        entries.append(OBJECT_CATALOG[obj_idx].poses[pose_idx])
    return entries

  
def get_active_pose_position_orientation_tensors(
    env: ManagerBasedEnv, env_ids: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return batched pose position/orientation tensors for env_ids from active assignments."""

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


def gather_active_object_tensor(
    env: ManagerBasedEnv, tensor_getter: Callable
) -> torch.Tensor:
    """Return the tensor from the single target_object asset (one chair per env)."""
    return tensor_getter(env.scene["target_object"])

