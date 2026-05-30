# enable postponed evaluation of type annotations.
from __future__ import annotations

import torch

import isaaclab.utils.math as math_utils

from SuperQ_ALORE.assets.object_catalog import OBJECT_CATALOG
from . import object_management as object_mgmt

# defines the public API of this module, anything not listed here is considered private and should not be used outside of this module
__all__ = [
    "pushable_keypoints_w",
    "goal_keypoints_w",
    "keypoint_mean_distance",
    "keypoint_yaw_error_deg_xy",
]


def pushable_keypoints_w(env) -> torch.Tensor:
    """Return world-frame keypoints for each env's active object."""

    # obtain keypoints in the object's own local frame 
    local_kps = _get_pushable_local_keypoints(env)

    # obtain the active object's world pose
    pos_w = object_mgmt.get_active_object_state_attr(env, "root_pos_w")
    quat_w = object_mgmt.get_active_object_state_attr(env, "root_quat_w")

    # transform the local keypoints to world frame using the object's pose
    keypoints_w = _transform_points(local_kps, pos_w, quat_w)

    # Debug print for validating per-env keypoint handling during tests.
    # active_indices = env.active_object_indices.detach().cpu().tolist()
    # local_kps_cpu = local_kps.detach().cpu()
    # keypoints_cpu = keypoints_w.detach().cpu()
    # pos_w_cpu = pos_w.detach().cpu()
    # quat_w_cpu = quat_w.detach().cpu()
    # for env_idx, object_idx in enumerate(active_indices):
    #     local_z = local_kps_cpu[env_idx, :, 2]
    #     world_z = keypoints_cpu[env_idx, :, 2]
    #     print(
    #         f"[DEBUG][pushable_keypoints_w] env={env_idx} active_object={object_idx} "
    #         f"root_pos_w={pos_w_cpu[env_idx].tolist()} "
    #         f"root_quat_w={quat_w_cpu[env_idx].tolist()} "
    #         f"local_z_range=({float(local_z.min()):.6f}, {float(local_z.max()):.6f}) "
    #         f"world_z_range=({float(world_z.min()):.6f}, {float(world_z.max()):.6f}) "
    #         f"keypoints_w={keypoints_cpu[env_idx].tolist()}"
    #     )

    return keypoints_w


def goal_keypoints_w(env, goal_term_name: str = "goal_pose") -> torch.Tensor:
    """Return world-frame keypoints for the goal marker using the active object geometry."""

    # obtain keypoints in the goal's local frame by indexing the corresponding active catalog object
    local_kps = _get_pushable_local_keypoints(env)

    # obtain the goal pose from the specified command term, 
    # and use it to transform the local keypoints to world frame
    term = env.command_manager.get_term(goal_term_name)
    pos_w = term.goal_w if hasattr(term, "goal_w") else term.command[:, :3]

    # if the command term does not have a goal_quat_w attribute, assume identity quaternion
    # but we should ensure that our command term has a consistent API and includes the goal_quat_w attribute to avoid this ambiguity
    if hasattr(term, "goal_quat_w"):
        quat_w = term.goal_quat_w
    else:
        quat_w = torch.zeros((pos_w.shape[0], 4), device=pos_w.device, dtype=pos_w.dtype)
        quat_w[:, 0] = 1.0

    # transform the local keypoints to world frame using the goal pose
    return _transform_points(local_kps, pos_w, quat_w)


def _get_pushable_local_keypoints(env) -> torch.Tensor:
    """Return per-env local keypoints by indexing the active catalog object."""

    # ensure that the catalog state is up to date before accessing the keypoints
    object_mgmt.ensure_catalog_state(env)

    # obtain the local keypoints for catalog objects
    object_kps = _get_catalog_object_local_keypoints(env)

    # a 1D tensor consists of integers representing the specific active object 
    active_indices = env.active_object_indices.to(device=object_kps.device)

    # obain the local keypoints at the active indices, resulting in a tensor of shape (num_envs, num_keypoints, 3)
    # using advanced indexing
    return object_kps[active_indices]


def _get_catalog_object_local_keypoints(env) -> torch.Tensor:
    '''This function should return a tensor containing local keypoints for each catalog object.'''
    cached = getattr(env, "_catalog_object_keypoints_local", None)

    if isinstance(cached, torch.Tensor) and cached.shape == (len(OBJECT_CATALOG), 8, 3): # 8: keypoints, 3: xyz
        return cached

    device = env.device
    object_kps = torch.zeros((len(OBJECT_CATALOG), 8, 3), device=device, dtype=torch.float32)

    # precomputes a fallback keypoint set, which is the 8 corners of a 1m^3 cube centered at the origin, 
    # in case we fail to obtain the actual keypoints from the USD stage for any reason 
    # (e.g. missing prim, invalid prim, non-mesh prim, failure in computing bounds, etc.)
    fallback = _bounds_corner_keypoints((-0.5, -0.5, -0.5), (0.5, 0.5, 0.5), device)

    # try to import USD helpers for geometry bounds queries.
    try:
        from pxr import Usd, UsdGeom
    except Exception:
        # if we fail to import the USD helpers, we won't be able to query the actual object geometry for keypoint generation,
        # so we return the fallback keypoints for all catalog objects.
        object_kps[:] = fallback

        # cache the fallback keypoints in the env and return immediately
        env._catalog_object_keypoints_local = object_kps
        return object_kps

    # Build bounds from each object's source USD asset (canonical space), not from live scene instances.
    for object_idx, object_entry in enumerate(OBJECT_CATALOG):
        min_v, max_v = _asset_local_bounds(object_entry.asset_path, Usd, UsdGeom)
        if min_v is None or max_v is None:
            object_kps[object_idx] = fallback
        else:
            # In this project all pushables are mesh USDs, so use the 8 corners of the local bounds.
            object_kps[object_idx] = _bounds_corner_keypoints(min_v, max_v, device)

    # shape: (num_objects, num_keypoints, 3)
    env._catalog_object_keypoints_local = object_kps
    return object_kps


def _get_object_prim(env, stage, object_idx: int):
    """Helper function that tries to find the USD prim for one catalog object in the scene."""
    try:
        prim_paths = list(env.scene[f"target_object_{object_idx}"].root_physx_view.prim_paths)
    except Exception:
        return None

    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if prim is not None and prim.IsValid():
            return prim
    return None


def _asset_local_bounds(asset_path: str, Usd, UsdGeom):
    """Compute local AABB min/max directly from an asset USD file."""
    try:
        stage = Usd.Stage.Open(asset_path)
        if stage is None:
            return None, None

        prim = stage.GetDefaultPrim()
        if prim is None or not prim.IsValid():
            children = stage.GetPseudoRoot().GetChildren()
            prim = children[0] if children else None
        if prim is None or not prim.IsValid():
            return None, None

        bbox_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        )
        return _local_bounds(prim, bbox_cache)
    except Exception:
        return None, None


def _local_bounds(prim, bbox_cache):
    """
    Helper function to get local AABB coordinates for a given prim using the provided bounding box cache. 
    Returns (min_v, max_v) tuple, where each is a 3D coordinate. If bounds cannot be computed, returns (None, None).
    """
    try:
        # compute local bounding box for this prim.
        bbox = bbox_cache.ComputeLocalBound(prim)

        # obtain min/max range object from the bbox.
        rng = bbox.GetRange()

        # read minimum corner.
        min_v = rng.GetMin()

        # read maximum corner.
        max_v = rng.GetMax()
        return (
            (float(min_v[0]), float(min_v[1]), float(min_v[2])),
            (float(max_v[0]), float(max_v[1]), float(max_v[2])),
        )
    except Exception:
        return None, None


def _bounds_corner_keypoints(min_v, max_v, device):
    """Helper function to generate 8 corner keypoints from AABB bounds."""
    xs = (min_v[0], max_v[0])
    ys = (min_v[1], max_v[1])
    zs = (min_v[2], max_v[2])
    pts = []

    # triple loop over all min/max combinations of x, y, z to generate the 8 corners of the bounding box
    for x in xs:
        for y in ys:
            for z in zs:
                pts.append((x, y, z))
    return torch.tensor(pts, device=device, dtype=torch.float32)


def _transform_points(points_local: torch.Tensor, pos_w: torch.Tensor, quat_w: torch.Tensor) -> torch.Tensor:
    """Helper function to transform local keypoints to world coordinates for batched envs."""

    # unpack the shapes of the input tensors
    num_envs, num_points, _ = points_local.shape

    # 1. quat_w has a shape of (num_envs, 4), representing one rotation quaternion per env.
    # 2. the None keywords is an alias for torch.unsqueeze(), inserts a new dimension of size 1 at that specific position, so quat_w[:, None, :] has a shape of (num_envs, 1, 4).
    # 3. the expand() function then replicates the data along the new dimension to match the number of points, resulting in a shape of (num_envs, num_points, 4).
    # 4. we reshape it back to (num_envs * num_points, 4) to prepare for the batched quaternion application.
    quat = quat_w[:, None, :].expand(num_envs, num_points, 4).reshape(-1, 4)

    # reshape local points to (num_envs * num_points, 3) to prepare for the batched quaternion application.
    points = points_local.reshape(-1, 3)

    # apply the quaternion rotation to the local points, resulting in rotated points of shape (num_envs * num_points, 3), 
    # then reshape back to (num_envs, num_points, 3).
    points_w = _quat_apply_safe(quat, points).reshape(num_envs, num_points, 3)

    # add the translation to the rotated points to get the final world coordinates of the keypoints.
    return points_w + pos_w[:, None, :]


def _quat_apply_safe(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    if hasattr(math_utils, "quat_apply"):
        return math_utils.quat_apply(quat, vec)

    # we imported math_utils so it should be fine.
    w = quat[:, 0:1]
    xyz = quat[:, 1:4]
    cross_term = 2.0 * torch.cross(xyz, vec, dim=-1)
    return vec + w * cross_term + torch.cross(xyz, cross_term, dim=-1)


def keypoint_mean_distance(k_a_w: torch.Tensor, k_b_w: torch.Tensor) -> torch.Tensor:

    # mean L2 distance between corresponding keypoints in k_a_w and k_b_w, 
    # where k_a_w and k_b_w have shape (num_envs, num_keypoints, 3)
    return torch.linalg.norm(k_a_w - k_b_w, dim=-1).mean(dim=-1)


def keypoint_yaw_error_deg_xy(k_src_w: torch.Tensor, k_tgt_w: torch.Tensor) -> torch.Tensor:
    """Calculates the yaw orientation in degrees between two sets of 2D keypoints using the Kabsch-Umeyama alignment algorithm."""
    src_xy = k_src_w[..., :2]
    tgt_xy = k_tgt_w[..., :2]

    src_centered = src_xy - src_xy.mean(dim=-2, keepdim=True)
    tgt_centered = tgt_xy - tgt_xy.mean(dim=-2, keepdim=True)

    cov = src_centered.transpose(-1, -2) @ tgt_centered
    u, _, vh = torch.linalg.svd(cov)
    v = vh.transpose(-1, -2)
    ut = u.transpose(-1, -2)

    det = torch.linalg.det(v @ ut)
    sign = torch.where(det < 0.0, -torch.ones_like(det), torch.ones_like(det))
    correction = torch.zeros_like(cov)
    correction[..., 0, 0] = 1.0
    correction[..., 1, 1] = sign

    rot = v @ correction @ ut
    yaw_err = torch.atan2(rot[..., 1, 0], rot[..., 0, 0])
    yaw_err = torch.remainder(yaw_err + torch.pi, 2.0 * torch.pi) - torch.pi
    return torch.rad2deg(yaw_err)