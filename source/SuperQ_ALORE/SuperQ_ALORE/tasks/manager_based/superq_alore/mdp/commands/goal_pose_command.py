from __future__ import annotations

from collections.abc import Sequence

# use TYPE_CHECKING to avoid circular imports for type hints
from typing import TYPE_CHECKING

import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

from SuperQ_ALORE.assets.object_catalog import OBJECT_CATALOG
from .. import keypoints as keypoints_mdp
from .. import object_management as object_mgmt

# When the program executes, it will not import the following modules at runtime, 
# but they are imported only for type checking, and not imported at runtime 
# to avoid circular import issues and reduce import overhead.
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from .commands_cfg import GoalPoseCommandCfg


class GoalPoseCommand(CommandTerm):
    """Sample a per-env goal object pose and visualize the matching catalog object."""

    cfg: GoalPoseCommandCfg

    def __init__(self, cfg: GoalPoseCommandCfg, env: ManagerBasedEnv):
        self._goal_vis = None
        self._kps_vis = None
        super().__init__(cfg, env)

        # stores the goal pose in world frame for each environment, represented as (x, y, z) position and (x, y, z, w) quaternion. 
        self.goal_w = torch.zeros((self.num_envs, 3), device=self.device)
        self.goal_quat_w = torch.zeros((self.num_envs, 4), device=self.device)
        self.goal_quat_w[:, 0] = 1.0

        if self.cfg.debug_vis:

            # initialize the goal material (green)
            goal_mat = sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.1, 0.8, 0.2),
                opacity=0.35,
            )

            # create a visualization marker for the goal pose, using the object asset paths from the catalog and the green material.
            # the marker will be parented under "/Visuals/Command/goal_pose" in the scene.
            marker_cfg = VisualizationMarkersCfg(
                prim_path="/Visuals/Command/goal_pose",
                markers={
                    f"object_{obj_idx}": sim_utils.UsdFileCfg(
                        usd_path=obj.asset_path,
                        rigid_props=None,
                        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
                        visual_material=goal_mat,
                    )
                    for obj_idx, obj in enumerate(OBJECT_CATALOG)
                },
            )

            # initialize the VisualizationMarkers instance for the goal pose visualization using the defined marker configuration.
            self._goal_vis = VisualizationMarkers(marker_cfg)

            # if the configuration flag for keypoint visualization is set, 
            # initialize the visualization markers for the keypoints with specified colors and sizes.
            if bool(getattr(self.cfg, "debug_vis_keypoints", True)):
                keypoint_radius = float(getattr(self.cfg, "debug_vis_keypoint_radius", 0.04))
                keypoint_push_mat = sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(1.0, 1.0, 0.0),
                    opacity=0.9,
                )
                keypoint_goal_mat = sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(1.0, 0.0, 0.0),
                    opacity=0.9,
                )
                kps_cfg = VisualizationMarkersCfg(
                    prim_path="/Visuals/Command/goal_pose_keypoints",
                    markers={
                        "push_kp": sim_utils.SphereCfg(
                            radius=keypoint_radius,
                            rigid_props=None,
                            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
                            visual_material=keypoint_push_mat,
                        ),
                        "goal_kp": sim_utils.SphereCfg(
                            radius=keypoint_radius,
                            rigid_props=None,
                            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
                            visual_material=keypoint_goal_mat,
                        ),
                    },
                )
                self._kps_vis = VisualizationMarkers(kps_cfg)

        # initially sample a goal pose for all environments by calling _resample_command with all environment indices.
        self._resample_command(torch.arange(self.num_envs, device=self.device, dtype=torch.long))

    @property
    def command(self) -> torch.Tensor:
        # concatenate the goal position and quaternion to return the full goal pose command of shape (num_envs, 7) 
        # where the first 3 dimensions are position and the next 4 dimensions are quaternion.
        return torch.cat((self.goal_w, self.goal_quat_w), dim=-1)

    def set_goal_world(self, env_ids, pos_w, quat_w=None):
        """External helper function to set the goal pose for specific environment"""

        # convert env_ids, pos_w, quat_w to tensors and ensure they have the correct shapes, 
        # then update the goal_w and goal_quat_w tensors for the specified environment indices.
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        pos = torch.as_tensor(pos_w, device=self.device, dtype=self.goal_w.dtype)
        if pos.dim() == 1:
            pos = pos.view(1, 3).repeat(env_ids.numel(), 1)

        # writes goal positions into the self.goal_w tensor at the specified environment indices.
        self.goal_w[env_ids] = pos

        if quat_w is None:
            quat = torch.zeros((env_ids.numel(), 4), device=self.device, dtype=self.goal_w.dtype)
            quat[:, 0] = 1.0
        else:
            quat = torch.as_tensor(quat_w, device=self.device, dtype=self.goal_w.dtype)
            if quat.dim() == 1:
                quat = quat.view(1, 4).repeat(env_ids.numel(), 1)
        # writes goal quaternions into the self.goal_quat_w tensor at the specified environment indices.
        self.goal_quat_w[env_ids] = quat

    def resample_on_reset(self, env_ids: Sequence[int]) -> None:
        """Public API to resample goal pose for reset environments."""
        self._resample_command(env_ids)

    def _resample_command(self, env_ids: Sequence[int]):

        # normalize env_ids to a device tensor
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        # null check
        if env_ids.numel() == 0:
            return
        
        # reads per-environment world origins offsets
        origins = self._env.scene.env_origins[env_ids]

        # for each environment index, sample a random goal pose by sampling offsets from the specified ranges in the configuration,
        # then add those offsets to the environment origins to get the final goal positions in world frame
        samples = torch.empty((env_ids.numel(), 4), device=self.device)
        samples[:, 0].uniform_(*self.cfg.ranges.pos_x)
        samples[:, 1].uniform_(*self.cfg.ranges.pos_y)
        samples[:, 2].uniform_(*self.cfg.ranges.pos_z)
        samples[:, 3].uniform_(*self.cfg.ranges.yaw)

        self.goal_w[env_ids, 0] = origins[:, 0] + samples[:, 0]
        self.goal_w[env_ids, 1] = origins[:, 1] + samples[:, 1]
        self.goal_w[env_ids, 2] = origins[:, 2] + samples[:, 2]

        zeros = torch.zeros(env_ids.numel(), device=self.device, dtype=self.goal_w.dtype)
        self.goal_quat_w[env_ids] = math_utils.quat_from_euler_xyz(zeros, zeros, samples[:, 3])

    def _update_command(self):
        return

    def _update_metrics(self):
        # object-to-goal distance in XY plane
        goal_xy = self.goal_w[:, :2]
        obj_pos_w = object_mgmt.get_active_object_state_attr(self._env, "root_pos_w")
        obj_xy = obj_pos_w[:, :2]
        object_to_goal_dist = torch.linalg.norm(goal_xy - obj_xy, dim=1)

        # keypoint alignment angle error (degrees)
        keypoint_angle_error_degree = torch.full(
            (self.num_envs,),
            180.0,
            device=self.device,
            dtype=self.goal_w.dtype,
        )
        push_kps = None
        goal_kps = None
        try:
            push_kps = keypoints_mdp.pushable_keypoints_w(self._env)
        except Exception:
            push_kps = None

        term_candidates = []
        configured_term = getattr(self.cfg, "goal_term_name", "goal_pose")
        for term_name in (configured_term, "goal_pose", "goal_region"):
            if term_name not in term_candidates:
                term_candidates.append(term_name)

        for term_name in term_candidates:
            try:
                goal_kps = keypoints_mdp.goal_keypoints_w(self._env, goal_term_name=term_name)
                break
            except Exception:
                goal_kps = None

        if push_kps is not None and goal_kps is not None:
            try:
                keypoint_angle_error_degree = torch.abs(
                    keypoints_mdp.keypoint_yaw_error_deg_xy(push_kps, goal_kps)
                )
            except Exception:
                pass

        # object and robot speed (m/s)
        obj_lin_vel_w = object_mgmt.get_active_object_state_attr(self._env, "root_lin_vel_w")
        object_speed = torch.linalg.norm(obj_lin_vel_w, dim=1)

        robot = self._env.scene["robot"]
        robot_lin_vel_w = robot.data.root_lin_vel_w
        robot_speed = torch.linalg.norm(robot_lin_vel_w, dim=1)

        # robot yaw in degree
        try:
            _, _, robot_yaw = math_utils.euler_xyz_from_quat(robot.data.root_quat_w)
            robot_yaw_degree = torch.rad2deg(robot_yaw)
        except Exception:
            robot_yaw_degree = torch.zeros(self.num_envs, device=self.device, dtype=self.goal_w.dtype)

        # success rate (1.0 success, 0.0 failure) using configurable thresholds
        success_dist_thresh = float(getattr(self.cfg, "success_object_to_goal_dist_thresh_m", 0.10))
        success_angle_thresh = float(getattr(self.cfg, "success_keypoint_angle_error_thresh_deg", 10.0))
        success_rate = (
            (object_to_goal_dist <= success_dist_thresh)
            & (keypoint_angle_error_degree <= success_angle_thresh)
        ).to(torch.float32)

        # requested metric names
        self.metrics["object_to_goal_dist"] = object_to_goal_dist
        self.metrics["keypoint_angle_error_degree"] = keypoint_angle_error_degree
        self.metrics["object_speed"] = object_speed
        self.metrics["robot_speed"] = robot_speed
        self.metrics["robot_yaw_degree"] = robot_yaw_degree
        self.metrics["success_rate"] = success_rate

    def _set_debug_vis_impl(self, debug_vis: bool):
        """set the visibility of the goal pose and keypoint visualizers based on the debug_vis flag."""
        if self._goal_vis is not None:
            self._goal_vis.set_visibility(debug_vis)
        if self._kps_vis is not None:
            self._kps_vis.set_visibility(debug_vis)

    def _debug_vis_callback(self, event):
        """visualize the goal pose and keypoints in the simulation for debugging purposes."""
        if self._goal_vis is None and self._kps_vis is None:
            return

        if hasattr(self._env, "active_object_indices"):
            marker_indices = self._env.active_object_indices.to(dtype=torch.int64, device=self.device)
        else:
            marker_indices = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)

        if self._goal_vis is not None:
            self._goal_vis.visualize(self.goal_w, self.goal_quat_w, marker_indices=marker_indices)

        if self._kps_vis is None:
            return

        push_kps = None
        goal_kps = None

        try:
            push_kps = keypoints_mdp.pushable_keypoints_w(self._env)
        except Exception:
            push_kps = None

        term_candidates = []
        configured_term = getattr(self.cfg, "goal_term_name", "goal_pose")
        for term_name in (configured_term, "goal_pose", "goal_region"):
            if term_name not in term_candidates:
                term_candidates.append(term_name)

        for term_name in term_candidates:
            try:
                goal_kps = keypoints_mdp.goal_keypoints_w(self._env, goal_term_name=term_name)
                break
            except Exception:
                goal_kps = None

        if push_kps is not None and goal_kps is not None:
            kps = torch.cat((push_kps, goal_kps), dim=1).reshape(-1, 3)
            marker_ids = torch.cat(
                (
                    torch.zeros((self.num_envs, 8), device=self.device, dtype=torch.int64),
                    torch.ones((self.num_envs, 8), device=self.device, dtype=torch.int64),
                ),
                dim=1,
            ).reshape(-1)
            self._kps_vis.visualize(kps, marker_indices=marker_ids)
            return

        if push_kps is not None:
            marker_ids = torch.zeros((self.num_envs, 8), device=self.device, dtype=torch.int64).reshape(-1)
            self._kps_vis.visualize(push_kps.reshape(-1, 3), marker_indices=marker_ids)
            return

        if goal_kps is not None:
            marker_ids = torch.ones((self.num_envs, 8), device=self.device, dtype=torch.int64).reshape(-1)
            self._kps_vis.visualize(goal_kps.reshape(-1, 3), marker_indices=marker_ids)