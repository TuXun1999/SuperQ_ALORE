"""Goal-pose command that follows the QiVi upper assembly rather than its root."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
import math
from typing import TYPE_CHECKING

import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import GREEN_ARROW_X_MARKER_CFG
from isaaclab.utils import configclass

from ..articulated_chair import upper_body_pose_w

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class ArticulatedChairGoalPoseCommand(CommandTerm):
    """Sample world-frame goals for the chair ``frame`` link.

    The command mirrors the existing goal-pose command's x/y/yaw interface, but
    never reads the rigid-object catalog.  Consequently the goal quaternion is
    based on the upper assembly's live orientation rather than the free base.

    When debug visualization is enabled, a hollow ground square and heading
    arrow show the target pose of the ``frame`` link.  This intentionally
    represents the tracked upper assembly rather than spawning a second,
    physically misleading copy of the whole articulated chair.
    """

    def __init__(self, cfg, env: ManagerBasedEnv):
        # This must exist before ``CommandTerm.__init__`` calls
        # ``_set_debug_vis_impl``.
        self._goal_outline_vis: VisualizationMarkers | None = None
        self._goal_heading_vis: VisualizationMarkers | None = None
        super().__init__(cfg, env)
        self.goal_w = torch.zeros((self.num_envs, 3), device=self.device)
        self.goal_quat_w = torch.zeros((self.num_envs, 4), device=self.device)
        self.goal_quat_w[:, 0] = 1.0
        self._active_yaw_range = tuple(self.cfg.ranges.yaw)
        if bool(getattr(self.cfg, "enable_yaw_curriculum", False)):
            self._active_yaw_range = tuple(
                getattr(self.cfg, "curriculum_initial_yaw_range", self.cfg.ranges.yaw)
            )
        half_extent = self.cfg.goal_outline_size * 0.5
        self._goal_outline_local_offsets = torch.tensor(
            # Two x-aligned edges followed by two y-aligned edges.
            ((0.0, half_extent, 0.0), (0.0, -half_extent, 0.0),
             (half_extent, 0.0, 0.0), (-half_extent, 0.0, 0.0)),
            dtype=self.goal_w.dtype,
            device=self.device,
        )
        self._goal_outline_marker_indices = torch.tensor((0, 0, 1, 1), dtype=torch.int64, device=self.device)
        self._resample_command(torch.arange(self.num_envs, device=self.device, dtype=torch.long))

    @property
    def command(self) -> torch.Tensor:
        return torch.cat((self.goal_w, self.goal_quat_w), dim=-1)

    def set_goal_world(self, env_ids, pos_w, quat_w=None) -> None:
        """Set absolute target poses for an external evaluation/control loop."""

        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        pos_w = torch.as_tensor(pos_w, device=self.device, dtype=self.goal_w.dtype)
        if pos_w.ndim == 1:
            pos_w = pos_w.unsqueeze(0).expand(env_ids.numel(), -1)
        self.goal_w[env_ids] = pos_w

        if quat_w is None:
            quat_w = torch.zeros((env_ids.numel(), 4), device=self.device, dtype=self.goal_w.dtype)
            quat_w[:, 0] = 1.0
        else:
            quat_w = torch.as_tensor(quat_w, device=self.device, dtype=self.goal_w.dtype)
            if quat_w.ndim == 1:
                quat_w = quat_w.unsqueeze(0).expand(env_ids.numel(), -1)
        self.goal_quat_w[env_ids] = quat_w

    def resample_on_reset(self, env_ids: Sequence[int]) -> None:
        """Public reset hook used by the environment event configuration."""

        self._resample_command(env_ids)

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if env_ids.numel() == 0:
            return

        origins = self._env.scene.env_origins[env_ids]
        frame_pos_w, frame_quat_w = upper_body_pose_w(self._env)
        samples = torch.empty((env_ids.numel(), 4), device=self.device)
        samples[:, 0].uniform_(*self.cfg.ranges.pos_x)
        samples[:, 1].uniform_(*self.cfg.ranges.pos_y)
        samples[:, 2].uniform_(*self.cfg.ranges.pos_z)
        samples[:, 3].uniform_(*self._active_yaw_range)

        self.goal_w[env_ids, :2] = origins[:, :2] + samples[:, :2]
        # Keep the target at the upper assembly's nominal height; the task is
        # planar, but this makes debug/external goals physically meaningful.
        self.goal_w[env_ids, 2] = frame_pos_w[env_ids, 2] + samples[:, 2]
        zeros = torch.zeros(env_ids.numel(), dtype=self.goal_w.dtype, device=self.device)
        yaw_delta = math_utils.quat_from_euler_xyz(zeros, zeros, samples[:, 3])
        self.goal_quat_w[env_ids] = math_utils.quat_mul(yaw_delta, frame_quat_w[env_ids])

    def _update_metrics(self) -> None:
        frame_pos_w, frame_quat_w = upper_body_pose_w(self._env)
        position_error = torch.linalg.vector_norm((self.goal_w - frame_pos_w)[:, :2], dim=-1)
        goal_yaw = math_utils.euler_xyz_from_quat(self.goal_quat_w)[2]
        frame_yaw = math_utils.euler_xyz_from_quat(frame_quat_w)[2]
        yaw_error = torch.atan2(torch.sin(goal_yaw - frame_yaw), torch.cos(goal_yaw - frame_yaw)).abs()
        self.metrics["upper_body_to_goal_dist"] = position_error
        self.metrics["upper_body_to_goal_yaw_diff"] = yaw_error
        self.metrics["success_rate"] = (
            (position_error <= float(getattr(self.cfg, "success_object_to_goal_dist_thresh_m", 0.10)))
            & (torch.rad2deg(yaw_error) <= float(getattr(self.cfg, "success_keypoint_angle_error_thresh_deg", 10.0)))
        ).to(torch.float32)

    def _update_command(self) -> None:
        # Goals only change when explicitly resampled.
        return

    def _set_debug_vis_impl(self, debug_vis: bool) -> None:
        """Create or hide the ground-projected upper-assembly goal marker."""

        if debug_vis:
            if self._goal_outline_vis is None:
                self._goal_outline_vis = VisualizationMarkers(self.cfg.goal_outline_visualizer_cfg)
                self._goal_heading_vis = VisualizationMarkers(self.cfg.goal_heading_visualizer_cfg)
            self._goal_outline_vis.set_visibility(True)
            self._goal_heading_vis.set_visibility(True)
        else:
            if self._goal_outline_vis is not None:
                self._goal_outline_vis.set_visibility(False)
            if self._goal_heading_vis is not None:
                self._goal_heading_vis.set_visibility(False)

    def _debug_vis_callback(self, event) -> None:
        """Show the target location and yaw of the chair's tracked ``frame`` link."""

        del event
        if self._goal_outline_vis is None or self._goal_heading_vis is None:
            return

        # The command is expressed at the elevated upper assembly.  The debug
        # marker instead lives on each environment's ground plane, while
        # retaining its x/y location and yaw.  This keeps it visible even when
        # the chair overlaps the target at success.
        goal_ground_w = self.goal_w.clone()
        goal_ground_w[:, 2] = self._env.scene.env_origins[:, 2] + self.cfg.goal_marker_ground_offset
        goal_yaw = math_utils.euler_xyz_from_quat(self.goal_quat_w)[2]
        zeros = torch.zeros_like(goal_yaw)
        goal_yaw_quat_w = math_utils.quat_from_euler_xyz(zeros, zeros, goal_yaw)

        # A two-prototype marker makes the four thin cuboid edges.  Transform
        # the local edge centers by the yaw-only target orientation so their
        # outline rotates with the desired chair heading.
        num_edges = self._goal_outline_local_offsets.shape[0]
        outline_quat_w = goal_yaw_quat_w.repeat_interleave(num_edges, dim=0)
        outline_offsets_w = math_utils.quat_apply(
            outline_quat_w,
            self._goal_outline_local_offsets.repeat(self.num_envs, 1),
        )
        outline_pos_w = goal_ground_w.repeat_interleave(num_edges, dim=0) + outline_offsets_w
        self._goal_outline_vis.visualize(
            outline_pos_w,
            outline_quat_w,
            marker_indices=self._goal_outline_marker_indices.repeat(self.num_envs),
        )

        # Put the arrow directly on the forward (+x) edge.  It shares the
        # outline's green material and ground height, so it reads as part of
        # the square rather than as a separate floating marker.
        heading_offset_b = torch.zeros_like(goal_ground_w)
        heading_offset_b[:, 0] = self.cfg.goal_outline_size * 0.5 + self.cfg.goal_heading_edge_offset
        goal_heading_w = goal_ground_w + math_utils.quat_apply(goal_yaw_quat_w, heading_offset_b)
        goal_heading_w[:, 2] += self.cfg.goal_heading_height_offset
        self._goal_heading_vis.visualize(goal_heading_w, goal_yaw_quat_w)


@configclass
class ArticulatedChairGoalPoseCommandCfg(CommandTermCfg):
    """Configuration for a goal expressed at the chair's ``frame`` link."""

    class_type: type = ArticulatedChairGoalPoseCommand

    @configclass
    class Ranges:
        pos_x: tuple[float, float] = MISSING
        pos_y: tuple[float, float] = MISSING
        pos_z: tuple[float, float] = (0.0, 0.0)
        yaw: tuple[float, float] = (-math.pi, math.pi)

    ranges: Ranges = MISSING
    success_object_to_goal_dist_thresh_m: float = 0.10
    success_keypoint_angle_error_thresh_deg: float = 10.0
    enable_yaw_curriculum: bool = False
    curriculum_initial_yaw_range: tuple[float, float] = (0.0, 0.0)
    curriculum_yaw_step: float = math.pi / 3.0
    curriculum_max_yaw: float = math.pi
    curriculum_iterations_per_level: int = 1000
    debug_vis_keypoints: bool = False
    debug_vis_keypoint_radius: float = 0.04

    # This marker lies on the ground rather than at the elevated ``frame``
    # link.  Four green bars form a hollow square; the arrow gives its yaw.
    goal_outline_size: float = 0.70
    goal_outline_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Command/articulated_chair_goal_outline",
        markers={
            "edge_x": sim_utils.CuboidCfg(
                size=(goal_outline_size, 0.022, 0.025),
                rigid_props=None,
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.0, 0.9, 0.18), emissive_color=(0.0, 0.12, 0.02), roughness=0.6
                ),
            ),
            "edge_y": sim_utils.CuboidCfg(
                size=(0.022, goal_outline_size, 0.025),
                rigid_props=None,
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.0, 0.9, 0.18), emissive_color=(0.0, 0.12, 0.02), roughness=0.6
                ),
            ),
        },
    )
    goal_heading_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/articulated_chair_goal_heading"
    )
    goal_heading_visualizer_cfg.markers["arrow"].scale = (0.55, 0.09, 0.025)
    goal_heading_visualizer_cfg.markers["arrow"].visual_material = sim_utils.PreviewSurfaceCfg(
        diffuse_color=(0.0, 0.9, 0.18), emissive_color=(0.0, 0.12, 0.02), roughness=0.6
    )
    goal_marker_ground_offset: float = 0.012
    goal_heading_height_offset: float = 0.0
    goal_heading_edge_offset: float = 0.0
