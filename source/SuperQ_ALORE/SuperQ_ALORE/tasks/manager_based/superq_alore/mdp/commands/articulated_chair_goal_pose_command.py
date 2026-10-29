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
from isaaclab.utils import configclass

from ..articulated_chair import upper_body_pose_w

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class ArticulatedChairGoalPoseCommand(CommandTerm):
    """Sample world-frame goals for the chair ``frame`` link.

    The command mirrors the existing goal-pose command's x/y/yaw interface, but
    never reads the rigid-object catalog.  Consequently the goal quaternion is
    based on the upper assembly's live orientation rather than the free base.

    When debug visualization is enabled, a hollow ground square joined to an
    outlined heading arrow shows the target pose of the ``frame`` link.  This
    represents the tracked upper assembly rather than spawning a second,
    physically misleading copy of the whole articulated chair.
    """

    def __init__(self, cfg, env: ManagerBasedEnv):
        # This must exist before ``CommandTerm.__init__`` calls
        # ``_set_debug_vis_impl``.
        self._goal_outline_vis: VisualizationMarkers | None = None
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
        shaft_half_width = self.cfg.goal_heading_shaft_width * 0.5
        head_half_width = self.cfg.goal_heading_head_width * 0.5
        head_base_x = half_extent + self.cfg.goal_heading_shaft_length
        tip_x = head_base_x + self.cfg.goal_heading_head_length
        # One closed outline: leave a gap in the square's forward (+x) edge
        # where the open arrow shaft joins it.  The arrow has no fill or base
        # line, matching the continuous square-and-arrow silhouette.
        outline_vertices = torch.tensor(
            ((-half_extent, -half_extent, 0.0),
             (half_extent, -half_extent, 0.0),
             (half_extent, -shaft_half_width, 0.0),
             (head_base_x, -shaft_half_width, 0.0),
             (head_base_x, -head_half_width, 0.0),
             (tip_x, 0.0, 0.0),
             (head_base_x, head_half_width, 0.0),
             (head_base_x, shaft_half_width, 0.0),
             (half_extent, shaft_half_width, 0.0),
             (half_extent, half_extent, 0.0),
             (-half_extent, half_extent, 0.0)),
            dtype=self.goal_w.dtype,
            device=self.device,
        )
        next_vertices = torch.roll(outline_vertices, shifts=-1, dims=0)
        edge_vectors = next_vertices - outline_vertices
        self._goal_outline_local_offsets = (outline_vertices + next_vertices) * 0.5
        edge_yaw = torch.atan2(edge_vectors[:, 1], edge_vectors[:, 0])
        zeros = torch.zeros_like(edge_yaw)
        self._goal_outline_local_quats = math_utils.quat_from_euler_xyz(zeros, zeros, edge_yaw)
        edge_scales = torch.empty_like(edge_vectors)
        # Slightly overlap adjoining stroke ends so corners have no gaps.
        edge_scales[:, 0] = torch.linalg.vector_norm(edge_vectors, dim=-1) + self.cfg.goal_outline_line_width
        edge_scales[:, 1] = self.cfg.goal_outline_line_width
        edge_scales[:, 2] = self.cfg.goal_outline_height
        self._goal_outline_scales = edge_scales.repeat(self.num_envs, 1)
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
            self._goal_outline_vis.set_visibility(True)
        else:
            if self._goal_outline_vis is not None:
                self._goal_outline_vis.set_visibility(False)

    def _debug_vis_callback(self, event) -> None:
        """Show the target location and yaw of the chair's tracked ``frame`` link."""

        del event
        # A queued render callback can run after the environment closes.
        if self._goal_outline_vis is None or not hasattr(self._env, "scene"):
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

        # Thin, flat strokes form both the square and the hollow arrow.
        # Rotate their centers and local edge orientations with the goal yaw.
        num_edges = self._goal_outline_local_offsets.shape[0]
        goal_quat_per_edge = goal_yaw_quat_w.repeat_interleave(num_edges, dim=0)
        outline_offsets_w = math_utils.quat_apply(
            goal_quat_per_edge,
            self._goal_outline_local_offsets.repeat(self.num_envs, 1),
        )
        outline_pos_w = goal_ground_w.repeat_interleave(num_edges, dim=0) + outline_offsets_w
        outline_quat_w = math_utils.quat_mul(
            goal_quat_per_edge, self._goal_outline_local_quats.repeat(self.num_envs, 1)
        )
        self._goal_outline_vis.visualize(
            outline_pos_w,
            outline_quat_w,
            scales=self._goal_outline_scales,
        )


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

    # A continuous green outline lies on the ground.  Its hollow arrow joins
    # the square's forward edge and indicates the desired upper-body yaw.
    goal_outline_size: float = 0.70
    goal_outline_line_width: float = 0.012
    goal_outline_height: float = 0.008
    goal_heading_shaft_length: float = 0.30
    goal_heading_shaft_width: float = 0.12
    goal_heading_head_length: float = 0.25
    goal_heading_head_width: float = 0.36
    goal_outline_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Command/articulated_chair_goal_outline",
        markers={
            "edge": sim_utils.CuboidCfg(
                # Each instance is scaled to the length, width and height of
                # its outline segment by the command's visualizer.
                size=(1.0, 1.0, 1.0),
                rigid_props=None,
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.0, 0.9, 0.18), emissive_color=(0.0, 0.12, 0.02), roughness=0.6
                ),
            ),
        },
    )
    goal_marker_ground_offset: float = 0.015
