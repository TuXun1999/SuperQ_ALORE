# Copyright (c) 2024 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

"""Custom velocity command term for object motion aligned with robot x-axis."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import omni.log
import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .commands_cfg import ObjectUniformVelocityRobotFrameCommandCfg


class ObjectUniformVelocityRobotFrameCommand(CommandTerm):
    r"""Uniform velocity command with configurable commanded asset and reference axis asset.

    The command is represented in the commanded asset's body frame. Linear direction can be
    aligned with the reference asset x-axis (default: robot), which allows the same term to
    command either object velocity or robot velocity.
    """

    cfg: ObjectUniformVelocityRobotFrameCommandCfg

    def __init__(self, cfg: ObjectUniformVelocityRobotFrameCommandCfg, env: ManagerBasedEnv):
        
        # initailize the base class
        super().__init__(cfg, env)

        # check the configuration
        if self.cfg.heading_command and self.cfg.ranges.heading is None:
            raise ValueError(
                "The velocity command has heading commands active (heading_command=True) but the `ranges.heading`"
                " parameter is set to None."
            )
        if self.cfg.ranges.heading and not self.cfg.heading_command:
            omni.log.warn(
                f"The velocity command has the 'ranges.heading' attribute set to '{self.cfg.ranges.heading}'"
                " but the heading command is not active. Consider setting the flag for the heading command to True."
            )

        # asset whose velocity is being commanded and tracked based on the error.
        self.command_asset: Articulation = env.scene[cfg.asset_name]

        # Asset providing the forward x-axis used for direction alignment.
        # In our setting we choose to be the robot, since we only want it move forward/backward.
        self.reference_asset: Articulation = env.scene[cfg.reference_asset_name]

        # Command storage: linear x/y and yaw in the commanded asset body frame.
        self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.heading_target = torch.zeros(self.num_envs, device=self.device)
        self.is_heading_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.is_standing_env = torch.zeros_like(self.is_heading_env)

        # Metrics follow the original uniform velocity command.
        self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        msg = "ObjectUniformVelocityRobotFrameCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        msg += f"\tHeading command: {self.cfg.heading_command}\n"
        if self.cfg.heading_command:
            msg += f"\tHeading probability: {self.cfg.rel_heading_envs}\n"
        msg += f"\tStanding probability: {self.cfg.rel_standing_envs}"
        return msg

    @property
    def command(self) -> torch.Tensor:
        return self.vel_command_b

    def _update_metrics(self):
        # obtain the max command time since the resampling_time_range is a tuple.
        max_command_time = self.cfg.resampling_time_range[1]

        # calculate the max command step based on the max command time and environment step time,
        # and use it to normalize the velocity error.
        max_command_step = max_command_time / self._env.step_dt
        self.metrics["error_vel_xy"] += (
            torch.norm(
                self.vel_command_b[:, :2] - self.command_asset.data.root_lin_vel_b[:, :2],
                dim=-1,
            )
            / max_command_step
        )

        # accumulate yaw velocity error, normalized by max command step. 
        self.metrics["error_vel_yaw"] += (
            torch.abs(self.vel_command_b[:, 2] - self.command_asset.data.root_ang_vel_b[:, 2]) / max_command_step
        )

    def _resample_command(self, env_ids: Sequence[int]):

        # sample velocity commands from the range
        r = torch.empty(len(env_ids), device=self.device)

        # sample linear and angular velocity commands from the specified ranges.
        self.vel_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)
        self.vel_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)
        self.vel_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z)
        if self.cfg.heading_command:
            self.heading_target[env_ids] = r.uniform_(*self.cfg.ranges.heading)
            self.is_heading_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs
        self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs

    def _update_command(self):
        """Post-process the command so linear motion follows the reference asset x-axis."""

        # compute heading velocity from heading direction.
        if self.cfg.heading_command:

            # resolve indices of heading envs
            env_ids = self.is_heading_env.nonzero(as_tuple=False).flatten()

            # compute angular velocity using a proportional contorller
            heading_error = math_utils.wrap_to_pi(
                self.heading_target[env_ids] - self.reference_asset.data.heading_w[env_ids]
            )
            self.vel_command_b[env_ids, 2] = torch.clip(
                self.cfg.heading_control_stiffness * heading_error,
                min=self.cfg.ranges.ang_vel_z[0],
                max=self.cfg.ranges.ang_vel_z[1],
            )

        # take the sampled linear velocity command (forward/backward direction)
        speed = self.vel_command_b[:, 0].clone()

        # construct a unit x-axis vector in reference asset body frame.
        robot_x_b = torch.zeros(self.num_envs, 3, device=self.device)
        robot_x_b[:, 0] = 1.0

        # rotate the x-axis vector to world frame, using the reference asset quaternion.
        robot_x_w = math_utils.quat_apply(self.reference_asset.data.root_quat_w, robot_x_b)

        # scale the x-axis vector by the magnitude of the linear velocity command to get the desired velocity in world frame.
        desired_lin_vel_w = robot_x_w * speed.unsqueeze(-1)

        # converts that world-frame velocity back into commanded asset body frame.
        desired_lin_vel_obj_b = math_utils.quat_apply_inverse(self.command_asset.data.root_quat_w, desired_lin_vel_w)
        self.vel_command_b[:, :2] = desired_lin_vel_obj_b[:, :2]

        # enforcing standing by zeroing out the velocity command for standing envs.
        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        self.vel_command_b[standing_env_ids, :] = 0.0

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer = VisualizationMarkers(self.cfg.goal_vel_visualizer_cfg)
                self.current_vel_visualizer = VisualizationMarkers(self.cfg.current_vel_visualizer_cfg)
            self.goal_vel_visualizer.set_visibility(True)
            self.current_vel_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer.set_visibility(False)
                self.current_vel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.command_asset.is_initialized:
            return

        base_pos_w = self.command_asset.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5

        vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_xy_velocity_to_arrow(self.command[:, :2])
        vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(self.command_asset.data.root_lin_vel_b[:, :2])
        self.goal_vel_visualizer.visualize(base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
        self.current_vel_visualizer.visualize(base_pos_w, vel_arrow_quat, vel_arrow_scale)

    def _resolve_xy_velocity_to_arrow(self, xy_velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        default_scale = self.goal_vel_visualizer.cfg.markers["arrow"].scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0
        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)
        base_quat_w = self.command_asset.data.root_quat_w
        arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)
        return arrow_scale, arrow_quat
