# Copyright (c) 2024 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from __future__ import annotations

import math
from dataclasses import MISSING

from isaaclab.managers import CommandTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.utils import configclass
from .arm_command import (
    ArmJointTrajectoryCommand,
    LegJointTrajectoryCommand,
    MultiLegJointTrajectoryCommand,
    BasePoseCommand,
    ArmLegJointBasePoseCommand,
)
from .goal_pose_command import GoalPoseCommand
from .object_velocity_command import ObjectUniformVelocityRobotFrameCommand


@configclass
class ObjectUniformVelocityRobotFrameCommandCfg(CommandTermCfg):
    """Configuration for the object velocity command aligned with robot x-axis."""

    class_type: type = ObjectUniformVelocityRobotFrameCommand

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""

    reference_asset_name: str = "robot"
    """Name of the reference asset whose x-axis defines forward direction for linear commands."""

    heading_command: bool = False
    """Whether to use heading command or angular velocity command. Defaults to False.

    If True, the angular velocity command is computed from the heading error, where the
    target heading is sampled uniformly from provided range. Otherwise, the angular velocity
    command is sampled uniformly from provided range.
    """

    heading_control_stiffness: float = 1.0
    """Scale factor to convert the heading error to angular velocity command. Defaults to 1.0."""

    rel_standing_envs: float = 0.0
    """The sampled probability of environments that should be standing still. Defaults to 0.0."""

    rel_heading_envs: float = 1.0
    """The sampled probability of environments where the robots follow the heading-based angular velocity command
    (the others follow the sampled angular velocity command). Defaults to 1.0.

    This parameter is only used if :attr:`heading_command` is True.
    """

    @configclass
    class Ranges:
        """Uniform distribution ranges for the velocity commands."""

        lin_vel_x: tuple[float, float] = MISSING
        """Range for the linear-x velocity command (in m/s)."""

        lin_vel_y: tuple[float, float] = MISSING
        """Range for the linear-y velocity command (in m/s)."""

        ang_vel_z: tuple[float, float] = MISSING
        """Range for the angular-z velocity command (in rad/s)."""

        heading: tuple[float, float] | None = None
        """Range for the heading command (in rad). Defaults to None.

        This parameter is only used if :attr:`~ObjectUniformVelocityRobotFrameCommandCfg.heading_command` is True.
        """

    ranges: Ranges = MISSING
    """Distribution ranges for the velocity commands."""

    goal_vel_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_goal"
    )
    """The configuration for the goal velocity visualization marker. Defaults to GREEN_ARROW_X_MARKER_CFG."""

    current_vel_visualizer_cfg: VisualizationMarkersCfg = BLUE_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_current"
    )
    """The configuration for the current velocity visualization marker. Defaults to BLUE_ARROW_X_MARKER_CFG."""

    # Set the scale of the visualization markers to (0.5, 0.5, 0.5)
    goal_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
    current_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)


@configclass
class GoalPoseCommandCfg(CommandTermCfg):
    """Configuration for a per-env sampled goal pose and corresponding object green marker."""

    class_type: type = GoalPoseCommand

    @configclass
    class Ranges:
        pos_x: tuple[float, float] = MISSING
        """Range for the x-coordinate of the goal pose."""

        pos_y: tuple[float, float] = MISSING
        """Range for the y-coordinate of the goal pose."""

        pos_z: tuple[float, float] = (0.0, 0.0)
        """Range for the z-coordinate of the goal pose, default to zero."""

        yaw: tuple[float, float] = (-3.141592653589793, 3.141592653589793)
        """Range for the yaw angle of the goal pose, default to [-pi, pi]."""

    ranges: Ranges = MISSING
    """Distribution ranges for sampling the goal pose."""

    goal_term_name: str = "goal_pose"
    """The name of the goal term that this command is associated with. Defaults to "goal_pose"."""

    success_object_to_goal_dist_thresh_m: float = 0.10
    """Distance threshold (meters) for success-rate computation."""

    success_keypoint_angle_error_thresh_deg: float = 10.0
    """Keypoint yaw-angle threshold (degrees) for success-rate computation."""

    enable_yaw_curriculum: bool = True
    """Whether to progressively widen the sampled yaw range based on success rate."""

    curriculum_success_rate_threshold: float = 0.60
    """Mean success-rate threshold required to unlock the next yaw difficulty level."""

    curriculum_initial_yaw_range: tuple[float, float] = (0.0, 0.0)
    """Initial yaw range used when the curriculum starts."""

    curriculum_yaw_step: float = math.pi / 3.0
    """Yaw expansion applied on each side when advancing a curriculum level."""

    curriculum_max_yaw: float = math.pi
    """Maximum absolute yaw magnitude allowed by the curriculum."""

    debug_vis: bool = True
    debug_vis_keypoints: bool = True
    debug_vis_keypoint_radius: float = 0.04


@configclass
class ArmJointTrajectoryCommandCfg(CommandTermCfg):
    """Configuration for the uniform 3D orientation command term.

    Please refer to the :class:`InHandReOrientationCommand` class for more details.
    """

    class_type: type = ArmJointTrajectoryCommand

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""

    trajectory_time: tuple[float, float] = MISSING
    """Length of the trajectory in seconds."""

    hold_time: tuple[float, float] = MISSING
    """Length of the arm holding in positon in seconds."""

    joint_names: tuple[str, ...] = MISSING

    @configclass
    class Ranges:
        """Uniform distribution ranges for the gripper commands."""

        init_range: float = MISSING
        final_range: float = MISSING
        noise_range: float = MISSING

    ranges: Ranges = MISSING


@configclass
class LegJointTrajectoryCommandCfg(CommandTermCfg):
    """Configuration for the uniform 3D orientation command term.

    Please refer to the :class:`InHandReOrientationCommand` class for more details.
    """

    class_type: type = LegJointTrajectoryCommand

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""

    trajectory_time: tuple[float, float] = MISSING
    """Length of the trajectory in seconds."""

    hold_time: tuple[float, float] = MISSING
    """Length of the arm holding in positon in seconds."""

    leg_joint_names: tuple[str, ...] = MISSING

    @configclass
    class Ranges:
        """Uniform distribution ranges for the gripper commands."""

        init_range: float = MISSING
        final_range: float = MISSING
        noise_range: float = MISSING

    ranges: Ranges = MISSING


@configclass
class MultiLegJointTrajectoryCommandCfg(CommandTermCfg):
    """Configuration for the uniform 3D orientation command term.

    Please refer to the :class:`InHandReOrientationCommand` class for more details.
    """

    class_type: type = MultiLegJointTrajectoryCommand

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""

    trajectory_time: tuple[float, float] = MISSING
    """Length of the trajectory in seconds."""

    hold_time: tuple[float, float] = MISSING
    """Length of the arm holding in positon in seconds."""

    leg_joint_names: dict = MISSING

    no_command_leg_prob: float = MISSING
    """Probability of not command any leg."""

    @configclass
    class Ranges:
        """Uniform distribution ranges for the gripper commands."""

        init_range: float = MISSING
        final_range: float = MISSING
        noise_range: float = MISSING

    ranges: Ranges = MISSING


@configclass
class BasePoseCommandCfg(CommandTermCfg):
    """Configuration for the uniform 3D orientation command term.

    Please refer to the :class:`InHandReOrientationCommand` class for more details.
    """

    class_type: type = BasePoseCommand

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""

    @configclass
    class Ranges:
        """Uniform distribution ranges for the gripper commands."""

        roll: tuple[float, float] = MISSING
        pitch: tuple[float, float] = MISSING
        height: tuple[float, float] = MISSING
        noise_range: float = MISSING

    ranges: Ranges = MISSING


@configclass
class ArmLegJointBasePoseCommandCfg(CommandTermCfg):
    """Configuration for the uniform 3D orientation command term.

    Please refer to the :class:`InHandReOrientationCommand` class for more details.
    """

    class_type: type = ArmLegJointBasePoseCommand

    asset_name: str = "robot"
    """Name of the asset in the environment for which the commands are generated."""

    trajectory_time: tuple[float, float] = (1.0, 3.0)
    """Length of the trajectory in seconds."""

    hold_time: tuple[float, float] = (0.5, 2.0)
    """Length of the arm holding in positon in seconds."""

    arm_joint_names: tuple[str, ...] = MISSING
    leg_joint_names: dict = MISSING

    command_range_roll: tuple[float, float] = (-0.35, 0.35)
    command_range_pitch: tuple[float, float] = (-0.35, 0.35)
    command_range_height: tuple[float, float] = (0.25, 0.65)
    command_range_arm_joint: tuple[list[float, ...], list[float, ...]] = (
        [-2.61799, -3.14159, 0.0, -2.79252, -1.8326, -2.87988, -1.5708],
        [3.14157, 0.52359, 3.14158, 2.79252, 1.83259, 2.87979, 0.0],
    )
    command_range_leg_joint: tuple[list[float, ...], list[float, ...]] = (
        [-0.7854, -0.89884, -2.7929],
        [0.78539, 2.29511, 0.0],
    )
    """Command sample ranges."""

    command_which_leg: int = 4
    """Which leg to command: -1: no leg; [0, 1, 2, 3]: [FL, FR, HL, HR]; 4: all leg"""
