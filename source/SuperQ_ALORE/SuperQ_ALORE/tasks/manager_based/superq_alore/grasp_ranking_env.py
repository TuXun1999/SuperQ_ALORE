# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from dataclasses import MISSING

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.terrains import TerrainImporterCfg
from . import mdp
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as isaac_mdp
from isaaclab.actuators import ActuatorNetMLPCfg, DCMotorCfg, ImplicitActuatorCfg
from SuperQ_ALORE.tasks.manager_based.superq_alore.mdp.scene import OBJECT_TELEOPERATION_INFO
##
# Pre-defined configs
##
from SuperQ_ALORE.assets.spot.spot import SPOT_ARM_CFG  # isort: skip
from SuperQ_ALORE.assets.spot.constants import ARM_JOINT_NAMES, LEG_JOINT_NAMES, FEET_NAMES, SPOT_BODY_LINKS
import SuperQ_ALORE.tasks.manager_based.superq_alore.mdp.scene as scene
##
# Scene definition
##


@configclass
class GraspRankingSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground plane
    # ground = AssetBaseCfg(
    #     prim_path="/World/ground",
    #     spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    # )
    # TODO: adopt the previous style of grid ground
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )

    # robots
    robot: ArticulationCfg = MISSING
    
    # object
    target_object_0: RigidObjectCfg = MISSING
    # contact sensors
    # TODO: are they really... helpful?
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, update_period=0.005, track_air_time=True
    )
    robot_to_ground_contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        update_period=0.005,
        track_air_time=True,
        filter_prim_paths_expr=["/World/ground/terrain/mesh"],
    )
    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
    )
    



##
# MDP settings
##
@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

@configclass
class CommandsEvalCfg:
    """Command specifications for the evaluation environment."""
    goal_pose = mdp.GoalPoseCommandPLAYCfg(
        resampling_time_range=(1e6, 1e6), # No need to change the command
        debug_vis=True,
        debug_vis_keypoints=True,
        debug_vis_keypoint_radius=0.04,
        enable_yaw_curriculum=False,
        ranges=mdp.GoalPoseCommandCfg.Ranges(
            pos_x=(-1.0, 1.0),
            pos_y=(-1.0, 1.0),
            pos_z=(0.0, 0.0),
            yaw=(-math.pi/2, math.pi/2),
        ),
    )
@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    
    ## Execute actions predicted from the high-level controller / agent
    """
    Actions: input to the sim environment, output from the agent
    In our design, the raw output from the high-level controller will be 
    a_{high} = (arm joints, base pose, base velocities)
    This is going to be the low-level command to track
    
    Our low-level controller, ReLIC, is originally trained to track the command
    c_{low} = (arm joints, base pose, base velocities)
    as long as all four legs are used and no leg joint tracking is enabled
    The output from ReLIC will be
    a_{low} = (arm joints, leg joints)
    
    So, we need to create an input for the low-level actor from obs & actions
    I.e. substitute the previous command obs in ReLIC with the action from high-level controller
    NOTE: Order is important!!
    
    """
    # This configuration `high_level_action` is defining an action specification for the MDP (Markov
    # Decision Process).
    high_level_action = mdp.MixedPDArmMultiLegJointPositionActionCfg(
        asset_name="robot",
        joint_names=["[fh].*"],
        command_name="arm_leg_joint_base_pose",
        arm_joint_names=ARM_JOINT_NAMES,
        leg_joint_names=LEG_JOINT_NAMES,
        scale=0.2,
    )

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""
    
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for the Actor / Policy agent of the high-level controller"""
        # Joint velocities & positions
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.0, n_max=0.0),
            scale = 1.0
        ) # dim: 18 (12 legs + 7 arm joints - 1 redundant joint) --- relative joint positions to the default pose
        joint_vel = ObsTerm(
            func=mdp.joint_vel, noise=Unoise(n_min=-0.0, n_max=0.0),
            scale = 0.05
        ) # dim: 18
        
        # Body orientation data
        body_orientation = ObsTerm(
            func = mdp.get_body_orientation,
            noise=Unoise(n_min=-0.1, n_max=0.1),
            scale = 1.0
        ) # dim: 2 (no yaw information)
        
        # Root angular velocity
        base_ang_vel = ObsTerm(
            func=isaac_mdp.base_ang_vel, noise=Unoise(n_min=-0.0, n_max=0.0),
            scale = 0.25
        ) # dim: 3, base_ang_vel is in robot's root frame
        
        # Last action (x, y, omega, \delta arm joints)
        last_action = ObsTerm(
            func = mdp.last_high_level_action, params={"clip_limit": 100},
            scale = 1.0,
        ) # dim: 9
        
        # End-effector in robot frame
        ee_pose_in_robot_frame = ObsTerm(
            func = mdp.ee_pose_in_robot_frame,
            params = {"end_effector_link_name": "arm_link_jaw"},
            scale = 1.0,
        ) # dim: 7 (position + quat) for the end-effector link

        # Object pose in robot frame
        obj_pose_in_robot_frame = ObsTerm(
            func = mdp.obj_pose_in_robot_frame_SE2_grasp_ranking,
            scale = 1.0,
        ) # dim: 3 (position + yaw) for the target object, SE2
        
        # Redundant placeholders to load the pretrained policy correctly in dimension
        redundant_placeholders = ObsTerm(
            func = mdp.redundant_placeholders_actor,
            scale = 1.0,
        ) # dim: 3 (placeholders for the missing dimensions in the pretrained policy)
        
        def __post_init__(self):
            self.enable_corruption = False
            self.history_length = 1
            self.concatenate_terms = True

    @configclass
    class LocomotionPolicyCfg(ObsGroup):
        """
        Observations for locomotion policy.
        
        This function summarizes all the observation inputs to ReLIC, so 
        that this low-level controller can perform normally
        
        """
        base_lin_vel = ObsTerm(
            func=isaac_mdp.base_lin_vel, noise=Unoise(n_min=-0.0, n_max=0.0)
        ) # dim: 3
        base_ang_vel = ObsTerm(
            func=isaac_mdp.base_ang_vel, noise=Unoise(n_min=-0.0, n_max=0.0)
        ) # dim: 3, base_ang_vel is in robot's root frame
        projected_gravity = ObsTerm(
            func=isaac_mdp.projected_gravity,
            noise=Unoise(n_min=-0.0, n_max=0.0),
        ) # dim: 3, projected gravity in robot's root frame
        
        # NOTE: the commands used to train ReLIC are no longer commands for
        # high-level controller. We need to obtain the base velocity & joint pose
        # to track from the predicted action from the high-level agent
        # velocity_commands = ObsTerm(
        #     func=isaac_mdp.generated_commands, params={"command_name": "base_velocity"}
        # )
        # commands = ObsTerm(
        #     func=isaac_mdp.generated_commands,
        #     params={"command_name": "arm_leg_joint_base_pose"},
        # )
        joint_pos = ObsTerm(
            func=isaac_mdp.joint_pos_rel, noise=Unoise(n_min=-0.0, n_max=0.0)
        ) # dim: 19
        joint_vel = ObsTerm(
            func=isaac_mdp.joint_vel_rel, noise=Unoise(n_min=-0.0, n_max=0.0)
        ) # dim: 19
        actions = ObsTerm(func=mdp.last_leg_action, params={"action_term_name": "high_level_action"})
        # dim: 12
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
    
    @configclass
    class CriticCfg(ObsGroup):
        redundant_placeholders = ObsTerm(
            func = mdp.redundant_placeholders_critic,
            scale = 1.0,
        ) # dim: 154 (placeholders for the missing dimensions in the pretrained policy)
        def __post_init__(self):
            self.enable_corruption = False
            self.history_length = 1
            self.concatenate_terms = True
    
    policy: PolicyCfg = PolicyCfg()
    locomotion_policy: LocomotionPolicyCfg = LocomotionPolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class ObservationsEvalCfg:
    """Observation specifications for the evaluation MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for the Actor / Policy agent of the high-level controller."""
        # Joint velocities & positions
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.0, n_max=0.0),
            scale=1.0
        )  # dim: 18 (12 legs + 7 arm joints - 1 redundant joint)
        joint_vel = ObsTerm(
            func=mdp.joint_vel, noise=Unoise(n_min=-0.0, n_max=0.0),
            scale=0.05
        )  # dim: 18

        # Body orientation data
        body_orientation = ObsTerm(
            func=mdp.get_body_orientation,
            noise=Unoise(n_min=-0.1, n_max=0.1),
            scale=1.0
        )  # dim: 2 (no yaw information)

        # Root angular velocity
        base_ang_vel = ObsTerm(
            func=isaac_mdp.base_ang_vel, noise=Unoise(n_min=-0.0, n_max=0.0),
            scale=0.25
        )  # dim: 3, base_ang_vel is in robot's root frame

        # Last action (x, y, omega, delta arm joints)
        last_action = ObsTerm(
            func=mdp.last_high_level_action, params={"clip_limit": 100},
            scale=1.0,
        )  # dim: 9

        # End-effector in robot frame
        ee_pose_in_robot_frame = ObsTerm(
            func=mdp.ee_pose_in_robot_frame,
            params={"end_effector_link_name": "arm_link_jaw"},
            scale=1.0,
        )  # dim: 7 (position + quat) for the end-effector link

        # Object pose in robot frame
        obj_pose_in_robot_frame = ObsTerm(
            func=mdp.obj_pose_in_robot_frame_SE2_grasp_ranking,
            scale=1.0,
        )  # dim: 3 (position + yaw) for the target object, SE2

        # Vector from active object to goal in active object frame.
        obj_to_goal_pos_local = ObsTerm(
            func=mdp.obj_to_goal_pos_local,
            params={"goal_term_name": "goal_pose"},
            noise=Unoise(n_min=-0.02, n_max=0.02),
            scale=1.0,
        )  # dim: 2 (xy components in active object frame)

        # Goal orientation represented in active object frame as a yaw angle.
        obj_to_goal_rot_local = ObsTerm(
            func=mdp.obj_to_goal_rot_local,
            params={"goal_term_name": "goal_pose"},
            noise=Unoise(n_min=-0.01, n_max=0.01),
            scale=1.0,
        )  # dim: 1 (yaw error in active object frame)

        def __post_init__(self):
            self.enable_corruption = False
            self.history_length = 1
            self.concatenate_terms = True

    @configclass
    class LocomotionPolicyCfg(ObsGroup):
        """Observations for locomotion policy."""
        base_lin_vel = ObsTerm(
            func=isaac_mdp.base_lin_vel, noise=Unoise(n_min=-0.0, n_max=0.0)
        )  # dim: 3
        base_ang_vel = ObsTerm(
            func=isaac_mdp.base_ang_vel, noise=Unoise(n_min=-0.0, n_max=0.0)
        )  # dim: 3, base_ang_vel is in robot's root frame
        projected_gravity = ObsTerm(
            func=isaac_mdp.projected_gravity,
            noise=Unoise(n_min=-0.0, n_max=0.0),
        )  # dim: 3, projected gravity in robot's root frame
        joint_pos = ObsTerm(
            func=isaac_mdp.joint_pos_rel, noise=Unoise(n_min=-0.0, n_max=0.0)
        )  # dim: 19
        joint_vel = ObsTerm(
            func=isaac_mdp.joint_vel_rel, noise=Unoise(n_min=-0.0, n_max=0.0)
        )  # dim: 19
        actions = ObsTerm(func=mdp.last_leg_action, params={"action_term_name": "high_level_action"})
        # dim: 12

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        redundant_placeholders = ObsTerm(
            func=mdp.redundant_placeholders_critic,
            scale=1.0,
        )  # dim: 154 (placeholders for missing dimensions in pretrained policy)

        def __post_init__(self):
            self.enable_corruption = False
            self.history_length = 1
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    locomotion_policy: LocomotionPolicyCfg = LocomotionPolicyCfg()
    critic: CriticCfg = CriticCfg()

@configclass
class EventCfg:
    """Configuration for events."""

    # Reset the object pose and the robot pose
    reset_object_robot_pose = EventTerm(
        func=mdp.reset_object_robot_pose_grasp_ranking,
        mode="reset",
        params={
            "object_idx": 0,
            "pose_idx": 0,
        },
    )
    
    # Resample the mass & friction & com of the object
    reset_object_physical_properties = EventTerm(
        func=mdp.reset_object_physical_properties_grasp_ranking,
        mode="reset",
        params={
            "mass_range": (5, 8),
            "friction_range": (0.15, 0.35),
            "com_range": {
                "x": (-0.15, 0.15),
                "y": (-0.0, 0.0),
                "z": (-0.15, 0.15),
            },
        },
    )

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    
@configclass
class RewardsEvalCfg:
    """Reward terms for the MDP."""
    """
    Section I: Task specific Rewards
    """
    sparse_completion = RewTerm(
        func=mdp.sparse_completion_reward,
        weight=10.0,
        params={
            "goal_term_name": "goal_pose",
            "dist_error": 0.05,
            "angular_error": 5.0,
            "success_reward": 1.0,
        },
    ) # Sparse success bonus when object-goal position and yaw errors are both within threshold

    keypoint_pose_match_exp = RewTerm(
        func=mdp.keypoint_pose_match_exp,
        weight=8.0,
        params={
            "goal_term_name": "goal_pose",
            "sigma": 1.0,
        },
    ) # Encourage object-goal pose matching using world-frame keypoint distance

    vel_toward_goal = RewTerm(
        func=mdp.velocity_toward_goal_exp,
        weight=1.0,
        params={
            "goal_term_name": "goal_pose",
            "sigma": 0.7071067812,
            "use_unit_vel": True,
            "use_xy": True,
        },
    ) # Encourage object velocity to align with the direction from object to goal
    
    is_alive = RewTerm(func=mdp.is_alive, weight=5.0) # The manipulation process should be alive

    """
    Section II: Smooth motion rewards
    """
    lin_vel_change_penalty = RewTerm(
        func=mdp.lin_vel_change_penalty,
        weight=2.0,
    ) # Penalize the change in linear velocity of the object to encourage smooth motion
    
    ang_vel_change_penalty = RewTerm(
        func=mdp.ang_vel_change_penalty,
        weight=2.0,
    ) # Penalize the change in angular velocity of the object to encourage smooth motion
    
    flat_orientation_l2 = RewTerm(
        func=mdp.flat_orientation_l2,
        weight=-10.0,
        params={
            "flat_threshold": 1e-3,
        },
    ) # Indicator penalty: subtract 1 when object is non-flat

    joint_positions_wrt_reference = RewTerm(
        func=mdp.joint_positions_wrt_reference,
        weight=5.0,
        params={
            "arm_joint_names": [
                "arm_sh0",
                "arm_sh1",
                "arm_el0",
                "arm_el1",
                "arm_wr0",
                "arm_wr1",], 
            "robot_name": "robot",
        },
    ) # Penalize the deviation of joint positions from the per-env active grasp pose reference
    
    undesired_contact_penalty = RewTerm(
        func=mdp.undesired_contact_penalty,
        weight=7.0,
        params={
            "undesired_contact_body_names": SPOT_BODY_LINKS,  # Replace with actual body names
            "contact_sensor_name": "contact_forces",
            "undesired_contact_threshold": 1.0,
        },
    ) # Penalize undesired contacts between the robot and the ground to encourage the robot to
@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""


@configclass
class GraspRankingEnvPlayCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: GraspRankingSceneCfg = GraspRankingSceneCfg(num_envs=1, env_spacing=4.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 1 / 200
        self.sim.render_interval = self.decimation
        
        # Import the robot (behind the chair)
        self.scene.robot = SPOT_ARM_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.joint_drive.gains.stiffness = None

        # Import the target object
        self.scene.target_object_0 = OBJECT_TELEOPERATION_INFO[0].replace(prim_path="{ENV_REGEX_NS}/TargetObject0")
@configclass
class GraspRankingEnvEvalCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: GraspRankingSceneCfg = GraspRankingSceneCfg(num_envs=20, env_spacing=4.0)
    # Basic settings
    observations: ObservationsEvalCfg = ObservationsEvalCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    commands: CommandsEvalCfg = CommandsEvalCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 1 / 200
        self.sim.render_interval = self.decimation
        
        # Import the robot
        self.scene.robot = SPOT_ARM_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.joint_drive.gains.stiffness = None
        
        # Import the target object
        self.scene.target_object_0 = OBJECT_TELEOPERATION_INFO[0].replace(prim_path="{ENV_REGEX_NS}/TargetObject0")