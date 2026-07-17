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
    target_object_0: RigidObjectCfg = OBJECT_TELEOPERATION_INFO[0]
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
            "mass_range": (11, 12),
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

