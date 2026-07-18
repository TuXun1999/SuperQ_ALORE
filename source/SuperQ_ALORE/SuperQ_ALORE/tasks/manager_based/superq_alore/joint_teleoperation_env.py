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
class SuperqAloreSceneCfg(InteractiveSceneCfg):
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
    target_object: RigidObjectCfg = OBJECT_TELEOPERATION_INFO[0]
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
    high_level_action = mdp.MixedPDArmMultiLegJointPositionActionTeleCfg(
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
        """Observations for critic."""
        # Joint velocities & positions
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.0, n_max=0.0),
            scale = 1.0
        ) # dim: 18
        joint_vel = ObsTerm(
            func=mdp.joint_vel, noise=Unoise(n_min=-0.0, n_max=0.0),
            scale = 0.05
        ) # dim: 18
        
        # Default joint positions
        default_joint_pos = ObsTerm(
            func=mdp.default_joint_pos, noise=Unoise(n_min=-0.0, n_max=0.0),
            scale=1.0
        ) # dim: 18
        # Robot joint positions (absolute, not relative to default pose)
        joint_pos_abs = ObsTerm(
            func = mdp.joint_pos, noise=Unoise(n_min=-0.0, n_max=0.0),
            scale=1.0
        ) # dim: 18
        
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            
    critic: CriticCfg = CriticCfg()
    locomotion_policy: LocomotionPolicyCfg = LocomotionPolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # reset
    reset_base = EventTerm(
        func=isaac_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (0.0, 0.0),
                "y": (-0.0, 0.0),
                "z": (-0.0, -0.0),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
            "velocity_range": {
                "x": (-0.0, 0.0),
                "y": (-0.0, 0.0),
                "z": (-0.0, 0.0),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-0.0, 0.0),
            },
        },
    )
    # reset the object to the assigned pose
    reset_object = EventTerm(
        func=mdp.reset_target_object_pose,
        mode="reset",
        params = {
            "asset_name": "target_object",
            "offset": OBJECT_TELEOPERATION_INFO[2],  # use the initial pose of the object in the catalog
            "rotation": OBJECT_TELEOPERATION_INFO[3],  # use the initial pose of the object in the catalog
        }
    )
    
    # Reset active object + robot consistently from sampled catalog pose.
    reset_object_and_robot_teleop = EventTerm(
        func=mdp.reset_joints_around_grasp_pose,
        mode="reset",
        params={
            "position_range": (-0.0, 0.0),
            "velocity_range": (-0.0, 0.0),
            "joint_position_ref": OBJECT_TELEOPERATION_INFO[1],  # use the joint positions corresponding to the initial pose of the object
            "asset_cfg": SceneEntityCfg("robot"),
        }
    )

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""


@configclass
class JointTeleoperationEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: SuperqAloreSceneCfg = SuperqAloreSceneCfg(num_envs=1, env_spacing=4.0)
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

