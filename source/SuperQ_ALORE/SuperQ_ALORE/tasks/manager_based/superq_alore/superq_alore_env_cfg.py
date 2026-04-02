# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from dataclasses import MISSING
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
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
##
# Pre-defined configs
##
from SuperQ_ALORE.assets.spot.spot import SPOT_ARM_CFG  # isort: skip
from SuperQ_ALORE.assets.spot.constants import ARM_JOINT_NAMES, LEG_JOINT_NAMES, FEET_NAMES
from SuperQ_ALORE.assets.spot.constants import GRASP_POSE_1_JOINT_POS
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
    
    # target objects to manipulate
    target_object = scene.CHAIR_RIGID_CFG
    # contact sensors
    # TODO: are they really... helpful?
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True
    )
    robot_to_ground_contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
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
    # TODO: modify it into the correct sampled command
    object_velocity = isaac_mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.05,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=isaac_mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-1.0, 1.0),
            ang_vel_z=(-1.0, 1.0),
            heading=(-math.pi, math.pi),
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
        # TODO: Add the other observations
        """Proprioceptive Data from robot"""
        base_lin_vel = ObsTerm(
            func=isaac_mdp.base_lin_vel, noise=Unoise(n_min=-0.01, n_max=0.01)
        )
        base_ang_vel = ObsTerm(
            func=isaac_mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2)
        )
        projected_gravity = ObsTerm(
            func=isaac_mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
        



    @configclass
    class LocomotionPolicyCfg(ObsGroup):
        """
        Observations for locomotion policy.
        
        This function summarizes all the observation inputs to ReLIC, so 
        that this low-level controller can perform normally
        
        """
        base_lin_vel = ObsTerm(
            func=isaac_mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1)
        )
        base_ang_vel = ObsTerm(
            func=isaac_mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2)
        )
        projected_gravity = ObsTerm(
            func=isaac_mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        
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
            func=isaac_mdp.joint_pos_rel, noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        joint_vel = ObsTerm(
            func=isaac_mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5)
        )
        actions = ObsTerm(func=mdp.last_leg_action, params={"action_term_name": "high_level_action"})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    # TODO: Define these configurations by borrowing codes from previous works
    # Now, remove them temporarily...
    
    # policy_deployable: PolicyDeployableCfg = PolicyDeployableCfg()
    # critic: CriticCfg = CriticCfg()
    # adapt_teacher: AdaptTeacherCfg = AdaptTeacherCfg()
    # adapt_student: AdaptStudentCfg = AdaptStudentCfg()
    locomotion_policy: LocomotionPolicyCfg = LocomotionPolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # reset
    # TODO: Reset the robot pose behind the target object
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
    reset_object = EventTerm(
        func=mdp.reset_target_object_position,
        mode="reset",
        params = {
            "asset_name": "target_object",
            "offset": (0.0, 0.0)
        }
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_around_grasp_pose,
        mode="reset",
        params={
            "position_range": (-0.0, 0.0),
            "velocity_range": (-0.0, 0.0),
            "joint_position_ref": GRASP_POSE_1_JOINT_POS,
        },
    )
    # reset_robot_joints = EventTerm(
    #     func=mdp.reset_joints_around_default,
    #     mode="reset",
    #     params={
    #         "position_range": (-0.0, 0.0),
    #         "velocity_range": (-0.0, 0.0),
    #     },
    # )

    
    # Move the object closer after a delay
    # move_object_delayed = EventTerm(
    #     func=mdp.move_target_object_closer,
    #     mode="interval",
    #     interval_range_s=(1.0, 1.0),
    #     params = {"asset_name": "target_object"},
    # )

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # # (1) Constant running reward
    # alive = RewTerm(func=mdp.is_alive, weight=1.0)
    # # (2) Failure penalty
    # terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)
    # # (3) Primary task: keep pole upright
    # pole_pos = RewTerm(
    #     func=mdp.joint_pos_target_l2,
    #     weight=-1.0,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]), "target": 0.0},
    # )
    # # (4) Shaping tasks: lower cart velocity
    # cart_vel = RewTerm(
    #     func=mdp.joint_vel_l1,
    #     weight=-0.01,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"])},
    # )
    # # (5) Shaping tasks: lower pole angular velocity
    # pole_vel = RewTerm(
    #     func=mdp.joint_vel_l1,
    #     weight=-0.005,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"])},
    # )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # # (2) Cart out of bounds
    # cart_out_of_bounds = DoneTerm(
    #     func=mdp.joint_pos_out_of_manual_limit,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]), "bounds": (-3.0, 3.0)},
    # )


##
# Environment configuration
##


@configclass
class SuperqAloreEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: SuperqAloreSceneCfg = SuperqAloreSceneCfg(num_envs=4096, env_spacing=4.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
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

        # Set up the initial state of robot
        # default_root_state = self.scene.robot.data.default_root_state.clone()
        
        # # Set robot positions
        # default_root_state[:, 0] = -0.5
        # default_root_state[:, 1] = 0.0
        # default_root_state[:, 2] = -0.2

        # self.scene.robot.write_root_state_to_sim(default_root_state)

        # TODO: Find a way to set up the robot's arm right at the time of initialization
        # (Or it can be achieved by a command from agent)
