# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from dataclasses import MISSING

import torch
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
from SuperQ_ALORE.assets.spot.constants import ARM_JOINT_NAMES, LEG_JOINT_NAMES, FEET_NAMES, SPOT_BODY_LINKS
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
    object_velocity = isaac_mdp.UniformVelocityCommandCfg(
        asset_name="target_object",
        resampling_time_range=(100.0, 100.0), # No need to change the command
        rel_standing_envs=0.0,
        rel_heading_envs=0.0,
        heading_command=False,
        heading_control_stiffness=1.0,
        debug_vis=True,
        # TODO: for objects of different frame conventions, we 
        # might need to consider different command ranges
        ranges=isaac_mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.0, 0.0),
            lin_vel_y=(-0.0, 0.5),
            ang_vel_z=(-0.5, 0.5),
            heading=(-math.pi, math.pi), # Not used if heading_command = False
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
            scale = 1.0
        ) # dim: 2 (no yaw information)
        
        # Root angular velocity
        base_ang_vel = ObsTerm(
            func=isaac_mdp.base_ang_vel, noise=Unoise(n_min=-0.0, n_max=0.0),
            scale = 0.25
        ) # dim: 3, base_ang_vel is in robot's root frame
        
        # Last action (x, y, omega, \delta arm joints)
        last_action = ObsTerm(
            func = mdp.last_high_level_action, params={"clip_limit": 100}
        ) # dim: 9
        
        # Commands (x, y, omega)
        commands = ObsTerm(
            func=isaac_mdp.generated_commands,
            params={"command_name": "object_velocity"},
        ) # dim: 3 # TODO: Uncomment this term. Now, due to a zero agent, this term is not applicable
        
        # End-effector in robot frame
        ee_pose_in_robot_frame = ObsTerm(
            func = mdp.ee_pose_in_robot_frame,
            params = {"end_effector_link_name": "arm_link_jaw"},
            scale = 1.0,
        ) # dim: 7 (position + quat) for the end-effector link
        
        # Object pose in robot frame
        obj_pose_in_robot_frame = ObsTerm(
            func = mdp.obj_pose_in_robot_frame,
            scale = 1.0,
        ) # dim: 7 (position + quat) for the target object
        
        # Category code? (TODO: Clarify this... it's constant zero in ALORE)
        category_encode = ObsTerm(
            func = mdp.category_encode,
            scale = 1.0,
        ) # dim 3, one-hot encoding for object category, not used in ALORE so just return zeros

        def __post_init__(self):
            self.enable_corruption = False
            self.history_length = 10
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
        
        # Robot base orientation
        body_orientation = ObsTerm(
            func = mdp.get_body_orientation,
            scale = 1.0
        ) # dim: 2 (no yaw information)
        
        
        # Robot base angular velocity
        base_ang_vel = ObsTerm(
            func=isaac_mdp.base_ang_vel, noise=Unoise(n_min=-0.0, n_max=0.0),
            scale = 0.25
        ) # dim: 3, base_ang_vel is in robot's root frame
        
        # Last action (x, y, omega, \delta arm joints)
        last_action = ObsTerm(
            func = mdp.last_high_level_action, params={"clip_limit": 100.0}
        ) # dim: 9
        
        # Commands
        commands = ObsTerm(
            func=isaac_mdp.generated_commands,
            params={"command_name": "object_velocity"},
        ) # dim: 3 # TODO: Uncomment this term. Now, due to a zero agent, this term is not applicable
        
        # Link pose in robot frame
        link_pose_in_robot_frame = ObsTerm(
            func = mdp.link_pose_in_robot_frame,
            params = {"link_names": 
                        ["arm_link_sh0",
                        "arm_link_sh1",
                        "arm_link_el0",
                        "arm_link_el1",
                        "arm_link_wr0",
                        "arm_link_wr1",
                        "arm_link_fngr",
                        "arm_link_jaw"]
                    },
            scale = 1.0,
        ) # dim: 8 *  7 (position + quat) for the arm links
         
        # End-effector contact state
        ee_contact_state = ObsTerm(
            func = mdp.ee_contact_state,
            params = {"contact_sensor_name": "contact_forces"},
            scale = 1.0,
        ) # dim: 1, binary contact state for the end-effector
        
        # Object pose in robot frame
        obj_pose_in_robot_frame = ObsTerm(
            func = mdp.obj_pose_in_robot_frame,
            scale = 1.0,
        ) # dim: 7 (position + quat) for the target object
        
        # Robot base velocity
        base_lin_vel = ObsTerm(
            func=isaac_mdp.base_lin_vel, noise=Unoise(n_min=-0.0, n_max=0.0),
            scale = 2.0,
        ) # dim: 3
        
       # Object velocity in robot frame
        obj_lin_vel_in_robot_frame = ObsTerm(
            func = mdp.obj_lin_vel_in_robot_frame,
            scale = 2.0,
        ) # dim: 3
        
        # Object angular velocity in robot frame
        obj_ang_vel_in_robot_frame = ObsTerm(
            func = mdp.obj_ang_vel_in_robot_frame,
            scale = 0.25,
        ) # dim: 3
        
        # Object physical properties (Static fric, mass, dynamic fric)
        obj_physical_properties = ObsTerm(
            func = mdp.obj_physical_properties,
            params={"object_name": "target_object"},
            scale = 1.0,
        ) # dim: 3
        
        def __post_init__(self):
            self.enable_corruption = False
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
    class RewardCalculationCfg(ObsGroup):
        """
        Observations for the reward calculations.
        This is going to be used by reward functions
        
        """
        object_velocity = ObsTerm(
            func=mdp.object_velocity,
            history_length = 2,
            flatten_history_dim = False,
            params={"asset_name": "target_object"},
        ) # dim: 3
        applied_actions = ObsTerm(
            func = mdp.last_high_level_action, 
            history_length = 2,
            flatten_history_dim = False,
            params={"clip_limit": 100.0},
        ) # dim: 9
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False
    policy: PolicyCfg = PolicyCfg()
    
    # policy_deployable: PolicyDeployableCfg = PolicyDeployableCfg()
    critic: CriticCfg = CriticCfg()
    # adapt_teacher: AdaptTeacherCfg = AdaptTeacherCfg()
    # adapt_student: AdaptStudentCfg = AdaptStudentCfg()
    locomotion_policy: LocomotionPolicyCfg = LocomotionPolicyCfg()
    reward_calculation: RewardCalculationCfg = RewardCalculationCfg()
    
    


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
    reset_object = EventTerm(
        func=mdp.reset_target_object_position,
        mode="reset",
        params = {
            "asset_name": "target_object",
            "offset": (0.0, 0.0)
        }
    )
    
    # TODO: figure out how to reset the initial grasp poses at different locations
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

    ## (1) Group 1: Object related rewards (primary task)
    # TODO: UUNCOMMENT THEM!!
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=5.0,
        params={"command_name": "object_velocity"},
    ) # Track the xy linear velocity of th object according to the command
    
    track_ang_vel_yaw_exp = RewTerm(
        func=mdp.track_ang_vel_yaw_exp,
        weight=5.0,
        params={"command_name": "object_velocity"},
    ) # Track the yaw angular velocity of the object according to the command
    
    is_alive = RewTerm(func=mdp.is_alive, weight=1.0) # The manipulation process should be alive
    
    
    ## Smooth movement of the object
    yaw_alignment = RewTerm(
        func=mdp.yaw_alignment_reward, 
        weight=5.0,
        params={
            "asset_name": "target_object",
            "robot_name": "robot",
        },
    ) # Align the yaw of the object with the desired direction (if applicable)
    
    lin_vel_z_l2 = RewTerm(
        func=mdp.lin_vel_z_l2,
        weight=2.0,
        params = {"asset_name": "target_object"}
    ) # Penalize the vertical velocity of the object to encourage it to stay on the ground
    
    ang_vel_xy_l2 = RewTerm(
        func=mdp.ang_vel_xy_l2,
        weight=0.05,
        params = {"asset_name": "target_object"}
    ) # Penalize the angular velocity in x and y axes to encourage the object not to topple
    
    flat_orientation_l2 = RewTerm(
        func=mdp.flat_orientation_l2,
        weight=10.0,
        params = {"asset_name": "target_object"}
    ) # Encourage the object to maintain a flat orientation (if applicable)

    lin_vel_change_penalty = RewTerm(
        func=mdp.lin_vel_change_penalty,
        weight=2.0,
        params = {"asset_name": "target_object"}
    ) # Penalize the change in linear velocity of the object to encourage smooth motion
    
    ang_vel_change_penalty = RewTerm(
        func=mdp.ang_vel_change_penalty,
        weight=2.0,
        params = {"asset_name": "target_object"}
    ) # Penalize the change in angular velocity of the object to encourage smooth motion
    
    # TODO: the distance may be tricky
    distance_penalty = RewTerm(
        func=mdp.distance_penalty,
        weight=10.0,
        params={
            "robot_name": "robot",
            "end_effector_link_name": "arm_link_jaw",
            "distance_threshold": 0.6,
        },
    ) # Penalize the distance between the end-effector and the robot
    
    ## Group 2: Robot related rewards (auxiliary task, to encourage better robot behavior)
    action_rate_l2 = RewTerm(
        func=mdp.action_rate_l2,
        weight=0.01,
    ) # Penalize large instantaneous changes in the action output to encourage smoother motion
    action_rate2_l2 = RewTerm(
        func=mdp.action_rate2_l2,
        weight=0.002,
    ) # Penalize large instantaneous changes in the action changes to encourage smoother motion
    
    joint_torques = RewTerm(
        func=mdp.joint_torques,
        weight=2.5e-7,
        params={"arm_joint_names": ARM_JOINT_NAMES, "robot_name": "robot"},
    ) # Penalize the torques in the arm joints to encourage energy-efficient motion
    
    joint_accel = RewTerm(
        func=mdp.joint_accel,
        weight=2.5e-7,
        params={"arm_joint_names": ARM_JOINT_NAMES, "robot_name": "robot"},
    ) # Penalize the acceleration in the arm joints to encourage smoother motion
    
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
            # TODO: Adapt it to multiple grasp poses
            "reference_joint_positions": GRASP_POSE_1_JOINT_POS
        },
    ) # Penalize the deviation of joint positions from the reference initial grasp pose to encourage a more natural pose
    
    undesired_contact_penalty = RewTerm(
        func=mdp.undesired_contact_penalty,
        weight=5.0,
        params={
            "undesired_contact_body_names": SPOT_BODY_LINKS,  # Replace with actual body names
            "contact_sensor_name": "contact_forces",
            "undesired_contact_threshold": 1.0,
        },
    ) # Penalize undesired contacts between the robot and the ground to encourage the robot to
@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    
    # (2) Terminate if illegal contact happens
    # Reset the environment if too large action / velocities are detected
    physics_explosion = DoneTerm(
        func=mdp.outlier_detected,
        params={"threshold": 1000.0} 
    )
    base_contact = DoneTerm(
        func=isaac_mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["body"]),
            "threshold": 2.0,
        },
    )
    undesired_ground_contact = DoneTerm(
        func=mdp.illegal_ground_contact,
        params={
            "sensor_cfg": SceneEntityCfg(
                "robot_to_ground_contact_forces", body_names=[".*leg"]
            ),
            "threshold": 1.0,
        },
    )
    
    # Terminate if any joint velocity exceeds 50.0 rad/s
    aggressive_joint_velocity = DoneTerm(
        func=mdp.joint_velocity_limits,
        params={"max_vel": 30.0},
    )

    # (3) Terminate if the object falls off the gripper
    object_slide_off = DoneTerm(
        func=mdp.object_slide_off,
        params={
            "contact_sensor_name": "contact_forces",
            "gripper_links_names": ["arm_link_fngr", "arm_link_jaw"],
        },
    )

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
        
        
    # Create a new buffer to store the previous object velocities & actions
    def _pre_physics_step(self, action):
        # Cache the current velocity before it gets updated
        prev_obj_lin_vel = self.scene["target_object"].data.root_lin_vel_b.clone()[:, :2]
        prev_obj_ang_vel = self.scene["target_object"].data.root_ang_vel_b.clone()[:, 2]
        self.prev_obj_vel = torch.cat([prev_obj_lin_vel, prev_obj_ang_vel.unsqueeze(-1)], dim=-1)
    
    # After the physics step, update prev_action and prev_prev_action for the next step
    def _post_physics_step(self, action):
        # Update the observed actions as well
        self.prev_prev_action = getattr(self, "prev_action", torch.zeros_like(action))
        self.prev_action = action.clone()

