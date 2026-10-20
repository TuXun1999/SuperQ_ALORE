"""Training configuration for the articulated QiVi office-chair task.

This module is intentionally parallel to ``superq_alore_env_cfg.py``.  It does
not alter the rigid-object scene, catalog spawning, rewards, or registration.
"""

from dataclasses import MISSING
import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from SuperQ_ALORE.assets.spot.spot import SPOT_ARM_CFG
from SuperQ_ALORE.tasks.manager_based.superq_alore import mdp
from SuperQ_ALORE.tasks.manager_based.superq_alore.mdp import scene as scene_mdp
from SuperQ_ALORE.tasks.manager_based.superq_alore.mdp import articulated_chair as chair_mdp
from SuperQ_ALORE.tasks.manager_based.superq_alore.mdp.commands.articulated_chair_goal_pose_command import (
    ArticulatedChairGoalPoseCommandCfg,
)

from .superq_alore_env_cfg import (
    ActionsCfg,
    CommandsCfg,
    EventCfg,
    ObservationsCfg,
    RewardsCfg,
    TerminationsCfg,
)


@configclass
class ArticulatedChairSceneCfg(InteractiveSceneCfg):
    """Scene containing one free-base QiVi articulation per environment."""

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

    robot: ArticulationCfg = MISSING
    target_object: ArticulationCfg = scene_mdp.create_articulated_target_object_cfg()

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, update_period=0.005, track_air_time=False
    )
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
    )


@configclass
class ArticulatedChairCommandsCfg(CommandsCfg):
    """Goal commands expressed at the chair's upper ``frame`` link."""

    goal_pose = ArticulatedChairGoalPoseCommandCfg(
        resampling_time_range=(1e6, 1e6),
        enable_yaw_curriculum=False,
        curriculum_iterations_per_level=1000,
        curriculum_initial_yaw_range=(0.0, 0.0),
        curriculum_yaw_step=math.pi / 18.0,
        curriculum_max_yaw=math.pi / 2.0,
        # The chair command's debug visualizer is a non-physical hollow green
        # ground square plus a heading arrow at the target ``frame``-link pose.
        # Keep it off during vectorized training; enable it for an interactive,
        # small-env debug run.
        debug_vis=True,
        debug_vis_keypoints=False,
        debug_vis_keypoint_radius=0.04,
        ranges=ArticulatedChairGoalPoseCommandCfg.Ranges(
            pos_x=(-1.0, 1.0),
            pos_y=(-1.0, 1.0),
            pos_z=(0.0, 0.0),
            yaw=(-math.pi / 2.0, math.pi / 2.0),
        ),
    )


@configclass
class ArticulatedChairObservationsCfg(ObservationsCfg):
    """Base observations with every object-state term redirected to ``frame``."""

    def __post_init__(self) -> None:
        self.policy.obj_pose_in_robot_frame = ObsTerm(
            func=chair_mdp.upper_body_pose_in_robot_frame_se2, scale=1.0
        )
        self.policy.obj_to_goal_pos_local = ObsTerm(
            func=chair_mdp.upper_body_to_goal_pos_local,
            params={"goal_term_name": "goal_pose"},
            noise=Unoise(n_min=-0.02, n_max=0.02),
            scale=1.0,
        )
        self.policy.obj_to_goal_rot_local = ObsTerm(
            func=chair_mdp.upper_body_to_goal_rot_local,
            params={"goal_term_name": "goal_pose"},
            noise=Unoise(n_min=-0.01, n_max=0.01),
            scale=1.0,
        )

        self.grasp_ranking.obj_init_pose_in_robot_frame = ObsTerm(
            func=chair_mdp.upper_body_init_pose_in_robot_frame_se2, scale=1.0
        )
        self.grasp_ranking.obj_to_goal_pos_local = ObsTerm(
            func=chair_mdp.upper_body_to_goal_pos_local,
            params={"goal_term_name": "goal_pose"},
            noise=Unoise(n_min=-0.02, n_max=0.02),
            scale=1.0,
        )
        self.grasp_ranking.obj_to_goal_rot_local = ObsTerm(
            func=chair_mdp.upper_body_to_goal_rot_local,
            params={"goal_term_name": "goal_pose"},
            noise=Unoise(n_min=-0.01, n_max=0.01),
            scale=1.0,
        )

        self.critic.obj_pose_in_robot_frame = ObsTerm(
            func=chair_mdp.upper_body_pose_in_robot_frame_se2, scale=1.0
        )
        self.critic.obj_to_goal_pos_local = ObsTerm(
            func=chair_mdp.upper_body_to_goal_pos_local,
            params={"goal_term_name": "goal_pose"},
            noise=Unoise(n_min=0.0, n_max=0.0),
        )
        self.critic.obj_to_goal_rot_local = ObsTerm(
            func=chair_mdp.upper_body_to_goal_rot_local,
            params={"goal_term_name": "goal_pose"},
            noise=Unoise(n_min=0.0, n_max=0.0),
        )
        self.critic.obj_lin_vel_in_body_frame = ObsTerm(
            func=chair_mdp.upper_body_lin_vel_in_body_frame, scale=2.0
        )
        self.critic.obj_ang_vel_in_body_frame = ObsTerm(
            func=chair_mdp.upper_body_ang_vel_in_body_frame, scale=0.25
        )
        self.critic.obj_physical_properties = ObsTerm(
            func=chair_mdp.upper_body_physical_properties, scale=1.0
        )

        self.reward_calculation.object_velocity = ObsTerm(
            func=chair_mdp.upper_body_velocity,
            history_length=2,
            flatten_history_dim=False,
        )
        self.object_idx.object_idx = ObsTerm(func=chair_mdp.articulated_chair_object_idx)
        self.com_estimation.obj_pose_in_robot_frame = ObsTerm(
            func=chair_mdp.upper_body_pose_in_robot_frame_se2, scale=1.0
        )


@configclass
class ArticulatedChairEventCfg(EventCfg):
    """Reset the chair articulation while retaining YAML-driven robot pre-grasps."""

    reset_object_and_robot = EventTerm(
        func=chair_mdp.reset_articulated_chair_and_robot_from_catalog_pose,
        mode="reset",
        params={
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )
    # Mass, CoM, and materials are authored individually for all 19 links in
    # the chair USD.  The rigid-object randomizer assumes one rigid body and is
    # deliberately not registered for this environment.
    reset_object_physical_properties = None


@configclass
class ArticulatedChairRewardsCfg(RewardsCfg):
    """Task rewards evaluated from the live upper-chair-frame pose."""

    sparse_completion = RewTerm(
        func=chair_mdp.upper_body_sparse_completion_reward,
        weight=10.0,
        params={
            "goal_term_name": "goal_pose",
            "dist_error": 0.05,
            "angular_error": 5.0,
            "success_reward": 1.0,
        },
    )
    keypoint_pose_match_exp = RewTerm(
        func=chair_mdp.upper_body_pose_match_exp,
        weight=8.0,
        params={"goal_term_name": "goal_pose", "sigma": 1.0},
    )
    vel_toward_goal = RewTerm(
        func=chair_mdp.upper_body_velocity_toward_goal_exp,
        weight=1.0,
        params={
            "goal_term_name": "goal_pose",
            "sigma": 0.7071067812,
            "use_unit_vel": True,
            "use_xy": True,
        },
    )
    lin_vel_change_penalty = RewTerm(
        func=chair_mdp.upper_body_lin_vel_change_penalty,
        weight=2.0,
    )
    ang_vel_change_penalty = RewTerm(
        func=chair_mdp.upper_body_ang_vel_change_penalty,
        weight=2.0,
    )
    flat_orientation_l2 = RewTerm(
        func=chair_mdp.upper_body_flat_orientation_l2,
        weight=-10.0,
        params={"flat_threshold": 1.0e-3},
    )


@configclass
class ArticulatedChairTerminationsCfg(TerminationsCfg):
    """Termination terms specific to the more expensive articulated-chair task."""

    # Keep the equivalent protection disabled for the rigid-object task to
    # preserve its training throughput.  The 20-body chair is more prone to
    # unstable contacts, so reset an affected environment before it can poison
    # a rollout with exploding or non-finite robot/chair state/action values.
    physics_explosion = DoneTerm(
        func=chair_mdp.articulated_chair_outlier_detected,
        params={"threshold": 1000.0},
    )


@configclass
class ArticulatedChairEnvCfg(ManagerBasedRLEnvCfg):
    """Parallel training environment for the articulated QiVi chair."""

    scene: ArticulatedChairSceneCfg = ArticulatedChairSceneCfg(num_envs=4096, env_spacing=4.0)
    observations: ArticulatedChairObservationsCfg = ArticulatedChairObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: ArticulatedChairEventCfg = ArticulatedChairEventCfg()
    commands: ArticulatedChairCommandsCfg = ArticulatedChairCommandsCfg()
    rewards: ArticulatedChairRewardsCfg = ArticulatedChairRewardsCfg()
    terminations: ArticulatedChairTerminationsCfg = ArticulatedChairTerminationsCfg()

    def __post_init__(self) -> None:
        self.decimation = 4
        self.episode_length_s = 20.0
        self.viewer.eye = (8.0, 0.0, 5.0)
        self.sim.dt = 1 / 200
        self.sim.render_interval = self.decimation

        if getattr(self.sim, "physx", None) is not None:
            self.sim.physx.gpu_max_rigid_patch_count = 3_000_000
            self.sim.physx.gpu_found_lost_pairs_capacity = 4_194_304
            self.sim.physx.gpu_heap_capacity = 256 * 1024 * 1024
            self.sim.physx.gpu_temp_buffer_capacity = 64 * 1024 * 1024

        self.scene.robot = SPOT_ARM_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.joint_drive.gains.stiffness = None
