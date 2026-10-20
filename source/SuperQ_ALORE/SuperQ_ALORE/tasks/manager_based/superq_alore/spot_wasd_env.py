# Copyright (c) 2026
# SPDX-License-Identifier: BSD-3-Clause

"""Object-free Spot/ReLIC environment for keyboard joint-position teleoperation."""

from dataclasses import MISSING

import isaaclab.sim as sim_utils
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as isaac_mdp
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from SuperQ_ALORE.assets.spot.constants import ARM_JOINT_NAMES, LEG_JOINT_NAMES
from SuperQ_ALORE.assets.spot.spot import SPOT_ARM_CFG
from . import mdp
from .mdp.scene import OBJECT_TELEOPERATION_INFO


# Use the joint angles at the default arm position
_WASD_ARM_JOINT_POS = {
    "arm_sh0": -0.000942,  # arm_sh0
    "arm_sh1": -3.114827,  # arm_sh1
    "arm_el0": 3.133373,   # arm_el0
    "arm_el1": 1.559898,   # arm_el1
    "arm_wr0": -0.002241,  # arm_wr0
    "arm_wr1": -1.569777,  # arm_wr1
    "arm_f1x": -1.556664,  # arm_f1x
}


@configclass
class SpotWASDSceneCfg(InteractiveSceneCfg):
    """A plane, lighting, and one or more independent Spot articulations."""

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
    )
    robot: ArticulationCfg = MISSING
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
    )


@configclass
class ActionsCfg:
    """Expose the 12-D base/arm/torso command; the term executes joint targets."""

    relic_joint_position = mdp.ReLICWASDJointPositionActionCfg(
        asset_name="robot",
        joint_names=["[fh].*"],
        arm_joint_names=tuple(ARM_JOINT_NAMES),
        leg_joint_names=LEG_JOINT_NAMES,
        arm_joint_position=_WASD_ARM_JOINT_POS,
        scale = 0.2,
        low_level_update_decimation=2,
    )


@configclass
class ObservationsCfg:
    @configclass
    class LocomotionPolicyCfg(ObsGroup):
        """The ReLIC state terms before the 25-D command insertion."""

        base_lin_vel = ObsTerm(func=isaac_mdp.base_lin_vel, noise=Unoise(n_min=0.0, n_max=0.0))
        base_ang_vel = ObsTerm(func=isaac_mdp.base_ang_vel, noise=Unoise(n_min=0.0, n_max=0.0))
        projected_gravity = ObsTerm(
            func=isaac_mdp.projected_gravity, noise=Unoise(n_min=0.0, n_max=0.0)
        )
        joint_pos = ObsTerm(func=isaac_mdp.joint_pos_rel, noise=Unoise(n_min=0.0, n_max=0.0))
        joint_vel = ObsTerm(func=isaac_mdp.joint_vel_rel, noise=Unoise(n_min=0.0, n_max=0.0))
        actions = ObsTerm(
            func=mdp.last_leg_action,
            params={"action_term_name": "relic_joint_position"},
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    locomotion_policy: LocomotionPolicyCfg = LocomotionPolicyCfg()


@configclass
class EventCfg:
    """Reset only the robot; there is no object state in this task."""

    reset_base = EventTerm(
        func=isaac_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                           "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0)},
            "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                               "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0)},
        },
    )
    reset_joints = EventTerm(
        func=mdp.reset_joints_around_grasp_pose,
        mode="reset",
        params={
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
            "joint_position_ref": _WASD_ARM_JOINT_POS,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


@configclass
class CommandsCfg:
    """The base-velocity policy command arrives through the action term."""


@configclass
class RewardsCfg:
    """No reward is needed for direct keyboard teleoperation."""


@configclass
class TerminationsCfg:
    """Episodes terminate only through the configured time limit."""


@configclass
class SpotWASDEnvCfg(ManagerBasedRLEnvCfg):
    """Simulation-only Spot locomotion environment driven by ReLIC."""

    scene: SpotWASDSceneCfg = SpotWASDSceneCfg(num_envs=1, env_spacing=4.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        self.decimation = 4
        self.episode_length_s = 120.0
        self.sim.dt = 1 / 200
        self.sim.render_interval = self.decimation
        self.viewer.eye = (5.0, 5.0, 3.0)
        self.scene.robot = SPOT_ARM_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.joint_drive.gains.stiffness = None
