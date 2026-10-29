import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
import isaaclab.utils.math as math_utils
from pathlib import Path
from SuperQ_ALORE.assets.object_catalog import ARM_JOINT_NAMES_IN_ORDER, OBJECT_CATALOG
import numpy as np
# Default configurations for the object
DEFAULT_RIGID_PROPS = sim_utils.RigidBodyPropertiesCfg(
    rigid_body_enabled=True,
    kinematic_enabled=False,
    disable_gravity=False,
    solver_position_iteration_count=12,
    solver_velocity_iteration_count=2,
)

DEFAULT_COLLISION_PROPS = sim_utils.CollisionPropertiesCfg(
    collision_enabled=True,
    contact_offset=0.02,
    rest_offset=0.0,
)

# The ground
GROUND_PATCH_THICKNESS = 0.02

GROUND_PATCH_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/GroundPatch",
    spawn=sim_utils.CuboidCfg(
        size=(4.0, 4.0, GROUND_PATCH_THICKNESS),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            kinematic_enabled=True,
            disable_gravity=True,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True, contact_offset=0.01, rest_offset=0.0
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.35, 0.35)),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -0.5 * GROUND_PATCH_THICKNESS)),
)


# Create the configuration for each object
def create_target_object_cfg():
    target_obj_cfgs = []
    for obj_id in range(len(OBJECT_CATALOG)):
        # Spawn each catalog object across all envs. Activation is handled at reset-time
        # using OBJECT_IDX_ENVS, while non-active objects are moved away.
        prim_path = f"{{ENV_REGEX_NS}}/target_object_{obj_id}"


        target_obj_cfgs.append(
            RigidObjectCfg(
                prim_path=prim_path,
                spawn=sim_utils.UsdFileCfg(
                    usd_path=OBJECT_CATALOG[obj_id].asset_path,
                    rigid_props=DEFAULT_RIGID_PROPS,
                    collision_props=DEFAULT_COLLISION_PROPS,

                    # VERY important for multi-env USD spawning
                    copy_from_source=True,
                ),
            )
        )
    # print(len(target_obj_cfgs))
    # input("Press to continue...")
    return target_obj_cfgs
CATALOG_OBJECT_CFGS = create_target_object_cfg()


# -----------------------------------------------------------------------------
# Articulated-chair scene asset
# -----------------------------------------------------------------------------
# Keep this configuration separate from ``CATALOG_OBJECT_CFGS``.  The rigid-body
# environments intentionally continue to spawn the full catalog of
# ``RigidObjectCfg`` instances above.
ARTICULATED_CHAIR_USD_PATH = str(
    Path(__file__).resolve().parents[4] / "assets" / "objects" / "articulated-chair" / "qivi.usdc"
)

# ``frame`` is the common parent of the QiVi seat and back.  It is the stable
# upper-assembly reference used by the articulated-chair environment for
# object-pose observations, commands, rewards, and success checks.
ARTICULATED_CHAIR_UPPER_BODY_NAME = "frame"

# The authored QiVi frame pose with the gas lift fixed at its lowest setting,
# expressed in the articulation-root frame. This forms the fixed pre-grasp
# observation; live tracking always reads the simulated ``frame`` link pose.
ARTICULATED_CHAIR_UPPER_BODY_INITIAL_POS_B = (0.0, 0.0, 0.3512)


def create_articulated_target_object_cfg() -> ArticulationCfg:
    """Create the free-base QiVi chair articulation used by the chair task.

    The USD supplies the collision shapes, masses, and joint limits.  The
    joints are deliberately passive in this task while a copied instance is
    spawned for every vectorized environment.
    """

    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/target_object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=ARTICULATED_CHAIR_USD_PATH,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                disable_gravity=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_depenetration_velocity=2.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                fix_root_link=False,
                # 20-body chair with 18 movable joints and a fixed gas lift,
                # simulated at 200 Hz: 12 position / 4 velocity iterations.
                solver_position_iteration_count=12,
                solver_velocity_iteration_count=4,
                sleep_threshold=0.0,
            ),
            # Required when the same USD is instantiated under every
            # environment namespace.
            copy_from_source=True,
        ),
        # The wheel geometry reaches slightly below the authored root frame;
        # this offset matches the supplied chair reference configuration.
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.02),
            joint_pos={".*": 0.0},
            joint_vel={".*": 0.0},
        ),
        soft_joint_pos_limit_factor=1.0,
        # Preserve the physical springs and damping authored in the USD:
        # the seat/back return springs and caster resistance. The gas lift
        # is a fixed joint at minimum height and has no actuator target.
        # None imports the USD gains; zero would disable those mechanisms.
        # Reset events restore the USD spring rest positions. The policy
        # commands only the robot, never these passive chair joints.
        actuators={
            "passive_joints": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=None,
                damping=None,
            ),
        },
    )

# Only for the teleoperation environment, where only one object is used
def create_target_obj_teleoperation_cfg(object_idx = 0, pose_idx = 0):
    # Object init pos & rot
    # obj_pos = OBJECT_CATALOG[object_idx].poses[pose_idx].position
    # obj_rot = OBJECT_CATALOG[object_idx].poses[pose_idx].orientation
    obj_pos = (10.0, 0.0, 0.0)
    obj_rot = (0.707, 0.0, 0.0, 0.707)
    # Construct the object
    target_obj_cfg = RigidObjectCfg(
        prim_path=f"/World/envs/env_0/target_object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=OBJECT_CATALOG[object_idx].asset_path,
            rigid_props=DEFAULT_RIGID_PROPS,
            collision_props=DEFAULT_COLLISION_PROPS,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=obj_pos,
            rot=obj_rot,
        ),
    )
    joint_position = OBJECT_CATALOG[object_idx].poses[pose_idx].joint_positions
    
    # An initial value to rise the robotic arm
    joint_angle_val = [
        -0.000942,  # arm_sh0
        -3.114827,  # arm_sh1
        3.133373,   # arm_el0
        1.559898,   # arm_el1
        -0.002241,  # arm_wr0
        -1.569777,  # arm_wr1
        -1.556664,  # arm_f1x
    ]
    joint_angle_ref = {ARM_JOINT_NAMES_IN_ORDER[i]: joint_angle_val[i] for i in range(len(ARM_JOINT_NAMES_IN_ORDER))}
    
    
    joint_angle_ref["fl_hx"] = +0.017192 + 0.12
    joint_angle_ref["fr_hx"] = +0.010593 - 0.12
    joint_angle_ref["hl_hx"] = +0.012298 +0.12
    joint_angle_ref["hr_hx"] = +0.011882 - 0.12

    joint_angle_ref["fl_hy"] = +0.193184 + 0.5
    joint_angle_ref["fr_hy"] = +0.191340 + 0.5
    joint_angle_ref["hl_hy"] = +0.190885 + 0.5
    joint_angle_ref["hr_hy"] = +0.191736 + 0.5

    joint_angle_ref["fl_kn"] = -0.437494 - 1.0
    joint_angle_ref["fr_kn"] = -0.429058 - 1.0
    joint_angle_ref["hl_kn"] = -0.437132 - 1.0
    joint_angle_ref["hr_kn"] = -0.429248 - 1.0

    return [target_obj_cfg, joint_angle_ref, obj_pos, obj_rot]

OBJECT_TELEOPERATION_INFO = create_target_obj_teleoperation_cfg(0, 0)
