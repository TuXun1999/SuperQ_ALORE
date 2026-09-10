import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.assets import ArticulationCfg
import isaaclab.utils.math as math_utils
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


# Only for the teleoperation environment, where only one object is used
def create_target_obj_teleoperation_cfg(object_idx = 0, pose_idx = 0):
    # Object init pos & rot TODO: Adapt it to the new configuration
    # obj_pos = OBJECT_CATALOG[object_idx].poses[pose_idx].position
    # obj_rot = OBJECT_CATALOG[object_idx].poses[pose_idx].orientation
    obj_pos = (0.0, 0.0, 0.0)
    obj_rot = (1.0, 0.0, 0.0, 0.0)
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
    joint_angle_val = [0.0, -2.05, 1.3366, 0.0, 1.2281, 0.0, -0.9]
    joint_angle_ref = {ARM_JOINT_NAMES_IN_ORDER[i]: joint_angle_val[i] for i in range(len(ARM_JOINT_NAMES_IN_ORDER))}
    return [target_obj_cfg, joint_angle_ref, obj_pos, obj_rot]

OBJECT_TELEOPERATION_INFO = create_target_obj_teleoperation_cfg(0, 0)