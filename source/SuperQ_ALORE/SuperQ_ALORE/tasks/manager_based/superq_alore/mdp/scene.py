import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.assets import ArticulationCfg

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


# The chair => spawn the chairs in order
# Global variable to store the object & pose in each parallel sub-env
OBJECT_IDX_ENVS = []
POSE_IDX_LOCAL_ENVS = []
GRASP_POSE_JOINT_POSITIONS = []
def build_target_objects(pool_size = 4096):
    pool_size = int(pool_size)
    # Step 1: Determine the number of poses initialized in the environment
    # OBJECT_CATALOG: a tuple of ObjectEntry,
    # each containing 
    # object_id, asset_path, and 
    # a tuple of PoseEntry (pose_id, position, orientation, joint_configuration)
    pose_num = []
    for obj in OBJECT_CATALOG:
        pose_num.append(len(obj.poses))
    pose_num = np.array(pose_num)

    total_pose_num = np.sum(pose_num)
    pose_num_cumsum = np.cumsum(pose_num)
    pose_idx_global = np.resize(np.arange(total_pose_num), pool_size) # global pose idx across all objects, resized to the pool size of envs
    pose_idx_global = np.sort(pose_idx_global) # sort to ensure the same object and pose are assigned together in adjacent envs, which can help with debugging and visualization
    # Step 2: Map the pose IDXs to object IDs and pose IDs within that object
    for pose_idx in pose_idx_global:
        obj_idx = np.searchsorted(pose_num_cumsum, pose_idx, side='right')
        pose_idx_within_obj = pose_idx - (pose_num_cumsum[obj_idx - 1] if obj_idx > 0 else 0)
        OBJECT_IDX_ENVS.append(obj_idx)
        POSE_IDX_LOCAL_ENVS.append(pose_idx_within_obj)
        joint_position = OBJECT_CATALOG[obj_idx].poses[pose_idx_within_obj].joint_positions
        joint_angle_val = [joint_position[name] for name in ARM_JOINT_NAMES_IN_ORDER]
        GRASP_POSE_JOINT_POSITIONS.append(joint_angle_val)



def create_target_object_cfg(pool_size = 4096):
    # assign object ids & pose ids to each environment
    build_target_objects(pool_size)
    target_obj_cfgs = []
    for obj_id in range(len(OBJECT_CATALOG)):

        env_ids = np.where(np.array(OBJECT_IDX_ENVS) == obj_id)[0]
        #
        # Build regex:
        #
        # /World/envs/(env_0|env_1|env_2)/target_object
        #
        env_regex = "|".join([f"env_{i}" for i in env_ids])

        prim_path = (
            f"/World/envs/({env_regex})/target_object"
        )


        target_obj_cfgs.append(
            RigidObjectCfg(
                prim_path=prim_path,
                spawn=sim_utils.UsdFileCfg(
                    usd_path=OBJECT_CATALOG[obj_id].asset_path,

                    # VERY important for multi-env USD spawning
                    copy_from_source=True,
                ),
            )
        )
    return target_obj_cfgs
CATALOG_OBJECT_CFGS = create_target_object_cfg(4096)