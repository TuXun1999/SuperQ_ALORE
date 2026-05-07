import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.assets import ArticulationCfg
from SuperQ_ALORE.assets import ASSET_DIR
from SuperQ_ALORE.assets.object_catalog import OBJECT_CATALOG
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


# previous: single chair 
# # The chair
# # CHAIR_USD_PATH = ASSET_DIR + "/objects/Shorter_Office_Chair.usd" 
# # CHAIR_USD_PATH = ASSET_DIR + "/objects/armchair_with_contact_sensor.usda" 
# # CHAIR_USD_PATH = ASSET_DIR + "/objects/meramic_chair_with_contact_sensor.usd"
# # CHAIR_USD_PATH = ASSET_DIR + "/objects/new_shorter_office_chair.usd"
# # CHAIR_USD_PATH = ASSET_DIR + "/objects/birch_seat.usd"
# CHAIR_USD_PATH = ASSET_DIR + "/objects/willow_bench.usd"


# CHAIR_RIGID_CFG = RigidObjectCfg(
#     prim_path="{ENV_REGEX_NS}/Pushable",
#     spawn=sim_utils.UsdFileCfg(
#         usd_path=CHAIR_USD_PATH,
#         rigid_props=DEFAULT_RIGID_PROPS,
#         collision_props=DEFAULT_COLLISION_PROPS,
#         mass_props=sim_utils.MassPropertiesCfg(mass=0.5),

# Multi-object target assets loaded from OBJECT_CATALOG.
_CATALOG_ASSETS_CFG = tuple(
    sim_utils.UsdFileCfg(
        usd_path=entry.asset_path,
        rigid_props=DEFAULT_RIGID_PROPS,
        collision_props=DEFAULT_COLLISION_PROPS,
        mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
    )
    for entry in OBJECT_CATALOG
)

# we set the random_choice to False so that asset configurations are spawned in the order
# defined in OBJECT_CATALOG, which allows us to have a deterministic mapping from env index to object 
# (i.e. env i will always have OBJECT_CATALOG[i % N], where N is the number of objects in the catalog)
CATALOG_MULTI_OBJECT_RIGID_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Pushable",
    spawn=sim_utils.MultiAssetSpawnerCfg(
        assets_cfg=_CATALOG_ASSETS_CFG,
        random_choice=False,
    ),

    # we reset the initail pose in event.py: reset_target_object_from_catalog_pose
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
    ),
)