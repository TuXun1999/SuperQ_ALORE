import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.assets import ArticulationCfg
from SuperQ_ALORE.assets import ASSET_DIR
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

# The chair
CHAIR_USD_PATH = ASSET_DIR + "/object-models/armchair_with_contact_sensor.usda"

CHAIR_RIGID_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Pushable",
    spawn=sim_utils.UsdFileCfg(
        usd_path=CHAIR_USD_PATH,
        rigid_props=DEFAULT_RIGID_PROPS,
        collision_props=DEFAULT_COLLISION_PROPS,
        mass_props=sim_utils.MassPropertiesCfg(mass=2.0),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        rot=(0.707, 0.0, 0.0, 0.707)),
)