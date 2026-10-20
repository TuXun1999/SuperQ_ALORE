"""Isaac Lab / PhysX configuration (validated with source release 2.3.1). Import after AppLauncher."""
from pathlib import Path
import json
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
ROOT=Path(__file__).resolve().parents[1]
MANIFEST=json.loads((ROOT/'model_manifest.json').read_text())

def make_qivi_cfg(prim_path='/World/Qivi',pos=(0.,0.,0.02),fixed=False,disable_gravity=False):
    # Cylinders must retain smooth analytic contacts in the simulation process.
    import carb
    carb.settings.get_settings().set_bool('/physics/collisionApproximateCylinders',False)
    joints=MANIFEST['joints']
    return ArticulationCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(ROOT/'isaaclab/qivi.usdc'),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=disable_gravity,linear_damping=0.,angular_damping=0.,max_depenetration_velocity=2.),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=False,fix_root_link=fixed,solver_position_iteration_count=16,solver_velocity_iteration_count=4,sleep_threshold=0.),
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=pos,joint_pos={'.*':0.},joint_vel={'.*':0.}),
        soft_joint_pos_limit_factor=1.,
        actuators={'passive_mechanisms':ImplicitActuatorCfg(
            joint_names_expr=['.*'],
            stiffness={j['name']:j['stiffness'] for j in joints},
            damping={j['name']:j['damping'] for j in joints},
            effort_limit_sim=1e6,
            armature=0.,friction=0.,
        )},
    )

def set_passive_targets(chair):
    """Apply physical spring rest positions; call again after reset/target writes.

    Lab tensor gains and targets use radians. Authored USD drive gains use degrees.
    Targets include backrest preload that balances the estimated unloaded weight.
    """
    refs={j['name']:j['springref'] for j in MANIFEST['joints']}
    target=torch.tensor([[refs[name] for name in chair.joint_names]],device=chair.device).expand(chair.num_instances,-1)
    chair.set_joint_position_target(target)
    chair.set_joint_velocity_target(torch.zeros_like(target))
    return target


def make_sim_cfg(device="cuda:0"):
    """Reference hard-floor contact settings, tested at 480 Hz."""
    return sim_utils.SimulationCfg(
        dt=MANIFEST["parameters"]["simulation"]["dt_s"],device=device,render_interval=8,
        physx=sim_utils.PhysxCfg(friction_offset_threshold=.001,friction_correlation_distance=.0005),
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=.7,dynamic_friction=.7,restitution=0.,friction_combine_mode="min",restitution_combine_mode="min"),
    )


def spawn_reference_floor(prim_path="/World/Floor"):
    """Local static floor: avoids dependencies on remote Nucleus assets."""
    cfg=sim_utils.CuboidCfg(size=(10.,10.,.1),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=.001,rest_offset=0.,torsional_patch_radius=0.,min_torsional_patch_radius=0.),
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=.7,dynamic_friction=.7,restitution=0.,friction_combine_mode="min"),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.18,.22,.25)),
    )
    return cfg.func(prim_path,cfg,translation=(0.,0.,-.05))


def set_held_height(chair,offset_m):
    """Set manual seat height relative to supplied pose; holds, never free falls.

    This represents choosing a pneumatic lock position. It is not a lever/valve model.
    Does not overwrite the other joints' spring rest positions.
    """
    spec=next(j for j in MANIFEST["joints"] if j["name"]=="height_hold")
    if not spec["range"][0]<=offset_m<=spec["range"][1]:
        raise ValueError(f"Height offset must be within {spec['range']} metres")
    idx=chair.joint_names.index("height_hold")
    target=torch.full((chair.num_instances,1),float(offset_m),device=chair.device)
    chair.set_joint_position_target(target,joint_ids=[idx])
