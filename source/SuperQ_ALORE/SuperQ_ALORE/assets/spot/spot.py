# Copyright (c) 2024 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

"""Configuration for the Boston Dynamics robot.

The following configuration parameters are available:

* :obj:`SPOT_ARM_CFG`: The Spot Arm robot with delay PD and remote PD actuators.
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg
from SuperQ_ALORE.actuators import SpotKneeActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from SuperQ_ALORE.assets import ASSET_DIR
from SuperQ_ALORE.assets.spot.constants import (
    SPOT_DEFAULT_POS,
    SPOT_DEFAULT_JOINT_POS,
    HIP_EFFORT_LIMIT,
    HIP_STIFFNESS,
    HIP_DAMPING,
    KNEE_STIFFNESS,
    KNEE_DAMPING,
    ARM_EFFORT_LIMIT,
    ARM_STIFFNESS,
    ARM_DAMPING,
    ARM_ARMATURE,
    JOINT_PARAMETER_LOOKUP_TABLE,
)

##
# Configuration
##

# TODO: Determine SPOT's configuration from SuperQ-GRASP
def spot_initial_pos():
    return (-1.0, 0.0, 0.45)
def spot_initial_joint_pos(spot_initial_joint_pos_ref):
    spot_initial_joint_pos = {}
    # Arm joints 
    # A "safe" initial pos from experiments;
    # [-0.1132, -2.4997,  1.7310, -0.1282, -0.7852, -0.1023, -1.4842,]
    spot_initial_joint_pos["arm_sh0"] = 1.0
    spot_initial_joint_pos["arm_sh1"] = -2.5
    spot_initial_joint_pos["arm_el0"] = 1.73
    spot_initial_joint_pos["arm_el1"] = -0.1282
    spot_initial_joint_pos["arm_wr0"] = -0.7852
    spot_initial_joint_pos["arm_wr1"] = -0.1023
    spot_initial_joint_pos["arm_f1x"] = -1.45

    # Leg joints
    spot_initial_joint_pos["fl_hx"] = spot_initial_joint_pos_ref["fl_hx"]
    spot_initial_joint_pos["fr_hx"] = spot_initial_joint_pos_ref["fr_hx"]
    spot_initial_joint_pos["hl_hx"] = spot_initial_joint_pos_ref["hl_hx"]
    spot_initial_joint_pos["hr_hx"] = spot_initial_joint_pos_ref["hr_hx"]
    spot_initial_joint_pos["fl_hy"] = spot_initial_joint_pos_ref["fl_hy"]
    spot_initial_joint_pos["fr_hy"] = spot_initial_joint_pos_ref["fr_hy"]
    spot_initial_joint_pos["hl_hy"] = spot_initial_joint_pos_ref["hl_hy"]
    spot_initial_joint_pos["hr_hy"] = spot_initial_joint_pos_ref["hr_hy"]
    spot_initial_joint_pos["fl_kn"] = spot_initial_joint_pos_ref["fl_kn"]
    spot_initial_joint_pos["fr_kn"] = spot_initial_joint_pos_ref["fr_kn"]
    spot_initial_joint_pos["hl_kn"] = spot_initial_joint_pos_ref["hl_kn"]
    spot_initial_joint_pos["hr_kn"] = spot_initial_joint_pos_ref["hr_kn"]
    return spot_initial_joint_pos

    
SPOT_ARM_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        merge_fixed_joints=False,
        make_instanceable=False,
        link_density=1.0e-8,
        asset_path=f"{ASSET_DIR}/spot/spot_with_arm.urdf",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=spot_initial_pos(),
        # NOTE: I don't know why, but if I enable this, the robot starts to ... rotate!!!
        # joint_pos=spot_initial_joint_pos(SPOT_DEFAULT_JOINT_POS),
        joint_pos = SPOT_DEFAULT_JOINT_POS,
        joint_vel={".*": 0.0},
    ),
    actuators={
        "spot_hip": DelayedPDActuatorCfg(
            joint_names_expr=[".*_h[xy]"],
            effort_limit=HIP_EFFORT_LIMIT,
            stiffness=HIP_STIFFNESS,
            damping=HIP_DAMPING,
            min_delay=0,  # physics time steps (min: 5.0*1=5.0ms)
            max_delay=3,  # physics time steps (max: 5.0*2=10.0ms)
        ),
        "spot_knee": SpotKneeActuatorCfg(
            joint_names_expr=[".*_kn"],
            joint_parameter_lookup=JOINT_PARAMETER_LOOKUP_TABLE,
            effort_limit=None,  # torque limits are handled based experimental data
            stiffness=KNEE_STIFFNESS,
            damping=KNEE_DAMPING,
            min_delay=0,  # physics time steps (min: 5.0*1=5.0ms)
            max_delay=3,  # physics time steps (max: 5.0*2=10.0ms)
            enable_torque_speed_limit=True,
        ),
        "spot_arm_sh0": DelayedPDActuatorCfg(
            joint_names_expr=["arm_sh0"],
            effort_limit=ARM_EFFORT_LIMIT[0],
            stiffness=ARM_STIFFNESS[0],
            damping=ARM_DAMPING[0],
            armature=ARM_ARMATURE[0],
            min_delay=0,  # physics time steps (min: 5.0*1=5.0ms)
            max_delay=3,  # physics time steps (max: 5.0*2=10.0ms)
        ),
        "spot_arm_sh1": DelayedPDActuatorCfg(
            joint_names_expr=["arm_sh1"],
            effort_limit=ARM_EFFORT_LIMIT[1],
            stiffness=ARM_STIFFNESS[1],
            damping=ARM_DAMPING[1],
            armature=ARM_ARMATURE[1],
            min_delay=0,  # physics time steps (min: 5.0*1=5.0ms)
            max_delay=3,  # physics time steps (max: 5.0*2=10.0ms)
        ),
        "spot_arm_el0": DelayedPDActuatorCfg(
            joint_names_expr=["arm_el0"],
            effort_limit=ARM_EFFORT_LIMIT[2],
            stiffness=ARM_STIFFNESS[2],
            damping=ARM_DAMPING[2],
            armature=ARM_ARMATURE[2],
            min_delay=0,  # physics time steps (min: 5.0*1=5.0ms)
            max_delay=3,  # physics time steps (max: 5.0*2=10.0ms)
        ),
        "spot_arm_el1": DelayedPDActuatorCfg(
            joint_names_expr=["arm_el1"],
            effort_limit=ARM_EFFORT_LIMIT[3],
            stiffness=ARM_STIFFNESS[3],
            damping=ARM_DAMPING[3],
            armature=ARM_ARMATURE[3],
            min_delay=0,  # physics time steps (min: 5.0*1=5.0ms)
            max_delay=3,  # physics time steps (max: 5.0*2=10.0ms)
        ),
        "spot_arm_wr0": DelayedPDActuatorCfg(
            joint_names_expr=["arm_wr0"],
            effort_limit=ARM_EFFORT_LIMIT[4],
            stiffness=ARM_STIFFNESS[4],
            damping=ARM_DAMPING[4],
            armature=ARM_ARMATURE[4],
            min_delay=0,  # physics time steps (min: 5.0*1=5.0ms)
            max_delay=3,  # physics time steps (max: 5.0*2=10.0ms)
        ),
        "spot_arm_wr1": DelayedPDActuatorCfg(
            joint_names_expr=["arm_wr1"],
            effort_limit=ARM_EFFORT_LIMIT[5],
            stiffness=ARM_STIFFNESS[5],
            damping=ARM_DAMPING[5],
            armature=ARM_ARMATURE[5],
            min_delay=0,  # physics time steps (min: 5.0*1=5.0ms)
            max_delay=3,  # physics time steps (max: 5.0*2=10.0ms)
        ),
        "spot_arm_f1x": DelayedPDActuatorCfg(
            joint_names_expr=["arm_f1x"],
            effort_limit=ARM_EFFORT_LIMIT[6],
            stiffness=ARM_STIFFNESS[6],
            damping=ARM_DAMPING[6],
            armature=ARM_ARMATURE[6],
            min_delay=0,  # physics time steps (min: 5.0*1=5.0ms)
            max_delay=3,  # physics time steps (max: 5.0*2=10.0ms)
        ),
    },
)
