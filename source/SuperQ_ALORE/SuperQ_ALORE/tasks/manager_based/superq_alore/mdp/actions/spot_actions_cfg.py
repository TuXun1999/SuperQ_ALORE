# Copyright (c) 2024 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from dataclasses import MISSING

from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass
from isaaclab.envs.mdp.actions import JointActionCfg

from . import spot_joint_actions

##
# Joint actions.
##


@configclass
class MixedPDArmMultiLegJointPositionActionCfg(JointActionCfg):
    """Configuration for the joint position action term.

    See :class:`JointPositionAction` for more details.
    """

    class_type: type[ActionTerm] = (
        spot_joint_actions.MixedPDArmMultiLegJointPositionAction
    )

    arm_joint_names: tuple[str, ...] = MISSING
    leg_joint_names: dict = MISSING

    command_name: str = MISSING

    use_default_offset: bool = True
    
    locomotion_policy_path = "./source/SuperQ_ALORE/SuperQ_ALORE/assets/spot/pretrained_relic/policy.pt"
    locomotion_obs_group: str = "locomotion_policy"
    low_level_update_decimation: int = 1
    """Run low-level locomotion inference every N simulation steps.

    Effective low-level frequency = physics_frequency / low_level_update_decimation.
    Example: with 200Hz physics and decimation=2, low-level runs at 100Hz.
    """

    gripper_vel: float = 0.008
    """Whether to use default joint positions configured in the articulation asset as offset.
    Defaults to True.

    If True, this flag results in overwriting the values of :attr:`offset` to the default joint positions
    from the articulation asset.
    """

@configclass
class MixedPDArmMultiLegJointPositionActionTeleCfg(JointActionCfg):
    """Configuration for the joint position action term used in teleoperation.

    See :class:`MixedPDArmMultiLegJointPositionActionCfg` for more details.
    """

    class_type: type[ActionTerm] = (
        spot_joint_actions.MixedPDArmMultiLegJointPositionActionTele
    )

    arm_joint_names: tuple[str, ...] = MISSING
    leg_joint_names: dict = MISSING

    command_name: str = MISSING

    use_default_offset: bool = True
    
    locomotion_policy_path = "./source/SuperQ_ALORE/SuperQ_ALORE/assets/spot/pretrained_relic/policy.pt"
    locomotion_obs_group: str = "locomotion_policy"
    low_level_update_decimation: int = 1
    """Run low-level locomotion inference every N simulation steps."""
    
    # Remove the gripper-related attributes

"""
(DEPRECATED)
Not used by the system, only kept here for programming 
debugging purpose
"""
# @configclass
# class PDArmJointPositionActionCfg(JointActionCfg):
#     """Configuration for the joint position action term.

#     See :class:`JointPositionAction` for more details.
#     """

#     class_type: type[ActionTerm] = spot_joint_actions.PDArmJointPositionAction

#     arm_joint_names: tuple[str, ...] = MISSING

#     arm_command_name: str = MISSING

#     use_default_offset: bool = True
#     """Whether to use default joint positions configured in the articulation asset as offset.
#     Defaults to True.

#     If True, this flag results in overwriting the values of :attr:`offset` to the default joint positions
#     from the articulation asset.
#     """


# @configclass
# class PDArmLegJointPositionActionCfg(JointActionCfg):
#     """Configuration for the joint position action term.

#     See :class:`JointPositionAction` for more details.
#     """

#     class_type: type[ActionTerm] = spot_joint_actions.PDArmLegJointPositionAction

#     arm_joint_names: tuple[str, ...] = MISSING
#     leg_joint_names: tuple[str, ...] = MISSING

#     arm_command_name: str = MISSING
#     leg_command_name: str = MISSING

#     use_default_offset: bool = True
#     """Whether to use default joint positions configured in the articulation asset as offset.
#     Defaults to True.

#     If True, this flag results in overwriting the values of :attr:`offset` to the default joint positions
#     from the articulation asset.
#     """


# @configclass
# class PDArmMultiLegJointPositionActionCfg(JointActionCfg):
#     """Configuration for the joint position action term.

#     See :class:`JointPositionAction` for more details.
#     """

#     class_type: type[ActionTerm] = spot_joint_actions.PDArmMultiLegJointPositionAction

#     arm_joint_names: tuple[str, ...] = MISSING
#     leg_joint_names: dict = MISSING

#     arm_command_name: str = MISSING
#     leg_command_name: dict = MISSING

#     use_default_offset: bool = True
#     """Whether to use default joint positions configured in the articulation asset as offset.
#     Defaults to True.

#     If True, this flag results in overwriting the values of :attr:`offset` to the default joint positions
#     from the articulation asset.
#     """