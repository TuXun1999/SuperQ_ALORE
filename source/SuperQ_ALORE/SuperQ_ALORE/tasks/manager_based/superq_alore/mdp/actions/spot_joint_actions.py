# Copyright (c) 2024 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from __future__ import annotations

import torch
from typing import TYPE_CHECKING
import os
from isaaclab.utils.io.torchscript import load_torchscript_model

from isaaclab.envs.mdp.actions import JointAction

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from . import spot_actions_cfg

# TODO: Modify it into the version that takes the
# input high-level controller action, and process it
# into low-level control actions
class MixedPDArmMultiLegJointPositionAction(JointAction):
    """Joint action term that applies the processed actions to the articulation's joints as position commands."""

    cfg: spot_actions_cfg.MixedPDArmMultiLegJointPositionActionCfg
    """The configuration of the action term."""

    def __init__(
        self,
        cfg: spot_actions_cfg.MixedPDArmMultiLegJointPositionActionCfg,
        env: ManagerBasedEnv,
    ):
        # initialize the action term
        super().__init__(cfg, env)
        # use default joint positions as offset
        if cfg.use_default_offset:
            self._offset = self._asset.data.default_joint_pos[
                :, self._joint_ids
            ].clone()

        # setup the arm command buffer
        self._arm_joint_ids, self._arm_joint_names = self._asset.find_joints(
            self.cfg.arm_joint_names
        )
        self._leg_joint_ids = {
            leg: self._asset.find_joints(names)[0]
            for leg, names in self.cfg.leg_joint_names.items()
        }

        # Pre-trained low-level controller weights
        self.policy_path = self.cfg.locomotion_policy_path
        if self.policy_path is None:
            policy_path = os.path.normpath(
                os.path.join(
                    os.path.dirname(__file__),
                    "..",
                    "..",
                    "..",
                    "..",
                    "assets",
                    "spot",
                    "low-level-controller.pt",
                )
            )
        self._locomotion_policy = load_torchscript_model(policy_path, device=self.device)
        self._locomotion_policy.eval()

        # Joint-level actions
        self._arm_raw_actions = torch.zeros(
            self.num_envs, len(self._arm_joint_ids), device=self.device
        )
        self._arm_processed_actions = torch.zeros_like(self.arm_raw_actions)

        self._leg_raw_actions = torch.zeros(
            self.num_envs, len(self._leg_joint_ids["fl"]), device=self.device
        )
        self._leg_processed_actions = torch.zeros_like(self._leg_raw_actions)

        self.batch_indices = torch.arange(self.num_envs).view(-1, 1).repeat(1, 3)
        self.action_joint_idxs = torch.tensor(self._joint_ids, device=self.device)

    def apply_actions(self):
        """Apply the actions."""
        # set position targets
        self._asset.set_joint_position_target(
            self.processed_actions, joint_ids=self._joint_ids
        )
        self._asset.set_joint_position_target(
            self.arm_processed_actions, joint_ids=self._arm_joint_ids
        )

    def process_actions(self, actions: torch.Tensor):
        """Process the actions."""
        """
        Originally: actions are leg joint actions
        The arm actions are directly following the commands using linear interpolation
        The leg joint actions are split into two groups:
            a) commanded leg: directly follow the commands using linear interpolation
            b) non-commanded leg: directly execute the predicted actions from agent after scaling
            
        Now: actions are commands for ReLIC to track (arm joints, base pose, base velocities)
        Order (following the convention in CommandCfg, no leg joint tracking):
        base velocity, arm joints, base pose
        (3 + 7 + 3)
        
        The program consists of several steps
        1. Load the pretrained weights
        2. Organize the obs to generate the input to the model
        3. Predict joint-level actions (for the legs)
        4. Predict the arm joint actions & leg joint actions
        """
        # Step 1: Load the pretrained weights ==> See the initialization
        # Step 2: Organize the obs as input to the model
        with torch.inference_mode():
            # The environmental policy observations
            policy_env_obs = self._env.observation_manager.compute_group(
                self.cfg._locomotion_obs_group, update_history=False
            ) # update_history: recommended to be set false
            print("Checking the env obs shape")
            print(policy_env_obs.shape)
            input("Press any key to continue")

            # Insert the commands into the policy obs
            """
            Locomotion policy obs: 
            base_lin_vel, base_ang_vel, projected_gravity, 
            (velocity commands, commands),
            joint_pos, joint_vel, actions
            
            velocity commands: base velocity from command (dim: 3)
            commands: arm leg joint & base pose from command (dim: 22)
            """
            # TODO: check out actions shape...
            base_velocity = actions[:, :3]
            arm_joints = actions[:, 3:10]
            base_pose = actions[:, -3:]
            
            # The following commands from ReLIC indicate that if 
            # the legs are not the commanded ones, just set up the 
            # leg_joint_command to be zero to de-activate the leg tracking 
            # functionality
            # self.leg_joint_start[~self.command_leg] = 0.0
            # self.leg_joint_sub_goal[~self.command_leg] = 0.0
            # self.leg_joint_goal[~self.command_leg] = 0.0
            leg_joints = torch.zeros(arm_joints.shape[0], 12).to(arm_joints.device)
            arm_leg_joint_base_pose = torch.cat(
                [
                    arm_joints,
                    leg_joints,
                    base_pose,
                ],
                dim=1,
            ) # dim: 22

            # the input to low-level controller consists of everything
            # TODO: check out the obs dim in ReLIC
            policy_env_obs = torch.cat(
                [
                    policy_env_obs[:, :12],
                    base_velocity, 
                    arm_leg_joint_base_pose, 
                    policy_env_obs[:, 9:]
                ]
            )
            ## Step 3: predict leg actions
            leg_actions = self._locomotion_policy(policy_env_obs)

        
        
        self._raw_actions[:] = leg_actions
        # apply the affine transformations
        self._processed_actions = self._raw_actions * self._scale + self._offset
        
        # Step 4: extract the raw actions
        # store the raw arm actions, which is the target joint pos
        
        # Extract it from the input actions
        arm_actions = actions[:, 3:10]
        # Execute the action directly (according to ALORE)
        self._arm_raw_actions[:] = arm_actions
        self._arm_processed_actions[:] = self._arm_raw_actions.clone()

        # TODO: understand the actions in ReLIC ==> seems confusing between 
        # leg actions & arm actions

    @property
    def arm_raw_actions(self) -> torch.Tensor:
        """Get the raw arm actions."""
        return self._arm_raw_actions

    @property
    def arm_processed_actions(self) -> torch.Tensor:
        """Get the processed arm actions."""
        return self._arm_processed_actions



"""
(DEPRECATED)
The following functions are deprecated objects from ReLIC.
Kept here just for programming reference
"""


# class PDArmJointPositionAction(JointAction):
#     """Joint action term that applies the processed actions to the articulation's joints as position commands."""

#     cfg: spot_actions_cfg.PDArmJointPositionActionCfg
#     """The configuration of the action term."""

#     def __init__(
#         self, cfg: spot_actions_cfg.PDArmJointPositionActionCfg, env: ManagerBasedEnv
#     ):
#         # initialize the action term
#         super().__init__(cfg, env)
#         # use default joint positions as offset
#         if cfg.use_default_offset:
#             self._offset = self._asset.data.default_joint_pos[
#                 :, self._joint_ids
#             ].clone()

#         # setup the arm command buffer
#         self._arm_joint_ids, self._arm_joint_names = self._asset.find_joints(
#             self.cfg.arm_joint_names
#         )
#         self.arm_command_name = cfg.arm_command_name
#         self.arm_command_manager = env.command_manager

#         self._arm_raw_actions = torch.zeros(
#             self.num_envs, len(self._arm_joint_ids), device=self.device
#         )
#         self._arm_processed_actions = torch.zeros_like(self.arm_raw_actions)

#     @property
#     def arm_raw_actions(self) -> torch.Tensor:
#         """Get the raw arm actions."""
#         return self._arm_raw_actions

#     @property
#     def arm_processed_actions(self) -> torch.Tensor:
#         """Get the processed arm actions."""
#         return self._arm_processed_actions

#     def apply_actions(self):
#         """Apply the actions."""
#         # set position targets
#         self._asset.set_joint_position_target(
#             self.processed_actions, joint_ids=self._joint_ids
#         )
#         self._asset.set_joint_position_target(
#             self.arm_processed_actions, joint_ids=self._arm_joint_ids
#         )

#     def process_actions(self, actions: torch.Tensor):
#         """Process the actions."""
#         # store the raw actions
#         self._raw_actions[:] = actions
#         # apply the affine transformations
#         self._processed_actions = self._raw_actions * self._scale + self._offset

#         # store the raw arm actions, which is the target joint pos
#         self._arm_raw_actions[:] = self.arm_command_manager.get_command(
#             self.arm_command_name
#         )
#         self._arm_processed_actions = self._arm_raw_actions.clone()


# class PDArmLegJointPositionAction(JointAction):
#     """Joint action term that applies the processed actions to the articulation's joints as position commands."""

#     cfg: spot_actions_cfg.PDArmLegJointPositionActionCfg
#     """The configuration of the action term."""

#     def __init__(
#         self, cfg: spot_actions_cfg.PDArmLegJointPositionActionCfg, env: ManagerBasedEnv
#     ):
#         # initialize the action term
#         super().__init__(cfg, env)
#         # use default joint positions as offset
#         if cfg.use_default_offset:
#             self._offset = self._asset.data.default_joint_pos[
#                 :, self._joint_ids
#             ].clone()

#         # setup the arm command buffer
#         self._arm_joint_ids, self._arm_joint_names = self._asset.find_joints(
#             self.cfg.arm_joint_names
#         )
#         self._leg_joint_ids, self._leg_joint_names = self._asset.find_joints(
#             self.cfg.leg_joint_names
#         )

#         self.arm_command_name = cfg.arm_command_name
#         self.leg_command_name = cfg.leg_command_name
#         self.command_manager = env.command_manager

#         self._arm_raw_actions = torch.zeros(
#             self.num_envs, len(self._arm_joint_ids), device=self.device
#         )
#         self._arm_processed_actions = torch.zeros_like(self.arm_raw_actions)

#         self._leg_raw_actions = torch.zeros(
#             self.num_envs, len(self._leg_joint_ids), device=self.device
#         )
#         self._leg_processed_actions = torch.zeros_like(self._leg_raw_actions)

#     def apply_actions(self):
#         """Apply the actions."""
#         # set position targets
#         self._asset.set_joint_position_target(
#             self.processed_actions, joint_ids=self._joint_ids
#         )
#         self._asset.set_joint_position_target(
#             self.arm_processed_actions, joint_ids=self._arm_joint_ids
#         )
#         self._asset.set_joint_position_target(
#             self.leg_processed_actions, joint_ids=self._leg_joint_ids
#         )

#     def process_actions(self, actions: torch.Tensor):
#         """Process the actions."""
#         # store the raw actions
#         self._raw_actions[:] = actions
#         # apply the affine transformations
#         self._processed_actions = self._raw_actions * self._scale + self._offset

#         # store the raw arm actions, which is the target joint pos
#         self._arm_raw_actions[:] = self.command_manager.get_command(
#             self.arm_command_name
#         )
#         self._arm_processed_actions[:] = self._arm_raw_actions.clone()

#         # store the raw leg actions, which is the target joint pos
#         self._leg_raw_actions[:] = self.command_manager.get_command(
#             self.leg_command_name
#         )
#         self._leg_processed_actions[:] = self._leg_raw_actions.clone()

#     @property
#     def arm_raw_actions(self) -> torch.Tensor:
#         """Get the raw arm actions."""
#         return self._arm_raw_actions

#     @property
#     def arm_processed_actions(self) -> torch.Tensor:
#         """Get the processed arm actions."""
#         return self._arm_processed_actions

#     @property
#     def leg_raw_actions(self) -> torch.Tensor:
#         """Get the raw leg actions."""
#         return self._leg_raw_actions

#     @property
#     def leg_processed_actions(self) -> torch.Tensor:
#         """Get the processed leg actions."""
#         return self._leg_processed_actions


# class PDArmMultiLegJointPositionAction(JointAction):
#     """Joint action term that applies the processed actions to the articulation's joints as position commands."""

#     cfg: spot_actions_cfg.PDArmMultiLegJointPositionActionCfg
#     """The configuration of the action term."""

#     def __init__(
#         self,
#         cfg: spot_actions_cfg.PDArmMultiLegJointPositionActionCfg,
#         env: ManagerBasedEnv,
#     ):
#         # initialize the action term
#         super().__init__(cfg, env)
#         # use default joint positions as offset
#         if cfg.use_default_offset:
#             self._offset = self._asset.data.default_joint_pos[
#                 :, self._joint_ids
#             ].clone()

#         # setup the arm command buffer
#         self._arm_joint_ids, self._arm_joint_names = self._asset.find_joints(
#             self.cfg.arm_joint_names
#         )
#         self._leg_joint_ids = {
#             leg: self._asset.find_joints(names)[0]
#             for leg, names in self.cfg.leg_joint_names.items()
#         }

#         self.arm_command_name = cfg.arm_command_name
#         self.leg_command_name = cfg.leg_command_name
#         self.command_manager = env.command_manager
#         self.leg_command = self.command_manager.get_term(self.leg_command_name)

#         self._arm_raw_actions = torch.zeros(
#             self.num_envs, len(self._arm_joint_ids), device=self.device
#         )
#         self._arm_processed_actions = torch.zeros_like(self.arm_raw_actions)

#         self._leg_raw_actions = torch.zeros(
#             self.num_envs, len(self._leg_joint_ids["fl"]), device=self.device
#         )
#         self._leg_processed_actions = torch.zeros_like(self._leg_raw_actions)

#         self.batch_indices = torch.arange(self.num_envs).view(-1, 1).repeat(1, 3)
#         self.action_joint_idxs = torch.tensor(self._joint_ids, device=self.device)

#     def apply_actions(self):
#         """Apply the actions."""
#         # set position targets
#         self._asset.set_joint_position_target(
#             self.processed_actions, joint_ids=self._joint_ids
#         )
#         self._asset.set_joint_position_target(
#             self.arm_processed_actions, joint_ids=self._arm_joint_ids
#         )

#     def process_actions(self, actions: torch.Tensor):
#         """Process the actions."""
#         # store the raw actions
#         self._raw_actions[:] = actions
#         # apply the affine transformations
#         self._processed_actions = self._raw_actions * self._scale + self._offset
#         # store the non-command leg actions
#         no_command_leg_processed_actions = self._processed_actions.clone()[
#             ~self.leg_command.command_leg
#         ]

#         # store the raw arm actions, which is the target joint pos
#         self._arm_raw_actions[:] = self.command_manager.get_command(
#             self.arm_command_name
#         )
#         self._arm_processed_actions[:] = self._arm_raw_actions.clone()

#         # store the raw leg actions
#         command_joint_idxs = self.leg_command.command_leg_joint_idxs[
#             self.leg_command.command_leg_idxs
#         ]
#         self._leg_raw_actions[:] = self.leg_command.command[
#             self.batch_indices, command_joint_idxs
#         ]
#         self._leg_processed_actions[:] = self._leg_raw_actions.clone()

#         # overwrite command leg actions
#         # --- order of command: [
#         # 'fl_hx', 'fl_hy', 'fl_kn', 'fr_hx', 'fr_hy', 'fr_kn',
#         # 'hl_hx', 'hl_hy', 'hl_kn', 'hr_hx', 'hr_hy', 'hr_kn'
#         # ]
#         # --- order of control: [
#         # 'fl_hx', 'fr_hx', 'hl_hx', 'hr_hx', 'fl_hy', 'fr_hy',
#         # 'hl_hy', 'hr_hy', 'fl_kn', 'fr_kn', 'hl_kn', 'hr_kn'
#         # ]
#         leg_joint_idxs = self.leg_command.leg_joint_idxs[
#             self.leg_command.command_leg_idxs
#         ]  # commanded leg joint idx in the simulation
#         action_joint_idxs = (
#             (
#                 leg_joint_idxs.view(-1).unsqueeze(1)
#                 == self.action_joint_idxs.unsqueeze(0)
#             )
#             .nonzero(as_tuple=True)[1]
#             .view(leg_joint_idxs.shape)
#         )
#         self._processed_actions[self.batch_indices, action_joint_idxs] = (
#             self._leg_processed_actions[:].clone()
#         )
#         # restore the non-command leg actions
#         self._processed_actions[~self.leg_command.command_leg] = (
#             no_command_leg_processed_actions.clone()
#         )

#     @property
#     def arm_raw_actions(self) -> torch.Tensor:
#         """Get the raw arm actions."""
#         return self._arm_raw_actions

#     @property
#     def arm_processed_actions(self) -> torch.Tensor:
#         """Get the processed arm actions."""
#         return self._arm_processed_actions

#     @property
#     def leg_raw_actions(self) -> torch.Tensor:
#         """Get the raw leg actions."""
#         return self._leg_raw_actions

#     @property
#     def leg_processed_actions(self) -> torch.Tensor:
#         """Get the processed leg actions."""
#         return self._leg_processed_actions


# class MixedPDArmMultiLegJointPositionAction(JointAction):
#     """Joint action term that applies the processed actions to the articulation's joints as position commands."""

#     cfg: spot_actions_cfg.MixedPDArmMultiLegJointPositionActionCfg
#     """The configuration of the action term."""

#     def __init__(
#         self,
#         cfg: spot_actions_cfg.MixedPDArmMultiLegJointPositionActionCfg,
#         env: ManagerBasedEnv,
#     ):
#         # initialize the action term
#         super().__init__(cfg, env)
#         # use default joint positions as offset
#         if cfg.use_default_offset:
#             self._offset = self._asset.data.default_joint_pos[
#                 :, self._joint_ids
#             ].clone()

#         # setup the arm command buffer
#         self._arm_joint_ids, self._arm_joint_names = self._asset.find_joints(
#             self.cfg.arm_joint_names
#         )
#         self._leg_joint_ids = {
#             leg: self._asset.find_joints(names)[0]
#             for leg, names in self.cfg.leg_joint_names.items()
#         }

#         self.command_name = cfg.command_name
#         self.command_manager = env.command_manager
#         self.command = self.command_manager.get_term(self.command_name)

#         self._arm_raw_actions = torch.zeros(
#             self.num_envs, len(self._arm_joint_ids), device=self.device
#         )
#         self._arm_processed_actions = torch.zeros_like(self.arm_raw_actions)

#         self._leg_raw_actions = torch.zeros(
#             self.num_envs, len(self._leg_joint_ids["fl"]), device=self.device
#         )
#         self._leg_processed_actions = torch.zeros_like(self._leg_raw_actions)

#         self.batch_indices = torch.arange(self.num_envs).view(-1, 1).repeat(1, 3)
#         self.action_joint_idxs = torch.tensor(self._joint_ids, device=self.device)

#     def apply_actions(self):
#         """Apply the actions."""
#         # set position targets
#         self._asset.set_joint_position_target(
#             self.processed_actions, joint_ids=self._joint_ids
#         )
#         self._asset.set_joint_position_target(
#             self.arm_processed_actions, joint_ids=self._arm_joint_ids
#         )

#     def process_actions(self, actions: torch.Tensor):
#         """Process the actions."""
#         # store the raw actions
#         self._raw_actions[:] = actions
#         # apply the affine transformations
#         self._processed_actions = self._raw_actions * self._scale + self._offset
#         # store the non-command leg actions
#         no_command_leg_processed_actions = self._processed_actions.clone()[
#             ~self.command.command_leg
#         ]

#         # store the raw arm actions, which is the target joint pos
#         self._arm_raw_actions[:] = self.command.arm_joint_sub_goal
#         self._arm_processed_actions[:] = self._arm_raw_actions.clone()

#         # store the raw leg actions
#         command_joint_idxs = self.command.command_leg_joint_idxs[
#             self.command.command_leg_idxs
#         ]
#         self._leg_raw_actions[:] = self.command.leg_joint_sub_goal[
#             self.batch_indices, command_joint_idxs
#         ]
#         self._leg_processed_actions[:] = self._leg_raw_actions.clone()

#         # overwrite command leg actions
#         # --- order of command: [
#         # 'fl_hx', 'fl_hy', 'fl_kn', 'fr_hx', 'fr_hy', 'fr_kn',
#         # 'hl_hx', 'hl_hy', 'hl_kn', 'hr_hx', 'hr_hy', 'hr_kn'
#         # ]
#         # --- order of control: [
#         # 'fl_hx', 'fr_hx', 'hl_hx', 'hr_hx', 'fl_hy', 'fr_hy', 'hl_hy', 'hr_hy',
#         # 'fl_kn', 'fr_kn', 'hl_kn', 'hr_kn'
#         # ]
#         leg_joint_idxs = self.command.leg_joint_idxs[
#             self.command.command_leg_idxs
#         ]  # commanded leg joint idx in the simulation
#         action_joint_idxs = (
#             (
#                 leg_joint_idxs.view(-1).unsqueeze(1)
#                 == self.action_joint_idxs.unsqueeze(0)
#             )
#             .nonzero(as_tuple=True)[1]
#             .view(leg_joint_idxs.shape)
#         )
#         self._processed_actions[self.batch_indices, action_joint_idxs] = (
#             self._leg_processed_actions[:].clone()
#         )
#         # restore the non-command leg actions
#         self._processed_actions[~self.command.command_leg] = (
#             no_command_leg_processed_actions.clone()
#         )

#     @property
#     def arm_raw_actions(self) -> torch.Tensor:
#         """Get the raw arm actions."""
#         return self._arm_raw_actions

#     @property
#     def arm_processed_actions(self) -> torch.Tensor:
#         """Get the processed arm actions."""
#         return self._arm_processed_actions

#     @property
#     def leg_raw_actions(self) -> torch.Tensor:
#         """Get the raw leg actions."""
#         return self._leg_raw_actions

#     @property
#     def leg_processed_actions(self) -> torch.Tensor:
#         """Get the processed leg actions."""
#         return self._leg_processed_actions
