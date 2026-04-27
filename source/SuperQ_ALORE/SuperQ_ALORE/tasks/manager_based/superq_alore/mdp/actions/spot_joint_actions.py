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
from SuperQ_ALORE.assets.spot.constants import GRASP_POSE_1_JOINT_POS
from SuperQ_ALORE.assets.spot.constants import HIP_STIFFNESS, HIP_DAMPING, KNEE_STIFFNESS, KNEE_DAMPING
# Input: high-level controller action
# Output: process it
# into low-level joint control actions
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
        # Load the pretrained policy as the low-level controller
        self._locomotion_policy = load_torchscript_model(self.policy_path, device=self.device)
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
        
        # Speed of closing the gripper
        self.gripper_vel = self.cfg.gripper_vel
        self.gripper_closing_steps = (int)(1.5 / self._env.step_dt) # The gripper will be closed after 1.2s, which is the time duration for the gripper to close from fully open to fully closed at the speed of self.gripper_vel

    def apply_actions(self):
        """Apply the actions."""
        # set position targets
        # (The reference of zero should be the ones in software when importing the robot
        # not the physical ones)
        self._asset.set_joint_position_target(
            self.processed_actions, joint_ids=self._joint_ids
        )
        self._asset.set_joint_position_target(
            self.arm_processed_actions, joint_ids=self._arm_joint_ids
        )

    def process_actions(self, actions: torch.Tensor):
        """Process the actions."""
        """
        Originally: actions are leg joint actions (dim: 12, verified)
        The arm actions are directly following the commands using linear interpolation
        The leg joint actions are split into two groups:
            a) commanded leg: directly follow the commands using linear interpolation
            b) non-commanded leg: directly execute the predicted actions from agent after scaling
            
        Now: actions are commands for ReLIC to track (arm joints, base pose, base velocities)
        Order (following the convention in CommandCfg, no leg joint tracking):
        base velocity, arm joints, base pose (input is pitch + height, roll is set to be zero)
        (3 + 7 + 3)
        
        The program consists of several steps
        1. Load the pretrained weights
        2. Organize the obs to generate the input to the model
        3. Predict joint-level actions (for the legs)
        4. Predict the arm joint actions & leg joint actions
        """
        # Extract the arm actions from the input actions
        arm_actions = actions[:, 3:10] # dim: 7
        # Extract the "command" for the low-level controller
        base_velocity = actions[:, :3]
        # Extract the base pose command (pitch and height, roll is set to be zero)
        base_pose = actions[:, 10:12] # dim: 2 (pitch, height)
        base_pose[:, 0] = 0.0 # zero roll command, which is not desired for the task
        base_pose[:, 1] = 0.55 # force the height
        leg_actions = torch.zeros(actions.shape[0], 12).to(actions.device) # dim: 12
        
        
        # Grip the object in the beginning, and maintain the gripper pose after that
        gripper_closing_mask = self._env.episode_length_buf > 1 
        if gripper_closing_mask.any(): 
            arm_actions[gripper_closing_mask, -1] = torch.clamp(
                -0.9 + self._env.episode_length_buf[gripper_closing_mask]* self.gripper_vel, max= -0.15
            )
            
        start_moving_mask = self._env.episode_length_buf < self.gripper_closing_steps
        
        """
        Section I: For the robot that are still closing the gripper
        Arm joint: use the default ones read from the pre-calculated files
        (FAILED) Leg joint: use the PD controller to force the robot to stand still
        """
        if start_moving_mask.any():
            # Obtain the Pre-calculated joint positions for the grasp pose
            arm_joint_names = [joint_name for joint_name in GRASP_POSE_1_JOINT_POS.keys() if joint_name.startswith("arm")]
            # leg_joint_names = ["fl_hx", "fr_hx", "hl_hx", "hr_hx", "fl_hy", "fr_hy", "hl_hy", "hr_hy", "fl_kn", "fr_kn", "hl_kn", "hr_kn"]
            arm_grasp_pose_by_default = torch.tensor([GRASP_POSE_1_JOINT_POS[joint_name] for joint_name in arm_joint_names], device=self._env.unwrapped.device)
            # leg_grasp_pose_by_default = torch.tensor([GRASP_POSE_1_JOINT_POS[joint_name] for joint_name in leg_joint_names], device=self._env.unwrapped.device)

            arm_actions[start_moving_mask] = arm_grasp_pose_by_default
            # leg_actions[start_moving_mask] = leg_grasp_pose_by_default
            
            # Also, force the robot to stand
            base_velocity[start_moving_mask] = torch.zeros(3, device=base_velocity.device) # zero base velocity
            


        """
        Section II: For the robot that have completed the gripper closing
        Arm joint: use the ones generated from the high-level controller
        Leg joint: use the predicted actions from the low-level controller
        """
        # if (~start_moving_mask).any():
            #     arm_joint_current_pos = self._asset.data.joint_pos[:, self._arm_joint_ids]
            #     arm_joint_current_pos = arm_joint_current_pos[~start_moving_mask]
            #     arm_actions[~start_moving_mask] = arm_actions[~start_moving_mask]
            
        with torch.inference_mode():
            # The environmental policy observations
            policy_env_obs = self._env.observation_manager.compute_group(
                self.cfg.locomotion_obs_group, update_history=False
            ) # update_history: recommended to be set false

            # Insert the commands into the policy obs
            """
            Locomotion policy obs: (dim: 84, verified) 
            base_lin_vel (3), base_ang_vel (3), projected_gravity (3), 
            (velocity commands (3), commands (22)),
            joint_pos (19), joint_vel (19)
            (last_actions (12))
            
            velocity commands: base velocity from command (dim: 3)
            commands: arm leg joint & base pose from command (dim: 7 + 12 + 3 = 22, verified)
            last_actions:
            For ReLIC, last_action are the leg joint actions predicted (12) by the agent
            and applied to the environment
            However, now the agent doesn't predict the leg joint actions directly...
            So, we need to extract last_action manually in observations.py
            """
            arm_joints = arm_actions # dim: 7]

            # In ReLIC indicate that if 
            # the legs are not the commanded ones, just set up the 
            # leg_joint_command to be zero to de-activate the leg tracking 
            # functionality
            leg_joints = torch.zeros(arm_joints.shape[0], 12).to(arm_joints.device)
            
            # We don't want a roll operation
            roll_target = torch.zeros(arm_joints.shape[0], 1).to(arm_joints.device)
            # Construct the "command" to track used in ReLIC
            arm_leg_joint_base_pose_command = torch.cat(
                [
                    arm_joints,
                    leg_joints,
                    roll_target,
                    base_pose,
                ],
                dim=1,
            ) # dim: 22
            assert arm_leg_joint_base_pose_command.shape[1] == 22, "Whole-body pose shape incorrect"
            
            # The input to low-level controller consists of everything
            policy_env_obs = torch.cat(
                [
                    policy_env_obs[:, :9],
                    base_velocity, 
                    arm_leg_joint_base_pose_command, 
                    policy_env_obs[:, 9:]
                ],
                dim = 1
            )
            
            ## Step 3: predict leg actions
            leg_actions = self._locomotion_policy(policy_env_obs)
    
        
        
        
        """
        Wrap up the actions & Execute them
        """
        # store the raw leg actions, which is used by the low-level controller
        self._raw_actions[:] = leg_actions
        # apply the affine transformations
        self._processed_actions = self._raw_actions * self._scale + self._offset
        
        
        # Execute the action directly (according to ALORE)
        # TODO: modify it into relative pose execution
        # arm_current = self._asset.data.joint_pos[:, self._arm_joint_ids]
        self._arm_raw_actions[:] = arm_actions
        self._arm_processed_actions[:] = self._arm_raw_actions.clone()


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
