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
from SuperQ_ALORE.tasks.manager_based.superq_alore.mdp.scene import ARM_JOINT_NAMES_IN_ORDER, OBJECT_TELEOPERATION_INFO
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

        # Latched high-level command buffers (updated in process_actions).
        self._cached_base_velocity = torch.zeros(self.num_envs, 3, device=self.device)
        self._cached_arm_leg_joint_base_pose_command = torch.zeros(self.num_envs, 22, device=self.device)

        # Low-level update cadence in sim steps.
        self._low_level_update_decimation = max(1, int(self.cfg.low_level_update_decimation))
        self._low_level_step_counter = 0
        
        # Speed of closing the gripper
        self.gripper_vel = self.cfg.gripper_vel
        self.gripper_closing_steps = (int)(1.5 / self._env.step_dt) # The gripper will be closed after 1.2s, which is the time duration for the gripper to close from fully open to fully closed at the speed of self.gripper_vel

    def _update_low_level_leg_actions(self):
        """Run low-level locomotion policy using the most recent latched high-level command."""
        with torch.inference_mode():
            policy_env_obs = self._env.observation_manager.compute_group(
                self.cfg.locomotion_obs_group, update_history=False
            )
            # print("Check policy_env_obs")
            # print(policy_env_obs.shape)
            # print("Check gravity proj")
            # print(policy_env_obs[[0, 10, 19], 6:9])
            # print("Check the joint pos & vel")
            # print(policy_env_obs[0, 9:28])
            # print(policy_env_obs[0, 28:47])
            # print("Theoretical index of joint")
            # print(self._joint_ids)
            # print(self._arm_joint_ids)
            # print("Name of joints")
            # print(self.cfg.leg_joint_names)
            # print(self._arm_joint_names)
            policy_env_obs = torch.cat(
                [
                    policy_env_obs[:, :9],
                    self._cached_base_velocity,
                    self._cached_arm_leg_joint_base_pose_command,
                    policy_env_obs[:, 9:],
                ],
                dim=1,
            )

            leg_actions = self._locomotion_policy(policy_env_obs)

        self._raw_actions[:] = leg_actions
        self._processed_actions = self._raw_actions * self._scale + self._offset

    def apply_actions(self):
        """Apply the actions."""
        # Run low-level policy every N sim steps (decoupled from high-level updates).
        if self._low_level_step_counter % self._low_level_update_decimation == 0:
            self._low_level_step_counter = 0
            self._update_low_level_leg_actions()
        self._low_level_step_counter += 1

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
        arm_actions_delta = actions[:, 3:10] # dim: 7
        
        # Extract the "command" for the low-level controller
        base_velocity = actions[:, :3]
        # Extract the base pose command (pitch and height, roll is set to be zero)
        base_pose = actions[:, 10:12] # dim: 2 (pitch, height)
        base_pose[:, 0] = 0.0 # zero roll command, which is not desired for the task
        base_pose[:, 1] = 0.55 # force the height
        # The joint angles to command for the arm
        arm_actions = torch.zeros_like(arm_actions_delta, device=arm_actions_delta.device) # dim: 7, which is the target joint position for the arm joints

        # assign per-env active arm joint reference
        arm_joint_names = [joint_name for joint_name in GRASP_POSE_1_JOINT_POS.keys() if joint_name.startswith("arm")]
        arm_reference = (
            self._env.active_arm_joint_reference[:, : arm_actions.shape[1]]
            if hasattr(self._env, "active_arm_joint_reference")
            else torch.tensor(
                [GRASP_POSE_1_JOINT_POS[joint_name] for joint_name in arm_joint_names],
                device=self._env.unwrapped.device,
            ).unsqueeze(0).expand(self.num_envs, -1)
        )
        
        
        # Grip the object in the beginning, and maintain the gripper pose after that
        gripper_closing_mask = self._env.episode_length_buf > 1
        gripper_target = torch.clamp(
            -0.9 + self._env.episode_length_buf * self.gripper_vel, max=-0.15
        )
        arm_actions[:, -1] = torch.where(
            gripper_closing_mask,
            gripper_target,
            arm_actions[:, -1],
        )
            
        start_moving_mask = self._env.episode_length_buf < self.gripper_closing_steps
        
        # """
        # Section I: For the robot that are still closing the gripper
        # Arm joint: use the default ones read from the pre-calculated files
        # (FAILED) Leg joint: use the PD controller to force the robot to stand still
        # """
        startup_arm_reference = arm_reference[:, :-1]
        moving_arm_reference = arm_reference[:, :-1] + arm_actions_delta[:, :-1]
        arm_actions[:, :-1] = torch.where(
            start_moving_mask.unsqueeze(-1),
            startup_arm_reference,
            moving_arm_reference,
        )

        # Also, force the robot to stand
        base_velocity = torch.where(
            start_moving_mask.unsqueeze(-1),
            torch.zeros_like(base_velocity),
            base_velocity,
        )

        # Latch high-level command for low-level controller.
        arm_joints = arm_actions
        leg_joints = torch.zeros(arm_joints.shape[0], 12, device=arm_joints.device)
        roll_target = torch.zeros(arm_joints.shape[0], 1, device=arm_joints.device)
        arm_leg_joint_base_pose_command = torch.cat(
            [
                arm_joints,
                leg_joints,
                roll_target,
                base_pose,
            ],
            dim=1,
        )
        assert arm_leg_joint_base_pose_command.shape[1] == 22, "Whole-body pose shape incorrect"
        self._cached_base_velocity[:] = base_velocity
        self._cached_arm_leg_joint_base_pose_command[:] = arm_leg_joint_base_pose_command
        
        
        # Execute the action directly (according to ALORE)
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

class MixedPDArmMultiLegJointPositionActionTele(JointAction):
    """Joint action term that applies the processed actions to the articulation's joints as position commands."""

    cfg: spot_actions_cfg.MixedPDArmMultiLegJointPositionActionTeleCfg
    """The configuration of the action term."""
    def __init__(
        self,
        cfg: spot_actions_cfg.MixedPDArmMultiLegJointPositionActionTeleCfg,
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

        # Latched high-level command buffers (updated in process_actions).
        self._cached_base_velocity = torch.zeros(self.num_envs, 3, device=self.device)
        self._cached_arm_leg_joint_base_pose_command = torch.zeros(self.num_envs, 22, device=self.device)

        # Low-level update cadence in sim steps.
        self._low_level_update_decimation = max(1, int(self.cfg.low_level_update_decimation))
        self._low_level_step_counter = 0

    def _update_low_level_leg_actions(self):
        """Run low-level locomotion policy using the most recent latched high-level command."""
        with torch.inference_mode():
            policy_env_obs = self._env.observation_manager.compute_group(
                self.cfg.locomotion_obs_group, update_history=False
            )
            # arm_indices = [0, 5, 10, 15, 16, 17]
            # leg_indices = [1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14]
            # critic_env_obs = self._env.observation_manager.compute_group(
            #     "critic", update_history=False
            # )
            # print("=== Test critic obs ===")
            # joints_info = critic_env_obs[0, 0:72]
            # joint_angles_rel = joints_info[0:18]
            # joint_angles_abs = joints_info[54:72]
            # print("Leg joints (abs, ?): ", joint_angles_abs[leg_indices])

            policy_env_obs = torch.cat(
                [
                    policy_env_obs[:, :9],
                    self._cached_base_velocity,
                    self._cached_arm_leg_joint_base_pose_command,
                    policy_env_obs[:, 9:],
                ],
                dim=1,
            )

            leg_actions = self._locomotion_policy(policy_env_obs)
            # print("=== Test leg actions ===")
            
        self._raw_actions[:] = leg_actions
        self._processed_actions = self._raw_actions * self._scale + self._offset
        # print("Executed actions")
        # print(self._processed_actions)
        
    def apply_actions(self):
        """Apply the actions."""
        # Run low-level policy every N sim steps (decoupled from high-level updates).
        if self._low_level_step_counter % self._low_level_update_decimation == 0:
            self._update_low_level_leg_actions()
        self._low_level_step_counter += 1

        # set position targets
        # (The reference of zero should be the ones in software when importing the robot
        # not the physical ones)
        self._asset.set_joint_position_target(
            self.processed_actions, joint_ids=self._joint_ids
        )
        self._asset.set_joint_position_target(
            self.arm_processed_actions, joint_ids=self._arm_joint_ids
        )
    
    """Overload the process_action term (Remove the initial closing stage)"""
    def process_actions(self, actions: torch.Tensor):
        # Extract the arm actions from the input actions
        arm_actions_delta = actions[:, 3:10] # dim: 7
        
        # Extract the "command" for the low-level controller
        base_velocity = actions[:, :3]
        # Extract the base pose command (pitch and height, roll is set to be zero)
        base_pose = actions[:, 10:12] # dim: 2 (pitch, height)
        base_pose[:, 0] = 0.0 # zero roll command, which is not desired for the task
        base_pose[:, 1] = 0.55 # force the height
        # The joint angles to command for the arm
        arm_actions = torch.zeros_like(arm_actions_delta, device=arm_actions_delta.device) # dim: 7, which is the target joint position for the arm joints

        # assign per-env active arm joint reference
        default_arm_ref = [OBJECT_TELEOPERATION_INFO[1][key] for key in ARM_JOINT_NAMES_IN_ORDER]
        arm_reference = torch.tensor(default_arm_ref, device=actions.device).unsqueeze(0).expand(self.num_envs, -1)
        
        
        # Gripper joint set as constantly open so gripper can stay open
        arm_actions[:, -1] = arm_reference[:, -1]

        # For moving episodes, offset the per-env reference with policy delta.
        arm_actions[:, :-1] = (
            arm_reference[:, :-1] + arm_actions_delta[:, :-1]
        )

        # Latch high-level command for low-level controller.
        arm_joints = arm_actions
        leg_joints = torch.zeros(arm_joints.shape[0], 12, device=arm_joints.device)
        roll_target = torch.zeros(arm_joints.shape[0], 1, device=arm_joints.device)
        arm_leg_joint_base_pose_command = torch.cat(
            [
                arm_joints,
                leg_joints,
                roll_target,
                base_pose,
            ],
            dim=1,
        )
        assert arm_leg_joint_base_pose_command.shape[1] == 22, "Whole-body pose shape incorrect"
        self._cached_base_velocity[:] = base_velocity
        self._cached_arm_leg_joint_base_pose_command[:] = arm_leg_joint_base_pose_command
        
        
        # Execute the action directly (according to ALORE)
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