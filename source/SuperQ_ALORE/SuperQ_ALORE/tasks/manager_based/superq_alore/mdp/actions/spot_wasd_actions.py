# Copyright (c) 2026
# SPDX-License-Identifier: BSD-3-Clause

"""Object-free joint-position action for the ReLIC Spot policy.

The 12-D input contains base velocity (3), arm deltas (7), and torso commands
(2). This term builds ReLIC's native 84-D observation, scales its 12 leg
outputs into joint-position targets, and applies the arm targets separately.
"""

from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.io.torchscript import load_torchscript_model
from isaaclab.envs.mdp.actions import JointAction
from isaaclab.envs.mdp.actions import JointActionCfg

from SuperQ_ALORE.assets import ASSET_DIR
import os

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


# ReLIC's raw 12-D output is joint-type-major, not Spot's native SDK order.
_RELIC_LEG_ACTION_JOINT_NAMES = (
    "fl_hx", "fr_hx", "hl_hx", "hr_hx",
    "fl_hy", "fr_hy", "hl_hy", "hr_hy",
    "fl_kn", "fr_kn", "hl_kn", "hr_kn",
)


class ReLICWASDJointPositionAction(JointAction):
    """Joint action term that applies the processed actions to the articulation's joints as position commands."""

    cfg: ReLICWASDJointPositionActionCfg
    """The configuration of the action term."""
    def __init__(
        self,
        cfg: ReLICWASDJointPositionActionCfg,
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
            
            # Combine the existing observations with the command from the high-level controller
            policy_env_obs = torch.cat(
                [
                    policy_env_obs[:, :9],
                    self._cached_base_velocity,
                    self._cached_arm_leg_joint_base_pose_command,
                    policy_env_obs[:, 9:],
                ],
                dim=1,
            )

            # Execute the locomotion policy to obtain leg actions
            leg_actions = self._locomotion_policy(policy_env_obs)
        
        # Raw actions & the processed actions (on the leg)
        self._raw_actions[:] = leg_actions
        self._processed_actions = self._raw_actions * self._scale + self._offset
        
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
        base_pose[:, 1] = self.cfg.base_height # force the height
        
        # The joint angles to command for the arm
        arm_actions = torch.zeros_like(arm_actions_delta, device=arm_actions_delta.device) # dim: 7, which is the target joint position for the arm joints

        # assign per-env active arm joint reference
        default_arm_ref = [self.cfg.arm_joint_position[name] for name in self._arm_joint_names]
        arm_reference = torch.tensor(
            default_arm_ref, device=actions.device, dtype=actions.dtype
        ).unsqueeze(0).expand(self.num_envs, -1)
        
        
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

@configclass
class ReLICWASDJointPositionActionCfg(JointActionCfg):
    """Configuration for object-free ReLIC leg-joint position control."""

    class_type: type[ActionTerm] = ReLICWASDJointPositionAction

    arm_joint_names: tuple[str, ...] = MISSING
    leg_joint_names: dict[str, list[str]] = MISSING
    arm_joint_position: dict[str, float] = MISSING
    """Absolute arm targets in radians, also used in ReLIC's pose command."""
    locomotion_policy_path: str = f"{ASSET_DIR}/spot/pretrained_relic/policy.pt"
    locomotion_obs_group: str = "locomotion_policy"
    low_level_update_decimation: int = 2
    
    use_default_offset: bool = True
    
    base_height: float = 0.55
