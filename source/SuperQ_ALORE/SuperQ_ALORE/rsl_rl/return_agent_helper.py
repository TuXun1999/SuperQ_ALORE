# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from tensordict import TensorDict


class ReturnAgentHelper:
    """Utility to train and checkpoint a small non-privileged return-regression agent."""

    def __init__(self, cfg: dict, device: str) -> None:
        self.device = device
        
        # Return estimation agent (during middle of PPO training) configuration
        self.return_agent_interval = int(cfg.get("return_agent_interval", 1000)) # By default, only at the end of the training
        self.return_agent_train_steps = int(cfg.get("return_agent_train_steps", 1000))
        self.return_agent_batch_size = int(cfg.get("return_agent_batch_size", 256))
        self.return_agent_lr = float(cfg.get("return_agent_lr", 1e-3))
        self.gamma = float(cfg.get("gamma", 0.99))
        self.return_agent_hidden_dims = cfg.get("return_agent_hidden_dims", [256, 128])
        self.save_return_agent_default = True # Note: Force to be true...

        # Return estimation agent (after final PPO rollout) configuration
        self.final_finetune_steps = int(cfg.get("return_agent_final_finetune_steps", 1000))
        self.final_finetune_batch_size = int(cfg.get("return_agent_final_finetune_batch_size", 256))
        self.final_finetune_lr = float(cfg.get("return_agent_final_finetune_lr", self.return_agent_lr))
        
        
        self.return_agent: nn.Module | None = None
        self.return_agent_optimizer: torch.optim.Optimizer | None = None
        self.writer: Any = None
        self.logger_type: str = "tensorboard"
        self.disable_logs: bool = True
        self.log_dir: str | None = None
        
        

    def bind_logger(self, writer: Any, logger_type: str, disable_logs: bool, log_dir: str | None) -> None:
        """Bind the runner logger so helper metrics use the same backend."""
        self.writer = writer
        self.logger_type = logger_type
        self.disable_logs = disable_logs
        self.log_dir = log_dir

    def bind_gamma(self, gamma: float) -> None:
        """Bind the PPO discount factor so final experiment returns use the same gamma."""
        self.gamma = float(gamma)

    def _build_return_agent(self, input_dim: int) -> nn.Module:
        """Create a small MLP that regresses critic-estimated returns from policy observations."""
        hidden_dims = [int(d) for d in self.return_agent_hidden_dims]
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        return nn.Sequential(*layers).to(self.device)

    def _ensure_return_agent(self, policy_obs: torch.Tensor) -> None:
        """Initialize the small return-regression agent lazily once observation dimensions are known."""
        if self.return_agent is not None:
            return
        input_dim = int(policy_obs.shape[-1])
        self.return_agent = self._build_return_agent(input_dim)
        self.return_agent_optimizer = torch.optim.Adam(self.return_agent.parameters(), lr=self.return_agent_lr)

    @staticmethod
    def _extract_policy_obs(obs: TensorDict) -> torch.Tensor:
        """Extract non-privileged observation tensor used as input for the small agent."""
        if isinstance(obs, TensorDict) and "grasp_ranking" in obs.keys():
            return obs["grasp_ranking"]
        if torch.is_tensor(obs):
            return obs
        raise ValueError("Unable to extract non-privileged policy observations from environment outputs.")

    @staticmethod
    def _reset_and_get_observations(env: Any, device: str) -> TensorDict:
        """Reset env and return observations in a robust way across reset API variants."""
        reset_out = env.reset()
        if isinstance(reset_out, tuple):
            reset_obs = reset_out[0]
        elif reset_out is None:
            reset_obs = env.get_observations()
        else:
            reset_obs = reset_out
        return reset_obs.to(device)

    def train_from_critic(
        self,
        it: int,
        env: Any,
        policy: Any,
    ) -> TensorDict | None:
        """Periodically distill critic-estimated initial-state returns into a small non-privileged agent."""
        if self.return_agent_interval <= 0 or self.return_agent_train_steps <= 0:
            return None
        if (it + 1) % self.return_agent_interval != 0:
            return None
        with torch.inference_mode():
            # Requirement: reset environment every interval before computing initial-state targets.
            obs_reset = self._reset_and_get_observations(env, self.device)

        with torch.inference_mode():
            target_returns = policy.evaluate(obs_reset).detach().view(-1, 1)
        policy_obs = self._extract_policy_obs(obs_reset).detach()

        self._ensure_return_agent(policy_obs)
        assert self.return_agent is not None
        assert self.return_agent_optimizer is not None

        self.return_agent.train()
        num_samples = policy_obs.shape[0]
        batch_size = min(self.return_agent_batch_size, num_samples)
        mse_loss = nn.MSELoss()
        final_loss = 0.0
        start_loss = 0.0

        for step in range(self.return_agent_train_steps):
            sample_ids = torch.randint(0, num_samples, (batch_size,), device=self.device)
            batch_obs = policy_obs[sample_ids]
            batch_target = target_returns[sample_ids]

            pred_returns = self.return_agent(batch_obs)
            loss = mse_loss(pred_returns, batch_target)

            self.return_agent_optimizer.zero_grad()
            loss.backward()
            self.return_agent_optimizer.step()
            loss_value = float(loss.item())
            if step == 0:
                start_loss = loss_value
            final_loss = loss_value

            if self.log_dir is not None and not self.disable_logs and self.writer is not None:
                self.writer.add_scalar("ReturnAgent/train_step_loss", loss_value, step)

        # if self.log_dir is not None and not self.disable_logs and self.writer is not None:
        #     with torch.inference_mode():
        #         pred_mean = float(self.return_agent(policy_obs).mean().item())
        #     self.writer.add_scalar("ReturnAgent/loss", final_loss, it)
        #     self.writer.add_scalar("ReturnAgent/start_loss", start_loss, it)
        #     self.writer.add_scalar("ReturnAgent/loss_delta", start_loss - final_loss, it)
        #     self.writer.add_scalar("ReturnAgent/target_mean", float(target_returns.mean().item()), it)
        #     self.writer.add_scalar("ReturnAgent/pred_mean", pred_mean, it)

        print(
            f"[SuperQ-ALORE] Return-agent distillation at iter {it}: "
            f"steps={self.return_agent_train_steps}, loss={final_loss:.6f}"
        )

        return obs_reset

    def collect_final_experiment_returns(self, env: Any, policy: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Roll out the final policy from a reset state and collect empirical episode returns.

        Returns:
            A tuple ``(policy_obs, returns)`` where ``policy_obs`` contains the initial non-privileged observations and
            ``returns`` contains the realized episodic return for each sampled environment.
        """
        
        # Reset the environment and get initial observations
        with torch.inference_mode():
            obs_reset = self._reset_and_get_observations(env, self.device)
        policy_obs = self._extract_policy_obs(obs_reset).detach()
        num_envs = int(policy_obs.shape[0])

        episode_returns = torch.zeros(num_envs, dtype=torch.float, device=self.device)
        finished = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        discounts = torch.ones(num_envs, dtype=torch.float, device=self.device)
        max_steps = int(getattr(env, "max_episode_length", 0))
        if max_steps <= 0:
            max_steps = 1

        # Freeze the actor policy temporarily to avoid any training updates
        policy_was_training = getattr(policy, "training", False)
        if hasattr(policy, "eval"):
            policy.eval()

        obs = obs_reset
        # Collect empirical returns by rolling out the policy in the environment
        with torch.inference_mode():
            for _ in range(max_steps):
                actions = policy.act_inference(obs)
                obs, rewards, dones, extras = env.step(actions.to(env.device))
                obs = obs.to(self.device)
                rewards = rewards.to(self.device)
                dones = dones.to(self.device)
                active_mask = (~finished).float()
                episode_returns += rewards * discounts * active_mask
                discounts = discounts * self.gamma * active_mask
                finished = torch.logical_or(finished, dones > 0)
                if bool(finished.all()):
                    break
        
        # Resume the training status of original policy if it was in training mode
        # (This should not be actually necessary, but we do it for safety.)
        if policy_was_training and hasattr(policy, "train"):
            policy.train()

        return policy_obs, episode_returns

    def finetune_from_experiment_returns(self, policy_obs: torch.Tensor, returns: torch.Tensor, it: int) -> None:
        """Finetune the small agent on real returns collected from a final experimental rollout."""
        self._ensure_return_agent(policy_obs)
        assert self.return_agent is not None

        finetune_lr = self.final_finetune_lr
        optimizer = torch.optim.Adam(self.return_agent.parameters(), lr=finetune_lr)

        self.return_agent.train()
        num_samples = int(policy_obs.shape[0])
        batch_size = min(self.final_finetune_batch_size, num_samples)
        mse_loss = nn.MSELoss()
        final_loss = 0.0
        start_loss = 0.0

        for step in range(self.final_finetune_steps):
            sample_ids = torch.randint(0, num_samples, (batch_size,), device=self.device)
            batch_obs = policy_obs[sample_ids]
            batch_target = returns[sample_ids].view(-1, 1)

            pred_returns = self.return_agent(batch_obs)
            loss = mse_loss(pred_returns, batch_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_value = float(loss.item())
            if step == 0:
                start_loss = loss_value
            final_loss = loss_value

            if self.log_dir is not None and not self.disable_logs and self.writer is not None:
                self.writer.add_scalar("ReturnAgent/final_finetune_step_loss", loss_value, step)

        if self.log_dir is not None and not self.disable_logs and self.writer is not None:
            with torch.inference_mode():
                pred_mean = float(self.return_agent(policy_obs).mean().item())
            self.writer.add_scalar("ReturnAgent/final_finetune_loss", final_loss, it)
            self.writer.add_scalar("ReturnAgent/final_finetune_start_loss", start_loss, it)
            self.writer.add_scalar("ReturnAgent/final_finetune_loss_delta", start_loss - final_loss, it)
            self.writer.add_scalar("ReturnAgent/final_real_return_mean", float(returns.mean().item()), it)
            self.writer.add_scalar("ReturnAgent/final_pred_mean", pred_mean, it)

        print(
            f"[SuperQ-ALORE] Return-agent finetune from experimental rollout at iter {it}: "
            f"steps={self.final_finetune_steps}, loss={final_loss:.6f}"
        )

    def augment_save_payload(self, saved_dict: dict, infos: dict | None) -> None:
        """Optionally attach small return-agent parameters to checkpoint payload."""
        save_return_agent = self.save_return_agent_default
        if isinstance(infos, dict):
            save_return_agent = bool(infos.get("save_return_agent", save_return_agent))
        if save_return_agent and self.return_agent is not None:
            saved_dict["return_agent_state_dict"] = self.return_agent.state_dict()
            if self.return_agent_optimizer is not None:
                saved_dict["return_agent_optimizer_state_dict"] = self.return_agent_optimizer.state_dict()

    def load_from_checkpoint(self, loaded_dict: dict, load_optimizer: bool) -> None:
        """Restore small return-agent state from checkpoint payload if available."""
        if "return_agent_state_dict" not in loaded_dict:
            raise ValueError("Checkpoint payload does not contain return-agent state dictionary.")
            

        state_dict = loaded_dict["return_agent_state_dict"]
        first_linear_key = next((k for k in state_dict.keys() if k.endswith("0.weight")), None)
        if first_linear_key is None:
            return

        input_dim = int(state_dict[first_linear_key].shape[1])
        if self.return_agent is None:
            self.return_agent = self._build_return_agent(input_dim)
            self.return_agent_optimizer = torch.optim.Adam(self.return_agent.parameters(), lr=self.return_agent_lr)
        self.return_agent.load_state_dict(state_dict)

        if (
            load_optimizer
            and self.return_agent_optimizer is not None
            and "return_agent_optimizer_state_dict" in loaded_dict
        ):
            self.return_agent_optimizer.load_state_dict(loaded_dict["return_agent_optimizer_state_dict"])
