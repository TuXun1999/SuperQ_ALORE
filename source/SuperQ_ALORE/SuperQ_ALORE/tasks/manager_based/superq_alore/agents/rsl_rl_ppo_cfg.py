# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 8000
    save_interval = 2000
    experiment_name = f"SuperQ_ALORE"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        class_name = "ActorCritic",
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        class_name = "PPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

@configclass
class PPORunnerGraspRankingCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 8000
    save_interval = 2000
    experiment_name = f"Grasp_Ranking"
    empirical_normalization = False
    class_name = "OnPolicyRunnerGraspRanking"
    return_agent_interval = 8000
    policy = RslRlPpoActorCriticCfg(
        class_name = "ActorCritic",
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        class_name = "PPOGraspRanking",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
    
@configclass
class PPOSuperQALORERunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 100
    experiment_name = "SuperQ_ALORE_GNN_3phase"
    empirical_normalization = False
    class_name = "OnPolicyRunnerSuperQALORE"
    enable_three_phase_training = True
    # Trial order for one-run curriculum.
    phase_trials = ["1", "2", "3"]
    # Iteration fractions for (phase 1, phase 2, phase 3) inside one learn() call.
    phase_fractions = {"1": 0.6, "2": 0.1, "3": 0.3}
    # Runner-side noise for privileged CoM slots in phase 2 (policy obs tail).
    phase2_objcom_noise_std = 0.03
    # Number of privileged CoM dimensions at the end of policy observation.
    policy_obj_com_dim = 3
    policy = RslRlPpoActorCriticCfg(
        class_name = "PhysicActorCritic",
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        noise_std_type="scalar",
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        class_name = "PhysicPPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        # normalize_advantage_per_mini_batch=True, NOTE: not significant
    )