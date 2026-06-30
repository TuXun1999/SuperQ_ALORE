# Copyright (c) 2024 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class PhaseSettings:
    """High-level curriculum phases for SuperQ-ALORE RL training."""

    phase: str
    description: str
    episode_length_s: float
    include_policy_obj_com: bool
    policy_obj_com_noise: float
    mass_range: tuple[float, float]
    friction_range: tuple[float, float]
    com_range: dict[str, tuple[float, float]]


PHASE_1 = PhaseSettings(
    phase="1",
    description="Privileged phase: policy sees object CoM directly.",
    episode_length_s=20.0,
    include_policy_obj_com=True,
    policy_obj_com_noise=0.0,
)

PHASE_2 = PhaseSettings(
    phase="2",
    description="Robustness phase: policy sees noisy object CoM.",
    episode_length_s=25.0,
    include_policy_obj_com=True,
    policy_obj_com_noise=0.03,
)

PHASE_3 = PhaseSettings(
    phase="3",
    description="Deployment-like phase: no privileged CoM in policy observation.",
    episode_length_s=30.0,
    include_policy_obj_com=False,
    policy_obj_com_noise=0.0,
)


