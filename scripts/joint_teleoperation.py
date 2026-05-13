# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Keyboard teleoperation for Spot arm joints with optional base motion mode.

Controls (Windows terminal):
- 1..7: select arm joint index
- a: increase selected joint angle by +step_size (arm mode)
- d: decrease selected joint angle by -step_size (arm mode)
- b: toggle base motion mode on/off
- w/a/s/d: control base velocity in motion mode
- r: reset arm targets to initial values
- p: print current targets
- q: quit

The script keeps non-arm action components fixed, so only Spot arm joint targets are changed.
"""

import argparse
import math
import msvcrt

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Spot arm joint teleoperation (keyboard).")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--step_size", type=float, default=0.05, help="Joint angle increment/decrement in radians.")
parser.add_argument(
    "--step_size_deg",
    type=float,
    default=None,
    help="Joint angle increment/decrement in degrees. If set, overrides --step_size.",
)
parser.add_argument(
    "--base_lin_vel",
    type=float,
    nargs=3,
    default=(0.0, 0.0, 0.0),
    metavar=("VX", "VY", "WZ"),
    help="Fixed base velocity command [vx, vy, wz].",
)
parser.add_argument(
    "--base_pose",
    type=float,
    nargs=2,
    default=(0.0, 0.0),
    metavar=("PITCH", "HEIGHT"),
    help="Fixed base pose command [pitch, height].",
),
parser.add_argument(
    "--motion_speed",
    type=float,
    default=0.2,
    help="Base speed magnitude (m/s) used by WASD while in motion mode.",
)
parser.add_argument(
    "--init_from_grasp_pose",
    action="store_true",
    default=True,
    help="Initialize arm targets from GRASP_POSE_1_JOINT_POS.",
)
parser.add_argument(
    "--init_from_zero",
    action="store_true",
    default=False,
    help="Initialize arm targets as zeros instead of GRASP_POSE_1_JOINT_POS.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import isaaclab_tasks  # noqa: F401
import SuperQ_ALORE.tasks  # noqa: F401
import torch
from isaaclab_tasks.utils import parse_env_cfg

from SuperQ_ALORE.assets.spot.constants import ARM_JOINT_NAMES, GRASP_POSE_1_JOINT_POS
from SuperQ_ALORE.assets.object_catalog import (
    ARM_JOINT_NAMES_IN_ORDER,
    OBJECT_CATALOG,
    OBJECT_IDS,
    POSE_IDS_BY_OBJECT,
)
from SuperQ_ALORE.tasks.manager_based.superq_alore.mdp import event as mdp_event
from SuperQ_ALORE.tasks.manager_based.superq_alore.mdp import object_management as mdp_object_management

# Teleoperation catalog selection (file-level settings; not CLI):
TELEOP_OBJECT_ID = "chair_1"
TELEOP_POSE_ID = "back"


def _build_arm_joint_names() -> list[str]:
    """Return ordered Spot arm joint names used by the action head (7 joints)."""
    names = [name for name in ARM_JOINT_NAMES if name.startswith("arm")]
    if len(names) >= 7:
        return names[:7]

    # Fallback to grasp-pose dictionary order if constants do not contain all joints.
    fallback = [name for name in GRASP_POSE_1_JOINT_POS.keys() if name.startswith("arm")]
    return fallback[:7]


def _apply_fixed_catalog_selection(env) -> None:
    """Force env 0 to a fixed (object, pose) from the YAML catalog.

    This operates through catalog state tensors directly, then applies object and robot
    reset for env 0 only, with zero reset noise.
    """
    if TELEOP_OBJECT_ID not in OBJECT_IDS:
        raise ValueError(
            f"Invalid TELEOP_OBJECT_ID='{TELEOP_OBJECT_ID}'. Available: {list(OBJECT_IDS)}"
        )
    if TELEOP_POSE_ID not in POSE_IDS_BY_OBJECT[TELEOP_OBJECT_ID]:
        raise ValueError(
            f"Invalid TELEOP_POSE_ID='{TELEOP_POSE_ID}' for object '{TELEOP_OBJECT_ID}'. "
            f"Available: {list(POSE_IDS_BY_OBJECT[TELEOP_OBJECT_ID])}"
        )

    mdp_object_management.ensure_catalog_state(env)

    env_id = torch.tensor([0], dtype=torch.long, device=env.device)
    obj_idx = OBJECT_IDS.index(TELEOP_OBJECT_ID)
    pose_idx = POSE_IDS_BY_OBJECT[TELEOP_OBJECT_ID].index(TELEOP_POSE_ID)
    pose_entry = OBJECT_CATALOG[obj_idx].poses[pose_idx]

    env.active_object_indices[env_id] = obj_idx
    env.active_pose_indices[env_id] = pose_idx
    env.active_arm_joint_reference[0] = torch.tensor(
        [pose_entry.joint_positions[name] for name in ARM_JOINT_NAMES_IN_ORDER],
        dtype=torch.float32,
        device=env.device,
    )
    env.target_assignment_ready[env_id] = True

    # Apply object placement and robot joint reset for env 0.
    origins = env.scene.env_origins[env_id]
    pose_pos = torch.tensor([pose_entry.position], dtype=torch.float32, device=env.device)
    pose_rot = torch.tensor([pose_entry.orientation], dtype=torch.float32, device=env.device)
    underground_pos = origins.clone()
    underground_pos[:, 2] = -100.0
    underground_rot = torch.zeros_like(pose_rot)
    underground_rot[:, 0] = 1.0

    for idx in range(len(OBJECT_CATALOG)):
        obj = env.scene[f"target_object_{idx}"]
        state = obj.data.default_root_state[env_id].clone()
        state[:, 7:] = 0.0
        if idx == obj_idx:
            state[:, 0:3] = origins + pose_pos
            state[:, 3:7] = pose_rot
        else:
            state[:, 0:3] = underground_pos
            state[:, 3:7] = underground_rot
        obj.write_root_state_to_sim(state, env_ids=env_id)

    mdp_event.reset_joints_around_grasp_pose(
        env=env,
        env_ids=env_id,
        position_range=(0.0, 0.0),
        velocity_range=(0.0, 0.0),
        joint_position_ref={
            name: float(pose_entry.joint_positions[name])
            for name in ARM_JOINT_NAMES_IN_ORDER
        },
    )


def _print_help(arm_joint_names: list[str], selected_idx: int, targets: torch.Tensor, step_size: float):
    print("\n[TELEOP] Controls:")
    print("  1a / 1d .. 7a / 7d : select joint and apply +/- delta in one sequence")
    print("  1..7               : select joint")
    print("  a / d              : increase/decrease selected joint (arm mode)")
    print("  b                  : toggle motion mode on/off")
    print("  w/a/s/d            : base control in motion mode")
    print("                       w:+vx, s:-vx, a:+vy, d:-vy")
    print("  r    : reset all joints to initial targets")
    print("  p    : print current joint targets")
    print("  q    : quit")
    print(f"[TELEOP] step_size = {step_size:.4f} rad")
    print(f"[TELEOP] selected joint = {selected_idx + 1} ({arm_joint_names[selected_idx]})")
    print("[TELEOP] initial targets:")
    for i, (name, value) in enumerate(zip(arm_joint_names, targets.tolist()), start=1):
        print(f"  {i}: {name:>12s} = {value:+.4f}")


def main():
    """Run keyboard teleoperation for arm joints only."""
    # Teleoperation is intended for a single environment.
    if args_cli.num_envs != 1:
        print(f"[TELEOP] overriding --num_envs={args_cli.num_envs} -> 1")

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=1,
        use_fabric=not args_cli.disable_fabric,
    )

    env = gym.make(args_cli.task, cfg=env_cfg)

    print(f"[INFO] Gym observation space: {env.observation_space}")
    print(f"[INFO] Gym action space: {env.action_space}")

    env.reset()
    _apply_fixed_catalog_selection(env.unwrapped)
    print(
        "[TELEOP] fixed catalog selection: "
        f"object='{TELEOP_OBJECT_ID}', pose='{TELEOP_POSE_ID}'"
    )

    arm_joint_names = _build_arm_joint_names()
    if len(arm_joint_names) != 7:
        raise RuntimeError(
            f"Expected 7 Spot arm joints, but got {len(arm_joint_names)} from constants: {arm_joint_names}"
        )

    if args_cli.init_from_zero:
        arm_targets = torch.zeros(7, device=env.unwrapped.device)
    else:
        arm_targets = env.unwrapped.active_arm_joint_reference[0, :7].clone()

    arm_targets_init = arm_targets.clone()
    selected_joint_idx = 0
    step_size = math.radians(args_cli.step_size_deg) if args_cli.step_size_deg is not None else args_cli.step_size

    fixed_base_lin_vel = torch.tensor(args_cli.base_lin_vel, device=env.unwrapped.device, dtype=torch.float32)
    fixed_base_pose = torch.tensor(args_cli.base_pose, device=env.unwrapped.device, dtype=torch.float32)
    motion_mode = False
    motion_speed = float(args_cli.motion_speed)

    _print_help(arm_joint_names, selected_joint_idx, arm_targets, step_size)

    def _apply_delta(joint_idx: int, direction: str):
        if direction == "a":
            arm_targets[joint_idx] += step_size
        elif direction == "d":
            arm_targets[joint_idx] -= step_size
        print(
            f"[TELEOP] {arm_joint_names[joint_idx]} -> "
            f"{arm_targets[joint_idx].item():+.4f} rad"
        )

    while simulation_app.is_running():
        with torch.inference_mode():
            # Consume all pending key presses (non-blocking).
            while msvcrt.kbhit():
                key = msvcrt.getwch().lower()

                if key.isdigit():
                    joint_num = int(key)
                    if 1 <= joint_num <= len(arm_joint_names):
                        selected_joint_idx = joint_num - 1

                        # Support compact combo control like: 1a, 1d, 5a, 5d.
                        if (not motion_mode) and msvcrt.kbhit():
                            maybe_dir = msvcrt.getwch().lower()
                            if maybe_dir in ("a", "d"):
                                _apply_delta(selected_joint_idx, maybe_dir)
                            else:
                                print(
                                    f"[TELEOP] selected joint {joint_num}: {arm_joint_names[selected_joint_idx]} "
                                    f"(current {arm_targets[selected_joint_idx].item():+.4f} rad)"
                                )
                        else:
                            print(
                                f"[TELEOP] selected joint {joint_num}: {arm_joint_names[selected_joint_idx]} "
                                f"(current {arm_targets[selected_joint_idx].item():+.4f} rad)"
                            )
                    else:
                        print(f"[TELEOP] invalid joint index '{joint_num}', expected 1..{len(arm_joint_names)}")

                elif key == "a":
                    if motion_mode:
                        fixed_base_lin_vel[:] = torch.tensor((0.0, motion_speed, 0.0), device=env.unwrapped.device)
                        print(
                            "[TELEOP] motion mode command: left "
                            f"(vx=0.0, vy={motion_speed:+.2f}, wz=0.0)"
                        )
                    else:
                        _apply_delta(selected_joint_idx, key)

                elif key == "d":
                    if motion_mode:
                        fixed_base_lin_vel[:] = torch.tensor((0.0, -motion_speed, 0.0), device=env.unwrapped.device)
                        print(
                            "[TELEOP] motion mode command: right "
                            f"(vx=0.0, vy={-motion_speed:+.2f}, wz=0.0)"
                        )
                    else:
                        _apply_delta(selected_joint_idx, key)

                elif key == "b":
                    motion_mode = not motion_mode
                    fixed_base_lin_vel[:] = torch.zeros(3, device=env.unwrapped.device)
                    if motion_mode:
                        print(
                            "[TELEOP] motion mode ON. Use WASD to move base at "
                            f"{motion_speed:.2f} m/s."
                        )
                    else:
                        print("[TELEOP] motion mode OFF. A/D now control selected arm joint.")

                elif key == "w":
                    if motion_mode:
                        fixed_base_lin_vel[:] = torch.tensor((motion_speed, 0.0, 0.0), device=env.unwrapped.device)
                        print(
                            "[TELEOP] motion mode command: forward "
                            f"(vx={motion_speed:+.2f}, vy=0.0, wz=0.0)"
                        )

                elif key == "s":
                    if motion_mode:
                        fixed_base_lin_vel[:] = torch.tensor((-motion_speed, 0.0, 0.0), device=env.unwrapped.device)
                        print(
                            "[TELEOP] motion mode command: backward "
                            f"(vx={-motion_speed:+.2f}, vy=0.0, wz=0.0)"
                        )

                elif key == "r":
                    arm_targets = arm_targets_init.clone()
                    print("[TELEOP] reset arm targets to initial values")

                elif key == "p":
                    print("[TELEOP] current arm targets:")
                    for i, (name, value) in enumerate(zip(arm_joint_names, arm_targets.tolist()), start=1):
                        marker = "<" if i - 1 == selected_joint_idx else " "
                        print(f" {marker} {i}: {name:>12s} = {value:+.4f}")

                elif key == "q":
                    print("[TELEOP] quitting teleoperation...")
                    env.close()
                    return

            # Keep all non-arm commands fixed and update only arm commands (action slice 3:10).
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            actions[:, :3] = fixed_base_lin_vel
            actions[:, 3:10] = arm_targets
            actions[:, -2:] = fixed_base_pose

            env.step(actions)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
