# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Slider-based teleoperation for Spot arm joints and base motion.

A tkinter window with:
- Sliders for each of the 7 arm joints (rad)
- Keyboard keys for base velocity (WASD / Q)

Base velocity keys (window must have focus):
    W : +vx (forward)     S : -vx (backward)
    A : +wz (turn left)   D : -wz (turn right)
    Space : zero velocity

Buttons: Reset (arm to init), Print, Quit.
"""

import argparse
import tkinter as tk


from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Spot arm joint teleoperation (sliders).")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--init_from_zero", action="store_true", default=False, help="Initialise arm targets to zero instead of grasp pose.")
parser.add_argument(
    "--motion_speed",
    type=float,
    default=0.5,
    help="Base speed magnitude (m/s or rad/s) applied while a velocity key is held.",
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
from SuperQ_ALORE.tasks.manager_based.superq_alore.mdp.scene import OBJECT_TELEOPERATION_INFO
from SuperQ_ALORE.tasks.manager_based.superq_alore.mdp.object_management import ARM_JOINT_NAMES_IN_ORDER
def _build_arm_joint_names() -> list[str]:
    """Return ordered Spot arm joint names used by the action head (7 joints)."""
    names = [name for name in ARM_JOINT_NAMES if name.startswith("arm")]
    if len(names) >= 7:
        return names[:7]

    # Fallback to grasp-pose dictionary order if constants do not contain all joints.
    fallback = [name for name in GRASP_POSE_1_JOINT_POS.keys() if name.startswith("arm")]
    return fallback[:7]


class _TkSliderGUI:
    """Tkinter window with sliders for arm joints and keyboard base-velocity control."""

    _ARM_RANGE = (-3.14, 3.14)
    _SLIDER_LEN = 380
    _LABEL_WIDTH = 22

    def __init__(
        self,
        arm_joint_names: list[str],
        initial_arm_targets: list[float],
        motion_speed: float,
    ) -> None:
        self._quit = False
        self._reset_requested = False
        self._print_requested = False
        self._arm_joint_names = arm_joint_names
        self._initial_arm_targets = list(initial_arm_targets)
        self._motion_speed = motion_speed
        # vx, vy, wz — updated by key press/release
        self._base_vel = [0.0, 0.0, 0.0]
        self._held_keys: set[str] = set()

        try:
            self.root = tk.Tk()
        except tk.TclError as exc:
            raise RuntimeError(
                "Failed to initialise tkinter. Ensure an X display is available."
            ) from exc

        self.root.title("Spot Arm Teleoperation — Sliders")
        self.root.resizable(True, True)
        self.root.protocol("WM_DELETE_WINDOW", self._on_quit)
        self.root.bind("<KeyPress>", self._on_key_press)
        self.root.bind("<KeyRelease>", self._on_key_release)

        # ── Arm joints ──────────────────────────────────────────────────────
        arm_frame = tk.LabelFrame(self.root, text="Arm Joints (rad)", padx=8, pady=4)
        arm_frame.pack(fill="both", expand=True, padx=10, pady=(8, 4))

        self._arm_vars: list[tk.DoubleVar] = []
        self._arm_value_labels: list[tk.Label] = []
        for i, name in enumerate(arm_joint_names):
            row = tk.Frame(arm_frame)
            row.pack(fill="x", pady=1)
            tk.Label(row, text=f"{i + 1}: {name}", width=self._LABEL_WIDTH, anchor="w").pack(side="left")
            var = tk.DoubleVar(value=initial_arm_targets[i])
            val_lbl = tk.Label(row, text=f"{var.get():+.4f}", width=9, anchor="e")
            val_lbl.pack(side="right")
            scale = tk.Scale(
                row, variable=var,
                from_=self._ARM_RANGE[0], to=self._ARM_RANGE[1],
                resolution=0.001, orient="horizontal", length=self._SLIDER_LEN,
                showvalue=False,
            )
            scale.pack(side="left", fill="x", expand=True)
            self._arm_vars.append(var)
            self._arm_value_labels.append(val_lbl)

        # ── Base velocity key legend ─────────────────────────────────────────
        key_frame = tk.LabelFrame(self.root, text="Base Velocity (keyboard)", padx=8, pady=4)
        key_frame.pack(fill="x", padx=10, pady=4)
        legend = (
            "W/S : +/-vx (fwd/back)    "
            "A/D : +/-wz (turn)    "
            "Space : stop"
        )
        tk.Label(key_frame, text=legend, anchor="w", justify="left").pack(fill="x")
        self._vel_label = tk.Label(key_frame, text="vx=+0.00  vy=+0.00  wz=+0.00", anchor="w", font=("Courier", 10))
        self._vel_label.pack(fill="x")

        # ── Buttons ──────────────────────────────────────────────────────────
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(fill="x", padx=10, pady=(4, 8))
        tk.Button(btn_frame, text="Reset Arm", width=12, command=self._on_reset).pack(side="left", padx=4)
        tk.Button(btn_frame, text="Print", width=8, command=self._on_print).pack(side="left", padx=4)
        tk.Button(btn_frame, text="Quit", width=8, command=self._on_quit).pack(side="right", padx=4)

        self.root.update_idletasks()

    # ── Button callbacks ─────────────────────────────────────────────────────

    def _on_reset(self) -> None:
        self._reset_requested = True

    def _on_print(self) -> None:
        self._print_requested = True

    def _on_quit(self) -> None:
        self._quit = True

    # ── Key callbacks ────────────────────────────────────────────────────────

    def _on_key_press(self, event: tk.Event) -> None:
        self._held_keys.add(event.keysym.lower())
        self._update_vel_from_keys()

    def _on_key_release(self, event: tk.Event) -> None:
        self._held_keys.discard(event.keysym.lower())
        self._update_vel_from_keys()

    def _update_vel_from_keys(self) -> None:
        keys = self._held_keys
        spd = self._motion_speed
        vx = (spd if "w" in keys else 0.0) + (-spd if "s" in keys else 0.0)
        vy = 0.0
        wz = (spd if "a" in keys else 0.0) + (-spd if "d" in keys else 0.0)
        if "space" in keys:
            vx, vy, wz = 0.0, 0.0, 0.0
        self._base_vel = [vx, vy, wz]
        self._vel_label.config(text=f"vx={vx:+.2f}  vy={vy:+.2f}  wz={wz:+.2f}")

    # ── Public interface ─────────────────────────────────────────────────────

    def poll_events(self) -> None:
        self.root.update_idletasks()
        self.root.update()

    def get_arm_targets(self) -> list[float]:
        return [v.get() for v in self._arm_vars]

    def update_arm_measured_values(self, measured_values: list[float]) -> None:
        for label, value in zip(self._arm_value_labels, measured_values):
            label.config(text=f"{value:+.4f}")
    
    def get_arm_displacements(self) -> list[float]:
        return [v.get() - init for v, init in zip(self._arm_vars, self._initial_arm_targets)]

    def get_base_vel(self) -> list[float]:
        return list(self._base_vel)

    def reset_arm_sliders(self) -> None:
        for var, val in zip(self._arm_vars, self._initial_arm_targets):
            var.set(val)

    def should_quit(self) -> bool:
        return self._quit

    def consume_reset(self) -> bool:
        val = self._reset_requested
        self._reset_requested = False
        return val

    def consume_print(self) -> bool:
        val = self._print_requested
        self._print_requested = False
        return val

    def close(self) -> None:
        try:
            if self.root.winfo_exists():
                self.root.destroy()
        except tk.TclError:
            pass


def main():
    """Run slider (arm) + keyboard (base velocity) teleoperation."""
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )

    env = gym.make(args_cli.task, cfg=env_cfg)

    print(f"[INFO] Gym observation space: {env.observation_space}")
    print(f"[INFO] Gym action space: {env.action_space}")

    env.reset()

    arm_joint_names = _build_arm_joint_names()
    if len(arm_joint_names) != 7:
        raise RuntimeError(
            f"Expected 7 Spot arm joints, but got {len(arm_joint_names)} from constants: {arm_joint_names}"
        )

    robot = env.unwrapped.scene["robot"]
    arm_joint_ids, _ = robot.find_joints(arm_joint_names)

    initial_arm_targets = [OBJECT_TELEOPERATION_INFO[1][key] for key in ARM_JOINT_NAMES_IN_ORDER]

    gui = _TkSliderGUI(
        arm_joint_names=arm_joint_names,
        initial_arm_targets=initial_arm_targets,
        motion_speed=args_cli.motion_speed,
    )

    print("[TELEOP] Slider GUI opened. Adjust sliders to control the robot.")
    print(f"[TELEOP] Base velocity: W/S=fwd/back  A/D=turn  Space=stop  (speed={args_cli.motion_speed} m/s)")

    while simulation_app.is_running():
        with torch.inference_mode():
            gui.poll_events()

            if gui.should_quit():
                print("[TELEOP] quitting teleoperation...")
                gui.close()
                env.close()
                return

            if gui.consume_reset():
                gui.reset_arm_sliders()
                print("[TELEOP] arm targets reset to initial values")

            if gui.consume_print():
                arm_vals = gui.get_arm_targets()
                print("[TELEOP] current arm targets:")
                for i, (name, value) in enumerate(zip(arm_joint_names, arm_vals), start=1):
                    print(f"  {i}: {name:>12s} = {value:+.4f} rad")
                vel_vals = gui.get_base_vel()
                print(f"[TELEOP] base vel: vx={vel_vals[0]:+.4f}  vy={vel_vals[1]:+.4f}  wz={vel_vals[2]:+.4f}")

            arm_vals = gui.get_arm_displacements()
            vel_vals = gui.get_base_vel()

            measured_arm = robot.data.joint_pos[0, arm_joint_ids].detach().cpu().tolist()
            gui.update_arm_measured_values(measured_arm)

            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            actions[:, :3] = torch.tensor(vel_vals, device=env.unwrapped.device, dtype=torch.float32)
            actions[:, 3:10] = torch.tensor(arm_vals, device=env.unwrapped.device, dtype=torch.float32)

            env.step(actions)

    gui.close()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()