# Copyright (c) 2026
# SPDX-License-Identifier: BSD-3-Clause

"""Simulation-only WASD locomotion teleoperation for ReLIC Spot.

Controls (the Tk window must have keyboard focus):
    Tap a movement key to select a continuous command. Releasing it does not stop.
    Another movement key replaces the command; Space clears it.
    W / S: forward / backward
    A / D: left / right
    Q / E: turn left / right
    Space / Stop: clear the movement command
    R / Reset Spot: stop and reset Spot; keep the panel open
    Esc: quit
"""

import argparse
import tkinter as tk

from isaaclab.app import AppLauncher


# Share the instructions between terminal, --help, and the control panel.
CONTROL_HELP = (
    "Focus the control panel to use the keyboard.",
    "W/S: forward/backward    A/D: left/right    Q/E: turn left/right",
    "Tap once to move continuously at the fixed speed; releasing the key does not stop.",
    "Another movement key replaces the current command. Green = active command.",
    "SPACE / Stop: clear the movement command.",
    "R / Reset Spot: stop and reset the robot; the panel stays open.",
    "ESC / close window: quit.",
)

parser = argparse.ArgumentParser(
    description="Tap-to-move keyboard control of the simulated ReLIC Spot robot.",
    epilog="\n".join(CONTROL_HELP),
    formatter_class=argparse.RawDescriptionHelpFormatter,
)
parser.add_argument("--disable_fabric", action="store_true", help="Disable Isaac Fabric.")
parser.add_argument("--num_envs", type=int, default=1, help="Must be one for keyboard control.")
parser.add_argument("--task", type=str, default="Spot-ReLIC-WASD", help="Gym task to run.")
parser.add_argument(
    "--linear_speed",
    "--motion_speed",
    dest="linear_speed",
    type=float,
    default=0.20,
    help="Fixed translation command in m/s when W/A/S/D is selected (--motion_speed is an alias).",
)
parser.add_argument(
    "--turn_speed", type=float, default=0.40,
    help="Fixed yaw-rate command in rad/s when Q/E is selected.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
import SuperQ_ALORE.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


class WASDController:
    """Latch one fixed ReLIC command until another key, Stop, or Reset."""

    _MOTION_KEYS = frozenset({"w", "a", "s", "d", "q", "e"})

    def __init__(self, linear_speed: float, turn_speed: float) -> None:
        self._linear_speed = linear_speed
        self._turn_speed = turn_speed
        self._active_key: str | None = None
        # The last selected direction stays active across key releases and
        # focus changes. Each selection replaces the entire fixed command.
        self._command = (0.0, 0.0, 0.0)
        self._reset_requested = False
        self._quit_requested = False

        self.root = tk.Tk()
        self.root.title("Simulated Spot ReLIC WASD")
        self.root.resizable(False, False)
        tk.Label(
            self.root,
            text="\n".join(CONTROL_HELP),
            justify="left",
            padx=16,
            pady=12,
        ).pack()
        key_frame = tk.Frame(self.root, padx=12, pady=4)
        key_frame.pack()
        self._key_labels: dict[str, tk.Label] = {}
        for key, caption, row, column in (
            ("q", "Turn left", 0, 0),
            ("w", "Forward", 0, 1),
            ("e", "Turn right", 0, 2),
            ("a", "Left", 1, 0),
            ("s", "Backward", 1, 1),
            ("d", "Right", 1, 2),
        ):
            label = tk.Label(
                key_frame,
                text=f"{key.upper()}\n{caption}",
                width=12,
                height=3,
                font=("TkDefaultFont", 12, "bold"),
                borderwidth=3,
            )
            label.grid(row=row, column=column, padx=5, pady=5)
            self._key_labels[key] = label
        self._command_label = tk.Label(self.root, font=("Courier", 11), padx=16, pady=4)
        self._command_label.pack()
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=(4, 12))
        tk.Button(button_frame, text="Stop (Space)", command=self._stop_motion).pack(side="left", padx=6)
        tk.Button(button_frame, text="Reset Spot (R)", command=self._request_reset).pack(side="left", padx=6)

        # Only presses select commands. KeyRelease (including X11 repeat
        # releases) and FocusOut must not clear a latched command.
        self.root.bind("<KeyPress>", self._on_key_press)
        self.root.protocol("WM_DELETE_WINDOW", self._request_quit)
        self._refresh_label()
        self.root.after(100, self.root.focus_force)

    def _request_reset(self) -> None:
        self._stop_motion()
        self._reset_requested = True
        self.root.focus_set()

    def _request_quit(self) -> None:
        self._stop_motion()
        self._quit_requested = True

    def _stop_motion(self) -> None:
        self._active_key = None
        self._update_command()
        self._refresh_label()

    def _on_key_press(self, event: tk.Event) -> str | None:
        key = event.keysym.lower()
        if key == "escape":
            self._request_quit()
        elif key == "r":
            self._request_reset()
        elif key == "space":
            self._stop_motion()
        elif key in self._MOTION_KEYS:
            self._active_key = key
        else:
            return None
        self._update_command()
        self._refresh_label()
        return "break"

    def _update_command(self) -> None:
        """Build the fixed command for the selected direction, or zero."""
        key = self._active_key
        vx = self._linear_speed * (int(key == "w") - int(key == "s"))
        vy = self._linear_speed * (int(key == "a") - int(key == "d"))
        yaw_rate = self._turn_speed * (int(key == "q") - int(key == "e"))
        self._command = (float(vx), float(vy), float(yaw_rate))

    def _refresh_label(self) -> None:
        for key, label in self._key_labels.items():
            active = key == self._active_key
            label.config(
                background="#15803d" if active else "#e5e7eb",
                foreground="#ffffff" if active else "#1f2937",
                relief=tk.SUNKEN if active else tk.RAISED,
            )
        vx, vy, yaw_rate = self.command
        selection = f"ACTIVE: {self._active_key.upper()}" if self._active_key else "STOPPED"
        self._command_label.config(
            text=f"{selection} | vx={vx:+.2f} m/s  vy={vy:+.2f} m/s  yaw={yaw_rate:+.2f} rad/s"
        )

    @property
    def command(self) -> tuple[float, float, float]:
        """Return the latched command, or zero after Stop or Reset."""
        return self._command

    def poll_events(self) -> None:
        self.root.update_idletasks()
        self.root.update()

    def consume_reset(self) -> bool:
        reset_requested = self._reset_requested
        self._reset_requested = False
        return reset_requested

    @property
    def should_quit(self) -> bool:
        return self._quit_requested

    def close(self) -> None:
        try:
            if self.root.winfo_exists():
                self.root.destroy()
        except tk.TclError:
            pass


def main() -> None:
    if args_cli.num_envs != 1:
        raise ValueError("WASD teleoperation controls one Spot instance; use --num_envs 1.")
    if args_cli.linear_speed < 0 or args_cli.turn_speed < 0:
        raise ValueError("--linear_speed and --turn_speed must be non-negative.")

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=1,
        use_fabric=not args_cli.disable_fabric,
    )
    env = gym.make(args_cli.task, cfg=env_cfg)
    # ManagerBasedRLEnv is vectorized: Gym includes the environment batch in
    # its action-space shape. The three values are ReLIC's base command; the
    # action term internally turns them into final leg joint-position targets.
    expected_action_shape = (env.unwrapped.num_envs, 3)
    if env.action_space.shape != expected_action_shape:
        raise RuntimeError(
            "Expected one 3-D ReLIC command per environment "
            f"(shape {expected_action_shape}), got {env.action_space}."
        )

    print("[WASD] Starting simulation-only ReLIC Spot control (tap-to-move).")
    print(
        f"[WASD] Fixed command speeds: translation {args_cli.linear_speed:.2f} m/s; "
        f"rotation {args_cli.turn_speed:.2f} rad/s."
    )
    for line in CONTROL_HELP:
        print(f"[WASD] {line}")
    with torch.inference_mode():
        env.reset()
    controller = WASDController(args_cli.linear_speed, args_cli.turn_speed)

    try:
        while simulation_app.is_running() and not controller.should_quit:
            controller.poll_events()
            if controller.should_quit:
                break

            with torch.inference_mode():
                if controller.consume_reset():
                    # Keep reset in the same inference context as stepping:
                    # IsaacLab may update tensors cached during prior steps.
                    env.reset()
                    print("[WASD] Spot reset; motion stopped. Panel remains open.")
                    continue
                actions = torch.tensor(
                    [controller.command], device=env.unwrapped.device, dtype=torch.float32
                )
                env.step(actions)
    finally:
        controller.close()
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
