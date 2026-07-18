# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Single-window dual-chair GRASP-RANKING with absolute pose sliders.

This script keeps two chairs in the scene:
- one static normal chair
- one green chair controlled by a single Tk window

The green chair pose is absolute and matches the slider values exactly:
- X (m)
- Y (m)
- Yaw (rad)
"""

import argparse
import importlib.util
import math
import os
import pathlib
import time
import tkinter as tk
import numpy as np

from isaaclab.app import AppLauncher


def _load_local_cli_args_module():
    """Load local scripts/rsl_rl/cli_args.py without import-name conflicts."""
    cli_args_path = pathlib.Path(__file__).resolve().parent / "rsl_rl" / "cli_args.py"
    spec = importlib.util.spec_from_file_location("superq_local_cli_args", str(cli_args_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load local cli_args module from: {cli_args_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cli_args = _load_local_cli_args_module()

# add argparse arguments
parser = argparse.ArgumentParser(description="Dual-chair GRASP-RANKING (absolute pose sliders).")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate (forced to 1).")
parser.add_argument("--task", type=str, default="Joint-GRASP-RANKING", help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_grasp_ranking_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--chair_object_id", type=str, default="chair_lab", help="Catalog chair object id used for both visual chairs.")
parser.add_argument("--green_offset_y", type=float, default=1.0, help="Initial y-offset (m) of the green chair from the normal chair.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--x_range", type=float, nargs=2, default=(-2.0, 2.0), metavar=("X_MIN", "X_MAX"), help="Slider range for X in meters.")
parser.add_argument("--y_range", type=float, nargs=2, default=(-2.0, 2.0), metavar=("Y_MIN", "Y_MAX"), help="Slider range for Y in meters.")
parser.add_argument(
    "--yaw_range",
    type=float,
    nargs=2,
    default=(-3.1415926535, 3.1415926535),
    metavar=("YAW_MIN", "YAW_MAX"),
    help="Slider range for yaw in radians.",
)
cli_args.add_rsl_rl_args(parser)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import isaaclab.sim as sim_utils
import isaaclab_tasks  # noqa: F401
import SuperQ_ALORE.tasks  # noqa: F401
import torch
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from rsl_rl.runners import DistillationRunner, OnPolicyRunner
import isaaclab.utils.math as math_utils
from SuperQ_ALORE.assets.object_catalog import OBJECT_CATALOG, OBJECT_IDS
from SuperQ_ALORE.rsl_rl.on_policy_runner_grasp_ranking import OnPolicyRunnerGraspRanking
from SuperQ_ALORE.rsl_rl.on_policy_runner_superqalore import OnPolicyRunnerSuperQALORE
from SuperQ_ALORE.tasks.manager_based.superq_alore.mdp.observations import \
    quat_inverse_safe, quat_mul, _euler_from_quat, quat_apply
from SuperQ_ALORE.tasks.manager_based.superq_alore.mdp.event import reset_object_robot_pose_grasp_ranking
from SuperQ_ALORE.tasks.manager_based.superq_alore.mdp.object_management import ARM_JOINT_NAMES_IN_ORDER
def _yaw_to_quat_wxyz(yaw: float, device: torch.device) -> torch.Tensor:
    """Convert yaw angle to wxyz quaternion tensor of shape (4,)."""
    half = 0.5 * yaw
    return torch.tensor([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=torch.float32, device=device)


def _quat_wxyz_to_yaw(quat_wxyz: torch.Tensor) -> float:
    """Extract yaw (around z) from a wxyz quaternion."""
    w, x, y, z = [float(v) for v in quat_wxyz]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)

def _goal_pose_local_frame(target_object, obj_goal_pos_w: torch.Tensor, obj_goal_quat_w: torch.Tensor) -> torch.Tensor:
    """Compute the goal object pose in local frame of the target object."""
    obj_pos_w = target_object.data.root_pos_w
    obj_quat_w = target_object.data.root_quat_w
    vec_w = obj_goal_pos_w - obj_pos_w
    objGoalPosLocal = math_utils.quat_apply_inverse(obj_quat_w, vec_w)[:, :2]
    objQuatInv = quat_inverse_safe(obj_quat_w)
    relQuat = quat_mul(objQuatInv, obj_goal_quat_w)
    objGoalYawLocal = _euler_from_quat(relQuat)[2] # Only the yaw difference
    return torch.cat([objGoalPosLocal, objGoalYawLocal.view(-1, 1)], dim=1)

def _obs_append_goal_pose(obs: torch.Tensor, target_object, obj_goal_pos_w: torch.Tensor, obj_goal_quat_w: torch.Tensor) -> torch.Tensor:
    """Append the goal object pose in local frame to the observation tensor."""
    objGoalPoseLocal = _goal_pose_local_frame(target_object, obj_goal_pos_w, obj_goal_quat_w)
    obs["policy"][:, -3:] = objGoalPoseLocal.view(1, 3)
    return obs

def _obj_init_pose_robot_frame(obj_init_pose_w: torch.Tensor, device = "cuda:0") -> torch.Tensor:
    # Hard-coded initial robot pose in world frame
    robot_init_pos = (-1.0, 0.0, 0.515)
    robot_init_quat = (1.0, 0.0, 0.0, 0.0)
    num_envs = obj_init_pose_w.shape[0]
    robot_base_pos_init = torch.tensor(robot_init_pos, device=device).unsqueeze(0).repeat(num_envs, 1)
    robot_quat_inv = quat_inverse_safe(torch.tensor(robot_init_quat, device=device).unsqueeze(0).repeat(num_envs, 1))

    # Initialized object initial pose in world frame
    obj_init_pose = obj_init_pose_w.clone().to(device) # shape (num_envs, 7), with position (3) + orientation (4)
    obj_pos_w_init = obj_init_pose[:, :3]
    obj_quat_w_init = obj_init_pose[:, 3:]

    obj_pos_relative = obj_pos_w_init - robot_base_pos_init.to(device)
    obj_pos_in_robot_frame = quat_apply(robot_quat_inv.to(device), obj_pos_relative)
    obj_quat_in_robot_frame = quat_mul(robot_quat_inv.to(device), obj_quat_w_init)
    # return torch.cat([obj_pos_in_robot_frame, obj_quat_in_robot_frame], dim=-1).to(env.device)  # (num_envs, 7)
    obj_pos_se2_in_robot_frame = obj_pos_in_robot_frame[:, :2]  # (num_envs, 2)
    obj_angle_yaw_in_robot_frame = _euler_from_quat(obj_quat_in_robot_frame)[2] 
    return torch.cat([obj_pos_se2_in_robot_frame, obj_angle_yaw_in_robot_frame.unsqueeze(-1)], dim=-1).to(device)  # (num_envs, 3)


def _grasp_pose_ranking(return_agent, obj_goal_pose_local, device = "cuda:0"):
    """Return the score of the grasp poses given the current obj_goal_pose_local"""
    obj_idx = 0 # Only consider one object
    return_est_list = []
    for pose_idx in range(len(OBJECT_CATALOG[obj_idx].poses)):
        pose = OBJECT_CATALOG[obj_idx].poses[pose_idx]
        obj_init_pose_w = torch.tensor(pose.position + pose.orientation, device=device)  # (7,)
        # Initial obj pose in robot frame
        obj_init_pose_robot_frame = _obj_init_pose_robot_frame(obj_init_pose_w.unsqueeze(0), device = device)  # (1, 3)
        
        # Initial arm joint positions
        arm_joint_pos_init_list = [pose.joint_positions[joint_name] for joint_name in ARM_JOINT_NAMES_IN_ORDER]
        arm_joint_pos_init = torch.tensor(arm_joint_pos_init_list, device=device)[:-1]  # (6,), no gripper
        
        grasp_ranking_input = torch.cat([obj_init_pose_robot_frame, arm_joint_pos_init.unsqueeze(0), obj_goal_pose_local], dim=1) # (1, 13)
        return_est = return_agent(grasp_ranking_input)
        return_est_list.append(return_est)
    return torch.tensor(return_est_list, device=device)
        
def _select_grasp_orientation(return_est_lst) -> tuple[int, int]:
    """Select the object index and pose index for grasping the chair."""
    best_pose_idx = torch.argmax(return_est_lst)
    return 0, best_pose_idx
class _TkChairGUI:
    """Tkinter window for absolute green-chair pose control."""

    _SLIDER_LEN = 380

    def __init__(
        self,
        x_init: float,
        y_init: float,
        yaw_init: float,
        x_range: tuple[float, float],
        y_range: tuple[float, float],
        yaw_range: tuple[float, float],
    ) -> None:
        self._quit = False
        self._reset_requested = False
        self._print_requested = False
        self._start_requested = False
        self._simulation_started = False

        self._x_init = float(x_init)
        self._y_init = float(y_init)
        self._yaw_init = float(yaw_init)

        try:
            self.root = tk.Tk()
        except tk.TclError as exc:
            raise RuntimeError("Failed to initialise tkinter. Ensure an X display is available.") from exc

        self.root.title("Dual Chair GRASP-RANKING (Absolute Pose)")
        self.root.resizable(True, True)
        self.root.protocol("WM_DELETE_WINDOW", self._on_quit)

        info_frame = tk.LabelFrame(self.root, text="Green Chair Pose (Absolute)", padx=8, pady=6)
        info_frame.pack(fill="both", expand=True, padx=10, pady=(8, 4))

        self._x_var = tk.DoubleVar(value=self._x_init)
        self._y_var = tk.DoubleVar(value=self._y_init)
        self._yaw_var = tk.DoubleVar(value=self._yaw_init)

        self._x_label = None
        self._y_label = None
        self._yaw_label = None
        self._x_entry = None
        self._y_entry = None
        self._yaw_entry = None

        self._add_slider_row(
            parent=info_frame,
            title="X (m)",
            var=self._x_var,
            bounds=x_range,
            label_attr="_x_label",
            entry_attr="_x_entry",
            callback=self._on_slider_changed,
        )
        self._add_slider_row(
            parent=info_frame,
            title="Y (m)",
            var=self._y_var,
            bounds=y_range,
            label_attr="_y_label",
            entry_attr="_y_entry",
            callback=self._on_slider_changed,
        )
        self._add_slider_row(
            parent=info_frame,
            title="Yaw (rad)",
            var=self._yaw_var,
            bounds=yaw_range,
            label_attr="_yaw_label",
            entry_attr="_yaw_entry",
            callback=self._on_slider_changed,
        )

        self._pose_label = tk.Label(self.root, text="", anchor="w", font=("Courier", 10))
        self._pose_label.pack(fill="x", padx=10, pady=4)

        btn_frame = tk.Frame(self.root)
        btn_frame.pack(fill="x", padx=10, pady=(4, 8))
        self._start_button = tk.Button(btn_frame, text="Start Simulation", width=16, command=self._on_start)
        self._start_button.pack(side="left", padx=4)
        tk.Button(btn_frame, text="Reset Green Chair", width=16, command=self._on_reset).pack(side="left", padx=4)
        tk.Button(btn_frame, text="Print", width=8, command=self._on_print).pack(side="left", padx=4)
        tk.Button(btn_frame, text="Quit", width=8, command=self._on_quit).pack(side="right", padx=4)

        self._on_slider_changed(None)
        self.root.update_idletasks()

    def _add_slider_row(
        self,
        parent: tk.Widget,
        title: str,
        var: tk.DoubleVar,
        bounds: tuple[float, float],
        label_attr: str,
        entry_attr: str,
        callback,
    ) -> None:
        row = tk.Frame(parent)
        row.pack(fill="x", pady=2)
        row.columnconfigure(1, weight=1)

        tk.Label(row, text=title, width=12, anchor="w").grid(row=0, column=0, sticky="w", padx=(0, 6))
        scale = tk.Scale(
            row,
            variable=var,
            from_=float(bounds[0]),
            to=float(bounds[1]),
            resolution=0.001,
            orient="horizontal",
            length=self._SLIDER_LEN,
            showvalue=False,
            command=callback,
        )
        scale.grid(row=0, column=1, sticky="ew")

        entry_widget = tk.Entry(row, width=10, justify="right")
        entry_widget.grid(row=0, column=2, sticky="ew", padx=(6, 4))

        value_label = tk.Label(row, text="", width=8, anchor="e")
        value_label.grid(row=0, column=3, sticky="e", padx=(4, 0))

        setattr(self, label_attr, value_label)
        setattr(self, entry_attr, entry_widget)

        entry_widget.bind("<Return>", lambda event, v=var, b=bounds, e=entry_widget: self._apply_entry_value(v, b, e))
        entry_widget.bind("<FocusOut>", lambda event, v=var, b=bounds, e=entry_widget: self._apply_entry_value(v, b, e))

    def _apply_entry_value(self, var: tk.DoubleVar, bounds: tuple[float, float], entry_widget: tk.Entry) -> None:
        try:
            value = float(entry_widget.get())
        except ValueError:
            value = var.get()
        value = max(float(bounds[0]), min(float(bounds[1]), value))
        var.set(value)
        self._on_slider_changed(None)

    def _on_slider_changed(self, _value) -> None:
        self._x_label.config(text=f"{self._x_var.get():+.3f}")
        self._y_label.config(text=f"{self._y_var.get():+.3f}")
        self._yaw_label.config(text=f"{self._yaw_var.get():+.3f}")
        self._x_entry.delete(0, tk.END)
        self._x_entry.insert(0, f"{self._x_var.get():+.3f}")
        self._y_entry.delete(0, tk.END)
        self._y_entry.insert(0, f"{self._y_var.get():+.3f}")
        self._yaw_entry.delete(0, tk.END)
        self._yaw_entry.insert(0, f"{self._yaw_var.get():+.3f}")
        self._pose_label.config(
            text=(
                f"green chair: x={self._x_var.get():+.3f} "
                f"y={self._y_var.get():+.3f} yaw={self._yaw_var.get():+.3f}"
            )
        )

    def _on_quit(self) -> None:
        self._quit = True

    def _on_start(self) -> None:
        if not self._simulation_started:
            self._start_requested = True

    def _on_reset(self) -> None:
        self._reset_requested = True

    def _on_print(self) -> None:
        self._print_requested = True

    def poll_events(self) -> None:
        self.root.update_idletasks()
        self.root.update()

    def get_absolute_pose(self) -> tuple[float, float, float]:
        return float(self._x_var.get()), float(self._y_var.get()), float(self._yaw_var.get())

    def reset_pose(self) -> None:
        self._x_var.set(self._x_init)
        self._y_var.set(self._y_init)
        self._yaw_var.set(self._yaw_init)
        self._on_slider_changed(None)

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

    def consume_start(self) -> bool:
        val = self._start_requested
        self._start_requested = False
        return val

    def mark_simulation_started(self) -> None:
        self._simulation_started = True
        self._start_button.config(state="disabled", text="Simulation Running")

    def close(self) -> None:
        try:
            if self.root.winfo_exists():
                self.root.destroy()
        except tk.TclError:
            pass


def _build_chair_markers(chair_asset_path: str) -> tuple[VisualizationMarkers, VisualizationMarkers]:
    """Create one movable green chair marker."""

    green_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Teleop/green_chair",
        markers={
            "chair": sim_utils.UsdFileCfg(
                usd_path=chair_asset_path,
                rigid_props=None,
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.10, 0.85, 0.20), opacity=1.0),
            )
        },
    )
    return VisualizationMarkers(green_cfg)


def _load_grasp_pose_mesh(mesh_name: str, color: tuple[float, float, float]) -> sim_utils.UsdFileCfg:
    """Load a custom mesh for grasp-pose markers.

    The actual mesh path/name is left as TODO and should be replaced with the required asset.
    """
    mesh_asset_path = f"source/SuperQ_ALORE/SuperQ_ALORE/assets/objects/{mesh_name}.usdc"
    # TODO: replace mesh_asset_path with the actual custom grasp mesh for `mesh_name`
    return sim_utils.UsdFileCfg(
        usd_path=mesh_asset_path,
        rigid_props=None,
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, opacity=1.0),
    )


def _build_grasp_pose_markers() -> VisualizationMarkers:
    """Create three custom mesh prototypes used to visualize the grasp poses."""
    grasp_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Teleop/grasp_poses",
        markers={
            "left_armrest": _load_grasp_pose_mesh("gripper", (0.0, 1.0, 0.0)),
            "right_armrest": _load_grasp_pose_mesh("gripper", (1.0, 1.0, 0.0)),
            "back": _load_grasp_pose_mesh("gripper", (1.0, 0.0, 0.0)),
        },
    )
    return VisualizationMarkers(grasp_cfg)

def _grasp_marker_local():
    """Return the location of the three markers in chair's local frame"""
    """_summary_

    Returns:
        _type_: a list, each component in the format (x, y, rw, rx, ry, rz)
    """
    # TODO: Read the grasp poses from pre-grasping.yaml
    # Compute local offsets of the three markers
    pose_local_offset = []
    
    # Back
    back_pose = [0.28, 0, 0.95, 0.696364, 0.122788, 0.122788, 0.696364]
    pose_local_offset.append(back_pose)
    
    # Right armrest
    right_armrest_pose = [0, 0.27, 0.73, 1, 0, 0, 0]
    pose_local_offset.append(right_armrest_pose)
    
    # Left armrest
    left_armrest_pose = [0, -0.27, 0.73, 1, 0, 0, 0]
    pose_local_offset.append(left_armrest_pose)
    
    
    
    return pose_local_offset


def _visualize_and_color_grasp_markers(grasp_markers, grasp_markers_local_offset, green_pos, green_quat, return_est_lst, device):
    """Place the three grasp markers in world using local offsets and color them continuously.

    Colors map return estimates to a red->yellow->green scale via linear interpolation.
    """
    # prepare translations and orientations
    translations = []
    orientations = []
    for off in grasp_markers_local_offset:
        # off: [x, y, z, qw, qx, qy, qz]
        local_pos = torch.tensor([off[0], off[1], off[2]], dtype=torch.float32, device=device).view(1, 3)
        local_quat = torch.tensor([off[3], off[4], off[5], off[6]], dtype=torch.float32, device=device).view(1, 4)
        world_pos = green_pos + quat_apply(green_quat, local_pos)
        world_quat = quat_mul(green_quat.view(1, 4), local_quat)
        translations.append(world_pos.view(3))
        orientations.append(world_quat.view(4))

    trans = torch.stack(translations, dim=0)
    orients = torch.stack(orientations, dim=0)
    # marker prototype indices fixed to 0..N-1
    marker_indices = list(range(len(translations)))
    grasp_markers.visualize(translations=trans, orientations=orients, marker_indices=marker_indices)

    # map return estimates to colors
    # convert to floats and handle NaNs
    ret_vals = []
    for v in return_est_lst:
        try:
            val = float(v)
        except Exception:
            val = float('nan')
        ret_vals.append(val)

    # normalization
    valid_vals = [v for v in ret_vals if not math.isnan(v)]
    if len(valid_vals) == 0:
        norm = [0.5] * len(ret_vals)
    else:
        vmin = min(valid_vals)
        vmax = max(valid_vals)
        if vmax - vmin < 1e-8:
            norm = [0.5 if not math.isnan(v) else 0.5 for v in ret_vals]
        else:
            norm = [0.0 if math.isnan(v) else (v - vmin) / (vmax - vmin) for v in ret_vals]

    # color mapping: t=0 -> red (1,0,0); t=0.5 -> yellow (1,1,0); t=1 -> green (0,1,0)
    colors = []
    for t in norm:
        # interpolate red->yellow (t in [0,0.5]) and yellow->green (t in (0.5,1])
        if t <= 0.5:
            # between red and yellow: R=1, G=2*t, B=0
            r = 1.0
            g = 2.0 * t
            b = 0.0
        else:
            # between yellow and green: R=2*(1-t), G=1, B=0
            r = 2.0 * (1.0 - t)
            g = 1.0
            b = 0.0
        colors.append((r, g, b))

    # Update prototype material diffuse color for each marker prototype.
    # For UsdFileCfg spawner, visual material is bound at: <prim_path>/<name>/material/Shader
    try:
        from pxr import Gf, Sdf
        import omni.kit.commands
        # marker names in the cfg order
        marker_names = list(grasp_markers.cfg.markers.keys())
        base = grasp_markers.prim_path
        num_markers = len(marker_names)
        if len(colors) < num_markers:
            colors = colors + [(0.5, 0.5, 0.5)] * (num_markers - len(colors))
        for name, col in zip(marker_names, colors[:num_markers]):
            material_shader_path = f"{base}/{name}/material/Shader"
            # property path: <prim>.inputs:diffuseColor
            prop_path = Sdf.Path(f"{material_shader_path}.inputs:diffuseColor")
            omni.kit.commands.execute(
                "ChangePropertyCommand",
                prop_path=prop_path,
                value=Gf.Vec3f(float(col[0]), float(col[1]), float(col[2])),
                prev=None,
                type_to_create_if_not_exist=Sdf.ValueTypeNames.Color3f,
            )
    except Exception:
        # silently ignore material update failures to avoid breaking runtime
        pass

def _set_target_object_pose(
    target_object,
    env_ids: torch.Tensor,
    x_val: float,
    y_val: float,
    z_val: float,
    yaw_val: float,
    device: torch.device,
) -> None:
    """Teleport target object to match the slider pose exactly."""
    state = target_object.data.default_root_state[env_ids].clone()
    state[:, 0] = float(x_val)
    state[:, 1] = float(y_val)
    state[:, 2] = float(z_val)
    state[:, 3:7] = _yaw_to_quat_wxyz(float(yaw_val), device=device).view(1, 4)
    state[:, 7:] = 0.0
    target_object.write_root_state_to_sim(state, env_ids=env_ids)


def _build_policy_agent(env, task_name: str):
    """Load policy runner/checkpoint following scripts/rsl_rl/play.py style."""
    agent_cfg = load_cfg_from_registry(task_name, args_cli.agent)
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)

    if args_cli.seed is not None:
        agent_cfg.seed = args_cli.seed

    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)

    if args_cli.use_pretrained_checkpoint:
        train_task_name = task_name.split(":")[-1].replace("-Play", "")
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            raise RuntimeError("No published pretrained checkpoint found for this task.")
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)
    vec_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    
    
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(vec_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "OnPolicyRunnerSuperQALORE":
        runner = OnPolicyRunnerSuperQALORE(vec_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(vec_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "OnPolicyRunnerGraspRanking":
        runner = OnPolicyRunnerGraspRanking(vec_env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")

    print(f"[INFO] Loading model checkpoint from: {resume_path}")

    runner.load(resume_path)
    
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    # The agent for return estimation
    return_agent = None
    if agent_cfg.class_name == "OnPolicyRunnerGraspRanking":
        return_agent = runner.return_agent_helper.return_agent
    action_clip = torch.tensor(
        [0.6, 0.0, 0.6, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.0, 0.0, 0.0],
        device=env.unwrapped.device,
    )

    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    return vec_env, policy, policy_nn, action_clip, return_agent


def main():
    """Run dual-chair GRASP-RANKING with absolute green-chair pose sliders."""
    if args_cli.num_envs != 1:
        print(f"[GRASP-RANKING] overriding --num_envs={args_cli.num_envs} -> 1")

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=1,
        use_fabric=not args_cli.disable_fabric,
    )

    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()

    if args_cli.chair_object_id not in OBJECT_IDS:
        raise ValueError(
            f"Invalid --chair_object_id='{args_cli.chair_object_id}'. Available: {list(OBJECT_IDS)}"
        )

    obj_idx = OBJECT_IDS.index(args_cli.chair_object_id)
    chair_asset_path = OBJECT_CATALOG[obj_idx].asset_path

    green_marker = _build_chair_markers(chair_asset_path)
    grasp_markers = _build_grasp_pose_markers()
    grasp_markers_local_offset = _grasp_marker_local()
    
    device = env.unwrapped.device
    target_object = env.unwrapped.scene["target_object_0"]
    base_root_state = target_object.data.default_root_state[0].clone().to(device)

    normal_pos = base_root_state[0:3].clone().view(1, 3)
    normal_quat = base_root_state[3:7].clone().view(1, 4)

    green_x_init = float(normal_pos[0, 0].item())
    green_y_init = float(normal_pos[0, 1].item() + float(args_cli.green_offset_y))
    green_yaw_init = _quat_wxyz_to_yaw(normal_quat[0])

    gui = _TkChairGUI(
        x_init=green_x_init,
        y_init=green_y_init,
        yaw_init=green_yaw_init,
        x_range=(float(args_cli.x_range[0]), float(args_cli.x_range[1])),
        y_range=(float(args_cli.y_range[0]), float(args_cli.y_range[1])),
        yaw_range=(float(args_cli.yaw_range[0]), float(args_cli.yaw_range[1])),
    )

    env_ids = torch.tensor([0], dtype=torch.long, device=device)

    vec_env, policy, policy_nn, action_clip, return_agent = _build_policy_agent(env, args_cli.task)
    obs = None
    simulation_started = False
    dt = float(env.unwrapped.step_dt)

    print("[GRASP-RANKING] Dual-chair GRASP-RANKING started.")
    print("[GRASP-RANKING] Green chair follows slider pose exactly (no velocity integration).")
    print("[GRASP-RANKING] Configure the green chair pose, then click 'Start Simulation'.")

    while simulation_app.is_running():
        with torch.inference_mode():
            gui.poll_events()

            if gui.should_quit():
                print("[GRASP-RANKING] quitting GRASP-RANKING...")
                gui.close()
                env.close()
                return

            if gui.consume_reset():
                gui.reset_pose()
                print("[GRASP-RANKING] reset green chair pose")

            x_val, y_val, yaw_val = gui.get_absolute_pose()
            green_pos = torch.tensor([[x_val, y_val, float(normal_pos[0, 2].item())]], dtype=torch.float32, device=device)
            green_quat = _yaw_to_quat_wxyz(yaw_val, device=device).view(1, 4)

            # The pose of the green chair is the target object pose in world frame
            obj_goal_pos_w = green_pos.clone()
            obj_goal_quat_w = green_quat.clone()
            # Find the goal of object pose in local frame
            obj_goal_pose_local = _goal_pose_local_frame(target_object, obj_goal_pos_w, obj_goal_quat_w)
            
            # Rank the grasp poses on the object
            return_est_lst = _grasp_pose_ranking(return_agent, obj_goal_pose_local)

            if gui.consume_print():
                print(f"[GRASP-RANKING] green chair pose: x={x_val:+.4f}, y={y_val:+.4f}, yaw={yaw_val:+.4f}")

            if (not simulation_started) and gui.consume_start():
                # Select the object/pose index
                object_idx, pose_idx = _select_grasp_orientation(return_est_lst)
                obs, _ = vec_env.unwrapped.reset(
                    env_ids = env_ids,
                )
                reset_object_robot_pose_grasp_ranking(
                    vec_env.unwrapped,
                    env_ids,
                    object_idx = object_idx,
                    pose_idx = pose_idx,
                )
                obs = vec_env.get_observations()
                gui.mark_simulation_started()
                simulation_started = True
                print("[GRASP-RANKING] simulation started with loaded policy agent")

            green_marker.visualize(green_pos, green_quat)
            if not simulation_started:
                # Visualize the three grasp markers and colour them continuously by return estimates
                _visualize_and_color_grasp_markers(
                    grasp_markers, grasp_markers_local_offset, green_pos, green_quat, return_est_lst, device
                )
            
            if simulation_started:
                start_time = time.time()
                # The raw observation doesn't contain the object goal pose in local frame
                obs = _obs_append_goal_pose(obs, target_object, obj_goal_pos_w, obj_goal_quat_w)
                actions = policy(obs)
                actions = torch.clamp(actions, -action_clip, action_clip)
            else:
                start_time = time.time()
                actions = torch.zeros(env.action_space.shape, dtype=torch.float32, device=device)

            obs, _, dones, _ = vec_env.step(actions)
            
            policy_nn.reset(dones)

            sleep_time = dt - (time.time() - start_time)
            if args_cli.real_time and sleep_time > 0:
                time.sleep(sleep_time)

    gui.close()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
