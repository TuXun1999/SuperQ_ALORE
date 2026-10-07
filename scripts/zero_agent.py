# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run an environment with zero action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Zero agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--video",
    action="store_true",
    default=False,
    help="Export one rendered video of the zero-action rollout.",
)
parser.add_argument("--video_length", type=int, default=200, help="Number of rollout frames to export.")
parser.add_argument(
    "--video_fps",
    type=int,
    default=None,
    help="Output video FPS. Defaults to the environment control frequency.",
)
parser.add_argument(
    "--video_folder",
    type=str,
    default=None,
    help="Output folder for videos. Defaults to './logs/videos'.",
)
parser.add_argument(
    "--video_name_prefix",
    type=str,
    default="",
    help="Optional prefix for the exported video filename.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# Cameras must be enabled before launching Isaac Sim for rgb-array recording.
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import isaaclab_tasks  # noqa: F401
import SuperQ_ALORE.tasks  # noqa: F401
import torch
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.utils.dict import print_dict


def _estimate_step_dt_from_cfg(env_cfg) -> float:
    """Estimate the environment control period for a sensible default video FPS."""
    sim_dt = float(getattr(getattr(env_cfg, "sim", None), "dt", 1.0 / 60.0))
    decimation = int(getattr(env_cfg, "decimation", 1) or 1)
    return sim_dt * decimation


def _filename_component(value: str) -> str:
    """Convert a user-supplied filename component to a portable form."""
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in value)


def _build_video_name() -> str:
    """Create a descriptive filename for the one zero-action rollout."""
    parts = ["zero-agent"]
    if args_cli.video_name_prefix:
        parts.insert(0, _filename_component(args_cli.video_name_prefix))
    return "-".join(parts)


def main():
    """Zero actions agent with Isaac Lab environment."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    if args_cli.video and args_cli.video_length <= 0:
        raise ValueError(f"--video_length must be positive when --video is set, got {args_cli.video_length}.")

    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    video_recorder = None
    if args_cli.video:
        video_folder = os.path.abspath(args_cli.video_folder or "./logs/videos")
        os.makedirs(video_folder, exist_ok=True)
        render_fps = (
            int(args_cli.video_fps)
            if args_cli.video_fps is not None and int(args_cli.video_fps) > 0
            else max(1, int(round(1.0 / max(1.0e-4, _estimate_step_dt_from_cfg(env_cfg)))))
        )
        try:
            if hasattr(env, "metadata") and isinstance(env.metadata, dict):
                env.metadata["render_fps"] = render_fps
            if hasattr(env, "unwrapped") and hasattr(env.unwrapped, "metadata") and isinstance(env.unwrapped.metadata, dict):
                env.unwrapped.metadata["render_fps"] = render_fps
        except Exception:
            pass

        video_kwargs = {
            "video_folder": video_folder,
            # Recording starts explicitly after the initial environment reset.
            "step_trigger": lambda _step: False,
            "video_length": args_cli.video_length,
            "name_prefix": "",
            "fps": render_fps,
            "disable_logger": True,
        }
        print(f"[INFO] Video FPS set to: {render_fps}")
        print(f"[INFO] Video output folder: {video_folder}")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
        video_recorder = env

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment
    env.reset()

    if video_recorder is not None:
        video_name = _build_video_name()
        video_recorder.start_recording(video_name)
        print(f"[INFO] Recording zero-action rollout: {video_name}.mp4")

    steps = env.unwrapped.max_episode_length

    timestep = 0
    recorded_steps = 0
    
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # compute zero actions
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            """
            actions: base velocity (3) + arm joint (7) + base pose (2: pitch, height)
            (Forced to match 12D action space of the pretrained locomotion policy)
            """

            # set the active arm joints to the reference joint positions for the active pose, 
            # so that the arm will hold the desired pose with zero actions
            # also notice that we now use relative action, so we send zero delta.
            actions[:, 3:10] = torch.zeros_like(actions[:, 3:10], device=actions.device) # command zero delta for the arm joints to hold the arm at the desired pose defined by the active arm joint reference

            # Only command the base to be at a suitable height & pitch 
            # (roll action not desired)
            actions[:, :3] = torch.tensor([-0.2, 0.0, 0.0], device=env.unwrapped.device) # command a base velocity to move forward after chair reset, to avoid the disturbance from chair reset and keep the grasping pose stable

            obs, _, _, _, _ = env.step(actions)

            timestep += 1
            timestep = timestep % steps
            recorded_steps += 1

            if (
                video_recorder is not None
                and video_recorder.recording
                and recorded_steps >= args_cli.video_length
            ):
                video_recorder.stop_recording()
                print(f"[INFO] Video export completed after {recorded_steps} frames.")

    # close the simulator
    if video_recorder is not None and video_recorder.recording:
        video_recorder.stop_recording()
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
