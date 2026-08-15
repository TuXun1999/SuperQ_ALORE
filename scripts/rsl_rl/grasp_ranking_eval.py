# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--video_fps",
    type=int,
    default=None,
    help="Output video FPS. Defaults to control frequency (1 / step_dt).",
)
parser.add_argument(
    "--video_folder",
    type=str,
    default=None,
    help="Optional output folder for videos. Defaults to '<checkpoint_dir>/videos/play'.",
)
parser.add_argument(
    "--video_name_prefix",
    type=str,
    default="play",
    help="Filename prefix for exported video files.",
)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_grasp_ranking_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--num_grasp_poses",
    type=int,
    default=3,
    help="Number of grasp poses/checkpoints to evaluate (pose indices start from 0).",
)
parser.add_argument(
    "--object_name",
    type=str,
    default="chair-lab",
    help="Object name used in checkpoint directory naming (e.g., '<object_name>-grasp-pose-1').",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import time

import gymnasium as gym
import isaaclab_tasks  # noqa: F401
import SuperQ_ALORE.tasks  # noqa: F401
import torch
import isaaclab.utils.math as math_utils
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from rsl_rl.runners import DistillationRunner, OnPolicyRunner
from SuperQ_ALORE.assets.object_catalog import OBJECT_CATALOG
from SuperQ_ALORE.rsl_rl.on_policy_runner_superqalore import OnPolicyRunnerSuperQALORE
from SuperQ_ALORE.rsl_rl.on_policy_runner_grasp_ranking import OnPolicyRunnerGraspRanking
from SuperQ_ALORE.tasks.manager_based.superq_alore.mdp.event import reset_object_robot_pose_grasp_ranking
from SuperQ_ALORE.tasks.manager_based.superq_alore.mdp.observations import quat_inverse_safe, quat_mul, _euler_from_quat, quat_apply
from SuperQ_ALORE.tasks.manager_based.superq_alore.mdp.object_management import ARM_JOINT_NAMES_IN_ORDER
from SuperQ_ALORE.tasks.manager_based.superq_alore.mdp import object_management as om
def _estimate_step_dt_from_cfg(env_cfg) -> float:
    """Estimate environment step time from sim dt and decimation."""
    sim_dt = float(getattr(getattr(env_cfg, "sim", None), "dt", 1.0 / 60.0))
    decimation = int(getattr(env_cfg, "decimation", 1) or 1)
    return sim_dt * decimation


def _build_env(env_cfg, log_dir: str):
    """Create and optionally wrap environment for video recording."""
    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    video_folder = None
    # wrap for video recording
    if args_cli.video:
        video_folder = args_cli.video_folder or os.path.join(log_dir, "videos", "play")
        video_folder = os.path.abspath(video_folder)
        os.makedirs(video_folder, exist_ok=True)

        est_step_dt = _estimate_step_dt_from_cfg(env_cfg)
        render_fps = (
            int(args_cli.video_fps)
            if args_cli.video_fps is not None and int(args_cli.video_fps) > 0
            else max(1, int(round(1.0 / max(1.0e-4, est_step_dt))))
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
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "name_prefix": args_cli.video_name_prefix,
            "disable_logger": True,
        }
        print(f"[INFO] Video FPS set to: {render_fps}")
        print(f"[INFO] Video output folder: {video_folder}")
        print(f"[INFO] Video name prefix: {args_cli.video_name_prefix}")
        print("[INFO] Recording videos during playback.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    return env, video_folder


def _build_policy_agent(vec_env, task_name: str, agent_name: str = None,agent_cfg = None, checkpoint: str = None):
    """Load policy runner/checkpoint following scripts/grasp_ranking_play.py style."""
    if agent_cfg is None:
        agent_cfg = load_cfg_from_registry(task_name, agent_name)
        agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)

    if args_cli.seed is not None:
        agent_cfg.seed = args_cli.seed

    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    
    
    if checkpoint is not None:
        resume_path = retrieve_file_path(checkpoint)
    elif args_cli.use_pretrained_checkpoint:
        train_task_name = task_name.split(":")[-1].replace("-Play", "")
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            raise RuntimeError("No published pretrained checkpoint found for this task.")
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

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

    policy = runner.get_inference_policy(device=vec_env.unwrapped.device)
    # The agent for return estimation
    return_agent = None
    if agent_cfg.class_name == "OnPolicyRunnerGraspRanking":
        return_agent = runner.return_agent_helper.return_agent
    action_clip = torch.tensor(
        [0.6, 0.0, 0.6, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.0, 0.0, 0.0],
        device=vec_env.unwrapped.device,
    )
    
    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    return policy, policy_nn, action_clip, return_agent, resume_path


def _read_goal_pose_metrics(vec_env) -> dict[str, float]:
    """Read and average goal-pose command metrics across all envs."""
    goal_term = vec_env.unwrapped.command_manager.get_term("goal_pose")
    metrics = goal_term.metrics
    keys = [
        "success_rate",
        "object_to_goal_dist",
        "object_to_goal_yaw_diff",
        "keypoint_angle_error_degree",
    ]
    summary: dict[str, float] = {}
    for key in keys:
        if key in metrics:
            summary[key] = float(torch.mean(metrics[key]).item())
    return summary


def _resolve_goal_pose_w(env, goal_term_name: str = "goal_pose") -> tuple[torch.Tensor, torch.Tensor]:
    """Resolve goal position/quaternion in world frame from command term."""
    term = env.command_manager.get_term(goal_term_name)
    goal_pos_w = term.goal_w if hasattr(term, "goal_w") else term.command[:, :3]
    if hasattr(term, "goal_quat_w"):
        goal_quat_w = term.goal_quat_w
    else:
        goal_quat_w = torch.zeros((goal_pos_w.shape[0], 4), device=goal_pos_w.device, dtype=goal_pos_w.dtype)
        goal_quat_w[:, 0] = 1.0
    return goal_pos_w, goal_quat_w


def _goal_pose_local_frame(env, obj_goal_pos_w: torch.Tensor, obj_goal_quat_w: torch.Tensor) -> torch.Tensor:
    """Compute goal object pose in active object local frame."""
    obj_pos_w = env.scene["target_object_0"].data.root_pos_w
    obj_quat_w = env.scene["target_object_0"].data.root_quat_w
    vec_w = obj_goal_pos_w - obj_pos_w
    obj_goal_pos_local = math_utils.quat_apply_inverse(obj_quat_w, vec_w)[:, :2]
    obj_quat_inv = quat_inverse_safe(obj_quat_w)
    rel_quat = quat_mul(obj_quat_inv, obj_goal_quat_w)
    obj_goal_yaw_local = _euler_from_quat(rel_quat)[2]
    return torch.cat([obj_goal_pos_local, obj_goal_yaw_local.view(-1, 1)], dim=1) # (num_envs, 3)


def _obj_init_pose_robot_frame(robot_init_pose_w: torch.Tensor, num_envs: int, device) -> torch.Tensor:
    """Convert robot initial world pose into object SE2 pose in robot frame and tile for env batch."""
    robot_init_pose = robot_init_pose_w.clone().to(device)
    robot_base_pos_init = robot_init_pose[:, :3]
    robot_quat_inv = quat_inverse_safe(robot_init_pose[:, 3:])

    obj_pos_w_init = torch.zeros((1, 3), dtype=robot_init_pose.dtype, device=device)
    obj_quat_w_init = torch.zeros((1, 4), dtype=robot_init_pose.dtype, device=device)
    obj_quat_w_init[:, 0] = 1.0

    obj_pos_relative = obj_pos_w_init - robot_base_pos_init
    obj_pos_in_robot_frame = quat_apply(robot_quat_inv, obj_pos_relative)
    obj_quat_in_robot_frame = quat_mul(robot_quat_inv, obj_quat_w_init)
    obj_pos_se2_in_robot_frame = obj_pos_in_robot_frame[:, :2]
    obj_angle_yaw_in_robot_frame = _euler_from_quat(obj_quat_in_robot_frame)[2]
    single = torch.cat([obj_pos_se2_in_robot_frame, obj_angle_yaw_in_robot_frame.unsqueeze(-1)], dim=-1)
    return single.repeat(num_envs, 1) # (num_envs, 3)

def _make_eval_plan(policy_test_type, vec_env, return_agent, env_ids, num_grasp_poses: int):
    """
    Make different evaluation plans based on the policy test type.
    
    grasp_ranking: w/ grasp ranking agent & fixed poses
    others: fixed poses
    """
    eval_plan: list[tuple[str, int | torch.Tensor]] = []
    if policy_test_type == "grasp_ranking":
        # Extra rollout: choose grasp pose via return-agent ranking against commanded goal pose.
        vec_env.unwrapped.reset(env_ids=env_ids)
        ranked_pose_idx, ranked_scores = _select_ranked_pose_idx(vec_env, return_agent, num_grasp_poses)
        
        eval_plan.append(("ranked", ranked_pose_idx))
        eval_plan.append(("random", torch.randint(0, num_grasp_poses, (vec_env.unwrapped.num_envs,), device=vec_env.unwrapped.device)))
    # Baseline fixed-pose rollouts.
    eval_plan.extend(("fixed", pose_idx) for pose_idx in range(num_grasp_poses))
    return eval_plan

def _select_ranked_pose_idx(vec_env, return_agent, num_grasp_poses: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Select pose_idx by ranking candidate grasp poses against current command goal."""
    env = vec_env.unwrapped
    device = env.device
    num_envs = env.num_envs

    obj_goal_pos_w, obj_goal_quat_w = _resolve_goal_pose_w(env, goal_term_name="goal_pose")
    obj_goal_pose_local = _goal_pose_local_frame(env, obj_goal_pos_w, obj_goal_quat_w)

    pose_scores: list[float] = []
    obj_idx = 0
    pose_count = min(num_grasp_poses, len(OBJECT_CATALOG[obj_idx].poses))
    with torch.inference_mode():
        # TODO: Incorrectly assumes single object and single goal; may need to generalize for multiple objects/goals
        # Fix it with multiple parallel-envs
        for pose_idx in range(pose_count):
            pose = OBJECT_CATALOG[obj_idx].poses[pose_idx]
            robot_init_pose_w = torch.tensor(pose.position + pose.orientation, device=device).view(1, 7)
            obj_init_pose_robot_frame = _obj_init_pose_robot_frame(robot_init_pose_w, num_envs, device)

            arm_joint_pos_init_list = [pose.joint_positions[joint_name] for joint_name in ARM_JOINT_NAMES_IN_ORDER]
            arm_joint_pos_init = torch.tensor(arm_joint_pos_init_list, device=device)[:-1].view(1, -1).repeat(num_envs, 1) # (num_envs, 6)

            grasp_ranking_input = torch.cat(
                [obj_init_pose_robot_frame, arm_joint_pos_init, obj_goal_pose_local],
                dim=1,
            )
            return_est = return_agent(grasp_ranking_input) # (num_envs, 1)
            pose_scores.append(return_est)
    pose_scores = torch.cat(pose_scores, dim=1) # (num_envs, num_grasp_poses)
    best_pose_idx = torch.argmax(pose_scores, dim=1)
    return best_pose_idx, pose_scores

@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")

    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)
    env, video_folder = _build_env(env_cfg, log_dir)
    vec_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    object_name = args_cli.object_name.strip()
    if not object_name:
        raise ValueError("object_name must be a non-empty string.")

    if args_cli.num_grasp_poses <= 0:
        raise ValueError(f"num_grasp_poses must be > 0, got {args_cli.num_grasp_poses}.")

    base_path = "./logs/rsl_rl/"
    policy_test_list = ["grasp_ranking"] + [f"ppo{pose_idx}" for pose_idx in range(args_cli.num_grasp_poses)]
    checkpoint_list = [os.path.join(base_path, "Grasp_Ranking", object_name, "model_7999.pt")]
    checkpoint_list.extend(
        os.path.join(base_path, "SuperQ_ALORE", f"{object_name}-grasp-pose-{pose_idx + 1}", "model_7999.pt")
        for pose_idx in range(args_cli.num_grasp_poses)
    )
    task_list = ["Grasp-Ranking-EVAL"] + ["Template-Superq-Alore-v0"] * args_cli.num_grasp_poses
    agent_list = ["rsl_rl_grasp_ranking_cfg_entry_point"] + ["rsl_rl_cfg_entry_point"] * args_cli.num_grasp_poses
    policy_list = []
    policy_nn_list = []
    return_agent = None
    for checkpoint, task_name, agent_name in zip(checkpoint_list, task_list, agent_list):
        policy, policy_nn, action_clip, return_agent_raw, _ = _build_policy_agent(
            vec_env, task_name, agent_name=agent_name, agent_cfg=None, checkpoint=checkpoint
        )
        policy_list.append(policy)
        policy_nn_list.append(policy_nn)
        if return_agent_raw is not None:
            return_agent = return_agent_raw

    if return_agent is None:
        raise RuntimeError(
            "Return agent is not available from loaded runner/checkpoint. "
            "Use an OnPolicyRunnerGraspRanking checkpoint."
        )


    dt = vec_env.unwrapped.step_dt
    rollout_steps = 750
    env_ids = torch.arange(vec_env.unwrapped.num_envs, device=vec_env.unwrapped.device, dtype=torch.long)
    object_idx = 0

    # Reset through the wrapper once so wrappers/video are initialized correctly.
    vec_env.reset()

    for policy_test_type, policy, policy_nn in zip(policy_test_list, policy_list, policy_nn_list):
        # Make up the evaluation plan for each policy
        eval_plan = _make_eval_plan(policy_test_type, vec_env, return_agent, env_ids, args_cli.num_grasp_poses)
    
        # Evaluate each policy according to the evaluation plan
        for mode, pose_idx in eval_plan:
            if not simulation_app.is_running():
                break

            # Reset env state for the current pose index and fetch fresh observations.
            with torch.inference_mode():
                vec_env.reset()
                reset_object_robot_pose_grasp_ranking(
                    vec_env.unwrapped,
                    env_ids,
                    object_idx=object_idx,
                    pose_idx=pose_idx,
                )
            obs = vec_env.get_observations()

            for step in range(rollout_steps):
                if not simulation_app.is_running():
                    break
                start_time = time.time()
                with torch.inference_mode():
                    actions = policy(obs)
                    actions = torch.clamp(actions, -action_clip, action_clip)
                    obs, _, dones, _ = vec_env.step(actions)
                    policy_nn.reset(dones)

                if args_cli.real_time:
                    sleep_time = dt - (time.time() - start_time)
                    if sleep_time > 0:
                        time.sleep(sleep_time)

                # Keep video export behavior bounded by video_length when recording.
                if args_cli.video and step + 1 >= args_cli.video_length:
                    break

            metric_summary = _read_goal_pose_metrics(vec_env)
            success_rate = metric_summary.get("success_rate", float("nan"))
            pos_dist = metric_summary.get("object_to_goal_dist", float("nan"))
            yaw_diff = metric_summary.get("object_to_goal_yaw_diff", float("nan"))
            kp_err = metric_summary.get("keypoint_angle_error_degree", float("nan"))
            
            if isinstance(pose_idx, torch.Tensor):
                pose_idx_print = "N/A"
            else:
                pose_idx_print = pose_idx
            print(
                "[EVAL] "
                f"policy_type={policy_test_type} "
                f"mode={mode} "
                f"pose_idx={pose_idx_print} "
                f"success_rate={success_rate:.4f} "
                f"object_to_goal_dist={pos_dist:.4f} "
                f"object_to_goal_yaw_diff={yaw_diff:.4f} "
                f"keypoint_angle_error_degree={kp_err:.4f}"
            )

    # close the simulator
    vec_env.close()
    if args_cli.video:
        print(f"[INFO] Video export completed. Check folder: {video_folder}")


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
