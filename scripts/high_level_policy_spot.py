"""Minimal hierarchical object-moving policy runner for physical Spot.

Use the high-level ``exported/policy.pt`` produced by scripts/rsl_rl/play.py
(TorchScript actor WITH its observation normalizer), not model_*.pt.

Hardware contract for SPOT.get_high_level_obs() (left unchanged): return a
dictionary with the fields in robot.OBS_TERMS, already ordered/scaled exactly
as ObservationsCfg.PolicyCfg. Also return ``object_pose_world`` = [x, y, yaw]
in the same fixed frame as GOAL_POSE_WORLD (e.g. Spot's vision frame). Angles
are radians. This wrapper supplies last_action; the object-local goal-error
construction is currently left commented out in robot.py.
The currently unfinished hardware method must satisfy this contract before use.

Optional --camera-pose-file reads FLIR latest-pose.json into env.camera_tag_pose.
This is a raw camera-relative TAG pose, not a replacement for the unfinished
world/object observations. Camera calibration and tag/object offsets are still
needed. Invalid/stale camera input raises an error and enters normal cleanup.

This runner assumes the object is ALREADY grasped and Spot is powered/standing.
It does not approach, grasp, or power on. --enable-joint-commands explicitly
enables actuation; without it, only observations and inference run. Live cleanup
uses the existing release/stow/self-right/power-off behavior.
"""

import argparse

import torch


# Fixed target object SE(2) pose in the hardware world/vision frame. Edit for
# your experiment; this is NOT a base-velocity command or an object-local offset.
GOAL_POSE_WORLD = (-0.2, 0.0, 0.0)



def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hostname", default="192.168.80.3")
    parser.add_argument("--high-level-policy", required=True, help="Exported high-level TorchScript policy.pt.")
    parser.add_argument("--steps", type=int, default=100, help="High-level steps (nominally 50 Hz).")
    parser.add_argument("--enable-joint-commands", action="store_true")
    parser.add_argument("--camera-pose-file", help="Path to the running FLIR process's latest-pose.json.")
    parser.add_argument("--camera-tag-id", type=int, default=7, help="Expected FLIR tag ID (default: 7).")
    parser.add_argument("--camera-tag-family", default="tag36h11")
    parser.add_argument("--camera-pose-max-age", type=float, default=0.5,
                        help="Maximum camera record age in seconds (host processing timestamp).")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main(args):
    if args.steps <= 0:
        raise ValueError("--steps must be positive.")
    camera_reader = None
    if args.camera_pose_file:
        from camera_tag_pose import CameraTagPoseReader
        camera_reader = CameraTagPoseReader(
            args.camera_pose_file, tag_id=args.camera_tag_id,
            max_age_s=args.camera_pose_max_age, tag_family=args.camera_tag_family,
        )
        camera_reader.read()  # Reject unusable camera input before touching Spot.
    # Load before touching hardware; the export includes actor normalization.
    policy = torch.jit.load(args.high_level_policy, map_location="cpu").eval()
    from robot import SPOT, SpotReLICEnvPLAY, HighLevelSpotEnv

    robot = SPOT(args)
    lease = low_level = None
    started = False
    try:
        if args.enable_joint_commands:
            lease = robot.lease_alive()
            if not robot.robot.is_powered_on() or robot.robot.is_estopped():
                raise RuntimeError("Live execution requires an already powered, non-E-stopped Spot.")
        started = True
        # Loads the same pretrained_relic/policy.pt as zero_agent_ReLIC_spot.py.
        low_level = SpotReLICEnvPLAY(
            robot, enable_latency_reporting=False,
            enable_observation_logging=False, enable_policy_trace_logging=False,
        )
        env = HighLevelSpotEnv(robot, low_level, GOAL_POSE_WORLD, camera_pose_reader=camera_reader)
        with torch.inference_mode():
            obs = env.get_observations()
            for _ in range(args.steps):
                obs = env.step(policy(obs))
    finally:
        try:
            if started:
                if args.enable_joint_commands:
                    if low_level is not None:
                        low_level.close()
                    else:
                        robot.shutdown_smooth()
                else:
                    # Observation-only mode must not send recovery/power commands.
                    robot.joint_level_control_stop()
        finally:
            if lease is not None:
                robot.lease_return()


if __name__ == "__main__":
    main(build_parser().parse_args())
