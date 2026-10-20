"""Camera JSON and environment integration checks with no camera/Spot access."""
import ast
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from camera_tag_pose import CameraPoseUnavailable, CameraTagPoseReader
from high_level_policy_spot import build_parser, main


def load_environment():
    path = Path(__file__).with_name("robot.py")
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if
             (isinstance(node, (ast.ClassDef, ast.FunctionDef))
              and node.name in {"HighLevelSpotEnv", "vector"})
             or (isinstance(node, ast.Assign) and any(
                 isinstance(target, ast.Name) and target.id == "OBS_TERMS" for target in node.targets))]
    ns = {"torch": torch, "math": math, "time": time}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), ns)
    return ns["HighLevelSpotEnv"], ns["OBS_TERMS"]


class CameraPoseTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "latest-pose.json"
        self.clock = patch("camera_tag_pose.time.time", return_value=1000.0)
        self.clock.start()
        self.addCleanup(self.clock.stop)
        self.matrix = np.eye(4)
        self.matrix[:3, 3] = [0.1, -0.2, 1.5]
        self.record = {
            "valid": True, "reason": "TRACKING", "tag_id": 7, "family": "tag36h11",
            "tag_size_m": 0.2, "frame_id": 23,
            "time_utc": datetime.fromtimestamp(999.9, timezone.utc).isoformat(),
            "T_camera_tag": self.matrix.tolist(),
        }
        self.reader = CameraTagPoseReader(self.path, tag_id=7)
        self.write()

    def write(self):
        # Same atomic replacement mechanism used by the actual camera writer.
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.record))
        temporary.replace(self.path)

    def test_valid_record_preserves_frame_units_and_metadata(self):
        pose = self.reader.read()
        np.testing.assert_array_equal(pose.T_camera_tag, self.matrix)
        self.assertEqual(pose.tag_id, 7)
        self.assertEqual(pose.tag_size_m, 0.2)
        self.assertEqual(pose.frame_id, 23)
        self.assertAlmostEqual(pose.age_s, 0.1)
        self.assertFalse(pose.T_camera_tag.flags.writeable)

    def test_atomic_replacement_is_read_without_cached_file_handle(self):
        self.reader.read()
        self.record["T_camera_tag"][0][3] = 0.3
        self.record["frame_id"] = 24
        self.write()
        pose = self.reader.read()
        self.assertEqual(pose.T_camera_tag[0, 3], 0.3)
        self.assertEqual(pose.frame_id, 24)

    def test_missing_file_and_malformed_json(self):
        for content in ("{", "[]", "null"):
            with self.subTest(content=content):
                self.path.write_text(content)
                with self.assertRaises(CameraPoseUnavailable):
                    self.reader.read()
        self.path.unlink()
        with self.assertRaises(CameraPoseUnavailable):
            self.reader.read()

    def test_invalid_records_do_not_reuse_previous_pose(self):
        self.reader.read()
        for reason in ("STOPPED", "TAG NOT FOUND", "NO CURRENT FRAME", "AMBIGUOUS / TILT TAG"):
            self.record = {"valid": False, "reason": reason}
            self.write()
            with self.assertRaisesRegex(CameraPoseUnavailable, reason):
                self.reader.read()

    def test_stale_future_and_timezone_missing(self):
        for stamp in ("1970-01-01T00:16:39+00:00", "1970-01-01T00:16:42Z",
                      "1970-01-01T00:16:39.9", "not a timestamp"):
            self.record["time_utc"] = stamp
            self.write()
            with self.subTest(stamp=stamp), self.assertRaises(CameraPoseUnavailable):
                self.reader.read()

    def test_unchanged_record_eventually_expires(self):
        self.reader.read()
        with patch("camera_tag_pose.time.time", return_value=1001.0):
            with self.assertRaisesRegex(CameraPoseUnavailable, "stale"):
                self.reader.read()

    def test_identity_size_and_required_fields(self):
        for key, value in (("tag_id", 8), ("tag_id", True), ("family", "tag16h5"),
                           ("duplicate_id", True), ("tag_size_m", 0), ("tag_size_m", float("nan"))):
            with self.subTest(key=key, value=value):
                previous = self.record.copy()
                self.record[key] = value
                self.write()
                with self.assertRaises(CameraPoseUnavailable):
                    self.reader.read()
                self.record = previous
        del self.record["T_camera_tag"]
        self.write()
        with self.assertRaises(CameraPoseUnavailable):
            self.reader.read()

    def test_transform_rejects_nonrigid_and_nonfinite_matrices(self):
        for matrix in (np.eye(3), np.full((4, 4), float("nan")),
                       np.diag([1, 1, 1, 0]), np.diag([2, 1, 1, 1]), np.diag([-1, 1, 1, 1])):
            self.record["T_camera_tag"] = matrix.tolist()
            self.write()
            with self.assertRaises(CameraPoseUnavailable):
                self.reader.read()

    def test_reader_configuration(self):
        for kwargs in ({"tag_id": -1}, {"tag_id": True}, {"tag_id": 7, "max_age_s": 0},
                       {"tag_id": 7, "max_age_s": float("inf")}):
            with self.assertRaises(ValueError):
                CameraTagPoseReader(self.path, **kwargs)

    def make_env(self, reader):
        env_class, terms = load_environment()
        robot = Mock()
        robot.get_high_level_obs.return_value = {name: torch.zeros(size) for name, size in terms}
        robot.get_high_level_obs.return_value["object_pose_world"] = [0, 0, 0]
        low_level = Mock(control_period_s=0.01)
        env = env_class(robot, low_level, [0, 0, 0], camera_pose_reader=reader)
        return env, robot, low_level

    def test_env_refreshes_camera_without_appending_to_policy_vector(self):
        env, robot, low_level = self.make_env(self.reader)
        obs = env.get_observations()
        self.assertEqual(obs.shape, (1, 63))
        np.testing.assert_array_equal(env.camera_tag_pose.T_camera_tag, self.matrix)
        self.assertNotIn("T_camera_tag", robot.get_high_level_obs.return_value)
        low_level.step.assert_not_called()
        self.record["valid"] = False
        self.record["reason"] = "STOPPED"
        self.write()
        with self.assertRaises(CameraPoseUnavailable):
            env.get_camera_tag_pose()
        self.assertIsNone(env.camera_tag_pose)

    def test_invalid_camera_blocks_action_before_relic(self):
        env, _, low_level = self.make_env(self.reader)
        self.record["valid"] = False
        self.write()
        with self.assertRaises(CameraPoseUnavailable):
            env.step(torch.zeros((1, 12)))
        low_level.step.assert_not_called()

    def test_no_reader_preserves_previous_interface(self):
        env, _, low_level = self.make_env(None)
        self.assertIsNone(env.get_camera_tag_pose())
        with patch.object(time, "sleep"):
            obs = env.step(torch.zeros((1, 12)))
        self.assertEqual(low_level.step.call_count, 2)
        self.assertEqual(obs.shape, (1, 63))

    def test_runner_flags_and_bad_camera_preflight(self):
        args = build_parser().parse_args([
            "--high-level-policy", "not-loaded.pt", "--camera-pose-file", str(self.path),
            "--camera-tag-id", "7", "--camera-pose-max-age", "0.3",
        ])
        self.assertEqual(args.camera_pose_max_age, 0.3)
        self.record["valid"] = False
        self.write()
        with patch("high_level_policy_spot.torch.jit.load") as load:
            with self.assertRaises(CameraPoseUnavailable):
                main(args)
            load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
