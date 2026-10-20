"""Hardware-free checks for the hierarchical observation/action adapter."""
import math
import ast
from pathlib import Path
import time
import unittest
from unittest.mock import Mock, patch

import torch

# Compile the actual adapter without importing robot.py's hardware/Isaac stack.
source = ast.parse(Path(__file__).with_name("robot.py").read_text())
nodes = [node for node in source.body if
         (isinstance(node, (ast.FunctionDef, ast.ClassDef))
          and node.name in {"vector", "HighLevelSpotEnv"})
         or (isinstance(node, ast.Assign)
             and any(isinstance(target, ast.Name) and target.id == "OBS_TERMS"
                     for target in node.targets))]
namespace = {"torch": torch, "time": time, "math": math}
exec(compile(ast.Module(body=nodes, type_ignores=[]), "robot.py", "exec"), namespace)
HighLevelSpotEnv = namespace["HighLevelSpotEnv"]
OBS_TERMS = namespace["OBS_TERMS"]


class HighLevelSpotTests(unittest.TestCase):
    def setUp(self):
        self.measured = {name: torch.zeros(size) for name, size in OBS_TERMS}
        self.measured["object_pose_world"] = [1, 2, math.pi / 2]
        self.robot = Mock()
        self.robot.get_high_level_obs.return_value = self.measured
        self.low = Mock(control_period_s=0.01)
        self.env = HighLevelSpotEnv(self.robot, self.low, [1, 3, math.pi])

    def test_policy_order_and_local_goal(self):
        self.measured["joint_vel"] = torch.ones(18)
        obs = self.env.get_observations()
        self.assertEqual(obs.shape, (1, 63))
        torch.testing.assert_close(obs[:, 18:36], torch.ones(1, 18))
        torch.testing.assert_close(obs[:, -3:], torch.tensor([[1., 0., math.pi / 2]]))

    def test_two_relic_updates_and_previous_action(self):
        action = torch.arange(12).float().unsqueeze(0)
        with patch.object(time, "sleep"):
            obs = self.env.step(action)
        self.assertEqual(self.low.step.call_count, 2)
        for call in self.low.step.call_args_list:
            torch.testing.assert_close(call.args[0], action)
        torch.testing.assert_close(obs[:, 41:50], action[:, :9])

    def test_missing_sensor_data_fails(self):
        del self.measured["joint_pos"]
        with self.assertRaisesRegex(ValueError, "joint_pos"):
            self.env.get_observations()

    def test_custom_low_level_frequency(self):
        env = HighLevelSpotEnv(self.robot, self.low, [1, 3, math.pi], low_level_frequency_hz=200)
        self.assertEqual(self.low.control_period_s, 0.005)
        with patch.object(time, "sleep"):
            env.step(torch.zeros(12))
        self.assertEqual(self.low.step.call_count, 4)

    def test_invalid_frequency(self):
        for frequency in (0, -50, 25, 75, float("nan"), float("inf")):
            with self.subTest(frequency=frequency), self.assertRaises(ValueError):
                HighLevelSpotEnv(self.robot, self.low, [0, 0, 0], low_level_frequency_hz=frequency)

    def test_bad_action_never_reaches_relic(self):
        for action in (torch.zeros(9), torch.full((12,), float("nan"))):
            with self.assertRaises(ValueError):
                self.env.step(action)
        self.low.step.assert_not_called()


if __name__ == "__main__":
    unittest.main()
