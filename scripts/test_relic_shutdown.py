"""Test the real cleanup methods with SDK calls mocked; no hardware imports."""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, call


def load_cleanup_methods():
    # Compile only these methods so tests do not need the robot's camera/SDK
    # dependencies and cannot accidentally instantiate a hardware connection.
    path = Path(__file__).with_name("robot.py")
    tree = ast.parse(path.read_text())
    selected = {
        "SPOT": {"shutdown_relic", "self_right", "estop", "_arm_stow"},
        "SpotReLICEnvPLAY": {"close"},
    }
    classes = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name in selected:
            node.body = [method for method in node.body if isinstance(method, ast.FunctionDef)
                         and method.name in selected[node.name]]
            classes.append(node)
    namespace = {
        "blocking_selfright": Mock(),
        "block_until_arm_arrives": Mock(return_value=True),
        "RobotCommandBuilder": SimpleNamespace(arm_stow_command=Mock(return_value="stow")),
    }
    exec(compile(ast.Module(body=classes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


class ReLICShutdownTests(unittest.TestCase):
    def setUp(self):
        self.ns = load_cleanup_methods()
        self.spot = self.ns["SPOT"]()
        self.spot.enable_joint_commands = True
        self.spot.robot = Mock()
        self.spot.robot.is_powered_on.return_value = True
        self.spot.robot.is_estopped.return_value = False
        self.order = Mock()
        self.spot.joint_level_control_stop = self.order.stop_stream
        self.spot.open_gripper = self.order.open_gripper
        self.spot._arm_stow = self.order.stow
        self.spot._arm_stow.return_value = True
        self.spot.self_right = self.order.self_right
        self.spot.power_off = self.order.power_off

    def test_normal_cleanup_order(self):
        self.spot.shutdown_relic()
        self.assertEqual(self.order.mock_calls, [
            call.stop_stream(), call.open_gripper(), call.stow(timeout_sec=10.0),
            call.self_right(), call.power_off(),
        ])

    def test_recovery_not_added_to_dry_run_or_estopped_or_unpowered_robot(self):
        for setting in ("dry_run", "estopped", "unpowered"):
            with self.subTest(setting=setting):
                self.setUp()
                if setting == "dry_run":
                    self.spot.enable_joint_commands = False
                elif setting == "estopped":
                    self.spot.robot.is_estopped.return_value = True
                else:
                    self.spot.robot.is_powered_on.return_value = False
                self.spot.shutdown_relic()
                self.assertEqual(self.order.mock_calls, [call.stop_stream(), call.power_off()])

    def test_failed_stow_skips_selfright_and_attempts_poweroff(self):
        self.spot._arm_stow.return_value = False
        with self.assertRaisesRegex(RuntimeError, "Arm stow did not complete"):
            self.spot.shutdown_relic()
        self.spot.self_right.assert_not_called()
        self.spot.power_off.assert_called_once()

    def test_stop_release_and_recovery_failures_still_attempt_poweroff(self):
        for stage in ("joint_level_control_stop", "open_gripper", "self_right"):
            with self.subTest(stage=stage):
                self.setUp()
                error = RuntimeError("test shutdown failure")
                getattr(self.spot, stage).side_effect = error
                with self.assertRaises(RuntimeError) as raised:
                    self.spot.shutdown_relic()
                self.assertIs(raised.exception, error)
                self.spot.power_off.assert_called_once()

    def test_sdk_helpers_wait_and_report_stow_result(self):
        spot = self.ns["SPOT"]()
        spot.command_client = Mock()
        spot.command_client.robot_command.return_value = 42
        spot.self_right()
        self.ns["blocking_selfright"].assert_called_once_with(spot.command_client, timeout_sec=10)
        self.assertTrue(spot._arm_stow(timeout_sec=10.0))
        spot.command_client.robot_command.assert_called_once_with("stow")
        self.ns["block_until_arm_arrives"].assert_called_once_with(spot.command_client, 42, 10.0)
        self.ns["block_until_arm_arrives"].return_value = False
        self.assertFalse(spot._arm_stow())

    def test_environment_close_uses_recovery_even_if_logging_or_visualizer_fails(self):
        for failing_stage in (None, "report_max_latency", "visualizer"):
            with self.subTest(failing_stage=failing_stage):
                env = self.ns["SpotReLICEnvPLAY"]()
                env.report_max_latency = Mock()
                env.save_observation_log = Mock()
                env.joint_pose_visualizer = Mock()
                env._policy_trace_file = Mock()
                env.robot = Mock()
                trace = env._policy_trace_file
                if failing_stage == "report_max_latency":
                    env.report_max_latency.side_effect = RuntimeError("test logging failure")
                elif failing_stage == "visualizer":
                    env.joint_pose_visualizer.close.side_effect = RuntimeError("test dashboard failure")
                if failing_stage:
                    with self.assertRaises(RuntimeError):
                        env.close()
                else:
                    env.close()
                trace.close.assert_called_once()
                env.robot.shutdown_relic.assert_called_once()


if __name__ == "__main__":
    unittest.main()
