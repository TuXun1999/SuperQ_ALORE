"""Chair-pull routing tests using fake robot objects; never connect to Spot."""

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

# Load outside patch.dict(sys.modules), so mock cleanup does not unload torch.
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import zero_agent_spot as agent


class ChairPullTests(unittest.TestCase):
    def run_agent(self, cli, *, step_error=None, grasp_error=None, init_error=None):
        events = []
        robot = Mock()
        robot.close_gripper.side_effect = lambda: events.append("gripper_closed")
        if grasp_error is not None:
            robot.close_gripper.side_effect = grasp_error
        env = Mock(control_period_s=.01)
        commands = []
        def step(action):
            events.append("policy_step")
            commands.append(action.clone())
            if step_error is not None:
                raise step_error
        env.step.side_effect = step
        def create_env(*args, **kwargs):
            events.append("create_env")
            if init_error is not None:
                raise init_error
            return env
        mpc_factory = Mock(side_effect=create_env)
        relic_factory = Mock(side_effect=create_env)
        robot_factory = Mock(return_value=robot)
        fake_module = SimpleNamespace(SPOT=robot_factory, SpotRLEnvPLAY=mpc_factory, SpotReLICEnvPLAY=relic_factory)
        args = agent.build_parser().parse_args(cli)
        error = None
        with patch.dict(sys.modules, {"robot": fake_module}), \
             patch.object(agent.time, "perf_counter", return_value=1.0), \
             patch.object(agent.time, "sleep") as sleep:
            try:
                agent.main(args)
            except BaseException as caught:
                error = caught
        return SimpleNamespace(
            args=args, robot=robot, env=env, events=events, commands=commands,
            mpc=mpc_factory, relic=relic_factory, factory=robot_factory, sleep=sleep, error=error,
        )

    def test_mpc_remains_default_without_joint_stream_or_power_off(self):
        result = self.run_agent(["--steps", "3"])
        self.assertIsNone(result.error)
        self.assertFalse(result.args.enable_joint_commands)
        result.mpc.assert_called_once_with(result.robot)
        result.relic.assert_not_called()
        self.assertEqual(result.env.step.call_count, 3)
        result.env.close.assert_not_called()
        result.robot.power_off.assert_not_called()
        result.robot.joint_level_control_stop.assert_not_called()
        result.robot.lease_return.assert_called_once()
        result.sleep.assert_called_once_with(1)

    def test_relic_starts_after_grip_with_logging_disabled(self):
        result = self.run_agent(["--use-relic", "--steps", "3"])
        self.assertIsNone(result.error)
        self.assertTrue(result.args.enable_joint_commands)
        result.mpc.assert_not_called()
        result.relic.assert_called_once_with(
            result.robot, enable_latency_reporting=False,
            enable_observation_logging=False, enable_policy_trace_logging=False,
        )
        self.assertEqual(result.events[:2], ["gripper_closed", "create_env"])
        for action in result.commands:
            self.assertEqual(tuple(action.shape), (1, 12))
            self.assertAlmostEqual(action[0, 0].item(), -.2)
            self.assertEqual(action[0, 1:].count_nonzero().item(), 0)
        self.assertEqual(result.env.step.call_count, 3)
        # One existing grasp wait plus an absolute 100 Hz deadline per update.
        self.assertEqual(result.sleep.call_count, 4)
        waits = [call.args[0] for call in result.sleep.call_args_list[1:]]
        for actual, expected in zip(waits, [.01, .02, .03]):
            self.assertAlmostEqual(actual, expected)
        result.env.close.assert_called_once()
        result.robot.lease_return.assert_called_once()

    def test_relic_cleans_up_on_policy_error_or_ctrl_c(self):
        for error in (RuntimeError("test policy error"), KeyboardInterrupt()):
            with self.subTest(error=type(error).__name__):
                result = self.run_agent(["--use-relic"], step_error=error)
                self.assertIs(result.error, error)
                result.env.close.assert_called_once()
                result.robot.lease_return.assert_called_once()

    def test_relic_cleans_up_if_grasp_or_initialization_fails(self):
        for stage in ("grasp_error", "init_error"):
            with self.subTest(stage=stage):
                error = RuntimeError("test startup error")
                result = self.run_agent(["--use-relic"], **{stage: error})
                self.assertIs(result.error, error)
                result.env.step.assert_not_called()
                result.robot.shutdown_relic.assert_called_once()
                result.robot.lease_return.assert_called_once()

    def test_hostname_spellings_and_validation(self):
        for cli in (["--hostname", "example-spot"], ["example-spot"]):
            result = self.run_agent(cli + ["--steps", "1"])
            self.assertIsNone(result.error)
            self.assertEqual(result.args.hostname, "example-spot")
        for cli in (["--steps", "0"], ["--steps", "-1"], ["host-a", "--hostname", "host-b"]):
            result = self.run_agent(cli)
            self.assertIsInstance(result.error, ValueError)
            result.factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()
