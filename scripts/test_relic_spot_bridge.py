"""Hardware-free regression tests; no Isaac launch, authentication, or Spot RPCs."""

from contextlib import nullcontext
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import relic_spot_bridge as bridge
import wasd_relic_spot as wasd


def fake_robot_worker(shared, stop, ready, messages):
    """Spawn target with an explicitly fake deployment backend, never the SDK."""
    class FakeRobot:
        def __init__(self, options):
            pass
        def lease_alive(self):
            return object()
        def power_on_stand(self):
            pass
        def lease_return(self):
            pass

    class FakeEnv:
        control_period_s = .01
        def __init__(self, robot, **kwargs):
            self.previous = None
        def step(self, action):
            command = tuple(action[0])
            if command != self.previous:
                messages.put(("sample", command))
                self.previous = command
        def close(self):
            pass

    torch = SimpleNamespace(inference_mode=nullcontext, tensor=lambda values, **kwargs: values, float32=None)
    bridge._load_deployment = lambda: (FakeRobot, FakeEnv, torch)
    bridge._robot_worker("unused-test-host", None, shared, stop, ready, messages)


class BridgeTests(unittest.TestCase):
    def test_import_and_flags_do_not_launch_or_import_robot(self):
        self.assertNotIn("robot", sys.modules)
        self.assertNotIn("bosdyn.client", sys.modules)
        self.assertNotIn("isaaclab.app", sys.modules)
        parser = wasd.build_parser()
        self.assertFalse(parser.parse_args([]).real_robot)
        self.assertTrue(parser.parse_args(["--real-robot"]).real_robot)
        self.assertTrue(parser.parse_args(["--real_robot"]).real_robot)

    def test_keyboard_command_is_the_same_for_sim_and_bridge(self):
        # Exercise the real callbacks without constructing a Tk window.
        controller = wasd.WASDController.__new__(wasd.WASDController)
        controller._linear_speed = .2
        controller._turn_speed = .4
        controller._active_key = None
        controller._on_command = Mock()
        controller._refresh_label = Mock()
        controller.root = Mock()
        expected = {"w": (.2, 0, 0), "s": (-.2, 0, 0), "a": (0, .2, 0),
                    "d": (0, -.2, 0), "q": (0, 0, .4), "e": (0, 0, -.4), "space": (0, 0, 0)}
        for key, velocity in expected.items():
            controller._on_key_press(SimpleNamespace(keysym=key))
            self.assertEqual(controller.command, velocity + (0.0,) * 9)
            controller._on_command.assert_called_with(controller.command)
        controller._on_key_press(SimpleNamespace(keysym="w"))
        controller._request_reset()
        self.assertEqual(controller.command, bridge.ZERO_COMMAND)
        self.assertTrue(controller.consume_reset())
        self.assertFalse(controller.consume_reset())
        controller._request_quit()
        self.assertTrue(controller.should_quit)
        controller._on_command.assert_called_with(bridge.ZERO_COMMAND)

    def test_mailbox_blocks_motion_before_ready_and_keeps_latest_command(self):
        real = bridge.RealSpotBridge("unused-test-host")
        try:
            real.publish((.2, 0, 0) + (0,) * 9)
            self.assertEqual(tuple(real._shared_command[:12]), bridge.ZERO_COMMAND)
            real._ready.set()
            real.publish((.2, 0, 0) + (0,) * 9)
            real.publish((0, -.2, 0) + (0,) * 9)
            self.assertEqual(tuple(real._shared_command[:12]), (0, -.2, 0) + (0,) * 9)
            generation = real._shared_command[13]
            real.publish((0, -.2, 0) + (0,) * 9)
            self.assertEqual(real._shared_command[13], generation)
            real.publish(bridge.ZERO_COMMAND)
            self.assertGreater(real._shared_command[13], generation)
        finally:
            real.close()

    def test_main_real_flag_shares_commands_and_reset_is_sim_only(self):
        real = Mock(ready=True)
        real.poll.side_effect = [[("ready", "fake ready")]] + [[]] * 10
        controller = wasd.WASDController.__new__(wasd.WASDController)
        controller._linear_speed, controller._turn_speed = .2, .4
        controller._active_key, controller._command = None, bridge.ZERO_COMMAND
        controller._reset_requested = controller._quit_requested = False
        controller._on_command = real.publish
        controller._refresh_label = controller.set_status = controller.close = Mock()
        controller.root = Mock()
        keys = iter(("space", "w", "space", "r", "s", "escape"))
        controller.poll_events = lambda: controller._on_key_press(SimpleNamespace(keysym=next(keys)))
        env = Mock(action_space=SimpleNamespace(shape=(1, 12)), num_envs=1, device="cpu", step_dt=.02)
        env.unwrapped = env
        env.sim.is_playing.return_value = True
        delivered = []
        def step(action):
            # Both consumers must get the identical command on every sim step.
            self.assertEqual(real.publish.call_args.args[0], tuple(action[0]))
            delivered.append(tuple(action[0]))
        env.step.side_effect = step
        def reset():
            if env.reset.call_count > 1:
                real.publish.assert_called_with(bridge.ZERO_COMMAND)
        env.reset.side_effect = reset
        torch = SimpleNamespace(inference_mode=nullcontext, tensor=lambda values, **kwargs: values, float32=None)
        tasks = SimpleNamespace()
        modules = {
            "torch": torch, "gymnasium": SimpleNamespace(make=lambda *a, **k: env),
            "isaaclab_tasks": tasks, "isaaclab_tasks.utils": SimpleNamespace(parse_env_cfg=lambda *a, **k: None),
            "SuperQ_ALORE": SimpleNamespace(tasks=tasks), "SuperQ_ALORE.tasks": tasks,
        }
        args = wasd.build_parser().parse_args(["--real-robot"])
        args.device = "cpu"
        with patch.dict(sys.modules, modules), patch.object(bridge, "RealSpotBridge", return_value=real), \
             patch.object(wasd, "WASDController", return_value=controller), patch.object(wasd.time, "sleep"):
            wasd.main(args, SimpleNamespace(is_running=lambda: True))
        self.assertIn((.2, 0, 0) + (0,) * 9, delivered)
        self.assertIn((-.2, 0, 0) + (0,) * 9, delivered)
        self.assertEqual(env.reset.call_count, 2)
        real.reset.assert_not_called()
        real.start.assert_called_once()
        real.close.assert_called_once()
        env.close.assert_called_once()

    def test_invalid_commands_rejected(self):
        for command in ((0,) * 3, (float("nan"),) + (0,) * 11,
                        (float("inf"),) + (0,) * 11, (0,) * 3 + (1,) + (0,) * 8):
            with self.assertRaises(ValueError):
                bridge.validate_command(command)

    def test_missing_credentials_cannot_spawn_worker(self):
        real = bridge.RealSpotBridge("unused-test-host")
        try:
            with patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(ValueError, "BOSDYN_CLIENT"):
                real.start()
            self.assertIsNone(real._process)
        finally:
            real.close()

    def test_stale_command_requires_a_new_selection(self):
        watchdog = bridge.CommandWatchdog(timeout_s=.5)
        moving = (.2, 0, 0) + (0,) * 9
        self.assertEqual(watchdog.select(moving, 10, 1, 10.1), (moving, False))
        self.assertEqual(watchdog.select(moving, 10, 1, 10.6), (bridge.ZERO_COMMAND, True))
        self.assertEqual(watchdog.select(moving, 10, 1, 10.7), (bridge.ZERO_COMMAND, False))
        # A resumed heartbeat cannot silently resume the old movement.
        self.assertEqual(watchdog.select(moving, 11, 1, 11), (bridge.ZERO_COMMAND, False))
        self.assertEqual(watchdog.select(bridge.ZERO_COMMAND, 11, 2, 11), (bridge.ZERO_COMMAND, False))
        self.assertEqual(watchdog.select(moving, 11, 3, 11), (moving, False))

    def test_policy_loop_uses_real_observations_via_env_and_stands_on_exit(self):
        ctx = mp.get_context("spawn")
        shared = ctx.Array("d", [*(-.2, 0, 0), *((0,) * 9), 1.0, 1.0])
        ready = ctx.Event()
        stop = ctx.Event()
        received = []
        def step(action):
            received.append(tuple(action[0]))
            if len(received) == 1:
                self.assertFalse(ready.is_set())
            if len(received) == 3:
                stop.set()
        env = SimpleNamespace(step=step, control_period_s=.01)
        torch = SimpleNamespace(inference_mode=nullcontext, tensor=lambda values, **kwargs: values, float32=None)
        with patch.object(bridge.time, "perf_counter", return_value=1.0), patch.object(bridge.time, "sleep"):
            bridge._run_policy_loop(env, torch, shared, stop, ready, Mock(), lambda: True)
        self.assertEqual(received[0], bridge.ZERO_COMMAND)
        self.assertEqual(received[1:3], [(-.2, 0, 0) + (0,) * 9] * 2)
        self.assertEqual(received[3:], [bridge.ZERO_COMMAND] * 25)

    def test_worker_cleanup_on_policy_failure(self):
        robot = Mock()
        env = Mock()
        ready = Mock()
        stop = Mock(is_set=Mock(return_value=False))
        messages = Mock()
        robot_factory, env_factory = Mock(return_value=robot), Mock(return_value=env)
        with patch.object(bridge.signal, "signal"), \
             patch.object(bridge, "_load_deployment", return_value=(robot_factory, env_factory, Mock())), \
             patch.object(bridge, "_run_policy_loop", side_effect=RuntimeError("test policy failure")):
            bridge._robot_worker("unused-test-host", None, Mock(), stop, ready, messages)
        robot.lease_alive.assert_called_once()
        robot.power_on_stand.assert_called_once()
        env.close.assert_called_once()
        robot.lease_return.assert_called_once()
        self.assertTrue(any(call.args[0][0] == "error" for call in messages.put.call_args_list))

    def test_worker_cleanup_on_environment_construction_failure(self):
        robot = Mock()
        with patch.object(bridge.signal, "signal"), patch.object(
            bridge, "_load_deployment",
            return_value=(Mock(return_value=robot), Mock(side_effect=RuntimeError("test init failure")), Mock()),
        ):
            bridge._robot_worker("unused-test-host", None, Mock(), Mock(is_set=Mock(return_value=False)), Mock(), Mock())
        robot.shutdown_relic.assert_called_once()
        robot.lease_return.assert_called_once()

    def test_spawned_worker_command_watchdog_and_shutdown(self):
        real = bridge.RealSpotBridge("unused-test-host")
        real._process = mp.get_context("spawn").Process(
            target=fake_robot_worker,
            args=(real._shared_command, real._stop, real._ready, real._messages),
        )
        real._process.start()
        samples = []
        def wait_for(predicate, command=None):
            deadline = time.perf_counter() + 5
            while time.perf_counter() < deadline:
                if command is not None:
                    real.publish(command)
                for kind, message in real.poll():
                    if kind == "sample":
                        samples.append(message)
                if predicate():
                    return
                time.sleep(.01)
            self.fail("Spawned fake worker did not reach expected state.")
        try:
            wait_for(lambda: real.ready)
            moving = (.2, 0, 0) + (0,) * 9
            wait_for(lambda: moving in samples, moving)
            # Stop heartbeats: the child must independently zero its command.
            wait_for(lambda: samples[-1] == bridge.ZERO_COMMAND)
            count = len(samples)
            # Fresh heartbeats alone must NOT resume the stale selection.
            for _ in range(5):
                real.publish(moving)
                time.sleep(.01)
            for kind, message in real.poll():
                if kind == "sample":
                    samples.append(message)
            self.assertEqual(len(samples), count)
            real.publish(bridge.ZERO_COMMAND)
            wait_for(lambda: len(samples) > count and samples[-1] == moving, moving)
        finally:
            process = real._process
            real.close()
            self.assertFalse(process.is_alive())


if __name__ == "__main__":
    unittest.main()
