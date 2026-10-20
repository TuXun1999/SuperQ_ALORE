"""Optional, wall-clock ReLIC deployment worker for the simulation WASD panel.

Importing this module never imports the Spot SDK or connects to hardware. The
worker reuses the deployment path from zero_agent_ReLIC_spot.py; it receives
high-level commands, never simulated observations or simulated joint targets.
"""

import math
import multiprocessing as mp
import os
from queue import Empty
import signal
import time
from types import SimpleNamespace


ZERO_COMMAND = (0.0,) * 12
COMMAND_TIMEOUT_S = 0.5


def validate_command(command):
    values = tuple(float(value) for value in command)
    if len(values) != 12 or not all(math.isfinite(value) for value in values):
        raise ValueError("Expected 12 finite ReLIC command values.")
    # This bridge is for base-only teleoperation. Keep the measured arm pose.
    if any(values[3:]):
        raise ValueError("WASD hardware commands must have zero arm/torso deltas.")
    return values


class CommandWatchdog:
    """After a lost heartbeat, require a new selection before moving again."""

    def __init__(self, timeout_s=COMMAND_TIMEOUT_S):
        self.timeout_s = timeout_s
        self.blocked_generation = None

    def select(self, command, timestamp, generation, now):
        stale = now - timestamp > self.timeout_s
        newly_stale = stale and self.blocked_generation != generation
        if stale:
            self.blocked_generation = generation
        if stale or generation == self.blocked_generation:
            return ZERO_COMMAND, newly_stale
        return validate_command(command), False


def _load_deployment():
    # Lazy imports: the spawned process does not launch Isaac Sim.
    from robot import SPOT, SpotReLICEnvPLAY
    import torch

    return SPOT, SpotReLICEnvPLAY, torch


def _run_policy_loop(env, torch, shared_command, stop, ready, messages, parent_alive):
    watchdog = CommandWatchdog()
    # Always enter joint control with the existing standing command, even if
    # someone selected a direction while the robot was connecting.
    with torch.inference_mode():
        env.step(torch.tensor([ZERO_COMMAND], dtype=torch.float32))
    ready.set()
    messages.put(("ready", "Real Spot ready; select a movement key. Space commands standing."))
    deadline = time.perf_counter()
    while not stop.is_set() and parent_alive():
        with shared_command.get_lock():
            snapshot = tuple(shared_command[:])
        command, timed_out = watchdog.select(
            snapshot[:12], snapshot[12], snapshot[13], time.perf_counter()
        )
        if timed_out:
            messages.put(("watchdog", "Panel heartbeat lost: commanding standing; select a new command to resume."))
        with torch.inference_mode():
            env.step(torch.tensor([command], dtype=torch.float32))
        deadline += env.control_period_s
        remaining = deadline - time.perf_counter()
        if remaining > 0:
            stop.wait(remaining)
        else:
            # Do not burst old control updates to catch up with wall time.
            deadline = time.perf_counter()

    # Briefly command standing before the existing stop-stream/power-off path.
    # This is a requested stop, not a guarantee of physical balance or an E-stop.
    for _ in range(max(1, round(0.25 / env.control_period_s))):
        with torch.inference_mode():
            env.step(torch.tensor([ZERO_COMMAND], dtype=torch.float32))
        time.sleep(env.control_period_s)


def _robot_worker(hostname, trace_path, shared_command, stop, ready, messages):
    # The parent handles terminal Ctrl-C and requests an orderly worker stop.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    robot = env = lease = None
    power_requested = False
    parent = mp.parent_process()
    def parent_alive():
        return parent is None or parent.is_alive()
    try:
        SPOT, SpotReLICEnvPLAY, torch = _load_deployment()
        if stop.is_set() or not parent_alive():
            return
        robot = SPOT(SimpleNamespace(hostname=hostname, enable_joint_commands=True))
        if stop.is_set() or not parent_alive():
            return
        lease = robot.lease_alive()
        if stop.is_set() or not parent_alive():
            return
        power_requested = True
        robot.power_on_stand()
        if stop.is_set() or not parent_alive():
            return
        env = SpotReLICEnvPLAY(
            robot,
            enable_latency_reporting=False,
            enable_observation_logging=False,
            enable_policy_trace_logging=trace_path is not None,
            policy_trace_log_path=trace_path or "relic_policy_trace.txt",
        )
        _run_policy_loop(
            env, torch, shared_command, stop, ready, messages,
            parent_alive=parent_alive,
        )
    except Exception as error:
        messages.put(("error", f"Real Spot worker failed: {type(error).__name__}: {error}"))
    finally:
        ready.clear()
        try:
            if env is not None:
                env.close()
            elif robot is not None and power_requested:
                # Also clean up if standing or environment initialization failed.
                robot.shutdown_relic()
        except Exception as error:
            messages.put(("error", f"Robot shutdown failed; use external E-stop: {error}"))
        finally:
            if lease is not None:
                try:
                    robot.lease_return()
                except Exception as error:
                    messages.put(("error", f"Could not return Spot lease: {error}"))


class RealSpotBridge:
    """Latest-command mailbox plus an independent 100 Hz deployment process."""

    def __init__(self, hostname, trace_path=None):
        self.hostname = hostname
        self.trace_path = trace_path
        ctx = mp.get_context("spawn")
        # command[12], heartbeat time, command generation. A mailbox avoids
        # queuing stale directions while connection or inference is slow.
        self._shared_command = ctx.Array("d", [*ZERO_COMMAND, time.perf_counter(), 0.0])
        self._stop = ctx.Event()
        self._ready = ctx.Event()
        self._messages = ctx.Queue()
        self._process = None
        self._closed = False

    @property
    def ready(self):
        return self._ready.is_set()

    def start(self):
        # Spawned workers cannot prompt on stdin. Use the same SDK environment
        # authentication as zero_agent_ReLIC_spot.py, without command-line secrets.
        if not all(os.environ.get(name) for name in ("BOSDYN_CLIENT_USERNAME", "BOSDYN_CLIENT_PASSWORD")):
            raise ValueError("Set BOSDYN_CLIENT_USERNAME and BOSDYN_CLIENT_PASSWORD before using --real-robot.")
        if self._process is not None:
            raise RuntimeError("Real Spot worker already started.")
        self._process = mp.get_context("spawn").Process(
            target=_robot_worker,
            args=(self.hostname, self.trace_path, self._shared_command, self._stop, self._ready, self._messages),
            name="wasd_relic_real_spot",
        )
        self._process.start()

    def publish(self, command):
        command = validate_command(command)
        if not self.ready:
            command = ZERO_COMMAND
        with self._shared_command.get_lock():
            if tuple(self._shared_command[:12]) != command:
                self._shared_command[13] += 1
            self._shared_command[:12] = command
            self._shared_command[12] = time.perf_counter()

    def poll(self):
        messages = []
        while True:
            try:
                kind, message = self._messages.get_nowait()
            except Empty:
                break
            if kind == "error":
                raise RuntimeError(message)
            messages.append((kind, message))
        if self._process is not None and self._process.exitcode is not None:
            raise RuntimeError(f"Real Spot worker exited (code {self._process.exitcode}); ending WASD control.")
        return messages

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        if self._process is not None and self._process.pid is not None:
            # Allow the release/stow/self-right sequence and motor power-off
            # to complete before considering the worker unresponsive.
            self._process.join(timeout=60.0)
            if self._process.is_alive():
                print("[REAL SPOT] Shutdown timed out; terminating worker. Power-off is NOT confirmed; use external E-stop.", flush=True)
                self._process.terminate()
                self._process.join(timeout=2.0)
            self._process = None
        while True:
            try:
                kind, message = self._messages.get_nowait()
            except Empty:
                break
            if kind == "error":
                print(f"[REAL SPOT] {message}", flush=True)
        self._messages.close()
