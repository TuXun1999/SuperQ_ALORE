"""Deployment-style Tk display of env 0's base commands sent to ReLIC.

The viewer runs as a standalone subprocess: multiprocessing's spawn would
re-import play.py and launch another Isaac Sim app. Small nonblocking pipe
writes keep Tk out of the simulation loop. History uses simulation seconds;
the stale-data indicator uses wall time. This shows commands, not measured
robot motion.

Preview without Isaac Sim: python base_command_visualizer.py --demo
"""

from collections import deque
import json
import math
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import time


class BaseCommandVisualizer:
    """Own an optional desktop viewer without blocking on display updates."""

    def __init__(self):
        self._process = None
        self._closed = False
        try:
            self._process = subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()), "--window"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                bufsize=0,
            )
            with selectors.DefaultSelector() as ready:
                ready.register(self._process.stdout, selectors.EVENT_READ)
                if not ready.select(timeout=10.0):
                    raise RuntimeError("Base command display did not start within 10 seconds.")
                message = self._process.stdout.readline()
            if not message:
                raise RuntimeError("Base command display exited before opening its window.")
            error = json.loads(message)
            if error is not None:
                raise RuntimeError(
                    f"Cannot open base command display: {error}. "
                    "Use a desktop display or omit --visualize-base-command."
                )
            os.set_blocking(self._process.stdin.fileno(), False)
        except Exception:
            self.close()
            raise
        finally:
            if self._process is not None:
                self._process.stdout.close()

    def update(self, *, vx, vtheta, simulation_time):
        """Publish one env-0 command; return False if the viewer has closed."""
        if self._closed:
            return False
        if self._process.poll() is not None:
            self.close()
            return False
        sample = (float(simulation_time), float(vx), float(vtheta))
        if not all(math.isfinite(value) for value in sample):
            raise ValueError("Base command display requires finite times and velocities.")
        # A short line fits within PIPE_BUF, so nonblocking writes are atomic.
        message = (json.dumps(sample) + "\n").encode("utf-8")
        try:
            os.write(self._process.stdin.fileno(), message)
        except BlockingIOError:
            pass  # Drop display samples when the GUI falls behind.
        except OSError:
            self.close()
            return False
        return True

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self._process is None:
            return
        self._process.stdin.close()  # EOF tells Tk to exit on its next refresh.
        try:
            self._process.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            self._process.terminate()
            try:
                self._process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=1.0)


def _step_points(samples, index, now, left, top, width, height, history_seconds):
    """Map timestamped commands to the deployment viewer's held-value trace."""
    points = []
    previous_y = None
    for sample in samples:
        x = left + width * max(0.0, min(1.0, (sample[0] - now + history_seconds) / history_seconds))
        y = top + height * (0.5 - sample[index] / 1.4)
        if previous_y is not None:
            points.extend((x, previous_y))
        points.extend((x, y))
        previous_y = y
    return points


class _CommandWindow:
    HISTORY_SECONDS = 10.0
    STALE_SECONDS = 0.5
    BACKGROUND = "#101821"
    MUTED = "#9aaebe"

    def __init__(self, root, sample_fd):
        import tkinter as tk

        self.root, self.sample_fd = root, sample_fd
        self.history = deque(maxlen=4096)
        self._buffer = b""
        self._last_received_at = None
        os.set_blocking(sample_fd, False)
        root.title("Isaac Sim | Env 0 high-level base command to ReLIC")
        root.geometry("960x640")
        root.minsize(700, 520)
        self.canvas = tk.Canvas(root, bg=self.BACKGROUND, highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        root.after(0, self._refresh)

    def _refresh(self):
        # Bound work per GUI refresh even when simulation runs faster than real time.
        for _ in range(4):
            try:
                chunk = os.read(self.sample_fd, 65536)
            except BlockingIOError:
                break
            if not chunk:
                self.root.destroy()
                return
            lines = (self._buffer + chunk).split(b"\n")
            self._buffer = lines.pop()
            for line in lines:
                self.history.append(tuple(json.loads(line)))
                self._last_received_at = time.monotonic()
        now = self.history[-1][0] if self.history else 0.0
        while len(self.history) > 1 and self.history[1][0] < now - self.HISTORY_SECONDS:
            self.history.popleft()
        self._draw(now)
        self.root.after(100, self._refresh)

    def _text(self, x, y, value, *, color=None, size=12, anchor="nw"):
        self.canvas.create_text(
            x, y, text=value, fill=color or self.MUTED, font=("TkDefaultFont", size), anchor=anchor
        )

    def _draw(self, now):
        c = self.canvas
        c.delete("all")
        width, height = max(700, c.winfo_width()), max(520, c.winfo_height())
        self._text(26, 18, "Env 0 | High-level base command → ReLIC", color="#eef4fa", size=20)
        self._text(26, 55, "Body frame  |  vy = 0 m/s  |  vx limit ±0.6 m/s  |  vtheta limit ±0.6 rad/s")
        age = None if self._last_received_at is None else max(0.0, time.monotonic() - self._last_received_at)
        if age is None:
            status = "Waiting for the first command (after simulation reset)…"
        elif age <= self.STALE_SECONDS:
            status = f"Receiving commands  |  simulation time {now:.2f} s"
        else:
            status = f"No new command for {age:.1f} s  |  readouts show the last received command"
        self._text(26, 82, status, color="#a7d5be" if age is not None and age <= self.STALE_SECONDS else "#e7b66d")
        panel_height = (height - 168) / 2
        for index, name, unit, color, positive, negative in (
            (1, "vx", "m/s", "#61bfff", "FORWARD", "BACKWARD"),
            (2, "vtheta", "rad/s", "#f7bc65", "LEFT / CCW", "RIGHT / CW"),
        ):
            panel_top = 118 + (index - 1) * panel_height
            value = self.history[-1][index] if self.history else None
            readout = "—" if value is None else f"{value:+.3f} {unit}"
            direction = "" if value is None else (positive if value > 1e-6 else negative if value < -1e-6 else "ZERO")
            self._text(26, panel_top, f"{name}   {readout}", color=color, size=22)
            self._text(width - 26, panel_top + 9, direction, color=color, anchor="ne")
            left, top, plot_width, plot_height = 76, panel_top + 53, width - 106, panel_height - 88
            for tick in (-0.6, 0.0, 0.6):
                y = top + plot_height * (0.5 - tick / 1.4)
                c.create_line(left, y, left + plot_width, y, fill="#304250", dash=(4, 4))
                self._text(left - 10, y, f"{tick:+.1f}", size=10, anchor="e")
            for seconds in (-10, -5, 0):
                x = left + plot_width * (1 + seconds / self.HISTORY_SECONDS)
                c.create_line(x, top, x, top + plot_height, fill="#24333f")
                self._text(x, top + plot_height + 6, f"{seconds} s", size=10, anchor="n")
            visible = [sample for sample in self.history if sample[0] >= now - self.HISTORY_SECONDS]
            if visible and self.history[0][0] < now - self.HISTORY_SECONDS:
                visible.insert(0, self.history[0])
            points = _step_points(visible, index, now, left, top, plot_width, plot_height, self.HISTORY_SECONDS)
            if len(points) >= 4:
                c.create_line(*points, fill=color, width=2)
            if points:
                x, y = points[-2:]
                c.create_oval(x - 3, y - 3, x + 3, y + 3, fill=color, outline="")
        self._text(
            26, height - 22,
            "Commanded velocity, not measured motion. History: simulation seconds. Closing this window leaves playback running.",
            size=10,
        )


def _run_window():
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    root = None
    ready = False
    try:
        import tkinter as tk

        root = tk.Tk()
        _CommandWindow(root, sys.stdin.fileno())
        print(json.dumps(None), flush=True)
        ready = True
        root.mainloop()
    except Exception as error:
        if ready:
            print(f"Base command display stopped: {error}", file=sys.stderr)
        else:
            print(json.dumps(str(error)), flush=True)
    finally:
        if root is not None:
            try:
                root.destroy()
            except tk.TclError:
                pass


def _demo():
    viewer = BaseCommandVisualizer()
    start = time.monotonic()
    try:
        while True:
            elapsed = time.monotonic() - start
            if not viewer.update(
                vx=0.6 * math.sin(elapsed), vtheta=0.6 * math.sin(0.7 * elapsed), simulation_time=elapsed
            ):
                break
            time.sleep(0.02)
    except KeyboardInterrupt:
        pass
    finally:
        viewer.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--demo", action="store_true", help="Preview synthetic commands without Isaac Sim.")
    mode.add_argument("--window", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.window:
        _run_window()
    else:
        _demo()
