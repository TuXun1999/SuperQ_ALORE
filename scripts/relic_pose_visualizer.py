"""Dependency-free live browser visualizer for ReLIC Spot joint targets.

The visualizer intentionally renders the kinematic skeleton from the checked-in
Spot URDF instead of a dynamics simulation.  It is therefore safe to run beside
the real robot: it only receives measured and target joint-angle snapshots.
"""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from threading import Lock, Thread
import time
from typing import Iterable
from urllib.parse import urlparse
from xml.etree import ElementTree as ET


def _vector(value: str | None, *, field: str, joint_name: str) -> list[float]:
    """Parse one URDF three-vector, defaulting omitted values to zero."""

    if value is None:
        return [0.0, 0.0, 0.0]
    try:
        parsed = [float(element) for element in value.split()]
    except ValueError as error:
        raise ValueError(f"Invalid {field} for URDF joint {joint_name!r}.") from error
    if len(parsed) != 3:
        raise ValueError(f"URDF joint {joint_name!r} {field} must have three values.")
    return parsed


def load_urdf_kinematic_model(urdf_path: str | Path) -> list[dict[str, object]]:
    """Return the parent-to-child joint tree needed by the browser renderer."""

    resolved_path = Path(urdf_path).expanduser().resolve()
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Spot URDF was not found: {resolved_path}")

    root = ET.parse(resolved_path).getroot()
    if root.tag != "robot":
        raise ValueError(f"Expected a URDF <robot> root in {resolved_path}.")

    joints: list[dict[str, object]] = []
    for joint in root.findall("joint"):
        joint_name = joint.get("name")
        joint_type = joint.get("type")
        parent = joint.find("parent")
        child = joint.find("child")
        if not joint_name or not joint_type or parent is None or child is None:
            raise ValueError("Every URDF joint must have a name, type, parent, and child.")
        parent_link = parent.get("link")
        child_link = child.get("link")
        if not parent_link or not child_link:
            raise ValueError(f"URDF joint {joint_name!r} has an invalid parent or child link.")

        origin = joint.find("origin")
        axis = joint.find("axis")
        joints.append(
            {
                "name": joint_name,
                "type": joint_type,
                "parent": parent_link,
                "child": child_link,
                "xyz": _vector(
                    None if origin is None else origin.get("xyz"),
                    field="origin xyz",
                    joint_name=joint_name,
                ),
                "rpy": _vector(
                    None if origin is None else origin.get("rpy"),
                    field="origin rpy",
                    joint_name=joint_name,
                ),
                "axis": _vector(
                    None if axis is None else axis.get("xyz"),
                    field="axis",
                    joint_name=joint_name,
                )
                if joint_type in {"revolute", "continuous"}
                else [1.0, 0.0, 0.0],
            }
        )
    if not joints:
        raise ValueError(f"Spot URDF contains no joints: {resolved_path}")
    return joints


def _urdf_joint_name(sdk_joint_name: str) -> str:
    """Convert the SDK's native dot-separated names to this URDF's names."""

    if sdk_joint_name.startswith("arm0."):
        return "arm_" + sdk_joint_name.removeprefix("arm0.")
    return sdk_joint_name.replace(".", "_")


class _PoseVisualizerRequestHandler(BaseHTTPRequestHandler):
    """Serve the static page and its latest pose snapshot."""

    visualizer: "ReLICPoseVisualizer"

    def do_GET(self) -> None:  # noqa: N802 - required BaseHTTPRequestHandler name
        path = urlparse(self.path).path
        if path in {"/", "/index.html"}:
            content = self.visualizer.page.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
            return
        if path == "/api/pose":
            content = json.dumps(self.visualizer.snapshot()).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
            return
        self.send_error(404, "Not found")

    def log_message(self, _format: str, *_args: object) -> None:
        """Keep the 5 Hz browser polling endpoint out of the control console."""


_PAGE_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ReLIC Spot Pose Visualizer</title>
  <style>
    :root { color-scheme: dark; font-family: Inter, system-ui, sans-serif; background: #111827; }
    body { max-width: 1440px; margin: 0 auto; padding: 18px; color: #e5e7eb; }
    h1 { margin: 0 0 4px; font-size: 1.45rem; }
    p { margin: 6px 0; color: #cbd5e1; }
    #status { font-weight: 650; } .waiting { color: #fbbf24; } .fresh { color: #6ee7b7; } .stale { color: #fb7185; }
    #poses { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; margin-top: 14px; }
    .card { background: #1f2937; border: 1px solid #374151; border-radius: 10px; overflow: hidden; }
    .card h2 { margin: 0; padding: 10px 14px; font-size: 1rem; background: #273449; }
    canvas { display: block; width: 100%; aspect-ratio: 1 / 0.82; touch-action: none; }
    #help { font-size: .85rem; }
    details { margin-top: 18px; background: #1f2937; border: 1px solid #374151; border-radius: 10px; }
    summary { cursor: pointer; padding: 11px 14px; font-weight: 700; background: #273449; }
    #policy-inputs { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; padding: 14px; }
    .diagnostic-card { border: 1px solid #475569; border-radius: 8px; padding: 10px; background: #111827; }
    .diagnostic-card h3 { margin: 0 0 9px; font-size: .94rem; color: #f8fafc; }
    .diagnostic-card p { margin: 4px 0 8px; font-size: .82rem; color: #94a3b8; }
    .vector { margin: 7px 0; } .vector-name { display: block; color: #c4b5fd; font-weight: 650; font-size: .84rem; }
    .vector-values { display: block; margin-top: 2px; color: #e2e8f0; font: .78rem ui-monospace, SFMono-Regular, Menlo, monospace; overflow-wrap: anywhere; line-height: 1.45; }
    table { width: 100%; border-collapse: collapse; margin-top: 18px; background: #1f2937; border-radius: 10px; overflow: hidden; }
    th, td { padding: 7px 10px; text-align: right; border-bottom: 1px solid #374151; font-variant-numeric: tabular-nums; }
    th:first-child, td:first-child { text-align: left; } th { background: #273449; } tr:last-child td { border-bottom: 0; }
    .current { color: #60a5fa; } .target { color: #fbbf24; } .delta { color: #c4b5fd; }
    @media (max-width: 800px) { #poses, #policy-inputs { grid-template-columns: 1fr; } body { padding: 10px; } }
  </style>
</head>
<body>
  <h1>ReLIC Spot pose visualizer</h1>
  <p id="status" class="waiting">Waiting for the first robot-state and policy-target snapshot…</p>
  <p id="help">Drag either pose to orbit both views. Scroll to zoom. Blue links are legs; orange links are the arm.</p>
  <section id="poses">
    <article class="card"><h2>Current measured joint pose</h2><canvas id="current"></canvas></article>
    <article class="card"><h2>ReLIC policy target pose</h2><canvas id="target"></canvas></article>
  </section>
  <details open>
    <summary>Live ReLIC inputs and output <span id="policy-update"></span></summary>
    <section id="policy-inputs"><p>Waiting for the first ReLIC policy input…</p></section>
  </details>
  <table>
    <thead><tr><th>SDK joint</th><th class="current">Current (°)</th><th class="target">Target (°)</th><th class="delta">Δ target-current (°)</th></tr></thead>
    <tbody id="joint-rows"></tbody>
  </table>
<script>
const MODEL = __KINEMATIC_MODEL__;
let latest = null;
const view = { yaw: -0.72, pitch: 0.44, zoom: 360 };

function matrixMultiply(a, b) {
  return [[0, 0, 0], [0, 0, 0], [0, 0, 0]].map((row, i) => row.map((_, j) =>
    a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j]));
}
function matrixVector(m, v) { return [
  m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
  m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
  m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
]; }
function identity() { return { p: [0, 0, 0], r: [[1, 0, 0], [0, 1, 0], [0, 0, 1]] }; }
function compose(a, b) {
  const translated = matrixVector(a.r, b.p);
  return { p: [a.p[0] + translated[0], a.p[1] + translated[1], a.p[2] + translated[2]], r: matrixMultiply(a.r, b.r) };
}
function rpy(roll, pitch, yaw) {
  const cr = Math.cos(roll), sr = Math.sin(roll), cp = Math.cos(pitch), sp = Math.sin(pitch), cy = Math.cos(yaw), sy = Math.sin(yaw);
  return [[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
          [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
          [-sp, cp * sr, cp * cr]];
}
function axisAngle(axis, angle) {
  const length = Math.hypot(axis[0], axis[1], axis[2]);
  if (length < 1e-12 || Math.abs(angle) < 1e-12) return identity().r;
  const [x, y, z] = axis.map(value => value / length), c = Math.cos(angle), s = Math.sin(angle), t = 1 - c;
  return [[t*x*x+c, t*x*y-s*z, t*x*z+s*y], [t*x*y+s*z, t*y*y+c, t*y*z-s*x], [t*x*z-s*y, t*y*z+s*x, t*z*z+c]];
}
function kinematics(angles) {
  const links = { body: identity() }, points = {};
  let pending = [...MODEL];
  while (pending.length) {
    const next = [];
    let progressed = false;
    for (const joint of pending) {
      const parent = links[joint.parent];
      if (!parent) { next.push(joint); continue; }
      const base = compose(parent, { p: joint.xyz, r: rpy(...joint.rpy) });
      points[joint.name] = base.p;
      const angle = (joint.type === 'revolute' || joint.type === 'continuous') ? (angles[joint.name] || 0) : 0;
      links[joint.child] = compose(base, { p: [0, 0, 0], r: axisAngle(joint.axis, angle) });
      progressed = true;
    }
    if (!progressed) throw new Error('URDF joint tree is disconnected.');
    pending = next;
  }
  return { links, points };
}
function projected(point, width, height) {
  const cy = Math.cos(view.yaw), sy = Math.sin(view.yaw), cp = Math.cos(view.pitch), sp = Math.sin(view.pitch);
  const x = cy * point[0] - sy * point[1], y = sy * point[0] + cy * point[1], z = point[2];
  return { x: width / 2 + view.zoom * x, y: height / 2 - view.zoom * (cp * z - sp * y), depth: sp * z + cp * y };
}
function line(context, a, b, color, width) {
  context.strokeStyle = color; context.lineWidth = width; context.beginPath(); context.moveTo(a.x, a.y); context.lineTo(b.x, b.y); context.stroke();
}
function drawBody(context, width, height) {
  const half = [[-.43, -.17, -.13], [.43, -.17, -.13], [.43, .17, -.13], [-.43, .17, -.13],
                [-.43, -.17, .13], [.43, -.17, .13], [.43, .17, .13], [-.43, .17, .13]].map(point => projected(point, width, height));
  const edges = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]];
  for (const [first, second] of edges) line(context, half[first], half[second], '#e5e7eb', 2);
}
function drawPose(canvas, sdkAngles) {
  const ratio = window.devicePixelRatio || 1, bounds = canvas.getBoundingClientRect();
  const width = Math.max(1, Math.round(bounds.width * ratio)), height = Math.max(1, Math.round(bounds.height * ratio));
  if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; }
  const context = canvas.getContext('2d'); context.setTransform(ratio, 0, 0, ratio, 0, 0);
  const cssWidth = bounds.width, cssHeight = bounds.height;
  context.fillStyle = '#111827'; context.fillRect(0, 0, cssWidth, cssHeight);
  const angles = {};
  for (const [sdkName, degrees] of Object.entries(sdkAngles || {})) angles[sdkName.replace(/^arm0\./, 'arm_').replaceAll('.', '_')] = degrees * Math.PI / 180;
  const state = kinematics(angles);
  drawBody(context, cssWidth, cssHeight);
  const segments = [];
  for (const joint of MODEL) {
    const parent = state.links[joint.parent], point = state.points[joint.name];
    if (!parent || !point) continue;
    const first = projected(parent.p, cssWidth, cssHeight), second = projected(point, cssWidth, cssHeight);
    segments.push({ first, second, arm: joint.name.startsWith('arm_'), depth: (first.depth + second.depth) / 2 });
  }
  segments.sort((a, b) => a.depth - b.depth);
  for (const segment of segments) line(context, segment.first, segment.second, segment.arm ? '#f59e0b' : '#60a5fa', segment.arm ? 6 : 7);
  for (const joint of MODEL) {
    const point = state.points[joint.name]; if (!point) continue;
    const pixel = projected(point, cssWidth, cssHeight); context.fillStyle = joint.name.startsWith('arm_') ? '#fde68a' : '#bfdbfe';
    context.beginPath(); context.arc(pixel.x, pixel.y, 3.5, 0, 2 * Math.PI); context.fill();
  }
  context.fillStyle = '#cbd5e1'; context.font = '12px system-ui'; context.fillText('body frame fixed; joint angles from live SDK state', 10, cssHeight - 12);
}
function refresh() {
  const current = document.getElementById('current'), target = document.getElementById('target');
  drawPose(current, latest?.current_degrees); drawPose(target, latest?.target_degrees);
}
function rows(data) {
  const body = document.getElementById('joint-rows');
  body.replaceChildren();
  for (const name of data.joint_names) {
    const current = data.current_degrees[name], target = data.target_degrees[name], delta = target - current;
    const row = document.createElement('tr');
    for (const [text, className] of [[name, ''], [current.toFixed(2), 'current'], [target.toFixed(2), 'target'], [delta.toFixed(2), 'delta']]) {
      const cell = document.createElement('td'); cell.textContent = text; cell.className = className; row.appendChild(cell);
    }
    body.appendChild(row);
  }
}
function valuesText(values) {
  if (Array.isArray(values)) return `[${values.map(value => Number(value).toFixed(5)).join(', ')}]`;
  return Object.entries(values).map(([name, value]) => `${name}=${Number(value).toFixed(5)}`).join(', ');
}
function vector(card, name, unit, values) {
  const wrapper = document.createElement('div'); wrapper.className = 'vector';
  const label = document.createElement('span'); label.className = 'vector-name'; label.textContent = `${name} (${unit})`;
  const content = document.createElement('span'); content.className = 'vector-values'; content.textContent = valuesText(values);
  wrapper.append(label, content); card.appendChild(wrapper);
}
function diagnosticCard(title, description, entries) {
  const card = document.createElement('article'); card.className = 'diagnostic-card';
  const heading = document.createElement('h3'); heading.textContent = title; card.appendChild(heading);
  const note = document.createElement('p'); note.textContent = description; card.appendChild(note);
  for (const [name, unit, values] of entries) vector(card, name, unit, values);
  return card;
}
function policyInputs(policy) {
  const container = document.getElementById('policy-inputs'), marker = document.getElementById('policy-update');
  container.replaceChildren();
  if (!policy) { marker.textContent = ''; const text = document.createElement('p'); text.textContent = 'Waiting for the first ReLIC policy input…'; container.appendChild(text); return; }
  marker.textContent = `(policy update ${policy.relic_update_count})`;
  container.append(
    diagnosticCard('High-level command/action (12-D)', 'The zero-agent runner supplies this command; it does not run the high-level ALORE observation/policy.', [
      ['base_velocity', 'vx m/s, vy m/s, yaw-rate rad/s', policy.high_level_action.base_velocity],
      ['arm_joint_delta', 'rad', policy.high_level_action.arm_joint_delta],
      ['raw_base_pose_input', 'pitch rad, height m — ignored by current physical path', policy.high_level_action.raw_base_pose_input],
    ]),
    diagnosticCard('Injected ReLIC command (22-D)', 'Inserted between the base state and joint observations to form the pretrained policy input.', [
      ['arm_joint_target', 'rad', policy.effective_command.arm_joint_target],
      ['leg_joint_command', 'raw ReLIC command', policy.effective_command.leg_joint_command],
      ['base_pose', 'roll rad, pitch rad, height m', policy.effective_command.base_pose],
    ]),
    diagnosticCard('LocomotionPolicyCfg group (59-D)', 'Exact term order in superq_alore_env_cfg.py.', [
      ['base_lin_vel', 'm/s', policy.locomotion_policy_cfg.base_lin_vel],
      ['base_ang_vel', 'rad/s', policy.locomotion_policy_cfg.base_ang_vel],
      ['projected_gravity', 'unit vector', policy.locomotion_policy_cfg.projected_gravity],
      ['joint_pos_rel', 'rad; Isaac/ReLIC 19-joint order', policy.locomotion_policy_cfg.joint_pos_rel],
      ['joint_vel_rel', 'rad/s; Isaac/ReLIC 19-joint order', policy.locomotion_policy_cfg.joint_vel_rel],
      ['actions', 'previous raw low-level leg action', policy.locomotion_policy_cfg.actions],
    ]),
    diagnosticCard('Low-level ReLIC output (12-D)', 'The current raw policy action. The complete input layout is [base 9 | base velocity 3 | command 22 | joint position 19 | joint velocity 19 | last action 12].', [
      ['leg_action', 'raw policy action', policy.low_level_output],
    ]),
  );
}
async function poll() {
  try {
    const response = await fetch('/api/pose', { cache: 'no-store' }); if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json(); latest = data;
    const status = document.getElementById('status');
    if (!data.current_degrees) { status.textContent = 'Waiting for the first robot-state and policy-target snapshot…'; status.className = 'waiting'; }
    else { const age = Date.now() / 1000 - data.updated_at; status.textContent = `Live snapshot: ${data.sequence}  |  age: ${(age * 1000).toFixed(0)} ms`;
      status.className = age > 0.5 ? 'stale' : 'fresh'; rows(data); policyInputs(data.policy_io); }
    refresh();
  } catch (error) { const status = document.getElementById('status'); status.textContent = `Visualizer connection error: ${error.message}`; status.className = 'stale'; }
}
let dragging = null;
for (const canvas of document.querySelectorAll('canvas')) {
  canvas.addEventListener('pointerdown', event => { dragging = { x: event.clientX, y: event.clientY }; canvas.setPointerCapture(event.pointerId); });
  canvas.addEventListener('pointermove', event => { if (!dragging) return; view.yaw += (event.clientX - dragging.x) * .012; view.pitch = Math.max(-1.3, Math.min(1.3, view.pitch + (event.clientY - dragging.y) * .012)); dragging = { x: event.clientX, y: event.clientY }; refresh(); });
  canvas.addEventListener('pointerup', () => { dragging = null; });
  canvas.addEventListener('wheel', event => { event.preventDefault(); view.zoom = Math.max(120, Math.min(900, view.zoom * (event.deltaY > 0 ? .91 : 1.1))); refresh(); }, { passive: false });
}
window.addEventListener('resize', refresh); poll(); setInterval(poll, 200);
</script>
</body>
</html>"""


class ReLICPoseVisualizer:
    """Serve two live, browser-rendered Spot poses from SDK joint snapshots."""

    def __init__(
        self,
        urdf_path: str | Path,
        joint_names: Iterable[str],
        host: str = "127.0.0.1",
        port: int = 8765,
    ) -> None:
        self.joint_names = tuple(joint_names)
        if len(self.joint_names) != 19 or len(set(self.joint_names)) != len(self.joint_names):
            raise ValueError("The pose visualizer requires the 19 unique SDK joint names.")
        if not 0 <= port <= 65535:
            raise ValueError("Visualizer port must be between 0 and 65535.")

        self._model = load_urdf_kinematic_model(urdf_path)
        urdf_names = {joint["name"] for joint in self._model}
        missing = {_urdf_joint_name(name) for name in self.joint_names} - urdf_names
        if missing:
            raise ValueError(f"Spot URDF is missing SDK-controlled joints: {sorted(missing)}")
        self._lock = Lock()
        self._sequence = 0
        self._snapshot: dict[str, object] = {
            "joint_names": self.joint_names,
            "current_degrees": None,
            "target_degrees": None,
            "policy_io": None,
            "updated_at": None,
            "sequence": self._sequence,
        }
        self.page = _PAGE_TEMPLATE.replace(
            "__KINEMATIC_MODEL__", json.dumps(self._model, separators=(",", ":"))
        )
        handler = type(
            "ReLICPoseVisualizerHandler",
            (_PoseVisualizerRequestHandler,),
            {"visualizer": self},
        )
        self._server = ThreadingHTTPServer((host, port), handler)
        self._server.daemon_threads = True
        self._thread = Thread(target=self._server.serve_forever, name="relic-pose-visualizer", daemon=True)
        self._thread.start()
        bound_host, bound_port = self._server.server_address[:2]
        display_host = "127.0.0.1" if bound_host in {"0.0.0.0", "::"} else bound_host
        self.url = f"http://{display_host}:{bound_port}"
        print(f"ReLIC pose visualizer | open {self.url}")

    @staticmethod
    def _named_values(names: Iterable[str], values: Iterable[float], field_name: str) -> dict[str, float]:
        named_values = tuple((str(name), float(value)) for name, value in zip(names, values))
        expected_count = len(tuple(names))
        if len(named_values) != expected_count:
            raise ValueError(f"{field_name} does not contain the expected number of values.")
        return dict(named_values)

    @staticmethod
    def _values(values: Iterable[float], count: int, field_name: str) -> tuple[float, ...]:
        converted = tuple(float(value) for value in values)
        if len(converted) != count:
            raise ValueError(f"{field_name} must contain {count} values; got {len(converted)}.")
        return converted

    def _build_policy_io(
        self,
        high_level_action: Iterable[float],
        policy_env_obs: Iterable[float],
        leg_actions: Iterable[float],
        arm_joint_names: Iterable[str],
        observation_joint_names: Iterable[str],
        leg_action_joint_names: Iterable[str],
        relic_update_count: int | None,
    ) -> dict[str, object]:
        """Split one exact 84-D ReLIC input into its configuration groups."""

        high_level = self._values(high_level_action, 12, "high-level action")
        policy_input = self._values(policy_env_obs, 84, "ReLIC policy input")
        output = self._values(leg_actions, 12, "ReLIC policy output")
        arm_names = tuple(arm_joint_names)
        observation_names = tuple(observation_joint_names)
        action_names = tuple(leg_action_joint_names)
        if len(arm_names) != 7 or len(observation_names) != 19 or len(action_names) != 12:
            raise ValueError("Invalid ReLIC diagnostic joint-name schema.")

        command = policy_input[12:34]
        return {
            "relic_update_count": relic_update_count,
            "high_level_action": {
                "base_velocity": high_level[0:3],
                "arm_joint_delta": self._named_values(arm_names, high_level[3:10], "arm_joint_delta"),
                "raw_base_pose_input": high_level[10:12],
            },
            "effective_command": {
                "arm_joint_target": self._named_values(arm_names, command[0:7], "arm_joint_target"),
                "leg_joint_command": self._named_values(action_names, command[7:19], "leg_joint_command"),
                "base_pose": command[19:22],
            },
            "locomotion_policy_cfg": {
                "base_lin_vel": policy_input[0:3],
                "base_ang_vel": policy_input[3:6],
                "projected_gravity": policy_input[6:9],
                "joint_pos_rel": self._named_values(
                    observation_names, policy_input[34:53], "joint_pos_rel"
                ),
                "joint_vel_rel": self._named_values(
                    observation_names, policy_input[53:72], "joint_vel_rel"
                ),
                "actions": self._named_values(action_names, policy_input[72:84], "actions"),
            },
            "low_level_output": self._named_values(action_names, output, "low_level_output"),
        }

    def update(
        self,
        current_angles_rad: Iterable[float],
        target_angles_rad: Iterable[float],
        *,
        high_level_action: Iterable[float] | None = None,
        policy_env_obs: Iterable[float] | None = None,
        leg_actions: Iterable[float] | None = None,
        arm_joint_names: Iterable[str] | None = None,
        observation_joint_names: Iterable[str] | None = None,
        leg_action_joint_names: Iterable[str] | None = None,
        relic_update_count: int | None = None,
    ) -> None:
        """Publish one measured/target pair and optional exact ReLIC I/O."""

        current = tuple(float(value) for value in current_angles_rad)
        target = tuple(float(value) for value in target_angles_rad)
        if len(current) != len(self.joint_names) or len(target) != len(self.joint_names):
            raise ValueError("Pose visualizer snapshots must contain all 19 SDK joint angles.")
        policy_arguments = (
            high_level_action,
            policy_env_obs,
            leg_actions,
            arm_joint_names,
            observation_joint_names,
            leg_action_joint_names,
        )
        if any(argument is not None for argument in policy_arguments) and any(
            argument is None for argument in policy_arguments
        ):
            raise ValueError("Provide every ReLIC policy-I/O field or none of them.")
        policy_io = (
            None
            if high_level_action is None
            else self._build_policy_io(
                high_level_action,
                policy_env_obs,
                leg_actions,
                arm_joint_names,
                observation_joint_names,
                leg_action_joint_names,
                relic_update_count,
            )
        )
        current_degrees = {name: value * 180.0 / 3.141592653589793 for name, value in zip(self.joint_names, current)}
        target_degrees = {name: value * 180.0 / 3.141592653589793 for name, value in zip(self.joint_names, target)}
        with self._lock:
            self._sequence += 1
            self._snapshot = {
                "joint_names": self.joint_names,
                "current_degrees": current_degrees,
                "target_degrees": target_degrees,
                "policy_io": policy_io,
                "updated_at": time.time(),
                "sequence": self._sequence,
            }

    def snapshot(self) -> dict[str, object]:
        """Return a detached serializable copy for the request handler."""

        with self._lock:
            return dict(self._snapshot)

    def close(self) -> None:
        """Stop serving the dashboard; safe to call repeatedly."""

        with self._lock:
            server = self._server
            self._server = None
        if server is None:
            return
        server.shutdown()
        server.server_close()
        self._thread.join(timeout=1.0)
