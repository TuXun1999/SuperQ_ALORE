# Standalone FLIR AprilTag pose example

Prepared for Xun by Adam, 18 October 2026. This folder works independently of
the FLIR/Spot repository: it contains capture code, marker detection/pose code,
a small CLI, pinned Python requirements, and the current camera's intrinsics.
It does not need ROS, Spot/Bosdyn, Git, or the original checkout.

## Hardware and calibration scope

The live capture adapter targets the lab's **Blackfly S BFS-PGE-50S5C**, serial
**19476401**, full native **2448 × 2048 BayerRG8** over GigE. It deliberately
rejects another serial/model. For a different FLIR, adapt the camera features,
image format/size and identity checks, and calibrate that camera/lens first.
This is an Ubuntu/Linux Aravis example, not a Windows/PySpin package.

Default marker: **AprilTag tag36h11, ID 7, black square 0.200 m on each side**.
Measure the outside edge of the black square; exclude the white margin.
Set `--tag-id` and `--tag-size-m` to match the actual printed target. Wrong
physical tag size scales the estimated translation incorrectly.

**The included calibration applies only to serial 19476401 with the unchanged
lens/focus/zoom from the 16 October calibration. It does not apply to the
proposed Canon EF-S lens, a different camera, a crop, binning, or resized input.**
The code cannot detect a physical lens/focus change. Recalibrate if any of those
changed; do not simply edit the serial or mark a new calibration accepted.
Moving the whole camera preserves intrinsics, but changes its world transform.

The calibration passed pixel-fit checks (training RMS 0.1883 px); independent
metric position/orientation accuracy has not been established. The exported
JSON preserves the original quality-check flags and source SHA256, omitting
training-image paths and unrelated session data.

## 1. Install dependencies on Ubuntu 24.04 / Python 3.12

Use UV with the distribution Python so the native `python3-gi` binding has the
same Python ABI. Install UV first if it is not already available. Run the system
package command only if you administer the machine or have approval to install
these packages. No installation is performed automatically by this package.

```bash
sudo apt-get update
sudo apt-get install python3-gi gir1.2-aravis-0.8 libgl1
cd flir-marker-standalone
uv venv --python /usr/bin/python3 .venv
uv pip sync --python .venv/bin/python requirements.txt
```

`camera.py` appends `/usr/lib/python3/dist-packages` for GI after loading the
venv's NumPy/OpenCV. Do not mix a UV-downloaded Python with a different system
GI ABI. Requirements pin the versions used in the existing RPM environment.
Aravis 0.8 is a native dependency installed separately; no native binaries are
redistributed in the ZIP.

On RPM, the existing project environment can run this extracted folder without
new installs. Because RPM uses an extracted Aravis runtime, optionally set:

```bash
export FLIR_ARAVIS_RUNTIME=/home/adam/repos/flir-apriltag/data/commissioning/2026-10-14/runtime/root/usr/lib/x86_64-linux-gnu
/home/adam/repos/flir-apriltag/.venv/bin/python run.py --help
```

That environment override is optional and is not needed with system Aravis.
The code itself contains no dependency on that RPM path or repository.

## 2. Discover the camera

Connect the camera through the lab's PoE/GigE network. The host interface must
be able to reach the camera's subnet. Close other applications that own it.
The code does not change host networking, camera IP, focus, or zoom.

```bash
.venv/bin/python run.py --list
```

Copy `device_id` (or the current `address`) for serial 19476401. Do not rely on
an old IP address. Discovery does not acquire images. If discovery is empty,
check PoE, cable, interface/subnet and firewall with the lab's network setup.
Do not force camera ownership or kill another user's process.

## 3. Capture and estimate the marker pose

Before running, confirm the camera/marker are physically ready, the calibrated
optics are unchanged, and the tag size is correct. Replace the camera argument
below with the result from discovery. Choose a new output directory each run.

```bash
.venv/bin/python run.py --live --camera 'DEVICE_ID_FROM_LIST' \
  --calibration calibration.json --tag-family tag36h11 --tag-id 7 \
  --tag-size-m 0.2 --seconds 300 --fps 12 --exposure-us 20000 \
  --preview --out runs/chair-tag7-001
```

Omit `--preview` on a headless machine. Optional `--gain-db` sets manual gain;
otherwise the prior gain value is retained. The requested FPS is a camera cap,
not a guarantee of detector/preview throughput. This compact example processes
full frames and is not the optimized hardware-ROI live demo.

Stop with Ctrl-C, Q/Esc, or closing the preview window. Live runs are bounded
(default 300 s, maximum 3600 s), stop after five seconds without a complete
frame, and stop at a sensor temperature of 70 C. SIGINT/SIGTERM and ordinary
errors enter the settings-restoration path. Unplugging the camera or killing
the process with SIGKILL can prevent restoration; inspect the saved status.
The program never commands Spot, acquires a robot lease, or operates an E-stop.

Camera acquisition temporarily changes exposure/auto modes, pixel format,
full-frame ROI and frame-rate settings. It snapshots original settings and
attempts to restore and verify them on exit. It shares RPM's serial-specific
lock `/tmp/flir-roi-sweep-19476401.lock` and refuses another camera owner.

## 4. Read the result

- `poses.jsonl`: one observation per processed frame, also emitted to stdout.
- `latest-pose.json`: current observation only. Missing frames invalidate it;
  stopping replaces it with `valid: false, reason: STOPPED`. Use `poses.jsonl`
  for the last recorded result after shutdown.
- `status.json`: final frame counts, error and any restoration errors.
- `settings/`: original, temporary capture and restored settings for live runs.
- `calibration.json` and `run.json`: the calibration/settings used by this run.

Use an observation only when `valid` is true. A valid result includes:

```text
tag_id, tag_size_m, corners_px, rms_px, alternative_gap_px,
rvec, tvec_m, distance_m, T_camera_tag, T_tag_camera
```

`T_camera_tag` is a 4×4 homogeneous transform using column vectors:

```python
p_camera = T_camera_tag @ [x_tag, y_tag, z_tag, 1]
```

All translations are **metres**. Camera axes: X right, Y down, Z forward.
Tag origin: center of its decoded black square; X right, Y up, +Z out of the
printed face in the tag's canonical orientation. `rvec` is an axis-angle
rotation vector in **radians**, not Euler angles. `T_tag_camera` is the inverse.
The package estimates camera-relative pose; it contains no world/Spot transform.
Timestamps are host processing time, not synchronized sensor acquisition time.

Detection uses OpenCV's AprilTag corner refinement, then IPPE_SQUARE with the
saved fisheye or pinhole model. It rejects small/degenerate targets, negative
depth, reprojection RMS above 2 px and competing-pose gaps below 0.1 px. A
repeated requested ID is rejected as ambiguous identity. It does not substitute
a stale pose when detection fails. An `AMBIGUOUS / TILT TAG` result means the
square's perspective is insufficient to distinguish planar pose candidates.
Do not disable that gate to present an arbitrary pose as a valid measurement.

## Offline image mode and Python use

Offline mode needs only NumPy/OpenCV. Use a full-resolution image corresponding
to the calibration. No sample lab photographs are included.

```bash
.venv/bin/python run.py --image /path/to/full-resolution.png \
  --tag-id 7 --tag-size-m 0.2 --out runs/offline-001
```

This writes a single observation to `poses.jsonl` and `annotated.png`; as with a
live run, `latest-pose.json` is invalidated on exit.

```python
import cv2
from marker_pose import load_calibration, TagDetector
cal, K, D, model = load_calibration('calibration.json')
detector = TagDetector('tag36h11', 0.2)
gray = cv2.imread('/path/to/full-resolution.png', cv2.IMREAD_GRAYSCALE)
assert gray is not None and [gray.shape[1], gray.shape[0]] == cal['image_size']
poses = detector.detect(gray, K, D, model)
selected = [p for p in poses if p['tag_id'] == 7]
if len(selected) == 1 and selected[0]['valid']:
    T_camera_tag = selected[0]['T_camera_tag']
```

## Troubleshooting and verification limits

- `Namespace Aravis not available`: install the matching Aravis GI package or
  set `FLIR_ARAVIS_RUNTIME` to an existing extracted runtime's library folder.
- GI import/ABI error: recreate the venv with the distribution Python 3.12.
- Camera-owned/lock error: coordinate with its current operator; do not force it.
- Incomplete payloads: the adapter rejects them even if transport says SUCCESS.
  Check network/cable/traffic; do not treat missing pixels as a valid frame.
- No marker: ensure family/ID, lighting, focus, full black border and sufficient
  pixel size; this is an AprilTag example, not ChArUco board tracking.
- Restoration error: read `status.json` and settings JSON before another run.

The capture and pose components are extracted/adapted from Adam's existing RPM
pipeline, including AprilTag-specific corner refinement. This standalone export
received syntax, dependency-reference and ZIP/hash inspection only. No new
camera acquisition, runtime test suite or physical accuracy test was run for
this handoff, as requested. Inspect `PROVENANCE.json` and `MANIFEST.sha256` for
exact source identifiers. No credentials, robot SDK, CAD or raw captures included.
