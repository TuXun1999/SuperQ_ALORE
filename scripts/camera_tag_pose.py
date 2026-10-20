"""Read FLIR latest-pose.json without importing Aravis, OpenCV, or the Spot SDK.

The writer atomically replaces one JSON record per processed frame. This reader
opens the path again on every read, never reusing a stale/invalid cached pose.
time_utc is HOST PROCESSING time, not exposure time. If the file is shared from
another machine, synchronize the hosts' clocks before using freshness checks.
"""

from dataclasses import dataclass
from datetime import datetime
import json
import math
from pathlib import Path
import time

import numpy as np


class CameraPoseUnavailable(RuntimeError):
    """No current, usable measurement; callers must not keep using an old pose."""


@dataclass(frozen=True)
class CameraTagPose:
    """Tag pose in optical camera axes: X right, Y down, Z forward.

    T_camera_tag maps tag points into the camera frame; translations are metres.
    These are the FLIR detector's tag axes, not automatically the SDK fiducial
    axes, the object origin, or a gravity-aligned world frame.
    """

    tag_id: int
    tag_size_m: float
    family: str
    T_camera_tag: np.ndarray
    time_utc: str
    age_s: float  # Age AT READ TIME; retaining this object does not refresh it.
    frame_id: int | None


class CameraTagPoseReader:
    """Read one selected tag from the camera's live JSON file, not poses.jsonl."""

    def __init__(self, path, tag_id, max_age_s=0.5, tag_family="tag36h11"):
        if isinstance(tag_id, bool) or not isinstance(tag_id, int) or tag_id < 0:
            raise ValueError("tag_id must be a nonnegative integer.")
        if not math.isfinite(max_age_s) or max_age_s <= 0:
            raise ValueError("max_age_s must be finite and positive.")
        self.path = Path(path).expanduser()
        self.tag_id = tag_id
        self.max_age_s = float(max_age_s)
        self.tag_family = tag_family

    def read(self):
        """Return CameraTagPose, or raise CameraPoseUnavailable without fallback."""
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError) as exc:
            raise CameraPoseUnavailable(f"Cannot read camera pose {self.path}: {exc}") from exc
        if not isinstance(data, dict):
            raise CameraPoseUnavailable("Expected a single JSON pose object.")
        if data.get("valid") is not True:
            raise CameraPoseUnavailable(f"Camera pose invalid: {data.get('reason', 'valid flag missing')}")
        if type(data.get("tag_id")) is not int or data["tag_id"] != self.tag_id:
            raise CameraPoseUnavailable(f"Expected camera tag ID {self.tag_id}, got {data.get('tag_id')}.")
        if data.get("family") != self.tag_family or data.get("duplicate_id", False):
            raise CameraPoseUnavailable("Wrong tag family or duplicate tag identity.")
        try:
            timestamp = datetime.fromisoformat(data["time_utc"].replace("Z", "+00:00"))
            if timestamp.utcoffset() is None:
                raise ValueError("time_utc must include a timezone")
            acquired = timestamp.timestamp()
            tag_size = float(data["tag_size_m"])
            transform = np.array(data["T_camera_tag"], dtype=np.float64)
        except (KeyError, TypeError, ValueError, OverflowError, AttributeError) as exc:
            raise CameraPoseUnavailable(f"Malformed camera pose: {exc}") from exc
        age = time.time() - acquired
        if not math.isfinite(age) or age < 0 or age > self.max_age_s:
            raise CameraPoseUnavailable(f"Camera pose timestamp is stale or in the future (age={age:.3f}s).")
        if not math.isfinite(tag_size) or tag_size <= 0:
            raise CameraPoseUnavailable("Invalid physical tag size.")
        if (transform.shape != (4, 4) or not np.isfinite(transform).all()
                or not np.allclose(transform[3], [0, 0, 0, 1], rtol=0, atol=1e-6)):
            raise CameraPoseUnavailable("T_camera_tag must be a finite 4x4 homogeneous transform.")
        rotation = transform[:3, :3]
        if (not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0, atol=1e-5)
                or not np.isclose(np.linalg.det(rotation), 1.0, rtol=0, atol=1e-5)):
            raise CameraPoseUnavailable("T_camera_tag does not contain a valid rotation matrix.")
        transform.setflags(write=False)
        return CameraTagPose(
            tag_id=self.tag_id, tag_size_m=tag_size, family=self.tag_family,
            T_camera_tag=transform, time_utc=data["time_utc"], age_s=age,
            frame_id=data.get("frame_id"),
        )
