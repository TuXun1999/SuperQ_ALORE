"""Model-aware AprilTag pose estimation; camera-from-tag, metres."""
from collections import Counter
import json
from pathlib import Path
import cv2
import numpy as np

FAMILIES={"tag36h11":cv2.aruco.DICT_APRILTAG_36h11,
          "tag25h9":cv2.aruco.DICT_APRILTAG_25h9,
          "tag16h5":cv2.aruco.DICT_APRILTAG_16h5,
          "tag36h10":cv2.aruco.DICT_APRILTAG_36h10}

def positive(value,name):
    if not np.isfinite(value) or value<=0:
        raise ValueError(name+" must be finite and positive")

def tag_points(size):
    positive(size, "tag_size_m")
    h = size / 2
    # Decoded canonical TL, TR, BR, BL, matching IPPE_SQUARE exactly.
    return np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], float)

def make_detector(family="tag36h11"):
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
    return cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(FAMILIES[family]), params)

def project(obj, rv, tv, k, d, model):
    if model == "fisheye":
        return cv2.fisheye.projectPoints(obj.reshape(-1, 1, 3), rv, tv, k, d)[0].reshape(-1, 2)
    return cv2.projectPoints(obj, rv, tv, k, d)[0].reshape(-1, 2)

def load_calibration(path):
    data = json.loads(Path(path).read_text())
    if data.get("accepted") is not True or not data.get("quality_checks") or not all(data["quality_checks"].values()):
        raise ValueError("Use an accepted calibration with all quality checks passed")
    models = {"opencv_fisheye": "fisheye", "opencv_pinhole_radtan": "pinhole"}
    if data.get("model") not in models:
        raise ValueError("Unsupported calibration model")
    k = np.asarray(data["K"], np.float64)
    d = np.asarray(data["distortion"], np.float64)
    size = data["image_size"]
    if k.shape != (3, 3) or not np.isfinite(k).all() or not np.isfinite(d).all():
        raise ValueError("Invalid intrinsic matrix or distortion")
    if k[0, 0] <= 0 or k[1, 1] <= 0 or not np.allclose(k[2], [0, 0, 1]):
        raise ValueError("Invalid camera matrix")
    if len(size) != 2 or any(type(v) is not int or v <= 0 for v in size):
        raise ValueError("Invalid calibrated image size")
    if (models[data["model"]] == "fisheye" and d.shape != (4,)) or d.ndim != 1:
        raise ValueError("Fisheye calibration requires exactly four coefficients")
    if models[data["model"]] == "pinhole" and len(d) not in (4, 5, 8, 12, 14):
        raise ValueError("Invalid pinhole coefficient count")
    return data, k, d, models[data["model"]]

def inverse_transform(transform):
    inverse = np.eye(4)
    inverse[:3, :3] = transform[:3, :3].T
    inverse[:3, 3] = -inverse[:3, :3] @ transform[:3, 3]
    return inverse

def estimate_tag_pose(corners, size, k, d, model, max_rms_px=2., min_gap_px=.1, min_side_px=15.):
    xy = np.asarray(corners, np.float64).reshape(4, 2)
    result = {"valid": False, "reason": "INVALID CORNERS", "corners_px": xy.tolist(), "tag_size_m": size}
    if not np.isfinite(xy).all():
        return result
    sides = np.linalg.norm(xy-np.roll(xy, 1, axis=0), axis=1)
    result["min_side_px"] = float(sides.min())
    if sides.min() < min_side_px or not cv2.isContourConvex(xy.astype(np.float32)):
        result["reason"] = "SMALL OR DEGENERATE TAG"
        return result
    obj = tag_points(size)
    pixels = xy.reshape(-1, 1, 2)
    normalized = (cv2.fisheye.undistortPoints(pixels, k, d) if model == "fisheye"
                  else cv2.undistortPoints(pixels, k, d))
    if not np.isfinite(normalized).all() or np.max(np.abs(normalized)) > 100:
        result["reason"] = "INVALID UNDISTORTED POINTS"
        return result
    try:
        solutions = cv2.solvePnPGeneric(obj, normalized, np.eye(3), None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        candidates = []
        for rv, tv in zip(solutions[1], solutions[2]):
            if not np.isfinite(rv).all() or not np.isfinite(tv).all():
                continue
            rotation = cv2.Rodrigues(rv)[0]
            if np.min((obj @ rotation.T + tv.ravel())[:, 2]) <= .001:
                continue
            rms = float(np.sqrt(np.mean(np.sum((project(obj, rv, tv, k, d, model)-xy)**2, axis=1))))
            if np.isfinite(rms):
                candidates.append((rms, rv, tv))
        candidates.sort(key=lambda item: item[0])
        if not candidates:
            result["reason"] = "NO FORWARD POSE"
            return result
        rms, rv, tv = candidates[0]
        gap = candidates[1][0]-rms if len(candidates) > 1 else None
        result.update(rms_px=rms, alternative_gap_px=gap)
        if rms > max_rms_px:
            result["reason"] = "HIGH REPROJECTION ERROR"
            return result
        if gap is not None and gap < min_gap_px:
            result["reason"] = "AMBIGUOUS / TILT TAG"
            return result
        rr, tt = cv2.solvePnPRefineLM(obj, normalized, np.eye(3), None, rv.copy(), tv.copy())
        refined = float(np.sqrt(np.mean(np.sum((project(obj, rr, tt, k, d, model)-xy)**2, axis=1))))
        if np.isfinite(refined) and refined <= rms and np.min((obj @ cv2.Rodrigues(rr)[0].T+tt.ravel())[:, 2]) > .001:
            rv, tv, rms = rr, tt, refined
    except cv2.error:
        result["reason"] = "POSE SOLVER REJECTED TAG"
        return result
    transform = np.eye(4)
    transform[:3, :3] = cv2.Rodrigues(rv)[0]
    transform[:3, 3] = tv.ravel()
    result.update(valid=True, reason="TRACKING", rms_px=rms, rvec=rv.ravel().tolist(),
                  tvec_m=tv.ravel().tolist(), distance_m=float(np.linalg.norm(tv)),
                  T_camera_tag=transform.tolist(), T_tag_camera=inverse_transform(transform).tolist())
    return result

class TagDetector:
    def __init__(self, family, size):
        if family not in FAMILIES:
            raise ValueError("Unknown AprilTag family")
        if not np.isfinite(size) or size <= 0:
            raise ValueError("Tag size must be a finite positive length in meters")
        self.family, self.size = family, size
        self.detector = make_detector(family)

    def detect(self, gray, k, d, model, max_rms_px=2., min_gap_px=.1):
        corners, ids, _ = self.detector.detectMarkers(gray)
        if ids is None:
            return []
        pairs = sorted(zip(ids.ravel(), corners), key=lambda pair: (int(pair[0]), float(pair[1][0, 0, 0])))
        counts = Counter(int(i) for i in ids.ravel())
        seen = Counter()
        tags = []
        for marker_id, quad in pairs:
            marker_id = int(marker_id)
            pose = estimate_tag_pose(quad, self.size, k, d, model, max_rms_px, min_gap_px)
            pose.update(tag_id=marker_id, family=self.family, duplicate_id=counts[marker_id] > 1,
                        instance_index=seen[marker_id])
            seen[marker_id] += 1
            tags.append(pose)
        return tags
