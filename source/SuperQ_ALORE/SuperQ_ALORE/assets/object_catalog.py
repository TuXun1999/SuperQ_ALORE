from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from SuperQ_ALORE.assets import ASSET_DIR
from SuperQ_ALORE.assets.spot.constants import SPOT_DEFAULT_JOINT_POS


# define arm joint names in the same order as in the YAML catalog
ARM_JOINT_NAMES_IN_ORDER: tuple[str, ...] = (
    "arm_sh0",
    "arm_sh1",
    "arm_el0",
    "arm_el1",
    "arm_wr0",
    "arm_wr1",
    "arm_f1x",
)

LEG_JOINT_NAMES: tuple[str, ...] = (
    "fl_hx", "fl_hy", "fl_kn",
    "fr_hx", "fr_hy", "fr_kn",
    "hl_hx", "hl_hy", "hl_kn",
    "hr_hx", "hr_hy", "hr_kn",
)

CATALOG_PATH = Path(ASSET_DIR) / "object-models" / "pre_grasping.yaml"


# one immutable class for each (object, object pose, joint configuration) tuple defined in the YAML catalog
@dataclass(frozen=True)
class PoseEntry:
    """One (chair, pose) pair loaded from pre_grasping.yaml."""
    object_id: str
    pose_id: str
    asset_path: str
    # Chair position relative to env origin (x, y, z) — wxyz quaternion
    position: tuple[float, float, float]
    # TODO: ensure that the defined orientation matches with the defintion of RigidObjectCfg in isaaclab (currently assumed to be wxyz)
    orientation: tuple[float, float, float, float]  # wxyz
    # All robot joints: arm joints from YAML, leg joints from SPOT defaults
    joint_positions: dict[str, float]

# immutable class for each object defined in the YAML catalog, containing all its poses
@dataclass(frozen=True)
class ObjectEntry:
    object_id: str
    asset_path: str
    poses: tuple[PoseEntry, ...]


def _resolve_asset_path(raw_path: str) -> str:
    '''takes the raw asset path from the YAML and resolves it to an absolute path on disk'''
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        # paths in the YAML are relative to the workspace root
        workspace_root = Path(ASSET_DIR).parents[3]
        candidate = workspace_root / candidate
    if not candidate.exists():
        raise FileNotFoundError(
            f"Asset path '{raw_path}' does not exist (resolved to '{candidate}')."
        )
    return str(candidate)


def _build_joint_positions(arm_values: list[float]) -> dict[str, float]:
    """Build a full joint-name → value dict: arm joints from YAML, legs from Spot defaults."""
    if len(arm_values) != len(ARM_JOINT_NAMES_IN_ORDER):
        raise ValueError(
            f"Expected {len(ARM_JOINT_NAMES_IN_ORDER)} arm joint values, got {len(arm_values)}."
        )
    joint_positions: dict[str, float] = dict(
        zip(ARM_JOINT_NAMES_IN_ORDER, arm_values, strict=True)
    )
    for leg_joint in LEG_JOINT_NAMES:
        joint_positions[leg_joint] = SPOT_DEFAULT_JOINT_POS[leg_joint]
    return joint_positions


def load_pregrasp_catalog(catalog_path: Path | None = None) -> tuple[ObjectEntry, ...]:

    # read from the YAML file
    catalog_file = catalog_path or CATALOG_PATH
    with catalog_file.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}

    # iterate through all the defined objects (value of the "objects": chair_1, chair_2, etc. in the YAML)
    objects_raw = data.get("objects", {})
    if not objects_raw:
        raise ValueError(f"No objects defined in {catalog_file}.")

    
    object_entries: list[ObjectEntry] = []

    # looks for the `objects` key and starts iterate through chair_1, chair_2 etc.
    for object_id, obj_data in objects_raw.items():

        # resolve the asset path for this object
        raw_path = obj_data.get("asset_path")
        if not raw_path:
            raise ValueError(f"Object '{object_id}' is missing asset_path.")
        asset_path = _resolve_asset_path(raw_path)

        # iterate through all the defined poses for this object (pose_1, pose_2, etc. in the YAML)
        poses_raw = obj_data.get("poses", {})
        if not poses_raw:
            raise ValueError(f"Object '{object_id}' must define at least one pose.")

        # for each pose, process the position, orientation, and arm joint configuration, and create a PoseEntry for it. 
        # Then create an ObjectEntry for this object containing all its poses.
        pose_entries: list[PoseEntry] = []
        for pose_id, pose_data in poses_raw.items():
            pos = pose_data.get("position")
            ori = pose_data.get("orientation")
            jcfg = pose_data.get("joint_configuration")
            if not isinstance(pos, list) or len(pos) != 3:
                raise ValueError(f"Pose '{object_id}/{pose_id}': position must be a 3-element list.")
            if not isinstance(ori, list) or len(ori) != 4:
                raise ValueError(f"Pose '{object_id}/{pose_id}': orientation must be a 4-element list (wxyz).")
            if not isinstance(jcfg, list):
                raise ValueError(f"Pose '{object_id}/{pose_id}': joint_configuration must be a list.")
            pose_entries.append(PoseEntry(
                object_id=object_id,
                pose_id=pose_id,
                asset_path=asset_path,
                position=tuple(float(v) for v in pos),
                orientation=tuple(float(v) for v in ori),
                joint_positions=_build_joint_positions(jcfg),
            ))

        # after processing all poses for this object, create an ObjectEntry 
        # and append it to the list of object entries
        object_entries.append(ObjectEntry(
            object_id=object_id,
            asset_path=asset_path,
            poses=tuple(pose_entries),
        ))

    return tuple(object_entries)


# Module-level singletons — loaded once at import time.
OBJECT_CATALOG: tuple[ObjectEntry, ...] = load_pregrasp_catalog()
OBJECT_IDS: tuple[str, ...] = tuple(e.object_id for e in OBJECT_CATALOG)
POSE_IDS_BY_OBJECT: dict[str, tuple[str, ...]] = {
    e.object_id: tuple(p.pose_id for p in e.poses) for e in OBJECT_CATALOG
}
POSE_ENTRIES_BY_OBJECT: dict[str, tuple[PoseEntry, ...]] = {
    e.object_id: e.poses for e in OBJECT_CATALOG
}


def get_pose_entry(object_id: str, pose_id: str) -> PoseEntry:
    for entry in POSE_ENTRIES_BY_OBJECT[object_id]:
        if entry.pose_id == pose_id:
            return entry
    raise KeyError(f"Pose '{pose_id}' not found for object '{object_id}'.")
