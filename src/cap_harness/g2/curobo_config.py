# Copyright (c) 2026 FF EAI Robotics Ecosystem Inc. All rights reserved. See LICENSE-FF.txt.
"""Convert GenieSim's v1 cuRobo description of the G2 to the service's v2 form.

GenieSim ships ``G2_omnipicker_fixed_dual.yml`` for cuRobo v1 with USD
kinematics: 34 c-space joints (waist 5, head 3, arms 14, fingers 12), inline
collision spheres, a self-collision ignore map and ``lock_joints`` that pin
everything but the arms. The cuRobo V2 service (``providers/curobo/service.py``)
needs the same description in v2 form, on a URDF, with exactly the 14 arm
joints free so a request's ``joint_names`` equal the planner's.

This module is pure Python (yaml + pathlib) so the conversion is testable on a
CPU-only box; the service wraps :func:`write_g2_curobo_config` in an
``lru_cache`` the way it does for the YAM station. The steps mirror
``_yam_real_robot_config``; each is load-bearing for the v2 loader:

* the USD keys go (``use_usd_kinematics``, ``usd_path``, ...): v2 reads the URDF;
* ``ee_link`` + ``link_names`` become ``tool_frames`` with *both* centre links,
  or one arm has no tool frame and every goal for it is unsolvable;
* ``retract_config`` becomes ``default_joint_position``;
* ``format_version: 2.0`` and an absolute ``urdf_path``;
* ``mesh_link_names`` is dropped: the URDF's ``${mesh_dir}`` FBX references are
  not shipped and collision checking uses the spheres, not meshes;
* ``lock_joints`` is rewritten with the configured waist/head values (default:
  the station's hold pose) and the finger values GenieSim locks, leaving the 14
  arm joints, in ``[l1..l7, r1..r7]`` order, as the only free ones;
* the URDF's ``mimic`` joints are dropped from the cspace (every per-joint
  array) and from the locks. Verified on the v2 loader (cuRobo a35a708): it
  refuses a mimic joint in ``lock_joints`` ("lock the active joint instead")
  and refuses an unlocked cspace joint that is not in the active tree, so a
  mimic joint cannot be kept either way. The sim URDF mimics the two inner
  ``joint1`` fingers from the outer ones (32-joint cspace, 18 locks); the real
  one mimics ten finger joints from the two inner ``joint1`` (24 and 10).
  A dropped lock whose value disagrees with its leader's through the mimic
  relation is warned about: the planner follows the leader, so the sphere
  model and the lock table would be assuming different finger poses.

The URDF defaults to the sim-matched ``G2_omnipicker_sim.urdf`` (what GenieSim
actually simulates; see ``kinematics.py``); ``CAP_HARNESS_CUROBO_G2_URDF`` can
point the same spheres at the identified real-robot URDF, whose arm is the
longer crsB one. The sphere set was tuned for the sim arm, so that pairing is
a stopgap until a crsB sphere set exists.
"""

from __future__ import annotations

from collections.abc import Mapping
import json
import os
from pathlib import Path
import tempfile
from typing import TYPE_CHECKING, Any
import warnings

import yaml

from cap_harness.g2.codec import ARMS
from cap_harness.g2.kinematics import (
    DESCRIPTION_ROOT,
    GENIESIM_URDF,
    UrdfJoint,
    load_urdf,
    resolve_description,
)

if TYPE_CHECKING:
    from cap_harness.g2.config import G2StationConfig

DEFAULT_SOURCE_YML = DESCRIPTION_ROOT / "geniesim" / "G2_omnipicker_fixed_dual.yml"
DEFAULT_URDF = GENIESIM_URDF
FORMAT_VERSION = 2.0

#: v1 keys the v2 loader rejects outright (an Isaac asset description).
LEGACY_USD_KEYS = (
    "use_usd_kinematics",
    "usd_path",
    "usd_robot_root",
    "isaac_usd_path",
    "usd_flip_joints",
    "usd_flip_joint_limits",
    "asset_root_path",
)
#: Finger joints GenieSim locks, at its values: the fingers are driven by the
#: gripper command, never planned, and the planner's spheres assume them open.
FINGER_LOCKS: Mapping[str, float] = {
    "idx32_gripper_l_inner_joint3": 0.0,
    "idx33_gripper_l_inner_joint4": 0.349,
    "idx41_gripper_l_outer_joint1": 0.785,
    "idx42_gripper_l_outer_joint3": 0.01,
    "idx43_gripper_l_outer_joint4": -0.35,
    "idx31_gripper_l_inner_joint1": -0.785,
    "idx72_gripper_r_inner_joint3": 0.0,
    "idx73_gripper_r_inner_joint4": 0.349,
    "idx81_gripper_r_outer_joint1": 0.785,
    "idx82_gripper_r_outer_joint3": 0.01,
    "idx83_gripper_r_outer_joint4": -0.35,
    "idx71_gripper_r_inner_joint1": -0.785,
}
ENV_SOURCE = "CAP_HARNESS_CUROBO_G2_CONFIG"
ENV_URDF = "CAP_HARNESS_CUROBO_G2_URDF"
ENV_STATION = "CAP_HARNESS_CUROBO_G2_STATION"
ENV_LOCK_JOINTS = "CAP_HARNESS_CUROBO_G2_LOCK_JOINTS"


def g2_lock_joints(
    station: G2StationConfig | None = None,
    *,
    waist: Any | None = None,
    head: Any | None = None,
    fingers: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """The joints the planner holds fixed and their values.

    Waist and head default to the station's ``hold`` pose (``geniesim-default``
    when no station is given); fingers to :data:`FINGER_LOCKS`. Values are
    plain floats so the mapping hashes and the service can cache on it.
    """
    if station is None:
        from cap_harness.g2.config import load_g2_station_config

        station = load_g2_station_config(os.environ.get(ENV_STATION, "geniesim-default"))
    waist_values = station.waist.hold if waist is None else waist
    head_values = station.head.hold if head is None else head
    locks: dict[str, float] = {}
    locks.update(_named(station.waist.joint_names, waist_values, "waist"))
    locks.update(_named(station.head.joint_names, head_values, "head"))
    locks.update({str(k): float(v) for k, v in (fingers or FINGER_LOCKS).items()})
    return locks


def _named(names: tuple[str, ...], values: Any, what: str) -> dict[str, float]:
    items = [float(v) for v in values]
    if len(items) != len(names):
        raise ValueError(f"{what} lock values must have {len(names)} entries, got {len(items)}")
    return dict(zip(names, items, strict=True))


def mimic_joints(urdf: str | Path) -> dict[str, UrdfJoint]:
    """The URDF's ``mimic`` joints by name; each carries its leader and multiplier."""
    model = load_urdf(str(resolve_description(urdf)))
    return {name: joint for name, joint in model.joints.items() if joint.mimic is not None}


def _drop_mimic_locks(
    locks: Mapping[str, float], mimics: Mapping[str, UrdfJoint]
) -> dict[str, float]:
    """Remove mimic joints from the lock map (v2 refuses them), warning on disagreement."""
    result = dict(locks)
    for name, joint in mimics.items():
        if name not in result:
            continue
        value = result.pop(name)
        leader = result.get(joint.mimic)
        if leader is None:
            continue
        implied = joint.mimic_multiplier * leader + joint.mimic_offset
        if abs(implied - value) > 1e-6:
            warnings.warn(
                f"lock for mimic joint {name!r} ({value:g}) is dropped; the planner follows "
                f"its leader {joint.mimic!r} ({leader:g} -> {implied:g}), so the finger pose "
                "the collision spheres were tuned for is not the one being planned",
                stacklevel=3,
            )
    return result


def build_g2_curobo_config(
    source_yml: str | Path | None = None,
    urdf: str | Path | None = None,
    *,
    lock_joints: Mapping[str, float] | None = None,
    arm_joint_names: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """The v2 ``robot_cfg`` document as a dict (see the module docstring)."""
    source = Path(source_yml) if source_yml is not None else DEFAULT_SOURCE_YML
    if not source.is_file():
        raise FileNotFoundError(f"G2 cuRobo v1 config not found: {source}")
    urdf_path = resolve_description(urdf) if urdf is not None else DEFAULT_URDF
    if not urdf_path.is_file():
        raise FileNotFoundError(f"G2 cuRobo URDF not found: {urdf_path}")

    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    kinematics = config["robot_cfg"]["kinematics"]
    for legacy in LEGACY_USD_KEYS:
        kinematics.pop(legacy, None)

    end_effector = kinematics.pop("ee_link", "gripper_r_center_link")
    link_names = kinematics.pop("link_names", None) or []
    kinematics["tool_frames"] = list(dict.fromkeys([end_effector, *link_names]))
    for side in ARMS:
        frame = f"gripper_{side[0]}_center_link"
        if frame not in kinematics["tool_frames"]:
            kinematics["tool_frames"].append(frame)

    kinematics.pop("mesh_link_names", None)

    cspace = kinematics.get("cspace") or {}
    if "retract_config" in cspace:
        cspace["default_joint_position"] = cspace.pop("retract_config")
    mimics = mimic_joints(urdf_path)
    all_names = [str(name) for name in cspace.get("joint_names", [])]
    keep = [index for index, name in enumerate(all_names) if name not in mimics]
    # Every per-joint array (names, default position, weights) loses the same rows.
    for key, value in list(cspace.items()):
        if isinstance(value, list) and len(value) == len(all_names):
            cspace[key] = [value[index] for index in keep]
    joint_names = [all_names[index] for index in keep]

    locks = dict(lock_joints) if lock_joints is not None else g2_lock_joints()
    locks = _drop_mimic_locks(locks, mimics)
    unknown = sorted(set(locks) - set(joint_names))
    if unknown:
        raise ValueError(f"lock_joints name joints outside the cspace: {unknown}")
    kinematics["lock_joints"] = {name: float(locks[name]) for name in joint_names if name in locks}

    free = tuple(name for name in joint_names if name not in locks)
    expected = arm_joint_names or _default_arm_joint_names()
    if free != tuple(expected):
        raise ValueError(
            f"free joints after locking must be the 14 arm joints in order; got {free}"
        )

    kinematics["format_version"] = FORMAT_VERSION
    kinematics["urdf_path"] = str(urdf_path)
    return config


def _default_arm_joint_names() -> tuple[str, ...]:
    left = tuple(f"idx2{i}_arm_l_joint{i}" for i in range(1, 8))
    right = tuple(f"idx6{i}_arm_r_joint{i}" for i in range(1, 8))
    return left + right


def free_joint_names(config: Mapping[str, Any]) -> tuple[str, ...]:
    """The planner's joint order: cspace names minus the locked ones."""
    kinematics = config["robot_cfg"]["kinematics"]
    locked = set(kinematics.get("lock_joints") or {})
    return tuple(name for name in kinematics["cspace"]["joint_names"] if name not in locked)


def lock_joints_from_env(station: G2StationConfig | None = None) -> dict[str, float]:
    """Service-side lock values: the station's, overridden by ``CAP_HARNESS_CUROBO_G2_LOCK_JOINTS``.

    The override is a JSON object ``{"idx01_body_joint1": -0.83, ...}`` so a
    deployment can pin the waist at the task's initial pose without editing
    the profile.
    """
    locks = g2_lock_joints(station)
    raw = os.environ.get(ENV_LOCK_JOINTS)
    if raw:
        override = json.loads(raw)
        if not isinstance(override, dict):
            raise ValueError(f"{ENV_LOCK_JOINTS} must be a JSON object of joint: value")
        locks.update({str(k): float(v) for k, v in override.items()})
    return locks


def write_g2_curobo_config(
    source_yml: str | Path | None = None,
    urdf: str | Path | None = None,
    *,
    lock_joints: Mapping[str, float] | None = None,
) -> str:
    """Write the converted document to a temp file and return its path."""
    config = build_g2_curobo_config(source_yml, urdf, lock_joints=lock_joints)
    output = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yml", prefix="cap_g2_curobo_", delete=False
    )
    with output:
        yaml.safe_dump(config, output, sort_keys=False)
    return output.name


__all__ = [
    "DEFAULT_SOURCE_YML",
    "DEFAULT_URDF",
    "ENV_LOCK_JOINTS",
    "ENV_SOURCE",
    "ENV_STATION",
    "ENV_URDF",
    "FINGER_LOCKS",
    "FORMAT_VERSION",
    "LEGACY_USD_KEYS",
    "build_g2_curobo_config",
    "free_joint_names",
    "g2_lock_joints",
    "lock_joints_from_env",
    "mimic_joints",
    "write_g2_curobo_config",
]
