# Copyright (c) 2026 FF EAI Robotics Ecosystem Inc. All rights reserved. See LICENSE-FF.txt.
"""URDF forward kinematics for the AgiBot G2 (pure numpy, no pinocchio).

The G2 moves its cameras: the head camera rides on the waist (5) + head (3)
chain and each wrist camera on waist + arm (7). Every observation therefore
needs FK for three camera frames and two tool frames, which this module
provides from a URDF with nothing but ``xml.etree`` and numpy.

Two robot descriptions are shipped under ``description/`` and the station
profile picks one (``model.fk_urdf``):

* ``real/G2_t2v2_crsB_omnipicker.urdf`` -- the identified URDF active on the
  robot (``/data/parameters/identify/urdf/G2_t2v2_crsB/``, backed up
  2026-09-18). Validated here against the GDK's own TF tree and
  ``motion_control_status.frame_poses`` from that backup (``tests/test_g2_kinematics.py``).
* ``geniesim/G2_omnipicker_sim.urdf`` -- the GenieSim 3.2 ``G2_omnipicker``
  as Isaac simulates it. Generated from the joint frames of
  ``robot/G2_omnipicker/robot_fix.usda`` because neither URDF GenieSim ships
  matches its own USD (one has the gripper base 5.4 mm short, the other a
  1 cm longer flange and a 7 mm shorter centre link). Validated against the
  flattened link poses authored in the same USD.

Conventions:

* All poses are ``T_base_from_link`` 4x4 homogeneous matrices, ``base`` being
  the station's ``base_frame`` (``base_link`` on the G2).
* ``mimic`` joints follow their leader when the leader is in the joint map and
  are otherwise held at zero; no arm, tool or camera chain passes through one
  on the G2, so they only matter for finger links.
* Camera poses are returned in the OpenCV optical convention (+X right, +Y
  down, +Z forward) by right-multiplying the configured
  ``T_parent_from_camera`` with the profile's ``optical_correction``. Which
  correction applies is a profile flag (``frame_convention``): the real
  robot's calibrated extrinsics are already optical (``identity``); Isaac
  camera prims look down -Z with +Y up (``opengl`` -> ``diag(1,-1,-1,1)``).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any
import xml.etree.ElementTree as ET

import numpy as np

from cap_harness.g2.codec import ARM_DOF, ARMS, CAMERAS, HEAD_DOF, WAIST_DOF

if TYPE_CHECKING:
    from cap_harness.g2.codec import G2Request
    from cap_harness.g2.config import G2StationConfig

DESCRIPTION_ROOT = Path(__file__).resolve().parent / "description"
#: Packaged descriptions by role; profiles may also name any absolute path.
REAL_URDF = DESCRIPTION_ROOT / "real" / "G2_t2v2_crsB_omnipicker.urdf"
GENIESIM_URDF = DESCRIPTION_ROOT / "geniesim" / "G2_omnipicker_sim.urdf"

_MOVING_TYPES = frozenset({"revolute", "continuous", "prismatic"})
_SUPPORTED_TYPES = _MOVING_TYPES | {"fixed"}

#: Right-multiplied onto a camera link pose to reach the OpenCV optical frame.
OPTICAL_CORRECTIONS: Mapping[str, np.ndarray] = {
    # Already +X right / +Y down / +Z forward (calibrated RGBD extrinsics).
    "optical": np.eye(4),
    # Isaac / OpenGL camera: +X right, +Y up, -Z forward.
    "opengl": np.diag([1.0, -1.0, -1.0, 1.0]),
    # ROS body frame: +X forward, +Y left, +Z up.
    "ros": np.array(
        [[0.0, 0.0, 1.0, 0.0], [-1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    ),
}


# -- small SE(3) helpers -------------------------------------------------------


def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """URDF ``rpy`` to a rotation matrix: ``Rz(yaw) @ Ry(pitch) @ Rx(roll)``."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def matrix_to_rpy(rotation: Any) -> tuple[float, float, float]:
    """Inverse of :func:`rpy_to_matrix` (URDF convention)."""
    r = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    pitch = math.atan2(-r[2, 0], math.hypot(r[0, 0], r[1, 0]))
    if abs(math.cos(pitch)) < 1e-9:
        roll = math.atan2(r[0, 1], r[1, 1]) if pitch < 0 else math.atan2(-r[0, 1], r[1, 1])
        yaw = 0.0
    else:
        roll = math.atan2(r[2, 1], r[2, 2])
        yaw = math.atan2(r[1, 0], r[0, 0])
    return roll, pitch, yaw


def axis_angle_matrix(axis: Any, angle: float) -> np.ndarray:
    """Rodrigues rotation about ``axis`` by ``angle`` radians."""
    a = np.asarray(axis, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(a))
    if norm < 1e-12:
        raise ValueError("joint axis must be non-zero")
    a = a / norm
    k = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
    return np.eye(3) + math.sin(angle) * k + (1.0 - math.cos(angle)) * (k @ k)


def transform(rotation: Any = None, translation: Any = None) -> np.ndarray:
    """Homogeneous 4x4 from an optional 3x3 rotation and 3-vector translation."""
    t = np.eye(4, dtype=np.float64)
    if rotation is not None:
        t[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    if translation is not None:
        t[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return t


def invert(t: Any) -> np.ndarray:
    """Inverse of a rigid transform without a general matrix inverse."""
    m = np.asarray(t, dtype=np.float64).reshape(4, 4)
    r = m[:3, :3].T
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = r
    out[:3, 3] = -r @ m[:3, 3]
    return out


def quat_wxyz_to_matrix(quat: Any) -> np.ndarray:
    """Unit quaternion (w, x, y, z) to a rotation matrix."""
    q = np.asarray(quat, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(q))
    if norm < 1e-12:
        raise ValueError("quaternion must be non-zero")
    w, x, y, z = q / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def pose_from_xyz_quat_xyzw(xyz: Any, quat_xyzw: Any) -> np.ndarray:
    """The GDK / ROS TF convention (translation + xyzw quaternion) to a 4x4."""
    q = np.asarray(quat_xyzw, dtype=np.float64).reshape(4)
    return transform(quat_wxyz_to_matrix([q[3], q[0], q[1], q[2]]), xyz)


def rotation_angle_deg(a: Any, b: Any) -> float:
    """Geodesic angle between two rotations (or 4x4 poses), in degrees."""
    ra = np.asarray(a, dtype=np.float64)[:3, :3]
    rb = np.asarray(b, dtype=np.float64)[:3, :3]
    cosine = (np.trace(ra.T @ rb) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, float(cosine)))))


# -- URDF model ----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class UrdfJoint:
    name: str
    type: str
    parent: str
    child: str
    origin: np.ndarray
    """``T_parent_from_joint`` (the URDF ``<origin>``)."""

    axis: np.ndarray
    lower: float | None
    upper: float | None
    mimic: str | None
    mimic_multiplier: float
    mimic_offset: float

    @property
    def moves(self) -> bool:
        return self.type in _MOVING_TYPES

    def motion(self, value: float) -> np.ndarray:
        """``T_joint_from_child`` for joint value ``value``."""
        if self.type in ("revolute", "continuous"):
            return transform(axis_angle_matrix(self.axis, value))
        if self.type == "prismatic":
            return transform(None, self.axis * value)
        return np.eye(4, dtype=np.float64)


def _floats(text: str | None, default: Sequence[float]) -> np.ndarray:
    if text is None or not text.strip():
        return np.asarray(default, dtype=np.float64)
    return np.asarray([float(item) for item in text.split()], dtype=np.float64)


class UrdfModel:
    """A parsed URDF kinematic tree with FK to any link.

    Only what FK needs is kept: joint origins, axes, types, limits and mimic
    relations. Visuals, collisions, inertials, ``loop_joint`` and ``transmission``
    elements are ignored.
    """

    def __init__(self, root: ET.Element, *, source: Path | None = None) -> None:
        self.source = source
        self.name = str(root.get("name", ""))
        joints: dict[str, UrdfJoint] = {}
        parent_joint: dict[str, str] = {}
        links = {str(link.get("name")) for link in root.findall("link")}
        for element in root.findall("joint"):
            name = str(element.get("name"))
            joint_type = str(element.get("type", "fixed"))
            if joint_type not in _SUPPORTED_TYPES:
                raise ValueError(f"joint {name!r} has unsupported type {joint_type!r}")
            origin = element.find("origin")
            xyz = _floats(None if origin is None else origin.get("xyz"), (0.0, 0.0, 0.0))
            rpy = _floats(None if origin is None else origin.get("rpy"), (0.0, 0.0, 0.0))
            axis_element = element.find("axis")
            axis = _floats(
                None if axis_element is None else axis_element.get("xyz"), (1.0, 0.0, 0.0)
            )
            if joint_type in _MOVING_TYPES and float(np.linalg.norm(axis)) < 1e-12:
                raise ValueError(f"joint {name!r} has a zero axis")
            parent = element.find("parent")
            child = element.find("child")
            if parent is None or child is None:
                raise ValueError(f"joint {name!r} lacks parent/child links")
            limit = element.find("limit")
            mimic = element.find("mimic")
            joint = UrdfJoint(
                name=name,
                type=joint_type,
                parent=str(parent.get("link")),
                child=str(child.get("link")),
                origin=transform(rpy_to_matrix(*rpy), xyz),
                axis=axis,
                lower=None
                if limit is None or limit.get("lower") is None
                else float(limit.get("lower")),
                upper=None
                if limit is None or limit.get("upper") is None
                else float(limit.get("upper")),
                mimic=None if mimic is None else str(mimic.get("joint")),
                mimic_multiplier=1.0 if mimic is None else float(mimic.get("multiplier", 1.0)),
                mimic_offset=0.0 if mimic is None else float(mimic.get("offset", 0.0)),
            )
            if joint.child in parent_joint:
                raise ValueError(f"link {joint.child!r} has two parent joints")
            joints[name] = joint
            parent_joint[joint.child] = name
            links.add(joint.parent)
            links.add(joint.child)
        self.joints: Mapping[str, UrdfJoint] = joints
        self.links: frozenset[str] = frozenset(links)
        self._parent_joint = parent_joint
        roots = [link for link in links if link not in parent_joint]
        if len(roots) != 1:
            raise ValueError(f"URDF must have exactly one root link, found {sorted(roots)}")
        self.root_link: str = roots[0]
        self._chains: dict[str, tuple[str, ...]] = {}

    @classmethod
    def from_file(cls, path: str | Path) -> UrdfModel:
        source = Path(path).expanduser().resolve()
        return cls(ET.parse(source).getroot(), source=source)

    @classmethod
    def from_string(cls, text: str) -> UrdfModel:
        return cls(ET.fromstring(text))

    @property
    def moving_joint_names(self) -> tuple[str, ...]:
        return tuple(
            name for name, joint in self.joints.items() if joint.moves and joint.mimic is None
        )

    def chain(self, link: str) -> tuple[str, ...]:
        """Joint names from the root link down to ``link``."""
        cached = self._chains.get(link)
        if cached is not None:
            return cached
        if link not in self.links:
            raise KeyError(f"unknown link {link!r}")
        names: list[str] = []
        current = link
        while current in self._parent_joint:
            joint_name = self._parent_joint[current]
            names.append(joint_name)
            current = self.joints[joint_name].parent
        result = tuple(reversed(names))
        self._chains[link] = result
        return result

    def joint_value(self, joint: UrdfJoint, positions: Mapping[str, float]) -> float:
        if joint.mimic is not None:
            leader = positions.get(joint.mimic)
            return (
                0.0
                if leader is None
                else joint.mimic_multiplier * float(leader) + joint.mimic_offset
            )
        return float(positions.get(joint.name, 0.0))

    def fk(self, link: str, positions: Mapping[str, float], base: str | None = None) -> np.ndarray:
        """``T_base_from_link``; ``base`` defaults to the root link."""
        pose = np.eye(4, dtype=np.float64)
        for joint_name in self.chain(link):
            joint = self.joints[joint_name]
            pose = pose @ joint.origin
            if joint.moves:
                pose = pose @ joint.motion(self.joint_value(joint, positions))
        if base is None or base == self.root_link:
            return pose
        return invert(self.fk(base, positions)) @ pose

    def joint_limits(self, names: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        lower = []
        upper = []
        for name in names:
            joint = self.joints[name]
            lower.append(-np.inf if joint.lower is None else joint.lower)
            upper.append(np.inf if joint.upper is None else joint.upper)
        return np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)


@lru_cache(maxsize=8)
def load_urdf(path: str) -> UrdfModel:
    """Parse once per path; the model is immutable apart from its chain cache."""
    return UrdfModel.from_file(path)


def resolve_description(path: str | Path) -> Path:
    """Absolute path of a description: absolute as given, else under ``description/``."""
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (DESCRIPTION_ROOT / candidate).resolve()


# -- G2 binding ----------------------------------------------------------------


class G2Kinematics:
    """FK bound to a station profile: joint names, tool frames and camera mounts.

    Joint vectors follow the profile: ``waist`` (5), ``head`` (3) and per-arm
    ``joints`` (7) in ``arms.<side>.joint_names`` order. Anything not supplied
    is held at the profile's ``hold`` pose for the waist/head and at zero for
    the fingers, which is exactly what the planner locks.
    """

    def __init__(self, station: G2StationConfig, model: UrdfModel) -> None:
        self.station = station
        self.model = model
        self.base_frame = str(station.base_frame)
        if self.base_frame not in model.links:
            raise ValueError(f"base_frame {self.base_frame!r} is not a link of {model.source}")
        for side in ARMS:
            arm = station.arms[side]
            for link in (arm.tool_frame, arm.end_link):
                if link not in model.links:
                    raise ValueError(f"arms.{side} names unknown link {link!r} in {model.source}")
            for name in arm.joint_names:
                if name not in model.joints:
                    raise ValueError(f"arms.{side}.joint_names has unknown joint {name!r}")
        for group, names in (
            ("waist", station.waist.joint_names),
            ("head", station.head.joint_names),
        ):
            for name in names:
                if name not in model.joints:
                    raise ValueError(f"{group}.joint_names has unknown joint {name!r}")
        for camera in CAMERAS:
            parent = station.cameras[camera].parent_link
            if parent not in model.links:
                raise ValueError(f"cameras.{camera}.parent_link {parent!r} is not a link")

    @classmethod
    def from_station(cls, station: G2StationConfig) -> G2Kinematics:
        urdf = resolve_description(station.model.fk_urdf)
        if not urdf.is_file():
            raise FileNotFoundError(f"G2 FK URDF not found: {urdf}")
        return cls(station, load_urdf(str(urdf)))

    # -- joint maps --------------------------------------------------------

    def joint_map(
        self,
        *,
        left: Any | None = None,
        right: Any | None = None,
        waist: Any | None = None,
        head: Any | None = None,
    ) -> dict[str, float]:
        station = self.station
        values: dict[str, float] = {}
        waist_values = station.waist.hold if waist is None else _vector(waist, WAIST_DOF, "waist")
        head_values = (
            station.head.hold if head is None else _vector(head, HEAD_DOF, "head", pad=True)
        )
        values.update(zip(station.waist.joint_names, waist_values.tolist(), strict=True))
        values.update(zip(station.head.joint_names, head_values.tolist(), strict=True))
        for side, joints in (("left", left), ("right", right)):
            if joints is None:
                continue
            arm = _vector(joints, ARM_DOF, f"{side} arm")
            values.update(zip(station.arms[side].joint_names, arm.tolist(), strict=True))
        return values

    # -- poses -------------------------------------------------------------

    def link_pose(self, link: str, positions: Mapping[str, float]) -> np.ndarray:
        """``T_base_from_link`` for an explicit joint map (see :meth:`joint_map`)."""
        return self.model.fk(link, positions, base=self.base_frame)

    def tool_pose(
        self, side: str, joints: Any, waist: Any | None = None, head: Any | None = None
    ) -> np.ndarray:
        """``T_base_from_tool`` of ``arms.<side>.tool_frame`` (the planner's frame)."""
        positions = self.joint_map(**{side: joints}, waist=waist, head=head)
        return self.link_pose(self.station.arms[side].tool_frame, positions)

    def end_link_pose(self, side: str, joints: Any, waist: Any | None = None) -> np.ndarray:
        """``T_base_from_end_link`` of ``arms.<side>.end_link`` (the GDK's reported frame)."""
        positions = self.joint_map(**{side: joints}, waist=waist)
        return self.link_pose(self.station.arms[side].end_link, positions)

    def camera_pose(
        self,
        camera: str,
        *,
        left: Any | None = None,
        right: Any | None = None,
        waist: Any | None = None,
        head: Any | None = None,
    ) -> np.ndarray:
        """``T_base_from_optical`` for one profile camera at the given joints."""
        config = self.station.cameras[camera]
        positions = self.joint_map(left=left, right=right, waist=waist, head=head)
        parent = self.link_pose(config.parent_link, positions)
        return parent @ config.T_parent_from_camera @ config.optical_correction

    def head_hold(self, task_name: str | None) -> np.ndarray:
        """The head pose to assume when the wire carries none: the task's init head (``head.task_holds``), else
        ``head.hold``. The manipulation board starts two tasks with another head pitch (GenieSim
        ``robot_init_states.py``: take_wrong_item_shelf G2_STATES_5 0.1745, scoop_popcorn G2_STATES_6 0)."""
        return np.asarray(self.station.head.hold_for(task_name), dtype=np.float64)

    # -- codec hooks -------------------------------------------------------

    def fk_hook(self) -> Callable[[str, Any, Any], np.ndarray]:
        """``fk(side, joints7, waist5) -> T_base_from_tool`` as ``codec.robot_state`` expects."""

        def fk(side: str, joints: Any, waist: Any) -> np.ndarray:
            return self.tool_pose(side, joints, waist)

        return fk

    def camera_fk_hook(self) -> Callable[[str, G2Request], np.ndarray]:
        """``camera_fk(camera, request) -> T_base_from_optical`` as ``codec.camera_observation`` expects.

        Upstream genie_sim sends ``head_joint_states`` empty on G2_omnipicker (the
        leaderboard, 2026-10-25 probe); the head then sits at the task's init pose,
        which the profile's ``head.hold`` must state (GenieSim ``G2_DEFAULT_STATES``
        ``head_state`` = [0, 0, 0.11464] on every instruction task). An empty head
        vector therefore means "use the hold", not zeros: zeros put the head camera
        6.6 deg / 17 mm off and the table 40 mm high.
        """

        def camera_fk(camera: str, request: G2Request) -> np.ndarray:
            head = request.head_joints
            return self.camera_pose(
                camera,
                left=request.left_joints,
                right=request.right_joints,
                waist=request.waist_joints,
                head=head if np.asarray(head).size else self.head_hold(getattr(request, "task_name", None)),
            )

        return camera_fk


def _vector(value: Any, length: int, name: str, *, pad: bool = False) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).reshape(-1)
    if pad and result.shape[0] < length:
        result = np.concatenate([result, np.zeros(length - result.shape[0])])
    if result.shape[0] != length:
        raise ValueError(f"{name} must have {length} values, got {result.shape[0]}")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    return result


__all__ = [
    "DESCRIPTION_ROOT",
    "GENIESIM_URDF",
    "OPTICAL_CORRECTIONS",
    "REAL_URDF",
    "G2Kinematics",
    "UrdfJoint",
    "UrdfModel",
    "axis_angle_matrix",
    "invert",
    "load_urdf",
    "matrix_to_rpy",
    "pose_from_xyz_quat_xyzw",
    "quat_wxyz_to_matrix",
    "resolve_description",
    "rotation_angle_deg",
    "rpy_to_matrix",
    "transform",
]
