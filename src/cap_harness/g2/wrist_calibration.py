# Copyright (c) 2026 FF EAI Robotics Ecosystem Inc. All rights reserved. See LICENSE-FF.txt.
"""Wrist-camera extrinsic self-check for the G2 GenieSim leaderboard (robust board, ``camposgen``).

The leaderboard wire carries no camera poses (``camera_pose_source: fk``): every camera is placed with FK and its
USD mount. The robust board's ``camposgen`` generator moves the two wrist cameras on their gripper mounts (x, y
+-1 cm, a fixed 1 cm lift, roll / pitch / yaw +-10 deg; upstream ``generalization_utils``
``_apply_camera_position_perturbation``), so FK alone places everything a wrist sees several centimetres off and
the planning scene fuses those points as phantom obstacles. The head camera is never moved.

At the episode's first request (head and both wrists with depth, the arms at their start pose) each wrist's true
pose is estimated against the head and the table:

1. the wrist depth beyond the gripper's self-view is back-projected through the lens model (as the planning scene);
2. the dominant plane within 40 deg of horizontal is the table: it is rotated level about the camera centre and
   moved to the head's table height (two rotations, one translation);
3. points 6 mm - 15 cm above the table (the objects) of head and wrist are rasterised (4 mm); the yaw about the
   vertical through the camera centre (+-20 deg) and the in-plane shift (+-12 cm) that overlap them best are found
   by FFT cross-correlation;
4. a few rounds of point-to-point ICP on objects and table refine all six degrees of freedom.

The result is ``E`` in the wrist's optical frame (``T_base_from_optical_true = T_nominal @ E``); it rides on the
gripper, so it holds for the whole episode. It is applied only when its rotation is at least ``MIN_ROTATION_DEG``
and the fit is sound (enough table and object points, the objects overlap better than with the nominal pose, a small
ICP residual, a plausible size). Otherwise the nominal pose stands, so an unperturbed scene is served exactly as
before. Live estimates (C5i, 922 wrist cameras): camposgen 6.6-14.6 deg (the platform's seed-1 draws 4.5-15.6 deg);
unperturbed scenes <= 0.2 deg, except backgen's room up to 1.2 deg / 13 mm (other structure in the wrist's view),
which the first threshold (0.6 deg or 12 mm) let through on 11 of 68 backgen wrist cameras.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from cap_harness.geometry import undistort_normalized

#: Below this the estimate is noise: <= 0.22 deg on 718 instruction-board wrist frames, <= 1.2 deg in backgen's room;
#: camposgen's mount rotations are >= 4.5 deg on the platform's draws (|r| < 3 deg on ~1.4 % of draws, then ~1 cm at
#: the hover). Translation alone never triggers a correction (backgen's room reached 13 mm).
MIN_ROTATION_DEG = 3.0
#: Above these the estimate is not a mount perturbation (camposgen: <= ~17.3 deg, <= ~17 mm).
MAX_ROTATION_DEG = 25.0
MAX_TRANSLATION_M = 0.06
#: Fit quality gates.
MIN_PLANE_INLIERS = 3000
MIN_OBJECT_POINTS = 150
MIN_OVERLAP = 0.55
MAX_ICP_RMS_M = 0.008

DEPTH_STEP = 4  # back-project every 4th pixel of the 1280 x 1056 wrist depth (~85k points)
HEAD_STEP = 2
SELF_VIEW_M = 0.16  # the wrist sees its own open gripper at 4-13 cm
FAR_M = 1.5
PLANE_TOL_M = 0.004
PLANE_MAX_TILT_DEG = 40.0
BAND_LOW_M = 0.006
BAND_HIGH_M = 0.15
CELL_M = 0.004
GRID_LO = (0.15, -0.8)
GRID_SIZE_M = (1.2, 1.6)
YAW_RANGE_DEG = 20.0
SHIFT_RANGE_M = 0.12
ICP_ROUNDS = 8
ICP_MATCH_M = 0.02
TABLE_SEARCH_M = 0.02


@dataclass(frozen=True)
class WristCorrection:
    """One wrist's estimate; ``correction`` is applied (right-multiplied) only when ``applied``."""

    camera: str
    correction: np.ndarray
    applied: bool
    reason: str
    rotation_deg: float = 0.0
    translation_m: float = 0.0
    info: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        out = {
            "applied": self.applied,
            "reason": self.reason,
            "rotation_deg": round(self.rotation_deg, 3),
            "translation_mm": round(1000.0 * self.translation_m, 2),
        }
        out.update(self.info)
        return out


def _rotation_angle_deg(R: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))))


def _axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    norm = float(np.linalg.norm(axis))
    if norm < 1e-12 or angle == 0.0:
        return np.eye(3)
    k = axis / norm
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def _about(R: np.ndarray, pivot: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = pivot - R @ pivot
    return T


def _apply(T: np.ndarray, P: np.ndarray) -> np.ndarray:
    return P @ T[:3, :3].T + T[:3, 3]


def back_project(
    depth: np.ndarray, K: np.ndarray, distortion: Any, step: int, near: float, far: float
) -> np.ndarray:
    """Optical-frame points of every ``step``-th pixel with ``near < depth < far`` (OpenCV lens model)."""
    d = np.asarray(depth, dtype=np.float64)[::step, ::step]
    h, w = d.shape
    v, u = np.mgrid[0 : h * step : step, 0 : w * step : step]
    ok = np.isfinite(d) & (d > near) & (d < far)
    z = d[ok]
    x = (u[ok] - K[0][2]) / K[0][0]
    y = (v[ok] - K[1][2]) / K[1][1]
    if distortion is not None and any(float(c) != 0.0 for c in distortion):
        x, y = undistort_normalized(x, y, distortion)
    return np.stack([x * z, y * z, z], axis=1)


def fit_table_plane(P: np.ndarray, *, iterations: int = 400, sample: int = 6000, seed: int = 0):
    """RANSAC over a subsample + least squares: (normal up, centroid, inlier mask) of the dominant near-level plane."""
    if len(P) < 100:
        return None
    rng = np.random.default_rng(seed)
    S = P[rng.integers(0, len(P), size=min(sample, len(P)))]
    idx = rng.integers(0, len(S), size=(iterations, 3))
    a, b, c = S[idx[:, 0]], S[idx[:, 1]], S[idx[:, 2]]
    n = np.cross(b - a, c - a)
    norm = np.linalg.norm(n, axis=1)
    ok = norm > 1e-9
    n[ok] /= norm[ok, None]
    n[n[:, 2] < 0] *= -1.0
    ok &= n[:, 2] >= np.cos(np.radians(PLANE_MAX_TILT_DEG))
    if not ok.any():
        return None
    n, a = n[ok], a[ok]
    d = -np.einsum("ij,ij->i", n, a)
    counts = (np.abs(S @ n.T + d) < PLANE_TOL_M).sum(axis=0)
    k = int(np.argmax(counts))
    inliers = np.abs(P @ n[k] + d[k]) < PLANE_TOL_M
    Q = P[inliers]
    centroid = Q.mean(axis=0)
    _, _, vt = np.linalg.svd(Q - centroid, full_matrices=False)
    normal = vt[2] if vt[2][2] >= 0 else -vt[2]
    inliers = np.abs((P - centroid) @ normal) < PLANE_TOL_M
    return normal, centroid, inliers


def _occupancy(P: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    ij = np.floor((P[:, :2] - np.asarray(GRID_LO)) / CELL_M).astype(int)
    ok = (ij[:, 0] >= 0) & (ij[:, 0] < shape[0]) & (ij[:, 1] >= 0) & (ij[:, 1] < shape[1])
    grid = np.zeros(shape)
    grid[ij[ok, 0], ij[ok, 1]] = 1.0
    return grid


def _best_shift(FH: np.ndarray, W: np.ndarray, shape: tuple[int, int]) -> tuple[float, int, int]:
    corr = np.fft.irfft2(FH * np.conj(np.fft.rfft2(W)), s=shape)
    lim = int(SHIFT_RANGE_M / CELL_M)
    window = np.full(corr.shape, -np.inf)
    rows = np.r_[0 : lim + 1, shape[0] - lim : shape[0]]
    cols = np.r_[0 : lim + 1, shape[1] - lim : shape[1]]
    window[np.ix_(rows, cols)] = corr[np.ix_(rows, cols)]
    k = np.unravel_index(int(np.argmax(window)), corr.shape)
    si = int(k[0]) if k[0] <= shape[0] // 2 else int(k[0]) - shape[0]
    sj = int(k[1]) if k[1] <= shape[1] // 2 else int(k[1]) - shape[1]
    return float(window[k]) / max(1.0, float(W.sum())), si, sj


def estimate_wrist_correction(
    *,
    camera: str,
    head_points_base: np.ndarray,
    wrist_points_optical: np.ndarray,
    wrist_pose: np.ndarray,
    table_z_hint: float,
) -> WristCorrection:
    """The wrist's correction ``E`` (optical frame) from head points (base frame) and its own optical-frame points."""
    identity = np.eye(4)
    Ph = np.asarray(head_points_base, dtype=np.float64)
    near_table = Ph[(np.abs(Ph[:, 2] - table_z_hint) < TABLE_SEARCH_M) & (Ph[:, 0] > 0.3)]
    table_z = float(np.median(near_table[:, 2])) if len(near_table) > 100 else float(table_z_hint)
    T_nom = np.asarray(wrist_pose, dtype=np.float64)
    P = _apply(T_nom, np.asarray(wrist_points_optical, dtype=np.float64))
    info: dict[str, Any] = {"table_z": round(table_z, 4), "wrist_points": int(len(P))}
    plane = fit_table_plane(P)
    if plane is None:
        return WristCorrection(camera, identity, False, "no table plane", info=info)
    normal, _, inliers = plane
    info["plane_inliers"] = int(inliers.sum())
    if inliers.sum() < MIN_PLANE_INLIERS:
        return WristCorrection(camera, identity, False, "too few table points", info=info)
    centre = T_nom[:3, 3]
    tilt = float(np.arccos(min(1.0, float(normal[2]))))
    info["plane_tilt_deg"] = round(float(np.degrees(tilt)), 3)
    T1 = _about(_axis_angle(np.cross(normal, [0.0, 0.0, 1.0]), tilt), centre)
    P1 = _apply(T1, P)
    T2 = np.eye(4)
    T2[2, 3] = table_z - float(np.median(P1[inliers][:, 2]))
    level = T2 @ T1

    def band(Q: np.ndarray) -> np.ndarray:
        keep = (
            (Q[:, 2] > table_z + BAND_LOW_M)
            & (Q[:, 2] < table_z + BAND_HIGH_M)
            & (Q[:, 0] > 0.25)
            & (Q[:, 0] < 1.2)
            & (np.abs(Q[:, 1]) < 0.7)
        )
        return Q[keep]

    Oh, Ow = band(Ph), band(_apply(level, P))
    info["object_points_head"], info["object_points_wrist"] = int(len(Oh)), int(len(Ow))
    if len(Oh) < MIN_OBJECT_POINTS or len(Ow) < MIN_OBJECT_POINTS:
        return WristCorrection(camera, identity, False, "too few object points", info=info)
    shape = (int(round(GRID_SIZE_M[0] / CELL_M)), int(round(GRID_SIZE_M[1] / CELL_M)))
    FH = np.fft.rfft2(_occupancy(Oh, shape))
    vertical = np.array([0.0, 0.0, 1.0])
    best: tuple[float, float, int, int] | None = None

    def search(angles: np.ndarray) -> None:
        nonlocal best
        for angle in angles:
            Tr = _about(_axis_angle(vertical, float(np.radians(angle))), centre)
            W = _occupancy(_apply(Tr, Ow), shape)
            if W.sum() < 10:
                continue
            score, si, sj = _best_shift(FH, W, shape)
            if best is None or score > best[0]:
                best = (score, float(angle), si, sj)

    search(np.arange(-YAW_RANGE_DEG, YAW_RANGE_DEG + 1e-9, 1.0))
    if best is None:
        return WristCorrection(camera, identity, False, "no object overlap", info=info)
    search(np.arange(best[1] - 1.0, best[1] + 1.0 + 1e-9, 0.2))
    score, yaw, si, sj = best
    # The nominal pose's own overlap (no levelling, yaw or shift): the correction must beat it.
    W0 = _occupancy(band(P), shape)
    corr0 = float((_occupancy(Oh, shape) * W0).sum()) / max(1.0, float(W0.sum()))
    info.update(
        {
            "yaw_deg": round(yaw, 2),
            "shift_mm": [round(1000.0 * si * CELL_M, 1), round(1000.0 * sj * CELL_M, 1)],
            "overlap": round(score, 3),
            "overlap_nominal": round(corr0, 3),
        }
    )
    Ts = np.eye(4)
    Ts[0, 3], Ts[1, 3] = si * CELL_M, sj * CELL_M
    T_world = Ts @ _about(_axis_angle(vertical, float(np.radians(yaw))), centre) @ level
    # ICP on objects + table, from the coarse estimate (points expressed back in the nominal frame).
    table_head = near_table[:: max(1, len(near_table) // 4000)]
    tree_points = np.vstack([Oh, table_head]) if len(table_head) else Oh
    src_nominal = np.vstack(
        [_apply(np.linalg.inv(level), Ow), P[inliers][:: max(1, int(inliers.sum()) // 4000)]]
    )
    rms = None
    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(tree_points)
        for _ in range(ICP_ROUNDS):
            S = _apply(T_world, src_nominal)
            dist, j = tree.query(S, distance_upper_bound=ICP_MATCH_M)
            ok = np.isfinite(dist)
            if ok.sum() < 50:
                break
            A, B = S[ok], tree_points[j[ok]]
            ca, cb = A.mean(axis=0), B.mean(axis=0)
            U, _, Vt = np.linalg.svd((A - ca).T @ (B - cb))
            D = np.eye(3)
            D[2, 2] = np.sign(np.linalg.det(Vt.T @ U.T))
            R = Vt.T @ D @ U.T
            dT = np.eye(4)
            dT[:3, :3] = R
            dT[:3, 3] = cb - R @ ca
            T_world = dT @ T_world
            rms = float(np.sqrt(np.mean(dist[ok] ** 2)))
    except Exception as exc:  # scipy missing or a degenerate fit: keep the coarse estimate
        info["icp_error"] = str(exc)[:80]
    info["icp_rms_mm"] = None if rms is None else round(1000.0 * rms, 2)
    E = np.linalg.inv(T_nom) @ T_world @ T_nom
    rotation = _rotation_angle_deg(E[:3, :3])
    translation = float(np.linalg.norm(E[:3, 3]))
    if rotation < MIN_ROTATION_DEG:
        reason = "within noise"
    elif rotation > MAX_ROTATION_DEG or translation > MAX_TRANSLATION_M:
        reason = "implausibly large"
    elif score < MIN_OVERLAP or score <= corr0:
        reason = "objects do not overlap better than nominal"
    elif rms is None or rms > MAX_ICP_RMS_M:
        reason = "ICP residual too large"
    else:
        return WristCorrection(camera, E, True, "applied", rotation, translation, info)
    return WristCorrection(camera, identity, False, reason, rotation, translation, info)


def calibrate_wrists(
    request: Any, station: Any, camera_fk: Any, *, table_z_hint: float = 0.731
) -> dict[str, WristCorrection]:
    """Per wrist camera of ``request`` (needs head and wrist depth): its correction, applied or not."""
    out: dict[str, WristCorrection] = {}
    if request.depths.get("head") is None:
        return out
    head_cfg = station.cameras["head"]
    head_pose = np.asarray(camera_fk("head", request), dtype=np.float64)
    Ph = _apply(
        head_pose,
        back_project(
            request.depths["head"],
            np.asarray(head_cfg.color_K),
            getattr(head_cfg, "distortion", None),
            HEAD_STEP,
            0.1,
            2.5,
        ),
    )
    for camera in ("hand_left", "hand_right"):
        if request.depths.get(camera) is None:
            continue
        cfg = station.cameras[camera]
        try:
            points = back_project(
                request.depths[camera],
                np.asarray(cfg.color_K),
                getattr(cfg, "distortion", None),
                DEPTH_STEP,
                SELF_VIEW_M,
                FAR_M,
            )
            pose = np.asarray(camera_fk(camera, request), dtype=np.float64)
            out[camera] = estimate_wrist_correction(
                camera=camera,
                head_points_base=Ph,
                wrist_points_optical=points,
                wrist_pose=pose,
                table_z_hint=table_z_hint,
            )
        except Exception as exc:  # never let the self-check break an episode
            out[camera] = WristCorrection(camera, np.eye(4), False, "error: " + str(exc)[:80])
    return out


__all__ = [
    "MAX_ROTATION_DEG",
    "MAX_TRANSLATION_M",
    "MIN_ROTATION_DEG",
    "WristCorrection",
    "back_project",
    "calibrate_wrists",
    "estimate_wrist_correction",
    "fit_table_plane",
]
