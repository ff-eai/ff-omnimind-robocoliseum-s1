# Copyright (c) 2026 FF EAI Robotics Ecosystem Inc. All rights reserved. See LICENSE-FF.txt.
"""``EnvironmentAdapter`` for the AgiBot G2 driven through GenieSim 3.2.

Thin by design: normalize, delegate, never touch the wire. Every motion becomes
one or more action chunks handed to the :class:`GenieSimBridge`, which is the
single chokepoint to the simulator. The adapter owns three things the bridge
does not: the episode/task bookkeeping, the gripper **latch** (the last width
commanded per arm, re-sent with every chunk so a closed jaw stays closed), and
the tick accounting that keeps the run recorder's budget equal to the
simulator's ``step_out`` budget.

Tick semantics: one row of a reply is one simulator tick. ``step`` sends one
row; ``execute_trajectory`` sends its waypoints in chunks of
``geniesim.chunk_len`` rows; ``set_gripper`` holds the jaw target for
``gripper.full_travel_s`` worth of rows. The recorder counts one step per
command, so the extra rows of a chunk are added to its ``step_count`` after the
fact, which keeps ``episode_steps_remaining`` honest.

Depth on the upstream (leaderboard) wire is requested reply by reply
(``need_depth``). Each chunk tells the bridge whether the program will look at
the observation that follows it (the last chunk of a motion: yes; the chunks
before it and single ``step`` ticks: no), and :meth:`get_observation` refreshes
an observation that came without depth with a one-row hold that asks for it, so
the program never reads a zero depth map it did not expect.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import dataclasses
import logging
import math
import time
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np

from cap_harness.contracts import (
    ArmCommand,
    ExecutionResult,
    Observation,
    RobotAction,
    RobotPlanningContext,
    RobotState,
    StepResult,
    SynchronizedTrajectory,
    TaskContext,
    Trajectory,
)
from cap_harness.errors import ApiError, ErrorCode
from cap_harness.g2 import codec
from cap_harness.g2.codec import ARM_DOF, ARMS, CAMERAS, HEAD_DOF, WAIST_DOF, G2Request
from cap_harness.g2.config import G2StationConfig, load_g2_station_config
from cap_harness.g2.geniesim_bridge import GenieSimBridge
from cap_harness.g2.registry import G2TaskMetadata, G2TaskRegistry

if TYPE_CHECKING:
    from cap_harness.g2.kinematics import G2Kinematics
    from cap_harness.g2.resolver import InstructionResolver

logger = logging.getLogger(__name__)

#: Robust board (camqualgen): upstream ``corobotpolicy`` drops a camera's frame with probability 0.1-0.5 per request by
#: sending it all black (``apply_camera_image_augmentation``). A frame whose brightest pixel is below this is dropped.
BLACK_FRAME_MAX = 3
#: One-row holds a single observation may spend waiting for cameras that came black and have no usable substitute.
BLACK_REFRESH_MAX = 6
#: A dropped frame is replaced by the camera's last good frame only while its carrier holds still: the largest joint
#: change, in radians, of the arm a wrist rides on (both arms for the head, whose view they cross).
SUBSTITUTE_JOINT_TOL_RAD = 0.01


def is_black(rgb: np.ndarray) -> bool:
    """True for a dropped (all-black) camera frame."""
    return int(np.max(rgb)) < BLACK_FRAME_MAX if np.size(rgb) else True


#: ``task_progress`` entries whose ``STATUS`` means the stage is complete. GenieSim
#: reports ``PENDING`` until a stage resolves; the completed token is accepted in
#: the spellings seen across its evaluators.
SUCCESS_STATUS_TOKENS = frozenset({"SUCCESS", "SUCCEED", "SUCCEEDED", "DONE", "FINISHED", "PASS"})


def _progress_entry_ok(item: Mapping[str, Any]) -> bool:
    status = str(item.get("status", "")).strip().upper()
    try:
        numeric = float(item.get("score", 0))
    except (TypeError, ValueError):
        numeric = 0.0
    return status in SUCCESS_STATUS_TOKENS or numeric >= 1.0


def progress_complete(task_progress: Sequence[Mapping[str, Any]]) -> bool:
    """Every named stage has at least one successful entry (``ActionSetWaitAny`` alternatives
    share a name); entries without a name must each be successful.
    """
    groups: dict[str, bool] = {}
    for item in task_progress:
        name = str(item.get("name") or item.get("class_name") or "").strip()
        ok = _progress_entry_ok(item)
        if not name:
            if not ok:
                return False
            continue
        groups[name] = groups.get(name, False) or ok
    return all(groups.values()) if groups else True


def end_pose_check(kinematics: G2Kinematics, request: G2Request) -> dict[str, Any] | None:
    """FK of ``arms.<side>.end_link`` against the client's ``end_pose`` (upstream only).

    The 2026-10-25 leaderboard probe frames agree to 0.002 mm; a large error means
    the server's robot (or waist) is not the one the station describes.
    """
    if not request.end_poses:
        return None
    from cap_harness.g2.kinematics import rotation_angle_deg

    check: dict[str, Any] = {}
    for side, reported in request.end_poses.items():
        try:
            fk = kinematics.end_link_pose(side, request.arm(side), request.waist_joints)
        except (KeyError, ValueError):
            continue
        check[side] = {
            "pos_mm": round(float(np.linalg.norm(fk[:3, 3] - reported[:3, 3]) * 1000.0), 3),
            "rot_deg": round(float(rotation_angle_deg(fk, reported)), 4),
        }
    return check or None


def stage_scores(task_progress: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """Best score per named stage (``ActionSetWaitAny`` alternatives share a name), in [0, 1]."""
    groups: dict[str, float] = {}
    for item in task_progress:
        name = str(item.get("name") or item.get("class_name") or "").strip() or "unnamed"
        score = 1.0 if _progress_entry_ok(item) else 0.0
        try:
            score = max(score, min(1.0, max(0.0, float(item.get("score", 0)))))
        except (TypeError, ValueError):
            pass
        groups[name] = max(groups.get(name, 0.0), score)
    return groups


HOME_DIAGNOSTIC_TOLERANCE_RAD = 0.03
CONTROL_CONTRACT_VERSION = "g2-geniesim-chunk-v1"
#: Serve mode: seconds without a client before an unanswerable motion reports TERMINATED.
CLIENT_GONE_GRACE_S = 5.0


class G2GenieSimAdapter:
    """Protocol-facing wrapper over the GenieSim policy bridge."""

    embodiment = codec.EMBODIMENT
    registry_prefix = "g2"
    #: The omnipicker takes a position target, so an intermediate width holds.
    gripper_holds_intermediate = True

    def __init__(
        self,
        station: G2StationConfig | str = "geniesim-default",
        *,
        bridge: GenieSimBridge | None = None,
        registry: G2TaskRegistry | None = None,
        run_observer: object | None = None,
        horizon: int | None = None,
        fk: Any | None = None,
        camera_fk: Any | None = None,
        config_root: str | None = None,
        host: str | None = None,
        port: int | None = None,
        autostart: bool = True,
        reset_timeout_s: float | None = None,
        step_timeout_s: float | None = None,
        resolver: InstructionResolver | None = None,
        consumed_episode: int = -1,
        **_ignored: object,
    ) -> None:
        self._station = (
            station
            if isinstance(station, G2StationConfig)
            else load_g2_station_config(str(station), config_root=config_root)
        )
        self._registry = registry or G2TaskRegistry()
        self._owns_bridge = bridge is None
        self._bridge = bridge or GenieSimBridge(self._station, host=host, port=port)
        if autostart and not self._bridge.running:
            self._bridge.start()
        self._run_observer = run_observer
        self._horizon = horizon
        # Real FK by default: EE poses are the planner's tool frame and camera
        # poses follow the waist/head/arms. Tests pass explicit hooks (or
        # ``False`` to keep the identity placeholders).
        self._kinematics: G2Kinematics | None = None
        if fk is None or camera_fk is None:
            from cap_harness.g2.kinematics import G2Kinematics

            self._kinematics = G2Kinematics.from_station(self._station)
        if fk is None:
            fk = self._kinematics.fk_hook()
        if camera_fk is None:
            camera_fk = self._kinematics.camera_fk_hook()
        self._fk = fk or None
        self._camera_fk = camera_fk or None
        # Robust board (camposgen): per-episode wrist mount corrections from the self-check at reset()
        # (cap_harness.g2.wrist_calibration). The wrapper returns FK's pose unchanged for any camera without one, so
        # an unperturbed scene is served exactly as before.
        self._wrist_corrections: dict[str, np.ndarray] = {}
        self.wrist_calibration: dict[str, Any] | None = None
        self._calibration_pending = False
        if self._camera_fk is not None:
            nominal_camera_fk = self._camera_fk

            def camera_fk_with_corrections(camera: str, request: G2Request) -> np.ndarray:
                pose = nominal_camera_fk(camera, request)
                correction = self._wrist_corrections.get(camera)
                return (
                    pose if correction is None else np.asarray(pose, dtype=np.float64) @ correction
                )

            self._camera_fk = camera_fk_with_corrections
        # Robust board (camqualgen): per camera the last good RGB frame and its carrier's joints (black_frame_guard).
        self._valid_frames: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._substituted_sequences: set[int] = set()
        self.black_frame_stats: dict[str, int] = {"substituted": 0, "refreshes": 0, "unresolved": 0}
        geniesim = self._station.geniesim
        self._reset_timeout_s = (
            float(geniesim.reset_timeout_s) if reset_timeout_s is None else float(reset_timeout_s)
        )
        self._step_timeout_s = (
            float(geniesim.step_timeout_s) if step_timeout_s is None else float(step_timeout_s)
        )
        self._chunk_len = int(geniesim.chunk_len)
        self._pad_last_chunk = bool(geniesim.pad_last_chunk)
        self._convention = self._station.gripper.convention
        self._task_metadata: G2TaskMetadata | None = None
        self._task_context: TaskContext | None = None
        self._seed = 0
        # A serve worker hands every episode's adapter the bridge's last consumed
        # episode, so reset() waits for the next one instead of re-running it.
        self._consumed_episode = int(consumed_episode)
        self._started = False
        self._last_seen_sequence = -1
        self._last_row: np.ndarray | None = None
        # Manipulation board (sorting): the waist target once the program commands one (move_torso); every row after
        # that carries it (21 columns), holds included, so the torso never falls back to an uncommanded state.
        self._waist_target: np.ndarray | None = None
        self._depth_refreshes = 0
        self._gone_since: float | None = None
        self._hold_last_command = geniesim.keepalive_hold_target == "last_command"
        # Any wire that sends depth on request (not our patched client, which always
        # does): an observation read without depth is refreshed. Under "always" this
        # only catches requests that lost depth elsewhere (e.g. after a relay hold).
        self._refresh_depth = geniesim.need_depth != "none"
        self.end_pose_check: dict[str, Any] | None = None
        self._gripper_latch: dict[str, float] = {}
        self._tick = 0
        self._chunks_sent = 0
        self._success_observed_step: int | None = None
        self._done_request: G2Request | None = None
        # The VLM client behind ``resolve_instruction``; built on first use from the
        # environment unless a test injects one.
        self._resolver = resolver
        self._closed = False
        self.control_frequency = float(self._station.controller.control_frequency_hz)
        self.control_period_s = 1.0 / self.control_frequency

    # -- identity ----------------------------------------------------------

    @property
    def native_env(self) -> GenieSimBridge:
        """The bridge, mirroring the simulator adapters' ``native_env``."""
        return self._bridge

    @property
    def bridge(self) -> GenieSimBridge:
        return self._bridge

    @property
    def station(self) -> G2StationConfig:
        return self._station

    @property
    def kinematics(self) -> G2Kinematics | None:
        """The FK model behind the default hooks (``None`` when both were injected)."""
        return self._kinematics

    @property
    def arms(self) -> tuple[str, ...]:
        return ARMS

    @property
    def dof(self) -> int:
        return ARM_DOF

    @property
    def tick_count(self) -> int:
        """Simulator ticks commanded since the episode started."""
        return self._tick

    @property
    def success_observed_step(self) -> int | None:
        return self._success_observed_step

    def resolve_arm(self, arm: str) -> str:
        """``left``/``right`` pass through; ``primary``/``secondary`` go via the profile."""
        return self._station.resolve_arm(arm)

    # -- helpers -----------------------------------------------------------

    def _require_request(self) -> G2Request:
        request = self._bridge.latest()
        if request is None:
            raise RuntimeError(
                "no observation yet: GenieSim has not sent an infer request; call reset() "
                f"and start the benchmark client against {self._bridge.url}"
            )
        return request

    def _timestamp(self) -> float:
        return self._tick / self.control_frequency

    def _observation_from(self, request: G2Request) -> Observation:
        # The wrist self-check needs every camera's depth: an episode's first request has it only when GenieSim
        # sends it (later instructions of one GenieSim process often start without), so it runs at the first
        # depth-complete request, before any observation is built from it.
        if self._calibration_pending and request.depth_complete:
            self._calibrate_wrists(request)
        return codec.observation(
            self._guarded(request),
            self._station,
            task_context=self._task_context,
            fk=self._fk,
            camera_fk=self._camera_fk,
            timestamp_s=self._timestamp(),
        )

    @staticmethod
    def _carrier_joints(camera: str, request: G2Request) -> np.ndarray:
        if camera == "hand_left":
            return np.asarray(request.left_joints, dtype=np.float64)
        if camera == "hand_right":
            return np.asarray(request.right_joints, dtype=np.float64)
        return np.asarray(request.arm_joints, dtype=np.float64)

    def _guarded(self, request: G2Request) -> G2Request:
        """Robust board (camqualgen): each black camera frame replaced by that camera's last good frame while its
        carrier still holds the pose it was taken from (station ``geniesim.black_frame_guard``). The request itself
        when the guard is off or nothing is black, so an unperturbed episode is served exactly as before.
        """
        if not self._station.geniesim.black_frame_guard:
            return request
        images = dict(request.images)
        replaced = []
        for camera, rgb in request.images.items():
            carrier = self._carrier_joints(camera, request)
            if not is_black(rgb):
                self._valid_frames[camera] = (rgb, carrier)
                continue
            stored = self._valid_frames.get(camera)
            if stored is not None and stored[1].shape == carrier.shape:
                if float(np.max(np.abs(stored[1] - carrier))) <= SUBSTITUTE_JOINT_TOL_RAD:
                    images[camera] = stored[0]
                    replaced.append(camera)
        if not replaced:
            return request
        if request.sequence not in self._substituted_sequences:
            self._substituted_sequences.add(request.sequence)
            self.black_frame_stats["substituted"] += 1
            logger.info(
                "black frame guard: request %d %s replaced by the last good frame",
                request.sequence,
                replaced,
            )
        return dataclasses.replace(request, images=MappingProxyType(images))

    def _refresh_black(self, request: G2Request) -> G2Request:
        """One-row holds until no camera of the (guarded) observation is black, at most ``BLACK_REFRESH_MAX``."""
        if not self._station.geniesim.black_frame_guard or not self._started:
            return request
        for _ in range(BLACK_REFRESH_MAX):
            black = [name for name, rgb in self._guarded(request).images.items() if is_black(rgb)]
            if not black:
                return request
            self.black_frame_stats["refreshes"] += 1
            logger.info(
                "black frame guard: request %d %s black, one-row hold", request.sequence, black
            )
            following, error = self._send_chunk(
                self._hold_row(request).reshape(1, -1),
                source="black_frame_refresh",
                force_depth=True,
            )
            if error is not None or following is None:
                return request
            request = following
        if any(is_black(rgb) for rgb in self._guarded(request).images.values()):
            self.black_frame_stats["unresolved"] += 1
        return request

    def _state_from(self, request: G2Request) -> RobotState:
        return codec.robot_state(request, self._station, fk=self._fk, timestamp_s=self._timestamp())

    def _latch_from(self, request: G2Request) -> None:
        for side in ARMS:
            self._gripper_latch.setdefault(side, request.gripper_norm(side, self._convention))

    def _latched(self, side: str, request: G2Request) -> float:
        if side not in self._gripper_latch:
            self._gripper_latch[side] = request.gripper_norm(side, self._convention)
        return self._gripper_latch[side]

    def _row(
        self,
        request: G2Request,
        joints: Mapping[str, np.ndarray] | None = None,
        grippers: Mapping[str, float] | None = None,
        waist: np.ndarray | None = None,
    ) -> np.ndarray:
        """One chunk row: commanded joints where given, the measured ones elsewhere; the waist target once set."""
        parts: list[np.ndarray] = []
        for side in ARMS:
            target = None if joints is None else joints.get(side)
            parts.append(
                np.asarray(request.arm(side), dtype=np.float64).reshape(ARM_DOF)
                if target is None
                else np.asarray(target, dtype=np.float64).reshape(ARM_DOF)
            )
        widths = []
        for side in ARMS:
            value = None if grippers is None else grippers.get(side)
            widths.append(self._latched(side, request) if value is None else float(value))
        if waist is None:
            waist = self._waist_target
        if waist is None:
            return np.concatenate([*parts, np.asarray(widths, dtype=np.float64)])
        return np.concatenate(
            [*parts, np.asarray(widths, dtype=np.float64), np.asarray(waist, dtype=np.float64).reshape(WAIST_DOF)]
        )

    def _row_width(self) -> int:
        """Columns of a chunk row now: 16, or 21 once the waist has been commanded this episode."""
        return codec.CHUNK_WIDTH if self._waist_target is None else codec.CHUNK_WIDTH_WITH_WAIST

    def _commanded_action(self, row: np.ndarray) -> RobotAction:
        """The ``RobotAction`` a chunk row commands, for the recorder."""
        return RobotAction(
            arms={
                side: ArmCommand(
                    mode="joint_position",
                    target=row[index * ARM_DOF : (index + 1) * ARM_DOF],
                    gripper_position=float(np.clip(row[2 * ARM_DOF + index], 0.0, 1.0)),
                    embodiment=self.embodiment,
                )
                for index, side in enumerate(ARMS)
            }
        )

    def _episode_over(self) -> str | None:
        """Why the episode this adapter runs is no longer the client's, or None.

        Only meaningful after :meth:`reset`. Catches what happens while the program
        is not replying: a keepalive hold, then ``episode_done``, then the next
        episode's first request -- which must never receive this episode's motion.
        """
        if not self._started:
            return None
        if self._bridge.episode_index != self._consumed_episode:
            return "GenieSim started a new episode"
        done = self._bridge.done_request()
        if done is not None and done.sequence > self._last_seen_sequence:
            self._done_request = done
            return "GenieSim ended the episode"
        return None

    def _client_gone(self) -> str | None:
        """Behind the tunnel relay a closed connection is a finished session: say so after a grace.

        Only with ``geniesim.new_connection_new_episode`` (serve mode); the local
        client reconnects within an episode, so there the adapter keeps waiting.
        """
        if not (self._started and self._bridge.new_connection_new_episode):
            return None
        if self._bridge.connected.is_set() or self._bridge.pending() is not None:
            self._gone_since = None
            return None
        now = time.monotonic()
        if self._gone_since is None:
            self._gone_since = now
        if now - self._gone_since < CLIENT_GONE_GRACE_S:
            return None
        return "the GenieSim client disconnected"

    def _answer(
        self, chunk: np.ndarray, *, observe_next: bool | None, force_depth: bool | None
    ) -> tuple[G2Request | None, str | None]:
        """Reply ``chunk`` to the pending request, waiting out a keepalive gap.

        After a keepalive hold the client replays that one row and sends a new
        request; a program that finishes thinking inside that gap must answer
        the new request, not fail. Returns ``(answered_request, None)`` or
        ``(None, reason)`` (``reason`` starting with ``ended:`` when the episode
        is over).
        """
        deadline = time.monotonic() + self._step_timeout_s
        while True:
            over = self._episode_over() or self._client_gone()
            if over is not None:
                return None, f"ended: {over}"
            pending, index = self._bridge.wait_for_pending(
                max(0.0, min(1.0, deadline - time.monotonic())) if self._started else 0.0
            )
            if pending is not None and self._started and index != self._consumed_episode:
                return None, "ended: GenieSim started a new episode"
            if pending is not None:
                try:
                    self._bridge.reply(chunk, observe_next=observe_next, force_depth=force_depth)
                except RuntimeError:
                    if not self._bridge.running or not self._started:
                        raise
                    # Held by the keepalive between pending() and reply(): wait again.
                    continue
                return pending, None
            if not self._started:
                # Before reset() nothing is waited for (and nothing guarded).
                return None, (
                    "no pending GenieSim infer request to answer; the client is not waiting "
                    "for an action (episode ended, not started, or already held)"
                )
            if not self._bridge.running:
                return None, "the GenieSim bridge is closed"
            if time.monotonic() >= deadline:
                return None, (
                    f"no infer request to answer within {self._step_timeout_s:.0f} s "
                    "(the client is not waiting for an action)"
                )

    def _send_chunk(
        self,
        chunk: np.ndarray,
        *,
        source: str,
        observe_next: bool | None = None,
        force_depth: bool | None = None,
    ) -> tuple[G2Request | None, ApiError | None]:
        """Reply one chunk and wait for the observation that follows its replay.

        Returns the next request, or the typed failure explaining why there is
        none. The recorder sees one step per chunk, so the remaining rows of the
        chunk are added to its count afterwards. ``observe_next`` tells the
        bridge's ``need_depth`` policy whether the program will look at the
        images of the next request.
        """
        rows = int(np.asarray(chunk).shape[0])
        action = self._commanded_action(np.asarray(chunk, dtype=np.float64)[-1])
        observer = self._run_observer
        if observer is not None:
            observer.before_step(action)  # may raise StepLimitReached
        try:
            answered, problem = self._answer(
                chunk, observe_next=observe_next, force_depth=force_depth
            )
        except RuntimeError as exc:
            answered, problem = None, str(exc)
        if problem is not None and problem.startswith("ended: "):
            final = self._done_request
            error = ApiError(
                code=ErrorCode.TERMINATED,
                message=f"{source}: {problem[len('ended: ') :]} before the chunk was sent",
                details={"tick": self._tick, "chunk_rows": rows},
            )
            self._record(
                observer,
                action,
                StepResult(
                    ok=False,
                    error=error,
                    observation=None if final is None else self._observation_from(final),
                    terminated=True,
                ),
            )
            return None, error
        if problem is not None:
            error = ApiError(
                code=ErrorCode.EXECUTION_FAILED,
                message=f"{source}: {problem}",
                details={"tick": self._tick, "chunk_rows": rows},
            )
            self._record(observer, action, StepResult(ok=False, error=error))
            return None, error
        self._last_row = np.asarray(chunk, dtype=np.float64)[-1].copy()
        self._tick += rows
        self._chunks_sent += 1
        deadline = time.monotonic() + self._step_timeout_s
        request: G2Request | None = None
        while True:
            request, index = self._bridge.wait_for_pending(
                max(0.0, min(1.0, deadline - time.monotonic()))
            )
            # The client answers a finished episode with one more request carrying
            # episode_done=True, which the bridge holds itself, and the request after
            # that already belongs to the next episode. Detect the end by the done
            # request's sequence rather than by what happens to be pending now --
            # and as soon as it arrives: the next episode may be a scene reset away.
            done = self._bridge.done_request()
            if done is not None and (answered is None or done.sequence > answered.sequence):
                self._done_request = done
                error = ApiError(
                    code=ErrorCode.TERMINATED,
                    message=f"{source}: GenieSim ended the episode during the chunk",
                    details={"tick": self._tick, "done_sequence": done.sequence},
                )
                self._record(
                    observer,
                    action,
                    StepResult(
                        ok=False,
                        error=error,
                        observation=self._observation_from(done),
                        terminated=True,
                    ),
                )
                return None, error
            gone = self._client_gone()
            if (request is not None and self._started and index != self._consumed_episode) or gone:
                # A new client connection (a new leaderboard session behind the relay)
                # started another episode without an episode_done for this one, or the
                # session's connection is gone for good.
                error = ApiError(
                    code=ErrorCode.TERMINATED,
                    message=f"{source}: "
                    + (gone or "GenieSim started a new episode during the chunk"),
                    details={"tick": self._tick, "episode_index": index},
                )
                self._record(observer, action, StepResult(ok=False, error=error, terminated=True))
                return None, error
            if request is not None or time.monotonic() >= deadline or not self._bridge.running:
                break
        if request is None:
            error = ApiError(
                code=ErrorCode.TIMEOUT,
                message=(
                    f"{source}: no infer request from GenieSim within {self._step_timeout_s:.0f} s "
                    f"after replying {rows} rows"
                ),
                details={"tick": self._tick, "chunk_rows": rows, "bridge": self._bridge.stats()},
            )
            self._record(observer, action, StepResult(ok=False, error=error))
            return None, error
        self._last_seen_sequence = max(self._last_seen_sequence, request.sequence)
        step = StepResult(
            ok=True,
            observation=self._observation_from(request),
            diagnostics={"tick": self._tick, "chunk_rows": rows, "source": source},
        )
        self._record(observer, action, step)
        if observer is not None and rows > 1:
            # One ``after_step`` counted one; the chunk consumed ``rows`` ticks.
            count = getattr(observer, "step_count", None)
            if isinstance(count, int):
                observer.step_count = count + rows - 1
        return request, None

    @staticmethod
    def _record(observer: object | None, action: RobotAction, step: StepResult) -> None:
        if observer is None:
            return
        observer.after_step(action, step)  # type: ignore[attr-defined]

    def _chunks(self, rows: np.ndarray) -> list[np.ndarray]:
        chunks = [
            rows[start : start + self._chunk_len] for start in range(0, len(rows), self._chunk_len)
        ]
        if self._pad_last_chunk and chunks and len(chunks[-1]) < self._chunk_len:
            last = chunks[-1]
            padding = np.tile(last[-1], (self._chunk_len - len(last), 1))
            chunks[-1] = np.vstack([last, padding])
        return chunks

    # -- lifecycle ---------------------------------------------------------

    def reset(self, task_ref: object = None, seed: int = 0) -> Observation:
        """Block until GenieSim starts a new episode, then return its first observation.

        Nothing is commanded: the simulator resets itself and sends the first
        ``infer`` request; that request *is* the reset observation. The reply to
        it is whatever the program's first motion turns out to be.
        """
        if isinstance(task_ref, TaskContext):
            self._task_context = task_ref
            self._task_metadata = None
        elif task_ref is not None:
            self._task_metadata = self._registry.resolve(task_ref)
        self._seed = int(seed)
        index = self._bridge.wait_for_episode(self._consumed_episode, self._reset_timeout_s)
        if index is None:
            raise RuntimeError(
                f"timed out after {self._reset_timeout_s:.0f} s waiting for GenieSim to start an "
                f"episode on {self._bridge.url}; is the benchmark client running with "
                f"--infer-host=<this host>:{self._bridge.port}?"
            )
        self._consumed_episode = index
        self._bridge.episode_started.clear()
        request = self._bridge.episode_request() or self._require_request()
        self._started = True
        self._last_seen_sequence = request.sequence
        self._last_row = None
        self._waist_target = None
        self._depth_refreshes = 0
        self._tick = 0
        self._chunks_sent = 0
        self._success_observed_step = None
        self._done_request = None
        self._gripper_latch = {}
        self._valid_frames = {}
        self._substituted_sequences = set()
        self.black_frame_stats = {"substituted": 0, "refreshes": 0, "unresolved": 0}
        self._latch_from(request)
        self.end_pose_check = self._check_end_poses(request)
        self._wrist_corrections = {}
        self.wrist_calibration = None
        self._calibration_pending = bool(
            self._station.geniesim.wrist_self_calibration and self._camera_fk is not None
        )
        if not isinstance(task_ref, TaskContext):
            self._task_context = self._task_context_for(request)
        observation = self._observation_from(request)
        if self._run_observer is not None:
            metadata = dict(self._task_context.metadata) if self._task_context else {}
            metadata.update(
                {
                    "embodiment": self.embodiment,
                    "station": self._station.station,
                    "backend": self._station.backend,
                    "bridge": self._bridge.stats(),
                }
            )
            if self.end_pose_check is not None:
                metadata["end_pose_fk_check"] = self.end_pose_check
            if self.wrist_calibration is not None:
                metadata["wrist_calibration"] = self.wrist_calibration
            elif self._calibration_pending:
                metadata["wrist_calibration"] = {"pending": "first request without depth"}
            self._run_observer.on_reset(observation, metadata)  # type: ignore[attr-defined]
        return observation

    def _calibrate_wrists(self, request: G2Request) -> None:
        """The wrist self-check on the episode's first depth-complete request (``geniesim.wrist_self_calibration``).

        Estimated with the nominal poses (corrections cleared first); only estimates that pass the module's gates
        become corrections. A failure leaves the nominal poses and is recorded, never raised.
        """
        self._calibration_pending = False
        self._wrist_corrections = {}
        self.wrist_calibration = None
        if not self._station.geniesim.wrist_self_calibration or self._camera_fk is None:
            return
        from cap_harness.g2.wrist_calibration import calibrate_wrists

        started = time.monotonic()
        try:
            results = calibrate_wrists(request, self._station, self._camera_fk)
        except Exception as exc:  # the self-check must never cost an episode
            self.wrist_calibration = {"error": f"{type(exc).__name__}: {exc}"[:160]}
            return
        self._wrist_corrections = {
            name: item.correction for name, item in results.items() if item.applied
        }
        self.wrist_calibration = {name: item.summary() for name, item in results.items()}
        self.wrist_calibration["seconds"] = round(time.monotonic() - started, 3)
        self.wrist_calibration["request"] = request.sequence
        logger.info(
            "wrist self-check (client episode %s, request %d): %s",
            request.episode_idx,
            request.sequence,
            {
                name: (
                    item.applied,
                    round(item.rotation_deg, 2),
                    round(1000.0 * item.translation_m, 1),
                    item.reason,
                )
                for name, item in results.items()
            },
        )

    def _check_end_poses(self, request: G2Request) -> dict[str, Any] | None:
        if self._kinematics is None:
            return None
        return end_pose_check(self._kinematics, request)

    def _task_context_for(self, request: G2Request) -> TaskContext:
        metadata = self._task_metadata
        listed = metadata or self._registry.by_task_name(request.task_name)
        task_name = metadata.task_name if metadata else (request.task_name or "unknown")
        language = request.prompt.strip() or (listed.description if listed else task_name)
        if listed is not None and metadata is not None and request.task_name:
            mismatch = request.task_name != metadata.task_name
        else:
            mismatch = False
        return TaskContext(
            suite=listed.board if listed else "geniesim",
            task_id=listed.task_id if listed else 0,
            task_name=task_name,
            language=language,
            family=listed.family if listed else "g2_geniesim",
            metadata={
                "seed": self._seed,
                "prompt": request.prompt,
                "client_task_name": request.task_name,
                "client_episode_idx": request.episode_idx,
                "bridge_episode_index": self._consumed_episode,
                "robot_type": request.robot_type,
                "station": self._station.station,
                "backend": self._station.backend,
                "horizon": self._horizon or (listed.horizon if listed else None),
                "task_name_mismatch": mismatch,
                "scene_reset": "simulator",
            },
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._owns_bridge:
            self._bridge.close()

    # -- observation -------------------------------------------------------

    def get_task_context(self) -> TaskContext:
        if self._task_context is None:
            raise RuntimeError("reset() must be called before reading task context")
        return self._task_context

    def get_observation(self) -> Observation:
        """The current observation; with depth, if the station's wire can be asked for it.

        Upstream genie_sim sends depth only while the previous reply asked for it.
        When the program reads an observation whose request came without depth
        (a servo ``step`` or a policy that skipped it), the adapter answers that
        request with a one-row hold carrying ``need_depth: true`` and returns the
        next one: one tick and one more round trip, instead of a zero depth map.
        """
        request = self._require_request()
        if self._refresh_depth and self._started and not request.depth_complete:
            refreshed = self._depth_refresh(request)
            if refreshed is not None:
                request = refreshed
        request = self._refresh_black(request)
        return self._observation_from(request)

    @property
    def depth_refreshes(self) -> int:
        """Observations this episode refreshed to get depth (one tick and one round trip each)."""
        return self._depth_refreshes

    def _hold_row(self, request: G2Request) -> np.ndarray:
        """One row that keeps the robot where it is going.

        The last commanded row when the profile holds on the last command, else the
        reported joints.
        """
        if self._hold_last_command and self._last_row is not None:
            return self._last_row.copy()
        return self._row(request)

    def _depth_refresh(self, request: G2Request) -> G2Request | None:
        # _send_chunk answers whatever request the client is waiting on (after a
        # keepalive hold that is the next one) and gives up if the episode is over.
        self._depth_refreshes += 1
        following, error = self._send_chunk(
            self._hold_row(request).reshape(1, -1), source="depth_refresh", force_depth=True
        )
        return following if error is None else None

    def get_robot_state(self) -> RobotState:
        return self._state_from(self._require_request())

    def check_success(self) -> bool:
        """True when the client's ``task_progress`` reports every stage complete.

        ``task_progress`` is host-side evidence: it is read here for the run
        outcome and never exposed to the program. GenieSim lists the alternatives
        of an ``ActionSetWaitAny`` group as separate entries sharing a name (for
        example ``PickUpOnGripper`` for the left and the right gripper), and only
        one of them can succeed, so entries are grouped by name and a group counts
        as complete when any of its entries is.
        """
        request = self._done_request or self._bridge.latest()
        if request is None or not request.task_progress:
            return False
        if not progress_complete(request.task_progress):
            return False
        if self._success_observed_step is None:
            self._success_observed_step = self._tick
        return True

    # -- actuation ---------------------------------------------------------

    def step(self, action: RobotAction) -> StepResult:
        try:
            request = self._require_request()
        except RuntimeError as exc:
            return StepResult(
                ok=False, error=ApiError(code=ErrorCode.ADAPTER_FAILED, message=str(exc))
            )
        joints: dict[str, np.ndarray] = {}
        grippers: dict[str, float] = {}
        for side, command in action.arms.items():
            name = self.resolve_arm(side)
            if name not in ARMS:
                return StepResult(
                    ok=False,
                    error=ApiError(
                        code=ErrorCode.INVALID_REQUEST,
                        message=f"action names unknown G2 arms: {[side]}",
                        details={"known_arms": list(ARMS)},
                    ),
                )
            joints[name] = np.asarray(command.target, dtype=np.float64).reshape(ARM_DOF)
            if command.gripper_position is not None:
                grippers[name] = float(command.gripper_position)
                self._gripper_latch[name] = grippers[name]
        row = self._row(request, joints, grippers)
        # A single tick is a servo step (move_to_joints reads joints, not images); an
        # observation read afterwards refreshes its depth if the policy skipped it.
        following, error = self._send_chunk(row.reshape(1, -1), source="step", observe_next=False)
        if error is not None:
            terminated = error.code is ErrorCode.TERMINATED
            final = self._done_request if terminated else None
            return StepResult(
                ok=False,
                error=error,
                observation=None if final is None else self._observation_from(final),
                terminated=terminated,
            )
        assert following is not None
        return StepResult(
            ok=True,
            observation=self._observation_from(following),
            diagnostics={"tick": self._tick, "chunk_rows": 1},
        )

    def execute_trajectory(
        self, trajectory: Trajectory | SynchronizedTrajectory
    ) -> ExecutionResult:
        try:
            request = self._require_request()
        except RuntimeError as exc:
            return ExecutionResult(
                ok=False,
                steps_executed=0,
                error=ApiError(code=ErrorCode.ADAPTER_FAILED, message=str(exc)),
            )
        if isinstance(trajectory, Trajectory):
            per_arm = {
                self.resolve_arm(trajectory.arm): np.asarray(
                    trajectory.joint_positions, dtype=np.float64
                )
            }
            per_arm_grippers = (
                {}
                if trajectory.gripper_positions is None
                else {
                    self.resolve_arm(trajectory.arm): np.asarray(
                        trajectory.gripper_positions, dtype=np.float64
                    ).reshape(-1)
                }
            )
        else:
            per_arm = {
                self.resolve_arm(side): np.asarray(value, dtype=np.float64)
                for side, value in trajectory.joint_positions.items()
            }
            per_arm_grippers = (
                {}
                if trajectory.gripper_positions is None
                else {
                    self.resolve_arm(side): np.asarray(value, dtype=np.float64).reshape(-1)
                    for side, value in trajectory.gripper_positions.items()
                }
            )
        unknown = set(per_arm) - set(ARMS)
        if unknown:
            return ExecutionResult(
                ok=False,
                steps_executed=0,
                error=ApiError(
                    code=ErrorCode.INVALID_REQUEST,
                    message=f"trajectory names unknown G2 arms: {sorted(unknown)}",
                ),
            )
        count = max(len(value) for value in per_arm.values())
        rows = np.empty((count, self._row_width()), dtype=np.float64)
        for index in range(count):
            joints = {side: value[min(index, len(value) - 1)] for side, value in per_arm.items()}
            grippers = {
                side: float(value[min(index, len(value) - 1)])
                for side, value in per_arm_grippers.items()
                if value.size
            }
            rows[index] = self._row(request, joints, grippers)
        for side, value in per_arm_grippers.items():
            if value.size:
                self._gripper_latch[side] = float(value[-1])

        chunks = self._chunks(rows)
        sent = 0
        padded = sum(len(chunk) for chunk in chunks) - count
        following: G2Request | None = None
        for number, chunk in enumerate(chunks):
            following, error = self._send_chunk(
                chunk, source="execute_trajectory", observe_next=number == len(chunks) - 1
            )
            if error is not None:
                terminated = error.code is ErrorCode.TERMINATED
                final = self._done_request if terminated else None
                return ExecutionResult(
                    ok=False,
                    steps_executed=sent,
                    final_observation=None if final is None else self._observation_from(final),
                    terminated=terminated,
                    error=error,
                    diagnostics={
                        "chunks_sent": sent // max(1, self._chunk_len),
                        "ticks": self._tick,
                    },
                )
            sent += len(chunk)
        assert following is not None

        residuals = {
            side: float(
                np.max(
                    np.abs(
                        np.asarray(following.arm(side), dtype=np.float64).reshape(ARM_DOF)
                        - np.asarray(value[-1], dtype=np.float64).reshape(ARM_DOF)
                    )
                )
            )
            for side, value in per_arm.items()
        }
        tolerance = float(self._station.controller.final_joint_tolerance_rad)
        worst_arm = max(residuals, key=residuals.get)
        worst = residuals[worst_arm]
        error = None
        ok = True
        if worst > tolerance:
            ok = False
            error = ApiError(
                code=ErrorCode.EXECUTION_FAILED,
                message=(
                    f"trajectory ended {worst:.3f} rad from its final waypoint on the {worst_arm} "
                    f"arm (tolerance {tolerance:.3f} rad); the simulated arm did not track it"
                ),
                details={"joint_residual_rad": residuals},
            )
        return ExecutionResult(
            ok=ok,
            steps_executed=sent,
            final_observation=self._observation_from(following),
            error=error,
            final_errors=residuals,
            diagnostics={
                "joint_residual_rad": residuals,
                "chunks": len(chunks),
                "chunk_len": self._chunk_len,
                "padded_rows": int(padded),
                "waypoints": int(count),
                "ticks": self._tick,
            },
        )

    def _gripper_rows(self) -> int:
        return max(1, int(math.ceil(self._station.gripper.full_travel_s * self.control_frequency)))

    def _command_grippers(self, targets: Mapping[str, float], *, source: str) -> ExecutionResult:
        try:
            request = self._require_request()
        except RuntimeError as exc:
            return ExecutionResult(
                ok=False,
                steps_executed=0,
                error=ApiError(code=ErrorCode.ADAPTER_FAILED, message=str(exc)),
            )
        normalized: dict[str, float] = {}
        for arm, value in targets.items():
            name = self.resolve_arm(arm)
            if name not in ARMS:
                return ExecutionResult(
                    ok=False,
                    steps_executed=0,
                    error=ApiError(code=ErrorCode.NOT_FOUND, message=f"unknown arm {arm!r}"),
                )
            numeric = float(value)
            if not np.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
                return ExecutionResult(
                    ok=False,
                    steps_executed=0,
                    error=ApiError(
                        code=ErrorCode.INVALID_REQUEST,
                        message=f"gripper position for {name!r} must be finite and in [0, 1]",
                    ),
                )
            normalized[name] = numeric
        self._gripper_latch.update(normalized)
        rows = self._gripper_rows()
        chunk = np.tile(self._row(request, None, normalized), (rows, 1))
        following, error = self._send_chunk(chunk, source=source, observe_next=True)
        if error is not None:
            return ExecutionResult(
                ok=False,
                steps_executed=0,
                terminated=error.code is ErrorCode.TERMINATED,
                error=error,
            )
        assert following is not None
        observed = {side: following.gripper_norm(side, self._convention) for side in normalized}
        return ExecutionResult(
            ok=True,
            steps_executed=rows,
            final_observation=self._observation_from(following),
            final_errors={side: abs(observed[side] - normalized[side]) for side in normalized},
            diagnostics={
                "commanded": dict(normalized),
                "observed": observed,
                "ticks": self._tick,
                "hold_rows": rows,
            },
        )

    def set_gripper(self, position: float, *, arm: str = "primary") -> ExecutionResult:
        return self._command_grippers({arm: position}, source="set_gripper")

    def set_grippers(self, positions: Mapping[str, float]) -> ExecutionResult:
        """Both grippers in one chunk, so they move in the same ticks."""
        return self._command_grippers(dict(positions), source="set_grippers")

    def _homing_rows(self, request: G2Request) -> int:
        controller = self._station.controller
        worst = 0.0
        for side in ARMS:
            measured = np.asarray(request.arm(side), dtype=np.float64).reshape(ARM_DOF)
            home = self._station.arms[side].home_joints
            worst = max(worst, float(np.max(np.abs(measured - home))))
        seconds = worst / controller.homing_joint_speed_rad_s
        seconds = min(controller.max_homing_s, max(controller.min_homing_s, seconds))
        return max(1, int(math.ceil(seconds * self.control_frequency)))

    def go_home(self, *, arm: str = "both", duration: float | None = None) -> ExecutionResult:
        """Interpolate both arms to the profile's ``arm_standard`` pose.

        Both arms move together: home is a whole-robot pose. ``arm`` is accepted
        for signature compatibility and ignored. The waist is not commanded
        (phase 0 keeps it at the task's initial pose).
        """
        try:
            request = self._require_request()
        except RuntimeError as exc:
            return ExecutionResult(
                ok=False,
                steps_executed=0,
                error=ApiError(code=ErrorCode.ADAPTER_FAILED, message=str(exc)),
            )
        rows_count = (
            self._homing_rows(request)
            if duration is None
            else max(1, int(math.ceil(float(duration) * self.control_frequency)))
        )
        alphas = np.linspace(0.0, 1.0, rows_count + 1)[1:]
        rows = np.empty((rows_count, self._row_width()), dtype=np.float64)
        for index, alpha in enumerate(alphas):
            joints = {
                side: (1.0 - alpha) * np.asarray(request.arm(side), dtype=np.float64)
                + alpha * self._station.arms[side].home_joints
                for side in ARMS
            }
            rows[index] = self._row(request, joints, None)
        sent = 0
        following: G2Request | None = None
        home_chunks = self._chunks(rows)
        for number, chunk in enumerate(home_chunks):
            following, error = self._send_chunk(
                chunk, source="go_home", observe_next=number == len(home_chunks) - 1
            )
            if error is not None:
                return ExecutionResult(
                    ok=False,
                    steps_executed=sent,
                    terminated=error.code is ErrorCode.TERMINATED,
                    error=error,
                )
            sent += len(chunk)
        assert following is not None
        observation = self._observation_from(following)
        residuals = {
            side: float(
                np.max(
                    np.abs(
                        observation.robot_state.joint_positions[side]
                        - self._station.arms[side].home_joints
                    )
                )
            )
            for side in ARMS
        }
        return ExecutionResult(
            ok=True,
            steps_executed=sent,
            final_observation=observation,
            final_errors=residuals,
            diagnostics={
                "joint_residual_rad": residuals,
                "home_tolerance_rad": HOME_DIAGNOSTIC_TOLERANCE_RAD,
                "home_verified": max(residuals.values()) <= HOME_DIAGNOSTIC_TOLERANCE_RAD,
                "requested_arm": str(arm),
                "rows": int(rows_count),
                "ticks": self._tick,
            },
        )

    # -- planning and metadata --------------------------------------------

    def get_planning_context(self) -> RobotPlanningContext:
        """Describe the G2 to the planners: 7-DOF arms, cuRobo names, centre-link tools.

        Base transforms are the identity because the cuRobo description is the
        whole robot and already places both arms; the request joint names must
        equal the planner's, which is why the profile carries the ``idx*`` names.
        """
        return RobotPlanningContext(
            embodiment=self.embodiment,
            model="g2",
            joint_names={side: self._station.arms[side].joint_names for side in ARMS},
            base_transforms={side: np.eye(4, dtype=np.float64) for side in ARMS},
            end_effector_links={side: self._station.arms[side].tool_frame for side in ARMS},
            lock_joints=self.planner_lock_joints(),
        )

    def planner_lock_joints(self) -> dict[str, float] | None:
        """The torso the planner should assume now (``geniesim.planner_lock_from_request``), else None.

        Waist: the commanded target once the program moved it, else the reported waist rounded to
        ``planner_lock_quantum_rad`` (one cached planner per distinct map; the reported waist sits at the task's start
        state). Head: the request's, else the task's init head (``head.task_holds``). FK keeps the exact values.
        """
        geniesim = self._station.geniesim
        if not geniesim.planner_lock_from_request:
            return None
        request = self._bridge.latest()
        if request is None:
            return None
        if self._waist_target is not None:
            waist = np.asarray(self._waist_target, dtype=np.float64)
        else:
            quantum = float(geniesim.planner_lock_quantum_rad)
            waist = np.round(np.asarray(request.waist_joints, dtype=np.float64) / quantum) * quantum
        head = np.asarray(request.head_joints, dtype=np.float64).reshape(-1)
        if head.size != HEAD_DOF:
            head = np.asarray(self._station.head.hold_for(request.task_name), dtype=np.float64)
        locks = {name: round(float(value), 6) for name, value in zip(self._station.waist.joint_names, waist)}
        locks.update({name: round(float(value), 6) for name, value in zip(self._station.head.joint_names, head)})
        return locks

    def planning_workspace(self) -> tuple[np.ndarray, np.ndarray]:
        return self._station.planning_workspace_lower, self._station.planning_workspace_upper

    def camera_distortion(self, camera_name: str) -> tuple[float, ...] | None:
        """The station's lens model for one camera (``cameras.<name>.distortion``), if any."""
        camera = self._station.cameras.get(camera_name)
        return None if camera is None else camera.distortion

    def get_task_metadata(self) -> dict[str, Any]:
        context = self._task_context
        return {
            "embodiment": self.embodiment,
            "station": self._station.station,
            "backend": self._station.backend,
            "arms": list(ARMS),
            "dof": ARM_DOF,
            "cameras": list(CAMERAS),
            "language": "" if context is None else context.language,
            "task_name": "" if context is None else context.task_name,
            "suite": "" if context is None else context.suite,
            "horizon": None if context is None else context.metadata.get("horizon"),
        }

    def get_controller_metadata(self) -> dict[str, Any]:
        station = self._station
        return {
            "control_frequency_hz": self.control_frequency,
            "control_period_s": self.control_period_s,
            "arm_names": list(ARMS),
            "action_lower_bounds": station.joint_limits_lower.tolist(),
            "action_upper_bounds": station.joint_limits_upper.tolist(),
            "chunk_len": self._chunk_len,
            "pad_last_chunk": self._pad_last_chunk,
            "gripper_hold_rows": self._gripper_rows(),
            "control_contract_version": CONTROL_CONTRACT_VERSION,
            "control_contract": {
                "version": CONTROL_CONTRACT_VERSION,
                "backend": station.backend,
                "station": station.station,
                "reply_timeout_s": station.geniesim.reply_timeout_s,
                "step_timeout_s": self._step_timeout_s,
                "tick_hz": self.control_frequency,
            },
            "tick_count": self._tick,
            "chunks_sent": self._chunks_sent,
        }

    # -- torso (manipulation board) ------------------------------------------

    def move_torso(
        self,
        target: Sequence[float],
        *,
        speed_rad_s: float = 0.3,
        settle_rows: int = 15,
    ) -> ExecutionResult:
        """Turn the waist to ``target`` (5 joints in ``waist.joint_names`` order, radians).

        Rows interpolate from the current waist target (the reported waist before the first command) with the joint
        that moves most at ``speed_rad_s``, then hold ``settle_rows`` rows. The arms hold their last command (else
        their reported joints) and the grippers stay latched. Every later row of the episode carries this target
        (holds included), and with ``planner_lock_from_request`` the planner's torso follows it. GenieSim's client
        applies a reply's ``waist`` block (``corobotpolicy._parse_result``; ``pi_env.step``).
        """
        try:
            request = self._require_request()
        except RuntimeError as exc:
            return ExecutionResult(
                ok=False,
                steps_executed=0,
                error=ApiError(code=ErrorCode.ADAPTER_FAILED, message=str(exc)),
            )
        waist = self._station.waist
        try:
            goal = np.asarray(target, dtype=np.float64).reshape(WAIST_DOF)
            speed = float(speed_rad_s)
            settle = int(settle_rows)
        except (TypeError, ValueError) as exc:
            return ExecutionResult(
                ok=False,
                steps_executed=0,
                error=ApiError(code=ErrorCode.INVALID_REQUEST, message=f"move_torso: {exc}"),
            )
        if (
            not np.all(np.isfinite(goal))
            or np.any(goal < waist.limits_lower)
            or np.any(goal > waist.limits_upper)
            or not np.isfinite(speed)
            or speed <= 0.0
            or settle < 0
        ):
            return ExecutionResult(
                ok=False,
                steps_executed=0,
                error=ApiError(
                    code=ErrorCode.INVALID_REQUEST,
                    message="move_torso: the target must lie within the waist limits, speed > 0, settle >= 0",
                    details={
                        "target": goal.tolist(),
                        "lower": waist.limits_lower.tolist(),
                        "upper": waist.limits_upper.tolist(),
                    },
                ),
            )
        start = (
            np.asarray(request.waist_joints, dtype=np.float64)
            if self._waist_target is None
            else np.asarray(self._waist_target, dtype=np.float64)
        )
        travel = float(np.max(np.abs(goal - start)))
        moving = max(1, int(math.ceil(travel / speed * self.control_frequency)))
        held = None
        if self._last_row is not None:
            held = {
                side: self._last_row[index * ARM_DOF : (index + 1) * ARM_DOF]
                for index, side in enumerate(ARMS)
            }
        alphas = np.linspace(0.0, 1.0, moving + 1)[1:]
        rows = [self._row(request, held, None, waist=(1.0 - alpha) * start + alpha * goal) for alpha in alphas]
        rows += [self._row(request, held, None, waist=goal)] * settle
        self._waist_target = goal.copy()
        chunks = self._chunks(np.vstack(rows))
        sent = 0
        following: G2Request | None = None
        for number, chunk in enumerate(chunks):
            following, error = self._send_chunk(
                chunk, source="move_torso", observe_next=number == len(chunks) - 1
            )
            if error is not None:
                terminated = error.code is ErrorCode.TERMINATED
                final = self._done_request if terminated else None
                return ExecutionResult(
                    ok=False,
                    steps_executed=sent,
                    final_observation=None if final is None else self._observation_from(final),
                    terminated=terminated,
                    error=error,
                )
            sent += len(chunk)
        assert following is not None
        measured = np.asarray(following.waist_joints, dtype=np.float64)
        residual = float(np.max(np.abs(measured - goal)))
        return ExecutionResult(
            ok=True,
            steps_executed=sent,
            final_observation=self._observation_from(following),
            final_errors={"waist": residual},
            diagnostics={
                "waist_start": start.tolist(),
                "waist_target": goal.tolist(),
                "waist_measured": measured.tolist(),
                "waist_residual_rad": residual,
                "rows": len(rows),
                "ticks": self._tick,
            },
        )

    def get_body_state(
        self,
        waist: Sequence[float] | None = None,
        left: Sequence[float] | None = None,
        right: Sequence[float] | None = None,
    ) -> dict[str, Any]:
        """Torso and arm geometry now, from the robot's own kinematics (joint readings, no scene state).

        ``waist`` (reported), ``waist_target`` (commanded this episode, or None), ``head`` (reported, else the
        task's init head), per arm the base-frame origins of its seven joints (``joint_origins``) with the tool
        position and approach axis, and the head camera's position and optical axis. Never raises.

        g2-v44 (C5n): ``waist`` / ``left`` / ``right`` (optional) evaluate the same FK for those body or arm joints
        instead of the reported ones (``hypothetical`` is then True) - e.g. to find a torso move that carries both
        held arms without turning them. Per arm ``tool_x_axis`` is reported too, and ``body_origins`` (the five body
        links' origins, body joint 1 first). Nothing moves.
        """
        kinematics = self._kinematics
        if kinematics is None:
            return {"ok": False, "error": "no kinematic model"}
        try:
            request = self._require_request()
            head = np.asarray(request.head_joints, dtype=np.float64).reshape(-1)
            if head.size != HEAD_DOF:
                head = kinematics.head_hold(request.task_name)
            hypothetical = waist is not None or left is not None or right is not None
            body = request.waist_joints if waist is None else np.asarray(waist, dtype=np.float64).reshape(WAIST_DOF)
            left_joints = request.left_joints if left is None else np.asarray(left, dtype=np.float64).reshape(ARM_DOF)
            right_joints = request.right_joints if right is None else np.asarray(right, dtype=np.float64).reshape(ARM_DOF)
            if not (np.all(np.isfinite(body)) and np.all(np.isfinite(left_joints)) and np.all(np.isfinite(right_joints))):
                return {"ok": False, "error": "non-finite joints"}
            positions = kinematics.joint_map(
                left=left_joints,
                right=right_joints,
                waist=body,
                head=head,
            )
            arms: dict[str, Any] = {}
            for side in ARMS:
                arm = self._station.arms[side]
                origins = [
                    kinematics.link_pose(kinematics.model.joints[name].child, positions)[:3, 3].tolist()
                    for name in arm.joint_names
                ]
                tool = kinematics.link_pose(arm.tool_frame, positions)
                arms[side] = {
                    "joint_origins": origins,
                    "tool_position": tool[:3, 3].tolist(),
                    "tool_z_axis": tool[:3, 2].tolist(),
                    "tool_x_axis": tool[:3, 0].tolist(),
                }
            camera = kinematics.camera_pose(
                "head",
                left=left_joints,
                right=right_joints,
                waist=body,
                head=head,
            )
            body_origins = [
                kinematics.link_pose(kinematics.model.joints[name].child, positions)[:3, 3].tolist()
                for name in self._station.waist.joint_names
            ]
        except Exception as exc:  # a program-facing tool never raises
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        return {
            "ok": True,
            "hypothetical": hypothetical,
            "waist": np.asarray(body, dtype=np.float64).tolist(),
            "waist_reported": np.asarray(request.waist_joints, dtype=np.float64).tolist(),
            "waist_target": None if self._waist_target is None else self._waist_target.tolist(),
            "head": head.tolist(),
            "arms": arms,
            "body_origins": body_origins,
            "head_camera": {"position": camera[:3, 3].tolist(), "forward": camera[:3, 2].tolist()},
        }

    # -- instruction resolution --------------------------------------------

    @property
    def resolver(self) -> InstructionResolver:
        """The VLM client (``CAP_HARNESS_G2_VLM_URL`` / ``_MODEL`` or the bench default)."""
        if self._resolver is None:
            from cap_harness.g2.resolver import InstructionResolver

            self._resolver = InstructionResolver()
        return self._resolver

    def resolve_instruction(
        self, instruction: str | None = None, *, camera_name: str = "head"
    ) -> dict[str, Any]:
        """Resolve an instruction-board sentence against the current camera image.

        Asks the local vision-language model which ONE concrete object on the
        table the sentence refers to, using only the public RGB image of
        ``camera_name`` (``head`` by default) and the text. ``instruction``
        defaults to the episode's task language. Nonprivileged: no simulator
        state is read, and nothing here raises into the program.

        Returns a JSON-like dict: ``ok``, ``instruction``, ``arm`` (``"left"``,
        ``"right"`` or ``None`` when the sentence names none), ``query`` (a
        short noun phrase for ``segment_text``), ``alternatives`` (up to three
        fallback phrases), ``box_xyxy_norm`` (the model's own box, 0-1000 on
        both axes) and ``box_xyxy`` (the same box in pixels of ``image_size``),
        ``selection_rule`` (``named``, ``attribute``, ``largest``, ``smallest``,
        ``category``, ``common_sense``, ``logic_or``, ``number`` or
        ``position``), ``place_back`` (straighten / return-to-origin tasks),
        ``visible``, ``confidence`` (0-1), ``rationale``, ``latency_s`` and, when
        ``ok`` is False, ``error``.
        """
        from cap_harness.g2.resolver import failure

        text = instruction
        if text is None or not str(text).strip():
            context = self._task_context
            text = "" if context is None else context.language
        text = " ".join(str(text).split())
        try:
            observation = self._observation_from(self._refresh_black(self._require_request()))
        except Exception as exc:  # a program-facing tool never raises
            return failure(text, f"no observation available: {exc}").to_dict()
        camera = observation.cameras.get(str(camera_name))
        if camera is None:
            available = ", ".join(sorted(observation.cameras))
            return failure(
                text, f"camera {camera_name!r} not in observation (have: {available})"
            ).to_dict()
        try:
            result = self.resolver.resolve_rgb(camera.rgb, text)
        except Exception as exc:  # a program-facing tool never raises
            return failure(text, f"resolver failed: {exc}").to_dict()
        return result.to_dict()

    def monitor_snapshot(self) -> dict[str, Any]:
        snapshot = dict(self._bridge.stats())
        snapshot.update({"tick": self._tick, "chunks_sent": self._chunks_sent})
        return snapshot


__all__ = [
    "CONTROL_CONTRACT_VERSION",
    "HOME_DIAGNOSTIC_TOLERANCE_RAD",
    "SUCCESS_STATUS_TOKENS",
    "G2GenieSimAdapter",
    "end_pose_check",
    "progress_complete",
    "stage_scores",
]
