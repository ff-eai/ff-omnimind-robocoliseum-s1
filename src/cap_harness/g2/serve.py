# Copyright (c) 2026 FF EAI Robotics Ecosystem Inc. All rights reserved. See LICENSE-FF.txt.
r"""Serve mode: one long-running GenieSim policy server that runs a program for every episode.

``cap_harness.cli run`` is one process per episode: it opens the bridge, waits
for GenieSim's first request, runs the program, and exits. The leaderboard
cannot be driven like that: one tunnel session carries many episodes, chosen by
the server, back to back. A serve worker keeps one :class:`GenieSimBridge` open
and loops:

1. wait for a new episode (its first ``infer`` request);
2. resolve the task from the request's ``task_name`` (registry name, a registry
   name inside it, else the client's name unchanged so the program reads the
   sentence) and run the program through :func:`cap_harness.run.run_program` --
   the exact wiring of ``cap_harness.cli run`` -- on the caller-owned bridge;
3. *drain*: until GenieSim ends the episode, answer every request with a hold of
   ``geniesim.drain_chunk_rows`` rows on the last commanded row (gripper latched,
   ``need_depth`` false). A program that raised or returned early is covered the
   same way: the episode always runs to GenieSim's own end;
4. append one JSON line per episode to the worker's episode log: run dir,
   program outcome, the final ``task_progress`` from the ``episode_done`` request
   with its stage scores, bridge counters (requests, depth requests, keepalive
   holds) and timings.

Run directories are the local runs' (``<output>/g2_geniesim/<board>/<NN-task>/<seed>/run-*``
with ``source/program-result.json``); ``seed`` is the worker's episode counter.
``evaluation/serve-final.json`` is added after the run is finalized (so the
run manifest does not list it), and ``media/keyframes`` is removed unless asked
to keep it (about 50 MB per episode).

    python -m cap_harness.cli serve --station geniesim-leaderboard --port 9101 \
        --program examples/g2/instruction_board_v3_4.py --output-root runs/serve/w9101
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import shutil
import threading
import time
from typing import Any

from cap_harness.g2.adapter import end_pose_check, progress_complete, stage_scores
from cap_harness.g2.codec import G2Request
from cap_harness.g2.config import G2StationConfig, load_g2_station_config
from cap_harness.g2.geniesim_bridge import GenieSimBridge
from cap_harness.g2.registry import G2TaskMetadata, G2TaskRegistry

logger = logging.getLogger(__name__)

#: task_id given to a task the registry does not list (directory naming only).
UNLISTED_TASK_ID = 99
#: A drain gives up when the client has been gone this long with nothing pending.
DISCONNECTED_GRACE_S = 5.0


@dataclass
class ServeConfig:
    program_path: Path
    output_root: Path
    station: str = "geniesim-leaderboard"
    station_config_root: Path | None = None
    host: str | None = None
    port: int | None = None
    board: str = "instruction"
    """Board given to a task name the registry does not list."""

    max_episodes: int | None = None
    """Exit after this many episodes (tests, smoke runs); None serves forever."""

    idle_exit_s: float | None = None
    """Exit after this long with no client connected and no episode running."""

    max_steps: int | None = None
    keep_keyframes: bool = False
    capture_videos: bool = False
    drain_timeout_s: float = 1800.0
    episode_log: Path | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def resolve_task(
    registry: G2TaskRegistry, request: G2Request, board: str
) -> tuple[G2TaskMetadata, str]:
    """The task metadata for a request and how it was found (``name``, ``contains``, ``unlisted``)."""
    name = request.task_name.strip()
    listed = registry.by_task_name(name)
    if listed is not None:
        return listed, "name"
    # A variant name that embeds a listed one (e.g. a robust-board suffix): the longest wins.
    contained = [
        task for task in registry.enumerate_tasks() if task.task_name and task.task_name in name
    ]
    if contained:
        return max(contained, key=lambda task: len(task.task_name)), "contains"
    return (
        G2TaskMetadata(
            task_name=name or "unknown",
            board=G2TaskRegistry._board(board),
            task_id=UNLISTED_TASK_ID,
            description=request.prompt or name or "unlisted GenieSim task",
        ),
        "unlisted",
    )


class G2ServeWorker:
    """One bridge, many episodes; see the module docstring."""

    def __init__(self, config: ServeConfig) -> None:
        self.config = config
        self.station: G2StationConfig = load_g2_station_config(
            config.station,
            config_root=None
            if config.station_config_root is None
            else str(config.station_config_root),
        )
        self.program_path = Path(config.program_path).expanduser().resolve()
        if not self.program_path.is_file():
            raise ValueError(f"program not found: {self.program_path}")
        self.output_root = Path(config.output_root).expanduser().resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.episode_log = (
            Path(config.episode_log).expanduser().resolve()
            if config.episode_log is not None
            else self.output_root / "serve_episodes.jsonl"
        )
        self.registry = G2TaskRegistry()
        self.bridge = GenieSimBridge(self.station, host=config.host, port=config.port)
        self._stop = threading.Event()
        self._consumed = -1
        self.served = 0
        self._kinematics = None

    # -- lifecycle -------------------------------------------------------------

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> int:
        """Serve until ``max_episodes``, ``idle_exit_s`` or :meth:`stop`; returns episodes served."""
        self.bridge.start()
        logger.info(
            "serve: %s listening on %s (station %s, need_depth %s), program %s",
            os.getpid(),
            self.bridge.url,
            self.station.station,
            self.bridge.need_depth_policy,
            self.program_path,
        )
        self._log_event("worker_started")
        idle_since = time.monotonic()
        try:
            while not self._stop.is_set():
                if self.config.max_episodes is not None and self.served >= self.config.max_episodes:
                    break
                index = self.bridge.wait_for_episode(self._consumed, timeout=1.0)
                if index is None:
                    if self.bridge.connected.is_set():
                        idle_since = time.monotonic()
                    elif (
                        self.config.idle_exit_s is not None
                        and time.monotonic() - idle_since > self.config.idle_exit_s
                    ):
                        logger.info("serve: idle for %.0f s, exiting", self.config.idle_exit_s)
                        break
                    continue
                self._serve_episode(index)
                idle_since = time.monotonic()
        finally:
            self._log_event("worker_stopped")
            self.bridge.close()
        return self.served

    # -- one episode -----------------------------------------------------------

    def _serve_episode(self, index: int) -> None:
        from cap_harness.run import run_program

        request = self.bridge.episode_request()
        if request is None:
            self._consumed = index
            return
        metadata, how = resolve_task(self.registry, request, self.config.board)
        seed = self.served
        record: dict[str, Any] = {
            "event": "episode",
            "worker": {
                "port": self.bridge.port,
                "pid": os.getpid(),
                "station": self.station.station,
            },
            "served_index": seed,
            "bridge_episode_index": index,
            "client_episode_idx": request.episode_idx,
            "client_task_name": request.task_name,
            "task_name": metadata.task_name,
            "board": metadata.board,
            "task_id": metadata.task_id,
            "task_resolved_by": how,
            "prompt": request.prompt,
            "first_request_depth": request.depth_complete,
            "end_pose_fk_check": self._end_pose_check(request),
            "started_at": datetime.now(timezone.utc).isoformat(),
            "program": str(self.program_path),
        }
        logger.info(
            "serve: episode %d (client idx %d) task %s [%s] prompt %r",
            index,
            request.episode_idx,
            metadata.task_name,
            how,
            request.prompt,
        )
        started = time.monotonic()
        run_dir: Path | None = None
        try:
            outcome = run_program(
                benchmark="g2_geniesim",
                suite=metadata.board,
                task_id=metadata.task_id,
                seed=seed,
                program_path=self.program_path,
                output_root=self.output_root,
                max_steps=self.config.max_steps,
                capture_videos=self.config.capture_videos,
                # The profile object itself, so a --need-depth override reaches the adapter.
                station=self.station,  # type: ignore[arg-type]
                station_config_root=self.config.station_config_root,
                g2_bridge=self.bridge,
                g2_consumed_episode=index - 1,
                g2_task=metadata,
            )
            run_dir = outcome.run_dir
            record.update(
                {
                    "run_dir": str(outcome.run_dir),
                    "program_ok": outcome.program_ok,
                    "task_success_at_program_end": outcome.task_success,
                    "termination_reason": outcome.termination_reason,
                }
            )
        except KeyboardInterrupt:
            record.update(
                {"error": "interrupted", "program_s": round(time.monotonic() - started, 3)}
            )
            self._append(record)
            raise
        except Exception as exc:  # the episode still has to reach GenieSim's end
            logger.exception("serve: episode %d harness error", index)
            record.update({"error": f"{type(exc).__name__}: {exc}"[:2000], "program_ok": False})
        record["program_s"] = round(time.monotonic() - started, 3)
        # The episode this worker just ran is consumed whatever happens next.
        self._consumed = index
        self.served += 1
        drain_started = time.monotonic()
        record["drain"] = self._drain(index, request.sequence)
        record["drain_s"] = round(time.monotonic() - drain_started, 3)
        record["total_s"] = round(time.monotonic() - started, 3)
        record["final"] = self._final(request.sequence)
        record["bridge_episode_stats"] = record["drain"].pop("episode_stats", None)
        if run_dir is not None:
            self._tidy_run_dir(run_dir, record)
        self._append(record)
        logger.info(
            "serve: episode %d done in %.0f s: program_ok=%s final=%s drain=%s",
            index,
            record["total_s"],
            record.get("program_ok"),
            record["final"],
            record["drain"].get("end"),
        )

    def _end_pose_check(self, request: G2Request) -> dict[str, Any] | None:
        if not request.end_poses:
            return None
        if self._kinematics is None:
            from cap_harness.g2.kinematics import G2Kinematics

            self._kinematics = G2Kinematics.from_station(self.station)
        return end_pose_check(self._kinematics, request)

    def _drain(self, index: int, first_sequence: int) -> dict[str, Any]:
        """Hold until GenieSim ends episode ``index`` (or another one starts, or the client is gone).

        ``first_sequence`` is the bridge sequence of the episode's first request:
        an ``episode_done`` request after it ends this episode.
        """
        bridge = self.bridge
        rows = int(self.station.geniesim.drain_chunk_rows)
        holds = 0
        deadline = time.monotonic() + self.config.drain_timeout_s
        gone_since: float | None = None
        end = "stopped"
        while not self._stop.is_set():
            if bridge.episode_index != index:
                end = "next_episode"
                break
            if self._done_after(first_sequence) is not None:
                end = "episode_done"
                break
            if time.monotonic() > deadline:
                end = "drain_timeout"
                break
            pending, current = bridge.wait_for_pending(1.0)
            if pending is None:
                if bridge.connected.is_set():
                    gone_since = None
                elif gone_since is None:
                    gone_since = time.monotonic()
                elif time.monotonic() - gone_since > DISCONNECTED_GRACE_S:
                    end = "disconnected"
                    break
                continue
            if current != index:
                end = "next_episode"
                break
            try:
                # Nobody looks at these observations: no depth (6.7 MB each on the leaderboard).
                bridge.reply_raw(bridge.hold_result(rows), force_depth=False, record=False)
            except RuntimeError:
                continue  # answered by a keepalive hold in between; look again
            holds += 1
        return {
            "end": end,
            "holds": holds,
            "rows": holds * rows,
            "episode_stats": bridge.episode_stats(index),
        }

    def _done_after(self, first_sequence: int) -> G2Request | None:
        """The ``episode_done`` request that ended the episode starting at ``first_sequence``."""
        done = self.bridge.done_request()
        if done is None or done.sequence <= first_sequence:
            return None
        return done

    def _final(self, first_sequence: int) -> dict[str, Any]:
        """What GenieSim reported at the episode's end (its ``episode_done`` request), if seen."""
        done = self._done_after(first_sequence)
        if done is None:
            return {"episode_done_seen": False}
        progress = [dict(item) for item in done.task_progress]
        stages = stage_scores(progress)
        return {
            "episode_done_seen": True,
            "task_progress": progress,
            "stages": stages,
            "stage_score": round(sum(stages.values()) / len(stages), 4) if stages else None,
            "success": bool(progress) and progress_complete(progress),
        }

    def _tidy_run_dir(self, run_dir: Path, record: dict[str, Any]) -> None:
        if not self.config.keep_keyframes:
            shutil.rmtree(run_dir / "media" / "keyframes", ignore_errors=True)
        try:
            (run_dir / "evaluation").mkdir(parents=True, exist_ok=True)
            (run_dir / "evaluation" / "serve-final.json").write_text(
                json.dumps(record, indent=1, default=str) + "\n", encoding="utf-8"
            )
        except OSError:
            logger.exception("serve: could not write serve-final.json into %s", run_dir)

    # -- log -------------------------------------------------------------------

    def _append(self, record: dict[str, Any]) -> None:
        self.episode_log.parent.mkdir(parents=True, exist_ok=True)
        with self.episode_log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=str) + "\n")

    def _log_event(self, event: str) -> None:
        self._append(
            {
                "event": event,
                "at": datetime.now(timezone.utc).isoformat(),
                "pid": os.getpid(),
                "port": self.bridge.port if self.bridge.running else self.config.port,
                "station": self.station.station,
                "need_depth": self.bridge.need_depth_policy,
                "program": str(self.program_path),
                "served": self.served,
            }
        )


__all__ = [
    "DISCONNECTED_GRACE_S",
    "UNLISTED_TASK_ID",
    "G2ServeWorker",
    "ServeConfig",
    "resolve_task",
]
