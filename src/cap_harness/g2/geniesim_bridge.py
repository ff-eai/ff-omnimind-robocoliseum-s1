# Copyright (c) 2026 FF EAI Robotics Ecosystem Inc. All rights reserved. See LICENSE-FF.txt.
"""The loop inversion: a policy *server* GenieSim's ``CoRobotPolicy`` connects to.

GenieSim evaluates a policy by being a WebSocket **client**: for every step that
needs a decision it sends one msgpack ``{"method": "infer", "params": ...}``
message and blocks until a ``{"result": ...}`` comes back carrying an action
chunk, which it then replays one row per simulator tick before asking again.
ASPIRE's ``EnvironmentAdapter`` is written the other way round -- a program
calls ``step`` and expects the world to advance. This bridge turns one into the
other:

* the server thread decodes each request into a :class:`~.codec.G2Request`
  and parks it as *pending*;
* the adapter thread picks it up with :meth:`wait_for_request`, reads the
  observation from it, and answers with :meth:`reply`;
* the reply resolves the handler's future, the handler sends it, and the next
  request arrives after the sim has replayed the chunk.

Two protocol facts shape the lifecycle handling. The client sends one last
request with ``episode_done=True`` after an episode ends and *replays whatever
that reply contains at the start of the next episode* (``update_task_status``
clears the buffer only when ``done`` flips), so the bridge answers it itself
with a one-row hold. And ``episode_idx`` is set only *after* an episode
finishes, so it lags by one; a new episode is therefore detected by the
``done`` flag falling, by the index changing, or by the first request ever.

A reply that takes longer than ``reply_timeout_s`` would trip the client's
120 s (leaderboard) / 180 s (local) inference timeout and drop the connection.
With ``keepalive_hold`` the bridge answers such a request with a one-row hold
instead, at the cost of one simulator tick; the next request supersedes it.
The hold commands the reported joints, or with ``keepalive_hold_target:
last_command`` the last row the program commanded in this episode, so an arm
still settling onto its target is not frozen short of it.

Upstream genie_sim (the leaderboard) sends depth on an episode's first request
and afterwards only while the previous reply carried ``need_depth: true``. The
station's ``need_depth`` policy sets that key on every reply this bridge sends
(see :data:`~cap_harness.g2.config.NEED_DEPTH_POLICIES`); ``none`` leaves the
reply exactly as our patched local client has always received it.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
import time
from typing import Any

import msgpack

from cap_harness.g2 import codec
from cap_harness.g2.codec import G2Request, GripperConvention
from cap_harness.g2.config import NEED_DEPTH_POLICIES

try:  # websockets >= 13: the new asyncio implementation.
    from websockets.asyncio.server import serve as _serve
except ImportError:  # pragma: no cover - websockets 12 keeps the legacy module.
    from websockets.server import serve as _serve  # type: ignore[no-redef]
from websockets.exceptions import ConnectionClosed

logger = logging.getLogger(__name__)

_INFER = "infer"
#: What ``corobotpolicy._parse_result`` requires of every reply.
_REPLY_KEYS = frozenset({"left_arm", "right_arm", "left_effector", "right_effector"})
_START_TIMEOUT_S = 10.0
_PING_INTERVAL_S = 20.0
_PING_TIMEOUT_S = 60.0


def _packb(value: Any) -> bytes:
    """Plain msgpack: the codec already produced lists and floats, never numpy."""
    return msgpack.packb(value, use_bin_type=True)


class GenieSimBridge:
    """msgpack-over-WebSocket policy server with a blocking, thread-safe adapter side."""

    def __init__(
        self,
        station: Any,
        *,
        host: str | None = None,
        port: int | None = None,
        reply_timeout_s: float | None = None,
        keepalive_hold: bool | None = None,
        decode_images: bool = True,
        depth_scales: dict[str, float] | None = None,
        need_depth: str | None = None,
        new_connection_new_episode: bool | None = None,
    ) -> None:
        geniesim = station.geniesim
        self._station = station
        self._host = geniesim.ws_host if host is None else str(host)
        self._port = geniesim.ws_port if port is None else int(port)
        self._reply_timeout_s = (
            float(geniesim.reply_timeout_s) if reply_timeout_s is None else float(reply_timeout_s)
        )
        self._keepalive_hold = (
            bool(geniesim.keepalive_hold) if keepalive_hold is None else bool(keepalive_hold)
        )
        self._decode_images = bool(decode_images)
        # The profile states each camera's wire scale (transport-specific: 1000/10000
        # for our local patch, 1000 everywhere upstream).
        self._depth_scales = (
            {name: float(camera.depth_scale) for name, camera in station.cameras.items()}
            if depth_scales is None
            else {name: float(scale) for name, scale in depth_scales.items()}
        )
        self._camera_poses = str(geniesim.camera_pose_source) == "reported"
        self._need_depth = str(geniesim.need_depth if need_depth is None else need_depth)
        if self._need_depth not in NEED_DEPTH_POLICIES:
            raise ValueError(f"need_depth must be one of {NEED_DEPTH_POLICIES}")
        self._hold_last_command = str(geniesim.keepalive_hold_target) == "last_command"
        self._new_connection_new_episode = (
            bool(geniesim.new_connection_new_episode)
            if new_connection_new_episode is None
            else bool(new_connection_new_episode)
        )
        self._convention: GripperConvention = station.gripper.convention
        self._include_waist = bool(geniesim.include_waist)

        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._latest: G2Request | None = None
        self._pending: G2Request | None = None
        self._pending_future: asyncio.Future[Any] | None = None
        self._pending_connection: Any = None
        self._previous_done = False
        self._episode_index = -1
        self._episode_request: G2Request | None = None
        self._done_request: G2Request | None = None
        self._request_count = 0
        self._reply_count = 0
        self._hold_count = 0
        self._error_count = 0
        self._connections = 0
        self._connection_count = 0
        self._closed = False
        self._force_new_episode = False
        #: Last row (wire units, 1-row result) the adapter replied in this episode.
        self._last_command: dict[str, Any] | None = None
        #: Counters since the current episode's first request (see :meth:`episode_stats`).
        self._episode_counts: dict[str, Any] = self._fresh_counts()
        self._finished_counts: dict[int, dict[str, Any]] = {}

        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._stop_event: asyncio.Event | None = None
        self._started = threading.Event()
        self._startup_error: BaseException | None = None
        self._bound_port: int | None = None

        #: Set when a new episode's first request arrives; cleared by whoever consumes it.
        self.episode_started = threading.Event()
        #: Set when a request carrying ``episode_done=True`` arrives.
        self.episode_ended = threading.Event()
        #: Set while at least one client is connected.
        self.connected = threading.Event()

    # -- lifecycle ---------------------------------------------------------

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive() and not self._closed

    @property
    def port(self) -> int:
        """The port actually bound (``port=0`` asks the OS for a free one)."""
        if self._bound_port is None:
            raise RuntimeError("the bridge is not started")
        return self._bound_port

    @property
    def host(self) -> str:
        return self._host

    @property
    def url(self) -> str:
        return f"ws://{self._host}:{self.port}"

    def start(self) -> GenieSimBridge:
        if self._closed:
            raise RuntimeError("the bridge is closed")
        if self._thread is not None:
            return self
        self._thread = threading.Thread(target=self._run, name="g2-geniesim-bridge", daemon=True)
        self._thread.start()
        if not self._started.wait(_START_TIMEOUT_S):
            raise RuntimeError("the GenieSim bridge did not start in time")
        if self._startup_error is not None:
            error = self._startup_error
            self._thread.join(timeout=1.0)
            self._thread = None
            self._started.clear()
            self._startup_error = None
            raise RuntimeError(
                f"the GenieSim bridge could not listen on {self._host}:{self._port}: {error}"
            )
        return self

    def close(self) -> None:
        """Stop serving. Idempotent; wakes every blocked adapter-side waiter."""
        with self._cond:
            if self._closed:
                return
            self._closed = True
            self._cond.notify_all()
        loop = self._loop
        if loop is not None and self._stop_event is not None:
            with contextlib.suppress(RuntimeError):
                loop.call_soon_threadsafe(self._stop_event.set)
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self.connected.clear()

    stop = close

    def __enter__(self) -> GenieSimBridge:
        return self.start()

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        del exc_type, exc, traceback
        self.close()

    # -- server thread -----------------------------------------------------

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        try:
            loop.run_until_complete(self._serve())
        finally:
            with contextlib.suppress(Exception):
                loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()
            self._loop = None
            self._started.set()

    async def _serve(self) -> None:
        self._stop_event = asyncio.Event()
        try:
            server_cm = _serve(
                self._handler,
                self._host,
                self._port,
                max_size=None,
                compression=None,
                ping_interval=_PING_INTERVAL_S,
                ping_timeout=_PING_TIMEOUT_S,
            )
            async with server_cm as server:
                sockets = getattr(server, "sockets", None) or ()
                self._bound_port = sockets[0].getsockname()[1] if sockets else self._port
                self._started.set()
                await self._stop_event.wait()
                self._cancel_pending()
        except Exception as exc:
            self._startup_error = exc
            self._started.set()

    def _cancel_pending(self) -> None:
        with self._cond:
            future = self._pending_future
            self._pending = None
            self._pending_future = None
            self._pending_connection = None
            self._cond.notify_all()
        if future is not None and not future.done():
            future.cancel()

    async def _handler(self, connection: Any) -> None:
        with self._cond:
            self._connections += 1
            self._connection_count += 1
            if self._new_connection_new_episode:
                # Behind the tunnel relay one connection is one leaderboard session:
                # whatever the last session left half-done is over.
                self._force_new_episode = True
        self.connected.set()
        try:
            async for message in connection:
                await self._handle_message(connection, message)
        except ConnectionClosed:
            pass
        finally:
            with self._cond:
                self._connections -= 1
                remaining = self._connections
                if self._pending_connection is connection:
                    # The request we were answering has no reader any more; the
                    # client reconnects and resends, so drop it rather than
                    # hand a dead request to the adapter.
                    self._pending = None
                    self._pending_future = None
                    self._pending_connection = None
                self._cond.notify_all()
            if remaining <= 0:
                self.connected.clear()

    async def _handle_message(self, connection: Any, message: Any) -> None:
        if isinstance(message, str):
            await self._send_error(connection, "expected a binary msgpack frame")
            return
        try:
            envelope = msgpack.unpackb(message, raw=False)
        except Exception as exc:
            await self._send_error(connection, f"undecodable msgpack frame: {exc}")
            return
        if not isinstance(envelope, dict):
            await self._send_error(connection, "expected a msgpack map")
            return
        method = envelope.get("method")
        if method != _INFER:
            await self._send_error(connection, f"unsupported method {method!r}")
            return
        with self._cond:
            sequence = self._request_count
        try:
            request = codec.decode_request(
                envelope.get("params") or {},
                decode_images=self._decode_images,
                depth_scales=self._depth_scales,
                sequence=sequence,
                received_at_s=time.time(),
                camera_poses=self._camera_poses,
            )
        except (ValueError, KeyError, TypeError) as exc:
            await self._send_error(connection, f"invalid infer params: {exc}")
            return

        loop = asyncio.get_running_loop()
        with self._cond:
            self._request_count += 1
            self._note_episode(request)
            self._latest = request
            counts = self._episode_counts
            counts["requests"] += 1
            counts["request_bytes"] += len(message)
            if any(request.depths.get(camera) is not None for camera in codec.CAMERAS):
                counts["depth_requests"] += 1
            if request.episode_done:
                # Answered here: the client replays this reply at the start of
                # the next episode, so it must be a single hold.
                self._pending = None
                self._pending_future = None
                self._pending_connection = None
                self._hold_count += 1
                counts["done_holds"] += 1
                self._done_request = request
                result = self._with_need_depth(self._hold_result(request), observe_next=True)
                self.episode_ended.set()
                self._cond.notify_all()
            else:
                future: asyncio.Future[Any] = loop.create_future()
                self._pending = request
                self._pending_future = future
                self._pending_connection = connection
                self._cond.notify_all()
                result = None
        if result is None:
            result = await self._await_reply(request, future)
            if result is None:
                return  # shutting down
        try:
            await connection.send(_packb({"result": result}))
        except ConnectionClosed:
            logger.warning("GenieSim client went away before reply %d was sent", request.sequence)

    def _note_episode(self, request: G2Request) -> None:
        """Called under the lock with the request about to become ``latest``."""
        previous = self._latest
        new_episode = (
            previous is None
            or self._force_new_episode
            or request.episode_idx != previous.episode_idx
            or (self._previous_done and not request.episode_done)
        )
        self._previous_done = request.episode_done
        if new_episode and not request.episode_done:
            self._force_new_episode = False
            if self._episode_index >= 0:
                # Kept for whoever reports on the episode after the next one started.
                self._finished_counts[self._episode_index] = self._episode_counts
                for stale in sorted(self._finished_counts)[:-8]:
                    del self._finished_counts[stale]
            self._episode_index += 1
            self._episode_request = request
            self._last_command = None
            self._episode_counts = self._fresh_counts()
            self.episode_ended.clear()
            self.episode_started.set()

    @staticmethod
    def _fresh_counts() -> dict[str, Any]:
        return {
            "requests": 0,
            "depth_requests": 0,
            "request_bytes": 0,
            "replies": 0,
            "reply_rows": 0,
            "keepalive_holds": 0,
            "done_holds": 0,
            "started_at_s": time.time(),
        }

    def _hold_result(self, request: G2Request, rows: int = 1) -> dict[str, Any]:
        """A ``rows``-long hold: the last commanded row when the profile says so, else ``request``'s joints."""
        last = self._last_command if self._hold_last_command else None
        if last is not None and set(last) >= _REPLY_KEYS:
            result: dict[str, Any] = {}
            for key, value in last.items():
                if isinstance(value, dict):
                    result[key] = {**value, "values": [list(value["values"][-1])] * int(rows)}
                else:
                    result[key] = [list(value[-1])] * int(rows)
            return result
        chunk = codec.hold_chunk(
            request,
            length=int(rows),
            convention=self._convention,
            include_waist=self._include_waist,
        )
        return codec.encode_action_chunk(chunk, convention=self._convention)

    def _with_need_depth(
        self, result: dict[str, Any], *, observe_next: bool | None = None, force: bool | None = None
    ) -> dict[str, Any]:
        """``result`` with ``need_depth`` set by the station's policy (never mutates ``result``).

        ``observe_next`` is the caller's hint that the program will look at the
        images of the next request; ``force`` overrides the policy either way
        (``True``: an observation refresh; ``False``: holds nobody looks at). The
        ``none`` policy never adds the key.
        """
        policy = self._need_depth
        if policy == "none":
            return result
        if force is not None:
            wanted = bool(force)
        elif policy == "always":
            wanted = True
        elif policy == "observe_next":
            wanted = True if observe_next is None else bool(observe_next)
        else:  # on_demand
            wanted = False
        return {**result, "need_depth": wanted}

    @staticmethod
    def _last_row(result: dict[str, Any]) -> dict[str, Any]:
        """The final row of an encoded reply, kept as a 1-row result (``need_depth`` dropped)."""
        last: dict[str, Any] = {}
        for key, value in result.items():
            if key in ("left_arm", "right_arm", "waist") and isinstance(value, dict):
                rows = value.get("values") or []
                if rows:
                    last[key] = {**value, "values": [list(rows[-1])]}
            elif key in ("left_effector", "right_effector") and isinstance(value, list) and value:
                last[key] = [list(value[-1])]
        return last

    async def _await_reply(self, request: G2Request, future: asyncio.Future[Any]) -> Any:
        assert self._stop_event is not None
        stop_waiter = asyncio.ensure_future(self._stop_event.wait())
        try:
            while True:
                timeout = self._reply_timeout_s if self._keepalive_hold else None
                done, _ = await asyncio.wait(
                    {future, stop_waiter}, timeout=timeout, return_when=asyncio.FIRST_COMPLETED
                )
                if future in done:
                    if future.cancelled():
                        return None
                    return future.result()
                if stop_waiter in done:
                    return None
                with self._cond:
                    if self._pending is not request:
                        # Answered or superseded between the timeout and now.
                        continue
                    self._pending = None
                    self._pending_future = None
                    self._pending_connection = None
                    self._hold_count += 1
                    self._episode_counts["keepalive_holds"] += 1
                    # The program may look at the next request once it has thought.
                    result = self._with_need_depth(self._hold_result(request), observe_next=True)
                    self._cond.notify_all()
                logger.warning(
                    "no reply to GenieSim request %d within %.1f s; holding for one tick",
                    request.sequence,
                    self._reply_timeout_s,
                )
                return result
        finally:
            stop_waiter.cancel()

    async def _send_error(self, connection: Any, message: str) -> None:
        with self._cond:
            self._error_count += 1
        logger.warning("GenieSim bridge rejected a message: %s", message)
        with contextlib.suppress(ConnectionClosed):
            await connection.send(_packb({"error": message}))

    # -- adapter side ------------------------------------------------------

    def latest(self) -> G2Request | None:
        """The most recent decoded request, answered or not."""
        with self._cond:
            return self._latest

    def pending(self) -> G2Request | None:
        """The request the client is currently blocked on, if any."""
        with self._cond:
            return self._pending

    def wait_for_request(self, timeout: float | None = None) -> G2Request | None:
        """Block until a request is pending; None on timeout or after close."""
        with self._cond:
            self._cond.wait_for(lambda: self._pending is not None or self._closed, timeout)
            return self._pending

    def wait_for_pending(self, timeout: float | None = None) -> tuple[G2Request | None, int]:
        """Like :meth:`wait_for_request`, plus the episode index read under the same lock.

        The index is the episode the pending request belongs to: a new episode's
        first request advances it before it becomes pending.
        """
        with self._cond:
            self._cond.wait_for(lambda: self._pending is not None or self._closed, timeout)
            return self._pending, self._episode_index

    def hold_result(self, rows: int = 1) -> dict[str, Any]:
        """A ``rows``-long hold for the current episode (same target rule as the keepalive hold)."""
        with self._cond:
            request = self._pending or self._latest
            if request is None:
                raise RuntimeError("no GenieSim request yet: nothing to hold")
            return self._hold_result(request, rows=max(1, int(rows)))

    def episode_stats(self, index: int | None = None) -> dict[str, Any] | None:
        """Counters of episode ``index`` (default: the current one), None once it has aged out.

        Requests, depth requests, request bytes, replies, reply rows, keepalive and
        done holds.
        """
        with self._cond:
            if index is None or index == self._episode_index:
                counts = dict(self._episode_counts)
                current = True
            elif index in self._finished_counts:
                counts = dict(self._finished_counts[index])
                current = False
            else:
                return None
        started = float(counts.pop("started_at_s"))
        counts["elapsed_s"] = round(time.time() - started, 3) if current else None
        return counts

    @property
    def episode_index(self) -> int:
        """Bridge-side count of episodes seen (-1 before the first request)."""
        with self._cond:
            return self._episode_index

    def episode_request(self) -> G2Request | None:
        """The first request of the current episode."""
        with self._cond:
            return self._episode_request

    def done_request(self) -> G2Request | None:
        """The most recent ``episode_done`` request: it carries the final ``task_progress``."""
        with self._cond:
            return self._done_request

    def wait_for_episode(self, after_index: int = -1, timeout: float | None = None) -> int | None:
        """Block until an episode newer than ``after_index`` has started."""
        with self._cond:
            self._cond.wait_for(lambda: self._episode_index > after_index or self._closed, timeout)
            if self._episode_index > after_index:
                return self._episode_index
            return None

    def reply(
        self,
        action_chunk: Any,
        *,
        kind: str = "JOINT_ABS",
        observe_next: bool | None = None,
        force_depth: bool | None = None,
    ) -> dict[str, Any]:
        """Answer the pending request with an ``(H, 16|21)`` chunk; returns the wire result.

        ``observe_next`` / ``force_depth`` feed the station's ``need_depth``
        policy (see :meth:`_with_need_depth`); they change nothing under ``none``.
        """
        result = codec.encode_action_chunk(action_chunk, convention=self._convention, kind=kind)
        return self.reply_raw(result, observe_next=observe_next, force_depth=force_depth)

    def reply_raw(
        self,
        result: dict[str, Any],
        *,
        observe_next: bool | None = None,
        force_depth: bool | None = None,
        record: bool = True,
    ) -> dict[str, Any]:
        """Answer the pending request with an already-encoded ``result`` mapping.

        ``record=False`` keeps the reply out of the "last command" a later hold
        repeats (a hold replying a hold changes nothing either way). Returns what
        was sent, ``need_depth`` included.
        """
        if not isinstance(result, dict):
            raise ValueError("result must be a mapping")
        sent = self._with_need_depth(result, observe_next=observe_next, force=force_depth)
        with self._cond:
            if self._closed:
                raise RuntimeError("the bridge is closed")
            if self._pending is None or self._pending_future is None:
                raise RuntimeError(
                    "no pending GenieSim infer request to answer; the client is not waiting "
                    "for an action (episode ended, not started, or already held)"
                )
            future = self._pending_future
            self._pending = None
            self._pending_future = None
            self._pending_connection = None
            self._reply_count += 1
            counts = self._episode_counts
            counts["replies"] += 1
            rows = (result.get("left_arm") or {}).get("values") or []
            counts["reply_rows"] += len(rows)
            if record:
                self._last_command = self._last_row(result)
            loop = self._loop
            self._cond.notify_all()
        if loop is None:
            raise RuntimeError("the bridge event loop is not running")

        def _resolve() -> None:
            if not future.done():
                future.set_result(sent)

        loop.call_soon_threadsafe(_resolve)
        return sent

    @property
    def need_depth_policy(self) -> str:
        return self._need_depth

    @property
    def new_connection_new_episode(self) -> bool:
        return self._new_connection_new_episode

    def stats(self) -> dict[str, Any]:
        with self._cond:
            return {
                "host": self._host,
                "port": self._bound_port,
                "connected": self._connections > 0,
                "connections": self._connections,
                "connections_total": self._connection_count,
                "requests": self._request_count,
                "replies": self._reply_count,
                "holds": self._hold_count,
                "errors": self._error_count,
                "episode_index": self._episode_index,
                "pending": self._pending is not None,
                "closed": self._closed,
                "need_depth": self._need_depth,
            }


__all__ = ["GenieSimBridge"]
