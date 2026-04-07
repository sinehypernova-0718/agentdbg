"""Asynchronous background writer for events.jsonl."""

from __future__ import annotations

import atexit
import io
import logging
import os
import queue
import threading
import weakref
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agentdbg.config import AgentDbgConfig
from agentdbg.events import EventType
from agentdbg.exceptions import AgentDbgStorageError

from agentdbg._tracing._redact import _serialize_event_for_storage


logger = logging.getLogger(__name__)

DEFAULT_QUEUE_MAXSIZE = 2048
DEFAULT_ENQUEUE_TIMEOUT_S = 0.05
DEFAULT_BARRIER_TIMEOUT_S = 2.0
DEFAULT_SHUTDOWN_TIMEOUT_S = 2.0
MAX_BATCH_SIZE = 512
_FORCE_STOP_POLL_INTERVAL_S = 0.1

_SHUTDOWN_SIGNAL = object()
# We keep a weak registry so a module-level atexit hook can stop any live workers.
# (The thread is daemon=True; without an explicit shutdown you can lose the tail
# of events on interpreter exit.)
_registered_workers: weakref.WeakSet["EventQueueWorker"] = weakref.WeakSet()
_registered_workers_lock = threading.Lock()


@dataclass(slots=True)
class EventItem:
    """One event waiting to be written to a run's JSONL file."""

    run_id: str
    path: Path
    event: dict[str, Any]
    config: AgentDbgConfig


@dataclass(slots=True)
class BarrierItem:
    """Synchronize producers with the worker after all prior writes are durable."""

    run_id: str | None
    done: threading.Event = field(default_factory=threading.Event)
    error: BaseException | None = None


@dataclass(slots=True)
class CloseHandleItem:
    """Close a run's file handle once prior writes have been processed."""

    run_id: str
    done: threading.Event = field(default_factory=threading.Event)
    error: BaseException | None = None


class EventQueueWorker(threading.Thread):
    """Single consumer that owns all open events.jsonl file handles."""

    def __init__(
        self,
        *,
        queue_maxsize: int = DEFAULT_QUEUE_MAXSIZE,
        enqueue_timeout_s: float = DEFAULT_ENQUEUE_TIMEOUT_S,
    ) -> None:
        super().__init__(name="agentdbg-event-writer", daemon=True)
        self._queue: queue.Queue[object] = queue.Queue(maxsize=queue_maxsize)
        self._enqueue_timeout_s = enqueue_timeout_s
        self._handles: dict[str, io.TextIOWrapper] = {}
        self._fatal_error: AgentDbgStorageError | None = None
        self._fatal_lock = threading.Lock()
        self._start_lock = threading.Lock()
        self._shutdown_requested = threading.Event()
        self._force_stop = threading.Event()
        self._disabled = False
        self._stopped = False
        with _registered_workers_lock:
            _registered_workers.add(self)

    def _get_health_state(self) -> tuple[AgentDbgStorageError | None, bool, bool]:
        """Read the worker's fatal state under lock."""
        with self._fatal_lock:
            return self._fatal_error, self._disabled, self._stopped

    def ensure_started(self) -> None:
        """Start the worker thread on first use."""
        if self.is_alive():
            return
        with self._start_lock:
            if self.is_alive():
                return
            if self._shutdown_requested.is_set():
                raise AgentDbgStorageError("storage worker has already been shut down")
            self.start()

    def ensure_healthy(self) -> None:
        """Raise the fatal worker error if one has already occurred."""
        fatal, disabled, stopped = self._get_health_state()
        if fatal is not None:
            raise fatal
        if disabled or stopped:
            raise AgentDbgStorageError("storage worker is disabled")

    def enqueue_event(self, item: EventItem) -> None:
        """Queue one event for asynchronous persistence."""
        self.ensure_started()
        # Fail fast if the worker already crashed, and check again after the enqueue
        # (the worker can die between those two points).
        self.ensure_healthy()
        self._put(item)
        self.ensure_healthy()

    def flush_run(
        self,
        run_id: str | None,
        timeout_s: float = DEFAULT_BARRIER_TIMEOUT_S,
    ) -> None:
        """Block until all prior writes are flushed/fsynced."""
        self.ensure_healthy()
        if not self.is_alive():
            return
        barrier = BarrierItem(run_id=run_id)
        self._put(barrier)
        if not barrier.done.wait(timeout_s):
            raise AgentDbgStorageError(
                f"timed out waiting for storage flush for run_id={run_id or '*'}"
            )
        if barrier.error is not None:
            raise barrier.error

    def close_run(
        self,
        run_id: str,
        timeout_s: float = DEFAULT_BARRIER_TIMEOUT_S,
    ) -> None:
        """Close a run-specific file handle after pending writes finish."""
        self.ensure_healthy()
        if not self.is_alive():
            return
        item = CloseHandleItem(run_id=run_id)
        self._put(item)
        if not item.done.wait(timeout_s):
            raise AgentDbgStorageError(
                f"timed out waiting to close storage handle for run_id={run_id}"
            )
        if item.error is not None:
            raise item.error

    def shutdown(self, timeout_s: float = DEFAULT_SHUTDOWN_TIMEOUT_S) -> None:
        """Drain the queue, close remaining handles, and stop the worker."""
        if self._shutdown_requested.is_set():
            return
        self._shutdown_requested.set()
        self._force_stop.set()
        if not self.is_alive():
            return
        try:
            # Best effort: if the queue is full we still want the worker to notice
            # _force_stop and exit on its own.
            self._put(_SHUTDOWN_SIGNAL, allow_when_shutting_down=True)
        except AgentDbgStorageError:
            logger.debug("storage shutdown signal could not be enqueued; forcing stop")
        self.join(timeout_s)
        if self.is_alive():
            raise AgentDbgStorageError("timed out waiting for storage worker shutdown")

    def run(self) -> None:
        """Consume queue items until shutdown is requested."""
        try:
            while True:
                try:
                    # Use a timeout so shutdown() can force-stop even if no sentinel
                    # can be enqueued (queue full) or producers are stuck.
                    first = self._queue.get(timeout=_FORCE_STOP_POLL_INTERVAL_S)
                except queue.Empty:
                    if self._force_stop.is_set():
                        break
                    continue
                batch: list[object] = [first]
                if first is not _SHUTDOWN_SIGNAL:
                    # Drain greedily, but cap the size so a burst can't blow up memory.
                    while len(batch) < MAX_BATCH_SIZE:
                        try:
                            nxt = self._queue.get_nowait()
                        except queue.Empty:
                            break
                        batch.append(nxt)
                        if nxt is _SHUTDOWN_SIGNAL:
                            break

                stop_requested = False
                try:
                    stop_requested = self._handle_batch(batch)
                finally:
                    for _ in batch:
                        self._queue.task_done()

                if stop_requested:
                    break
        finally:
            self._close_all_handles()

    def _put(self, item: object, *, allow_when_shutting_down: bool = False) -> None:
        self.ensure_healthy()
        if self._shutdown_requested.is_set() and not allow_when_shutting_down:
            raise AgentDbgStorageError("storage worker is shutting down")
        try:
            self._queue.put(item, timeout=self._enqueue_timeout_s)
        except queue.Full as exc:
            raise AgentDbgStorageError("event queue is full; refusing unbounded growth") from exc

    def _handle_batch(self, batch: list[object]) -> bool:
        """Process a FIFO batch of drained queue items. Returns True when a shutdown signal was encountered."""
        idx = 0
        stop_requested = False
        while idx < len(batch):
            item = batch[idx]
            if item is _SHUTDOWN_SIGNAL:
                stop_requested = True
                break
            if isinstance(item, EventItem):
                run_id = item.run_id
                path = item.path
                # Coalesce consecutive EventItems for the same file. This keeps the
                # common "lots of events for one run" case fast and reduces flush/fsync calls.
                events: list[EventItem] = [item]
                idx += 1
                while idx < len(batch):
                    nxt = batch[idx]
                    if not isinstance(nxt, EventItem):
                        break
                    if nxt.run_id != run_id or nxt.path != path:
                        break
                    events.append(nxt)
                    idx += 1
                # fsync is expensive; only do it when we know the queue is drained.
                pending_after = idx < len(batch) or self._has_pending_outside_batch(len(batch))
                self._handle_events(events, pending_after=pending_after)
                continue
            if isinstance(item, BarrierItem):
                self._handle_barrier(item)
            elif isinstance(item, CloseHandleItem):
                self._handle_close(item)
            idx += 1
        return stop_requested

    def _handle_events(self, items: list[EventItem], *, pending_after: bool) -> None:
        if not items:
            return
        fatal, disabled, stopped = self._get_health_state()
        if fatal is not None or disabled or stopped:
            return

        head = items[0]
        serialization_failure: tuple[EventItem, BaseException] | None = None
        wrote_any = False
        fsync_required = False
        try:
            handle = self._ensure_handle(head.run_id, head.path)
            # Stream writes directly to the handle so large batches do not require
            # building an intermediate concatenated string in memory.
            for item in items:
                fatal, disabled, stopped = self._get_health_state()
                if fatal is not None or disabled or stopped:
                    return
                try:
                    serialized = _serialize_event_for_storage(item.event, item.config)
                except Exception as exc:
                    if serialization_failure is None:
                        serialization_failure = (item, exc)
                    logger.exception(
                        "failed serializing event for run_id=%s path=%s",
                        item.run_id,
                        item.path,
                    )
                    continue
                handle.write(serialized)
                handle.write("\n")
                wrote_any = True
                if self._should_fsync(item.event, pending_after=pending_after):
                    fsync_required = True
            if wrote_any:
                handle.flush()
                if fsync_required:
                    os.fsync(handle.fileno())
        except Exception as exc:
            self._record_fatal(exc, head.run_id, head.path)
            return

        if serialization_failure is not None:
            failed_item, exc = serialization_failure
            self._record_fatal(exc, failed_item.run_id, failed_item.path)

    def _handle_barrier(self, item: BarrierItem) -> None:
        try:
            fatal, disabled, stopped = self._get_health_state()
            if fatal is not None:
                item.error = fatal
                return
            if disabled or stopped:
                item.error = AgentDbgStorageError("storage worker is disabled")
                return
            # This is the producer-facing "make my writes durable" barrier.
            self._fsync_handles(item.run_id)
        except Exception as exc:
            failure = self._record_fatal(exc, item.run_id, None)
            item.error = failure
        finally:
            item.done.set()

    def _handle_close(self, item: CloseHandleItem) -> None:
        try:
            fatal, disabled, stopped = self._get_health_state()
            if fatal is not None:
                item.error = fatal
                return
            if disabled or stopped:
                item.error = AgentDbgStorageError("storage worker is disabled")
                return
            self._close_handle(item.run_id)
        except Exception as exc:
            failure = self._record_fatal(exc, item.run_id, None)
            item.error = failure
        finally:
            item.done.set()

    def _ensure_handle(self, run_id: str, path: Path) -> io.TextIOWrapper:
        # Only the worker thread should call this, but producers can race with
        # fatal-state updates, so we lock around reads/writes of _fatal_error/_disabled/_handles.
        with self._fatal_lock:
            handle = self._handles.get(run_id)
            fatal = self._fatal_error
            disabled = self._disabled
            stopped = self._stopped
            if handle is not None and not handle.closed:
                return handle
            if fatal is not None:
                raise fatal
            if disabled or stopped:
                raise AgentDbgStorageError(
                    f"storage worker disabled before opening handle for run_id={run_id} path={path}"
                )
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except PermissionError as exc:
            raise AgentDbgStorageError(
                f"permission denied creating events directory for run_id={run_id} path={path.parent}"
            ) from exc
        except FileNotFoundError as exc:
            raise AgentDbgStorageError(
                f"events directory disappeared while preparing handle for run_id={run_id} path={path.parent}"
            ) from exc
        try:
            handle = open(path, "a", encoding="utf-8")
        except PermissionError as exc:
            raise AgentDbgStorageError(
                f"permission denied opening events file for run_id={run_id} path={path}"
            ) from exc
        except FileNotFoundError as exc:
            raise AgentDbgStorageError(
                f"events file parent disappeared while opening handle for run_id={run_id} path={path}"
            ) from exc
        with self._fatal_lock:
            fatal = self._fatal_error
            disabled = self._disabled
            stopped = self._stopped
            if fatal is not None or disabled or stopped:
                # Don't leak file descriptors if we lost a fatal race after opening.
                handle.close()
                if fatal is not None:
                    raise fatal
                raise AgentDbgStorageError(
                    f"storage worker disabled before caching handle for run_id={run_id} path={path}"
                )
            self._handles[run_id] = handle
            return handle

    def _should_fsync(self, event: dict[str, Any], *, pending_after: bool) -> bool:
        event_type = str(event.get("event_type") or "")
        return event_type in (EventType.ERROR.value, EventType.RUN_END.value) or not pending_after

    def _fsync_handles(self, run_id: str | None) -> None:
        if run_id is None:
            handles = list(self._handles.values())
        else:
            handle = self._handles.get(run_id)
            handles = [handle] if handle is not None else []
        for handle in handles:
            if handle is None or handle.closed:
                continue
            handle.flush()
            os.fsync(handle.fileno())

    def _close_handle(self, run_id: str) -> None:
        handle = self._handles.pop(run_id, None)
        if handle is None:
            return
        try:
            if not handle.closed:
                handle.flush()
                os.fsync(handle.fileno())
        finally:
            handle.close()

    def _close_all_handles(self) -> None:
        run_ids = list(self._handles)
        for run_id in run_ids:
            try:
                self._close_handle(run_id)
            except Exception:
                logger.exception("failed closing events handle for run_id=%s", run_id)

    def _record_fatal(
        self,
        exc: BaseException,
        run_id: str | None,
        path: Path | None,
    ) -> AgentDbgStorageError:
        # Fatal means "stop doing I/O". We keep the first failure and disable the worker.
        # Handle cleanup happens outside the lock so we don't hold locks across filesystem calls.
        should_close_handles = False
        with self._fatal_lock:
            fatal = self._fatal_error
            if fatal is None:
                target = str(path) if path is not None else "events.jsonl"
                fatal = AgentDbgStorageError(
                    f"background storage worker failed for run_id={run_id or '?'} path={target}"
                )
                fatal.__cause__ = exc
                self._fatal_error = fatal
                self._disabled = True
                self._stopped = True
                should_close_handles = True
            else:
                target = str(path) if path is not None else "events.jsonl"
                self._stopped = True
        if should_close_handles:
            logger.error(
                "background storage worker failed for run_id=%s path=%s",
                run_id,
                target,
                exc_info=(type(exc), exc, exc.__traceback__),
            )
            self._close_all_handles()
        return fatal

    def _has_pending_outside_batch(self, batch_size: int) -> bool:
        return self._queue.unfinished_tasks > batch_size


def _shutdown_registered_workers() -> None:
    with _registered_workers_lock:
        workers = list(_registered_workers)
    for worker in workers:
        try:
            worker.shutdown(timeout_s=DEFAULT_SHUTDOWN_TIMEOUT_S)
        except Exception:
            logger.debug("failed to shut down storage worker during interpreter exit", exc_info=True)


atexit.register(_shutdown_registered_workers)


def event_path_for_run(run_id: str, config: AgentDbgConfig) -> Path:
    """Build the path to events.jsonl without importing storage in the worker."""
    return config.data_dir.expanduser() / "runs" / run_id / "events.jsonl"


def is_agentdbg_event_file(path: Path) -> bool:
    """Return True for the canonical events.jsonl file name."""
    return path.name == "events.jsonl"
