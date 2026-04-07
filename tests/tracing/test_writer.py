"""Focused unit tests for the hardened async event writer."""

import builtins
import json
import threading

import pytest

from agentdbg._tracing.writer import EventItem, EventQueueWorker
from agentdbg.config import load_config
from agentdbg.events import EventType, new_event
from agentdbg.exceptions import AgentDbgStorageError
from agentdbg.storage import create_run


def test_worker_disables_future_io_after_fatal(temp_data_dir, monkeypatch):
    """A fatal write error disables the worker and blocks future I/O attempts."""
    import agentdbg._tracing.writer as writer_mod

    config = load_config()
    meta = create_run("fatal-disable", config)
    run_id = meta["run_id"]
    path = config.data_dir / "runs" / run_id / "events.jsonl"
    event = new_event(EventType.TOOL_CALL, run_id, "tool", {"tool_name": "tool"})
    worker = EventQueueWorker()

    def boom(event_dict, event_config):
        raise RuntimeError("boom")

    monkeypatch.setattr(writer_mod, "_serialize_event_for_storage", boom)
    worker._handle_events(
        [EventItem(run_id=run_id, path=path, event=event, config=config)],
        pending_after=False,
    )

    with pytest.raises(AgentDbgStorageError, match="background storage worker failed"):
        worker.ensure_healthy()
    assert worker._disabled is True
    assert worker._stopped is True

    def fail_open(*args, **kwargs):
        raise AssertionError("writer should not reopen handles after fatal disable")

    monkeypatch.setattr(builtins, "open", fail_open)
    with pytest.raises(AgentDbgStorageError, match="background storage worker failed"):
        worker._ensure_handle(run_id, path)
    worker._handle_events(
        [EventItem(run_id=run_id, path=path, event=event, config=config)],
        pending_after=False,
    )


def test_ensure_handle_wraps_permission_error(temp_data_dir, monkeypatch):
    """Permission errors are converted into contextual storage failures."""
    config = load_config()
    meta = create_run("locked-file", config)
    run_id = meta["run_id"]
    path = config.data_dir / "runs" / run_id / "events.jsonl"
    worker = EventQueueWorker()

    def locked_open(*args, **kwargs):
        raise PermissionError("locked")

    monkeypatch.setattr(builtins, "open", locked_open)

    with pytest.raises(
        AgentDbgStorageError, match="permission denied opening events file"
    ):
        worker._ensure_handle(run_id, path)


def test_ensure_handle_wraps_directory_race(temp_data_dir, monkeypatch):
    """Directory deletion races surface as contextual storage failures."""
    config = load_config()
    meta = create_run("missing-dir", config)
    run_id = meta["run_id"]
    path = config.data_dir / "runs" / run_id / "events.jsonl"
    worker = EventQueueWorker()

    def missing_dir(self, parents=False, exist_ok=False):
        raise FileNotFoundError("gone")

    monkeypatch.setattr(type(path.parent), "mkdir", missing_dir)

    with pytest.raises(AgentDbgStorageError, match="events directory disappeared"):
        worker._ensure_handle(run_id, path)


def test_shutdown_forces_worker_exit_when_signal_enqueue_fails(monkeypatch):
    """shutdown still stops the worker when the sentinel cannot be enqueued."""
    worker = EventQueueWorker()
    worker.ensure_started()

    def fail_put(*args, **kwargs):
        raise AgentDbgStorageError("queue full")

    monkeypatch.setattr(worker, "_put", fail_put)

    worker.shutdown(timeout_s=1.0)

    assert not worker.is_alive()


def test_run_limits_greedy_batch_size(monkeypatch):
    """The worker never drains more than MAX_BATCH_SIZE items per batch."""
    import agentdbg._tracing.writer as writer_mod

    worker = EventQueueWorker(queue_maxsize=8)
    batch_sizes: list[int] = []
    done = threading.Event()

    def record_batch(batch):
        batch_sizes.append(len(batch))
        done.set()
        return any(item is writer_mod._SHUTDOWN_SIGNAL for item in batch)

    monkeypatch.setattr(writer_mod, "MAX_BATCH_SIZE", 3)
    monkeypatch.setattr(worker, "_handle_batch", record_batch)

    for _ in range(5):
        worker._queue.put(object())
    worker._queue.put(writer_mod._SHUTDOWN_SIGNAL)

    worker.start()
    assert done.wait(1.0)
    worker.join(2.0)

    assert batch_sizes == [3, 3]


def test_handle_events_skips_bad_serializations_without_disabling_worker(
    temp_data_dir, monkeypatch
):
    """One bad event is skipped without disabling later writes for other events."""
    import agentdbg._tracing.writer as writer_mod

    config = load_config()
    meta = create_run("serialize-partial", config)
    run_id = meta["run_id"]
    path = config.data_dir / "runs" / run_id / "events.jsonl"
    original_serialize = writer_mod._serialize_event_for_storage
    worker = EventQueueWorker()

    good_one = new_event(EventType.TOOL_CALL, run_id, "tool", {"tool_name": "good-one"})
    bad_one = new_event(EventType.TOOL_CALL, run_id, "tool", {"tool_name": "bad-one"})
    good_two = new_event(EventType.RUN_END, run_id, "run", {"tool_name": "good-two"})

    def flaky_serialize(event_dict, event_config):
        if event_dict["payload"].get("tool_name") == "bad-one":
            raise RuntimeError("cannot serialize bad event")
        return original_serialize(event_dict, event_config)

    monkeypatch.setattr(writer_mod, "_serialize_event_for_storage", flaky_serialize)

    worker._handle_events(
        [
            EventItem(run_id=run_id, path=path, event=good_one, config=config),
            EventItem(run_id=run_id, path=path, event=bad_one, config=config),
            EventItem(run_id=run_id, path=path, event=good_two, config=config),
        ],
        pending_after=False,
    )

    worker.ensure_healthy()

    lines = path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["payload"]["tool_name"] for line in lines] == [
        "good-one",
        "good-two",
    ]
