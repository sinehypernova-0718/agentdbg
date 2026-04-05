"""Benchmarks for loop detection: signature computation and repeated-pattern detection."""

import pytest

from agentdbg.loopdetect import compute_signature, detect_loop, pattern_key


def _make_event(event_id: str, event_type: str, payload: dict) -> dict:
    return {
        "event_id": event_id,
        "event_type": event_type,
        "payload": payload,
    }


# -- compute_signature --------------------------------------------------------


@pytest.mark.benchmark
def test_bench_compute_signature_llm_call(benchmark):
    event = _make_event("id-0", "LLM_CALL", {"model": "gpt-4o"})
    benchmark(compute_signature, event)


@pytest.mark.benchmark
def test_bench_compute_signature_tool_call(benchmark):
    event = _make_event("id-0", "TOOL_CALL", {"tool_name": "search_db"})
    benchmark(compute_signature, event)


# -- detect_loop --------------------------------------------------------------


def _build_repeating_events(pattern_len: int, repetitions: int) -> list[dict]:
    """Build a list of events forming a repeating pattern."""
    types = ["TOOL_CALL", "LLM_CALL"]
    payloads = [{"tool_name": "foo"}, {"model": "gpt"}]
    events = []
    total = pattern_len * repetitions
    for i in range(total):
        idx = i % pattern_len
        t = types[idx % len(types)]
        p = payloads[idx % len(payloads)]
        events.append(_make_event(f"id-{i}", t, p))
    return events


@pytest.mark.benchmark
def test_bench_detect_loop_small_pattern(benchmark):
    """Detect a 2-event pattern repeated 3 times in a 12-event window."""
    events = _build_repeating_events(pattern_len=2, repetitions=3)
    benchmark(detect_loop, events, window=12, repetitions=3)


@pytest.mark.benchmark
def test_bench_detect_loop_large_window(benchmark):
    """Detect a 2-event pattern repeated 5 times in a 50-event window."""
    events = _build_repeating_events(pattern_len=2, repetitions=5)
    # Add noise at the beginning so the window is larger
    noise = [_make_event(f"noise-{i}", "STATE_UPDATE", {}) for i in range(40)]
    all_events = noise + events
    benchmark(detect_loop, all_events, window=50, repetitions=5)


@pytest.mark.benchmark
def test_bench_detect_loop_no_loop(benchmark):
    """No loop present: all unique signatures."""
    events = [
        _make_event(f"id-{i}", "TOOL_CALL", {"tool_name": f"tool_{i}"})
        for i in range(20)
    ]
    benchmark(detect_loop, events, window=20, repetitions=3)


# -- pattern_key --------------------------------------------------------------


@pytest.mark.benchmark
def test_bench_pattern_key(benchmark):
    payload = {
        "pattern": "TOOL_CALL:foo -> LLM_CALL:gpt",
        "repetitions": 3,
        "window_size": 12,
        "evidence_event_ids": [f"id-{i}" for i in range(6)],
    }
    benchmark(pattern_key, payload)
