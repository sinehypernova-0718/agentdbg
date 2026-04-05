"""Benchmarks for event creation and JSON-safety normalization."""

import pytest

from agentdbg.events import EventType, _ensure_json_safe, new_event


# -- new_event ----------------------------------------------------------------


@pytest.mark.benchmark
def test_bench_new_event_simple(benchmark):
    """Create a simple LLM_CALL event with minimal payload."""
    benchmark(
        new_event,
        EventType.LLM_CALL,
        "run-id-placeholder",
        "gpt-4o",
        {"model": "gpt-4o", "prompt": "Hello", "response": "Hi there"},
    )


@pytest.mark.benchmark
def test_bench_new_event_with_meta(benchmark):
    """Create an event with both payload and meta dicts."""
    benchmark(
        new_event,
        EventType.TOOL_CALL,
        "run-id-placeholder",
        "search_db",
        {"tool_name": "search_db", "args": {"query": "users"}, "result": {"count": 42}},
        meta={"framework": "langchain", "version": "0.1"},
    )


# -- _ensure_json_safe --------------------------------------------------------


@pytest.mark.benchmark
def test_bench_ensure_json_safe_flat_dict(benchmark):
    """Normalize a flat dict with primitive values."""
    data = {
        "model": "gpt-4o",
        "prompt": "Hello world",
        "tokens": 150,
        "temperature": 0.7,
        "stream": False,
    }
    benchmark(_ensure_json_safe, data)


@pytest.mark.benchmark
def test_bench_ensure_json_safe_nested_dict(benchmark):
    """Normalize a deeply nested dict structure."""
    data = {
        "level1": {
            "level2": {
                "level3": {
                    "level4": {
                        "value": "deep",
                        "numbers": [1, 2, 3, 4, 5],
                    }
                }
            }
        }
    }
    benchmark(_ensure_json_safe, data)


@pytest.mark.benchmark
def test_bench_ensure_json_safe_large_list(benchmark):
    """Normalize a list with 100 dict entries."""
    data = [{"key": f"item_{i}", "value": i} for i in range(100)]
    benchmark(_ensure_json_safe, data)
