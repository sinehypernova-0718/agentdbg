"""Benchmarks for guardrail checks and parameter merging."""

import pytest

from agentdbg.guardrails import (
    GuardrailParams,
    check_after_event,
    merge_guardrail_params,
)


# -- merge_guardrail_params ---------------------------------------------------


@pytest.mark.benchmark
def test_bench_merge_guardrail_params_no_overrides(benchmark):
    """Merge with no overrides (identity operation)."""
    base = GuardrailParams()
    benchmark(merge_guardrail_params, base)


@pytest.mark.benchmark
def test_bench_merge_guardrail_params_all_overrides(benchmark):
    """Merge with all guardrail fields overridden."""
    base = GuardrailParams()
    benchmark(
        merge_guardrail_params,
        base,
        stop_on_loop=True,
        stop_on_loop_min_repetitions=5,
        max_llm_calls=100,
        max_tool_calls=50,
        max_events=500,
        max_duration_s=120.0,
    )


# -- check_after_event --------------------------------------------------------


@pytest.mark.benchmark
def test_bench_check_after_event_no_guardrails(benchmark):
    """Check with all guardrails disabled (fast path)."""
    params = GuardrailParams()
    event = {"event_type": "LLM_CALL", "payload": {}}
    counts = {"llm_calls": 10, "tool_calls": 5, "errors": 0, "loop_warnings": 0}

    benchmark(
        check_after_event,
        event,
        counts,
        15,
        "2026-01-01T12:00:00.000Z",
        params,
        now_iso="2026-01-01T12:00:05.000Z",
    )


@pytest.mark.benchmark
def test_bench_check_after_event_all_guardrails_within_limits(benchmark):
    """Check with all guardrails enabled but within limits."""
    params = GuardrailParams(
        stop_on_loop=True,
        stop_on_loop_min_repetitions=3,
        max_llm_calls=100,
        max_tool_calls=100,
        max_events=500,
        max_duration_s=300.0,
    )
    event = {"event_type": "LLM_CALL", "payload": {}}
    counts = {"llm_calls": 10, "tool_calls": 5, "errors": 0, "loop_warnings": 0}

    benchmark(
        check_after_event,
        event,
        counts,
        15,
        "2026-01-01T12:00:00.000Z",
        params,
        now_iso="2026-01-01T12:00:05.000Z",
    )
