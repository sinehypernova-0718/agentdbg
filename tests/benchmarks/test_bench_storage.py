"""Benchmarks for storage utilities: run ID validation and ISO timestamp parsing."""

import pytest

from agentdbg.storage import validate_run_id_format


@pytest.mark.benchmark
def test_bench_validate_run_id_valid(benchmark):
    """Validate a well-formed UUIDv4 run ID."""
    import uuid

    valid_id = str(uuid.uuid4())
    benchmark(validate_run_id_format, valid_id)


@pytest.mark.benchmark
def test_bench_validate_run_id_invalid(benchmark):
    """Validate an invalid run ID (expected to raise ValueError)."""

    def validate_invalid():
        try:
            validate_run_id_format("not-a-valid-uuid")
        except ValueError:
            pass

    benchmark(validate_invalid)
