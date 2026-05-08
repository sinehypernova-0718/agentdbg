"""
Shared pytest fixtures and helpers for AgentDbg tests.
"""

import os
import shutil
import uuid
from pathlib import Path

import pytest

from agentdbg.storage import finalize_storage, list_runs


@pytest.fixture
def temp_data_dir():
    """Create a temporary directory and set AGENTDBG_DATA_DIR to it for the test."""
    tmp = Path.cwd() / f".tmp-agentdbg-{uuid.uuid4().hex}"
    tmp.mkdir(parents=True, exist_ok=False)
    old = os.environ.get("AGENTDBG_DATA_DIR")
    try:
        os.environ["AGENTDBG_DATA_DIR"] = str(tmp)
        yield tmp
    finally:
        try:
            finalize_storage()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
            if old is not None:
                os.environ["AGENTDBG_DATA_DIR"] = old
            elif "AGENTDBG_DATA_DIR" in os.environ:
                os.environ.pop("AGENTDBG_DATA_DIR")


def get_latest_run_id(config):
    """
    Return run_id of the most recent run for the given config.

    Use when the test has just created a single run in a temp dir (so the
    latest run is the one we care about). If the code under test starts
    writing multiple runs, prefer selecting by run_name or another stable
    attribute instead.
    """
    runs = list_runs(limit=1, config=config)
    assert runs, "expected at least one run"
    return runs[0]["run_id"]
