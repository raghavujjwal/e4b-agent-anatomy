"""Paths and tunables. Everything can be overridden by an environment variable, so the same code
runs on Colab and in WSL.

ROOT  fast local disk: venv templates, pristine task trees, per-run directories.
DATA  where tasks.jsonl, snapshots/ and wheels/ live (a Drive folder on Colab).
"""
from __future__ import annotations
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent


def _path(name: str, default) -> Path:
    return Path(os.environ.get(name, str(default)))


ROOT = _path("GRADER_ROOT", "/content/grader")
DATA = _path("GRADER_DATA", "/content/drive/MyDrive/gemma4-data")
TASKS_JSONL = _path("GRADER_TASKS", DATA / "tasks.jsonl")
SNAPSHOTS = _path("GRADER_SNAPSHOTS", DATA / "snapshots")
WHEELS = _path("GRADER_WHEELS", DATA / "wheels")
# Supplementary wheelhouse for packages the dataset wheelhouse omits (h11, typing_inspection, ...). Copied from
# localeval/extra_wheels on branch `main`. Without it pydantic 2.x cannot install and uv silently picks
# pydantic 1.10, which crashes on Python 3.13.
EXTRA_WHEELS = _path("GRADER_EXTRA_WHEELS", HERE / "extra_wheels")   # shipped in the repo
# interpreter chosen per task (the first of 3.13 / 3.12 / 3.11 under which the gold patch passes)
ENV_MAP = _path("GRADER_ENV_MAP", HERE / "env_map.json")
# The official sandbox/setup.py, imported for its pytest.ini / conftest / git-exclude helpers.
SETUP_PY = _path("GRADER_SETUP_PY", REPO / "rules-raw" / "dataset" / "sandbox_setup.py")
SHIMS = HERE / "shims"

DEFAULT_PYTHON = "3.13"          # the official image
PYTHON_LADDER = ("3.13", "3.12", "3.11")

# HARNESS_README section 4.1 / 8.2
PYTEST_TIMEOUT = 300             # verification pytest timeout on the scorer
COMMAND_TIMEOUT = 300            # single agent command
MEM_GB = 4
CPUS = 2

# Sandbox user for agent and test commands. None = run as the current user.
SANDBOX_USER = os.environ.get("GRADER_SANDBOX_USER") or None
BLOCK_NETWORK = os.environ.get("GRADER_BLOCK_NETWORK", "0") == "1"
# Fall back to PyPI for packages that neither WHEELS nor EXTRA_WHEELS provide.
ALLOW_PYPI = os.environ.get("GRADER_ALLOW_PYPI", "1") == "1"
