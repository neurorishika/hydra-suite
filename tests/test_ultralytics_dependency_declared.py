"""Packaging guard for the Ultralytics MPS target-assignment fix."""

from __future__ import annotations

import tomllib
from pathlib import Path

_FIXED_ULTRALYTICS_REQUIREMENT = "ultralytics>=8.4.138"


def test_ultralytics_mps_target_assigner_fix_is_required() -> None:
    """Fresh installs must not resolve the MPS-indexing-buggy 8.4.34 build."""
    root = Path(__file__).resolve().parents[1]
    config = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))

    assert _FIXED_ULTRALYTICS_REQUIREMENT in config["project"]["dependencies"]
    assert (
        _FIXED_ULTRALYTICS_REQUIREMENT
        in config["project"]["optional-dependencies"]["sam3"]
    )
