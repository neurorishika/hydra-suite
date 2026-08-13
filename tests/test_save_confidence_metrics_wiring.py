"""Historical wiring guard for the (now-retired) confidence-metrics toggle.

A 2026-08-11 user decision retired ``save_confidence_metrics`` entirely: the
tracking worker (``core/tracking/worker.py``) now *always* computes and
appends the three confidence columns to every CSV row, unconditionally. The
``SAVE_CONFIDENCE_METRICS`` param key is no longer read anywhere in the
worker (see ``docs/superpowers/sdd/2026-08-11-trackerkit-debug-mode``,
Task 1). The CLI/GUI param plumbing (``cli_config.py``, ``engine_params.py``)
still carries the lowercase field/uppercase key for now -- that surface is
retired later in the Debug Mode work (Task 8 removes the GUI checkbox) -- but
nothing downstream gates on it any more.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from hydra_suite.trackerkit.cli_config import (
    TrackerCliVideoProbe,
    load_tracker_cli_session,
)

_PROBE = TrackerCliVideoProbe(fps=30.0, total_frames=120, width=640, height=480)


def _session(tmp_path, *, save_confidence: bool):
    return load_tracker_cli_session(
        str(tmp_path / "clip.mp4"),
        config_data={
            "file_path": str(tmp_path / "clip.mp4"),
            "fps": 30.0,
            "save_confidence_metrics": save_confidence,
        },
        video_probe=_PROBE,
    )


@pytest.mark.parametrize("save_confidence", [True, False])
def test_cli_params_carry_the_toggle(tmp_path, save_confidence):
    session = _session(tmp_path, save_confidence=save_confidence)
    assert session.params["SAVE_CONFIDENCE_METRICS"] is save_confidence


def test_cli_params_agree_with_the_header_field(tmp_path):
    """The row gate and the header must read the same decision."""
    for choice in (True, False):
        session = _session(tmp_path, save_confidence=choice)
        assert session.save_confidence_metrics is choice
        assert session.params["SAVE_CONFIDENCE_METRICS"] is choice


def _repo_src() -> Path:
    return Path(__file__).resolve().parents[1] / "src" / "hydra_suite"


def _assigned_string_keys(path: Path) -> set[str]:
    """Every string key assigned in a dict literal or by subscript in *path*."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    keys: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for k in node.keys:
                if isinstance(k, ast.Constant) and isinstance(k.value, str):
                    keys.add(k.value)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Subscript)
                    and isinstance(target.slice, ast.Constant)
                    and isinstance(target.slice.value, str)
                ):
                    keys.add(target.slice.value)
    return keys


def test_gui_parameter_builder_also_writes_the_key():
    """The GUI path must not regress to the CLI-only fix.

    Asserted structurally rather than by constructing a MainWindow: this repo
    has known modal-dialog hangs that prevent GUI tests from completing.
    """
    # The shared param-builder refactor moved the SAVE_CONFIDENCE_METRICS
    # assignment out of the GUI orchestrator into the shared engine-params
    # builder (trackerkit/engine_params.py), which the GUI path feeds through.
    builder = _repo_src() / "trackerkit" / "engine_params.py"
    assert "SAVE_CONFIDENCE_METRICS" in _assigned_string_keys(builder)


def test_worker_no_longer_gates_on_the_toggle():
    """Task 1 of the Debug Mode work retires the row-level gate entirely.

    The worker must always compute and emit the three confidence columns;
    ``SAVE_CONFIDENCE_METRICS``/``save_confidence`` must not appear anywhere
    in worker.py any more (see task-1 brief verification requirement).
    """
    worker = _repo_src() / "core" / "tracking" / "worker.py"
    text = worker.read_text(encoding="utf-8")
    assert "SAVE_CONFIDENCE_METRICS" not in text
    assert "save_confidence" not in text
