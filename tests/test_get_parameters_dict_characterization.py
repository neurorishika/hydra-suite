"""Characterization gate for the ``get_parameters_dict()`` collapse (Task 6).

Before ``get_parameters_dict()`` is rewritten from its ~580-line widget scrape
into a thin wrapper over the shared ``build_engine_params`` + a GUI-only
overlay, this test snapshots its FULL output (every key -- tracking, display,
and runtime-overlay) for the ``fly_obb`` and ``ant_cnn_identity`` gate configs
loaded into a real, offscreen ``MainWindow``. After the rewrite the same call
must reproduce the snapshot key-for-key -- proving the collapse is
behaviour-preserving for the live GUI, including the display + runtime keys the
Task-2 oracle intentionally excludes.

The snapshot (golden) is a pickle written to a FIXED temp path on first run and
compared on every subsequent run. It is intentionally NOT committed: several
keys (``YOLO_DEVICE`` / ``ENABLE_TENSORRT`` / ``ENABLE_GPU_BACKGROUND``) are
host-resolver-dependent, so a committed golden would be platform-locked. The
gate is meaningful within one dev session on one host: capture with the
pre-rewrite code, then re-run against the post-rewrite code. On a fresh host
(temp wiped) the first run writes the golden and passes as a no-op -- the
durable anti-drift guard is the Task-2 oracle, not this file.
"""

import os
import pickle
import tempfile
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

from hydra_suite.trackerkit.gui.main_window import MainWindow  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES_CONFIG_DIR = REPO_ROOT / "tools" / "equivalence" / "fixtures" / "configs"

# Fixed (not per-run) temp path so the golden survives between the pre-rewrite
# capture and the post-rewrite comparison within one dev session.
GOLDEN_PATH = (
    Path(tempfile.gettempdir()) / "hydra_get_params_characterization_golden.pkl"
)

CLIPS = ["fly_obb", "ant_cnn_identity"]


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def main_window(monkeypatch, qapp):
    monkeypatch.setattr(MainWindow, "_save_advanced_config", lambda self: None)
    monkeypatch.setattr(MainWindow, "_load_advanced_config", lambda self: {})
    window = MainWindow()
    try:
        yield window
    finally:
        window.close()


def _load_gate_config(main_window: MainWindow, clip: str) -> None:
    config_path = FIXTURES_CONFIG_DIR / f"{clip}.json"
    assert config_path.is_file(), f"missing gate fixture config: {config_path}"
    main_window._config_orch._load_config_from_file(str(config_path), preset_mode=False)


def _capture_all_params(main_window: MainWindow) -> dict:
    out = {}
    for clip in CLIPS:
        _load_gate_config(main_window, clip)
        out[clip] = dict(main_window._config_orch.get_parameters_dict())
    return out


def _assert_params_equal(clip: str, reference: dict, candidate: dict) -> None:
    ref_keys = set(reference)
    cand_keys = set(candidate)
    assert ref_keys == cand_keys, (
        f"{clip}: key set changed. "
        f"missing={sorted(ref_keys - cand_keys)} "
        f"added={sorted(cand_keys - ref_keys)}"
    )
    mismatches = []
    for key in sorted(ref_keys):
        ref_val = reference[key]
        cand_val = candidate[key]
        if isinstance(ref_val, np.ndarray) or isinstance(cand_val, np.ndarray):
            if not np.array_equal(np.asarray(ref_val), np.asarray(cand_val)):
                mismatches.append((key, ref_val, cand_val))
        elif ref_val != cand_val:
            mismatches.append((key, ref_val, cand_val))
    assert not mismatches, f"{clip}: {len(mismatches)} keys diverged: " + "; ".join(
        f"{k}: ref={r!r} cand={c!r}" for k, r, c in mismatches
    )


def test_get_parameters_dict_matches_characterization_snapshot(main_window):
    """Post-rewrite ``get_parameters_dict()`` must equal the pre-rewrite golden.

    First run (golden absent) captures + writes the golden and passes. Every
    later run compares the live output against the golden, key-for-key.
    """
    candidate = _capture_all_params(main_window)

    if not GOLDEN_PATH.exists():
        GOLDEN_PATH.write_bytes(pickle.dumps(candidate))
        pytest.skip(
            f"characterization golden written to {GOLDEN_PATH} "
            "(first run captures baseline behaviour)"
        )

    # Safe: the golden is written by this same test (self-generated, fixed
    # local temp path), never an untrusted/external source.
    reference = pickle.loads(GOLDEN_PATH.read_bytes())
    for clip in CLIPS:
        assert clip in reference, f"golden missing clip {clip}; delete {GOLDEN_PATH}"
        _assert_params_equal(clip, reference[clip], candidate[clip])
