"""Characterization gate for the ``get_parameters_dict()`` collapse (Task 6).

``get_parameters_dict()`` is a thin wrapper over the shared
``build_engine_params`` + a GUI-only display overlay + a GUI-only runtime
overlay (``_gui_display_overlay`` / ``_gui_runtime_context`` in
``trackerkit/gui/orchestrators/config.py``). The Task-2 param-equality oracle
(``tests/test_gui_cli_param_equivalence.py``) proves the GUI and CLI agree on
*tracking* params, but it intentionally EXCLUDES the display keys (``SHOW_*``,
``zoom_factor``, ``VISUALIZATION_FREE_MODE``, ``TRACKING_REALTIME_MODE``,
``TRACKING_WORKFLOW_MODE``, ``TRAJECTORY_COLORS``) and the runtime-overlay
keys (``ROI_MASK``, ``START_FRAME``, ``END_FRAME``, the output-dir/cache-path
keys) that only the GUI wrapper produces.

This test closes that gap: it snapshots the FULL ``get_parameters_dict()``
output -- every key, including the ones the oracle excludes -- for the
``fly_obb`` and ``ant_cnn_identity`` gate configs loaded into a real,
offscreen ``MainWindow``, and compares against a COMMITTED golden captured
from known-correct (behavior-preserving, gate-verified) code. A future edit
to either overlay -- or to ``build_engine_params`` -- that changes any
non-excluded key's value will fail this test.

Host-dependent keys are dropped from the golden rather than pinned:

* ``ROI_MASK`` is an ndarray-or-None; both sides are normalized to
  ``None`` or ``{"__ndarray_shape__": [...]}`` before comparing (see
  ``_normalize_roi_mask``).
* Model-path keys whose value embeds this machine's absolute
  ``HYDRA_DATA_DIR``/home-relative model path are dropped entirely -- see
  ``HOST_DEPENDENT_DROPPED_KEYS`` below. These were determined empirically
  (see fix report) by diffing a live capture against the string "contains an
  absolute path" predicate; every other key (all tracking keys, ALL display
  keys, START_FRAME/END_FRAME, POSE_*, identity keys, the output-dir keys
  which are empty strings for these fixtures) is kept and asserted verbatim.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

from hydra_suite.trackerkit.gui.main_window import MainWindow  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES_CONFIG_DIR = REPO_ROOT / "tools" / "equivalence" / "fixtures" / "configs"
GOLDEN_DIR = Path(__file__).resolve().parent / "data" / "get_parameters_dict_golden"

CLIPS = ["fly_obb", "ant_cnn_identity"]

# Keys dropped from the golden because their VALUE embeds an absolute,
# machine-specific filesystem path (the resolved model file under
# HYDRA_DATA_DIR / the platformdirs models directory). Determined empirically
# by capturing get_parameters_dict() on a live host and flagging every string
# value containing the repo root or home directory prefix. Every other key,
# including all other model-identifying keys, is committed to the golden.
HOST_DEPENDENT_DROPPED_KEYS = {
    "YOLO_MODEL_PATH",
    "YOLO_OBB_DIRECT_MODEL_PATH",
    "YOLO_HEADTAIL_MODEL_PATH",
    "POSE_MODEL_DIR",
}

# Keys whose VALUE is derived by the runtime resolver from the detected
# platform/accelerator (mps vs cuda vs cpu), NOT from the GUI param wrapper.
# The golden was captured on the hydra-mps host; these would legitimately
# differ on a CUDA/CPU box or CI, so they are excluded to keep the golden
# host-portable. The resolver logic that produces them is covered by the
# runtime/resolver tests, not by this GUI-wrapper characterization guard.
RESOLVER_DEPENDENT_DROPPED_KEYS = {
    "YOLO_DEVICE",
    "ENABLE_GPU_BACKGROUND",
    "ENABLE_TENSORRT",
    "ENABLE_ONNX_RUNTIME",
    "TENSORRT_MAX_BATCH_SIZE",
}

DROPPED_KEYS = HOST_DEPENDENT_DROPPED_KEYS | RESOLVER_DEPENDENT_DROPPED_KEYS


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


def _normalize_roi_mask(value):
    if value is None:
        return None
    if isinstance(value, dict) and "__ndarray_shape__" in value:
        # Already-normalized golden value (loaded from JSON).
        return tuple(value["__ndarray_shape__"])
    arr = np.asarray(value)
    return tuple(arr.shape)


def _normalize_params(params: dict) -> dict:
    """Drop host-dependent keys and normalize ``ROI_MASK`` for comparison.

    Also round-trips every remaining value through JSON so tuple/list
    differences between the live (in-memory) candidate and the
    JSON-deserialized golden don't produce spurious mismatches.
    """
    normalized = {}
    for key, value in params.items():
        if key in DROPPED_KEYS:
            continue
        if key == "ROI_MASK":
            normalized[key] = _normalize_roi_mask(value)
            continue
        normalized[key] = json.loads(json.dumps(value, default=str))
    return normalized


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
        if ref_val != cand_val:
            mismatches.append((key, ref_val, cand_val))
    assert not mismatches, f"{clip}: {len(mismatches)} keys diverged: " + "; ".join(
        f"{k}: ref={r!r} cand={c!r}" for k, r, c in mismatches
    )


def test_get_parameters_dict_matches_committed_golden(main_window):
    """``get_parameters_dict()`` must equal the committed golden, key-for-key.

    The golden is captured from known-correct, gate-verified behavior (see
    the fix report referenced in the module docstring) and committed under
    ``tests/data/get_parameters_dict_golden/``. This test never skips: a
    missing golden file is a hard failure, not a "first run, capture and
    pass" no-op, so it durably guards the display + runtime overlay keys the
    Task-2 params-equality oracle excludes.
    """
    candidate = _capture_all_params(main_window)

    for clip in CLIPS:
        golden_path = GOLDEN_DIR / f"{clip}.json"
        assert golden_path.is_file(), f"missing committed golden: {golden_path}"
        reference = json.loads(golden_path.read_text())
        ref_norm = _normalize_params(reference)
        cand_norm = _normalize_params(candidate[clip])
        _assert_params_equal(clip, ref_norm, cand_norm)
