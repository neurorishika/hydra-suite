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

from hydra_suite.trackerkit import cli_config  # noqa: E402
from hydra_suite.trackerkit.engine_params import (  # noqa: E402
    RuntimeContext,
    build_engine_params,
)
from hydra_suite.trackerkit.gui.main_window import MainWindow  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES_CONFIG_DIR = REPO_ROOT / "tools" / "equivalence" / "fixtures" / "configs"
GOLDEN_DIR = Path(__file__).resolve().parent / "data" / "get_parameters_dict_golden"

CLIPS = ["fly_obb", "ant_cnn_identity"]

# Task 5: full 7-clip gate matrix, used only by the identity-keys
# characterization guard below (does not need a live MainWindow -- it drives
# ``build_engine_params`` directly, same pattern as
# ``tests/test_engine_params_extraction.py``).
IDENTITY_GATE_CLIPS = [
    "fly_obb",
    "worm_bgsub",
    "emi_obb_identity",
    "ant_pose_headtail",
    "ant_obb_sleap",
    "ant_obb_sequential",
    "ant_cnn_identity",
]

IDENTITY_KEYS = [
    "IDENTITY_DISAGREE_MIN_RUN",
    "IDENTITY_GATES_TRAJECTORY_STRUCTURE",
    "ENABLE_IDENTITY_IN_TRACKING",
    "ENABLE_IDENTITY_ONLINE_DECODER",
    "IDENTITY_POSTPROCESS_MODE",
    "ENABLE_IDENTITY_FRAGMENT_SOLVER",
    "ASSOCIATION_IDENTITY_HINT_SCALE",
    "IDENTITY_COMMIT_THRESHOLD",
    "IDENTITY_DISPLAY_THRESHOLD",
    "IDENTITY_TRANSITION_EPSILON",
    "IDENTITY_UNKNOWN_PRIOR",
    "IDENTITY_REJOIN_THRESHOLD",
    "IDENTITY_SWAP_ENABLED",
    "IDENTITY_SWAP_MIN_FRAMES",
    "IDENTITY_SWAP_CONF_MARGIN",
    "IDENTITY_REJOIN_VELOCITY_BUDGET",
    "IDENTITY_REJOIN_DIST_FLOOR",
    "IDENTITY_CALIBRATION_REQUIRED",
    "IDENTITY_CALIBRATION_OVERRIDE",
]

# Captured from the UN-MODIFIED (pre-Task-5) ``build_engine_params`` via
# ``cli_config.load_tracker_cli_config`` + a synthetic ``RuntimeContext``
# (fps=100.0, total_frames=500, width=640, height=480 -- same probe shape as
# ``tests/test_engine_params_extraction.py``), advanced_config left at its
# default. This pins the pre-refactor baseline so Task 5's swap to
# ``IdentityConfig``-derived reads is provably inert.
EXPECTED_IDENTITY = {
    "fly_obb": {
        "IDENTITY_DISAGREE_MIN_RUN": 5,
        "IDENTITY_GATES_TRAJECTORY_STRUCTURE": True,
        "ENABLE_IDENTITY_IN_TRACKING": False,
        "ENABLE_IDENTITY_ONLINE_DECODER": False,
        "IDENTITY_POSTPROCESS_MODE": "Fragment Solver",
        "ENABLE_IDENTITY_FRAGMENT_SOLVER": True,
        "ASSOCIATION_IDENTITY_HINT_SCALE": 1.0,
        "IDENTITY_COMMIT_THRESHOLD": 0.85,
        "IDENTITY_DISPLAY_THRESHOLD": 0.6,
        "IDENTITY_TRANSITION_EPSILON": 0.02,
        "IDENTITY_UNKNOWN_PRIOR": 0.05,
        "IDENTITY_REJOIN_THRESHOLD": 0.5,
        "IDENTITY_SWAP_ENABLED": True,
        "IDENTITY_SWAP_MIN_FRAMES": 8,
        "IDENTITY_SWAP_CONF_MARGIN": 0.2,
        "IDENTITY_REJOIN_VELOCITY_BUDGET": 1.5,
        "IDENTITY_REJOIN_DIST_FLOOR": None,
        "IDENTITY_CALIBRATION_REQUIRED": False,
        "IDENTITY_CALIBRATION_OVERRIDE": False,
    },
    "worm_bgsub": {
        "IDENTITY_DISAGREE_MIN_RUN": 5,
        "IDENTITY_GATES_TRAJECTORY_STRUCTURE": True,
        "ENABLE_IDENTITY_IN_TRACKING": True,
        "ENABLE_IDENTITY_ONLINE_DECODER": False,
        "IDENTITY_POSTPROCESS_MODE": "Fragment Solver",
        "ENABLE_IDENTITY_FRAGMENT_SOLVER": True,
        "ASSOCIATION_IDENTITY_HINT_SCALE": 1.0,
        "IDENTITY_COMMIT_THRESHOLD": 0.85,
        "IDENTITY_DISPLAY_THRESHOLD": 0.6,
        "IDENTITY_TRANSITION_EPSILON": 0.02,
        "IDENTITY_UNKNOWN_PRIOR": 0.05,
        "IDENTITY_REJOIN_THRESHOLD": 0.5,
        "IDENTITY_SWAP_ENABLED": True,
        "IDENTITY_SWAP_MIN_FRAMES": 8,
        "IDENTITY_SWAP_CONF_MARGIN": 0.2,
        "IDENTITY_REJOIN_VELOCITY_BUDGET": 1.5,
        "IDENTITY_REJOIN_DIST_FLOOR": None,
        "IDENTITY_CALIBRATION_REQUIRED": False,
        "IDENTITY_CALIBRATION_OVERRIDE": False,
    },
    "emi_obb_identity": {
        "IDENTITY_DISAGREE_MIN_RUN": 100,
        "IDENTITY_GATES_TRAJECTORY_STRUCTURE": True,
        "ENABLE_IDENTITY_IN_TRACKING": True,
        "ENABLE_IDENTITY_ONLINE_DECODER": True,
        "IDENTITY_POSTPROCESS_MODE": "Fragment Solver",
        "ENABLE_IDENTITY_FRAGMENT_SOLVER": True,
        "ASSOCIATION_IDENTITY_HINT_SCALE": 0.0,
        "IDENTITY_COMMIT_THRESHOLD": 0.95,
        "IDENTITY_DISPLAY_THRESHOLD": 0.95,
        "IDENTITY_TRANSITION_EPSILON": 0.02,
        "IDENTITY_UNKNOWN_PRIOR": 0.05,
        "IDENTITY_REJOIN_THRESHOLD": 0.95,
        "IDENTITY_SWAP_ENABLED": True,
        "IDENTITY_SWAP_MIN_FRAMES": 8,
        "IDENTITY_SWAP_CONF_MARGIN": 0.2,
        "IDENTITY_REJOIN_VELOCITY_BUDGET": 1.5,
        "IDENTITY_REJOIN_DIST_FLOOR": None,
        "IDENTITY_CALIBRATION_REQUIRED": False,
        "IDENTITY_CALIBRATION_OVERRIDE": False,
    },
    "ant_pose_headtail": {
        "IDENTITY_DISAGREE_MIN_RUN": 5,
        "IDENTITY_GATES_TRAJECTORY_STRUCTURE": True,
        "ENABLE_IDENTITY_IN_TRACKING": True,
        "ENABLE_IDENTITY_ONLINE_DECODER": False,
        "IDENTITY_POSTPROCESS_MODE": "Fragment Solver",
        "ENABLE_IDENTITY_FRAGMENT_SOLVER": True,
        "ASSOCIATION_IDENTITY_HINT_SCALE": 1.0,
        "IDENTITY_COMMIT_THRESHOLD": 0.85,
        "IDENTITY_DISPLAY_THRESHOLD": 0.6,
        "IDENTITY_TRANSITION_EPSILON": 0.02,
        "IDENTITY_UNKNOWN_PRIOR": 0.05,
        "IDENTITY_REJOIN_THRESHOLD": 0.5,
        "IDENTITY_SWAP_ENABLED": True,
        "IDENTITY_SWAP_MIN_FRAMES": 8,
        "IDENTITY_SWAP_CONF_MARGIN": 0.2,
        "IDENTITY_REJOIN_VELOCITY_BUDGET": 1.5,
        "IDENTITY_REJOIN_DIST_FLOOR": None,
        "IDENTITY_CALIBRATION_REQUIRED": False,
        "IDENTITY_CALIBRATION_OVERRIDE": False,
    },
    "ant_obb_sleap": {
        "IDENTITY_DISAGREE_MIN_RUN": 100,
        "IDENTITY_GATES_TRAJECTORY_STRUCTURE": True,
        "ENABLE_IDENTITY_IN_TRACKING": True,
        "ENABLE_IDENTITY_ONLINE_DECODER": True,
        "IDENTITY_POSTPROCESS_MODE": "Fragment Solver",
        "ENABLE_IDENTITY_FRAGMENT_SOLVER": True,
        "ASSOCIATION_IDENTITY_HINT_SCALE": 0.0,
        "IDENTITY_COMMIT_THRESHOLD": 0.95,
        "IDENTITY_DISPLAY_THRESHOLD": 0.95,
        "IDENTITY_TRANSITION_EPSILON": 0.02,
        "IDENTITY_UNKNOWN_PRIOR": 0.05,
        "IDENTITY_REJOIN_THRESHOLD": 0.95,
        "IDENTITY_SWAP_ENABLED": True,
        "IDENTITY_SWAP_MIN_FRAMES": 8,
        "IDENTITY_SWAP_CONF_MARGIN": 0.2,
        "IDENTITY_REJOIN_VELOCITY_BUDGET": 1.5,
        "IDENTITY_REJOIN_DIST_FLOOR": None,
        "IDENTITY_CALIBRATION_REQUIRED": False,
        "IDENTITY_CALIBRATION_OVERRIDE": False,
    },
    "ant_obb_sequential": {
        "IDENTITY_DISAGREE_MIN_RUN": 100,
        "IDENTITY_GATES_TRAJECTORY_STRUCTURE": True,
        "ENABLE_IDENTITY_IN_TRACKING": True,
        "ENABLE_IDENTITY_ONLINE_DECODER": True,
        "IDENTITY_POSTPROCESS_MODE": "Fragment Solver",
        "ENABLE_IDENTITY_FRAGMENT_SOLVER": True,
        "ASSOCIATION_IDENTITY_HINT_SCALE": 0.0,
        "IDENTITY_COMMIT_THRESHOLD": 0.95,
        "IDENTITY_DISPLAY_THRESHOLD": 0.95,
        "IDENTITY_TRANSITION_EPSILON": 0.02,
        "IDENTITY_UNKNOWN_PRIOR": 0.05,
        "IDENTITY_REJOIN_THRESHOLD": 0.95,
        "IDENTITY_SWAP_ENABLED": True,
        "IDENTITY_SWAP_MIN_FRAMES": 8,
        "IDENTITY_SWAP_CONF_MARGIN": 0.2,
        "IDENTITY_REJOIN_VELOCITY_BUDGET": 1.5,
        "IDENTITY_REJOIN_DIST_FLOOR": None,
        "IDENTITY_CALIBRATION_REQUIRED": False,
        "IDENTITY_CALIBRATION_OVERRIDE": False,
    },
    "ant_cnn_identity": {
        "IDENTITY_DISAGREE_MIN_RUN": 5,
        "IDENTITY_GATES_TRAJECTORY_STRUCTURE": True,
        "ENABLE_IDENTITY_IN_TRACKING": True,
        "ENABLE_IDENTITY_ONLINE_DECODER": True,
        "IDENTITY_POSTPROCESS_MODE": "Fragment Solver",
        "ENABLE_IDENTITY_FRAGMENT_SOLVER": True,
        "ASSOCIATION_IDENTITY_HINT_SCALE": 0.05,
        "IDENTITY_COMMIT_THRESHOLD": 0.95,
        "IDENTITY_DISPLAY_THRESHOLD": 0.95,
        "IDENTITY_TRANSITION_EPSILON": 0.02,
        "IDENTITY_UNKNOWN_PRIOR": 0.05,
        "IDENTITY_REJOIN_THRESHOLD": 0.5,
        "IDENTITY_SWAP_ENABLED": True,
        "IDENTITY_SWAP_MIN_FRAMES": 8,
        "IDENTITY_SWAP_CONF_MARGIN": 0.2,
        "IDENTITY_REJOIN_VELOCITY_BUDGET": 1.5,
        "IDENTITY_REJOIN_DIST_FLOOR": None,
        "IDENTITY_CALIBRATION_REQUIRED": False,
        "IDENTITY_CALIBRATION_OVERRIDE": False,
    },
}


def test_identity_keys_byte_identical():
    """Task 5 golden: the ~15 scalar ``IDENTITY_*`` engine keys must stay
    byte-identical across the ``IdentityConfig``-derivation refactor of
    ``build_engine_params``.
    """
    for clip in IDENTITY_GATE_CLIPS:
        cfg = cli_config.load_tracker_cli_config(
            str(FIXTURES_CONFIG_DIR / f"{clip}.json")
        )
        probe = cli_config.TrackerCliVideoProbe(
            fps=100.0, total_frames=500, width=640, height=480
        )
        rt = RuntimeContext(
            fps=probe.fps,
            total_frames=probe.total_frames,
            frame_width=probe.width,
            frame_height=probe.height,
        )
        params = build_engine_params(cfg, runtime=rt)
        got = {k: params[k] for k in IDENTITY_KEYS}
        assert got == EXPECTED_IDENTITY[clip], clip


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
