"""Task 1 of the shared engine-param-builder program: the CLI's

``build_tracking_parameters`` is now a thin shim over the Qt-free
``engine_params.build_engine_params``. This test proves the extraction was a
pure refactor (byte-identical output) and that the new module stays
importable with PySide6 blocked.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from hydra_suite.trackerkit import cli_config
from hydra_suite.trackerkit.engine_params import RuntimeContext, build_engine_params

FIXTURE_CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "equivalence"
    / "fixtures"
    / "configs"
    / "fly_obb.json"
)


@pytest.fixture()
def fly_obb_cfg() -> dict:
    """Real fly_obb tracking config from the equivalence-harness fixtures."""
    assert FIXTURE_CONFIG_PATH.exists(), f"missing fixture: {FIXTURE_CONFIG_PATH}"
    return cli_config.load_tracker_cli_config(str(FIXTURE_CONFIG_PATH))


@pytest.fixture()
def fly_obb_probe() -> cli_config.TrackerCliVideoProbe:
    # fly_obb.json declares fps=100.0, end_frame=499 (500 frames); width/height
    # aren't recorded in the fixture config, so use representative values --
    # the shim/direct-builder equivalence doesn't depend on their specifics.
    return cli_config.TrackerCliVideoProbe(
        fps=100.0, total_frames=500, width=640, height=480
    )


def test_shim_matches_direct_builder(fly_obb_cfg, fly_obb_probe):
    """The CLI shim must reproduce build_engine_params's output exactly."""
    via_shim = cli_config.build_tracking_parameters(
        fly_obb_cfg, video_probe=fly_obb_probe
    )
    rt = RuntimeContext(
        fps=fly_obb_probe.fps,
        total_frames=fly_obb_probe.total_frames,
        frame_width=fly_obb_probe.width,
        frame_height=fly_obb_probe.height,
    )
    via_direct = build_engine_params(fly_obb_cfg, runtime=rt)

    for k in via_shim:
        assert via_shim[k] == via_direct.get(k, "__MISSING__"), f"key {k} diverged"
    # And the reverse: the direct call must not produce keys the shim omits.
    for k in via_direct:
        assert k in via_shim, f"direct-only key {k} missing from shim output"


def test_shim_matches_direct_builder_minimal_config():
    """Same equivalence check on a minimal hand-built config (no fixture I/O)."""
    cfg = {"detection_method": "background_subtraction", "max_targets": 2}
    probe = cli_config.TrackerCliVideoProbe(
        fps=30.0, total_frames=100, width=640, height=480
    )
    via_shim = cli_config.build_tracking_parameters(cfg, video_probe=probe)
    rt = RuntimeContext(
        fps=probe.fps,
        total_frames=probe.total_frames,
        frame_width=probe.width,
        frame_height=probe.height,
    )
    via_direct = build_engine_params(cfg, runtime=rt)
    assert via_shim == via_direct


def test_no_output_dir_keys_emitted(fly_obb_cfg, fly_obb_probe):
    """Task-1 decision #2: output-dir keys stay absent when runtime fields are None."""
    rt = RuntimeContext(
        fps=fly_obb_probe.fps,
        total_frames=fly_obb_probe.total_frames,
        frame_width=fly_obb_probe.width,
        frame_height=fly_obb_probe.height,
    )
    params = build_engine_params(fly_obb_cfg, runtime=rt)
    for key in (
        "DATASET_OUTPUT_DIR",
        "FINAL_MEDIA_EXPORT_VIDEO_OUTPUT_DIR",
        "INDIVIDUAL_DATASET_OUTPUT_DIR",
        "INDIVIDUAL_DATASET_NAME",
        "INDIVIDUAL_DATASET_RUN_ID",
        "INDIVIDUAL_PROPERTIES_CACHE_PATH",
    ):
        assert key not in params, f"unexpected output-dir key emitted: {key}"


def test_dataset_export_knobs_reach_engine_params():
    """Task 15: export levels / dedup / class-names flow from TrackerConfig,
    through its own to_dict round-trip, through the shared build_engine_params --
    never a parallel path."""
    from hydra_suite.trackerkit.config.schemas import TrackerConfig

    tracker_cfg = TrackerConfig()
    tracker_cfg.dataset_export_levels = ["polygon", "obb"]
    tracker_cfg.dataset_dedup_method = "dhash"
    tracker_cfg.dataset_dedup_threshold = 12
    tracker_cfg.dataset_class_names = "ant, larva"

    cfg = tracker_cfg.to_dict()
    rt = RuntimeContext(fps=30.0, total_frames=100, frame_width=640, frame_height=480)
    params = build_engine_params(cfg, runtime=rt)

    assert params["DATASET_EXPORT_LEVELS"] == ["polygon", "obb"]
    assert params["DATASET_DEDUP_METHOD"] == "dhash"
    assert params["DATASET_DEDUP_THRESHOLD"] == 12
    assert params["DATASET_CLASS_NAMES"] == ["ant", "larva"]


def test_dataset_class_names_falls_back_to_single_class_name():
    """No dataset_class_names -> falls back to the legacy single class name."""
    rt = RuntimeContext(fps=30.0, total_frames=100, frame_width=640, frame_height=480)
    cfg = {"dataset_class_name": "bee", "dataset_class_names": ""}
    params = build_engine_params(cfg, runtime=rt)
    assert params["DATASET_CLASS_NAMES"] == ["bee"]


def test_dataset_class_names_falls_back_to_object_when_nothing_set():
    rt = RuntimeContext(fps=30.0, total_frames=100, frame_width=640, frame_height=480)
    params = build_engine_params({}, runtime=rt)
    assert params["DATASET_CLASS_NAMES"] == ["object"]


def test_dataset_export_knob_defaults():
    rt = RuntimeContext(fps=30.0, total_frames=100, frame_width=640, frame_height=480)
    params = build_engine_params({}, runtime=rt)
    assert params["DATASET_EXPORT_LEVELS"] == ["polygon", "obb", "aabb"]
    assert params["DATASET_DEDUP_METHOD"] == "phash"
    assert params["DATASET_DEDUP_THRESHOLD"] == 8
    assert params["DATASET_DETECTKIT_PROJECT"] == ""


def test_tracker_config_dataset_fields_round_trip():
    from hydra_suite.trackerkit.config.schemas import TrackerConfig

    cfg = TrackerConfig()
    cfg.dataset_export_levels = ["obb"]
    cfg.dataset_dedup_method = "dhash"
    cfg.dataset_dedup_threshold = 4
    cfg.dataset_class_names = "x,y"
    cfg.dataset_detectkit_project = "proj1"

    restored = TrackerConfig.from_dict(cfg.to_dict())
    assert restored.dataset_export_levels == ["obb"]
    assert restored.dataset_dedup_method == "dhash"
    assert restored.dataset_dedup_threshold == 4
    assert restored.dataset_class_names == "x,y"
    assert restored.dataset_detectkit_project == "proj1"

    # And a fresh default TrackerConfig round-trips the documented defaults.
    default_restored = TrackerConfig.from_dict(TrackerConfig().to_dict())
    assert default_restored.dataset_export_levels == ["polygon", "obb", "aabb"]
    assert default_restored.dataset_dedup_method == "phash"
    assert default_restored.dataset_dedup_threshold == 8
    assert default_restored.dataset_class_names == ""
    assert default_restored.dataset_detectkit_project == ""


def test_engine_params_module_is_qt_free(monkeypatch):
    """engine_params.py must import cleanly with PySide6 unimportable."""
    monkeypatch.setitem(sys.modules, "PySide6", None)
    for mod_name in list(sys.modules):
        if mod_name == "hydra_suite.trackerkit.engine_params" or mod_name.startswith(
            "hydra_suite.trackerkit.engine_params."
        ):
            monkeypatch.delitem(sys.modules, mod_name, raising=False)
    import importlib

    module = importlib.import_module("hydra_suite.trackerkit.engine_params")
    assert hasattr(module, "build_engine_params")
    assert hasattr(module, "RuntimeContext")
    assert hasattr(module, "build_roi_mask")
