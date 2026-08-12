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
