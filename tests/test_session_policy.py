"""Pure policy predicates agree with the GUI widget-reading methods across all fixture configs."""

import json
import os
import pathlib

import pytest

from hydra_suite.core.tracking import session_policy as sp

FIXTURES = sorted(
    (
        pathlib.Path(__file__).resolve().parents[1]
        / "tools"
        / "equivalence"
        / "fixtures"
        / "configs"
    ).glob("*.json")
)

PREDICATES = [
    "is_individual_pipeline_enabled",
    "is_pose_inference_enabled",
    "is_headtail_compute_enabled",
    "should_export_final_canonical_images",
    "should_export_final_media_videos",
    "should_run_interpolated_postpass",
    "is_pose_export_enabled",
]


@pytest.mark.parametrize("cfg_path", FIXTURES, ids=lambda p: p.stem)
def test_predicates_callable_and_boolean(cfg_path):
    cfg = json.loads(cfg_path.read_text())
    for name in PREDICATES:
        assert isinstance(getattr(sp, name)(cfg), bool), name
    assert sp.workflow_mode_key(cfg) in ("realtime", "non_realtime")


def test_fly_obb_expected_values():
    cfg = json.loads((FIXTURES[0].parent / "fly_obb.json").read_text())
    # fly_obb: yolo_obb detection, no pose extractor, no identity.
    assert sp.is_individual_pipeline_enabled(cfg) is True
    assert sp.is_pose_export_enabled(cfg) is False
    assert sp.is_pose_inference_enabled(cfg) is False
    # NOTE: the fly_obb fixture has "realtime_tracking_mode": true (verified in
    # tools/equivalence/fixtures/configs/fly_obb.json), so the correct expected
    # value is "realtime", not "non_realtime" as the task brief's test literal
    # claimed. workflow_mode_key is a straight pass-through of that flag.
    assert sp.workflow_mode_key(cfg) == "realtime"


def test_build_trajectory_colors_pinned_values():
    assert sp.build_trajectory_colors(3) == [
        (102, 179, 92),
        (14, 106, 71),
        (188, 20, 102),
    ]


def test_build_trajectory_colors_does_not_leak_global_seed():
    import numpy as np

    state_before = np.random.get_state()
    sp.build_trajectory_colors(5)
    state_after = np.random.get_state()
    # The saved/restored guard must leave the global RNG state exactly as found.
    assert state_before[0] == state_after[0]
    assert (state_before[1] == state_after[1]).all()


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    import PySide6  # noqa: F401

    _HAS_QT = True
except ImportError:
    _HAS_QT = False


if _HAS_QT:
    from PySide6.QtWidgets import QApplication

    from hydra_suite.trackerkit.gui.main_window import MainWindow

    @pytest.fixture(scope="module")
    def qapp() -> QApplication:
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        return app

    @pytest.fixture
    def qtbot_config_stub(monkeypatch, qapp):
        """A real, offscreen MainWindow (no disk/network I/O), per test_config_build_dict.py."""
        monkeypatch.setattr(MainWindow, "_save_advanced_config", lambda self: None)
        monkeypatch.setattr(MainWindow, "_load_advanced_config", lambda self: {})
        window = MainWindow()
        try:
            yield window
        finally:
            window.close()

    GUI_METHOD_MAP = {
        "is_individual_pipeline_enabled": "_is_individual_pipeline_enabled",
        "is_pose_inference_enabled": "_is_pose_inference_enabled",
        "is_headtail_compute_enabled": "_is_headtail_compute_enabled",
        "should_export_final_canonical_images": "_should_export_final_canonical_images",
        "should_export_final_media_videos": "_should_export_final_media_videos",
        "should_run_interpolated_postpass": "_should_run_interpolated_postpass",
        "is_pose_export_enabled": "_is_pose_export_enabled",
        "workflow_mode_key": "_workflow_mode_key",
    }

    @pytest.mark.parametrize("cfg_path", FIXTURES, ids=lambda p: p.stem)
    def test_predicates_agree_with_gui(qtbot_config_stub, cfg_path):
        window = qtbot_config_stub
        window._config_orch._load_config_from_file(str(cfg_path))
        cfg = window._config_orch.build_config_dict()

        for pure_name, gui_name in GUI_METHOD_MAP.items():
            pure_fn = getattr(sp, pure_name)
            if gui_name == "_is_pose_export_enabled":
                gui_value = window._is_pose_export_enabled()
            else:
                gui_value = getattr(window._session_orch, gui_name)()
            assert (
                pure_fn(cfg) == gui_value
            ), f"{pure_name} disagrees with {gui_name} for {cfg_path.stem}"
