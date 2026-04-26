"""Smoke tests: each panel instantiates and exposes expected key widgets."""

import os
import sys
from unittest.mock import MagicMock

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QWidget  # noqa: E402

from hydra_suite.trackerkit.config.schemas import TrackerConfig  # noqa: E402

# All panels — used for signal-presence checks.
_PANEL_MAP = {
    "DatasetPanel": "hydra_suite.trackerkit.gui.panels.dataset_panel",
    "DetectionPanel": "hydra_suite.trackerkit.gui.panels.detection_panel",
    "IdentityPanel": "hydra_suite.trackerkit.gui.panels.identity_panel",
    "PostProcessPanel": "hydra_suite.trackerkit.gui.panels.postprocess_panel",
    "SetupPanel": "hydra_suite.trackerkit.gui.panels.setup_panel",
    "TrackingPanel": "hydra_suite.trackerkit.gui.panels.tracking_panel",
}

# All panels are now fully extracted — no stubs remain.
_STUB_PANEL_MAP: dict[str, str] = {}


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def main_window(qapp):
    from hydra_suite.trackerkit.gui.main_window import MainWindow

    w = MainWindow()
    yield w
    w.close()


@pytest.mark.parametrize("class_name,module_path", list(_STUB_PANEL_MAP.items()))
def test_panel_instantiates(qapp, class_name, module_path):
    """Stub panels must instantiate without raising an exception (mock main_window is safe)."""
    import importlib

    mock_mw = MagicMock()
    config = TrackerConfig()
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    panel = cls(main_window=mock_mw, config=config)
    assert isinstance(panel, QWidget)


@pytest.mark.parametrize("class_name,module_path", list(_PANEL_MAP.items()))
def test_panel_has_config_changed_signal(class_name, module_path):
    """Each panel class must declare a config_changed signal."""
    import importlib

    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    assert hasattr(cls, "config_changed")


def test_dataset_panel_wired_in_main_window(main_window):
    """DatasetPanel is accessible on MainWindow and exposes key widgets."""
    from hydra_suite.trackerkit.gui.panels.dataset_panel import DatasetPanel

    assert hasattr(main_window, "_dataset_panel")
    assert isinstance(main_window._dataset_panel, DatasetPanel)
    assert hasattr(main_window._dataset_panel, "chk_enable_dataset_gen")
    assert hasattr(main_window._dataset_panel, "g_oriented_videos")
    assert hasattr(
        main_window._dataset_panel,
        "chk_suppress_foreign_obb_individual_dataset",
    )
    assert hasattr(
        main_window._dataset_panel,
        "chk_suppress_foreign_obb_oriented_videos",
    )
    assert hasattr(
        main_window._dataset_panel,
        "chk_fix_oriented_video_direction_flips",
    )
    assert hasattr(
        main_window._dataset_panel,
        "chk_enable_oriented_video_affine_stabilization",
    )
    assert hasattr(main_window._dataset_panel, "g_downstream_tools")


def test_setup_panel_wired_in_main_window(main_window):
    """SetupPanel is accessible on MainWindow and exposes key widgets."""
    from hydra_suite.trackerkit.gui.panels.setup_panel import SetupPanel

    assert hasattr(main_window, "_setup_panel")
    assert isinstance(main_window._setup_panel, SetupPanel)
    assert hasattr(main_window._setup_panel, "combo_presets")
    assert hasattr(main_window._setup_panel, "btn_file")
    assert not main_window._setup_panel.spin_start_frame.keyboardTracking()
    assert not main_window._setup_panel.spin_end_frame.keyboardTracking()
    assert main_window._setup_panel.slider_timeline.hasTracking() is (
        not sys.platform.startswith("linux")
    )
    assert main_window._setup_panel.spin_traj_hist.minimum() == -1


def test_controls_panel_has_wider_minimum_width(main_window):
    """The right-side controls panel keeps enough width for tab content."""
    assert main_window.splitter.widget(1).minimumWidth() >= 560


def test_tracking_panel_wired_in_main_window(main_window):
    """TrackingPanel is accessible on MainWindow and exposes key widgets."""
    from hydra_suite.trackerkit.gui.panels.tracking_panel import TrackingPanel

    assert hasattr(main_window, "_tracking_panel")
    assert isinstance(main_window._tracking_panel, TrackingPanel)
    assert hasattr(main_window._tracking_panel, "g_density")
    assert hasattr(main_window._tracking_panel, "chk_enable_confidence_density_map")


def test_postprocess_panel_wired_in_main_window(main_window):
    """PostProcessPanel is accessible on MainWindow and exposes key widgets."""
    from hydra_suite.trackerkit.gui.panels.postprocess_panel import PostProcessPanel

    assert hasattr(main_window, "_postprocess_panel")
    assert isinstance(main_window._postprocess_panel, PostProcessPanel)
    assert hasattr(main_window._postprocess_panel, "enable_postprocessing")
    assert hasattr(main_window._postprocess_panel, "combo_interpolation_method")
    assert hasattr(main_window._postprocess_panel, "spin_identity_interpolation_max_gap")
    assert hasattr(main_window._postprocess_panel, "g_refinekit")
    assert hasattr(main_window._postprocess_panel, "chk_prompt_open_refinekit")


def test_identity_panel_wired_in_main_window(main_window):
    """IdentityPanel is accessible on MainWindow and exposes key widgets."""
    from hydra_suite.trackerkit.gui.panels.identity_panel import IdentityPanel

    assert hasattr(main_window, "_identity_panel")
    assert isinstance(main_window._identity_panel, IdentityPanel)
    assert hasattr(main_window._identity_panel, "g_headtail")
    assert hasattr(main_window._identity_panel, "combo_yolo_headtail_model")
    assert hasattr(main_window._identity_panel, "btn_remove_yolo_headtail_model")
    assert hasattr(main_window._identity_panel, "btn_remove_pose_model")


def test_identity_panel_cnn_row_exposes_unique_identifier_toggle(main_window):
    """Each CNN classifier row exposes the unique-identifier toggle."""
    row = main_window._identity_panel._add_cnn_classifier_row()
    try:
        assert hasattr(row, "chk_unique_identifier")
        assert row.chk_unique_identifier.isChecked() is False
    finally:
        main_window._identity_panel._remove_cnn_classifier_row(row)


def test_detection_panel_wired_in_main_window(main_window):
    """DetectionPanel is accessible on MainWindow and exposes key widgets."""
    from hydra_suite.trackerkit.gui.panels.detection_panel import DetectionPanel

    assert hasattr(main_window, "_detection_panel")
    assert isinstance(main_window._detection_panel, DetectionPanel)
    assert hasattr(main_window._detection_panel, "combo_detection_method")
    assert hasattr(main_window._detection_panel, "stack_detection")
    assert hasattr(main_window._detection_panel, "btn_remove_yolo_model")
    assert hasattr(main_window._detection_panel, "btn_remove_yolo_detect_model")
    assert hasattr(main_window._detection_panel, "btn_remove_yolo_crop_obb_model")


def test_preview_detection_context_keeps_identity_overlays_without_master_toggle(
    main_window,
    monkeypatch,
):
    """Preview detection should include configured CNN/AprilTag overlays in YOLO mode."""
    detection_panel = main_window._detection_panel
    identity_panel = main_window._identity_panel

    original_detection_index = detection_panel.combo_detection_method.currentIndex()
    original_identity_enabled = identity_panel.g_identity.isChecked()
    original_apriltags_enabled = identity_panel.g_apriltags.isChecked()

    detection_panel.combo_detection_method.setCurrentIndex(1)
    identity_panel.g_identity.setChecked(False)
    identity_panel.g_apriltags.setChecked(True)

    row = identity_panel._add_cnn_classifier_row()
    monkeypatch.setattr(
        row,
        "to_config",
        lambda: {
            "model_path": "/tmp/cnn_identity.onnx",
            "label": "cnn_identity",
            "confidence": 0.61,
            "window": 5,
            "batch_size": 16,
            "scoring_mode": "atomic",
        },
    )

    try:
        runtime_cfg = detection_panel._identity_config()
        preview_cfg = detection_panel._preview_identity_config()
        preview_context = detection_panel._collect_preview_detection_context()

        assert runtime_cfg == {"use_apriltags": False, "cnn_classifiers": []}
        assert preview_cfg["use_apriltags"] is True
        assert len(preview_cfg["cnn_classifiers"]) == 1
        assert preview_context["use_apriltags"] is True
        assert len(preview_context["cnn_classifiers"]) == 1
        assert preview_context["cnn_classifiers"][0]["label"] == "cnn_identity"
    finally:
        identity_panel._remove_cnn_classifier_row(row)
        identity_panel.g_apriltags.setChecked(original_apriltags_enabled)
        identity_panel.g_identity.setChecked(original_identity_enabled)
        detection_panel.combo_detection_method.setCurrentIndex(original_detection_index)
