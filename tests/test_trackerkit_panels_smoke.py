"""Smoke tests: each panel instantiates and exposes expected key widgets."""

import os
import sys

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

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
@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def main_window(qapp):
    from hydra_suite.trackerkit.gui.main_window import MainWindow

    w = MainWindow()
    yield w
    w.close()


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
    assert hasattr(main_window._setup_panel, "combo_inference_autotune")
    assert hasattr(main_window._setup_panel, "spin_inference_autotune_budget")
    assert hasattr(main_window._setup_panel, "lbl_inference_autotune_status")
    assert hasattr(main_window._setup_panel, "btn_continue_inference_settings")
    assert not main_window._setup_panel.btn_continue_inference_settings.isVisible()


def test_setup_inference_autotune_policy_persists_and_status_is_read_only(main_window):
    """The one UI control owns policy; runtime outcomes only update its label.

    ``TrackerConfig`` now stores one boolean (``apply_tuned_inference``); the
    combo box's "record" and "automatic" entries both persist as ``True`` --
    they no longer round-trip as distinct values (see
    ``.superpowers/sdd/2026-09-08-oneclick-inference-calibration``: "record"
    is retired, "calibrate without applying" is now expressed by calibrating
    and leaving apply off). This test still exercises the status-label text
    for both, since that copy is unchanged, but persists the boolean.
    """
    panel = main_window._setup_panel
    original_index = panel.combo_inference_autotune.currentIndex()
    try:
        panel._set_inference_autotune_combo_mode("record")
        panel._on_inference_autotune_mode_changed(
            panel.combo_inference_autotune.currentIndex()
        )
        config = main_window._config_orch.build_config_dict()
        assert config["apply_tuned_inference"] is True
        assert "Record-only" in panel.lbl_inference_autotune_status.text()

        panel._set_inference_autotune_combo_mode("automatic")
        panel._on_inference_autotune_mode_changed(
            panel.combo_inference_autotune.currentIndex()
        )
        config = main_window._config_orch.build_config_dict()
        assert config["apply_tuned_inference"] is True
        assert "validated profile" in panel.lbl_inference_autotune_status.text()

        panel.set_inference_autotune_status("Cache hit — profile abc123")
        assert panel.combo_inference_autotune.currentData() == "automatic"
        assert (
            panel.lbl_inference_autotune_status.text() == "Cache hit — profile abc123"
        )
    finally:
        panel.combo_inference_autotune.setCurrentIndex(original_index)


def test_setup_inference_autotune_combo_signal_is_actually_connected(main_window):
    """IMPORTANT 2 (round 1 review): the previous test only ever called
    ``_on_inference_autotune_mode_changed`` by hand, so a severed
    ``currentIndexChanged.connect(...)`` in ``setup_panel.py`` would still
    pass it. Drive the combo box the way a real user does -- an unblocked
    ``setCurrentIndex`` -- and observe the persisted config change through
    that signal alone, so a disconnected wire fails this test.

    "off" is the only combo entry that persists ``apply_tuned_inference`` as
    ``False``; "record" and "automatic" both persist ``True``.
    """
    panel = main_window._setup_panel
    original_index = panel.combo_inference_autotune.currentIndex()
    try:
        off_index = panel.combo_inference_autotune.findData("off")
        assert off_index >= 0
        panel.combo_inference_autotune.setCurrentIndex(off_index)
        config = main_window._config_orch.build_config_dict()
        assert config["apply_tuned_inference"] is False

        record_index = panel.combo_inference_autotune.findData("record")
        assert record_index >= 0
        panel.combo_inference_autotune.setCurrentIndex(record_index)
        config = main_window._config_orch.build_config_dict()
        assert config["apply_tuned_inference"] is True

        automatic_index = panel.combo_inference_autotune.findData("automatic")
        panel.combo_inference_autotune.setCurrentIndex(automatic_index)
        config = main_window._config_orch.build_config_dict()
        assert config["apply_tuned_inference"] is True
    finally:
        panel.combo_inference_autotune.setCurrentIndex(original_index)


def test_setup_inference_autotune_budget_spinbox_persists(main_window):
    """The bounded calibration-time budget (previously widget-less) must be
    both visible and persisted through the config dict."""
    panel = main_window._setup_panel
    original = panel.spin_inference_autotune_budget.value()
    try:
        panel.spin_inference_autotune_budget.setValue(45.0)
        panel._on_inference_autotune_budget_changed(45.0)
        config = main_window._config_orch.build_config_dict()
        assert config["inference_autotune_budget_seconds"] == 45.0
    finally:
        panel.spin_inference_autotune_budget.setValue(original)


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
    assert hasattr(main_window._postprocess_panel, "spin_changepoint_penalty")
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
        saved_cfg = main_window._config_orch.build_config_dict()

        assert runtime_cfg == {"use_apriltags": False, "cnn_classifiers": []}
        assert saved_cfg["enable_identity_analysis"] is False
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
