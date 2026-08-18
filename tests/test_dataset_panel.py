import pytest

from hydra_suite.utils.geometry_levels import GeometryLevel

pytest.importorskip("PySide6")


@pytest.mark.parametrize(
    "params,expected_enabled",
    [
        (
            {"DETECTION_METHOD": "yolo_obb", "YOLO_OBB_DIRECT_TASK": "segment"},
            {"polygon", "obb", "aabb"},
        ),
        (
            {"DETECTION_METHOD": "yolo_obb", "YOLO_OBB_DIRECT_TASK": "obb"},
            {"obb", "aabb"},
        ),
        (
            {"DETECTION_METHOD": "yolo_obb", "YOLO_OBB_DIRECT_TASK": "detect"},
            {"aabb"},
        ),
    ],
)
def test_level_checkboxes_reflect_detector_capability(params, expected_enabled):
    """Pure logic: which level checkboxes a given detector config enables."""
    from hydra_suite.data.al.escalation import achievable_levels
    from hydra_suite.data.dataset_generation import resolve_native_level

    enabled = {lvl.label for lvl in achievable_levels(resolve_native_level(params))}
    assert enabled == expected_enabled


def test_level_status_text_names_the_missing_requirement():
    from hydra_suite.trackerkit.gui.panels.dataset_panel import format_level_status

    text = format_level_status(GeometryLevel.OBB)
    assert "obb" in text and "aabb" in text
    assert "segmentation" in text.lower()

    assert "polygon" in format_level_status(GeometryLevel.POLYGON)


def test_level_status_text_no_levels_checked_says_nothing_will_export():
    """A deliberately all-unchecked panel must say plainly that nothing will
    be exported, not silently imply the capability-derived default is active."""
    from hydra_suite.trackerkit.gui.panels.dataset_panel import (
        format_level_status,
        level_status_text,
    )

    text = level_status_text(GeometryLevel.OBB, any_checked=False)
    assert "no" in text.lower()
    assert "export" in text.lower()

    # With at least one level checked, falls back to the normal capability text.
    checked_text = level_status_text(GeometryLevel.OBB, any_checked=True)
    assert checked_text == format_level_status(GeometryLevel.OBB)


# =============================================================================
# FINDING 6: the level refresh must not run inside config loading's try
# =============================================================================


class _Combo:
    def __init__(self, index):
        self._index = index

    def currentIndex(self):
        return self._index


class _Check:
    def __init__(self):
        self.enabled = True
        self.checked = True

    def setEnabled(self, v):
        self.enabled = bool(v)

    def setChecked(self, v):
        self.checked = bool(v)

    def isChecked(self):
        return self.checked


class _Label:
    def __init__(self):
        self.text = ""
        self.visible = True

    def setText(self, t):
        self.text = t

    def setVisible(self, v):
        self.visible = bool(v)


def _fake_panel_self(method_index=1, mode_index=0, task_index=2):
    from types import SimpleNamespace

    detection = SimpleNamespace(
        combo_detection_method=_Combo(method_index),
        combo_yolo_obb_mode=_Combo(mode_index),
        combo_yolo_direct_task=_Combo(task_index),
    )

    def _boom():  # pragma: no cover - must never be called
        raise AssertionError("refresh_export_levels called get_parameters_dict()")

    main_window = SimpleNamespace(_detection_panel=detection, get_parameters_dict=_boom)
    from hydra_suite.trackerkit.gui.panels.dataset_panel import DatasetPanel

    fake = SimpleNamespace(
        _main_window=main_window,
        chk_level_polygon=_Check(),
        chk_level_obb=_Check(),
        chk_level_aabb=_Check(),
        lbl_export_level_status=_Label(),
        lbl_bgsub_notice=_Label(),
    )
    fake._detection_level_params = lambda: DatasetPanel._detection_level_params(fake)
    return fake


def test_detection_level_params_reads_the_detection_panel_directly():
    from hydra_suite.trackerkit.gui.panels.dataset_panel import DatasetPanel

    panel = _fake_panel_self(method_index=1, mode_index=1, task_index=2)
    params = DatasetPanel._detection_level_params(panel)
    assert params == {
        "DETECTION_METHOD": "yolo_obb",
        "YOLO_OBB_MODE": "sequential",
        "YOLO_OBB_DIRECT_TASK": "segment",
    }


def test_refresh_export_levels_does_not_build_the_full_param_dict():
    """`get_parameters_dict()` commits pending spinbox edits and can rasterize
    an ROI mask; this refresh runs on every detector and runtime change."""
    from hydra_suite.trackerkit.gui.panels.dataset_panel import DatasetPanel

    panel = _fake_panel_self(method_index=1, mode_index=0, task_index=2)
    DatasetPanel.refresh_export_levels(panel)  # would raise if it built params
    assert panel.chk_level_polygon.enabled is True
    assert "polygon" in panel.lbl_export_level_status.text

    panel = _fake_panel_self(method_index=1, mode_index=0, task_index=0)  # obb
    DatasetPanel.refresh_export_levels(panel)
    assert panel.chk_level_polygon.enabled is False
    assert panel.chk_level_polygon.checked is False


def test_export_level_refresh_cannot_skip_identity_config_loading(tmp_path):
    """A throw in the derived-UI refresh used to unwind into the caller's
    single try/except and silently skip `_load_config_individual_analysis`."""
    import json
    from types import SimpleNamespace

    from hydra_suite.trackerkit.gui.orchestrators.config import ConfigOrchestrator

    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps({"fps": 30.0}))

    orch = object.__new__(ConfigOrchestrator)
    called: list[str] = []

    class _BoomDataset:
        def refresh_export_levels(self):
            called.append("refresh")
            raise RuntimeError("panel not ready")

    orch._panels = SimpleNamespace(dataset=_BoomDataset())
    for name in (
        "_load_config_file_paths",
        "_load_config_reference_params",
        "_load_config_system_performance",
        "_load_config_detection",
        "_load_config_yolo",
        "_load_config_core_tracking",
        "_load_config_orientation_and_lifecycle",
        "_load_config_postprocessing",
        "_load_config_visualization",
        "_load_config_dataset",
        "_load_config_individual_analysis",
    ):
        setattr(orch, name, (lambda n: (lambda *a, **kw: called.append(n)))(name))

    orch._load_config_from_file(str(cfg_path))

    assert "_load_config_individual_analysis" in called
    assert called.index("_load_config_individual_analysis") < called.index("refresh")
