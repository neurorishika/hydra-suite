"""Tests for DetectKit ToolsPanel."""

from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication(sys.argv)
    yield app


def _make_proj(tmp_path):
    from hydra_suite.detectkit.gui.models import DetectKitProject, OBBSource

    proj = DetectKitProject(project_dir=tmp_path, class_names=["ant", "bee"])
    proj.sources = [
        OBBSource(path=str(tmp_path / "dataset1"), name="ds1"),
        OBBSource(path=str(tmp_path / "dataset2"), name="ds2"),
    ]
    return proj


def test_tools_panel_imports(qapp):
    pass  # noqa: F401


def test_overlay_settings_namedtuple():
    from hydra_suite.detectkit.gui.panels.tools_panel import OverlaySettings

    s = OverlaySettings(
        show_gt=True,
        show_pred=False,
        show_derived_levels=True,
        show_escalation=True,
        confidence_threshold=0.5,
        visible_class_ids=set(),
        active_model_path="",
    )
    assert s.show_gt is True
    assert s.show_pred is False
    assert s.show_derived_levels is True


def test_tools_panel_has_show_derived_levels_checkbox(qapp):
    from hydra_suite.detectkit.gui.panels.tools_panel import ToolsPanel

    panel = ToolsPanel()
    assert hasattr(panel, "_chk_show_derived_levels")


def test_overlay_settings_includes_show_derived_levels_default_true(qapp):
    from hydra_suite.detectkit.gui.panels.tools_panel import ToolsPanel

    panel = ToolsPanel()
    settings = panel.get_overlay_settings()
    assert settings.show_derived_levels is True


def test_overlay_settings_reflects_unchecked_show_derived_levels(qapp):
    from hydra_suite.detectkit.gui.panels.tools_panel import ToolsPanel

    panel = ToolsPanel()
    panel._chk_show_derived_levels.setChecked(False)
    settings = panel.get_overlay_settings()
    assert settings.show_derived_levels is False


def test_main_window_on_overlay_changed_wires_derived_levels_to_canvas():
    import inspect

    from hydra_suite.detectkit.gui.main_window import MainWindow

    source = inspect.getsource(MainWindow._on_overlay_changed)
    assert "set_derived_levels_visible" in source


def test_tools_panel_is_resizable_with_readable_minimum(qapp):
    from hydra_suite.detectkit.gui.panels.tools_panel import ToolsPanel

    panel = ToolsPanel()
    assert panel.minimumWidth() == 300
    assert panel.maximumWidth() > panel.minimumWidth()
    assert panel.sizePolicy().horizontalPolicy().name == "Preferred"


def test_tools_panel_uses_compact_escalation_button_labels(qapp):
    from hydra_suite.detectkit.gui.panels.tools_panel import ToolsPanel

    panel = ToolsPanel()
    assert panel._btn_escalate_sam2.text() == "Geometry escalation (SAM2)…"
    assert panel._btn_semantic.text() == "Semantic escalation (SAM3)…"


def test_tools_panel_set_project(qapp, tmp_path):
    from hydra_suite.detectkit.gui.panels.tools_panel import ToolsPanel

    panel = ToolsPanel()
    proj = _make_proj(tmp_path)
    panel.set_project(proj)  # Must not raise


def test_tools_panel_project_switch_replaces_the_previous_active_model(qapp, tmp_path):
    from hydra_suite.detectkit.gui.panels.tools_panel import ToolsPanel

    first = _make_proj(tmp_path / "first")
    first.active_model_path = "/models/first.pt"
    second = _make_proj(tmp_path / "second")

    panel = ToolsPanel()
    panel.set_project(first)
    panel.set_active_model_path(first.active_model_path)
    panel.set_project(second)

    assert panel.get_overlay_settings().active_model_path == ""


def test_model_status_text_never_becomes_part_of_the_model_path(qapp):
    from hydra_suite.detectkit.gui.panels.tools_panel import ToolsPanel

    panel = ToolsPanel()
    panel.set_active_model_path("/models/detect.pt", status="missing OBB head")

    assert panel.get_overlay_settings().active_model_path == "/models/detect.pt"
    assert "missing OBB head" in panel._model_display.text()


def test_main_window_passes_incomplete_model_status_separately_from_path():
    import inspect

    from hydra_suite.detectkit.gui.main_window import MainWindow

    load_source = inspect.getsource(MainWindow._load_project)
    history_source = inspect.getsource(MainWindow._open_history_dialog)
    assert 'status="missing OBB head"' in load_source
    assert 'status="missing OBB head"' in history_source
    assert "active_model_path} (missing OBB head)" not in load_source
    assert "active_model_path} (missing OBB head)" not in history_source


def test_tools_panel_refresh_overview(qapp, tmp_path):
    from hydra_suite.detectkit.gui.panels.tools_panel import ToolsPanel

    panel = ToolsPanel()
    proj = _make_proj(tmp_path)
    panel.set_project(proj)
    panel.refresh_overview()  # Must not raise


def test_tools_panel_refresh_model_selector(qapp):
    from hydra_suite.detectkit.gui.panels.tools_panel import ToolsPanel

    panel = ToolsPanel()
    panel.refresh_model_selector(["/some/model.pt", "/other/model.pt"])
    # First path auto-selected when none is currently active.
    assert panel._active_model_path == "/some/model.pt"


def test_tools_panel_get_overlay_settings_default(qapp):
    from hydra_suite.detectkit.gui.panels.tools_panel import OverlaySettings, ToolsPanel

    panel = ToolsPanel()
    settings = panel.get_overlay_settings()
    assert isinstance(settings, OverlaySettings)
    assert settings.show_gt is True
    assert settings.show_pred is True


def test_tools_panel_overlay_gt_toggle(qapp):
    from hydra_suite.detectkit.gui.panels.tools_panel import ToolsPanel

    panel = ToolsPanel()
    panel._chk_show_gt.setChecked(False)
    s = panel.get_overlay_settings()
    assert s.show_gt is False


def test_tools_panel_overlay_pred_toggle(qapp):
    from hydra_suite.detectkit.gui.panels.tools_panel import ToolsPanel

    panel = ToolsPanel()
    panel._chk_show_pred.setChecked(False)
    s = panel.get_overlay_settings()
    assert s.show_pred is False


def test_tools_panel_confidence_threshold(qapp):
    from hydra_suite.detectkit.gui.panels.tools_panel import ToolsPanel

    panel = ToolsPanel()
    # Slider range 0–100 maps to 0.0–1.0
    panel._conf_slider.setValue(75)
    s = panel.get_overlay_settings()
    assert abs(s.confidence_threshold - 0.75) < 0.01


def test_tools_panel_class_checkboxes_after_set_project(qapp, tmp_path):
    from hydra_suite.detectkit.gui.panels.tools_panel import ToolsPanel

    panel = ToolsPanel()
    proj = _make_proj(tmp_path)
    panel.set_project(proj)
    # Should have checkboxes for "ant" and "bee"
    assert len(panel._class_checkboxes) == 2


def test_tools_panel_collapsible_section(qapp):
    from PySide6.QtWidgets import QLabel

    from hydra_suite.detectkit.gui.panels.tools_panel import _CollapsibleSection

    section = _CollapsibleSection("Test Section")
    content = QLabel("hello")
    section.set_content(content)
    # Initially collapsed
    assert not section.is_expanded()
    section.toggle()
    assert section.is_expanded()
    section.toggle()
    assert not section.is_expanded()


def test_tools_panel_has_overview_progress(qapp):
    from hydra_suite.detectkit.gui.panels.tools_panel import ToolsPanel

    panel = ToolsPanel()
    assert hasattr(panel, "_overview_progress")


def test_overlay_settings_includes_show_escalation_default_true(qapp):
    """Default ON: a pending escalation is transient and awaiting review, so
    the whole point is to see it without hunting for a toggle."""
    from hydra_suite.detectkit.gui.panels.tools_panel import ToolsPanel

    panel = ToolsPanel()
    assert panel.get_overlay_settings().show_escalation is True
    panel._chk_show_escalation.setChecked(False)
    assert panel.get_overlay_settings().show_escalation is False


def test_main_window_on_overlay_changed_wires_escalation_to_canvas():
    import inspect

    from hydra_suite.detectkit.gui.main_window import MainWindow

    source = inspect.getsource(MainWindow._on_overlay_changed)
    assert 'set_layer_visible("staged"' in source
