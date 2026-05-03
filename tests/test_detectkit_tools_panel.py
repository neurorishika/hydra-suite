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
        confidence_threshold=0.5,
        visible_class_ids=set(),
        active_model_path="",
    )
    assert s.show_gt is True
    assert s.show_pred is False
    assert s.confidence_threshold == 0.5


def test_tools_panel_fixed_width(qapp):
    from hydra_suite.detectkit.gui.panels.tools_panel import ToolsPanel

    panel = ToolsPanel()
    assert panel.maximumWidth() == 280
    assert panel.minimumWidth() == 280


def test_tools_panel_signals(qapp):
    from hydra_suite.detectkit.gui.panels.tools_panel import ToolsPanel

    panel = ToolsPanel()
    assert hasattr(panel, "overlay_settings_changed")
    assert hasattr(panel, "run_inference_requested")
    assert hasattr(panel, "prev_requested")
    assert hasattr(panel, "next_requested")
    assert hasattr(panel, "train_requested")
    assert hasattr(panel, "evaluate_requested")
    assert hasattr(panel, "history_requested")


def test_tools_panel_has_overlay_action_buttons(qapp):
    from hydra_suite.detectkit.gui.panels.tools_panel import ToolsPanel

    panel = ToolsPanel()
    assert hasattr(panel, "_btn_run_inference")
    assert hasattr(panel, "_btn_overlay_evaluate")


def test_tools_panel_set_project(qapp, tmp_path):
    from hydra_suite.detectkit.gui.panels.tools_panel import ToolsPanel

    panel = ToolsPanel()
    proj = _make_proj(tmp_path)
    panel.set_project(proj)  # Must not raise


def test_tools_panel_refresh_overview(qapp, tmp_path):
    from hydra_suite.detectkit.gui.panels.tools_panel import ToolsPanel

    panel = ToolsPanel()
    proj = _make_proj(tmp_path)
    panel.set_project(proj)
    panel.refresh_overview()  # Must not raise


def test_tools_panel_set_image_counter(qapp):
    from hydra_suite.detectkit.gui.panels.tools_panel import ToolsPanel

    panel = ToolsPanel()
    panel.set_image_counter(3, 10)  # Must not raise


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


def test_tools_panel_update_model_metrics(qapp):
    from hydra_suite.detectkit.gui.panels.tools_panel import ToolsPanel

    panel = ToolsPanel()
    panel.update_model_metrics({"mAP50": 0.85, "mAP50-95": 0.62})  # Must not raise
