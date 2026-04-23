"""Tests for DatasetPanel widget refactor (source combo + manage signal)."""

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


def test_dataset_panel_has_source_combo(qapp):
    from hydra_suite.detectkit.gui.panels.dataset_panel import DatasetPanel

    panel = DatasetPanel()
    assert hasattr(panel, "source_combo")


def test_dataset_panel_has_manage_btn(qapp):
    from hydra_suite.detectkit.gui.panels.dataset_panel import DatasetPanel

    panel = DatasetPanel()
    assert hasattr(panel, "btn_manage_sources")


def test_dataset_panel_manage_signal(qapp):
    from hydra_suite.detectkit.gui.panels.dataset_panel import DatasetPanel

    panel = DatasetPanel()
    assert hasattr(panel, "manage_sources_requested")


def test_dataset_panel_refresh_sources(qapp, tmp_path):
    from hydra_suite.detectkit.gui.models import OBBSource
    from hydra_suite.detectkit.gui.panels.dataset_panel import DatasetPanel

    panel = DatasetPanel()

    class FakeProj:
        sources = [OBBSource(path=str(tmp_path), name="ds1")]

    panel.refresh_sources(FakeProj())
    assert panel.source_combo.count() == 1


def test_dataset_panel_no_source_list(qapp):
    """Old QListWidget-based source_list must be gone."""
    from hydra_suite.detectkit.gui.panels.dataset_panel import DatasetPanel

    panel = DatasetPanel()
    assert not hasattr(panel, "source_list"), "old source_list widget must not exist"


def test_dataset_panel_navigate_prev_next(qapp):
    from hydra_suite.detectkit.gui.panels.dataset_panel import DatasetPanel

    panel = DatasetPanel()
    # Methods should exist without raising
    panel.navigate_prev()
    panel.navigate_next()


def test_dataset_panel_has_analysis_controls(qapp):
    from hydra_suite.detectkit.gui.panels.dataset_panel import DatasetPanel

    panel = DatasetPanel()
    assert hasattr(panel, "btn_analyze_dataset")
    assert hasattr(panel, "_analysis_view")


def test_dataset_panel_analysis_without_project_shows_message(qapp):
    from hydra_suite.detectkit.gui.panels.dataset_panel import DatasetPanel

    panel = DatasetPanel()
    panel._run_dataset_analysis()
    assert "No dataset sources" in panel._analysis_view.toPlainText()
