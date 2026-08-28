"""Smoke tests for the SAM2-escalation GUI entry points (Task 10 wiring)."""

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


def test_tools_panel_exposes_escalation_buttons(qapp):
    from hydra_suite.detectkit.gui.panels.tools_panel import ToolsPanel

    panel = ToolsPanel()
    assert hasattr(panel, "_btn_escalate_sam2")
    assert hasattr(panel, "_btn_mark_reviewed")
    assert hasattr(panel, "escalate_sam2_requested")
    assert hasattr(panel, "mark_reviewed_requested")


def test_main_window_class_has_escalation_handlers():
    from hydra_suite.detectkit.gui.main_window import MainWindow

    assert callable(getattr(MainWindow, "_on_escalate_to_segment_sam2", None))
    assert callable(getattr(MainWindow, "_on_mark_reviewed", None))


def test_escalate_dialog_preselect_source(qapp):
    from hydra_suite.detectkit.gui.dialogs.escalate_sam2_dialog import (
        EscalateSam2Dialog,
    )

    class _Src:
        def __init__(self, name, level="obb"):
            self.name = name
            self.level = level

    dlg = EscalateSam2Dialog([_Src("a"), _Src("b")])
    dlg.preselect_source("b")
    assert dlg.selected_sources() == ["b"]


def test_tools_panel_exposes_review_escalations_button(qapp):
    from hydra_suite.detectkit.gui.panels.tools_panel import ToolsPanel

    panel = ToolsPanel()
    assert hasattr(panel, "_btn_review_escalations")
    assert hasattr(panel, "review_escalations_requested")


def test_main_window_has_review_escalations_handler():
    from hydra_suite.detectkit.gui.main_window import MainWindow

    assert callable(getattr(MainWindow, "_on_review_escalations", None))


def test_main_window_escalation_finish_defers_dialog_past_progress_close():
    """The review dialog (and any post-run message box) must NOT be opened
    from _handle_result, which BaseWorker's result_ready signal can deliver
    on the worker thread, and which fires while the application-modal
    progress dialog is still open -- both would make the dialog undismissable
    or crash Qt. It must be deferred to _finish, which runs after
    progress.close()."""
    import inspect

    from hydra_suite.detectkit.gui.main_window import MainWindow

    source = inspect.getsource(MainWindow._on_escalate_to_segment_sam2)
    assert "_on_review_escalations" in source
    assert 'getattr(result, "staged"' in source

    handle_result_start = source.index("def _handle_result")
    finish_start = source.index("def _finish")
    handle_result_body = source[handle_result_start:finish_start]
    finish_body = source[finish_start:]

    assert "ReviewEscalationsDialog" not in handle_result_body
    assert "QMessageBox" not in handle_result_body
    assert (
        "ReviewEscalationsDialog" in finish_body
        or "_on_review_escalations" in finish_body
    )
