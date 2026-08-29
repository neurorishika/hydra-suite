"""Smoke tests for the escalation GUI entry points (Tasks 10-11 wiring).

Task 11 renamed the SAM2 signal/button (geometry escalation) and added a
SAM3 semantic-escalation sibling, and moved the escalate/review handlers out
of MainWindow into module-level functions in ``escalation_actions.py`` --
these tests were updated in that same task to point at the new names and
locations rather than the retired ``MainWindow`` methods.
"""

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
    assert hasattr(panel, "escalate_geometry_requested")
    assert hasattr(panel, "mark_reviewed_requested")


def test_tools_panel_exposes_semantic_escalation_button(qapp):
    from hydra_suite.detectkit.gui.panels.tools_panel import ToolsPanel

    panel = ToolsPanel()
    assert hasattr(panel, "_btn_semantic")
    assert hasattr(panel, "semantic_escalation_requested")


def test_escalation_actions_module_has_handlers():
    from hydra_suite.detectkit.gui import escalation_actions
    from hydra_suite.detectkit.gui.main_window import MainWindow

    assert callable(escalation_actions.on_escalate_geometry)
    assert callable(escalation_actions.on_semantic_escalation)
    assert callable(escalation_actions.on_review_escalations)
    # _on_mark_reviewed was not moved -- it stays a MainWindow method.
    assert callable(getattr(MainWindow, "_on_mark_reviewed", None))
    # The moved methods must no longer exist on MainWindow.
    assert getattr(MainWindow, "_on_escalate_to_segment_sam2", None) is None
    assert getattr(MainWindow, "_on_review_escalations", None) is None


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


def test_escalation_finish_defers_dialog_past_progress_close():
    """The review dialog (and any post-run message box) must NOT be opened
    from _handle_result, which fires while the application-modal progress
    dialog is still open -- a dialog opened there would stack under it and be
    undismissable. It must be deferred to _finish, which runs after
    progress.close()."""
    import inspect

    from hydra_suite.detectkit.gui import escalation_actions

    source = inspect.getsource(escalation_actions.on_escalate_geometry)
    assert "on_review_escalations" in source
    assert 'getattr(result, "staged"' in source

    handle_result_start = source.index("def _handle_result")
    finish_start = source.index("def _finish")
    # Without this, a reordering that puts _finish first would make
    # handle_result_body an empty string and every assertion below pass
    # vacuously -- silently disarming the one guard for this bug class.
    assert finish_start > handle_result_start
    handle_result_body = source[handle_result_start:finish_start]
    finish_body = source[finish_start:]

    assert "ReviewEscalationsDialog" not in handle_result_body
    assert "QMessageBox" not in handle_result_body
    assert (
        "ReviewEscalationsDialog" in finish_body
        or "on_review_escalations" in finish_body
    )


def test_escalation_error_deferred_past_progress_close():
    """A worker error must not pop a QMessageBox directly from the error
    signal connection (which fires before progress.close()) -- it must be
    stashed and shown from _finish, after progress.close(), matching the
    result path's fix in this same task."""
    import inspect

    from hydra_suite.detectkit.gui import escalation_actions

    source = inspect.getsource(escalation_actions.on_escalate_geometry)
    assert "_last_escalation_error" in source

    error_connect_idx = source.index("worker.error.connect(")
    finish_start = source.index("def _finish")

    # The connect statement itself must not construct a QMessageBox inline.
    error_connect_snippet = source[error_connect_idx : error_connect_idx + 120]
    assert "QMessageBox" not in error_connect_snippet

    finish_body = source[finish_start:]
    assert "QMessageBox" in finish_body  # still shown, just deferred here


def test_semantic_escalation_finish_defers_dialog_past_progress_close():
    """Mirrors test_escalation_finish_defers_dialog_past_progress_close for the
    SAM3 path: _handle_result must not pop a blocking dialog while the
    (WindowModal, cancellable) progress dialog is still open -- that has to
    wait for _finish, which runs after progress.close()."""
    import inspect

    from hydra_suite.detectkit.gui import escalation_actions

    source = inspect.getsource(escalation_actions.on_semantic_escalation)

    handle_result_start = source.index("def _handle_result")
    finish_start = source.index("def _finish")
    assert finish_start > handle_result_start
    handle_result_body = source[handle_result_start:finish_start]
    finish_body = source[finish_start:]

    # _handle_result is allowed its own QMessageBox calls today (prompt-failure
    # warning / success info) -- unlike the SAM2 path it does not open a
    # second dialog on top of the progress dialog, so this is not a bug; the
    # regression this guards is specifically that it must never open the
    # (unrelated, heavier) ReviewEscalationsDialog before the progress dialog
    # closes, and that the error path checked below stays deferred.
    assert "ReviewEscalationsDialog" not in handle_result_body
    assert "progress.close()" in finish_body
    assert "window._escalation_worker = None" in finish_body


def test_semantic_escalation_error_deferred_past_progress_close():
    """A SAM3 worker error must not pop a QMessageBox directly from the error
    signal connection (which fires before progress.close()) -- it must be
    stashed and shown from _finish, after progress.close(). This is the exact
    gap a code review caught in Task 11: on_semantic_escalation originally
    connected only result_ready/finished, so an exception mid-run (a
    multi-hour job) silently closed the progress dialog with no error shown."""
    import inspect

    from hydra_suite.detectkit.gui import escalation_actions

    source = inspect.getsource(escalation_actions.on_semantic_escalation)

    # The worker's `error` signal must be connected at all.
    assert "worker.error.connect(" in source
    assert "_last_escalation_error" in source

    error_connect_idx = source.index("worker.error.connect(")
    finish_start = source.index("def _finish")

    # The connect statement itself must not construct a QMessageBox inline.
    error_connect_snippet = source[error_connect_idx : error_connect_idx + 120]
    assert "QMessageBox" not in error_connect_snippet

    finish_body = source[finish_start:]
    assert "progress.close()" in finish_body
    assert "QMessageBox" in finish_body  # still shown, just deferred here
    # Progress dialog handle must be cleared too, matching on_escalate_geometry.
    assert "window._escalation_progress_dialog = None" in finish_body


def test_semantic_escalation_dialog_refuses_empty_prompt_or_no_source(qapp):
    """Headless behavioral check for SemanticEscalationDialog.accept(): an
    empty prompt or no selected source must refuse to close-as-accepted.
    QMessageBox.warning is monkeypatched to a no-op because triggering a real
    modal QMessageBox under QT_QPA_PLATFORM=offscreen opens a real (invisible)
    event loop that hangs waiting for a click that will never come -- the
    same class of hang test_detectkit_review_escalations_dialog.py already
    works around. No dlg.exec() is called anywhere in this test."""
    import hydra_suite.detectkit.gui.dialogs.semantic_escalation_dialog as mod
    from hydra_suite.detectkit.gui.dialogs.semantic_escalation_dialog import (
        SemanticEscalationDialog,
    )

    mod.QMessageBox.warning = staticmethod(lambda *args, **kwargs: None)

    class _Src:
        def __init__(self, name, level="obb", path="/tmp/nonexistent"):
            self.name = name
            self.level = level
            self.path = path

    dlg = SemanticEscalationDialog([_Src("a"), _Src("b")], reference_body_px=40.0)

    # Empty prompt, no source selected: accept() must refuse (dialog stays
    # not-accepted -- QDialog.Accepted == 1).
    dlg._prompt.setText("")
    dlg.accept()
    assert dlg.result() != 1

    # Non-empty prompt but still no source selected: still refused.
    dlg._prompt.setText("ant")
    dlg.accept()
    assert dlg.result() != 1

    # Select a source: now accept() should succeed.
    dlg._list.setCurrentRow(0)
    dlg.accept()
    assert dlg.result() == 1
