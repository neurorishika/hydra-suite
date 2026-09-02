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
    # on_review_escalations was retired by the frame-granular review task:
    # its one irreplaceable feature (SAM3 re-thresholding) moved to the
    # review bar, and "jump to a staged review" is now
    # MainWindow._on_go_to_staged_review.
    assert getattr(escalation_actions, "on_review_escalations", None) is None
    # _on_mark_reviewed was not moved -- it stays a MainWindow method.
    assert callable(getattr(MainWindow, "_on_mark_reviewed", None))
    # The moved methods must no longer exist on MainWindow.
    assert getattr(MainWindow, "_on_escalate_to_segment_sam2", None) is None
    assert getattr(MainWindow, "_on_review_escalations", None) is None
    assert callable(getattr(MainWindow, "_on_go_to_staged_review", None))


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
    assert "_on_go_to_staged_review" in source
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
        or "_on_go_to_staged_review" in finish_body
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

    # Fixed by this task: _handle_result used to pop the prompt-failure
    # warning / success info QMessageBox directly, stacking it under the
    # still-open progress dialog -- the same bug class already fixed for
    # worker.error. Both must now be deferred to _finish, exactly like
    # on_escalate_geometry's result path.
    assert "ReviewEscalationsDialog" not in handle_result_body
    assert "QMessageBox" not in handle_result_body
    assert "window._last_escalation_result = result" in handle_result_body
    assert "QMessageBox" in finish_body
    assert "is_prompt_failure" in finish_body
    assert "progress.close()" in finish_body
    assert "window._escalation_worker = None" in finish_body


def test_semantic_escalation_result_messages_deferred_past_progress_close():
    """Specific regression guard for the Task-11-review fix: both the
    prompt-failure warning and the success info box must come from _finish,
    strictly after progress.close(), not from _handle_result. Reverting the
    fix (moving either QMessageBox call back into _handle_result) must break
    this test."""
    import inspect

    from hydra_suite.detectkit.gui import escalation_actions

    source = inspect.getsource(escalation_actions.on_semantic_escalation)

    finish_start = source.index("def _finish")
    finish_body = source[finish_start:]
    close_idx = finish_body.index("progress.close()")

    prompt_failure_idx = finish_body.index("matched nothing on")
    success_idx = finish_body.index("{len(result.staged)} source(s) over")

    # Both result-driven messages must textually appear after progress.close()
    # within _finish -- i.e. cannot fire before the progress dialog closes.
    assert prompt_failure_idx > close_idx
    assert success_idx > close_idx


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


def test_semantic_escalation_dialog_refuses_empty_prompt_or_no_source(
    qapp, monkeypatch
):
    """Headless behavioral check for SemanticEscalationDialog.accept(): an
    empty prompt or no selected source must refuse to close-as-accepted.
    QMessageBox.warning is monkeypatched to a no-op because triggering a real
    modal QMessageBox under QT_QPA_PLATFORM=offscreen opens a real (invisible)
    event loop that hangs waiting for a click that will never come -- the
    same class of hang test_detectkit_review_escalations_dialog.py already
    works around. No dlg.exec() is called anywhere in this test.

    The patch goes through `monkeypatch`, NOT a bare assignment onto the real
    PySide6 QMessageBox class: that assignment was never undone and silenced
    QMessageBox.warning for every later test module in the same pytest
    process."""
    import hydra_suite.detectkit.gui.dialogs.semantic_escalation_dialog as mod
    from hydra_suite.detectkit.gui.dialogs.semantic_escalation_dialog import (
        SemanticEscalationDialog,
    )

    monkeypatch.setattr(mod.QMessageBox, "warning", staticmethod(lambda *a, **k: None))
    # The dialog now probes the checkpoint on accept() (C1); a machine without
    # the 3.45 GB weights must not turn that into a modal question here.
    monkeypatch.setattr(
        mod, "probe_checkpoint", lambda *a, **k: _Availability(True, "")
    )

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


from hydra_suite.detectkit.gui.dialogs.semantic_escalation_dialog import (  # noqa: E402
    SemanticEscalationDialog,
)


class _Src:
    def __init__(self, name, level="obb", path="/tmp/nonexistent"):
        self.name = name
        self.level = level
        self.path = path


class _Availability:
    """Stand-in for checkpoints.Sam3Availability in dialog tests."""

    def __init__(self, usable, reason, checkpoint_missing=False):
        self.usable = usable
        self.reason = reason
        self.checkpoint_missing = checkpoint_missing

    @property
    def actionable(self):
        return self.usable or self.checkpoint_missing


def test_semantic_dialog_asks_before_a_missing_checkpoint_download(qapp, monkeypatch):
    """C1: the 3.45 GB download is surfaced BEFORE the run starts.

    Previously the tools-panel button was disabled whenever the checkpoint
    was absent, and the only place that could have offered the download was
    this dialog -- behind that disabled button. So the feature was
    unreachable on any machine without a pre-placed checkpoint.
    """
    import hydra_suite.detectkit.gui.dialogs.semantic_escalation_dialog as mod
    from hydra_suite.core.inference.semantic.checkpoints import CHECKPOINT_SIZE_GB

    monkeypatch.setattr(
        mod,
        "probe_checkpoint",
        lambda *a, **k: _Availability(False, "not here yet", checkpoint_missing=True),
    )
    asked = []
    monkeypatch.setattr(
        mod.QMessageBox,
        "question",
        staticmethod(
            lambda parent, title, text, *a, **k: (
                asked.append(text),
                mod.QMessageBox.No,
            )[1]
        ),
    )
    monkeypatch.setattr(mod.QMessageBox, "warning", staticmethod(lambda *a, **k: None))

    dlg = SemanticEscalationDialog([_Src("a")], reference_body_px=40.0)
    # The pending download is visible in the dialog, with its size, before
    # anything is clicked.
    assert f"{CHECKPOINT_SIZE_GB:.2f} GB" in dlg._checkpoint_note.text()

    dlg._prompt.setText("ant")
    dlg._list.setCurrentRow(0)
    dlg.accept()
    assert asked, "accept() must ask before starting a 3.45 GB download"
    assert f"{CHECKPOINT_SIZE_GB:.2f}" in asked[0]
    assert dlg.result() != 1, "declining the download must not start the run"

    # Confirming lets the run start.
    monkeypatch.setattr(
        mod.QMessageBox, "question", staticmethod(lambda *a, **k: mod.QMessageBox.Yes)
    )
    dlg.accept()
    assert dlg.result() == 1


def test_semantic_dialog_still_refuses_a_broken_install(qapp, monkeypatch):
    """A missing DEPENDENCY is not a missing checkpoint: no download offer."""
    import hydra_suite.detectkit.gui.dialogs.semantic_escalation_dialog as mod

    monkeypatch.setattr(
        mod, "probe_checkpoint", lambda *a, **k: _Availability(False, "no ftfy")
    )
    warned = []
    monkeypatch.setattr(
        mod.QMessageBox,
        "warning",
        staticmethod(lambda parent, title, text, *a, **k: warned.append(text)),
    )
    monkeypatch.setattr(
        mod.QMessageBox,
        "question",
        staticmethod(
            lambda *a, **k: (_ for _ in ()).throw(
                AssertionError("must not offer a download")
            )
        ),
    )
    dlg = SemanticEscalationDialog([_Src("a")], reference_body_px=40.0)
    dlg._prompt.setText("ant")
    dlg._list.setCurrentRow(0)
    dlg.accept()
    assert dlg.result() != 1
    assert warned and "no ftfy" in warned[0]


def test_tools_panel_enables_the_button_when_only_the_checkpoint_is_missing(
    qapp, monkeypatch
):
    """C1 at the other end: the button must not gate on `usable` alone."""
    import hydra_suite.core.inference.semantic.checkpoints as ck
    from hydra_suite.detectkit.gui.panels.tools_panel import ToolsPanel

    monkeypatch.setattr(
        ck,
        "probe_availability",
        lambda *a, **k: ck.Sam3Availability(
            False, "not downloaded yet", checkpoint_missing=True
        ),
    )
    panel = ToolsPanel()
    assert panel._btn_semantic.isEnabled() is True
    assert "download" in panel._btn_semantic.toolTip().lower()

    monkeypatch.setattr(
        ck, "probe_availability", lambda *a, **k: ck.Sam3Availability(False, "no ftfy")
    )
    broken = ToolsPanel()
    assert broken._btn_semantic.isEnabled() is False
    assert broken._btn_semantic.toolTip() == "no ftfy"


def test_semantic_dialog_exposes_an_editable_reference_body_size(qapp, monkeypatch):
    """I6 link 3: the chain used to dead-end with no way to enter a value."""
    import hydra_suite.detectkit.gui.dialogs.semantic_escalation_dialog as mod

    monkeypatch.setattr(
        mod, "probe_checkpoint", lambda *a, **k: _Availability(True, "")
    )
    dlg = SemanticEscalationDialog(
        [_Src("a")],
        reference_body_px=0.0,
        body_px_origin="nothing found — enter one, or tiling stays off",
    )
    assert dlg.parameters()["reference_body_px"] == 0.0
    assert "tiling is off" in dlg._tile_label.text()
    assert "nothing found" in dlg._body_origin_label.text()

    dlg._reference_body.setValue(80.0)
    assert dlg.parameters()["reference_body_px"] == 80.0
    # 80 px body / 0.05 fraction = 1600 px tiles: tiling is now ON.
    assert "1600 px" in dlg._tile_label.text()


def test_semantic_dialog_preview_button_is_connected(qapp, monkeypatch):
    """I3: the button was created, tooltipped, laid out -- and never wired.

    The shipped user guide tells users to press it.
    """
    import hydra_suite.detectkit.gui.dialogs.semantic_escalation_dialog as mod

    monkeypatch.setattr(
        mod, "probe_checkpoint", lambda *a, **k: _Availability(True, "")
    )
    dlg = SemanticEscalationDialog([_Src("a")], reference_body_px=40.0)
    assert dlg._btn_preview.text() == "Test random image…"
    assert "complete image" in dlg._btn_preview.toolTip()
    # The button really is connected: disconnecting succeeds (it raises
    # RuntimeError on an unconnected signal, which is how the dead button
    # would be caught).
    dlg._btn_preview.clicked.disconnect()

    calls = []
    monkeypatch.setattr(dlg, "_run_preview", lambda: calls.append(1))
    dlg._btn_preview.clicked.connect(dlg._run_preview)
    dlg._btn_preview.click()
    assert calls == [1]

    # And it reaches _run_preview specifically, not some unrelated slot.
    import inspect

    assert "_run_preview" in inspect.getsource(mod.SemanticEscalationDialog.__init__)
    assert "FramePreviewWorker" in inspect.getsource(
        mod.SemanticEscalationDialog._run_preview
    )


# --- I2 / I9 / I6: the semantic handler's seams ------------------------------


def test_semantic_handler_never_hardcodes_overwrite_true():
    """I2: `overwrite=True` disarmed the job's already-pending guard.

    A resume needs no overwrite (the job compares staging directories), so a
    hardcoded True only ever meant "silently wipe whatever else is staged" --
    including an unreviewed SAM2 escalation or a previous prompt's SAM3
    result. The behavioural half of this is pinned in
    tests/test_semantic_escalation_job.py
    (test_a_different_pending_escalation_is_skipped_without_overwrite,
    test_an_unreviewed_sam2_escalation_is_not_silently_destroyed); this pins
    that the GUI asks rather than deciding for the user.
    """
    import inspect

    from hydra_suite.detectkit.gui import escalation_actions

    source = inspect.getsource(escalation_actions.on_semantic_escalation)
    assert "overwrite=True" not in source
    assert "sources_pending_replacement" in source
    # The consent, and the fact overwrite is only set after it.
    question_idx = source.index("QMessageBox.question")
    set_idx = source.index("request.overwrite = True")
    assert set_idx > question_idx
    assert "will be destroyed" in source


def test_semantic_handler_reports_a_cancelled_run_distinctly():
    """I9: a cancelled run was reported as unqualified success."""
    import inspect

    from hydra_suite.detectkit.gui import escalation_actions

    source = inspect.getsource(escalation_actions.on_semantic_escalation)
    assert "result.cancelled" in source
    assert "CANCELLED" in source


def test_semantic_handler_refreshes_the_overview_like_the_sam2_path():
    import inspect

    from hydra_suite.detectkit.gui import escalation_actions

    source = inspect.getsource(escalation_actions.on_semantic_escalation)
    assert "window._tools_panel.refresh_overview()" in source


def test_reference_body_px_resolution_chain(tmp_path):
    """I6: project setting -> median of existing labels -> the user."""
    import cv2
    import numpy as np

    from hydra_suite.detectkit.gui.escalation_actions import resolve_reference_body_px
    from hydra_suite.detectkit.gui.models import OBBSource

    root = tmp_path / "sources" / "s"
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir(parents=True)
    cv2.imwrite(
        str(root / "images" / "f0.png"), np.zeros((400, 400, 3), dtype=np.uint8)
    )
    src = OBBSource(path=str(root), name="s", level="obb")

    class _Slice:
        reference_body_px = 0.0

    class _Project:
        project_dir = str(tmp_path)
        sources = [src]
        slice_settings = _Slice()

    project = _Project()

    # Link 3: nothing resolves -- and the dialog must say so, not proceed
    # quietly with tiling off.
    value, origin = resolve_reference_body_px(project)
    assert value == 0.0
    assert "enter one" in origin

    # Link 2: the median longest side of the source's existing labels.
    # 400 px frame, AABB w=0.2 -> 80 px.
    (root / "labels" / "f0.txt").write_text("0 0.5 0.5 0.2 0.05\n")
    value, origin = resolve_reference_body_px(project)
    assert value == pytest.approx(80.0, abs=1.0)
    assert "existing labels" in origin

    # Link 1: the project setting wins outright.
    _Slice.reference_body_px = 55.0
    value, origin = resolve_reference_body_px(project)
    assert value == 55.0
    assert "sliced-training" in origin


def test_reference_body_px_reads_the_real_project_slice_settings(tmp_path):
    from hydra_suite.detectkit.gui.escalation_actions import resolve_reference_body_px
    from hydra_suite.detectkit.gui.models import DetectKitProject, SliceTrainingSettings

    project = DetectKitProject(
        project_dir=tmp_path,
        slice_settings=SliceTrainingSettings(reference_body_px=55.0),
    )

    value, origin = resolve_reference_body_px(project)

    assert value == 55.0
    assert "sliced-training" in origin


def test_gui_handler_uses_frames_processed_as_the_denominator(tmp_path):
    """Goes through the CALLER's arithmetic, which the unit test above cannot.

    The existing unit test passed the denominator by hand, so it could never
    catch the caller computing it wrongly.
    """
    import inspect

    from hydra_suite.detectkit.gui import escalation_actions

    src_text = inspect.getsource(escalation_actions.on_semantic_escalation)
    assert "result.frames_processed" in src_text
    assert "result.labelled + result.empty_images" not in src_text


def test_the_dialog_asks_for_labels_without_decoding_images():
    """I8, other end: the has-labels check must not be labelled_frames_for."""
    import inspect

    from hydra_suite.detectkit.gui.dialogs import semantic_escalation_dialog as mod

    source = inspect.getsource(
        mod.SemanticEscalationDialog._refresh_calibration_enabled
    )
    assert "has_labelled_frames" in source
    assert "labelled_frames_for" not in source


def test_accepting_a_sam3_review_does_not_create_a_sibling_source(tmp_path):
    """The originally reported symptom: accept used to spawn a new source."""
    import hydra_suite.detectkit.jobs.semantic_escalation as se

    assert not hasattr(se, "accept_pending_semantic_escalation")
    assert not hasattr(se, "_unique_source_name")


def test_semantic_handler_jumps_to_the_review_like_the_sam2_path():
    """Staging is worthless until the user is in front of the review bar.

    The SAM2 path has called ``_on_go_to_staged_review()`` on success since
    it shipped; the SAM3 path did not, so a successful run ended at an
    info box with the review bar nowhere in sight.
    """
    import inspect

    from hydra_suite.detectkit.gui import escalation_actions

    source = inspect.getsource(escalation_actions.on_semantic_escalation)
    assert "window._on_go_to_staged_review()" in source


def test_both_escalation_handlers_persist_the_pointer_as_it_is_written():
    """A staged_review written in memory but never saved orphans the whole
    staging directory: the review bar keys off the pointer, so the staged
    frames become unreviewable even though every label is on disk."""
    import inspect

    from hydra_suite.detectkit.gui import escalation_actions

    for handler in (
        escalation_actions.on_semantic_escalation,
        escalation_actions.on_escalate_geometry,
    ):
        source = inspect.getsource(handler)
        assert (
            "worker.project_mutated.connect(window._persist_staged_pointer)" in source
        ), handler.__name__


def test_the_pointer_persist_slot_is_a_bound_method_of_the_window():
    """A functor with no receiver QObject is delivered DIRECTLY on the
    emitting thread. This slot touches the dataset panel, so it must be a
    bound method of the window for AutoConnection to queue it onto the main
    thread -- hence a MainWindow method, not a lambda in the handler.

    It also must not be _save_current_project: it fires once per source, and
    that method flashes "Project saved." over the run's progress messages.
    """
    import ast
    import inspect
    import textwrap

    from hydra_suite.detectkit.gui.main_window import MainWindow

    fn = ast.parse(
        textwrap.dedent(inspect.getsource(MainWindow._persist_staged_pointer))
    ).body[0]
    # The docstring names _save_current_project to explain the choice, so
    # compare against the BODY, not the raw source text.
    body = ast.unparse(ast.Module(body=fn.body[1:], type_ignores=[]))
    assert "save_project(self._project)" in body
    assert "_save_current_project" not in body


def test_semantic_dialog_accepts_a_published_finetuned_model_key(qapp, monkeypatch):
    """The dropdown offers registry keys, so the gate must accept them.

    ``probe_availability`` rejects anything outside ``SAM3_VARIANTS`` as
    "Unknown SAM3 variant", which made every published finetuned model
    unselectable: the dialog refused the key its own dropdown had listed.
    The gate is ``probe_checkpoint``, which spans both key spaces.
    """
    import hydra_suite.detectkit.gui.dialogs.semantic_escalation_dialog as mod

    key = "sam3_finetuned/whatever.pt"
    seen = []

    def _probe(k=None, *a, **kw):
        seen.append(k)
        return _Availability(True, "")

    monkeypatch.setattr(mod, "probe_checkpoint", _probe)
    dlg = SemanticEscalationDialog([_Src("a")], reference_body_px=40.0)
    monkeypatch.setattr(dlg, "selected_variant", lambda: key)

    dlg._refresh_checkpoint_note()
    assert dlg._checkpoint_note.text() == ""
    assert dlg.confirm_checkpoint() is True
    assert key in seen, "the gate never probed the selected finetuned key"
