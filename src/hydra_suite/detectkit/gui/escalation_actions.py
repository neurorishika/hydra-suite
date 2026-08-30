"""Escalation handlers, lifted out of main_window.py.

Honest accounting: this removes ~205 of main_window.py's 2152 lines, which
does NOT bring it near CLAUDE.md's thin-coordinator target. It is done
because the new semantic handler would otherwise add a THIRD escalation
flow to an already-oversized file.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMessageBox, QProgressDialog


def on_escalate_geometry(window, preselect: str | None = None) -> None:
    """Open the SAM2 escalate dialog and run a Sam2EscalationWorker."""
    if window._project is None:
        QMessageBox.information(
            window,
            "Escalate to segment (SAM2)",
            "Open a project before escalating sources.",
        )
        return
    if window._escalation_worker is not None:
        QMessageBox.information(
            window,
            "Escalate to segment (SAM2)",
            "An escalation run is already in progress.",
        )
        return

    try:
        from hydra_suite.detectkit.jobs.sam2_escalation import (
            EscalationRequest,
            Sam2EscalationWorker,
        )

        from .dialogs.escalate_sam2_dialog import EscalateSam2Dialog
    except Exception as exc:  # pragma: no cover - optional SAM2 assets
        QMessageBox.warning(
            window,
            "Escalate to segment (SAM2)",
            f"SAM2 escalation is unavailable: {exc}",
        )
        return

    dlg = EscalateSam2Dialog(window._project.sources, parent=window)
    if preselect:
        dlg.preselect_source(preselect)
    if not dlg.exec():
        return

    source_names = dlg.selected_sources()
    if not source_names:
        QMessageBox.information(
            window,
            "Escalate to segment (SAM2)",
            "Select at least one source to escalate.",
        )
        return

    existing_by_name = {s.name: s for s in window._project.sources}
    would_conflict = [
        n
        for n in source_names
        if existing_by_name.get(n) is not None
        and existing_by_name[n].pending_escalation is not None
    ]
    overwrite = False
    if would_conflict:
        reply = QMessageBox.question(
            window,
            "Escalate to segment (SAM2)",
            (
                "The following source(s) already have a pending escalation "
                "awaiting review, which will be replaced:\n\n"
                f"{', '.join(would_conflict)}\n\n"
                "Continue and replace the staged result?"
            ),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            source_names = [n for n in source_names if n not in would_conflict]
            if not source_names:
                return
        else:
            overwrite = True

    request = EscalationRequest(
        project=window._project,
        source_names=source_names,
        variant=dlg.selected_variant(),
        overwrite=overwrite,
    )

    progress = QProgressDialog(
        f"Escalating {len(source_names)} source(s) to segment…",
        None,
        0,
        100,
        window,
    )
    progress.setWindowTitle("Escalate to segment (SAM2)")
    progress.setMinimumDuration(0)
    progress.setWindowModality(Qt.ApplicationModal)
    progress.setAttribute(Qt.WA_DeleteOnClose, True)
    progress.setValue(0)

    worker = Sam2EscalationWorker(request)
    worker.progress.connect(progress.setValue)
    worker.status.connect(progress.setLabelText)
    worker.status.connect(lambda msg: window.statusBar().showMessage(msg, 3000))

    def _stash_error(msg: str) -> None:
        # error fires before finished/progress.close() -- stash and show
        # from _finish, same reasoning as _handle_result below: showing a
        # QMessageBox here would stack it under the still-open
        # application-modal progress dialog.
        window._last_escalation_error = msg

    worker.error.connect(_stash_error)

    def _handle_result(result: object) -> None:
        # The worker set pending_escalation on existing sources on a
        # background thread; persist + refresh immediately. Everything
        # UI-facing (message boxes, the review dialog) is deferred to
        # _finish, because the application-modal progress dialog is still
        # open here -- a dialog opened from this slot would stack under it
        # and be undismissable.
        window._save_current_project()
        window._dataset_panel.refresh_sources(window._project)
        window._tools_panel.refresh_overview()
        window._last_escalation_result = result

    def _finish() -> None:
        progress.close()
        window._escalation_worker = None
        window._escalation_progress_dialog = None

        error_msg = window._last_escalation_error
        window._last_escalation_error = None
        if error_msg is not None:
            QMessageBox.warning(window, "Escalate to segment (SAM2)", error_msg)
            return

        result = window._last_escalation_result
        window._last_escalation_result = None
        if result is None:
            return

        staged = list(getattr(result, "staged", []) or [])
        primed = int(getattr(result, "primed", 0))
        fell_back = int(getattr(result, "fell_back", 0))
        skipped = list(getattr(result, "skipped", []) or [])
        skipped_note = (
            (
                "\n\nSkipped (already has a pending escalation, not "
                "overwritten): "
                + ", ".join(f"{name} ({reason})" for name, reason in skipped)
            )
            if skipped
            else ""
        )

        if staged:
            QMessageBox.information(
                window,
                "Escalate to segment (SAM2)",
                (
                    f"Staged {len(staged)} source(s) for review: "
                    f"{', '.join(staged)}.\n\n"
                    f"{primed} instance(s) primed, {fell_back} fell back "
                    f"to the original box.{skipped_note}"
                ),
            )
            on_review_escalations(window)
        else:
            QMessageBox.information(
                window,
                "Escalate to segment (SAM2)",
                f"No sources were staged for escalation.{skipped_note}",
            )

    worker.result_ready.connect(_handle_result)
    worker.finished.connect(_finish)
    window._escalation_worker = worker
    window._escalation_progress_dialog = progress
    progress.show()
    worker.start()


def on_semantic_escalation(window) -> None:
    """Open the SAM3 semantic escalation dialog and run the worker."""
    if window._project is None:
        QMessageBox.information(
            window,
            "Semantic escalation",
            "Open a project before escalating sources.",
        )
        return
    if window._escalation_worker is not None:
        QMessageBox.information(
            window,
            "Semantic escalation",
            "An escalation run is already in progress.",
        )
        return

    from hydra_suite.detectkit.jobs.semantic_escalation import (
        SemanticEscalationRequest,
        SemanticEscalationWorker,
        is_prompt_failure,
        sources_pending_replacement,
    )

    from .dialogs.semantic_escalation_dialog import SemanticEscalationDialog

    reference_body_px, body_px_origin = resolve_reference_body_px(window._project)
    dlg = SemanticEscalationDialog(
        window._project.sources,
        reference_body_px,
        parent=window,
        body_px_origin=body_px_origin,
        project=window._project,
        persist_callback=window._save_current_project,
    )
    if not dlg.exec():
        return

    sources = dlg.selected_sources()
    params = dlg.parameters()
    request = SemanticEscalationRequest(
        project=window._project,
        source_names=[s.name for s in sources],
        variant=dlg.selected_variant(),
        prompt=dlg.prompt(),
        **params,
    )

    # I2: overwrite is NOT hardcoded. A resume of this same run needs no
    # overwrite (the job compares staging directories), so a True here only
    # ever meant "silently wipe whatever else is staged" -- including an
    # unreviewed SAM2 escalation. Mirror on_escalate_geometry and ask.
    would_replace = sources_pending_replacement(request)
    if would_replace:
        reply = QMessageBox.question(
            window,
            "Semantic escalation",
            (
                "The following source(s) already have a DIFFERENT pending "
                "escalation awaiting review, which will be destroyed:\n\n"
                f"{', '.join(would_replace)}\n\n"
                "Continue and replace the staged result?"
            ),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            request.overwrite = True
        else:
            request.source_names = [
                n for n in request.source_names if n not in would_replace
            ]
            if not request.source_names:
                return

    # A REAL cancel button (the SAM2 dialog passes None here and cannot be
    # cancelled) -- this run takes tens of seconds PER FRAME.
    progress = QProgressDialog(
        f"Segmenting '{request.prompt}'…", "Cancel", 0, 100, window
    )
    progress.setWindowTitle("Semantic escalation (SAM3)")
    progress.setMinimumDuration(0)
    progress.setWindowModality(Qt.WindowModal)
    progress.setAttribute(Qt.WA_DeleteOnClose, True)
    progress.setValue(0)

    worker = SemanticEscalationWorker(request)
    progress.canceled.connect(worker.cancel)
    worker.progress.connect(progress.setValue)
    worker.status.connect(progress.setLabelText)

    def _stash_error(msg: str) -> None:
        # error fires before finished/progress.close() -- stash and show
        # from _finish, same reasoning as on_escalate_geometry's _stash_error:
        # showing a QMessageBox here would stack it under the still-open
        # progress dialog.
        window._last_escalation_error = msg

    worker.error.connect(_stash_error)

    def _handle_result(result) -> None:
        # Everything UI-facing (the prompt-failure warning, the success info
        # box) is deferred to _finish, because the progress dialog is still
        # open here -- a modal dialog opened from this slot would stack
        # under it, same reasoning as on_escalate_geometry's _handle_result.
        window._save_current_project()
        window._dataset_panel.refresh_sources(window._project)
        # The overview counts pending escalations; the SAM2 path refreshes it
        # here and this one did not, leaving the panel stale after a run.
        window._tools_panel.refresh_overview()
        window._last_escalation_result = result

    def _finish() -> None:
        progress.close()
        window._escalation_worker = None
        window._escalation_progress_dialog = None

        error_msg = window._last_escalation_error
        window._last_escalation_error = None
        if error_msg is not None:
            QMessageBox.warning(window, "Semantic escalation", error_msg)
            return

        result = window._last_escalation_result
        window._last_escalation_result = None
        if result is None:
            return

        skipped_note = (
            (
                "\n\nSkipped (a different escalation is already pending, not "
                "replaced): "
                + ", ".join(f"{name} ({reason})" for name, reason in result.skipped)
            )
            if result.skipped
            else ""
        )
        # I1: FRAMES, from the run itself. `labelled` counts INSTANCES, so
        # `labelled + empty_images` inflated the denominator (40 frames x 10
        # instances -> 460) and the rule below could only fire on a total
        # shutout.
        frames = result.frames_processed
        if is_prompt_failure(result):
            QMessageBox.warning(
                window,
                "Semantic escalation",
                f"'{request.prompt}' matched nothing on "
                f"{result.empty_images} of {frames} frame(s). This is usually a "
                "prompt that the model does not recognise — try a different "
                "noun phrase in the preview before re-running." + skipped_note,
            )
            return
        # I9: a cancelled run is a PARTIAL result, and saying "Staged N
        # source(s)" without that word invites the user to accept it as
        # complete.
        headline = (
            "CANCELLED — partial result. Staged so far"
            if result.cancelled
            else "Staged"
        )
        orphan_note = (
            f" {result.orphaned} label(s) had no matching image."
            if result.orphaned
            else ""
        )
        QMessageBox.information(
            window,
            "Semantic escalation",
            f"{headline} {len(result.staged)} source(s) over {frames} frame(s): "
            f"{result.labelled} instance(s), "
            f"{result.empty_images} empty frame(s), {result.degenerate} degenerate "
            f"contour(s) dropped.{orphan_note}"
            + (
                " Re-run with the same settings to carry on from here."
                if result.cancelled
                else " Review them before training."
            )
            + skipped_note,
        )

    worker.result_ready.connect(_handle_result)
    worker.finished.connect(_finish)
    window._escalation_worker = worker
    window._escalation_progress_dialog = progress
    worker.start()


def resolve_reference_body_px(project) -> tuple[float, str]:
    """(reference_body_px, provenance) for the semantic escalation dialog.

    The spec's chain, in order: the DetectKit project setting, then the
    MEDIAN LONGEST SIDE of the source's existing labels, then the user (the
    dialog's editable field, which this only prefills). Only link 1 existed,
    so any project without a slice-training reference silently ran with
    tiling OFF -- the measured-worst configuration.
    """
    from hydra_suite.detectkit.jobs.semantic_escalation import measure_median_body_px

    slice_settings = getattr(project, "slice_training", None)
    from_project = float(getattr(slice_settings, "reference_body_px", 0.0) or 0.0)
    if from_project > 0:
        return from_project, "the project's sliced-training reference body size"
    try:
        # F4: this decodes images on the GUI thread while the dialog opens, so
        # the sample is capped project-wide. The cap is reported, not hidden:
        # the field is editable and the user must be able to see the median
        # rests on a sample rather than on every labelled frame.
        measured, sampled, truncated = measure_median_body_px(
            getattr(project, "sources", []) or []
        )
    except Exception:  # pragma: no cover - unreadable labels
        measured, sampled, truncated = 0.0, 0, False
    if measured > 0:
        note = f"the median longest side of your existing labels ({sampled} frame"
        note += "s" if sampled != 1 else ""
        note += ", a capped sample)" if truncated else ")"
        return measured, note
    return 0.0, "nothing found — enter one, or tiling stays off"


def on_review_escalations(window) -> None:
    """Open the review dialog for every source with a pending escalation,
    regardless of when it was staged. Reachable independent of an
    escalation run having just finished, so closing the dialog without
    acting never strands a pending escalation."""
    if window._project is None:
        QMessageBox.information(window, "Review Escalations", "Open a project first.")
        return

    pending_sources = [
        s for s in window._project.sources if s.pending_escalation is not None
    ]
    if not pending_sources:
        QMessageBox.information(
            window,
            "Review Escalations",
            "There are no pending escalations to review.",
        )
        return

    from .dialogs.review_escalations_dialog import ReviewEscalationsDialog

    review_dlg = ReviewEscalationsDialog(
        pending_sources,
        parent=window,
        project=window._project,
        project_dir=window._project.project_dir,
    )
    review_dlg.exec()
    window._save_current_project()
    window._dataset_panel.refresh_sources(window._project)
    window._tools_panel.refresh_overview()
