"""Escalation handlers, lifted out of main_window.py.

Honest accounting: this removes ~205 of main_window.py's 2152 lines, which
does NOT bring it near CLAUDE.md's thin-coordinator target. It is done
because the new semantic handler would otherwise add a THIRD escalation
flow to an already-oversized file.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMessageBox, QProgressDialog, QPushButton


def retain_semantic_containment_recovery(window, progress: QProgressDialog) -> None:
    """Keep the SAM3 cleanup owner and retry control visible after failure."""
    progress.setLabelText(
        "Containment recovery required. Click Retry cleanup to confirm the "
        "protected SAM3 process tree has stopped."
    )
    progress.setCancelButtonText("Retry cleanup")
    progress.show()
    window.statusBar().showMessage(
        "SAM3 containment recovery required; resources remain reserved."
    )


def retry_semantic_containment_cleanup(
    window, worker, progress: QProgressDialog
) -> bool:
    """Release the SAM3 worker only after its sidecar proves quiescence."""
    progress.setLabelText("Retrying safe SAM3 containment cleanup…")
    progress.setCancelButtonText("Retrying…")
    button = progress.findChild(QPushButton)
    if button is not None:
        button.setEnabled(False)
    if not worker.retry_containment_cleanup():
        progress.setLabelText(
            "Containment is still unproven. Retry cleanup again before "
            "starting another SAM3 run."
        )
        progress.setCancelButtonText("Retry cleanup")
        if button is not None:
            button.setEnabled(True)
        progress.show()
        window.statusBar().showMessage(
            "SAM3 containment recovery is still required; ownership is retained."
        )
        return False

    progress.close()
    if window._escalation_worker is worker:
        window._escalation_worker = None
        window._escalation_progress_dialog = None
    window._last_escalation_error = None
    window._last_escalation_result = None
    cleanup_note = (
        f" Temporary cleanup issue: {worker.recovery_cleanup_error}"
        if worker.recovery_cleanup_error
        else ""
    )
    QMessageBox.information(
        window,
        "SAM3 Containment Recovery Complete",
        "The protected SAM3 process has stopped and its resources were released. "
        "This escalation did not complete; run it again." + cleanup_note,
    )
    window.statusBar().showMessage("SAM3 containment recovery completed safely.", 5000)
    return True


def _tiling_note(tile_px, tiled_frames: int, untiled_frames: int, seams: int) -> str:
    """What the run actually did with tiling, for the completion message."""
    if not tile_px:
        return ""
    if not tiled_frames:
        return (
            f"\n\nTiling at {tile_px} px could not apply to any frame (the "
            "tile covered the frame, or was too small for it), so every frame "
            "was segmented whole."
        )
    note = f"\n\nSegmented in {tile_px} px tiles on {tiled_frames} frame(s)"
    if untiled_frames:
        note += f"; {untiled_frames} frame(s) could not be tiled and ran whole"
    if seams:
        note += (
            f"; {seams} animal(s) crossed a tile seam and were segmented on "
            "the full frame instead"
        )
    return note + "."


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

    body_px, _note, body_source = resolve_reference_body(window._project)
    dlg = EscalateSam2Dialog(
        window._project.sources,
        parent=window,
        project=window._project,
        reference_body_px=body_px,
        body_px_source=body_source,
        persist_callback=window._save_current_project,
    )
    if preselect:
        dlg.preselect_source(preselect)
    if not dlg.exec():
        return

    source_names = dlg.selected_sources()
    source_paths = dlg.selected_source_paths()
    if not source_names:
        QMessageBox.information(
            window,
            "Escalate to segment (SAM2)",
            "Select at least one source to escalate.",
        )
        return

    existing_by_path = {str(s.path): s for s in window._project.sources}
    conflicting_paths = [
        path
        for path in source_paths
        if existing_by_path.get(path) is not None
        and existing_by_path[path].staged_review is not None
    ]
    would_conflict = [existing_by_path[path].name for path in conflicting_paths]
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
            selected = [
                (name, path)
                for name, path in zip(source_names, source_paths)
                if path not in conflicting_paths
            ]
            source_names = [name for name, _path in selected]
            source_paths = [path for _name, path in selected]
            if not source_names:
                return
        else:
            overwrite = True

    request = EscalationRequest(
        project=window._project,
        source_names=source_names,
        source_paths=source_paths,
        variant=dlg.selected_variant(),
        overwrite=overwrite,
        device=dlg.selected_device(),
        **dlg.tiling_parameters(),
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
    # A BOUND METHOD of the window, not a lambda: a functor with no
    # receiver QObject is delivered DIRECTLY on the emitting (worker)
    # thread, and this slot touches the dataset panel. Binding it to the
    # window -- a QObject living in the main thread -- is what makes Qt's
    # AutoConnection resolve to a queued, main-thread call.
    worker.project_mutated.connect(window._persist_staged_pointer)
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
        # The worker set staged_review on existing sources on a
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
        seam_fallbacks = int(getattr(result, "seam_fallbacks", 0))
        tile_px = getattr(result, "tile_px", None)
        tiling_note = _tiling_note(
            tile_px,
            int(getattr(result, "tiled_frames", 0)),
            int(getattr(result, "untiled_frames", 0)),
            seam_fallbacks,
        )
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
                    f"to the original box.{tiling_note}{skipped_note}\n\n"
                    "Use the review bar on the annotation preview to accept "
                    "or reject each frame."
                ),
            )
            window._on_go_to_staged_review()
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
        source_paths_pending_replacement,
        sources_pending_replacement,
    )

    from .dialogs.semantic_escalation_dialog import SemanticEscalationDialog

    reference_body_px, body_px_origin, body_px_source = resolve_reference_body(
        window._project
    )
    dlg = SemanticEscalationDialog(
        window._project.sources,
        reference_body_px,
        parent=window,
        body_px_origin=body_px_origin,
        body_px_source=body_px_source,
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
        source_paths=[s.path for s in sources],
        variant=dlg.selected_variant(),
        prompt=dlg.prompt(),
        **params,
    )

    # I2: overwrite is NOT hardcoded. A resume of this same run needs no
    # overwrite (the job compares staging directories), so a True here only
    # ever meant "silently wipe whatever else is staged" -- including an
    # unreviewed SAM2 escalation. Mirror on_escalate_geometry and ask.
    would_replace = sources_pending_replacement(request)
    replacement_paths = source_paths_pending_replacement(request)
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
            selected = [
                (name, path)
                for name, path in zip(request.source_names, request.source_paths)
                if path not in replacement_paths
            ]
            request.source_names = [name for name, _path in selected]
            request.source_paths = [path for _name, path in selected]
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
    progress.setAutoClose(False)
    progress.setAutoReset(False)
    progress.setValue(0)

    worker = SemanticEscalationWorker(request)

    def _request_cancel() -> None:
        if worker.containment_recovery_required:
            retry_semantic_containment_cleanup(window, worker, progress)
            return
        worker.cancel()
        progress.setLabelText("Cancelling SAM3 escalation…")
        progress.setCancelButtonText("Cancelling…")
        button = progress.findChild(QPushButton)
        if button is not None:
            button.setEnabled(False)
        progress.show()

    progress.canceled.connect(_request_cancel)
    worker.progress.connect(progress.setValue)
    worker.status.connect(progress.setLabelText)

    def _stash_error(msg: str) -> None:
        # error fires before finished/progress.close() -- stash and show
        # from _finish, same reasoning as on_escalate_geometry's _stash_error:
        # showing a QMessageBox here would stack it under the still-open
        # progress dialog.
        window._last_escalation_error = msg

    worker.error.connect(_stash_error)

    # A BOUND METHOD of the window, not a lambda: a functor with no
    # receiver QObject is delivered DIRECTLY on the emitting (worker)
    # thread, and this slot touches the dataset panel. Binding it to the
    # window -- a QObject living in the main thread -- is what makes Qt's
    # AutoConnection resolve to a queued, main-thread call.
    worker.project_mutated.connect(window._persist_staged_pointer)

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
        if worker.containment_recovery_required:
            retain_semantic_containment_recovery(window, progress)
            return
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
        # Mirror the SAM2 path: staging is worthless until the user is
        # standing in front of the review bar. A CANCELLED run still staged
        # real frames, so it jumps too.
        if result.staged:
            window._on_go_to_staged_review()

    worker.result_ready.connect(_handle_result)
    worker.finished.connect(_finish)
    window._escalation_worker = worker
    window._escalation_progress_dialog = progress
    worker.start()


def resolve_reference_body_px(project) -> tuple[float, str]:
    """(reference_body_px, provenance note); see :func:`resolve_reference_body`."""
    value, note, _source = resolve_reference_body(project)
    return value, note


def resolve_reference_body(project) -> tuple[float, str, str]:
    """(reference_body_px, provenance note, badge source) for the escalation dialogs.

    The source is the widget badge: ``project`` (the project setting),
    ``dataset`` (label median) or ``user`` (nothing found -- the user enters it).

    The spec's chain, in order: the DetectKit project setting, then the
    MEDIAN LONGEST SIDE of the source's existing labels, then the user (the
    dialog's editable field, which this only prefills). Only link 1 existed,
    so any project without a slice-training reference silently ran with
    tiling OFF -- the measured-worst configuration.
    """
    from hydra_suite.detectkit.jobs.calibration_frames import (
        has_polygon_frames,
        measure_median_body_px,
    )

    slice_settings = getattr(project, "slice_settings", None)
    from_project = float(getattr(slice_settings, "reference_body_px", 0.0) or 0.0)
    if from_project > 0:
        return (
            from_project,
            "the project's sliced-training reference body size",
            "project",
        )
    try:
        # F4: this decodes images on the GUI thread while the dialog opens, so
        # the sample is capped project-wide. The cap is reported, not hidden:
        # the field is editable and the user must be able to see the median
        # rests on a sample rather than on every labelled frame.
        # Polygon ground truth first: it is what calibration measures on, so
        # the tile size it fits is relative to THESE animals' size.
        sources = list(getattr(project, "sources", []) or [])
        measured, sampled, truncated = measure_median_body_px(
            [s for s in sources if has_polygon_frames(s)] or sources
        )
    except Exception:  # pragma: no cover - unreadable labels
        measured, sampled, truncated = 0.0, 0, False
    if measured > 0:
        note = f"the median longest side of your existing labels ({sampled} frame"
        note += "s" if sampled != 1 else ""
        note += ", a capped sample)" if truncated else ")"
        return measured, note, "dataset"
    return 0.0, "nothing found — enter one, or tiling stays off", "user"
