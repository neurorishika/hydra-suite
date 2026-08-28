"""SAM2 escalation orchestrator: existing OBB/box labels -> staged polygon review."""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from datetime import datetime
from hashlib import sha1
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from PySide6.QtCore import Signal

from hydra_suite.core.inference.sam2.masks import mask_to_contour
from hydra_suite.data.al.escalation import LabelRecord
from hydra_suite.data.al.labels import write_label_file
from hydra_suite.data.project_bundle import ensure_bundle_subdirectory
from hydra_suite.detectkit.gui.models import OBBSource, PendingEscalation
from hydra_suite.utils.geometry_levels import GeometryLevel
from hydra_suite.widgets.workers import BaseWorker

from .sam2_prompts import build_prompts, read_boxes_from_label


@dataclass
class EscalationRequest:
    project: object  # has .project_dir and .sources (list[OBBSource])
    source_names: list[str]
    variant: str
    overwrite: bool = False


@dataclass
class EscalationResult:
    # Names of sources that received a fresh `pending_escalation` this run.
    staged: list[str] = field(default_factory=list)
    primed: int = 0
    fell_back: int = 0
    # (source_name, reason) pairs for sources skipped because they already
    # have a pending escalation and overwrite was not requested.
    skipped: list[tuple[str, str]] = field(default_factory=list)


class Sam2EscalationWorker(BaseWorker):
    """QThread wrapper around run_escalation (BaseWorker signals + result_ready)."""

    result_ready = Signal(object)  # EscalationResult

    def __init__(self, request: EscalationRequest, executor=None, parent=None) -> None:
        super().__init__(parent)
        self._request = request
        self._executor = executor

    def execute(self) -> None:
        from hydra_suite.core.inference.sam2.executor import Sam2SegmentExecutor

        executor = self._executor or Sam2SegmentExecutor.from_variant(
            self._request.variant
        )
        self.status.emit(f"Escalating {len(self._request.source_names)} source(s)...")
        result = run_escalation(
            self._request,
            executor,
            overwrite=self._request.overwrite,
            progress=lambda pct, msg: (
                self.progress.emit(pct),
                self.status.emit(msg),
            ),
        )
        self.status.emit(
            f"Done: {len(result.staged)} staged, {result.primed} primed, "
            f"{result.fell_back} fell back (review these first)."
        )
        self.result_ready.emit(result)


def _sources_by_name(project) -> dict[str, OBBSource]:
    return {s.name: s for s in project.sources}


def run_escalation(
    req: EscalationRequest,
    executor,
    *,
    overwrite: bool = False,
    progress: Callable[[int, str], None] | None = None,
) -> EscalationResult:
    """Stage each named source's SAM2-primed polygon labels for review.

    Writes the primed result to a per-source staging directory under
    ``artifacts/pending_escalations/`` and records it on the source's
    ``pending_escalation`` field. It does NOT touch the source's own
    canonical labels and does NOT register any new source -- a caller (the
    escalation review dialog) must call ``accept_pending_escalation`` or
    ``reject_pending_escalation`` to promote or discard the staged result.

    Re-running escalation over a source that already has a pending
    escalation is guarded: by default it's skipped (recorded in
    ``result.skipped``) rather than silently clobbering a staged result the
    user hasn't reviewed yet. Pass ``overwrite=True`` to re-stage (replaces
    the staging directory in place).
    """
    result = EscalationResult()
    by_name = _sources_by_name(req.project)
    todo = [
        by_name[n]
        for n in req.source_names
        if n in by_name and by_name[n].level != "polygon"
    ]
    project_root = Path(req.project.project_dir)
    for si, src in enumerate(todo):
        if src.pending_escalation is not None and not overwrite:
            result.skipped.append(
                (
                    src.name,
                    f"'{src.name}' already has a pending escalation; review it, "
                    "or re-run with overwrite to replace it.",
                )
            )
            continue

        src_root = Path(src.path)
        images_dir = src_root / "images"
        labels_dir = src_root / "labels"

        content_hash = sha1(
            (str(src_root.resolve()) + req.variant).encode("utf-8")
        ).hexdigest()[:10]
        staged_dirname = f"{src.name}-{req.variant}-{content_hash}"
        staged_root = ensure_bundle_subdirectory(
            project_root, f"artifacts/pending_escalations/{staged_dirname}"
        )

        # A source with an existing pending escalation under a DIFFERENT
        # staging path (e.g. this is a re-escalation with a different SAM2
        # variant, which hashes to a different directory) must have its old
        # staging dir cleaned up -- otherwise it's orphaned forever, since
        # nothing else ever revisits a replaced pending_escalation.
        old_pending = src.pending_escalation
        if old_pending is not None and old_pending.staged_path != str(staged_root):
            shutil.rmtree(Path(old_pending.staged_path), ignore_errors=True)

        shutil.rmtree(staged_root, ignore_errors=True)
        (staged_root / "labels").mkdir(parents=True, exist_ok=True)

        # Recursive + path-mirroring: a source's images/labels can be nested
        # (e.g. images/train/..., images/val/...) -- source_import.py's
        # materializer can produce this layout. A flat top-level glob would
        # silently stage ZERO labels for such a source, and accept() would
        # then delete every real label with nothing to replace it.
        images = sorted(
            p
            for p in images_dir.rglob("*")
            if p.suffix.lower() in (".jpg", ".jpeg", ".png")
        )
        for ii, img_path in enumerate(images):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            relative_label = img_path.relative_to(images_dir).with_suffix(".txt")
            label_path = labels_dir / relative_label
            boxes = read_boxes_from_label(label_path, w, h)
            records: list[LabelRecord] = []
            if boxes:
                prompts = build_prompts(boxes)
                executor.set_image(img)
                for box, prompt in zip(boxes, prompts):
                    mask, _iou = executor.segment(
                        prompt.box_xyxy, prompt.positive_points, prompt.negative_points
                    )
                    contour = mask_to_contour(mask)
                    if contour is not None:
                        result.primed += 1
                        poly = contour
                    else:  # fallback: original OBB corners as the polygon
                        result.fell_back += 1
                        poly = box.polygon_px
                    records.append(
                        LabelRecord(
                            class_id=0,
                            confidence=1.0,
                            points=np.asarray(poly, dtype=np.float32).reshape(-1, 2),
                            level=GeometryLevel.POLYGON,
                        )
                    )
            staged_label_path = staged_root / "labels" / relative_label
            staged_label_path.parent.mkdir(parents=True, exist_ok=True)
            write_label_file(
                staged_label_path,
                records,
                frame_size=(h, w),
                level=GeometryLevel.POLYGON,
            )
            if progress:
                progress(
                    int(100 * (si + (ii + 1) / max(len(images), 1)) / len(todo)),
                    f"{src.name}: {ii + 1}/{len(images)}",
                )

        (staged_root / "classes.txt").write_text(
            (src_root / "classes.txt").read_text()
            if (src_root / "classes.txt").exists()
            else "object\n"
        )
        src.pending_escalation = PendingEscalation(
            staged_path=str(staged_root),
            target_level=GeometryLevel.POLYGON.label,
            sam2_variant=req.variant,
            created_at=datetime.now().isoformat(),
        )
        result.staged.append(src.name)
    return result


def accept_pending_escalation(source: OBBSource) -> None:
    """Promote *source*'s staged escalation result to its canonical labels.

    Overwrites the source's ``labels/`` + ``classes.txt`` from the staged
    copy, sets ``level``/``sam2_variant`` from the pending record, resets
    ``reviewed`` to ``False`` (same meaning as any other machine-derived,
    not-yet-human-confirmed result -- just attached to the existing source
    instead of a new sibling), removes the staging directory, and clears
    ``pending_escalation``.

    Validates BEFORE deleting anything: refuses (raising ``RuntimeError``,
    source left untouched) if the staging directory is missing on disk, or
    if it is missing a label file for an image the source currently has a
    label for (e.g. an image that failed to decode during escalation and was
    silently skipped by ``run_escalation``) -- accepting such a staged result
    would otherwise delete real labels with nothing staged to replace them.

    Raises ValueError if the source has no pending escalation.
    """
    pending = source.pending_escalation
    if pending is None:
        raise ValueError(f"Source '{source.name}' has no pending escalation.")

    staged_root = Path(pending.staged_path)
    staged_labels = staged_root / "labels"
    if not staged_labels.is_dir():
        raise RuntimeError(
            f"Staged escalation for '{source.name}' is missing on disk "
            f"({staged_labels}); nothing was changed. Reject this escalation "
            "and re-run it."
        )

    source_root = Path(source.path)
    source_labels = source_root / "labels"
    existing_rel = (
        {p.relative_to(source_labels) for p in source_labels.rglob("*.txt")}
        if source_labels.is_dir()
        else set()
    )
    staged_rel = {p.relative_to(staged_labels) for p in staged_labels.rglob("*.txt")}
    missing = sorted(str(p) for p in existing_rel - staged_rel)
    if missing:
        raise RuntimeError(
            f"Staged escalation for '{source.name}' is missing "
            f"{len(missing)} label file(s) that exist in the source (likely "
            "an unreadable image during escalation) -- refusing to accept, "
            f"as this would delete those labels: {missing[:5]}"
        )

    shutil.rmtree(source_labels, ignore_errors=True)
    shutil.copytree(staged_labels, source_labels)
    classes_src = staged_root / "classes.txt"
    if classes_src.exists():
        shutil.copyfile(classes_src, source_root / "classes.txt")

    source.level = pending.target_level
    source.reviewed = False
    source.sam2_variant = pending.sam2_variant

    shutil.rmtree(staged_root, ignore_errors=True)
    source.pending_escalation = None


def reject_pending_escalation(source: OBBSource) -> None:
    """Discard *source*'s staged escalation result, leaving it untouched.

    Raises ValueError if the source has no pending escalation.
    """
    pending = source.pending_escalation
    if pending is None:
        raise ValueError(f"Source '{source.name}' has no pending escalation.")
    shutil.rmtree(Path(pending.staged_path), ignore_errors=True)
    source.pending_escalation = None
