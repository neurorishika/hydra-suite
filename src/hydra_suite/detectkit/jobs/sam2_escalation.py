"""SAM2 escalation orchestrator: existing OBB/box labels -> staged polygon review."""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from hashlib import sha1
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from PySide6.QtCore import Signal

from hydra_suite.core.inference.masks import clip_mask_to_polygon, mask_to_contour
from hydra_suite.data.al.escalation import LabelRecord
from hydra_suite.data.al.labels import write_label_file
from hydra_suite.data.project_bundle import ensure_bundle_subdirectory
from hydra_suite.detectkit.gui.constants import IMG_EXTS
from hydra_suite.detectkit.gui.models import OBBSource, StagedReview
from hydra_suite.utils.geometry_levels import GeometryLevel
from hydra_suite.widgets.workers import BaseWorker

from .sam2_prompts import build_prompts, read_boxes_from_label

logger = logging.getLogger(__name__)

# Every staging directory lives at
# ``<project_dir>/artifacts/pending_escalations/<dirname>``.
PENDING_ESCALATIONS_RELDIR = Path("artifacts") / "pending_escalations"


def _is_safe_to_delete(
    path: str | Path | None,
    project_dir: str | Path | None = None,
) -> bool:
    """True if *path* is a staging directory this module may recursively delete.

    ``StagedReview.staged_path`` round-trips through the saved project
    file, so it is untrusted input from disk: a hand-edited or corrupted
    project could point it at ``/`` or at the source's own directory, and
    every deletion here is a recursive ``rmtree``. When *project_dir* is
    known the path must resolve strictly inside that project's
    ``artifacts/pending_escalations/``; when it is not (callers holding only
    an ``OBBSource``), the path must at least have the structural shape every
    staging directory has.
    """
    text = str(path or "").strip()
    if not text:
        return False
    try:
        resolved = Path(text).expanduser().resolve()
    except OSError:
        return False
    if resolved == Path(resolved.anchor):  # filesystem root
        return False

    if project_dir is not None:
        try:
            allowed_root = (
                Path(project_dir).expanduser().resolve() / PENDING_ESCALATIONS_RELDIR
            )
        except OSError:
            return False
        return resolved != allowed_root and resolved.is_relative_to(allowed_root)

    return (
        resolved.parent.name == PENDING_ESCALATIONS_RELDIR.name
        and resolved.parent.parent.name == PENDING_ESCALATIONS_RELDIR.parent.name
    )


def remove_staged_escalation_dir(
    path: str | Path | None,
    project_dir: str | Path | None = None,
) -> bool:
    """Recursively delete a staging directory, refusing anything out of bounds.

    Returns True if a delete was attempted. An out-of-bounds path is logged
    and skipped rather than raising -- this guards a destructive operation,
    and a source carrying a currently-invalid ``staged_path`` should just
    leave that directory alone, not crash the caller.
    """
    if not _is_safe_to_delete(path, project_dir):
        logger.warning(
            "Refusing to delete staged escalation path outside %s: %r",
            PENDING_ESCALATIONS_RELDIR,
            str(path or ""),
        )
        return False
    shutil.rmtree(Path(str(path)).expanduser(), ignore_errors=True)
    return True


@dataclass
class EscalationRequest:
    project: object  # has .project_dir and .sources (list[OBBSource])
    source_names: list[str]
    variant: str
    overwrite: bool = False


@dataclass
class EscalationResult:
    # Names of sources that received a fresh `staged_review` this run.
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
    ``staged_review`` field. It does NOT touch the source's own
    canonical labels and does NOT register any new source -- a caller (the
    frame-granular review flow in ``jobs/staged_review.py``) accepts or
    rejects individual staged frames to promote or discard the staged result.

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
        if src.staged_review is not None and not overwrite:
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
        # nothing else ever revisits a replaced staged_review.
        old_pending = src.staged_review
        if old_pending is not None and old_pending.staged_path != str(staged_root):
            remove_staged_escalation_dir(old_pending.staged_path, project_root)

        remove_staged_escalation_dir(staged_root, project_root)
        (staged_root / "labels").mkdir(parents=True, exist_ok=True)

        # Recursive + path-mirroring: a source's images/labels can be nested
        # (e.g. images/train/..., images/val/...) -- source_import.py's
        # materializer can produce this layout. A flat top-level glob would
        # silently stage ZERO labels for such a source, and accept() would
        # then delete every real label with nothing to replace it.
        # IMG_EXTS (not a hardcoded jpg/png tuple): DetectKit's canonical
        # extension set also covers .bmp/.tif/.tiff/.webp. Missing one of
        # those staged no label for that image, and accept() then refused
        # forever on the missing-labels check.
        images = sorted(
            p for p in images_dir.rglob("*") if p.suffix.lower() in IMG_EXTS
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
                    # SAM2's box prompt is soft guidance, not a hard crop -- the
                    # predicted mask can extend past the source OBB. Clip to the
                    # OBB's own polygon (not just its aabb) before contouring so
                    # a rotated OBB's escalated mask stays bounded correctly.
                    mask = clip_mask_to_polygon(mask, box.polygon_px)
                    contour = mask_to_contour(mask)
                    if contour is not None:
                        result.primed += 1
                        poly = contour
                    else:  # fallback: original OBB corners as the polygon
                        result.fell_back += 1
                        poly = box.polygon_px
                    records.append(
                        LabelRecord(
                            # Preserve the original per-instance class id: a
                            # hardcoded 0 would collapse every class of a
                            # multi-class source on accept, which now
                            # overwrites the source's own labels in place.
                            class_id=box.class_id,
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
        src.staged_review = StagedReview(
            staged_path=str(staged_root),
            target_level=GeometryLevel.POLYGON.label,
            producer="sam2",
            producer_variant=req.variant,
            created_at=datetime.now().isoformat(),
        )
        result.staged.append(src.name)
    return result
