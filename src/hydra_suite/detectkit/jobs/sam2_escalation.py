"""SAM2 escalation orchestrator: existing OBB/box labels -> primed seg source."""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from PySide6.QtCore import Signal

from hydra_suite.core.inference.sam2.masks import mask_to_contour
from hydra_suite.data.al.escalation import LabelRecord
from hydra_suite.data.al.labels import write_label_file
from hydra_suite.detectkit.gui.models import OBBSource
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
    derived: list[str] = field(default_factory=list)
    primed: int = 0
    fell_back: int = 0
    # (source_name, reason) pairs for sources skipped because a "<name>_seg"
    # already exists and overwrite was not requested.
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
            f"Done: {result.primed} primed, {result.fell_back} fell back "
            f"(review these first)."
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
    """Escalate each named source to a new <name>_seg source (reviewed=False).

    Re-running escalation over a source whose "<name>_seg" already exists (as a
    project source entry and/or an on-disk directory) is guarded: by default the
    source is skipped (recorded in ``result.skipped``) rather than silently
    clobbering a derived source the user may already have reviewed. Pass
    ``overwrite=True`` to replace it in place (no duplicate source entries).
    """
    result = EscalationResult()
    by_name = _sources_by_name(req.project)
    todo = [
        by_name[n]
        for n in req.source_names
        if n in by_name and by_name[n].level != "polygon"
    ]
    for si, src in enumerate(todo):
        src_root = Path(src.path)
        images_dir = src_root / "images"
        labels_dir = src_root / "labels"
        out_name = f"{src.name}_seg"
        out_root = Path(req.project.project_dir) / "sources" / out_name

        existing = next((s for s in req.project.sources if s.name == out_name), None)
        if (existing is not None or out_root.exists()) and not overwrite:
            result.skipped.append(
                (
                    src.name,
                    f"'{out_name}' already exists; re-run with overwrite to replace it.",
                )
            )
            continue
        if existing is not None:
            # Replace in place: never leave a duplicate "<name>_seg" entry.
            req.project.sources.remove(existing)
        if overwrite and out_root.exists():
            # Clean overwrite: drop stale files (e.g. images removed upstream)
            # rather than merging with whatever was there before.
            shutil.rmtree(out_root)

        (out_root / "images").mkdir(parents=True, exist_ok=True)
        (out_root / "labels").mkdir(parents=True, exist_ok=True)

        images = sorted(
            p
            for p in images_dir.glob("*")
            if p.suffix.lower() in (".jpg", ".jpeg", ".png")
        )
        for ii, img_path in enumerate(images):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            label_path = labels_dir / f"{img_path.stem}.txt"
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
            write_label_file(
                out_root / "labels" / f"{img_path.stem}.txt",
                records,
                frame_size=(h, w),
                level=GeometryLevel.POLYGON,
            )
            shutil.copy2(img_path, out_root / "images" / img_path.name)
            if progress:
                progress(
                    int(100 * (si + (ii + 1) / max(len(images), 1)) / len(todo)),
                    f"{src.name}: {ii + 1}/{len(images)}",
                )

        (out_root / "classes.txt").write_text(
            (src_root / "classes.txt").read_text()
            if (src_root / "classes.txt").exists()
            else "object\n"
        )
        req.project.sources.append(
            OBBSource(
                path=str(out_root),
                name=out_name,
                level="polygon",
                reviewed=False,
                derived_from=src.name,
                sam2_variant=req.variant,
                source_kind="detectkit_sam2",
                imported=True,
            )
        )
        result.derived.append(out_name)
    return result
