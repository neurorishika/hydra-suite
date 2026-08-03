"""SAM2 escalation orchestrator: existing OBB/box labels -> primed seg source."""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2

from hydra_suite.core.inference.sam2.masks import mask_to_contour
from hydra_suite.detectkit.gui.models import OBBSource
from hydra_suite.detectkit.jobs.al_worker import _write_geometry_label

from .sam2_prompts import build_prompts, read_boxes_from_label


@dataclass
class EscalationRequest:
    project: object  # has .project_dir and .sources (list[OBBSource])
    source_names: list[str]
    variant: str


@dataclass
class EscalationResult:
    derived: list[str] = field(default_factory=list)
    primed: int = 0
    fell_back: int = 0


def _sources_by_name(project) -> dict[str, OBBSource]:
    return {s.name: s for s in project.sources}


def run_escalation(
    req: EscalationRequest,
    executor,
    *,
    progress: Callable[[int, str], None] | None = None,
) -> EscalationResult:
    """Escalate each named source to a new <name>_seg source (reviewed=False)."""
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
            records = []
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
                    records.append((0.0, 0.0, 0.0, 0.0, 0.0, 1.0, poly))
            _write_geometry_label(
                out_root / "labels" / f"{img_path.stem}.txt", records, (h, w)
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
