"""Pure prompt geometry for SAM2 escalation (no SAM2 import)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class SourceBox:
    aabb: tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels
    center: tuple[float, float]
    polygon_px: list[tuple[float, float]]  # original label polygon (fallback)
    # The label line's own class id. Carried through escalation so a
    # multi-class source keeps its per-instance class assignments when a
    # staged escalation is accepted over its canonical labels.
    class_id: int = 0


@dataclass
class Prompt:
    box_xyxy: tuple[float, float, float, float]
    positive_points: list[tuple[float, float]]
    negative_points: list[tuple[float, float]]


def _aabb_of(points: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (min(xs), min(ys), max(xs), max(ys))


def read_boxes_from_label(label_path: Path, img_w: int, img_h: int) -> list[SourceBox]:
    """Parse normalized YOLO aabb (5-field) / obb (9-field) lines to pixel boxes."""
    out: list[SourceBox] = []
    try:
        text = Path(label_path).read_text(encoding="utf-8")
    except Exception:
        return out
    for line in text.splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        try:
            class_id = int(float(parts[0]))
            vals = [float(v) for v in parts[1:]]
        except ValueError:
            continue
        if len(vals) == 4:  # aabb: cx cy w h (normalized)
            cx, cy, w, h = vals
            poly = [
                ((cx - w / 2) * img_w, (cy - h / 2) * img_h),
                ((cx + w / 2) * img_w, (cy - h / 2) * img_h),
                ((cx + w / 2) * img_w, (cy + h / 2) * img_h),
                ((cx - w / 2) * img_w, (cy + h / 2) * img_h),
            ]
        elif len(vals) == 8:  # obb: x1 y1 .. x4 y4 (normalized)
            poly = [(vals[i] * img_w, vals[i + 1] * img_h) for i in range(0, 8, 2)]
        else:
            continue
        aabb = _aabb_of(poly)
        center = (
            sum(p[0] for p in poly) / len(poly),
            sum(p[1] for p in poly) / len(poly),
        )
        out.append(
            SourceBox(aabb=aabb, center=center, polygon_px=poly, class_id=int(class_id))
        )
    return out


def _overlaps(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> bool:
    ix = min(a[2], b[2]) - max(a[0], b[0])
    iy = min(a[3], b[3]) - max(a[1], b[1])
    return ix > 0 and iy > 0


def build_prompts(boxes: list[SourceBox]) -> list[Prompt]:
    """Box + center-positive + overlapping-neighbor-center negatives, per box."""
    prompts: list[Prompt] = []
    for i, box in enumerate(boxes):
        negatives = [
            other.center
            for j, other in enumerate(boxes)
            if j != i and _overlaps(box.aabb, other.aabb)
        ]
        prompts.append(
            Prompt(
                box_xyxy=box.aabb,
                positive_points=[box.center],
                negative_points=negatives,
            )
        )
    return prompts
