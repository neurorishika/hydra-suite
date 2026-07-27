"""Sliced (tiled) training-data builder for DetectKit direct OBB models.

Tiles a merged OBB dataset so a direct model learns to detect at the SAME scale
SAHI feeds at inference. Tiles through ``utils.slice_geometry`` — the exact grid
the inference path uses (Approach B). See
docs/superpowers/specs/2026-07-27-detectkit-sahi-sliced-training-design.md.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from hydra_suite.utils.slice_geometry import (
    clip_polygon_to_tile,
    plan_tiles,
    polygon_area,
    tile_size_for_mode,
)

from .contracts import DatasetBuildResult
from .dataset_builders import (
    IMAGE_EXTS,
    _find_label_for_obb_image,
    _parse_geometry_label_lines,
)
from .geometry_levels import GeometryLevel


def measure_reference_body_px(labels, frame_wh) -> float:
    """Median OBB major axis (px) over a frame's normalized-point labels."""
    w, h = float(frame_wh[0]), float(frame_wh[1])
    majors: list[float] = []
    for _cls_id, pts_norm in labels:
        pts = np.asarray(pts_norm, dtype=np.float32).copy()
        pts[:, 0] *= w
        pts[:, 1] *= h
        if pts.shape[0] < 3:
            continue
        _c, (bw, bh), _a = cv2.minAreaRect(pts.astype(np.float32))
        majors.append(float(max(bw, bh)))
    if not majors:
        return 0.0
    return float(np.median(np.asarray(majors, dtype=np.float64)))


def project_to_level(poly_norm: np.ndarray, level: GeometryLevel) -> np.ndarray:
    """Re-derive a normalized (M,2) contour DOWN to ``level`` (contour space kept)."""
    poly = np.asarray(poly_norm, dtype=np.float32)
    if level == GeometryLevel.POLYGON:
        return poly
    if level == GeometryLevel.OBB:
        box = cv2.boxPoints(cv2.minAreaRect(poly))
        return np.asarray(box, dtype=np.float32)
    # AABB: axis-aligned envelope corners.
    x1, y1 = float(poly[:, 0].min()), float(poly[:, 1].min())
    x2, y2 = float(poly[:, 0].max()), float(poly[:, 1].max())
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)


def label_line_for_level(
    class_id: int, pts_norm: np.ndarray, level: GeometryLevel
) -> str:
    """Format one YOLO label line for ``level`` (coords clipped to [0,1], %.6f)."""
    pts = np.clip(np.asarray(pts_norm, dtype=np.float32), 0.0, 1.0)
    if level == GeometryLevel.AABB:
        x1, y1 = float(pts[:, 0].min()), float(pts[:, 1].min())
        x2, y2 = float(pts[:, 0].max()), float(pts[:, 1].max())
        cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
        bw, bh = max(0.0, x2 - x1), max(0.0, y2 - y1)
        return f"{int(class_id)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
    coords = " ".join(f"{float(v):.6f}" for v in pts.reshape(-1))
    return f"{int(class_id)} {coords}"


@dataclass
class SliceBuildParams:
    geometry_mode: str = "auto_object"
    imgsz: int = 640
    object_tile_fraction: float = 0.15
    slice_width: int = 0
    slice_height: int = 0
    overlap: float = 0.2
    min_area_ratio: float = 0.1
    negative_tile_fraction: float = 0.15
    target_sizes: list[float] = field(default_factory=lambda: [200.0, 300.0, 400.0])
    full_frame_mix: bool = True
    reference_body_px: float = 0.0


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _tile_one_image(img, labels, tile, level, min_area_ratio):
    """Crop one tile and emit kept label lines. ``labels`` = (cls, (P,2) frame-norm)."""
    x0, y0, x1, y1 = tile
    fh, fw = img.shape[:2]
    xi0, yi0 = max(0, int(x0)), max(0, int(y0))
    xi1, yi1 = min(fw, int(x1)), min(fh, int(y1))
    crop = img[yi0:yi1, xi0:xi1]
    tw, th = float(xi1 - xi0), float(yi1 - yi0)
    lines: list[str] = []
    if tw <= 0 or th <= 0:
        return crop, lines
    for cls_id, poly_norm in labels:
        poly_px = np.asarray(poly_norm, dtype=np.float32).copy()
        poly_px[:, 0] *= fw
        poly_px[:, 1] *= fh
        full_area = polygon_area(poly_px)
        if full_area <= 1e-6:
            continue
        clipped = clip_polygon_to_tile(poly_px, (xi0, yi0, xi1, yi1))
        if clipped is None:
            continue
        if polygon_area(clipped) / full_area < min_area_ratio:
            continue
        local = clipped.copy()
        local[:, 0] = (local[:, 0] - xi0) / tw
        local[:, 1] = (local[:, 1] - yi0) / th
        derived = project_to_level(np.clip(local, 0.0, 1.0), level)
        lines.append(label_line_for_level(int(cls_id), derived, level))
    return crop, lines


def _iter_dataset_items(merged_dir: Path):
    """Yield (split, image_path, label_path) for a merged OBB dataset."""
    for split in ("train", "val", "test"):
        src_img = merged_dir / "images" / split
        src_lbl = merged_dir / "labels" / split
        if not src_img.exists():
            continue
        for img_path in sorted(src_img.rglob("*")):
            if img_path.suffix.lower() not in IMAGE_EXTS:
                continue
            lbl_path = _find_label_for_obb_image(img_path, src_img, src_lbl)
            if lbl_path is not None:
                yield split, img_path, lbl_path


def _tile_sizes_for_params(params, reference_body_px) -> list[tuple[int, int]]:
    """Resolve the (deduped) list of (w,h) tile sizes to emit.

    ``auto_object`` with a measured reference and a non-empty ``target_sizes``
    fans out one square tile per target apparent size (target/imgsz -> fraction);
    otherwise a single size from the geometry mode.
    """
    if (
        params.geometry_mode == "auto_object"
        and reference_body_px > 0
        and params.target_sizes
    ):
        sizes: list[tuple[int, int]] = []
        for target in params.target_sizes:
            frac = max(0.01, min(0.9, float(target) / max(1, params.imgsz)))
            w, h = tile_size_for_mode(
                geometry_mode="auto_object",
                imgsz=params.imgsz,
                reference_body_px=reference_body_px,
                object_tile_fraction=frac,
                slice_width=0,
                slice_height=0,
            )
            if (w, h) not in sizes:
                sizes.append((w, h))
        if sizes:
            return sizes
    w, h = tile_size_for_mode(
        geometry_mode=params.geometry_mode,
        imgsz=params.imgsz,
        reference_body_px=reference_body_px,
        object_tile_fraction=params.object_tile_fraction,
        slice_width=params.slice_width,
        slice_height=params.slice_height,
    )
    return [(w, h)]


def build_sliced_obb_dataset(
    merged_obb_dataset_dir, output_root, *, level, params, seed=42
):
    merged_dir = Path(merged_obb_dataset_dir).expanduser().resolve()
    out_root = Path(output_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    out_dir = out_root / f"sliced_obb_{_timestamp()}"
    for split in ("train", "val", "test"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    rng = random.Random(int(seed))
    counts = {"train": 0, "val": 0, "test": 0, "tiles": 0, "negatives": 0, "objects": 0}
    class_names = _read_class_names(merged_dir)

    for split, img_path, lbl_path in _iter_dataset_items(merged_dir):
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None or img.size == 0:
            continue
        fh, fw = img.shape[:2]
        labels = _parse_geometry_label_lines(lbl_path)
        ref_px = params.reference_body_px or measure_reference_body_px(labels, (fw, fh))
        for tile_w, tile_h in _tile_sizes_for_params(params, ref_px):
            try:
                plan = plan_tiles(
                    (fh, fw), tile_w, tile_h, params.overlap, params.overlap
                )
            except ValueError:
                continue
            for ti, tile in enumerate(plan.tiles):
                crop, lines = _tile_one_image(
                    img, labels, tile, level, params.min_area_ratio
                )
                if crop.size == 0:
                    continue
                is_negative = not lines
                if is_negative and rng.random() >= params.negative_tile_fraction:
                    continue
                stem = f"{img_path.stem}_t{tile_w}x{tile_h}_{ti:04d}"
                cv2.imwrite(str(out_dir / "images" / split / f"{stem}.jpg"), crop)
                (out_dir / "labels" / split / f"{stem}.txt").write_text(
                    ("\n".join(lines) + "\n") if lines else "", encoding="utf-8"
                )
                counts[split] += 1
                counts["tiles"] += 1
                counts["objects"] += len(lines)
                if is_negative:
                    counts["negatives"] += 1

        if params.full_frame_mix:
            lines = []
            for cls_id, poly_norm in labels:
                derived = project_to_level(
                    np.clip(np.asarray(poly_norm, np.float32), 0, 1), level
                )
                lines.append(label_line_for_level(int(cls_id), derived, level))
            stem = f"{img_path.stem}_full"
            cv2.imwrite(str(out_dir / "images" / split / f"{stem}.jpg"), img)
            (out_dir / "labels" / split / f"{stem}.txt").write_text(
                ("\n".join(lines) + "\n") if lines else "", encoding="utf-8"
            )
            counts[split] += 1
            counts["objects"] += len(lines)

    _write_sliced_yaml(out_dir, class_names)
    manifest = {
        "type": "sliced_obb",
        "source": str(merged_dir),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "level": level.label,
        "counts": counts,
        "slice_geometry": _slice_geometry_manifest(params),
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return DatasetBuildResult(
        dataset_dir=str(out_dir), stats=manifest, manifest_path=str(manifest_path)
    )


def _read_class_names(merged_dir: Path) -> list[str]:
    yaml_path = merged_dir / "dataset.yaml"
    if not yaml_path.exists():
        return ["object"]
    try:
        import yaml

        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
        names = data.get("names", {})
        if isinstance(names, dict):
            return [str(names[k]) for k in sorted(names, key=lambda x: int(x))] or [
                "object"
            ]
        if isinstance(names, list):
            return [str(n) for n in names] or ["object"]
    except Exception:
        pass
    return ["object"]


def _write_sliced_yaml(out_dir: Path, class_names: list[str]) -> None:
    lines = [
        f"path: {out_dir.resolve()}",
        "train: images/train",
        "val: images/val",
        "names:",
    ]
    lines.extend(f"  {i}: {n}" for i, n in enumerate(class_names))
    (out_dir / "dataset.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _slice_geometry_manifest(params) -> dict:
    return {
        "geometry_mode": params.geometry_mode,
        "imgsz": params.imgsz,
        "object_tile_fraction": params.object_tile_fraction,
        "slice_width": params.slice_width,
        "slice_height": params.slice_height,
        "overlap": params.overlap,
        "min_area_ratio": params.min_area_ratio,
        "negative_tile_fraction": params.negative_tile_fraction,
        "target_sizes": list(params.target_sizes),
        "full_frame_mix": params.full_frame_mix,
        "reference_body_px": params.reference_body_px,
    }
