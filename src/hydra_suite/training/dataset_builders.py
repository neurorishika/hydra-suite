"""Dataset builders for MAT multi-role training."""

from __future__ import annotations

import hashlib
import json
import math
import random
import shutil
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .class_mapping import build_class_id_map, resolve_dataset_class_names
from .contracts import DatasetBuildResult, SourceDataset, SplitConfig, TrainingRole
from .dataset_inspector import (
    DatasetInspection,
    inspect_obb_or_detect_dataset,
    split_items_for_training,
    stratified_split_items,
)
from .geometry_levels import GeometryLevel


def _normalize_class_names(
    class_names: list[str] | None = None,
    class_name: str | None = None,
) -> list[str]:
    """Normalize single-class and multi-class inputs into an ordered list."""
    if class_names is not None:
        candidates = class_names
    elif class_name is not None:
        candidates = [class_name]
    else:
        candidates = []

    resolved = [str(name).strip() for name in candidates if str(name).strip()]
    return resolved or ["object"]


def _validate_class_name_coverage(
    class_names: list[str],
    class_ids: set[int],
    *,
    dataset_label: str,
) -> None:
    """Ensure dataset class ids are covered by the provided class-name list."""
    missing = sorted(class_id for class_id in class_ids if class_id >= len(class_names))
    if missing:
        raise RuntimeError(
            f"{dataset_label} contains class ids {missing} but only {len(class_names)} class names were configured."
        )


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _hash_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _normalize_split_cfg(cfg: SplitConfig) -> tuple[float, float, float]:
    train = float(cfg.train)
    val = float(cfg.val)
    test = float(cfg.test)
    if train < 0 or val < 0 or test < 0:
        raise RuntimeError("Split ratios must be non-negative.")
    total = train + val + test
    if total <= 0:
        raise RuntimeError("Split ratios sum to zero.")
    return train / total, val / total, test / total


def _write_dataset_yaml(
    root_dir: Path,
    class_names: list[str] | None = None,
    *,
    class_name: str | None = None,
    include_test: bool = False,
) -> Path:
    resolved_class_names = _normalize_class_names(
        class_names=class_names,
        class_name=class_name,
    )
    data = {
        "path": str(root_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {idx: name for idx, name in enumerate(resolved_class_names)},
    }
    if include_test:
        data["test"] = "images/test"

    out = root_dir / "dataset.yaml"
    try:
        import yaml  # type: ignore

        out.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    except Exception:
        lines = [
            f"path: {root_dir.resolve()}",
            "train: images/train",
            "val: images/val",
        ]
        if include_test:
            lines.append("test: images/test")
        lines.append("names:")
        lines.extend(
            f"  {idx}: {name}" for idx, name in enumerate(resolved_class_names)
        )
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def _write_manifest(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _safe_name(text: str) -> str:
    return (
        "".join(
            ch if ch.isalnum() or ch in "-_" else "_" for ch in str(text or "")
        ).strip("_")
        or "src"
    )


def _parse_obb_label_lines(lbl_path: Path) -> list[tuple[int, np.ndarray]]:
    out: list[tuple[int, np.ndarray]] = []
    for ln in lbl_path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        parts = ln.split()
        if len(parts) != 9:
            raise RuntimeError(f"Invalid OBB label format in {lbl_path}: {ln}")
        cls_id = int(float(parts[0]))
        vals = np.asarray([float(v) for v in parts[1:]], dtype=np.float32).reshape(4, 2)
        out.append((cls_id, vals))
    return out


def _parse_geometry_label_lines(lbl_path: Path) -> list[tuple[int, np.ndarray]]:
    """Parse a label file of mixed geometry into (class_id, (P,2) points).

    - 5 fields (cx cy w h) expand to the 4-corner axis-aligned quad.
    - >=7 odd fields are read as a normalized point list of P = (fields-1)/2 points.
    Raises on malformed lines.
    """
    out: list[tuple[int, np.ndarray]] = []
    for raw in lbl_path.read_text(encoding="utf-8").splitlines():
        ln = raw.strip()
        if not ln:
            continue
        parts = ln.split()
        cls_id = int(float(parts[0]))
        coords = [float(v) for v in parts[1:]]
        if len(parts) == 5:
            cx, cy, w, h = coords
            x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
            pts = np.asarray([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
        elif len(coords) >= 6 and len(coords) % 2 == 0:
            pts = np.asarray(coords, dtype=np.float32).reshape(-1, 2)
        else:
            raise RuntimeError(f"Invalid geometry label line in {lbl_path}: {ln}")
        out.append((cls_id, pts))
    return out


def _render_filtered_obb_label(
    lbl: Path,
    *,
    class_id_map: dict[int, int] | None = None,
    remap_single_class: bool = False,
) -> tuple[str, set[int]]:
    """Return filtered/remapped OBB label text and the kept class ids."""
    detections = _parse_obb_label_lines(lbl)
    lines: list[str] = []
    kept_class_ids: set[int] = set()
    for cls_id, poly in detections:
        mapped_id = int(cls_id)
        if class_id_map is not None:
            mapped = class_id_map.get(mapped_id)
            if mapped is None:
                continue
            mapped_id = int(mapped)
        elif remap_single_class:
            mapped_id = 0

        coords = " ".join(f"{float(v):.6f}" for v in poly.reshape(-1))
        lines.append(f"{mapped_id} {coords}")
        kept_class_ids.add(mapped_id)

    text = "\n".join(lines) + ("\n" if lines else "")
    return text, kept_class_ids


def eligible_sources(sources):
    """Drop reviewed=False sources from a training build, returning skip messages."""
    kept, messages = [], []
    for s in sources:
        if getattr(s, "reviewed", True):
            kept.append(s)
        else:
            messages.append(
                f"Source '{s.name}' is unreviewed (SAM2-primed) — review it in "
                f"X-AnyLabeling and Mark reviewed before it can be used for training."
            )
    return kept, messages


def merge_obb_sources(
    sources: list[SourceDataset],
    output_root: str | Path,
    split_cfg: SplitConfig,
    class_name: str | None = None,
    seed: int = 42,
    dedup: bool = True,
    remap_single_class: bool = True,
    class_names: list[str] | None = None,
) -> DatasetBuildResult:
    """Merge OBB sources into one canonical dataset (non-destructive)."""
    resolved_class_names = _normalize_class_names(
        class_names=class_names,
        class_name=class_name,
    )
    if remap_single_class:
        resolved_class_names = resolved_class_names[:1]

    sources, skipped_sources = eligible_sources(sources)

    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    out_dir = root / f"combined_obb_{_timestamp()}"

    for split in ("train", "val", "test"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    split_tuple = _normalize_split_cfg(split_cfg)

    rng = random.Random(int(seed))
    seen_hashes: set[str] = set()
    split_counts = {"train": 0, "val": 0, "test": 0}
    source_items: dict[str, int] = defaultdict(int)
    duplicate_skipped = 0
    encountered_class_ids: set[int] = set()

    for src in sources:
        src_name = _safe_name(src.name or Path(src.path).name)
        inspection: DatasetInspection = inspect_obb_or_detect_dataset(src.path)
        class_id_map: dict[int, int] | None = None
        try:
            source_class_names = resolve_dataset_class_names(
                src.path, inspection.class_names
            )
            class_id_map = build_class_id_map(source_class_names, resolved_class_names)
        except RuntimeError:
            if not remap_single_class:
                raise
        if "all" in inspection.splits:
            split_items = stratified_split_items(
                list(inspection.splits["all"]), split_tuple, seed=seed
            )
        else:
            split_items = split_items_for_training(inspection, split_tuple, seed=seed)

        for split in ("train", "val", "test"):
            items = list(split_items.get(split, []))
            rng.shuffle(items)
            for idx, item in enumerate(items):
                img = Path(item.image_path)
                lbl = Path(item.label_path)
                label_text, kept_class_ids = _render_filtered_obb_label(
                    lbl,
                    class_id_map=class_id_map,
                    remap_single_class=remap_single_class,
                )
                if not label_text:
                    continue
                if dedup:
                    file_hash = _hash_file(img)
                    if file_hash in seen_hashes:
                        duplicate_skipped += 1
                        continue
                    seen_hashes.add(file_hash)

                stem = f"{src_name}_{img.stem}_{idx:06d}"
                img_dst = out_dir / "images" / split / f"{stem}{img.suffix.lower()}"
                lbl_dst = out_dir / "labels" / split / f"{stem}.txt"

                shutil.copy2(img, img_dst)
                lbl_dst.write_text(label_text, encoding="utf-8")
                encountered_class_ids.update(kept_class_ids)

                split_counts[split] += 1
                source_items[src_name] += 1

    if split_counts["train"] <= 0 or split_counts["val"] <= 0:
        raise RuntimeError(
            "Merged dataset has empty split(s). Ensure source datasets provide enough labeled frames."
        )

    include_test = split_counts["test"] > 0
    _validate_class_name_coverage(
        resolved_class_names,
        encountered_class_ids,
        dataset_label="Merged OBB dataset",
    )
    _write_dataset_yaml(
        out_dir,
        class_names=resolved_class_names,
        include_test=include_test,
    )
    manifest = {
        "type": "merged_obb",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "sources": [asdict(s) for s in sources],
        "split_cfg": asdict(split_cfg),
        "seed": int(seed),
        "dedup": bool(dedup),
        "remap_single_class": bool(remap_single_class),
        "class_names": resolved_class_names,
        "counts": split_counts,
        "source_items": dict(source_items),
        "duplicates_skipped": int(duplicate_skipped),
        "skipped_sources": list(skipped_sources),
    }
    manifest_path = out_dir / "manifest.json"
    _write_manifest(manifest_path, manifest)

    return DatasetBuildResult(
        dataset_dir=str(out_dir),
        stats=manifest,
        manifest_path=str(manifest_path),
    )


def _convert_obb_to_aabb(poly: np.ndarray) -> tuple[float, float, float, float]:
    xs = poly[:, 0]
    ys = poly[:, 1]
    x1 = float(np.clip(np.min(xs), 0.0, 1.0))
    y1 = float(np.clip(np.min(ys), 0.0, 1.0))
    x2 = float(np.clip(np.max(xs), 0.0, 1.0))
    y2 = float(np.clip(np.max(ys), 0.0, 1.0))
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    cx = x1 + w * 0.5
    cy = y1 + h * 0.5
    return cx, cy, w, h


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _find_label_for_obb_image(
    img_path: Path, src_img: Path, src_lbl: Path
) -> Path | None:
    """Locate the label file corresponding to an image, or return None."""
    rel = img_path.relative_to(src_img)
    lbl_path = (src_lbl / rel).with_suffix(".txt")
    if lbl_path.exists():
        return lbl_path
    lbl_path = src_lbl / f"{img_path.stem}.txt"
    return lbl_path if lbl_path.exists() else None


def _unique_dst_pair(out_dir: Path, split: str, img_path: Path) -> tuple[Path, Path]:
    """Generate a unique (image, label) destination path pair."""
    dst_img = out_dir / "images" / split / img_path.name
    dst_lbl = out_dir / "labels" / split / f"{img_path.stem}.txt"
    counter = 1
    while dst_img.exists() or dst_lbl.exists():
        dst_img = (
            out_dir
            / "images"
            / split
            / f"{img_path.stem}_{counter}{img_path.suffix.lower()}"
        )
        dst_lbl = out_dir / "labels" / split / f"{img_path.stem}_{counter}.txt"
        counter += 1
    return dst_img, dst_lbl


def derive_detect_dataset_from_obb(
    obb_dataset_dir: str | Path,
    output_root: str | Path,
    class_name: str | None = None,
    *,
    class_names: list[str] | None = None,
) -> DatasetBuildResult:
    """Create YOLO-detect dataset by converting OBB polygons to AABB labels."""
    resolved_class_names = _normalize_class_names(
        class_names=class_names,
        class_name=class_name,
    )

    src = Path(obb_dataset_dir).expanduser().resolve()
    out_root = Path(output_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    out_dir = out_root / f"derived_detect_{_timestamp()}"

    for split in ("train", "val", "test"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    counts = {"train": 0, "val": 0, "test": 0, "objects": 0}
    encountered_class_ids: set[int] = set()

    for split in ("train", "val", "test"):
        src_img = src / "images" / split
        src_lbl = src / "labels" / split
        if not src_img.exists():
            continue
        for img_path in sorted(src_img.rglob("*")):
            if img_path.suffix.lower() not in IMAGE_EXTS:
                continue
            lbl_path = _find_label_for_obb_image(img_path, src_img, src_lbl)
            if lbl_path is None:
                continue

            detections = _parse_geometry_label_lines(lbl_path)
            out_lines = []
            for cls_id, pts in detections:
                encountered_class_ids.add(int(cls_id))
                cx, cy, bw, bh = _convert_obb_to_aabb(pts)
                out_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            if not out_lines:
                continue

            dst_img, dst_lbl = _unique_dst_pair(out_dir, split, img_path)
            shutil.copy2(img_path, dst_img)
            dst_lbl.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
            counts[split] += 1
            counts["objects"] += len(out_lines)

    include_test = counts["test"] > 0
    _validate_class_name_coverage(
        resolved_class_names,
        encountered_class_ids,
        dataset_label="Derived detect dataset",
    )
    _write_dataset_yaml(
        out_dir,
        class_names=resolved_class_names,
        include_test=include_test,
    )
    manifest = {
        "type": "derived_detect",
        "source": str(src),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "counts": counts,
    }
    manifest_path = out_dir / "manifest.json"
    _write_manifest(manifest_path, manifest)
    return DatasetBuildResult(
        str(out_dir), stats=manifest, manifest_path=str(manifest_path)
    )


def _clip_crop(
    x1: float, y1: float, x2: float, y2: float, w: int, h: int
) -> tuple[int, int, int, int] | None:
    xi1 = int(max(0, math.floor(x1)))
    yi1 = int(max(0, math.floor(y1)))
    xi2 = int(min(w, math.ceil(x2)))
    yi2 = int(min(h, math.ceil(y2)))
    if xi2 <= xi1 or yi2 <= yi1:
        return None
    return xi1, yi1, xi2, yi2


def _extract_crop_for_object(
    img: np.ndarray,
    poly_norm: np.ndarray,
    pad_ratio: float,
    min_crop_size_px: int,
    enforce_square: bool,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Extract a padded crop and re-normalize the polygon into crop space.

    Returns (crop_image, normalized_polygon) or None if the crop is empty.
    """
    h, w = img.shape[:2]
    poly_px = np.zeros((4, 2), dtype=np.float32)
    poly_px[:, 0] = poly_norm[:, 0] * float(w)
    poly_px[:, 1] = poly_norm[:, 1] * float(h)

    x1, x2 = float(np.min(poly_px[:, 0])), float(np.max(poly_px[:, 0]))
    y1, y2 = float(np.min(poly_px[:, 1])), float(np.max(poly_px[:, 1]))

    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    cx, cy = x1 + bw * 0.5, y1 + bh * 0.5

    crop_w = max(float(min_crop_size_px), bw * (1.0 + 2.0 * max(0.0, pad_ratio)))
    crop_h = max(float(min_crop_size_px), bh * (1.0 + 2.0 * max(0.0, pad_ratio)))
    if enforce_square:
        crop_w = crop_h = max(crop_w, crop_h)

    c = _clip_crop(
        cx - crop_w * 0.5, cy - crop_h * 0.5, cx + crop_w * 0.5, cy + crop_h * 0.5, w, h
    )
    if c is None:
        return None
    xi1, yi1, xi2, yi2 = c
    crop = img[yi1:yi2, xi1:xi2]
    if crop is None or crop.size == 0:
        return None
    ch, cw = crop.shape[:2]
    if ch <= 0 or cw <= 0:
        return None

    poly_crop = poly_px.copy()
    poly_crop[:, 0] = np.clip((poly_crop[:, 0] - float(xi1)) / float(cw), 0.0, 1.0)
    poly_crop[:, 1] = np.clip((poly_crop[:, 1] - float(yi1)) / float(ch), 0.0, 1.0)
    return crop, poly_crop


def _unique_crop_output_paths(
    out_dir: Path,
    split: str,
    stem: str,
) -> tuple[Path, Path]:
    """Generate a unique destination pair for one cropped OBB object."""
    dst_img = out_dir / "images" / split / f"{stem}.jpg"
    dst_lbl = out_dir / "labels" / split / f"{stem}.txt"
    counter = 1
    while dst_img.exists() or dst_lbl.exists():
        dst_img = out_dir / "images" / split / f"{stem}_{counter}.jpg"
        dst_lbl = out_dir / "labels" / split / f"{stem}_{counter}.txt"
        counter += 1
    return dst_img, dst_lbl


def _process_crop_obb_image(
    img_path: Path,
    lbl_path: Path,
    out_dir: Path,
    split: str,
    pad_ratio: float,
    min_crop_size_px: int,
    enforce_square: bool,
) -> tuple[int, set[int]]:
    """Create crop-domain OBB samples for one source image."""
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None or img.size == 0:
        return 0, set()

    written = 0
    class_ids: set[int] = set()
    detections = _parse_obb_label_lines(lbl_path)
    for obj_idx, (cls_id, poly_norm) in enumerate(detections):
        class_ids.add(int(cls_id))
        result = _extract_crop_for_object(
            img, poly_norm, pad_ratio, min_crop_size_px, enforce_square
        )
        if result is None:
            continue

        crop, poly_crop = result
        stem = f"{img_path.stem}__obj{obj_idx:03d}"
        dst_img, dst_lbl = _unique_crop_output_paths(out_dir, split, stem)
        cv2.imwrite(str(dst_img), crop)
        coords = " ".join(f"{float(v):.6f}" for v in poly_crop.reshape(-1))
        dst_lbl.write_text(f"{cls_id} {coords}\n", encoding="utf-8")
        written += 1

    return written, class_ids


def derive_crop_obb_dataset_from_obb(
    obb_dataset_dir: str | Path,
    output_root: str | Path,
    class_name: str | None = None,
    *,
    class_names: list[str] | None = None,
    pad_ratio: float = 0.15,
    min_crop_size_px: int = 64,
    enforce_square: bool = True,
) -> DatasetBuildResult:
    """Create crop-domain OBB dataset for sequential stage-2 training."""
    resolved_class_names = _normalize_class_names(
        class_names=class_names,
        class_name=class_name,
    )

    src = Path(obb_dataset_dir).expanduser().resolve()
    out_root = Path(output_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    out_dir = out_root / f"derived_crop_obb_{_timestamp()}"

    for split in ("train", "val", "test"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    counts = {"train": 0, "val": 0, "test": 0, "objects": 0}
    encountered_class_ids: set[int] = set()

    for split in ("train", "val", "test"):
        src_img = src / "images" / split
        src_lbl = src / "labels" / split
        if not src_img.exists():
            continue
        for img_path in sorted(src_img.rglob("*")):
            if img_path.suffix.lower() not in IMAGE_EXTS:
                continue
            lbl_path = _find_label_for_obb_image(img_path, src_img, src_lbl)
            if lbl_path is None:
                continue
            written, class_ids = _process_crop_obb_image(
                img_path,
                lbl_path,
                out_dir,
                split,
                pad_ratio,
                min_crop_size_px,
                enforce_square,
            )
            encountered_class_ids.update(class_ids)
            counts[split] += written
            counts["objects"] += written

    include_test = counts["test"] > 0
    _validate_class_name_coverage(
        resolved_class_names,
        encountered_class_ids,
        dataset_label="Derived crop OBB dataset",
    )
    _write_dataset_yaml(
        out_dir,
        class_names=resolved_class_names,
        include_test=include_test,
    )
    manifest = {
        "type": "derived_crop_obb",
        "source": str(src),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "pad_ratio": float(pad_ratio),
        "min_crop_size_px": int(min_crop_size_px),
        "enforce_square": bool(enforce_square),
        "counts": counts,
    }
    manifest_path = out_dir / "manifest.json"
    _write_manifest(manifest_path, manifest)
    return DatasetBuildResult(
        str(out_dir), stats=manifest, manifest_path=str(manifest_path)
    )


def derive_segment_dataset_from_source(
    src_dataset_dir: str | Path,
    output_root: str | Path,
    class_name: str | None = None,
    *,
    class_names: list[str] | None = None,
) -> DatasetBuildResult:
    """YOLO-seg passthrough: copy images and normalized-contour labels verbatim."""
    resolved_class_names = _normalize_class_names(
        class_names=class_names,
        class_name=class_name,
    )

    src = Path(src_dataset_dir).expanduser().resolve()
    out_root = Path(output_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    out_dir = out_root / f"derived_segment_{_timestamp()}"

    for split in ("train", "val", "test"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    counts = {"train": 0, "val": 0, "test": 0, "objects": 0}
    encountered_class_ids: set[int] = set()

    for split in ("train", "val", "test"):
        src_img = src / "images" / split
        src_lbl = src / "labels" / split
        if not src_img.exists():
            continue
        for img_path in sorted(src_img.rglob("*")):
            if img_path.suffix.lower() not in IMAGE_EXTS:
                continue
            lbl_path = _find_label_for_obb_image(img_path, src_img, src_lbl)
            if lbl_path is None:
                continue

            detections = _parse_geometry_label_lines(lbl_path)
            if not detections:
                continue
            for cls_id, _pts in detections:
                encountered_class_ids.add(int(cls_id))

            dst_img, dst_lbl = _unique_dst_pair(out_dir, split, img_path)
            shutil.copy2(img_path, dst_img)
            shutil.copy2(lbl_path, dst_lbl)
            counts[split] += 1
            counts["objects"] += len(detections)

    include_test = counts["test"] > 0
    _validate_class_name_coverage(
        resolved_class_names,
        encountered_class_ids,
        dataset_label="Derived segment dataset",
    )
    _write_dataset_yaml(
        out_dir,
        class_names=resolved_class_names,
        include_test=include_test,
    )
    manifest = {
        "type": "derived_segment",
        "source": str(src),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "counts": counts,
    }
    manifest_path = out_dir / "manifest.json"
    _write_manifest(manifest_path, manifest)
    return DatasetBuildResult(
        str(out_dir), stats=manifest, manifest_path=str(manifest_path)
    )


def _extract_crop_for_contour(
    img: np.ndarray,
    poly_norm: np.ndarray,
    pad_ratio: float,
    min_crop_size_px: int,
    enforce_square: bool,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Crop around a contour's AABB; return (crop, contour re-normalized+clipped to crop)."""
    h, w = img.shape[:2]
    pts_px = poly_norm.astype(np.float32).copy()
    pts_px[:, 0] *= float(w)
    pts_px[:, 1] *= float(h)

    x1, x2 = float(pts_px[:, 0].min()), float(pts_px[:, 0].max())
    y1, y2 = float(pts_px[:, 1].min()), float(pts_px[:, 1].max())
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    cx, cy = x1 + bw * 0.5, y1 + bh * 0.5

    crop_w = max(float(min_crop_size_px), bw * (1.0 + 2.0 * max(0.0, pad_ratio)))
    crop_h = max(float(min_crop_size_px), bh * (1.0 + 2.0 * max(0.0, pad_ratio)))
    if enforce_square:
        crop_w = crop_h = max(crop_w, crop_h)

    c = _clip_crop(
        cx - crop_w * 0.5, cy - crop_h * 0.5, cx + crop_w * 0.5, cy + crop_h * 0.5, w, h
    )
    if c is None:
        return None
    xi1, yi1, xi2, yi2 = c
    crop = img[yi1:yi2, xi1:xi2]
    if crop is None or crop.size == 0:
        return None
    ch, cw = crop.shape[:2]
    if ch <= 0 or cw <= 0:
        return None

    contour = pts_px.copy()
    contour[:, 0] = np.clip((contour[:, 0] - float(xi1)) / float(cw), 0.0, 1.0)
    contour[:, 1] = np.clip((contour[:, 1] - float(yi1)) / float(ch), 0.0, 1.0)
    return crop, contour


def _process_crop_segment_image(
    img_path: Path,
    lbl_path: Path,
    out_dir: Path,
    split: str,
    pad_ratio: float,
    min_crop_size_px: int,
    enforce_square: bool,
) -> tuple[int, set[int]]:
    """Create crop-domain segmentation samples for one source image."""
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None or img.size == 0:
        return 0, set()

    written = 0
    class_ids: set[int] = set()
    for obj_idx, (cls_id, poly_norm) in enumerate(
        _parse_geometry_label_lines(lbl_path)
    ):
        class_ids.add(int(cls_id))
        result = _extract_crop_for_contour(
            img, poly_norm, pad_ratio, min_crop_size_px, enforce_square
        )
        if result is None:
            continue

        crop, contour = result
        stem = f"{img_path.stem}__obj{obj_idx:03d}"
        dst_img, dst_lbl = _unique_crop_output_paths(out_dir, split, stem)
        cv2.imwrite(str(dst_img), crop)
        coords = " ".join(f"{float(v):.6f}" for v in contour.reshape(-1))
        dst_lbl.write_text(f"{cls_id} {coords}\n", encoding="utf-8")
        written += 1

    return written, class_ids


def derive_crop_segment_dataset_from_source(
    src_dataset_dir: str | Path,
    output_root: str | Path,
    class_name: str | None = None,
    *,
    class_names: list[str] | None = None,
    pad_ratio: float = 0.15,
    min_crop_size_px: int = 64,
    enforce_square: bool = True,
) -> DatasetBuildResult:
    """Crop-domain segmentation dataset: clip each contour to its padded crop."""
    resolved_class_names = _normalize_class_names(
        class_names=class_names,
        class_name=class_name,
    )

    src = Path(src_dataset_dir).expanduser().resolve()
    out_root = Path(output_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    out_dir = out_root / f"derived_crop_segment_{_timestamp()}"

    for split in ("train", "val", "test"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    counts = {"train": 0, "val": 0, "test": 0, "objects": 0}
    encountered_class_ids: set[int] = set()

    for split in ("train", "val", "test"):
        src_img = src / "images" / split
        src_lbl = src / "labels" / split
        if not src_img.exists():
            continue
        for img_path in sorted(src_img.rglob("*")):
            if img_path.suffix.lower() not in IMAGE_EXTS:
                continue
            lbl_path = _find_label_for_obb_image(img_path, src_img, src_lbl)
            if lbl_path is None:
                continue
            written, class_ids = _process_crop_segment_image(
                img_path,
                lbl_path,
                out_dir,
                split,
                pad_ratio,
                min_crop_size_px,
                enforce_square,
            )
            encountered_class_ids.update(class_ids)
            counts[split] += written
            counts["objects"] += written

    include_test = counts["test"] > 0
    _validate_class_name_coverage(
        resolved_class_names,
        encountered_class_ids,
        dataset_label="Derived crop segment dataset",
    )
    _write_dataset_yaml(
        out_dir,
        class_names=resolved_class_names,
        include_test=include_test,
    )
    manifest = {
        "type": "derived_crop_segment",
        "source": str(src),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "pad_ratio": float(pad_ratio),
        "min_crop_size_px": int(min_crop_size_px),
        "enforce_square": bool(enforce_square),
        "counts": counts,
    }
    manifest_path = out_dir / "manifest.json"
    _write_manifest(manifest_path, manifest)
    return DatasetBuildResult(
        str(out_dir), stats=manifest, manifest_path=str(manifest_path)
    )


_ROLE_MIN_LEVEL = {
    TrainingRole.OBB_DIRECT: GeometryLevel.OBB,
    TrainingRole.DETECT_DIRECT: GeometryLevel.AABB,
    TrainingRole.SEGMENT_DIRECT: GeometryLevel.POLYGON,
    TrainingRole.SEQ_DETECT: GeometryLevel.AABB,
    TrainingRole.SEQ_CROP_OBB: GeometryLevel.OBB,
    TrainingRole.SEQ_CROP_SEGMENT: GeometryLevel.POLYGON,
}


def role_min_level(role: TrainingRole) -> GeometryLevel:
    """Minimum geometry level a training role requires."""
    try:
        return _ROLE_MIN_LEVEL[role]
    except KeyError as exc:
        raise RuntimeError(f"Role has no geometry-level requirement: {role}") from exc


def blocked_roles_for_level(
    merged_level: GeometryLevel, roles: list[TrainingRole]
) -> dict[TrainingRole, GeometryLevel]:
    """Roles whose minimum level exceeds the merged dataset's level."""
    blocked: dict[TrainingRole, GeometryLevel] = {}
    for role in roles:
        required = role_min_level(role)
        if required > merged_level:
            blocked[role] = required
    return blocked


def prepare_role_dataset(
    role: TrainingRole,
    merged_obb_dataset_dir: str,
    role_output_root: str | Path,
    class_name: str | None = None,
    *,
    class_names: list[str] | None = None,
    crop_pad_ratio: float = 0.15,
    min_crop_size_px: int = 64,
    enforce_square: bool = True,
    merged_level: GeometryLevel = GeometryLevel.POLYGON,
) -> DatasetBuildResult:
    """Prepare role-specific dataset from the merged source."""
    required = role_min_level(role)
    if required > merged_level:
        raise RuntimeError(
            f"Role {role.value} requires {required.label}-level data but the merged "
            f"dataset is {merged_level.label}-level."
        )

    out_root = Path(role_output_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    if role == TrainingRole.OBB_DIRECT:
        manifest_path = Path(merged_obb_dataset_dir) / "manifest.json"
        return DatasetBuildResult(
            dataset_dir=str(Path(merged_obb_dataset_dir).resolve()),
            stats={"type": "passthrough_obb"},
            manifest_path=str(manifest_path) if manifest_path.exists() else "",
        )
    if role in (TrainingRole.SEQ_DETECT, TrainingRole.DETECT_DIRECT):
        return derive_detect_dataset_from_obb(
            merged_obb_dataset_dir,
            out_root,
            class_name=class_name,
            class_names=class_names,
        )
    if role == TrainingRole.SEGMENT_DIRECT:
        return derive_segment_dataset_from_source(
            merged_obb_dataset_dir,
            out_root,
            class_name=class_name,
            class_names=class_names,
        )
    if role == TrainingRole.SEQ_CROP_OBB:
        return derive_crop_obb_dataset_from_obb(
            merged_obb_dataset_dir,
            out_root,
            class_name=class_name,
            class_names=class_names,
            pad_ratio=crop_pad_ratio,
            min_crop_size_px=min_crop_size_px,
            enforce_square=enforce_square,
        )
    if role == TrainingRole.SEQ_CROP_SEGMENT:
        return derive_crop_segment_dataset_from_source(
            merged_obb_dataset_dir,
            out_root,
            class_name=class_name,
            class_names=class_names,
            pad_ratio=crop_pad_ratio,
            min_crop_size_px=min_crop_size_px,
            enforce_square=enforce_square,
        )

    raise RuntimeError(f"Unsupported training role for dataset preparation: {role}")
