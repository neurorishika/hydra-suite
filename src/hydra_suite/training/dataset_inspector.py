"""Dataset inspection and layout discovery for MAT training."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass(slots=True)
class DatasetItem:
    """Single image/label pair in a split."""

    image_path: str
    label_path: str
    split: str


@dataclass(slots=True)
class DatasetInspection:
    """Inspection result for OBB/detect-style datasets."""

    root_dir: str
    splits: dict[str, list[DatasetItem]] = field(default_factory=dict)
    class_names: dict[int, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dep branch
        raise RuntimeError("PyYAML is required to parse dataset.yaml") from exc
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise RuntimeError(f"Invalid dataset.yaml structure: {path}")
    return data


def _resolve_data_path(root: Path, value: Any) -> Path | None:
    if value is None:
        return None
    p = Path(str(value).strip())
    if not p:
        return None
    if p.is_absolute():
        return p
    return (root / p).resolve()


def _find_label_for_image(image_path: Path, labels_root: Path) -> Path:
    # Preferred: preserve relative path from images_root into labels_root.
    rel = None
    if "images" in image_path.parts:
        idx = image_path.parts.index("images")
        rel = Path(*image_path.parts[idx + 1 :])
    if rel is not None and rel.parts:
        cand = (labels_root / rel).with_suffix(".txt")
        if cand.exists():
            return cand

    # Fallback: same basename under labels root (recursive search).
    stem = image_path.stem
    matches = list(labels_root.rglob(f"{stem}.txt"))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        # Deterministic tie-break
        return sorted(matches)[0]

    # Final fallback: sibling txt
    return image_path.with_suffix(".txt")


def _collect_dir_split(
    images_dir: Path, labels_dir: Path, split: str
) -> list[DatasetItem]:
    items: list[DatasetItem] = []
    if not images_dir.exists():
        return items
    for image_path in sorted(images_dir.rglob("*")):
        if image_path.suffix.lower() not in IMAGE_EXTS:
            continue
        label_path = _find_label_for_image(image_path, labels_dir)
        items.append(
            DatasetItem(
                image_path=str(image_path.resolve()),
                label_path=str(label_path.resolve()),
                split=split,
            )
        )
    return items


def _infer_label_path_from_image(
    root: Path, image_path: Path, labels_root: Path | None = None
) -> Path:
    labels_root = (labels_root or (root / "labels")).resolve()
    image_posix = image_path.as_posix()
    if "/images/" in image_posix:
        image_parts = image_path.parts
        idx = image_parts.index("images")
        rel = Path(*image_parts[idx + 1 :])
        return (labels_root / rel).with_suffix(".txt")
    # Fallback to sibling labels folder
    return (labels_root / image_path.name).with_suffix(".txt")


def _collect_list_split(
    root: Path, list_path: Path, split: str, labels_root: Path | None = None
) -> list[DatasetItem]:
    items: list[DatasetItem] = []
    if not list_path.exists():
        return items
    lines = [ln.strip() for ln in list_path.read_text(encoding="utf-8").splitlines()]
    for ln in lines:
        if not ln:
            continue
        p = Path(ln)
        if not p.is_absolute():
            p = (root / p).resolve()
        if p.suffix.lower() not in IMAGE_EXTS:
            continue
        lbl = _infer_label_path_from_image(root, p, labels_root=labels_root)
        if not lbl.is_absolute():
            lbl = (root / lbl).resolve()
        items.append(DatasetItem(image_path=str(p), label_path=str(lbl), split=split))
    return items


def _extract_class_names(data: dict[str, Any]) -> dict[int, str]:
    names = data.get("names", {})
    out: dict[int, str] = {}
    if isinstance(names, dict):
        for k, v in names.items():
            try:
                out[int(k)] = str(v)
            except Exception:
                continue
    elif isinstance(names, list):
        for i, v in enumerate(names):
            out[i] = str(v)
    return out


def _resolve_yaml_labels_dir(
    root: Path, data: dict[str, Any], split_path: Path
) -> Path:
    """Resolve the labels directory for one YAML-defined split."""
    labels_dir = _resolve_data_path(root, data.get("labels"))
    if labels_dir is None:
        labels_dir = root / "labels"
    if split_path.is_dir() and split_path.name in {"train", "val", "test"}:
        split_labels = labels_dir / split_path.name
        if split_labels.exists():
            return split_labels
    return labels_dir


def _collect_yaml_split_items(
    root: Path,
    data: dict[str, Any],
    split: str,
) -> list[DatasetItem]:
    """Collect items for one split declared in dataset.yaml."""
    split_ref = data.get(split)
    if split_ref is None:
        return []

    split_path = _resolve_data_path(root, split_ref)
    if split_path is None:
        return []

    labels_dir = _resolve_yaml_labels_dir(root, data, split_path)
    if split_path.suffix.lower() == ".txt":
        return _collect_list_split(
            root, split_path, split=split, labels_root=labels_dir
        )
    return _collect_dir_split(split_path, labels_dir, split=split)


def _inspect_from_yaml(
    root: Path, yaml_path: Path, inspection: DatasetInspection
) -> bool:
    """Try to populate inspection from dataset.yaml; return True if splits found."""
    if not yaml_path.exists():
        return False
    data = _read_yaml(yaml_path)
    inspection.class_names = _extract_class_names(data)

    for split in ("train", "val", "test"):
        items = _collect_yaml_split_items(root, data, split)
        if items:
            inspection.splits[split] = items

    if inspection.splits:
        inspection.metadata["source"] = "dataset.yaml"
        return True
    return False


def _inspect_from_directory_layout(root: Path, inspection: DatasetInspection) -> bool:
    """Try to populate inspection from standard images/labels directory layout."""
    images_root = root / "images"
    labels_root = root / "labels"
    if not (images_root.exists() and labels_root.exists()):
        return False

    split_items: dict[str, list[DatasetItem]] = {}
    split_found = False
    for split in ("train", "val", "test"):
        img_dir = images_root / split
        lbl_dir = labels_root / split if (labels_root / split).exists() else labels_root
        if img_dir.exists():
            split_items[split] = _collect_dir_split(img_dir, lbl_dir, split)
            split_found = True
    if split_found:
        inspection.splits = split_items
        inspection.metadata["source"] = "images/labels split"
        return True

    # Unsplit dataset (images + labels roots)
    inspection.splits = {
        "all": _collect_dir_split(images_root, labels_root, split="all"),
    }
    inspection.metadata["source"] = "images/labels unsplit"
    return True


def inspect_obb_or_detect_dataset(root_dir: str | Path) -> DatasetInspection:
    """Inspect a YOLO OBB/detect dataset and return resolved split items."""

    root = Path(root_dir).expanduser().resolve()
    if not root.exists():
        raise RuntimeError(f"Dataset root not found: {root}")

    inspection = DatasetInspection(root_dir=str(root))

    if _inspect_from_yaml(root, root / "dataset.yaml", inspection):
        return inspection
    if _inspect_from_directory_layout(root, inspection):
        return inspection

    raise RuntimeError(f"No valid OBB/detect dataset layout found in {root}")


@dataclass(slots=True)
class OBBSizeStats:
    """Object and crop size statistics for an OBB dataset."""

    n_objects: int = 0
    n_images: int = 0
    # Object bounding-box sizes in pixels (axis-aligned envelope of the OBB).
    obj_widths: list[float] = field(default_factory=list)
    obj_heights: list[float] = field(default_factory=list)
    # Crop sizes that would result from the given pad/min/square settings.
    crop_sizes: list[float] = field(default_factory=list)
    # Image dimensions encountered.
    img_widths: list[int] = field(default_factory=list)
    img_heights: list[int] = field(default_factory=list)
    # Per-object longest image dimension for full-image resize analysis.
    obj_image_longest_dims: list[float] = field(default_factory=list)


def _parse_obb_object_from_line(ln: str, w: int, h: int):
    """Parse one OBB label line and return (bw, bh) in pixels, or None."""
    import numpy as np

    ln = ln.strip()
    if not ln:
        return None
    parts = ln.split()
    if len(parts) != 9:
        return None
    try:
        coords = np.asarray([float(v) for v in parts[1:]], dtype=np.float32).reshape(
            4, 2
        )
    except Exception:
        return None
    px = coords[:, 0] * float(w)
    py = coords[:, 1] * float(h)
    bw = max(1.0, float(np.max(px)) - float(np.min(px)))
    bh = max(1.0, float(np.max(py)) - float(np.min(py)))
    return bw, bh


def _compute_crop_size(
    bw: float, bh: float, pad_ratio: float, min_crop_size_px: int, enforce_square: bool
) -> float:
    """Compute the crop size for an object with the given dimensions."""
    crop_w = max(float(min_crop_size_px), bw * (1.0 + 2.0 * max(0.0, pad_ratio)))
    crop_h = max(float(min_crop_size_px), bh * (1.0 + 2.0 * max(0.0, pad_ratio)))
    if enforce_square:
        crop_w = crop_h = max(crop_w, crop_h)
    return max(crop_w, crop_h)


def _analyze_obb_item(
    item: DatasetItem,
    stats: OBBSizeStats,
    pad_ratio: float,
    min_crop_size_px: int,
    enforce_square: bool,
) -> None:
    """Accumulate size statistics from one dataset item."""
    lbl_path = Path(item.label_path)
    img_path = Path(item.image_path)
    if not lbl_path.exists() or not img_path.exists():
        return

    import cv2

    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None or img.size == 0:
        return
    h, w = img.shape[:2]
    image_longest_dim = float(max(w, h))
    stats.n_images += 1
    stats.img_widths.append(w)
    stats.img_heights.append(h)

    try:
        lines = lbl_path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return

    for ln in lines:
        result = _parse_obb_object_from_line(ln, w, h)
        if result is None:
            continue
        bw, bh = result
        stats.obj_widths.append(bw)
        stats.obj_heights.append(bh)
        stats.obj_image_longest_dims.append(image_longest_dim)
        stats.n_objects += 1
        stats.crop_sizes.append(
            _compute_crop_size(bw, bh, pad_ratio, min_crop_size_px, enforce_square)
        )


def analyze_obb_sizes(
    inspection: DatasetInspection,
    pad_ratio: float = 0.15,
    min_crop_size_px: int = 64,
    enforce_square: bool = True,
    max_images: int = 500,
) -> OBBSizeStats:
    """Compute object and derived crop size statistics from an OBB dataset.

    Samples up to *max_images* items (deterministic) to keep the analysis fast.
    """
    import random

    stats = OBBSizeStats()
    all_items: list[DatasetItem] = []
    for split_items in inspection.splits.values():
        all_items.extend(split_items)
    if not all_items:
        return stats

    rng = random.Random(0)
    if len(all_items) > max_images:
        all_items = rng.sample(all_items, max_images)

    for item in all_items:
        _analyze_obb_item(
            item,
            stats,
            pad_ratio,
            min_crop_size_px,
            enforce_square,
        )

    return stats


def format_size_analysis(
    stats: OBBSizeStats,
    training_imgsz: int = 160,
    pipeline_mode: Literal["crop", "full_image"] = "crop",
) -> tuple[str, list[str]]:
    """Format a human-readable analysis and return (report_text, warnings).

    *warnings* contains actionable suggestions when settings look problematic.
    *pipeline_mode* controls whether imgsz is compared to derived crops or to
    full-image resize behavior.
    """
    import numpy as np

    lines: list[str] = []
    warnings: list[str] = []

    if stats.n_objects == 0:
        return "No objects found in dataset for analysis.", warnings

    obj_w = np.asarray(stats.obj_widths)
    obj_h = np.asarray(stats.obj_heights)
    crops = np.asarray(stats.crop_sizes)
    img_w = np.asarray(stats.img_widths) if stats.img_widths else np.array([0])
    img_h = np.asarray(stats.img_heights) if stats.img_heights else np.array([0])

    lines.append(f"Dataset: {stats.n_images} images, {stats.n_objects} objects")
    lines.append("")

    lines.append("Image dimensions:")
    lines.append(
        f"  width : min={int(np.min(img_w))}, median={int(np.median(img_w))}, "
        f"max={int(np.max(img_w))}"
    )
    lines.append(
        f"  height: min={int(np.min(img_h))}, median={int(np.median(img_h))}, "
        f"max={int(np.max(img_h))}"
    )
    lines.append("")

    lines.append("Object sizes (px, axis-aligned envelope of OBB):")
    lines.append(
        f"  width : min={obj_w.min():.0f}, median={np.median(obj_w):.0f}, "
        f"max={obj_w.max():.0f}"
    )
    lines.append(
        f"  height: min={obj_h.min():.0f}, median={np.median(obj_h):.0f}, "
        f"max={obj_h.max():.0f}"
    )
    lines.append("")

    if pipeline_mode == "crop":
        lines.append("Crop sizes after padding (px, largest dimension):")
        lines.append(
            f"  min={crops.min():.0f}, median={np.median(crops):.0f}, "
            f"max={crops.max():.0f}"
        )
        lines.append("")

        if training_imgsz > 0:
            upscaled = float(np.sum(crops < training_imgsz)) / len(crops) * 100.0
            downscaled = float(np.sum(crops > training_imgsz)) / len(crops) * 100.0
            matched = 100.0 - upscaled - downscaled
            lines.append(f"Relative to training imgsz={training_imgsz}:")
            lines.append(
                f"  {upscaled:.0f}% of crops will be upscaled (smaller than imgsz)"
            )
            lines.append(
                f"  {downscaled:.0f}% of crops will be downscaled (larger than imgsz)"
            )
            lines.append(f"  {matched:.0f}% are approximately the right size")
            lines.append("")

            median_crop = float(np.median(crops))
            scale_ratio = training_imgsz / max(1.0, median_crop)

            if upscaled > 80:
                warnings.append(
                    f"WARNING: {upscaled:.0f}% of crops are smaller than imgsz={training_imgsz} "
                    f"and will be heavily upscaled (median crop={median_crop:.0f}px). "
                    f"Consider reducing imgsz to ~{int(median_crop)} or increasing pad ratio."
                )
            if downscaled > 80:
                warnings.append(
                    f"WARNING: {downscaled:.0f}% of crops are larger than imgsz={training_imgsz} "
                    f"and will lose detail when downscaled (median crop={median_crop:.0f}px). "
                    f"Consider increasing imgsz to ~{int(median_crop)}."
                )
            if scale_ratio > 3.0:
                warnings.append(
                    f"WARNING: Median crop ({median_crop:.0f}px) is {scale_ratio:.1f}x smaller "
                    f"than imgsz={training_imgsz}. This extreme upscaling introduces blur "
                    f"artifacts. Strongly consider reducing imgsz."
                )
            if scale_ratio < 0.3:
                warnings.append(
                    f"WARNING: Median crop ({median_crop:.0f}px) is {1.0 / scale_ratio:.1f}x larger "
                    f"than imgsz={training_imgsz}. Significant detail loss from downscaling. "
                    f"Consider increasing imgsz."
                )
    elif pipeline_mode == "full_image":
        if training_imgsz > 0:
            obj_long = np.maximum(obj_w, obj_h)
            if stats.obj_image_longest_dims:
                image_longest = np.asarray(stats.obj_image_longest_dims, dtype=float)
            else:
                image_longest = np.full(
                    len(obj_long),
                    max(1.0, float(np.median(np.maximum(img_w, img_h)))),
                )
            resized_obj = (
                obj_long * float(training_imgsz) / np.maximum(1.0, image_longest)
            )
            very_small = float(np.sum(resized_obj < 16.0)) / len(resized_obj) * 100.0
            small = float(np.sum(resized_obj < 24.0)) / len(resized_obj) * 100.0
            median_resized = float(np.median(resized_obj))

            lines.append(f"At full-image training imgsz={training_imgsz}:")
            lines.append(
                "  object largest dimension after resize: "
                f"min={resized_obj.min():.0f}px, median={median_resized:.0f}px, "
                f"max={resized_obj.max():.0f}px"
            )
            lines.append(
                f"  {very_small:.0f}% of objects will be under 16px after resize"
            )
            lines.append(f"  {small:.0f}% of objects will be under 24px after resize")
            lines.append("")

            if very_small > 80:
                warnings.append(
                    f"WARNING: {very_small:.0f}% of objects will be under 16px at imgsz={training_imgsz} "
                    "after full-image resize. Direct OBB will likely miss fine details; "
                    "consider increasing imgsz substantially or using sequential detection."
                )
            if median_resized < 16.0:
                warnings.append(
                    f"WARNING: Median object shrinks to {median_resized:.0f}px at imgsz={training_imgsz} "
                    "after full-image resize. This is too small for reliable direct OBB learning; "
                    "prefer sequential detection or a much larger imgsz."
                )
    else:
        raise ValueError(f"Unsupported pipeline_mode: {pipeline_mode}")

    # Object-to-image ratio.
    median_obj = float(np.median(np.maximum(obj_w, obj_h)))
    median_img = float(np.median(np.maximum(img_w, img_h)))
    if median_img > 0:
        obj_frac = median_obj / median_img
        lines.append(
            f"Object-to-image ratio: median object is {obj_frac:.1%} of image size"
        )
        if pipeline_mode == "full_image" and obj_frac < 0.02:
            warnings.append(
                "WARNING: Objects are very small relative to images (<2%). "
                "Sequential detection mode is strongly recommended over direct OBB."
            )

    return "\n".join(lines), warnings


def split_items_for_training(
    inspection: DatasetInspection, split_cfg: tuple[float, float, float], seed: int
) -> dict[str, list[DatasetItem]]:
    """Normalize to train/val/test using provided ratios when source is unsplit."""

    import random

    if "all" not in inspection.splits:
        out = {
            "train": list(inspection.splits.get("train", [])),
            "val": list(inspection.splits.get("val", [])),
            "test": list(inspection.splits.get("test", [])),
        }
        return out

    items = list(inspection.splits.get("all", []))
    rng = random.Random(int(seed))
    rng.shuffle(items)

    train_r, val_r, test_r = split_cfg
    total = max(1e-8, float(train_r) + float(val_r) + float(test_r))
    train_r, val_r, test_r = train_r / total, val_r / total, test_r / total

    n = len(items)
    n_train = int(round(n * train_r))
    n_val = int(round(n * val_r))

    # Guardrails for non-empty train/val when feasible
    if n >= 2:
        n_train = max(1, min(n - 1, n_train))
        n_val = max(1, min(n - n_train, n_val))

    train = items[:n_train]
    val = items[n_train : n_train + n_val]
    test = items[n_train + n_val :]

    for it in train:
        it.split = "train"
    for it in val:
        it.split = "val"
    for it in test:
        it.split = "test"

    return {"train": train, "val": val, "test": test}


def _read_class_ids_from_label(label_path: str) -> set[int]:
    """Read class IDs from an OBB/detect label file.

    Each line: ``class_id x1 y1 x2 y2 x3 y3 x4 y4``.
    Returns the set of integer class IDs found.
    """
    try:
        text = Path(label_path).read_text(encoding="utf-8").strip()
        if not text:
            return set()
        ids: set[int] = set()
        for line in text.splitlines():
            parts = line.split()
            if parts:
                ids.add(int(float(parts[0])))
        return ids
    except Exception:
        return set()


def _split_by_ratio(
    items: list[DatasetItem],
    train_r: float,
    val_r: float,
) -> tuple[list[DatasetItem], list[DatasetItem], list[DatasetItem]]:
    """Split items list into (train, val, test) by ratio with guardrails."""
    n = len(items)
    n_train = int(round(n * train_r))
    n_val = int(round(n * val_r))
    if n >= 2:
        n_train = max(1, min(n - 1, n_train))
        n_val = max(1, min(n - n_train, n_val))
    return items[:n_train], items[n_train : n_train + n_val], items[n_train + n_val :]


def _label_splits(
    train: list[DatasetItem],
    val: list[DatasetItem],
    test: list[DatasetItem],
) -> dict[str, list[DatasetItem]]:
    """Assign split labels and return the standard split dict."""
    for it in train:
        it.split = "train"
    for it in val:
        it.split = "val"
    for it in test:
        it.split = "test"
    return {"train": train, "val": val, "test": test}


def stratified_split_items(
    items: list[DatasetItem],
    split_cfg: tuple[float, float, float],
    seed: int,
) -> dict[str, list[DatasetItem]]:
    """Split items with stratified class balance.

    Groups items by their dominant (most frequent) class ID, then splits each
    group proportionally according to *split_cfg* ``(train, val, test)``.
    Falls back to simple random shuffle when labels are unreadable or all items
    share one class.
    """
    import random
    from collections import Counter, defaultdict

    rng = random.Random(int(seed))

    train_r, val_r, test_r = split_cfg
    total = max(1e-8, float(train_r) + float(val_r) + float(test_r))
    train_r, val_r, test_r = train_r / total, val_r / total, test_r / total

    # Determine dominant class per item
    buckets: dict[int, list[DatasetItem]] = defaultdict(list)
    fallback_items: list[DatasetItem] = []

    for item in items:
        cls_ids = _read_class_ids_from_label(item.label_path)
        if not cls_ids:
            fallback_items.append(item)
        else:
            counter = Counter(cls_ids)
            dominant = min(counter, key=lambda c: (-counter[c], c))
            buckets[dominant].append(item)

    # If only one class (or no readable labels), fall back to simple shuffle
    if len(buckets) <= 1:
        all_items = list(items)
        rng.shuffle(all_items)
        train, val, test = _split_by_ratio(all_items, train_r, val_r)
        return _label_splits(train, val, test)

    # Put fallback items into an artificial bucket
    if fallback_items:
        buckets[-1] = fallback_items

    train_out: list[DatasetItem] = []
    val_out: list[DatasetItem] = []
    test_out: list[DatasetItem] = []

    for _cls_id, bucket in sorted(buckets.items()):
        rng.shuffle(bucket)
        tr, va, te = _split_by_ratio(bucket, train_r, val_r)
        train_out.extend(tr)
        val_out.extend(va)
        test_out.extend(te)

    # Shuffle within each split to avoid class clustering
    rng.shuffle(train_out)
    rng.shuffle(val_out)
    rng.shuffle(test_out)

    return _label_splits(train_out, val_out, test_out)
