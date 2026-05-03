"""DetectKit source inspection and project-local standardization helpers."""

from __future__ import annotations

import json
import logging
import re
import shutil
from dataclasses import dataclass
from hashlib import sha1
from pathlib import Path
from typing import Any

from hydra_suite.data.project_bundle import ensure_bundle_subdirectory
from hydra_suite.training.class_mapping import (
    normalize_declared_class_names,
    resolve_dataset_class_names,
)
from hydra_suite.training.dataset_inspector import inspect_obb_or_detect_dataset

logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class DetectKitSourceInspection:
    """Summary of a source that DetectKit can consume."""

    dataset_root: Path
    source_kind: str
    images_count: int
    annotation_count: int
    discovered_labels: list[str]
    requires_import: bool


@dataclass(slots=True, frozen=True)
class MaterializedDetectKitSource:
    """Result of resolving a selected source into DetectKit's canonical layout."""

    source_root: Path
    canonical_path: Path
    source_kind: str
    display_name: str
    images_count: int
    annotation_count: int
    discovered_labels: list[str]
    imported: bool


def _slugify_name(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-._")
    return slug or "source"


def _is_detectkit_source_root(root: Path) -> bool:
    return (
        (root / "images").is_dir()
        and (root / "labels").is_dir()
        and (root / "classes.txt").is_file()
    )


def _count_nonempty_lines(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        return sum(
            1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
        )
    except Exception:
        return 0


def _flatten_inspection_items(inspection) -> list:
    items = []
    for split_items in inspection.splits.values():
        items.extend(split_items)
    return items


def _infer_yolo_source_kind(inspection) -> str:
    for item in _flatten_inspection_items(inspection):
        label_path = Path(item.label_path)
        if not label_path.exists():
            continue
        for raw_line in label_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) == 5:
                return "yolo_detect"
            if len(parts) == 9:
                return "yolo_obb"
    return "yolo_obb"


def _inspect_yolo_like_source(root: Path) -> DetectKitSourceInspection:
    inspection = inspect_obb_or_detect_dataset(root)
    items = _flatten_inspection_items(inspection)
    class_names = resolve_dataset_class_names(root, inspection.class_names)
    annotation_count = sum(
        _count_nonempty_lines(Path(item.label_path)) for item in items
    )
    source_kind = (
        "detectkit"
        if _is_detectkit_source_root(root)
        else _infer_yolo_source_kind(inspection)
    )
    return DetectKitSourceInspection(
        dataset_root=root,
        source_kind=source_kind,
        images_count=len(items),
        annotation_count=annotation_count,
        discovered_labels=list(class_names),
        requires_import=not _is_detectkit_source_root(root),
    )


def _is_coco_payload(payload: Any) -> bool:
    return isinstance(payload, dict) and all(
        isinstance(payload.get(key), list)
        for key in ("images", "annotations", "categories")
    )


def _iter_coco_json_candidates(root: Path) -> list[Path]:
    candidates: list[Path] = []
    preferred_names = (
        "annotations.json",
        "instances.json",
        "instances_train.json",
        "instances_val.json",
    )
    for name in preferred_names:
        path = root / name
        if path.is_file():
            candidates.append(path)

    annotations_dir = root / "annotations"
    if annotations_dir.is_dir():
        candidates.extend(sorted(annotations_dir.glob("*.json")))

    candidates.extend(sorted(root.glob("*.coco.json")))
    candidates.extend(sorted(root.glob("*.json")))

    unique: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(resolved)
    return unique


def _load_coco_dataset(root: Path) -> tuple[Path, dict[str, Any]] | None:
    for candidate in _iter_coco_json_candidates(root):
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except Exception:
            continue
        if _is_coco_payload(payload):
            return candidate, payload
    return None


def _resolve_coco_image_path(root: Path, file_name: str) -> Path:
    raw_path = Path(str(file_name))
    candidates = [root / raw_path, root / "images" / raw_path]
    if raw_path.name != str(raw_path):
        candidates.extend([root / raw_path.name, root / "images" / raw_path.name])
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise RuntimeError(f"COCO image not found for entry: {file_name}")


def _inspect_coco_source(root: Path) -> DetectKitSourceInspection | None:
    loaded = _load_coco_dataset(root)
    if loaded is None:
        return None

    _json_path, payload = loaded
    sorted_categories = sorted(
        (
            (int(entry.get("id")), str(entry.get("name")))
            for entry in payload.get("categories", [])
            if entry.get("id") is not None and entry.get("name") is not None
        ),
        key=lambda item: item[0],
    )
    declared_labels = normalize_declared_class_names(
        [name for _category_id, name in sorted_categories],
        source_label=f"COCO categories for {root}",
    )
    return DetectKitSourceInspection(
        dataset_root=root,
        source_kind="coco",
        images_count=len(payload.get("images", [])),
        annotation_count=len(payload.get("annotations", [])),
        discovered_labels=declared_labels,
        requires_import=True,
    )


def inspect_detectkit_source(source_root: str | Path) -> DetectKitSourceInspection:
    """Inspect a selected source and describe how DetectKit should handle it."""
    root = Path(source_root).expanduser().resolve()
    if not root.exists():
        raise ValueError(f"Source root not found: {root}")

    try:
        return _inspect_yolo_like_source(root)
    except Exception:
        pass

    coco_inspection = _inspect_coco_source(root)
    if coco_inspection is not None:
        return coco_inspection

    raise ValueError(
        "Selected source folder must be a DetectKit source, a YOLO detect/obb dataset root, "
        "or a COCO annotations root.\n\n"
        f"{root}"
    )


def _standardized_source_dir(source_root: Path, project_dir: Path) -> Path:
    project_root = project_dir.expanduser().resolve()
    imported_root = ensure_bundle_subdirectory(
        project_root, "artifacts/imported_sources"
    )
    source_hash = sha1(str(source_root.resolve()).encode("utf-8")).hexdigest()[:10]
    return imported_root / f"{_slugify_name(source_root.name)}-{source_hash}"


def _relative_target_path(source_root: Path, image_path: Path) -> Path:
    candidates = [source_root / "images", source_root]
    for anchor in candidates:
        try:
            rel = image_path.relative_to(anchor)
            if rel.parts:
                return rel
        except ValueError:
            continue
    digest = sha1(str(image_path.resolve()).encode("utf-8")).hexdigest()[:12]
    return Path(f"{digest}_{image_path.name}")


def _clamp_normalized(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _format_obb_line(class_id: int, coords: list[float]) -> str:
    formatted = " ".join(f"{_clamp_normalized(value):.6f}" for value in coords)
    return f"{int(class_id)} {formatted}"


def _convert_yolo_label_text(label_path: Path) -> str:
    if not label_path.exists():
        return ""

    lines: list[str] = []
    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        try:
            class_id = int(float(parts[0]))
        except Exception as exc:
            raise RuntimeError(
                f"Invalid YOLO annotation line in {label_path}: {raw_line}"
            ) from exc

        if len(parts) == 5:
            cx, cy, width, height = (float(value) for value in parts[1:5])
            x1 = cx - (width * 0.5)
            y1 = cy - (height * 0.5)
            x2 = cx + (width * 0.5)
            y2 = cy + (height * 0.5)
            coords = [x1, y1, x2, y1, x2, y2, x1, y2]
        elif len(parts) == 9:
            coords = [float(value) for value in parts[1:9]]
        else:
            raise RuntimeError(
                "Unsupported YOLO annotation format in "
                f"{label_path}: expected 5 or 9 fields, got {len(parts)}"
            )
        lines.append(_format_obb_line(class_id, coords))
    return "\n".join(lines) + ("\n" if lines else "")


def _copy_file(source_path: Path, dest_path: Path) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, dest_path)


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_classes_txt(dest_root: Path, class_names: list[str]) -> None:
    _write_text(dest_root / "classes.txt", "\n".join(class_names) + "\n")


def _materialize_yolo_source(source_root: Path, dest_root: Path) -> list[str]:
    inspection = inspect_obb_or_detect_dataset(source_root)
    class_names = resolve_dataset_class_names(source_root, inspection.class_names)
    _write_classes_txt(dest_root, class_names)

    for item in _flatten_inspection_items(inspection):
        image_path = Path(item.image_path).resolve()
        label_path = Path(item.label_path).resolve()
        relative_path = _relative_target_path(source_root, image_path)
        _copy_file(image_path, dest_root / "images" / relative_path)
        _write_text(
            dest_root / "labels" / relative_path.with_suffix(".txt"),
            _convert_yolo_label_text(label_path),
        )

    return class_names


def _coerce_coco_image_size(
    image_entry: dict[str, Any], image_path: Path
) -> tuple[int, int]:
    width = int(image_entry.get("width") or 0)
    height = int(image_entry.get("height") or 0)
    if width > 0 and height > 0:
        return width, height

    import cv2

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise RuntimeError(f"Could not determine image size for {image_path}")
    return int(image.shape[1]), int(image.shape[0])


def _points_to_min_area_rect(
    points: list[tuple[float, float]], width: int, height: int
) -> list[float] | None:
    if len(points) < 3:
        return None

    import cv2
    import numpy as np

    rect = cv2.minAreaRect(np.asarray(points, dtype=np.float32))
    box = cv2.boxPoints(rect).astype(float)
    coords: list[float] = []
    for x_pos, y_pos in box:
        coords.extend([x_pos / float(width), y_pos / float(height)])
    return coords


def _coco_segmentation_points(segmentation: Any) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    if not isinstance(segmentation, list):
        return points
    for segment in segmentation:
        if not isinstance(segment, list) or len(segment) < 6:
            continue
        if len(segment) % 2 != 0:
            continue
        for index in range(0, len(segment), 2):
            points.append((float(segment[index]), float(segment[index + 1])))
    return points


def _coco_bbox_to_polygon(bbox: Any, width: int, height: int) -> list[float] | None:
    if not isinstance(bbox, list) or len(bbox) < 4:
        return None
    x_pos, y_pos, box_width, box_height = (float(value) for value in bbox[:4])
    return [
        x_pos / float(width),
        y_pos / float(height),
        (x_pos + box_width) / float(width),
        y_pos / float(height),
        (x_pos + box_width) / float(width),
        (y_pos + box_height) / float(height),
        x_pos / float(width),
        (y_pos + box_height) / float(height),
    ]


def _coco_annotation_to_obb(
    annotation: dict[str, Any], width: int, height: int
) -> list[float] | None:
    points = _coco_segmentation_points(annotation.get("segmentation"))
    if points:
        polygon = _points_to_min_area_rect(points, width, height)
        if polygon is not None:
            return polygon
    return _coco_bbox_to_polygon(annotation.get("bbox"), width, height)


def _materialize_coco_source(source_root: Path, dest_root: Path) -> list[str]:
    loaded = _load_coco_dataset(source_root)
    if loaded is None:
        raise RuntimeError(f"No COCO annotations found in {source_root}")

    _json_path, payload = loaded
    sorted_categories = sorted(
        (
            (int(entry.get("id")), str(entry.get("name")))
            for entry in payload.get("categories", [])
            if entry.get("id") is not None and entry.get("name") is not None
        ),
        key=lambda item: item[0],
    )
    class_names = normalize_declared_class_names(
        [name for _category_id, name in sorted_categories],
        source_label=f"COCO categories for {source_root}",
    )
    category_to_dense = {
        category_id: dense_id
        for dense_id, (category_id, _name) in enumerate(sorted_categories)
    }
    _write_classes_txt(dest_root, class_names)

    annotations_by_image: dict[int, list[dict[str, Any]]] = {}
    for annotation in payload.get("annotations", []):
        image_id = annotation.get("image_id")
        if image_id is None:
            continue
        annotations_by_image.setdefault(int(image_id), []).append(annotation)

    for image_entry in payload.get("images", []):
        image_id = image_entry.get("id")
        file_name = image_entry.get("file_name")
        if image_id is None or not file_name:
            continue
        image_path = _resolve_coco_image_path(source_root, str(file_name))
        width, height = _coerce_coco_image_size(image_entry, image_path)
        relative_path = _relative_target_path(source_root, image_path)
        _copy_file(image_path, dest_root / "images" / relative_path)

        lines: list[str] = []
        for annotation in annotations_by_image.get(int(image_id), []):
            category_id = annotation.get("category_id")
            if category_id is None:
                continue
            dense_id = category_to_dense.get(int(category_id))
            if dense_id is None:
                continue
            polygon = _coco_annotation_to_obb(annotation, width, height)
            if polygon is None:
                continue
            lines.append(_format_obb_line(dense_id, polygon))

        _write_text(
            dest_root / "labels" / relative_path.with_suffix(".txt"),
            "\n".join(lines) + ("\n" if lines else ""),
        )

    return class_names


def compute_positional_class_remap(
    source_classes: list[str],
    project_classes: list[str],
) -> dict[int, int]:
    """Map source class ids onto project class ids by list position.

    Rules:
    - If both lists are empty, returns an empty mapping.
    - If the project has a single class, every source class maps to 0.
    - If the source has a single class, that source class maps to 0 (the
      first project class).
    - Otherwise, source class *i* maps to project class *i* if *i* is within
      bounds. Source classes beyond the project list are dropped.
    """
    if not source_classes or not project_classes:
        return {}
    if len(project_classes) == 1:
        return {i: 0 for i in range(len(source_classes))}
    if len(source_classes) == 1:
        return {0: 0}
    return {i: i for i in range(min(len(source_classes), len(project_classes)))}


def remap_materialized_source_classes(
    canonical_path: Path,
    project_classes: list[str],
    remap: dict[int, int],
) -> None:
    """Rewrite *canonical_path*/classes.txt and labels to use project class ids."""
    canonical_root = Path(canonical_path)
    _write_classes_txt(canonical_root, list(project_classes))

    labels_dir = canonical_root / "labels"
    if not labels_dir.is_dir():
        return

    for label_file in labels_dir.rglob("*.txt"):
        new_lines: list[str] = []
        for raw_line in label_file.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            try:
                source_class_id = int(float(parts[0]))
            except (ValueError, IndexError):
                continue
            mapped = remap.get(source_class_id)
            if mapped is None:
                continue
            parts[0] = str(int(mapped))
            new_lines.append(" ".join(parts))
        label_file.write_text(
            "\n".join(new_lines) + ("\n" if new_lines else ""),
            encoding="utf-8",
        )


def materialize_detectkit_source(
    source_root: str | Path,
    project_dir: str | Path,
    *,
    force_import: bool = False,
) -> MaterializedDetectKitSource:
    """Resolve *source_root* into a DetectKit-ready source for *project_dir*."""
    root = Path(source_root).expanduser().resolve()
    inspection = inspect_detectkit_source(root)
    if not inspection.requires_import and not force_import:
        return MaterializedDetectKitSource(
            source_root=root,
            canonical_path=root,
            source_kind=inspection.source_kind,
            display_name=root.name,
            images_count=inspection.images_count,
            annotation_count=inspection.annotation_count,
            discovered_labels=list(inspection.discovered_labels),
            imported=False,
        )

    dest_root = _standardized_source_dir(root, Path(project_dir))
    if dest_root.exists():
        shutil.rmtree(dest_root)
    dest_root.mkdir(parents=True, exist_ok=True)

    if inspection.source_kind == "coco":
        _materialize_coco_source(root, dest_root)
    else:
        _materialize_yolo_source(root, dest_root)

    return MaterializedDetectKitSource(
        source_root=root,
        canonical_path=dest_root,
        source_kind=inspection.source_kind,
        display_name=root.name,
        images_count=inspection.images_count,
        annotation_count=inspection.annotation_count,
        discovered_labels=list(inspection.discovered_labels),
        imported=True,
    )
