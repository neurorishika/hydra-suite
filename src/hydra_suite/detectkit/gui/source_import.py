"""DetectKit source inspection and project-local standardization helpers."""

from __future__ import annotations

import json
import logging
import re
import shutil
from dataclasses import dataclass, replace
from hashlib import sha1
from pathlib import Path
from typing import Any

from hydra_suite.data.project_bundle import ensure_bundle_subdirectory
from hydra_suite.training.class_mapping import (
    normalize_declared_class_names,
    resolve_dataset_class_names,
)
from hydra_suite.training.dataset_inspector import inspect_obb_or_detect_dataset
from hydra_suite.training.geometry_levels import GeometryLevel, scan_source_levels

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
    level: str = "obb"


IMPORT_MODE_PORTABLE = "portable"
IMPORT_MODE_LINKED = "linked"


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


def _load_al_round_roots(root: Path) -> list[dict[str, Any]] | None:
    """Return the ``roots`` list from an AL round manifest.json at *root*, if any.

    Matches the container format written by ``hydra_suite.data.al.export.
    export_al_dataset``: a round directory holding ``manifest.json`` plus one
    sibling dataset root per geometry level (e.g. ``obb/``, ``aabb/``). The
    round directory itself has no ``images``/``labels``, so it never satisfies
    ``_is_detectkit_source_root`` -- callers must resolve into one of the
    listed roots instead of treating *root* as a dataset itself.
    """
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    roots = payload.get("roots")
    if not isinstance(roots, list) or not roots:
        return None
    if not all(isinstance(entry, dict) and entry.get("path") for entry in roots):
        return None
    return roots


def _resolve_al_round_entry_path(round_dir: Path, entry: dict[str, Any]) -> Path | None:
    """Resolve one manifest root entry to an existing directory, tolerating a
    moved/renamed round.

    The manifest records each root's ABSOLUTE path at export time
    (``data/al/export.py`` writes ``path = round_dir / level.label``). If the
    round directory has since been copied, moved, or renamed, that recorded
    path is stale even though the round's own internal structure -- one
    subfolder per level, named after the level -- is unchanged. Fall back to
    ``round_dir / level`` before giving up.
    """
    candidates = [Path(str(entry["path"])).expanduser()]
    level = entry.get("level")
    if level:
        candidates.append(round_dir / str(level))
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved.exists():
            return resolved
    return None


def _select_al_round_authoritative_root(
    round_dir: Path, roots: list[dict[str, Any]]
) -> Path | None:
    for entry in roots:
        if entry.get("authoritative"):
            return _resolve_al_round_entry_path(round_dir, entry)
    if roots:
        return _resolve_al_round_entry_path(round_dir, roots[0])
    return None


def resolve_al_round_authoritative_level(source_root: str | Path) -> str | None:
    """Return an AL round's manifest-declared authoritative-root level.

    An AL-export root's labels are always stored as 9-field quads regardless
    of level (see `_detect_source_level`'s `intended_level=OBB` re-scan,
    which cannot distinguish a genuine OBB from an axis-aligned-quad-encoded
    AABB by re-scanning). Only the manifest recorded which is which at
    export time -- callers that need an AL round's true level (rather than a
    re-scanned guess) must go through this function instead of
    `_detect_source_level`.

    Returns ``None`` if *source_root* is not an AL round container (no
    ``manifest.json`` with a ``roots`` list) or has no authoritative entry.
    """
    root = Path(source_root).expanduser().resolve()
    al_roots = _load_al_round_roots(root)
    if al_roots is None:
        return None
    for entry in al_roots:
        if entry.get("authoritative"):
            level = entry.get("level")
            return str(level) if level else None
    return None


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

    al_roots = _load_al_round_roots(root)
    if al_roots is not None:
        authoritative_path = _select_al_round_authoritative_root(root, al_roots)
        if authoritative_path is not None and authoritative_path != root:
            inspection = inspect_detectkit_source(authoritative_path)
            return replace(inspection, source_kind="detectkit_al")

    raise ValueError(
        "Selected source folder must be a DetectKit source, a YOLO detect/obb dataset root, "
        "a COCO annotations root, or an active-learning export round (a folder "
        "containing manifest.json) -- select that round folder's authoritative "
        "level subfolder if this error persists.\n\n"
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
    if source_path.resolve() == dest_path.resolve():
        return
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


def _coco_annotation_to_points(
    annotation: dict[str, Any], width: int, height: int
) -> tuple[list[float], str] | None:
    """Return (normalized_coords, evidence). Segmentation is preserved as a full
    contour ("polygon"); a bbox-only annotation yields an axis-aligned quad ("aabb")."""
    points = _coco_segmentation_points(annotation.get("segmentation"))
    if len(points) >= 3:
        coords: list[float] = []
        for x_pos, y_pos in points:
            coords.extend([x_pos / float(width), y_pos / float(height)])
        return coords, "polygon"
    quad = _coco_bbox_to_polygon(annotation.get("bbox"), width, height)
    if quad is not None:
        return quad, "aabb"
    return None


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
            converted = _coco_annotation_to_points(annotation, width, height)
            if converted is None:
                continue
            coords, _evidence = converted
            lines.append(_format_obb_line(dense_id, coords))

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


def _coco_source_level(source_root: Path) -> str:
    """Resolve a COCO source's geometry level from its annotation payload evidence."""
    loaded = _load_coco_dataset(source_root)
    if loaded is None:
        return GeometryLevel.OBB.label
    _json_path, payload = loaded
    has_polygon = False
    has_box = False
    for annotation in payload.get("annotations", []):
        if len(_coco_segmentation_points(annotation.get("segmentation"))) >= 3:
            has_polygon = True
        elif isinstance(annotation.get("bbox"), list) and len(annotation["bbox"]) >= 4:
            has_box = True
    if has_polygon:
        return GeometryLevel.POLYGON.label
    if has_box:
        return GeometryLevel.AABB.label
    return GeometryLevel.OBB.label


def _detect_source_level(source_root: Path, inspection) -> str:
    """Resolve a source's geometry level from its ORIGINAL input evidence,
    before label conversion collapses detect boxes into quads."""
    if inspection.source_kind == "coco":
        return _coco_source_level(source_root)
    scan = scan_source_levels(source_root / "labels", intended_level=GeometryLevel.OBB)
    return scan.resolved_level.label


def materialize_detectkit_source(
    source_root: str | Path,
    project_dir: str | Path,
    *,
    import_mode: str = IMPORT_MODE_PORTABLE,
    force_import: bool = False,
) -> MaterializedDetectKitSource:
    """Resolve *source_root* into a DetectKit-ready source for *project_dir*."""
    root = Path(source_root).expanduser().resolve()
    inspection = inspect_detectkit_source(root)
    # inspect_detectkit_source may redirect an AL round container (manifest.json
    # + sibling level roots) to its authoritative level subfolder; operate on
    # that resolved dataset root from here on, not the container directory.
    root = inspection.dataset_root
    level = _detect_source_level(root, inspection)
    if import_mode not in {IMPORT_MODE_PORTABLE, IMPORT_MODE_LINKED}:
        raise ValueError(f"Unsupported DetectKit import mode: {import_mode}")

    if import_mode == IMPORT_MODE_LINKED and not force_import:
        if inspection.requires_import:
            if inspection.source_kind == "coco":
                _materialize_coco_source(root, root)
            else:
                _materialize_yolo_source(root, root)
        return MaterializedDetectKitSource(
            source_root=root,
            canonical_path=root,
            source_kind=inspection.source_kind,
            display_name=root.name,
            images_count=inspection.images_count,
            annotation_count=inspection.annotation_count,
            discovered_labels=list(inspection.discovered_labels),
            imported=False,
            level=level,
        )

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
            level=level,
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
        level=level,
    )
