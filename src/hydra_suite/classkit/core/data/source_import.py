"""Source import helpers for ClassKit image and annotation-backed datasets."""

from __future__ import annotations

import json
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass
from hashlib import sha1
from pathlib import Path
from typing import Any

from hydra_suite.data.project_bundle import (
    bundle_root_for_path,
    ensure_bundle_subdirectory,
)
from hydra_suite.training.dataset_inspector import inspect_obb_or_detect_dataset

from .ingest import scan_images

_SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
_LABEL_METADATA_KEYS = {
    "annotation_count",
    "annotation_path",
    "category_ids",
    "category_names",
    "class_ids",
    "class_name",
    "class_folder",
    "label_path",
}


@dataclass(slots=True)
class ExternalSourceInspection:
    """Summary of an annotation-backed dataset that ClassKit can import."""

    source_kind: str
    images_count: int
    annotation_count: int
    discovered_labels: list[str]


@dataclass(slots=True)
class SourceImportPlan:
    """Normalized source import plan for one selected source root."""

    source_root: Path
    source_kind: str
    image_paths: list[Path]
    label_updates: dict[str, tuple[str, float]]
    metadata_by_path: dict[str, dict[str, Any]]
    discovered_labels: list[str]


def _iter_supported_images(folder: Path) -> list[Path]:
    return [
        path.resolve()
        for path in sorted(folder.iterdir())
        if path.is_file() and path.suffix.lower() in _SUPPORTED_IMAGE_EXTS
    ]


def _slugify_name(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-._")
    return slug or "source"


def _strip_label_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in metadata.items() if key not in _LABEL_METADATA_KEYS
    }


def _materialize_metadata(
    metadata: dict[str, Any],
    *,
    original_image_path: Path,
    standardized_source_dir: Path,
    include_label_metadata: bool,
) -> dict[str, Any]:
    payload = dict(metadata)
    if not include_label_metadata:
        payload = _strip_label_metadata(payload)
    payload["original_image_path"] = str(original_image_path.resolve())
    payload["standardized_source_dir"] = str(standardized_source_dir.resolve())
    return payload


def _standardized_source_dir(source_root: Path, project_anchor: Path) -> Path:
    project_root = bundle_root_for_path(project_anchor)
    ensure_bundle_subdirectory(project_root, "artifacts/imported_sources")
    source_hash = sha1(str(source_root.resolve()).encode("utf-8")).hexdigest()[:10]
    folder_name = f"{_slugify_name(source_root.name)}-{source_hash}"
    return ensure_bundle_subdirectory(
        project_root, f"artifacts/imported_sources/{folder_name}"
    )


def materialize_source_import_plan(
    plan: SourceImportPlan,
    project_anchor: Path,
    *,
    include_labels: bool = True,
) -> SourceImportPlan:
    """Copy one source into the project's internal canonical image store."""
    standardized_root = _standardized_source_dir(plan.source_root, project_anchor)
    images_root = standardized_root / "images"
    images_root.mkdir(parents=True, exist_ok=True)

    image_paths: list[Path] = []
    label_updates: dict[str, tuple[str, float]] = {}
    metadata_by_path: dict[str, dict[str, Any]] = {}
    discovered_labels = list(plan.discovered_labels) if include_labels else []

    for index, source_path in enumerate(plan.image_paths, start=1):
        source_key = str(source_path.resolve())
        source_metadata = plan.metadata_by_path.get(source_key, {})
        original_name = _slugify_name(source_path.stem)[:48]
        target_name = f"{index:06d}_{original_name}{source_path.suffix.lower()}"
        target_path = (images_root / target_name).resolve()
        shutil.copy2(source_path, target_path)

        target_key = str(target_path)
        image_paths.append(target_path)
        metadata_by_path[target_key] = _materialize_metadata(
            source_metadata,
            original_image_path=source_path,
            standardized_source_dir=standardized_root,
            include_label_metadata=include_labels,
        )
        if include_labels and source_key in plan.label_updates:
            label_updates[target_key] = plan.label_updates[source_key]

    return SourceImportPlan(
        source_root=plan.source_root,
        source_kind=plan.source_kind,
        image_paths=image_paths,
        label_updates=label_updates,
        metadata_by_path=metadata_by_path,
        discovered_labels=discovered_labels,
    )


def _metadata_payload(
    source_root: Path, source_kind: str, extra: dict[str, Any] | None = None
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "source_root": str(source_root.resolve()),
        "source_kind": source_kind,
    }
    if extra:
        payload.update(extra)
    return payload


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
    candidates = [root / raw_path]
    candidates.append(root / "images" / raw_path)
    if raw_path.name != str(raw_path):
        candidates.append(root / raw_path.name)
        candidates.append(root / "images" / raw_path.name)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise ValueError(f"COCO image not found for entry: {file_name}")


def inspect_external_source(root: Path) -> ExternalSourceInspection | None:
    """Return an external-dataset summary when *root* matches a supported format."""
    root = root.expanduser().resolve()

    class_folder_summary = _inspect_class_folder_source(root)
    if class_folder_summary is not None:
        return class_folder_summary

    coco_dataset = _load_coco_dataset(root)
    if coco_dataset is not None:
        _json_path, payload = coco_dataset
        categories = {
            int(entry.get("id")): str(entry.get("name"))
            for entry in payload.get("categories", [])
            if entry.get("id") is not None and entry.get("name") is not None
        }
        discovered = sorted({name for name in categories.values() if name})
        return ExternalSourceInspection(
            source_kind="coco",
            images_count=len(payload.get("images", [])),
            annotation_count=len(payload.get("annotations", [])),
            discovered_labels=discovered,
        )

    try:
        inspection = inspect_obb_or_detect_dataset(root)
    except Exception:
        return None

    items = [item for split_items in inspection.splits.values() for item in split_items]
    label_set = sorted(
        {str(name) for name in inspection.class_names.values() if str(name)}
    )
    annotation_count = 0
    for item in items:
        label_path = Path(item.label_path)
        if not label_path.exists():
            continue
        try:
            annotation_count += sum(
                1
                for line in label_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        except Exception:
            continue

    return ExternalSourceInspection(
        source_kind="yolo_obb",
        images_count=len(items),
        annotation_count=annotation_count,
        discovered_labels=label_set,
    )


def _inspect_class_folder_source(root: Path) -> ExternalSourceInspection | None:
    split_dirs = [
        root / split for split in ("train", "val", "test") if (root / split).is_dir()
    ]
    if not split_dirs:
        return None

    images_count = 0
    discovered_labels: set[str] = set()
    for split_dir in split_dirs:
        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            images = _iter_supported_images(class_dir)
            if not images:
                continue
            images_count += len(images)
            discovered_labels.add(class_dir.name)

    if images_count == 0 or not discovered_labels:
        return None

    return ExternalSourceInspection(
        source_kind="class_folders",
        images_count=images_count,
        annotation_count=images_count,
        discovered_labels=sorted(discovered_labels),
    )


def _parse_yolo_label_file(
    label_path: Path, class_names: dict[int, str]
) -> tuple[str | None, dict[str, Any]]:
    if not label_path.exists():
        return None, {"annotation_count": 0, "class_ids": [], "category_names": []}

    class_ids: list[int] = []
    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        try:
            class_ids.append(int(float(parts[0])))
        except Exception as exc:
            raise ValueError(
                f"Invalid YOLO label line in {label_path}: {raw_line}"
            ) from exc

    category_names = sorted(
        {class_names.get(class_id, f"class_{class_id}") for class_id in class_ids}
    )
    if len(category_names) > 1:
        raise ValueError(
            "ClassKit imports one label per image. "
            f"{label_path} contains multiple categories: {', '.join(category_names)}"
        )

    label = category_names[0] if category_names else None
    return label, {
        "annotation_count": len(class_ids),
        "class_ids": sorted(set(class_ids)),
        "category_names": category_names,
    }


def _build_yolo_plan(root: Path) -> SourceImportPlan:
    inspection = inspect_obb_or_detect_dataset(root)
    label_updates: dict[str, tuple[str, float]] = {}
    metadata_by_path: dict[str, dict[str, Any]] = {}
    discovered_labels: list[str] = []
    image_paths: list[Path] = []
    seen_paths: set[str] = set()

    for split_name, split_items in inspection.splits.items():
        for item in split_items:
            image_path = Path(item.image_path).resolve()
            label_path = Path(item.label_path).resolve()
            image_key = str(image_path)
            if image_key not in seen_paths:
                image_paths.append(image_path)
                seen_paths.add(image_key)

            label, label_meta = _parse_yolo_label_file(
                label_path, inspection.class_names
            )
            metadata_by_path[image_key] = _metadata_payload(
                root,
                "yolo_obb",
                {
                    "split": split_name,
                    "label_path": str(label_path),
                    **label_meta,
                },
            )
            if label:
                label_updates[image_key] = (label, 1.0)
                if label not in discovered_labels:
                    discovered_labels.append(label)

    return SourceImportPlan(
        source_root=root,
        source_kind="yolo_obb",
        image_paths=image_paths,
        label_updates=label_updates,
        metadata_by_path=metadata_by_path,
        discovered_labels=discovered_labels,
    )


def _build_coco_plan(root: Path) -> SourceImportPlan:
    loaded = _load_coco_dataset(root)
    if loaded is None:
        raise ValueError(f"No COCO annotations found in {root}")

    json_path, payload = loaded
    categories = {
        int(entry.get("id")): str(entry.get("name"))
        for entry in payload.get("categories", [])
        if entry.get("id") is not None and entry.get("name") is not None
    }

    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in payload.get("annotations", []):
        image_id = annotation.get("image_id")
        if image_id is None:
            continue
        annotations_by_image[int(image_id)].append(annotation)

    image_paths: list[Path] = []
    label_updates: dict[str, tuple[str, float]] = {}
    metadata_by_path: dict[str, dict[str, Any]] = {}
    discovered_labels: list[str] = []

    for image_entry in payload.get("images", []):
        image_id = image_entry.get("id")
        file_name = image_entry.get("file_name")
        if image_id is None or not file_name:
            continue
        image_path = _resolve_coco_image_path(root, str(file_name))
        image_key = str(image_path)
        image_paths.append(image_path)

        annotations = annotations_by_image.get(int(image_id), [])
        category_names = sorted(
            {
                categories.get(
                    int(annotation.get("category_id")),
                    f"class_{annotation.get('category_id')}",
                )
                for annotation in annotations
                if annotation.get("category_id") is not None
            }
        )
        if len(category_names) > 1:
            raise ValueError(
                "ClassKit imports one label per image. "
                f"{file_name} contains multiple categories: {', '.join(category_names)}"
            )
        label = category_names[0] if category_names else None
        metadata_by_path[image_key] = _metadata_payload(
            root,
            "coco",
            {
                "annotation_path": str(json_path),
                "image_id": int(image_id),
                "file_name": str(file_name),
                "annotation_count": len(annotations),
                "category_names": category_names,
                "category_ids": sorted(
                    {
                        int(annotation.get("category_id"))
                        for annotation in annotations
                        if annotation.get("category_id") is not None
                    }
                ),
            },
        )
        if label:
            label_updates[image_key] = (label, 1.0)
            if label not in discovered_labels:
                discovered_labels.append(label)

    return SourceImportPlan(
        source_root=root,
        source_kind="coco",
        image_paths=image_paths,
        label_updates=label_updates,
        metadata_by_path=metadata_by_path,
        discovered_labels=discovered_labels,
    )


def _build_class_folder_plan(root: Path) -> SourceImportPlan:
    image_paths: list[Path] = []
    label_updates: dict[str, tuple[str, float]] = {}
    metadata_by_path: dict[str, dict[str, Any]] = {}
    discovered_labels: list[str] = []

    for split_name in ("train", "val", "test"):
        split_dir = root / split_name
        if not split_dir.is_dir():
            continue
        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            label = class_dir.name.strip()
            if not label:
                continue
            images = _iter_supported_images(class_dir)
            if not images:
                continue
            if label not in discovered_labels:
                discovered_labels.append(label)
            for image_path in images:
                image_key = str(image_path)
                image_paths.append(image_path)
                label_updates[image_key] = (label, 1.0)
                metadata_by_path[image_key] = _metadata_payload(
                    root,
                    "class_folders",
                    {
                        "split": split_name,
                        "class_name": label,
                        "class_folder": str(class_dir.resolve()),
                    },
                )

    return SourceImportPlan(
        source_root=root,
        source_kind="class_folders",
        image_paths=image_paths,
        label_updates=label_updates,
        metadata_by_path=metadata_by_path,
        discovered_labels=discovered_labels,
    )


def detect_multihead_label_structure(
    labels: list[str], separator: str = "_"
) -> list[list[str]] | None:
    """Return per-factor value lists if labels consistently encode multiple factors.

    All labels must split into the same number of components (>= 2) and each
    component position must have at least 2 distinct values.  Returns None when
    the structure does not look like a multihead encoding.
    """
    if not labels or len(labels) < 2:
        return None

    split_labels = [label.split(separator) for label in labels]
    n_factors = len(split_labels[0])
    if n_factors < 2:
        return None
    if not all(len(parts) == n_factors for parts in split_labels):
        return None

    factor_values: list[list[str]] = []
    for i in range(n_factors):
        seen = list(dict.fromkeys(parts[i] for parts in split_labels))
        if len(seen) < 2:
            return None
        factor_values.append(seen)

    return factor_values


def recode_plan_labels(
    plan: SourceImportPlan, from_sep: str, to_sep: str
) -> SourceImportPlan:
    """Return a copy of *plan* with all label strings re-encoded from *from_sep* to *to_sep*."""

    def _recode(label: str) -> str:
        return to_sep.join(label.split(from_sep))

    return SourceImportPlan(
        source_root=plan.source_root,
        source_kind=plan.source_kind,
        image_paths=list(plan.image_paths),
        label_updates={
            path: (_recode(label), conf)
            for path, (label, conf) in plan.label_updates.items()
        },
        metadata_by_path=dict(plan.metadata_by_path),
        discovered_labels=[_recode(label) for label in plan.discovered_labels],
    )


def build_source_import_plan(source_root: Path) -> SourceImportPlan:
    """Build a normalized import plan for a selected ClassKit source."""
    source_root = source_root.expanduser().resolve()

    external = inspect_external_source(source_root)
    if external is not None:
        if external.source_kind == "class_folders":
            return _build_class_folder_plan(source_root)
        if external.source_kind == "coco":
            return _build_coco_plan(source_root)
        return _build_yolo_plan(source_root)

    image_paths = list(scan_images(source_root))
    metadata_by_path = {
        str(path.resolve()): _metadata_payload(source_root, "images")
        for path in image_paths
    }
    return SourceImportPlan(
        source_root=source_root,
        source_kind="images",
        image_paths=image_paths,
        label_updates={},
        metadata_by_path=metadata_by_path,
        discovered_labels=[],
    )
