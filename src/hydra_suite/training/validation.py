"""Strict non-destructive validation for MAT training datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .contracts import TrainingRole, ValidationIssue, ValidationReport
from .dataset_inspector import DatasetInspection, inspect_obb_or_detect_dataset
from .dataset_io import (
    DEFAULT_DATASET_IO_LIMITS,
    DatasetLimitError,
    iter_bounded_text_lines,
)

MAX_VALIDATION_ISSUES = 1000
MAX_JSON_NESTING = 128


def _record_class_id(stats: dict[str, object], class_id: int) -> None:
    values = stats["class_ids"]
    if isinstance(values, set):
        values.add(class_id)
        if len(values) > DEFAULT_DATASET_IO_LIMITS.max_classes:
            raise DatasetLimitError(
                "Dataset labels exceed the safe cap of "
                f"{DEFAULT_DATASET_IO_LIMITS.max_classes} distinct class ids"
            )


def _count_coco_arrays(path: Path) -> tuple[int, int]:
    """Count top-level COCO array items with bounded streaming state."""

    counts = {"images": 0, "annotations": 0}
    found: set[str] = set()
    stack: list[str] = []
    in_string = False
    escaped = False
    string_chars: list[str] = []
    string_overflow = False
    pending_string: str | None = None
    pending_key: str | None = None
    target: str | None = None
    target_depth = -1
    expecting_item = False
    with path.open("r", encoding="utf-8") as stream:
        while chunk := stream.read(64 * 1024):
            for char in chunk:
                if in_string:
                    if escaped:
                        escaped = False
                    elif char == "\\":
                        escaped = True
                    elif char == '"':
                        in_string = False
                        pending_string = (
                            None if string_overflow else "".join(string_chars)
                        )
                    elif len(string_chars) < 64:
                        string_chars.append(char)
                    else:
                        string_overflow = True
                    continue
                if char == '"':
                    if (
                        target is not None
                        and len(stack) == target_depth
                        and expecting_item
                    ):
                        counts[target] += 1
                        expecting_item = False
                    in_string = True
                    escaped = False
                    string_chars = []
                    string_overflow = False
                    continue
                if char.isspace():
                    continue
                if char == ":" and len(stack) == 1 and stack[0] == "object":
                    pending_key = pending_string
                    pending_string = None
                    continue
                if char in "[{":
                    if (
                        target is not None
                        and len(stack) == target_depth
                        and expecting_item
                    ):
                        counts[target] += 1
                        expecting_item = False
                    if (
                        char == "["
                        and len(stack) == 1
                        and stack[0] == "object"
                        and pending_key in counts
                    ):
                        target = pending_key
                        target_depth = len(stack) + 1
                        expecting_item = True
                        found.add(target)
                    stack.append("array" if char == "[" else "object")
                    if len(stack) > MAX_JSON_NESTING:
                        raise DatasetLimitError(
                            f"COCO JSON exceeds nesting cap {MAX_JSON_NESTING}: {path}"
                        )
                    pending_key = None
                    continue
                if char in "]}":
                    expected = "array" if char == "]" else "object"
                    if not stack or stack[-1] != expected:
                        raise RuntimeError(f"Malformed COCO JSON structure: {path}")
                    if target is not None and len(stack) == target_depth:
                        target = None
                        target_depth = -1
                        expecting_item = False
                    stack.pop()
                    pending_key = None
                    continue
                if target is not None and len(stack) == target_depth:
                    if char == ",":
                        expecting_item = True
                    elif expecting_item:
                        counts[target] += 1
                        expecting_item = False
                pending_string = None
    if in_string or stack or found != set(counts):
        raise RuntimeError(f"Malformed or incomplete COCO JSON: {path}")
    return counts["images"], counts["annotations"]


def validate_coco_dataset(
    dataset_dir: str | Path, *, min_train: int = 1, min_val: int = 0
) -> ValidationReport:
    """Validate a COCO instance-segmentation layout (train/valid/_annotations.coco.json)."""
    root = Path(dataset_dir)
    issues: list[ValidationIssue] = []
    stats: dict[str, Any] = {"root_dir": str(root)}
    for split, floor in (("train", min_train), ("valid", min_val)):
        ann = root / split / "_annotations.coco.json"
        if not ann.exists():
            if floor > 0:
                issues.append(
                    ValidationIssue(
                        "error", "coco_missing_split", f"missing {ann}", str(ann)
                    )
                )
            continue
        n_img, n_ann = _count_coco_arrays(ann)
        stats[f"{split}_images"] = n_img
        stats[f"{split}_annotations"] = n_ann
        if n_img < floor or (floor > 0 and n_ann == 0):
            issues.append(
                ValidationIssue(
                    "error",
                    "coco_empty_split",
                    f"{split}: {n_img} images, {n_ann} annotations",
                    str(ann),
                )
            )
    return ValidationReport(
        valid=not any(i.severity == "error" for i in issues), issues=issues, stats=stats
    )


def _parse_label_lines(path: Path) -> list[list[str]]:
    lines = []
    for ln in iter_bounded_text_lines(path):
        ln = ln.strip()
        if not ln:
            continue
        lines.append(ln.split())
    return lines


def _validate_split_counts(
    inspection: DatasetInspection,
    min_train: int,
    min_val: int,
) -> list[ValidationIssue]:
    """Check that train/val splits meet minimum item requirements."""
    issues: list[ValidationIssue] = []
    train_count = len(inspection.splits.get("train", []))
    val_count = len(inspection.splits.get("val", []))
    if train_count < min_train:
        issues.append(
            ValidationIssue(
                severity="error",
                code="empty_train",
                message=f"Train split has {train_count} items; require >= {min_train}.",
            )
        )
    if val_count < min_val:
        issues.append(
            ValidationIssue(
                severity="error",
                code="empty_val",
                message=f"Val split has {val_count} items; require >= {min_val}.",
            )
        )
    return issues


def _validate_obb_line(
    parts: list[str], lbl: Path, stats: dict[str, object]
) -> list[ValidationIssue]:
    """Validate a single OBB label line and return any issues."""
    issues: list[ValidationIssue] = []
    if len(parts) != 9:
        stats["invalid_lines"] = int(stats["invalid_lines"]) + 1
        issues.append(
            ValidationIssue(
                severity="error",
                code="invalid_obb_format",
                message=f"Expected 9 fields for OBB line, got {len(parts)} fields.",
                path=str(lbl),
            )
        )
        return issues
    try:
        class_id = int(float(parts[0]))
        coords = [float(v) for v in parts[1:]]
    except Exception:
        issues.append(
            ValidationIssue(
                severity="error",
                code="invalid_numeric",
                message="Non-numeric OBB label values.",
                path=str(lbl),
            )
        )
        return issues
    _record_class_id(stats, class_id)
    for coord in coords:
        if coord < -1e-6 or coord > 1.0 + 1e-6:
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="coord_out_of_range",
                    message="Normalized OBB coordinate out of [0,1] range.",
                    path=str(lbl),
                )
            )
            break
    return issues


def _validate_detect_line(
    parts: list[str], lbl: Path, stats: dict[str, object]
) -> list[ValidationIssue]:
    """Validate a single axis-aligned YOLO detect label line."""
    issues: list[ValidationIssue] = []
    if len(parts) != 5:
        stats["invalid_lines"] = int(stats["invalid_lines"]) + 1
        issues.append(
            ValidationIssue(
                severity="error",
                code="invalid_detect_format",
                message=f"Expected 5 fields for detect line, got {len(parts)} fields.",
                path=str(lbl),
            )
        )
        return issues
    try:
        class_id = int(float(parts[0]))
        cx, cy, width, height = (float(v) for v in parts[1:5])
    except Exception:
        issues.append(
            ValidationIssue(
                severity="error",
                code="invalid_numeric",
                message="Non-numeric detect label values.",
                path=str(lbl),
            )
        )
        return issues
    _record_class_id(stats, class_id)
    for coord in (cx, cy, width, height):
        if coord < -1e-6 or coord > 1.0 + 1e-6:
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="coord_out_of_range",
                    message="Normalized detect coordinate out of [0,1] range.",
                    path=str(lbl),
                )
            )
            return issues
    if width <= 0.0 or height <= 0.0:
        issues.append(
            ValidationIssue(
                severity="error",
                code="non_positive_bbox",
                message="Detect bbox width/height must be positive.",
                path=str(lbl),
            )
        )
    return issues


def _validate_segment_line(
    parts: list[str], lbl: Path, stats: dict[str, object]
) -> list[ValidationIssue]:
    """Validate one YOLO instance-segmentation label line."""
    issues: list[ValidationIssue] = []
    if len(parts) < 7 or (len(parts) - 1) % 2:
        stats["invalid_lines"] = int(stats["invalid_lines"]) + 1
        issues.append(
            ValidationIssue(
                severity="error",
                code="invalid_segment_format",
                message="Expected class id followed by at least three x/y points.",
                path=str(lbl),
            )
        )
        return issues
    try:
        class_id = int(float(parts[0]))
        coords = [float(value) for value in parts[1:]]
    except Exception:
        issues.append(
            ValidationIssue(
                severity="error",
                code="invalid_numeric",
                message="Non-numeric segmentation label values.",
                path=str(lbl),
            )
        )
        return issues
    _record_class_id(stats, class_id)
    if any(coord < -1e-6 or coord > 1.0 + 1e-6 for coord in coords):
        issues.append(
            ValidationIssue(
                severity="error",
                code="coord_out_of_range",
                message="Normalized segmentation coordinate out of [0,1] range.",
                path=str(lbl),
            )
        )
    return issues


def _validate_item_file_pair(
    item, split: str, stats: dict[str, object]
) -> list[ValidationIssue]:
    """Validate one image/label pair and return issues."""
    issues: list[ValidationIssue] = []
    img = Path(item.image_path)
    lbl = Path(item.label_path)
    if not img.exists():
        issues.append(
            ValidationIssue(
                severity="error",
                code="missing_image",
                message="Image file missing.",
                path=str(img),
            )
        )
        return issues
    if not lbl.exists():
        stats["missing_labels"] = int(stats["missing_labels"]) + 1
        issues.append(
            ValidationIssue(
                severity="error",
                code="missing_label",
                message=f"Missing label for split '{split}'.",
                path=str(lbl),
            )
        )
        return issues

    try:
        parsed = _parse_label_lines(lbl)
    except Exception as exc:
        issues.append(
            ValidationIssue(
                severity="error",
                code="label_read_error",
                message=f"Cannot read label file: {exc}",
                path=str(lbl),
            )
        )
        return issues

    if not parsed:
        # Empty labels are valid YOLO negative examples.  In particular, SAHI
        # deliberately writes them for sampled background tiles.
        stats["empty_labels"] = int(stats.get("empty_labels", 0)) + 1
        return issues

    for parts in parsed:
        issues.extend(_validate_obb_line(parts, lbl, stats))
    return issues


def _validate_item_file_pair_for_mode(
    item,
    split: str,
    stats: dict[str, object],
    *,
    label_mode: str,
) -> list[ValidationIssue]:
    """Validate one image/label pair for the requested YOLO label mode."""
    issues: list[ValidationIssue] = []
    img = Path(item.image_path)
    lbl = Path(item.label_path)
    if not img.exists():
        issues.append(
            ValidationIssue(
                severity="error",
                code="missing_image",
                message="Image file missing.",
                path=str(img),
            )
        )
        return issues
    if not lbl.exists():
        stats["missing_labels"] = int(stats["missing_labels"]) + 1
        issues.append(
            ValidationIssue(
                severity="error",
                code="missing_label",
                message=f"Missing label for split '{split}'.",
                path=str(lbl),
            )
        )
        return issues

    try:
        parsed = _parse_label_lines(lbl)
    except Exception as exc:
        issues.append(
            ValidationIssue(
                severity="error",
                code="label_read_error",
                message=f"Cannot read label file: {exc}",
                path=str(lbl),
            )
        )
        return issues

    if not parsed:
        # Empty labels are valid YOLO negative examples.  In particular, SAHI
        # deliberately writes them for sampled background tiles.
        stats["empty_labels"] = int(stats.get("empty_labels", 0)) + 1
        return issues

    validators = {
        "obb": _validate_obb_line,
        "detect": _validate_detect_line,
        "segment": _validate_segment_line,
    }
    validator = validators[label_mode]
    for parts in parsed:
        issues.extend(validator(parts, lbl, stats))
    return issues


def validate_obb_dataset(
    inspection: DatasetInspection,
    *,
    require_train_val: bool = True,
    min_train: int = 1,
    min_val: int = 1,
) -> ValidationReport:
    """Validate OBB-label source dataset with strict fail-fast checks."""

    issues: list[ValidationIssue] = []
    stats: dict[str, object] = {
        "root_dir": inspection.root_dir,
        "split_counts": {k: len(v) for k, v in inspection.splits.items()},
        "missing_labels": 0,
        "empty_labels": 0,
        "invalid_lines": 0,
        "class_ids": set(),
    }

    if require_train_val:
        issues.extend(_validate_split_counts(inspection, min_train, min_val))

    for split, items in inspection.splits.items():
        for item in items:
            issues.extend(_validate_item_file_pair(item, split, stats))
            if len(issues) >= MAX_VALIDATION_ISSUES:
                stats["validation_truncated"] = True
                issues = issues[:MAX_VALIDATION_ISSUES]
                break
        if stats.get("validation_truncated"):
            break

    class_ids = sorted(int(x) for x in stats.get("class_ids", set()))
    if not class_ids:
        issues.append(
            ValidationIssue(
                severity="error",
                code="no_labeled_objects",
                message="Dataset contains no labeled objects.",
            )
        )
    if len(class_ids) > 1:
        issues.append(
            ValidationIssue(
                severity="error",
                code="multi_class_source",
                message=(
                    "Dataset contains multiple class IDs; expected a single-class "
                    "training source."
                ),
            )
        )
    stats["class_ids"] = class_ids

    return ValidationReport(
        valid=not any(i.severity == "error" for i in issues), issues=issues, stats=stats
    )


def validate_ultralytics_dataset(
    inspection: DatasetInspection,
    *,
    label_mode: str,
    require_single_class: bool = False,
    require_train_val: bool = True,
    min_train: int = 1,
    min_val: int = 1,
) -> ValidationReport:
    """Validate an Ultralytics OBB, detect, or segment dataset."""
    if label_mode not in {"obb", "detect", "segment"}:
        raise RuntimeError(f"Unsupported Ultralytics label mode: {label_mode}")

    issues: list[ValidationIssue] = []
    stats: dict[str, object] = {
        "root_dir": inspection.root_dir,
        "split_counts": {k: len(v) for k, v in inspection.splits.items()},
        "missing_labels": 0,
        "empty_labels": 0,
        "invalid_lines": 0,
        "class_ids": set(),
        "label_mode": label_mode,
    }

    if require_train_val:
        issues.extend(_validate_split_counts(inspection, min_train, min_val))

    for split, items in inspection.splits.items():
        for item in items:
            issues.extend(
                _validate_item_file_pair_for_mode(
                    item,
                    split,
                    stats,
                    label_mode=label_mode,
                )
            )
            if len(issues) >= MAX_VALIDATION_ISSUES:
                stats["validation_truncated"] = True
                issues = issues[:MAX_VALIDATION_ISSUES]
                break
        if stats.get("validation_truncated"):
            break

    class_ids = sorted(int(x) for x in stats.get("class_ids", set()))
    if not class_ids:
        issues.append(
            ValidationIssue(
                severity="error",
                code="no_labeled_objects",
                message="Dataset contains no labeled objects.",
            )
        )
    if require_single_class and len(class_ids) > 1:
        issues.append(
            ValidationIssue(
                severity="error",
                code="multi_class_source",
                message=(
                    "Dataset contains multiple class IDs; expected a single-class "
                    "training source."
                ),
            )
        )
    stats["class_ids"] = class_ids
    return ValidationReport(
        valid=not any(i.severity == "error" for i in issues),
        issues=issues,
        stats=stats,
    )


def validate_role_dataset(
    dataset_dir: str | Path,
    role: TrainingRole,
    *,
    require_train_val: bool = True,
    min_train: int = 1,
    min_val: int = 1,
) -> ValidationReport:
    """Inspect and validate a derived dataset for the requested training role."""
    # MUST precede inspect_obb_or_detect_dataset: that inspector RAISES on a
    # COCO layout, so branching after it is a crash, not a fall-through.
    if role is TrainingRole.SEMANTIC_SAM3:
        return validate_coco_dataset(dataset_dir, min_train=min_train, min_val=0)
    inspection = inspect_obb_or_detect_dataset(dataset_dir)
    if role in {TrainingRole.OBB_DIRECT, TrainingRole.SEQ_CROP_OBB}:
        return validate_ultralytics_dataset(
            inspection,
            label_mode="obb",
            require_train_val=require_train_val,
            min_train=min_train,
            min_val=min_val,
        )
    if role in {TrainingRole.DETECT_DIRECT, TrainingRole.SEQ_DETECT}:
        return validate_ultralytics_dataset(
            inspection,
            label_mode="detect",
            require_train_val=require_train_val,
            min_train=min_train,
            min_val=min_val,
        )
    if role in {TrainingRole.SEGMENT_DIRECT, TrainingRole.SEQ_CROP_SEGMENT}:
        return validate_ultralytics_dataset(
            inspection,
            label_mode="segment",
            require_train_val=require_train_val,
            min_train=min_train,
            min_val=min_val,
        )
    return ValidationReport(valid=True, stats={"root_dir": str(Path(dataset_dir))})


def format_validation_report(report: ValidationReport) -> str:
    """Format validation report for UI logs."""

    lines = [
        f"Validation: {'PASS' if report.valid else 'FAIL'}",
        f"Stats: {report.stats}",
    ]
    for issue in report.issues:
        where = f" [{issue.path}]" if issue.path else ""
        lines.append(f"- {issue.severity.upper()} {issue.code}: {issue.message}{where}")
    return "\n".join(lines)
