"""Strict non-destructive validation for MAT training datasets."""

from __future__ import annotations

from pathlib import Path

from .contracts import TrainingRole, ValidationIssue, ValidationReport
from .dataset_inspector import DatasetInspection, inspect_obb_or_detect_dataset


def _parse_label_lines(path: Path) -> list[list[str]]:
    lines = []
    txt = path.read_text(encoding="utf-8").splitlines()
    for ln in txt:
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
    cast = stats["class_ids"]
    if isinstance(cast, set):
        cast.add(class_id)
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
    cast = stats["class_ids"]
    if isinstance(cast, set):
        cast.add(class_id)
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
        issues.append(
            ValidationIssue(
                severity="error",
                code="empty_label",
                message="Label file has no objects.",
                path=str(lbl),
            )
        )
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
        issues.append(
            ValidationIssue(
                severity="error",
                code="empty_label",
                message="Label file has no objects.",
                path=str(lbl),
            )
        )
        return issues

    validator = _validate_obb_line if label_mode == "obb" else _validate_detect_line
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
        "invalid_lines": 0,
        "class_ids": set(),
    }

    if require_train_val:
        issues.extend(_validate_split_counts(inspection, min_train, min_val))

    for split, items in inspection.splits.items():
        for item in items:
            issues.extend(_validate_item_file_pair(item, split, stats))

    class_ids = sorted(int(x) for x in stats.get("class_ids", set()))
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
    """Validate an Ultralytics detect/OBB dataset against the expected label mode."""
    if label_mode not in {"obb", "detect"}:
        raise RuntimeError(f"Unsupported Ultralytics label mode: {label_mode}")

    issues: list[ValidationIssue] = []
    stats: dict[str, object] = {
        "root_dir": inspection.root_dir,
        "split_counts": {k: len(v) for k, v in inspection.splits.items()},
        "missing_labels": 0,
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

    class_ids = sorted(int(x) for x in stats.get("class_ids", set()))
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
    inspection = inspect_obb_or_detect_dataset(dataset_dir)
    if role in {TrainingRole.OBB_DIRECT, TrainingRole.SEQ_CROP_OBB}:
        return validate_ultralytics_dataset(
            inspection,
            label_mode="obb",
            require_train_val=require_train_val,
            min_train=min_train,
            min_val=min_val,
        )
    if role == TrainingRole.SEQ_DETECT:
        return validate_ultralytics_dataset(
            inspection,
            label_mode="detect",
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
