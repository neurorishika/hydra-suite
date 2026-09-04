"""Helpers for dataset class-name declarations and class-id remapping."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterable

from .dataset_io import (
    DEFAULT_DATASET_IO_LIMITS,
    DatasetLimitError,
    iter_bounded_text_lines,
)


def _dedupe_and_validate_class_names(
    class_names: Iterable[str],
    *,
    source_label: str,
) -> list[str]:
    names: list[str] = []
    for name in class_names:
        normalized = str(name).strip()
        if not normalized:
            continue
        names.append(normalized)
        if len(names) > DEFAULT_DATASET_IO_LIMITS.max_classes:
            raise DatasetLimitError(
                "Class declarations exceed the safe cap of "
                f"{DEFAULT_DATASET_IO_LIMITS.max_classes} names"
            )
    if not names:
        raise RuntimeError(f"{source_label} does not declare any class names.")

    duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
    if duplicates:
        dup_list = ", ".join(duplicates)
        raise RuntimeError(
            f"{source_label} declares duplicate class names: {dup_list}."
        )
    return names


def normalize_declared_class_names(
    declared: list[str] | tuple[str, ...] | dict[int, str] | None,
    *,
    source_label: str,
) -> list[str]:
    """Normalize declared class names into a dense ordered list."""
    if declared is None:
        raise RuntimeError(f"{source_label} does not declare any class names.")

    if isinstance(declared, dict):
        if not declared:
            raise RuntimeError(f"{source_label} does not declare any class names.")
        normalized: dict[int, str] = {}
        for class_id, class_name in declared.items():
            try:
                normalized[int(class_id)] = str(class_name).strip()
            except Exception as exc:  # pragma: no cover - defensive branch
                raise RuntimeError(
                    f"{source_label} has an invalid class declaration: {class_id!r} -> {class_name!r}."
                ) from exc
        ordered_names: list[str] = []
        for class_id in range(max(normalized) + 1):
            if class_id not in normalized or not normalized[class_id]:
                raise RuntimeError(
                    f"{source_label} is missing a class name for class id {class_id}."
                )
            ordered_names.append(normalized[class_id])
        return _dedupe_and_validate_class_names(
            ordered_names, source_label=source_label
        )

    return _dedupe_and_validate_class_names(declared, source_label=source_label)


def read_classes_txt(dataset_root: str | Path) -> list[str]:
    """Read class names from ``classes.txt`` in *dataset_root*."""
    root = Path(dataset_root).expanduser().resolve()
    classes_path = root / "classes.txt"
    if not classes_path.exists():
        raise RuntimeError(f"Missing required classes.txt in {root}.")
    return normalize_declared_class_names(
        iter_bounded_text_lines(classes_path),
        source_label=str(classes_path),
    )


def resolve_dataset_class_names(
    dataset_root: str | Path,
    fallback_declared: list[str] | tuple[str, ...] | dict[int, str] | None = None,
) -> list[str]:
    """Resolve dataset class names, preferring ``classes.txt`` when present."""
    root = Path(dataset_root).expanduser().resolve()
    classes_path = root / "classes.txt"
    if classes_path.exists():
        return read_classes_txt(root)
    if fallback_declared is None:
        raise RuntimeError(
            f"{root} does not provide classes.txt or any fallback class names."
        )
    return normalize_declared_class_names(
        fallback_declared,
        source_label=f"class declarations for {root}",
    )


def build_class_id_map(
    source_class_names: list[str] | tuple[str, ...] | dict[int, str],
    target_class_names: list[str] | tuple[str, ...] | dict[int, str],
) -> dict[int, int]:
    """Map source class ids into target class ids by class name.

    The source class list must be an exact match or a strict superset of the
    target list. Extra source classes are intentionally omitted from the
    returned mapping so callers can drop those labels.
    """
    source_names = normalize_declared_class_names(
        source_class_names,
        source_label="source dataset classes",
    )
    target_names = normalize_declared_class_names(
        target_class_names,
        source_label="project classes",
    )

    source_name_to_id = {name: class_id for class_id, name in enumerate(source_names)}
    missing = [name for name in target_names if name not in source_name_to_id]
    if missing:
        missing_list = ", ".join(missing)
        raise RuntimeError(
            "Source classes must match the project class scheme or be a superset of it. "
            f"Missing project classes: {missing_list}."
        )

    return {
        source_name_to_id[class_name]: target_class_id
        for target_class_id, class_name in enumerate(target_names)
    }
