"""Shared source-folder validation helpers for ClassKit dialogs."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from hydra_suite.classkit.core.data.source_import import inspect_external_source

CLASSKIT_IMAGES_SUBDIR = "images"
CLASSKIT_IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class ClassKitSourceInspection:
    """Inspection result for a candidate ClassKit source folder."""

    dataset_root: Path
    images_dir: Path
    images_count: int
    source_kind: str = "images"
    annotation_count: int = 0
    needs_standardization: bool = False


def list_classkit_images(folder: Path) -> list[Path]:
    """Return supported image files directly inside *folder*."""
    return [
        path
        for path in sorted(folder.iterdir())
        if path.is_file() and path.suffix.lower() in CLASSKIT_IMAGE_EXTS
    ]


def count_classkit_images(folder: Path) -> int:
    """Return the number of supported image files directly inside *folder*."""
    direct_images = list_classkit_images(folder)
    if direct_images:
        return len(direct_images)
    try:
        inspection = inspect_classkit_source_dir(folder)
    except ValueError:
        return 0
    return inspection.images_count


def has_classkit_images(folder: Path) -> bool:
    """Return whether *folder* contains any supported image files."""
    return count_classkit_images(folder) > 0


def inspect_classkit_source_dir(dataset_root: Path) -> ClassKitSourceInspection:
    """Inspect *dataset_root* and describe whether it can be used as a source."""
    external = inspect_external_source(dataset_root)
    if external is not None:
        return ClassKitSourceInspection(
            dataset_root=dataset_root,
            images_dir=dataset_root,
            images_count=external.images_count,
            source_kind=external.source_kind,
            annotation_count=external.annotation_count,
            needs_standardization=False,
        )

    images_dir = dataset_root / CLASSKIT_IMAGES_SUBDIR
    if images_dir.is_dir() and has_classkit_images(images_dir):
        return ClassKitSourceInspection(
            dataset_root=dataset_root,
            images_dir=images_dir,
            images_count=count_classkit_images(images_dir),
            source_kind="images",
            needs_standardization=False,
        )

    root_images = list_classkit_images(dataset_root)
    if root_images:
        return ClassKitSourceInspection(
            dataset_root=dataset_root,
            images_dir=images_dir,
            images_count=len(root_images),
            source_kind="images",
            needs_standardization=True,
        )

    raise ValueError(
        "Selected source folder must contain compatible .jpg / .jpeg / .png files, "
        "an images/ subdirectory, a supported COCO or YOLO OBB/detect dataset root, "
        "or a train/val class-folder dataset root.\n\n"
        f"{dataset_root}"
    )


def standardize_classkit_source_dir(dataset_root: Path) -> Path:
    """Create *dataset_root/images* and copy supported root images into it."""
    root_images = list_classkit_images(dataset_root)
    if not root_images:
        raise ValueError(
            f"No compatible images were found in:\n{dataset_root}\n\n"
            "Please select a source folder containing .jpg / .jpeg / .png files."
        )

    images_dir = dataset_root / CLASSKIT_IMAGES_SUBDIR
    if images_dir.exists() and not images_dir.is_dir():
        raise ValueError(
            f"Cannot create images/ because a file already exists at:\n{images_dir}"
        )

    images_dir.mkdir(parents=True, exist_ok=True)
    for source_path in root_images:
        shutil.copy2(source_path, images_dir / source_path.name)
    return images_dir


def resolve_classkit_images_dir(dataset_root: Path) -> Path:
    """Return the required images directory for a selected ClassKit source."""
    inspection = inspect_classkit_source_dir(dataset_root)
    if inspection.needs_standardization:
        raise ValueError(
            "Selected source folder does not contain an images/ subdirectory, "
            "but compatible images were found directly in the folder.\n\n"
            "Standardize the folder first or use the dialog prompt to create "
            "an images/ folder automatically.\n\n"
            f"{dataset_root}"
        )
    return inspection.images_dir
