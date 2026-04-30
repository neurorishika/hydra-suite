"""Export labeled images to PyTorch ImageFolder directory structure."""

import os
import shutil
from pathlib import Path
from typing import List, Tuple

from tqdm import tqdm


def _resolve_destination_path(dest_dir: Path, src_path: Path) -> Path:
    """Return a collision-safe destination path within one split/class directory."""
    dest_path = dest_dir / src_path.name
    counter = 1
    while dest_path.exists():
        dest_path = dest_dir / f"{src_path.stem}_{counter}{src_path.suffix}"
        counter += 1
    return dest_path


def export_to_imagefolder(
    dataset_root: Path,
    images: List[Tuple[Path, str, str]],  # (source_path, label, split)
    copy: bool = True,
):
    """
    Export dataset to PyTorch ImageFolder structure.

    Args:
        dataset_root: The root directory for the export.
        images: List of tuples (source_path, label, split).
                split should be 'train', 'val', or 'test'.
        copy: If True, copies files. If False, symlinks (ln -s).
    """
    dataset_root = Path(dataset_root)
    if dataset_root.exists():
        # Safety check: avoid deleting something unintended
        # shutil.rmtree(dataset_root)
        pass

    dataset_root.mkdir(parents=True, exist_ok=True)

    for src_path, label, split in tqdm(images, desc="Exporting ImageFolder"):
        if not label:
            continue

        dest_dir = dataset_root / split / label
        dest_dir.mkdir(parents=True, exist_ok=True)

        dest_path = _resolve_destination_path(dest_dir, src_path)
        if copy:
            shutil.copy2(src_path, dest_path)
        else:
            if not dest_path.exists():
                os.symlink(src_path.absolute(), dest_path)

    print(f"Exported ImageFolder to {dataset_root}")
