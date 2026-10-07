"""Utilities to validate, normalize, and merge YOLO-OBB datasets.

The functions in this module are used by the GUI dataset builder to combine
multiple sources (including converted X-AnyLabeling projects) into a single,
train/val-ready output directory.
"""

import hashlib
import json
import logging
import os
import random
import shutil
from pathlib import Path
from typing import Any

from hydra_suite.utils.hidden_files import is_hidden_file

logger = logging.getLogger(__name__)


def _hash_file(path, chunk_size=1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _read_dataset_yaml(yaml_path):
    try:
        import yaml  # type: ignore
    except Exception as e:
        raise RuntimeError("PyYAML is required to parse dataset.yaml") from e

    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)


def _normalize_path(root, p):
    if p is None:
        return None
    p = str(p)
    return p if os.path.isabs(p) else os.path.join(root, p)


def _detect_layout_from_yaml(root_dir: str) -> dict[str, tuple[str, str]]:
    """Try to detect layout from dataset.yaml; return splits dict (may be empty)."""
    yaml_path = os.path.join(root_dir, "dataset.yaml")
    if not os.path.exists(yaml_path):
        return {}
    data = _read_dataset_yaml(yaml_path)
    splits = {}
    for split_name in ("train", "val", "test"):
        path = _normalize_path(root_dir, data.get(split_name))
        if path:
            splits[split_name] = (path, _labels_dir_for_images(path))
    return splits


def _detect_layout_from_dirs(root_dir: str) -> dict[str, tuple[str, str]]:
    """Try to detect layout from standard images/labels directories."""
    images_root = os.path.join(root_dir, "images")
    labels_root = os.path.join(root_dir, "labels")
    if not (os.path.isdir(images_root) and os.path.isdir(labels_root)):
        return {}

    # Check for split subfolders
    if any(os.path.isdir(os.path.join(images_root, s)) for s in ["train", "val"]):
        splits = {}
        for s in ["train", "val"]:
            img_dir = os.path.join(images_root, s)
            lbl_dir = os.path.join(labels_root, s)
            if os.path.isdir(img_dir):
                splits[s] = (img_dir, lbl_dir)
        return splits
    # Unsplit
    return {"all": (images_root, labels_root)}


def detect_dataset_layout(root_dir: str | Path) -> dict[str, tuple[str, str]]:
    """Detect supported dataset folder layout.

    Args:
        root_dir: Dataset root directory.

    Returns:
        Mapping of split name to `(images_dir, labels_dir)`.

    Raises:
        RuntimeError: If no recognized dataset structure is found.
    """
    root_dir = str(root_dir)

    splits = _detect_layout_from_yaml(root_dir)
    if splits:
        return splits

    splits = _detect_layout_from_dirs(root_dir)
    if splits:
        return splits

    raise RuntimeError(f"No valid dataset layout found in {root_dir}")


def get_dataset_class_name(root_dir: str | Path) -> str | None:
    """Read class name from `dataset.yaml` or `classes.txt` if available."""
    yaml_path = os.path.join(root_dir, "dataset.yaml")
    if os.path.exists(yaml_path):
        try:
            data = _read_dataset_yaml(yaml_path)
            names = data.get("names")
            if isinstance(names, dict) and 0 in names:
                return names[0]
        except Exception:
            pass
    classes_path = os.path.join(root_dir, "classes.txt")
    if os.path.exists(classes_path):
        with open(classes_path, "r") as f:
            line = f.readline().strip()
            return line or None
    return None


def update_dataset_class_name(root_dir: str | Path, class_name: str) -> None:
    """Update dataset class metadata in yaml/txt files."""
    yaml_path = os.path.join(root_dir, "dataset.yaml")
    if os.path.exists(yaml_path):
        try:
            data = _read_dataset_yaml(yaml_path)
            data["names"] = {0: class_name}
            import yaml  # type: ignore

            with open(yaml_path, "w") as f:
                yaml.safe_dump(data, f, sort_keys=False)
        except Exception:
            pass
    classes_path = os.path.join(root_dir, "classes.txt")
    if os.path.exists(classes_path):
        with open(classes_path, "w") as f:
            f.write(f"{class_name}\n")


def _labels_dir_for_images(images_dir):
    # Common YOLO layout: images/<split> and labels/<split>
    parts = Path(images_dir).parts
    if "images" in parts:
        idx = parts.index("images")
        labels_parts = list(parts)
        labels_parts[idx] = "labels"
        return str(Path(*labels_parts))
    return str(Path(images_dir).parent / "labels")


def validate_labels(labels_dir: str | Path) -> tuple[set[int], int]:
    """Validate YOLO-OBB labels and return discovered class IDs and file count."""
    class_ids = set()
    total = 0
    for root, _, files in os.walk(labels_dir):
        for fn in files:
            if not fn.endswith(".txt") or is_hidden_file(fn):
                continue
            total += 1
            fp = os.path.join(root, fn)
            with open(fp, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) != 9:
                        raise RuntimeError(f"Invalid label format in {fp}: {line}")
                    try:
                        class_id = int(float(parts[0]))
                    except Exception:
                        raise RuntimeError(f"Invalid class id in {fp}: {line}")
                    class_ids.add(class_id)
    return class_ids, total


def rewrite_labels_to_single_class(labels_dir: str | Path, class_id: int = 0) -> None:
    """Rewrite all label files so every object uses the same class ID."""
    for root, _, files in os.walk(labels_dir):
        for fn in files:
            if not fn.endswith(".txt") or is_hidden_file(fn):
                continue
            fp = os.path.join(root, fn)
            lines = []
            with open(fp, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) != 9:
                        raise RuntimeError(f"Invalid label format in {fp}: {line}")
                    parts[0] = str(class_id)
                    lines.append(" ".join(parts))
            with open(fp, "w") as f:
                f.write("\n".join(lines) + ("\n" if lines else ""))


def write_classes_txt(root_dir: str | Path, class_name: str) -> None:
    """Write `classes.txt` with a single class name."""
    with open(os.path.join(root_dir, "classes.txt"), "w") as f:
        f.write(f"{class_name}\n")


def merge_datasets(
    sources: list[dict[str, Any]],
    output_dir: str | Path,
    class_name: str,
    split_cfg: dict[str, float],
    seed: int = 42,
    dedup: bool = True,
) -> str:
    """Merge multiple YOLO-OBB datasets.

    sources: list of dicts {"name": str, "path": str}
    output_dir: base output dir
    split_cfg: dict with train/val/test ratios

    Returns merged dataset path.
    """
    random.seed(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    merged_dir = output_dir / f"combined_dataset_{_timestamp()}"
    images_dir = merged_dir / "images"
    labels_dir = merged_dir / "labels"
    for split in ["train", "val"]:
        (images_dir / split).mkdir(parents=True, exist_ok=True)
        (labels_dir / split).mkdir(parents=True, exist_ok=True)

    seen_hashes = set()
    manifest = {
        "sources": sources,
        "dedup": dedup,
        "split": split_cfg,
        "class_name": class_name,
        "stats": {"total_images": 0, "unique_images": 0, "duplicates_skipped": 0},
    }

    for src in sources:
        layout = detect_dataset_layout(src["path"])
        if "all" in layout:
            # split here
            img_dir, lbl_dir = layout["all"]
            img_files = _collect_images(img_dir)
            random.shuffle(img_files)
            train_n, val_n = _split_counts(len(img_files), split_cfg)
            splits = {
                "train": img_files[:train_n],
                "val": img_files[train_n : train_n + val_n],
            }
            for split, files in splits.items():
                _copy_split(
                    files,
                    img_dir,
                    lbl_dir,
                    images_dir / split,
                    labels_dir / split,
                    dedup,
                    seen_hashes,
                    manifest,
                )
        else:
            for split, (img_dir, lbl_dir) in layout.items():
                files = _collect_images(img_dir)
                _copy_split(
                    files,
                    img_dir,
                    lbl_dir,
                    images_dir / split,
                    labels_dir / split,
                    dedup,
                    seen_hashes,
                    manifest,
                )

    # Write dataset.yaml + classes
    write_classes_txt(merged_dir, class_name)
    _write_dataset_yaml(merged_dir, class_name, include_test=False)
    _write_manifest(merged_dir, manifest)

    return str(merged_dir)


def _collect_images(img_dir: str | Path) -> list[str]:
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    files = []
    for root, _, fnames in os.walk(img_dir):
        for fn in fnames:
            if Path(fn).suffix.lower() in exts and not is_hidden_file(fn):
                files.append(os.path.join(root, fn))
    return files


def _copy_split(
    files: list[str],
    img_dir: str | Path,
    lbl_dir: str | Path,
    out_img_dir: str | Path,
    out_lbl_dir: str | Path,
    dedup: bool,
    seen_hashes: set[str],
    manifest: dict[str, Any],
) -> None:
    for fp in files:
        manifest["stats"]["total_images"] += 1
        if dedup:
            h = _hash_file(fp)
            if h in seen_hashes:
                manifest["stats"]["duplicates_skipped"] += 1
                continue
            seen_hashes.add(h)
        manifest["stats"]["unique_images"] += 1

        rel = os.path.relpath(fp, img_dir)
        base = os.path.splitext(os.path.basename(rel))[0]
        lbl_src = os.path.join(lbl_dir, base + ".txt")
        if not os.path.exists(lbl_src):
            raise RuntimeError(f"Missing label for image: {fp}")

        dest_img = _unique_dest(out_img_dir, os.path.basename(fp))
        dest_lbl = os.path.join(
            out_lbl_dir, os.path.splitext(os.path.basename(dest_img))[0] + ".txt"
        )
        shutil.copy2(fp, dest_img)
        shutil.copy2(lbl_src, dest_lbl)


def _unique_dest(out_dir, filename):
    out_dir = str(out_dir)
    base, ext = os.path.splitext(filename)
    dest = os.path.join(out_dir, filename)
    if not os.path.exists(dest):
        return dest
    idx = 1
    while True:
        candidate = os.path.join(out_dir, f"{base}_{idx}{ext}")
        if not os.path.exists(candidate):
            return candidate
        idx += 1


def _split_counts(n, split_cfg):
    train = split_cfg.get("train", 0.8)
    val = split_cfg.get("val", 0.2)
    train_n = int(n * train)
    val_n = int(n * val)
    return train_n, val_n


def _write_dataset_yaml(root_dir, class_name, include_test=True):
    path = Path(root_dir)
    data = {
        "path": str(path),
        "train": "images/train",
        "val": "images/val",
        "names": {0: class_name},
    }
    if include_test:
        data["test"] = "images/test"
    try:
        import yaml  # type: ignore

        with open(path / "dataset.yaml", "w") as f:
            yaml.safe_dump(data, f, sort_keys=False)
    except Exception:
        # fallback minimal
        with open(path / "dataset.yaml", "w") as f:
            f.write(f"path: {path}\n")
            f.write("train: images/train\n")
            f.write("val: images/val\n")
            if include_test:
                f.write("test: images/test\n")
            f.write("names:\n  0: %s\n" % class_name)


def _write_manifest(root_dir, manifest):
    with open(Path(root_dir) / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


def _timestamp():
    import datetime

    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
