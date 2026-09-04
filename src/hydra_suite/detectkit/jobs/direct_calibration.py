"""DetectKit-side adapter and worker for direct-detector SAHI calibration.

Core owns the grid, the sweep and the scoring; this module supplies labelled
frames, drives the production runner, and persists project-local evidence.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml as _yaml

_LOGGER = logging.getLogger(__name__)

from hydra_suite.core.inference.direct_calibration_grid import label_set_fingerprint
from hydra_suite.detectkit.jobs.semantic_escalation import stratified_calibration_frames

EXHAUSTIVE_LABEL_WARNING = (
    "Confirm these frames are exhaustively labelled. A real animal missing "
    "from the labels looks like a false positive and biases calibration "
    "toward settings that are too strict."
)
MIN_MATCHED_NOTE = (
    "Too few matched instances for a recommendation. The measurements are "
    "still shown, but label a few more frames before trusting them."
)
_IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class EvidenceSet:
    frames: list
    split: str
    instances: int
    size_range: tuple
    sampled_from: int
    fingerprint: str


def _recording_key(image_path: Path) -> tuple[str, str]:
    """Group frames by their recording: parent dir + filename stem prefix.

    Neighbouring frames from one video must stay together -- scattering them
    across recordings makes the evidence set look more diverse than it is.
    """
    stem = image_path.stem
    prefix = stem.rsplit("_", 1)[0] if "_" in stem else stem
    return (str(image_path.parent), prefix)


def _labels_dir_for(images_dir: Path) -> Path:
    """Resolve the labels directory that mirrors ``images_dir``.

    Replaces only the LAST ``images`` path segment with ``labels`` -- naive
    string substitution (``str.replace("/images/", "/labels/")``) rewrites
    every occurrence, which corrupts paths where an ancestor directory is
    also named ``images`` (e.g. a dataset rooted at ``/data/images/pilot1``).
    """
    parts = list(images_dir.parts)
    for index in range(len(parts) - 1, -1, -1):
        if parts[index] == "images":
            parts[index] = "labels"
            return Path(*parts)
    return images_dir.parent.parent / "labels" / images_dir.name


def _split_frames(dataset_yaml: Path, split: str) -> list:
    from hydra_suite.data.al.escalation import LabelRecord
    from hydra_suite.detectkit.gui.utils import parse_obb_label
    from hydra_suite.utils.geometry_levels import GeometryLevel

    document = _yaml.safe_load(Path(dataset_yaml).read_text(encoding="utf-8")) or {}
    root = Path(document.get("path") or Path(dataset_yaml).parent)
    rel = document.get(split)
    images_dir = (root / rel) if rel else (root / "images" / split)
    if not images_dir.is_dir():
        return []
    labels_dir = _labels_dir_for(images_dir)
    if not labels_dir.is_dir():
        _LOGGER.warning(
            "No labels directory found for images dir %s (expected %s); "
            "treating split '%s' as having zero labelled frames.",
            images_dir,
            labels_dir,
            split,
        )
        return []
    out = []
    for image_path in sorted(
        p for p in images_dir.rglob("*") if p.suffix.lower() in _IMG_EXTS
    ):
        label_path = labels_dir / (image_path.stem + ".txt")
        if not label_path.exists() or not label_path.read_text().strip():
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        height, width = image.shape[:2]
        parsed = parse_obb_label(label_path, width, height)
        if not parsed:
            continue
        out.append(
            (
                image_path,
                [
                    LabelRecord(
                        class_id=int(d["class_id"]),
                        confidence=1.0,
                        points=np.asarray(d["polygon_px"], dtype=np.float32).reshape(
                            -1, 2
                        ),
                        level=GeometryLevel.POLYGON,
                    )
                    for d in parsed
                ],
            )
        )
    return out


def _bounded_by_recording(frames: list, budget: int) -> list:
    """Take whole recordings until the budget is reached.

    A single recording that alone exceeds the budget is truncated to the
    budget -- the whole-group rule protects against scattering, not against
    an oversized first group swallowing the entire run unbounded.
    """
    if not budget or len(frames) <= budget:
        return frames
    grouped: dict[tuple[str, str], list] = {}
    for item in frames:
        grouped.setdefault(_recording_key(Path(item[0])), []).append(item)
    output: list = []
    for _key, group in sorted(grouped.items()):
        if not output and len(group) > budget:
            return group[:budget]
        if output and len(output) + len(group) > budget:
            break
        output.extend(group)
    return output or frames[:budget]


def collect_evidence(
    *,
    dataset_yaml: Path | None,
    sources: list,
    split: str = "val",
    budget: int = 80,
) -> EvidenceSet:
    """Labelled full-resolution evidence, defaulting to the held-out val split.

    Tuning on frames the model took gradient steps on reports optimistic
    numbers, so ``val`` is the default and any fallback is reported in
    ``EvidenceSet.split`` for the UI to show.
    """
    used_split = split
    frames: list = []
    if dataset_yaml is not None:
        frames = _split_frames(Path(dataset_yaml), split)
        if not frames and split != "train":
            frames = _split_frames(Path(dataset_yaml), "train")
            if frames:
                used_split = "train"
    if not frames and sources:
        frames = stratified_calibration_frames(sources, budget=budget)
        used_split = "sources"
    total = len(frames)
    frames = _bounded_by_recording(frames, budget)
    sizes = []
    for image_path, _labels in frames:
        image = cv2.imread(str(image_path))
        if image is not None:
            sizes.append(tuple(image.shape[:2]))
    size_range = (min(sizes), max(sizes)) if sizes else ((0, 0), (0, 0))
    return EvidenceSet(
        frames=frames,
        split=used_split,
        instances=sum(len(labels) for _p, labels in frames),
        size_range=size_range,
        sampled_from=total,
        fingerprint=label_set_fingerprint(frames),
    )
