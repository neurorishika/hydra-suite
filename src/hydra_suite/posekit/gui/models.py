"""Data models for PoseKit: Keypoint, FrameAnn, DataSource, and Project."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from hydra_suite.core.pose_flip import normalize_flip_pairs, parse_flip_pairs

from .constants import DEFAULT_EDGE_WIDTH, DEFAULT_KPT_RADIUS, DEFAULT_LABEL_FONT_SIZE
from .utils import _relativize_path, _resolve_project_path

logger = logging.getLogger("pose_label")


@dataclass
class Keypoint:
    """Single keypoint annotation in pixel space."""

    x: float = 0.0  # pixel coords
    y: float = 0.0
    v: int = 0  # 0 missing, 1 occluded, 2 visible


@dataclass
class FrameAnn:
    """Single-frame annotation containing class, box, and keypoints."""

    cls: int
    bbox_xyxy: Optional[Tuple[float, float, float, float]]  # pixel coords
    kpts: List[Keypoint]


@dataclass
class DataSource:
    """A single source dataset ingested into a multi-source PoseKit project."""

    source_id: str  # unique within project, e.g. "source_0"
    dataset_root: Path  # original dataset root (for metadata.json lookup)
    images_dir: Path  # where this source's images live
    labels_dir: Path  # per-source label subdirectory (avoids filename collisions)
    description: str = ""  # human-readable label, e.g. "Species A – Jan 2025"

    def to_dict(self, base: Path) -> dict:
        """Serialize the source to a JSON-compatible dict with paths made relative to *base*."""
        return {
            "source_id": self.source_id,
            "dataset_root": str(_relativize_path(self.dataset_root, base)),
            "images_dir": str(_relativize_path(self.images_dir, base)),
            "labels_dir": str(_relativize_path(self.labels_dir, base)),
            "description": self.description,
        }

    @staticmethod
    def from_dict(d: dict, base: Path) -> "DataSource":
        """Reconstruct a DataSource from a serialized dict, resolving paths relative to *base*."""

        def _rp(key: str) -> Path:
            raw = Path(d[key])
            return _resolve_project_path(raw, base, {})

        return DataSource(
            source_id=str(d["source_id"]),
            dataset_root=_rp("dataset_root"),
            images_dir=_rp("images_dir"),
            labels_dir=_rp("labels_dir"),
            description=str(d.get("description", "")),
        )


@dataclass
class Project:
    """Persistent PoseKit project configuration and UI/session state."""

    images_dir: Path
    out_root: Path
    labels_dir: Path
    project_path: Path

    class_names: List[str]
    keypoint_names: List[str]
    skeleton_edges: List[Tuple[int, int]]  # 0-based indices

    dims: int = 3
    bbox_pad_frac: float = 0.03
    autosave: bool = True
    last_index: int = 0
    labeling_frames: List[int] = field(default_factory=list)

    # Display enhancement (CLAHE + unsharp mask)
    enhance_enabled: bool = False
    clahe_clip: float = 2.0
    clahe_grid: Tuple[int, int] = (8, 8)
    sharpen_amt: float = 1.2
    blur_sigma: float = 1.0

    # Display sizing
    kpt_radius: float = DEFAULT_KPT_RADIUS
    label_font_size: int = DEFAULT_LABEL_FONT_SIZE

    # Opacity
    kpt_opacity: float = 1.0
    edge_opacity: float = 0.7
    edge_width: float = DEFAULT_EDGE_WIDTH
    latest_pose_weights: Optional[Path] = None
    latest_sleap_dataset: Optional[Path] = None

    # Multi-source support — list of ingested dataset sources.
    # When non-empty, images/labels are routed per-source.
    # Legacy single-source projects have sources=[] and use images_dir/labels_dir.
    sources: List["DataSource"] = field(default_factory=list)
    last_source_id: str = ""

    # Smart Select's last batch, in pick order (image paths).  Drives the
    # "Smart Select batch" sort and frame navigation in that order.
    smart_batch: List[str] = field(default_factory=list)

    # The labeling set by image path.  ``labeling_frames`` (indices) shifts
    # when images are added or removed between sessions; paths do not, so
    # they win on load when present.
    labeling_frame_paths: List[str] = field(default_factory=list)

    # Left/right keypoint pairs (0-based) swapped when training mirrors an
    # image; see ``hydra_suite.core.pose_flip``.
    flip_pairs: List[Tuple[int, int]] = field(default_factory=list)

    def to_json(self) -> dict:
        """Serialize project state to a JSON-compatible dictionary."""
        base = self.project_path.parent
        images_dir = _relativize_path(self.images_dir, base)
        labels_dir = _relativize_path(self.labels_dir, base)
        out_root = _relativize_path(self.out_root, base)
        latest_weights = (
            _relativize_path(self.latest_pose_weights, base)
            if self.latest_pose_weights
            else None
        )
        latest_sleap_dataset = (
            _relativize_path(self.latest_sleap_dataset, base)
            if self.latest_sleap_dataset
            else None
        )
        paths_relative = (
            not Path(images_dir).is_absolute()
            and not Path(labels_dir).is_absolute()
            and not Path(out_root).is_absolute()
        )
        return {
            "images_dir": str(images_dir),
            "out_root": str(out_root),
            "labels_dir": str(labels_dir),
            "latest_pose_weights": str(latest_weights) if latest_weights else "",
            "latest_sleap_dataset": (
                str(latest_sleap_dataset) if latest_sleap_dataset else ""
            ),
            "paths_relative": bool(paths_relative),
            "class_names": self.class_names,
            "keypoint_names": self.keypoint_names,
            "skeleton_edges": [[a, b] for a, b in self.skeleton_edges],
            "flip_pairs": [[a, b] for a, b in self.flip_pairs],
            "dims": self.dims,
            "bbox_pad_frac": self.bbox_pad_frac,
            "autosave": self.autosave,
            "last_index": self.last_index,
            "labeling_frames": sorted({int(i) for i in self.labeling_frames}),
            "enhance_enabled": self.enhance_enabled,
            "clahe_clip": self.clahe_clip,
            "clahe_grid": list(self.clahe_grid),
            "sharpen_amt": self.sharpen_amt,
            "blur_sigma": self.blur_sigma,
            "kpt_radius": self.kpt_radius,
            "label_font_size": self.label_font_size,
            "kpt_opacity": self.kpt_opacity,
            "edge_opacity": self.edge_opacity,
            "edge_width": self.edge_width,
            "sources": [s.to_dict(base) for s in self.sources],
            "last_source_id": self.last_source_id,
            "smart_batch": [
                str(_relativize_path(Path(p), base)) for p in self.smart_batch
            ],
            "labeling_frame_paths": [
                str(_relativize_path(Path(p), base)) for p in self.labeling_frame_paths
            ],
        }

    @staticmethod
    def from_json(project_path: Path) -> "Project":
        """Load project configuration from disk."""
        data = json.loads(project_path.read_text(encoding="utf-8"))
        grid = data.get("clahe_grid", [8, 8])
        if not isinstance(grid, (list, tuple)) or len(grid) != 2:
            grid = [8, 8]
        base = project_path.parent
        images_dir = _resolve_project_path(Path(data["images_dir"]), base, data)
        labels_dir = _resolve_project_path(Path(data["labels_dir"]), base, data)
        out_root_raw = Path(data.get("out_root", labels_dir.parent))
        out_root = _resolve_project_path(out_root_raw, base, data)
        latest_raw = Path(data.get("latest_pose_weights", "") or "")
        latest_pose_weights = (
            _resolve_project_path(latest_raw, base, data) if str(latest_raw) else None
        )
        if latest_pose_weights and latest_pose_weights.suffix != ".pt":
            latest_pose_weights = None
        sleap_raw = Path(data.get("latest_sleap_dataset", "") or "")
        latest_sleap_dataset = (
            _resolve_project_path(sleap_raw, base, data) if str(sleap_raw) else None
        )
        if latest_sleap_dataset and latest_sleap_dataset.suffix != ".slp":
            latest_sleap_dataset = None
        return Project(
            images_dir=images_dir,
            out_root=out_root,
            labels_dir=labels_dir,
            project_path=project_path,
            class_names=list(data.get("class_names", ["object"])),
            keypoint_names=list(data.get("keypoint_names", ["kp1", "kp2"])),
            skeleton_edges=[
                (int(a), int(b)) for a, b in data.get("skeleton_edges", [])
            ],
            flip_pairs=normalize_flip_pairs(
                parse_flip_pairs(data),
                len(data.get("keypoint_names", ["kp1", "kp2"])),
            ),
            dims=int(data.get("dims", 3)),
            bbox_pad_frac=float(data.get("bbox_pad_frac", 0.03)),
            autosave=bool(data.get("autosave", True)),
            last_index=int(data.get("last_index", 0)),
            labeling_frames=[int(i) for i in data.get("labeling_frames", [])],
            enhance_enabled=bool(data.get("enhance_enabled", False)),
            clahe_clip=float(data.get("clahe_clip", 2.0)),
            clahe_grid=(int(grid[0]), int(grid[1])),
            sharpen_amt=float(data.get("sharpen_amt", 1.2)),
            blur_sigma=float(data.get("blur_sigma", 1.0)),
            kpt_radius=float(data.get("kpt_radius", DEFAULT_KPT_RADIUS)),
            label_font_size=int(data.get("label_font_size", DEFAULT_LABEL_FONT_SIZE)),
            kpt_opacity=float(data.get("kpt_opacity", 1.0)),
            edge_opacity=float(data.get("edge_opacity", 0.7)),
            edge_width=float(data.get("edge_width", DEFAULT_EDGE_WIDTH)),
            latest_pose_weights=latest_pose_weights,
            latest_sleap_dataset=latest_sleap_dataset,
            sources=[
                DataSource.from_dict(s, base)
                for s in data.get("sources", [])
                if isinstance(s, dict)
            ],
            last_source_id=str(data.get("last_source_id", "")),
            smart_batch=[
                str(_resolve_project_path(Path(str(p)), base, data))
                for p in data.get("smart_batch", []) or []
                if str(p).strip()
            ],
            labeling_frame_paths=[
                str(_resolve_project_path(Path(str(p)), base, data))
                for p in data.get("labeling_frame_paths", []) or []
                if str(p).strip()
            ],
        )


def compute_bbox_from_kpts(
    kpts: List[Keypoint], pad_frac: float, w: int, h: int
) -> Optional[Tuple[float, float, float, float]]:
    """Compute bounding box from visible keypoints with minimum size guarantee."""
    MIN_BBOX_SIZE = 2.0  # Minimum bbox size in pixels

    pts = [(kp.x, kp.y) for kp in kpts if kp.v > 0]
    if not pts:
        return None

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]

    # Check for valid coordinates
    if not all(np.isfinite(x) for x in xs) or not all(np.isfinite(y) for y in ys):
        logger.error("Non-finite keypoint coordinates detected")
        return None

    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)

    # Add padding
    pad = pad_frac * max((x2 - x1), (y2 - y1), 1.0)
    x1 = max(0.0, x1 - pad)
    y1 = max(0.0, y1 - pad)
    x2 = min(float(w - 1), x2 + pad)
    y2 = min(float(h - 1), y2 + pad)

    # Ensure minimum bbox size to prevent numerical issues
    if x2 - x1 < MIN_BBOX_SIZE:
        center_x = (x1 + x2) / 2.0
        x1 = max(0.0, center_x - MIN_BBOX_SIZE / 2.0)
        x2 = min(float(w - 1), center_x + MIN_BBOX_SIZE / 2.0)

    if y2 - y1 < MIN_BBOX_SIZE:
        center_y = (y1 + y2) / 2.0
        y1 = max(0.0, center_y - MIN_BBOX_SIZE / 2.0)
        y2 = min(float(h - 1), center_y + MIN_BBOX_SIZE / 2.0)

    # Final validation
    if x2 <= x1 + 1 or y2 <= y1 + 1:
        logger.warning(
            f"Computed bbox too small: ({x1:.1f}, {y1:.1f}) to ({x2:.1f}, {y2:.1f})"
        )
        return None

    return (x1, y1, x2, y2)
