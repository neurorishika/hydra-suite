"""DetectKit project model dataclasses."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

DEFAULT_CLASS_NAME = "object"


@dataclass
class PendingEscalation:
    """A staged (not-yet-reviewed) SAM2 escalation result awaiting accept/reject."""

    staged_path: str = ""
    target_level: str = "polygon"
    sam2_variant: str = ""
    created_at: str = ""

    def to_dict(self) -> dict:
        """Serialize to a plain dictionary."""
        return {
            "staged_path": self.staged_path,
            "target_level": self.target_level,
            "sam2_variant": self.sam2_variant,
            "created_at": self.created_at,
        }

    @staticmethod
    def from_dict(d: dict) -> "PendingEscalation":
        """Restore a PendingEscalation from a dictionary."""
        return PendingEscalation(
            staged_path=str(d.get("staged_path", "")),
            target_level=str(d.get("target_level", "polygon") or "polygon"),
            sam2_variant=str(d.get("sam2_variant", "")),
            created_at=str(d.get("created_at", "")),
        )


def normalize_class_names(values: Any) -> list[str]:
    """Normalize class-name input into a non-empty ordered list."""
    if isinstance(values, str):
        candidates = [values]
    elif isinstance(values, (list, tuple)):
        candidates = list(values)
    else:
        candidates = []

    class_names = [str(name).strip() for name in candidates if str(name).strip()]
    return class_names or [DEFAULT_CLASS_NAME]


@dataclass
class OBBSource:
    """Represents one source dataset directory."""

    path: str = ""
    name: str = ""
    validated: bool = False
    original_path: str = ""
    source_kind: str = "detectkit"
    imported: bool = False
    level: str = "obb"  # GeometryLevel.label; "obb" for pre-migration sources
    reviewed: bool = True  # False only for un-reviewed SAM2-primed derived sources
    derived_from: str | None = None  # origin source name for derived sources
    sam2_variant: str | None = None  # SAM2 version that primed a derived source
    pending_escalation: PendingEscalation | None = None  # staged, unreviewed escalation

    def to_dict(self) -> dict:
        """Serialize to a plain dictionary."""
        return {
            "path": self.path,
            "name": self.name,
            "validated": self.validated,
            "original_path": self.original_path,
            "source_kind": self.source_kind,
            "imported": self.imported,
            "level": self.level,
            "reviewed": self.reviewed,
            "derived_from": self.derived_from,
            "sam2_variant": self.sam2_variant,
            "pending_escalation": (
                self.pending_escalation.to_dict()
                if self.pending_escalation is not None
                else None
            ),
        }

    @staticmethod
    def from_dict(d: dict) -> OBBSource:
        """Restore an OBBSource from a dictionary."""
        return OBBSource(
            path=str(d.get("path", "")),
            name=str(d.get("name", "")),
            validated=bool(d.get("validated", False)),
            original_path=str(d.get("original_path", "")),
            source_kind=str(d.get("source_kind", "detectkit") or "detectkit"),
            imported=bool(d.get("imported", False)),
            level=str(d.get("level", "obb") or "obb"),
            reviewed=bool(d.get("reviewed", True)),
            derived_from=(d.get("derived_from") or None),
            sam2_variant=(d.get("sam2_variant") or None),
            pending_escalation=(
                PendingEscalation.from_dict(d["pending_escalation"])
                if d.get("pending_escalation")
                else None
            ),
        )


@dataclass
class SliceTrainingSettings:
    """Shared SAHI sliced-training + preview geometry, persisted with the project."""

    enabled: bool = False
    geometry_mode: str = "auto_object"  # auto_model | auto_object | custom
    object_tile_fraction: float = 0.15
    reference_body_px: float = 0.0
    slice_width: int = 0
    slice_height: int = 0
    overlap: float = 0.2
    min_area_ratio: float = 0.1
    negative_tile_fraction: float = 0.15
    target_sizes: list[float] = field(default_factory=lambda: [200.0, 300.0, 400.0])
    full_frame_mix: bool = True
    merge_threshold: float = 0.5

    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "geometry_mode": self.geometry_mode,
            "object_tile_fraction": self.object_tile_fraction,
            "reference_body_px": self.reference_body_px,
            "slice_width": self.slice_width,
            "slice_height": self.slice_height,
            "overlap": self.overlap,
            "min_area_ratio": self.min_area_ratio,
            "negative_tile_fraction": self.negative_tile_fraction,
            "target_sizes": list(self.target_sizes),
            "full_frame_mix": self.full_frame_mix,
            "merge_threshold": self.merge_threshold,
        }

    @staticmethod
    def from_dict(d: dict) -> "SliceTrainingSettings":
        base = SliceTrainingSettings()
        if not isinstance(d, dict):
            return base
        return SliceTrainingSettings(
            enabled=bool(d.get("enabled", base.enabled)),
            geometry_mode=str(
                d.get("geometry_mode", base.geometry_mode) or base.geometry_mode
            ),
            object_tile_fraction=float(
                d.get("object_tile_fraction", base.object_tile_fraction)
            ),
            reference_body_px=float(d.get("reference_body_px", base.reference_body_px)),
            slice_width=int(d.get("slice_width", base.slice_width)),
            slice_height=int(d.get("slice_height", base.slice_height)),
            overlap=float(d.get("overlap", base.overlap)),
            min_area_ratio=float(d.get("min_area_ratio", base.min_area_ratio)),
            negative_tile_fraction=float(
                d.get("negative_tile_fraction", base.negative_tile_fraction)
            ),
            target_sizes=[
                float(x) for x in (d.get("target_sizes") or base.target_sizes)
            ],
            full_frame_mix=bool(d.get("full_frame_mix", base.full_frame_mix)),
            merge_threshold=float(d.get("merge_threshold", base.merge_threshold)),
        )


def populate_measured_reference(
    settings: SliceTrainingSettings, measured: float
) -> bool:
    """Set settings.reference_body_px from a measured value only when currently unset.

    Returns True iff it changed the value (settings.reference_body_px was 0.0 and
    measured > 0). A user-set value is never overwritten.
    """
    if settings.reference_body_px == 0.0 and float(measured) > 0.0:
        settings.reference_body_px = float(measured)
        return True
    return False


@dataclass
class DetectKitProject:
    """Full project state, persisted as JSON."""

    # Core
    project_dir: Path = field(default_factory=lambda: Path("."))
    class_names: list[str] = field(default_factory=lambda: [DEFAULT_CLASS_NAME])
    sources: list[OBBSource] = field(default_factory=list)

    # Split
    split_train: float = 0.8
    split_val: float = 0.2
    seed: int = 42
    dedup: bool = True

    # Crop
    crop_pad_ratio: float = 0.15
    min_crop_size_px: int = 64
    enforce_square: bool = True

    # Per-role imgsz
    imgsz_obb_direct: int = 640
    imgsz_detect_direct: int = 640
    imgsz_segment_direct: int = 640
    imgsz_seq_detect: int = 640
    imgsz_seq_crop_obb: int = 160
    imgsz_seq_crop_segment: int = 160

    # Base models
    model_obb_direct: str = "yolo26s-obb.pt"
    model_detect_direct: str = "yolo26s.pt"
    model_segment_direct: str = "yolo26s-seg.pt"
    model_seq_detect: str = "yolo26s.pt"
    model_seq_crop_obb: str = "yolo26s-obb.pt"
    model_seq_crop_segment: str = "yolo26s-seg.pt"

    # Hyperparams
    epochs: int = 100
    batch: int = 16
    lr0: float = 0.01
    patience: int = 30
    workers: int = 8
    cache: bool = False
    auto_batch: bool = False

    # Augmentation
    aug_enabled: bool = True
    aug_fliplr: float = 0.5
    aug_flipud: float = 0.0
    aug_degrees: float = 0.0
    aug_mosaic: float = 1.0
    aug_mixup: float = 0.0
    aug_hsv_h: float = 0.015
    aug_hsv_s: float = 0.7
    aug_hsv_v: float = 0.4

    # Roles
    role_obb_direct: bool = True
    role_detect_direct: bool = False
    role_segment_direct: bool = False
    role_seq_detect: bool = True
    role_seq_crop_obb: bool = True
    role_seq_crop_segment: bool = False

    # Simple training-plan selection. The role booleans remain persisted for
    # backward compatibility with existing projects and run history.
    training_mode: str = "direct"
    training_task: str = "obb"

    # Device
    device: str = "auto"

    # Publish
    species: str = ""
    model_tag: str = "train"
    auto_import: bool = True
    auto_select: bool = False

    # Session
    last_source_index: int = 0
    last_image_index: int = 0
    active_model_path: str = ""
    training_history: list[dict[str, Any]] = field(default_factory=list)
    slice_settings: SliceTrainingSettings = field(default_factory=SliceTrainingSettings)

    @property
    def class_name(self) -> str:
        """Backward-compatible access to the primary class name."""
        return normalize_class_names(self.class_names)[0]

    @class_name.setter
    def class_name(self, value: str) -> None:
        """Backward-compatible setter that replaces the project class list."""
        self.class_names = normalize_class_names([value])

    def to_dict(self) -> dict[str, Any]:
        """Serialize all fields to a dictionary."""
        d: dict[str, Any] = {"version": 1}
        for f in fields(self):
            val = getattr(self, f.name)
            if f.name == "project_dir":
                d[f.name] = str(val)
            elif f.name == "sources":
                d[f.name] = [s.to_dict() for s in val]
            elif f.name == "slice_settings":
                d[f.name] = val.to_dict()
            else:
                d[f.name] = val
        return d

    def save(self, path: Path) -> None:
        """Write project state as JSON to *path*."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @staticmethod
    def load(path: Path) -> DetectKitProject:
        """Read JSON from *path* and return a DetectKitProject."""
        raw: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))

        # Build a defaults instance to know field names and types.
        proj = DetectKitProject()
        known = {f.name: f for f in fields(proj)}

        for name in known:
            if name not in raw:
                continue
            val = raw[name]

            if name == "project_dir":
                proj.project_dir = Path(val)
            elif name == "class_names":
                proj.class_names = normalize_class_names(val)
            elif name == "sources":
                proj.sources = [OBBSource.from_dict(s) for s in val]
            elif name == "slice_settings":
                proj.slice_settings = SliceTrainingSettings.from_dict(val)
            else:
                # Type-cast based on the default type.
                default_val = getattr(proj, name)
                if isinstance(default_val, bool):
                    setattr(proj, name, bool(val))
                elif isinstance(default_val, int):
                    setattr(proj, name, int(val))
                elif isinstance(default_val, float):
                    setattr(proj, name, float(val))
                elif isinstance(default_val, str):
                    setattr(proj, name, str(val))
                else:
                    setattr(proj, name, val)

        if "class_names" not in raw and "class_name" in raw:
            proj.class_names = normalize_class_names([raw.get("class_name", "")])
        else:
            proj.class_names = normalize_class_names(proj.class_names)

        return proj
