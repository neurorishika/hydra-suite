"""DetectKit project model dataclasses."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

DEFAULT_CLASS_NAME = "object"


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

    def to_dict(self) -> dict:
        """Serialize to a plain dictionary."""
        return {
            "path": self.path,
            "name": self.name,
            "validated": self.validated,
            "original_path": self.original_path,
            "source_kind": self.source_kind,
            "imported": self.imported,
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
        )


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
    imgsz_seq_detect: int = 640
    imgsz_seq_crop_obb: int = 160

    # Base models
    model_obb_direct: str = "yolo26s-obb.pt"
    model_seq_detect: str = "yolo26s.pt"
    model_seq_crop_obb: str = "yolo26s-obb.pt"

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
    role_seq_detect: bool = True
    role_seq_crop_obb: bool = True

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
