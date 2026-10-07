"""Utility functions for PoseKit: path helpers, image enhancement, palettes, and UI settings."""

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PySide6.QtGui import QColor

from hydra_suite.utils.hidden_files import is_hidden_file

from .constants import IMG_EXTS

logger = logging.getLogger("pose_label")


def _relativize_path(path: Path, base: Path) -> Path:
    try:
        return Path(os.path.relpath(path, base))
    except Exception:
        return path


def _resolve_project_path(path: Path, base: Path, data: dict) -> Path:
    if path.is_absolute():
        return path
    if bool(data.get("paths_relative", False)):
        return (base / path).resolve()
    return (base / path).resolve()


def list_images(images_dir: Path) -> List[Path]:
    """Recursively list supported image files in sorted order."""
    paths: List[Path] = []
    for p in sorted(images_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMG_EXTS and not is_hidden_file(p):
            paths.append(p)
    return paths


def enhance_for_pose(
    img_bgr: np.ndarray,
    clahe_clip: float = 2.0,
    clahe_grid: Tuple[int, int] = (8, 8),
    sharpen_amt: float = 1.2,
    blur_sigma: float = 1.0,
) -> np.ndarray:
    """Apply CLAHE and unsharp masking to improve pose keypoint visibility."""
    # 1) CLAHE on L channel (LAB)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_grid)
    L2 = clahe.apply(L)
    lab2 = cv2.merge([L2, A, B])
    img = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

    # 2) Unsharp mask (mild sharpening)
    blur = cv2.GaussianBlur(img, (0, 0), blur_sigma)
    sharp = cv2.addWeighted(img, 1.0 + sharpen_amt, blur, -sharpen_amt, 0)

    return sharp


def get_default_skeleton_dir() -> Optional[Path]:
    """Return the user-level skeleton config directory if available."""
    from hydra_suite.paths import get_skeleton_dir

    d = get_skeleton_dir()
    if d.exists() and d.is_dir():
        return d
    return None


def get_keypoint_palette() -> List[QColor]:
    """Return a stable high-contrast color palette for keypoint overlays."""
    return [
        QColor(255, 99, 71),
        QColor(30, 144, 255),
        QColor(60, 179, 113),
        QColor(238, 130, 238),
        QColor(255, 165, 0),
        QColor(0, 206, 209),
        QColor(255, 215, 0),
        QColor(199, 21, 133),
        QColor(127, 255, 0),
        QColor(70, 130, 180),
        QColor(255, 105, 180),
        QColor(64, 224, 208),
    ]


def get_ui_settings_path() -> Path:
    """Get path to UI settings file in user's home directory."""
    config_dir = Path.home() / "posekit"
    config_dir.mkdir(exist_ok=True)
    return config_dir / "ui_settings.json"


def load_ui_settings() -> Dict[str, Any]:
    """Load persistent UI settings."""
    path = get_ui_settings_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_ui_settings(settings: Dict[str, Any]) -> None:
    """Save persistent UI settings."""
    path = get_ui_settings_path()
    try:
        path.write_text(json.dumps(settings, indent=2), encoding="utf-8")
    except Exception:
        pass


def _clamp01(value: float) -> float:
    """Clamp value to [0, 1] range with NaN protection."""
    if not np.isfinite(value):
        logger.warning(f"Non-finite value clamped to 0: {value}")
        return 0.0
    return max(0.0, min(1.0, value))


def _xyxy_to_cxcywh(
    x1: float, y1: float, x2: float, y2: float
) -> Tuple[float, float, float, float]:
    """Convert bounding box from (x1, y1, x2, y2) to (center_x, center_y, width, height)."""
    # Validate inputs
    if not all(np.isfinite(v) for v in (x1, y1, x2, y2)):
        logger.error(f"Non-finite bbox coordinates: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
        return (0.5, 0.5, 0.1, 0.1)  # Return safe default

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    bw = x2 - x1
    bh = y2 - y1

    # Ensure positive dimensions
    bw = max(bw, 0.001)
    bh = max(bh, 0.001)

    return cx, cy, bw, bh


def _stable_hash_dict(d: dict) -> str:
    s = json.dumps(d, sort_keys=True).encode("utf-8")
    return hashlib.sha1(s).hexdigest()[:12]


def _choose_device(pref: str = "auto") -> str:
    if pref and pref != "auto":
        return pref
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _maybe_empty_cuda_cache() -> None:
    """Release unused CUDA memory back to the allocator if CUDA is available."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
