#!/usr/bin/env python3
"""
Utilities for PoseKit Dialogs.
"""

import logging
import math
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPen

from hydra_suite.utils.conda_utils import run_conda


# Settings helpers
def load_ui_settings():
    """Load the persistent PoseKit UI settings dictionary from disk."""
    from ..utils import load_ui_settings as _load

    return _load()


def save_ui_settings(settings):
    """Write the PoseKit UI settings dictionary back to disk."""
    from ..utils import save_ui_settings as _save

    return _save(settings)


# GPU utils
try:
    from hydra_suite.utils.gpu_utils import (
        CUDA_AVAILABLE,
        MPS_AVAILABLE,
        TORCH_CUDA_AVAILABLE,
    )
except ImportError:
    CUDA_AVAILABLE = False
    MPS_AVAILABLE = False
    TORCH_CUDA_AVAILABLE = False

logger = logging.getLogger("pose_label.dialogs.utils")

# PoseInferenceService — optional; guarded at call sites
try:
    from hydra_suite.integrations.sleap.service import PoseInferenceService
except ImportError:
    PoseInferenceService = None  # type: ignore[assignment]

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _load_dialog_settings(key: str) -> Dict:
    settings = load_ui_settings()
    return settings.get("dialogs", {}).get(key, {})


def _save_dialog_settings(key: str, data: Dict) -> None:
    settings = load_ui_settings()
    dialogs = settings.get("dialogs", {})
    dialogs[key] = data
    settings["dialogs"] = dialogs
    save_ui_settings(settings)


def make_pose_infer(out_root: Path, keypoint_names: List[str]):
    """Construct and return a PoseInferenceService configured for the given project output root."""
    if PoseInferenceService is None:
        raise RuntimeError("PoseInferenceService not available. Check imports.")
    return PoseInferenceService(out_root, keypoint_names)


def get_available_devices() -> List[str]:
    """Get list of available compute devices based on gpu_utils flags."""
    devices: List[str] = ["auto", "cpu"]
    if CUDA_AVAILABLE or TORCH_CUDA_AVAILABLE:
        devices.append("cuda")
    if MPS_AVAILABLE:
        devices.append("mps")
    return devices


def list_sleap_envs() -> Tuple[List[str], str]:
    """Return (envs, error_message). envs contains conda envs starting with 'sleap'."""
    if shutil.which("conda") is None:
        return [], "Conda not found on PATH."
    try:
        res = run_conda(
            ["conda", "env", "list"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if res.returncode != 0:
            return [], "Unable to list conda environments."
        envs: List[str] = []
        for line in (res.stdout or "").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if not parts:
                continue
            name = parts[0]
            if name.lower().startswith("sleap"):
                envs.append(name)
        return envs, ""
    except Exception as e:
        return [], f"Env scan failed: {e}"


def make_histogram_image(values, bins: int = 20, width: int = 360, height: int = 120):
    """Render a bar-chart histogram of *values* into a QImage of the requested dimensions."""
    img = QImage(width, height, QImage.Format_ARGB32)
    img.fill(QColor(248, 248, 248))
    if not values:
        return img

    hist, _ = np.histogram(values, bins=bins)
    max_count = int(hist.max()) if hist.size else 1
    if max_count <= 0:
        return img

    painter = QPainter(img)
    bar_w = max(1, width // bins)
    for i, count in enumerate(hist):
        x = i * bar_w
        h = int((count / max_count) * (height - 8))
        painter.fillRect(x, height - h, bar_w - 1, h, QColor(60, 120, 200))
    painter.end()
    return img


def format_float(val, digits: int = 3):
    """Format a numeric value to a fixed number of decimal places, returning ``'n/a'`` for non-finite inputs."""
    if val is None or (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
        return "n/a"
    try:
        return f"{float(val):.{digits}f}"
    except Exception:
        return "n/a"


def get_yolo_pose_base_models() -> List[str]:
    """Known YOLO Pose base models from Ultralytics docs."""
    return [
        "yolo26n-pose.pt",
        "yolo26s-pose.pt",
        "yolo26m-pose.pt",
        "yolo26l-pose.pt",
        "yolo26x-pose.pt",
    ]


def list_images_in_dir(images_dir: Path) -> List[Path]:
    """List images in directory with known extensions."""
    paths: List[Path] = []
    for p in sorted(images_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            paths.append(p)
    return paths


def _label_path_for_image(base_dir: Path, img_path: Path) -> Optional[Path]:
    parts = list(img_path.parts)
    if "images" in parts:
        idx = parts.index("images")
        parts[idx] = "labels"
        lbl = Path(*parts).with_suffix(".txt")
        if lbl.exists():
            return lbl
    # fallback to base/labels/<stem>.txt
    lbl = base_dir / "labels" / f"{img_path.stem}.txt"
    if lbl.exists():
        return lbl
    return None


def load_yolo_dataset_items(
    dataset_yaml: Path,
) -> Tuple[List[Tuple[Path, Path]], Dict[str, object]]:
    """Load items from a YOLO dataset YAML."""
    data = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8"))
    base = Path(data.get("path", dataset_yaml.parent)).expanduser().resolve()
    train = data.get("train")
    val = data.get("val")
    names = data.get("names", {})
    kpt_shape = data.get("kpt_shape")
    kpt_names = data.get("kpt_names")

    def _resolve_source(src):
        if src is None:
            return []
        p = Path(src)
        if not p.is_absolute():
            p = (base / p).resolve()
        if p.is_file() and p.suffix.lower() in [".txt"]:
            lines = [
                line.strip() for line in p.read_text(encoding="utf-8").splitlines()
            ]
            out = []
            for line in lines:
                if not line:
                    continue
                lp = Path(line)
                if not lp.is_absolute():
                    lp = (base / lp).resolve()
                out.append(lp)
            return out
        if p.is_dir():
            return list_images_in_dir(p)
        return []

    images = _resolve_source(train) + _resolve_source(val)
    items: List[Tuple[Path, Path]] = []
    for img in images:
        lbl = _label_path_for_image(base, img)
        if lbl is not None:
            items.append((img, lbl))

    info = {
        "base": str(base),
        "names": names,
        "kpt_shape": kpt_shape,
        "kpt_names": kpt_names,
    }
    return items, info


def make_loss_plot_image(
    train_vals: Dict[str, List[float]],
    val_vals: Dict[str, List[float]],
    width: int = 520,
    height: int = 220,
) -> QImage:
    """Create a QImage rendering of loss training curves."""
    img = QImage(width, height, QImage.Format_ARGB32)
    img.fill(QColor(30, 30, 30))
    painter = QPainter(img)
    painter.setRenderHint(QPainter.Antialiasing, True)

    pad_left = 40
    pad_right = 12
    pad_top = 18
    pad_bottom = 32
    w = width - pad_left - pad_right
    h = height - pad_top - pad_bottom

    # Background frame
    painter.setPen(QPen(QColor(80, 80, 80), 1))
    painter.drawRect(pad_left, pad_top, w, h)

    all_vals = []
    for series in list(train_vals.values()) + list(val_vals.values()):
        all_vals.extend([v for v in series if v is not None and np.isfinite(v)])

    if not all_vals:
        painter.end()
        return img

    vmin = min(all_vals)
    vmax = max(all_vals)
    if vmax <= vmin:
        vmax = vmin + 1e-6

    # Plot helper
    keys = sorted(set(train_vals) | set(val_vals))
    palette = [
        QColor(80, 200, 120),
        QColor(255, 140, 80),
        QColor(120, 180, 255),
        QColor(255, 200, 80),
        QColor(200, 120, 255),
        QColor(120, 220, 220),
    ]

    for idx, key in enumerate(keys):
        color = palette[idx % len(palette)]
        pen_solid = QPen(color, 2)
        pen_dash = QPen(color, 2, Qt.DashLine)

        # Train (solid)
        vals = train_vals.get(key, [])
        if len(vals) > 1:
            painter.setPen(pen_solid)
            n = len(vals)
            for i in range(1, n):
                v0 = vals[i - 1]
                v1 = vals[i]
                if (
                    v0 is None
                    or v1 is None
                    or not np.isfinite(v0)
                    or not np.isfinite(v1)
                ):
                    continue
                x0 = pad_left + (w * (i - 1) / max(1, n - 1))
                x1 = pad_left + (w * i / max(1, n - 1))
                y0 = pad_top + h * (1.0 - (v0 - vmin) / (vmax - vmin))
                y1 = pad_top + h * (1.0 - (v1 - vmin) / (vmax - vmin))
                painter.drawLine(int(x0), int(y0), int(x1), int(y1))

        # Val (dashed)
        vals_val = val_vals.get(key, [])
        if len(vals_val) > 1:
            painter.setPen(pen_dash)
            n = len(vals_val)
            for i in range(1, n):
                v0 = vals_val[i - 1]
                v1 = vals_val[i]
                if (
                    v0 is None
                    or v1 is None
                    or not np.isfinite(v0)
                    or not np.isfinite(v1)
                ):
                    continue
                x0 = pad_left + (w * (i - 1) / max(1, n - 1))
                x1 = pad_left + (w * i / max(1, n - 1))
                y0 = pad_top + h * (1.0 - (v0 - vmin) / (vmax - vmin))
                y1 = pad_top + h * (1.0 - (v1 - vmin) / (vmax - vmin))
                painter.drawLine(int(x0), int(y0), int(x1), int(y1))

    # Text (Legend)
    painter.setPen(QColor(200, 200, 200))
    # Note: QFont might need import if not already

    # Legend area (bottom)
    lx = pad_left
    ly = height - 8
    for idx, key in enumerate(keys):
        color = palette[idx % len(palette)]
        painter.setPen(color)
        painter.drawText(lx, ly, key)
        lx += (len(key) * 7) + 20
        if lx > width - 20:
            break

    painter.end()
    return img
