"""Read a model's .slice_meta.json sidecar and map it to SAHI panel/config values.

Pure (stdlib + numpy). The sidecar is written by training/model_publish.py for
OBB-direct models trained with sliced data; TrackerKit reads it to pre-fill the
SAHI panel. Mirrors the <artifact>.runtime_meta.json sidecar convention in
runtime_artifacts.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def read_slice_meta(model_path) -> "dict | None":
    """Return the parsed <model_path>.slice_meta.json dict, or None on absent/bad."""
    try:
        sidecar = Path(model_path).with_suffix(
            Path(model_path).suffix + ".slice_meta.json"
        )
        if not sidecar.exists():
            return None
        data = json.loads(sidecar.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def slice_meta_to_panel_values(meta: dict) -> dict:
    """Translate a slice_meta dict into SAHI panel/config values.

    object_tile_fraction = median(target_sizes)/imgsz (clamped [0.01, 0.9]);
    falls back to meta['object_tile_fraction'] (default 0.15) when target_sizes is
    empty/absent or imgsz is absent/0. reference stays model-internal: trained_body_px.
    """
    stored_fraction = float(meta.get("object_tile_fraction", 0.15) or 0.15)
    target_sizes = [float(t) for t in (meta.get("target_sizes") or [])]
    imgsz = int(meta.get("imgsz", 0) or 0)
    if target_sizes and imgsz > 0:
        frac = float(np.median(np.asarray(target_sizes, dtype=np.float64))) / float(
            imgsz
        )
        object_tile_fraction = max(0.01, min(0.9, frac))
    else:
        object_tile_fraction = stored_fraction
    return {
        "enabled": True,
        "geometry_mode": str(meta.get("geometry_mode", "auto_object") or "auto_object"),
        "overlap": float(meta.get("overlap", 0.2) or 0.2),
        "object_tile_fraction": object_tile_fraction,
        "trained_body_px": float(meta.get("reference_body_px", 0.0) or 0.0),
    }
