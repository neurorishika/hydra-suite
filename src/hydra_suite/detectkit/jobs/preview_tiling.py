"""Resolve the DetectKit preview's SAHI tiling from the model sidecar (F2).

The preview tiles the way TrackerKit would for the same model with a fresh
config: geometry, scale, overlap, tile size and merge settings come from the
model's ``.slice_meta.json`` through ``slice_meta_to_panel_values`` -- the
SAME fresh-load ladder TrackerKit uses (primary profile, else the training
geometry). DetectKit owns only what is the user's knob (deviation 16):

* ``enabled`` -- the project's SAHI toggle;
* the reference body -- the project's label-measured value first, then the
  stamped ``trained_body_px`` (spec §3.3: dataset before stamp);
* ``imgsz`` -- the project's direct-OBB input size.

An open inference-settings override wins outright. Never raises: an absent
or corrupt sidecar resolves to the project settings.
"""

from __future__ import annotations

import copy
import logging
import math
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

from hydra_suite.core.inference.slice_meta import (
    read_slice_meta,
    sidecar_path,
    slice_meta_to_panel_values,
)
from hydra_suite.utils.tiling_spec import DEFAULT_YOLO_IMGSZ

from ..gui.models import InferenceRunSettings, SliceTrainingSettings

logger = logging.getLogger(__name__)

# The preview's historical merge (prediction_preview._preview_slice_merge_config).
DEFAULT_PREVIEW_MERGE_POLICY = "greedy_nmm"
DEFAULT_PREVIEW_MERGE_METRIC = "ios"
# What SliceConfig accepts; ``nmm`` passes through raw (F6).
_MERGE_POLICIES = ("nms", "nmm", "greedy_nmm")
_MERGE_METRICS = ("iou", "ios")


@dataclass(frozen=True)
class PreviewTiling:
    """The tiling one DetectKit preview run uses, and where it came from."""

    slice_settings: SliceTrainingSettings
    imgsz: int
    merge_policy: str = DEFAULT_PREVIEW_MERGE_POLICY
    merge_metric: str = DEFAULT_PREVIEW_MERGE_METRIC
    source: str = "project"  # override | profile:<name> | training | project


@lru_cache(maxsize=32)
def _read_slice_meta_cached(model_path: str, mtime_ns: int, size: int):
    """Parse once per (path, mtime_ns, size); a recalibration changes the key."""
    del mtime_ns, size  # cache-key only
    return read_slice_meta(model_path)


def _read_slice_meta(model_path: str | Path) -> dict | None:
    """The sidecar, memoized on its stat so overlay refreshes do not re-parse.

    ``_dataset_signature`` resolves the tiling several times per keypress (m2).
    A copy is returned: callers must never mutate the shared cached document.
    """
    try:
        stat = sidecar_path(model_path).stat()
    except OSError:
        return None
    meta = _read_slice_meta_cached(str(model_path), stat.st_mtime_ns, stat.st_size)
    return copy.deepcopy(meta) if meta is not None else None


def _finite_float(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if math.isfinite(parsed) else None


def _choice(value: object, allowed: tuple[str, ...], default: str) -> str:
    text = str(value).strip().lower() if value is not None else ""
    return text if text in allowed else default


def resolve_preview_tiling(
    model_path: str | Path | None,
    project_settings: SliceTrainingSettings,
    *,
    project_imgsz: int,
    override: SliceTrainingSettings | None,
) -> PreviewTiling:
    """Return the preview tiling for ``model_path`` (see module docstring)."""
    imgsz = max(1, int(project_imgsz or DEFAULT_YOLO_IMGSZ))
    if override is not None:
        return PreviewTiling(slice_settings=override, imgsz=imgsz, source="override")
    project_only = PreviewTiling(
        slice_settings=project_settings, imgsz=imgsz, source="project"
    )
    if not model_path:
        return project_only
    try:
        meta = _read_slice_meta(model_path)
        if meta is None:
            return project_only
        values = slice_meta_to_panel_values(meta, None)
        fraction = float(values["object_tile_fraction"])
        project_body = _finite_float(project_settings.reference_body_px) or 0.0
        body = (
            project_body
            if project_body > 0.0
            else float(values.get("trained_body_px") or 0.0)
        )
        merge_threshold = _finite_float(values.get("merge_threshold"))
        settings = replace(
            project_settings,
            # ``enabled`` stays the project's (deviation 16).
            geometry_mode=str(values["geometry_mode"]),
            overlap=float(values["overlap"]),
            slice_width=int(values["slice_width"]),
            slice_height=int(values["slice_height"]),
            object_tile_fraction=fraction,
            target_size_fractions=[fraction],
            reference_body_px=body,
            merge_threshold=(
                merge_threshold
                if merge_threshold is not None
                else project_settings.merge_threshold
            ),
        )
        profile_id = values.get("profile_id")
        source = f"profile:{values.get('profile_name')}" if profile_id else "training"
        return PreviewTiling(
            slice_settings=settings,
            imgsz=imgsz,
            merge_policy=_choice(
                values.get("merge_policy"),
                _MERGE_POLICIES,
                DEFAULT_PREVIEW_MERGE_POLICY,
            ),
            merge_metric=_choice(
                values.get("merge_metric"),
                _MERGE_METRICS,
                DEFAULT_PREVIEW_MERGE_METRIC,
            ),
            source=source,
        )
    except Exception:  # never raise into the GUI: fall back to the project
        logger.warning(
            "Could not resolve SAHI preview tiling from %s; using project settings.",
            model_path,
            exc_info=True,
        )
        return project_only


def preview_tiling_label(tiling: PreviewTiling) -> str:
    """Human-readable origin for the status bar."""
    if tiling.source == "override":
        return "inference-settings override"
    if tiling.source == "training":
        return "training geometry"
    if tiling.source.startswith("profile:"):
        return f"profile '{tiling.source.split(':', 1)[1]}'"
    return "project settings"


def dataset_signature(
    source_path: str,
    model_path: str,
    device: str,
    tiling: PreviewTiling,
) -> tuple[object, ...]:
    """The in-session reuse key for dataset predictions (plan review M3).

    Built from the RESOLVED tiling, so recalibrating the model's sidecar (a new
    primary profile) changes it. A disabled run keeps the historical shape.
    """
    key = (str(source_path), str(model_path)) + InferenceRunSettings(
        device=device, slice_settings=tiling.slice_settings
    ).cache_key()
    if not tiling.slice_settings.enabled:
        return key
    return key + (int(tiling.imgsz), tiling.merge_policy, tiling.merge_metric)
