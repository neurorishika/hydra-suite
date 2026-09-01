"""Read direct-detector SAHI metadata and calibrated profile sidecars.

The original flat ``<model>.slice_meta.json`` payload carries sliced-training
geometry. Version 2 adds user-approved operating profiles without turning one
checkpoint into several registry models. This module stays pure so both kits
can consume it without importing each other.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

SLICE_META_SCHEMA_VERSION = 2
_GEOMETRY_MODES = {"auto_model", "auto_object", "custom"}


def sidecar_path(model_path: str | Path) -> Path:
    """Return the canonical append-style metadata sidecar path."""
    path = Path(model_path)
    return path.with_suffix(path.suffix + ".slice_meta.json")


def read_slice_meta(model_path: str | Path) -> dict[str, Any] | None:
    """Return parsed sidecar data, or ``None`` for absent/corrupt metadata."""
    try:
        data = json.loads(sidecar_path(model_path).read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def write_slice_meta(model_path: str | Path, meta: dict[str, Any]) -> Path:
    """Atomically write caller-owned metadata beside ``model_path``."""
    target = sidecar_path(model_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    staged = target.with_name(target.name + ".tmp")
    staged.write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    staged.replace(target)
    return target


def training_geometry(meta: dict[str, Any]) -> dict[str, Any]:
    """Return v2 training geometry or a legacy flat payload, without mutation."""
    nested = meta.get("training_geometry")
    return dict(nested) if isinstance(nested, dict) else dict(meta)


def available_slice_profiles(meta: dict[str, Any]) -> list[dict[str, Any]]:
    """Return valid v2 profiles in sidecar order, dropping malformed entries."""
    profiles = meta.get("profiles")
    if not isinstance(profiles, list):
        return []
    valid: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in profiles:
        if not isinstance(raw, dict):
            continue
        profile_id = str(raw.get("id", "")).strip()
        name = str(raw.get("name", "")).strip()
        settings = raw.get("settings")
        if (
            not profile_id
            or not name
            or profile_id in seen
            or not isinstance(settings, dict)
        ):
            continue
        seen.add(profile_id)
        valid.append(
            {
                "id": profile_id,
                "name": name,
                "note": str(raw.get("note", "") or ""),
                "settings": dict(settings),
                "measurement": dict(raw.get("measurement") or {}),
            }
        )
    return valid


def primary_slice_profile(meta: dict[str, Any]) -> dict[str, Any] | None:
    """Return the explicitly chosen primary profile, if it is still valid."""
    primary_id = str(meta.get("primary_profile_id", "") or "")
    return next(
        (
            profile
            for profile in available_slice_profiles(meta)
            if profile["id"] == primary_id
        ),
        None,
    )


def profile_by_id(
    meta: dict[str, Any], profile_id: str | None
) -> dict[str, Any] | None:
    """Resolve a profile, falling back only to the explicit primary profile."""
    if profile_id:
        profile = next(
            (
                item
                for item in available_slice_profiles(meta)
                if item["id"] == str(profile_id)
            ),
            None,
        )
        if profile is not None:
            return profile
    return primary_slice_profile(meta)


def _clamped_float(value: object, default: float, lo: float, hi: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, parsed)) if np.isfinite(parsed) else default


def _clamped_int(value: object, default: int, lo: int, hi: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, parsed))


def _training_values(geometry: dict[str, Any]) -> dict[str, Any]:
    stored_fraction = _clamped_float(
        geometry.get("object_tile_fraction"), 0.15, 0.01, 0.9
    )
    targets: list[float] = []
    for target in geometry.get("target_sizes") or []:
        try:
            targets.append(float(target))
        except (TypeError, ValueError):
            pass
    imgsz = _clamped_int(geometry.get("imgsz"), 0, 0, 8192)
    if targets and imgsz > 0:
        stored_fraction = max(
            0.01, min(0.9, float(np.median(np.asarray(targets))) / imgsz)
        )
    mode = str(geometry.get("geometry_mode", "auto_object") or "auto_object")
    return {
        "enabled": True,
        "geometry_mode": mode if mode in _GEOMETRY_MODES else "auto_object",
        "overlap": _clamped_float(geometry.get("overlap"), 0.2, 0.0, 0.9),
        "object_tile_fraction": stored_fraction,
        "trained_body_px": _clamped_float(
            geometry.get("reference_body_px"), 0.0, 0.0, 8192.0
        ),
        "slice_width": _clamped_int(geometry.get("slice_width"), 0, 0, 8192),
        "slice_height": _clamped_int(geometry.get("slice_height"), 0, 0, 8192),
        "confidence_threshold": None,
        "merge_policy": None,
        "merge_metric": None,
        "merge_threshold": None,
        "merge_backend": None,
    }


def slice_meta_to_panel_values(
    meta: dict[str, Any], profile_id: str | None = None
) -> dict[str, Any]:
    """Translate metadata (and an optional profile) into safe TrackerKit values."""
    values = _training_values(training_geometry(meta))
    profile = profile_by_id(meta, profile_id)
    if profile is None:
        values["profile_id"] = None
        values["profile_name"] = "Training geometry"
        return values

    settings = profile["settings"]
    values.update(
        {
            "enabled": bool(settings.get("enabled", True)),
            "geometry_mode": str(
                settings.get("geometry_mode", values["geometry_mode"])
            ),
            "overlap": _clamped_float(
                settings.get("overlap"), values["overlap"], 0.0, 0.9
            ),
            "object_tile_fraction": _clamped_float(
                settings.get("object_tile_fraction"),
                values["object_tile_fraction"],
                0.01,
                0.9,
            ),
            "trained_body_px": _clamped_float(
                settings.get("trained_body_px"), values["trained_body_px"], 0.0, 8192.0
            ),
            "slice_width": _clamped_int(
                settings.get("slice_width"), values["slice_width"], 0, 8192
            ),
            "slice_height": _clamped_int(
                settings.get("slice_height"), values["slice_height"], 0, 8192
            ),
            "confidence_threshold": settings.get("confidence_threshold"),
            "merge_policy": settings.get("merge_policy"),
            "merge_metric": settings.get("merge_metric"),
            "merge_threshold": settings.get("merge_threshold"),
            "merge_backend": settings.get("merge_backend"),
            "profile_id": profile["id"],
            "profile_name": profile["name"],
        }
    )
    if values["geometry_mode"] not in _GEOMETRY_MODES:
        values["geometry_mode"] = "auto_object"
    return values
