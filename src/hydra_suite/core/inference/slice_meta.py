"""Read direct-detector SAHI metadata and calibrated profile sidecars.

The original flat ``<model>.slice_meta.json`` payload carries sliced-training
geometry. Version 2 adds user-approved operating profiles without turning one
checkpoint into several registry models. This module stays pure so both kits
can consume it without importing each other.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

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


def new_profile_id(name: str) -> str:
    """Create a stable-looking, collision-resistant user-visible profile id."""
    slug = "-".join(
        part
        for part in "".join(ch.lower() if ch.isalnum() else " " for ch in name).split()
        if part
    )
    return f"{slug[:32] or 'profile'}-{uuid4().hex[:8]}"


def normalized_slice_meta(meta: dict[str, Any]) -> dict[str, Any]:
    """Promote legacy metadata to the v2 document shape without inventing profiles."""
    return {
        "schema_version": SLICE_META_SCHEMA_VERSION,
        "training_geometry": training_geometry(meta),
        "primary_profile_id": str(meta.get("primary_profile_id", "") or ""),
        "profiles": available_slice_profiles(meta),
    }


def upsert_slice_profile(
    meta: dict[str, Any],
    *,
    name: str,
    settings: dict[str, Any],
    profile_id: str | None = None,
    note: str = "",
    measurement: dict[str, Any] | None = None,
    primary: bool = False,
) -> dict[str, Any]:
    """Return v2 metadata with one explicitly user-approved profile upserted.

    The caller owns persistence. This pure helper intentionally preserves every
    other profile and never promotes a generated training default to calibration.
    """
    clean_name = str(name or "").strip()
    if not clean_name:
        raise ValueError("A SAHI profile needs a name.")
    result = normalized_slice_meta(meta)
    profiles = list(result["profiles"])
    selected_id = str(profile_id or new_profile_id(clean_name)).strip()
    if not selected_id:
        raise ValueError("A SAHI profile needs an id.")
    if any(
        p["name"].casefold() == clean_name.casefold() and p["id"] != selected_id
        for p in profiles
    ):
        raise ValueError(f"A SAHI profile named {clean_name!r} already exists.")
    profile = {
        "id": selected_id,
        "name": clean_name,
        "note": str(note or ""),
        "settings": dict(settings),
        "measurement": dict(measurement or {}),
    }
    replacement = next(
        (i for i, old in enumerate(profiles) if old["id"] == selected_id), None
    )
    if replacement is None:
        profiles.append(profile)
    else:
        profiles[replacement] = profile
    result["profiles"] = profiles
    # Primary is a user designation, never a side effect of saving the first
    # profile: an inferred default is exactly what the spec forbids.
    if primary:
        result["primary_profile_id"] = selected_id
    return result


def remove_slice_profile(
    meta: dict[str, Any],
    profile_id: str,
    *,
    new_primary_id: str | None = None,
) -> dict[str, Any]:
    """Remove one profile, preserving weights, geometry and every other profile.

    Removing the PRIMARY requires an explicit decision: ``new_primary_id=""``
    clears the designation, an id promotes that profile. Silently promoting
    whatever happened to be first would hand the user an operating point they
    never chose.
    """
    result = normalized_slice_meta(meta)
    target = str(profile_id)
    remaining = [p for p in result["profiles"] if p["id"] != target]
    if len(remaining) == len(result["profiles"]):
        return result
    if result["primary_profile_id"] == target:
        if new_primary_id is None:
            raise ValueError(
                "Removing the primary SAHI profile needs a replacement "
                "(pass new_primary_id) or an explicit clear (new_primary_id='')."
            )
        chosen = str(new_primary_id)
        if chosen and chosen not in {p["id"] for p in remaining}:
            raise ValueError(f"Unknown replacement primary profile {chosen!r}.")
        result["primary_profile_id"] = chosen
    result["profiles"] = remaining
    return result


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


def profile_summary(meta: dict[str, Any]) -> dict[str, Any]:
    """Inventory summary the registry stores; the sidecar stays the source of truth."""
    profiles = available_slice_profiles(meta)
    return {
        "count": len(profiles),
        "primary_profile_id": str(meta.get("primary_profile_id", "") or ""),
        "names": [profile["name"] for profile in profiles],
    }


def merge_training_geometry(
    existing: dict[str, Any] | None, training_geometry: dict[str, Any]
) -> dict[str, Any]:
    """Replace training geometry while preserving user-approved profiles.

    Publishing must never destroy calibration a user did before registering.
    """
    result = normalized_slice_meta(existing or {})
    result["training_geometry"] = dict(training_geometry)
    return result


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


def profile_evidence_state(
    profile: dict[str, Any], *, checkpoint_path: str | Path
) -> tuple[bool, str]:
    """Is this profile's measured evidence still about THESE weights?

    Missing or unreadable provenance is not fatal -- an imported or legacy
    profile simply has nothing to contradict. A fingerprint that DISAGREES is:
    applying settings measured on other weights silently misdescribes the
    operating point.
    """
    recorded = str((profile.get("measurement") or {}).get("checkpoint_fingerprint", ""))
    if not recorded:
        return True, ""
    try:
        digest = hashlib.sha256(Path(checkpoint_path).read_bytes()).hexdigest()
    except Exception:
        return True, ""
    if recorded.split(":")[-1] == digest:
        return True, ""
    return False, (
        f"'{profile.get('name', 'profile')}' was measured before the model's "
        "weights changed; its numbers no longer describe this checkpoint."
    )


def profile_by_id(
    meta: dict[str, Any], profile_id: str | None
) -> dict[str, Any] | None:
    """Resolve a profile, falling back only to the explicit primary profile."""
    if profile_id == "__training__":
        return None
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
    """Translate metadata (and an optional profile) into safe TrackerKit values.

    ``values["resolution"]`` explains WHY the returned profile is the one it
    is, so callers never have to infer a fallback by comparing ids:
    ``"requested"`` (the asked-for profile id was found), ``"primary"``
    (the requested id -- if any -- was not found, so the sidecar's explicit
    primary profile was used instead), or ``"training"`` (no profile applies
    at all -- either ``"__training__"`` was requested, or nothing resolved).
    """
    values = _training_values(training_geometry(meta))
    profile = profile_by_id(meta, profile_id)
    if profile is None:
        values["profile_id"] = None
        values["profile_name"] = "Training geometry"
        values["resolution"] = "training"
        return values

    if profile_id and str(profile_id) == str(profile.get("id", "")):
        values["resolution"] = "requested"
    else:
        values["resolution"] = "primary"

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


def slice_meta_values_from_settings(
    meta: dict[str, Any], settings: dict[str, Any]
) -> dict[str, Any]:
    """Translate an explicit effective-settings snapshot into panel values.

    Used to restore a saved session's SAHI configuration when the profile
    that produced it is no longer present in the sidecar: the settings
    themselves are the source of truth, not the (now-missing) profile's
    identity. ``values["resolution"]`` is always ``"saved_settings"``.
    """
    values = _training_values(training_geometry(meta))
    values.update(
        {
            "enabled": bool(settings.get("enabled", values["enabled"])),
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
            "profile_id": None,
            "profile_name": "Saved settings",
        }
    )
    if values["geometry_mode"] not in _GEOMETRY_MODES:
        values["geometry_mode"] = "auto_object"
    values["resolution"] = "saved_settings"
    return values
