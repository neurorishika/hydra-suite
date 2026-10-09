"""Unified v3 tiling sidecar: geometry builders and one reader for every family.

Spec: docs/superpowers/specs/2026-10-09-sahi-unification-design.md (§4).
The file is still ``<model>.<ext>.slice_meta.json`` (slice_meta.sidecar_path).
YOLO v3 geometry is ADDITIVE over v2 (every v2 key verbatim) so the v2
reader, the baseline drift guard and the calibration grid are unchanged
until S3 moves them onto ``read_tiling_meta``. SAM3 v3 is a new file and is
canonical-only. Builders stamp only what the input states -- never defaults.
"""

from __future__ import annotations

import math
from typing import Any

from hydra_suite.utils.slice_geometry import LEGACY_TARGET_SIZE_IMGSZ
from hydra_suite.utils.tiling_spec import canonicalize, operating_fraction

from .geometry_drift import stamped_object_tile_fraction

MODEL_FAMILIES = ("yolo", "sam3")
# SAM3 build manifests carry build bookkeeping; only these survive into the stamp.
_SAM3_EXTRAS = ("full_frame_mix", "scale_range_px", "keep_empty_tiles")
_SAM3_PRESENT_ONLY = (
    "geometry_mode",
    "reference_body_px",
    "slice_width",
    "slice_height",
    "overlap",
    "min_area_ratio",
)


def _real(value: Any) -> float | None:
    """A finite real number (never bool or str) as a float, else None."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        parsed = float(value)
    except (OverflowError, ValueError):  # float(10**400) overflows
        return None
    return parsed if math.isfinite(parsed) else None


def _tile_pairs(raw: Any) -> list[tuple[int, int]]:
    """Parse a tile size / tile-size set; skip anything not a positive finite size."""
    items = list(raw) if isinstance(raw, (list, tuple)) else [raw]
    if len(items) == 2 and all(_real(v) is not None for v in items):
        items = [items]  # a bare [w, h] pair
    pairs: list[tuple[int, int]] = []
    for item in items:
        if isinstance(item, (list, tuple)):
            if len(item) != 2:
                continue
            w, h = _real(item[0]), _real(item[1])
        else:
            w = h = _real(item)
        if w is None or h is None or w < 1 or h < 1:
            continue
        pairs.append((int(w), int(h)))
    return pairs


def _v2_fraction_claim(geometry: dict[str, Any]) -> float | None:
    """What the v2 readers (calibration grid, prefill) read as the fraction."""
    try:
        return stamped_object_tile_fraction(geometry)
    except Exception:  # e.g. OverflowError on a huge int; treat as no claim
        return None


def training_geometry_from_yolo_manifest(
    slice_geometry: dict[str, Any],
) -> dict[str, Any]:
    """v3 block for a YOLO sliced build: the v2 manifest verbatim + canonical keys."""
    geometry = dict(slice_geometry or {})
    # Legacy YOLO pixel target_sizes without imgsz were expressed at 640.
    canonical, _ = canonicalize(geometry, legacy_px_imgsz=LEGACY_TARGET_SIZE_IMGSZ)
    fractions = list(canonical.get("object_tile_fractions") or ())
    if fractions:
        geometry.setdefault("object_tile_fractions", fractions)
    operating = canonical.get("operating_fraction")
    # stamped_object_tile_fraction (calibration grid, prefill) reads the bare
    # scalar first and falls back to the prefill. Adding a prefill where the
    # manifest has no positive bare scalar would change what that v2 reader
    # sees -- and would write the reader's 0.15 fallback to disk as if stated.
    if operating is not None and _v2_fraction_claim(slice_geometry or {}) is not None:
        geometry.setdefault("prefill_object_tile_fraction", float(operating))
    if "reference_body_px" in canonical:
        geometry.setdefault("trained_body_px", float(canonical["reference_body_px"]))
    geometry.setdefault("fragment_policy", "drop")
    return geometry


def training_geometry_from_sam3_manifest(
    build_manifest: dict[str, Any], *, imgsz: int
) -> dict[str, Any]:
    """v3 block for a SAM3 tile build: canonical names, present values only."""
    manifest = build_manifest if isinstance(build_manifest, dict) else {}
    # A build manifest's tile_px is a realized MEASUREMENT, not the escalation
    # override canonicalize aliases to custom slice_width/height (S1 NEW-3);
    # it is mapped to tile_px_set below instead.
    canonical, extras = canonicalize(
        {k: v for k, v in manifest.items() if k != "tile_px"}
    )
    geometry: dict[str, Any] = {"fragment_policy": "crowd", "imgsz": int(imgsz)}
    for key in _SAM3_PRESENT_ONLY:
        if key in canonical:
            geometry[key] = canonical[key]
    if "reference_body_px" in canonical:
        geometry["trained_body_px"] = float(canonical["reference_body_px"])
    fractions = list(canonical.get("object_tile_fractions") or ())
    if fractions:
        geometry["object_tile_fractions"] = fractions
    operating = canonical.get("operating_fraction")
    if operating is None:
        operating = operating_fraction(fractions)
    if operating is not None:
        geometry["prefill_object_tile_fraction"] = float(operating)
        # A median under a measurement's name reads as "the" training tile
        # size downstream; multi-scale stamps it only as the named prefill.
        if len(fractions) <= 1:
            geometry["object_tile_fraction"] = float(operating)
    raw_set = manifest.get("tile_px_set")
    if not raw_set and manifest.get("tile_px") is not None:
        raw_set = manifest["tile_px"]
    tiles = _tile_pairs(raw_set) if raw_set is not None else []
    if tiles:
        geometry["tile_px_set"] = [[w, h] for w, h in tiles]
    geometry.update({k: extras[k] for k in _SAM3_EXTRAS if k in extras})
    return geometry
