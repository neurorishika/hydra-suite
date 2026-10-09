"""Unified v3 tiling sidecar: geometry builders and one reader for every family.

Spec: docs/superpowers/specs/2026-10-09-sahi-unification-design.md (§4).
The file is still ``<model>.<ext>.slice_meta.json`` (slice_meta.sidecar_path).
YOLO v3 geometry is ADDITIVE over v2 (every v2 key verbatim) so the v2
reader, the baseline drift guard and the calibration grid are unchanged
until S3 moves them onto ``read_tiling_meta``. SAM3 v3 is a new file and is
canonical-only. Builders stamp only what the input states -- never defaults.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any

from hydra_suite.utils.slice_geometry import LEGACY_TARGET_SIZE_IMGSZ
from hydra_suite.utils.tiling_spec import (
    FRACTION_MAX,
    FRACTION_MIN,
    TilingSpec,
    canonicalize,
    operating_fraction,
)

from .geometry_drift import stamped_object_tile_fraction, stamped_tile_px_set
from .slice_meta import available_slice_profiles, read_slice_meta, training_geometry

logger = logging.getLogger(__name__)

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
_SPEC_FIELD_NAMES = frozenset(f.name for f in fields(TilingSpec))
_SAM3_META_EXTRAS = (
    "full_frame_mix",
    "scale_range_px",
    "scale_grouped_batching",
    "augmentation",
)
_GEOMETRY_ONLY_KEYS = (
    "imgsz",
    "tile_px_set",
    "train_tile_px",
    "train_tile_px_set",
    "prefill_train_tile_px",
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


def _states_a_fraction(canonical: dict[str, Any], source: dict[str, Any]) -> bool:
    """Did the document STATE a fraction (a real set/scalar or a positive prefill)?

    canonicalize's operating ladder ends in a bare-scalar fallback (None ->
    0.15, 0 -> 0.01, 1.5 -> 0.9) that mirrors TrackerKit's YOLO reader. For
    SAM3 that fallback is not a measurement and must never be stamped/read.
    """
    if canonical.get("object_tile_fractions"):
        return True
    prefill = _real(source.get("prefill_object_tile_fraction"))
    return prefill is not None and prefill > 0


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
    operating = None
    if _states_a_fraction(canonical, manifest):
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


def sam3_meta_path(model_path: str | Path) -> Path:
    """``<artifact>.sam3_meta.json`` (append-style, as publish_worker writes it)."""
    path = Path(model_path)
    return path.with_name(path.name + ".sam3_meta.json")


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _imgsz(raw: Any) -> int:
    value = _real(raw)
    return int(value) if value is not None and value >= 1 else 0


@dataclass(frozen=True)
class TilingMeta:
    """A model's SAHI training geometry and calibration profiles, any family."""

    model_family: str
    source: str  # "slice_meta" | "sam3_meta"
    training: TilingSpec | None
    imgsz: int
    tile_px_set: tuple[tuple[int, int], ...]
    operating_fraction: float | None
    extras: dict[str, Any] = field(default_factory=dict)
    primary_profile_id: str = ""
    profiles: tuple[dict[str, Any], ...] = ()


def read_tiling_meta(model_path: str | Path) -> TilingMeta | None:
    """The ONE reader for a model's SAHI geometry and calibration profiles. Never raises."""
    try:
        return _read_tiling_meta(model_path)
    except Exception:
        logger.warning("Unreadable SAHI metadata beside %s", model_path, exc_info=True)
        return None


def _family_base(family: str) -> TilingSpec:
    """Family defaults that never claim a fraction the document did not state."""
    if family == "sam3":
        return replace(TilingSpec.defaults("sam3"), object_tile_fractions=())
    # yolo_infer, with _training_values' auto_object fallback for the mode.
    return replace(
        TilingSpec.defaults("yolo_infer"),
        geometry_mode="auto_object",
        object_tile_fractions=(),
    )


def _stamped_tiles(geometry: dict[str, Any]) -> tuple[tuple[int, int], ...]:
    if geometry.get("tile_px_set") is not None:
        return tuple(_tile_pairs(geometry["tile_px_set"]))
    try:
        stamped = stamped_tile_px_set(geometry) or ()
    except Exception:  # e.g. OverflowError on a huge int: no usable claim
        stamped = ()
    return tuple(_tile_pairs([list(pair) for pair in stamped]))


def _read_tiling_meta(model_path: str | Path) -> TilingMeta | None:
    slice_doc = read_slice_meta(model_path) or {}
    geometry = training_geometry(slice_doc) if slice_doc else {}
    family = slice_doc.get("model_family") or "yolo"
    source = "slice_meta"
    if not geometry:
        sam3_doc = _read_json(sam3_meta_path(model_path))
        if sam3_doc:
            geometry, family, source = sam3_doc, "sam3", "sam3_meta"
    if not geometry and not slice_doc:
        return None
    if not isinstance(family, str) or family not in MODEL_FAMILIES:
        logger.warning(
            "Unknown SAHI model_family %r beside %s; reading as yolo",
            family,
            model_path,
        )
        family = "yolo"

    training: TilingSpec | None = None
    operating: float | None = None
    extras: dict[str, Any] = {}
    tiles: tuple[tuple[int, int], ...] = ()
    imgsz = 0
    if geometry:
        legacy = LEGACY_TARGET_SIZE_IMGSZ if family == "yolo" else None
        canonical, extras = canonicalize(geometry, legacy_px_imgsz=legacy)
        try:
            training = replace(
                _family_base(family),
                **{k: v for k, v in canonical.items() if k in _SPEC_FIELD_NAMES},
            )
        except Exception:
            logger.warning("Invalid SAHI geometry beside %s", model_path, exc_info=True)
            training = None
        if training is not None:
            training = replace(training, enabled=True)
            # YOLO mirrors _training_values (incl. its 0.15 fallback); SAM3
            # reads a fraction only when the document states one.
            if family == "yolo" or _states_a_fraction(canonical, geometry):
                operating = canonical.get("operating_fraction")
                if operating is None:
                    operating = training.operating_fraction()
                else:
                    operating = max(FRACTION_MIN, min(FRACTION_MAX, float(operating)))
        imgsz = _imgsz(geometry.get("imgsz"))
        tiles = _stamped_tiles(geometry)
        if source == "sam3_meta":
            extras = {k: extras[k] for k in _SAM3_META_EXTRAS if k in extras}
        for key in _GEOMETRY_ONLY_KEYS:
            extras.pop(key, None)

    return TilingMeta(
        model_family=family,
        source=source,
        training=training,
        imgsz=imgsz,
        tile_px_set=tiles,
        operating_fraction=operating,
        extras=extras,
        primary_profile_id=str(slice_doc.get("primary_profile_id", "") or ""),
        profiles=tuple(available_slice_profiles(slice_doc)),
    )
