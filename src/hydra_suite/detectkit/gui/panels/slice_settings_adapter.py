"""DetectKit SAHI state <-> the shared widget's ``(TilingSpec, extras)``.

The widget holds the S1 contract plus role extras; DetectKit persists
``SliceTrainingSettings`` (YOLO training/preview) and the nine SAM3 tiling
kwargs of ``Sam3LoraParams``. These four functions are the only translation
point, so the host-facing contracts (``to_settings()``,
``to_sam3_tiling()``) are unchanged by the widget swap.

Reads are lenient the way the old spin boxes were: an out-of-range saved
value is clamped into the widget's (= the contract's) range rather than
raising inside a GUI load.
"""

from __future__ import annotations

from dataclasses import replace
from statistics import median
from typing import Any

from hydra_suite.utils.tiling_spec import GEOMETRY_MODES, OVERLAP_MAX, TilingSpec
from hydra_suite.widgets.slice_settings_parts import SLICE_SIZE_MAX

from ..models import SliceTrainingSettings

# The nine kwargs SliceSettingsGroup.to_sam3_tiling() has always returned.
SAM3_TILING_KEYS = (
    "geometry_mode",
    "object_tile_fraction",
    "object_tile_fractions",
    "full_frame_mix",
    "slice_width",
    "slice_height",
    "tile_overlap",
    "keep_empty_tiles",
    "min_area_ratio",
)


def _clamp(value: Any, lo: float, hi: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return lo
    if number != number:  # NaN
        return lo
    return max(lo, min(hi, number))


def _size(value: Any) -> int:
    return int(_clamp(value, 0, SLICE_SIZE_MAX))


def _mode(value: Any) -> str:
    # The pre-S4 combo fell back to its first item, "auto_object".
    return value if value in GEOMETRY_MODES else "auto_object"


def _fractions(values) -> tuple[float, ...]:
    out = []
    for value in values or ():
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if 0.0 < number <= 1.0:
            out.append(number)
    return tuple(out)


# -- YOLO training / preview ------------------------------------------------


def settings_to_spec(s: SliceTrainingSettings) -> tuple[TilingSpec, dict]:
    """``SliceTrainingSettings`` -> ``(TilingSpec, train_yolo extras)``."""
    spec = replace(
        TilingSpec.defaults("yolo_train"),
        enabled=bool(s.enabled),
        geometry_mode=_mode(s.geometry_mode),
        object_tile_fractions=_fractions(s.target_fractions()),
        reference_body_px=_clamp(s.reference_body_px, 0.0, float("inf")),
        slice_width=_size(s.slice_width),
        slice_height=_size(s.slice_height),
        overlap=_clamp(s.overlap, 0.0, OVERLAP_MAX),
        min_area_ratio=_clamp(s.min_area_ratio, 0.0, 1.0),
        merge_threshold=_clamp(s.merge_threshold, 0.0, 1.0),
    )
    extras = {
        "negative_tile_fraction": _clamp(s.negative_tile_fraction, 0.0, 1.0),
        "full_frame_mix": bool(s.full_frame_mix),
        "balance_multiscale_loss": bool(s.balance_multiscale_loss),
        "balance_multiscale_loss_power": _clamp(
            s.balance_multiscale_loss_power, 0.0, 1.0
        ),
        "min_area_ratio": spec.min_area_ratio,
    }
    return spec, extras


def spec_to_settings(
    spec: TilingSpec, extras: dict, base: SliceTrainingSettings | None = None
) -> SliceTrainingSettings:
    """``(TilingSpec, extras)`` -> ``SliceTrainingSettings`` (fractions only, F1).

    ``target_sizes`` keeps ``base``'s value (by default the dataclass default,
    exactly as before S4) and is ignored whenever fractions are present.
    """
    base = base or SliceTrainingSettings()
    fractions = list(spec.object_tile_fractions) or list(
        SliceTrainingSettings().target_fractions()
    )
    defaults = SliceTrainingSettings()
    return replace(
        base,
        enabled=bool(spec.enabled),
        geometry_mode=spec.geometry_mode,
        object_tile_fraction=float(median(fractions)),
        # Output metadata from the previous build, not a user override:
        # dataset preparation always remeasures labels.
        reference_body_px=float(spec.reference_body_px),
        slice_width=int(spec.slice_width),
        slice_height=int(spec.slice_height),
        overlap=float(spec.overlap if spec.overlap is not None else defaults.overlap),
        min_area_ratio=float(extras.get("min_area_ratio", spec.min_area_ratio)),
        negative_tile_fraction=float(
            extras.get("negative_tile_fraction", defaults.negative_tile_fraction)
        ),
        target_size_fractions=fractions,
        full_frame_mix=bool(extras.get("full_frame_mix", defaults.full_frame_mix)),
        merge_threshold=float(spec.merge_threshold),
        balance_multiscale_loss=bool(
            extras.get("balance_multiscale_loss", defaults.balance_multiscale_loss)
        ),
        balance_multiscale_loss_power=float(
            extras.get(
                "balance_multiscale_loss_power",
                defaults.balance_multiscale_loss_power,
            )
        ),
    )


# -- SAM3 LoRA tiling -------------------------------------------------------


def sam3_tiling_to_spec(tiling: dict) -> tuple[TilingSpec, dict]:
    """The nine SAM3 tiling kwargs -> ``(TilingSpec, train_sam3 extras)``.

    ``tile_overlap`` rides in the extras at full precision: SAM3's own
    contract is [0, 1) while the shared spec stops at OVERLAP_MAX (decision
    27), so a saved 0.95 must not be clamped on its way through.
    """
    overlap = _clamp(tiling.get("tile_overlap", 0.0), 0.0, 0.99)
    min_area = _clamp(tiling.get("min_area_ratio", 0.0), 0.0, 1.0)
    spec = replace(
        TilingSpec.defaults("sam3"),
        enabled=True,
        geometry_mode=_mode(tiling.get("geometry_mode")),
        object_tile_fractions=_fractions(tiling.get("object_tile_fractions")),
        slice_width=_size(tiling.get("slice_width", 0)),
        slice_height=_size(tiling.get("slice_height", 0)),
        overlap=min(overlap, OVERLAP_MAX),
        min_area_ratio=min_area,
    )
    extras = {
        "object_tile_fraction": _clamp(
            tiling.get("object_tile_fraction", 0.0), 0.0, 1.0
        ),
        "keep_empty_tiles": bool(tiling.get("keep_empty_tiles", False)),
        "full_frame_mix": bool(tiling.get("full_frame_mix", False)),
        "min_area_ratio": min_area,
        "tile_overlap": overlap,
    }
    return spec, extras


def spec_to_sam3_tiling(spec: TilingSpec, extras: dict) -> dict:
    """``(TilingSpec, extras)`` -> exactly the nine ``Sam3LoraParams`` kwargs.

    Deliberately emits NO pixel list: ``resolve_scales`` takes no pixel
    parameter, so SAM3 consumes only fractions (a 640-anchored list would be
    a silent 1.575x shift against SAM3's 1008px input).
    """
    overlap = extras.get("tile_overlap")
    return {
        "geometry_mode": spec.geometry_mode,
        "object_tile_fraction": float(extras.get("object_tile_fraction", 0.0)),
        "object_tile_fractions": tuple(spec.object_tile_fractions),
        "full_frame_mix": bool(extras.get("full_frame_mix", False)),
        "slice_width": int(spec.slice_width),
        "slice_height": int(spec.slice_height),
        "tile_overlap": float(spec.overlap if overlap is None else overlap),
        "keep_empty_tiles": bool(extras.get("keep_empty_tiles", False)),
        "min_area_ratio": float(extras.get("min_area_ratio", spec.min_area_ratio)),
    }
