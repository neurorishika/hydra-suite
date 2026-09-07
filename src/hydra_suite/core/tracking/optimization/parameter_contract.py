"""Shared, Qt-free contract for tracking auto-tuner parameters.

The optimiser evaluates a proposal before TrackerKit writes it into a
``QDoubleSpinBox``.  Those widgets deliberately expose different decimal
precisions, so evaluating an arbitrary Optuna float would otherwise evaluate a
different configuration from the one production receives.  This module keeps
the search bounds, storage units, and UI precision in one lower-layer contract
that both the core optimiser and TrackerKit can use without a core-to-GUI
dependency.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, is_dataclass
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any, Literal

import numpy as np

SearchKind = Literal["float", "log_float", "int"]


@dataclass(frozen=True)
class TrackingAutotuneParameter:
    """One search dimension and its production/UI representation."""

    key: str
    search_kind: SearchKind
    minimum: float
    maximum: float
    widget_decimals: int
    stored_as_frames: bool = False


# Keep this tuple ordered: it drives deterministic candidate serialization as
# well as all core proposal-generation paths.
TRACKING_AUTOTUNE_PARAMETERS = (
    TrackingAutotuneParameter("YOLO_CONFIDENCE_THRESHOLD", "float", 0.01, 1.0, 2),
    TrackingAutotuneParameter("YOLO_IOU_THRESHOLD", "float", 0.01, 1.0, 2),
    TrackingAutotuneParameter("MAX_DISTANCE_MULTIPLIER", "float", 0.1, 20.0, 2),
    TrackingAutotuneParameter("KALMAN_NOISE_COVARIANCE", "log_float", 0.0001, 1.0, 4),
    TrackingAutotuneParameter(
        "KALMAN_MEASUREMENT_NOISE_COVARIANCE", "log_float", 0.0001, 1.0, 4
    ),
    TrackingAutotuneParameter("W_POSITION", "float", 0.0, 10.0, 2),
    TrackingAutotuneParameter("W_ORIENTATION", "float", 0.0, 10.0, 2),
    TrackingAutotuneParameter("W_AREA", "float", 0.0, 2.0, 4),
    TrackingAutotuneParameter("W_ASPECT", "float", 0.0, 10.0, 2),
    TrackingAutotuneParameter("KALMAN_DAMPING", "float", 0.5, 0.999, 3),
    TrackingAutotuneParameter(
        "KALMAN_LONGITUDINAL_NOISE_MULTIPLIER", "float", 0.1, 20.0, 1
    ),
    TrackingAutotuneParameter(
        "KALMAN_INITIAL_VELOCITY_RETENTION", "float", 0.0, 1.0, 2
    ),
    # These values are integer frame counts in core. TrackerKit displays them
    # in seconds in four-decimal QDoubleSpinBoxes.
    TrackingAutotuneParameter(
        "LOST_THRESHOLD_FRAMES", "int", 2, 25, 4, stored_as_frames=True
    ),
    TrackingAutotuneParameter(
        "KALMAN_MATURITY_AGE", "int", 1, 20, 4, stored_as_frames=True
    ),
)

TRACKING_AUTOTUNE_PARAMETER_SPECS = {
    parameter.key: parameter for parameter in TRACKING_AUTOTUNE_PARAMETERS
}
TRACKING_AUTOTUNE_CANDIDATE_KEYS = tuple(
    parameter.key for parameter in TRACKING_AUTOTUNE_PARAMETERS
)

# ``build_engine_params`` derives these engine-only values from the public
# controls.  Keep the derivation at the candidate/base merge boundary too: a
# candidate is evaluated before TrackerKit writes it into those controls.
_MAX_DISTANCE_MULTIPLIER = "MAX_DISTANCE_MULTIPLIER"
_MAX_DISTANCE_THRESHOLD = "MAX_DISTANCE_THRESHOLD"
_KALMAN_LONGITUDINAL_NOISE = "KALMAN_LONGITUDINAL_NOISE_MULTIPLIER"
_KALMAN_LATERAL_NOISE = "KALMAN_LATERAL_NOISE_MULTIPLIER"
_KALMAN_ANISOTROPY_RATIO = "KALMAN_ANISOTROPY_RATIO"
_KALMAN_LATERAL_NOISE_FLOOR = 1e-6
_KALMAN_MIN_ANISOTROPY_RATIO = 1.0
_DEFAULT_KALMAN_ANISOTROPY_RATIO = 50.0

# Retained as a simple mapping for the core search implementation and external
# callers that have historically imported ``optimizer._PARAM_RANGES``.
PARAM_RANGES = {
    parameter.key: (parameter.search_kind, parameter.minimum, parameter.maximum)
    for parameter in TRACKING_AUTOTUNE_PARAMETERS
}


def tracking_autotune_parameter(key: str) -> TrackingAutotuneParameter:
    """Return the typed contract for *key*, or raise a useful error."""

    try:
        return TRACKING_AUTOTUNE_PARAMETER_SPECS[key]
    except KeyError as exc:
        raise KeyError(f"{key} is not a tracking auto-tuner parameter") from exc


def _quantize_decimal(value: float, decimals: int) -> float:
    """Match ``QDoubleSpinBox``'s fixed-decimal rounding for a float value.

    ``Decimal(value)`` intentionally receives the binary float rather than a
    string. Qt rounds that same binary value to the configured decimal places;
    using a decimal string would disagree at values such as ``0.995``.
    """

    quantum = Decimal(1).scaleb(-decimals)
    return float(Decimal(value).quantize(quantum, rounding=ROUND_HALF_UP))


def quantize_tracking_autotune_value(key: str, value: Any) -> int | float:
    """Return the exact value representable by TrackerKit for one candidate.

    This intentionally does not clamp values to their range. Callers validate
    ranges before applying to widgets, preserving the atomic no-partial-write
    behavior of the GUI contract. Core proposal paths generate values inside
    their search ranges and use this function only to remove unrepresentable
    fractional precision before evaluating them.
    """

    parameter = tracking_autotune_parameter(key)
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} is not numeric") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"{key} is not a finite number")
    if parameter.search_kind == "int":
        if not numeric.is_integer():
            raise ValueError(f"{key} must be an integer frame count")
        return int(numeric)
    return _quantize_decimal(numeric, parameter.widget_decimals)


def quantize_tracking_autotune_params(
    params: Mapping[str, Any],
) -> dict[str, Any]:
    """Canonicalize every recognized candidate value without touching extras."""

    return {
        key: (
            quantize_tracking_autotune_value(key, value)
            if key in TRACKING_AUTOTUNE_PARAMETER_SPECS
            else value
        )
        for key, value in params.items()
    }


def merge_tracking_autotune_candidate(
    base_params: Mapping[str, Any], candidate_params: Mapping[str, Any]
) -> dict[str, Any]:
    """Return engine parameters for a public candidate applied to *base_params*.

    Candidate results intentionally contain only public, applyable controls.
    Engine-only values derived from those controls must be recalculated after
    the merge, rather than carried forward from an earlier proposal or from
    the baseline.  This is particularly important for longitudinal Kalman
    noise: TrackerKit writes only that public control and keeps its hidden
    lateral multiplier unchanged.  Production subsequently derives the
    anisotropy ratio from those two effective values, so evaluation must do
    exactly the same thing.

    The ratio calculation deliberately mirrors ``trackerkit.engine_params``:
    its lateral denominator has a ``1e-6`` floor and its ratio has a ``1.0``
    floor.  The fallback covers hand-built/legacy engine mappings which omit
    the retained hidden lateral key; normal TrackerKit engine params always
    include it.
    """

    canonical_candidate = quantize_tracking_autotune_params(candidate_params)
    merged = dict(base_params)
    # Do not let historical or engine-only result fields influence replay.
    # Applying a selected row writes only this public contract, and a restored
    # candidate must therefore evaluate with the same surface.
    selected = {
        key: canonical_candidate[key]
        for key in TRACKING_AUTOTUNE_CANDIDATE_KEYS
        if key in canonical_candidate
    }
    merged.update(selected)

    if _MAX_DISTANCE_MULTIPLIER in selected:
        merged[_MAX_DISTANCE_THRESHOLD] = (
            float(merged[_MAX_DISTANCE_MULTIPLIER])
            * float(merged.get("REFERENCE_BODY_SIZE", 20.0))
            * float(merged.get("RESIZE_FACTOR", 1.0))
        )

    if _KALMAN_LONGITUDINAL_NOISE in selected:
        longitudinal = float(merged[_KALMAN_LONGITUDINAL_NOISE])
        if _KALMAN_LATERAL_NOISE in merged:
            lateral = float(merged[_KALMAN_LATERAL_NOISE])
        else:
            # An engine mapping without the retained hidden lateral setting
            # still has an effective baseline lateral noise through its ratio.
            baseline_longitudinal = float(
                base_params.get(_KALMAN_LONGITUDINAL_NOISE, longitudinal)
            )
            baseline_ratio = max(
                float(
                    base_params.get(
                        _KALMAN_ANISOTROPY_RATIO,
                        _DEFAULT_KALMAN_ANISOTROPY_RATIO,
                    )
                ),
                _KALMAN_MIN_ANISOTROPY_RATIO,
            )
            lateral = baseline_longitudinal / baseline_ratio
        merged[_KALMAN_ANISOTROPY_RATIO] = max(
            _KALMAN_MIN_ANISOTROPY_RATIO,
            longitudinal / max(lateral, _KALMAN_LATERAL_NOISE_FLOOR),
        )

    return merged


def tracking_autotune_widget_value(
    key: str, value: Any, *, fps: float | None = None
) -> int | float:
    """Convert a canonical core value into the exact TrackerKit widget value."""

    parameter = tracking_autotune_parameter(key)
    canonical = quantize_tracking_autotune_value(key, value)
    if not parameter.stored_as_frames:
        return canonical
    if fps is None or not math.isfinite(float(fps)) or float(fps) <= 0.0:
        raise ValueError("FPS must be a positive finite value")
    return _quantize_decimal(float(canonical) / float(fps), parameter.widget_decimals)


def canonicalize_evaluation_value(value: Any) -> Any:
    """Return a deterministic, JSON-safe snapshot of an evaluation input.

    In particular, ndarray fingerprints include dtype, shape, and contents;
    ``default=str`` would preserve only an unstable/truncated representation and
    allow replay-affecting ROI/arena arrays to reuse stale recommendations.
    """

    if isinstance(value, np.generic):
        return canonicalize_evaluation_value(value.item())
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            return {
                "__type__": "ndarray-object",
                "dtype": value.dtype.str,
                "shape": list(value.shape),
                "values": canonicalize_evaluation_value(value.tolist()),
            }
        contiguous = np.ascontiguousarray(value)
        return {
            "__type__": "ndarray",
            "dtype": contiguous.dtype.str,
            "shape": list(contiguous.shape),
            "sha256": hashlib.sha256(contiguous.tobytes()).hexdigest(),
        }
    if isinstance(value, Mapping):
        return {
            str(key): canonicalize_evaluation_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [canonicalize_evaluation_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        members = [canonicalize_evaluation_value(item) for item in value]
        return sorted(
            members,
            key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")),
        )
    if isinstance(value, Path):
        return {"__type__": "path", "value": str(value)}
    if isinstance(value, bytes):
        return {
            "__type__": "bytes",
            "sha256": hashlib.sha256(value).hexdigest(),
            "size": len(value),
        }
    if isinstance(value, float):
        if math.isnan(value):
            return {"__type__": "float", "value": "nan"}
        if math.isinf(value):
            return {"__type__": "float", "value": "inf" if value > 0 else "-inf"}
        return value
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "__type__": f"{type(value).__module__}.{type(value).__qualname__}",
            "values": canonicalize_evaluation_value(vars(value)),
        }
    if value is None or isinstance(value, (bool, int, str)):
        return value
    # Runtime objects are not expected in TrackerKit's parameter dict, but a
    # stable type/value fallback is safer than ``default=str`` silently hashing
    # a process-specific object address.
    return {
        "__type__": f"{type(value).__module__}.{type(value).__qualname__}",
        "value": str(value),
    }


def canonical_evaluation_params(params: Mapping[str, Any]) -> dict[str, Any]:
    """Snapshot the complete fixed-and-tunable production replay contract."""

    canonical = canonicalize_evaluation_value(params)
    if not isinstance(canonical, dict):  # defensive: ``params`` is a Mapping
        raise TypeError("evaluation parameters must canonicalize to a mapping")
    return canonical
