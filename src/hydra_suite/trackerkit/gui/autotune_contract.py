"""GUI contract for applying tracking auto-tuner candidates.

The core optimizer owns its complete search space. This module deliberately
lists only the candidate fields that TrackerKit can present and write back to
the current UI. It keeps frame-based core values separate from the seconds-
based controls exposed to users.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from hydra_suite.core.tracking.optimization.parameter_contract import (
    TRACKING_AUTOTUNE_CANDIDATE_KEYS,
    quantize_tracking_autotune_params,
    tracking_autotune_widget_value,
)

_DIRECT_WIDGETS = {
    "YOLO_CONFIDENCE_THRESHOLD": ("detection", "spin_yolo_confidence"),
    "YOLO_IOU_THRESHOLD": ("detection", "spin_yolo_iou"),
    "MAX_DISTANCE_MULTIPLIER": ("tracking", "spin_max_dist"),
    "W_POSITION": ("tracking", "spin_Wp"),
    "W_ORIENTATION": ("tracking", "spin_Wo"),
    "W_AREA": ("tracking", "spin_Wa"),
    "W_ASPECT": ("tracking", "spin_Wasp"),
    "KALMAN_NOISE_COVARIANCE": ("tracking", "spin_kalman_noise"),
    "KALMAN_MEASUREMENT_NOISE_COVARIANCE": ("tracking", "spin_kalman_meas"),
    "KALMAN_DAMPING": ("tracking", "spin_kalman_damping"),
    "KALMAN_LONGITUDINAL_NOISE_MULTIPLIER": (
        "tracking",
        "spin_kalman_longitudinal_noise",
    ),
    "KALMAN_INITIAL_VELOCITY_RETENTION": (
        "tracking",
        "spin_kalman_initial_velocity_retention",
    ),
}

_FRAME_WIDGETS = {
    "KALMAN_MATURITY_AGE": ("tracking", "spin_kalman_maturity_age"),
    "LOST_THRESHOLD_FRAMES": ("tracking", "spin_lost_thresh"),
}


class AutotuneCandidateApplicationError(ValueError):
    """A selected candidate cannot be represented by the active UI controls."""


def applicable_candidate_params(params: Mapping[str, Any]) -> dict[str, Any]:
    """Return only candidate values that TrackerKit can apply faithfully."""
    return {
        key: params[key] for key in TRACKING_AUTOTUNE_CANDIDATE_KEYS if key in params
    }


def _valid_fps(panels: Any) -> float:
    fps = float(panels.setup.spin_fps.value())
    if not math.isfinite(fps) or fps <= 0.0:
        raise AutotuneCandidateApplicationError(
            "Cannot apply the selected candidate because FPS must be a positive finite value."
        )
    return fps


def _widget(panels: Any, section_name: str, widget_name: str) -> Any:
    return getattr(getattr(panels, section_name), widget_name)


def _validate_widget_value(key: str, widget: Any, value: float) -> None:
    if not math.isfinite(value):
        raise AutotuneCandidateApplicationError(
            f"{key} is not a finite number and cannot be applied."
        )
    minimum = float(widget.minimum())
    maximum = float(widget.maximum())
    if not minimum <= value <= maximum:
        raise AutotuneCandidateApplicationError(
            f"{key}={value:g} is outside this control's supported range "
            f"({minimum:g}–{maximum:g}); the candidate was not applied."
        )


def apply_tracking_autotune_candidate(params: Mapping[str, Any], panels: Any) -> None:
    """Apply a selected candidate without implicit clamping or partial writes.

    The optimizer stores ``KALMAN_MATURITY_AGE`` and
    ``LOST_THRESHOLD_FRAMES`` as frame counts; TrackerKit's UI stores the same
    controls in seconds. The lower-layer typed contract first quantizes every
    candidate to the same fixed precision supported by those controls. Values
    are then validated before any widget is changed, so an unsupported
    candidate cannot partially apply or silently clamp.
    """
    try:
        candidate = quantize_tracking_autotune_params(
            applicable_candidate_params(params)
        )
    except ValueError as exc:
        raise AutotuneCandidateApplicationError(
            f"{exc} and cannot be applied."
        ) from exc
    fps = _valid_fps(panels) if any(key in candidate for key in _FRAME_WIDGETS) else 1.0
    updates: list[tuple[Any, float, str]] = []

    for key, (section_name, widget_name) in _DIRECT_WIDGETS.items():
        if key in candidate:
            updates.append(
                (
                    _widget(panels, section_name, widget_name),
                    float(tracking_autotune_widget_value(key, candidate[key])),
                    key,
                )
            )
    for key, (section_name, widget_name) in _FRAME_WIDGETS.items():
        if key in candidate:
            frames = candidate[key]
            updates.append(
                (
                    _widget(panels, section_name, widget_name),
                    float(tracking_autotune_widget_value(key, frames, fps=fps)),
                    f"{key} ({frames:g} frames at {fps:g} FPS)",
                )
            )

    for widget, value, key in updates:
        _validate_widget_value(key, widget, value)
    for widget, value, _key in updates:
        widget.setValue(value)
