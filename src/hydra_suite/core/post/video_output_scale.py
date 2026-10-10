"""The ``video_output_scale`` knob for the final annotated video.

One place owns the default, the valid range and the output-size rule so the
GUI spinbox, the ``--video-scale`` CLI flag and the renderer cannot drift.
Out-of-range values are a loud ``ValueError`` -- never a silent clamp: a user
who typed 1.5 asked for something this renderer cannot do.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

VIDEO_OUTPUT_SCALE_KEY = "video_output_scale"
DEFAULT_VIDEO_OUTPUT_SCALE = 0.5
MIN_VIDEO_OUTPUT_SCALE = 0.1
MAX_VIDEO_OUTPUT_SCALE = 1.0


def validate_video_output_scale(value: Any) -> float:
    """Return ``value`` as a float in [0.1, 1.0] or raise ``ValueError``."""
    try:
        scale = float(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"{VIDEO_OUTPUT_SCALE_KEY} must be a number in "
            f"[{MIN_VIDEO_OUTPUT_SCALE}, {MAX_VIDEO_OUTPUT_SCALE}], got {value!r}"
        ) from None
    if not math.isfinite(scale) or not (
        MIN_VIDEO_OUTPUT_SCALE <= scale <= MAX_VIDEO_OUTPUT_SCALE
    ):
        raise ValueError(
            f"{VIDEO_OUTPUT_SCALE_KEY} must be in "
            f"[{MIN_VIDEO_OUTPUT_SCALE}, {MAX_VIDEO_OUTPUT_SCALE}], got {value!r}"
        )
    return scale


def resolve_video_output_scale(config: Mapping[str, Any]) -> float:
    """The validated scale from a config dict; a missing/None key is 0.5."""
    value = config.get(VIDEO_OUTPUT_SCALE_KEY) if config else None
    if value is None:
        return DEFAULT_VIDEO_OUTPUT_SCALE
    return validate_video_output_scale(value)


def _even_dim(source: int, scale: float) -> int:
    dim = int(round(source * scale))
    dim -= dim % 2  # H.264/HEVC 4:2:0 needs even sizes; never exceed round()
    return max(2, dim)


def scaled_output_size(width: int, height: int, scale: float) -> tuple[int, int]:
    """Output ``(width, height)``: round(src * scale), forced even, min 2."""
    return _even_dim(int(width), scale), _even_dim(int(height), scale)
