"""Arena overlay styling: palette, sizing, alpha constants.

Deliberately Qt-free -- returns plain ``(r, g, b)`` tuples and ints. The
caller converts to ``QColor``. Keeping these rules importable without a
display is what makes them testable (see ``project_main_suite_blockers``).
"""

from __future__ import annotations

from dataclasses import dataclass

from hydra_suite.utils.arena_overlay_style import BOUNDARY_COLOR_RGB
from hydra_suite.utils.arena_overlay_style import (
    boundary_line_width_px as _shared_boundary_line_width_px,
)

VEIL_ALPHA = 0.15
TEXT_ALPHA = 0.70
CLICK_DRAG_THRESHOLD_PX = 3
GLYPH_MIN_PX = 10
GLYPH_MAX_PX = 64

# Below this mean luminance the frame counts as dark and the polarity flips.
_LUMINANCE_MIDPOINT = 0.5
# Divisor turning the viewport's short edge into a line width. 260 gives 2 px
# on a 520 px viewport and 4 px on a 1080 px one -- visible without being fat.
_LINE_WIDTH_DIVISOR = 260


@dataclass(frozen=True)
class ArenaPalette:
    """Colours for one frame's overlay, all as (r, g, b) 0-255 tuples."""

    line_include: tuple[int, int, int]
    line_exclude: tuple[int, int, int]
    line_preview: tuple[int, int, int]
    line_boundary: tuple[int, int, int]
    veil: tuple[int, int, int]
    glyph: tuple[int, int, int]
    halo: tuple[int, int, int]


# Fixed per-role colours -- learnable across footage, independent of
# frame polarity. Only the arena-number glyph/halo still adapt to
# luminance, since legibility against arbitrary footage still matters there.
_LINE_INCLUDE = (0, 160, 60)  # green
_LINE_EXCLUDE = (210, 30, 30)  # red
_LINE_PREVIEW = (255, 140, 0)  # orange -- distinct from committed include (green)
_LINE_BOUNDARY = BOUNDARY_COLOR_RGB  # blue -- the combined-ROI outline
_VEIL = BOUNDARY_COLOR_RGB  # same blue as the boundary, filled at VEIL_ALPHA


def frame_palette(mean_luminance: float) -> ArenaPalette:
    """Palette for a frame of the given mean luminance (0.0-1.0).

    Zone/boundary/veil colours are fixed regardless of footage polarity, so
    green/red/blue stay learnable across videos. Only the arena-number
    glyph/halo still flip for contrast against the actual frame content.
    """
    dark_frame = mean_luminance < _LUMINANCE_MIDPOINT
    glyph, halo = (
        ((255, 255, 255), (20, 20, 20))
        if dark_frame
        else ((20, 20, 20), (255, 255, 255))
    )
    return ArenaPalette(
        line_include=_LINE_INCLUDE,
        line_exclude=_LINE_EXCLUDE,
        line_preview=_LINE_PREVIEW,
        line_boundary=_LINE_BOUNDARY,
        veil=_VEIL,
        glyph=glyph,
        halo=halo,
    )


def line_width_px(viewport_min_dim: int) -> int:
    """Outline width in DEVICE pixels for the given viewport short edge.

    Deriving this from the viewport rather than the image is the whole point:
    a width in image pixels scales with zoom, which is why the current 2 px
    cyan pen vanishes when zoomed out.
    """
    return max(2, int(round(int(viewport_min_dim) / _LINE_WIDTH_DIVISOR)))


def zone_line_width_px(viewport_min_dim: int) -> int:
    """Thin outline width for individual inclusion/exclusion zone shapes --
    secondary detail relative to the thicker combined-ROI boundary."""
    return max(1, line_width_px(viewport_min_dim) // 2)


def boundary_line_width_px(viewport_min_dim: int) -> int:
    """Thick outline width for an arena's combined-ROI boundary stroke."""
    return _shared_boundary_line_width_px(viewport_min_dim)


def glyph_size_px(on_screen_radius: float) -> int:
    """Arena-number point size for an arena of the given on-screen radius.

    Clamped so 96 wells stay readable and a single large arena does not get an
    absurd number.
    """
    raw = float(on_screen_radius) * 0.8
    return int(max(GLYPH_MIN_PX, min(GLYPH_MAX_PX, raw)))
