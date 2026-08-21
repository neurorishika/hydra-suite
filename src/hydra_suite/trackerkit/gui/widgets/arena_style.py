"""Arena overlay styling: palette, sizing, alpha constants.

Deliberately Qt-free -- returns plain ``(r, g, b)`` tuples and ints. The
caller converts to ``QColor``. Keeping these rules importable without a
display is what makes them testable (see ``project_main_suite_blockers``).
"""

from __future__ import annotations

from dataclasses import dataclass

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
    veil: tuple[int, int, int]
    glyph: tuple[int, int, int]
    halo: tuple[int, int, int]


def frame_palette(mean_luminance: float) -> ArenaPalette:
    """Palette for a frame of the given mean luminance (0.0-1.0).

    Hue is fixed per ROLE -- include blue, exclude red, in-progress green --
    so the meaning stays learnable across videos. Only the light/dark variant
    changes with the footage, along with veil, glyph and halo polarity.
    """
    dark_frame = mean_luminance < _LUMINANCE_MIDPOINT
    if dark_frame:
        return ArenaPalette(
            line_include=(120, 190, 255),
            line_exclude=(255, 120, 110),
            line_preview=(140, 255, 150),
            veil=(255, 255, 255),
            glyph=(255, 255, 255),
            halo=(20, 20, 20),
        )
    return ArenaPalette(
        line_include=(0, 80, 200),
        line_exclude=(200, 25, 25),
        line_preview=(20, 130, 40),
        veil=(0, 0, 0),
        glyph=(20, 20, 20),
        halo=(255, 255, 255),
    )


def line_width_px(viewport_min_dim: int) -> int:
    """Outline width in DEVICE pixels for the given viewport short edge.

    Deriving this from the viewport rather than the image is the whole point:
    a width in image pixels scales with zoom, which is why the current 2 px
    cyan pen vanishes when zoomed out.
    """
    return max(2, int(round(int(viewport_min_dim) / _LINE_WIDTH_DIVISOR)))


def glyph_size_px(on_screen_radius: float) -> int:
    """Arena-number point size for an arena of the given on-screen radius.

    Clamped so 96 wells stay readable and a single large arena does not get an
    absurd number.
    """
    raw = float(on_screen_radius) * 0.8
    return int(max(GLYPH_MIN_PX, min(GLYPH_MAX_PX, raw)))
