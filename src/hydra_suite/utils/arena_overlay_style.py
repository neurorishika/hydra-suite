"""Arena boundary overlay styling shared between core (cv2) and the
TrackerKit GUI (Qt). Deliberately framework-free -- plain ints/tuples.

Lives in utils/ (not core/ or trackerkit/) because it must be importable
from core/tracking/worker.py (cv2 drawing on raw video frames) AND from
trackerkit/gui/widgets/arena_style.py (Qt drawing on the setup canvas)
without either layer importing the other -- see CLAUDE.md's Dependency
Direction rule.
"""

from __future__ import annotations

# The combined-ROI/arena boundary stroke color, as (r, g, b) 0-255.
# Matches trackerkit/gui/widgets/arena_style.py's _LINE_BOUNDARY -- kept
# here as the single source of truth so the live tracking preview and the
# arena-setup canvas render the same boundary color.
BOUNDARY_COLOR_RGB = (20, 100, 220)

# Same color as a BGR tuple, for cv2 drawing calls (cv2 expects BGR).
BOUNDARY_COLOR_BGR = (
    BOUNDARY_COLOR_RGB[2],
    BOUNDARY_COLOR_RGB[1],
    BOUNDARY_COLOR_RGB[0],
)

# Divisor turning an image's shortest edge into a line width (260 -> 2px @
# 520px short edge, ~4px @ 1080px, ~8px @ 2160px) so boundary thickness
# stays visible instead of a fixed pixel count vanishing on high-resolution
# frames. Public (no leading underscore): arena_style.py imports this value
# rather than keeping its own copy, so the two never drift out of sync.
LINE_WIDTH_DIVISOR = 260


def boundary_line_width_px(min_dim: int) -> int:
    """Boundary stroke width in pixels for an image/viewport of the given
    shortest edge. Mirrors arena_style.py's boundary_line_width_px (2x the
    base line width) so the preview boundary is at least as thick as the
    setup canvas's, never a bare hardcoded ``2``."""
    base = max(2, int(round(int(min_dim) / LINE_WIDTH_DIVISOR)))
    return base * 2
