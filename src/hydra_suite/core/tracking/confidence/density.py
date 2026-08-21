"""Density-aware detection filtering utilities."""

import numpy as np


def get_density_region_flags(
    meas,
    regions,
    frame_idx: int,
    meas_arena=None,
) -> np.ndarray:
    """Return a boolean mask indicating which detections fall inside a density region.

    Parameters
    ----------
    meas:
        List/array of detection measurements.  Each element must be indexable
        with ``[0]`` (x) and ``[1]`` (y).
    regions:
        List of :class:`DensityRegion` to test against.
    frame_idx:
        Current frame index.
    meas_arena:
        Optional ``(M,)`` int array of arena ids per detection (``-1`` outside
        every arena).  A region carrying ``arena=<id>`` then flags ONLY
        detections whose arena id equals it, so an arena's flags stay a pure
        function of its own detections even where a region's rectangular
        bounding box overlaps a neighbour.  A detection outside every arena
        (``-1``) matches no arena-tagged region and is therefore never flagged
        by one.  Regions with ``arena is None`` -- every single-arena run and
        every pre-multi-arena sidecar -- are whole-frame and match regardless.
        Passing ``None`` (the default, and what single-arena callers pass)
        ignores region arena tags entirely, i.e. exactly the original
        behaviour.

    Returns
    -------
    np.ndarray
        Shape ``(M,)`` bool array — ``True`` for detections inside a flagged
        region.
    """
    M = len(meas)
    flags = np.zeros(M, dtype=bool)
    if not regions:
        return flags

    for j in range(M):
        cx, cy = float(meas[j][0]), float(meas[j][1])
        det_arena = None if meas_arena is None else int(meas_arena[j])
        for region in regions:
            if (
                det_arena is not None
                and region.arena is not None
                and int(region.arena) != det_arena
            ):
                continue
            if region.contains(frame_idx, cx, cy):
                flags[j] = True
                break
    return flags
