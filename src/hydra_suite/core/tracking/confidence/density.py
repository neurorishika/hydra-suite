"""Density-aware detection filtering utilities."""

import numpy as np


def get_density_region_flags(
    meas,
    regions,
    frame_idx: int,
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
        for region in regions:
            if region.contains(frame_idx, cx, cy):
                flags[j] = True
                break
    return flags
