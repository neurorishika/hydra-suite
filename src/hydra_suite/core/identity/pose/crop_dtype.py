"""Crop dtype normalisation shared by pose backends.

Pose backends expect crops as uint8 ``[0, 255]``: that is what
``cv2.imread`` yields, and it is the convention every backend's preprocessing
assumes when it divides by 255.

The tracking pipeline does not produce that.  ``stages/crops.py`` builds
canonical crops as float32 ``[0, 1]`` (``torch.from_numpy(...).float() / 255.0``)
so the GPU warp can run on them.  Handing such a crop to a backend unchanged
divides by 255 a second time, leaving the model an essentially constant image;
casting it straight to uint8 instead floors every pixel to black.

``backends/sleap.py`` already guards both ways at each of its entry points.
This module is that guard, in one place, for any backend to reuse.
"""

from __future__ import annotations

import numpy as np

# Floats at or below this maximum are read as a unit-range image and scaled.
# The tolerance absorbs rounding in crops that reach exactly 1.0.
_UNIT_RANGE_MAX = 1.0 + 1e-3


def to_uint8_image(crop: np.ndarray) -> np.ndarray:
    """Return *crop* as uint8 ``[0, 255]``, preserving shape and channel order.

    uint8 input is returned unchanged.  Float input is scaled by 255 when it
    looks unit-range, passed through when it already spans ``[0, 255]``, then
    sanitised and clipped.  Non-finite values become 0.
    """
    arr = np.asarray(crop)
    if arr.dtype == np.uint8:
        return arr

    a = arr.astype(np.float32)
    finite = a[np.isfinite(a)]
    if finite.size and float(finite.max()) <= _UNIT_RANGE_MAX:
        a = a * 255.0
    return np.clip(np.nan_to_num(a), 0.0, 255.0).astype(np.uint8)
