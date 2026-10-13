"""A degenerate OBB must be counted, not crash the run.

ClippingStats is a diagnostic tally, but its record() let overflow_ratio's
'Degenerate OBB (zero-length edge)' ValueError escape, and the inference
pipeline calls it for every detection that feeds a crop stage -- so one
zero-length box from the detector killed a whole video's batch pass
(Libby X5_d12, 2026-10-12, frame ~3290 of 47025).
"""

from __future__ import annotations

import math

import numpy as np

from hydra_suite.core.canonicalization.geometry import CanonicalGeometry, ClippingStats


def _geometry():
    return CanonicalGeometry.from_reference(
        reference_body_px=40.0, aspect_ratio=2.0, margin=1.3
    )


def test_degenerate_obb_is_tallied_not_raised():
    stats = ClippingStats()
    collapsed = np.array([[10, 10], [10, 10], [30, 20], [30, 20]], dtype=np.float32)
    ratio = stats.record(collapsed, _geometry())
    assert math.isnan(ratio)
    assert stats.degenerate_skipped_count == 1
    assert stats.total_count == 0 and stats.clipped_count == 0


def test_regular_obb_still_counted():
    stats = ClippingStats()
    box = np.array([[0, 0], [40, 0], [40, 20], [0, 20]], dtype=np.float32)
    assert stats.record(box, _geometry()) > 0
    assert stats.total_count == 1 and stats.degenerate_skipped_count == 0
