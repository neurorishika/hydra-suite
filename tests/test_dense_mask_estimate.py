"""I4: the dense segment-mask term of the memory estimate is realistic.

N left extraction, so the candidate count fed to the estimator is the
per-frame limit (1025 = MAX_DETECTIONS_PER_FRAME + probe row). Budgeting
1025 full-resolution masks per item collapsed every segment-task batch to 1.
Masks really are emitted at the model's letterbox resolution (ultralytics
``process_mask(..., upsample=True)``; measured 24 masks at imgsz 1024 =
25.2 MB uint8), so the pixel term stays ``imgsz**2``; what is bounded is the
number of masks the dense term assumes per item
(``DENSE_MASK_ESTIMATE_CANDIDATES``) -- an estimate, never a detection cap.
"""

from __future__ import annotations

import pytest

from hydra_suite.core.inference.limits import (
    DENSE_MASK_ESTIMATE_CANDIDATES,
    MAX_DETECTIONS_PER_FRAME,
)
from hydra_suite.core.inference.stages import slicing
from hydra_suite.core.inference.stages.slicing import (
    COMPACT_OUTPUT_BYTES_PER_DETECTION,
    admitted_prediction_chunk_size,
    admitted_tile_chunk_size,
    estimated_prediction_job_bytes,
)
from hydra_suite.utils.slice_geometry import SlicePlan

_MIB = 1024 * 1024
_CANDIDATES = MAX_DETECTIONS_PER_FRAME + 1  # effective_raw_detection_cap
_BASE_CANDIDATES = 52  # the pre-branch 2N at N=26


def _old_estimate(imgsz, task, candidates, source_bytes=0):
    """The estimator before this fix: every candidate gets a dense mask."""
    dense = candidates * imgsz * imgsz if task == "segment" else 0
    return source_bytes + imgsz * imgsz * 12 + candidates * 128 + dense


@pytest.fixture(autouse=True)
def _fresh_oversize_warnings(monkeypatch):
    monkeypatch.setattr(slicing, "_OVERSIZE_WARNED", set())


def test_dense_term_bounds_masks_not_pixels():
    side = 1024
    est = estimated_prediction_job_bytes(
        imgsz=side, task="segment", max_detections=_CANDIDATES
    )
    assert est == (
        side * side * 3 * 4
        + _CANDIDATES * COMPACT_OUTPUT_BYTES_PER_DETECTION  # compact: all 1025
        + DENSE_MASK_ESTIMATE_CANDIDATES * side * side  # dense: bounded, uint8
    )
    # Fewer candidates than the bound: exact, unchanged.
    small = estimated_prediction_job_bytes(
        imgsz=side, task="segment", max_detections=24
    )
    assert small == _old_estimate(side, "segment", 24)
    # Compact tasks never had a dense term.
    for task in ("obb", "detect"):
        assert estimated_prediction_job_bytes(
            imgsz=side, task=task, max_detections=_CANDIDATES
        ) == _old_estimate(side, task, _CANDIDATES)


def _tile_plan(tile=1024, frame=4096):
    return SlicePlan(
        tiles=[(0, 0, tile, tile)] * 16,
        slice_wh=(tile, tile),
        frame_wh=(frame, frame),
        full_frame=False,
    )


def _admitted_tile(candidates, device_tiles, estimator=None):
    plan = _tile_plan()
    kwargs = dict(
        imgsz=1024,
        device_tiles=device_tiles,
        requested=16,
        byte_budget=256 * _MIB,
        task="segment",
        max_detections=candidates,
    )
    if estimator is None:
        return admitted_tile_chunk_size(plan, **kwargs)
    source = 0 if device_tiles else 1024 * 1024 * 3
    per_job = estimator(1024, "segment", candidates, source)
    return 1 if per_job > 256 * _MIB else min(16, (256 * _MIB) // per_job)


@pytest.mark.parametrize("device_tiles", [True, False])
def test_sliced_segment_tile_admits_at_least_the_base_chunk(caplog, device_tiles):
    base = _admitted_tile(_BASE_CANDIDATES, device_tiles, _old_estimate)
    before = _admitted_tile(_CANDIDATES, device_tiles, _old_estimate)
    with caplog.at_level("WARNING"):
        after = _admitted_tile(_CANDIDATES, device_tiles)
    assert (base, before) == (3, 1)  # the review's table
    assert after >= base
    assert "batch size 1" not in caplog.text  # no per-run oversize WARNING


def test_sequential_stage2_segment_crops_admit_the_requested_batch():
    side, requested = 256, 63

    def _admit(estimate):
        return max(1, min(requested, 128, (1024 * _MIB) // estimate))

    src = side * side * 3
    base = _admit(_old_estimate(side, "segment", _BASE_CANDIDATES, src))
    before = _admit(_old_estimate(side, "segment", _CANDIDATES, src))
    after = admitted_prediction_chunk_size(
        imgsz=side,
        task="segment",
        max_detections=_CANDIDATES,
        requested=requested,
        source_bytes=src,
        description="Sequential stage-2 crop batch",
    )
    assert (base, before) == (63, 15)  # the review's table
    assert after >= base


def test_bound_is_an_estimate_assumption_not_a_cap():
    assert 0 < DENSE_MASK_ESTIMATE_CANDIDATES < MAX_DETECTIONS_PER_FRAME
