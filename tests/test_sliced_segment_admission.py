"""Tile admission must budget segment masks at their real size.

`estimated_prediction_job_bytes` modelled dense segment masks at 4 bytes per
pixel. Ultralytics returns `masks.data` as **uint8** -- 1 byte per pixel --
so the estimate was 4x the truth. Combined with budgeting for `max_det`
candidates rather than the detections a frame actually has, a 1024px tile with
a 600-detection ceiling was estimated at 2.5 GB against a measured 25 MB, and
sliced segment inference was refused outright.

Measured on a real checkpoint: `imgsz=1024, max_det=600` returned
`masks.data.shape=(24, 1024, 1024), dtype=torch.uint8` = 25.2 MB actual.
"""

import pytest

from hydra_suite.core.inference.stages.slicing import (
    DENSE_MASK_BYTES_PER_PIXEL,
    MAX_TILE_BATCH_BYTES,
    admitted_tile_chunk_size,
    estimated_prediction_job_bytes,
)
from hydra_suite.utils.slice_geometry import SlicePlan


def test_dense_mask_bytes_match_ultralytics_uint8_masks():
    """masks.data is uint8; anything else overstates the budget by that factor."""
    assert DENSE_MASK_BYTES_PER_PIXEL == 1


def test_segment_estimate_tracks_uint8_mask_size():
    imgsz, max_det = 1024, 600
    est = estimated_prediction_job_bytes(
        imgsz=imgsz, task="segment", max_detections=max_det
    )
    dense = max_det * imgsz * imgsz  # uint8
    assert dense <= est < dense * 1.1


def test_detect_task_budgets_no_dense_masks():
    """Only the segment task allocates dense masks."""
    est = estimated_prediction_job_bytes(imgsz=1024, task="detect", max_detections=600)
    assert est < 1024 * 1024 * 3 * 4 * 1.1


def _plan(tile=1931, frame=4512):
    return SlicePlan(
        tiles=[(0, 0, tile, tile)] * 9,
        slice_wh=(tile, tile),
        frame_wh=(frame, frame),
        full_frame=False,
    )


def test_sliced_segment_admissible_at_the_al_detection_ceiling():
    """The regression: an AL-ceiling export refused sliced inference outright."""
    chunk = admitted_tile_chunk_size(
        _plan(),
        imgsz=1024,
        device_tiles=True,
        requested=16,
        byte_budget=MAX_TILE_BATCH_BYTES,
        task="segment",
        max_detections=600,
    )
    assert chunk >= 1


def test_geometry_genuinely_too_large_is_still_refused():
    """The admission guard must still bite -- this is a safety bound, not a no-op."""
    with pytest.raises(ValueError, match="not resource-admissible"):
        admitted_tile_chunk_size(
            _plan(tile=8192, frame=16384),
            imgsz=8192,
            device_tiles=False,
            requested=1,
            byte_budget=MAX_TILE_BATCH_BYTES,
            task="segment",
            max_detections=600,
        )
