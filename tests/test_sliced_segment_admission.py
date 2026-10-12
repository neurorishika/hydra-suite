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

from hydra_suite.core.inference.limits import (
    DENSE_MASK_ESTIMATE_CANDIDATES,
    MAX_DETECTIONS_PER_FRAME,
)
from hydra_suite.core.inference.stages.slicing import (
    DENSE_MASK_BYTES_PER_PIXEL,
    MAX_TILE_BATCH_BYTES,
    admitted_prediction_chunk_size,
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
    # uint8 masks, for at most DENSE_MASK_ESTIMATE_CANDIDATES masks per item
    dense = min(max_det, DENSE_MASK_ESTIMATE_CANDIDATES) * imgsz * imgsz
    assert est == imgsz * imgsz * 3 * 4 + max_det * 128 + dense


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


@pytest.fixture(autouse=True)
def _fresh_oversize_warnings(monkeypatch):
    from hydra_suite.core.inference.stages import slicing

    monkeypatch.setattr(slicing, "_OVERSIZE_WARNED", set())


def test_oversized_single_tile_is_admitted_at_batch_one_and_warned_once(caplog):
    """Admission never refuses the minimal unit of work (controller ruling R3).

    A segment model at imgsz 4096 (the bounded dense-mask term alone is 1 GiB)
    exceeds the 1 GiB hard ceiling; the tile is admitted at chunk size 1 and
    the WARNING (estimate + ceiling) is logged once, not per call.
    """
    est = estimated_prediction_job_bytes(
        imgsz=4096,
        task="segment",
        max_detections=MAX_DETECTIONS_PER_FRAME + 1,
        source_bytes=1931 * 1931 * 3,
    )
    assert est > MAX_TILE_BATCH_BYTES
    with caplog.at_level("WARNING"):
        for _ in range(3):
            chunk = admitted_tile_chunk_size(
                _plan(),
                imgsz=4096,
                device_tiles=False,
                requested=16,
                byte_budget=MAX_TILE_BATCH_BYTES,
                task="segment",
                max_detections=MAX_DETECTIONS_PER_FRAME + 1,
            )
            assert chunk == 1
    warnings = [r for r in caplog.records if "batch size 1" in r.getMessage()]
    assert len(warnings) == 1
    assert str(est) in warnings[0].getMessage()
    assert str(MAX_TILE_BATCH_BYTES) in warnings[0].getMessage()


def test_oversized_stage2_crop_is_admitted_at_batch_one(caplog):
    """Sequential stage-2 with a segment model: same rule, no ValueError."""
    with caplog.at_level("WARNING"):
        size = admitted_prediction_chunk_size(
            imgsz=4096,
            task="segment",
            max_detections=MAX_DETECTIONS_PER_FRAME + 1,
            requested=8,
            source_bytes=4096 * 4096 * 3,
            description="Sequential stage-2 crop batch",
        )
    assert size == 1
    assert "Sequential stage-2 crop batch" in caplog.text


def test_tight_budget_still_shrinks_the_batch():
    """The budget still bites: it shrinks the chunk below what a loose one allows."""
    kwargs = dict(
        imgsz=640,
        device_tiles=True,
        requested=16,
        task="segment",
        max_detections=64,
    )
    loose = admitted_tile_chunk_size(
        _plan(), byte_budget=MAX_TILE_BATCH_BYTES, **kwargs
    )
    tight = admitted_tile_chunk_size(_plan(), byte_budget=64 * 1024 * 1024, **kwargs)
    assert loose > tight >= 1
    assert tight == (64 * 1024 * 1024) // estimated_prediction_job_bytes(
        imgsz=640, task="segment", max_detections=64
    )


def test_detect_and_obb_admission_unchanged(caplog):
    """Compact-output tasks never hit the oversize path at normal geometry."""
    for task in ("detect", "obb"):
        with caplog.at_level("WARNING"):
            chunk = admitted_tile_chunk_size(
                _plan(),
                imgsz=1024,
                device_tiles=False,
                requested=16,
                byte_budget=256 * 1024 * 1024,
                task=task,
                max_detections=MAX_DETECTIONS_PER_FRAME + 1,
            )
        per_job = estimated_prediction_job_bytes(
            imgsz=1024,
            task=task,
            max_detections=MAX_DETECTIONS_PER_FRAME + 1,
            source_bytes=1931 * 1931 * 3,
        )
        assert chunk == min(16, (256 * 1024 * 1024) // per_job) > 1
    assert "batch size 1" not in caplog.text


def test_oversize_warning_is_once_per_run_not_once_per_process(caplog):
    """The GUI runs tracking in-process: each new InferenceRunner (= run) must
    log the oversize WARNING again, once, even for identical geometry."""
    from unittest.mock import MagicMock, patch

    from hydra_suite.core.inference.config import build_obb_only_config
    from hydra_suite.core.inference.runner import InferenceRunner

    cfg = build_obb_only_config(
        "seg.pt",
        confidence_threshold=0.25,
        iou_threshold=0.7,
        mode="direct",
        model_task="segment",
    )

    def _admit():
        return admitted_tile_chunk_size(
            _plan(),
            imgsz=4096,
            device_tiles=False,
            requested=16,
            byte_budget=MAX_TILE_BATCH_BYTES,
            task="segment",
            max_detections=MAX_DETECTIONS_PER_FRAME + 1,
        )

    per_run = []
    with patch("hydra_suite.core.inference.runner._load_all_models") as ml:
        ml.return_value = MagicMock(
            obb=MagicMock(), headtail=None, cnn=[], pose=None, apriltag=None
        )
        for _run in range(2):
            caplog.clear()
            with caplog.at_level("WARNING"):
                InferenceRunner(cfg)
                assert _admit() == 1
                assert _admit() == 1
            per_run.append(
                sum("batch size 1" in r.getMessage() for r in caplog.records)
            )
    assert per_run == [1, 1]
