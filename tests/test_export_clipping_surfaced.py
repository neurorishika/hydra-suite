"""F4: crop-dataset export must surface (not silently swallow) canvas clipping.

Both the crop-dataset exporter (``IndividualDatasetGenerator``) and the
oriented-video exporter accumulate per-detection overflow stats via the
shared ``ClippingStats`` and must emit a ``logger.warning`` at finalize when
any detection was clipped by the fixed canonical canvas.
"""

import logging

import numpy as np

from hydra_suite.core.individual.dataset.generator import IndividualDatasetGenerator
from hydra_suite.core.individual.dataset.oriented_video import (
    FrameBundle,
    OrientedTrackVideoExporter,
)


def _make_generator(tmp_path):
    params = {
        "ENABLE_INDIVIDUAL_DATASET": True,
        "ENABLE_INDIVIDUAL_IMAGE_SAVE": True,
        "INDIVIDUAL_CROP_PADDING": 0.1,
        "INDIVIDUAL_DATASET_RUN_ID": "run1",
        # Tiny reference body size -> tiny canonical canvas, so a
        # normal-sized OBB below will badly overflow it.
        "REFERENCE_BODY_SIZE": 2.0,
        "RESIZE_FACTOR": 1.0,
        "ADVANCED_CONFIG": {
            "reference_aspect_ratio": 2.0,
            "canonical_margin": 1.0,
        },
    }
    return IndividualDatasetGenerator(
        params=params,
        output_dir=str(tmp_path),
        video_name="test_video",
        dataset_name="ds",
    )


def test_overflowing_obb_warns_at_finalize(caplog, tmp_path):
    gen = _make_generator(tmp_path)
    assert gen.enabled and gen.crops_dir is not None

    frame = np.zeros((500, 500, 3), dtype=np.uint8)
    # A large OBB (100x50 px) against a canvas sized off a 2px reference body
    # -- guaranteed overflow_ratio >> 1.
    cx, cy = 250.0, 250.0
    theta = 0.0
    half_w, half_h = 50.0, 25.0
    corners = np.array(
        [
            [cx - half_w, cy - half_h],
            [cx + half_w, cy - half_h],
            [cx + half_w, cy + half_h],
            [cx - half_w, cy + half_h],
        ],
        dtype=np.float32,
    )

    with caplog.at_level(logging.WARNING):
        num_saved = gen.process_frame(
            frame,
            frame_id=0,
            meas=[[cx, cy, theta]],
            obb_corners=[corners],
            track_ids=[1],
            trajectory_ids=[1],
        )
        assert num_saved == 1

        result_path = gen.finalize()

    assert result_path is not None
    assert gen._clipping_stats.clipped_count == 1
    assert gen._clipping_stats.worst_overflow_ratio > 1.0

    warnings = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert any("CLIPPED" in msg for msg in warnings), warnings


def _make_oriented_exporter(tmp_path, **kwargs):
    dataset_dir = tmp_path / "individual_crops" / "run_20260311"
    exporter = OrientedTrackVideoExporter(
        dataset_dir,
        tmp_path / "final.csv",
        video_path=tmp_path / "source.mp4",
        detection_cache_path=tmp_path / "detections.npz",
        fps=5.0,
        padding_fraction=0.0,
        **kwargs,
    )
    return exporter, dataset_dir


def test_oriented_video_overflowing_obb_warns_at_finalize(caplog, tmp_path):
    """Same F4 guard for the oriented-video exporter: a task whose OBB badly
    overflows the fixed canvas must surface a CLIPPED warning when the
    canonical provenance is written."""
    exporter, dataset_dir = _make_oriented_exporter(tmp_path)

    # Box far larger than the tiny default fallback canvas (canvas derived
    # from reference_body_px=20, aspect_ratio=2.0 -> ~ (28, 14)) -- guaranteed
    # overflow_ratio >> 1.
    task = exporter._build_task(
        frame_id=0,
        trajectory_id=1,
        center_x=100.0,
        center_y=100.0,
        width=300.0,
        height=150.0,
        theta=0.0,
        corners=np.array(
            [[-50, -50], [250, -50], [250, 100], [-50, 100]], dtype=np.float32
        ),
        polygon_index=0,
        detection_id=1,
    )
    assert task is not None
    assert exporter._clipping_stats.clipped_count == 1
    assert exporter._clipping_stats.total_count == 1

    with caplog.at_level(logging.WARNING):
        exporter._write_canonical_metadata()

    warnings = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert any("CLIPPED" in msg for msg in warnings), warnings


def test_oriented_video_postprocessing_does_not_double_count_clipping(tmp_path):
    """Regression guard for the double-count bug: when fix_direction_flips
    (or enable_affine_stabilization) is on, _apply_track_postprocessing
    re-derives the same tasks' affines but must NOT record them again in
    self._clipping_stats -- total_count must equal the number of detections,
    not 2x."""
    exporter, _dataset_dir = _make_oriented_exporter(
        tmp_path, fix_direction_flips=True, heading_flip_max_burst=1
    )

    num_detections = 3
    frame_bundles: dict[int, FrameBundle] = {}
    track_sizes: dict[int, tuple[int, int]] = {}
    for i in range(num_detections):
        task = exporter._build_task(
            frame_id=i,
            trajectory_id=7,
            center_x=100.0,
            center_y=100.0,
            width=300.0,
            height=150.0,
            theta=0.0,
            corners=np.array(
                [[-50, -50], [250, -50], [250, 100], [-50, 100]], dtype=np.float32
            ),
            polygon_index=0,
            detection_id=i,
        )
        assert task is not None
        bundle = frame_bundles.setdefault(i, FrameBundle())
        bundle.tasks.append(task)
        bundle.polygons.append(task.corners)
        track_sizes[7] = (task.out_w, task.out_h)

    # Sanity: recording happened exactly once per detection during the
    # initial _build_task pass, before any postprocessing.
    assert exporter._clipping_stats.total_count == num_detections
    assert exporter._clipping_stats.clipped_count == num_detections

    # This re-derives each task's affine (heading-flip fixing is enabled) --
    # it must not add to the tally a second time.
    exporter._apply_track_postprocessing(frame_bundles, track_sizes)

    assert exporter._clipping_stats.total_count == num_detections
    assert exporter._clipping_stats.clipped_count == num_detections
