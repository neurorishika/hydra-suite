"""F4: crop-dataset export must surface (not silently swallow) canvas clipping.

Both the crop-dataset exporter (``IndividualDatasetGenerator``) and the
oriented-video exporter accumulate per-detection overflow stats via the
shared ``ClippingStats`` and must emit a ``logger.warning`` at finalize when
any detection was clipped by the fixed canonical canvas.
"""

import logging

import numpy as np

from hydra_suite.core.identity.dataset.generator import IndividualDatasetGenerator


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
