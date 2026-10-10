import numpy as np
import pytest

from hydra_suite.core.inference.config import OBBConfig, OBBDirectConfig
from hydra_suite.core.inference.limits import (
    MAX_DETECTIONS_PER_FRAME,
    DetectionLimitError,
    DetectionLimitStats,
    require_target_count_within_limit,
)
from hydra_suite.core.inference.result import OBBResult
from hydra_suite.core.inference.stages.filtering import filter_with_indices


def _spread(n: int, conf: float = 0.9) -> OBBResult:
    xs = np.arange(n, dtype=np.float32) * 40.0
    centroids = np.stack([xs, np.zeros(n, np.float32)], axis=1)
    corners = np.stack(
        [centroids + d for d in ([-5, -5], [5, -5], [5, 5], [-5, 5])], axis=1
    ).astype(np.float32)
    return OBBResult(
        frame_idx=0,
        centroids=centroids,
        angles=np.zeros(n, np.float32),
        sizes=np.full(n, 100.0, np.float32),
        shapes=np.tile(np.array([[100.0, 1.0]], np.float32), (n, 1)),
        confidences=np.full(n, conf, np.float32),
        corners=corners,
        detection_ids=OBBResult.make_detection_ids(0, n),
    )


def _cfg(max_detections: int) -> OBBConfig:
    return OBBConfig(
        mode="direct",
        direct=OBBDirectConfig(model_path="/m.pt"),
        max_detections=max_detections,
        confidence_threshold=0.1,
        iou_threshold=1.0,
    )


def test_limit_is_1024():
    assert MAX_DETECTIONS_PER_FRAME == 1024


def test_require_target_count_rejects_above_limit():
    assert require_target_count_within_limit(1024) == 1024
    with pytest.raises(DetectionLimitError, match="1024"):
        require_target_count_within_limit(1025)


def test_n_200_keeps_200_detections_regression_for_128_clamp():
    _, idx = filter_with_indices(_spread(300), _cfg(200))
    assert len(idx) == 200


def test_filter_rejects_n_above_limit():
    with pytest.raises(DetectionLimitError):
        filter_with_indices(_spread(3), _cfg(MAX_DETECTIONS_PER_FRAME + 1))


def test_stats_summary_counts_frames():
    stats = DetectionLimitStats()
    assert stats.summary() is None
    stats.record(7, 1500)
    stats.record(9, 2048)
    msg = stats.summary()
    assert "2 frame(s)" in msg and "1024" in msg and "7" in msg


# --- M12: a threshold below the extraction floor is loud ----------------------


def test_confidence_below_extraction_floor_warns_once(caplog):
    import logging

    from hydra_suite.core.inference.config import build_inference_config_from_params

    with caplog.at_level(logging.WARNING):
        build_inference_config_from_params(
            {"YOLO_CONFIDENCE_THRESHOLD": 0.005, "MAX_TARGETS": 4}
        )
    hits = [r for r in caplog.records if "extraction floor" in r.getMessage()]
    assert len(hits) == 1 and "0.01" in hits[0].getMessage()

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        build_inference_config_from_params(
            {"YOLO_CONFIDENCE_THRESHOLD": 0.25, "MAX_TARGETS": 4}
        )
    assert not [r for r in caplog.records if "extraction floor" in r.getMessage()]


# --- M13: per-frame limit warnings collapse after the first 20 ---------------


def test_per_frame_limit_warnings_collapse_after_twenty(caplog):
    import logging

    from hydra_suite.core.inference.limits import DetectionLimitStats

    stats = DetectionLimitStats()
    with caplog.at_level(logging.WARNING):
        for f in range(50):
            stats.record(f, 2000)
    msgs = [r.getMessage() for r in caplog.records]
    per_frame = [m for m in msgs if m.startswith("Frame ")]
    assert len(per_frame) == 20
    suppressed = [m for m in msgs if "suppressed" in m]
    assert len(suppressed) == 1 and "see summary" in suppressed[0]
    assert len(stats.frames) == 50  # every hit still counted
    assert stats.summary().startswith("50 frame(s)")
