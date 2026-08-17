"""Tests for dataset generation and active-learning export metadata."""

from pathlib import Path

import numpy as np
import pytest

from hydra_suite.data.dataset_generation import FrameQualityScorer
from hydra_suite.utils.geometry_levels import GeometryLevel


class _FakeCap:
    """Minimal cv2.VideoCapture stand-in returning one constant frame."""

    def __init__(self, frame, total):
        self._frame = frame
        self._total = total

    def isOpened(self):
        return True

    def get(self, prop):
        import cv2

        return {
            cv2.CAP_PROP_FRAME_COUNT: self._total,
            cv2.CAP_PROP_FRAME_WIDTH: self._frame.shape[1],
            cv2.CAP_PROP_FRAME_HEIGHT: self._frame.shape[0],
        }.get(prop, 0)

    def set(self, prop, value):
        return True

    def read(self):
        return True, self._frame.copy()

    def release(self):
        return None


def _seg_obb_result():
    import numpy as np

    from hydra_suite.core.inference.result import OBBResult

    corners = np.array(
        [[[90.0, 40.0], [110.0, 40.0], [110.0, 60.0], [90.0, 60.0]]], dtype=np.float32
    )
    result = OBBResult(
        frame_idx=0,
        centroids=np.array([[100.0, 50.0]], dtype=np.float32),
        angles=np.array([0.0], dtype=np.float32),
        sizes=np.array([400.0], dtype=np.float32),
        shapes=np.array([[400.0, 1.0]], dtype=np.float32),
        confidences=np.array([0.8], dtype=np.float32),
        corners=corners,
        detection_ids=OBBResult.make_detection_ids(0, 1),
        class_ids=np.array([0], dtype=np.int64),
    )
    result.polygons = [
        np.array([[90.0, 40.0], [110.0, 42.0], [108.0, 60.0]], dtype=np.float32)
    ]
    return result


class TestFrameQualityScorer:
    """Test suite for FrameQualityScorer class."""

    def test_initialization(self):
        """Test scorer initialization with default parameters."""
        params = {
            "MAX_TARGETS": 4,
            "DATASET_CONF_THRESHOLD": 0.5,
            "METRIC_LOW_CONFIDENCE": True,
            "METRIC_COUNT_MISMATCH": True,
            "METRIC_HIGH_ASSIGNMENT_COST": True,
            "METRIC_TRACK_LOSS": True,
            "METRIC_HIGH_UNCERTAINTY": False,
            "METRIC_FRAGMENTED_DETECTIONS": True,
        }

        scorer = FrameQualityScorer(params)

        assert scorer.max_targets == 4
        assert scorer.conf_threshold == 0.5
        assert scorer.use_confidence is True
        assert scorer.use_count_mismatch is True

    def test_score_frame_perfect_detections(self):
        """Test scoring of frame with perfect detections."""
        params = {
            "MAX_TARGETS": 4,
            "DATASET_CONF_THRESHOLD": 0.5,
            "METRIC_LOW_CONFIDENCE": True,
            "METRIC_COUNT_MISMATCH": True,
        }

        scorer = FrameQualityScorer(params)

        detection_data = {
            "confidences": [0.9, 0.85, 0.88, 0.92],
            "count": 4,
        }

        score = scorer.score_frame(frame_id=0, detection_data=detection_data)

        # Perfect frame should have low score
        assert score == 0.0

    def test_score_frame_low_confidence(self):
        """Test scoring when detections have low confidence."""
        params = {
            "MAX_TARGETS": 4,
            "DATASET_CONF_THRESHOLD": 0.7,
            "METRIC_LOW_CONFIDENCE": True,
            "METRIC_COUNT_MISMATCH": False,
        }

        scorer = FrameQualityScorer(params)

        detection_data = {
            "confidences": [0.3, 0.4, 0.5, 0.6],  # All below threshold
            "count": 4,
        }

        score = scorer.score_frame(frame_id=0, detection_data=detection_data)

        # Low confidence should increase score
        assert score > 0

    def test_score_frame_count_mismatch_under(self):
        """Test scoring when detection count is below expected."""
        params = {
            "MAX_TARGETS": 4,
            "DATASET_CONF_THRESHOLD": 0.5,
            "METRIC_LOW_CONFIDENCE": False,
            "METRIC_COUNT_MISMATCH": True,
        }

        scorer = FrameQualityScorer(params)

        detection_data = {
            "confidences": [0.8, 0.9],
            "count": 2,  # Only 2 instead of 4
        }

        score = scorer.score_frame(frame_id=0, detection_data=detection_data)

        # Under-detection should significantly increase score
        assert score > 0

    def test_score_frame_count_mismatch_over(self):
        """Test scoring when detection count is above expected."""
        params = {
            "MAX_TARGETS": 4,
            "DATASET_CONF_THRESHOLD": 0.5,
            "METRIC_LOW_CONFIDENCE": False,
            "METRIC_COUNT_MISMATCH": True,
        }

        scorer = FrameQualityScorer(params)

        detection_data = {
            "confidences": [0.8, 0.9, 0.85, 0.88, 0.82, 0.87],
            "count": 6,  # 6 instead of 4
        }

        score = scorer.score_frame(frame_id=0, detection_data=detection_data)

        # Over-detection should increase score (but less than under-detection)
        assert score > 0

    def test_score_frame_high_assignment_cost(self):
        """Test scoring when tracking assignment costs are high."""
        params = {
            "MAX_TARGETS": 4,
            "DATASET_CONF_THRESHOLD": 0.5,
            "METRIC_LOW_CONFIDENCE": False,
            "METRIC_COUNT_MISMATCH": False,
            "METRIC_HIGH_ASSIGNMENT_COST": True,
        }

        scorer = FrameQualityScorer(params)

        tracking_data = {
            "assignment_costs": [80, 90, 75, 85],  # High costs
        }

        score = scorer.score_frame(frame_id=0, tracking_data=tracking_data)

        # High costs should increase score
        assert score > 0

    def test_score_frame_combined_metrics(self):
        """Test scoring with multiple problematic metrics."""
        params = {
            "MAX_TARGETS": 4,
            "DATASET_CONF_THRESHOLD": 0.7,
            "METRIC_LOW_CONFIDENCE": True,
            "METRIC_COUNT_MISMATCH": True,
            "METRIC_HIGH_ASSIGNMENT_COST": True,
        }

        scorer = FrameQualityScorer(params)

        detection_data = {
            "confidences": [0.4, 0.5, 0.3],  # Low confidence
            "count": 3,  # Count mismatch
        }

        tracking_data = {
            "assignment_costs": [60, 70, 55],  # High costs
        }

        score = scorer.score_frame(
            frame_id=0, detection_data=detection_data, tracking_data=tracking_data
        )

        # Multiple issues should compound the score
        assert score > 0.3

    def test_score_frame_nan_confidences_ignored(self):
        """Test that NaN confidences (from background subtraction) are handled."""
        params = {
            "MAX_TARGETS": 4,
            "DATASET_CONF_THRESHOLD": 0.5,
            "METRIC_LOW_CONFIDENCE": True,
        }

        scorer = FrameQualityScorer(params)

        detection_data = {
            "confidences": [np.nan, np.nan, np.nan, np.nan],
            "count": 4,
        }

        # Should not crash and should handle gracefully
        score = scorer.score_frame(frame_id=0, detection_data=detection_data)

        assert isinstance(score, (int, float))

    def test_score_frame_mixed_confidences(self):
        """Test scoring with mix of valid and NaN confidences."""
        params = {
            "MAX_TARGETS": 4,
            "DATASET_CONF_THRESHOLD": 0.7,
            "METRIC_LOW_CONFIDENCE": True,
        }

        scorer = FrameQualityScorer(params)

        detection_data = {
            "confidences": [0.8, 0.3, np.nan, 0.6],  # Mixed
            "count": 4,
        }

        score = scorer.score_frame(frame_id=0, detection_data=detection_data)

        # Should score based on valid confidences only
        assert score > 0  # 0.3 is below threshold

    def test_score_frame_no_data(self):
        """Test scoring with no detection or tracking data."""
        params = {
            "MAX_TARGETS": 4,
            "DATASET_CONF_THRESHOLD": 0.5,
            "METRIC_LOW_CONFIDENCE": True,
            "METRIC_COUNT_MISMATCH": True,
        }

        scorer = FrameQualityScorer(params)

        score = scorer.score_frame(frame_id=0)

        # Should return 0 or low score when no data
        assert score == 0.0

    def test_score_frame_empty_detection_data(self):
        """Test scoring with empty detection data dict."""
        params = {
            "MAX_TARGETS": 4,
            "DATASET_CONF_THRESHOLD": 0.5,
            "METRIC_LOW_CONFIDENCE": True,
        }

        scorer = FrameQualityScorer(params)

        score = scorer.score_frame(frame_id=0, detection_data={})

        assert score == 0.0

    def test_metrics_can_be_disabled(self):
        """Test that individual metrics can be disabled."""
        params = {
            "MAX_TARGETS": 4,
            "DATASET_CONF_THRESHOLD": 0.7,
            "METRIC_LOW_CONFIDENCE": False,  # Disabled
            "METRIC_COUNT_MISMATCH": False,  # Disabled
        }

        scorer = FrameQualityScorer(params)

        detection_data = {
            "confidences": [0.3, 0.4],  # Low confidence (but metric disabled)
            "count": 2,  # Count mismatch (but metric disabled)
        }

        score = scorer.score_frame(frame_id=0, detection_data=detection_data)

        # Should be 0 since all metrics are disabled
        assert score == 0.0


def test_export_dataset_writes_three_roots_for_segmentation(tmp_path, monkeypatch):
    """A segmentation source yields polygon + obb + aabb roots."""
    import numpy as np
    import pandas as pd

    import hydra_suite.data.dataset_generation as dg

    video = tmp_path / "clip.mp4"
    video.write_bytes(b"")
    csv_path = tmp_path / "tracks.csv"
    pd.DataFrame(
        {
            "FrameID": [0, 0],
            "TrackID": [1, 2],
            "X": [100.0, 150.0],
            "Y": [50.0, 60.0],
            "Theta": [0.0, 0.5],
            "State": ["tracked", "tracked"],
        }
    ).to_csv(csv_path, index=False)

    frame = np.zeros((200, 400, 3), dtype=np.uint8)
    monkeypatch.setattr(dg, "_open_video", lambda p: _FakeCap(frame, total=3))
    monkeypatch.setattr(dg, "_init_detection_runner", lambda params: None)
    monkeypatch.setattr(
        dg,
        "_detect_records_for_frames",
        lambda runner, frames, params, level: {
            0: dg.records_from_obb_result(_seg_obb_result(), level)
        },
    )

    manifest = dg.export_dataset(
        video_path=str(video),
        csv_path=str(csv_path),
        frame_ids=[0],
        output_dir=str(tmp_path / "out"),
        dataset_name="round",
        class_name="ant",
        params={"DETECTION_METHOD": "yolo_obb", "YOLO_OBB_DIRECT_TASK": "segment"},
        include_context=False,
    )

    assert manifest["native_level"] == "polygon"
    assert {r["level"] for r in manifest["roots"]} == {"polygon", "obb", "aabb"}
    round_dir = Path(manifest["round_dir"])
    assert (round_dir / "polygon" / "labels").is_dir()
    assert (round_dir / "aabb" / "labels").is_dir()


def test_export_dataset_writes_two_roots_for_obb_model(tmp_path, monkeypatch):
    """An OBB model must never produce a polygon root."""
    import numpy as np
    import pandas as pd

    import hydra_suite.data.dataset_generation as dg

    video = tmp_path / "clip.mp4"
    video.write_bytes(b"")
    csv_path = tmp_path / "tracks.csv"
    pd.DataFrame(
        {
            "FrameID": [0, 0],
            "TrackID": [1, 2],
            "X": [100.0, 150.0],
            "Y": [50.0, 60.0],
            "Theta": [0.0, 0.5],
            "State": ["tracked", "tracked"],
        }
    ).to_csv(csv_path, index=False)

    frame = np.zeros((200, 400, 3), dtype=np.uint8)
    monkeypatch.setattr(dg, "_open_video", lambda p: _FakeCap(frame, total=3))
    monkeypatch.setattr(dg, "_init_detection_runner", lambda params: None)
    monkeypatch.setattr(
        dg,
        "_detect_records_for_frames",
        lambda runner, frames, params, level: {
            0: dg.records_from_obb_result(_seg_obb_result(), level)
        },
    )

    manifest = dg.export_dataset(
        video_path=str(video),
        csv_path=str(csv_path),
        frame_ids=[0],
        output_dir=str(tmp_path / "out"),
        dataset_name="round",
        class_name="ant",
        params={"DETECTION_METHOD": "yolo_obb", "YOLO_OBB_DIRECT_TASK": "obb"},
        include_context=False,
    )

    assert manifest["native_level"] == "obb"
    assert {r["level"] for r in manifest["roots"]} == {"obb", "aabb"}
    round_dir = Path(manifest["round_dir"])
    assert (round_dir / "obb" / "labels").is_dir()
    assert (round_dir / "aabb" / "labels").is_dir()
    assert not (round_dir / "polygon").exists()


def test_score_frame_zero_count():
    """Test scoring when no objects detected."""
    params = {
        "MAX_TARGETS": 4,
        "DATASET_CONF_THRESHOLD": 0.5,
        "METRIC_COUNT_MISMATCH": True,
    }

    scorer = FrameQualityScorer(params)

    detection_data = {
        "confidences": [],
        "count": 0,
    }

    score = scorer.score_frame(frame_id=0, detection_data=detection_data)

    # Zero detections should be highly problematic
    assert score > 0.2


def test_score_normalization():
    """Test that scores are in reasonable range."""
    params = {
        "MAX_TARGETS": 4,
        "DATASET_CONF_THRESHOLD": 0.7,
        "METRIC_LOW_CONFIDENCE": True,
        "METRIC_COUNT_MISMATCH": True,
        "METRIC_HIGH_ASSIGNMENT_COST": True,
    }

    scorer = FrameQualityScorer(params)

    detection_data = {
        "confidences": [0.1, 0.2],
        "count": 2,
    }
    tracking_data = {
        "assignment_costs": [100, 120],
    }

    score = scorer.score_frame(
        frame_id=0, detection_data=detection_data, tracking_data=tracking_data
    )

    # Score should be finite and positive
    assert 0 <= score <= 5  # Allow some headroom for compounded scores


def test_multiple_frames_independent():
    """Test that scoring multiple frames maintains independence."""
    params = {
        "MAX_TARGETS": 4,
        "DATASET_CONF_THRESHOLD": 0.5,
        "METRIC_LOW_CONFIDENCE": True,
    }

    scorer = FrameQualityScorer(params)

    score1 = scorer.score_frame(
        frame_id=0,
        detection_data={"confidences": [0.3, 0.4, 0.5, 0.6], "count": 4},
    )
    score2 = scorer.score_frame(
        frame_id=1,
        detection_data={"confidences": [0.9, 0.8, 0.85, 0.87], "count": 4},
    )

    assert score1 != score2
    assert score1 > score2


def test_empty_confidences_list():
    """Test handling of empty confidence list."""
    params = {
        "MAX_TARGETS": 4,
        "DATASET_CONF_THRESHOLD": 0.5,
        "METRIC_LOW_CONFIDENCE": True,
    }

    scorer = FrameQualityScorer(params)

    detection_data = {
        "confidences": [],
        "count": 0,
    }

    score = scorer.score_frame(frame_id=0, detection_data=detection_data)

    assert isinstance(score, (int, float))


def test_low_confidence_uses_frame_average_not_minimum():
    """Low-confidence score should use average confidence across detections."""
    params = {
        "MAX_TARGETS": 4,
        "DATASET_CONF_THRESHOLD": 0.5,
        "METRIC_LOW_CONFIDENCE": True,
        "METRIC_COUNT_MISMATCH": False,
    }

    scorer = FrameQualityScorer(params)

    detection_data = {
        "confidences": [0.1, 0.9, 0.9, 0.9],
        "count": 4,
    }

    score = scorer.score_frame(frame_id=0, detection_data=detection_data)

    assert score == 0.0


def _ellipse_shape(width: float, height: float) -> tuple[float, float]:
    area = np.pi * (width / 2.0) * (height / 2.0)
    aspect_ratio = width / max(height, 1e-6)
    return area, aspect_ratio


def test_score_frame_uses_assignment_confidence_when_costs_missing():
    params = {
        "MAX_TARGETS": 4,
        "DATASET_CONF_THRESHOLD": 0.5,
        "METRIC_LOW_CONFIDENCE": False,
        "METRIC_COUNT_MISMATCH": False,
        "METRIC_HIGH_ASSIGNMENT_COST": True,
        "METRIC_TRACK_LOSS": False,
    }

    scorer = FrameQualityScorer(params)
    score = scorer.score_frame(
        frame_id=0,
        tracking_data={"assignment_confidences": [0.15, 0.25, 0.35]},
    )

    assert score > 0.0
    assert (
        scorer.frame_scores[0]["metrics"]["high_assignment_cost"]["source"]
        == "assignment_confidence"
    )


def test_score_frame_prioritizes_split_detections_over_clean_overcount():
    params = {
        "MAX_TARGETS": 4,
        "DATASET_CONF_THRESHOLD": 0.5,
        "METRIC_LOW_CONFIDENCE": False,
        "METRIC_COUNT_MISMATCH": True,
        "METRIC_FRAGMENTED_DETECTIONS": True,
        "METRIC_HIGH_ASSIGNMENT_COST": False,
        "METRIC_TRACK_LOSS": False,
    }

    scorer = FrameQualityScorer(params)

    normal_shape = _ellipse_shape(20.0, 8.0)
    split_shape = _ellipse_shape(8.0, 4.0)

    split_score = scorer.score_frame(
        frame_id=0,
        detection_data={
            "confidences": [0.9] * 5,
            "count": 5,
            "measurements": [
                np.array([20.0, 20.0, 0.0], dtype=np.float32),
                np.array([80.0, 20.0, 0.0], dtype=np.float32),
                np.array([20.0, 80.0, 0.0], dtype=np.float32),
                np.array([48.0, 48.0, 0.0], dtype=np.float32),
                np.array([54.0, 50.0, 0.0], dtype=np.float32),
            ],
            "shapes": [
                normal_shape,
                normal_shape,
                normal_shape,
                split_shape,
                split_shape,
            ],
            "obb_corners": [],
        },
    )

    clean_overcount_score = scorer.score_frame(
        frame_id=1,
        detection_data={
            "confidences": [0.9] * 5,
            "count": 5,
            "measurements": [
                np.array([20.0, 20.0, 0.0], dtype=np.float32),
                np.array([80.0, 20.0, 0.0], dtype=np.float32),
                np.array([20.0, 80.0, 0.0], dtype=np.float32),
                np.array([80.0, 80.0, 0.0], dtype=np.float32),
                np.array([50.0, 50.0, 0.0], dtype=np.float32),
            ],
            "shapes": [
                normal_shape,
                normal_shape,
                normal_shape,
                normal_shape,
                normal_shape,
            ],
            "obb_corners": [],
        },
    )

    assert split_score > clean_overcount_score
    assert "fragmented_detections" in scorer.frame_scores[0]["metrics"]


def test_frame_quality_scorer_uses_tracker_default_preset_after_refactor():
    """After refactor, scorer routes through data/al/acquisition with tracker_default."""
    from hydra_suite.data.dataset_generation import FrameQualityScorer

    params = {
        "MAX_TARGETS": 4,
        "DATASET_CONF_THRESHOLD": 0.5,
        "REFERENCE_BODY_SIZE": 20.0,
        "METRIC_LOW_CONFIDENCE": True,
        "METRIC_COUNT_MISMATCH": True,
        "METRIC_HIGH_ASSIGNMENT_COST": True,
        "METRIC_TRACK_LOSS": True,
        "METRIC_HIGH_UNCERTAINTY": False,
        "METRIC_FRAGMENTED_DETECTIONS": False,
    }
    scorer = FrameQualityScorer(params)

    scorer.score_frame(
        0,
        detection_data={"confidences": [0.9, 0.9, 0.9, 0.9], "count": 4},
        tracking_data={"lost_tracks": 0},
    )
    scorer.score_frame(
        100,
        detection_data={"confidences": [0.2, 0.3], "count": 2},
        tracking_data={
            "lost_tracks": 2,
            "assignment_confidences": [0.3, 0.3],
        },
    )

    picks = scorer.get_worst_frames(
        max_frames=1, diversity_window=0, probabilistic=False
    )
    assert picks == [100]


def test_frame_quality_scorer_honors_dataset_al_preset_param():
    """Custom DATASET_AL_PRESET param swaps the weight preset."""
    from hydra_suite.data.al.acquisition import PRESETS, AcquisitionWeights
    from hydra_suite.data.dataset_generation import FrameQualityScorer

    params = {
        "MAX_TARGETS": 4,
        "DATASET_CONF_THRESHOLD": 0.5,
        "DATASET_AL_PRESET": "uncertainty_heavy",
        "REFERENCE_BODY_SIZE": 20.0,
    }
    scorer = FrameQualityScorer(params)
    expected = PRESETS["uncertainty_heavy"]
    # uncertainty channel should match the preset's uncertainty weight,
    # renormalized: only enabled-channel weights flow through (uncertainty is
    # enabled by default; nms_instability is always zeroed here), and
    # `._weights` stores the already-normalized weights.
    expected_norm = AcquisitionWeights(
        uncertainty=expected.uncertainty,
        nms_instability=0.0,
        count=expected.count,
        crowd=expected.crowd,
        edge=expected.edge,
        assignment=expected.assignment,
        track_loss=expected.track_loss,
        position_uncertainty=expected.position_uncertainty,
    ).normalized()
    assert scorer._weights.uncertainty == pytest.approx(expected_norm.uncertainty)


def test_frame_quality_scorer_unknown_preset_falls_back_to_tracker_default():
    from hydra_suite.data.al.acquisition import PRESETS, AcquisitionWeights
    from hydra_suite.data.dataset_generation import FrameQualityScorer

    params = {"MAX_TARGETS": 4, "DATASET_AL_PRESET": "no_such_preset"}
    scorer = FrameQualityScorer(params)
    expected = PRESETS["tracker_default"]
    # `._weights` is stored pre-normalized; fragmentation isn't wired into the
    # scorer's weight construction (out of scope here) and METRIC_HIGH_
    # UNCERTAINTY (-> position_uncertainty) defaults off, so both drop out.
    expected_norm = AcquisitionWeights(
        uncertainty=expected.uncertainty,
        nms_instability=0.0,
        count=expected.count,
        crowd=expected.crowd,
        edge=expected.edge,
        fragmentation=0.0,
        assignment=expected.assignment,
        track_loss=expected.track_loss,
        position_uncertainty=0.0,
    ).normalized()
    assert scorer._weights.uncertainty == pytest.approx(expected_norm.uncertainty)


@pytest.mark.parametrize(
    "params,expected",
    [
        (
            {"DETECTION_METHOD": "yolo_obb", "YOLO_OBB_DIRECT_TASK": "segment"},
            GeometryLevel.POLYGON,
        ),
        (
            {"DETECTION_METHOD": "yolo_obb", "YOLO_OBB_DIRECT_TASK": "obb"},
            GeometryLevel.OBB,
        ),
        (
            {"DETECTION_METHOD": "yolo_obb", "YOLO_OBB_DIRECT_TASK": "detect"},
            GeometryLevel.AABB,
        ),
        ({"DETECTION_METHOD": "background_subtraction"}, GeometryLevel.POLYGON),
    ],
)
def test_resolve_native_level(params, expected):
    from hydra_suite.data.dataset_generation import resolve_native_level

    assert resolve_native_level(params) is expected


def test_resolve_native_level_uses_stage2_task_in_sequential_mode():
    from hydra_suite.data.dataset_generation import resolve_native_level

    params = {
        "DETECTION_METHOD": "yolo_obb",
        "YOLO_OBB_MODE": "sequential",
        "YOLO_OBB_DIRECT_TASK": "detect",
        "YOLO_OBB_STAGE2_TASK": "segment",
    }
    assert resolve_native_level(params) is GeometryLevel.POLYGON


def test_resolve_native_level_defaults_to_obb_for_unknown_method():
    from hydra_suite.data.dataset_generation import resolve_native_level

    assert resolve_native_level({"DETECTION_METHOD": "mystery"}) is GeometryLevel.OBB


def test_bgsub_detection_runner_config_enables_native_geometry():
    """The bgsub branch of _init_detection_runner must build a real InferenceConfig
    with bgsub set (not obb) and emit_native_geometry True for POLYGON native level.

    This exercises the exact construction path used by _init_detection_runner,
    mirroring core/tracking/worker.py's live bgsub InferenceConfig construction,
    rather than asserting against a mock.
    """
    from hydra_suite.core.inference.config import (
        BgSubConfig,
        InferenceConfig,
        migrate_runtime_to_tier,
    )

    params = {"DETECTION_METHOD": "background_subtraction"}

    compute_runtime = str(params.get("COMPUTE_RUNTIME", "cpu"))
    raw_tier = str(params.get("RUNTIME_TIER", "") or "").strip().lower()
    runtime_tier = (
        raw_tier
        if raw_tier in {"cpu", "gpu", "gpu_fast"}
        else migrate_runtime_to_tier({compute_runtime})
    )
    bgsub_cfg = BgSubConfig.from_params(params)
    bgsub_cfg.emit_native_geometry = True
    cfg = InferenceConfig(
        obb=None,
        bgsub=bgsub_cfg,
        runtime_tier=runtime_tier,
        detection_batch_size=int(params.get("DETECTION_BATCH_SIZE", 1) or 1),
    )

    assert cfg.obb is None
    assert cfg.bgsub is not None
    assert cfg.bgsub.emit_native_geometry is True
    assert cfg.detection_source == "bgsub"


def test_init_detection_runner_bgsub_sets_emit_native_geometry(monkeypatch):
    """_init_detection_runner itself must build a bgsub-backed InferenceConfig
    (not None, not an obb-only config) and set emit_native_geometry, given
    the corrected implementation (controller ruling: build_inference_config_from_params
    cannot build a bgsub config, so _init_detection_runner must build BgSubConfig
    + InferenceConfig directly, mirroring worker.py:1118-1132).
    """
    from hydra_suite.data import dataset_generation

    captured = {}

    class _FakeRunner:
        def __init__(self, cfg):
            captured["cfg"] = cfg

    monkeypatch.setattr(
        "hydra_suite.core.inference.runner.InferenceRunner", _FakeRunner
    )

    runner = dataset_generation._init_detection_runner(
        {"DETECTION_METHOD": "background_subtraction"}
    )

    assert runner is not None
    cfg = captured["cfg"]
    assert cfg.obb is None
    assert cfg.bgsub is not None
    assert cfg.bgsub.emit_native_geometry is True


def test_edge_score_uses_real_frame_shape():
    """Regression: frame_shape=(1, 1) made edge_score ~1900 instead of [0, 1]."""
    from hydra_suite.data.dataset_generation import FrameQualityScorer

    scorer = FrameQualityScorer({"MAX_TARGETS": 2}, frame_shape=(1080, 1920))
    scorer.score_frame(
        0,
        detection_data={
            "confidences": [0.9, 0.9],
            "count": 2,
            "obb_corners": [
                [[100, 100], [140, 100], [140, 120], [100, 120]],
                [[900, 500], [940, 500], [940, 520], [900, 520]],
            ],
        },
        tracking_data={},
    )
    assert 0.0 <= scorer.frame_signals[0].edge_score <= 1.0


def test_edge_score_is_high_for_a_detection_at_the_border():
    from hydra_suite.data.dataset_generation import FrameQualityScorer

    scorer = FrameQualityScorer({"MAX_TARGETS": 1}, frame_shape=(1080, 1920))
    scorer.score_frame(
        0,
        detection_data={
            "confidences": [0.9],
            "count": 1,
            "obb_corners": [[[0, 0], [40, 0], [40, 20], [0, 20]]],
        },
        tracking_data={},
    )
    assert scorer.frame_signals[0].edge_score > 0.9


def test_bgsub_zeroes_uncertainty_weight_and_renormalizes():
    """All-NaN confidences must not silently dilute the other channels."""
    from hydra_suite.data.dataset_generation import FrameQualityScorer

    scorer = FrameQualityScorer(
        {"MAX_TARGETS": 4, "DETECTION_METHOD": "background_subtraction"},
        frame_shape=(1080, 1920),
    )
    assert scorer._weights.uncertainty == 0.0
    total = sum(
        getattr(scorer._weights, f)
        for f in (
            "uncertainty",
            "nms_instability",
            "count",
            "crowd",
            "fragmentation",
            "edge",
            "assignment",
            "track_loss",
            "position_uncertainty",
        )
    )
    assert total == pytest.approx(1.0)
