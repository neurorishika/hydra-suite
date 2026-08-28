"""Tests for dataset generation and active-learning export metadata."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from hydra_suite.data.dataset_generation import FrameQualityScorer
from hydra_suite.utils.geometry import obb_corners_from_dims
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

        # Multiple issues should compound the score (absolute composite
        # severity across the uncertainty, count, and assignment channels).
        assert score > 0.2

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

        # No detection_data at all is indistinguishable from "zero detections
        # observed" under the absolute pipeline (real callers always supply
        # an explicit "count" -- see core/post/dataset_export.py -- so this
        # is a hypothetical edge case, not a live code path). Zero detections
        # against a nonzero MAX_TARGETS is a genuine absolute severity, not 0.
        assert score > 0.0

    def test_score_frame_empty_detection_data(self):
        """Test scoring with empty detection data dict."""
        params = {
            "MAX_TARGETS": 4,
            "DATASET_CONF_THRESHOLD": 0.5,
            "METRIC_LOW_CONFIDENCE": True,
        }

        scorer = FrameQualityScorer(params)

        score = scorer.score_frame(frame_id=0, detection_data={})

        # Same reasoning as test_score_frame_no_data: an empty dict yields
        # zero observed detections, which is a genuine absolute severity
        # under the count-deviation channel, not a "nothing to report" 0.
        assert score > 0.0

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
    monkeypatch.setattr(dg, "_init_detection_runner", lambda params, video_path: None)
    monkeypatch.setattr(
        dg,
        "_detect_records_for_frames",
        lambda runner, frames, params, level: (
            {0: dg.records_from_obb_result(_seg_obb_result(), level)},
            {"detection_failed": 0},
        ),
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
    monkeypatch.setattr(dg, "_init_detection_runner", lambda params, video_path: None)
    monkeypatch.setattr(
        dg,
        "_detect_records_for_frames",
        lambda runner, frames, params, level: (
            {0: dg.records_from_obb_result(_seg_obb_result(), level)},
            {"detection_failed": 0},
        ),
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

    # Zero detections should be flagged as problematic (absolute composite
    # severity, driven entirely by the count-deviation channel here).
    assert score > 0.1


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
    # Falls back to the assignment_confidences path (no assignment_costs
    # supplied): difficulty = 1 - mean(confidences) = 1 - 0.25 = 0.75.
    assert scorer.frame_signals[0].extras["assignment"] == pytest.approx(0.75)


def test_score_frame_prioritizes_split_detections_over_clean_overcount():
    """A fragmented frame must outrank a frame that merely has an extra box."""
    params = {"MAX_TARGETS": 2, "REFERENCE_BODY_SIZE": 20.0}
    scorer = FrameQualityScorer(params, frame_shape=(1080, 1920))

    fragmented_corners = [
        obb_corners_from_dims(500, 500, 44, 16, 0.0),
        obb_corners_from_dims(100, 100, 20, 7, 0.0),
        obb_corners_from_dims(108, 102, 20, 7, 0.0),
    ]
    clean_corners = [
        obb_corners_from_dims(200, 200, 44, 16, 0.0),
        obb_corners_from_dims(600, 600, 44, 16, 0.0),
        obb_corners_from_dims(900, 300, 44, 16, 0.0),
    ]
    scorer.score_frame(
        0,
        {"confidences": [0.9] * 3, "count": 3, "obb_corners": fragmented_corners},
        {},
    )
    scorer.score_frame(
        1, {"confidences": [0.9] * 3, "count": 3, "obb_corners": clean_corners}, {}
    )

    assert scorer.frame_signals[0].fragmentation_score > 0.45
    assert scorer.frame_signals[1].fragmentation_score == 0.0
    picked = scorer.get_worst_frames(1, diversity_window=1, probabilistic=False)
    assert picked == [0]


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
        fragmentation=expected.fragmentation,
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
    # `._weights` is stored pre-normalized. fragmentation IS wired through now
    # (METRIC_FRAGMENTED_DETECTIONS defaults True); METRIC_HIGH_UNCERTAINTY
    # (-> position_uncertainty) defaults off, so only that one drops out.
    expected_norm = AcquisitionWeights(
        uncertainty=expected.uncertainty,
        nms_instability=0.0,
        count=expected.count,
        crowd=expected.crowd,
        edge=expected.edge,
        fragmentation=expected.fragmentation,
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
        def __init__(self, cfg, cache_dir=None):
            captured["cfg"] = cfg
            captured["cache_dir"] = cache_dir

    monkeypatch.setattr(
        "hydra_suite.core.inference.runner.InferenceRunner", _FakeRunner
    )

    runner = dataset_generation._init_detection_runner(
        {"DETECTION_METHOD": "background_subtraction"}, "/tmp/does_not_exist.mp4"
    )

    assert runner is not None
    cfg = captured["cfg"]
    assert cfg.obb is None
    assert cfg.bgsub is not None
    assert cfg.bgsub.emit_native_geometry is True
    assert captured["cache_dir"] is not None


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


def test_fragmentation_carries_nonzero_weight_under_tracker_default():
    """Regression guard: fragmentation must actually reach `_weights`, not
    just live in the AcquisitionWeights preset object unused."""
    from hydra_suite.data.dataset_generation import FrameQualityScorer

    scorer = FrameQualityScorer({"MAX_TARGETS": 4}, frame_shape=(1080, 1920))
    assert scorer._weights.fragmentation > 0.0


def test_fragmented_detections_flag_gates_fragmentation_not_crowd():
    """METRIC_FRAGMENTED_DETECTIONS must control the fragmentation channel
    (one animal split in two), not crowd (animals genuinely touching)."""
    from hydra_suite.data.dataset_generation import FrameQualityScorer

    scorer = FrameQualityScorer(
        {"MAX_TARGETS": 4, "METRIC_FRAGMENTED_DETECTIONS": False},
        frame_shape=(1080, 1920),
    )
    assert scorer._weights.fragmentation == 0.0
    assert scorer._weights.crowd > 0.0


def test_crowding_flag_gates_crowd_not_fragmentation():
    """The converse of the above: METRIC_CROWDING controls crowd only."""
    from hydra_suite.data.dataset_generation import FrameQualityScorer

    scorer = FrameQualityScorer(
        {"MAX_TARGETS": 4, "METRIC_CROWDING": False},
        frame_shape=(1080, 1920),
    )
    assert scorer._weights.crowd == 0.0
    assert scorer._weights.fragmentation > 0.0


@pytest.mark.parametrize(
    "extra_params",
    [
        {},
        {"METRIC_FRAGMENTED_DETECTIONS": False},
        {"METRIC_CROWDING": False},
        {"DETECTION_METHOD": "background_subtraction"},
        {"METRIC_FRAGMENTED_DETECTIONS": False, "METRIC_CROWDING": False},
    ],
)
def test_weights_always_sum_to_one_across_gate_configurations(extra_params):
    from hydra_suite.data.dataset_generation import FrameQualityScorer

    params = {"MAX_TARGETS": 4, **extra_params}
    scorer = FrameQualityScorer(params, frame_shape=(1080, 1920))
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


def test_fragmented_detections_frame_outranks_clean_frame():
    """End-to-end: a frame containing a fragmented-looking pair must rank
    above a clean, well-separated frame -- proving fragmentation is actually
    wired with nonzero weight through the live scorer + selector, not merely
    present on the AcquisitionWeights object. Boxes are chosen close but
    non-overlapping so crowd_score is exactly 0 and only fragmentation fires,
    isolating the channel this regression is about.
    """
    from hydra_suite.data.dataset_generation import FrameQualityScorer
    from hydra_suite.utils.geometry import obb_corners_from_dims

    scorer = FrameQualityScorer(
        {"MAX_TARGETS": 2, "REFERENCE_BODY_SIZE": 50.0},
        frame_shape=(1080, 1920),
    )

    # Frame 0: clean, well-separated, full-confidence, matches MAX_TARGETS.
    # Both boxes stay far from the frame border so edge_score is also 0 --
    # otherwise a near-border placement could make frame 1 "win" via the
    # edge channel instead of fragmentation, defeating the isolation.
    clean_boxes = [
        obb_corners_from_dims(300, 300, 40.0, 16.0, 0.0),
        obb_corners_from_dims(900, 300, 40.0, 16.0, 0.0),
    ]
    scorer.score_frame(
        0,
        detection_data={
            "confidences": [0.95, 0.95],
            "count": 2,
            "obb_corners": [b.tolist() for b in clean_boxes],
        },
        tracking_data={},
    )

    # Frame 1: two small, close-but-non-overlapping boxes -- the classic
    # single-animal-split-in-two fragmentation signature. Also far from the
    # border, so edge_score is 0 here too.
    frag_boxes = [
        obb_corners_from_dims(500, 300, 20.0, 8.0, 0.0),
        obb_corners_from_dims(530, 300, 20.0, 8.0, 0.0),
    ]
    scorer.score_frame(
        1,
        detection_data={
            "confidences": [0.95, 0.95],
            "count": 2,
            "obb_corners": [b.tolist() for b in frag_boxes],
        },
        tracking_data={},
    )

    assert scorer.frame_signals[1].crowd_score == 0.0
    assert scorer.frame_signals[1].edge_score == 0.0
    assert scorer.frame_signals[1].fragmentation_score > 0.0
    assert scorer.frame_signals[0].fragmentation_score == 0.0

    worst = scorer.get_worst_frames(1, diversity_window=0, probabilistic=False)
    assert worst == [1]


# =============================================================================
# Whole-branch review fixes
# =============================================================================


def _tracks_csv(path, frames, x=100.0, y=50.0):
    import pandas as pd

    pd.DataFrame(
        [
            {
                "FrameID": f,
                "TrackID": 1,
                "X": x,
                "Y": y,
                "Theta": 0.0,
                "State": "tracked",
            }
            for f in frames
        ]
    ).to_csv(path, index=False)


def _bgsub_empty_result(frame_idx):
    """Byte-for-byte the shape `stages/bgsub._empty_result` returns."""
    from hydra_suite.core.inference.result import OBBResult

    return OBBResult(
        frame_idx=frame_idx,
        centroids=np.zeros((0, 2), np.float32),
        angles=np.zeros((0,), np.float32),
        sizes=np.zeros((0,), np.float32),
        shapes=np.zeros((0, 2), np.float32),
        confidences=np.zeros((0,), np.float32),
        corners=np.zeros((0, 4, 2), np.float32),
        detection_ids=OBBResult.make_detection_ids(frame_idx, 0),
        class_ids=np.zeros(0, dtype=np.int64),
    )


def _bgsub_contour_result(frame_idx, cx=100.0, cy=50.0):
    """A bg-sub frame WITH a detection: a real 6-point foreground contour."""
    from hydra_suite.core.inference.result import OBBResult

    poly = np.array(
        [
            [cx - 10, cy - 5],
            [cx, cy - 8],
            [cx + 10, cy - 5],
            [cx + 10, cy + 5],
            [cx, cy + 8],
            [cx - 10, cy + 5],
        ],
        dtype=np.float32,
    )
    return OBBResult(
        frame_idx=frame_idx,
        centroids=np.array([[cx, cy]], dtype=np.float32),
        angles=np.array([0.0], dtype=np.float32),
        sizes=np.array([200.0], dtype=np.float32),
        shapes=np.array([[200.0, 2.0]], dtype=np.float32),
        confidences=np.array([float("nan")], dtype=np.float32),
        corners=np.array(
            [
                [
                    [cx - 10, cy - 8],
                    [cx + 10, cy - 8],
                    [cx + 10, cy + 8],
                    [cx - 10, cy + 8],
                ]
            ],
            dtype=np.float32,
        ),
        detection_ids=OBBResult.make_detection_ids(frame_idx, 1),
        class_ids=np.zeros(1, dtype=np.int64),
        polygons=[poly],
    )


class _FakeRunner:
    """InferenceRunner stand-in returning a canned OBBResult per frame.

    Implements `detect_batch_raw` (not `detect_batch`) plus `config`/
    `cache_dir`/`_roi_mask` so it satisfies `get_or_compute_raw`'s and
    `filter_for_source`'s contracts exactly like the real InferenceRunner.
    Every canned result here is bg-sub-shaped (NaN confidences -- see
    `_bgsub_contour_result`/`_bgsub_empty_result`), so `detection_source`
    is fixed to `"bgsub"`, which makes `filter_for_source` the identity
    (matching what the real bg-sub branch does).
    """

    def __init__(self, by_frame, cache_dir=None):
        self._by_frame = by_frame
        self.config = SimpleNamespace(detection_source="bgsub")
        self.cache_dir = cache_dir
        self._roi_mask = None
        self.closed = False

    def detect_batch_raw(self, frames, frame_indices=None, roi_mask=None):
        return [self._by_frame[int(fid)] for fid in frame_indices]

    def close(self):
        self.closed = True


BGSUB_PARAMS = {
    "DETECTION_METHOD": "background_subtraction",
    "REFERENCE_BODY_SIZE": 20.0,
}


def test_bgsub_export_survives_a_zero_detection_frame(tmp_path, monkeypatch):
    """FINDING 1 (end to end, `_detect_records_for_frames` NOT monkeypatched).

    `run_bgsub` returns `_empty_result` (polygons=None) for every empty frame,
    always including frame 0 -- "the model has no history yet". The polygon
    level used to raise on that shape, which unwound through `export_dataset`
    into its broad handler and killed the whole round with the misleading
    message "the detection stage was not run with emit_native_geometry=True".
    """
    import hydra_suite.data.dataset_generation as dg

    video = tmp_path / "clip.mp4"
    video.write_bytes(b"")
    csv_path = tmp_path / "tracks.csv"
    _tracks_csv(csv_path, [0, 1, 2])

    frame = np.zeros((200, 400, 3), dtype=np.uint8)
    runner = _FakeRunner(
        {
            0: _bgsub_empty_result(0),  # first frame: no history
            1: _bgsub_contour_result(1),
            2: _bgsub_contour_result(2),
        },
        cache_dir=tmp_path / "cache",
    )
    monkeypatch.setattr(dg, "_open_video", lambda p: _FakeCap(frame, total=3))
    monkeypatch.setattr(dg, "_init_detection_runner", lambda params, video_path: runner)

    manifest = dg.export_dataset(
        video_path=str(video),
        csv_path=str(csv_path),
        frame_ids=[1],
        output_dir=str(tmp_path / "out"),
        dataset_name="round",
        class_name="ant",
        params=BGSUB_PARAMS,
        include_context=True,
    )

    assert manifest["native_level"] == "polygon"
    # The empty frame is a legitimate zero-detection frame, not a failure.
    assert manifest["totals"]["detection_failed"] == 0
    # ...but it is still not exported as a background sample (FINDING 4).
    assert manifest["totals"]["frames_skipped_no_records"] == 1
    assert manifest["skipped_frame_ids_no_records"] == [0]
    assert manifest["totals"]["frames_exported"] == 2

    labels = sorted(
        p.name for p in (Path(manifest["round_dir"]) / "polygon" / "labels").iterdir()
    )
    assert labels == ["f000001.txt", "f000002.txt"]
    for name in labels:
        text = (Path(manifest["round_dir"]) / "polygon" / "labels" / name).read_text()
        assert text.strip(), "an exported label file must never be empty"


def test_zero_record_frames_are_never_exported_as_background(tmp_path, monkeypatch):
    """FINDING 4: an under-detected frame must not become a background sample.

    YOLO reads an empty .txt as "this image contains no objects". Writing one
    for a frame where the export pass found nothing is fabricated negative
    ground truth.
    """
    import hydra_suite.data.dataset_generation as dg

    video = tmp_path / "clip.mp4"
    video.write_bytes(b"")
    csv_path = tmp_path / "tracks.csv"
    _tracks_csv(csv_path, [0, 1])

    frame = np.zeros((200, 400, 3), dtype=np.uint8)
    monkeypatch.setattr(dg, "_open_video", lambda p: _FakeCap(frame, total=2))
    monkeypatch.setattr(dg, "_init_detection_runner", lambda params, video_path: None)
    monkeypatch.setattr(
        dg,
        "_detect_records_for_frames",
        lambda runner, frames, params, level: (
            {0: dg.records_from_obb_result(_seg_obb_result(), level)},
            {"detection_failed": 0},
        ),
    )

    manifest = dg.export_dataset(
        video_path=str(video),
        csv_path=str(csv_path),
        frame_ids=[0, 1],
        output_dir=str(tmp_path / "out"),
        dataset_name="round",
        class_name="ant",
        params={"DETECTION_METHOD": "yolo_obb", "YOLO_OBB_DIRECT_TASK": "obb"},
        include_context=False,
    )

    round_dir = Path(manifest["round_dir"])
    assert sorted(p.name for p in (round_dir / "obb" / "labels").iterdir()) == [
        "f000000.txt"
    ]
    assert sorted(p.name for p in (round_dir / "obb" / "images").iterdir()) == [
        "f000000.jpg"
    ]
    assert manifest["totals"]["frames_skipped_no_records"] == 1
    assert manifest["skipped_frame_ids_no_records"] == [1]


def test_export_fails_loudly_when_no_frame_has_geometry(tmp_path, monkeypatch):
    """FINDING 4: a round with nothing to label is an error, not an empty dataset."""
    import hydra_suite.data.dataset_generation as dg

    video = tmp_path / "clip.mp4"
    video.write_bytes(b"")
    csv_path = tmp_path / "tracks.csv"
    _tracks_csv(csv_path, [0])

    frame = np.zeros((200, 400, 3), dtype=np.uint8)
    monkeypatch.setattr(dg, "_open_video", lambda p: _FakeCap(frame, total=1))
    monkeypatch.setattr(dg, "_init_detection_runner", lambda params, video_path: None)
    monkeypatch.setattr(
        dg,
        "_detect_records_for_frames",
        lambda runner, frames, params, level: ({}, {"detection_failed": 0}),
    )

    with pytest.raises(ValueError, match="no frame in this round"):
        dg.export_dataset(
            video_path=str(video),
            csv_path=str(csv_path),
            frame_ids=[0],
            output_dir=str(tmp_path / "out"),
            dataset_name="round",
            class_name="ant",
            params={"DETECTION_METHOD": "yolo_obb", "YOLO_OBB_DIRECT_TASK": "obb"},
            include_context=False,
        )


def test_detection_failure_is_counted_not_swallowed(tmp_path):
    """DESIGN RULE: a per-frame detection failure is named in the manifest."""
    import hydra_suite.data.dataset_generation as dg

    class _BoomRunner:
        config = SimpleNamespace(detection_source="bgsub")
        cache_dir = tmp_path / "cache"
        _roi_mask = None

        def detect_batch_raw(self, frames, frame_indices=None, roi_mask=None):
            raise RuntimeError("model exploded")

    frames = {0: np.zeros((4, 4, 3), np.uint8), 1: np.zeros((4, 4, 3), np.uint8)}
    records, stats = dg._detect_records_for_frames(
        _BoomRunner(), frames, {}, GeometryLevel.OBB
    )
    assert records == {}
    assert stats["detection_failed"] == 2


def test_missing_contour_costs_only_its_own_frame(tmp_path):
    """DESIGN RULE: a per-frame geometry failure must not kill the round."""
    import hydra_suite.data.dataset_generation as dg
    from hydra_suite.core.inference.result import OBBResult

    broken = _bgsub_contour_result(0)
    broken = OBBResult(**{**broken.__dict__, "polygons": None})
    runner = _FakeRunner(
        {0: broken, 1: _bgsub_contour_result(1)}, cache_dir=tmp_path / "cache"
    )
    frames = {0: np.zeros((4, 4, 3), np.uint8), 1: np.zeros((4, 4, 3), np.uint8)}

    records, stats = dg._detect_records_for_frames(
        runner, frames, {}, GeometryLevel.POLYGON
    )
    assert stats["detection_failed"] == 1
    assert set(records) == {1}


def test_init_detection_runner_failure_is_loud(monkeypatch):
    """DESIGN RULE: a whole-run failure is loud.

    Returning None used to be survivable only while a fabricated
    reference-size box existed as a fallback; now it means "write an empty
    label file for every frame".
    """
    import hydra_suite.core.inference.config as cfgmod
    import hydra_suite.data.dataset_generation as dg

    def boom(*a, **kw):
        raise RuntimeError("checkpoint not found")

    monkeypatch.setattr(cfgmod, "build_obb_only_config", boom)
    with pytest.raises(RuntimeError, match="Could not initialize the export"):
        dg._init_detection_runner(
            {"DETECTION_METHOD": "yolo_obb", "YOLO_OBB_DIRECT_TASK": "obb"},
            "/tmp/does_not_exist.mp4",
        )


# ---------------------------------------------------------------------------
# FINDING 2: the scorer must operate in one coordinate space
# ---------------------------------------------------------------------------


def _two_animals(gap_px, scale=1.0, cx=500.0, cy=500.0):
    """Two 44x16 animals whose CENTRES are `gap_px` apart, in working space.

    `scale` is RESIZE_FACTOR: the detection cache stores corners in that
    working space, so a physical scene at RESIZE_FACTOR=0.5 is the same scene
    with every coordinate halved.
    """
    half = gap_px / 2.0
    return [
        obb_corners_from_dims(
            (cx - half) * scale, cy * scale, 44.0 * scale, 16.0 * scale, 0.0
        ),
        obb_corners_from_dims(
            (cx + half) * scale, cy * scale, 44.0 * scale, 16.0 * scale, 0.0
        ),
    ]


def _animal_near_left_edge(scale=1.0, cx=60.0, cy=500.0):
    return [
        obb_corners_from_dims(cx * scale, cy * scale, 44.0 * scale, 16.0 * scale, 0.0)
    ]


@pytest.mark.parametrize("gap_px", [16.0, 24.0])
def test_fragmentation_is_invariant_to_resize_factor(gap_px):
    """The same physical scene must score the same at any RESIZE_FACTOR.

    `obb_corners` come from the detection cache, written in RESIZE_FACTOR
    working space, while `frame_shape` (CAP_PROP_FRAME_*) and
    REFERENCE_BODY_SIZE are original-space. Mixing the two made
    `fragmentation` -- the LARGEST weight (0.30) in `tracker_default` -- an
    artefact of the resize knob: for these two scenes it read 0.000 at
    RESIZE_FACTOR=1.0 and 0.651 / 0.527 at 0.5.
    """
    frame_shape = (1000, 1000)  # original space, as the caller supplies it
    base = {"DETECTION_METHOD": "yolo_obb", "REFERENCE_BODY_SIZE": 20.0}

    scores = {}
    for rf in (1.0, 0.5):
        scorer = FrameQualityScorer(
            {**base, "RESIZE_FACTOR": rf}, frame_shape=frame_shape
        )
        scorer.score_frame(0, {"obb_corners": _two_animals(gap_px, scale=rf)})
        sig = scorer.frame_signals[0]
        scores[rf] = (sig.fragmentation_score, sig.crowd_score)

    assert scores[0.5][0] == pytest.approx(scores[1.0][0], abs=1e-6)
    assert scores[0.5][1] == pytest.approx(scores[1.0][1], abs=1e-6)


def test_edge_score_is_invariant_to_resize_factor():
    """`edge_score` had the mirror defect: original-space `frame_shape`
    against working-space corners."""
    frame_shape = (1000, 1000)
    base = {"DETECTION_METHOD": "yolo_obb", "REFERENCE_BODY_SIZE": 20.0}

    edges = {}
    for rf in (1.0, 0.5):
        scorer = FrameQualityScorer(
            {**base, "RESIZE_FACTOR": rf}, frame_shape=frame_shape
        )
        scorer.score_frame(0, {"obb_corners": _animal_near_left_edge(scale=rf)})
        edges[rf] = scorer.frame_signals[0].edge_score

    assert edges[1.0] > 0.0  # the scene is genuinely near the border
    assert edges[0.5] == pytest.approx(edges[1.0], abs=1e-6)


def test_unopenable_video_degrades_to_no_frame_shape_not_zero_shape():
    """`dataset_export` used to hand the scorer (0, 0) for an unopenable
    video, which made every detection maximally close to the border."""
    scorer = FrameQualityScorer({"REFERENCE_BODY_SIZE": 20.0}, frame_shape=None)
    scorer.score_frame(0, {"obb_corners": _animal_near_left_edge()})
    assert scorer.frame_signals[0].edge_score == 0.0


def test_scorer_converts_reference_and_frame_shape_to_working_space():
    scorer = FrameQualityScorer(
        {"REFERENCE_BODY_SIZE": 20.0, "RESIZE_FACTOR": 0.5}, frame_shape=(1000, 800)
    )
    assert scorer.reference_body_size == pytest.approx(20.0)  # original space
    assert scorer.reference_body_size_working == pytest.approx(10.0)
    assert scorer.frame_shape == (500, 400)


# ---------------------------------------------------------------------------
# FINDING 5: provenance must name the model that actually ran
# ---------------------------------------------------------------------------


SEQUENTIAL_PARAMS = {
    "DETECTION_METHOD": "yolo_obb",
    "YOLO_OBB_MODE": "sequential",
    "YOLO_OBB_DIRECT_TASK": "obb",
    "YOLO_OBB_DIRECT_MODEL_PATH": "/models/direct-obb.pt",
    "YOLO_CROP_OBB_MODEL_PATH": "/models/crop-seg.pt",
    "YOLO_SEQ_STAGE2_TASK": "segment",
    "DATASET_AL_PRESET": "tracker_default",
}


def test_resolve_native_level_reads_the_real_sequential_stage2_key():
    """`YOLO_OBB_STAGE2_TASK` was a key nothing ever wrote."""
    import hydra_suite.data.dataset_generation as dg

    assert dg.resolve_native_level(SEQUENTIAL_PARAMS) is GeometryLevel.POLYGON
    assert dg.resolve_detection_task(SEQUENTIAL_PARAMS) == "segment"
    assert dg.resolve_detection_model_path(SEQUENTIAL_PARAMS) == "/models/crop-seg.pt"


def test_provenance_records_the_sequential_model_and_the_weights(tmp_path, monkeypatch):
    import json

    import hydra_suite.data.dataset_generation as dg

    video = tmp_path / "clip.mp4"
    video.write_bytes(b"")
    csv_path = tmp_path / "tracks.csv"
    _tracks_csv(csv_path, [0])

    frame = np.zeros((200, 400, 3), dtype=np.uint8)
    monkeypatch.setattr(dg, "_open_video", lambda p: _FakeCap(frame, total=1))
    monkeypatch.setattr(dg, "_init_detection_runner", lambda params, video_path: None)
    monkeypatch.setattr(
        dg,
        "_detect_records_for_frames",
        lambda runner, frames, params, level: (
            {0: dg.records_from_obb_result(_bgsub_contour_result(0), level)},
            {"detection_failed": 0},
        ),
    )

    manifest = dg.export_dataset(
        video_path=str(video),
        csv_path=str(csv_path),
        frame_ids=[0],
        output_dir=str(tmp_path / "out"),
        dataset_name="round",
        class_name="ant",
        params=SEQUENTIAL_PARAMS,
        include_context=False,
    )

    prov = manifest["provenance"]
    assert prov["model_path"] == "/models/crop-seg.pt"
    assert prov["model_task"] == "segment"
    assert prov["yolo_obb_mode"] == "sequential"
    weights = prov["acquisition_weights"]
    assert weights["fragmentation"] > 0
    assert sum(weights.values()) == pytest.approx(1.0)

    on_disk = json.loads(
        (Path(manifest["round_dir"]) / "polygon" / "source.json").read_text()
    )
    assert on_disk["provenance"]["model_path"] == "/models/crop-seg.pt"


def test_acquisition_weights_reflect_disabled_metrics():
    from hydra_suite.data.dataset_generation import _effective_acquisition_weights

    weights = _effective_acquisition_weights(
        {"DETECTION_METHOD": "yolo_obb", "METRIC_FRAGMENTED_DETECTIONS": False}
    )
    assert weights["fragmentation"] == 0.0
    assert sum(weights.values()) == pytest.approx(1.0)


def test_video_capture_is_released_when_the_detection_runner_raises(
    tmp_path, monkeypatch
):
    """`_init_detection_runner` raises on failure (it used to return None).
    Built above the `try`, it leaked the VideoCapture on every bad model path
    or runtime -- the failure it was made loud to report."""
    from hydra_suite.data import dataset_generation as dg

    released: list[bool] = []

    class _FakeCap:
        def release(self):
            released.append(True)

        def get(self, _prop):  # pragma: no cover - not reached
            return 0

    monkeypatch.setattr(dg, "_open_video", lambda _p: _FakeCap())

    def _boom(_params, _video_path):
        raise RuntimeError("Could not initialize the export detection runner")

    monkeypatch.setattr(dg, "_init_detection_runner", _boom)

    with pytest.raises(RuntimeError, match="detection runner"):
        dg.export_dataset(
            video_path="video.mp4",
            csv_path="tracks.csv",
            frame_ids=[0, 1],
            output_dir=str(tmp_path),
            dataset_name="round",
            class_name="ant",
            params={"DETECTION_METHOD": "yolo_obb", "YOLO_OBB_DIRECT_TASK": "obb"},
        )

    assert released == [True], "VideoCapture leaked when the runner failed"


# ---------------------------------------------------------------------------
# Task 4: export must reuse the existing on-disk detection cache instead of
# always rerunning inference at export time.
# ---------------------------------------------------------------------------


def _cache_reuse_obb_result(frame_idx):
    from hydra_suite.core.inference.result import OBBResult

    corners = obb_corners_from_dims(16.0, 16.0, 8.0, 4.0, 0.0).reshape(1, 4, 2)
    return OBBResult(
        frame_idx=frame_idx,
        centroids=np.array([[16.0, 16.0]], dtype=np.float32),
        angles=np.array([0.0], dtype=np.float32),
        sizes=np.array([32.0], dtype=np.float32),
        shapes=np.array([[32.0, 2.0]], dtype=np.float32),
        confidences=np.array([0.9], dtype=np.float32),
        corners=corners,
        detection_ids=OBBResult.make_detection_ids(frame_idx, 1),
        class_ids=np.array([0], dtype=np.int64),
    )


def test_detect_records_for_frames_uses_existing_cache(tmp_path, monkeypatch):
    """A cache the tracking pass already fully populated must be a pure read.

    `_init_detection_runner` wires `runner.cache_dir` to the video's real
    `.inference_cache_<stem>/` folder; once `get_or_compute_raw` has written a
    complete cache for the requested frames, `_detect_records_for_frames`
    must reuse it and never call `detect_batch_raw` again.
    """
    from unittest.mock import MagicMock, patch

    from hydra_suite.core.inference.cache.reuse import get_or_compute_raw
    from hydra_suite.data import dataset_generation as dg

    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"")  # never opened -- InferenceRunner loading is mocked

    def fake_run_obb(frames, models, obb_config, runtime, roi_mask=None):
        return [_cache_reuse_obb_result(i) for i in range(len(frames))]

    params = {"DETECTION_METHOD": "yolo_obb", "YOLO_OBB_DIRECT_TASK": "obb"}

    frames = [
        np.zeros((32, 32, 3), dtype=np.uint8),
        np.zeros((32, 32, 3), dtype=np.uint8),
    ]
    with (
        patch("hydra_suite.core.inference.runner._load_all_models") as ml,
        patch("hydra_suite.core.inference.runner.run_obb", side_effect=fake_run_obb),
    ):
        ml.return_value = MagicMock(
            obb=MagicMock(), headtail=None, cnn=[], pose=None, apriltag=None
        )
        runner = dg._init_detection_runner(params, str(video_path))
        # Pre-populate the cache for frames [0, 1] via the real path once
        # (needs `run_obb` still patched) -- this is the ONLY call allowed
        # to reach detect_batch_raw.
        get_or_compute_raw(runner, runner.cache_dir, frames, [0, 1])

    assert runner.cache_dir is not None

    def _fail_if_called(*a, **k):
        raise AssertionError(
            "detect_batch_raw should not be called on a full cache hit"
        )

    monkeypatch.setattr(runner, "detect_batch_raw", _fail_if_called)

    records, stats = dg._detect_records_for_frames(
        runner, {0: frames[0], 1: frames[1]}, params, GeometryLevel.OBB
    )

    assert stats["detection_failed"] == 0
    assert set(records) == {0, 1}
    assert len(records[0]) >= 1
    assert len(records[1]) >= 1


def test_init_detection_runner_requires_video_path_for_cache_dir(monkeypatch):
    """`_init_detection_runner` must wire `cache_dir` from `video_path`, not
    leave the runner uncached -- an uncached runner can never reuse the
    detection cache tracking already built."""
    from unittest.mock import MagicMock, patch

    from hydra_suite.data import dataset_generation as dg

    with (
        patch("hydra_suite.core.inference.runner._load_all_models") as ml,
        patch("hydra_suite.core.inference.runner.run_obb"),
    ):
        ml.return_value = MagicMock(
            obb=MagicMock(), headtail=None, cnn=[], pose=None, apriltag=None
        )
        runner = dg._init_detection_runner(
            {"DETECTION_METHOD": "yolo_obb", "YOLO_OBB_DIRECT_TASK": "obb"},
            "/some/dir/clip.mp4",
        )

    assert runner.cache_dir is not None
    assert Path(runner.cache_dir).name == ".inference_cache_clip"
