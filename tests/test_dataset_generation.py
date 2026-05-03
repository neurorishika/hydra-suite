"""Tests for dataset generation and active-learning export metadata."""

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from hydra_suite.data.dataset_generation import FrameQualityScorer, export_dataset


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


def test_export_dataset_writes_active_learning_metadata(tmp_path: Path):
    video_path = tmp_path / "video.mp4"
    csv_path = tmp_path / "tracks.csv"
    out_root = tmp_path / "out"

    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        5.0,
        (64, 48),
    )
    assert writer.isOpened()
    frame = np.full((48, 64, 3), 127, dtype=np.uint8)
    writer.write(frame)
    writer.release()

    pd.DataFrame(
        [
            {
                "FrameID": 0,
                "X": 32.0,
                "Y": 24.0,
                "Theta": 0.5,
                "TrackID": 7,
                "State": "tracked",
            }
        ]
    ).to_csv(csv_path, index=False)

    dataset_dir = export_dataset(
        video_path=video_path,
        csv_path=csv_path,
        frame_ids=[0],
        output_dir=out_root,
        dataset_name="al_test",
        class_name="ant",
        params={
            "DETECTION_METHOD": "background_subtraction",
            "REFERENCE_BODY_SIZE": 10.0,
        },
        include_context=False,
    )

    metadata_path = Path(dataset_dir) / "metadata.json"
    raw = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert int(raw["schema_version"]) == 1
    ann = raw["frames"][0]["annotations"][0]
    assert ann["track_id"] == 7
    assert ann["dimension_source"] == "reference_size"
    assert "canonicalization_support" not in raw
    assert "canonicalization" not in ann
    assert "obb_corners_px" not in ann


def test_measurements_to_detections_scales_back_raw_obb_corners() -> None:
    import hydra_suite.data.dataset_generation as dg

    detections = dg._measurements_to_detections(
        [np.array([25.0, 25.0, 0.0], dtype=np.float32)],
        [(100.0, 1.0)],
        0.5,
        obb_corners=[
            np.array(
                [[20.0, 20.0], [30.0, 20.0], [30.0, 30.0], [20.0, 30.0]],
                dtype=np.float32,
            )
        ],
    )

    match = detections[(50.0, 50.0)]
    assert match["corners"] is not None
    assert np.allclose(
        match["corners"],
        np.array(
            [[40.0, 40.0], [60.0, 40.0], [60.0, 60.0], [40.0, 60.0]],
            dtype=np.float32,
        ),
    )


def test_write_frame_annotations_prefers_matched_detector_corners(tmp_path: Path):
    import hydra_suite.data.dataset_generation as dg

    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    df = pd.DataFrame(
        [
            {
                "FrameID": 0,
                "X": 50.0,
                "Y": 50.0,
                "Theta": 0.0,
                "TrackID": 3,
                "State": "tracked",
            }
        ]
    )
    matched_corners = np.array(
        [[45.0, 25.0], [75.0, 45.0], [55.0, 75.0], [25.0, 55.0]],
        dtype=np.float32,
    )
    yolo_detections = {
        (50.0, 50.0): {
            "width": 40.0,
            "height": 20.0,
            "theta": 1.0,
            "corners": matched_corners,
        }
    }

    _image_filename, label_filename, annotations = dg._write_frame_annotations(
        0,
        frame,
        df,
        yolo_detections,
        {"REFERENCE_BODY_SIZE": 10.0},
        images_dir,
        labels_dir,
        100,
        100,
        1.0,
    )

    text = (labels_dir / label_filename).read_text(encoding="utf-8")
    assert (
        text
        == "0 0.450000 0.250000 0.750000 0.450000 0.550000 0.750000 0.250000 0.550000\n"
    )
    assert annotations[0]["dimension_source"] == "yolo_match"


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
    from hydra_suite.data.al.acquisition import PRESETS
    from hydra_suite.data.dataset_generation import FrameQualityScorer

    params = {
        "MAX_TARGETS": 4,
        "DATASET_CONF_THRESHOLD": 0.5,
        "DATASET_AL_PRESET": "uncertainty_heavy",
        "REFERENCE_BODY_SIZE": 20.0,
    }
    scorer = FrameQualityScorer(params)
    expected = PRESETS["uncertainty_heavy"]
    # uncertainty channel should match the preset's uncertainty weight
    # (only enabled-channel weights flow through; uncertainty is enabled by default).
    assert scorer._weights.uncertainty == expected.uncertainty


def test_frame_quality_scorer_unknown_preset_falls_back_to_tracker_default():
    from hydra_suite.data.al.acquisition import PRESETS
    from hydra_suite.data.dataset_generation import FrameQualityScorer

    params = {"MAX_TARGETS": 4, "DATASET_AL_PRESET": "no_such_preset"}
    scorer = FrameQualityScorer(params)
    expected = PRESETS["tracker_default"]
    assert scorer._weights.uncertainty == expected.uncertainty
