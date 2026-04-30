from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hydra_suite.core.identity.classification.cnn import (
    ClassPrediction,
    CNNIdentityCache,
)
from hydra_suite.core.identity.properties.cache import IndividualPropertiesCache
from hydra_suite.core.identity.properties.detected_cache import DetectedPropertiesCache
from hydra_suite.core.identity.properties.export import (
    DETECTED_HEADING_COLUMNS,
    POSE_SUMMARY_COLUMNS,
    augment_trajectories_with_detected_apriltag_cache,
    augment_trajectories_with_detected_apriltag_df,
    augment_trajectories_with_detected_cnn_cache,
    augment_trajectories_with_detected_properties_cache,
    augment_trajectories_with_pose_cache,
    build_detected_apriltag_lookup_dataframe,
    merge_interpolated_cnn_df,
    merge_interpolated_pose_df,
)


def test_augment_trajectories_with_pose_cache_merges_by_frame_and_detection(tmp_path):
    cache_path = tmp_path / "props.npz"
    writer = IndividualPropertiesCache(str(cache_path), mode="w")
    writer.add_frame(
        10,
        [101.0, 102.0],
        pose_keypoints=[
            np.array([[1, 2, 0.9], [3, 4, 0.8]], dtype=np.float32),
            None,
        ],
    )
    writer.add_frame(
        11,
        [201.0],
        pose_keypoints=[np.array([[7, 8, 0.5]], dtype=np.float32)],
    )
    writer.save(
        metadata={
            "individual_properties_id": "abc",
            "pose_keypoint_names": ["head", "tail"],
        }
    )
    writer.close()

    trajectories = pd.DataFrame(
        [
            {"FrameID": 10, "DetectionID": 101, "TrajectoryID": 1, "X": 1, "Y": 2},
            {"FrameID": 10, "DetectionID": 102, "TrajectoryID": 2, "X": 3, "Y": 4},
            {"FrameID": 11, "DetectionID": 201, "TrajectoryID": 3, "X": 5, "Y": 6},
            {"FrameID": 11, "DetectionID": 999, "TrajectoryID": 4, "X": 7, "Y": 8},
            {"FrameID": 12, "DetectionID": np.nan, "TrajectoryID": 5, "X": 9, "Y": 10},
        ]
    )
    out = augment_trajectories_with_pose_cache(trajectories, str(cache_path))

    for col in POSE_SUMMARY_COLUMNS:
        assert col in out.columns

    # First detection: keypoints [[1, 2, 0.9], [3, 4, 0.8]]
    # Mean conf = (0.9 + 0.8) / 2 = 0.85
    # With min_valid_conf=0.2 (default), both keypoints valid: num_valid=2, valid_fraction=1.0
    first = out.iloc[0]
    assert first["PoseMeanConf"] == pytest.approx(0.85)
    assert first["PoseNumValid"] == 2
    assert first["PoseNumKeypoints"] == 2
    assert first["PoseValidFraction"] == pytest.approx(1.0)
    assert first["PoseKpt_head_X"] == pytest.approx(1.0)
    assert first["PoseKpt_head_Y"] == pytest.approx(2.0)
    assert first["PoseKpt_head_Conf"] == pytest.approx(0.9)
    assert first["PoseKpt_tail_X"] == pytest.approx(3.0)
    assert first["PoseKpt_tail_Y"] == pytest.approx(4.0)
    assert first["PoseKpt_tail_Conf"] == pytest.approx(0.8)

    # Second detection: no keypoints (None)
    second = out.iloc[1]
    assert second["PoseMeanConf"] == pytest.approx(0.0)
    assert np.isnan(second["PoseKpt_head_X"])

    # Third detection: keypoints [[7, 8, 0.5]]
    # Mean conf = 0.5, num_valid=1, valid_fraction=1.0, num_keypoints=1
    third = out.iloc[2]
    assert third["PoseMeanConf"] == pytest.approx(0.5)
    assert third["PoseNumKeypoints"] == 1
    assert third["PoseNumValid"] == 1
    assert third["PoseValidFraction"] == pytest.approx(1.0)

    unmatched = out.iloc[3]
    assert np.isnan(unmatched["PoseMeanConf"])
    assert np.isnan(unmatched["PoseKpt_head_X"])


def test_augment_trajectories_with_pose_cache_requires_detection_columns(tmp_path):
    cache_path = tmp_path / "props_empty.npz"
    writer = IndividualPropertiesCache(str(cache_path), mode="w")
    writer.save(metadata={"individual_properties_id": "abc"})
    writer.close()

    trajectories = pd.DataFrame([{"FrameID": 1, "TrajectoryID": 1, "X": 1, "Y": 2}])
    out = augment_trajectories_with_pose_cache(trajectories, str(cache_path))
    assert list(out.columns) == list(trajectories.columns)


def test_augment_trajectories_with_pose_cache_applies_ignore_posthoc(tmp_path):
    cache_path = tmp_path / "props_ignore.npz"
    writer = IndividualPropertiesCache(str(cache_path), mode="w")
    writer.add_frame(
        3,
        [301.0],
        pose_keypoints=[np.array([[11, 12, 0.9], [13, 14, 0.7]], dtype=np.float32)],
    )
    writer.save(
        metadata={
            "individual_properties_id": "abc",
            "pose_keypoint_names": ["head", "tail"],
        }
    )
    writer.close()

    trajectories = pd.DataFrame(
        [{"FrameID": 3, "DetectionID": 301, "TrajectoryID": 1, "X": 1, "Y": 2}]
    )
    out = augment_trajectories_with_pose_cache(
        trajectories,
        str(cache_path),
        ignore_keypoints=["tail"],
    )

    assert "PoseKpt_head_X" in out.columns
    assert "PoseKpt_tail_X" not in out.columns
    assert out.iloc[0]["PoseKpt_head_X"] == pytest.approx(11.0)
    assert out.iloc[0]["PoseKpt_head_Conf"] == pytest.approx(0.9)


def test_merge_interpolated_pose_fills_only_missing_detection_pose():
    trajectories = pd.DataFrame(
        [
            {
                "FrameID": 1,
                "TrajectoryID": 10,
                "DetectionID": 100,
                "PoseMeanConf": 0.9,
                "PoseValidFraction": 1.0,
                "PoseNumValid": 5,
                "PoseNumKeypoints": 5,
                "PoseKpt_head_X": 1.0,
                "PoseKpt_head_Y": 2.0,
                "PoseKpt_head_Conf": 0.9,
            },
            {
                "FrameID": 2,
                "TrajectoryID": 10,
                "DetectionID": np.nan,
                "PoseMeanConf": np.nan,
                "PoseValidFraction": np.nan,
                "PoseNumValid": np.nan,
                "PoseNumKeypoints": np.nan,
                "PoseKpt_head_X": np.nan,
                "PoseKpt_head_Y": np.nan,
                "PoseKpt_head_Conf": np.nan,
            },
        ]
    )
    interp_pose = pd.DataFrame(
        [
            {
                "frame_id": 1,
                "trajectory_id": 10,
                "PoseMeanConf": 0.2,
                "PoseValidFraction": 0.3,
                "PoseNumValid": 1,
                "PoseNumKeypoints": 5,
                "PoseKpt_head_X": 9.0,
                "PoseKpt_head_Y": 9.0,
                "PoseKpt_head_Conf": 0.2,
            },
            {
                "frame_id": 2,
                "trajectory_id": 10,
                "PoseMeanConf": 0.7,
                "PoseValidFraction": 0.8,
                "PoseNumValid": 4,
                "PoseNumKeypoints": 5,
                "PoseKpt_head_X": 3.0,
                "PoseKpt_head_Y": 4.0,
                "PoseKpt_head_Conf": 0.7,
            },
        ]
    )

    out = merge_interpolated_pose_df(trajectories, interp_pose)

    # Existing detection-keyed pose should be preserved.
    assert out.iloc[0]["PoseMeanConf"] == pytest.approx(0.9)
    assert out.iloc[0]["PoseKpt_head_X"] == pytest.approx(1.0)
    assert out.iloc[0]["PoseKpt_head_Y"] == pytest.approx(2.0)
    assert out.iloc[0]["PoseKpt_head_Conf"] == pytest.approx(0.9)

    # Missing pose for interpolated row should be filled from interpolated table.
    assert out.iloc[1]["PoseMeanConf"] == pytest.approx(0.7)
    assert out.iloc[1]["PoseNumValid"] == 4
    assert out.iloc[1]["PoseKpt_head_X"] == pytest.approx(3.0)
    assert out.iloc[1]["PoseKpt_head_Y"] == pytest.approx(4.0)
    assert out.iloc[1]["PoseKpt_head_Conf"] == pytest.approx(0.7)


def test_augment_trajectories_with_pose_cache_coordinate_scale(tmp_path):
    """When coordinate_scale != 1.0, PoseKpt_*_X/Y are rescaled."""
    cache_path = tmp_path / "props_scale.npz"
    writer = IndividualPropertiesCache(str(cache_path), mode="w")
    writer.add_frame(
        5,
        [501.0],
        pose_keypoints=[np.array([[10, 20, 0.9], [30, 40, 0.8]], dtype=np.float32)],
    )
    writer.save(
        metadata={
            "individual_properties_id": "abc",
            "pose_keypoint_names": ["head", "tail"],
        }
    )
    writer.close()

    trajectories = pd.DataFrame(
        [{"FrameID": 5, "DetectionID": 501, "TrajectoryID": 1, "X": 20, "Y": 40}]
    )

    # No scaling (default)
    out_noscale = augment_trajectories_with_pose_cache(trajectories, str(cache_path))
    assert out_noscale.iloc[0]["PoseKpt_head_X"] == pytest.approx(10.0)
    assert out_noscale.iloc[0]["PoseKpt_head_Y"] == pytest.approx(20.0)
    assert out_noscale.iloc[0]["PoseKpt_tail_X"] == pytest.approx(30.0)
    assert out_noscale.iloc[0]["PoseKpt_tail_Y"] == pytest.approx(40.0)

    # With coordinate_scale=2.0 (simulates RESIZE_FACTOR=0.5 → scale=1/0.5=2)
    out_scaled = augment_trajectories_with_pose_cache(
        trajectories, str(cache_path), coordinate_scale=2.0
    )
    assert out_scaled.iloc[0]["PoseKpt_head_X"] == pytest.approx(20.0)
    assert out_scaled.iloc[0]["PoseKpt_head_Y"] == pytest.approx(40.0)
    assert out_scaled.iloc[0]["PoseKpt_tail_X"] == pytest.approx(60.0)
    assert out_scaled.iloc[0]["PoseKpt_tail_Y"] == pytest.approx(80.0)
    # Confidence should NOT be scaled
    assert out_scaled.iloc[0]["PoseKpt_head_Conf"] == pytest.approx(0.9)
    assert out_scaled.iloc[0]["PoseKpt_tail_Conf"] == pytest.approx(0.8)
    # Summary columns should remain unchanged
    assert out_scaled.iloc[0]["PoseMeanConf"] == pytest.approx(0.85)


def test_augment_trajectories_with_detected_properties_cache_merges_by_detection(
    tmp_path,
):
    cache_path = tmp_path / "detected_props.npz"
    with DetectedPropertiesCache(cache_path, mode="w") as cache:
        cache.add_frame(
            7,
            detection_ids=[70001],
            theta_raw=[0.1],
            theta_resolved=[0.2],
            heading_source=["headtail"],
            heading_directed=[1],
            headtail_heading=[0.2],
            headtail_confidence=[0.92],
            headtail_directed=[1],
        )
        cache.save(metadata={"cache_id": "abc"})

    trajectories = pd.DataFrame(
        [
            {"FrameID": 7, "DetectionID": 70001, "TrajectoryID": 1},
            {"FrameID": 7, "DetectionID": 79999, "TrajectoryID": 2},
        ]
    )
    out = augment_trajectories_with_detected_properties_cache(
        trajectories, str(cache_path)
    )

    for col in DETECTED_HEADING_COLUMNS:
        assert col in out.columns

    assert out.iloc[0]["ThetaRaw"] == pytest.approx(0.1)
    assert out.iloc[0]["ThetaResolved"] == pytest.approx(0.2)
    assert out.iloc[0]["HeadingSource"] == "headtail"
    assert out.iloc[0]["HeadingDirected"] == 1
    assert out.iloc[0]["HeadTailConfidence"] == pytest.approx(0.92)
    assert np.isnan(out.iloc[1]["ThetaRaw"])


def test_augment_trajectories_with_detected_cnn_cache_merges_by_detection(tmp_path):
    cache_path = tmp_path / "cnn_cache.npz"
    cache = CNNIdentityCache(cache_path)
    cache.save(
        5,
        [
            ClassPrediction(
                det_index=0,
                factor_names=("flat",),
                class_names=("alpha",),
                confidences=(0.83,),
            ),
            ClassPrediction(
                det_index=2,
                factor_names=("flat",),
                class_names=(None,),
                confidences=(0.1,),
            ),
        ],
    )
    cache.flush()

    trajectories = pd.DataFrame(
        [
            {"FrameID": 5, "DetectionID": 50000, "TrajectoryID": 1},
            {"FrameID": 5, "DetectionID": 50002, "TrajectoryID": 2},
            {"FrameID": 6, "DetectionID": 60000, "TrajectoryID": 3},
        ]
    )
    out = augment_trajectories_with_detected_cnn_cache(
        trajectories, str(cache_path), label="idA"
    )

    assert out.iloc[0]["CNN_idA_Class"] == "alpha"
    assert out.iloc[0]["CNN_idA_Conf"] == pytest.approx(0.83)
    assert pd.isna(out.iloc[1]["CNN_idA_Class"])
    assert out.iloc[1]["CNN_idA_Conf"] == pytest.approx(0.1)
    assert pd.isna(out.iloc[2]["CNN_idA_Class"])


def test_augment_trajectories_with_detected_cnn_cache_expands_multihead_columns(
    tmp_path,
):
    cache_path = tmp_path / "cnn_cache_multi.npz"
    cache = CNNIdentityCache(cache_path, factor_names=("color", "side"))
    cache.save(
        5,
        [
            ClassPrediction(
                det_index=0,
                factor_names=("color", "side"),
                class_names=("red", "left"),
                confidences=(0.83, 0.74),
            ),
            ClassPrediction(
                det_index=1,
                factor_names=("color", "side"),
                class_names=(None, "right"),
                confidences=(0.1, 0.91),
            ),
        ],
    )
    cache.flush()

    trajectories = pd.DataFrame(
        [
            {"FrameID": 5, "DetectionID": 50000, "TrajectoryID": 1},
            {"FrameID": 5, "DetectionID": 50001, "TrajectoryID": 2},
            {"FrameID": 6, "DetectionID": 60000, "TrajectoryID": 3},
        ]
    )
    out = augment_trajectories_with_detected_cnn_cache(
        trajectories, str(cache_path), label="idA"
    )

    assert out.iloc[0]["CNN_idA_color_Class"] == "red"
    assert out.iloc[0]["CNN_idA_color_Conf"] == pytest.approx(0.83)
    assert out.iloc[0]["CNN_idA_side_Class"] == "left"
    assert out.iloc[0]["CNN_idA_side_Conf"] == pytest.approx(0.74)
    assert pd.isna(out.iloc[1]["CNN_idA_color_Class"])
    assert out.iloc[1]["CNN_idA_color_Conf"] == pytest.approx(0.1)
    assert out.iloc[1]["CNN_idA_side_Class"] == "right"
    assert out.iloc[1]["CNN_idA_side_Conf"] == pytest.approx(0.91)
    assert pd.isna(out.iloc[2]["CNN_idA_color_Class"])
    assert pd.isna(out.iloc[2]["CNN_idA_side_Class"])


def test_merge_interpolated_cnn_df_backfills_multihead_columns():
    trajectories = pd.DataFrame(
        [
            {"FrameID": 7, "TrajectoryID": 3},
            {"FrameID": 8, "TrajectoryID": 4},
        ]
    )
    interp = pd.DataFrame(
        [
            {
                "frame_id": 7,
                "trajectory_id": 3,
                "CNN_idA_color_Class": "red",
                "CNN_idA_color_Conf": 0.92,
                "CNN_idA_side_Class": "left",
                "CNN_idA_side_Conf": 0.81,
            }
        ]
    )

    out = merge_interpolated_cnn_df(trajectories, interp, label="idA")

    assert out.iloc[0]["CNN_idA_color_Class"] == "red"
    assert out.iloc[0]["CNN_idA_color_Conf"] == pytest.approx(0.92)
    assert out.iloc[0]["CNN_idA_side_Class"] == "left"
    assert out.iloc[0]["CNN_idA_side_Conf"] == pytest.approx(0.81)
    assert pd.isna(out.iloc[1]["CNN_idA_color_Class"])
    assert pd.isna(out.iloc[1]["CNN_idA_side_Class"])


# ---------------------------------------------------------------------------
# AprilTag detection-level augmentation
# ---------------------------------------------------------------------------


class FakeTagObservationCache:
    """Minimal fake for TagObservationCache in read mode."""

    def __init__(self, data: dict):
        self._data = data

    def get_frame(self, frame_idx: int) -> dict:
        empty = {
            "tag_ids": np.array([], dtype=np.int32),
            "det_indices": np.array([], dtype=np.int32),
            "hammings": np.array([], dtype=np.int32),
        }
        return self._data.get(frame_idx, empty)

    def get_frame_range(self):
        if not self._data:
            return (0, 0)
        frames = list(self._data.keys())
        return (min(frames), max(frames))

    def close(self):
        pass


def _make_tag_cache(data: dict) -> FakeTagObservationCache:
    return FakeTagObservationCache(data)


def test_build_detected_apriltag_lookup_dataframe_basic():
    cache = _make_tag_cache(
        {
            5: {
                "tag_ids": np.array([0, 1], dtype=np.int32),
                "det_indices": np.array([0, 2], dtype=np.int32),
                "hammings": np.array([0, 1], dtype=np.int32),
            }
        }
    )
    tag_labels = ["ant1", "ant2"]
    df = build_detected_apriltag_lookup_dataframe(cache, tag_labels)

    assert len(df) == 2
    assert set(df.columns) >= {
        "_apt_frame_id",
        "_apt_detection_id",
        "DetectedTagID",
        "DetectedTagLabel",
        "DetectedTagConf",
        "DetectedTagHamming",
    }

    row0 = df[df["_apt_detection_id"] == 5 * 10000 + 0].iloc[0]
    assert row0["DetectedTagID"] == pytest.approx(0.0)
    assert row0["DetectedTagLabel"] == "ant1"
    assert row0["DetectedTagConf"] == pytest.approx(1.0)
    assert row0["DetectedTagHamming"] == pytest.approx(0.0)

    row2 = df[df["_apt_detection_id"] == 5 * 10000 + 2].iloc[0]
    assert row2["DetectedTagID"] == pytest.approx(1.0)
    assert row2["DetectedTagLabel"] == "ant2"
    assert row2["DetectedTagConf"] == pytest.approx(1.0 / 2.0)
    assert row2["DetectedTagHamming"] == pytest.approx(1.0)


def test_build_detected_apriltag_lookup_dataframe_empty_cache():
    cache = _make_tag_cache({})
    df = build_detected_apriltag_lookup_dataframe(cache, ["ant1"])
    assert df.empty


def test_augment_trajectories_with_detected_apriltag_df_join():
    lookup = pd.DataFrame(
        [
            {
                "_apt_frame_id": 10,
                "_apt_detection_id": 10 * 10000 + 1,
                "DetectedTagID": 0.0,
                "DetectedTagLabel": "ant1",
                "DetectedTagConf": 1.0,
                "DetectedTagHamming": 0.0,
            }
        ]
    )
    trajectories = pd.DataFrame(
        [
            {"FrameID": 10, "DetectionID": 10 * 10000 + 1, "TrajectoryID": 1},
            {"FrameID": 10, "DetectionID": 10 * 10000 + 3, "TrajectoryID": 2},
            {"FrameID": 11, "DetectionID": 11 * 10000 + 0, "TrajectoryID": 3},
        ]
    )
    out = augment_trajectories_with_detected_apriltag_df(trajectories, lookup)

    # Row matched to the tag
    matched = out[out["TrajectoryID"] == 1].iloc[0]
    assert matched["DetectedTagID"] == pytest.approx(0.0)
    assert matched["DetectedTagLabel"] == "ant1"
    assert matched["DetectedTagConf"] == pytest.approx(1.0)
    assert matched["DetectedTagHamming"] == pytest.approx(0.0)

    # Row not matched — all NaN
    unmatched = out[out["TrajectoryID"] == 2].iloc[0]
    assert pd.isna(unmatched["DetectedTagID"])
    assert pd.isna(unmatched["DetectedTagLabel"])


def test_augment_trajectories_with_detected_apriltag_cache(tmp_path):
    from hydra_suite.data.tag_observation_cache import TagObservationCache

    cache_path = tmp_path / "tags.npz"
    cache = TagObservationCache(str(cache_path), mode="w")
    cache.add_frame(
        3,
        tag_ids=[0],
        centers_xy=np.array([[50.0, 60.0]], dtype=np.float32),
        corners=np.zeros((1, 4, 2), dtype=np.float32),
        det_indices=[2],
        hammings=[0],
    )
    cache.save()
    cache.close()

    trajectories = pd.DataFrame(
        [
            {"FrameID": 3, "DetectionID": 3 * 10000 + 2, "TrajectoryID": 1},
            {"FrameID": 3, "DetectionID": 3 * 10000 + 5, "TrajectoryID": 2},
        ]
    )
    out = augment_trajectories_with_detected_apriltag_cache(
        trajectories, str(cache_path), tag_labels=["ant1", "ant2"]
    )

    matched = out[out["TrajectoryID"] == 1].iloc[0]
    assert matched["DetectedTagID"] == pytest.approx(0.0)
    assert matched["DetectedTagLabel"] == "ant1"
    assert matched["DetectedTagConf"] == pytest.approx(1.0)
    assert matched["DetectedTagHamming"] == pytest.approx(0.0)

    unmatched = out[out["TrajectoryID"] == 2].iloc[0]
    assert pd.isna(unmatched["DetectedTagID"])
