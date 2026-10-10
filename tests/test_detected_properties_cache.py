from __future__ import annotations

import numpy as np

from hydra_suite.core.individual.properties.detected_cache import (
    DetectedPropertiesCache,
)


def test_detected_properties_cache_roundtrip(tmp_path) -> None:
    cache_path = tmp_path / "detected_props.npz"

    with DetectedPropertiesCache(cache_path, mode="w") as cache:
        cache.add_frame(
            12,
            detection_ids=[120001, 120002],
            theta_raw=[0.1, 0.2],
            theta_resolved=[0.15, 0.25],
            heading_source=["pose", "obb_axis"],
            heading_directed=[1, 0],
            headtail_heading=[0.14, np.nan],
            headtail_confidence=[0.91, 0.0],
            headtail_directed=[1, 0],
        )
        cache.save(metadata={"cache_id": "abc"})

    with DetectedPropertiesCache(cache_path, mode="r") as cache:
        assert cache.is_compatible()
        frame = cache.get_frame(12)

    # Heading schema was renamed (detected_cache.py): ThetaRaw dropped as
    # redundant; ThetaResolved->HeadingResolved; HeadingSource->HeadingMethod;
    # HeadingDirected/HeadTailDirected merged into the boolean HeadingIsDirected;
    # HeadTailConfidence->HeadTailClassifierConf; headtail_heading->HeadTailAngleRad.
    assert frame["detection_ids"] == [120001, 120002]
    assert frame["HeadingResolved"] == [
        0.15000000596046448,
        0.25,
    ] or frame[
        "HeadingResolved"
    ] == [0.15, 0.25]
    assert frame["HeadingMethod"] == ["pose", "obb_axis"]
    assert frame["HeadingIsDirected"] == [True, False]
    assert frame["HeadTailClassifierConf"] == [
        0.9100000262260437,
        0.0,
    ] or frame[
        "HeadTailClassifierConf"
    ] == [0.91, 0.0]


def _write_detected(path, max_targets):
    with DetectedPropertiesCache(path, mode="w") as cache:
        cache.add_frame(
            0,
            detection_ids=[1],
            theta_raw=[0.1],
            theta_resolved=[0.1],
            heading_source=["obb_axis"],
            heading_directed=[0],
            headtail_heading=[np.nan],
            headtail_confidence=[0.0],
            headtail_directed=[0],
        )
        meta = {"cache_id": "x"}
        if max_targets is not None:
            meta["max_targets"] = max_targets
        cache.save(metadata=meta)


def test_detected_props_cache_is_gated_by_n(tmp_path):
    """M3 hardening: final-N artifact, readers can require the same N."""
    path = tmp_path / "d.npz"
    _write_detected(path, 10)
    with DetectedPropertiesCache(path, mode="r") as cache:
        assert cache.is_compatible()
        assert cache.is_compatible(max_targets=10)
        assert not cache.is_compatible(max_targets=20)


def test_export_readers_require_the_runs_n(tmp_path):
    import pandas as pd
    import pytest

    from hydra_suite.core.individual.properties import cache as props
    from hydra_suite.core.individual.properties.export import (
        augment_trajectories_with_detected_properties_cache,
        augment_trajectories_with_pose_cache,
    )

    df = pd.DataFrame({"FrameID": [0], "DetectionID": [1.0], "TrajectoryID": [0]})
    det = tmp_path / "d.npz"
    _write_detected(det, 10)
    augment_trajectories_with_detected_properties_cache(df, str(det), max_targets=10)
    with pytest.raises(RuntimeError, match="Incompatible"):
        augment_trajectories_with_detected_properties_cache(
            df, str(det), max_targets=20
        )

    pose = tmp_path / "p.npz"
    w = props.IndividualPropertiesCache(str(pose), mode="w")
    w.add_frame(0, [1.0], pose_keypoints=[np.zeros((2, 3), np.float32)])
    w.save(metadata={"max_targets": 10, "pose_keypoint_names": ["a", "b"]})
    with pytest.raises(RuntimeError, match="Incompatible"):
        augment_trajectories_with_pose_cache(df, str(pose), max_targets=20)
    augment_trajectories_with_pose_cache(df, str(pose), max_targets=10)
