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
