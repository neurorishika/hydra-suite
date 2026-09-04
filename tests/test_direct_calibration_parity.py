"""collect + merge must be run_obb, exactly. This is the load-bearing claim."""

import numpy as np

from hydra_suite.core.inference.stages import obb as obb_stage


def test_collect_then_merge_equals_run_obb(direct_obb_fixture):
    frames, models, config, runtime = direct_obb_fixture
    expected = obb_stage.run_obb(frames, models, config.obb, runtime)
    parts_by_frame, source = obb_stage.collect_obb_parts_by_frame(
        frames, models, config.obb, runtime
    )
    actual = []
    for index, parts in enumerate(parts_by_frame):
        if not parts:
            actual.append(obb_stage._empty_obb_result(index))
            continue
        actual.append(
            obb_stage.merge_per_frame(
                parts,
                source.merge_policy,
                source.merge_plan(index),
                config.obb,
                runtime,
            )
        )
    assert len(expected) == len(actual)
    for want, got in zip(expected, actual):
        if isinstance(want, obb_stage._RawOBBTensors):
            want = obb_stage.materialize_tensors(want, config.obb.raw_detection_cap)
            got = obb_stage.materialize_tensors(got, config.obb.raw_detection_cap)
        assert want.num_detections == got.num_detections
        np.testing.assert_array_equal(want.centroids, got.centroids)
        np.testing.assert_array_equal(want.angles, got.angles)
        np.testing.assert_array_equal(want.confidences, got.confidences)


def test_parts_are_in_frame_coordinates(direct_obb_fixture):
    """Tile-local coordinates would silently mis-score every candidate."""
    frames, models, config, runtime = direct_obb_fixture
    height, width = frames[0].shape[:2]
    parts_by_frame, _source = obb_stage.collect_obb_parts_by_frame(
        frames, models, config.obb, runtime
    )
    for parts in parts_by_frame:
        for part in parts:
            if isinstance(part, obb_stage._RawOBBTensors):
                part = obb_stage.materialize_tensors(part, 0)
            if part.num_detections:
                assert part.centroids[:, 0].max() <= width + 1
                assert part.centroids[:, 1].max() <= height + 1
