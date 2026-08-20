import numpy as np

from hydra_suite.core.canonicalization.geometry import CanonicalGeometry, ClippingStats
from hydra_suite.core.post.synthetic_detections import (
    build_synthetic_obb_result,
    filter_degenerate_tasks,
)


def _task(
    cx=50.0, cy=50.0, w=20.0, h=8.0, theta=0.0, frame_id=1, traj_id=3, interp_index=1
):
    return {
        "frame_id": frame_id,
        "cx": cx,
        "cy": cy,
        "w": w,
        "h": h,
        "theta": theta,
        "traj_id": traj_id,
        "interp_index": interp_index,
        "interp_from": (0, 2),
        "interp_total": 1,
    }


def test_build_synthetic_obb_result_shapes():
    tasks = [_task(traj_id=1), _task(traj_id=2, cx=80.0)]
    obb = build_synthetic_obb_result(frame_idx=1, tasks=tasks)
    assert obb.num_detections == 2
    assert obb.corners.shape == (2, 4, 2)
    assert obb.detection_ids.shape == (2,)
    assert (obb.detection_ids < 0).all()  # negative synthetic ids
    assert obb.detection_ids[0] != obb.detection_ids[1]


def test_build_synthetic_obb_result_empty():
    obb = build_synthetic_obb_result(frame_idx=1, tasks=[])
    assert obb.num_detections == 0
    assert obb.corners.shape == (0, 4, 2)


def test_build_synthetic_obb_result_matches_ellipse_to_obb_corners():
    from hydra_suite.core.individual.geometry import ellipse_to_obb_corners

    task = _task()
    obb = build_synthetic_obb_result(frame_idx=1, tasks=[task])
    expected = ellipse_to_obb_corners(
        task["cx"], task["cy"], task["w"], task["h"], task["theta"]
    )
    np.testing.assert_allclose(obb.corners[0], expected)


def test_filter_degenerate_tasks_drops_zero_length_edge_and_tallies():
    geometry = CanonicalGeometry(canvas_wh=(64, 64), margin=1.3, aspect_ratio=2.0)
    good = _task()
    degenerate = _task(w=0.0, h=0.0, traj_id=99)
    stats = ClippingStats()
    kept = filter_degenerate_tasks([good, degenerate], geometry, stats)
    assert len(kept) == 1
    assert kept[0]["traj_id"] == 3
    assert stats.degenerate_skipped_count == 1


def test_filter_degenerate_tasks_none_clipping_stats_is_safe():
    geometry = CanonicalGeometry(canvas_wh=(64, 64), margin=1.3, aspect_ratio=2.0)
    kept = filter_degenerate_tasks([_task()], geometry, None)
    assert len(kept) == 1
