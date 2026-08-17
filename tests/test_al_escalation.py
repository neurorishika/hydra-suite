import numpy as np
import pytest

from hydra_suite.core.inference.result import OBBResult
from hydra_suite.data.al.escalation import (
    achievable_levels,
    derive_down,
    records_from_obb_result,
)
from hydra_suite.utils.geometry_levels import GeometryLevel


def _obb_result(polygons=None):
    """One detection: a 40x20 box centred at (100, 50), unrotated."""
    corners = np.array(
        [[[80.0, 40.0], [120.0, 40.0], [120.0, 60.0], [80.0, 60.0]]],
        dtype=np.float32,
    )
    return OBBResult(
        frame_idx=0,
        centroids=np.array([[100.0, 50.0]], dtype=np.float32),
        angles=np.array([0.0], dtype=np.float32),
        sizes=np.array([800.0], dtype=np.float32),
        shapes=np.array([[800.0, 2.0]], dtype=np.float32),
        confidences=np.array([0.9], dtype=np.float32),
        corners=corners,
        detection_ids=OBBResult.make_detection_ids(0, 1),
        class_ids=np.array([3], dtype=np.int64),
        polygons=polygons,
    )


def test_records_from_obb_uses_corners_at_obb_level():
    records = records_from_obb_result(_obb_result(), GeometryLevel.OBB)
    assert len(records) == 1
    assert records[0].level is GeometryLevel.OBB
    assert records[0].class_id == 3
    assert records[0].confidence == pytest.approx(0.9)
    assert records[0].points.shape == (4, 2)


def test_records_from_obb_uses_native_polygons_at_polygon_level():
    poly = [np.array([[80.0, 40.0], [120.0, 45.0], [110.0, 60.0]], dtype=np.float32)]
    records = records_from_obb_result(_obb_result(polygons=poly), GeometryLevel.POLYGON)
    assert records[0].level is GeometryLevel.POLYGON
    assert records[0].points.shape == (3, 2)


def test_records_from_obb_rejects_polygon_level_without_polygons():
    with pytest.raises(ValueError, match="native polygons"):
        records_from_obb_result(_obb_result(), GeometryLevel.POLYGON)


def test_records_from_obb_honours_keep_indices():
    obb = _obb_result()
    assert records_from_obb_result(obb, GeometryLevel.OBB, keep=[]) == []


def test_derive_down_polygon_to_obb_gives_four_points():
    poly = [
        np.array(
            [[80.0, 40.0], [120.0, 40.0], [120.0, 60.0], [80.0, 60.0], [100.0, 62.0]],
            dtype=np.float32,
        )
    ]
    records = records_from_obb_result(_obb_result(polygons=poly), GeometryLevel.POLYGON)
    derived = derive_down(records, GeometryLevel.OBB)
    assert derived[0].level is GeometryLevel.OBB
    assert derived[0].points.shape == (4, 2)
    # minAreaRect must enclose every source point.
    assert derived[0].points[:, 0].min() <= 80.0
    assert derived[0].points[:, 1].max() >= 62.0


def test_derive_down_to_aabb_is_axis_aligned():
    poly = [np.array([[80.0, 40.0], [125.0, 45.0], [110.0, 65.0]], dtype=np.float32)]
    records = records_from_obb_result(_obb_result(polygons=poly), GeometryLevel.POLYGON)
    derived = derive_down(records, GeometryLevel.AABB)
    pts = derived[0].points
    assert derived[0].level is GeometryLevel.AABB
    assert pts.shape == (4, 2)
    assert sorted(set(np.round(pts[:, 0], 4))) == [80.0, 125.0]
    assert sorted(set(np.round(pts[:, 1], 4))) == [40.0, 65.0]


def test_derive_down_refuses_to_escalate_upward():
    records = records_from_obb_result(_obb_result(), GeometryLevel.OBB)
    with pytest.raises(ValueError, match="upward"):
        derive_down(records, GeometryLevel.POLYGON)


def test_derive_down_to_same_level_is_identity():
    records = records_from_obb_result(_obb_result(), GeometryLevel.OBB)
    derived = derive_down(records, GeometryLevel.OBB)
    np.testing.assert_array_equal(derived[0].points, records[0].points)


def test_achievable_levels_is_highest_first():
    assert achievable_levels(GeometryLevel.POLYGON) == [
        GeometryLevel.POLYGON,
        GeometryLevel.OBB,
        GeometryLevel.AABB,
    ]
    assert achievable_levels(GeometryLevel.AABB) == [GeometryLevel.AABB]
