import numpy as np

from hydra_suite.training.geometry_levels import GeometryLevel
from hydra_suite.training.sliced_dataset import (
    label_line_for_level,
    measure_reference_body_px,
    project_to_level,
)


def test_measure_reference_body_px_median_major_axis():
    # Two objects: 40x20 and 80x20 (px) at frame 100x100 -> majors 40, 80 -> median 60.
    def rect_norm(cx, cy, w, h):
        pts = np.array(
            [
                [cx - w / 2, cy - h / 2],
                [cx + w / 2, cy - h / 2],
                [cx + w / 2, cy + h / 2],
                [cx - w / 2, cy + h / 2],
            ],
            dtype=np.float32,
        )
        pts[:, 0] /= 100.0
        pts[:, 1] /= 100.0
        return pts

    labels = [(0, rect_norm(50, 50, 40, 20)), (0, rect_norm(50, 50, 80, 20))]
    ref = measure_reference_body_px(labels, (100, 100))
    assert abs(ref - 60.0) < 1.0


def test_project_to_level_aabb_from_polygon():
    poly = np.array([[0.1, 0.1], [0.5, 0.2], [0.4, 0.6], [0.05, 0.4]], dtype=np.float32)
    aabb = project_to_level(poly, GeometryLevel.AABB)
    assert aabb.shape == (4, 2)
    assert abs(aabb[:, 0].min() - 0.05) < 1e-4
    assert abs(aabb[:, 0].max() - 0.5) < 1e-4


def test_project_to_level_obb_returns_four_corners():
    poly = np.array(
        [[0.1, 0.1], [0.5, 0.1], [0.5, 0.3], [0.1, 0.3], [0.3, 0.35]], dtype=np.float32
    )
    obb = project_to_level(poly, GeometryLevel.OBB)
    assert obb.shape == (4, 2)


def test_project_to_level_polygon_keeps_contour():
    poly = np.array([[0.1, 0.1], [0.5, 0.1], [0.3, 0.5]], dtype=np.float32)
    out = project_to_level(poly, GeometryLevel.POLYGON)
    assert np.allclose(out, poly)


def test_label_line_field_counts():
    aabb = np.array([[0.1, 0.1], [0.3, 0.1], [0.3, 0.3], [0.1, 0.3]], dtype=np.float32)
    assert len(label_line_for_level(2, aabb, GeometryLevel.AABB).split()) == 5
    assert len(label_line_for_level(0, aabb, GeometryLevel.OBB).split()) == 9
    tri = np.array([[0.1, 0.1], [0.3, 0.1], [0.2, 0.4]], dtype=np.float32)
    assert len(label_line_for_level(1, tri, GeometryLevel.POLYGON).split()) == 7
