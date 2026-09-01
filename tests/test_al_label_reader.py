import numpy as np

from hydra_suite.data.al.escalation import LabelRecord
from hydra_suite.data.al.labels import read_label_file, write_label_file
from hydra_suite.utils.geometry_levels import GeometryLevel

FRAME = (100, 200)  # (height, width)


def _rec(points, level, class_id=0):
    return LabelRecord(
        class_id=class_id,
        confidence=1.0,
        points=np.asarray(points, dtype=np.float32),
        level=level,
    )


def test_reads_an_obb_line_as_obb(tmp_path):
    path = tmp_path / "a.txt"
    quad = [[10, 10], [50, 10], [50, 40], [10, 40]]
    write_label_file(path, [_rec(quad, GeometryLevel.OBB)], FRAME, GeometryLevel.OBB)

    out = read_label_file(path, FRAME)

    assert len(out) == 1
    assert out[0].level is GeometryLevel.OBB
    assert out[0].class_id == 0
    assert out[0].confidence == 1.0
    np.testing.assert_allclose(
        out[0].points, np.array(quad, dtype=np.float32), atol=0.05
    )


def test_reads_an_aabb_line_as_aabb(tmp_path):
    path = tmp_path / "a.txt"
    quad = [[10, 10], [50, 10], [50, 40], [10, 40]]
    write_label_file(path, [_rec(quad, GeometryLevel.AABB)], FRAME, GeometryLevel.AABB)

    out = read_label_file(path, FRAME)

    assert out[0].level is GeometryLevel.AABB
    np.testing.assert_allclose(
        out[0].points, np.array(quad, dtype=np.float32), atol=0.05
    )


def test_reads_a_five_point_polygon_as_polygon(tmp_path):
    path = tmp_path / "a.txt"
    poly = [[10, 10], [50, 12], [60, 40], [30, 55], [12, 38]]
    write_label_file(
        path, [_rec(poly, GeometryLevel.POLYGON)], FRAME, GeometryLevel.POLYGON
    )

    out = read_label_file(path, FRAME)

    assert out[0].level is GeometryLevel.POLYGON
    assert out[0].points.shape == (5, 2)


def test_a_promoted_quad_round_trips_as_polygon_not_obb(tmp_path):
    """_polygon_points repeats the last vertex; the reader must see 5 points."""
    path = tmp_path / "a.txt"
    quad = [[10, 10], [50, 10], [50, 40], [10, 40]]
    write_label_file(
        path, [_rec(quad, GeometryLevel.OBB)], FRAME, GeometryLevel.POLYGON
    )

    out = read_label_file(path, FRAME)

    assert out[0].level is GeometryLevel.POLYGON
    assert out[0].points.shape == (5, 2)


def test_class_ids_are_preserved(tmp_path):
    path = tmp_path / "a.txt"
    quad = [[10, 10], [50, 10], [50, 40], [10, 40]]
    write_label_file(
        path, [_rec(quad, GeometryLevel.OBB, class_id=3)], FRAME, GeometryLevel.OBB
    )

    assert read_label_file(path, FRAME)[0].class_id == 3


def test_malformed_and_empty_lines_are_skipped(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("\n0 0.1 0.2\n\nnot a label\n0 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2\n")

    out = read_label_file(path, FRAME)

    assert len(out) == 1


def test_a_missing_file_reads_as_empty(tmp_path):
    assert read_label_file(tmp_path / "nope.txt", FRAME) == []
