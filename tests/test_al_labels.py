import numpy as np

from hydra_suite.data.al.escalation import LabelRecord
from hydra_suite.data.al.labels import write_label_file
from hydra_suite.utils.geometry_levels import GeometryLevel, classify_label_line


def _record(points, level, class_id=0):
    return LabelRecord(
        class_id=class_id,
        confidence=0.9,
        points=np.asarray(points, dtype=np.float32),
        level=level,
    )


def test_obb_level_writes_nine_fields(tmp_path):
    path = tmp_path / "f.txt"
    rec = _record([[10, 20], [30, 20], [30, 40], [10, 40]], GeometryLevel.OBB)
    write_label_file(path, [rec], frame_size=(100, 200), level=GeometryLevel.OBB)

    fields = path.read_text().strip().split()
    assert len(fields) == 9
    assert classify_label_line(len(fields)) == "four_point"
    assert fields[0] == "0"
    # x normalized by width=200, y by height=100
    assert float(fields[1]) == 0.05
    assert float(fields[2]) == 0.20


def test_polygon_level_writes_point_list(tmp_path):
    path = tmp_path / "f.txt"
    rec = _record(
        [[10, 20], [30, 20], [30, 40], [20, 50], [10, 40]], GeometryLevel.POLYGON
    )
    write_label_file(path, [rec], frame_size=(100, 200), level=GeometryLevel.POLYGON)

    fields = path.read_text().strip().split()
    assert len(fields) == 11
    assert classify_label_line(len(fields)) == "polygon"


def test_aabb_level_writes_five_field_yolo_detect(tmp_path):
    path = tmp_path / "f.txt"
    rec = _record([[10, 20], [30, 20], [30, 40], [10, 40]], GeometryLevel.AABB)
    write_label_file(path, [rec], frame_size=(100, 200), level=GeometryLevel.AABB)

    fields = path.read_text().strip().split()
    assert len(fields) == 5
    assert classify_label_line(len(fields)) == "aabb"
    assert float(fields[1]) == 0.10  # cx = 20/200
    assert float(fields[2]) == 0.30  # cy = 30/100
    assert float(fields[3]) == 0.10  # w  = 20/200
    assert float(fields[4]) == 0.20  # h  = 20/100


def test_class_id_is_preserved(tmp_path):
    path = tmp_path / "f.txt"
    rec = _record(
        [[10, 20], [30, 20], [30, 40], [10, 40]], GeometryLevel.OBB, class_id=7
    )
    write_label_file(path, [rec], frame_size=(100, 200), level=GeometryLevel.OBB)
    assert path.read_text().split()[0] == "7"


def test_coordinates_are_clamped_to_unit_range(tmp_path):
    path = tmp_path / "f.txt"
    rec = _record([[-50, -50], [400, -50], [400, 400], [-50, 400]], GeometryLevel.OBB)
    write_label_file(path, [rec], frame_size=(100, 200), level=GeometryLevel.OBB)
    values = [float(v) for v in path.read_text().split()[1:]]
    assert all(0.0 <= v <= 1.0 for v in values)


def test_empty_records_writes_empty_file(tmp_path):
    path = tmp_path / "f.txt"
    write_label_file(path, [], frame_size=(100, 200), level=GeometryLevel.OBB)
    assert path.exists()
    assert path.read_text() == ""


def test_polygon_record_derived_to_obb_writes_nine_fields(tmp_path):
    """A POLYGON-native record written at OBB level must produce a 9-field
    line, never a raw point-list line — writing the source's native points
    directly into a file stamped a lower level would silently misrepresent
    the level (scan_source_levels would classify it as "polygon").
    """
    from hydra_suite.data.al.escalation import derive_down

    path = tmp_path / "f.txt"
    rec = _record(
        [[10, 20], [30, 20], [30, 40], [20, 50], [10, 40]], GeometryLevel.POLYGON
    )
    derived = derive_down([rec], GeometryLevel.OBB)
    write_label_file(path, derived, frame_size=(100, 200), level=GeometryLevel.OBB)

    fields = path.read_text().strip().split()
    assert len(fields) == 9
    assert classify_label_line(len(fields)) == "four_point"


# =============================================================================
# FINDING 7: level honesty on the ENCODING path, not just on records
# =============================================================================


def test_polygon_level_never_emits_a_nine_field_line(tmp_path):
    """A native 4-point contour in a `level=polygon` root must not read as OBB.

    `classify_label_line` treats ANY 9-field line as `four_point`, so a 4-point
    polygon written verbatim would make `scan_source_levels` disagree with that
    root's own `source.json`. bgsub is accidentally safe (its `len(c) < 5`
    contour filter); YOLO's `masks.xy` and SAM2 carry no such guarantee.
    """
    path = tmp_path / "f.txt"
    quad = [[10, 20], [30, 20], [30, 40], [10, 45]]
    rec = _record(quad, GeometryLevel.POLYGON)
    write_label_file(path, [rec], frame_size=(100, 200), level=GeometryLevel.POLYGON)
    fields = path.read_text().split()
    assert len(fields) == 11
    assert classify_label_line(len(fields)) == "polygon"
    # Geometry is unchanged: the padding repeats the final vertex.
    coords = np.array([float(v) for v in fields[1:]]).reshape(-1, 2)
    np.testing.assert_allclose(coords[-1], coords[-2])
    np.testing.assert_allclose(
        coords[:4], np.array(quad, dtype=np.float64) / [200.0, 100.0], atol=1e-6
    )


def test_polygon_level_refuses_a_degenerate_two_point_record(tmp_path):
    import pytest

    path = tmp_path / "f.txt"
    rec = _record([[10, 20], [30, 40]], GeometryLevel.POLYGON)
    with pytest.raises(ValueError, match="needs at least 3"):
        write_label_file(
            path, [rec], frame_size=(100, 200), level=GeometryLevel.POLYGON
        )


def test_five_point_polygon_is_untouched(tmp_path):
    path = tmp_path / "f.txt"
    rec = _record(
        [[10, 20], [30, 20], [35, 30], [30, 40], [10, 45]], GeometryLevel.POLYGON
    )
    write_label_file(path, [rec], frame_size=(100, 200), level=GeometryLevel.POLYGON)
    assert len(path.read_text().split()) == 11
