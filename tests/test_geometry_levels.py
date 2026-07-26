import pytest

from hydra_suite.training.geometry_levels import GeometryLevel, classify_label_line


def test_level_ordering_and_labels():
    assert GeometryLevel.AABB < GeometryLevel.OBB < GeometryLevel.POLYGON
    assert GeometryLevel.AABB.label == "aabb"
    assert GeometryLevel.OBB.label == "obb"
    assert GeometryLevel.POLYGON.label == "polygon"
    assert GeometryLevel.from_str("Polygon") is GeometryLevel.POLYGON


def test_from_str_rejects_unknown():
    with pytest.raises(ValueError):
        GeometryLevel.from_str("blob")


@pytest.mark.parametrize(
    "field_count,expected",
    [
        (5, "aabb"),  # class + cx cy w h
        (9, "four_point"),  # class + 8 coords (obb OR quad polygon)
        (7, "polygon"),  # class + 3 points
        (11, "polygon"),  # class + 5 points
        (13, "polygon"),  # class + 6 points
        (4, "invalid"),
        (8, "invalid"),  # even field count => odd coord count
        (1, "invalid"),
    ],
)
def test_classify_label_line(field_count, expected):
    assert classify_label_line(field_count) == expected
