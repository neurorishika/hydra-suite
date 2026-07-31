from pathlib import Path

import pytest

from hydra_suite.training.geometry_levels import (
    GeometryLevel,
    classify_label_line,
    scan_source_levels,
)


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


def _write(labels: Path, name: str, text: str) -> None:
    labels.mkdir(parents=True, exist_ok=True)
    (labels / name).write_text(text, encoding="utf-8")


def test_scan_all_polygon(tmp_path):
    labels = tmp_path / "labels"
    _write(labels, "a.txt", "0 0.1 0.1 0.5 0.1 0.5 0.5 0.1 0.5 0.3 0.7\n")  # 5 pts
    scan = scan_source_levels(labels)
    assert scan.resolved_level is GeometryLevel.POLYGON
    assert scan.is_homogeneous


def test_scan_four_point_uses_intended(tmp_path):
    labels = tmp_path / "labels"
    _write(labels, "a.txt", "0 0.1 0.1 0.5 0.1 0.5 0.5 0.1 0.5\n")  # 4 pts
    assert (
        scan_source_levels(labels, GeometryLevel.OBB).resolved_level
        is GeometryLevel.OBB
    )
    assert (
        scan_source_levels(labels, GeometryLevel.POLYGON).resolved_level
        is GeometryLevel.POLYGON
    )


def test_scan_aabb(tmp_path):
    labels = tmp_path / "labels"
    _write(labels, "a.txt", "0 0.5 0.5 0.2 0.2\n")  # cx cy w h
    scan = scan_source_levels(labels)
    assert scan.resolved_level is GeometryLevel.AABB
    assert scan.is_homogeneous


def test_scan_mixed_polygon_and_fourpoint_blocks(tmp_path):
    labels = tmp_path / "labels"
    _write(labels, "poly.txt", "0 0.1 0.1 0.5 0.1 0.5 0.5 0.1 0.5 0.3 0.7\n")
    _write(labels, "quad.txt", "0 0.1 0.1 0.5 0.1 0.5 0.5 0.1 0.5\n")
    scan = scan_source_levels(labels)
    assert not scan.is_homogeneous
    assert scan.needs_confirmation
    assert "quad.txt" in scan.conflict_files


def test_scan_mixed_resolved_by_confirm(tmp_path):
    labels = tmp_path / "labels"
    _write(labels, "poly.txt", "0 0.1 0.1 0.5 0.1 0.5 0.5 0.1 0.5 0.3 0.7\n")
    _write(labels, "quad.txt", "0 0.1 0.1 0.5 0.1 0.5 0.5 0.1 0.5\n")
    scan = scan_source_levels(labels, confirm_quads_are_polygons=True)
    assert scan.is_homogeneous
    assert scan.resolved_level is GeometryLevel.POLYGON


def test_scan_file_mixing_aabb_and_polygon_blocks(tmp_path):
    labels = tmp_path / "labels"
    _write(
        labels,
        "a.txt",
        "0 0.5 0.5 0.2 0.2\n0 0.1 0.1 0.5 0.1 0.5 0.5 0.1 0.5 0.3 0.7\n",
    )
    scan = scan_source_levels(labels)
    assert not scan.is_homogeneous
    assert "a.txt" in scan.conflict_files


def test_scan_aabb_file_with_obb_file_blocks(tmp_path):
    labels = tmp_path / "labels"
    _write(labels, "box.txt", "0 0.5 0.5 0.2 0.2\n")  # aabb
    _write(labels, "obb.txt", "0 0.1 0.1 0.5 0.1 0.5 0.5 0.1 0.5\n")  # four-point
    scan = scan_source_levels(labels)
    assert not scan.is_homogeneous
    assert (
        not scan.needs_confirmation
    )  # aabb/oriented conflict cannot be confirmed away


def test_scan_malformed_line_blocks(tmp_path):
    labels = tmp_path / "labels"
    _write(labels, "bad.txt", "0 0.1 0.2 0.3\n")  # 4 fields => invalid line
    scan = scan_source_levels(labels)
    assert not scan.is_homogeneous
    assert "bad.txt" in scan.conflict_files


from hydra_suite.detectkit.gui.dialogs.source_validation import (
    resolve_source_level_or_block,
)


def test_resolve_blocks_on_mixed(tmp_path):
    labels = tmp_path / "labels"
    _write(labels, "poly.txt", "0 0.1 0.1 0.5 0.1 0.5 0.5 0.1 0.5 0.3 0.7\n")
    _write(labels, "quad.txt", "0 0.1 0.1 0.5 0.1 0.5 0.5 0.1 0.5\n")
    scan = resolve_source_level_or_block(labels, GeometryLevel.OBB, confirm=False)
    assert not scan.is_homogeneous and scan.needs_confirmation


def test_resolve_confirm_override(tmp_path):
    labels = tmp_path / "labels"
    _write(labels, "poly.txt", "0 0.1 0.1 0.5 0.1 0.5 0.5 0.1 0.5 0.3 0.7\n")
    _write(labels, "quad.txt", "0 0.1 0.1 0.5 0.1 0.5 0.5 0.1 0.5\n")
    scan = resolve_source_level_or_block(labels, GeometryLevel.OBB, confirm=True)
    assert scan.is_homogeneous and scan.resolved_level is GeometryLevel.POLYGON


def test_xal_mode_for_level():
    from hydra_suite.detectkit.gui.panels.dataset_panel import xal_mode_for_level

    assert xal_mode_for_level(GeometryLevel.AABB) == "detect"
    assert xal_mode_for_level(GeometryLevel.OBB) == "obb"
    assert xal_mode_for_level(GeometryLevel.POLYGON) == "segment"
