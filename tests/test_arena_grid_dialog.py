"""Tests for the arena grid generator: pure geometry + a thin dialog wrapper.

``generate_grid_shapes`` is tested directly (no Qt). ``ArenaGridDialog`` is
tested by constructing a real instance under ``QT_QPA_PLATFORM=offscreen``
and reading its state -- never ``exec()``'d, per this repo's "no modal
dialogs in tests" rule (some GUI tests crash the interpreter).
"""

import pytest

from hydra_suite.trackerkit.gui.dialogs.arena_grid_dialog import (
    ArenaGridDialog,
    generate_grid_shapes,
)


def test_grid_produces_rows_times_cols_shapes():
    shapes = generate_grid_shapes(2, 3, 50, 50, 100, 100, 40)
    assert len(shapes) == 6


def test_arena_ids_are_sequential_and_unique():
    shapes = generate_grid_shapes(2, 3, 50, 50, 100, 100, 40)
    assert [s["arena_id"] for s in shapes] == [0, 1, 2, 3, 4, 5]


def test_shapes_are_row_major():
    """Ids increase across a row before moving down -- matches well-plate naming."""
    shapes = generate_grid_shapes(2, 2, 50, 50, 100, 100, 40)
    centers = [(s["params"][0], s["params"][1]) for s in shapes]
    assert centers == [(50, 50), (150, 50), (50, 150), (150, 150)]


def test_circle_geometry_uses_radius_half_of_size():
    shapes = generate_grid_shapes(1, 1, 50, 50, 100, 100, 40)
    assert shapes[0]["type"] == "circle"
    assert shapes[0]["params"] == [50, 50, 20]


def test_polygon_grid_emits_four_corner_squares():
    shapes = generate_grid_shapes(1, 1, 50, 50, 100, 100, 40, shape_type="polygon")
    assert shapes[0]["type"] == "polygon"
    assert shapes[0]["params"] == [[30, 30], [70, 30], [70, 70], [30, 70]]


def test_first_arena_id_offsets_the_numbering():
    shapes = generate_grid_shapes(1, 2, 50, 50, 100, 100, 40, first_arena_id=7)
    assert [s["arena_id"] for s in shapes] == [7, 8]


def test_all_shapes_are_include_mode():
    shapes = generate_grid_shapes(2, 2, 50, 50, 100, 100, 40)
    assert all(s["mode"] == "include" for s in shapes)


def test_ninety_six_well_layout_is_supported():
    shapes = generate_grid_shapes(8, 12, 30, 30, 25, 25, 20)
    assert len(shapes) == 96
    assert len({s["arena_id"] for s in shapes}) == 96


# --- Asymmetric-parameter tests: catch row/col swaps and pitch off-by-ones ---


def test_asymmetric_rows_cols_not_confused():
    """3 rows x 5 cols must NOT equal 5 rows x 3 cols -- a square grid can't
    catch a rows/cols swap bug, so use an asymmetric shape here."""
    shapes = generate_grid_shapes(3, 5, 0, 0, 10, 10, 4)
    assert len(shapes) == 15
    # last shape (row=2, col=4) must be at x=40, y=20 -- NOT x=20, y=40.
    last = shapes[-1]
    assert last["params"][0] == 40
    assert last["params"][1] == 20


def test_asymmetric_pitch_x_y_not_confused():
    """pitch_x != pitch_y: a swapped pitch would move the wrong axis."""
    shapes = generate_grid_shapes(2, 2, 0, 0, pitch_x=30, pitch_y=7, size=4)
    centers = [(s["params"][0], s["params"][1]) for s in shapes]
    assert centers == [(0, 0), (30, 0), (0, 7), (30, 7)]


def test_pitch_not_equal_to_size_exposes_spacing_off_by_one():
    """pitch (17) deliberately != size (10) so an off-by-one in spacing math
    (e.g. using size instead of pitch, or pitch+1) is visible."""
    shapes = generate_grid_shapes(1, 3, 100, 100, pitch_x=17, pitch_y=17, size=10)
    xs = [s["params"][0] for s in shapes]
    assert xs == [100, 117, 134]


def test_origin_x_y_not_confused():
    """origin_x != origin_y: a swapped origin would misplace the whole grid
    along the wrong axis."""
    shapes = generate_grid_shapes(
        1, 1, origin_x=13, origin_y=91, pitch_x=1, pitch_y=1, size=2
    )
    assert shapes[0]["params"][0] == 13
    assert shapes[0]["params"][1] == 91


def test_polygon_half_size_uses_asymmetric_origin():
    """Non-square-friendly origin/size combo for the polygon branch, so a
    half-size or corner-order bug is visible in x AND y independently."""
    shapes = generate_grid_shapes(
        1,
        1,
        origin_x=11,
        origin_y=23,
        pitch_x=1,
        pitch_y=1,
        size=6,
        shape_type="polygon",
    )
    assert shapes[0]["params"] == [[8, 20], [14, 20], [14, 26], [8, 26]]


@pytest.fixture()
def qapp():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_dialog_accepted_shapes_matches_pure_function(qapp):
    dialog = ArenaGridDialog(parent=None, reference_frame=None, first_arena_id=3)
    dialog.spin_rows.setValue(2)
    dialog.spin_cols.setValue(4)
    dialog.spin_origin_x.setValue(20)
    dialog.spin_origin_y.setValue(30)
    dialog.spin_pitch_x.setValue(15)
    dialog.spin_pitch_y.setValue(12)
    dialog.spin_size.setValue(8)

    expected = generate_grid_shapes(2, 4, 20, 30, 15, 12, 8, first_arena_id=3)
    assert dialog.accepted_shapes() == expected


def test_dialog_first_arena_id_offsets_generated_ids(qapp):
    dialog = ArenaGridDialog(parent=None, reference_frame=None, first_arena_id=42)
    shapes = dialog.accepted_shapes()
    assert shapes[0]["arena_id"] == 42


def test_dialog_shape_type_combo_switches_geometry(qapp):
    dialog = ArenaGridDialog(parent=None, reference_frame=None, first_arena_id=0)
    dialog.spin_rows.setValue(1)
    dialog.spin_cols.setValue(1)
    dialog.combo_shape_type.setCurrentText("polygon")
    shapes = dialog.accepted_shapes()
    assert shapes[0]["type"] == "polygon"
    assert isinstance(shapes[0]["params"][0], list)
