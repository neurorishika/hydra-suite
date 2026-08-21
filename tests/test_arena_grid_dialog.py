"""Tests for the arena grid generator: pure geometry + a thin dialog wrapper.

``generate_grid_shapes`` is tested directly (no Qt). ``ArenaGridDialog`` is
tested by constructing a real instance under ``QT_QPA_PLATFORM=offscreen``
and reading its state -- never ``exec()``'d, per this repo's "no modal
dialogs in tests" rule (some GUI tests crash the interpreter).
"""

import pytest

from hydra_suite.trackerkit.arena_geometry import generate_grid_shapes
from hydra_suite.trackerkit.gui.dialogs.arena_grid_dialog import ArenaGridDialog


@pytest.fixture(scope="module")
def qt_app():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


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
    dialog.spin_radius.setValue(4)  # diameter 8, matching the old spin_size=8
    dialog.spin_rows.setValue(2)
    dialog.spin_cols.setValue(4)
    dialog.spin_origin_x.setValue(20)
    dialog.spin_origin_y.setValue(30)
    dialog.spin_pitch_x.setValue(15)
    dialog.spin_pitch_y.setValue(12)

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
    dialog.combo_shape_type.setCurrentText("Rectangle")
    shapes = dialog.accepted_shapes()
    assert shapes[0]["type"] == "polygon"
    assert isinstance(shapes[0]["params"][0], list)


# --- MainWindow._on_generate_grid_clicked integration glue (Finding 1, fix
# round 1): the next-free-id computation over roi_shapes, the merge into
# existing shapes, and the mask/label refresh were previously unverified by
# CI. Driven directly on the unbound method with a MagicMock stand-in for
# MainWindow (the ``test_arena_session_assignment.py`` precedent), with
# ``ArenaGridDialog`` itself patched out so no modal dialog is ever
# exec()'d. ---


def _make_mock_main_window(roi_shapes):
    from unittest.mock import MagicMock

    mw = MagicMock()
    mw.roi_shapes = list(roi_shapes)
    mw.roi_base_frame = None
    return mw


def test_generate_grid_handler_offsets_past_existing_arenas_and_merges(
    qapp, monkeypatch
):
    """Existing shapes -- including an EXCLUDE-only high arena id -- must
    both (a) push the generated grid's first id past them and (b) survive
    the merge untouched, and n_arenas_from_shapes must report the combined
    total afterwards."""
    from hydra_suite.trackerkit.engine_params import n_arenas_from_shapes
    from hydra_suite.trackerkit.gui.main_window import MainWindow

    existing = [
        {"type": "circle", "params": [1, 1, 1], "mode": "include", "arena_id": 0},
        # exclude-only "arena" at a HIGH id: must still push next_id past it,
        # even though n_arenas_from_shapes doesn't count it as its own arena.
        {"type": "circle", "params": [9, 9, 1], "mode": "exclude", "arena_id": 5},
    ]
    mw = _make_mock_main_window(existing)

    class _FakeDialog:
        captured_first_arena_id = None

        def __init__(self, parent=None, reference_frame=None, first_arena_id=0):
            _FakeDialog.captured_first_arena_id = first_arena_id
            self._first_arena_id = first_arena_id

        def exec(self):
            from PySide6.QtWidgets import QDialog

            return QDialog.Accepted

        def accepted_shapes(self):
            return generate_grid_shapes(
                1, 2, 50, 50, 100, 100, 40, first_arena_id=self._first_arena_id
            )

    monkeypatch.setattr(
        "hydra_suite.trackerkit.gui.dialogs.arena_grid_dialog.ArenaGridDialog",
        _FakeDialog,
    )

    MainWindow._on_generate_grid_clicked(mw)

    # next id must be pushed past arena 5 (the exclude-only arena), i.e. 6 --
    # NOT past arena 0 alone (which would wrongly give 1).
    assert _FakeDialog.captured_first_arena_id == 6

    # existing shapes survive untouched, generated ones appended after.
    assert mw.roi_shapes[0] == existing[0]
    assert mw.roi_shapes[1] == existing[1]
    assert len(mw.roi_shapes) == 2 + 2
    generated_ids = [s["arena_id"] for s in mw.roi_shapes[2:]]
    assert generated_ids == [6, 7]

    # combined arena count: arena 0 (include) + 2 generated include arenas
    # (6, 7). The exclude-only arena 5 doesn't count as its own arena (it
    # renders nothing on its own), matching n_arenas_from_shapes semantics.
    assert n_arenas_from_shapes(mw.roi_shapes) == 3

    mw._generate_combined_roi_mask.assert_not_called()  # no roi_base_frame
    mw.arena_panel.set_shapes.assert_called_with(mw.roi_shapes)
    mw._update_animals_per_arena_total_label.assert_called_once()
    mw.update_roi_preview.assert_called_once()


# --- Task 10: shape choice, rotation, spacing floors, extent caps, shared
# preview renderer ---


def _dialog(app, width=640, height=480):
    from PySide6.QtGui import QImage

    frame = QImage(width, height, QImage.Format_RGB888)
    frame.fill(0)
    return ArenaGridDialog(reference_frame=frame, first_arena_id=0)


def test_dialog_starts_at_one_by_one(qt_app):
    dialog = _dialog(qt_app)
    assert dialog.spin_rows.value() == 1
    assert dialog.spin_cols.value() == 1


def test_spacing_controls_hidden_at_one_by_one(qt_app):
    """Spacing is meaningless with a single arena, so it must not be shown."""
    dialog = _dialog(qt_app)
    assert dialog.spin_pitch_x.isVisibleTo(dialog) is False
    assert dialog.spin_pitch_y.isVisibleTo(dialog) is False


def test_spacing_controls_appear_once_a_column_is_added(qt_app):
    dialog = _dialog(qt_app)
    dialog.spin_cols.setValue(2)
    assert dialog.spin_pitch_x.isVisibleTo(dialog) is True


def test_spacing_defaults_to_the_non_overlapping_minimum(qt_app):
    dialog = _dialog(qt_app)
    dialog.combo_shape_type.setCurrentText("Circle")
    dialog.spin_radius.setValue(30)
    dialog.spin_cols.setValue(2)
    assert dialog.spin_pitch_x.value() == 60
    assert dialog.spin_pitch_x.minimum() == 60


def test_rectangle_spacing_uses_width_and_height_separately(qt_app):
    dialog = _dialog(qt_app)
    dialog.combo_shape_type.setCurrentText("Rectangle")
    dialog.spin_width.setValue(40)
    dialog.spin_height.setValue(20)
    dialog.spin_rows.setValue(2)
    dialog.spin_cols.setValue(2)
    assert dialog.spin_pitch_x.minimum() == 40
    assert dialog.spin_pitch_y.minimum() == 20


def test_circle_shows_radius_only(qt_app):
    dialog = _dialog(qt_app)
    dialog.combo_shape_type.setCurrentText("Circle")
    assert dialog.spin_radius.isVisibleTo(dialog) is True
    assert dialog.spin_width.isVisibleTo(dialog) is False


def test_rectangle_shows_width_and_height(qt_app):
    dialog = _dialog(qt_app)
    dialog.combo_shape_type.setCurrentText("Rectangle")
    assert dialog.spin_width.isVisibleTo(dialog) is True
    assert dialog.spin_height.isVisibleTo(dialog) is True
    assert dialog.spin_radius.isVisibleTo(dialog) is False


def test_rotation_range_and_step(qt_app):
    dialog = _dialog(qt_app)
    assert dialog.spin_rotation.minimum() == -45.0
    assert dialog.spin_rotation.maximum() == 45.0
    assert dialog.spin_rotation.singleStep() == 0.5


def test_rotation_slider_and_spinbox_stay_in_sync(qt_app):
    dialog = _dialog(qt_app)
    dialog.spin_rotation.setValue(12.5)
    assert dialog.slider_rotation.value() == 25  # half-degree ticks
    dialog.slider_rotation.setValue(-30)
    assert dialog.spin_rotation.value() == -15.0


def test_rows_and_cols_capped_so_centres_stay_in_frame(qt_app):
    dialog = _dialog(qt_app, width=400, height=300)
    dialog.spin_origin_x.setValue(50)
    dialog.spin_origin_y.setValue(50)
    dialog.spin_cols.setValue(2)
    dialog.spin_pitch_x.setValue(100)
    dialog.spin_pitch_y.setValue(100)
    assert dialog.spin_cols.maximum() == 4
    assert dialog.spin_rows.maximum() == 3


def test_generated_grid_never_overlaps_at_default_spacing(qt_app):
    from hydra_suite.trackerkit.arena_geometry import overlapping_arena_pairs

    dialog = _dialog(qt_app)
    dialog.combo_shape_type.setCurrentText("Circle")
    dialog.spin_radius.setValue(20)
    dialog.spin_rows.setValue(3)
    dialog.spin_cols.setValue(3)
    shapes = dialog.accepted_shapes()
    assert overlapping_arena_pairs(shapes, 640, 480) == []


def test_accepted_shapes_carry_the_rotation(qt_app):
    dialog = _dialog(qt_app)
    dialog.spin_cols.setValue(2)
    dialog.spin_rotation.setValue(45.0)
    shapes = dialog.accepted_shapes()
    assert shapes[0]["params"][1] != shapes[1]["params"][1]
