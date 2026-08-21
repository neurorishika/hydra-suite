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

    # The dialog's "Arena 1 Centre" field is passed straight through to
    # generate_grid_shapes, which also wants the CENTRE -- no conversion.
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

        def current_params(self):
            return {"shape_type": "Circle", "radius": 40}

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
    # Fix Wave 6 gives this 400x300 frame a much bigger default circle
    # (radius ~70, so a default 140px min-pitch floor) than this test's
    # pitch=100 wants -- pin a small radius explicitly so the floor doesn't
    # clamp the pitch values this test is actually exercising.
    dialog.spin_radius.setValue(10)
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


# --- Fix Wave 5: "Arena 1 Centre" is the arena's CENTRE, not its bounding-box
# corner -- generate_grid_shapes/max_grid_extent already want the centre, so
# the dialog passes the spinbox values straight through with no conversion.
# (Fix Wave 3 had this backwards based on a misreading of the original bug
# report; the user has since confirmed centre semantics are correct.) ---


def test_top_left_position_zero_puts_circle_bounding_box_at_origin(qt_app):
    """Radius=20 (diameter 40) at Centre=(0, 0), 1x1 grid: the circle's
    centre must be exactly (0, 0)."""
    dialog = _dialog(qt_app)
    dialog.combo_shape_type.setCurrentText("Circle")
    dialog.spin_radius.setValue(20)
    dialog.spin_origin_x.setValue(0)
    dialog.spin_origin_y.setValue(0)
    shapes = dialog.accepted_shapes()
    assert shapes[0]["params"] == [0, 0, 20]


def test_top_left_position_zero_puts_rectangle_bounding_box_at_origin(qt_app):
    """Width=40, Height=20 at Centre=(0, 0): the polygon's min-x/min-y
    vertex must be half the width/height back from the centre."""
    dialog = _dialog(qt_app)
    dialog.combo_shape_type.setCurrentText("Rectangle")
    dialog.spin_width.setValue(40)
    dialog.spin_height.setValue(20)
    dialog.spin_origin_x.setValue(0)
    dialog.spin_origin_y.setValue(0)
    shapes = dialog.accepted_shapes()
    polygon = shapes[0]["params"]
    min_x = min(pt[0] for pt in polygon)
    min_y = min(pt[1] for pt in polygon)
    assert (min_x, min_y) == (-20, -10)


def test_top_left_position_nonzero_offset_carries_through_circle(qt_app):
    """Centre=(100, 50), Radius=20 (diameter 40): a non-zero centre must
    pass straight through unchanged, not just the zero case."""
    dialog = _dialog(qt_app)
    dialog.combo_shape_type.setCurrentText("Circle")
    dialog.spin_radius.setValue(20)
    dialog.spin_origin_x.setValue(100)
    dialog.spin_origin_y.setValue(50)
    shapes = dialog.accepted_shapes()
    assert shapes[0]["params"] == [100, 50, 20]


def test_top_left_position_nonzero_offset_carries_through_rectangle(qt_app):
    """Centre=(100, 50), Width=40, Height=20: the polygon's min-x/min-y
    vertex must be half the width/height back from the centre."""
    dialog = _dialog(qt_app)
    dialog.combo_shape_type.setCurrentText("Rectangle")
    dialog.spin_width.setValue(40)
    dialog.spin_height.setValue(20)
    dialog.spin_origin_x.setValue(100)
    dialog.spin_origin_y.setValue(50)
    shapes = dialog.accepted_shapes()
    polygon = shapes[0]["params"]
    min_x = min(pt[0] for pt in polygon)
    min_y = min(pt[1] for pt in polygon)
    assert (min_x, min_y) == (80, 40)


# --- Fix Wave 6 ---
#
# Fix 1: the preview must fit BOTH width and height (a dialog window wider
# than the reference frame's own aspect ratio previously cropped the
# top/bottom, since the zoom was computed from width alone), and a
# resizeEvent handler must re-fit the preview on resize.
#
# Fix 2: default radius/width/height/origin must derive from the reference
# frame's own size (20% of its average dimension), not hardcoded 20/40/50px.
#
# Fix 3: every numeric field (radius, width, height, centre X/Y, rows,
# columns, X/Y spacing) gets a slider alongside its spinbox, like Rotation.


def test_preview_fits_the_full_height_of_a_tall_narrow_frame(qt_app):
    """A reference frame much taller than it is wide, relative to the
    preview label's fixed minimum size (320x240), forces the height to be
    the binding constraint. Under the old width-only zoom calculation this
    frame would render at full scale (zoom clamped to 1.0 because
    target_w / image.width() > 1), producing a scaled image far taller than
    the label -- cropped top/bottom. The new fit-both-dimensions zoom must
    keep the whole frame inside the label's height.
    """
    dialog = _dialog(qt_app, width=100, height=2000)
    dialog._update_preview()
    pixmap = dialog.preview_label.pixmap()
    assert pixmap is not None
    assert pixmap.height() <= dialog.preview_label.height()
    assert pixmap.width() <= dialog.preview_label.width()


def test_resize_event_refits_the_preview(qt_app):
    """There was previously no resizeEvent override at all, so the preview
    never re-fit after the window was resized."""
    dialog = _dialog(qt_app, width=100, height=2000)
    assert hasattr(dialog, "resizeEvent")
    from PySide6.QtCore import QSize
    from PySide6.QtGui import QResizeEvent

    event = QResizeEvent(QSize(500, 500), QSize(320, 240))
    dialog.resizeEvent(event)
    pixmap = dialog.preview_label.pixmap()
    assert pixmap is not None
    assert pixmap.height() <= dialog.preview_label.height()
    assert pixmap.width() <= dialog.preview_label.width()


def test_default_radius_derives_from_reference_frame_average_dimension(qt_app):
    """1000x1000 reference frame: default radius = round(0.20 * 1000) = 200,
    and the default centre (Arena 1 Centre X/Y) equals that same radius so
    arena 1's edge is tangent to the frame's top-left corner."""
    dialog = _dialog(qt_app, width=1000, height=1000)
    assert dialog.spin_radius.value() == 200
    assert dialog.spin_origin_x.value() == 200
    assert dialog.spin_origin_y.value() == 200


def test_default_radius_scales_with_a_different_frame_size(qt_app):
    """2000x500 reference frame: average dimension 1250, default radius =
    round(0.20 * 1250) = 250 -- proportional to the frame, not a fixed
    number."""
    dialog = _dialog(qt_app, width=2000, height=500)
    assert dialog.spin_radius.value() == 250
    assert dialog.spin_origin_x.value() == 250
    assert dialog.spin_origin_y.value() == 250


def test_default_width_and_height_match_the_radius_based_footprint(qt_app):
    dialog = _dialog(qt_app, width=1000, height=1000)
    assert dialog.spin_width.value() == 400
    assert dialog.spin_height.value() == 400


def test_slider_radius_moves_the_spinbox(qt_app):
    dialog = _dialog(qt_app)
    assert hasattr(dialog, "slider_radius")
    dialog.slider_radius.setValue(75)
    assert dialog.spin_radius.value() == 75


def test_spinbox_radius_moves_the_slider(qt_app):
    dialog = _dialog(qt_app)
    dialog.spin_radius.setValue(50)
    assert dialog.slider_radius.value() == 50


def test_spinbox_radius_beyond_slider_range_clamps_the_slider(qt_app):
    """Typing a value outside the slider's range must still work in the
    spinbox -- the slider just clamps to its own end."""
    dialog = _dialog(qt_app, width=1000, height=1000)
    dialog.spin_radius.setValue(dialog.spin_radius.maximum())
    assert dialog.spin_radius.value() == dialog.spin_radius.maximum()
    assert dialog.slider_radius.value() == dialog.slider_radius.maximum()


def test_raising_the_pitch_floor_past_the_slider_max_does_not_clobber_the_spinbox(
    qt_app,
):
    """Regression: _sync_pitch_floors' slider.setMinimum(...) for the pitch
    sliders used to run unblocked. When the new floor exceeds the slider's
    CURRENT maximum, Qt's QSlider.setMinimum raises both the maximum and the
    current value to the new minimum -- which fires valueChanged into the
    paired spinbox (slider.valueChanged is connected to spin.setValue) and
    silently knocks a larger user-typed spinbox value down to the floor.

    Reproduction (1000x1000 frame, 2-column grid): set pitch_x to an
    arbitrarily large 5000, then raise the radius to 900 so the circle
    min-pitch floor becomes 1800 (comfortably above the slider's prior
    maximum). The spinbox must keep the user's 5000 -- only the slider's own
    minimum/value may visibly clamp to the floor.
    """
    dialog = _dialog(qt_app, width=1000, height=1000)
    dialog.spin_cols.setValue(2)
    dialog.spin_pitch_x.setValue(5000)
    assert dialog.spin_pitch_x.value() == 5000

    dialog.spin_radius.setValue(900)  # circle min-pitch floor -> 1800

    assert dialog.spin_pitch_x.value() == 5000
    assert dialog.slider_pitch_x.minimum() == 1800


def test_slider_rows_moves_the_spinbox(qt_app):
    dialog = _dialog(qt_app)
    assert hasattr(dialog, "slider_rows")
    # Small radius/origin so the extent cap doesn't clamp rows.maximum()
    # below the value this test wants to set.
    dialog.spin_radius.setValue(5)
    dialog.spin_origin_x.setValue(0)
    dialog.spin_origin_y.setValue(0)
    dialog.slider_rows.setValue(4)
    assert dialog.spin_rows.value() == 4


def test_spinbox_rows_moves_the_slider(qt_app):
    dialog = _dialog(qt_app)
    dialog.spin_radius.setValue(5)
    dialog.spin_origin_x.setValue(0)
    dialog.spin_origin_y.setValue(0)
    dialog.spin_rows.setValue(6)
    assert dialog.slider_rows.value() == 6


def test_slider_rows_maximum_tracks_the_extent_cap(qt_app):
    """slider_rows.maximum() must move in lockstep with spin_rows.maximum()
    when _sync_extent_caps shrinks it (mirroring
    test_rows_and_cols_capped_so_centres_stay_in_frame)."""
    dialog = _dialog(qt_app, width=400, height=300)
    dialog.spin_radius.setValue(10)  # see test_rows_and_cols_capped... above
    dialog.spin_origin_x.setValue(50)
    dialog.spin_origin_y.setValue(50)
    dialog.spin_cols.setValue(2)
    dialog.spin_pitch_x.setValue(100)
    dialog.spin_pitch_y.setValue(100)
    assert dialog.spin_cols.maximum() == 4
    assert dialog.spin_rows.maximum() == 3
    assert dialog.slider_cols.maximum() == 4
    assert dialog.slider_rows.maximum() == 3


@pytest.mark.parametrize(
    "dialog_w, dialog_h, frame_w, frame_h",
    [
        (1200, 700, 2000, 2000),  # wide dialog, square frame
        (700, 1200, 2000, 2000),  # tall/narrow dialog, square frame
        (1400, 900, 3000, 1200),  # wide dialog, wide/rectangular frame
        (900, 1400, 1200, 3000),  # tall dialog, tall/rectangular frame
    ],
)
def test_preview_pixmap_never_exceeds_the_labels_settled_size(
    qt_app, dialog_w, dialog_h, frame_w, frame_h
):
    """Regression test for Fix Wave 7: the previous fix round set
    preview_label's size policy to Ignored/Ignored (to solve a resize
    hysteresis bug) but gave neither preview_col nor preview_label any
    stretch factor, so the label never actually claimed the extra space a
    QSizePolicy of Ignored merely permits it to take. Combined with
    _update_preview() running at construction time -- before the layout had
    settled into its final geometry -- this produced a pixmap sized for a
    box the label never actually grew into, and QLabel (with no
    setScaledContents) silently clipped the overflow.

    The fix gives preview_col/preview_label real stretch factors and defers
    the first _update_preview() call by one event-loop tick via
    QTimer.singleShot(0, ...). This asserts the exact invariant that was
    violated: the rendered pixmap must fit inside the label's own settled
    width/height in both dimensions, across dialog sizes and reference
    frame aspect ratios.
    """
    from PySide6.QtGui import QImage

    frame = QImage(frame_w, frame_h, QImage.Format_RGB888)
    frame.fill(200)
    dialog = ArenaGridDialog(reference_frame=frame, first_arena_id=0)
    dialog.resize(dialog_w, dialog_h)
    dialog.show()
    # Flush the event loop so the deferred singleShot(0, self._update_preview)
    # from __init__ actually runs against the layout's real, settled geometry
    # (not a mid-construction snapshot).
    qt_app.processEvents()
    qt_app.processEvents()

    pixmap = dialog.preview_label.pixmap()
    assert pixmap is not None
    assert pixmap.width() <= dialog.preview_label.width()
    assert pixmap.height() <= dialog.preview_label.height()

    dialog.close()


# --- Fix Wave 8: initial_params prefill + current_params() reporter ---


def test_initial_params_prefill_overrides_frame_derived_defaults(qapp):
    initial_params = {
        "radius": 77,
        "origin_x": 10,
        "origin_y": 20,
        "rows": 3,
        "cols": 2,
    }
    dialog = ArenaGridDialog(
        parent=None,
        reference_frame=None,
        first_arena_id=0,
        initial_params=initial_params,
    )
    assert dialog.spin_radius.value() == 77
    assert dialog.spin_origin_x.value() == 10
    assert dialog.spin_origin_y.value() == 20
    assert dialog.spin_rows.value() == 3
    assert dialog.spin_cols.value() == 2


def test_current_params_round_trips_into_a_second_dialog(qapp):
    dialog = ArenaGridDialog(parent=None, reference_frame=None, first_arena_id=0)
    dialog.spin_radius.setValue(33)
    dialog.spin_rows.setValue(4)
    dialog.spin_cols.setValue(5)
    dialog.spin_origin_x.setValue(12)
    dialog.spin_origin_y.setValue(34)
    dialog.spin_pitch_x.setValue(50)
    dialog.spin_pitch_y.setValue(60)
    dialog.spin_rotation.setValue(5.5)

    params = dialog.current_params()

    dialog2 = ArenaGridDialog(
        parent=None,
        reference_frame=None,
        first_arena_id=0,
        initial_params=params,
    )
    assert dialog2.current_params() == params
    assert (
        dialog2.combo_shape_type.currentText() == dialog.combo_shape_type.currentText()
    )
    assert dialog2.spin_radius.value() == dialog.spin_radius.value()
    assert dialog2.spin_width.value() == dialog.spin_width.value()
    assert dialog2.spin_height.value() == dialog.spin_height.value()
    assert dialog2.spin_origin_x.value() == dialog.spin_origin_x.value()
    assert dialog2.spin_origin_y.value() == dialog.spin_origin_y.value()
    assert dialog2.spin_rows.value() == dialog.spin_rows.value()
    assert dialog2.spin_cols.value() == dialog.spin_cols.value()
    assert dialog2.spin_pitch_x.value() == dialog.spin_pitch_x.value()
    assert dialog2.spin_pitch_y.value() == dialog.spin_pitch_y.value()
    assert dialog2.spin_rotation.value() == dialog.spin_rotation.value()
