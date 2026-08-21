"""Arena bar: two-state machine and the overlap lock's enable/disable matrix."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from hydra_suite.trackerkit.gui.panels.arena_panel import ArenaPanel  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def panel(app):
    widget = ArenaPanel()
    widget.set_frame_size(400, 400)
    return widget


def _circle(cx, cy, r, arena_id, mode="include"):
    return {"type": "circle", "params": (cx, cy, r), "mode": mode, "arena_id": arena_id}


def test_empty_state_shows_only_the_two_add_buttons(panel):
    panel.set_shapes([])
    assert panel.stack.currentWidget() is panel.empty_widget
    assert panel.btn_add_single.isEnabled()
    assert panel.btn_add_grid.isEnabled()


def test_empty_state_message(panel):
    panel.set_shapes([])
    assert panel.lbl_default.text() == "By default, the whole video is used."


def test_adding_a_shape_enters_editing_state(panel):
    panel.set_shapes([_circle(100, 100, 30, 0)])
    assert panel.lbl_current.text() == "Currently labelling: Arena 1"


def test_previous_is_disabled_on_the_first_arena(panel):
    panel.set_shapes([_circle(50, 50, 20, 0), _circle(200, 200, 20, 1)])
    panel.set_current_arena(0)
    assert panel.btn_prev.isEnabled() is False
    assert panel.btn_next.isEnabled() is True


def test_next_is_disabled_on_the_last_arena(panel):
    panel.set_shapes([_circle(50, 50, 20, 0), _circle(200, 200, 20, 1)])
    panel.set_current_arena(1)
    assert panel.btn_next.isEnabled() is False


def test_add_new_arena_blocked_while_current_arena_is_empty(panel):
    """Empty arenas inflate MAX_TARGETS (n_arenas * animals_per_arena)."""
    panel.set_shapes([_circle(50, 50, 20, 0)])
    panel.begin_new_arena()
    assert panel.btn_add_new.isEnabled() is False


def test_navigation_locked_while_the_current_arena_overlaps(panel):
    panel.set_shapes([_circle(100, 100, 50, 0), _circle(130, 100, 50, 1)])
    panel.set_current_arena(0)
    assert panel.btn_next.isEnabled() is False
    assert panel.btn_add_new.isEnabled() is False
    assert "overlap" in panel.lbl_warning.text().lower()


def test_warning_names_the_conflicting_arena(panel):
    panel.set_shapes([_circle(100, 100, 50, 0), _circle(130, 100, 50, 1)])
    panel.set_current_arena(0)
    assert "2" in panel.lbl_warning.text()


def test_navigation_free_when_a_distant_pair_overlaps(panel):
    """The lock must never strand the user away from the arenas they must fix."""
    shapes = [
        _circle(30, 30, 15, 0),
        _circle(200, 200, 50, 1),
        _circle(230, 200, 50, 2),
    ]
    panel.set_shapes(shapes)
    panel.set_current_arena(0)
    assert panel.btn_next.isEnabled() is True


def test_tracking_blocked_by_any_overlap_anywhere(panel):
    shapes = [
        _circle(30, 30, 15, 0),
        _circle(200, 200, 50, 1),
        _circle(230, 200, 50, 2),
    ]
    panel.set_shapes(shapes)
    panel.set_current_arena(0)
    allowed, reason = panel.can_track()
    assert allowed is False
    assert "2" in reason and "3" in reason


def test_tracking_allowed_when_nothing_overlaps(panel):
    panel.set_shapes([_circle(50, 50, 20, 0), _circle(300, 300, 20, 1)])
    allowed, _reason = panel.can_track()
    assert allowed is True


def test_clear_arena_keeps_the_arena_and_its_number(panel):
    shapes = [_circle(50, 50, 20, 0), _circle(300, 300, 20, 1)]
    panel.set_shapes(shapes)
    panel.set_current_arena(1)
    remaining = panel.shapes_after_clearing(1)
    assert remaining == [shapes[0]]
    assert panel.current_arena == 1


def test_finish_disabled_until_a_shape_is_valid(panel):
    panel.set_shapes([_circle(50, 50, 20, 0)])
    panel.set_shape_valid(False)
    assert panel.btn_finish.isEnabled() is False
    panel.set_shape_valid(True)
    assert panel.btn_finish.isEnabled() is True
