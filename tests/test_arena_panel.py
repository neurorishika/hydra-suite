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


def test_done_button_switches_to_done_state_and_emits_signal(panel):
    panel.set_shapes([_circle(50, 50, 20, 0)])
    fired = []
    panel.done_requested.connect(lambda: fired.append(True))
    panel.btn_done.click()
    assert panel.stack.currentWidget() is panel.done_widget
    assert fired == [True]


def test_done_state_shows_exactly_the_three_expected_widgets(panel):
    panel.set_shapes([_circle(50, 50, 20, 0)])
    panel.btn_done.click()
    assert panel.stack.currentWidget() is panel.done_widget
    children = panel.done_widget.children()
    assert panel.lbl_done in children
    assert panel.btn_modify_existing in children
    assert panel.btn_start_fresh in children


def test_start_fresh_reuses_the_existing_clear_all_signal(panel):
    panel.set_shapes([_circle(50, 50, 20, 0)])
    panel.btn_done.click()
    fired = []
    panel.clear_all_requested.connect(lambda: fired.append(True))
    panel.btn_start_fresh.click()
    assert fired == [True]


def test_mark_grid_generated_and_mark_hand_drawn(panel):
    assert panel.made_via_grid is False
    assert panel.last_grid_params is None
    params = {"rows": 3, "cols": 4}
    panel.mark_grid_generated(params)
    assert panel.made_via_grid is True
    assert panel.last_grid_params == params
    panel.mark_hand_drawn()
    assert panel.made_via_grid is False
    assert panel.last_grid_params is None


def test_done_disabled_while_drawing_active(panel):
    panel.set_shapes([_circle(50, 50, 20, 0)])
    panel.set_drawing_active(True)
    assert panel.btn_done.isEnabled() is False
    panel.set_drawing_active(False)
    assert panel.btn_done.isEnabled() is True


def test_done_disabled_while_current_arena_overlaps(panel):
    panel.set_shapes([_circle(100, 100, 50, 0), _circle(130, 100, 50, 1)])
    panel.set_current_arena(0)
    assert panel.btn_done.isEnabled() is False


def test_done_disabled_with_no_shapes_yet(panel):
    panel.set_shapes([])
    panel.begin_new_arena()
    assert panel.btn_done.isEnabled() is False


def test_start_fresh_cycle_resets_done_flag(panel):
    panel.set_shapes([_circle(50, 50, 20, 0)])
    panel.btn_done.click()
    assert panel.stack.currentWidget() is panel.done_widget
    # Simulate the "Remove All Arenas"/Start Fresh outcome: shapes go to zero.
    panel.set_shapes([])
    assert panel.stack.currentWidget() is panel.empty_widget
    # A later set_shapes([...]) call must land in editing, not a stale done state.
    panel.set_shapes([_circle(60, 60, 20, 0)])
    assert panel.stack.currentWidget() is panel.editing_widget


def test_resume_editing_returns_from_done_state(panel):
    panel.set_shapes([_circle(50, 50, 20, 0)])
    panel.btn_done.click()
    assert panel.stack.currentWidget() is panel.done_widget
    panel.resume_editing()
    assert panel.stack.currentWidget() is panel.editing_widget


def test_open_in_done_state_if_shapes_exist_lands_in_done_state(panel):
    panel.set_shapes([_circle(50, 50, 20, 0)])
    assert panel.stack.currentWidget() is panel.editing_widget
    panel.open_in_done_state_if_shapes_exist()
    assert panel.stack.currentWidget() is panel.done_widget


def test_open_in_done_state_if_shapes_exist_is_a_noop_with_no_shapes(panel):
    assert panel.stack.currentWidget() is panel.empty_widget
    panel.open_in_done_state_if_shapes_exist()
    assert panel.stack.currentWidget() is panel.empty_widget
    assert not panel._done


def test_editing_bar_is_split_into_two_rows(panel):
    from PySide6.QtWidgets import QHBoxLayout, QVBoxLayout

    outer = panel.editing_widget.layout()
    assert isinstance(outer, QVBoxLayout)
    assert outer.count() == 2
    nav_row = outer.itemAt(0).layout()
    tools_row = outer.itemAt(1).layout()
    assert isinstance(nav_row, QHBoxLayout)
    assert isinstance(tools_row, QHBoxLayout)

    def _layout_containing(layout, widget):
        for i in range(layout.count()):
            item = layout.itemAt(i)
            if item.widget() is widget:
                return layout
        return None

    # Navigation and drawing-tool controls really are on separate rows,
    # not just visually reordered within one row.
    assert _layout_containing(nav_row, panel.lbl_current) is nav_row
    assert _layout_containing(tools_row, panel.lbl_hint) is tools_row
    assert _layout_containing(nav_row, panel.lbl_hint) is None
    assert _layout_containing(tools_row, panel.lbl_current) is None
