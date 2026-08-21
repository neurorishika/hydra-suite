"""Fix Wave 2: regression tests for the ten interaction-completeness fixes.

Each test below is named after the fix it pins; see
``.superpowers/sdd/2026-08-21-multi-arena-ux-redesign/fixwave-2-interaction-brief.md``
for the full rationale of each bug.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from hydra_suite.trackerkit.gui.main_window import MainWindow  # noqa: E402
from hydra_suite.trackerkit.gui.widgets.arena_canvas import ArenaCanvas  # noqa: E402


def _circle(cx, cy, r, arena_id, mode="include"):
    return {"type": "circle", "params": (cx, cy, r), "mode": mode, "arena_id": arena_id}


def _write_tiny_video(path, *, width=64, height=48, n_frames=3, fps=10.0):
    import cv2

    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    for _ in range(n_frames):
        writer.write(frame)
    writer.release()


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(app, tmp_path_factory):
    """A real MainWindow with a real tiny video wired onto the file line."""
    win = MainWindow()
    video_path = tmp_path_factory.mktemp("fixwave2") / "tiny.mp4"
    _write_tiny_video(video_path)
    win._setup_panel.file_line.setText(str(video_path))
    win.current_video_path = str(video_path)
    yield win
    win.roi_base_frame = None


# ---------------------------------------------------------------------------
# Fix 1: ArenaPanel.set_frame_size wired into production, via
# start_roi_selection's real (not test-injected) call path.
# ---------------------------------------------------------------------------


def test_fix1_start_roi_selection_wires_frame_size_into_production(window):
    assert window.arena_panel._frame_size == (0, 0)
    window._session_orch.start_roi_selection()
    assert window.arena_panel._frame_size == (64, 48)


def test_fix1_ensure_roi_base_frame_wires_frame_size_when_already_loaded(window):
    """Second call path: roi_base_frame already set from a previous call --
    set_frame_size must still fire (it lives outside the lazy-load `if`)."""
    window._session_orch.start_roi_selection()
    window.arena_panel.set_frame_size(0, 0)
    assert window._session_orch._ensure_roi_base_frame() is True
    assert window.arena_panel._frame_size == (64, 48)


# ---------------------------------------------------------------------------
# Fix 2 / Fix 3: the grid dialog gets a real reference frame from the empty
# state, and manual drawing / grid generation are mutually exclusive in the
# overflow menu.
# ---------------------------------------------------------------------------


def test_fix2_grid_dialog_has_a_reference_frame_from_the_empty_state(
    window, monkeypatch
):
    from PySide6.QtWidgets import QDialog

    captured = {}

    class _FakeDialog:
        def __init__(self, parent, reference_frame, first_arena_id):
            captured["reference_frame"] = reference_frame

        def exec(self):
            return QDialog.Rejected

    monkeypatch.setattr(
        "hydra_suite.trackerkit.gui.dialogs.arena_grid_dialog.ArenaGridDialog",
        _FakeDialog,
    )
    window.roi_base_frame = None
    window._on_generate_grid_clicked()
    assert captured["reference_frame"] is not None


def test_fix3_overflow_menu_has_no_grid_action(window):
    menu = window.arena_panel.btn_overflow.menu()
    labels = [a.text() for a in menu.actions()]
    assert "Add Grid of Arenas" not in labels
    assert "Remove All Arenas" in labels
    assert "Crop Video to ROI" in labels


# ---------------------------------------------------------------------------
# Fix 4: the dead facade is gone.
# ---------------------------------------------------------------------------


def test_fix4_record_roi_click_facade_removed(window):
    assert not hasattr(window, "record_roi_click")


# ---------------------------------------------------------------------------
# Fix 5: zone-button gating, drawing lock, navigation lock.
# ---------------------------------------------------------------------------


def test_fix5_add_single_arena_preselects_no_zone_tool(window):
    window._on_add_single_arena()
    panel = window.arena_panel
    assert all(not b.isChecked() for b in panel._zone_buttons)
    assert panel._drawing_active is False


def test_fix5_zone_button_press_locks_navigation_and_other_buttons(window):
    window._on_add_single_arena()
    panel = window.arena_panel
    panel.begin_new_arena()
    panel._shapes = [_circle(50, 50, 20, 0)]
    panel.set_current_arena(0)
    panel._on_zone_button_clicked(panel.btn_add_circle, "circle", "include")

    assert panel.btn_add_circle.isChecked() is True
    assert panel.btn_sub_circle.isChecked() is False
    # All four are disabled while drawing, including the checked one.
    assert all(not b.isEnabled() for b in panel._zone_buttons)
    assert panel.btn_prev.isEnabled() is False
    assert panel.btn_next.isEnabled() is False
    assert panel.btn_add_new.isEnabled() is False

    panel.set_drawing_active(False)
    assert all(not b.isChecked() for b in panel._zone_buttons)
    assert all(b.isEnabled() for b in panel._zone_buttons)


def test_fix5_failed_start_roi_selection_releases_the_drawing_lock(window, monkeypatch):
    """Regression for the hard-lock dead-end: a zone-button press sets
    _drawing_active True before start_roi_selection can fail. If
    _ensure_roi_base_frame() then returns False (e.g. no video selected),
    _drawing_active must not stay stuck True forever -- Escape, Undo, and
    Clear Arena were all gated on it, and the only escape hatch was
    destroying every arena via Remove All Arenas."""
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: None)
    window._on_add_single_arena()
    panel = window.arena_panel

    # No video selected -- the "No Video" branch of _ensure_roi_base_frame.
    window._setup_panel.file_line.setText("")

    panel._on_zone_button_clicked(panel.btn_add_circle, "circle", "include")

    assert panel._drawing_active is False
    assert all(b.isEnabled() for b in panel._zone_buttons)
    assert window.roi_selection_active is False

    # Escape (cancel_roi_shape called directly) must also be a safe no-op
    # release of the lock, even when drawing never actually started.
    window._session_orch.cancel_roi_shape()
    assert panel._drawing_active is False


# ---------------------------------------------------------------------------
# Fix 6: Escape cancels only the in-progress shape.
# ---------------------------------------------------------------------------


def test_fix6_escape_cancels_only_the_in_progress_shape(window):
    from PySide6.QtCore import QEvent, Qt
    from PySide6.QtGui import QKeyEvent

    orch = window._session_orch
    window.roi_shapes = [_circle(50, 50, 20, 0)]
    window.arena_panel.set_shapes(window.roi_shapes)
    window.roi_selection_active = True
    window.roi_points = [(1.0, 2.0), (3.0, 4.0)]

    event = QKeyEvent(QEvent.KeyPress, Qt.Key_Escape, Qt.NoModifier)
    orch.keyPressEvent(event)

    assert window.roi_points == []
    assert window.roi_shapes == [_circle(50, 50, 20, 0)]
    assert window.roi_selection_active is False


# ---------------------------------------------------------------------------
# Fix 8: _sync_contextual_controls re-asserts the panel's own state.
# ---------------------------------------------------------------------------


def test_fix8_sync_contextual_controls_reasserts_drawing_lock(window):
    panel = window.arena_panel
    panel._shapes = [_circle(50, 50, 20, 0)]
    panel._pending_new = True
    panel.set_drawing_active(True)
    assert all(not b.isEnabled() for b in panel._zone_buttons)

    # Simulate the enable-sweep blanket re-enabling every QAbstractButton.
    for b in panel._zone_buttons:
        b.setEnabled(True)
    assert panel.btn_add_circle.isEnabled() is True

    window._session_orch._sync_contextual_controls()
    assert all(not b.isEnabled() for b in panel._zone_buttons)


# ---------------------------------------------------------------------------
# Fix 9: Remove All Arenas requires confirmation; toast replaces the
# blocking full-screen message.
# ---------------------------------------------------------------------------


def test_fix9a_remove_all_arenas_confirmation_gates_clear(window, monkeypatch):
    window.roi_shapes = [_circle(50, 50, 20, 0)]
    window.arena_panel.set_shapes(window.roi_shapes)

    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.No)
    window._on_remove_all_arenas()
    assert window.roi_shapes == [_circle(50, 50, 20, 0)]

    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.Yes)
    window._on_remove_all_arenas()
    assert window.roi_shapes == []


def test_fix9b_clear_roi_shows_a_toast_not_a_blocking_message(window, monkeypatch):
    calls = {}
    monkeypatch.setattr(
        window.video_label,
        "show_toast",
        lambda text, **k: calls.setdefault("text", text),
    )
    monkeypatch.setattr(
        window,
        "_set_video_message",
        lambda *a, **k: calls.setdefault("blocking_message_called", True),
    )
    window.clear_roi()
    assert calls.get("text") == "ROI Cleared"
    assert "blocking_message_called" not in calls


def test_fixwave12_clear_roi_show_toast_false_suppresses_the_toast(window, monkeypatch):
    """Fix Wave 12, Fix 1: the internal per-video-open reset (show_toast=False)
    must not show the "ROI Cleared" toast that the user-initiated Start Fresh /
    Remove All Arenas path (show_toast=True, the default) correctly shows."""
    calls = {}
    monkeypatch.setattr(
        window.video_label,
        "show_toast",
        lambda text, **k: calls.setdefault("text", text),
    )
    window.clear_roi(show_toast=False)
    assert "text" not in calls


def test_fixwave12_mainwindow_clear_roi_forwards_show_toast_flag(window, monkeypatch):
    """Fix Wave 12, Fix 1: MainWindow.clear_roi(show_toast=...) must forward the
    flag through to SessionOrchestrator.clear_roi, not just swallow it."""
    received = {}

    def _fake_session_clear_roi(show_toast=True):
        received["show_toast"] = show_toast

    monkeypatch.setattr(window._session_orch, "clear_roi", _fake_session_clear_roi)
    window.clear_roi(show_toast=False)
    assert received["show_toast"] is False

    window.clear_roi()
    assert received["show_toast"] is True


def test_fix9b_toast_pauses_canvas_input():
    canvas = ArenaCanvas()
    assert canvas.is_input_paused() is False
    canvas.show_toast("hello", duration_ms=60_000)
    assert canvas.is_input_paused() is True
    canvas._clear_toast()
    assert canvas.is_input_paused() is False


# ---------------------------------------------------------------------------
# Fix 10: live shape preview, and immediate arena visibility after Finish.
# ---------------------------------------------------------------------------


def test_fix10a_update_roi_preview_pushes_a_circle_preview_shape(window):
    orch = window._session_orch
    window.roi_current_mode = "circle"
    window.roi_points = [(0.0, 0.0), (10.0, 0.0), (0.0, 10.0), (5.0, 8.0)]
    orch.update_roi_preview()
    preview = window.video_label._preview_shape
    assert preview is not None
    assert preview["type"] == "circle"
    assert preview["params"] == window.roi_fitted_circle


def test_fix10a_update_roi_preview_pushes_a_polygon_preview_shape(window):
    orch = window._session_orch
    window.roi_current_mode = "polygon"
    window.roi_points = [(0.0, 0.0), (10.0, 0.0), (5.0, 10.0)]
    orch.update_roi_preview()
    preview = window.video_label._preview_shape
    assert preview == {"type": "polygon", "params": list(window.roi_points)}


def test_fix10a_preview_clears_with_too_few_points(window):
    orch = window._session_orch
    window.roi_current_mode = "polygon"
    window.roi_points = [(0.0, 0.0)]
    orch.update_roi_preview()
    assert window.video_label._preview_shape is None


def test_fix10b_finish_roi_selection_pushes_the_new_shape_to_the_canvas(
    window, monkeypatch
):
    orch = window._session_orch
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: None)
    window._session_orch.start_roi_selection()
    window.roi_current_mode = "circle"
    window.roi_current_zone_type = "include"
    window.roi_points = [(10.0, 10.0), (30.0, 10.0), (10.0, 30.0), (20.0, 25.0)]
    orch.update_roi_preview()
    assert window.roi_fitted_circle is not None

    orch.finish_roi_selection()

    assert len(window.video_label._shapes) == 1
    assert window.video_label._shapes[0]["type"] == "circle"
    assert window.video_label._preview_shape is None
