"""Focused tests for TrackerKit preview seeking behavior."""

import os
from unittest.mock import MagicMock

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture()
def main_window(qapp):
    from hydra_suite.trackerkit.gui.main_window import MainWindow

    window = MainWindow()
    yield window
    window.close()


def test_timeline_navigation_renders_once(main_window, monkeypatch):
    main_window.video_total_frames = 10
    main_window.video_current_frame_idx = 1
    main_window._setup_panel.slider_timeline.setMaximum(9)
    main_window._setup_panel.slider_timeline.setValue(1)

    render_spy = MagicMock()
    monkeypatch.setattr(main_window._session_orch, "_display_current_frame", render_spy)

    main_window._goto_next_frame()

    assert main_window.video_current_frame_idx == 2
    assert main_window._setup_panel.slider_timeline.value() == 2
    assert render_spy.call_count == 1


def test_timeline_move_updates_label_without_render(main_window, monkeypatch):
    render_spy = MagicMock()
    monkeypatch.setattr(main_window._session_orch, "_display_current_frame", render_spy)

    main_window.video_total_frames = 10
    main_window._session_orch._on_timeline_moved(7)

    assert not main_window._setup_panel.slider_timeline.hasTracking()
    assert (
        main_window._setup_panel.lbl_current_frame.text()
        == "Frame: 7/9 (release to seek)"
    )
    assert render_spy.call_count == 0


def test_scrubbing_decodes_only_selected_frame(main_window, monkeypatch):
    slider = main_window._setup_panel.slider_timeline
    slider.setMaximum(99)
    main_window.video_total_frames = 100
    render_spy = MagicMock()
    monkeypatch.setattr(main_window._session_orch, "_display_current_frame", render_spy)

    slider.setSliderDown(True)
    for frame in (10, 25, 50):
        slider.setSliderPosition(frame)
        main_window._on_timeline_moved(frame)
    assert render_spy.call_count == 0
    assert main_window.video_current_frame_idx != 50

    slider.setSliderDown(False)
    assert main_window.video_current_frame_idx == 50
    assert render_spy.call_count == 1


def test_preview_controls_live_below_video_and_keep_actions(main_window):
    from PySide6.QtWidgets import QVBoxLayout

    left_layout = main_window.scroll.parentWidget().layout()
    assert isinstance(left_layout, QVBoxLayout)
    video_index = left_layout.indexOf(main_window.scroll)
    preview_index = left_layout.indexOf(main_window._setup_panel.g_video_player)
    assert left_layout.indexOf(main_window.interaction_help) == video_index + 1
    assert preview_index == video_index + 2
    assert (
        main_window._setup_panel.g_video_player.parentWidget()
        is main_window.scroll.parentWidget()
    )
    assert (
        main_window._setup_panel.g_video_player
        not in main_window._setup_panel.findChildren(
            type(main_window._setup_panel.g_video_player)
        )
    )

    controls = main_window._setup_panel
    for name in (
        "slider_timeline",
        "btn_first_frame",
        "btn_prev_frame",
        "btn_play_pause",
        "btn_next_frame",
        "btn_last_frame",
        "btn_random_seek",
        "combo_playback_speed",
        "spin_start_frame",
        "btn_set_start_current",
        "spin_end_frame",
        "btn_set_end_current",
        "btn_reset_range",
    ):
        assert getattr(controls, name).parentWidget() is not None
