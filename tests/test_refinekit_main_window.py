"""Focused RefineKit main-window behaviour tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QApplication

from hydra_suite.refinekit.core.event_types import EventType, SuspicionEvent
from hydra_suite.refinekit.core.track_editor_model import TrackEditorModel
from hydra_suite.refinekit.gui import main_window as main_window_module
from hydra_suite.refinekit.gui import overlay_utils
from hydra_suite.refinekit.gui.dialogs import (
    track_editor_dialog as track_editor_dialog_module,
)
from hydra_suite.refinekit.gui.overlay_utils import FrameDetections
from hydra_suite.refinekit.gui.widgets.kinematics_viewer import build_kinematics_cache
from hydra_suite.refinekit.gui.widgets.timeline_editor import TimelineEditorWidget
from hydra_suite.refinekit.gui.widgets.timeline_panel import TimelinePanelWidget
from hydra_suite.refinekit.gui.widgets.video_player import VideoPlayerWidget


@pytest.fixture
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


def test_open_current_session_reveals_main_view_before_merge_wizard(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    del qapp

    video_path = tmp_path / "session.mp4"
    csv_path = tmp_path / "session_tracking_final.csv"
    video_path.write_bytes(b"")
    csv_path.write_text("FrameID,TrajectoryID,X,Y\n", encoding="utf-8")

    df = pd.DataFrame(
        {
            "FrameID": [10, 11, 12],
            "TrajectoryID": [1, 1, 1],
            "X": [100.0, 101.0, 102.0],
            "Y": [50.0, 51.0, 52.0],
        }
    )

    class _FakeWriter:
        def __init__(self, _path: Path) -> None:
            self.df = df.copy()

        def open(self) -> None:
            return None

        def close(self) -> None:
            return None

    window = main_window_module.MainWindow()
    window._sessions = [str(video_path)]
    window._session_idx = 0

    monkeypatch.setattr(main_window_module, "CorrectionWriter", _FakeWriter)
    monkeypatch.setattr(window, "_discover_csv", lambda _path: csv_path)
    monkeypatch.setattr(window._player, "load_video", lambda _path: None)
    monkeypatch.setattr(window._player, "load_trajectories", lambda _df: None)
    monkeypatch.setattr(window._timeline, "load_trajectories", lambda _df: None)
    monkeypatch.setattr(window, "_start_kinematics_precompute", lambda _df: None)
    monkeypatch.setattr(window, "_run_scorer", lambda: None)

    seen = {}

    def _capture_merge_wizard() -> None:
        seen["stack_index"] = window._content_stack.currentIndex()

    monkeypatch.setattr(window, "_maybe_run_merge_wizard", _capture_merge_wizard)

    window._open_current_session()

    assert seen["stack_index"] == 1
    assert window._content_stack.currentIndex() == 1


def test_merge_wizard_auto_expands_radius_from_tracking_config(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    del qapp

    video_path = tmp_path / "session.mp4"
    config_path = tmp_path / "session_config.json"
    video_path.write_bytes(b"")
    config_path.write_text(json.dumps({"MAX_TARGETS": 2}), encoding="utf-8")

    window = main_window_module.MainWindow()
    window._video_path = str(video_path)
    window._df = pd.DataFrame(
        {
            "FrameID": [0, 20, 25, 50, 0, 50],
            "TrajectoryID": [1, 1, 2, 2, 3, 3],
            "X": [10.0, 10.0, 260.0, 260.0, 500.0, 500.0],
            "Y": [10.0, 10.0, 10.0, 10.0, 300.0, 300.0],
        }
    )

    seen = {}

    def _capture_run_merge_wizard(
        segments,
        candidates,
        swap_candidates,
        merge_tuning=None,
        max_animals=None,
    ) -> None:
        seen["segments"] = segments
        seen["candidates"] = candidates
        seen["swap_candidates"] = swap_candidates
        seen["merge_tuning"] = merge_tuning
        seen["max_animals"] = max_animals

    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "question",
        lambda *args, **kwargs: main_window_module.QMessageBox.StandardButton.Yes,
    )
    monkeypatch.setattr(window, "_run_merge_wizard", _capture_run_merge_wizard)

    window._maybe_run_merge_wizard()

    assert seen["max_animals"] == 2
    assert seen["merge_tuning"].max_dist > 200.0
    assert 1 in seen["candidates"]
    assert any(c.target_id == 2 for c in seen["candidates"][1])


def test_video_player_limits_slider_to_loaded_track_span(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del qapp

    from hydra_suite.refinekit.gui.widgets import video_player as video_player_module

    class _FakeCapture:
        def __init__(self, _path: str) -> None:
            self._props = {
                video_player_module.cv2.CAP_PROP_FRAME_COUNT: 200,
                video_player_module.cv2.CAP_PROP_FPS: 25,
            }

        def isOpened(self) -> bool:
            return True

        def get(self, prop: int) -> float:
            return float(self._props.get(prop, 0))

        def set(self, _prop: int, _value: float) -> None:
            return None

        def read(self):
            return False, None

        def release(self) -> None:
            return None

    monkeypatch.setattr(video_player_module.cv2, "VideoCapture", _FakeCapture)

    widget = VideoPlayerWidget()
    widget.load_video("dummy.mp4")
    widget.load_trajectories(
        pd.DataFrame(
            {
                "FrameID": [10, 15, 20],
                "TrajectoryID": [1, 1, 1],
                "X": [1.0, 2.0, 3.0],
                "Y": [1.0, 2.0, 3.0],
            }
        )
    )

    assert widget._slider.minimum() == 10
    assert widget._slider.maximum() == 20


def test_timeline_panel_uses_loaded_track_span(qapp: QApplication) -> None:
    del qapp
    widget = TimelinePanelWidget()
    widget.load_trajectories(
        pd.DataFrame(
            {
                "FrameID": [10, 15, 20],
                "TrajectoryID": [1, 1, 2],
                "X": [1.0, 2.0, 3.0],
                "Y": [1.0, 2.0, 3.0],
            }
        )
    )

    assert widget._canvas._frame_start == 10
    assert widget._canvas._frame_end == 20


def test_main_window_timeline_is_not_hard_capped_to_200px() -> None:
    window = main_window_module.MainWindow()

    assert window._timeline.maximumHeight() > 200


def test_manual_timeline_reassign_moves_non_overlapping_track(
    qapp: QApplication,
    tmp_path: Path,
) -> None:
    del qapp

    df = pd.DataFrame(
        {
            "FrameID": [0, 1, 2, 10, 11, 12],
            "TrajectoryID": [1, 1, 1, 2, 2, 2],
            "X": [0.0] * 6,
            "Y": [0.0] * 6,
            "Theta": [0.0] * 6,
        }
    )
    src = tmp_path / "tracked.csv"
    df.to_csv(src, index=False)
    (tmp_path / "tracked_config.json").write_text(
        json.dumps(
            {
                "fps": 10,
                "interpolation_method": "Linear",
                "interpolation_max_gap_seconds": 1.0,
            }
        ),
        encoding="utf-8",
    )

    writer = main_window_module.CorrectionWriter(src)
    writer.open()

    window = main_window_module.MainWindow()
    window._df = writer.df.copy()
    window._writer = writer
    window._scorer = None

    reloaded = {}

    def monkeypatch_player(updated_df):
        return reloaded.setdefault("player", updated_df.copy())

    def monkeypatch_timeline(updated_df):
        return reloaded.setdefault("timeline", updated_df.copy())

    window._player.load_trajectories = monkeypatch_player
    window._timeline.load_trajectories = monkeypatch_timeline
    window._queue.remove_events_for_tracks = lambda *args, **kwargs: None
    window._queue.show_rescore_button = lambda *args, **kwargs: None

    window._on_manual_track_reassign(1, 2)

    assert set(window._df["TrajectoryID"].unique()) == {2}
    assert set(reloaded["player"]["TrajectoryID"].unique()) == {2}
    assert set(reloaded["timeline"]["TrajectoryID"].unique()) == {2}
    assert sorted(window._df["FrameID"].tolist()) == list(range(13))


def test_correction_writer_interpolates_merged_track_using_tracking_config(
    tmp_path: Path,
) -> None:
    df = pd.DataFrame(
        {
            "FrameID": [0, 1, 4, 5],
            "TrajectoryID": [1, 1, 2, 2],
            "X": [0.0, 1.0, 4.0, 5.0],
            "Y": [0.0, 1.0, 4.0, 5.0],
            "Theta": [0.0, 0.1, 0.4, 0.5],
        }
    )
    src = tmp_path / "session_tracking_final.csv"
    df.to_csv(src, index=False)
    (tmp_path / "session_config.json").write_text(
        json.dumps(
            {
                "fps": 10,
                "interpolation_method": "Linear",
                "interpolation_max_gap_seconds": 0.3,
                "heading_flip_max_burst": 5,
            }
        ),
        encoding="utf-8",
    )

    writer = main_window_module.CorrectionWriter(src)
    writer.open()
    writer.apply_merge([1, 2])

    merged = writer.df[writer.df["TrajectoryID"] == 1].sort_values("FrameID")

    assert merged["FrameID"].tolist() == [0, 1, 2, 3, 4, 5]
    assert merged["X"].isna().sum() == 0
    assert merged["Y"].isna().sum() == 0


def test_manual_timeline_reassign_blocks_overlapping_track(
    qapp: QApplication,
    tmp_path: Path,
) -> None:
    del qapp

    df = pd.DataFrame(
        {
            "FrameID": [0, 1, 2, 2, 3, 4],
            "TrajectoryID": [1, 1, 1, 2, 2, 2],
            "X": [0.0] * 6,
            "Y": [0.0] * 6,
        }
    )
    src = tmp_path / "tracked.csv"
    df.to_csv(src, index=False)

    writer = main_window_module.CorrectionWriter(src)
    writer.open()

    window = main_window_module.MainWindow()
    window._df = writer.df.copy()
    window._writer = writer
    window._scorer = None

    window._on_manual_track_reassign(1, 2)

    assert set(window._df["TrajectoryID"].unique()) == {1, 2}


def test_track_editor_supports_play_button_and_spacebar(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeCapture:
        def __init__(self, _path: str) -> None:
            self._frame = np.zeros((40, 60, 3), dtype=np.uint8)
            self._props = {
                track_editor_dialog_module.cv2.CAP_PROP_FRAME_COUNT: 6.0,
                track_editor_dialog_module.cv2.CAP_PROP_FPS: 20.0,
                track_editor_dialog_module.cv2.CAP_PROP_FRAME_WIDTH: 60.0,
                track_editor_dialog_module.cv2.CAP_PROP_FRAME_HEIGHT: 40.0,
            }

        def isOpened(self) -> bool:
            return True

        def get(self, prop: int) -> float:
            return float(self._props.get(prop, 0.0))

        def set(self, _prop: int, _value: float) -> None:
            return None

        def read(self):
            return True, self._frame.copy()

        def release(self) -> None:
            return None

    monkeypatch.setattr(track_editor_dialog_module.cv2, "VideoCapture", _FakeCapture)
    monkeypatch.setattr(
        track_editor_dialog_module, "load_frame_detections", lambda _path: None
    )
    monkeypatch.setattr(
        track_editor_dialog_module._FrameLoader, "start", lambda self: None
    )

    df = pd.DataFrame(
        {
            "FrameID": [0, 1, 2],
            "TrajectoryID": [1, 1, 1],
            "X": [10.0, 20.0, 30.0],
            "Y": [10.0, 10.0, 10.0],
        }
    )
    event = SuspicionEvent(
        event_type=EventType.MANUAL,
        involved_tracks=[1],
        frame_peak=1,
        frame_range=(0, 2),
        score=1.0,
    )

    dialog = track_editor_dialog_module.TrackEditorDialog("dummy.mp4", df, event)
    dialog._loader.frames = {
        frame_idx: np.zeros((40, 60, 3), dtype=np.uint8) for frame_idx in range(6)
    }
    dialog._on_load_finished()
    qapp.processEvents()

    assert dialog._btn_play.isEnabled()
    assert dialog._is_playing is False

    dialog._btn_play.click()
    assert dialog._is_playing is True
    assert dialog._btn_play.text() == "⏸"

    dialog._advance_playback()
    assert dialog._current_frame == 2

    dialog.keyPressEvent(
        QKeyEvent(
            QEvent.Type.KeyPress, Qt.Key.Key_Space, Qt.KeyboardModifier.NoModifier
        )
    )
    assert dialog._is_playing is False
    assert dialog._btn_play.text() == "▶"

    dialog.close()


def test_track_editor_model_supports_view_window_zoom() -> None:
    model = TrackEditorModel(
        pd.DataFrame(
            {
                "FrameID": [10, 11, 12, 20, 21],
                "TrajectoryID": [1, 1, 1, 2, 2],
                "X": [1.0, 2.0, 3.0, 4.0, 5.0],
                "Y": [1.0, 2.0, 3.0, 4.0, 5.0],
            }
        ),
        [1, 2],
        (10, 21),
    )

    assert model.frame_range == (10, 21)
    assert model.set_view_range(12, 18) is True
    assert model.frame_range == (12, 18)
    assert model.reset_view_range() is True
    assert model.frame_range == (10, 21)


def test_track_editor_model_adds_new_lane_and_drops_emptied_source_lane() -> None:
    model = TrackEditorModel(
        pd.DataFrame(
            {
                "FrameID": [0, 1, 2, 10, 11],
                "TrajectoryID": [1, 1, 1, 2, 2],
                "X": [1.0, 2.0, 3.0, 4.0, 5.0],
                "Y": [1.0, 2.0, 3.0, 4.0, 5.0],
            }
        ),
        [1, 2],
        (0, 11),
    )

    new_track_id = model.add_track_lane()
    frag = next(fragment for fragment in model.fragments if fragment.track_id == 1)

    assert new_track_id == 3
    assert 3 in model.visible_tracks
    assert model.reassign(frag.frag_id, new_track_id) is True
    assert 1 not in model.visible_tracks
    assert new_track_id in model.visible_tracks


def test_track_editor_add_track_button_confirms_before_creating_lane(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeCapture:
        def __init__(self, _path: str) -> None:
            self._frame = np.zeros((40, 60, 3), dtype=np.uint8)
            self._props = {
                track_editor_dialog_module.cv2.CAP_PROP_FRAME_COUNT: 6.0,
                track_editor_dialog_module.cv2.CAP_PROP_FPS: 20.0,
                track_editor_dialog_module.cv2.CAP_PROP_FRAME_WIDTH: 60.0,
                track_editor_dialog_module.cv2.CAP_PROP_FRAME_HEIGHT: 40.0,
            }

        def isOpened(self) -> bool:
            return True

        def get(self, prop: int) -> float:
            return float(self._props.get(prop, 0.0))

        def set(self, _prop: int, _value: float) -> None:
            return None

        def read(self):
            return True, self._frame.copy()

        def release(self) -> None:
            return None

    monkeypatch.setattr(track_editor_dialog_module.cv2, "VideoCapture", _FakeCapture)
    monkeypatch.setattr(
        track_editor_dialog_module, "load_frame_detections", lambda _path: None
    )
    monkeypatch.setattr(
        track_editor_dialog_module._FrameLoader, "start", lambda self: None
    )

    seen = {"asked": 0}

    def _confirm(*args, **kwargs):
        seen["asked"] += 1
        return track_editor_dialog_module.QMessageBox.StandardButton.Yes

    monkeypatch.setattr(track_editor_dialog_module.QMessageBox, "question", _confirm)

    df = pd.DataFrame(
        {
            "FrameID": [0, 1, 2],
            "TrajectoryID": [1, 1, 1],
            "X": [10.0, 20.0, 30.0],
            "Y": [10.0, 10.0, 10.0],
        }
    )
    event = SuspicionEvent(
        event_type=EventType.MANUAL,
        involved_tracks=[1],
        frame_peak=1,
        frame_range=(0, 2),
        score=1.0,
    )

    dialog = track_editor_dialog_module.TrackEditorDialog("dummy.mp4", df, event)
    dialog._loader.frames = {
        frame_idx: np.zeros((40, 60, 3), dtype=np.uint8) for frame_idx in range(6)
    }
    dialog._on_load_finished()
    qapp.processEvents()

    dialog._btn_add_track.click()

    assert seen["asked"] == 1
    assert max(dialog._model.visible_tracks) == 2

    dialog.close()


def test_main_window_shows_kinematics_progress_during_precompute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        main_window_module._KinematicsWorker, "start", lambda self: None
    )

    window = main_window_module.MainWindow()
    df = pd.DataFrame(
        {
            "FrameID": [0, 1, 2],
            "TrajectoryID": [1, 1, 1],
            "X": [0.0, 1.0, 2.0],
            "Y": [0.0, 0.5, 1.0],
        }
    )

    window._load_review_dataframe(df)

    assert window._kinematics_progress.isHidden() is False
    assert window._kinematics.is_loading is True


def test_main_window_wires_kinematics_viewer_to_track_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = main_window_module.MainWindow()
    df = pd.DataFrame(
        {
            "FrameID": [0, 1, 2, 0, 1, 2],
            "TrajectoryID": [1, 1, 1, 2, 2, 2],
            "X": [0.0, 1.0, 2.0, 2.0, 2.5, 3.0],
            "Y": [0.0, 0.5, 1.0, 1.0, 1.5, 2.0],
            "Theta": [0.0, 0.1, 0.2, 0.2, 0.3, 0.4],
            "DetectionConfidence": [0.8, 0.9, 0.85, 0.7, 0.75, 0.78],
            "AssignmentConfidence": [0.95, 0.97, 0.96, 0.88, 0.9, 0.91],
        }
    )

    def _precompute_sync(current_df: pd.DataFrame) -> None:
        cache, frame_range = build_kinematics_cache(current_df)
        window._kinematics.set_loading(True)
        window._kinematics.set_precomputed_data(cache, frame_range)
        window._kinematics_progress.setVisible(False)

    monkeypatch.setattr(window, "_start_kinematics_precompute", _precompute_sync)

    window._load_review_dataframe(df)
    window._player.frame_changed.emit(2)
    window._timeline.track_selected.emit(2)

    assert window._review_splitter.count() == 3
    assert window._kinematics.active_track_id == 2
    assert window._kinematics.current_frame == 2
    assert window._timeline._canvas._current_frame == 2

    window._kinematics_toggles["velocity"].setChecked(False)

    assert window._kinematics.enabled_series["velocity"] is False


def test_main_window_blocks_more_than_four_kinematics_series(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = main_window_module.MainWindow()
    df = pd.DataFrame(
        {
            "FrameID": [0, 1, 2, 0, 1, 2],
            "TrajectoryID": [1, 1, 1, 2, 2, 2],
            "X": [0.0, 1.0, 2.0, 2.0, 2.5, 3.0],
            "Y": [0.0, 0.5, 1.0, 1.0, 1.5, 2.0],
            "Theta": [0.0, 0.1, 0.2, 0.2, 0.3, 0.4],
        }
    )

    def _precompute_sync(current_df: pd.DataFrame) -> None:
        cache, frame_range = build_kinematics_cache(current_df)
        window._kinematics.set_precomputed_data(cache, frame_range)
        window._kinematics_progress.setVisible(False)

    monkeypatch.setattr(window, "_start_kinematics_precompute", _precompute_sync)

    window._load_review_dataframe(df)

    checked = [
        key
        for key, checkbox in window._kinematics_toggles.items()
        if checkbox.isChecked()
    ]
    unchecked = [
        key
        for key, checkbox in window._kinematics_toggles.items()
        if not checkbox.isChecked()
    ]

    assert len(checked) == 4
    assert unchecked
    assert window._kinematics_toggles[unchecked[0]].isEnabled() is False

    window._kinematics_toggles[unchecked[0]].setChecked(True)

    assert window._kinematics_toggles[unchecked[0]].isChecked() is False
    assert (
        sum(
            1
            for checkbox in window._kinematics_toggles.values()
            if checkbox.isChecked()
        )
        == 4
    )


def test_frame_axis_alignment_matches_slider_kinematics_and_timeline(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = main_window_module.MainWindow()
    window.resize(1200, 700)
    window.show()
    qapp.processEvents()

    df = pd.DataFrame(
        {
            "FrameID": [10, 11, 12, 13, 14],
            "TrajectoryID": [1, 1, 1, 1, 1],
            "X": [0.0, 1.0, 2.0, 3.0, 4.0],
            "Y": [0.0, 1.0, 2.0, 3.0, 4.0],
            "Theta": [0.0, 0.1, 0.2, 0.3, 0.4],
        }
    )

    def _precompute_sync(current_df: pd.DataFrame) -> None:
        cache, frame_range = build_kinematics_cache(current_df)
        window._kinematics.set_precomputed_data(cache, frame_range)
        window._kinematics_progress.setVisible(False)

    monkeypatch.setattr(window, "_start_kinematics_precompute", _precompute_sync)

    window._player._total_frames = 20
    window._player.load_trajectories(df)
    window._load_review_dataframe(df)
    qapp.processEvents()

    left, right = window._player.frame_axis_margins()
    window._on_player_frame_axis_changed(left, right)
    qapp.processEvents()

    frame = 12
    slider_x = window._player.frame_to_slider_x(frame)
    kinematics_x = window._kinematics._frame_to_x(frame)
    timeline_x = window._timeline._canvas._frame_to_x(frame)

    assert abs(slider_x - kinematics_x) <= 1
    assert abs(slider_x - timeline_x) <= 1


def test_timeline_panel_paints_visible_track_bars(qapp: QApplication) -> None:
    widget = TimelinePanelWidget()
    widget.resize(480, 120)
    widget.load_trajectories(
        pd.DataFrame(
            {
                "FrameID": [10, 15, 20],
                "TrajectoryID": [1, 1, 2],
                "X": [1.0, 2.0, 3.0],
                "Y": [1.0, 2.0, 3.0],
            }
        )
    )

    widget.show()
    qapp.processEvents()

    canvas = widget._canvas
    image = canvas.grab().toImage()
    sample_x = (canvas._frame_to_x(10) + canvas._frame_to_x(16)) // 2
    sample_y = canvas._row_height // 2
    color = image.pixelColor(sample_x, sample_y)

    assert (color.red(), color.green(), color.blue()) == tuple(
        overlay_utils.TAB20_RGB[1]
    )


def test_timeline_editor_paints_fragment_bars(qapp: QApplication) -> None:
    model = TrackEditorModel(
        pd.DataFrame(
            {
                "FrameID": [10, 11, 12, 20, 21],
                "TrajectoryID": [1, 1, 1, 2, 2],
                "X": [1.0, 2.0, 3.0, 4.0, 5.0],
                "Y": [1.0, 2.0, 3.0, 4.0, 5.0],
            }
        ),
        [1, 2],
        (10, 21),
    )

    widget = TimelineEditorWidget()
    widget.resize(480, 140)
    widget.set_model(model)
    widget.show()
    qapp.processEvents()

    canvas = widget._canvas
    canvas.resize(480, canvas.minimumHeight())
    qapp.processEvents()
    frag = next(frag for frag in model.fragments if frag.track_id == 1)
    rect = canvas._frag_rect(frag)
    image = canvas.grab().toImage()
    color = image.pixelColor(rect.center())

    assert (color.red(), color.green(), color.blue()) == tuple(
        overlay_utils.TAB20_RGB[1]
    )


def test_frame_detections_reads_current_cache_tuple_shape() -> None:
    class _FakeCache:
        def get_frame(self, _frame_idx: int):
            return (
                [[10.0, 20.0, 0.25]],
                [],
                [[100.0, 2.0]],
                [],
                [np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])],
                [],
                [],
                [],
                [],
                [],
                None,
                None,
            )

    dets = FrameDetections(_FakeCache(), inv_resize=2.0)
    result = dets.get(12)

    assert result is not None
    meas_arr, _semi_axes, obb_corners = result
    assert tuple(meas_arr[0]) == pytest.approx((20.0, 40.0, 0.25))
    assert np.allclose(
        obb_corners[0],
        np.array([[2.0, 4.0], [6.0, 8.0], [10.0, 12.0], [14.0, 16.0]]),
    )


def test_load_frame_detections_skips_incompatible_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeDetectionCache:
        def __init__(self, _path: Path, mode: str = "r") -> None:
            self._mode = mode

        def is_compatible(self) -> bool:
            return False

    monkeypatch.setattr(
        overlay_utils,
        "discover_detection_cache",
        lambda _path: Path("fake_cache.npz"),
    )
    monkeypatch.setattr(overlay_utils, "DetectionCache", _FakeDetectionCache)

    assert overlay_utils.load_frame_detections("video.mp4") is None
