"""Regression tests for PoseKit main-window frame switching."""

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QMessageBox  # noqa: E402

from hydra_suite.posekit.gui.main_window import MainWindow  # noqa: E402
from hydra_suite.posekit.gui.models import FrameAnn  # noqa: E402


class _DummyCombo:
    def blockSignals(self, _blocked: bool) -> None:
        return None

    def setCurrentIndex(self, _index: int) -> None:
        return None

    def currentIndex(self) -> int:
        return 0

    def count(self) -> int:
        return 1


class _DummyCanvas:
    def set_current_keypoint(self, _index: int) -> None:
        return None


def test_load_frame_defers_previous_frame_list_refresh_after_save() -> None:
    saved_refresh_flags = []
    scheduled_indices = []
    cache_calls = []

    window = SimpleNamespace(
        image_paths=[Path("frame_0.png"), Path("frame_1.png")],
        current_index=0,
        current_kpt=0,
        _dirty=True,
        _ann=object(),
        _frame_cache={1: FrameAnn(cls=0, bbox_xyxy=None, kpts=[])},
        _img_bgr=None,
        _img_display=None,
        _img_wh=(1, 1),
        mode="frame",
        class_combo=_DummyCombo(),
        canvas=_DummyCanvas(),
        _undo_stack=[],
        _cache_current_frame=lambda: cache_calls.append("cached"),
        save_current=lambda refresh_ui=False: saved_refresh_flags.append(refresh_ui),
        _read_image=lambda _path: np.zeros((8, 8, 3), dtype=np.uint8),
        _load_ann_from_disk=lambda _idx: FrameAnn(cls=0, bbox_xyxy=None, kpts=[]),
        _refresh_canvas_image=lambda: None,
        _rebuild_canvas=lambda: None,
        _update_info=lambda: None,
        _load_metadata_ui=lambda: None,
        _schedule_frame_item_refresh=lambda idx: scheduled_indices.append(idx),
        _update_frame_item=lambda _idx: (_ for _ in ()).throw(
            AssertionError("load_frame should defer list-item refresh")
        ),
    )

    MainWindow.load_frame(window, 1)

    assert cache_calls == ["cached"]
    assert saved_refresh_flags == [False]
    assert scheduled_indices == [0]
    assert window.current_index == 1


def test_open_recent_project_uses_posekit_gui_project_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_path = tmp_path / "pose_project.json"
    project_path.write_text("{}", encoding="utf-8")

    opened = []
    switched = []
    sentinel_project = object()

    monkeypatch.setattr(
        "hydra_suite.posekit.gui.main_window.open_project_from_path",
        lambda path: opened.append(path) or sentinel_project,
    )

    window = SimpleNamespace(
        _switch_project_window=lambda project: switched.append(project),
    )

    MainWindow._open_recent_project(window, str(project_path))

    assert opened == [project_path]
    assert switched == [sentinel_project]


def test_recent_project_display_name_prefers_project_directory() -> None:
    path = "/Users/example/projects/ant_pose_project/pose_project.json"

    assert MainWindow._recent_project_display_name(path) == "ant_pose_project"


def test_recent_project_display_name_handles_bundle_state_path() -> None:
    path = "/Users/example/projects/ant_pose_project/state/pose_project.json"

    assert MainWindow._recent_project_display_name(path) == "ant_pose_project"


def test_switch_project_window_allows_empty_projects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created = []
    shown = []
    closed = []

    class _StubWindow:
        def __init__(self, project, image_paths, *, show_welcome_when_empty=True):
            created.append((project, list(image_paths), show_welcome_when_empty))

        def resize(self, _size) -> None:
            return None

        def showMaximized(self) -> None:
            shown.append(True)

    class _StubApp:
        pass

    app = _StubApp()
    monkeypatch.setattr(
        "hydra_suite.posekit.gui.main_window.MainWindow",
        _StubWindow,
    )
    monkeypatch.setattr(
        "hydra_suite.posekit.gui.main_window.build_image_list",
        lambda _project: [],
    )
    monkeypatch.setattr(
        "hydra_suite.posekit.gui.main_window.QApplication.instance",
        lambda: app,
    )

    window = SimpleNamespace(
        _recents_store=SimpleNamespace(add=lambda _path: None),
        _perform_autosave=lambda: None,
        save_project=lambda: None,
        size=lambda: object(),
        close=lambda: closed.append(True),
    )
    project = SimpleNamespace(project_path=Path("/tmp/pose_project.json"))

    MainWindow._switch_project_window(
        window, project, open_source_manager_if_empty=False
    )

    assert created == [(project, [], False)]
    assert shown == [True]
    assert closed == [True]
    assert hasattr(app, "_posekit_windows")


def test_on_frame_mode_toggled_updates_config():
    from hydra_suite.posekit.config.schemas import PoseKitConfig

    window = SimpleNamespace(config=PoseKitConfig())

    MainWindow._on_frame_mode_toggled(window, True)
    assert window.config.frame_mode is True

    MainWindow._on_frame_mode_toggled(window, False)
    assert window.config.frame_mode is False


def test_frame_expansion_groups_by_source_and_frame():
    from hydra_suite.posekit.config.schemas import PoseKitConfig

    window = SimpleNamespace(
        config=PoseKitConfig(frame_mode=True),
        image_paths=[
            Path("did10000.jpg"),
            Path("did10001.jpg"),
            Path("did20000.jpg"),
        ],
        _source_id_for_index=lambda idx: "src_a",
    )

    expanded, frame_count = MainWindow._frame_expansion(window, {0})

    assert expanded == {0, 1}
    assert frame_count == 1


def test_add_indices_to_labeling_frame_mode_expands_and_confirms(monkeypatch):
    from hydra_suite.posekit.config.schemas import PoseKitConfig

    calls = []

    def fake_question(*args, **kwargs):
        calls.append("asked")
        return QMessageBox.Yes

    monkeypatch.setattr(QMessageBox, "question", staticmethod(fake_question))
    monkeypatch.setattr(QMessageBox, "information", staticmethod(lambda *a, **k: None))

    window = SimpleNamespace(
        config=PoseKitConfig(frame_mode=True),
        image_paths=[
            Path("did10000.jpg"),
            Path("did10001.jpg"),
            Path("did20000.jpg"),
        ],
        _source_id_for_index=lambda idx: "src_a",
        labeling_frames=set(),
        current_index=0,
        _populate_frames=lambda: None,
        _select_frame_in_list=lambda *a, **k: None,
        _frame_expansion=lambda indices: MainWindow._frame_expansion(window, indices),
    )

    result = MainWindow._add_indices_to_labeling(window, [0], "Test")

    assert result is True
    assert window.labeling_frames == {0, 1}
    assert calls == ["asked"]


def test_add_indices_to_labeling_frame_mode_cancel_adds_nothing(monkeypatch):
    from hydra_suite.posekit.config.schemas import PoseKitConfig

    monkeypatch.setattr(
        QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.No)
    )

    window = SimpleNamespace(
        config=PoseKitConfig(frame_mode=True),
        image_paths=[Path("did10000.jpg"), Path("did10001.jpg")],
        _source_id_for_index=lambda idx: "src_a",
        labeling_frames=set(),
        current_index=0,
        _populate_frames=lambda: (_ for _ in ()).throw(
            AssertionError("must not refresh UI on cancel")
        ),
        _select_frame_in_list=lambda *a, **k: None,
        _frame_expansion=lambda indices: MainWindow._frame_expansion(window, indices),
    )

    result = MainWindow._add_indices_to_labeling(window, [0], "Test")

    assert result is False
    assert window.labeling_frames == set()


def test_add_indices_to_labeling_frame_mode_disclosed_skips_confirmation(monkeypatch):
    from hydra_suite.posekit.config.schemas import PoseKitConfig

    monkeypatch.setattr(
        QMessageBox,
        "question",
        staticmethod(
            lambda *a, **k: (_ for _ in ()).throw(
                AssertionError("must not confirm when disclosed=True")
            )
        ),
    )
    monkeypatch.setattr(QMessageBox, "information", staticmethod(lambda *a, **k: None))

    window = SimpleNamespace(
        config=PoseKitConfig(frame_mode=True),
        image_paths=[Path("did10000.jpg"), Path("did10001.jpg")],
        _source_id_for_index=lambda idx: "src_a",
        labeling_frames=set(),
        current_index=0,
        _populate_frames=lambda: None,
        _select_frame_in_list=lambda *a, **k: None,
        _frame_expansion=lambda indices: MainWindow._frame_expansion(window, indices),
    )

    result = MainWindow._add_indices_to_labeling(window, [0], "Test", disclosed=True)

    assert result is True
    assert window.labeling_frames == {0, 1}


def test_add_indices_to_labeling_individual_mode_unchanged(monkeypatch):
    from hydra_suite.posekit.config.schemas import PoseKitConfig

    monkeypatch.setattr(
        QMessageBox,
        "question",
        staticmethod(
            lambda *a, **k: (_ for _ in ()).throw(
                AssertionError("must not confirm outside frame mode")
            )
        ),
    )
    monkeypatch.setattr(QMessageBox, "information", staticmethod(lambda *a, **k: None))

    window = SimpleNamespace(
        config=PoseKitConfig(frame_mode=False),
        image_paths=[Path("did10000.jpg"), Path("did10001.jpg")],
        _source_id_for_index=lambda idx: "src_a",
        labeling_frames=set(),
        current_index=0,
        _populate_frames=lambda: None,
        _select_frame_in_list=lambda *a, **k: None,
    )

    result = MainWindow._add_indices_to_labeling(window, [0], "Test")

    assert result is True
    assert window.labeling_frames == {0}


def _make_save_current_window(monkeypatch, frame_mode, current_index=0):
    from hydra_suite.posekit.config.schemas import PoseKitConfig

    save_calls = []
    monkeypatch.setattr(
        "hydra_suite.posekit.gui.main_window.save_yolo_pose_label",
        lambda **kwargs: save_calls.append(kwargs),
    )
    monkeypatch.setattr(
        "hydra_suite.posekit.gui.main_window.compute_bbox_from_kpts",
        lambda *a, **k: (0, 0, 1, 1),
    )

    window = SimpleNamespace(
        config=PoseKitConfig(frame_mode=frame_mode),
        image_paths=[Path("did10000.jpg"), Path("did10001.jpg")],
        _source_id_for_index=lambda idx: "src_a",
        labeling_frames=set(),
        current_index=current_index,
        _ann=FrameAnn(cls=0, bbox_xyxy=None, kpts=[]),
        _cache_current_frame=lambda: None,
        _label_path_for=lambda p: Path(f"/labels/{p.stem}.txt"),
        class_combo=_DummyCombo(),
        _kpts_to_save_space=lambda kpts, path: (kpts, 10, 10),
        project=SimpleNamespace(bbox_pad_frac=0.1),
        _autosave_timer=SimpleNamespace(isActive=lambda: False, stop=lambda: None),
        _populate_frames=lambda: None,
        _select_frame_in_list=lambda *a, **k: None,
        statusBar=lambda: SimpleNamespace(showMessage=lambda *a, **k: None),
        _set_saved_status=lambda: None,
        save_project=lambda: None,
        _load_ann_from_disk=lambda idx: FrameAnn(cls=0, bbox_xyxy=None, kpts=[]),
        _rebuild_canvas=lambda: None,
    )
    window._frame_expansion = lambda indices: MainWindow._frame_expansion(
        window, indices
    )
    return window, save_calls


def test_save_current_frame_mode_confirms_and_adds_companions(monkeypatch):
    monkeypatch.setattr(
        QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.Yes)
    )
    window, save_calls = _make_save_current_window(monkeypatch, frame_mode=True)

    MainWindow.save_current(window)

    assert len(save_calls) == 1
    assert window.labeling_frames == {1}  # companion added explicitly


def test_save_current_frame_mode_cancel_discards_edits(monkeypatch):
    monkeypatch.setattr(
        QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.No)
    )
    window, save_calls = _make_save_current_window(monkeypatch, frame_mode=True)
    reload_calls = []
    window._load_ann_from_disk = lambda idx: (
        reload_calls.append(idx) or FrameAnn(cls=0, bbox_xyxy=None, kpts=[])
    )

    MainWindow.save_current(window)

    assert save_calls == []
    assert window.labeling_frames == set()
    assert reload_calls == [0]


def test_save_current_individual_mode_unchanged(monkeypatch):
    monkeypatch.setattr(
        QMessageBox,
        "question",
        staticmethod(
            lambda *a, **k: (_ for _ in ()).throw(
                AssertionError("must not confirm outside frame mode")
            )
        ),
    )
    window, save_calls = _make_save_current_window(monkeypatch, frame_mode=False)

    MainWindow.save_current(window)

    assert len(save_calls) == 1
    # Individual mode never explicitly adds anything; only _populate_frames'
    # own auto-promotion (not exercised by this fake) would do so.
    assert window.labeling_frames == set()


def test_save_current_already_in_labeling_set_skips_confirmation(monkeypatch):
    monkeypatch.setattr(
        QMessageBox,
        "question",
        staticmethod(
            lambda *a, **k: (_ for _ in ()).throw(
                AssertionError("must not confirm when frame already in labeling set")
            )
        ),
    )
    window, save_calls = _make_save_current_window(monkeypatch, frame_mode=True)
    window.labeling_frames = {0}

    MainWindow.save_current(window)

    assert len(save_calls) == 1
