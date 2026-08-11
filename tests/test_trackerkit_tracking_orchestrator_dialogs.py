from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from hydra_suite.core.post import media_export
from hydra_suite.trackerkit.gui.orchestrators import config as config_module
from hydra_suite.trackerkit.gui.orchestrators import tracking as tracking_module
from hydra_suite.trackerkit.gui.orchestrators.config import ConfigOrchestrator
from hydra_suite.trackerkit.gui.orchestrators.tracking import TrackingOrchestrator


def _make_orchestrator() -> tuple[TrackingOrchestrator, object]:
    main_window = object()
    panels = SimpleNamespace(
        setup=SimpleNamespace(file_line=SimpleNamespace(text=lambda: "video.mp4"))
    )
    orchestrator = TrackingOrchestrator(
        main_window=main_window,
        config=object(),
        panels=panels,
    )
    return orchestrator, main_window


def test_show_gpu_info_uses_main_window_parent(monkeypatch) -> None:
    orchestrator, main_window = _make_orchestrator()
    captured: dict[str, object] = {}

    class FakeMessageBox:
        Information = object()

        def __init__(self, parent=None):
            captured["parent"] = parent

        def setWindowTitle(self, title):
            captured["title"] = title

        def setTextFormat(self, text_format):
            captured["text_format"] = text_format

        def setText(self, text):
            captured["text"] = text

        def setIcon(self, icon):
            captured["icon"] = icon

        def exec(self):
            captured["executed"] = True

    monkeypatch.setattr(
        tracking_module,
        "QMessageBox",
        FakeMessageBox,
    )
    monkeypatch.setattr(
        "hydra_suite.utils.gpu_utils.get_device_info",
        lambda: {
            "cuda_available": False,
            "mps_available": True,
            "numba_available": True,
            "tensorrt_available": False,
            "torch_available": False,
        },
    )

    orchestrator.show_gpu_info()

    assert captured["parent"] is main_window
    assert captured["title"] == "GPU & Acceleration Info"
    assert captured["executed"] is True


def test_on_tracking_warning_uses_main_window_parent(monkeypatch) -> None:
    orchestrator, main_window = _make_orchestrator()
    orchestrator._mw = SimpleNamespace(_stop_all_requested=False)
    captured: dict[str, object] = {}

    def fake_information(parent, title, message):
        captured["parent"] = parent
        captured["title"] = title
        captured["message"] = message

    monkeypatch.setattr(tracking_module.QMessageBox, "information", fake_information)

    orchestrator._mw = SimpleNamespace(_stop_all_requested=False)
    main_window = orchestrator._mw
    orchestrator.on_tracking_warning("Heads up", "Check this")

    assert captured == {
        "parent": main_window,
        "title": "Heads up",
        "message": "Check this",
    }


def test_stop_tracking_stops_preview_detection_worker(monkeypatch) -> None:
    orchestrator, _ = _make_orchestrator()
    stopped_workers: list[str] = []
    cleaned_workers: list[str] = []

    class FakeControl:
        def setVisible(self, _visible: bool) -> None:
            return None

        def setValue(self, _value: int) -> None:
            return None

        def setText(self, _text: str) -> None:
            return None

        def setChecked(self, _checked: bool) -> None:
            return None

        def setEnabled(self, _enabled: bool) -> None:
            return None

        def blockSignals(self, _blocked: bool) -> None:
            return None

    orchestrator._mw = SimpleNamespace(
        _stop_all_requested=False,
        _pending_finish_after_interp=True,
        _pending_finish_after_track_videos=True,
        _pending_pose_export_csv_path="pose.csv",
        _pending_video_csv_path="video.csv",
        _pending_video_generation=True,
        _cache_builder_worker=object(),
        merge_worker=object(),
        postprocess_worker=object(),
        dataset_worker=object(),
        interp_worker=object(),
        final_media_export_worker=object(),
        preview_detection_worker=object(),
        tracking_worker=object(),
        session_worker=object(),
        progress_bar=FakeControl(),
        progress_label=FakeControl(),
        _set_ui_controls_enabled=lambda _enabled: None,
        current_video_path="video.mp4",
        _apply_ui_state=lambda _state: None,
        btn_preview=FakeControl(),
        btn_start=FakeControl(),
        _individual_dataset_run_id="run-id",
        current_detection_cache_path="cache.npz",
        current_individual_properties_cache_path="individual_props.npz",
        current_detected_properties_cache_path="detected_props.npz",
        current_detected_cnn_cache_paths={"model": "cnn.npz"},
        current_interpolated_roi_npz_path="roi.npz",
        current_interpolated_pose_csv_path="pose_interp.csv",
        current_interpolated_pose_df=object(),
        current_interpolated_tag_csv_path="tag_interp.csv",
        current_interpolated_tag_df=object(),
        current_interpolated_cnn_csv_paths={"model": "cnn.csv"},
        current_interpolated_cnn_dfs={"model": object()},
        current_interpolated_headtail_csv_path="headtail.csv",
        current_interpolated_headtail_df=object(),
        label_current_fps=FakeControl(),
        label_elapsed_time=FakeControl(),
        label_eta=FakeControl(),
        _tracking_frame_size=(640, 480),
        _cleanup_session_logging=lambda: None,
    )

    monkeypatch.setattr(
        orchestrator,
        "_request_qthread_stop",
        lambda _worker, worker_name, **_kwargs: stopped_workers.append(worker_name),
    )
    monkeypatch.setattr(
        orchestrator,
        "_stop_csv_writer",
        lambda timeout_sec=2.0: None,
    )
    monkeypatch.setattr(
        orchestrator,
        "_cleanup_thread_reference",
        lambda attr_name: cleaned_workers.append(attr_name),
    )

    orchestrator.stop_tracking()

    assert "PreviewDetectionWorker" in stopped_workers
    assert "preview_detection_worker" in cleaned_workers
    assert "tracking_worker" in cleaned_workers
    assert "SessionWorker" in stopped_workers
    assert "session_worker" in cleaned_workers


def test_load_config_uses_main_window_parent(monkeypatch) -> None:
    main_window = object()
    panels = SimpleNamespace(setup=SimpleNamespace(config_status_label=None))
    orchestrator = ConfigOrchestrator(
        main_window=main_window,
        config=object(),
        panels=panels,
    )
    captured: dict[str, object] = {}

    def fake_get_open_file_name(parent, title, directory, file_filter):
        captured["parent"] = parent
        captured["title"] = title
        captured["directory"] = directory
        captured["file_filter"] = file_filter
        return "", ""

    monkeypatch.setattr(
        config_module.QFileDialog,
        "getOpenFileName",
        fake_get_open_file_name,
    )

    orchestrator.load_config()

    assert captured == {
        "parent": main_window,
        "title": "Load Configuration",
        "directory": "",
        "file_filter": "JSON Files (*.json)",
    }


def test_open_parameter_helper_uses_main_window_parent(monkeypatch) -> None:
    main_window = object()
    panels = SimpleNamespace(
        setup=SimpleNamespace(
            file_line=SimpleNamespace(text=lambda: "video.mp4"),
            spin_start_frame=SimpleNamespace(value=lambda: 0),
            spin_end_frame=SimpleNamespace(value=lambda: 100),
            spin_fps=SimpleNamespace(value=lambda: 30.0),
        )
    )
    orchestrator = ConfigOrchestrator(
        main_window=main_window,
        config=object(),
        panels=panels,
    )
    captured: dict[str, object] = {}

    class FakeDialog:
        def __init__(
            self,
            video_path: str,
            cache_path: str,
            start_frame: int,
            end_frame: int,
            params: dict[str, object],
            parent=None,
        ) -> None:
            captured.update(
                {
                    "video_path": video_path,
                    "cache_path": cache_path,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "params": params,
                    "parent": parent,
                }
            )

        def exec(self) -> int:
            return config_module.QDialog.Rejected

    monkeypatch.setattr(
        "hydra_suite.trackerkit.gui.dialogs.parameter_helper.ParameterHelperDialog",
        FakeDialog,
    )
    monkeypatch.setattr(
        ConfigOrchestrator,
        "get_parameters_dict",
        lambda self: {"YOLO_CONFIDENCE_THRESHOLD": 0.5},
    )
    monkeypatch.setattr(
        ConfigOrchestrator,
        "_find_or_plan_optimizer_cache_path",
        lambda self, video_path, params, start_frame, end_frame: (
            "/tmp/cache.npz",
            True,
        ),
    )
    monkeypatch.setattr(config_module.os.path, "exists", lambda path: True)

    orchestrator._open_parameter_helper()

    assert captured == {
        "video_path": "video.mp4",
        "cache_path": "/tmp/cache.npz",
        "start_frame": 0,
        "end_frame": 100,
        "params": {"YOLO_CONFIDENCE_THRESHOLD": 0.5},
        "parent": main_window,
    }


def test_open_parameter_helper_range_warning_uses_main_window_parent(
    monkeypatch,
) -> None:
    main_window = object()
    panels = SimpleNamespace(
        setup=SimpleNamespace(
            file_line=SimpleNamespace(text=lambda: "video.mp4"),
            spin_start_frame=SimpleNamespace(value=lambda: 0),
            spin_end_frame=SimpleNamespace(value=lambda: 9001),
            spin_fps=SimpleNamespace(value=lambda: 30.0),
        )
    )
    orchestrator = ConfigOrchestrator(
        main_window=main_window,
        config=object(),
        panels=panels,
    )
    captured: dict[str, object] = {}

    def fake_warning(parent, title, message, *args, **kwargs):
        captured["parent"] = parent
        captured["title"] = title
        captured["message"] = message
        return config_module.QMessageBox.Ok

    monkeypatch.setattr(config_module.os.path, "exists", lambda path: True)
    monkeypatch.setattr(config_module.QMessageBox, "warning", fake_warning)

    orchestrator._open_parameter_helper()

    assert captured == {
        "parent": main_window,
        "title": "Range Too Large",
        "message": "The selected range spans more than 5 minutes at the current "
        "FPS. For faster optimization, please select a smaller slice "
        "using the 'Start Frame' and 'End Frame' boxes.",
    }


def test_open_parameter_helper_allows_large_frame_count_when_under_five_minutes(
    monkeypatch,
) -> None:
    main_window = object()
    panels = SimpleNamespace(
        setup=SimpleNamespace(
            file_line=SimpleNamespace(text=lambda: "video.mp4"),
            spin_start_frame=SimpleNamespace(value=lambda: 0),
            spin_end_frame=SimpleNamespace(value=lambda: 2000),
            spin_fps=SimpleNamespace(value=lambda: 120.0),
        )
    )
    orchestrator = ConfigOrchestrator(
        main_window=main_window,
        config=object(),
        panels=panels,
    )
    captured: dict[str, object] = {}

    class FakeDialog:
        def __init__(
            self,
            video_path: str,
            cache_path: str,
            start_frame: int,
            end_frame: int,
            params: dict[str, object],
            parent=None,
        ) -> None:
            captured.update(
                {
                    "video_path": video_path,
                    "cache_path": cache_path,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "params": params,
                    "parent": parent,
                }
            )

        def exec(self) -> int:
            return config_module.QDialog.Rejected

    monkeypatch.setattr(
        "hydra_suite.trackerkit.gui.dialogs.parameter_helper.ParameterHelperDialog",
        FakeDialog,
    )
    monkeypatch.setattr(
        ConfigOrchestrator,
        "get_parameters_dict",
        lambda self: {"YOLO_CONFIDENCE_THRESHOLD": 0.5},
    )
    monkeypatch.setattr(
        ConfigOrchestrator,
        "_find_or_plan_optimizer_cache_path",
        lambda self, video_path, params, start_frame, end_frame: (
            "/tmp/cache.npz",
            True,
        ),
    )
    monkeypatch.setattr(config_module.os.path, "exists", lambda path: True)

    orchestrator._open_parameter_helper()

    assert captured == {
        "video_path": "video.mp4",
        "cache_path": "/tmp/cache.npz",
        "start_frame": 0,
        "end_frame": 2000,
        "params": {"YOLO_CONFIDENCE_THRESHOLD": 0.5},
        "parent": main_window,
    }


def test_open_parameter_helper_detection_prompt_uses_main_window_parent(
    monkeypatch,
) -> None:
    main_window = object()
    panels = SimpleNamespace(
        setup=SimpleNamespace(
            file_line=SimpleNamespace(text=lambda: "video.mp4"),
            spin_start_frame=SimpleNamespace(value=lambda: 10),
            spin_end_frame=SimpleNamespace(value=lambda: 100),
            spin_fps=SimpleNamespace(value=lambda: 30.0),
        )
    )
    orchestrator = ConfigOrchestrator(
        main_window=main_window,
        config=object(),
        panels=panels,
    )
    captured: dict[str, object] = {}

    def fake_question(parent, title, message, buttons):
        captured["parent"] = parent
        captured["title"] = title
        captured["message"] = message
        captured["buttons"] = buttons
        return config_module.QMessageBox.No

    monkeypatch.setattr(config_module.os.path, "exists", lambda path: True)
    monkeypatch.setattr(
        ConfigOrchestrator,
        "get_parameters_dict",
        lambda self: {"YOLO_CONFIDENCE_THRESHOLD": 0.5},
    )
    monkeypatch.setattr(
        ConfigOrchestrator,
        "_find_or_plan_optimizer_cache_path",
        lambda self, video_path, params, start_frame, end_frame: (
            "/tmp/cache.npz",
            False,
        ),
    )
    monkeypatch.setattr(config_module.QMessageBox, "question", fake_question)

    orchestrator._open_parameter_helper()

    assert captured["parent"] is main_window
    assert captured["title"] == "Detection Required"
    assert (
        "No detection cache covering frames 10\u2013100 was found."
        in captured["message"]
    )


def test_setup_video_file_adds_recent_video_to_main_window_store(monkeypatch) -> None:
    captured: dict[str, object] = {}
    recent_paths: list[str] = []

    class FakeLineEdit:
        def __init__(self) -> None:
            self.value = ""

        def setText(self, value: str) -> None:
            self.value = value

    class FakeCheckBox:
        def __init__(self) -> None:
            self.checked = False

        def setChecked(self, value: bool) -> None:
            self.checked = value

    class FakeButton:
        def __init__(self) -> None:
            self.enabled = False

        def setEnabled(self, value: bool) -> None:
            self.enabled = value

    class FakeSpinBox:
        def __init__(self) -> None:
            self.value = None

        def setValue(self, value: int) -> None:
            self.value = value

    class FakeLabel:
        def __init__(self) -> None:
            self.text = ""
            self.style = ""

        def setText(self, value: str) -> None:
            self.text = value

        def setStyleSheet(self, value: str) -> None:
            self.style = value

    panels = SimpleNamespace(
        setup=SimpleNamespace(
            file_line=FakeLineEdit(),
            csv_line=FakeLineEdit(),
            btn_detect_fps=FakeButton(),
            spin_start_frame=FakeSpinBox(),
            spin_end_frame=FakeSpinBox(),
            config_status_label=FakeLabel(),
        ),
        postprocess=SimpleNamespace(
            video_out_line=FakeLineEdit(),
            check_video_output=FakeCheckBox(),
        ),
    )
    main_window = SimpleNamespace(
        current_video_path=None,
        current_detection_cache_path="stale-detections.npz",
        current_individual_properties_cache_path="stale-properties.npz",
        roi_selection_active=False,
        btn_test_detection=FakeButton(),
        video_total_frames=240,
        _recents_store=SimpleNamespace(add=recent_paths.append),
        _init_video_player=lambda path: captured.setdefault("video_path", path),
        setWindowTitle=lambda title: captured.setdefault("window_title", title),
        _apply_ui_state=lambda state: captured.setdefault("ui_state", state),
        _show_workspace=lambda: captured.setdefault("workspace_shown", True),
    )
    orchestrator = ConfigOrchestrator(
        main_window=main_window,
        config=object(),
        panels=panels,
    )

    monkeypatch.setattr(config_module.os.path, "isfile", lambda _path: False)

    orchestrator._setup_video_file("/tmp/example.mp4")

    assert recent_paths == ["/tmp/example.mp4"]
    assert main_window.current_detection_cache_path is None
    assert main_window.current_individual_properties_cache_path is None
    assert captured["video_path"] == "/tmp/example.mp4"
    assert captured["window_title"] == "HYDRA - example.mp4"
    assert captured["ui_state"] == "idle"
    assert captured["workspace_shown"] is True


def test_start_tracking_on_video_restores_csv_and_worker_imports(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    class FakeSignal:
        def connect(self, _callback) -> None:
            return None

    class FakeProgress:
        def setVisible(self, _visible: bool) -> None:
            return None

        def setValue(self, _value: int) -> None:
            return None

        def setText(self, _text: str) -> None:
            return None

    class FakeCSVWriterThread:
        def __init__(self, path: str, header=None) -> None:
            captured["csv_path"] = path
            captured["csv_header"] = list(header or [])

        def start(self) -> None:
            captured["csv_started"] = True

    class FakeTrackingWorker:
        def __init__(self, *args, **kwargs) -> None:
            captured["worker_args"] = args
            captured["worker_kwargs"] = kwargs
            self.frame_signal = FakeSignal()
            self.finished_signal = FakeSignal()
            self.progress_signal = FakeSignal()
            self.stats_signal = FakeSignal()
            self.warning_signal = FakeSignal()
            self.pose_exported_model_resolved_signal = FakeSignal()

        def set_parameters(self, params) -> None:
            captured["params"] = dict(params)

        def start(self) -> None:
            captured["worker_started"] = True

        def isRunning(self) -> bool:
            return False

        def update_parameters(self, _params) -> None:
            return None

    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"video")
    csv_path = tmp_path / "tracks.csv"

    setup_panel = SimpleNamespace(
        csv_line=SimpleNamespace(text=lambda: str(csv_path)),
        check_save_confidence=SimpleNamespace(isChecked=lambda: False),
        chk_use_cached_detections=SimpleNamespace(isChecked=lambda: False),
        file_line=SimpleNamespace(text=lambda: str(video_path)),
    )
    tracking_panel = SimpleNamespace(
        chk_enable_backward=SimpleNamespace(isChecked=lambda: False)
    )
    panels = SimpleNamespace(setup=setup_panel, tracking=tracking_panel)

    main_window = SimpleNamespace(
        tracking_worker=None,
        _stop_all_requested=False,
        _pending_finish_after_interp=False,
        _session_result_dataset=None,
        _dataset_was_started=False,
        _show_summary_on_dataset_done=False,
        _session_wall_start=None,
        _session_final_csv_path=None,
        _session_fps_list=[],
        _session_frames_processed=0,
        is_playing=False,
        _tracking_first_frame=False,
        csv_writer_thread=None,
        current_detection_cache_path=None,
        parameters_changed=FakeSignal(),
        progress_bar=FakeProgress(),
        progress_label=FakeProgress(),
        _selected_identity_method=lambda: "",
        get_parameters_dict=lambda: {"DETECTION_METHOD": "background_subtraction"},
        _prepare_tracking_display=lambda: captured.setdefault("prepared", True),
        _apply_ui_state=lambda state: captured.setdefault("ui_state", state),
        _stop_playback=lambda: None,
    )

    orchestrator = TrackingOrchestrator(
        main_window=main_window,
        config=object(),
        panels=panels,
    )

    monkeypatch.setattr(
        "hydra_suite.data.csv_writer.CSVWriterThread",
        FakeCSVWriterThread,
    )
    monkeypatch.setattr(
        "hydra_suite.trackerkit.gui.workers.tracking_worker.TrackingWorker",
        FakeTrackingWorker,
    )
    monkeypatch.setattr(
        tracking_module,
        "candidate_artifact_base_dirs",
        lambda _video_path, preferred_base_dirs=None: [tmp_path],
    )
    monkeypatch.setattr(
        tracking_module,
        "choose_writable_artifact_base_dir",
        lambda _video_path, preferred_base_dirs=None: tmp_path,
    )
    monkeypatch.setattr(
        tracking_module,
        "find_existing_detection_cache_path",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        tracking_module,
        "build_detection_cache_path",
        lambda *_args, **_kwargs: tmp_path / "cache.npz",
    )

    orchestrator.start_tracking_on_video(str(video_path), backward_mode=False)

    assert captured["csv_path"] == str(csv_path)
    assert captured["csv_started"] is True
    assert captured["worker_started"] is True
    assert main_window.csv_writer_thread is not None
    assert main_window.tracking_worker is not None
    assert captured["worker_kwargs"]["detection_cache_path"] == str(
        tmp_path / "cache.npz"
    )
    assert captured["worker_kwargs"]["preview_mode"] is False
    from hydra_suite.core.individual.identity import columns as C
    from hydra_suite.trackerkit.headless_tracking import build_tracking_csv_header

    assert captured["csv_header"] == build_tracking_csv_header(
        False, identity_method=main_window._selected_identity_method()
    )
    assert C.REALTIME_LABEL in captured["csv_header"]
    assert "IdentityAssignedLabel" not in captured["csv_header"]


def test_start_preview_on_video_uses_tracking_worker_when_cache_is_valid(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    class FakeSignal:
        def connect(self, _callback) -> None:
            return None

    class FakeProgress:
        def setVisible(self, _visible: bool) -> None:
            return None

        def setValue(self, _value: int) -> None:
            return None

        def setText(self, _text: str) -> None:
            return None

    class FakeTrackingWorker:
        def __init__(self, *args, **kwargs) -> None:
            captured["worker_args"] = args
            captured["worker_kwargs"] = kwargs
            self.frame_signal = FakeSignal()
            self.finished_signal = FakeSignal()
            self.progress_signal = FakeSignal()
            self.stats_signal = FakeSignal()
            self.warning_signal = FakeSignal()
            self.pose_exported_model_resolved_signal = FakeSignal()

        def set_parameters(self, params) -> None:
            captured["params"] = dict(params)

        def start(self) -> None:
            captured["worker_started"] = True

        def isRunning(self) -> bool:
            return False

    video_path = tmp_path / "video.mp4"
    cache_path = tmp_path / "preview_cache.npz"
    video_path.write_bytes(b"video")
    cache_path.write_bytes(b"cache")

    panels = SimpleNamespace(
        setup=SimpleNamespace(file_line=SimpleNamespace(text=lambda: str(video_path)))
    )

    main_window = SimpleNamespace(
        tracking_worker=None,
        _stop_all_requested=False,
        _pending_finish_after_interp=False,
        is_playing=False,
        _tracking_first_frame=False,
        csv_writer_thread="stale-writer",
        progress_bar=FakeProgress(),
        progress_label=FakeProgress(),
        get_parameters_dict=lambda: {"COMPUTE_RUNTIME": "cpu"},
        _preview_safe_runtime=lambda runtime: runtime,
        _find_or_plan_optimizer_cache_path=lambda *_args, **_kwargs: (
            str(cache_path),
            True,
        ),
        _prepare_tracking_display=lambda: captured.setdefault("prepared", True),
        _apply_ui_state=lambda state: captured.setdefault("ui_state", state),
        _stop_playback=lambda: captured.setdefault("playback_stopped", True),
    )

    orchestrator = TrackingOrchestrator(
        main_window=main_window,
        config=object(),
        panels=panels,
    )
    monkeypatch.setattr(
        orchestrator,
        "_validate_yolo_model_requirements",
        lambda params, mode_label="": True,
    )
    monkeypatch.setattr(
        "hydra_suite.trackerkit.gui.workers.tracking_worker.TrackingWorker",
        FakeTrackingWorker,
    )

    orchestrator.start_preview_on_video(str(video_path))

    assert captured["worker_started"] is True
    assert captured["worker_args"][0] == str(video_path)
    assert captured["worker_kwargs"]["detection_cache_path"] == str(cache_path)
    assert captured["worker_kwargs"]["preview_mode"] is True
    assert captured["worker_kwargs"]["use_cached_detections"] is True
    assert captured["params"]["VISUALIZATION_FREE_MODE"] is False
    assert main_window.csv_writer_thread is None
    assert main_window.tracking_worker is not None
    assert captured["prepared"] is True
    assert captured["ui_state"] == "preview"


def test_start_preview_on_video_downgrades_auxiliary_runtimes(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    class FakeSignal:
        def connect(self, _callback) -> None:
            return None

    class FakeProgress:
        def setVisible(self, _visible: bool) -> None:
            return None

        def setValue(self, _value: int) -> None:
            return None

        def setText(self, _text: str) -> None:
            return None

    class FakeTrackingWorker:
        def __init__(self, *args, **kwargs) -> None:
            self.frame_signal = FakeSignal()
            self.finished_signal = FakeSignal()
            self.progress_signal = FakeSignal()
            self.stats_signal = FakeSignal()
            self.warning_signal = FakeSignal()
            self.pose_exported_model_resolved_signal = FakeSignal()

        def set_parameters(self, params) -> None:
            captured["params"] = dict(params)

        def start(self) -> None:
            return None

        def isRunning(self) -> bool:
            return False

    video_path = tmp_path / "video.mp4"
    cache_path = tmp_path / "preview_cache.npz"
    video_path.write_bytes(b"video")
    cache_path.write_bytes(b"cache")

    panels = SimpleNamespace(
        setup=SimpleNamespace(file_line=SimpleNamespace(text=lambda: str(video_path)))
    )

    main_window = SimpleNamespace(
        tracking_worker=None,
        _stop_all_requested=False,
        _pending_finish_after_interp=False,
        is_playing=False,
        _tracking_first_frame=False,
        csv_writer_thread=None,
        progress_bar=FakeProgress(),
        progress_label=FakeProgress(),
        get_parameters_dict=lambda: {
            "COMPUTE_RUNTIME": "tensorrt",
            "HEADTAIL_COMPUTE_RUNTIME": "onnx_coreml",
            "CNN_COMPUTE_RUNTIME": "onnx_cuda",
            "POSE_MODEL_TYPE": "yolo",
        },
        _preview_safe_runtime=lambda runtime: {
            "onnx_cpu": "cpu",
            "onnx_coreml": "mps",
            "onnx_cuda": "cuda",
            "tensorrt": "cuda",
        }.get(runtime, runtime),
        _find_or_plan_optimizer_cache_path=lambda *_args, **_kwargs: (
            str(cache_path),
            True,
        ),
        _prepare_tracking_display=lambda: None,
        _apply_ui_state=lambda _state: None,
        _stop_playback=lambda: None,
    )

    orchestrator = TrackingOrchestrator(
        main_window=main_window,
        config=object(),
        panels=panels,
    )
    monkeypatch.setattr(
        orchestrator,
        "_validate_yolo_model_requirements",
        lambda params, mode_label="": True,
    )
    monkeypatch.setattr(
        "hydra_suite.trackerkit.gui.workers.tracking_worker.TrackingWorker",
        FakeTrackingWorker,
    )

    orchestrator.start_preview_on_video(str(video_path))

    assert captured["params"]["COMPUTE_RUNTIME"] == "cuda"
    assert captured["params"]["HEADTAIL_COMPUTE_RUNTIME"] == "mps"
    assert captured["params"]["CNN_COMPUTE_RUNTIME"] == "cuda"
    assert captured["params"]["POSE_RUNTIME_FLAVOR"] == "cuda"


def test_clear_detection_caches_deletes_all_current_video_cache_files(
    monkeypatch,
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "clip.mp4"
    cache_dir = tmp_path / "clip_caches"
    cache_dir.mkdir()
    detection_cache = cache_dir / "clip_detection_cache_model123.npz"
    optimizer_cache = cache_dir / "clip_yolo_model123_r100_opt_cache.npz"
    pose_cache = cache_dir / "clip_pose_cache_keep_me_0_10.npz"
    tag_cache = cache_dir / "clip_apriltag_cache_keep_me_0_10.npz"
    classify_cache = cache_dir / "clip_classify_cache_demo_keep_me_0_10.npz"
    detected_props_cache = cache_dir / "clip_detected_props_cache_keep_me_0_10.npz"
    other_file = cache_dir / "clip_interpolated_headtail.csv"
    detection_cache.write_bytes(b"cache")
    optimizer_cache.write_bytes(b"cache")
    pose_cache.write_bytes(b"cache")
    tag_cache.write_bytes(b"cache")
    classify_cache.write_bytes(b"cache")
    detected_props_cache.write_bytes(b"cache")
    other_file.write_text("keep")
    detection_cache.with_suffix(".autotune_state.json").write_text("{}")
    detection_cache.with_name(
        detection_cache.stem + "_confidence_regions.json"
    ).write_text("{}")

    orchestrator, _main_window = _make_orchestrator()
    orchestrator._mw = SimpleNamespace(
        _has_active_progress_task=lambda: False,
        current_detection_cache_path=str(detection_cache),
        current_individual_properties_cache_path=str(pose_cache),
    )
    orchestrator._panels = SimpleNamespace(
        setup=SimpleNamespace(
            file_line=SimpleNamespace(text=lambda: str(video_path)),
            csv_line=SimpleNamespace(text=lambda: ""),
        )
    )

    info_calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        tracking_module,
        "candidate_artifact_base_dirs",
        lambda _video_path, preferred_base_dirs=None: [tmp_path],
    )
    monkeypatch.setattr(
        tracking_module.QMessageBox,
        "question",
        lambda *args, **kwargs: tracking_module.QMessageBox.Yes,
    )
    monkeypatch.setattr(
        tracking_module.QMessageBox,
        "information",
        lambda _parent, title, message: info_calls.append((title, message)),
    )

    orchestrator.clear_detection_caches()

    assert not detection_cache.exists()
    assert not optimizer_cache.exists()
    assert not pose_cache.exists()
    assert not tag_cache.exists()
    assert not classify_cache.exists()
    assert not detected_props_cache.exists()
    assert not detection_cache.with_suffix(".autotune_state.json").exists()
    assert not detection_cache.with_name(
        detection_cache.stem + "_confidence_regions.json"
    ).exists()
    assert other_file.exists()
    assert orchestrator._mw.current_detection_cache_path is None
    assert orchestrator._mw.current_individual_properties_cache_path is None
    assert info_calls == [
        (
            "Caches Cleared",
            "Deleted 6 cache file(s) for the current video.",
        )
    ]


def test_collect_worker_props_path_stores_detected_export_caches(
    tmp_path: Path,
) -> None:
    orchestrator, _main_window = _make_orchestrator()
    detected_props_path = tmp_path / "detected_props.npz"
    orchestrator._mw = SimpleNamespace(
        tracking_worker=SimpleNamespace(
            individual_properties_cache_path="",
            detected_properties_cache_path=str(detected_props_path),
        ),
        current_individual_properties_cache_path=None,
        current_detected_properties_cache_path=None,
    )

    orchestrator._collect_worker_props_path()

    assert orchestrator._mw.current_detected_properties_cache_path == str(
        detected_props_path
    )


def test_format_video_track_label_prefers_unique_identity_key() -> None:
    assert media_export.format_video_track_label(7, "apriltag=12") == "Tag 12"
    assert (
        media_export.format_video_track_label(
            7,
            "cnn:uid:color=red|cnn:uid:shape=circle",
        )
        == "red / circle"
    )
    assert (
        media_export.format_video_track_label(
            7,
            "cnn:uid=color:red+shape:circle",
        )
        == "red / circle"
    )
    assert media_export.format_video_track_label(7, np.nan) == "ID7"


def test_preextract_traj_arrays_uses_unique_identity_labels_when_available() -> None:
    trajectories_df = pd.DataFrame(
        [
            {
                "FrameID": 1,
                "TrajectoryID": 3,
                "X": 10.0,
                "Y": 20.0,
                "Theta": 0.0,
                "UniqueIdentityKey": "apriltag=8",
            },
            {
                "FrameID": 2,
                "TrajectoryID": 4,
                "X": 11.0,
                "Y": 21.0,
                "Theta": 0.1,
            },
        ]
    )

    arrays = media_export.preextract_traj_arrays(
        trajectories_df,
        show_pose=False,
        pose_column_triplets=[],
        show_trails=False,
    )

    label_texts = arrays[4]
    assert list(label_texts) == ["Tag 8", "ID4"]


def test_build_video_track_color_key_array_prefers_identity_when_available() -> None:
    trajectories_df = pd.DataFrame(
        [
            {
                "FrameID": 1,
                "TrajectoryID": 3,
                "UniqueIdentityKey": "apriltag=8",
            },
            {
                "FrameID": 2,
                "TrajectoryID": 4,
                "IdentityFinalLabel": "worker_a",
            },
            {
                "FrameID": 3,
                "TrajectoryID": 9,
            },
        ]
    )

    color_keys = media_export.build_video_track_color_key_array(trajectories_df)

    assert list(color_keys) == [
        "identity:apriltag=8",
        "identity:worker_a",
        "trajectory:9",
    ]


def test_build_precomputed_color_palette_reuses_identity_colors_across_tracks() -> None:
    colors = [(10, 20, 30), (40, 50, 60), (70, 80, 90)]
    track_ids = np.asarray([3, 8, 2], dtype=np.int32)
    color_keys = np.asarray(
        ["identity:worker_a", "identity:worker_a", "trajectory:2"],
        dtype=object,
    )

    row_colors = media_export.build_precomputed_color_palette(
        colors,
        track_ids,
        color_keys,
    )

    assert row_colors[0] == (10, 20, 30)
    assert row_colors[1] == (10, 20, 30)
    assert row_colors[2] == (70, 80, 90)
