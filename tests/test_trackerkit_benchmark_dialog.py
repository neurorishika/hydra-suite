from __future__ import annotations

import gc
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

from hydra_suite.trackerkit.benchmarking import (  # noqa: E402
    BenchmarkRecommendation,
    BenchmarkResult,
    BenchmarkTargetSpec,
)
from hydra_suite.trackerkit.gui.dialogs import (  # noqa: E402
    benchmark_dialog as dialog_module,
)
from hydra_suite.widgets.dialogs import HYDRA_DIALOG_TEXT_COLOR  # noqa: E402


@pytest.fixture()
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


@pytest.fixture(autouse=True)
def cleanup_qt_widgets(qapp):
    yield
    for widget in list(qapp.topLevelWidgets()):
        widget.close()
        widget.deleteLater()
    qapp.processEvents()
    gc.collect()


def _make_main_window() -> SimpleNamespace:
    return SimpleNamespace(
        _setup_panel=SimpleNamespace(
            file_line=SimpleNamespace(text=lambda: str(Path("/tmp/video.mp4"))),
            spin_resize=SimpleNamespace(value=lambda: 1.0),
        ),
        _detection_panel=SimpleNamespace(
            spin_reference_body_size=SimpleNamespace(value=lambda: 40.0),
            spin_reference_aspect_ratio=SimpleNamespace(value=lambda: 2.0),
        ),
        _identity_panel=SimpleNamespace(
            spin_individual_padding=SimpleNamespace(value=lambda: 0.25),
        ),
        _is_realtime_tracking_mode_enabled=lambda: False,
    )


def _make_geometry() -> SimpleNamespace:
    return SimpleNamespace(
        frame_width=1920,
        frame_height=1080,
        effective_frame_width=960,
        effective_frame_height=540,
        canonical_crop_width=64,
        canonical_crop_height=32,
        resize_factor=0.5,
        reference_body_size=40.0,
        reference_aspect_ratio=2.0,
        padding_fraction=0.25,
    )


def test_runtime_selection_widget_returns_checked_runtimes(qapp) -> None:
    widget = dialog_module._RuntimeSelectionWidget(["cpu", "mps", "cuda"])

    widget._checkboxes["mps"].setChecked(False)

    assert widget.selected_runtimes() == ["cpu", "cuda"]


def test_batch_size_selection_widget_adds_removes_and_resets(qapp) -> None:
    widget = dialog_module._BatchSizeSelectionWidget([1, 4, 8])

    widget._batch_spin.setValue(16)
    widget._add_current_value()
    assert widget.values() == [1, 4, 8, 16]

    widget._selected_combo.setCurrentIndex(widget._selected_combo.findData(4))
    widget._remove_selected_value()
    assert widget.values() == [1, 8, 16]

    widget._reset_defaults()
    assert widget.values() == [1, 4, 8]


def test_tracker_benchmark_dialog_collects_validated_target_selections(
    qapp,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = BenchmarkTargetSpec(
        key="detection_direct",
        label="Detection",
        pipeline="obb",
        model_path="/tmp/model.pt",
        runtimes=["cpu", "mps"],
        batch_sizes=[1, 4],
        current_runtime="cpu",
        current_batch_size=4,
    )

    monkeypatch.setattr(
        dialog_module,
        "collect_active_targets",
        lambda _main_window: ([target], []),
    )
    monkeypatch.setattr(
        dialog_module,
        "derive_benchmark_geometry_from_video",
        lambda *args, **kwargs: _make_geometry(),
    )
    monkeypatch.setattr(
        dialog_module,
        "lookup_cached_recommendation",
        lambda *args, **kwargs: None,
    )

    dialog = dialog_module.TrackerBenchmarkDialog(_make_main_window())
    row = dialog._row_widgets["detection_direct"]
    runtime_selector = row["runtimes"]
    batch_selector = row["batches"]

    runtime_selector._checkboxes["cpu"].setChecked(False)
    runtime_selector._checkboxes["mps"].setChecked(False)
    selected, errors = dialog._selected_targets()
    assert selected == []
    assert errors == ["Detection: select at least one runtime."]

    runtime_selector._checkboxes["cpu"].setChecked(True)
    batch_selector._batch_spin.setValue(16)
    batch_selector._add_current_value()
    selected, errors = dialog._selected_targets()

    assert errors == []
    assert len(selected) == 1
    assert selected[0].runtimes == ["cpu"]
    assert selected[0].batch_sizes == [1, 4, 16]
    assert target.runtimes == ["cpu", "mps"]
    assert target.batch_sizes == [1, 4]


def test_tracker_benchmark_dialog_uses_bright_text_on_dark_cards(
    qapp,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = BenchmarkTargetSpec(
        key="detection_direct",
        label="Detection",
        pipeline="obb",
        model_path="/tmp/model.pt",
        runtimes=["cpu", "mps"],
        batch_sizes=[1, 4],
        current_runtime="cpu",
        current_batch_size=4,
    )

    monkeypatch.setattr(
        dialog_module,
        "collect_active_targets",
        lambda _main_window: ([target], []),
    )
    monkeypatch.setattr(
        dialog_module,
        "derive_benchmark_geometry_from_video",
        lambda *args, **kwargs: _make_geometry(),
    )
    monkeypatch.setattr(
        dialog_module,
        "lookup_cached_recommendation",
        lambda *args, **kwargs: None,
    )

    dialog = dialog_module.TrackerBenchmarkDialog(_make_main_window())
    stylesheet = dialog.styleSheet()

    assert "QFrame#BenchmarkTargetCard {" in stylesheet
    assert "background-color: #252526;" in stylesheet
    assert "QTableWidget#BenchmarkResultsTable::item:alternate {" in stylesheet
    assert "alternate-background-color: #2d2d30;" in stylesheet
    assert f"color: {HYDRA_DIALOG_TEXT_COLOR};" in stylesheet


def test_tracker_benchmark_dialog_collects_sequential_frame_and_crop_batches(
    qapp,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = BenchmarkTargetSpec(
        key="detection_sequential",
        label="Detection (Sequential)",
        pipeline="sequential",
        model_path="/tmp/model.pt",
        runtimes=["cpu", "mps"],
        batch_sizes=[1, 4],
        individual_batch_sizes=[8, 12],
        current_runtime="cpu",
        current_batch_size=4,
        current_individual_batch_size=12,
        benchmark_context={"max_targets": 12},
    )

    monkeypatch.setattr(
        dialog_module,
        "collect_active_targets",
        lambda _main_window: ([target], []),
    )
    monkeypatch.setattr(
        dialog_module,
        "derive_benchmark_geometry_from_video",
        lambda *args, **kwargs: _make_geometry(),
    )
    monkeypatch.setattr(
        dialog_module,
        "lookup_cached_recommendation",
        lambda *args, **kwargs: None,
    )

    dialog = dialog_module.TrackerBenchmarkDialog(_make_main_window())
    row = dialog._row_widgets["detection_sequential"]

    assert row["individual_batches"] is not None
    row["batches"]._batch_spin.setValue(6)
    row["batches"]._add_current_value()
    row["individual_batches"]._batch_spin.setValue(16)
    row["individual_batches"]._add_current_value()

    selected, errors = dialog._selected_targets()

    assert errors == []
    assert len(selected) == 1
    assert selected[0].batch_sizes == [1, 4, 6]
    assert selected[0].individual_batch_sizes == [8, 12, 16]


def test_tracker_benchmark_dialog_results_table_shows_dual_batch_recommendation(
    qapp,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = BenchmarkTargetSpec(
        key="detection_sequential",
        label="Detection (Sequential)",
        pipeline="sequential",
        model_path="/tmp/model.pt",
        runtimes=["cpu"],
        batch_sizes=[4],
        individual_batch_sizes=[12],
        benchmark_context={"max_targets": 12},
    )

    monkeypatch.setattr(
        dialog_module,
        "collect_active_targets",
        lambda _main_window: ([target], []),
    )
    monkeypatch.setattr(
        dialog_module,
        "derive_benchmark_geometry_from_video",
        lambda *args, **kwargs: _make_geometry(),
    )
    monkeypatch.setattr(
        dialog_module,
        "lookup_cached_recommendation",
        lambda *args, **kwargs: None,
    )

    dialog = dialog_module.TrackerBenchmarkDialog(_make_main_window())
    dialog._target_labels = {target.key: target.label}
    dialog._recommendations = {
        target.key: BenchmarkRecommendation(
            target_key=target.key,
            target_label=target.label,
            runtime="cpu",
            runtime_label="CPU",
            batch_size=4,
            individual_batch_size=12,
            mean_ms=10.0,
            throughput_fps=400.0,
            reason="test",
            model_path=target.model_path,
            mean_per_frame_ms=2.5,
        )
    }
    dialog._result_payload = {
        target.key: [
            BenchmarkResult(
                model_type="sequential",
                model_path=target.model_path,
                runtime="cpu",
                runtime_label="CPU",
                batch_size=4,
                individual_batch_size=12,
                input_shape=(540, 960),
                warmup_iters=1,
                bench_iters=2,
                mean_ms=10.0,
                mean_per_frame_ms=2.5,
                throughput_fps=400.0,
            )
        ]
    }

    dialog._populate_results_table()

    assert dialog.results_table.item(0, 2).text() == "F4/I12"
    assert dialog.results_table.item(0, 9).text() == "Yes"


def test_tracker_benchmark_dialog_clarifies_sleap_pose_runtime_paths(
    qapp,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = BenchmarkTargetSpec(
        key="pose_sleap",
        label="Pose Extraction",
        pipeline="pose",
        model_path="/tmp/pose_model",
        runtimes=["mps", "onnx_coreml"],
        batch_sizes=[1],
        backend_family="sleap",
        benchmark_context={"keypoint_names": ["nose", "tail"]},
    )

    monkeypatch.setattr(
        dialog_module,
        "collect_active_targets",
        lambda _main_window: ([target], []),
    )
    monkeypatch.setattr(
        dialog_module,
        "derive_benchmark_geometry_from_video",
        lambda *args, **kwargs: _make_geometry(),
    )
    monkeypatch.setattr(
        dialog_module,
        "lookup_cached_recommendation",
        lambda *args, **kwargs: None,
    )

    dialog = dialog_module.TrackerBenchmarkDialog(_make_main_window())
    dialog._result_payload = {
        target.key: [
            BenchmarkResult(
                model_type="pose",
                model_path=target.model_path,
                runtime="mps",
                runtime_label="MPS",
                batch_size=1,
                input_shape=(32, 64),
                warmup_iters=1,
                bench_iters=2,
                mean_ms=50.0,
                mean_per_frame_ms=50.0,
                throughput_fps=20.0,
            ),
            BenchmarkResult(
                model_type="pose",
                model_path=target.model_path,
                runtime="onnx_coreml",
                runtime_label="ONNX (CoreML)",
                batch_size=1,
                input_shape=(32, 64),
                warmup_iters=1,
                bench_iters=2,
                mean_ms=12.0,
                mean_per_frame_ms=12.0,
                throughput_fps=83.3,
            ),
        ]
    }

    dialog._populate_results_table()

    assert "service path" in dialog._target_comparison_note(target)
    assert dialog.results_table.item(0, 1).text() == "MPS [Native Service]"
    assert dialog.results_table.item(1, 1).text() == "ONNX (CoreML) [Exported Direct]"


def test_partial_rerun_preserves_cached_recommendations_for_other_targets(
    qapp,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detection_target = BenchmarkTargetSpec(
        key="detection_direct",
        label="Detection",
        pipeline="obb",
        model_path="/tmp/detect.pt",
        runtimes=["cpu"],
        batch_sizes=[4],
    )
    pose_target = BenchmarkTargetSpec(
        key="pose_yolo",
        label="Pose",
        pipeline="pose",
        model_path="/tmp/pose.pt",
        runtimes=["cpu"],
        batch_sizes=[8],
    )
    cached_detection = BenchmarkRecommendation(
        target_key=detection_target.key,
        target_label=detection_target.label,
        runtime="cpu",
        runtime_label="CPU",
        batch_size=4,
        mean_ms=12.0,
        throughput_fps=333.0,
        reason="cached detection",
        model_path=detection_target.model_path,
        mean_per_frame_ms=3.0,
    )
    cached_pose = BenchmarkRecommendation(
        target_key=pose_target.key,
        target_label=pose_target.label,
        runtime="mps",
        runtime_label="MPS",
        batch_size=8,
        mean_ms=9.0,
        throughput_fps=888.0,
        reason="cached pose",
        model_path=pose_target.model_path,
        mean_per_frame_ms=1.125,
    )
    rerun_detection = BenchmarkRecommendation(
        target_key=detection_target.key,
        target_label=detection_target.label,
        runtime="mps",
        runtime_label="MPS",
        batch_size=6,
        mean_ms=10.0,
        throughput_fps=600.0,
        reason="rerun detection",
        model_path=detection_target.model_path,
        mean_per_frame_ms=1.67,
    )

    monkeypatch.setattr(
        dialog_module,
        "collect_active_targets",
        lambda _main_window: ([detection_target, pose_target], []),
    )
    monkeypatch.setattr(
        dialog_module,
        "derive_benchmark_geometry_from_video",
        lambda *args, **kwargs: _make_geometry(),
    )

    def fake_lookup(target, *_args, **_kwargs):
        if target.key == detection_target.key:
            return cached_detection
        if target.key == pose_target.key:
            return cached_pose
        return None

    monkeypatch.setattr(dialog_module, "lookup_cached_recommendation", fake_lookup)
    monkeypatch.setattr(
        dialog_module,
        "store_cached_results",
        lambda target, *_args, **_kwargs: (
            rerun_detection if target.key == detection_target.key else None
        ),
    )

    dialog = dialog_module.TrackerBenchmarkDialog(_make_main_window())
    dialog._active_targets = [detection_target]
    dialog._on_completed(
        {
            "cancelled": False,
            "results": {
                detection_target.key: [
                    BenchmarkResult(
                        model_type="obb",
                        model_path=detection_target.model_path,
                        runtime="mps",
                        runtime_label="MPS",
                        batch_size=6,
                        input_shape=(540, 960),
                        warmup_iters=1,
                        bench_iters=2,
                        mean_ms=10.0,
                        mean_per_frame_ms=1.67,
                        throughput_fps=600.0,
                    )
                ]
            },
        }
    )

    recommendations = dialog.recommendations()

    assert recommendations[detection_target.key].runtime == "mps"
    assert recommendations[detection_target.key].batch_size == 6
    assert recommendations[pose_target.key].runtime == "mps"
    assert recommendations[pose_target.key].batch_size == 8


def test_tracker_benchmark_dialog_results_table_shows_extended_metrics(
    qapp,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = BenchmarkTargetSpec(
        key="detection_direct",
        label="Detection",
        pipeline="obb",
        model_path="/tmp/model.pt",
        runtimes=["cpu"],
        batch_sizes=[1],
    )

    monkeypatch.setattr(
        dialog_module,
        "collect_active_targets",
        lambda _main_window: ([target], []),
    )
    monkeypatch.setattr(
        dialog_module,
        "derive_benchmark_geometry_from_video",
        lambda *args, **kwargs: _make_geometry(),
    )
    monkeypatch.setattr(
        dialog_module,
        "lookup_cached_recommendation",
        lambda *args, **kwargs: None,
    )

    dialog = dialog_module.TrackerBenchmarkDialog(_make_main_window())

    assert dialog.results_table.columnCount() == 10
    assert dialog.results_table.horizontalHeaderItem(4).text() == "Per Frame (ms)"
    assert dialog.results_table.horizontalHeaderItem(6).text() == "RAM Peak (MB)"
    assert dialog.results_table.horizontalHeaderItem(7).text() == "Accel Peak (MB)"
    assert (
        dialog.results_table.horizontalHeaderItem(3).toolTip()
        == "Mean wall-clock time for one timed batch inference call. Setup, artifact export, and warmup are excluded."
    )
    assert (
        dialog.results_table.horizontalHeaderItem(4).toolTip()
        == "Mean runtime per frame or crop, computed as batch runtime divided by batch size."
    )
    assert (
        dialog.results_table.horizontalHeaderItem(7).toolTip()
        == "Peak accelerator memory observed during timed iterations when the backend exposes it."
    )


def test_tracker_benchmark_dialog_shows_detection_comparison_note(
    qapp,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = BenchmarkTargetSpec(
        key="detection_sequential",
        label="Detection (Sequential)",
        pipeline="sequential",
        model_path="/tmp/model.pt",
        runtimes=["cpu"],
        batch_sizes=[1],
        benchmark_context={"max_targets": 12},
    )

    monkeypatch.setattr(
        dialog_module,
        "collect_active_targets",
        lambda _main_window: ([target], []),
    )
    monkeypatch.setattr(
        dialog_module,
        "derive_benchmark_geometry_from_video",
        lambda *args, **kwargs: _make_geometry(),
    )
    monkeypatch.setattr(
        dialog_module,
        "lookup_cached_recommendation",
        lambda *args, **kwargs: None,
    )

    dialog = dialog_module.TrackerBenchmarkDialog(_make_main_window())
    note_label = dialog._row_widgets["detection_sequential"]["comparison_note"]

    assert note_label.isHidden() is False
    assert (
        "stage-1 full-frame box detection plus stage-2 crop OBB per frame"
        in note_label.text()
    )
    assert "12 animals/frame" in note_label.text()
