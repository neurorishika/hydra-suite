from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace

from hydra_suite.runtime import resolver as benchmarking_resolver
from hydra_suite.trackerkit import benchmarking


def test_build_benchmark_geometry_scales_frame_and_crop() -> None:
    geometry = benchmarking.build_benchmark_geometry_from_dimensions(
        frame_width=1920,
        frame_height=1080,
        resize_factor=0.5,
        reference_body_size=40.0,
        reference_aspect_ratio=2.0,
        padding_fraction=0.25,
    )

    assert geometry.effective_frame_width == 960
    assert geometry.effective_frame_height == 540
    assert geometry.canonical_crop_width == 50
    assert geometry.canonical_crop_height == 25


def test_choose_recommendation_prefers_best_per_frame_latency() -> None:
    target = benchmarking.BenchmarkTargetSpec(
        key="detection_direct",
        label="Detection",
        pipeline="obb",
        model_path="/tmp/model.pt",
        runtimes=["cpu"],
        batch_sizes=[4, 8],
    )
    smaller = benchmarking.BenchmarkResult(
        model_type="obb",
        model_path=target.model_path,
        runtime="cpu",
        runtime_label="CPU",
        batch_size=4,
        input_shape=(640, 640),
        warmup_iters=2,
        bench_iters=5,
        mean_ms=102.0,
        mean_per_frame_ms=25.5,
        throughput_fps=39.2,
    )
    larger = benchmarking.BenchmarkResult(
        model_type="obb",
        model_path=target.model_path,
        runtime="cpu",
        runtime_label="CPU",
        batch_size=8,
        input_shape=(640, 640),
        warmup_iters=2,
        bench_iters=5,
        mean_ms=100.0,
        mean_per_frame_ms=12.5,
        throughput_fps=80.0,
    )

    recommendation = benchmarking.choose_recommendation(target, [smaller, larger])

    assert recommendation is not None
    assert recommendation.batch_size == 8
    assert recommendation.mean_per_frame_ms == 12.5


def test_store_and_lookup_cached_recommendation(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(benchmarking, "get_config_dir", lambda: tmp_path)
    monkeypatch.setattr(benchmarking, "build_hardware_fingerprint", lambda: "hw")

    target = benchmarking.BenchmarkTargetSpec(
        key="pose_yolo",
        label="Pose Extraction",
        pipeline="pose",
        model_path=str(tmp_path / "pose.pt"),
        runtimes=["cpu", "mps"],
        batch_sizes=[1, 4],
    )
    Path(target.model_path).write_text("stub", encoding="utf-8")
    geometry = benchmarking.build_benchmark_geometry_from_dimensions(
        frame_width=1280,
        frame_height=720,
        resize_factor=1.0,
        reference_body_size=32.0,
        reference_aspect_ratio=2.0,
        padding_fraction=0.1,
    )
    result = benchmarking.BenchmarkResult(
        model_type="pose",
        model_path=target.model_path,
        runtime="mps",
        runtime_label="MPS",
        batch_size=4,
        input_shape=(32, 16),
        warmup_iters=2,
        bench_iters=5,
        mean_ms=12.0,
        mean_per_frame_ms=3.0,
        throughput_fps=333.3,
    )

    stored = benchmarking.store_cached_results(
        target,
        geometry,
        [result],
        realtime_enabled=False,
    )
    loaded = benchmarking.lookup_cached_recommendation(
        target,
        geometry,
        realtime_enabled=False,
    )

    assert stored is not None
    assert loaded is not None
    assert loaded.runtime == "mps"
    assert loaded.batch_size == 4


def test_benchmark_result_compute_stats_tracks_per_frame_metric() -> None:
    result = benchmarking.BenchmarkResult(
        model_type="obb",
        model_path="/tmp/model.pt",
        runtime="cpu",
        runtime_label="CPU",
        batch_size=8,
        input_shape=(640, 640),
        warmup_iters=0,
        bench_iters=2,
        latencies_ms=[80.0, 120.0],
    )

    result.compute_stats()

    assert result.mean_ms == 100.0
    assert result.mean_per_frame_ms == 12.5
    assert result.throughput_fps == 80.0


def test_collect_active_targets_includes_sleap_pose_target(tmp_path: Path) -> None:
    sleap_model_dir = tmp_path / "sleap_model"
    sleap_model_dir.mkdir()

    main_window = SimpleNamespace(
        _detection_panel=SimpleNamespace(
            spin_detection_batch_size=SimpleNamespace(value=lambda: 4),
            combo_yolo_obb_mode=SimpleNamespace(currentIndex=lambda: 0),
        ),
        _identity_panel=SimpleNamespace(
            spin_pose_batch=SimpleNamespace(value=lambda: 6),
        ),
        _setup_panel=SimpleNamespace(spin_max_targets=SimpleNamespace(value=lambda: 8)),
        _compute_runtime_options_for_current_ui=lambda: [
            ("CPU", "cpu"),
            ("MPS", "mps"),
        ],
        _selected_compute_runtime=lambda: "mps",
        _is_yolo_detection_mode=lambda: False,
        _get_selected_yolo_model_path=lambda: "",
        _get_selected_yolo_detect_model_path=lambda: "",
        _get_selected_yolo_crop_obb_model_path=lambda: "",
        _identity_config=lambda: {"cnn_classifiers": []},
        _is_pose_inference_enabled=lambda: True,
        _current_pose_backend_key=lambda: "sleap",
        _get_resolved_pose_model_dir=lambda backend: str(sleap_model_dir),
        _selected_pose_runtime_flavor=lambda: "onnx_cpu",
        _load_pose_skeleton_keypoint_names=lambda: ["nose", "tail"],
        _selected_pose_sleap_env=lambda: "sleap-mps",
        _selected_cnn_runtime=lambda: "cpu",
    )

    targets, notices = benchmarking.collect_active_targets(main_window)

    assert notices == ["Detection benchmarking currently supports YOLO OBB mode only."]
    pose_targets = [target for target in targets if target.key == "pose_sleap"]
    assert len(pose_targets) == 1
    pose_target = pose_targets[0]
    assert pose_target.backend_family == "sleap"
    assert pose_target.runtimes
    assert pose_target.benchmark_context["sleap_env"] == "sleap-mps"
    assert pose_target.benchmark_context["keypoint_names"] == ["nose", "tail"]


def test_collect_active_targets_resolves_relative_direct_model_path(
    monkeypatch,
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "models" / "obb" / "detector.pt"
    model_path.parent.mkdir(parents=True)
    model_path.write_text("stub", encoding="utf-8")

    monkeypatch.setattr(
        benchmarking,
        "resolve_model_path",
        lambda model: (
            str(model_path) if str(model) == "obb/detector.pt" else str(model)
        ),
    )

    main_window = SimpleNamespace(
        _detection_panel=SimpleNamespace(
            spin_detection_batch_size=SimpleNamespace(value=lambda: 4),
            combo_yolo_obb_mode=SimpleNamespace(currentIndex=lambda: 0),
        ),
        _identity_panel=SimpleNamespace(
            spin_pose_batch=SimpleNamespace(value=lambda: 4)
        ),
        _setup_panel=SimpleNamespace(spin_max_targets=SimpleNamespace(value=lambda: 8)),
        _compute_runtime_options_for_current_ui=lambda: [("CPU", "cpu")],
        _selected_compute_runtime=lambda: "cpu",
        _is_yolo_detection_mode=lambda: True,
        _get_selected_yolo_model_path=lambda: "obb/detector.pt",
        _get_selected_yolo_detect_model_path=lambda: "",
        _get_selected_yolo_crop_obb_model_path=lambda: "",
        _identity_config=lambda: {"cnn_classifiers": []},
        _is_pose_inference_enabled=lambda: False,
        _selected_cnn_runtime=lambda: "cpu",
    )

    targets, notices = benchmarking.collect_active_targets(main_window)

    assert not notices
    assert len(targets) == 1
    assert targets[0].key == "detection_direct"
    assert targets[0].model_path == str(model_path)
    assert targets[0].benchmark_context["max_targets"] == 8


def test_collect_active_targets_skips_vitpose_pose_target() -> None:
    main_window = SimpleNamespace(
        _detection_panel=SimpleNamespace(
            spin_detection_batch_size=SimpleNamespace(value=lambda: 4),
            combo_yolo_obb_mode=SimpleNamespace(currentIndex=lambda: 0),
        ),
        _identity_panel=SimpleNamespace(
            spin_pose_batch=SimpleNamespace(value=lambda: 4),
        ),
        _setup_panel=SimpleNamespace(spin_max_targets=SimpleNamespace(value=lambda: 8)),
        _compute_runtime_options_for_current_ui=lambda: [("CPU", "cpu")],
        _selected_compute_runtime=lambda: "cpu",
        _is_yolo_detection_mode=lambda: False,
        _get_selected_yolo_model_path=lambda: "",
        _get_selected_yolo_detect_model_path=lambda: "",
        _get_selected_yolo_crop_obb_model_path=lambda: "",
        _identity_config=lambda: {"cnn_classifiers": []},
        _is_pose_inference_enabled=lambda: True,
        _current_pose_backend_key=lambda: "vitpose",
        _get_resolved_pose_model_dir=lambda backend: "/tmp/vitpose_model",
        _selected_pose_runtime_flavor=lambda: "cpu",
        _load_pose_skeleton_keypoint_names=lambda: ["nose", "tail"],
        _selected_pose_sleap_env=lambda: "sleap",
        _selected_cnn_runtime=lambda: "cpu",
    )

    targets, notices = benchmarking.collect_active_targets(main_window)

    assert not any(target.key.startswith("pose_") for target in targets)
    assert "Pose benchmarking does not support ViTPose yet." in notices


def test_bench_pose_streams_phase_messages(monkeypatch, tmp_path: Path) -> None:
    messages = []
    call_counts = {"warmup": 0, "predict": 0, "close": 0}

    class FakeBackend:
        def __init__(self, model_path, device, batch_size, **_kwargs):
            self.model_path = model_path
            self.device = device
            self.batch_size = batch_size

        def warmup(self):
            call_counts["warmup"] += 1

        def predict_batch(self, crops):
            call_counts["predict"] += 1
            return [object() for _ in crops]

        def close(self):
            call_counts["close"] += 1

    monkeypatch.setitem(
        sys.modules,
        "hydra_suite.core.identity.pose.backends.yolo",
        types.SimpleNamespace(
            YoloNativeBackend=FakeBackend,
            auto_export_yolo_model=lambda config, flavor, runtime_device=None: str(
                tmp_path / "pose.onnx"
            ),
        ),
    )

    result = benchmarking.bench_pose(
        model_path=str(tmp_path / "pose.pt"),
        tier="cpu",
        warmup=1,
        iterations=2,
        batch_size=4,
        crop_size=96,
        backend_family="yolo",
        status_callback=messages.append,
    )

    assert result.success is True
    assert call_counts == {"warmup": 1, "predict": 3, "close": 1}
    assert any("Pose benchmark setup started" in message for message in messages)
    assert any("Creating YOLO pose backend" in message for message in messages)
    assert any("Running pose backend warmup" in message for message in messages)
    assert any("Pose benchmark warmup 1/1" in message for message in messages)
    assert any("Pose benchmark timed iteration 1/2" in message for message in messages)
    assert any("Pose benchmark complete" in message for message in messages)


def test_bench_pose_reuses_native_backend_across_batch_sweeps(
    monkeypatch,
    tmp_path: Path,
) -> None:
    call_counts = {"create": 0, "warmup": 0, "predict": 0, "close": 0}

    class FakeBackend:
        def __init__(self, model_path, device, batch_size, **_kwargs):
            call_counts["create"] += 1
            self.model_path = model_path
            self.device = device
            self.batch_size = batch_size

        def warmup(self):
            call_counts["warmup"] += 1

        def predict_batch(self, crops):
            call_counts["predict"] += 1
            return [object() for _ in crops]

        def close(self):
            call_counts["close"] += 1

    monkeypatch.setitem(
        sys.modules,
        "hydra_suite.core.identity.pose.backends.yolo",
        types.SimpleNamespace(
            YoloNativeBackend=FakeBackend,
            auto_export_yolo_model=lambda config, flavor, runtime_device=None: str(
                tmp_path / "pose.onnx"
            ),
        ),
    )

    backend_cache = benchmarking._PoseBenchmarkBackendCache()
    try:
        first = benchmarking.bench_pose(
            model_path=str(tmp_path / "pose.pt"),
            tier="cpu",
            warmup=0,
            iterations=1,
            batch_size=4,
            crop_size=96,
            backend_family="yolo",
            pose_backend_cache=backend_cache,
        )
        second = benchmarking.bench_pose(
            model_path=str(tmp_path / "pose.pt"),
            tier="cpu",
            warmup=0,
            iterations=1,
            batch_size=8,
            crop_size=96,
            backend_family="yolo",
            pose_backend_cache=backend_cache,
        )
    finally:
        backend_cache.close()

    assert first.success is True
    assert second.success is True
    assert call_counts == {"create": 1, "warmup": 1, "predict": 2, "close": 1}


def test_bench_pose_yolo_gpu_fast_uses_native_device_not_export(monkeypatch, tmp_path):
    import hydra_suite.trackerkit.benchmarking as benchmarking

    export_calls = []

    def fake_auto_export(*args, **kwargs):
        export_calls.append((args, kwargs))
        raise AssertionError(
            "YOLO pose must not export to ONNX/TensorRT under gpu_fast"
        )

    monkeypatch.setattr(
        "hydra_suite.core.identity.pose.backends.yolo.auto_export_yolo_model",
        fake_auto_export,
    )

    created = {}

    class _FakeBackend:
        def __init__(self, model_path, device, batch_size):
            created["device"] = device

        def predict_batch(self, crops):
            return [None] * len(crops)

        def close(self):
            pass

    monkeypatch.setattr(
        "hydra_suite.core.identity.pose.backends.yolo.YoloNativeBackend", _FakeBackend
    )

    result = benchmarking.bench_pose(
        str(tmp_path / "pose.pt"),
        "gpu_fast",
        warmup=1,
        iterations=1,
        batch_size=1,
        crop_size=64,
        backend_family="yolo",
    )

    assert result.success, result.error
    assert not export_calls
    # Whatever accelerator this machine actually has (mps here), gpu_fast must
    # resolve to the *native* device for that accelerator, never "cpu".
    platform = benchmarking.detect_platform()
    if platform.has_cuda:
        assert created["device"] == "cuda:0"
    elif platform.has_mps:
        assert created["device"] == "mps"
    else:
        assert created["device"] == "cpu"


def test_bench_pose_yolo_gpu_fast_uses_cuda_device_on_cuda_platform(
    monkeypatch, tmp_path
):
    """Regression test for the gpu_fast->cpu silent-fallback bug.

    Forces a simulated CUDA platform (no real GPU hardware required) and
    asserts device == "cuda:0". Against the pre-fix code (which called
    resolve_compute_runtime(tier, platform, stage="yolo_pose") with no
    artifact_available override, so gpu_fast + has_cuda=True resolved to the
    literal compute-runtime string "tensorrt", which the device-mapping
    ternary has no branch for and silently fell through to "cpu"), this
    assertion fails.
    """
    import hydra_suite.trackerkit.benchmarking as benchmarking

    monkeypatch.setattr(
        benchmarking,
        "detect_platform",
        lambda: benchmarking_resolver.PlatformInfo(has_cuda=True, has_mps=False),
    )

    export_calls = []

    def fake_auto_export(*args, **kwargs):
        export_calls.append((args, kwargs))
        raise AssertionError(
            "YOLO pose must not export to ONNX/TensorRT under gpu_fast"
        )

    monkeypatch.setattr(
        "hydra_suite.core.identity.pose.backends.yolo.auto_export_yolo_model",
        fake_auto_export,
    )

    created = {}

    class _FakeBackend:
        def __init__(self, model_path, device, batch_size):
            created["device"] = device

        def predict_batch(self, crops):
            return [None] * len(crops)

        def close(self):
            pass

    monkeypatch.setattr(
        "hydra_suite.core.identity.pose.backends.yolo.YoloNativeBackend", _FakeBackend
    )

    result = benchmarking.bench_pose(
        str(tmp_path / "pose.pt"),
        "gpu_fast",
        warmup=1,
        iterations=1,
        batch_size=1,
        crop_size=64,
        backend_family="yolo",
    )

    assert result.success, result.error
    assert not export_calls
    assert created["device"] == "cuda:0"


def test_run_target_benchmark_pose_preserves_rectangular_crop_geometry_for_sleap(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured = {
        "export_input_hw": "unset",
        "crop_shapes": [],
        "runtime_flavor": None,
        "device": None,
    }

    class FakeBackend:
        def warmup(self):
            return None

        def predict_batch(self, crops):
            captured["crop_shapes"].append(tuple(crops[0].shape))
            return [object() for _ in crops]

        def close(self):
            return None

    def fake_create_pose_backend_from_config(config):
        captured["export_input_hw"] = config.sleap_export_input_hw
        captured["runtime_flavor"] = config.runtime_flavor
        captured["device"] = config.sleap_device
        return FakeBackend()

    monkeypatch.setitem(
        sys.modules,
        "hydra_suite.core.identity.pose.api",
        types.SimpleNamespace(
            create_pose_backend_from_config=fake_create_pose_backend_from_config
        ),
    )

    target = benchmarking.BenchmarkTargetSpec(
        key="pose_sleap",
        label="Pose Extraction",
        pipeline="pose",
        model_path=str(tmp_path / "pose_model"),
        runtimes=["gpu_fast"],
        batch_sizes=[3],
        backend_family="sleap",
        benchmark_context={
            "keypoint_names": ["nose", "tail"],
            "sleap_env": "sleap",
        },
    )
    geometry = benchmarking.BenchmarkGeometry(
        frame_width=1920,
        frame_height=1080,
        resize_factor=1.0,
        reference_body_size=32.0,
        reference_aspect_ratio=2.0,
        padding_fraction=0.1,
        effective_frame_width=1920,
        effective_frame_height=1080,
        canonical_crop_width=104,
        canonical_crop_height=72,
    )

    result = benchmarking.run_target_benchmark(
        target,
        geometry,
        runtime="gpu_fast",
        batch_size=3,
        warmup=0,
        iterations=1,
    )

    assert result.success is True
    assert result.input_shape == (72, 104)
    assert captured["export_input_hw"] is None
    # On this platform (Apple Silicon, no CUDA), gpu_fast for SLEAP has no
    # exported CoreML path (see stages/pose.py) and resolves to the native
    # mps compute_runtime, which derive_pose_runtime_settings maps to the
    # "mps" flavor.
    assert captured["runtime_flavor"] == "mps"
    assert captured["device"] == "mps"
    assert captured["crop_shapes"] == [(72, 104, 3)]


def test_collect_active_targets_skips_headtail_when_pipeline_disabled(
    monkeypatch,
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "orientation.pth"
    model_path.write_text("stub", encoding="utf-8")

    monkeypatch.setattr(
        benchmarking,
        "resolve_model_path",
        lambda model: (
            str(model_path)
            if str(model) == "classification/orientation/model.pth"
            else str(model)
        ),
    )

    main_window = SimpleNamespace(
        _detection_panel=SimpleNamespace(
            spin_detection_batch_size=SimpleNamespace(value=lambda: 4),
            combo_yolo_obb_mode=SimpleNamespace(currentIndex=lambda: 0),
        ),
        _identity_panel=SimpleNamespace(
            spin_pose_batch=SimpleNamespace(value=lambda: 4),
            _get_selected_yolo_headtail_model_path=lambda: "classification/orientation/model.pth",
        ),
        _setup_panel=SimpleNamespace(spin_max_targets=SimpleNamespace(value=lambda: 8)),
        _compute_runtime_options_for_current_ui=lambda: [("CPU", "cpu")],
        _selected_compute_runtime=lambda: "cpu",
        _is_yolo_detection_mode=lambda: False,
        _get_selected_yolo_model_path=lambda: "",
        _get_selected_yolo_detect_model_path=lambda: "",
        _get_selected_yolo_crop_obb_model_path=lambda: "",
        _identity_config=lambda: {"cnn_classifiers": []},
        _is_pose_inference_enabled=lambda: False,
        _is_headtail_compute_enabled=lambda: False,
        _selected_cnn_runtime=lambda: "cpu",
    )

    targets, notices = benchmarking.collect_active_targets(main_window)

    assert not any(target.key == "headtail" for target in targets)
    assert (
        "Head-tail benchmark skipped because the selected orientation model could not be resolved."
        not in notices
    )


def test_bench_obb_uses_load_obb_executor_not_legacy_detector(monkeypatch, tmp_path):
    import hydra_suite.trackerkit.benchmarking as benchmarking_module

    calls = []

    class _FakeExecutor:
        def predict(self, frames, **kwargs):
            class _R:
                obb = None
                boxes = None

            return [_R() for _ in frames]

    def fake_load_obb_executor(model_path, compute_runtime, **kwargs):
        calls.append((model_path, compute_runtime, kwargs))
        return _FakeExecutor()

    monkeypatch.setattr(
        "hydra_suite.core.inference.runtime_artifacts.load_obb_executor",
        fake_load_obb_executor,
    )

    # If bench_obb still imports YOLOOBBDetector at all, fail loudly.
    def _fail_if_called(*_a, **_k):
        raise AssertionError("bench_obb must not construct YOLOOBBDetector")

    monkeypatch.setattr(
        "hydra_suite.core.detectors.YOLOOBBDetector", _fail_if_called, raising=False
    )

    model_path = str(tmp_path / "model.pt")
    result = benchmarking_module.bench_obb(
        model_path,
        "gpu_fast",
        warmup=1,
        iterations=1,
        batch_size=1,
        frame_size=(128, 128),
    )

    assert result.success, result.error
    assert calls, "load_obb_executor was never called"
    assert calls[0][0] == model_path


def test_bench_sequential_uses_load_obb_executor(monkeypatch, tmp_path):
    import hydra_suite.trackerkit.benchmarking as benchmarking_module

    calls = []

    class _FakeExecutor:
        def predict(self, frames, **kwargs):
            class _R:
                obb = None
                boxes = None

            return [_R() for _ in frames]

    def fake_load_obb_executor(model_path, compute_runtime, **kwargs):
        calls.append((model_path, compute_runtime, kwargs.get("task")))
        return _FakeExecutor()

    monkeypatch.setattr(
        "hydra_suite.core.inference.runtime_artifacts.load_obb_executor",
        fake_load_obb_executor,
    )

    detect_path = str(tmp_path / "detect.pt")
    crop_path = str(tmp_path / "crop.pt")
    result = benchmarking_module.bench_sequential(
        detect_path,
        crop_path,
        "gpu",
        warmup=1,
        iterations=1,
        batch_size=1,
        individual_batch_size=1,
        frame_size=(128, 128),
        crop_size=128,
    )

    assert result.success, result.error
    # Two executors: stage-1 detect (task="detect") and stage-2 crop OBB (task="obb").
    assert {c[2] for c in calls} == {"detect", "obb"}


def test_collect_active_targets_includes_sequential_crop_settings(
    tmp_path: Path,
) -> None:
    detect_model = tmp_path / "detect.pt"
    crop_model = tmp_path / "crop.pt"
    detect_model.write_text("stub", encoding="utf-8")
    crop_model.write_text("stub", encoding="utf-8")

    main_window = SimpleNamespace(
        _detection_panel=SimpleNamespace(
            spin_detection_batch_size=SimpleNamespace(value=lambda: 16),
            spin_yolo_seq_individual_batch_size=SimpleNamespace(value=lambda: 12),
            combo_yolo_obb_mode=SimpleNamespace(currentIndex=lambda: 1),
            spin_yolo_seq_crop_pad=SimpleNamespace(value=lambda: 0.22),
            spin_yolo_seq_min_crop_px=SimpleNamespace(value=lambda: 88),
            chk_yolo_seq_square_crop=SimpleNamespace(isChecked=lambda: False),
            spin_yolo_seq_stage2_imgsz=SimpleNamespace(value=lambda: 192),
            chk_yolo_seq_stage2_pow2_pad=SimpleNamespace(isChecked=lambda: True),
            spin_yolo_seq_detect_conf=SimpleNamespace(value=lambda: 0.14),
        ),
        _identity_panel=SimpleNamespace(
            spin_pose_batch=SimpleNamespace(value=lambda: 4)
        ),
        _setup_panel=SimpleNamespace(spin_max_targets=SimpleNamespace(value=lambda: 8)),
        _compute_runtime_options_for_current_ui=lambda: [
            ("CPU", "cpu"),
            ("MPS", "mps"),
        ],
        _selected_compute_runtime=lambda: "mps",
        _is_yolo_detection_mode=lambda: True,
        _get_selected_yolo_model_path=lambda: "",
        _get_selected_yolo_detect_model_path=lambda: str(detect_model),
        _get_selected_yolo_crop_obb_model_path=lambda: str(crop_model),
        _identity_config=lambda: {"cnn_classifiers": []},
        _is_pose_inference_enabled=lambda: False,
        _selected_cnn_runtime=lambda: "cpu",
    )

    targets, notices = benchmarking.collect_active_targets(main_window)

    assert notices == []
    assert len(targets) == 1
    target = targets[0]
    assert target.key == "detection_sequential"
    assert target.individual_batch_sizes is not None
    assert 12 in target.individual_batch_sizes
    assert target.current_individual_batch_size == 12
    assert target.benchmark_context["max_targets"] == 8
    assert target.benchmark_context["yolo_seq_individual_batch_size"] == 12
    assert target.benchmark_context["yolo_seq_crop_pad_ratio"] == 0.22
    assert target.benchmark_context["yolo_seq_min_crop_size_px"] == 88
    assert target.benchmark_context["yolo_seq_enforce_square_crop"] is False
    assert target.benchmark_context["yolo_seq_stage2_imgsz"] == 192
    assert target.benchmark_context["yolo_seq_stage2_pow2_pad"] is True
    assert target.benchmark_context["yolo_seq_detect_conf_threshold"] == 0.14


def test_run_target_benchmark_uses_setup_target_count_for_detection_modes(
    monkeypatch,
) -> None:
    geometry = benchmarking.build_benchmark_geometry_from_dimensions(
        frame_width=1920,
        frame_height=1080,
        resize_factor=1.0,
        reference_body_size=40.0,
        reference_aspect_ratio=2.0,
        padding_fraction=0.25,
    )
    calls = {}

    def fake_bench_obb(
        model_path,
        runtime,
        warmup,
        iterations,
        batch_size,
        frame_size,
        *,
        max_targets,
    ):
        calls["direct"] = {
            "model_path": model_path,
            "runtime": runtime,
            "batch_size": batch_size,
            "frame_size": frame_size,
            "max_targets": max_targets,
        }
        return benchmarking.BenchmarkResult(
            model_type="obb",
            model_path=model_path,
            runtime=runtime,
            runtime_label=runtime,
            batch_size=batch_size,
            input_shape=frame_size,
            warmup_iters=warmup,
            bench_iters=iterations,
        )

    def fake_bench_sequential(
        detect_model_path,
        crop_obb_model_path,
        runtime,
        warmup,
        iterations,
        batch_size,
        individual_batch_size,
        frame_size,
        crop_size,
        **kwargs,
    ):
        calls["sequential"] = {
            "detect_model_path": detect_model_path,
            "crop_obb_model_path": crop_obb_model_path,
            "runtime": runtime,
            "batch_size": batch_size,
            "individual_batch_size": individual_batch_size,
            "frame_size": frame_size,
            "crop_size": crop_size,
            **kwargs,
        }
        return benchmarking.BenchmarkResult(
            model_type="sequential",
            model_path=crop_obb_model_path,
            runtime=runtime,
            runtime_label=runtime,
            batch_size=batch_size,
            input_shape=frame_size,
            warmup_iters=warmup,
            bench_iters=iterations,
        )

    monkeypatch.setattr(benchmarking, "bench_obb", fake_bench_obb)
    monkeypatch.setattr(benchmarking, "bench_sequential", fake_bench_sequential)

    direct_target = benchmarking.BenchmarkTargetSpec(
        key="detection_direct",
        label="Detection (Direct OBB)",
        pipeline="obb",
        model_path="/tmp/direct.pt",
        runtimes=["cpu"],
        batch_sizes=[1],
        benchmark_context={"max_targets": 9},
    )
    sequential_target = benchmarking.BenchmarkTargetSpec(
        key="detection_sequential",
        label="Detection (Sequential)",
        pipeline="sequential",
        model_path="/tmp/crop.pt",
        extra_model_paths=["/tmp/detect.pt"],
        runtimes=["cpu"],
        batch_sizes=[1],
        individual_batch_sizes=[9],
        benchmark_context={
            "max_targets": 9,
            "yolo_seq_stage2_imgsz": 192,
            "yolo_seq_individual_batch_size": 9,
            "yolo_seq_crop_pad_ratio": 0.2,
            "yolo_seq_min_crop_size_px": 72,
            "yolo_seq_enforce_square_crop": False,
            "yolo_seq_stage2_pow2_pad": True,
            "yolo_seq_detect_conf_threshold": 0.12,
        },
    )

    benchmarking.run_target_benchmark(
        direct_target,
        geometry,
        "cpu",
        4,
        warmup=1,
        iterations=2,
    )
    benchmarking.run_target_benchmark(
        sequential_target,
        geometry,
        "cpu",
        4,
        9,
        warmup=1,
        iterations=2,
    )

    assert calls["direct"]["max_targets"] == 9
    assert calls["sequential"]["individual_batch_size"] == 9
    assert calls["sequential"]["max_targets"] == 9
    assert calls["sequential"]["crop_size"] == 192


def test_collect_active_targets_includes_tier_headtail_runtimes(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """headtail target now speaks the 3-value tier vocabulary, not canonical runtimes."""
    model_path = tmp_path / "orientation.onnx"
    model_path.write_text("stub", encoding="utf-8")

    monkeypatch.setattr(
        benchmarking,
        "resolve_model_path",
        lambda model: (
            str(model_path)
            if str(model) == "classification/orientation/model.onnx"
            else str(model)
        ),
    )

    main_window = SimpleNamespace(
        _detection_panel=SimpleNamespace(
            spin_detection_batch_size=SimpleNamespace(value=lambda: 4),
            combo_yolo_obb_mode=SimpleNamespace(currentIndex=lambda: 0),
        ),
        _identity_panel=SimpleNamespace(
            spin_pose_batch=SimpleNamespace(value=lambda: 4),
            spin_headtail_batch=SimpleNamespace(value=lambda: 32),
            _get_selected_yolo_headtail_model_path=lambda: "classification/orientation/model.onnx",
        ),
        _setup_panel=SimpleNamespace(
            spin_max_targets=SimpleNamespace(value=lambda: 8),
            combo_runtime_tier=SimpleNamespace(currentData=lambda: "gpu_fast"),
        ),
        _is_yolo_detection_mode=lambda: True,
        _get_selected_yolo_model_path=lambda: "",
        _get_selected_yolo_detect_model_path=lambda: "",
        _get_selected_yolo_crop_obb_model_path=lambda: "",
        _identity_config=lambda: {"cnn_classifiers": []},
        _is_pose_inference_enabled=lambda: False,
        _is_headtail_compute_enabled=lambda: True,
    )

    targets, notices = benchmarking.collect_active_targets(main_window)

    assert notices == []
    assert len(targets) == 1
    headtail_targets = [target for target in targets if target.key == "headtail"]
    assert len(headtail_targets) == 1
    allowed_tiers = {"cpu", "gpu", "gpu_fast"}
    assert set(headtail_targets[0].runtimes) <= allowed_tiers
    assert headtail_targets[0].current_runtime == "gpu_fast"
    assert headtail_targets[0].current_batch_size == 32
    assert headtail_targets[0].supports_batch_apply is True


def test_collect_active_targets_no_longer_calls_deleted_compute_runtime_options_method() -> (
    None
):
    """Regression: collect_active_targets must NOT call _compute_runtime_options_for_current_ui.

    Before the tier-model fix this raised AttributeError because the method was
    deleted in Task 6.  The mock below has NO such attribute — if the code tries
    to call it, AttributeError is raised and the test fails.
    """
    main_window = SimpleNamespace(
        _detection_panel=SimpleNamespace(
            spin_detection_batch_size=SimpleNamespace(value=lambda: 4),
            combo_yolo_obb_mode=SimpleNamespace(currentIndex=lambda: 0),
        ),
        _identity_panel=SimpleNamespace(
            spin_pose_batch=SimpleNamespace(value=lambda: 4),
        ),
        _setup_panel=SimpleNamespace(spin_max_targets=SimpleNamespace(value=lambda: 8)),
        # _compute_runtime_options_for_current_ui intentionally absent
        _selected_compute_runtime=lambda: "cpu",
        _is_yolo_detection_mode=lambda: False,
        _get_selected_yolo_model_path=lambda: "",
        _get_selected_yolo_detect_model_path=lambda: "",
        _get_selected_yolo_crop_obb_model_path=lambda: "",
        _identity_config=lambda: {"cnn_classifiers": []},
        _is_pose_inference_enabled=lambda: False,
        _selected_cnn_runtime=lambda: "cpu",
    )

    targets, notices = benchmarking.collect_active_targets(main_window)

    assert isinstance(targets, list)
    assert isinstance(notices, list)


def test_collect_active_targets_uses_tier_vocabulary(tmp_path: Path) -> None:
    """Every target type (detection, head-tail, pose, CNN) must speak the
    3-value tier vocabulary ("cpu"/"gpu"/"gpu_fast"), not canonical runtime
    strings such as onnx_cpu/onnx_cuda/onnx_coreml/tensorrt/mps/cuda.
    """
    direct_model = tmp_path / "detector.pt"
    direct_model.write_text("stub", encoding="utf-8")
    headtail_model = tmp_path / "orientation.pth"
    headtail_model.write_text("stub", encoding="utf-8")
    sleap_model_dir = tmp_path / "sleap_model"
    sleap_model_dir.mkdir()

    main_window = SimpleNamespace(
        _detection_panel=SimpleNamespace(
            spin_detection_batch_size=SimpleNamespace(value=lambda: 4),
            combo_yolo_obb_mode=SimpleNamespace(currentIndex=lambda: 0),
        ),
        _identity_panel=SimpleNamespace(
            spin_pose_batch=SimpleNamespace(value=lambda: 6),
            spin_headtail_batch=SimpleNamespace(value=lambda: 8),
            _get_selected_yolo_headtail_model_path=lambda: str(headtail_model),
        ),
        _setup_panel=SimpleNamespace(
            spin_max_targets=SimpleNamespace(value=lambda: 8),
            combo_runtime_tier=SimpleNamespace(currentData=lambda: "gpu"),
        ),
        _is_yolo_detection_mode=lambda: True,
        _get_selected_yolo_model_path=lambda: str(direct_model),
        _get_selected_yolo_detect_model_path=lambda: "",
        _get_selected_yolo_crop_obb_model_path=lambda: "",
        _identity_config=lambda: {
            "cnn_classifiers": [
                {"label": "species", "model_path": str(direct_model), "batch_size": 2}
            ]
        },
        _is_pose_inference_enabled=lambda: True,
        _is_headtail_compute_enabled=lambda: True,
        _current_pose_backend_key=lambda: "sleap",
        _get_resolved_pose_model_dir=lambda backend: str(sleap_model_dir),
        _load_pose_skeleton_keypoint_names=lambda: ["nose", "tail"],
        _selected_pose_sleap_env=lambda: "sleap-mps",
    )

    targets, notices = benchmarking.collect_active_targets(main_window)

    assert {target.key for target in targets} == {
        "detection_direct",
        "headtail",
        "pose_sleap",
        "cnn_0",
    }

    allowed_tiers = {"cpu", "gpu", "gpu_fast"}
    for target in targets:
        assert (
            set(target.runtimes) <= allowed_tiers
        ), f"target {target.key!r} exposed non-tier runtimes: {target.runtimes}"
        assert target.current_runtime in allowed_tiers
        assert target.current_runtime == "gpu"


def test_synchronize_runtime_calls_cuda_when_available(monkeypatch) -> None:
    import torch

    cuda_sync_calls: list[None] = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: cuda_sync_calls.append(None))
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

    # Passing a tier string ("gpu") must not affect which branch runs: only
    # real hardware availability should matter.
    benchmarking._synchronize_runtime("gpu")

    assert cuda_sync_calls == [None]


def test_synchronize_runtime_calls_mps_when_available_and_cuda_absent(
    monkeypatch,
) -> None:
    import torch

    mps_sync_calls: list[None] = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(torch.mps, "synchronize", lambda: mps_sync_calls.append(None))

    benchmarking._synchronize_runtime("gpu_fast")

    assert mps_sync_calls == [None]


def test_synchronize_runtime_no_accelerator_does_nothing(monkeypatch) -> None:
    import torch

    cuda_sync_calls: list[None] = []
    mps_sync_calls: list[None] = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: cuda_sync_calls.append(None))
    monkeypatch.setattr(torch.mps, "synchronize", lambda: mps_sync_calls.append(None))

    # Should not raise even though no accelerator is available.
    benchmarking._synchronize_runtime("cpu")

    assert cuda_sync_calls == []
    assert mps_sync_calls == []


def test_sample_accelerator_memory_mb_reads_cuda_when_available(monkeypatch) -> None:
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda: 2 * 1024 * 1024)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda: 4 * 1024 * 1024)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

    result = benchmarking._sample_accelerator_memory_mb("gpu")

    assert result == 4.0


def test_sample_accelerator_memory_mb_reads_mps_when_available_and_cuda_absent(
    monkeypatch,
) -> None:
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(torch.mps, "current_allocated_memory", lambda: 1024 * 1024)
    monkeypatch.setattr(torch.mps, "driver_allocated_memory", lambda: 3 * 1024 * 1024)

    result = benchmarking._sample_accelerator_memory_mb("gpu_fast")

    assert result == 3.0


def test_sample_accelerator_memory_mb_returns_none_without_accelerator(
    monkeypatch,
) -> None:
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

    assert benchmarking._sample_accelerator_memory_mb("cpu") is None
