from hydra_suite.trackerkit.obb_runtime_probe import (
    build_detector_params,
    build_parser,
    compare_metric_summaries,
    normalize_raw_detector_output,
    summarize_series,
)


def test_build_detector_params_for_tensorrt_defaults_to_runtime_batch_size() -> None:
    params = build_detector_params(
        model_path="/tmp/model.pt",
        runtime="tensorrt",
        batch_size=2,
        max_targets=11,
    )

    assert params["YOLO_DEVICE"] == "cuda:0"
    assert params["ENABLE_TENSORRT"] is True
    assert params["ENABLE_ONNX_RUNTIME"] is False
    assert params["YOLO_MANUAL_BATCH_SIZE"] == 2
    assert params["TENSORRT_MAX_BATCH_SIZE"] == 2
    assert params["TENSORRT_BUILD_BATCH_SIZE"] == 2
    assert params["TRACKING_REALTIME_MODE"] is True
    assert params["TRACKING_WORKFLOW_MODE"] == "realtime"
    assert params["MAX_TARGETS"] == 11
    assert params["YOLO_OBB_EXECUTION_MODE"] == "auto"


def test_build_detector_params_for_onnx_cpu_preserves_optional_overrides() -> None:
    params = build_detector_params(
        model_path="/tmp/model.pt",
        runtime="onnx_cpu",
        batch_size=1,
        max_targets=5,
        tensorrt_max_batch_size=7,
        tensorrt_build_batch_size=9,
        imgsz=960,
        headtail_model_path="/tmp/headtail.pth",
        headtail_runtime="cuda",
        headtail_batch_size=32,
        tracking_realtime_mode=False,
    )

    assert params["YOLO_DEVICE"] == "cpu"
    assert params["ENABLE_TENSORRT"] is False
    assert params["ENABLE_ONNX_RUNTIME"] is True
    assert params["TENSORRT_MAX_BATCH_SIZE"] == 7
    assert params["TENSORRT_BUILD_BATCH_SIZE"] == 9
    assert params["YOLO_IMGSZ"] == 960
    assert params["YOLO_HEADTAIL_MODEL_PATH"] == "/tmp/headtail.pth"
    assert params["HEADTAIL_COMPUTE_RUNTIME"] == "cuda"
    assert params["HEADTAIL_BATCH_SIZE"] == 32
    assert params["TRACKING_REALTIME_MODE"] is False
    assert params["TRACKING_WORKFLOW_MODE"] == "non_realtime"


def test_summarize_series_computes_expected_statistics() -> None:
    summary = summarize_series([1.0, 2.0, 3.0, 4.0])

    assert summary["count"] == 4
    assert summary["mean_ms"] == 2.5
    assert summary["median_ms"] == 2.5
    assert summary["min_ms"] == 1.0
    assert summary["max_ms"] == 4.0
    assert summary["p95_ms"] == 4.0


def test_parser_defaults_to_realtime_mode() -> None:
    args = build_parser().parse_args(
        ["--video", "/tmp/video.mp4", "--model", "/tmp/model.pt"]
    )

    assert args.tracking_realtime_mode is True
    assert args.headtail_runtime is None
    assert args.read_mode == "direct"
    assert args.prefetch_buffer_size == 2
    assert args.execution_mode == "auto"
    assert args.compare_execution_modes is False


def test_compare_metric_summaries_reports_delta_and_speedup() -> None:
    comparison = compare_metric_summaries(
        {"detector_call": {"mean_ms": 10.0, "p95_ms": 20.0}},
        {"detector_call": {"mean_ms": 8.0, "p95_ms": 16.0}},
    )

    detector_call = comparison["detector_call"]
    assert detector_call["mean_delta_ms"] == -2.0
    assert detector_call["mean_delta_pct"] == -20.0
    assert detector_call["p95_delta_ms"] == -4.0
    assert detector_call["p95_delta_pct"] == -20.0
    assert detector_call["speedup_vs_baseline"] == 1.25


def test_normalize_raw_detector_output_accepts_single_and_batched_shapes() -> None:
    single = (
        [1],
        [2],
        [3],
        object(),
        [4],
        [5],
        [6],
        [7],
        [8],
        [9],
    )
    batched = ([1], [2], [3], [4], [5], [6], [7], [8], [9])

    assert normalize_raw_detector_output(single) == batched
    assert normalize_raw_detector_output(batched) == batched
