from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from hydra_suite.core.inference.config import (
    OBBConfig,
    OBBDirectConfig,
    OBBSequentialConfig,
)
from hydra_suite.core.inference.result import OBBResult
from hydra_suite.core.inference.runtime import RuntimeContext
from hydra_suite.core.inference.stages.obb import (
    OBBModels,
    _empty_obb_result,
    _extract_raw_tensors,
    _RawOBBTensors,
    extract_obb_result,
    merge_obb_results,
    run_obb,
)


def _cpu_rt() -> RuntimeContext:
    return RuntimeContext(
        cuda_mode=False,
        device="cpu",
        use_nvdec=False,
        tensor_on_cuda=False,
    )


def _cuda_rt() -> RuntimeContext:
    # tensor_on_cuda=True for native PyTorch CUDA runtime
    return RuntimeContext(
        cuda_mode=True,
        device="cuda:0",
        use_nvdec=False,
        tensor_on_cuda=True,
        requested_gpu=True,
    )


def _onnx_cuda_rt() -> RuntimeContext:
    # ONNX CUDA: cuda_mode=True but tensor_on_cuda=False (CPU numpy outputs)
    return RuntimeContext(
        cuda_mode=True,
        device="cuda:0",
        use_nvdec=False,
        tensor_on_cuda=False,
        requested_gpu=True,
    )


def _mock_ul_result_tensors(n: int = 2) -> MagicMock:
    """Fake ultralytics OBB result with PyTorch tensors (CPU for testing)."""
    xywhr = torch.tensor([[100.0, 100.0, 20.0, 10.0, 0.5]] * n)
    corners = torch.zeros(n, 4, 2)
    conf = torch.full((n,), 0.8)
    r = MagicMock()
    r.obb.xywhr = xywhr
    r.obb.xyxyxyxy = corners
    r.obb.conf = conf
    r.obb.__len__ = lambda self: n
    return r


def _mock_ul_result_numpy_compat(n: int = 2) -> MagicMock:
    """Fake ultralytics OBB result via .cpu().numpy() chain (CPU-path test)."""

    def _t(arr):
        m = MagicMock()
        m.cpu.return_value.numpy.return_value = arr
        return m

    xywhr = np.array([[100.0, 100.0, 20.0, 10.0, 0.5]] * n, dtype=np.float32)
    corners = np.zeros((n, 4, 2), dtype=np.float32)
    conf = np.full(n, 0.8, dtype=np.float32)
    r = MagicMock()
    r.obb.xywhr = _t(xywhr)
    r.obb.xyxyxyxy = _t(corners)
    r.obb.conf = _t(conf)
    r.obb.__len__ = lambda self: n
    return r


def test_empty_obb_result_shape():
    r = _empty_obb_result(0)
    assert r.num_detections == 0
    assert r.centroids.shape == (0, 2)
    assert r.corners.shape == (0, 4, 2)
    # NEW: empty result must still carry a (zero-length) detection_ids array
    assert r.detection_ids.shape == (0,)


def test_extract_obb_result_n_detections():
    result = extract_obb_result(_mock_ul_result_numpy_compat(n=3), frame_idx=0)
    assert result.num_detections == 3
    assert result.centroids.shape == (3, 2)
    assert result.angles.shape == (3,)
    assert result.sizes.shape == (3,)
    assert result.corners.shape == (3, 4, 2)


def test_extract_obb_result_offset_shifts_centroids():
    result = extract_obb_result(
        _mock_ul_result_numpy_compat(n=1), frame_idx=0, offset=(50.0, 30.0)
    )
    assert result.centroids[0, 0] == pytest.approx(150.0)
    assert result.centroids[0, 1] == pytest.approx(130.0)


def test_extract_obb_result_sizes_computed():
    result = extract_obb_result(_mock_ul_result_numpy_compat(n=1), frame_idx=0)
    assert result.sizes[0] == pytest.approx(20.0 * 10.0)


def test_extract_obb_result_carries_detection_ids():
    """Per Correction 14: every constructed OBBResult must include detection_ids."""
    result = extract_obb_result(_mock_ul_result_numpy_compat(n=3), frame_idx=7)
    assert result.detection_ids.shape == (3,)
    assert result.detection_ids.dtype == np.int64
    assert result.detection_ids[0] == 7 * 10000  # DETECTION_ID_STRIDE
    assert result.detection_ids[2] == 7 * 10000 + 2


def test_extract_raw_tensors_returns_named_tuple():
    r = _mock_ul_result_tensors(n=2)
    raw = _extract_raw_tensors(r, frame_idx=5, device="cpu")
    assert isinstance(raw, _RawOBBTensors)
    assert raw.frame_idx == 5
    assert raw.xywhr.shape == (2, 5)
    assert raw.corners.shape == (2, 4, 2)
    assert raw.conf.shape == (2,)


def test_extract_raw_tensors_no_cpu_call():
    """_extract_raw_tensors must not call .cpu() on any tensor field."""
    xywhr_mock = MagicMock(spec=torch.Tensor)
    corners_mock = MagicMock(spec=torch.Tensor)
    conf_mock = MagicMock(spec=torch.Tensor)
    r = MagicMock()
    r.obb.xywhr = xywhr_mock
    r.obb.xyxyxyxy = corners_mock
    r.obb.conf = conf_mock
    r.obb.__len__ = lambda self: 2
    _extract_raw_tensors(r, frame_idx=0, device="cpu")
    xywhr_mock.cpu.assert_not_called()
    corners_mock.cpu.assert_not_called()
    conf_mock.cpu.assert_not_called()


def test_merge_obb_results_concatenates():
    r1 = OBBResult(
        frame_idx=0,
        centroids=np.ones((2, 2), dtype=np.float32),
        angles=np.ones(2, dtype=np.float32),
        sizes=np.ones(2, dtype=np.float32),
        shapes=np.ones((2, 2), dtype=np.float32),
        confidences=np.ones(2, dtype=np.float32),
        corners=np.zeros((2, 4, 2), dtype=np.float32),
        detection_ids=OBBResult.make_detection_ids(0, 2),
    )
    r2 = OBBResult(
        frame_idx=0,
        centroids=np.ones((3, 2), dtype=np.float32),
        angles=np.ones(3, dtype=np.float32),
        sizes=np.ones(3, dtype=np.float32),
        shapes=np.ones((3, 2), dtype=np.float32),
        confidences=np.ones(3, dtype=np.float32),
        corners=np.zeros((3, 4, 2), dtype=np.float32),
        # offset by 2 to mimic post-merge IDs from a sequential second pass
        detection_ids=OBBResult.make_detection_ids(0, 3) + 2,
    )
    merged = merge_obb_results(0, [r1, r2])
    assert merged.num_detections == 5
    assert merged.detection_ids.shape == (5,)


def test_run_obb_cpu_returns_obb_result():
    config = OBBConfig(
        mode="direct",
        direct=OBBDirectConfig(model_path="/m.pt"),
    )
    mock_model = MagicMock()
    mock_model.predict.return_value = [_mock_ul_result_numpy_compat(n=2)]
    models = OBBModels(mode="direct", direct_model=mock_model)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    results = run_obb([frame], models, config, _cpu_rt())
    assert len(results) == 1
    assert isinstance(results[0], OBBResult)
    assert results[0].num_detections == 2


def test_run_obb_native_cuda_returns_raw_tensors():
    """Native PyTorch CUDA → _RawOBBTensors (no .cpu() pull)."""
    config = OBBConfig(
        mode="direct",
        direct=OBBDirectConfig(model_path="/m.pt"),
    )
    mock_model = MagicMock()
    mock_model.predict.return_value = [_mock_ul_result_tensors(n=2)]
    models = OBBModels(mode="direct", direct_model=mock_model)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    results = run_obb([frame], models, config, _cuda_rt())
    assert len(results) == 1
    assert isinstance(results[0], _RawOBBTensors)


def test_run_obb_onnx_cuda_returns_obb_result():
    """Per Correction 2: onnx_cuda must NOT route through _RawOBBTensors —
    onnx_cuda returns CPU numpy from predict(), so we extract OBBResult."""
    config = OBBConfig(
        mode="direct",
        direct=OBBDirectConfig(model_path="/m.onnx"),
    )
    mock_model = MagicMock()
    mock_model.predict.return_value = [_mock_ul_result_numpy_compat(n=2)]
    models = OBBModels(mode="direct", direct_model=mock_model)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    results = run_obb([frame], models, config, _onnx_cuda_rt())
    assert len(results) == 1
    # onnx_cuda is NOT tensor_on_cuda → extract to OBBResult, not _RawOBBTensors
    assert isinstance(results[0], OBBResult)


def test_load_yolo_calls_to_for_native_pt(monkeypatch):
    """Native cuda and mps DO call .to(); cpu does not."""
    calls = []

    class FakeYOLO:
        def to(self, device):
            calls.append(device)
            return self

    import hydra_suite.core.inference.runtime_artifacts as ra

    monkeypatch.setattr(ra, "_load_torch_model", lambda p: FakeYOLO())
    from hydra_suite.core.inference.stages.obb import _load_yolo

    _load_yolo("/m.pt", "cuda")
    assert calls == ["cuda:0"]
    calls.clear()
    _load_yolo("/m.pt", "mps")
    assert calls == ["mps"]
    calls.clear()
    _load_yolo("/m.pt", "cpu")
    assert calls == []


def test_load_yolo_forwards_batch_size_to_load_obb_executor(monkeypatch):
    """_load_yolo must forward its batch_size kwarg to load_obb_executor
    unchanged -- this is how the configured window size reaches the
    TensorRT dynamic-vs-static export decision (Task 1)."""
    import hydra_suite.core.inference.stages.obb as obb_mod

    captured = {}

    def fake_load_obb_executor(model_path, compute_runtime, **kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(obb_mod, "load_obb_executor", fake_load_obb_executor)

    obb_mod._load_yolo("/m.pt", "tensorrt", auto_export=False, batch_size=8)
    assert captured["batch_size"] == 8


def test_load_obb_models_direct_mode_uses_detection_batch_size(monkeypatch):
    """Direct-mode OBB must be loaded with the caller's batch_size (the
    pipeline's configured detection_batch_size), not a hardcoded 1."""
    import hydra_suite.core.inference.stages.obb as obb_mod
    from hydra_suite.core.inference.config import OBBConfig, OBBDirectConfig
    from hydra_suite.core.inference.runtime import RuntimeContext

    captured = {}

    def fake_load_yolo(model_path, compute_runtime, **kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(obb_mod, "_load_yolo", fake_load_yolo)

    config = OBBConfig(mode="direct", direct=OBBDirectConfig(model_path="/m.pt"))
    runtime = RuntimeContext(
        cuda_mode=True,
        device="cuda:0",
        use_nvdec=False,
        tensor_on_cuda=False,
        requested_gpu=True,
    )
    obb_mod.load_obb_models(config, runtime, batch_size=8)
    assert captured["batch_size"] == 8


def test_load_obb_models_sequential_mode_uses_stage2_batch_size_for_obb_model(
    monkeypatch,
):
    """Sequential mode: stage-1 (detect) uses the frame-window batch_size;
    stage-2 (obb/crop) uses OBBSequentialConfig.stage2_batch_size when set,
    falling back to the frame-window batch_size otherwise."""
    import hydra_suite.core.inference.stages.obb as obb_mod
    from hydra_suite.core.inference.config import OBBConfig, OBBSequentialConfig
    from hydra_suite.core.inference.runtime import RuntimeContext

    calls = []

    def fake_load_yolo(model_path, compute_runtime, **kwargs):
        calls.append((model_path, kwargs.get("batch_size")))
        return object()

    monkeypatch.setattr(obb_mod, "_load_yolo", fake_load_yolo)

    runtime = RuntimeContext(
        cuda_mode=True,
        device="cuda:0",
        use_nvdec=False,
        tensor_on_cuda=False,
        requested_gpu=True,
    )

    # stage2_batch_size explicitly set -> obb model uses it, not batch_size.
    config = OBBConfig(
        mode="sequential",
        sequential=OBBSequentialConfig(
            detect_model_path="/detect.pt",
            obb_model_path="/obb.pt",
            stage2_batch_size=16,
        ),
    )
    obb_mod.load_obb_models(config, runtime, batch_size=8)
    assert calls == [("/detect.pt", 8), ("/obb.pt", 16)]

    # stage2_batch_size unset (None) -> obb model falls back to batch_size.
    calls.clear()
    config2 = OBBConfig(
        mode="sequential",
        sequential=OBBSequentialConfig(
            detect_model_path="/detect.pt", obb_model_path="/obb.pt"
        ),
    )
    obb_mod.load_obb_models(config2, runtime, batch_size=8)
    assert calls == [("/detect.pt", 8), ("/obb.pt", 8)]


def test_load_obb_models_sequential_uses_stage2_imgsz_not_checkpoint(monkeypatch):
    """Regression (ported from the deleted legacy seq-OBB imgsz test): the
    stage-2 (crop/OBB) model must be built at the configured stage-2 crop imgsz,
    NOT the checkpoint's own embedded imgsz. Using the wrong imgsz here fed
    already-resized crops into an executor letterboxed for a different square
    size, inflating CUDA confidences and producing ~10x over-detection.

    Direct mode is unaffected: it passes no imgsz_override, so the model runs
    on full frames at the checkpoint's own imgsz (via _resolve_imgsz).
    """
    import hydra_suite.core.inference.stages.obb as obb_mod
    from hydra_suite.core.inference.config import (
        OBBConfig,
        OBBDirectConfig,
        OBBSequentialConfig,
    )
    from hydra_suite.core.inference.runtime import RuntimeContext

    calls = []

    def fake_load_yolo(model_path, compute_runtime, **kwargs):
        calls.append((model_path, kwargs.get("imgsz_override")))
        return object()

    monkeypatch.setattr(obb_mod, "_load_yolo", fake_load_yolo)

    runtime = RuntimeContext(
        cuda_mode=True,
        device="cuda:0",
        use_nvdec=False,
        tensor_on_cuda=False,
        requested_gpu=True,
    )

    # Sequential: stage-2 OBB model built at the configured stage-2 crop imgsz.
    config = OBBConfig(
        mode="sequential",
        sequential=OBBSequentialConfig(
            detect_model_path="/detect.pt",
            obb_model_path="/obb.pt",
            stage2_image_size=128,
        ),
    )
    obb_mod.load_obb_models(config, runtime, batch_size=1)
    # Stage-2 (/obb.pt) uses the stage-2 crop imgsz, not the checkpoint's own.
    assert ("/obb.pt", 128) in calls

    # Direct mode passes no imgsz_override (checkpoint imgsz is used at runtime).
    calls.clear()
    direct = OBBConfig(mode="direct", direct=OBBDirectConfig(model_path="/m.pt"))
    obb_mod.load_obb_models(direct, runtime, batch_size=1)
    assert calls == [("/m.pt", None)]


import warnings


def test_extract_obb_result_drops_zero_height_without_divzero_warning():
    """H6 parity: zero-height (non-positive geometry) detections are dropped
    (legacy _obb_geometry:303-312), and dropping them must not raise a
    divide-by-zero RuntimeWarning along the way."""

    def _t(arr):
        from unittest.mock import MagicMock

        m = MagicMock()
        m.cpu.return_value.numpy.return_value = arr
        return m

    # One detection with h=0 (degenerate but possible from YOLO) plus one valid.
    xywhr = np.array(
        [[100.0, 100.0, 20.0, 0.0, 0.5], [50.0, 50.0, 18.0, 9.0, 0.3]],
        dtype=np.float32,
    )
    corners = np.zeros((2, 4, 2), dtype=np.float32)
    conf = np.array([0.8, 0.7], dtype=np.float32)
    from unittest.mock import MagicMock

    r = MagicMock()
    r.obb.xywhr = _t(xywhr)
    r.obb.xyxyxyxy = _t(corners)
    r.obb.conf = _t(conf)
    r.obb.__len__ = lambda self: 2

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        # Must NOT raise a RuntimeWarning
        result = extract_obb_result(r, frame_idx=0)
    # The degenerate h=0 detection is dropped; the valid one survives.
    assert result.num_detections == 1
    assert result.confidences[0] == pytest.approx(0.7)


def test_extract_raw_tensors_uses_runtime_device_for_empty_path():
    """Per code-quality fix: empty-result tensors must land on the runtime's device,
    not hardcoded cuda:0."""
    r = MagicMock()
    r.obb = None
    raw = _extract_raw_tensors(r, frame_idx=0, device="cpu")
    assert str(raw.xywhr.device) == "cpu"
    assert str(raw.corners.device) == "cpu"
    assert str(raw.conf.device) == "cpu"


def test_extract_obb_from_boxes_applies_fixed_angle():
    from types import SimpleNamespace

    from hydra_suite.core.inference.stages.obb import _extract_obb_from_boxes

    # One box: x1,y1,x2,y2 = 10,20,30,60 -> cx=20,cy=40,w=20,h=40
    result = SimpleNamespace(
        boxes=SimpleNamespace(
            xyxy=torch.tensor([[10.0, 20.0, 30.0, 60.0]]),
            conf=torch.tensor([0.9]),
        )
    )

    out = _extract_obb_from_boxes(result, frame_idx=3, fixed_angle_rad=0.0)

    assert out.num_detections == 1
    assert out.frame_idx == 3
    np.testing.assert_allclose(out.centroids[0], [20.0, 40.0], atol=1e-4)
    # w=20 < h=40, so _normalize_obb_geometry swaps to major=h=40, minor=w=20
    # and adds 90deg to the (here, 0deg) fixed angle.
    np.testing.assert_allclose(out.angles[0], np.pi / 2, atol=1e-4)
    np.testing.assert_allclose(out.sizes[0], 800.0, atol=1e-3)  # 20*40
    np.testing.assert_allclose(out.confidences[0], 0.9, atol=1e-4)


def test_extract_obb_from_boxes_empty_boxes_returns_empty_result():
    from types import SimpleNamespace

    from hydra_suite.core.inference.stages.obb import _extract_obb_from_boxes

    result = SimpleNamespace(
        boxes=SimpleNamespace(xyxy=torch.zeros((0, 4)), conf=torch.zeros(0))
    )
    out = _extract_obb_from_boxes(result, frame_idx=0, fixed_angle_rad=0.0)
    assert out.num_detections == 0


def test_extract_raw_tensors_from_boxes_keeps_everything_on_device():
    from types import SimpleNamespace

    from hydra_suite.core.inference.stages.obb import _extract_raw_tensors_from_boxes

    result = SimpleNamespace(
        boxes=SimpleNamespace(
            xyxy=torch.tensor([[10.0, 20.0, 30.0, 60.0]]),
            conf=torch.tensor([0.9]),
        )
    )

    raw = _extract_raw_tensors_from_boxes(
        result, frame_idx=3, fixed_angle_rad=0.5, device="cpu"
    )

    assert raw.frame_idx == 3
    assert isinstance(raw.xywhr, torch.Tensor)
    assert raw.xywhr.shape == (1, 5)
    torch.testing.assert_close(raw.xywhr[0, :4], torch.tensor([20.0, 40.0, 20.0, 40.0]))
    torch.testing.assert_close(raw.xywhr[0, 4], torch.tensor(0.5))
    torch.testing.assert_close(raw.conf, torch.tensor([0.9]))


def test_extract_raw_tensors_from_boxes_empty_boxes():
    from types import SimpleNamespace

    from hydra_suite.core.inference.stages.obb import _extract_raw_tensors_from_boxes

    result = SimpleNamespace(
        boxes=SimpleNamespace(xyxy=torch.zeros((0, 4)), conf=torch.zeros(0))
    )
    raw = _extract_raw_tensors_from_boxes(
        result, frame_idx=0, fixed_angle_rad=0.0, device="cpu"
    )
    assert raw.xywhr.shape == (0, 5)


def test_obb_models_has_callable_close():
    """Regression: OBBModels.close() must be a real method (it was once defined
    after a return inside _normalize_obb_geometry, so the class lacked it and
    InferenceRunner.close() crashed at session teardown)."""
    models = OBBModels(mode="direct")
    assert callable(models.close)
    models.close()  # must not raise


def test_run_direct_forwards_target_classes_to_predict():
    """H1 parity: OBBConfig.target_classes is passed as `classes=` to predict;
    empty target_classes maps to None (all classes) — legacy yolo_detector.py."""
    from hydra_suite.core.inference.config import OBBConfig, OBBDirectConfig
    from hydra_suite.core.inference.runtime import RuntimeContext
    from hydra_suite.core.inference.stages.obb import _run_direct

    rt = RuntimeContext(
        cuda_mode=False,
        device="cpu",
        use_nvdec=False,
        tensor_on_cuda=False,
    )
    captured = {}

    class _Model:
        def predict(self, frames, **kwargs):
            captured.update(kwargs)
            return []

    frames = [np.zeros((8, 8, 3), dtype=np.uint8)]

    cfg = OBBConfig(
        mode="direct",
        direct=OBBDirectConfig(model_path="/m.pt"),
        target_classes=[2, 3],
    )
    _run_direct(frames, _Model(), cfg, rt)
    assert captured["classes"] == [2, 3]

    captured.clear()
    cfg_all = OBBConfig(mode="direct", direct=OBBDirectConfig(model_path="/m.pt"))
    _run_direct(frames, _Model(), cfg_all, rt)
    assert captured["classes"] is None


def test_load_obb_models_sequential_dynamic_batching_warning(monkeypatch, caplog):
    """Sequential-mode OBB with batch_size>1 must log a WARNING about known
    detection discrepancy issues (documented in the TensorRT/CoreML spec).
    Warning must NOT fire for:
    - sequential mode with batch_size=1
    - direct mode with any batch_size
    """
    import logging

    import hydra_suite.core.inference.stages.obb as obb_mod

    def fake_load_yolo(model_path, compute_runtime, **kwargs):
        return object()

    monkeypatch.setattr(obb_mod, "_load_yolo", fake_load_yolo)

    runtime = RuntimeContext(
        cuda_mode=True,
        device="cuda:0",
        use_nvdec=False,
        tensor_on_cuda=False,
        requested_gpu=True,
    )

    # Test 1: sequential mode with batch_size > 1 → WARNING
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        config_seq = OBBConfig(
            mode="sequential",
            sequential=OBBSequentialConfig(
                detect_model_path="/detect.pt", obb_model_path="/obb.pt"
            ),
        )
        obb_mod.load_obb_models(config_seq, runtime, batch_size=4)
    assert any(
        "Sequential-mode OBB" in record.message and "dynamic-batch" in record.message
        for record in caplog.records
    ), "Expected warning about sequential-mode dynamic batching with batch_size>1"

    # Test 2: sequential mode with batch_size = 1 → NO WARNING
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        obb_mod.load_obb_models(config_seq, runtime, batch_size=1)
    assert not any(
        "Sequential-mode OBB" in record.message and "dynamic-batch" in record.message
        for record in caplog.records
    ), "Should NOT warn for sequential mode with batch_size=1"

    # Test 3: direct mode with batch_size > 1 → NO WARNING
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        config_direct = OBBConfig(
            mode="direct",
            direct=OBBDirectConfig(model_path="/m.pt"),
        )
        obb_mod.load_obb_models(config_direct, runtime, batch_size=4)
    assert not any(
        "Sequential-mode OBB" in record.message and "dynamic-batch" in record.message
        for record in caplog.records
    ), "Should NOT warn for direct mode"


def test_extract_obb_from_masks_computes_rotated_rect():
    import math
    from types import SimpleNamespace

    import numpy as np
    import torch

    from hydra_suite.core.inference.stages.obb import _extract_obb_from_masks

    # A 40x20 axis-aligned rectangle mask at (50, 30) in a 100x60 "mask-space"
    # canvas that is ALSO treated as the original frame (gain=1, no padding)
    # for this test -- Task 3 already covers the gain/pad math independently.
    mh = mw = 100
    ys, xs = torch.meshgrid(
        torch.arange(mh, dtype=torch.float32),
        torch.arange(mw, dtype=torch.float32),
        indexing="ij",
    )
    mask = (
        ((xs >= 30) & (xs <= 70) & (ys >= 20) & (ys <= 40)).float().unsqueeze(0)
    )  # (1, 100, 100)

    result = SimpleNamespace(
        masks=SimpleNamespace(data=mask),
        boxes=SimpleNamespace(
            xyxy=torch.tensor([[30.0, 20.0, 70.0, 40.0]]),
            conf=torch.tensor([0.8]),
        ),
        orig_shape=(100, 100),
    )

    out = _extract_obb_from_masks(result, frame_idx=5)

    assert out.num_detections == 1
    assert out.frame_idx == 5
    np.testing.assert_allclose(out.centroids[0], [50.0, 30.0], atol=1.5)
    np.testing.assert_allclose(out.sizes[0], 800.0, atol=60.0)  # ~40*20
    assert out.angles[0] < math.radians(8) or out.angles[0] > math.radians(172)
    np.testing.assert_allclose(out.confidences[0], 0.8, atol=1e-4)


def test_extract_obb_from_masks_no_masks_returns_empty_result():
    from types import SimpleNamespace

    from hydra_suite.core.inference.stages.obb import _extract_obb_from_masks

    result = SimpleNamespace(
        masks=None, boxes=SimpleNamespace(conf=None), orig_shape=(10, 10)
    )
    out = _extract_obb_from_masks(result, frame_idx=1)
    assert out.num_detections == 0


def test_extract_raw_tensors_from_masks_keeps_everything_on_device():
    from types import SimpleNamespace

    import torch

    from hydra_suite.core.inference.stages.obb import _extract_raw_tensors_from_masks

    mh = mw = 100
    ys, xs = torch.meshgrid(
        torch.arange(mh, dtype=torch.float32),
        torch.arange(mw, dtype=torch.float32),
        indexing="ij",
    )
    mask = ((xs >= 30) & (xs <= 70) & (ys >= 20) & (ys <= 40)).float().unsqueeze(0)

    result = SimpleNamespace(
        masks=SimpleNamespace(data=mask),
        boxes=SimpleNamespace(
            xyxy=torch.tensor([[30.0, 20.0, 70.0, 40.0]]),
            conf=torch.tensor([0.8]),
        ),
        orig_shape=(100, 100),
    )

    raw = _extract_raw_tensors_from_masks(result, frame_idx=5, device="cpu")

    assert raw.frame_idx == 5
    assert isinstance(raw.xywhr, torch.Tensor)
    assert raw.xywhr.shape == (1, 5)
    assert raw.xywhr.device.type == "cpu"  # sanity: still a tensor, no numpy conversion
    torch.testing.assert_close(raw.conf, torch.tensor([0.8]))


def test_extract_raw_tensors_from_masks_no_masks_returns_empty():
    from types import SimpleNamespace

    from hydra_suite.core.inference.stages.obb import _extract_raw_tensors_from_masks

    result = SimpleNamespace(
        masks=None, boxes=SimpleNamespace(conf=None), orig_shape=(10, 10)
    )
    raw = _extract_raw_tensors_from_masks(result, frame_idx=1, device="cpu")
    assert raw.xywhr.shape == (0, 5)


def test_run_direct_dispatches_to_detect_extraction(monkeypatch):
    from types import SimpleNamespace

    import numpy as np
    import torch

    from hydra_suite.core.inference.config import OBBConfig, OBBDirectConfig
    from hydra_suite.core.inference.stages.obb import OBBModels, run_obb

    class _FakeDetectModel:
        def predict(self, frames, **kwargs):
            return [
                SimpleNamespace(
                    boxes=SimpleNamespace(
                        xyxy=torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
                        conf=torch.tensor([0.7]),
                    )
                )
                for _ in frames
            ]

    config = OBBConfig(
        mode="direct",
        direct=OBBDirectConfig(
            model_path="fake.pt", model_task="detect", fixed_angle_deg=45.0
        ),
    )
    models = OBBModels(mode="direct", direct_model=_FakeDetectModel())
    runtime = SimpleNamespace(tensor_on_cuda=False, device="cpu")

    results = run_obb([np.zeros((20, 20, 3), dtype=np.uint8)], models, config, runtime)

    assert len(results) == 1
    assert results[0].num_detections == 1


def test_run_direct_detect_uses_raw_tensor_fast_path_when_tensor_on_cuda():
    from types import SimpleNamespace

    import numpy as np
    import torch

    from hydra_suite.core.inference.config import OBBConfig, OBBDirectConfig
    from hydra_suite.core.inference.stages.obb import OBBModels, run_obb

    class _FakeDetectModel:
        def predict(self, frames, **kwargs):
            return [
                SimpleNamespace(
                    boxes=SimpleNamespace(
                        xyxy=torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
                        conf=torch.tensor([0.7]),
                    )
                )
                for _ in frames
            ]

    config = OBBConfig(
        mode="direct",
        direct=OBBDirectConfig(model_path="fake.pt", model_task="detect"),
    )
    models = OBBModels(mode="direct", direct_model=_FakeDetectModel())
    runtime = SimpleNamespace(tensor_on_cuda=True, device="cpu")

    results = run_obb([np.zeros((20, 20, 3), dtype=np.uint8)], models, config, runtime)

    # tensor_on_cuda=True must return _RawOBBTensors (a torch-tensor
    # namedtuple), NOT an already-materialized OBBResult.
    assert hasattr(results[0], "xywhr")
    assert not hasattr(results[0], "corners") or isinstance(
        results[0].xywhr, torch.Tensor
    )


# ---------------------------------------------------------------------------
# Final-review CRITICAL 1: letterbox inversion on the native-CUDA/NVDEC path
# must also apply to detect (boxes) and segment (masks) results, not just OBB.
# ---------------------------------------------------------------------------


class _FakeBoxes:
    """Duck-type of ultralytics Boxes: `.data` is the single source of truth."""

    def __init__(self, data: torch.Tensor):
        self.data = data

    def __len__(self) -> int:
        return int(self.data.shape[0])

    @property
    def xyxy(self) -> torch.Tensor:
        return self.data[:, :4]

    @property
    def conf(self) -> torch.Tensor:
        return self.data[:, 4]


def _letterbox_params(h: int, w: int, imgsz: int):
    r = min(imgsz / h, imgsz / w)
    new_h, new_w = int(h * r), int(w * r)
    return r, (imgsz - new_w) // 2, (imgsz - new_h) // 2


def _force_cuda_frames(monkeypatch):
    """Fake the 'frames are CUDA tensors' branch so it runs on CPU tensors."""
    import hydra_suite.core.inference.stages.obb as obb_mod

    monkeypatch.setattr(obb_mod, "_frames_are_cuda_tensors", lambda frames: True)
    return obb_mod


def test_run_direct_detect_cuda_frames_returns_original_frame_coords(monkeypatch):
    """CRITICAL 1: detect results must be un-letterboxed back to frame coords."""
    from types import SimpleNamespace

    obb_mod = _force_cuda_frames(monkeypatch)

    H, W, IMGSZ = 40, 80, 64
    r, pad_left, pad_top = _letterbox_params(H, W, IMGSZ)
    # True original-frame box, and its letterbox-space image.
    x1, y1, x2, y2 = 10.0, 10.0, 30.0, 20.0
    lb = torch.tensor(
        [
            [
                x1 * r + pad_left,
                y1 * r + pad_top,
                x2 * r + pad_left,
                y2 * r + pad_top,
                0.9,
                0.0,
            ]
        ]
    )

    class _FakeModel:
        imgsz = IMGSZ

        def predict(self, batched, **kwargs):
            assert batched.shape[-2:] == (IMGSZ, IMGSZ)
            return [
                SimpleNamespace(
                    obb=None,
                    masks=None,
                    boxes=_FakeBoxes(lb.clone()),
                    orig_shape=(IMGSZ, IMGSZ),
                )
            ]

    cfg = OBBConfig(
        mode="direct",
        direct=OBBDirectConfig(model_path="/m.pt", model_task="detect"),
    )
    frames = [torch.zeros((H, W, 3), dtype=torch.uint8)]
    out = obb_mod._run_direct(frames, _FakeModel(), cfg, _cpu_rt())

    assert out[0].num_detections == 1
    np.testing.assert_allclose(out[0].centroids[0], [20.0, 15.0], atol=0.6)
    # w=20, h=10 -> size = 200 in ORIGINAL-frame pixels (not r**2-scaled).
    np.testing.assert_allclose(out[0].sizes[0], 200.0, rtol=0.06)


def test_run_direct_segment_cuda_frames_returns_original_frame_coords(monkeypatch):
    """CRITICAL 1: segment masks must map back to original-frame coordinates."""
    from types import SimpleNamespace

    obb_mod = _force_cuda_frames(monkeypatch)

    H, W, IMGSZ = 40, 80, 64
    r, pad_left, pad_top = _letterbox_params(H, W, IMGSZ)
    x1, y1, x2, y2 = 10.0, 12.0, 30.0, 22.0  # original-frame box (w=20, h=10)
    lx1, ly1 = x1 * r + pad_left, y1 * r + pad_top
    lx2, ly2 = x2 * r + pad_left, y2 * r + pad_top
    lb = torch.tensor([[lx1, ly1, lx2, ly2, 0.9, 0.0]])

    # Mask in letterbox space, at letterbox resolution.
    ys, xs = torch.meshgrid(
        torch.arange(IMGSZ, dtype=torch.float32),
        torch.arange(IMGSZ, dtype=torch.float32),
        indexing="ij",
    )
    mask = ((xs >= lx1) & (xs <= lx2) & (ys >= ly1) & (ys <= ly2)).float().unsqueeze(0)

    class _FakeModel:
        imgsz = IMGSZ

        def predict(self, batched, **kwargs):
            return [
                SimpleNamespace(
                    obb=None,
                    boxes=_FakeBoxes(lb.clone()),
                    masks=SimpleNamespace(data=mask.clone()),
                    orig_shape=(IMGSZ, IMGSZ),
                )
            ]

    cfg = OBBConfig(
        mode="direct",
        direct=OBBDirectConfig(model_path="/m.pt", model_task="segment"),
    )
    frames = [torch.zeros((H, W, 3), dtype=torch.uint8)]
    out = obb_mod._run_direct(frames, _FakeModel(), cfg, _cpu_rt())

    assert out[0].num_detections == 1
    np.testing.assert_allclose(out[0].centroids[0], [20.0, 17.0], atol=2.0)
    # Original-frame extent ~20 x 10 -> size ~200 px^2 (letterbox space would
    # report ~200 * r**2 == 128).
    np.testing.assert_allclose(out[0].sizes[0], 200.0, rtol=0.25)


# ---------------------------------------------------------------------------
# Final-review IMPORTANT 4: a mismatched checkpoint task must fail loudly.
# ---------------------------------------------------------------------------


def test_load_obb_models_rejects_checkpoint_task_mismatch(monkeypatch):
    import hydra_suite.core.inference.stages.obb as obb_mod

    class _SegCheckpoint:
        task = "segment"

    monkeypatch.setattr(obb_mod, "_load_yolo", lambda *a, **k: _SegCheckpoint())
    runtime = RuntimeContext(
        cuda_mode=False,
        device="cpu",
        use_nvdec=False,
        tensor_on_cuda=False,
    )
    cfg = OBBConfig(
        mode="direct",
        direct=OBBDirectConfig(model_path="/seg.pt", model_task="obb"),
    )
    with pytest.raises(ValueError, match="segment"):
        obb_mod.load_obb_models(cfg, runtime)


def test_load_obb_models_accepts_matching_checkpoint_task(monkeypatch):
    import hydra_suite.core.inference.stages.obb as obb_mod

    class _SegCheckpoint:
        task = "segment"

    monkeypatch.setattr(obb_mod, "_load_yolo", lambda *a, **k: _SegCheckpoint())
    runtime = RuntimeContext(
        cuda_mode=False,
        device="cpu",
        use_nvdec=False,
        tensor_on_cuda=False,
    )
    cfg = OBBConfig(
        mode="direct",
        direct=OBBDirectConfig(model_path="/seg.pt", model_task="segment"),
    )
    models = obb_mod.load_obb_models(cfg, runtime)
    assert models.mode == "direct"


def test_load_obb_models_warns_when_direct_artifact_task_unverifiable(
    monkeypatch, caplog
):
    import logging

    import hydra_suite.core.inference.stages.obb as obb_mod

    class _TasklessExecutor:  # a direct executor / adapter: no `.task`
        pass

    monkeypatch.setattr(obb_mod, "_load_yolo", lambda *a, **k: _TasklessExecutor())
    runtime = RuntimeContext(
        cuda_mode=False,
        device="cpu",
        use_nvdec=False,
        tensor_on_cuda=False,
    )
    cfg = OBBConfig(
        mode="direct",
        direct=OBBDirectConfig(model_path="/eng.engine", model_task="segment"),
    )
    with caplog.at_level(logging.WARNING):
        obb_mod.load_obb_models(cfg, runtime)
    assert any(
        "no task metadata" in r.message.lower() or "cannot" in r.message.lower()
        for r in caplog.records
    ), "expected an unverifiable-task warning"


# ---------------------------------------------------------------------------
# Segment pre-cap optimization: cap detections by confidence BEFORE the
# expensive rotated_rect_from_masks kernel runs. Pure optimization -- the final
# OBBResult must be byte-identical to the old post-cap path.
# ---------------------------------------------------------------------------


def _make_segment_result(confs, frame_idx=0, canvas=120):
    """Build a fake ultralytics segment result with len(confs) valid detections.

    Each detection is a distinct axis-aligned filled rectangle so that every
    row produces finite geometry (no valid-mask drops) and centroids differ,
    letting equivalence assertions catch any mis-ordering. orig_shape == mask
    shape (gain=1, no pad).
    """
    from types import SimpleNamespace

    n = len(confs)
    ys, xs = torch.meshgrid(
        torch.arange(canvas, dtype=torch.float32),
        torch.arange(canvas, dtype=torch.float32),
        indexing="ij",
    )
    masks = []
    boxes = []
    for i in range(n):
        cx = 20.0 + i * 8.0
        cy = 60.0
        x0, x1, y0, y1 = cx - 10, cx + 10, cy - 15, cy + 15
        m = ((xs >= x0) & (xs <= x1) & (ys >= y0) & (ys <= y1)).float()
        masks.append(m)
        boxes.append([x0, y0, x1, y1])
    return SimpleNamespace(
        masks=SimpleNamespace(data=torch.stack(masks)),
        boxes=SimpleNamespace(
            xyxy=torch.tensor(boxes, dtype=torch.float32),
            conf=torch.tensor(confs, dtype=torch.float32),
        ),
        orig_shape=(canvas, canvas),
    )


def _spy_kernel(monkeypatch):
    """Wrap rotated_rect_from_masks to record the N it is invoked with."""
    import hydra_suite.core.inference.stages.obb as obb_mod

    real = obb_mod.rotated_rect_from_masks
    calls = []

    def spy(mask_tensor, boxes_mask_space, **kwargs):
        calls.append(int(mask_tensor.shape[0]))
        # sanity: inputs must still be device tensors, never numpy-converted
        assert isinstance(mask_tensor, torch.Tensor)
        assert isinstance(boxes_mask_space, torch.Tensor)
        return real(mask_tensor, boxes_mask_space, **kwargs)

    monkeypatch.setattr(obb_mod, "rotated_rect_from_masks", spy)
    return calls


def _assert_obb_equal(a, b):
    assert a.frame_idx == b.frame_idx
    assert a.num_detections == b.num_detections
    np.testing.assert_allclose(a.centroids, b.centroids, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(a.angles, b.angles, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(a.sizes, b.sizes, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(a.confidences, b.confidences, rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(a.detection_ids, b.detection_ids)


def test_extract_obb_from_masks_precaps_before_kernel(monkeypatch):
    from hydra_suite.core.inference.stages.obb import (
        _apply_raw_detection_cap,
        _extract_obb_from_masks,
    )

    confs = [0.1, 0.9, 0.3, 0.7, 0.5, 0.8, 0.2, 0.6]
    cap = 3

    # OLD behaviour: kernel on all N, then post-cap.
    expected = _apply_raw_detection_cap(
        _extract_obb_from_masks(_make_segment_result(confs), frame_idx=5), cap
    )

    # NEW behaviour: pre-cap to top-`cap`, kernel sees only `cap`, then the
    # caller's post-cap (a no-op re-sort) is applied.
    calls = _spy_kernel(monkeypatch)
    new_final = _apply_raw_detection_cap(
        _extract_obb_from_masks(
            _make_segment_result(confs), frame_idx=5, raw_detection_cap=cap
        ),
        cap,
    )

    assert calls == [cap]  # optimization fired: kernel processed only `cap`
    _assert_obb_equal(new_final, expected)


def test_extract_obb_from_masks_cap_disabled_processes_all(monkeypatch):
    from hydra_suite.core.inference.stages.obb import _extract_obb_from_masks

    confs = [0.1, 0.9, 0.3, 0.7, 0.5]
    calls = _spy_kernel(monkeypatch)
    out = _extract_obb_from_masks(
        _make_segment_result(confs), frame_idx=0, raw_detection_cap=0
    )
    assert calls == [len(confs)]  # cap<=0 disables pre-cap; kernel sees all N
    assert out.num_detections == len(confs)


def test_extract_raw_tensors_from_masks_precaps_before_kernel(monkeypatch):
    from hydra_suite.core.inference.stages.obb import (
        _apply_raw_detection_cap,
        _extract_raw_tensors_from_masks,
        materialize_tensors,
    )

    confs = [0.1, 0.9, 0.3, 0.7, 0.5, 0.8, 0.2, 0.6]
    cap = 3

    # OLD: raw tensors for all N, materialize (which applies its own cap).
    raw_all = _extract_raw_tensors_from_masks(
        _make_segment_result(confs, frame_idx=7), frame_idx=7, device="cpu"
    )
    expected = materialize_tensors(raw_all, raw_detection_cap=cap)

    # NEW: pre-cap so the kernel sees only `cap` detections.
    calls = _spy_kernel(monkeypatch)
    raw_new = _extract_raw_tensors_from_masks(
        _make_segment_result(confs, frame_idx=7),
        frame_idx=7,
        device="cpu",
        raw_detection_cap=cap,
    )
    new_final = materialize_tensors(raw_new, raw_detection_cap=cap)

    assert calls == [cap]
    # The raw pre-cap must keep tensors on-device (no numpy/host conversion).
    assert isinstance(raw_new.conf, torch.Tensor)
    assert raw_new.conf.shape[0] == cap
    _assert_obb_equal(new_final, expected)

    # Sanity: also matches a direct post-cap of the CPU-materializing path.
    from hydra_suite.core.inference.stages.obb import _extract_obb_from_masks

    cpu_expected = _apply_raw_detection_cap(
        _extract_obb_from_masks(_make_segment_result(confs), frame_idx=7), cap
    )
    _assert_obb_equal(new_final, cpu_expected)


def test_extract_raw_tensors_from_masks_precap_is_cpu_free(monkeypatch):
    """The raw path (pre-cap + kernel) must not sync: no .cpu/.item/.numpy/.tolist."""
    from hydra_suite.core.inference.stages.obb import _extract_raw_tensors_from_masks

    for name in ("cpu", "item", "numpy", "tolist"):

        def _raise(self, *a, _n=name, **k):
            raise AssertionError(f"raw path invoked Tensor.{_n}() -- host sync!")

        monkeypatch.setattr(torch.Tensor, name, _raise, raising=True)

    result = _make_segment_result([0.1, 0.9, 0.3, 0.7, 0.5], frame_idx=2)
    raw = _extract_raw_tensors_from_masks(
        result, frame_idx=2, device="cpu", raw_detection_cap=3
    )
    assert raw.conf.shape[0] == 3


# ---------------------------------------------------------------------------
# Segment-as-OBB kernel knobs: seg_num_angles/seg_crop_size/seg_pad_ratio/
# seg_mask_threshold must be forwarded from OBBDirectConfig all the way to
# rotated_rect_from_masks, not silently dropped in favor of its own defaults.
# ---------------------------------------------------------------------------


def test_extract_obb_from_masks_forwards_configured_kernel_params(monkeypatch):
    import hydra_suite.core.inference.stages.obb as obb_mod

    recorded = {}
    real = obb_mod.rotated_rect_from_masks

    def spy(mask_tensor, boxes_mask_space, **kwargs):
        recorded.update(kwargs)
        return real(mask_tensor, boxes_mask_space, **kwargs)

    monkeypatch.setattr(obb_mod, "rotated_rect_from_masks", spy)

    result = _make_segment_result([0.9])
    obb_mod._extract_obb_from_masks(
        result,
        frame_idx=0,
        num_angles=48,
        crop_size=128,
        pad_ratio=0.25,
        mask_threshold=0.35,
    )

    assert recorded == {
        "num_angles": 48,
        "crop_size": 128,
        "pad_ratio": 0.25,
        "mask_threshold": 0.35,
        # CPU-materializing path opts into foreground-only projection.
        "foreground_only": True,
    }


def test_extract_raw_tensors_from_masks_forwards_configured_kernel_params(
    monkeypatch,
):
    import hydra_suite.core.inference.stages.obb as obb_mod

    recorded = {}
    real = obb_mod.rotated_rect_from_masks

    def spy(mask_tensor, boxes_mask_space, **kwargs):
        recorded.update(kwargs)
        return real(mask_tensor, boxes_mask_space, **kwargs)

    monkeypatch.setattr(obb_mod, "rotated_rect_from_masks", spy)

    result = _make_segment_result([0.9])
    obb_mod._extract_raw_tensors_from_masks(
        result,
        frame_idx=0,
        device="cpu",
        num_angles=48,
        crop_size=128,
        pad_ratio=0.25,
        mask_threshold=0.35,
    )

    assert recorded == {
        "num_angles": 48,
        "crop_size": 128,
        "pad_ratio": 0.25,
        "mask_threshold": 0.35,
    }


def test_extract_obb_from_masks_enables_foreground_only(monkeypatch):
    """CPU-materializing path must opt into foreground_only=True."""
    import hydra_suite.core.inference.stages.obb as obb_mod

    recorded = {}
    real = obb_mod.rotated_rect_from_masks

    def spy(mask_tensor, boxes_mask_space, **kwargs):
        recorded.update(kwargs)
        return real(mask_tensor, boxes_mask_space, **kwargs)

    monkeypatch.setattr(obb_mod, "rotated_rect_from_masks", spy)
    obb_mod._extract_obb_from_masks(_make_segment_result([0.9]), frame_idx=0)
    assert recorded.get("foreground_only") is True


def test_extract_raw_tensors_from_masks_keeps_full_pixel_projection(monkeypatch):
    """The zero-CPU-sync raw path must NOT opt into foreground_only (default
    False / omitted), keeping its full-pixel, host-sync-free projection."""
    import hydra_suite.core.inference.stages.obb as obb_mod

    recorded = {}
    real = obb_mod.rotated_rect_from_masks

    def spy(mask_tensor, boxes_mask_space, **kwargs):
        recorded.update(kwargs)
        return real(mask_tensor, boxes_mask_space, **kwargs)

    monkeypatch.setattr(obb_mod, "rotated_rect_from_masks", spy)
    obb_mod._extract_raw_tensors_from_masks(
        _make_segment_result([0.9]), frame_idx=0, device="cpu"
    )
    # Either omitted entirely or explicitly False -- never True.
    assert recorded.get("foreground_only", False) is False


def test_extract_obb_from_masks_defaults_match_kernel_defaults(monkeypatch):
    """Omitting the kwargs must reproduce the kernel's own defaults exactly."""
    import hydra_suite.core.inference.stages.obb as obb_mod

    recorded = {}
    real = obb_mod.rotated_rect_from_masks

    def spy(mask_tensor, boxes_mask_space, **kwargs):
        recorded.update(kwargs)
        return real(mask_tensor, boxes_mask_space, **kwargs)

    monkeypatch.setattr(obb_mod, "rotated_rect_from_masks", spy)

    result = _make_segment_result([0.9])
    obb_mod._extract_obb_from_masks(result, frame_idx=0)

    assert recorded == {
        "num_angles": 24,
        "crop_size": 64,
        "pad_ratio": 0.15,
        "mask_threshold": 0.5,
        "foreground_only": True,
    }


def test_run_direct_segment_threads_configured_kernel_params(monkeypatch):
    """_run_direct's segment branch must pull the four knobs from config.direct."""
    import hydra_suite.core.inference.stages.obb as obb_mod

    recorded = {}
    real = obb_mod.rotated_rect_from_masks

    def spy(mask_tensor, boxes_mask_space, **kwargs):
        recorded.update(kwargs)
        return real(mask_tensor, boxes_mask_space, **kwargs)

    monkeypatch.setattr(obb_mod, "rotated_rect_from_masks", spy)

    result = _make_segment_result([0.9])

    class _FakeModel:
        def predict(self, frames, **kwargs):
            return [result]

    cfg = OBBConfig(
        mode="direct",
        direct=OBBDirectConfig(
            model_path="/m.pt",
            model_task="segment",
            seg_num_angles=48,
            seg_crop_size=128,
            seg_pad_ratio=0.25,
            seg_mask_threshold=0.35,
        ),
    )
    frames = [np.zeros((120, 120, 3), dtype=np.uint8)]
    obb_mod._run_direct(frames, _FakeModel(), cfg, _cpu_rt())

    assert recorded == {
        "num_angles": 48,
        "crop_size": 128,
        "pad_ratio": 0.25,
        "mask_threshold": 0.35,
        # _run_direct's segment branch under a CPU (materializing) runtime
        # routes through _extract_obb_from_masks, which opts into foreground-only.
        "foreground_only": True,
    }
