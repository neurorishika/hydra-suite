from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from hydra_suite.core.inference.config import OBBConfig, OBBDirectConfig
from hydra_suite.core.inference.result import OBBResult
from hydra_suite.core.inference.runtime import RuntimeContext
from hydra_suite.core.inference.stages.obb import (
    OBBModels,
    _empty_obb_result,
    _extract_obb_result,
    _extract_raw_tensors,
    _merge_obb_results,
    _RawOBBTensors,
    run_obb,
)


def _cpu_rt() -> RuntimeContext:
    return RuntimeContext(
        cuda_mode=False,
        device="cpu",
        use_nvdec=False,
        default_runtime="cpu",
        tensor_on_cuda=False,
    )


def _cuda_rt() -> RuntimeContext:
    # tensor_on_cuda=True for native PyTorch CUDA runtime
    return RuntimeContext(
        cuda_mode=True,
        device="cuda:0",
        use_nvdec=False,
        default_runtime="cuda",
        tensor_on_cuda=True,
    )


def _onnx_cuda_rt() -> RuntimeContext:
    # ONNX CUDA: cuda_mode=True but tensor_on_cuda=False (CPU numpy outputs)
    return RuntimeContext(
        cuda_mode=True,
        device="cuda:0",
        use_nvdec=False,
        default_runtime="cuda",
        tensor_on_cuda=False,
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
    result = _extract_obb_result(_mock_ul_result_numpy_compat(n=3), frame_idx=0)
    assert result.num_detections == 3
    assert result.centroids.shape == (3, 2)
    assert result.angles.shape == (3,)
    assert result.sizes.shape == (3,)
    assert result.corners.shape == (3, 4, 2)


def test_extract_obb_result_offset_shifts_centroids():
    result = _extract_obb_result(
        _mock_ul_result_numpy_compat(n=1), frame_idx=0, offset=(50.0, 30.0)
    )
    assert result.centroids[0, 0] == pytest.approx(150.0)
    assert result.centroids[0, 1] == pytest.approx(130.0)


def test_extract_obb_result_sizes_computed():
    result = _extract_obb_result(_mock_ul_result_numpy_compat(n=1), frame_idx=0)
    assert result.sizes[0] == pytest.approx(20.0 * 10.0)


def test_extract_obb_result_carries_detection_ids():
    """Per Correction 14: every constructed OBBResult must include detection_ids."""
    result = _extract_obb_result(_mock_ul_result_numpy_compat(n=3), frame_idx=7)
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
    merged = _merge_obb_results(0, [r1, r2])
    assert merged.num_detections == 5
    assert merged.detection_ids.shape == (5,)


def test_run_obb_cpu_returns_obb_result():
    config = OBBConfig(
        mode="direct",
        direct=OBBDirectConfig(model_path="/m.pt", compute_runtime="cpu"),
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
        direct=OBBDirectConfig(model_path="/m.pt", compute_runtime="cuda"),
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
        direct=OBBDirectConfig(model_path="/m.onnx", compute_runtime="onnx_cuda"),
    )
    mock_model = MagicMock()
    mock_model.predict.return_value = [_mock_ul_result_numpy_compat(n=2)]
    models = OBBModels(mode="direct", direct_model=mock_model)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    results = run_obb([frame], models, config, _onnx_cuda_rt())
    assert len(results) == 1
    # onnx_cuda is NOT tensor_on_cuda → extract to OBBResult, not _RawOBBTensors
    assert isinstance(results[0], OBBResult)


def test_load_yolo_routes_onnx_trt_to_direct_executor(monkeypatch, tmp_path):
    """ONNX/TRT runtimes route through the direct-executor path (no torch .to()).

    Task 15 (H4): _load_yolo now delegates to load_obb_executor, which builds a
    direct ONNX/TRT executor for onnx_*/tensorrt rather than loading a .pt and
    silently running PyTorch. We inject fakes so no real ORT/TRT is needed.
    """
    import hydra_suite.core.inference.runtime_artifacts as ra

    created = []

    def fake_executor(*, runtime, artifact_path, imgsz, class_names=None):
        created.append(runtime)
        return object()

    monkeypatch.setattr(ra, "_create_direct_executor", fake_executor)
    from hydra_suite.core.inference.stages.obb import _load_yolo

    onnx = tmp_path / "m.onnx"
    onnx.write_bytes(b"x")
    engine = tmp_path / "m.engine"
    engine.write_bytes(b"x")

    _load_yolo(str(onnx), "onnx_cuda", auto_export=False)
    _load_yolo(str(engine), "tensorrt", auto_export=False)
    _load_yolo(str(onnx), "onnx_coreml", auto_export=False)
    assert created == ["onnx", "tensorrt", "onnx"]


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
        result = _extract_obb_result(r, frame_idx=0)
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
        default_runtime="cpu",
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
