"""Unit tests for the GPU-native classifier crop + forward path.

These run on the CPU torch device (grid_sample works there), so most of the
feature's correctness is provable off the CUDA box; only the end-to-end
determinism/agreement/perf gate needs mehek (see the implementation plan Task 6).
"""

import numpy as np
import pytest
import torch

from hydra_suite.core.inference.result import OBBResult


def _toy_obb(n=3, frame_idx=0):
    """Axis-aligned boxes with valid, non-degenerate 4x2 corners."""
    corners = np.zeros((n, 4, 2), np.float32)
    centroids = np.zeros((n, 2), np.float32)
    for i in range(n):
        x0, y0, w, h = 10 + 40 * i, 12, 30, 16
        corners[i] = [[x0, y0], [x0 + w, y0], [x0 + w, y0 + h], [x0, y0 + h]]
        centroids[i] = [x0 + w / 2, y0 + h / 2]
    return OBBResult(
        frame_idx=frame_idx,
        centroids=centroids,
        angles=np.zeros(n, np.float32),
        sizes=np.full(n, 30 * 16, np.float32),
        shapes=np.tile([30 * 16, 30 / 16], (n, 1)).astype(np.float32),
        confidences=np.full(n, 0.9, np.float32),
        corners=corners,
        detection_ids=np.arange(n, dtype=np.int64) + frame_idx * 10000,
    )


def test_gpu_classifier_crop_shape_device():
    from hydra_suite.core.inference.stages.crops import extract_classifier_crops_gpu

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    frame = (
        torch.randint(0, 256, (3, 200, 300), dtype=torch.uint8).float().div(255).to(dev)
    )
    crops = extract_classifier_crops_gpu(frame, _toy_obb(3), (128, 128), 2.0, 1.3, dev)
    assert crops.shape == (3, 3, 128, 128)
    assert str(crops.device).startswith(dev)
    assert crops.dtype == torch.float32


def test_gpu_classifier_crop_empty():
    from hydra_suite.core.inference.stages.crops import extract_classifier_crops_gpu

    frame = torch.zeros((3, 50, 50))
    empty = OBBResult(
        frame_idx=0,
        centroids=np.zeros((0, 2), np.float32),
        angles=np.zeros(0, np.float32),
        sizes=np.zeros(0, np.float32),
        shapes=np.zeros((0, 2), np.float32),
        confidences=np.zeros(0, np.float32),
        corners=np.zeros((0, 4, 2), np.float32),
        detection_ids=np.zeros(0, np.int64),
    )
    crops = extract_classifier_crops_gpu(frame, empty, (128, 128), 2.0, 1.3, "cpu")
    assert crops.shape == (0, 3, 128, 128)


def test_gpu_vs_cpu_classifier_crop_close():
    """grid_sample != cv2, but the crops must be close (guards affine mistakes)."""
    from hydra_suite.core.inference.stages.crops import (
        extract_classifier_crops,
        extract_classifier_crops_gpu,
    )

    frame = np.random.default_rng(0).integers(0, 256, (200, 300, 3), np.uint8)
    obb = _toy_obb(3)
    cpu = extract_classifier_crops(frame, obb, (128, 128), 2.0, 1.3)  # list HWC uint8
    cpu_t = np.stack([c.astype(np.float32) / 255.0 for c in cpu])  # (N,H,W,C)
    ft = torch.from_numpy(frame.transpose(2, 0, 1)).float().div(255.0)
    gpu = extract_classifier_crops_gpu(ft, obb, (128, 128), 2.0, 1.3, "cpu")
    gpu_hwc = gpu.permute(0, 2, 3, 1).numpy()
    assert gpu_hwc.shape == cpu_t.shape
    assert float(np.abs(gpu_hwc - cpu_t).mean()) < 0.03  # < ~8/255 mean abs


# ---- Task 2/3: GPU factor-bundle forward ------------------------------------


def _fake_bundle(monkeypatch, active_backend="native"):
    from hydra_suite.core.identity.classification import backend as bk

    class _FakeFactor:
        _active_execution_backend = active_backend

        def predict_batch(self, crops):
            return [[np.array([0.2, 0.8], np.float32)] for _ in crops]

        def predict_batch_cuda(self, crops, input_is_bgr=True):
            return [[np.array([0.2, 0.8], np.float32)] for _ in crops]

    b = bk.ClassifierBackend.__new__(bk.ClassifierBackend)
    b._model = [_FakeFactor(), _FakeFactor()]
    monkeypatch.setattr(b, "_uses_factor_backends", lambda: True, raising=False)
    return b


def test_forward_multi_cuda_shape_matches_numpy(monkeypatch):
    b = _fake_bundle(monkeypatch)
    crops = [object(), object()]  # 2 crops
    numpy_out = b._forward_yolo_multi(crops)
    cuda_out = b._forward_multi_cuda(crops, True)
    assert cuda_out.shape == numpy_out.shape  # (2 crops, 4 = 2 factors x 2)
    np.testing.assert_allclose(cuda_out, numpy_out, rtol=0, atol=1e-6)


def test_supports_cuda_forward_bundle():
    b_native = _supports_helper("native")
    b_coreml = _supports_helper("coreml")
    assert b_native.supports_cuda_forward() is True
    assert b_coreml.supports_cuda_forward() is False


def _supports_helper(active_backend):
    from hydra_suite.core.identity.classification import backend as bk

    class _F:
        _active_execution_backend = active_backend

        def predict_batch_cuda(self, crops, input_is_bgr=True):
            return []

    b = bk.ClassifierBackend.__new__(bk.ClassifierBackend)
    b._model = [_F(), _F()]
    b._uses_factor_backends = lambda: True  # type: ignore[method-assign]
    return b


def test_predict_batch_cuda_uses_gpu_forward_for_capable_bundle(monkeypatch):

    b = _fake_bundle(monkeypatch)
    called = {"numpy_fallback": False, "multi_cuda": False}
    monkeypatch.setattr(b, "supports_cuda_forward", lambda: True, raising=False)
    monkeypatch.setattr(b, "_ensure_loaded", lambda: None, raising=False)
    monkeypatch.setattr(b, "_cardinalities", lambda: [2, 2], raising=False)
    monkeypatch.setattr(b, "_softmax", lambda row: np.asarray(row), raising=False)
    orig = b._forward_multi_cuda

    def _spy(c, bgr):
        called["multi_cuda"] = True
        return orig(c, bgr)

    monkeypatch.setattr(b, "_forward_multi_cuda", _spy, raising=False)

    def _no_numpy(crops):
        called["numpy_fallback"] = True
        return []

    monkeypatch.setattr(b, "predict_batch", _no_numpy, raising=False)

    out = b.predict_batch_cuda([object(), object()], input_is_bgr=True)
    assert called["multi_cuda"] and not called["numpy_fallback"]
    assert len(out) == 2 and len(out[0]) == 2  # 2 crops, 2 factors


# ---- Task 4: strict gpu-tier capability check -------------------------------


def test_load_cnn_strict_raises_without_cuda_forward(monkeypatch, tmp_path):
    from hydra_suite.core.identity.classification import backend as bk
    from hydra_suite.core.inference.stages import cnn as cnn_stage

    class _NoCudaBackend:
        def supports_cuda_forward(self):
            return False

        def close(self):
            pass

    monkeypatch.setattr(bk, "ClassifierBackend", lambda *a, **k: _NoCudaBackend())
    monkeypatch.setattr(
        cnn_stage,
        "resolved_backend_for",
        lambda rt: type("R", (), {"backend": "torch", "device": "cuda"})(),
    )
    rt = type("RT", (), {"tensor_on_cuda": True})()
    cfg = type(
        "C", (), {"model_path": str(tmp_path / "m.multihead.json"), "label": "x"}
    )()
    with pytest.raises(RuntimeError, match="CUDA-native"):
        cnn_stage.load_cnn_model(cfg, rt)


def test_load_cnn_no_raise_when_not_tensor_on_cuda(monkeypatch, tmp_path):
    # On MPS/CPU (tensor_on_cuda False), a non-CUDA classifier loads fine.
    from hydra_suite.core.identity.classification import backend as bk
    from hydra_suite.core.inference.stages import cnn as cnn_stage

    class _Backend:
        metadata = type(
            "M",
            (),
            {
                "input_size": (128, 128),
                "factor_names": ["f"],
                "class_names_per_factor": [["a", "b"]],
            },
        )()

        def supports_cuda_forward(self):
            return False

        def close(self):
            pass

    monkeypatch.setattr(bk, "ClassifierBackend", lambda *a, **k: _Backend())
    monkeypatch.setattr(
        cnn_stage,
        "resolved_backend_for",
        lambda rt: type("R", (), {"backend": "torch", "device": "mps"})(),
    )
    rt = type("RT", (), {"tensor_on_cuda": False})()
    cfg = type("C", (), {"model_path": str(tmp_path / "m.json"), "label": "x"})()
    model = cnn_stage.load_cnn_model(cfg, rt)  # must NOT raise
    assert model.input_size == (128, 128)


# ---- Task 5: stage routing (GPU path when tensor_on_cuda) --------------------


def test_run_cnn_batch_routes_by_tensor_on_cuda(monkeypatch):
    from hydra_suite.core.inference.stages import cnn as cnn_stage
    from hydra_suite.core.inference.stages import crops as crops_mod

    used = {"gpu": False, "cpu": False, "cuda_fwd": False, "numpy_fwd": False}

    class _FakeBatch:
        crops = torch.zeros((1, 3, 8, 8))
        obb_by_frame = {0: _toy_obb(1)}

        def select_frame(self, f):
            return np.array([0])

    monkeypatch.setattr(
        crops_mod,
        "extract_classifier_crops_batch_gpu",
        lambda *a, **k: (used.__setitem__("gpu", True) or _FakeBatch()),
    )
    monkeypatch.setattr(
        crops_mod,
        "extract_classifier_crops_batch",
        lambda *a, **k: (used.__setitem__("cpu", True) or _FakeBatch()),
    )

    class _Backend:
        def predict_batch_cuda(self, crops, input_is_bgr=True):
            used["cuda_fwd"] = True
            return [[np.array([0.5, 0.5], np.float32)]]

        def predict_batch(self, crops):
            used["numpy_fwd"] = True
            return [[np.array([0.5, 0.5], np.float32)]]

    model = cnn_stage.CNNModel(
        backend=_Backend(),
        input_size=(8, 8),
        factor_names=["f"],
        factor_class_names=[["a", "b"]],
    )
    cfg = type("C", (), {"label": "x"})()

    rt_gpu = type("RT", (), {"tensor_on_cuda": True, "device": "cpu"})()
    cnn_stage.run_cnn_batch([None], [_toy_obb(1)], model, cfg, rt_gpu)
    assert used["gpu"] and used["cuda_fwd"]
    assert not used["cpu"] and not used["numpy_fwd"]

    for k in used:
        used[k] = False
    rt_cpu = type("RT", (), {"tensor_on_cuda": False, "device": "cpu"})()
    cnn_stage.run_cnn_batch([None], [_toy_obb(1)], model, cfg, rt_cpu)
    assert used["cpu"] and used["numpy_fwd"]
    assert not used["gpu"] and not used["cuda_fwd"]
