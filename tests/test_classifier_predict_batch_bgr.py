import inspect

import numpy as np
import torch

from hydra_suite.core.individual.classification import backend


def test_predict_batch_accepts_input_is_bgr():
    sig = inspect.signature(
        backend.ClassifierBackend.predict_batch
    )  # actual class name
    assert "input_is_bgr" in sig.parameters


# ---- F2 functional: predict_batch_cuda's two CPU-fallback branches ---------
#
# ``predict_batch_cuda`` (backend.py ~1255) has two paths that fall back to
# the numpy ``predict_batch`` (~1348) instead of running a CUDA-native
# forward:
#   (a) a factor bundle where at least one factor lacks a CUDA-native
#       forward (``supports_cuda_forward()`` False for a multihead backend),
#   (b) a flat backend whose ``_active_execution_backend`` is neither
#       "native" nor "onnx" (an unrecognised/unhandled backend string).
# Both branches build ``numpy_crops`` the same way (``permute(1,2,0).cpu()
# .numpy()``) and call ``self.predict_batch(numpy_crops, input_is_bgr=...)``.
# Everything here (``_preprocess_cuda`` included) is plain torch tensor code
# with no CUDA-specific op, so it is fully exercisable on CPU tensors --
# preferred over a CUDA-only skip so this is actually SEEN passing here.


def _metadata(arch="resnet18", multi=False):
    return backend.ClassifierMetadata(
        arch="yolo_multihead" if multi else arch,
        input_size=(16, 16),
        is_multihead=multi,
        factor_names=["f1", "f2"] if multi else ["flat"],
        class_names_per_factor=[["a", "b"], ["c", "d"]] if multi else [["a", "b"]],
        monochrome=False,
        recommended_confidence_threshold=None,
        source_path="dummy.json",
    )


def _bgr_crop(r, g, b, h=8, w=8):
    """A (3, h, w) CHW float32 tensor with a known, distinct value per
    channel (BGR order, matching cv2), so a caller can tell whether the
    fallback re-permutes/re-flips channels correctly."""
    crop = torch.zeros(3, h, w, dtype=torch.float32)
    crop[0] = b  # channel 0 = B
    crop[1] = g  # channel 1 = G
    crop[2] = r  # channel 2 = R
    return crop


def test_cuda_fallback_factor_missing_cuda_forward_runs_without_typeerror(
    monkeypatch,
):
    """Branch (a): a multihead bundle where a factor lacks CUDA-native forward."""
    be = backend.ClassifierBackend.__new__(backend.ClassifierBackend)
    be._model_path = "bundle.multihead.json"
    be._metadata = _metadata(multi=True)

    class _CudaCapableFactor:
        _active_execution_backend = "native"

        def predict_batch_cuda(self, crops, input_is_bgr=True):
            return [[np.array([0.5, 0.5], np.float32)] for _ in crops]

    class _NoCudaFactor:
        _active_execution_backend = "onnx_only"  # not "native"/"onnx"

        # deliberately no predict_batch_cuda -> supports_cuda_forward() is False

    be._model = [_CudaCapableFactor(), _NoCudaFactor()]
    monkeypatch.setattr(be, "_ensure_loaded", lambda: None, raising=False)

    seen = {}

    def _fake_predict_batch(crops, input_is_bgr=True):
        seen["crops"] = crops
        seen["input_is_bgr"] = input_is_bgr
        return [[np.array([0.2, 0.8], np.float32), np.array([0.4, 0.6], np.float32)]]

    monkeypatch.setattr(be, "predict_batch", _fake_predict_batch, raising=False)

    crop = _bgr_crop(r=30, g=20, b=10)
    result = be.predict_batch_cuda([crop], input_is_bgr=True)

    assert len(result) == 1 and len(result[0]) == 2  # 1 crop, 2 factors
    np.testing.assert_allclose(result[0][0], [0.2, 0.8])
    np.testing.assert_allclose(result[0][1], [0.4, 0.6])
    assert seen["input_is_bgr"] is True  # forwarded unchanged, no re-flip

    # numpy_crops must be HWC with channel order preserved (BGR stays BGR --
    # the fallback only permutes axes, the BGR->RGB flip is predict_batch's job).
    numpy_crops = seen["crops"]
    assert len(numpy_crops) == 1
    hwc = numpy_crops[0]
    assert hwc.shape == (8, 8, 3)
    assert np.allclose(hwc[..., 0], 10.0)  # B
    assert np.allclose(hwc[..., 1], 20.0)  # G
    assert np.allclose(hwc[..., 2], 30.0)  # R


def test_cuda_fallback_unknown_execution_backend_runs_without_typeerror(monkeypatch):
    """Branch (b): a flat backend whose active_execution_backend is unrecognised."""
    be = backend.ClassifierBackend.__new__(backend.ClassifierBackend)
    be._model_path = "flat.json"
    be._metadata = _metadata(multi=False)
    be._model = object()  # never touched: predict_batch is monkeypatched below
    be._active_execution_backend = "some_unhandled_backend"
    monkeypatch.setattr(be, "_ensure_loaded", lambda: None, raising=False)

    seen = {}

    def _fake_predict_batch(crops, input_is_bgr=True):
        seen["crops"] = crops
        seen["input_is_bgr"] = input_is_bgr
        return [[np.array([0.9, 0.1], np.float32)]]

    monkeypatch.setattr(be, "predict_batch", _fake_predict_batch, raising=False)

    crop = _bgr_crop(r=99, g=88, b=77, h=16, w=16)  # matches metadata.input_size
    result = be.predict_batch_cuda([crop], input_is_bgr=False)

    assert len(result) == 1 and len(result[0]) == 1
    assert seen["input_is_bgr"] is False  # forwarded unchanged, no re-flip

    numpy_crops = seen["crops"]
    assert len(numpy_crops) == 1
    hwc = numpy_crops[0]
    assert hwc.shape == (16, 16, 3)
    assert np.allclose(hwc[..., 0], 77.0)  # B
    assert np.allclose(hwc[..., 1], 88.0)  # G
    assert np.allclose(hwc[..., 2], 99.0)  # R
