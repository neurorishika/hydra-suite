"""Tests for the CUDA-tensor frame path in _run_direct (obb.py).

These tests run entirely on CPU (no CUDA required) by:
  - Testing the pure letterbox + inverse-transform math as isolated functions.
  - Monkeypatching `.is_cuda` on CPU tensors to exercise the CUDA branch of
    `_run_direct` on CPU hardware.

CUDA verification (equivalence between numpy-list path and tensor path on
real GPU hardware with NVDEC) is done separately on the mehek box.
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Import helpers from obb.py
# ---------------------------------------------------------------------------
from hydra_suite.core.inference.stages.obb import (
    _gpu_letterbox_batch,
    _invert_letterbox_on_result,
    _resolve_imgsz,
    _run_direct,
)

# ---------------------------------------------------------------------------
# Helpers / stubs
# ---------------------------------------------------------------------------


def _xywhr_to_corners(xywhr: torch.Tensor) -> torch.Tensor:
    """(N,5) cx,cy,w,h,angle -> (N,4,2) rotated corners (tl,tr,br,bl).

    Mirrors ultralytics' xywhr2xyxyxyxy so the test's xyxyxyxy property is a
    faithful recomputation from the backing data.
    """
    cx, cy, w, h, ang = (xywhr[:, i] for i in range(5))
    dx, dy = w / 2, h / 2
    cos, sin = torch.cos(ang), torch.sin(ang)
    ox = torch.stack([-dx, dx, dx, -dx], dim=1)  # (N,4)
    oy = torch.stack([-dy, -dy, dy, dy], dim=1)
    x = cx[:, None] + ox * cos[:, None] - oy * sin[:, None]
    y = cy[:, None] + ox * sin[:, None] + oy * cos[:, None]
    return torch.stack([x, y], dim=2)  # (N,4,2)


class _FakeOBB:
    """Ultralytics-like OBB backed by a single ``data`` tensor.

    Faithfully models ultralytics: ``xywhr`` is a VIEW of ``data[:, :5]`` (so
    in-place mutation persists) while ``xyxyxyxy`` is RECOMPUTED from ``data``
    on every access (so mutating the returned corners is discarded). This is
    exactly why the fix writes through ``data`` — the test validates that.
    """

    def __init__(self, data: torch.Tensor):
        self.data = data  # (N, 7): cx, cy, w, h, angle, conf, cls

    @property
    def xywhr(self) -> torch.Tensor:
        return self.data[:, :5]

    @property
    def xyxyxyxy(self) -> torch.Tensor:
        return _xywhr_to_corners(self.data[:, :5])

    @property
    def conf(self) -> torch.Tensor:
        return self.data[:, 5]

    def __len__(self) -> int:
        return self.data.shape[0]


class _FakeResult:
    """Ultralytics-like Results object."""

    def __init__(self, data: torch.Tensor):
        self.obb = _FakeOBB(data)


def _make_result_in_imgsz_space(
    cx_lb: float,
    cy_lb: float,
    w_lb: float,
    h_lb: float,
    angle: float,
    conf: float = 0.9,
) -> _FakeResult:
    """Build a single-detection FakeResult whose coordinates are in imgsz space."""
    data = torch.tensor(
        [[cx_lb, cy_lb, w_lb, h_lb, angle, conf, 0.0]], dtype=torch.float32
    )
    return _FakeResult(data)


# ---------------------------------------------------------------------------
# Test 1: pure letterbox math — forward then inverse is identity
# ---------------------------------------------------------------------------


class TestGpuLetterboxBatch:
    """Unit tests for _gpu_letterbox_batch (CPU tensors only)."""

    def test_output_shape(self):
        """Batch has shape (B, 3, imgsz, imgsz)."""
        H, W, imgsz = 200, 400, 640
        frames = [torch.zeros((H, W, 3), dtype=torch.uint8) for _ in range(3)]
        batched, params = _gpu_letterbox_batch(frames, imgsz)
        assert batched.shape == (3, 3, imgsz, imgsz), batched.shape

    def test_params_non_square_frame(self):
        """For H=200, W=400, imgsz=640: r=min(640/200, 640/400)=1.6."""
        H, W, imgsz = 200, 400, 640
        frames = [torch.zeros((H, W, 3), dtype=torch.uint8)]
        _, params = _gpu_letterbox_batch(frames, imgsz)
        r, pad_left, pad_top = params[0]
        expected_r = min(imgsz / H, imgsz / W)  # min(3.2, 1.6) = 1.6
        assert abs(r - expected_r) < 1e-5, f"r={r} expected {expected_r}"
        # new_h = int(200*1.6) = 320, pad_top = (640-320)//2 = 160
        # new_w = int(400*1.6) = 640, pad_left = (640-640)//2 = 0
        assert pad_top == (imgsz - int(H * expected_r)) // 2
        assert pad_left == (imgsz - int(W * expected_r)) // 2

    def test_output_normalised(self):
        """All pixel values in [0, 1]."""
        frames = [torch.full((100, 100, 3), 255, dtype=torch.uint8)]
        batched, _ = _gpu_letterbox_batch(frames, 128)
        assert float(batched.max()) <= 1.0 + 1e-6

    def test_dtype_float32(self):
        frames = [torch.zeros((50, 50, 3), dtype=torch.uint8)]
        batched, _ = _gpu_letterbox_batch(frames, 64)
        assert batched.dtype == torch.float32


# ---------------------------------------------------------------------------
# Test 2: inverse letterbox math — known box, hand-computed expected result
# ---------------------------------------------------------------------------


class TestInvertLetterbox:
    """Unit tests for _invert_letterbox_on_result.

    Hand-computed test case
    -----------------------
    Frame: H=200, W=400, imgsz=640.
    Scale:  r = min(640/200, 640/400) = min(3.2, 1.6) = 1.6
    new_h = int(200*1.6) = 320
    new_w = int(400*1.6) = 640
    pad_top  = (640 - 320) // 2 = 160
    pad_left = (640 - 640) // 2 = 0

    A detection centred at the middle of the original frame (cx_orig=200,
    cy_orig=100) should map to imgsz-space as:
        cx_lb = cx_orig * r + pad_left = 200 * 1.6 + 0 = 320
        cy_lb = cy_orig * r + pad_top  = 100 * 1.6 + 160 = 320

    Box size w_orig=40, h_orig=20 maps to:
        w_lb = 40 * 1.6 = 64
        h_lb = 20 * 1.6 = 32

    Inversion must recover (200, 100, 40, 20) from (320, 320, 64, 32).
    """

    H, W, imgsz = 200, 400, 640

    @staticmethod
    def _lb_params():
        H, W, imgsz = 200, 400, 640
        r = min(imgsz / H, imgsz / W)
        new_h = int(H * r)
        new_w = int(W * r)
        pad_top = (imgsz - new_h) // 2
        pad_left = (imgsz - new_w) // 2
        return r, pad_left, pad_top

    def test_centroid_recovered(self):
        r, pad_left, pad_top = self._lb_params()
        # Forward: place detection at original-frame centre
        cx_orig, cy_orig = 200.0, 100.0
        cx_lb = cx_orig * r + pad_left
        cy_lb = cy_orig * r + pad_top
        result = _make_result_in_imgsz_space(cx_lb, cy_lb, 64.0, 32.0, 0.5)
        _invert_letterbox_on_result(result, r, pad_left, pad_top)
        got_cx = float(result.obb.xywhr[0, 0])
        got_cy = float(result.obb.xywhr[0, 1])
        assert abs(got_cx - cx_orig) < 1e-4, f"cx={got_cx} expected {cx_orig}"
        assert abs(got_cy - cy_orig) < 1e-4, f"cy={got_cy} expected {cy_orig}"

    def test_size_recovered(self):
        r, pad_left, pad_top = self._lb_params()
        w_orig, h_orig = 40.0, 20.0
        result = _make_result_in_imgsz_space(320.0, 320.0, w_orig * r, h_orig * r, 0.3)
        _invert_letterbox_on_result(result, r, pad_left, pad_top)
        got_w = float(result.obb.xywhr[0, 2])
        got_h = float(result.obb.xywhr[0, 3])
        assert abs(got_w - w_orig) < 1e-4, f"w={got_w} expected {w_orig}"
        assert abs(got_h - h_orig) < 1e-4, f"h={got_h} expected {h_orig}"

    def test_angle_unchanged(self):
        r, pad_left, pad_top = self._lb_params()
        angle = math.pi / 4
        result = _make_result_in_imgsz_space(320.0, 320.0, 64.0, 32.0, angle)
        _invert_letterbox_on_result(result, r, pad_left, pad_top)
        got_angle = float(result.obb.xywhr[0, 4])
        assert abs(got_angle - angle) < 1e-6, f"angle={got_angle} expected {angle}"

    def test_corners_recovered(self):
        r, pad_left, pad_top = self._lb_params()
        cx_orig, cy_orig, w_orig, h_orig = 200.0, 100.0, 40.0, 20.0
        cx_lb = cx_orig * r + pad_left
        cy_lb = cy_orig * r + pad_top
        w_lb, h_lb = w_orig * r, h_orig * r
        result = _make_result_in_imgsz_space(cx_lb, cy_lb, w_lb, h_lb, 0.0)
        _invert_letterbox_on_result(result, r, pad_left, pad_top)
        corners = result.obb.xyxyxyxy  # (1, 4, 2)
        expected_tl_x = cx_orig - w_orig / 2
        expected_tl_y = cy_orig - h_orig / 2
        got_tl_x = float(corners[0, 0, 0])
        got_tl_y = float(corners[0, 0, 1])
        assert (
            abs(got_tl_x - expected_tl_x) < 1e-4
        ), f"tl_x={got_tl_x} expected {expected_tl_x}"
        assert (
            abs(got_tl_y - expected_tl_y) < 1e-4
        ), f"tl_y={got_tl_y} expected {expected_tl_y}"

    def test_empty_obb_no_crash(self):
        """_invert_letterbox_on_result must not crash on an empty OBB."""
        result = MagicMock()
        result.obb = None
        r, pad_left, pad_top = self._lb_params()
        _invert_letterbox_on_result(result, r, pad_left, pad_top)  # should not raise

    def test_inference_tensor_inplace_allowed(self):
        """Regression: ultralytics predict() runs under torch.inference_mode(), so
        result.obb.data is an *inference tensor*. Mutating it in-place outside
        InferenceMode raises 'Inplace update to inference tensor ...'. The invert
        helper must handle this (it re-enters inference_mode)."""
        r, pad_left, pad_top = self._lb_params()
        # Build the backing data AS AN INFERENCE TENSOR (created inside the
        # context), exactly as ultralytics returns it.
        with torch.inference_mode():
            data = torch.tensor(
                [[320.0, 320.0, 64.0, 32.0, 0.5, 0.9, 0.0]], dtype=torch.float32
            )
        assert data.is_inference()  # precondition: reproduces the real state
        result = _FakeResult(data)
        # Must not raise, and must actually apply the inversion.
        _invert_letterbox_on_result(result, r, pad_left, pad_top)
        got_cx = float(result.obb.xywhr[0, 0])
        assert abs(got_cx - (320.0 - pad_left) / r) < 1e-4


# ---------------------------------------------------------------------------
# Test 3: _resolve_imgsz reads from model.overrides
# ---------------------------------------------------------------------------


class TestResolveImgsz:
    def test_reads_from_overrides(self):
        model = MagicMock()
        model.overrides = {"imgsz": 1280}
        assert _resolve_imgsz(model) == 1280

    def test_reads_from_model_args(self):
        model = MagicMock()
        model.overrides = {}
        model.model.args = {"imgsz": 640}
        assert _resolve_imgsz(model) == 640

    def test_falls_back_to_default(self):
        model = MagicMock()
        model.overrides = {}
        model.model.args = {}
        result = _resolve_imgsz(model)
        from hydra_suite.core.inference.stages.obb import _FALLBACK_IMGSZ

        assert result == _FALLBACK_IMGSZ

    def test_list_imgsz_in_overrides(self):
        model = MagicMock()
        model.overrides = {"imgsz": [640, 640]}
        assert _resolve_imgsz(model) == 640

    def test_reads_from_direct_executor_adapter_imgsz(self):
        """DirectExecutorAdapter (gpu_fast direct-mode) has no .overrides/
        .model.args — a plain object with just .imgsz, like the real
        adapter — must resolve from that attribute instead of silently
        falling back to _FALLBACK_IMGSZ and mismatching the TRT engine."""

        class _FakeDirectExecutorAdapter:
            imgsz = 640

        assert _resolve_imgsz(_FakeDirectExecutorAdapter()) == 640


# ---------------------------------------------------------------------------
# Test 4: _run_direct — numpy-list path passes list unchanged to model.predict
# ---------------------------------------------------------------------------


class TestRunDirectNumpyPath:
    def _make_runtime(self, tensor_on_cuda=False):
        rt = MagicMock()
        rt.tensor_on_cuda = tensor_on_cuda
        rt.device = "cpu"
        return rt

    def _make_config(self):
        cfg = MagicMock()
        cfg.direct.confidence_floor = 0.001
        cfg.target_classes = None
        cfg.raw_detection_cap = 0
        return cfg

    def test_numpy_path_passes_list_to_predict(self):
        """The numpy list must be forwarded to model.predict as-is."""
        frames = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(2)]

        # Build a fake result (empty detections)
        fake_result = MagicMock()
        fake_result.obb = None

        model = MagicMock()
        model.predict.return_value = [fake_result, fake_result]

        rt = self._make_runtime(tensor_on_cuda=False)
        cfg = self._make_config()

        _run_direct(frames, model, cfg, rt)

        # predict must have been called with the list, not a single tensor
        call_args = model.predict.call_args
        first_arg = call_args[0][0]
        assert (
            first_arg is frames
        ), "numpy list must be passed directly to model.predict"


# ---------------------------------------------------------------------------
# Test 5: _run_direct — CUDA-tensor path batches frames and inverts letterbox
# ---------------------------------------------------------------------------


class TestRunDirectCudaTensorPath:
    """Exercise the CUDA-tensor branch of _run_direct on CPU hardware.

    We monkeypatch ``torch.Tensor.is_cuda`` (via a property on the tensor's
    dtype-specific class) — which is fragile across PyTorch versions — so
    instead we patch the branch guard inside _run_direct by using a subclass
    of Tensor that overrides is_cuda.

    Alternative used here: replace the branch check with a direct patch of the
    helper ``_gpu_letterbox_batch`` so we can control what enters predict, and
    separately verify the inverse transform via TestInvertLetterbox. This keeps
    the test robust across PyTorch versions.
    """

    def _make_runtime(self, tensor_on_cuda=True):
        rt = MagicMock()
        rt.tensor_on_cuda = tensor_on_cuda
        rt.device = "cuda:0"
        return rt

    def _make_config(self, imgsz: int = 640):
        cfg = MagicMock()
        cfg.direct.confidence_floor = 0.001
        cfg.target_classes = None
        cfg.raw_detection_cap = 0
        return cfg

    def _build_fake_cuda_frames(self, n: int = 2, H: int = 200, W: int = 400):
        """Build CPU tensors that look like CUDA HWC frames for patching purposes."""
        return [torch.zeros((H, W, 3), dtype=torch.uint8) for _ in range(n)]

    def test_predict_called_with_single_batched_tensor(self):
        """When frames are CUDA tensors, predict must receive a single (B,3,imgsz,imgsz) tensor."""
        H, W, imgsz, B = 200, 400, 640, 2
        frames = self._build_fake_cuda_frames(n=B, H=H, W=W)

        # Fake Results with non-empty OBB
        cx_lb, cy_lb = 320.0, 320.0
        w_lb, h_lb, angle = 64.0, 32.0, 0.5
        fake_results = [
            _make_result_in_imgsz_space(cx_lb, cy_lb, w_lb, h_lb, angle)
            for _ in range(B)
        ]

        model = MagicMock()
        model.overrides = {"imgsz": imgsz}
        model.predict.return_value = fake_results
        rt = self._make_runtime(tensor_on_cuda=True)
        cfg = self._make_config(imgsz=imgsz)

        # Patch is_cuda to True on these CPU tensors
        with patch.object(
            type(frames[0]),
            "is_cuda",
            new_callable=lambda: property(lambda self: True),
        ):
            _run_direct(frames, model, cfg, rt)

        call_args = model.predict.call_args
        first_arg = call_args[0][0]
        assert isinstance(
            first_arg, torch.Tensor
        ), "predict must be called with a single Tensor, not a list"
        assert first_arg.shape == (
            B,
            3,
            imgsz,
            imgsz,
        ), f"batched tensor shape {first_arg.shape} != ({B}, 3, {imgsz}, {imgsz})"

    def test_predict_called_with_float32_tensor(self):
        """The batched tensor must be float32 (no fp16)."""
        H, W, imgsz, B = 200, 400, 640, 1
        frames = self._build_fake_cuda_frames(n=B, H=H, W=W)
        cx_lb, cy_lb = 320.0, 320.0
        fake_results = [_make_result_in_imgsz_space(cx_lb, cy_lb, 64.0, 32.0, 0.0)]
        model = MagicMock()
        model.overrides = {"imgsz": imgsz}
        model.predict.return_value = fake_results
        rt = self._make_runtime(tensor_on_cuda=True)
        cfg = self._make_config()

        with patch.object(
            type(frames[0]),
            "is_cuda",
            new_callable=lambda: property(lambda self: True),
        ):
            _run_direct(frames, model, cfg, rt)

        call_args = model.predict.call_args
        batched = call_args[0][0]
        assert batched.dtype == torch.float32, f"expected float32, got {batched.dtype}"

    def test_inverse_letterbox_applied_before_extract(self):
        """After _run_direct, the returned tensors must be in original-frame coords."""
        H, W, imgsz = 200, 400, 640
        # Compute expected params
        r = min(imgsz / H, imgsz / W)  # 1.6
        new_h = int(H * r)
        new_w = int(W * r)
        pad_top = (imgsz - new_h) // 2  # 160
        pad_left = (imgsz - new_w) // 2  # 0

        cx_orig, cy_orig = 200.0, 100.0
        w_orig, h_orig = 40.0, 20.0
        angle = math.pi / 6

        # Build result as if model returned letterbox-space coords
        cx_lb = cx_orig * r + pad_left
        cy_lb = cy_orig * r + pad_top
        w_lb, h_lb = w_orig * r, h_orig * r
        fake_results = [_make_result_in_imgsz_space(cx_lb, cy_lb, w_lb, h_lb, angle)]

        frames = [torch.zeros((H, W, 3), dtype=torch.uint8)]
        model = MagicMock()
        model.overrides = {"imgsz": imgsz}
        model.predict.return_value = fake_results
        rt = self._make_runtime(tensor_on_cuda=True)
        cfg = self._make_config()

        with patch.object(
            type(frames[0]),
            "is_cuda",
            new_callable=lambda: property(lambda self: True),
        ):
            out = _run_direct(frames, model, cfg, rt)

        # tensor_on_cuda=True → returns _RawOBBTensors
        raw = out[0]
        got_cx = float(raw.xywhr[0, 0])
        got_cy = float(raw.xywhr[0, 1])
        got_w = float(raw.xywhr[0, 2])
        got_h = float(raw.xywhr[0, 3])
        got_angle = float(raw.xywhr[0, 4])

        assert abs(got_cx - cx_orig) < 1e-3, f"cx={got_cx}, expected {cx_orig}"
        assert abs(got_cy - cy_orig) < 1e-3, f"cy={got_cy}, expected {cy_orig}"
        assert abs(got_w - w_orig) < 1e-3, f"w={got_w}, expected {w_orig}"
        assert abs(got_h - h_orig) < 1e-3, f"h={got_h}, expected {h_orig}"
        assert abs(got_angle - angle) < 1e-6, f"angle={got_angle}, expected {angle}"
