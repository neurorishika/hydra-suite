"""Unit tests for the unified torch canonical-crop seam (resample.py).

The crop path is unified on ``canonical_warp``/``canonical_warp_batch``/
``letterbox_fit`` in ``core/canonicalization/resample.py`` -- there is no
separate ``extract_classifier_crops_gpu``/``apply_fit_gpu`` code path any
more. These tests prove:

1. CPU-torch and device-torch (MPS here; CUDA on mehek) agree to a tight
   tolerance running the SAME ``canonical_warp``/``letterbox_fit`` calls.
2. A landmark placed at a known frame location lands within ~0.5px of the
   analytic canonical location (``m_align @ point``) on BOTH devices -- using
   a NON-square canvas, so an (H, W)/(W, H) axis swap or anisotropic scale
   error would be caught (a square canvas can't distinguish these).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from hydra_suite.core.canonicalization.geometry import (
    CanonicalGeometry,
    canonical_affine,
)
from hydra_suite.core.canonicalization.resample import (
    canonical_warp,
    canonical_warp_batch,
    letterbox_fit,
)
from hydra_suite.core.inference.result import OBBResult

# Non-square canvas: from_reference(60, aspect_ratio=2.0, margin=1.3) yields
# canvas_w != canvas_h (long edge holds margin * major axis; short edge is
# canvas_w / aspect_ratio) -- this is the geometry a square (128, 128) test
# cannot exercise.
_GEOM = CanonicalGeometry.from_reference(60.0, 2.0, 1.3)

_MPS_AVAILABLE = torch.backends.mps.is_available()


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


def _single_obb_corners(cx, cy, major, minor, angle_rad):
    """Build one rotated-rectangle OBB's 4x2 corners around (cx, cy)."""
    import math

    hw, hh = major / 2.0, minor / 2.0
    local = np.array([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]], dtype=np.float64)
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    rot = np.array([[c, -s], [s, c]])
    world = local @ rot.T + np.array([cx, cy])
    return world.astype(np.float32)


assert _GEOM.canvas_w != _GEOM.canvas_h, "test geometry must be non-square"


def test_geometry_is_non_square():
    assert _GEOM.canvas_wh[0] != _GEOM.canvas_wh[1]


def test_canonical_warp_shape_and_dtype():
    frame = torch.rand(3, 200, 300, dtype=torch.float32)
    obb = _toy_obb(1)
    m_align, _theta, _clipped = canonical_affine(obb.corners[0], _GEOM)
    crop = canonical_warp(frame, m_align, _GEOM)
    assert crop.shape == (3, _GEOM.canvas_h, _GEOM.canvas_w)
    assert crop.dtype == torch.float32


def test_canonical_warp_batch_empty():
    frame = torch.zeros(3, 50, 50)
    crops = canonical_warp_batch(frame, [], _GEOM)
    assert crops.shape == (0, 3, _GEOM.canvas_h, _GEOM.canvas_w)


def test_canonical_warp_batch_shape():
    frame = torch.rand(3, 200, 300, dtype=torch.float32)
    obb = _toy_obb(3)
    m_aligns = [canonical_affine(c, _GEOM)[0] for c in obb.corners]
    crops = canonical_warp_batch(frame, m_aligns, _GEOM)
    assert crops.shape == (3, 3, _GEOM.canvas_h, _GEOM.canvas_w)


# ---- CPU vs device pixel-parity (Step 1) ------------------------------------


@pytest.mark.parametrize("device", ["cpu"] + (["mps"] if _MPS_AVAILABLE else []))
def test_cpu_vs_device_crop_close(device):
    """Same canonical_warp on cpu vs an accelerator device must agree tightly."""
    rng = np.random.default_rng(0)
    frame_np = rng.integers(0, 256, (200, 300, 3), np.uint8)
    frame_cpu = torch.from_numpy(frame_np.transpose(2, 0, 1)).float().div(255.0)

    obb = _toy_obb(3)
    m_aligns = [canonical_affine(c, _GEOM)[0] for c in obb.corners]

    crop_cpu = canonical_warp_batch(frame_cpu, m_aligns, _GEOM)

    frame_dev = frame_cpu.to(device)
    crop_dev = canonical_warp_batch(frame_dev, m_aligns, _GEOM).to("cpu")

    assert crop_dev.shape == crop_cpu.shape
    max_abs_diff = float((crop_cpu - crop_dev).abs().max())
    assert (
        max_abs_diff < 2e-2
    ), f"cpu vs {device} crop diverged: max|diff|={max_abs_diff}"


@pytest.mark.skipif(not _MPS_AVAILABLE, reason="MPS not available on this box")
def test_letterbox_fit_cpu_vs_mps_close():
    """letterbox_fit (Layer 2) must also agree tightly across devices."""
    rng = np.random.default_rng(1)
    crop_cpu = (
        torch.from_numpy(
            rng.integers(0, 256, (2, 3, _GEOM.canvas_h, _GEOM.canvas_w), np.uint8)
        )
        .float()
        .div(255.0)
    )

    model_wh = (96, 64)  # deliberately non-square model input too
    fitted_cpu = letterbox_fit(crop_cpu, model_wh)
    fitted_mps = letterbox_fit(crop_cpu.to("mps"), model_wh).to("cpu")

    assert fitted_cpu.shape == fitted_mps.shape
    max_abs_diff = float((fitted_cpu - fitted_mps).abs().max())
    assert max_abs_diff < 2e-2


# ---- Landmark (anisotropy / axis-swap) check --------------------------------


def _weighted_centroid(channel_hw: np.ndarray) -> tuple[float, float]:
    """Intensity-weighted centroid (x, y) of a single-channel 2D array."""
    ys, xs = np.indices(channel_hw.shape)
    total = float(channel_hw.sum())
    assert total > 0, "landmark not found in warped crop"
    cx = float((xs * channel_hw).sum() / total)
    cy = float((ys * channel_hw).sum() / total)
    return cx, cy


@pytest.mark.parametrize("device", ["cpu"] + (["mps"] if _MPS_AVAILABLE else []))
def test_landmark_lands_at_analytic_canonical_location(device):
    """A bright dot at a known frame location must land at ``m_align @ point``.

    Non-square canvas + non-square, rotated, off-centre OBB: this is the
    minimal fixture that fails if the canvas (H, W) vs (W, H) axes are ever
    swapped, or if the two axes get resampled with different effective scale
    (anisotropy) -- a square canvas cannot distinguish either failure mode.
    """
    h, w = 240, 360
    frame_np = np.zeros((h, w, 3), np.float32)

    # A rotated, off-centre, non-square OBB (major != minor breaks symmetry).
    cx, cy, major, minor, angle = 150.0, 90.0, 80.0, 40.0, 0.4
    corners = _single_obb_corners(cx, cy, major, minor, angle)

    # Bright landmark dot inside the OBB's footprint, offset from the centroid
    # (not at the geometric centre -- that would also pass for a mirrored crop).
    landmark_xy = (cx + 15.0, cy - 8.0)
    lx, ly = int(round(landmark_xy[0])), int(round(landmark_xy[1]))
    # 3x3 bright patch (sub-pixel centroid, robust to a 1px quantization jitter).
    frame_np[ly - 1 : ly + 2, lx - 1 : lx + 2, :] = 1.0

    frame_cpu = torch.from_numpy(frame_np.transpose(2, 0, 1)).contiguous()

    m_align, _theta, clipped = canonical_affine(corners, _GEOM)
    assert not clipped

    frame_dev = frame_cpu.to(device)
    crop = canonical_warp(frame_dev, m_align, _GEOM).to("cpu").numpy()  # (3, H, W)

    got_x, got_y = _weighted_centroid(crop[0])

    analytic = m_align @ np.array([landmark_xy[0], landmark_xy[1], 1.0])
    exp_x, exp_y = float(analytic[0]), float(analytic[1])

    dist = float(np.hypot(got_x - exp_x, got_y - exp_y))
    assert dist < 0.5, (
        f"landmark centroid ({got_x:.3f}, {got_y:.3f}) on {device} strayed "
        f"{dist:.3f}px from the analytic canonical location ({exp_x:.3f}, {exp_y:.3f})"
    )


# ---- NVDEC HWC-layout regression (single-frame path) ------------------------


def test_extract_canonical_crops_hwc_nvdec_layout():
    """NvdecFrameReader yields (H, W, 3) HWC uint8 (RGB); the extractor must
    permute to CHW and produce 3-channel crops (regression for the shape bug
    that crashed the first real NVDEC run: 'tensor a (4512) must match b (3)')."""
    from hydra_suite.core.inference.stages.crops import extract_canonical_crops

    hwc = torch.randint(0, 256, (200, 300, 3), dtype=torch.uint8)  # (H, W, 3)
    obb = _toy_obb(3)
    runtime = type("RT", (), {"cuda_mode": False, "device": "cpu"})()
    crops = extract_canonical_crops(hwc, obb, _GEOM, runtime)
    assert crops.shape == (3, 3, _GEOM.canvas_h, _GEOM.canvas_w)
    assert crops.dtype == torch.float32


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


def _supports_helper(active_backend):
    from hydra_suite.core.identity.classification import backend as bk

    class _F:
        _active_execution_backend = active_backend

        def predict_batch_cuda(self, crops, input_is_bgr=True):
            return []

    b = bk.ClassifierBackend.__new__(bk.ClassifierBackend)
    b._model = [_F(), _F()]
    b._uses_factor_backends = lambda: True  # type: ignore[method-assign]
    b._ensure_loaded = lambda: None  # type: ignore[method-assign]
    return b


def test_supports_cuda_forward_bundle():
    b_native = _supports_helper("native")
    b_coreml = _supports_helper("coreml")
    assert b_native.supports_cuda_forward() is True
    assert b_coreml.supports_cuda_forward() is False


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
    rt = type("RT", (), {"cuda_mode": True})()
    cfg = type(
        "C", (), {"model_path": str(tmp_path / "m.multihead.json"), "label": "x"}
    )()
    with pytest.raises(RuntimeError, match="CUDA-native"):
        cnn_stage.load_cnn_model(cfg, rt)


def test_load_cnn_no_raise_when_not_cuda_mode(monkeypatch, tmp_path):
    # On MPS/CPU (cuda_mode False), a non-CUDA classifier loads fine.
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
    rt = type("RT", (), {"cuda_mode": False})()
    cfg = type("C", (), {"model_path": str(tmp_path / "m.json"), "label": "x"})()
    model = cnn_stage.load_cnn_model(cfg, rt)  # must NOT raise
    assert model.input_size == (128, 128)


# ---- Task 5: stage routing (GPU path only when frames are actually on CUDA) --


def test_frames_on_cuda_gate():
    from hydra_suite.core.inference.stages.crops import frames_on_cuda

    # requested_gpu gates the path (True on gpu AND gpu_fast); tensor_on_cuda is
    # irrelevant here (it is False on gpu_fast, where NVDEC frames still belong
    # on the GPU crop path).
    rt_gpu = type("RT", (), {"requested_gpu": True})()
    rt_cpu = type("RT", (), {"requested_gpu": False})()
    cpu_tensor = torch.zeros((3, 8, 8))
    np_frame = np.zeros((8, 8, 3), np.uint8)

    # Not a gpu tier -> never.
    assert frames_on_cuda(rt_cpu, [cpu_tensor]) is False
    # gpu tier but the frame is CPU (NVDEC fell back to CpuFrameReader) -> False:
    # uploading a CPU frame to GPU just to crop is slower than cv2.
    assert frames_on_cuda(rt_gpu, [cpu_tensor]) is False
    assert frames_on_cuda(rt_gpu, [np_frame]) is False
    assert frames_on_cuda(rt_gpu, []) is False
    if torch.cuda.is_available():
        assert frames_on_cuda(rt_gpu, [cpu_tensor.cuda()]) is True


def test_run_cnn_batch_routes_by_frame_device(monkeypatch):
    """GPU path uses extract_canonical_crops_batch + letterbox_fit + predict_batch_cuda;
    CPU path uses extract_classifier_crops_batch_np + apply_fit_batch + predict_batch
    (the unified-seam successors of the deleted *_gpu crop helpers)."""
    from hydra_suite.core.inference.result import CropBatch, NumpyCropBatch
    from hydra_suite.core.inference.stages import cnn as cnn_stage
    from hydra_suite.core.inference.stages import crops as crops_mod

    used = {"gpu": False, "cpu": False, "cuda_fwd": False, "numpy_fwd": False}

    def _fake_canonical_batch(*a, **k):
        used["gpu"] = True
        return CropBatch(
            crops=torch.zeros((1, 3, 8, 8)),
            detection_ids=np.array([0]),
            frame_index=np.array([0]),
            obb_by_frame={0: _toy_obb(1)},
            native_sizes=np.array([[8, 8]]),
        )

    def _fake_np_batch(*a, **k):
        used["cpu"] = True
        return NumpyCropBatch(
            crops=[np.zeros((8, 8, 3), np.uint8)],
            detection_ids=np.array([0]),
            frame_index=np.array([0]),
            obb_by_frame={0: _toy_obb(1)},
            native_sizes=np.array([[8, 8]]),
        )

    monkeypatch.setattr(
        crops_mod, "extract_canonical_crops_batch", _fake_canonical_batch
    )
    monkeypatch.setattr(crops_mod, "extract_classifier_crops_batch_np", _fake_np_batch)

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
    rt = type("RT", (), {"tensor_on_cuda": True, "device": "cpu", "cuda_mode": False})()
    geometry = CanonicalGeometry.from_reference(20.0, 2.0, 1.3)

    # Gate True -> GPU path.
    monkeypatch.setattr(crops_mod, "frames_on_cuda", lambda r, f: True)
    cnn_stage.run_cnn_batch([None], [_toy_obb(1)], model, cfg, rt, geometry)
    assert used["gpu"] and used["cuda_fwd"]
    assert not used["cpu"] and not used["numpy_fwd"]

    # Gate False (e.g. NVDEC fell back to CPU frames) -> CPU path.
    for k in used:
        used[k] = False
    monkeypatch.setattr(crops_mod, "frames_on_cuda", lambda r, f: False)
    cnn_stage.run_cnn_batch([None], [_toy_obb(1)], model, cfg, rt, geometry)
    assert used["cpu"] and used["numpy_fwd"]
    assert not used["gpu"] and not used["cuda_fwd"]


def test_predict_batch_cuda_fallback_forwards_input_is_bgr(monkeypatch):
    """When a factor lacks CUDA forward, the numpy fallback must NOT re-flip RGB."""
    from hydra_suite.core.identity.classification import backend as bk

    be = bk.ClassifierBackend.__new__(bk.ClassifierBackend)
    be._model_path = "x.multihead.json"
    seen = {}

    monkeypatch.setattr(be, "_ensure_loaded", lambda: None)
    monkeypatch.setattr(be, "_uses_factor_backends", lambda: True)
    monkeypatch.setattr(be, "supports_cuda_forward", lambda: False)  # force fallback

    def _fake_predict_batch(crops, input_is_bgr=True):
        seen["input_is_bgr"] = input_is_bgr
        return [[np.array([1.0], np.float32)]]

    monkeypatch.setattr(be, "predict_batch", _fake_predict_batch)

    crop = torch.zeros((3, 4, 4))
    be.predict_batch_cuda([crop], input_is_bgr=False)
    assert seen["input_is_bgr"] is False  # RGB stays RGB through the fallback
