import numpy as np
import pytest
import torch

from hydra_suite.core.inference.result import OBBResult
from hydra_suite.core.inference.runtime import RuntimeContext
from hydra_suite.core.inference.stages.crops import (
    extract_aabb_crops,
    extract_canonical_crops,
)


def _cpu_rt() -> RuntimeContext:
    return RuntimeContext(
        cuda_mode=False,
        device="cpu",
        use_nvdec=False,
        default_runtime="cpu",
        tensor_on_cuda=False,
    )


def _onnx_cuda_rt() -> RuntimeContext:
    """ONNX CUDA: cuda_mode=True but tensor_on_cuda=False — must use CPU crop path."""
    return RuntimeContext(
        cuda_mode=True,
        device="cuda:0",
        use_nvdec=False,
        default_runtime="cuda",
        tensor_on_cuda=False,
        requested_gpu=True,
    )


def _obb_result(n: int = 2, frame_idx: int = 0) -> OBBResult:
    base = [[320.0, 240.0], [100.0, 100.0], [200.0, 300.0], [400.0, 150.0]]
    centroids = np.array(
        [base[i % len(base)] for i in range(n)], dtype=np.float32
    ).reshape(n, 2)
    corners = np.zeros((n, 4, 2), dtype=np.float32)
    for i in range(n):
        cx, cy = centroids[i]
        corners[i] = [
            [cx - 20, cy - 10],
            [cx + 20, cy - 10],
            [cx + 20, cy + 10],
            [cx - 20, cy + 10],
        ]
    return OBBResult(
        frame_idx=frame_idx,
        centroids=centroids,
        angles=np.zeros(n, dtype=np.float32),
        sizes=np.full(n, 400.0, dtype=np.float32),
        shapes=np.ones((n, 2), dtype=np.float32),
        confidences=np.ones(n, dtype=np.float32),
        corners=corners,
        detection_ids=OBBResult.make_detection_ids(frame_idx, n),
    )


def test_extract_canonical_crops_returns_tensor():
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    obb = _obb_result(n=2)
    crops = extract_canonical_crops(
        frame, obb, canonical_aspect_ratio=2.0, canonical_margin=1.3, runtime=_cpu_rt()
    )
    assert isinstance(crops, torch.Tensor)
    assert crops.shape[0] == 2
    assert crops.ndim == 4  # (N, C, H, W)


def test_extract_canonical_crops_empty_obb():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    obb = _obb_result(n=0)
    crops = extract_canonical_crops(
        frame, obb, canonical_aspect_ratio=2.0, canonical_margin=1.3, runtime=_cpu_rt()
    )
    assert crops.shape[0] == 0


def test_extract_aabb_crops_returns_list():
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    obb = _obb_result(n=2)
    crops = extract_aabb_crops(frame, obb, padding=0.1)
    assert len(crops) == 2
    for crop in crops:
        assert isinstance(crop, np.ndarray)
        assert crop.ndim == 3


def test_extract_aabb_crops_empty_obb():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    obb = _obb_result(n=0)
    crops = extract_aabb_crops(frame, obb, padding=0.1)
    assert len(crops) == 0


def test_canonical_and_aabb_same_count():
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    obb = _obb_result(n=3)
    canonical = extract_canonical_crops(frame, obb, 2.0, 1.3, _cpu_rt())
    aabb = extract_aabb_crops(frame, obb, padding=0.1)
    assert canonical.shape[0] == len(aabb) == 3


def test_onnx_cuda_uses_cpu_path(monkeypatch):
    """Per Correction 3 fix 3: tensor_on_cuda=False must NOT trigger _extract_canonical_gpu
    even when cuda_mode=True. ONNX-CUDA returns CPU numpy from the downstream model, so
    GPU crop extraction would be wasted upload+download."""
    gpu_called = []
    cpu_called = []

    def _fake_gpu(*a, **kw):
        gpu_called.append(True)
        return torch.zeros((1, 3, 4, 4))

    def _fake_cpu(*a, **kw):
        cpu_called.append(True)
        return torch.zeros((1, 3, 4, 4))

    import hydra_suite.core.inference.stages.crops as mod

    monkeypatch.setattr(mod, "_extract_canonical_gpu", _fake_gpu)
    monkeypatch.setattr(mod, "_extract_canonical_cpu", _fake_cpu)

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    obb = _obb_result(n=1)
    extract_canonical_crops(frame, obb, 2.0, 1.3, _onnx_cuda_rt())

    assert cpu_called == [True]
    assert gpu_called == []


def test_canonical_crops_dtype_normalized():
    """CPU crops are normalised to [0, 1] float32."""
    frame = np.full((480, 640, 3), 255, dtype=np.uint8)  # white frame
    obb = _obb_result(n=1)
    crops = extract_canonical_crops(frame, obb, 2.0, 1.3, _cpu_rt())
    assert crops.dtype == torch.float32
    # White pixels normalise to ~1.0
    assert crops.max().item() == pytest.approx(1.0, abs=1e-3)


def _overlapping_obb_result() -> OBBResult:
    """Two close, overlapping OBBs so detection 1's polygon lands inside
    detection 0's canonical crop bounds — the scenario the realtime path's
    missing foreign-masking gap manifested in (see 2026-07-10 equivalence
    investigation: legacy always masks foreign OBBs via suppress_foreign_obb,
    but extract_canonical_crops previously had no masking support at all)."""
    centroids = np.array([[100.0, 100.0], [110.0, 100.0]], dtype=np.float32)
    corners = np.zeros((2, 4, 2), dtype=np.float32)
    for i in range(2):
        cx, cy = centroids[i]
        corners[i] = [
            [cx - 20, cy - 10],
            [cx + 20, cy - 10],
            [cx + 20, cy + 10],
            [cx - 20, cy + 10],
        ]
    return OBBResult(
        frame_idx=0,
        centroids=centroids,
        angles=np.zeros(2, dtype=np.float32),
        sizes=np.full(2, 400.0, dtype=np.float32),
        shapes=np.ones((2, 2), dtype=np.float32),
        confidences=np.ones(2, dtype=np.float32),
        corners=corners,
        detection_ids=OBBResult.make_detection_ids(0, 2),
    )


def test_extract_canonical_crops_defaults_to_no_masking():
    """suppress_foreign defaults to False: unchanged behavior for existing callers."""
    frame = np.full((480, 640, 3), 255, dtype=np.uint8)  # white frame
    obb = _overlapping_obb_result()
    crops = extract_canonical_crops(frame, obb, 2.0, 1.3, _cpu_rt())
    # No masking requested -> no background-color pixels introduced; still ~white.
    assert crops.min().item() == pytest.approx(1.0, abs=1e-3)


def test_extract_canonical_crops_suppress_foreign_masks_overlapping_detection():
    """suppress_foreign=True must black out a neighboring detection's OBB —
    the realtime pose-crop path's parity gap with legacy's suppress_foreign_obb
    (always-on, no realtime/batch split there)."""
    frame = np.full((480, 640, 3), 255, dtype=np.uint8)  # white frame
    obb = _overlapping_obb_result()
    crops = extract_canonical_crops(
        frame, obb, 2.0, 1.3, _cpu_rt(), suppress_foreign=True
    )
    # With the neighbor's OBB masked to black, at least one crop must contain
    # background-color (0) pixels that a fully-white unmasked frame wouldn't produce.
    assert crops.min().item() == pytest.approx(0.0, abs=1e-3)


def test_extract_canonical_crops_suppress_foreign_noop_for_single_detection():
    """suppress_foreign=True with only one detection must not crash or alter output
    (mirrors extract_canonical_crops_batch's num_detections > 1 gate)."""
    frame = np.full((480, 640, 3), 255, dtype=np.uint8)
    obb = _obb_result(n=1)
    crops = extract_canonical_crops(
        frame, obb, 2.0, 1.3, _cpu_rt(), suppress_foreign=True
    )
    assert crops.min().item() == pytest.approx(1.0, abs=1e-3)
