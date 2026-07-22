"""Unit tests for the GPU-native classifier crop + forward path.

These run on the CPU torch device (grid_sample works there), so most of the
feature's correctness is provable off the CUDA box; only the end-to-end
determinism/agreement/perf gate needs mehek (see the implementation plan Task 6).
"""

import numpy as np
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
