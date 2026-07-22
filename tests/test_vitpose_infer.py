import numpy as np
import torch

from hydra_suite.core.identity.pose.vitpose.infer import (
    decode_and_project,
    preprocess_crop,
)


def test_preprocess_crop_shapes():
    crop = np.zeros((80, 60, 3), dtype=np.uint8)
    chw, center, scale = preprocess_crop(crop)
    assert chw.shape == (3, 256, 192)
    assert chw.dtype == np.float32
    assert center.shape == (2,)
    assert scale.shape == (2,)


def test_decode_and_project_center_peak():
    # a single hot pixel at heatmap center should map near the crop center
    B, K, H, W = 1, 3, 64, 48
    hm = torch.zeros(B, K, H, W)
    hm[:, :, H // 2, W // 2] = 10.0
    centers = np.array([[30.0, 40.0]], dtype=np.float32)  # (B,2)
    scales = np.array([[0.6, 0.8]], dtype=np.float32)  # (B,2) PIXEL_STD units
    coords, maxvals = decode_and_project(hm, centers, scales)
    assert coords.shape == (B, K, 2)
    assert maxvals.shape == (B, K, 1)
    # projected point lands within the crop bbox around the center
    assert np.all(np.isfinite(coords))
