from __future__ import annotations

import numpy as np


def generate_udp_gaussian(
    joints_hm: np.ndarray,
    vis: np.ndarray,
    heatmap_size_wh: tuple[int, int],
    sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    """UDP GaussianHeatmap target: full-map Gaussian at the subpixel joint
    center (mmpose encoding='UDP', target_type='GaussianHeatmap').

    joints_hm: (K, 2) keypoint coords already in heatmap pixel space.
    vis: (K,) visibility (>0 => labelled).
    Returns target (K, H, W) float32 and target_weight (K, 1) float32.
    """
    w, h = heatmap_size_wh
    k = joints_hm.shape[0]
    target = np.zeros((k, h, w), dtype=np.float32)
    weight = (np.asarray(vis).reshape(k) > 0).astype(np.float32).reshape(k, 1)
    xs = np.arange(w, dtype=np.float32)[None, :]  # (1, W)
    ys = np.arange(h, dtype=np.float32)[:, None]  # (H, 1)
    two_s2 = 2.0 * sigma * sigma
    for j in range(k):
        if weight[j, 0] == 0.0:
            continue
        mu_x, mu_y = float(joints_hm[j, 0]), float(joints_hm[j, 1])
        target[j] = np.exp(-(((xs - mu_x) ** 2) + ((ys - mu_y) ** 2)) / two_s2)
    return target, weight
