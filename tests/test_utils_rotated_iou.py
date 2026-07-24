import cv2
import numpy as np
import torch

from hydra_suite.utils.rotated_iou import pairwise_obb_overlap


def _cv2_overlap(ca, cb, metric):
    pa = cv2.convexHull(ca.astype(np.float32)).reshape(-1, 2)
    pb = cv2.convexHull(cb.astype(np.float32)).reshape(-1, 2)
    aa = abs(cv2.contourArea(pa))
    ab = abs(cv2.contourArea(pb))
    inter, _ = cv2.intersectConvexConvex(pa, pb)
    inter = max(0.0, inter)
    denom = min(aa, ab) if metric == "ios" else aa + ab - inter
    return inter / denom if denom > 1e-9 else 0.0


def _rect(cx, cy, w, h, deg):
    box = cv2.boxPoints(((cx, cy), (w, h), deg))
    return box.astype(np.float32)


def test_matches_cv2_within_tolerance_axis_aligned():
    corners = np.stack([_rect(100, 100, 40, 40, 0), _rect(120, 100, 40, 40, 0)])
    t = torch.from_numpy(corners)
    for metric in ("iou", "ios"):
        m = pairwise_obb_overlap(t, metric=metric).numpy()
        expected = _cv2_overlap(corners[0], corners[1], metric)
        assert abs(m[0, 1] - expected) < 1e-2
        assert abs(m[1, 0] - expected) < 1e-2


def test_matches_cv2_rotated_random():
    rng = np.random.default_rng(0)
    corners = np.stack(
        [
            _rect(*rng.uniform([80, 80, 30, 30, 0], [140, 140, 60, 60, 90]))
            for _ in range(6)
        ]
    ).astype(np.float32)
    m = pairwise_obb_overlap(torch.from_numpy(corners), metric="iou").numpy()
    for i in range(6):
        for j in range(6):
            if i == j:
                continue
            assert abs(m[i, j] - _cv2_overlap(corners[i], corners[j], "iou")) < 3e-2


def test_diagonal_is_one_and_empty_ok():
    corners = np.stack([_rect(100, 100, 40, 40, 15)])
    m = pairwise_obb_overlap(torch.from_numpy(corners), metric="iou")
    assert abs(float(m[0, 0]) - 1.0) < 1e-3
    empty = pairwise_obb_overlap(torch.zeros((0, 4, 2)), metric="iou")
    assert empty.shape == (0, 0)


def test_degenerate_box_yields_zero_not_nan():
    # A zero-area (collinear) "box" paired with a real box must yield 0.0
    # overlap, not NaN -- a NaN would silently poison downstream merge
    # decisions in the SAHI band-overlap gate.
    degenerate = np.array(
        [[0.0, 0.0], [0.0, 0.0], [10.0, 0.0], [10.0, 0.0]], dtype=np.float32
    )
    real = _rect(5, 0, 20, 20, 0)
    corners = np.stack([degenerate, real])
    m = pairwise_obb_overlap(torch.from_numpy(corners), metric="iou")
    assert torch.isfinite(m).all()
    assert abs(float(m[0, 1])) < 1e-6
    assert abs(float(m[1, 0])) < 1e-6
    # Diagonal is always defined as 1 by convention, even for a degenerate box.
    assert abs(float(m[0, 0]) - 1.0) < 1e-3


def test_symmetric():
    rng = np.random.default_rng(1)
    corners = np.stack(
        [
            _rect(*rng.uniform([80, 80, 30, 30, 0], [140, 140, 60, 60, 90]))
            for _ in range(5)
        ]
    ).astype(np.float32)
    m = pairwise_obb_overlap(torch.from_numpy(corners), metric="iou").numpy()
    assert np.allclose(m, m.T, atol=1e-5)


def test_preserves_device_and_dtype():
    corners = np.stack([_rect(100, 100, 40, 40, 0), _rect(120, 100, 40, 40, 0)]).astype(
        np.float32
    )
    t = torch.from_numpy(corners)
    m = pairwise_obb_overlap(t, metric="iou")
    assert m.device == t.device
    assert m.dtype == t.dtype
