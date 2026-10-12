"""I3: the foreign-mask AABB culls are exact (byte-identical masks).

``_apply_foreign_mask_canonical`` skips foreign polygons whose integer AABB
misses the canvas and skips the own-mask raster when a foreign AABB misses
the own polygon's. ``_reference`` below is the pre-cull implementation,
verbatim; randomized dense layouts (overlaps, off-canvas, sub-pixel
negative coordinates) must produce identical crops through both, for the
helper and for the batch crop path.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np
import pytest
import torch

from hydra_suite.core.canonicalization import crop as crop_mod
from hydra_suite.core.canonicalization.geometry import (
    CanonicalGeometry,
    canonical_affine,
)
from hydra_suite.core.inference.result import OBBResult
from hydra_suite.core.inference.stages import crops as crops_mod


def _reference(
    crop: np.ndarray,
    M_align: np.ndarray,
    foreign_corners_list: List[np.ndarray],
    bg_color: Tuple[int, int, int],
    own_corners: Optional[np.ndarray] = None,
) -> None:
    M = np.asarray(M_align, dtype=np.float64)
    R = M[:, :2]
    t = M[:, 2:]

    def _to_canonical(corners: np.ndarray) -> np.ndarray:
        pts = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
        return (R @ pts.T + t).T

    own_poly = None
    if own_corners is not None:
        own_poly = _to_canonical(own_corners).astype(np.int32).reshape(-1, 1, 2)

    h, w = crop.shape[:2]
    for corners in foreign_corners_list:
        poly = _to_canonical(corners).astype(np.int32).reshape(-1, 1, 2)
        if own_poly is None:
            cv2.fillPoly(crop, [poly], bg_color)
            continue
        foreign_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(foreign_mask, [poly], 1)
        own_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(own_mask, [own_poly], 1)
        mask = (foreign_mask & ~own_mask).astype(bool)
        if mask.any():
            crop[mask] = bg_color


_GEOM = CanonicalGeometry(canvas_wh=(96, 48), margin=1.3, aspect_ratio=2.0)


def _random_obbs(rng, m, side):
    cx = rng.uniform(0, side, m)
    cy = rng.uniform(0, side, m)
    w = rng.uniform(6, 60, m)
    h = rng.uniform(4, 30, m)
    a = rng.uniform(-np.pi, np.pi, m)
    corners = []
    for i in range(m):
        center = (float(cx[i]), float(cy[i]))
        size = (float(w[i]), float(h[i]))
        corners.append(cv2.boxPoints((center, size, float(np.degrees(a[i])))))
    return np.asarray(corners, np.float32)


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("with_own", [True, False])
def test_helper_matches_reference(seed, with_own):
    rng = np.random.default_rng(seed)
    side = float(rng.choice([150.0, 400.0, 1200.0]))  # dense .. sparse
    corners = _random_obbs(rng, int(rng.integers(2, 120)), side)
    for own in range(0, len(corners), max(1, len(corners) // 6)):
        m_align, _, _ = canonical_affine(corners[own], _GEOM)
        # Sub-pixel jitter puts vertices near the int-truncation boundaries.
        m_align = m_align + np.array([[0, 0, rng.uniform(-1, 1)]] * 2)
        foreign = np.delete(corners, own, axis=0)
        own_corners = corners[own] if with_own else None
        base = rng.integers(1, 255, (48, 96, 3), dtype=np.uint8)
        got, want = base.copy(), base.copy()
        crop_mod._apply_foreign_mask_canonical(
            got, m_align, foreign, (0, 0, 0), own_corners=own_corners
        )
        _reference(want, m_align, list(foreign), (0, 0, 0), own_corners=own_corners)
        np.testing.assert_array_equal(got, want)


def test_cull_keeps_subpixel_negative_vertices():
    """Canonical x in (-1, 0) truncates to column 0: such a polygon draws."""
    crop = np.full((10, 10, 3), 200, np.uint8)
    want = crop.copy()
    m = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    poly = np.array([[-0.5, 2.0], [-0.2, 2.0], [-0.2, 6.0], [-0.5, 6.0]])
    crop_mod._apply_foreign_mask_canonical(crop, m, [poly], (0, 0, 0))
    _reference(want, m, [poly], (0, 0, 0))
    np.testing.assert_array_equal(crop, want)
    assert (crop[2:7, 0] == 0).all()


@pytest.mark.parametrize("seed", range(4))
def test_batch_crop_path_matches_reference(monkeypatch, seed):
    rng = np.random.default_rng(100 + seed)
    corners = _random_obbs(rng, 64, 300.0)
    n = len(corners)
    obb = OBBResult(
        frame_idx=0,
        centroids=corners.mean(axis=1),
        angles=np.zeros(n, np.float32),
        sizes=np.full(n, 100.0, np.float32),
        shapes=np.ones((n, 2), np.float32),
        confidences=np.ones(n, np.float32),
        corners=corners,
        detection_ids=np.arange(n, dtype=np.int64),
    )
    crops = torch.from_numpy(
        rng.integers(0, 256, (n, 3, 48, 96)).astype(np.float32) / 255.0
    )
    got = crops_mod._apply_foreign_mask_canonical_batch(crops, obb, _GEOM, (0, 0, 0))
    monkeypatch.setattr(crops_mod, "_apply_foreign_mask_canonical", _reference)
    want = crops_mod._apply_foreign_mask_canonical_batch(crops, obb, _GEOM, (0, 0, 0))
    assert torch.equal(got, want)
