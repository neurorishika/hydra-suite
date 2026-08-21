"""``extract_canonical_crop`` must stay bit-identical after the lazy-frame change.

It used to convert the WHOLE frame to a float32 CHW tensor on every call, and
every caller loops it over a frame's detections -- so the conversion was
O(frame area) per crop. It now slices the detection's canvas footprint out of
the raw frame first. That is only a legitimate substitution if the returned
uint8 crop is exactly what the full-frame path returned, so this compares
against a frozen copy of the old implementation with ``array_equal``.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from hydra_suite.core.canonicalization.crop import (
    _apply_foreign_mask_canonical,
    extract_canonical_crop,
)
from hydra_suite.core.canonicalization.geometry import CanonicalGeometry
from hydra_suite.core.canonicalization.resample import canonical_warp


def _geom(cw: int = 96, ch: int = 48) -> CanonicalGeometry:
    return CanonicalGeometry(canvas_wh=(cw, ch), margin=2.0, aspect_ratio=cw / ch)


def _ref_extract(frame, M_align, geometry, bg_color=(0, 0, 0), foreign=None, own=None):
    """Frozen copy of the pre-change full-frame implementation."""
    arr = np.asarray(frame)
    frame_chw = torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1).float()
    crop_chw = canonical_warp(frame_chw, M_align, geometry)
    crop = (
        crop_chw.round()
        .clamp_(0, 255)
        .to(torch.uint8)
        .permute(1, 2, 0)
        .contiguous()
        .numpy()
    )
    if foreign:
        _apply_foreign_mask_canonical(crop, M_align, foreign, bg_color, own_corners=own)
    return crop


def _rand_affine(rng, w, h):
    ang = rng.uniform(-np.pi, np.pi)
    s = rng.uniform(0.5, 2.0)
    ca, sa = np.cos(ang) * s, np.sin(ang) * s
    return np.array(
        [[ca, -sa, rng.uniform(0, w)], [sa, ca, rng.uniform(0, h)]], dtype=np.float64
    )


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_matches_full_frame_reference_bitwise(seed):
    rng = np.random.default_rng(seed)
    h, w = 200, 260
    geo = _geom()
    frame = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    m = _rand_affine(rng, w, h)

    got = extract_canonical_crop(frame, m, geometry=geo)
    ref = _ref_extract(frame, m, geo)

    assert got.dtype == ref.dtype == np.uint8
    assert got.shape == ref.shape == (geo.canvas_h, geo.canvas_w, 3)
    assert np.array_equal(got, ref)


def test_matches_reference_with_foreign_mask():
    rng = np.random.default_rng(11)
    h, w = 180, 180
    geo = _geom(64, 32)
    frame = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    m = _rand_affine(rng, w, h)
    own = np.array([[40, 40], [80, 40], [80, 70], [40, 70]], dtype=np.float64)
    foreign = [np.array([[60, 50], [110, 50], [110, 90], [60, 90]], dtype=np.float64)]

    got = extract_canonical_crop(
        frame,
        m,
        bg_color=(7, 9, 11),
        foreign_corners=foreign,
        own_corners=own,
        geometry=geo,
    )
    ref = _ref_extract(frame, m, geo, (7, 9, 11), foreign, own)
    assert np.array_equal(got, ref)


def test_matches_reference_for_off_frame_detection():
    """A detection whose footprint lies outside the frame must still zero-fill."""
    rng = np.random.default_rng(5)
    h, w = 64, 64
    geo = _geom(32, 16)
    frame = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    m = np.array([[1.0, 0.0, -5000.0], [0.0, 1.0, -5000.0]], dtype=np.float64)
    assert np.array_equal(
        extract_canonical_crop(frame, m, geometry=geo), _ref_extract(frame, m, geo)
    )
