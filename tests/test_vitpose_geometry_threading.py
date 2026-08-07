"""Geometry threading: non-default geometry must reach every stage."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from hydra_suite.core.individual.pose.vitpose.geometry import (
    DEFAULT_GEOMETRY,
    PoseGeometry,
)
from hydra_suite.core.individual.pose.vitpose.heads import build_head
from hydra_suite.core.individual.pose.vitpose.infer import preprocess_crop
from hydra_suite.core.individual.pose.vitpose.transforms import box2cs, top_down_affine
from hydra_suite.core.individual.pose.vitpose.vitpose import build_vitpose

SQUARE = PoseGeometry((256, 256))


def test_box2cs_uses_the_geometry_aspect_not_the_default():
    box = np.array([0.0, 0.0, 100.0, 100.0], dtype=np.float32)
    _, scale_default = box2cs(box)
    _, scale_square = box2cs(box, geom=SQUARE)
    # Default aspect 0.75 grows a square box's height; square aspect does not.
    assert scale_default[1] > scale_square[1]
    assert scale_square[0] == pytest.approx(scale_square[1])


def test_box2cs_default_argument_is_unchanged():
    box = np.array([0.0, 0.0, 100.0, 100.0], dtype=np.float32)
    center_a, scale_a = box2cs(box)
    center_b, scale_b = box2cs(box, geom=DEFAULT_GEOMETRY)
    # Compared element-wise (not as a tuple) to sidestep a pytest 9.x
    # ValueError when pytest.approx wraps a tuple of same-shape ndarrays.
    assert center_a == pytest.approx(center_b)
    assert scale_a == pytest.approx(scale_b)


def test_top_down_affine_warps_to_the_geometry_size():
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    center, scale = box2cs(
        np.array([0.0, 0.0, 200.0, 200.0], dtype=np.float32), geom=SQUARE
    )
    out = top_down_affine(img, center, scale, geom=SQUARE)
    assert out.shape == (256, 256, 3)


def test_top_down_affine_default_is_still_192x256():
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    center, scale = box2cs(np.array([0.0, 0.0, 200.0, 200.0], dtype=np.float32))
    assert top_down_affine(img, center, scale).shape == (256, 192, 3)


def test_preprocess_crop_emits_the_geometry_shaped_tensor():
    crop = np.zeros((120, 120, 3), dtype=np.uint8)
    chw, _, _ = preprocess_crop(crop, geom=SQUARE)
    assert chw.shape == (3, 256, 256)
    chw_default, _, _ = preprocess_crop(crop)
    assert chw_default.shape == (3, 256, 192)


def test_simple_head_follows_the_geometry():
    # SimpleHead used to hardcode (64, 48); it must now track the geometry.
    head = build_head("simple", 768, 9, geom=SQUARE)
    out = head(torch.zeros(1, 768, 16, 16))
    assert out.shape == (1, 9, 64, 64)


def test_simple_head_default_is_unchanged():
    head = build_head("simple", 768, 9)
    out = head(torch.zeros(1, 768, 16, 12))
    assert out.shape == (1, 9, 64, 48)


def test_classic_head_scales_naturally_with_the_token_grid():
    head = build_head("classic", 768, 9, geom=SQUARE)
    out = head(torch.zeros(1, 768, 16, 16))
    assert out.shape == (1, 9, 64, 64)


def test_build_vitpose_constructs_the_backbone_at_the_geometry():
    model = build_vitpose("B", "classic", num_keypoints=9, geom=SQUARE)
    assert model.backbone.pos_embed.shape == (1, 257, 768)
    with torch.no_grad():
        out = model(torch.zeros(1, 3, 256, 256))
    assert out.shape == (1, 9, 64, 64)


def test_build_vitpose_default_is_unchanged():
    model = build_vitpose("B", "classic", num_keypoints=9)
    assert model.backbone.pos_embed.shape == (1, 193, 768)
