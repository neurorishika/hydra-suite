"""Patch-grid recovery and pos_embed interpolation."""

from __future__ import annotations

import pytest
import torch

from hydra_suite.core.identity.pose.vitpose.geometry import PoseGeometry
from hydra_suite.core.identity.pose.vitpose.pos_embed import (
    resize_pos_embed,
    resolve_patch_grid,
)


def test_stored_geometry_wins_over_inference():
    # 256 patches could be 16x16 or 8x32; a stored geometry settles it.
    assert resolve_patch_grid(256, PoseGeometry((512, 128))) == (8, 32)


def test_stored_geometry_must_agree_with_the_token_count():
    with pytest.raises(ValueError, match="does not match"):
        resolve_patch_grid(192, PoseGeometry((256, 256)))


def test_perfect_square_resolves_to_a_square_grid():
    # The collaborator's checkpoints: 257 pos_embed tokens -> 256 patches.
    assert resolve_patch_grid(256) == (16, 16)


def test_default_aspect_resolves_the_upstream_vitpose_grid():
    # Every upstream ViTPose release: 193 tokens -> 192 patches -> 12x16.
    assert resolve_patch_grid(192) == (16, 12)


def test_unresolvable_count_raises_rather_than_guessing():
    with pytest.raises(ValueError, match="cannot determine|input_size"):
        resolve_patch_grid(150)


def test_error_names_the_token_count_and_asks_for_input_size():
    with pytest.raises(ValueError) as exc:
        resolve_patch_grid(150)
    assert "150" in str(exc.value)
    assert "input_size" in str(exc.value)


def test_resize_is_identity_when_grids_match():
    pe = torch.randn(1, 193, 768)
    out = resize_pos_embed(pe, (16, 12), (16, 12))
    assert torch.equal(out, pe)


def test_resize_produces_the_target_token_count():
    pe = torch.randn(1, 193, 768)
    out = resize_pos_embed(pe, (16, 12), (16, 16))
    assert out.shape == (1, 257, 768)


def test_resize_preserves_the_cls_slot_exactly():
    pe = torch.randn(1, 193, 768)
    out = resize_pos_embed(pe, (16, 12), (16, 16))
    assert torch.equal(out[:, :1], pe[:, :1])


def test_resize_rejects_a_source_grid_that_contradicts_the_tensor():
    pe = torch.randn(1, 193, 768)
    with pytest.raises(ValueError, match="does not match"):
        resize_pos_embed(pe, (16, 16), (16, 12))


def test_resize_round_trip_approximately_recovers_a_smooth_field():
    # Bicubic up-then-down on a smooth field should be close to the original.
    gh, gw, dim = 16, 12, 8
    ramp = torch.linspace(0, 1, gh * gw).reshape(1, gh * gw, 1).repeat(1, 1, dim)
    pe = torch.cat([torch.zeros(1, 1, dim), ramp], dim=1)
    up = resize_pos_embed(pe, (gh, gw), (32, 24))
    back = resize_pos_embed(up, (32, 24), (gh, gw))
    assert torch.allclose(back, pe, atol=2e-2)
