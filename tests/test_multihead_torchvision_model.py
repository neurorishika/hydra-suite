"""Unit tests for the shared-trunk multi-head torchvision wrapper."""

from __future__ import annotations

import torch

from hydra_suite.training.multihead_torchvision_model import (
    build_multihead_torchvision_classifier,
)


def _set_inference_mode(m):
    m.train(False)


def test_forward_returns_concat_logits_with_correct_width():
    model = build_multihead_torchvision_classifier(
        backbone="resnet18",
        class_names_per_factor=[["a", "b", "c"], ["x", "y"]],
        trainable_layers=-1,
        head_hidden_dim=64,
        head_dropout=0.1,
        input_size=64,
    )
    _set_inference_mode(model)
    batch = torch.zeros(2, 3, 64, 64)
    with torch.no_grad():
        out = model(batch)
    assert out.shape == (2, 5)  # 3 + 2 concatenated


def test_per_factor_logits_helper_returns_list():
    model = build_multihead_torchvision_classifier(
        backbone="resnet18",
        class_names_per_factor=[["a", "b"], ["x", "y", "z"]],
        trainable_layers=-1,
        head_hidden_dim=32,
        head_dropout=0.0,
        input_size=64,
    )
    _set_inference_mode(model)
    batch = torch.zeros(2, 3, 64, 64)
    with torch.no_grad():
        per_factor = model.forward_per_factor(batch)
    assert isinstance(per_factor, list) and len(per_factor) == 2
    assert per_factor[0].shape == (2, 2)
    assert per_factor[1].shape == (2, 3)


def test_factor_widths_property():
    model = build_multihead_torchvision_classifier(
        backbone="resnet18",
        class_names_per_factor=[["a", "b", "c", "d"], ["x"], ["y", "z"]],
        trainable_layers=-1,
        head_hidden_dim=16,
        head_dropout=0.0,
        input_size=64,
    )
    assert model.factor_widths == [4, 1, 2]


def test_freeze_unfreeze_backbone_round_trip():
    model = build_multihead_torchvision_classifier(
        backbone="resnet18",
        class_names_per_factor=[["a", "b"], ["x", "y"]],
        trainable_layers=0,  # head-only
        head_hidden_dim=16,
        head_dropout=0.0,
        input_size=64,
    )
    backbone_params = list(model.backbone.parameters())
    head_params = list(model.heads.parameters())
    assert all(not p.requires_grad for p in backbone_params)
    assert all(p.requires_grad for p in head_params)
    model.unfreeze_all()
    assert all(p.requires_grad for p in backbone_params)
