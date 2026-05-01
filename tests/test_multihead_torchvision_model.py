"""Unit tests for the shared-trunk multi-head torchvision wrapper."""

from __future__ import annotations

import torch

from hydra_suite.training.multihead_torchvision_model import (
    MultiHeadTorchvisionClassifier,
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


def test_forward_concat_equals_per_factor_slices():
    """The concatenated logits must equal per-factor logits when split by
    cardinality — this is what ClassifierBackend.predict_batch relies on."""
    model = build_multihead_torchvision_classifier(
        backbone="resnet18",
        class_names_per_factor=[["a", "b"], ["x", "y", "z"], ["m", "n"]],
        trainable_layers=-1,
        head_hidden_dim=24,
        head_dropout=0.0,
        input_size=64,
    )
    model.train(False)
    batch = torch.randn(3, 3, 64, 64)
    with torch.no_grad():
        concat = model(batch)
        per_factor = model.forward_per_factor(batch)
    widths = model.factor_widths
    assert sum(widths) == concat.shape[-1]
    offset = 0
    for k, w in enumerate(widths):
        slice_ = concat[:, offset : offset + w]
        assert slice_.shape == per_factor[k].shape
        assert torch.allclose(slice_, per_factor[k], atol=1e-6)
        offset += w


def test_save_load_round_trip_preserves_head_kind(tmp_path):
    from hydra_suite.training.torchvision_model import (
        load_torchvision_classifier,
        save_torchvision_checkpoint,
    )

    model = build_multihead_torchvision_classifier(
        backbone="resnet18",
        class_names_per_factor=[["a", "b"], ["x", "y", "z"]],
        trainable_layers=-1,
        head_hidden_dim=32,
        head_dropout=0.1,
        input_size=64,
    )
    ckpt_path = tmp_path / "shared_trunk.pth"
    save_torchvision_checkpoint(
        model=model,
        backbone="resnet18",
        class_names=[],
        factor_names=["color1", "color2"],
        class_names_per_factor=[["a", "b"], ["x", "y", "z"]],
        input_size=(64, 64),
        best_val_acc=0.91,
        history={"train_loss": [], "val_acc": []},
        trainable_layers=-1,
        backbone_lr_scale=0.1,
        monochrome=False,
        extra_meta={
            "head_kind": "multihead_shared_trunk",
            "head_hidden_dim": 32,
            "head_dropout": 0.1,
        },
        path=ckpt_path,
    )

    loaded, ckpt = load_torchvision_classifier(str(ckpt_path), device="cpu")
    assert isinstance(loaded, MultiHeadTorchvisionClassifier)
    assert loaded.factor_widths == [2, 3]
    assert ckpt["head_kind"] == "multihead_shared_trunk"
    assert ckpt["factor_names"] == ["color1", "color2"]
    # Same input -> same output after round-trip
    batch = torch.randn(1, 3, 64, 64)
    model.train(False)
    loaded.train(False)
    with torch.no_grad():
        a = model(batch)
        b = loaded(batch)
    assert torch.allclose(a, b, atol=1e-5)


def test_contracts_expose_shared_trunk_role_and_params():
    from hydra_suite.training.contracts import CustomCNNParams, TrainingRole

    assert hasattr(TrainingRole, "CLASSIFY_MULTIHEAD_CUSTOM_SHARED")
    assert (
        TrainingRole.CLASSIFY_MULTIHEAD_CUSTOM_SHARED.value
        == "classify_multihead_custom_shared"
    )
    p = CustomCNNParams()
    assert p.head_kind == "flat"
    assert p.head_hidden_dim > 0
    assert 0.0 <= p.head_dropout < 1.0
    p2 = CustomCNNParams(
        head_kind="multihead_shared_trunk", head_hidden_dim=128, head_dropout=0.2
    )
    assert p2.head_kind == "multihead_shared_trunk"


def test_strip_classifier_head_uses_input_size_for_timm_probe(monkeypatch):
    """Fixed-resolution timm models (ViT/EVA02) reject the legacy 64x64 probe.

    ``_strip_classifier_head`` must use the configured ``input_size`` so the
    feat_dim probe matches the model's expected resolution.
    """
    import torch.nn as nn

    from hydra_suite.training import multihead_torchvision_model as mod

    captured: dict = {}

    class _StubVitOnly(nn.Module):
        """Mimics a fixed-resolution ViT: rejects anything other than 224x224."""

        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Identity()

        def reset_classifier(self, num_classes: int = 0) -> None:
            self.fc = nn.Identity()

        def forward(self, x):
            captured["probe_shape"] = tuple(x.shape)
            if x.shape[-1] != 224 or x.shape[-2] != 224:
                raise RuntimeError(
                    f"Input height ({x.shape[-2]}) doesn't match model (224)."
                )
            return torch.zeros(x.shape[0], 768)

    monkeypatch.setattr(mod, "is_timm_backbone", lambda name: name == "timm/eva02_stub")

    stub = _StubVitOnly()
    out, feat_dim = mod._strip_classifier_head(stub, "timm/eva02_stub", input_size=224)
    assert feat_dim == 768
    assert captured["probe_shape"] == (1, 3, 224, 224)
    assert out is stub
