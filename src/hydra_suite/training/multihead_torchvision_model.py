"""Shared-trunk multi-head torchvision classifier.

One pretrained backbone + N parallel MLP heads. Forward returns concatenated
logits ``[batch, sum(C_k)]`` so ``ClassifierBackend.predict_batch`` can keep
splitting per-factor by ``_cardinalities()`` without modification.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from hydra_suite.training.torchvision_model import (
    TORCHVISION_BACKBONES,
    _load_pretrained,
    freeze_backbone,
    is_timm_backbone,
)


def _strip_classifier_head(model: nn.Module, backbone: str) -> tuple[nn.Module, int]:
    """Replace the model's classifier head with ``nn.Identity`` and return the
    backbone-feature dimension."""
    if is_timm_backbone(backbone):
        if hasattr(model, "reset_classifier"):
            model.reset_classifier(num_classes=0)
            with torch.no_grad():
                dummy = torch.zeros(1, 3, 64, 64)
                feat_dim = int(model(dummy).shape[1])
            return model, feat_dim
        raise ValueError(f"unsupported timm backbone for trunk strip: {backbone!r}")

    if (
        backbone.startswith("convnext")
        or backbone.startswith("mobilenet")
        or backbone.startswith("efficientnet")
    ):
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Identity()
        return model, in_features
    if backbone.startswith("resnet") or backbone.startswith("shufflenet"):
        in_features = model.fc.in_features
        model.fc = nn.Identity()
        return model, in_features
    if backbone == "vit_b_16":
        in_features = model.heads.head.in_features
        model.heads.head = nn.Identity()
        return model, in_features
    raise ValueError(f"unsupported backbone for trunk strip: {backbone!r}")


def _build_head(
    in_features: int, num_classes: int, hidden_dim: int, dropout: float
) -> nn.Module:
    """One hidden-layer MLP head: Linear -> GELU -> Dropout -> Linear."""
    return nn.Sequential(
        nn.Linear(in_features, hidden_dim),
        nn.GELU(),
        nn.Dropout(float(dropout)),
        nn.Linear(hidden_dim, num_classes),
    )


class MultiHeadTorchvisionClassifier(nn.Module):
    """Shared backbone + per-factor MLP heads.

    Forward pass returns logits concatenated along the last dim:
    ``[B, C_0 + C_1 + ... + C_{K-1}]``. The backend splits these by
    cardinality.
    """

    def __init__(
        self,
        backbone_module: nn.Module,
        feat_dim: int,
        class_counts_per_factor: list[int],
        head_hidden_dim: int,
        head_dropout: float,
    ) -> None:
        super().__init__()
        if not class_counts_per_factor:
            raise ValueError("class_counts_per_factor must be non-empty")
        if any(c <= 0 for c in class_counts_per_factor):
            raise ValueError("each factor must have >=1 classes")
        self.backbone = backbone_module
        self.heads = nn.ModuleList(
            [
                _build_head(feat_dim, c, head_hidden_dim, head_dropout)
                for c in class_counts_per_factor
            ]
        )
        self._factor_widths = list(class_counts_per_factor)

    @property
    def factor_widths(self) -> list[int]:
        return list(self._factor_widths)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        return torch.cat([h(feat) for h in self.heads], dim=-1)

    def forward_per_factor(self, x: torch.Tensor) -> list[torch.Tensor]:
        feat = self.backbone(x)
        return [h(feat) for h in self.heads]

    def unfreeze_all(self) -> None:
        for p in self.parameters():
            p.requires_grad = True


def build_multihead_torchvision_classifier(
    *,
    backbone: str,
    class_names_per_factor: list[list[str]],
    trainable_layers: int,
    head_hidden_dim: int,
    head_dropout: float,
    input_size: int | tuple[int, int] | None = None,
) -> MultiHeadTorchvisionClassifier:
    """Build a pretrained torchvision backbone with N parallel MLP heads."""
    if backbone == "tinyclassifier":
        raise ValueError(
            "tinyclassifier backbone is not supported in shared-trunk mode"
        )
    if backbone not in TORCHVISION_BACKBONES and not is_timm_backbone(backbone):
        raise ValueError(f"unknown backbone {backbone!r}")
    if not class_names_per_factor:
        raise ValueError("class_names_per_factor must be non-empty")
    base = _load_pretrained(backbone, input_size=input_size)
    base, feat_dim = _strip_classifier_head(base, backbone)
    class_counts = [len(inner) for inner in class_names_per_factor]
    model = MultiHeadTorchvisionClassifier(
        backbone_module=base,
        feat_dim=feat_dim,
        class_counts_per_factor=class_counts,
        head_hidden_dim=int(head_hidden_dim),
        head_dropout=float(head_dropout),
    )
    if trainable_layers != -1:
        # ``freeze_backbone`` operates on the bare torchvision model; we passed
        # ``base`` (the backbone with Identity classifier head) so any head
        # un-freeze it does is harmless. Then we unconditionally enable grads
        # on our MLP heads.
        freeze_backbone(model.backbone, backbone, trainable_layers)
    for p in model.heads.parameters():
        p.requires_grad = True
    return model
