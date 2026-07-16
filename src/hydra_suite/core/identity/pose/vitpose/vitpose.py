"""Top-level ViTPose: backbone + keypoint head.

Attribute names `backbone` and `keypoint_head` are deliberate: they equal the
upstream checkpoint's state_dict prefixes, so strict loading needs no rename map.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import VARIANTS
from .heads import build_head
from .model import ViT


class ViTPose(nn.Module):
    def __init__(self, backbone: ViT, keypoint_head: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.keypoint_head = keypoint_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.keypoint_head(self.backbone(x))


def build_vitpose(variant: str, head: str, num_keypoints: int = 17) -> ViTPose:
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r} (expected one of SBLH)")
    v = VARIANTS[variant]
    backbone = ViT(embed_dim=v.embed_dim, depth=v.depth, num_heads=v.num_heads)
    return ViTPose(backbone, build_head(head, v.embed_dim, num_keypoints))


# out_channels for the 5 associate (non-COCO) datasets, in checkpoint order.
# AiC, MPII, AP-10K, APT-36K, WholeBody
ASSOCIATE_HEAD_CHANNELS = (14, 16, 17, 17, 133)


class ViTPoseMoE(nn.Module):
    def __init__(
        self,
        backbone: ViT,
        keypoint_head: nn.Module,
        associate_keypoint_heads: nn.ModuleList,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.keypoint_head = keypoint_head
        self.associate_keypoint_heads = associate_keypoint_heads

    def forward(self, x: torch.Tensor, dataset_index: int = 0) -> torch.Tensor:
        feat = self.backbone(x, dataset_index=dataset_index)
        if dataset_index == 0:
            return self.keypoint_head(feat)
        return self.associate_keypoint_heads[dataset_index - 1](feat)


def build_vitpose_moe(variant: str, num_keypoints: int = 17) -> ViTPoseMoE:
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r} (expected one of SBLH)")
    v = VARIANTS[variant]
    backbone = ViT(
        embed_dim=v.embed_dim,
        depth=v.depth,
        num_heads=v.num_heads,
        part_features=v.part_features,
    )
    head = build_head("classic", v.embed_dim, num_keypoints)
    associates = nn.ModuleList(
        [build_head("classic", v.embed_dim, c) for c in ASSOCIATE_HEAD_CHANNELS]
    )
    return ViTPoseMoE(backbone, head, associates)
