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
