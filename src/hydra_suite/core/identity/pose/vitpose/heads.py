"""ViTPose heatmap heads.

Both are upstream's TopdownHeatmapSimpleHead; the config chooses between them.
Input (B, D, 16, 12) -> output (B, K, 64, 48).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import HEATMAP_SIZE_WH


class ClassicHead(nn.Module):
    """num_deconv_layers=2, filters=(256, 256), kernels=(4, 4),
    final_conv_kernel=1."""

    def __init__(self, embed_dim: int, num_keypoints: int) -> None:
        super().__init__()
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.final_layer = nn.Conv2d(256, num_keypoints, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.final_layer(self.deconv_layers(x))


class SimpleHead(nn.Module):
    """num_deconv_layers=0, upsample=4, final_conv_kernel=3.

    Upstream applies ReLU inside _transform_inputs, i.e. BEFORE the upsample.
    """

    def __init__(self, embed_dim: int, num_keypoints: int) -> None:
        super().__init__()
        self.deconv_layers = nn.Identity()
        self.final_layer = nn.Conv2d(embed_dim, num_keypoints, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(x)
        w, h = HEATMAP_SIZE_WH
        # Explicit size (not scale_factor): scale_factor traces to a Resize with
        # computed sizes and is the classic ONNX shape-mismatch source. Same
        # result here, exportable later. align_corners=False is upstream's.
        x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
        return self.final_layer(self.deconv_layers(x))


def build_head(kind: str, embed_dim: int, num_keypoints: int) -> nn.Module:
    if kind == "classic":
        return ClassicHead(embed_dim, num_keypoints)
    if kind == "simple":
        return SimpleHead(embed_dim, num_keypoints)
    raise ValueError(f"unknown head kind: {kind!r} (expected 'classic'|'simple')")
