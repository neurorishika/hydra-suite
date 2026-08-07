"""ViT backbone for ViTPose.

Plain, non-hierarchical ViT: absolute learned pos-embed only, no relative
position bias, no window attention in any variant (L/H are dense global
attention at every layer). Attribute names mirror the upstream checkpoint's
state_dict keys so load_state_dict(strict=True) needs no remapping.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from timm.layers import DropPath, trunc_normal_

from .config import NUM_EXPERTS, PATCH_PADDING, PATCH_SIZE


class PatchEmbed(nn.Module):
    def __init__(self, embed_dim: int, in_chans: int = 3) -> None:
        super().__init__()
        # padding=2 is upstream's `4 + 2*(ratio//2 - 1)` with ratio=1, NOT the
        # stock ViT padding=0. See test_patch_embed_uses_padding_two_not_zero.
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=PATCH_SIZE,
            stride=PATCH_SIZE,
            padding=PATCH_PADDING,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        x = self.proj(x)
        _, _, hp, wp = x.shape
        return x.flatten(2).transpose(1, 2), hp, wp


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = True) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        return self.proj(x)


class Mlp(nn.Module):
    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class MoEMlp(nn.Module):
    """ViTPose+ FFN. ONLY the FFN differs from classic; attention, patch embed,
    pos embed and norms are byte-identical.

    Routing is NOT learned -- `indices` is the dataset index threaded in from
    outside. Upstream runs all experts and masks (a DDP workaround); for
    single-dataset inference (a plain Python `int`) we index the expert
    directly on the whole batch, which is numerically identical and avoids
    6x the expert-branch compute -- and, critically, never reads a tensor
    value back to the host, so no GPU sync happens on the hot path. A
    per-sample `torch.Tensor` of indices (multi-dataset training) takes the
    masked/gathered path instead.
    """

    def __init__(
        self,
        dim: int,
        hidden: int,
        part_features: int,
        num_expert: int = NUM_EXPERTS,
    ) -> None:
        super().__init__()
        self.part_features = part_features
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim - part_features)
        self.experts = nn.ModuleList(
            [nn.Linear(hidden, part_features) for _ in range(num_expert)]
        )

    def forward(self, x: torch.Tensor, indices: torch.Tensor | int) -> torch.Tensor:
        x = self.act(self.fc1(x))
        shared = self.fc2(x)
        if isinstance(indices, int):
            # Single dataset for the whole batch: no tensor, no host sync.
            expert = self.experts[indices](x)
        else:
            expert = torch.zeros(
                *x.shape[:-1], self.part_features, device=x.device, dtype=x.dtype
            )
            for i, e in enumerate(self.experts):
                mask = indices == i
                if mask.any():
                    expert[mask] = e(x[mask])
        return torch.cat([shared, expert], dim=-1)


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        part_features: int | None = None,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden = int(dim * mlp_ratio)
        if part_features is None:
            self.mlp = Mlp(dim, hidden)
        else:
            self.mlp = MoEMlp(dim, hidden, part_features)

    def forward(
        self, x: torch.Tensor, dataset_index: torch.Tensor | int = 0
    ) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        if isinstance(self.mlp, MoEMlp):
            mlp_out = self.mlp(self.norm2(x), dataset_index)
        else:
            mlp_out = self.mlp(self.norm2(x))
        x = x + self.drop_path(mlp_out)
        return x


class ViT(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        depth: int,
        num_heads: int,
        img_size_hw: tuple[int, int] = (256, 192),
        drop_path_rate: float = 0.0,
        part_features: int | None = None,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(embed_dim)
        hp = img_size_hw[0] // PATCH_SIZE
        wp = img_size_hw[1] // PATCH_SIZE
        num_patches = hp * wp
        # +1 keeps the MAE cls slot; there is no cls_token module.
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    drop_path=dpr[i],
                    part_features=part_features,
                )
                for i in range(depth)
            ]
        )
        self.last_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        trunc_normal_(self.pos_embed, std=0.02)

    def forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Patch-embed and add positional embeddings. Exposed for testing the
        cls-broadcast behaviour without running the blocks."""
        x, _, _ = self.patch_embed(x)
        # Both terms are required: patch pos-embeds PLUS the cls pos-embed
        # broadcast to every token.
        return x + self.pos_embed[:, 1:] + self.pos_embed[:, :1]

    def forward(
        self, x: torch.Tensor, dataset_index: torch.Tensor | int = 0
    ) -> torch.Tensor:
        x, hp, wp = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:] + self.pos_embed[:, :1]
        for blk in self.blocks:
            x = blk(x, dataset_index)
        x = self.last_norm(x)
        b, _, c = x.shape
        return x.permute(0, 2, 1).reshape(b, c, hp, wp)
