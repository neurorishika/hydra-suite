"""Patch-grid recovery and bicubic pos_embed resizing.

A bare token count does not determine the patch grid -- 256 patches could be
16x16 or 8x32 -- so recovery follows a fixed order and RAISES rather than
guessing. A wrong grid does not fail loudly at load time; it produces a
plausible model that is subtly wrong everywhere.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .geometry import PoseGeometry


def resolve_patch_grid(
    num_patches: int, stored: PoseGeometry | None = None
) -> tuple[int, int]:
    """Recover (grid_h, grid_w) from a patch count.

    Order: a stored geometry is authoritative; then a perfect square; then the
    default 0.75 aspect (which covers every upstream ViTPose release); then
    raise.
    """
    if num_patches <= 0:
        raise ValueError(f"num_patches must be positive; got {num_patches}")

    if stored is not None:
        gh, gw = stored.patch_grid_hw
        if gh * gw != num_patches:
            raise ValueError(
                f"stored geometry {stored.image_size_wh} implies a {gh}x{gw} patch "
                f"grid ({gh * gw} patches) but the checkpoint has {num_patches}; "
                "the recorded input_size does not match the weights"
            )
        return (gh, gw)

    root = math.isqrt(num_patches)
    if root * root == num_patches:
        return (root, root)

    # Default aspect: width/height == 0.75, so grid_h/grid_w == 4/3.
    gh_sq = num_patches * 4
    if gh_sq % 3 == 0:
        gh = math.isqrt(gh_sq // 3)
        if gh > 0 and gh % 4 == 0:
            gw = (gh * 3) // 4
            if gh * gw == num_patches:
                return (gh, gw)

    raise ValueError(
        f"cannot determine the patch grid for {num_patches} patches: it is "
        "neither square nor the default 0.75 aspect. Record an explicit "
        "input_size [H, W] in the checkpoint."
    )


def resize_pos_embed(
    pos_embed: torch.Tensor,
    src_grid_hw: tuple[int, int],
    dst_grid_hw: tuple[int, int],
) -> torch.Tensor:
    """Bicubically resample the patch grid of a (1, 1 + gh*gw, D) pos_embed.

    The leading cls slot is carried through untouched. Returns the input
    object unchanged when the grids already match, so the default path is
    bit-for-bit unaffected.
    """
    src_h, src_w = src_grid_hw
    expected = src_h * src_w + 1
    if pos_embed.shape[1] != expected:
        raise ValueError(
            f"pos_embed has {pos_embed.shape[1]} tokens which does not match the "
            f"declared {src_h}x{src_w} source grid ({expected} tokens with the "
            "cls slot)"
        )
    if src_grid_hw == dst_grid_hw:
        return pos_embed

    cls_token = pos_embed[:, :1]
    grid = pos_embed[:, 1:]
    dim = grid.shape[-1]
    grid = grid.reshape(1, src_h, src_w, dim).permute(0, 3, 1, 2)
    grid = F.interpolate(
        grid.float(), size=dst_grid_hw, mode="bicubic", align_corners=False
    )
    grid = grid.permute(0, 2, 3, 1).reshape(1, dst_grid_hw[0] * dst_grid_hw[1], dim)
    return torch.cat([cls_token, grid.to(pos_embed.dtype)], dim=1)


def grid_for_state(
    pos_embed: torch.Tensor, stored: PoseGeometry | None = None
) -> tuple[int, int]:
    """Convenience: resolve the grid for a checkpoint's pos_embed tensor."""
    return resolve_patch_grid(int(pos_embed.shape[1]) - 1, stored)
