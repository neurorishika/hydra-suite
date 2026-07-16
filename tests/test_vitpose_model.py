import torch

from hydra_suite.core.identity.pose.vitpose.config import VARIANTS
from hydra_suite.core.identity.pose.vitpose.model import PatchEmbed, ViT


def _vit_b() -> ViT:
    v = VARIANTS["B"]
    return ViT(embed_dim=v.embed_dim, depth=v.depth, num_heads=v.num_heads)


def test_patch_embed_uses_padding_two_not_zero():
    """TRAP 1. Upstream computes padding = 4 + 2*(ratio//2 - 1) = 2 for ratio=1.
    Stock timm uses padding=0. The output grid is coincidentally identical
    (floor((256+4-16)/16)+1 == 16), so a wrong padding loads with NO shape error
    and silently samples a shifted pixel grid."""
    pe = PatchEmbed(embed_dim=768)
    assert pe.proj.padding == (2, 2)
    assert pe.proj.kernel_size == (16, 16)
    assert pe.proj.stride == (16, 16)


def test_backbone_output_shape():
    m = _vit_b().eval()
    with torch.no_grad():
        out = m(torch.zeros(2, 3, 256, 192))
    assert out.shape == (2, 768, 16, 12)


def test_pos_embed_retains_cls_slot():
    """TRAP 2. pos_embed is (1, num_patches+1, D) -- the MAE cls slot is kept
    even though no cls_token module exists."""
    m = _vit_b()
    assert m.pos_embed.shape == (1, 16 * 12 + 1, 768)


def test_pos_embed_adds_cls_slot_to_every_token():
    """TRAP 2, the part that silently changes outputs. Upstream does
        x = x + pos_embed[:, 1:] + pos_embed[:, :1]
    i.e. the cls positional embedding is broadcast onto EVERY patch token.
    Dropping that second term still runs and still has the right shape.

    Test: zero the patch slots, set the cls slot to a known constant, and feed a
    zero image with identity-ish blocks bypassed. If the cls term is applied, the
    pre-block token tensor equals that constant.
    """
    m = _vit_b().eval()
    with torch.no_grad():
        m.pos_embed.zero_()
        m.pos_embed[:, :1].fill_(0.5)
        m.patch_embed.proj.weight.zero_()
        m.patch_embed.proj.bias.zero_()
        tokens = m.forward_tokens(torch.zeros(1, 3, 256, 192))
    assert torch.allclose(
        tokens, torch.full_like(tokens, 0.5)
    ), "cls positional embedding is not being broadcast onto patch tokens"


def test_layernorm_eps():
    m = _vit_b()
    assert m.blocks[0].norm1.eps == 1e-6
    assert m.last_norm.eps == 1e-6


def test_attention_head_dim_for_small():
    v = VARIANTS["S"]
    m = ViT(embed_dim=v.embed_dim, depth=1, num_heads=v.num_heads)
    assert m.blocks[0].attn.num_heads == 12
    assert m.blocks[0].attn.head_dim == 32
