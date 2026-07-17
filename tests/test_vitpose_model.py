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


from hydra_suite.core.identity.pose.vitpose.model import MoEMlp


def test_moe_shapes_for_base():
    """B: fc1 768->3072, fc2 3072->576 (D - part_features), 6 experts 3072->192.
    Concat of shared (576) + expert (192) restores 768."""
    m = MoEMlp(dim=768, hidden=3072, part_features=192, num_expert=6)
    assert m.fc1.out_features == 3072
    assert m.fc2.out_features == 768 - 192
    assert len(m.experts) == 6
    assert m.experts[0].out_features == 192
    out = m(torch.zeros(2, 10, 768), torch.zeros(2, dtype=torch.long))
    assert out.shape == (2, 10, 768)


def test_moe_routing_is_by_dataset_index_not_learned():
    """Routing is NOT learned: `indices` is the dataset index supplied from
    outside. Different indices must select different experts."""
    m = MoEMlp(dim=8, hidden=16, part_features=4, num_expert=6).eval()
    with torch.no_grad():
        for i, e in enumerate(m.experts):
            e.weight.fill_(float(i + 1))
            e.bias.zero_()
        m.fc1.weight.zero_()
        m.fc1.bias.fill_(1.0)
        m.fc2.weight.zero_()
        m.fc2.bias.zero_()
        x = torch.zeros(1, 1, 8)
        out0 = m(x, torch.zeros(1, dtype=torch.long))
        out3 = m(x, torch.full((1,), 3, dtype=torch.long))
    assert not torch.allclose(out0[..., -4:], out3[..., -4:])


def test_moe_int_index_takes_no_host_sync_path():
    """`dataset_index` as a plain int must never touch torch.unique/.item()/
    .cpu()/.tolist() -- those force a GPU->CPU sync every block. Assert on the
    source so a future edit can't silently reintroduce the sync."""
    import inspect

    src = inspect.getsource(MoEMlp.forward)
    for banned in (".item()", ".cpu()", ".tolist()", "torch.unique"):
        assert banned not in src, f"{banned!r} must not appear in MoEMlp.forward"


def test_moe_multi_index_matches_per_sample_single_index():
    """FINDING 4: the masked/gathered multi-index path is otherwise untested.
    For a batch with mixed dataset indices, each sample's output must equal
    what that sample alone would get through the fast int path."""
    m = MoEMlp(dim=8, hidden=16, part_features=4, num_expert=6).eval()
    with torch.no_grad():
        for i, e in enumerate(m.experts):
            e.weight.fill_(float(i + 1))
            e.bias.zero_()
        x = torch.randn(4, 3, 8)
        indices = torch.tensor([0, 2, 5, 2], dtype=torch.long)
        out_mixed = m(x, indices)
        for i in range(4):
            out_single = m(x[i : i + 1], int(indices[i]))
            assert torch.allclose(out_mixed[i : i + 1], out_single), (
                f"sample {i} (dataset {int(indices[i])}) diverged from its "
                "solo pass through the masked multi-index path"
            )
