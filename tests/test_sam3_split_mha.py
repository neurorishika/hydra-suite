"""SplitMultiheadAttention: forward parity, injection scope, merge round-trip.

The D1 consolidation slice replaces every in-scope torch nn.MultiheadAttention
with a split q/k/v/out module so LoRA can reach the fused projections (the
empirically validated SAM3 spike's mechanism).  These tests pin:

- numerical parity with stock nn.MultiheadAttention for the exact argument
  shapes SAM3's call sites use (decoder self_attn: seq-first + attn_mask;
  decoder ca_text: seq-first + key_padding_mask; text encoder: batch_first +
  causal attn_mask + need_weights=False);
- merge exactness back into the fused ``in_proj_weight`` row slices, with a
  key set identical to stock;
- zero-initialised adapters merging as a bit-identical no-op;
- the scope rules: SAM3's model_misc-style MHA clone is never replaced and
  its out_proj is never wrapped;
- the trainable-parameter invariant after replacement.
"""

import pytest
import torch
from torch import nn

from hydra_suite.training.sam3_lora.lora import (
    TARGET_SUFFIXES,
    LoraConfig,
    LoraLinear,
    SplitMultiheadAttention,
    adapter_state_dict,
    adapter_touched_keys,
    inject_adapters,
    merge_adapters,
)


def _cfg(rank=4, alpha=8, include=()):
    return LoraConfig(
        rank=rank,
        alpha=alpha,
        dropout=0.0,
        target_suffixes=TARGET_SUFFIXES,
        include_prefixes=include,
    )


def _pair(embed, heads, *, batch_first, dropout=0.0, seed=0):
    """A stock MHA and its split replacement, identical weights, eval mode."""
    torch.manual_seed(seed)
    mha = nn.MultiheadAttention(embed, heads, dropout=dropout, batch_first=batch_first)
    split = SplitMultiheadAttention.from_torch_mha(mha)
    mha.eval()
    split.eval()
    return mha, split


def _assert_forward_parity(mha, split, q, k, v, **kwargs):
    # need_weights=True forces the stock module off the fused fast path onto
    # the eager math the split module mirrors; the attention OUTPUT is the
    # same tensor either way, so this comparison also covers SAM3's
    # need_weights=False call sites.
    ref_out, ref_w = mha(q, k, v, need_weights=True, **kwargs)
    got_out, got_w = split(q, k, v, need_weights=True, **kwargs)
    torch.testing.assert_close(got_out, ref_out, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(got_w, ref_w, rtol=1e-5, atol=1e-6)
    # And the shape SAM3 actually calls with.
    got_out2, none_w = split(q, k, v, need_weights=False, **kwargs)
    torch.testing.assert_close(got_out2, ref_out, rtol=1e-5, atol=1e-6)
    assert none_w is None


def test_split_weights_are_the_fused_row_slices():
    mha, split = _pair(32, 4, batch_first=False)
    E = 32
    assert torch.equal(split.q_proj.weight, mha.in_proj_weight[0:E])
    assert torch.equal(split.k_proj.weight, mha.in_proj_weight[E : 2 * E])
    assert torch.equal(split.v_proj.weight, mha.in_proj_weight[2 * E : 3 * E])
    assert torch.equal(split.out_proj.weight, mha.out_proj.weight)
    assert torch.equal(split.q_proj.bias, mha.in_proj_bias[0:E])
    assert torch.equal(split.k_proj.bias, mha.in_proj_bias[E : 2 * E])
    assert torch.equal(split.v_proj.bias, mha.in_proj_bias[2 * E : 3 * E])
    assert torch.equal(split.out_proj.bias, mha.out_proj.bias)


def test_forward_parity_decoder_self_attn_shape():
    """Seq-first self-attention with q/k carrying pos-embed, distinct v."""
    E, H, L, B = 256, 8, 7, 2
    mha, split = _pair(E, H, batch_first=False, dropout=0.1, seed=1)
    torch.manual_seed(2)
    tgt = torch.randn(L, B, E)
    qk = tgt + torch.randn(L, B, E)  # with_pos_embed
    _assert_forward_parity(mha, split, qk, qk, tgt)


@pytest.mark.parametrize("mask_kind", ["float2d", "bool2d", "float3d", "bool3d"])
def test_forward_parity_with_attn_mask(mask_kind):
    E, H, L, B = 64, 4, 5, 3
    mha, split = _pair(E, H, batch_first=False, seed=3)
    torch.manual_seed(4)
    q = torch.randn(L, B, E)
    if mask_kind == "float2d":
        mask = torch.randn(L, L) * 0.1
    elif mask_kind == "bool2d":
        mask = torch.rand(L, L) > 0.7
        mask.fill_diagonal_(False)  # keep every row attendable
    elif mask_kind == "float3d":
        mask = torch.randn(B * H, L, L) * 0.1
    else:
        mask = torch.rand(B * H, L, L) > 0.7
        mask[:, torch.arange(L), torch.arange(L)] = False
    _assert_forward_parity(mha, split, q, q, q, attn_mask=mask)


def test_forward_parity_ca_text_key_padding_mask():
    """Seq-first cross-attention against text memory with padding mask."""
    E, H, L, S, B = 256, 8, 6, 11, 2
    mha, split = _pair(E, H, batch_first=False, seed=5)
    torch.manual_seed(6)
    q = torch.randn(L, B, E)
    mem = torch.randn(S, B, E)
    kpm = torch.zeros(B, S, dtype=torch.bool)
    kpm[0, 7:] = True
    kpm[1, 3:] = True
    _assert_forward_parity(mha, split, q, mem, mem, key_padding_mask=kpm)


def test_forward_parity_text_encoder_batch_first_causal():
    """batch_first + additive causal float mask, the CLIP text tower shape."""
    E, H, B, S = 128, 8, 2, 9
    mha, split = _pair(E, H, batch_first=True, seed=7)
    torch.manual_seed(8)
    x = torch.randn(B, S, E)
    causal = torch.full((S, S), float("-inf")).triu(1)
    _assert_forward_parity(mha, split, x, x, x, attn_mask=causal)


def test_forward_parity_both_masks_combined():
    E, H, L, S, B = 64, 4, 5, 8, 2
    mha, split = _pair(E, H, batch_first=False, seed=9)
    torch.manual_seed(10)
    q = torch.randn(L, B, E)
    mem = torch.randn(S, B, E)
    mask = torch.randn(L, S) * 0.1
    kpm = torch.zeros(B, S, dtype=torch.bool)
    kpm[:, -2:] = True
    _assert_forward_parity(
        mha, split, q, mem, mem, attn_mask=mask, key_padding_mask=kpm
    )


def test_from_torch_mha_refuses_unvalidated_variants():
    with pytest.raises(ValueError, match="separate q/k/v"):
        SplitMultiheadAttention.from_torch_mha(
            nn.MultiheadAttention(32, 4, kdim=16, vdim=16)
        )
    with pytest.raises(ValueError, match="bias_k"):
        SplitMultiheadAttention.from_torch_mha(
            nn.MultiheadAttention(32, 4, add_bias_kv=True)
        )
    with pytest.raises(TypeError):
        SplitMultiheadAttention.from_torch_mha(nn.Linear(32, 32))


# ---------------------------------------------------------------------------
# Injection scope
# ---------------------------------------------------------------------------


class _CloneLikeMHA(nn.Module):
    """Mimics SAM3's model_misc clone: has in_proj_weight, is NOT torch MHA."""

    def __init__(self, embed):
        super().__init__()
        self.in_proj_weight = nn.Parameter(torch.randn(3 * embed, embed))
        self.out_proj = nn.Linear(embed, embed)


class _Host(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(16, 4)
        self.clone_attn = _CloneLikeMHA(16)
        self.linear1 = nn.Linear(16, 16)


def test_inject_replaces_torch_mha_and_wraps_its_projections():
    model = _Host()
    n = inject_adapters(model, _cfg())
    assert isinstance(model.self_attn, SplitMultiheadAttention)
    for proj in ("q_proj", "k_proj", "v_proj", "out_proj"):
        assert isinstance(getattr(model.self_attn, proj), LoraLinear)
    # 4 projections + linear1
    assert n == 5


def test_inject_still_skips_the_sam3_clone():
    model = _Host()
    inject_adapters(model, _cfg())
    assert isinstance(model.clone_attn, _CloneLikeMHA)
    assert isinstance(model.clone_attn.out_proj, nn.Linear)
    assert not isinstance(model.clone_attn.out_proj, LoraLinear)


def test_inject_leaves_out_of_scope_torch_mha_untouched():
    model = nn.ModuleDict({"inside": _Host(), "outside": _Host()})
    inject_adapters(model, _cfg(include=("inside",)))
    assert isinstance(model["inside"].self_attn, SplitMultiheadAttention)
    assert isinstance(model["outside"].self_attn, nn.MultiheadAttention)
    assert not isinstance(model["outside"].self_attn.out_proj, LoraLinear)


def test_trainable_parameter_invariant_after_replacement():
    """Exactly 2 tensors and rank*(in+out) params per wrapped Linear."""
    model = _Host()
    model.requires_grad_(False)
    cfg = _cfg(rank=4)
    n = inject_adapters(model, cfg)
    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    assert len(trainable) == 2 * n
    assert all(name.endswith(("lora_A", "lora_B")) for name, _p in trainable)
    # Each of the 5 wrapped 16x16 Linears contributes rank*(16+16).
    assert sum(p.numel() for _n, p in trainable) == n * cfg.rank * 32


# ---------------------------------------------------------------------------
# Merge round-trip
# ---------------------------------------------------------------------------


def _mha_base(embed=16):
    """A stock-shaped base checkpoint for one fused MHA."""
    return {
        "detector.self_attn.in_proj_weight": torch.randn(3 * embed, embed),
        "detector.self_attn.in_proj_bias": torch.randn(3 * embed),
        "detector.self_attn.out_proj.weight": torch.randn(embed, embed),
        "detector.self_attn.out_proj.bias": torch.randn(embed),
    }


def test_merge_folds_qkv_deltas_into_the_in_proj_row_slices():
    E, rank = 16, 4
    torch.manual_seed(11)
    base = _mha_base(E)
    original = {k: v.clone() for k, v in base.items()}
    cfg = _cfg(rank=rank)
    adapters = {}
    for leaf in ("q_proj", "k_proj", "v_proj", "out_proj"):
        adapters[f"self_attn.{leaf}.lora_A"] = torch.randn(rank, E)
        adapters[f"self_attn.{leaf}.lora_B"] = torch.randn(E, rank)

    merged = merge_adapters(base, adapters, cfg)

    assert set(merged) == set(original)  # key-identical to stock
    for row, leaf in enumerate(("q_proj", "k_proj", "v_proj")):
        delta = (
            adapters[f"self_attn.{leaf}.lora_B"] @ adapters[f"self_attn.{leaf}.lora_A"]
        ) * cfg.scaling
        torch.testing.assert_close(
            merged["detector.self_attn.in_proj_weight"][row * E : (row + 1) * E],
            original["detector.self_attn.in_proj_weight"][row * E : (row + 1) * E]
            + delta,
        )
    out_delta = (
        adapters["self_attn.out_proj.lora_B"] @ adapters["self_attn.out_proj.lora_A"]
    ) * cfg.scaling
    torch.testing.assert_close(
        merged["detector.self_attn.out_proj.weight"],
        original["detector.self_attn.out_proj.weight"] + out_delta,
    )
    # LoRA touches weights only.
    assert torch.equal(
        merged["detector.self_attn.in_proj_bias"],
        original["detector.self_attn.in_proj_bias"],
    )
    assert torch.equal(
        merged["detector.self_attn.out_proj.bias"],
        original["detector.self_attn.out_proj.bias"],
    )


def test_freshly_injected_adapters_merge_bit_identically():
    """lora_B is zero-initialised, so an untrained merge must be a no-op."""
    torch.manual_seed(12)
    model = _Host()
    model.requires_grad_(False)
    cfg = _cfg(rank=4)
    inject_adapters(model, cfg)
    adapters = adapter_state_dict(model)
    assert adapters  # the split projections are represented

    E = 16
    base = _mha_base(E)
    base["detector.linear1.weight"] = torch.randn(E, E)
    # The clone's out_proj was never adapted, so drop keys it would need.
    adapters = {k: v for k, v in adapters.items() if not k.startswith("clone_attn")}
    original = {k: v.clone() for k, v in base.items()}
    merged = merge_adapters(base, adapters, cfg)
    for key in original:
        assert torch.equal(merged[key], original[key]), key


def test_merge_with_tied_in_proj_storage_clones_once_and_keeps_the_alias():
    """q/k/v hit one key; an aliased target must be cloned exactly once and
    the alias partner left untouched (regression: the pre-merge storage
    census never saw the clone, so the second slice update raised KeyError).
    """
    E, rank = 16, 2
    torch.manual_seed(13)
    base = _mha_base(E)
    base["detector.tied_copy"] = base["detector.self_attn.in_proj_weight"]
    partner_before = base["detector.tied_copy"].clone()
    cfg = _cfg(rank=rank)
    adapters = {}
    for leaf in ("q_proj", "k_proj", "v_proj"):
        adapters[f"self_attn.{leaf}.lora_A"] = torch.randn(rank, E)
        adapters[f"self_attn.{leaf}.lora_B"] = torch.randn(E, rank)
    merged = merge_adapters(base, adapters, cfg)
    assert torch.equal(merged["detector.tied_copy"], partner_before)
    assert not torch.equal(merged["detector.self_attn.in_proj_weight"], partner_before)


def test_unresolvable_qkv_adapter_is_still_a_hard_error():
    cfg = _cfg(rank=2)
    base = {"detector.something_else.weight": torch.randn(4, 4)}
    adapters = {
        "self_attn.q_proj.lora_A": torch.randn(2, 4),
        "self_attn.q_proj.lora_B": torch.randn(4, 2),
    }
    with pytest.raises(KeyError, match="refusing a partial merge"):
        merge_adapters(base, adapters, cfg)


def test_qkv_delta_shape_is_validated_against_the_fused_slice():
    cfg = _cfg(rank=2)
    E = 8
    base = {"detector.self_attn.in_proj_weight": torch.randn(3 * E, E)}
    adapters = {  # non-square delta cannot be an in_proj slice
        "self_attn.q_proj.lora_A": torch.randn(2, E),
        "self_attn.q_proj.lora_B": torch.randn(E + 1, 2),
    }
    with pytest.raises(ValueError, match="square"):
        merge_adapters(base, adapters, cfg)


def test_touched_keys_match_the_merge_resolution():
    E = 8
    base = {
        "detector.self_attn.in_proj_weight": torch.randn(3 * E, E),
        "detector.self_attn.out_proj.weight": torch.randn(E, E),
        # A free-standing q_proj Linear whose own weight key must win.
        "detector.text.q_proj.weight": torch.randn(E, E),
    }
    adapters = {}
    for path in (
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.out_proj",
        "text.q_proj",
    ):
        adapters[f"{path}.lora_A"] = torch.randn(2, E)
        adapters[f"{path}.lora_B"] = torch.randn(E, 2)
    assert adapter_touched_keys(adapters, base) == {
        "detector.self_attn.in_proj_weight",
        "detector.self_attn.out_proj.weight",
        "detector.text.q_proj.weight",
    }


def test_adapters_on_the_split_projections_are_alive():
    """The whole point of D1: activations must ROUTE THROUGH the wrapped
    projections.  A wrapper the parent never calls has identically-zero
    gradient (the audit's rejected D2 failure mode); prove gradient reaches
    lora_B on all four projections and that a nonzero lora_B changes the
    attention output.
    """
    torch.manual_seed(14)
    model = _Host()
    model.requires_grad_(False)
    inject_adapters(model, _cfg(rank=4))
    attn = model.self_attn
    x = torch.randn(5, 2, 16)

    out, _ = attn(x, x, x, need_weights=False)
    out.sum().backward()
    for leaf in ("q_proj", "k_proj", "v_proj", "out_proj"):
        grad = getattr(attn, leaf).lora_B.grad
        # lora_B is zero-initialised, so lora_A's grad is legitimately zero;
        # lora_B's grad is the liveness signal.
        assert grad is not None and grad.abs().sum() > 0, leaf

    with torch.no_grad():
        baseline, _ = attn(x, x, x, need_weights=False)
        attn.q_proj.lora_B.add_(1.0)
        perturbed, _ = attn(x, x, x, need_weights=False)
    assert not torch.equal(perturbed, baseline)
