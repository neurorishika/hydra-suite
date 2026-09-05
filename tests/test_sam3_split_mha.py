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
    SplitSam3Attention,
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


def test_inject_skips_a_fused_module_with_no_attn_bias_contract():
    """`_CloneLikeMHA` fuses q/k/v but takes no `attn_bias`, so it is NOT the
    SAM3 clone this injector reinterprets -- it stands for any third-party
    fused attention, which must be left alone. (Before the clone-splitting
    pass this test was named "still skips the sam3 clone"; SAM3's real clone
    IS now replaced, and `_Sam3CloneAttention` below is what stands for it.)
    """
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


# ---------------------------------------------------------------------------
# SAM3's own clone MHA (model_misc.MultiheadAttention)
#
# The spike adapted 314 modules to our 206; 100 of the 108 missing ones live
# inside SAM3's own fused attention clone, which `inject_adapters` used to
# skip.  The stub below is a FAITHFUL TRANSCRIPTION of that clone's forward
# from the real source read on the CUDA box
# (`~/sam3_spike/sam3/sam3/model/model_misc.py:230-470,586-733`, verified
# 2026-09-05), restricted to the configuration every one of the 25 clones in a
# live `build_sam3_image_model` actually uses (Vanilla attention, no fa3, no
# activation checkpointing, fused qkv, no bias_k/add_zero_attn -- measured on
# mehek the same day).  It is the oracle these parity tests compare against;
# `import sam3` is unavailable on macOS (triton), so the live-model check is a
# separate `importorskip` test plus an out-of-band run on the CUDA box.
# ---------------------------------------------------------------------------


class _Sam3CloneAttention(nn.Module):
    """Transcription of ``sam3.model.model_misc.MultiheadAttention``."""

    def __init__(self, embed_dim, num_heads, *, dropout=0.0, batch_first=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.kdim = self.vdim = embed_dim
        self._qkv_same_embed_dim = True
        self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.bias_k = self.bias_v = None
        self.add_zero_attn = False
        self.attn_type = "Vanilla"
        self.sparsity = 0.0
        self.use_fa3 = False
        self.use_act_checkpoint = False
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.normal_(self.in_proj_bias, std=0.1)

    def forward(
        self,
        query,
        key,
        value,
        key_padding_mask=None,
        need_weights=False,
        attn_mask=None,
        average_attn_weights=True,
        attn_bias=None,
    ):
        if self.batch_first:
            query, key, value = (x.transpose(1, 0) for x in (query, key, value))
        tgt_len, bsz, embed_dim = query.shape
        src_len = key.shape[0]
        num_heads, head_dim = self.num_heads, self.head_dim
        q, k, v = torch.nn.functional._in_projection_packed(
            query, key, value, self.in_proj_weight, self.in_proj_bias
        )
        if attn_mask is not None and attn_mask.dim() == 2:
            attn_mask = attn_mask.unsqueeze(0)
        q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
        k = k.contiguous().view(src_len, bsz * num_heads, head_dim).transpose(0, 1)
        v = v.contiguous().view(src_len, bsz * num_heads, head_dim).transpose(0, 1)
        if key_padding_mask is not None:
            key_padding_mask = (
                key_padding_mask.view(bsz, 1, 1, src_len)
                .expand(-1, num_heads, -1, -1)
                .reshape(bsz * num_heads, 1, src_len)
            )
            if attn_mask is None:
                attn_mask = key_padding_mask
            elif attn_mask.dtype == torch.bool:
                attn_mask = attn_mask.logical_or(key_padding_mask)
            else:
                attn_mask = attn_mask.masked_fill(key_padding_mask, float("-inf"))
        if attn_mask is not None and attn_mask.dtype == torch.bool:
            new_attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
            new_attn_mask.masked_fill_(attn_mask, float("-inf"))
            attn_mask = new_attn_mask
        dropout_p = self.dropout if self.training else 0.0
        if attn_mask is not None:
            if attn_mask.size(0) == 1:
                attn_mask = attn_mask.unsqueeze(0)
            else:
                attn_mask = attn_mask.view(bsz, num_heads, -1, src_len)
        if attn_bias is not None:
            assert attn_bias.shape == (bsz, num_heads, tgt_len, src_len)
            attn_mask = attn_bias if attn_mask is None else attn_mask + attn_bias
        q = q.view(bsz, num_heads, tgt_len, head_dim)
        k = k.view(bsz, num_heads, src_len, head_dim)
        v = v.view(bsz, num_heads, src_len, head_dim)
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask, dropout_p, False
        )
        attn_output = (
            attn_output.permute(2, 0, 1, 3).contiguous().view(bsz * tgt_len, embed_dim)
        )
        attn_output = torch.nn.functional.linear(
            attn_output, self.out_proj.weight, self.out_proj.bias
        )
        attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))
        weights = None
        if need_weights:
            weights = (q * head_dim**-0.5) @ k.transpose(-2, -1)
            weights = weights.softmax(dim=-1)
            weights = weights.view(bsz, num_heads, tgt_len, src_len)
            if average_attn_weights:
                weights = weights.sum(dim=1) / num_heads
        if self.batch_first:
            return attn_output.transpose(1, 0), weights
        return attn_output, weights


def _clone_pair(embed, heads, *, batch_first, dropout=0.0, seed=0):
    torch.manual_seed(seed)
    clone = _Sam3CloneAttention(embed, heads, dropout=dropout, batch_first=batch_first)
    split = SplitSam3Attention.from_sam3_mha(clone)
    return clone.eval(), split.eval()


# Tolerance rationale: both sides run the SAME `F.scaled_dot_product_attention`
# on the same fp32 CPU inputs, so the only divergence is the in-projection --
# stock packs q/k/v into ONE (3E, E) matmul, the split module runs three (E, E)
# matmuls, which differ only in GEMM tiling/reduction order (~1e-7 relative on
# E=256).  Bitwise equality is therefore unattainable by construction (and even
# less so on CUDA, where SDPA may pick a different backend for a different
# input shape), so these assert a tolerance one order above the observed float32
# noise rather than equality.
_RTOL, _ATOL = 1e-5, 1e-6


def _assert_clone_parity(clone, split, q, k, v, **kwargs):
    ref_out, ref_w = clone(q, k, v, need_weights=True, **kwargs)
    got_out, got_w = split(q, k, v, need_weights=True, **kwargs)
    torch.testing.assert_close(got_out, ref_out, rtol=_RTOL, atol=_ATOL)
    torch.testing.assert_close(got_w, ref_w, rtol=_RTOL, atol=_ATOL)
    # The arity the real call sites rely on (`...(...)[0]`, need_weights left
    # at the clone's False default).
    got_default = split(q, k, v, **kwargs)
    assert len(got_default) == 2 and got_default[1] is None
    torch.testing.assert_close(got_default[0], ref_out, rtol=_RTOL, atol=_ATOL)


def test_sam3_clone_split_weights_are_the_fused_row_slices():
    clone, split = _clone_pair(32, 4, batch_first=False)
    E = 32
    assert torch.equal(split.q_proj.weight, clone.in_proj_weight[0:E])
    assert torch.equal(split.k_proj.weight, clone.in_proj_weight[E : 2 * E])
    assert torch.equal(split.v_proj.weight, clone.in_proj_weight[2 * E : 3 * E])
    assert torch.equal(split.out_proj.weight, clone.out_proj.weight)
    assert torch.equal(split.q_proj.bias, clone.in_proj_bias[0:E])
    assert torch.equal(split.out_proj.bias, clone.out_proj.bias)
    # A retained `in_proj_weight` attribute (even a None-registered one) would
    # make `_parent_uses_weights_directly` skip the new projections.
    assert not hasattr(split, "in_proj_weight")


def test_sam3_clone_parity_decoder_cross_attn_with_attn_bias():
    """`transformer.decoder.layers.*.cross_attn`: seq-first + attn_bias."""
    E, H, L, S, B = 256, 8, 7, 11, 2
    clone, split = _clone_pair(E, H, batch_first=False, dropout=0.1, seed=21)
    torch.manual_seed(22)
    q = torch.randn(L, B, E)
    mem = torch.randn(S, B, E)
    bias = torch.randn(B, H, L, S) * 0.3
    _assert_clone_parity(clone, split, q, mem, mem, attn_bias=bias)


def test_sam3_clone_parity_attn_bias_combined_with_masks():
    E, H, L, S, B = 64, 4, 5, 9, 3
    clone, split = _clone_pair(E, H, batch_first=False, seed=23)
    torch.manual_seed(24)
    q = torch.randn(L, B, E)
    mem = torch.randn(S, B, E)
    bias = torch.randn(B, H, L, S) * 0.3
    kpm = torch.zeros(B, S, dtype=torch.bool)
    kpm[:, -2:] = True
    _assert_clone_parity(
        clone, split, q, mem, mem, attn_mask=torch.randn(L, S) * 0.1, attn_bias=bias
    )
    _assert_clone_parity(clone, split, q, mem, mem, key_padding_mask=kpm)


@pytest.mark.parametrize("mask_kind", ["float2d", "bool2d", "float3d", "bool3d"])
def test_sam3_clone_parity_with_attn_mask(mask_kind):
    E, H, L, B = 64, 4, 5, 3
    clone, split = _clone_pair(E, H, batch_first=False, seed=25)
    torch.manual_seed(26)
    q = torch.randn(L, B, E)
    if mask_kind == "float2d":
        mask = torch.randn(L, L) * 0.1
    elif mask_kind == "bool2d":
        mask = torch.rand(L, L) > 0.7
        mask.fill_diagonal_(False)
    elif mask_kind == "float3d":
        mask = torch.randn(B * H, L, L) * 0.1
    else:
        mask = torch.rand(B * H, L, L) > 0.7
        mask[:, torch.arange(L), torch.arange(L)] = False
    _assert_clone_parity(clone, split, q, q, q, attn_mask=mask)


def test_sam3_clone_parity_encoder_batch_first():
    """`transformer.encoder.layers.*`: the clone with batch_first=True."""
    E, H, B, S = 256, 8, 2, 6
    clone, split = _clone_pair(E, H, batch_first=True, seed=27)
    torch.manual_seed(28)
    x = torch.randn(B, S, E)
    _assert_clone_parity(clone, split, x, x, x)
    assert split(x, x, x)[0].shape == (B, S, E)


def test_sam3_clone_conversion_refuses_unvalidated_variants():
    clone = _Sam3CloneAttention(32, 4)
    clone.use_fa3 = True
    with pytest.raises(ValueError, match="use_fa3"):
        SplitSam3Attention.from_sam3_mha(clone)
    clone = _Sam3CloneAttention(32, 4)
    clone.attn_type = "Xformer"
    with pytest.raises(ValueError, match="attn_type"):
        SplitSam3Attention.from_sam3_mha(clone)
    clone = _Sam3CloneAttention(32, 4)
    clone.use_act_checkpoint = True
    with pytest.raises(ValueError, match="use_act_checkpoint"):
        SplitSam3Attention.from_sam3_mha(clone)
    clone = _Sam3CloneAttention(32, 4)
    clone._qkv_same_embed_dim = False
    with pytest.raises(ValueError, match="separate q/k/v"):
        SplitSam3Attention.from_sam3_mha(clone)
    with pytest.raises(TypeError):
        SplitSam3Attention.from_sam3_mha(nn.Linear(4, 4))


# ---------------------------------------------------------------------------
# Injection scope for the clone
# ---------------------------------------------------------------------------


class _CloneHost(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _Sam3CloneAttention(16, 4)
        self.cross_attn_image = _Sam3CloneAttention(16, 4, batch_first=True)
        self.linear1 = nn.Linear(16, 16)
        self.linear2 = nn.Linear(16, 16)


def test_inject_replaces_the_sam3_clone_and_wraps_its_projections():
    model = _CloneHost()
    n = inject_adapters(model, _cfg())
    for attn in (model.self_attn, model.cross_attn_image):
        assert isinstance(attn, SplitSam3Attention)
        for proj in ("q_proj", "k_proj", "v_proj", "out_proj"):
            assert isinstance(getattr(attn, proj), LoraLinear)
    assert n == 10  # 2 clones x 4 projections + linear1 + linear2


def test_inject_leaves_out_of_scope_clones_untouched():
    model = nn.ModuleDict({"inside": _CloneHost(), "outside": _CloneHost()})
    inject_adapters(model, _cfg(include=("inside",)))
    assert isinstance(model["inside"].self_attn, SplitSam3Attention)
    assert isinstance(model["outside"].self_attn, _Sam3CloneAttention)
    assert not isinstance(model["outside"].self_attn.out_proj, LoraLinear)


def test_a_fused_module_without_attn_bias_is_still_skipped():
    """The conservative fallback: only the clone's exact forward contract is
    reinterpreted.  Any OTHER module that fuses q/k/v is left alone rather
    than reinterpreted through semantics it may not share."""
    model = _Host()  # holds `_CloneLikeMHA`, whose forward takes no attn_bias
    inject_adapters(model, _cfg())
    assert isinstance(model.clone_attn, _CloneLikeMHA)
    assert not isinstance(model.clone_attn.out_proj, LoraLinear)


def test_clone_adapters_are_alive_and_merge_into_in_proj():
    torch.manual_seed(29)
    model = _CloneHost()
    model.requires_grad_(False)
    inject_adapters(model, _cfg(rank=4))
    x = torch.randn(5, 2, 16)
    out, _ = model.self_attn(x, x, x)
    out.sum().backward()
    for leaf in ("q_proj", "k_proj", "v_proj", "out_proj"):
        grad = getattr(model.self_attn, leaf).lora_B.grad
        assert grad is not None and grad.abs().sum() > 0, leaf
    # The clone stores its weights under exactly the stock fused key names, so
    # the existing q/k/v -> in_proj_weight row-slice resolution applies.
    adapters = {
        k: v for k, v in adapter_state_dict(model).items() if k.startswith("self_attn")
    }
    base = _mha_base(16)
    assert adapter_touched_keys(adapters, base) == {
        "detector.self_attn.in_proj_weight",
        "detector.self_attn.out_proj.weight",
    }


# ---------------------------------------------------------------------------
# Geometry-encoder plain Linears (the last 6 of the spike's 314)
# ---------------------------------------------------------------------------


class _GeometryEncoder(nn.Module):
    """Module paths mirror `sam3.model.geometry_encoders` VERBATIM."""

    def __init__(self):
        super().__init__()
        self.points_direct_project = nn.Linear(2, 16)
        self.points_pool_project = nn.Linear(16, 16)
        self.points_pos_enc_project = nn.Linear(16, 16)
        self.boxes_direct_project = nn.Linear(4, 16)
        self.boxes_pos_enc_project = nn.Linear(18, 16)
        # A Conv2d in the real model -- it must NOT be counted or wrapped.
        self.boxes_pool_project = nn.Conv2d(16, 16, 1)
        self.final_proj = nn.Linear(16, 16)
        self.encode = nn.ModuleList([_CloneHost() for _ in range(3)])


def test_geometry_scope_reaches_the_spikes_36_modules():
    from hydra_suite.training.sam3_lora.lora import SUBMODULE_PATHS

    model = nn.ModuleDict({"geometry_encoder": _GeometryEncoder()})
    paths = SUBMODULE_PATHS["adapt_geometry_encoder"]
    assert len(paths) == 6
    cfg = LoraConfig(
        rank=4,
        alpha=8,
        dropout=0.0,
        target_suffixes=TARGET_SUFFIXES,
        include_prefixes=("geometry_encoder",),
        include_module_paths=paths,
    )
    n = inject_adapters(model, cfg)
    # 3 layers x (2 clones x 4 + linear1 + linear2) = 30, plus the 6 plain
    # Linears = the spike checkpoint's 36 `geometry_encoder.*` modules.
    assert n == 36
    geo = model["geometry_encoder"]
    for leaf in (p.split(".")[-1] for p in paths):
        assert isinstance(getattr(geo, leaf), LoraLinear), leaf
    assert isinstance(geo.boxes_pool_project, nn.Conv2d)


def test_geometry_paths_are_exactly_the_spike_set():
    from hydra_suite.training.sam3_lora.lora import SUBMODULE_PATHS

    assert set(SUBMODULE_PATHS["adapt_geometry_encoder"]) == {
        "geometry_encoder.boxes_direct_project",
        "geometry_encoder.boxes_pos_enc_project",
        "geometry_encoder.points_direct_project",
        "geometry_encoder.points_pool_project",
        "geometry_encoder.points_pos_enc_project",
        "geometry_encoder.final_proj",
    }


def test_a_torch_mha_subclass_is_skipped_loudly():
    """An older vendored SAM3 tree shipped the clone as a torch-MHA subclass.
    It matches neither pass-1 branch (exact-type rejects it; the clone
    duck-type rejects every nn.MultiheadAttention instance on purpose), so the
    skip must be announced here rather than surfacing later as an
    estimator-drift refusal that names the wrong cause."""

    class _MultiheadAttentionWrapper(nn.MultiheadAttention):
        pass

    model = nn.Module()
    model.attn = _MultiheadAttentionWrapper(16, 4)
    model.linear1 = nn.Linear(16, 16)
    with pytest.warns(RuntimeWarning, match="subclass of nn.MultiheadAttention"):
        n = inject_adapters(model, _cfg())
    assert type(model.attn) is _MultiheadAttentionWrapper
    assert not isinstance(model.attn.out_proj, LoraLinear)
    assert n == 1  # linear1 only


# ---------------------------------------------------------------------------
# The live gate: 308 after the clone pass, 314 with the geometry Linears.
# 314 is SPIKE parity and includes the two `adapt_scoring_head` modules; the
# production ceiling today is 312, because `cli.py` still refuses that flag.
# Skipped everywhere `sam3` cannot be imported (macOS: triton).  The numbers
# below were also verified out-of-band on the CUDA box -- see the task report.
# ---------------------------------------------------------------------------


def test_live_model_reaches_the_spike_module_counts():
    pytest.importorskip("sam3")
    from sam3.model_builder import build_sam3_image_model

    from hydra_suite.training.sam3_lora.lora import (
        SUBMODULE_PATHS,
        SUBMODULE_PREFIXES,
    )

    prefixes = tuple(
        p
        for flag, pref in SUBMODULE_PREFIXES.items()
        if flag != "adapt_text_encoder"  # OFF, as the spike ran it
        for p in pref
    )
    scoring = SUBMODULE_PATHS["adapt_scoring_head"]
    geometry = SUBMODULE_PATHS["adapt_geometry_encoder"]

    def _count(paths):
        model = build_sam3_image_model(eval_mode=False)
        model.requires_grad_(False)
        return inject_adapters(
            model,
            LoraConfig(
                rank=4,
                alpha=8,
                dropout=0.0,
                target_suffixes=TARGET_SUFFIXES,
                include_prefixes=prefixes,
                include_module_paths=paths,
            ),
        )

    assert _count(scoring) == 308
    assert _count(scoring + geometry) == 314


def test_an_unreproducible_clone_is_skipped_loudly_not_approximated():
    """The plan's ruling: a site whose forward cannot be reproduced faithfully
    is SKIPPED and RECORDED, never approximated -- and never fatal to the ~100
    sites that are reproducible.  The wrapped count must exclude it, which is
    what makes the `cli.py` estimator-drift check the backstop."""
    model = _CloneHost()
    model.self_attn.use_fa3 = True
    with pytest.warns(RuntimeWarning, match="left unadapted"):
        n = inject_adapters(model, _cfg())
    assert isinstance(model.self_attn, _Sam3CloneAttention)
    assert not isinstance(model.self_attn.out_proj, LoraLinear)
    assert isinstance(model.cross_attn_image, SplitSam3Attention)
    # 1 reproducible clone x 4 projections + linear1 + linear2; the fa3 clone
    # contributes nothing.
    assert n == 6
