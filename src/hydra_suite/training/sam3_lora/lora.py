"""Low-rank adapters: inject, extract, merge.

Deliberately free of any SAM3 import so the whole seam is testable on a toy
nn.Module without a GPU or the licence-gated checkpoint.
"""

from __future__ import annotations

import inspect
import warnings
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class LoraConfig:
    rank: int
    alpha: int
    dropout: float
    target_suffixes: tuple[str, ...]
    # The six adapt_* flags select submodules by PREFIX, not suffix. Without
    # these the flags cannot be expressed and every matching Linear is adapted
    # -- including the text encoder we deliberately freeze.
    include_prefixes: tuple[str, ...] = ()
    exclude_prefixes: tuple[str, ...] = ()
    # Exact dotted module paths adapted regardless of their leaf name. Some
    # spike-trained Linears (`dot_prod_scoring.prompt_proj`, and Task 2's
    # geometry projections) match no `target_suffixes` entry, and broadening
    # the suffix list to reach them would change what every OTHER scope wraps.
    include_module_paths: tuple[str, ...] = ()

    @property
    def scaling(self) -> float:
        return float(self.alpha) / float(self.rank)


# Dotted module-path prefixes for each adapt_* flag, MEASURED against a live
# build_sam3_image_model (293 nn.Linear modules). Two natural guesses are WRONG
# and would match nothing, silently turning their flag into a no-op: the
# geometry encoder is top-level `geometry_encoder`, NOT
# `backbone.geometry_encoder`, and the mask decoder is `segmentation_head`, NOT
# `mask_decoder`.
#
# Adapted-module counts per flag AFTER both attention-splitting passes
# (re-measured 2026-09-05 on a live build; every fused attention module --
# torch's own and SAM3's `model_misc` clone -- contributes four wrapped
# projections):
#   adapt_vision_encoder             128
#   adapt_text_encoder               144   (24 torch MHA in the language
#                                          backbone)
#   adapt_detr_encoder                60   (12 -> +12 clones)
#   adapt_detr_decoder                84   (60 -> +6 clones)
#   adapt_geometry_encoder            36   (6 -> +6 clones, +6 exact paths)
#   adapt_mask_decoder                 4   (0 -> +1 clone)
#   adapt_scoring_head                 2   (exact paths, opt-in, default OFF)
# With the text encoder OFF, as the research spike ran it, that totals the
# spike checkpoint's 314 adapted modules exactly (308 before the six geometry
# paths are added). NOTE the production ceiling is 312, not 314: the last two
# are `adapt_scoring_head`, which `cli.py` still refuses by name because it
# has no measured LORA_PARAMS_PER_RANK coefficient (deliberate -- landing one
# would un-refuse an unvalidated scope). Full 314 parity with the spike is
# reachable only once that flag is validated.
#
# `dot_prod_scoring` (the text/vision similarity head that emits detection
# confidence) previously carried a comment here saying it was "covered by NO
# flag, deliberately". RECONSIDERED 2026-09-05: the finetune quality audit
# (docs/superpowers/specs/2026-09-05-sam3-finetune-quality-audit.md, N1)
# compared our adapted set against the research spike's checkpoint and found
# `dot_prod_scoring.prompt_proj` has the LARGEST lora_B norm (0.174) of every
# module the spike adapts and we do not, with `hs_proj` (0.097) also well
# above the spike-only median. The old reasoning assumed the spike left the
# head alone; it did not. It is now reachable via the opt-in
# `adapt_scoring_head` flag, DEFAULT OFF: a large lora_B norm proves the head
# MOVED, not that it moved toward better precision, so the flag stays off
# until the paired retrain ("RUN A") validates it.
SUBMODULE_PREFIXES: dict[str, tuple[str, ...]] = {
    "adapt_vision_encoder": ("backbone.vision_backbone",),
    "adapt_text_encoder": ("backbone.language_backbone",),
    "adapt_geometry_encoder": ("geometry_encoder",),
    "adapt_detr_encoder": ("transformer.encoder",),
    "adapt_detr_decoder": ("transformer.decoder",),
    "adapt_mask_decoder": ("segmentation_head",),
}


# Exact dotted module paths per adapt_* flag, for spike-trained Linears whose
# leaf name is in no TARGET_SUFFIXES entry. Keyed by flag exactly like
# SUBMODULE_PREFIXES, and unioned with it by `lora_config_from_params`.
#
# DEVIATION from the plan's stated interface, justified here: the plan also
# asked for a `SUBMODULE_PREFIXES["adapt_scoring_head"] = ("dot_prod_scoring",)`
# entry alongside this allowlist. It is deliberately absent. The head holds 4
# Linears but the spike trained exactly 2; a prefix entry ALSO wraps any of the
# other two whose leaf happens to be a target suffix (`proj` is one), silently
# exceeding the spike surface and breaking the module arithmetic (316 vs 314).
# An exact-path list makes "exactly 2" unconditional. Cost of the deviation:
# the prefix-only sentinel in `inject_adapters` must now account for
# path-only scopes (it does -- see `_scoped`), and a future scope that really
# is prefix-shaped still belongs in SUBMODULE_PREFIXES.
#
# `sam3_image.py:83-85` builds `instance_dot_prod_scoring = deepcopy(...)`.
# Exact-path matching excludes it, and that exclusion IS intended parity: the
# spike checkpoint carries no adapters for it.
#
# `adapt_geometry_encoder` appears in BOTH dicts: its attention layers are
# prefix-shaped (`geometry_encoder.encode.*`) while these six projections are
# plain Linears whose leaf names match no target suffix. Note in particular
# that suffix matching is exact on the LAST dotted component, so the existing
# `proj` entry does not reach `final_proj`; broadening the suffix list instead
# would silently change what every other scope wraps. The six are measured
# against the spike checkpoint's `geometry_encoder.*` keys (mehek
# `~/sam3_spike/out/fold_all_r16/best_lora_weights.pt`, 2026-09-05) and
# `geometry_encoders.py:543-566`. `boxes_pool_project` is deliberately absent:
# it is a Conv2d, not a Linear, and the spike did not adapt it.
SUBMODULE_PATHS: dict[str, tuple[str, ...]] = {
    "adapt_scoring_head": (
        "dot_prod_scoring.prompt_proj",
        "dot_prod_scoring.hs_proj",
    ),
    "adapt_geometry_encoder": (
        "geometry_encoder.boxes_direct_project",
        "geometry_encoder.boxes_pos_enc_project",
        "geometry_encoder.points_direct_project",
        "geometry_encoder.points_pool_project",
        "geometry_encoder.points_pos_enc_project",
        "geometry_encoder.final_proj",
    ),
}


# The Linear leaf names LoRA attaches to, across SAM3's ViT, CLIP-style text
# tower and DETR transformer.
TARGET_SUFFIXES: tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "out_proj",
    "qkv",
    "proj",
    "fc1",
    "fc2",
    "c_fc",
    "c_proj",
    "linear1",
    "linear2",
)


def lora_config_from_params(params) -> "LoraConfig":
    """Turn the six adapt_* booleans into an include-prefix list.

    Always return the explicit declared-prefix union. The empty-prefix sentinel
    means "everything" to the generic injector and would include unbudgeted
    modules such as ``dot_prod_scoring``.

    Path-keyed scopes (``SUBMODULE_PATHS``) are read with a ``False`` default:
    they are opt-in additions, so a params object predating the flag means
    "off", whereas a missing prefix flag is still a programming error.
    """
    enabled = [f for f in SUBMODULE_PREFIXES if getattr(params, f)]
    enabled_paths = [f for f in SUBMODULE_PATHS if getattr(params, f, False)]
    if not enabled and not enabled_paths:
        raise ValueError("at least one SAM3 LoRA adapter scope must be enabled")
    include = tuple(pref for flag in enabled for pref in SUBMODULE_PREFIXES[flag])
    paths = tuple(path for flag in enabled_paths for path in SUBMODULE_PATHS[flag])
    return LoraConfig(
        rank=params.rank,
        alpha=params.alpha,
        dropout=params.dropout,
        target_suffixes=TARGET_SUFFIXES,
        include_prefixes=include,
        include_module_paths=paths,
    )


class LoraLinear(nn.Module):
    """Frozen base Linear plus a trainable rank-r branch."""

    def __init__(self, base: nn.Linear, cfg: LoraConfig) -> None:
        super().__init__()
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        self.lora_A = nn.Parameter(torch.zeros(cfg.rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, cfg.rank))
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        # lora_B stays zero: an untrained adapter must be an exact no-op.
        self.dropout = nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity()
        self.scaling = cfg.scaling

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
        return self.base(x) + delta * self.scaling


class SplitMultiheadAttention(nn.Module):
    """Eager, split-projection replacement for torch ``nn.MultiheadAttention``.

    Torch's MHA never CALLS its projections -- it feeds ``in_proj_weight``
    and ``out_proj.weight`` into ``F.multi_head_attention_forward`` -- so a
    ``LoraLinear`` wrapped onto its Linears is mathematically dead (its
    forward never runs).  The empirically validated SAM3 spike solved this by
    replacing the whole module with separate ``q_proj/k_proj/v_proj/out_proj``
    Linears initialised from the fused ``in_proj`` row slices ``[0:E]``,
    ``[E:2E]``, ``[2E:3E]``; the four projections are then real call sites
    the adapters can attach to.  This is the same mechanism in this repo's
    idiom.

    The forward mirrors torch's *eager* ``F.multi_head_attention_forward``
    math (q pre-scaled before the matmul; key_padding_mask merged into a
    float additive mask) so parity with stock is tight, and covers exactly
    the argument shapes SAM3's call sites use: decoder ``self_attn``
    (seq-first, ``attn_mask``), decoder ``ca_text`` (seq-first,
    ``key_padding_mask``) and the text encoder (``batch_first``, causal
    ``attn_mask``, ``need_weights=False``).  MHA variants SAM3 never builds
    (separate kdim/vdim, ``bias_k``/``bias_v``, ``add_zero_attn``) are
    refused loudly at conversion time rather than approximated.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        *,
        dropout: float = 0.0,
        bias: bool = True,
        batch_first: bool = False,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = float(dropout)
        self.batch_first = batch_first
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    @classmethod
    def from_torch_mha(cls, mha: nn.MultiheadAttention) -> "SplitMultiheadAttention":
        """Build from a torch MHA, refusing any shape outside SAM3's usage."""
        if not isinstance(mha, nn.MultiheadAttention):
            raise TypeError(
                "SplitMultiheadAttention only replaces torch nn.MultiheadAttention; "
                f"got {type(mha).__name__}"
            )
        if not mha._qkv_same_embed_dim or mha.in_proj_weight is None:
            raise ValueError(
                "in-scope nn.MultiheadAttention has separate q/k/v projection "
                "weights (kdim/vdim != embed_dim); this is outside the "
                "empirically validated SAM3 surface -- refusing to adapt it"
            )
        if mha.bias_k is not None or mha.bias_v is not None or mha.add_zero_attn:
            raise ValueError(
                "in-scope nn.MultiheadAttention uses bias_k/bias_v/add_zero_attn; "
                "SAM3 never builds this variant -- refusing to adapt it"
            )
        has_bias = mha.in_proj_bias is not None
        if has_bias != (mha.out_proj.bias is not None):
            raise ValueError(
                "nn.MultiheadAttention with mismatched in/out projection bias "
                "is outside the validated surface"
            )
        split = cls(
            mha.embed_dim,
            mha.num_heads,
            dropout=mha.dropout,
            bias=has_bias,
            batch_first=mha.batch_first,
        )
        embed = mha.embed_dim
        with torch.no_grad():
            weight = mha.in_proj_weight
            split.q_proj.weight.copy_(weight[0:embed])
            split.k_proj.weight.copy_(weight[embed : 2 * embed])
            split.v_proj.weight.copy_(weight[2 * embed : 3 * embed])
            split.out_proj.weight.copy_(mha.out_proj.weight)
            if has_bias:
                in_bias = mha.in_proj_bias
                split.q_proj.bias.copy_(in_bias[0:embed])
                split.k_proj.bias.copy_(in_bias[embed : 2 * embed])
                split.v_proj.bias.copy_(in_bias[2 * embed : 3 * embed])
                split.out_proj.bias.copy_(mha.out_proj.bias)
        split.to(dtype=weight.dtype, device=weight.device)
        return split

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        need_weights: bool = True,
        attn_mask: torch.Tensor | None = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # ``is_causal`` in torch's API is a kernel hint that requires the
        # matching mask alongside it; SAM3 never passes it. Accepting and
        # ignoring it (like the eager path does when a mask is present) is
        # only safe with a mask, so refuse the maskless-hint combination.
        if is_causal and attn_mask is None:
            raise ValueError(
                "is_causal=True without attn_mask is not supported by the "
                "SplitMultiheadAttention eager path"
            )
        if self.batch_first:
            bsz, tgt_len, _ = query.shape
            src_len = key.shape[1]
        else:
            tgt_len, bsz, _ = query.shape
            src_len = key.shape[0]
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        # Projections: the LoraLinear call sites this module exists for.
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # (bsz, len, E) -> (bsz, heads, len, head_dim)
        q = q.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, src_len, self.num_heads, self.head_dim).transpose(1, 2)

        merged_mask = self._merged_float_mask(
            attn_mask, key_padding_mask, bsz, tgt_len, src_len, q.dtype, q.device
        )

        # Mirror torch's eager order: scale q BEFORE the matmul.
        q = q * (1.0 / (self.head_dim**0.5))
        attn_weights = torch.matmul(q, k.transpose(-2, -1))
        if merged_mask is not None:
            attn_weights = attn_weights + merged_mask
        attn_weights = torch.softmax(attn_weights, dim=-1)
        if self.dropout > 0.0:
            attn_probs = torch.nn.functional.dropout(
                attn_weights, p=self.dropout, training=self.training
            )
        else:
            attn_probs = attn_weights

        attn_output = torch.matmul(attn_probs, v)
        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(bsz, tgt_len, self.embed_dim)
        )
        attn_output = self.out_proj(attn_output)
        if not self.batch_first:
            attn_output = attn_output.transpose(0, 1)

        if need_weights:
            weights = attn_weights
            if average_attn_weights:
                weights = weights.mean(dim=1)
            return attn_output, weights
        return attn_output, None

    def _merged_float_mask(
        self,
        attn_mask: torch.Tensor | None,
        key_padding_mask: torch.Tensor | None,
        bsz: int,
        tgt_len: int,
        src_len: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor | None:
        """Merge both masks into one float additive (bsz, H, L, S) mask."""

        def _to_float(mask: torch.Tensor) -> torch.Tensor:
            if mask.dtype == torch.bool:
                out = torch.zeros(mask.shape, dtype=dtype, device=device)
                return out.masked_fill(mask, float("-inf"))
            return mask.to(dtype)

        merged: torch.Tensor | None = None
        if attn_mask is not None:
            mask = _to_float(attn_mask)
            if mask.dim() == 2:
                mask = mask.view(1, 1, tgt_len, src_len)
            elif mask.dim() == 3:
                if mask.shape[0] == bsz * self.num_heads:
                    mask = mask.view(bsz, self.num_heads, tgt_len, src_len)
                else:
                    raise ValueError(
                        "3-D attn_mask must have shape "
                        f"(bsz*num_heads, tgt_len, src_len); got {tuple(mask.shape)}"
                    )
            else:
                raise ValueError(f"attn_mask must be 2-D or 3-D; got {mask.dim()}-D")
            merged = mask
        if key_padding_mask is not None:
            kpm = _to_float(key_padding_mask).view(bsz, 1, 1, src_len)
            merged = kpm if merged is None else merged + kpm
        return merged


def is_sam3_clone_attention(module: nn.Module) -> bool:
    """Duck-type SAM3's own fused attention (``model_misc.MultiheadAttention``).

    This module stays deliberately free of any SAM3 import, so the class
    cannot be named.  A lazy ``import sam3`` behind a function was the
    alternative and was rejected: the same clone ships under at least two
    import paths on the training boxes (``sam3.model.model_misc`` and the
    vendored ``ultralytics.models.sam.sam3.model_misc``), so class identity
    against one of them would silently miss the other -- the exact
    zero-match-scope failure mode the rest of this file hard-errors on.

    The discriminator is the forward CONTRACT, not the shape: a fused
    ``in_proj_weight`` plus an ``attn_bias`` keyword.  ``attn_bias`` is the
    clone's own extension (torch's MHA has no such parameter, nor does any
    subclass of it), and every real call site passes it
    (``decoder.py:892,925``).  Anything else that merely fuses q/k/v is left
    alone rather than reinterpreted through semantics it may not share.
    """
    if isinstance(module, nn.MultiheadAttention):
        return False
    if not isinstance(getattr(module, "out_proj", None), nn.Linear):
        return False
    if getattr(module, "in_proj_weight", None) is None:
        # A separate-qkv clone has `in_proj_weight` registered as None. It is
        # outside the validated surface either way; report it as not-a-clone
        # so injection skips it instead of hard-erroring on a live model.
        return False
    if any(not hasattr(module, a) for a in ("embed_dim", "num_heads", "batch_first")):
        return False
    try:
        parameters = inspect.signature(module.forward).parameters
    except (TypeError, ValueError):  # pragma: no cover - exotic callables
        return False
    return "attn_bias" in parameters and "key_padding_mask" in parameters


class SplitSam3Attention(nn.Module):
    """Split-projection replacement for SAM3's OWN fused attention clone.

    Same motivation as :class:`SplitMultiheadAttention` -- the clone feeds
    ``in_proj_weight`` and ``out_proj.weight`` into a functional kernel, so a
    ``LoraLinear`` wrapped onto its projections would never run -- but the
    clone is NOT torch's MHA and cannot share that class:

    * its forward takes an extra ``attn_bias`` kwarg and has no ``is_causal``;
    * ``need_weights`` defaults to ``False`` (torch defaults to ``True``);
    * its Vanilla path calls ``F.scaled_dot_product_attention``, not torch's
      eager math, so the two are equal only to float tolerance;
    * its returned attention weights are computed OUTSIDE the SDPA call, from
      unmasked logits.

    The forward below is a transcription of
    ``sam3.model.model_misc.multi_head_attention_forward`` (read on the CUDA
    box, 2026-09-05) restricted to the configuration all 25 clones in a live
    ``build_sam3_image_model`` actually use; every other configuration is
    refused at conversion time rather than approximated.

    Reaching these sites is the bulk of the spike's adapter surface: 100 of
    the 108 modules the spike adapted and we did not are clone projections
    (`docs/superpowers/specs/2026-09-05-sam3-finetune-quality-audit.md`, N1).
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        *,
        dropout: float = 0.0,
        bias: bool = True,
        batch_first: bool = False,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = float(dropout)
        self.batch_first = batch_first
        # NOTE: no `in_proj_weight` attribute, not even a None-registered one.
        # `_parent_uses_weights_directly` keys on `hasattr(parent,
        # "in_proj_weight")`, so retaining it would make the injector skip the
        # very projections this class exists to expose.
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    @classmethod
    def from_sam3_mha(cls, mha: nn.Module) -> "SplitSam3Attention":
        """Build from a SAM3 clone, refusing any configuration it never uses."""
        if not is_sam3_clone_attention(mha):
            raise TypeError(
                "SplitSam3Attention only replaces SAM3's fused attention clone "
                f"(in_proj_weight + attn_bias forward); got {type(mha).__name__}"
            )
        if not getattr(mha, "_qkv_same_embed_dim", True):
            raise ValueError(
                "in-scope SAM3 attention clone has separate q/k/v projection "
                "weights; this is outside the validated surface"
            )
        if (
            getattr(mha, "bias_k", None) is not None
            or getattr(mha, "bias_v", None) is not None
        ):
            raise ValueError(
                "in-scope SAM3 attention clone uses bias_k/bias_v; refusing"
            )
        if getattr(mha, "add_zero_attn", False):
            raise ValueError(
                "in-scope SAM3 attention clone uses add_zero_attn; refusing"
            )
        attn_type = getattr(mha, "attn_type", "Vanilla")
        # AttentionType is a SAM3 enum this module must not import; compare on
        # its name, which is what the enum's __str__ yields too.
        if getattr(attn_type, "name", str(attn_type)) != "Vanilla":
            raise ValueError(
                f"in-scope SAM3 attention clone has attn_type {attn_type!r}; only "
                "the Vanilla (SDPA) path is reproduced -- refusing to adapt it"
            )
        if float(getattr(mha, "sparsity", 0.0)) != 0.0:
            raise ValueError("in-scope SAM3 attention clone is sparse; refusing")
        if getattr(mha, "use_fa3", False):
            raise ValueError(
                "in-scope SAM3 attention clone has use_fa3=True; the FlashAttention-3 "
                "kernel is not reproduced here -- refusing to adapt it"
            )
        if getattr(mha, "use_act_checkpoint", False):
            raise ValueError(
                "in-scope SAM3 attention clone has use_act_checkpoint=True; the "
                "replacement does not re-enter torch.utils.checkpoint -- refusing"
            )
        in_bias = getattr(mha, "in_proj_bias", None)
        has_bias = in_bias is not None
        if has_bias != (mha.out_proj.bias is not None):
            raise ValueError(
                "SAM3 attention clone with mismatched in/out projection bias "
                "is outside the validated surface"
            )
        split = cls(
            int(mha.embed_dim),
            int(mha.num_heads),
            dropout=float(getattr(mha, "dropout", 0.0)),
            bias=has_bias,
            batch_first=bool(mha.batch_first),
        )
        embed = int(mha.embed_dim)
        with torch.no_grad():
            weight = mha.in_proj_weight
            split.q_proj.weight.copy_(weight[0:embed])
            split.k_proj.weight.copy_(weight[embed : 2 * embed])
            split.v_proj.weight.copy_(weight[2 * embed : 3 * embed])
            split.out_proj.weight.copy_(mha.out_proj.weight)
            if has_bias:
                split.q_proj.bias.copy_(in_bias[0:embed])
                split.k_proj.bias.copy_(in_bias[embed : 2 * embed])
                split.v_proj.bias.copy_(in_bias[2 * embed : 3 * embed])
                split.out_proj.bias.copy_(mha.out_proj.bias)
        split.to(dtype=weight.dtype, device=weight.device)
        return split

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        need_weights: bool = False,
        attn_mask: torch.Tensor | None = None,
        average_attn_weights: bool = True,
        attn_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if query.dim() != 3:
            raise ValueError(
                "SplitSam3Attention requires batched 3-D query/key/value; the "
                "clone's unbatched path is dead code in every SAM3 call site"
            )
        if self.batch_first:
            query, key, value = (x.transpose(1, 0) for x in (query, key, value))
        tgt_len, bsz, _ = query.shape
        src_len = key.shape[0]
        heads, head_dim = self.num_heads, self.head_dim

        # Projections: the LoraLinear call sites this module exists for. Stock
        # runs one packed (3E, E) matmul; three (E, E) matmuls are the same
        # math up to GEMM reduction order (~1e-7 relative in float32).
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        if attn_mask is not None and attn_mask.dim() == 2:
            attn_mask = attn_mask.unsqueeze(0)
        q = q.contiguous().view(tgt_len, bsz * heads, head_dim).transpose(0, 1)
        k = k.contiguous().view(src_len, bsz * heads, head_dim).transpose(0, 1)
        v = v.contiguous().view(src_len, bsz * heads, head_dim).transpose(0, 1)

        if key_padding_mask is not None:
            key_padding_mask = (
                key_padding_mask.view(bsz, 1, 1, src_len)
                .expand(-1, heads, -1, -1)
                .reshape(bsz * heads, 1, src_len)
            )
            if attn_mask is None:
                attn_mask = key_padding_mask
            elif attn_mask.dtype == torch.bool:
                attn_mask = attn_mask.logical_or(key_padding_mask)
            else:
                attn_mask = attn_mask.masked_fill(key_padding_mask, float("-inf"))
        if attn_mask is not None and attn_mask.dtype == torch.bool:
            float_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
            float_mask.masked_fill_(attn_mask, float("-inf"))
            attn_mask = float_mask
        if attn_mask is not None:
            if attn_mask.size(0) == 1:
                attn_mask = attn_mask.unsqueeze(0)
            else:
                attn_mask = attn_mask.view(bsz, heads, -1, src_len)
        if attn_bias is not None:
            expected = (bsz, heads, tgt_len, src_len)
            if tuple(attn_bias.shape) != expected:
                raise ValueError(
                    f"expecting attn_bias shape of {expected}, but got "
                    f"{tuple(attn_bias.shape)}"
                )
            attn_mask = attn_bias if attn_mask is None else attn_mask + attn_bias

        q = q.view(bsz, heads, tgt_len, head_dim)
        k = k.view(bsz, heads, src_len, head_dim)
        v = v.view(bsz, heads, src_len, head_dim)
        dropout_p = self.dropout if self.training else 0.0
        # DEVIATION from the clone: it flips the three global
        # torch.backends.cuda.enable_*_sdp switches on immediately before this
        # call. Those are process-wide mutable state, and turning them on is
        # already the torch default; reproducing the side effect would let an
        # adapter injection silently re-enable a backend the caller disabled.
        # Cost: if a caller HAS disabled a backend, SDPA may dispatch to a
        # different kernel here than in stock, which moves the result by
        # kernel-level float noise only (the math is identical).
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask, dropout_p, False
        )
        attn_output = (
            attn_output.permute(2, 0, 1, 3).contiguous().view(bsz * tgt_len, -1)
        )
        attn_output = self.out_proj(attn_output)
        attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))

        weights = None
        if need_weights:
            # Faithful to the clone, INCLUDING its quirk: these weights are
            # recomputed from unmasked logits and are not the distribution
            # SDPA actually used. Every SAM3 call site indexes [0], so this
            # branch exists for parity, not for consumption.
            weights = ((q * head_dim**-0.5) @ k.transpose(-2, -1)).softmax(dim=-1)
            weights = weights.view(bsz, heads, tgt_len, src_len)
            if average_attn_weights:
                weights = weights.sum(dim=1) / heads
        if self.batch_first:
            return attn_output.transpose(1, 0), weights
        return attn_output, weights


def inject_adapters(model: nn.Module, cfg: LoraConfig) -> int:
    """Attach LoRA to every in-scope Linear the model actually calls.

    Two passes. First, every in-scope fused attention module is replaced with
    a split-projection equivalent so its q/k/v/out become real ``nn.Linear``
    call sites: torch ``nn.MultiheadAttention`` becomes
    :class:`SplitMultiheadAttention`, and SAM3's own ``model_misc`` clone
    (duck-typed by :func:`is_sam3_clone_attention`) becomes
    :class:`SplitSam3Attention`.  Second, every ``nn.Linear`` whose dotted
    path ends in a target suffix -- or which is named exactly in
    ``cfg.include_module_paths`` -- is wrapped in :class:`LoraLinear`.
    Returns the number of wrapped Linears.

    The clone was previously skipped as "outside the empirically validated
    surface".  The spike's checkpoint disproved that: 100 of the 108 modules
    it adapted and we did not are clone projections, and 97 of the 108
    carry a trained (nonzero) ``lora_B``
    (`docs/superpowers/specs/2026-09-05-sam3-finetune-quality-audit.md`, N1).
    In the spike's OLDER vendored tree the clone was a thin subclass of
    torch's MHA, which is how its isinstance-based replacement reached them;
    the current tree renamed a genuine ``nn.Module`` clone to the same alias.
    """

    def _excluded(name: str) -> bool:
        return bool(cfg.exclude_prefixes) and name.startswith(cfg.exclude_prefixes)

    def _scoped(name: str) -> bool:
        if _excluded(name):
            return False
        if cfg.include_prefixes:
            return name.startswith(cfg.include_prefixes)
        # The "no includes at all" sentinel still means "the whole model" (the
        # raw-LoraConfig callers rely on it), but a config that declares ONLY
        # exact module paths is a real scope: falling back to everything there
        # would adapt the entire model off a two-module flag.
        return not cfg.include_module_paths

    # Pass 1: split in-scope fused attention. torch MHA is matched by EXACT
    # type so a subclass with a different forward can never be silently
    # reinterpreted through torch-MHA semantics; SAM3's clone is matched by
    # its forward contract (see `is_sam3_clone_attention`) because the class
    # itself cannot be named from this deliberately sam3-free module.
    fused = [
        (name, mod, type(mod) is nn.MultiheadAttention)
        for name, mod in model.named_modules()
        if _scoped(name)
        and (type(mod) is nn.MultiheadAttention or is_sam3_clone_attention(mod))
    ]
    # A torch-MHA SUBCLASS falls through both matchers: exact-type matching
    # rejects it, and `is_sam3_clone_attention` rejects every
    # `nn.MultiheadAttention` instance on purpose (its forward semantics are
    # the subclass's, not the clone's). That is the correct conservative
    # outcome, but it must not be a SILENT one: the research spike's OLDER
    # vendored SAM3 tree shipped its clone as exactly such a subclass
    # (`MultiheadAttentionWrapper(nn.MultiheadAttention)`), so on that tree
    # ~100 projections would go unadapted and the only symptom would be the
    # estimator-drift refusal in `cli.py`, whose message names the wrong
    # cause. Say it here, where the cause is known.
    for name, mod in model.named_modules():
        if (
            isinstance(mod, nn.MultiheadAttention)
            and type(mod) is not nn.MultiheadAttention
            and _scoped(name)
        ):
            warnings.warn(
                f"in-scope attention module {name!r} is a subclass of "
                f"nn.MultiheadAttention ({type(mod).__name__}), not the class "
                "itself, and does not carry SAM3's attn_bias forward contract; "
                "its fused projections are left unadapted because its forward "
                "semantics are unknown. If this is an older vendored SAM3 tree "
                "whose clone subclasses torch's MHA, the adapter surface will "
                "be short by four projections per site.",
                RuntimeWarning,
                stacklevel=2,
            )
    for name, mod, is_torch_mha in fused:
        *parent_path, attr = name.split(".")
        parent = model
        for part in parent_path:
            parent = getattr(parent, part)
        if is_torch_mha:
            setattr(parent, attr, SplitMultiheadAttention.from_torch_mha(mod))
            continue
        try:
            replacement = SplitSam3Attention.from_sam3_mha(mod)
        except ValueError as exc:
            # The ruling for a site whose forward cannot be reproduced
            # faithfully: SKIP IT and say so, loudly, with the module named.
            # Shipping an attention module that is only "close enough" would
            # poison training in a way no test catches. Not silent -- and not
            # fatal either, because one exotic attention configuration
            # elsewhere in a future SAM3 build must not block the ~100 sites
            # that ARE reproducible.
            warnings.warn(
                f"SAM3 attention clone {name!r} left unadapted: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        setattr(parent, attr, replacement)

    # Rebuilt AFTER pass 1 so the parent guard sees the post-replacement tree
    # (a replaced MHA no longer has in_proj_weight; its projections are live).
    by_name = dict(model.named_modules())

    def _parent_uses_weights_directly(name: str) -> bool:
        """True when wrapping this Linear would break its parent.

        SAM3's MultiheadAttention (``model_misc.py``) does not CALL
        ``self.out_proj``; it passes ``self.out_proj.weight`` and
        ``.bias`` into a functional attention kernel. A ``LoraLinear``
        wrapper exposes neither (its base lives at ``.base``), so wrapping
        raised ``AttributeError: 'LoraLinear' object has no attribute
        'weight'`` on the first forward. ``in_proj_weight`` is the reliable
        marker for that fused-attention shape -- both torch's own
        ``nn.MultiheadAttention`` and SAM3's clone carry it -- and the
        adapter would be dead weight there regardless, since the parent
        never routes activations through the module.

        After pass 1, in-scope fused attention no longer trips this guard
        (both torch MHAs and SAM3 clones have been replaced, and neither
        replacement keeps an ``in_proj_weight`` attribute).  It still
        protects OUT-of-scope fused attention whose ``out_proj`` Linear would
        otherwise match a target suffix, and any in-scope clone pass 1
        refused and skipped.
        """
        parent_path = name.rsplit(".", 1)[0] if "." in name else ""
        parent = by_name.get(parent_path)
        return parent is not None and hasattr(parent, "in_proj_weight")

    targets = [
        (name, mod)
        for name, mod in model.named_modules()
        if isinstance(mod, nn.Linear)
        and (
            (name in cfg.include_module_paths and not _excluded(name))
            or (name.split(".")[-1] in cfg.target_suffixes and _scoped(name))
        )
        and not _parent_uses_weights_directly(name)
    ]
    # An exact dotted path that matches nothing is ALWAYS a bug, and a silent
    # one in the worst direction: the scope wraps zero modules, the trainable
    # count then equals the estimate that omits it, every downstream parity
    # check passes, and the resulting training run reads as "the hypothesis was
    # refuted" when in truth nothing was ever adapted. Prefixes have legitimate
    # zero-match cases (adapt_mask_decoder is 0 on the real model); an exact
    # path does not. Refuse, mirroring `merge_adapters`' hard error on an
    # adapter key that resolves to no base weight.
    matched = {name for name, _ in targets}
    missing = [path for path in cfg.include_module_paths if path not in matched]
    if missing:
        raise KeyError(
            "LoRA scope names exact module path(s) "
            f"{missing!r} that match no nn.Linear in this model; refusing an "
            "inert adapter scope"
        )

    for name, mod in targets:
        *parent_path, attr = name.split(".")
        parent = model
        for part in parent_path:
            parent = getattr(parent, part)
        setattr(parent, attr, LoraLinear(mod, cfg))
    return len(targets)


def adapter_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for name, mod in model.named_modules():
        if isinstance(mod, LoraLinear):
            out[f"{name}.lora_A"] = mod.lora_A.detach().cpu()
            out[f"{name}.lora_B"] = mod.lora_B.detach().cpu()
    return out


def merge_adapters(
    base: dict[str, torch.Tensor],
    adapters: dict[str, torch.Tensor],
    cfg: LoraConfig,
    *,
    prefix: str = "detector.",
) -> dict[str, torch.Tensor]:
    """Fold every adapter into ``base`` one tensor at a time.

    Adapters are trained against Meta's un-prefixed model; the published
    checkpoint is `detector.`-prefixed. An adapter that resolves to no base key
    is a HARD ERROR: skipping it silently produces a checkpoint that differs
    from base in bytes but not in behaviour, which is indistinguishable from a
    successful merge.

    Split-attention adapters (``{mha}.q_proj`` / ``.k_proj`` / ``.v_proj``
    from :class:`SplitMultiheadAttention`) have no ``.weight`` key in the
    stock checkpoint; their deltas fold back into the fused
    ``{prefix}{mha}.in_proj_weight`` row slices ``[0:E]``/``[E:2E]``/
    ``[2E:3E]``.  ``out_proj`` resolves through the normal formula (torch MHA
    stores ``out_proj.weight``).  Biases are untouched -- LoRA touches
    weights only -- so the merged state dict stays key-identical to stock.
    """
    validated = _validated_adapter_pairs(base, adapters, cfg, prefix=prefix)

    # A state dict can contain tied tensors. The previous full-dict clone broke
    # those aliases before updating one key; preserve that observable behaviour
    # without cloning unrelated tensors by separating only a touched alias.
    storage_owners: dict[int, int] = {}
    for tensor in base.values():
        storage_id = tensor.untyped_storage().data_ptr()
        storage_owners[storage_id] = storage_owners.get(storage_id, 0) + 1

    with torch.no_grad():
        for _path, key, row_slice, matrix_a, matrix_b in validated:
            target = base[key]
            # ``.get(ptr, 1)``: a clone made for an earlier adapter of the
            # same key (q/k/v share one in_proj_weight) has a storage the
            # pre-merge census never saw; it is by construction unaliased.
            if storage_owners.get(target.untyped_storage().data_ptr(), 1) > 1:
                target = target.clone()
                base[key] = target
            # matmul owns the sole output-sized temporary. ``add_`` performs
            # destination-dtype conversion internally, avoiding both a second
            # converted delta and a replacement tensor for the base weight.
            delta = torch.matmul(matrix_b, matrix_a)
            delta.mul_(cfg.scaling)
            if row_slice is None:
                target.add_(delta)
            else:
                # Split-attention q/k/v delta lands on its fused in_proj row
                # slice; add_ on the view mutates the stored tensor in place.
                target[row_slice].add_(delta)
            del delta
    return base


# Fused in_proj_weight row-slice index per split-attention projection leaf.
_IN_PROJ_ROW: dict[str, int] = {"q_proj": 0, "k_proj": 1, "v_proj": 2}


def _resolve_target_key(path: str, base_keys, *, prefix: str) -> tuple[str, int | None]:
    """Map one adapter path to its base-checkpoint key.

    Single source of truth for target resolution, shared by
    ``_validated_adapter_pairs`` and ``adapter_touched_keys`` so the merge
    and the publish-side "untouched keys" report can never disagree.

    Precedence: a real ``{prefix}{path}.weight`` key always wins (a
    free-standing ``q_proj`` Linear merges onto its own weight).  Only when
    that key is absent AND the leaf is one of the split-attention q/k/v
    projections does the adapter fold into the parent's fused
    ``in_proj_weight``, returning the row index of its slice.  Anything else
    is the existing hard error.
    """
    key = f"{prefix}{path}.weight"
    if key in base_keys:
        return key, None
    parent, _, leaf = path.rpartition(".")
    if leaf in _IN_PROJ_ROW and parent:
        fused_key = f"{prefix}{parent}.in_proj_weight"
        if fused_key in base_keys:
            return fused_key, _IN_PROJ_ROW[leaf]
    raise KeyError(
        f"adapter {path!r} resolves to {key!r}, which is not in the "
        "base checkpoint; refusing a partial merge"
    )


def _validated_adapter_pairs(
    base: dict[str, torch.Tensor],
    adapters: dict[str, torch.Tensor],
    cfg: LoraConfig,
    *,
    prefix: str,
) -> list[tuple[str, str, slice | None, torch.Tensor, torch.Tensor]]:
    """Resolve and validate the complete adapter plan before mutation."""

    if not isinstance(base, dict) or not base:
        raise ValueError("base checkpoint state must be a non-empty mapping")
    if not isinstance(adapters, dict) or not adapters:
        raise ValueError("adapter state must be a non-empty mapping")

    by_path: dict[str, dict[str, torch.Tensor]] = {}
    for adapter_key, tensor in adapters.items():
        if not isinstance(adapter_key, str) or not torch.is_tensor(tensor):
            raise ValueError("adapter state contains a non-tensor entry")
        path, separator, suffix = adapter_key.rpartition(".")
        if not separator or suffix not in {"lora_A", "lora_B"}:
            raise ValueError(f"unexpected adapter key {adapter_key!r}")
        if suffix in by_path.setdefault(path, {}):
            raise ValueError(f"duplicate adapter tensor {adapter_key!r}")
        by_path[path][suffix] = tensor

    validated: list[tuple[str, str, slice | None, torch.Tensor, torch.Tensor]] = []
    for path in sorted(by_path):
        pair = by_path[path]
        if set(pair) != {"lora_A", "lora_B"}:
            raise ValueError(f"adapter {path!r} has an incomplete LoRA A/B pair")
        key, row = _resolve_target_key(path, base, prefix=prefix)
        target = base[key]
        matrix_a = pair["lora_A"]
        matrix_b = pair["lora_B"]
        if not torch.is_tensor(target):
            raise ValueError(f"base checkpoint key {key!r} is not a tensor")
        if target.ndim != 2 or matrix_a.ndim != 2 or matrix_b.ndim != 2:
            raise ValueError(f"adapter {path!r} and its base weight must be 2-D")
        if not (
            target.is_floating_point()
            and matrix_a.is_floating_point()
            and matrix_b.is_floating_point()
        ):
            raise ValueError(
                f"adapter {path!r} and its base weight must be floating point"
            )
        if matrix_a.shape[0] != cfg.rank or matrix_b.shape[1] != cfg.rank:
            raise ValueError(
                f"adapter {path!r} rank does not match configured rank {cfg.rank}"
            )
        delta_shape = (matrix_b.shape[0], matrix_a.shape[1])
        if row is None:
            expected_shape = delta_shape
        else:
            # Fused in_proj_weight stacks the three E x E projections: the
            # delta must be square and the target exactly (3E, E).
            embed = delta_shape[1]
            if delta_shape[0] != embed:
                raise ValueError(
                    f"adapter {path!r} folds into a fused in_proj slice and "
                    f"must be square; got delta shape {delta_shape}"
                )
            expected_shape = (3 * embed, embed)
        if (
            matrix_a.shape[0] != matrix_b.shape[1]
            or tuple(target.shape) != expected_shape
        ):
            raise ValueError(
                f"adapter {path!r} shapes {tuple(matrix_b.shape)} @ "
                f"{tuple(matrix_a.shape)} do not match base weight "
                f"{tuple(target.shape)}"
            )
        row_slice = None
        if row is not None:
            embed = delta_shape[1]
            row_slice = slice(row * embed, (row + 1) * embed)
        validated.append((path, key, row_slice, matrix_a, matrix_b))
    return validated


def adapter_touched_keys(
    adapters: dict[str, torch.Tensor],
    base_keys,
    *,
    prefix: str = "detector.",
) -> set[str]:
    """Return the base-checkpoint keys `merge_adapters` would modify.

    Routed through the same ``_resolve_target_key`` the merge itself uses, so
    a caller reporting "N keys carried across untouched" (publish.py) cannot
    silently disagree with the key set `merge_adapters` touches.
    ``base_keys`` is required because split-attention q/k/v adapters resolve
    to their parent's fused ``in_proj_weight`` only when no free-standing
    ``.weight`` key exists -- a distinction only the base checkpoint decides.
    """
    paths = sorted({k.rsplit(".", 1)[0] for k in adapters})
    return {_resolve_target_key(path, base_keys, prefix=prefix)[0] for path in paths}
