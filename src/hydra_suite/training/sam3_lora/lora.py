"""Low-rank adapters: inject, extract, merge.

Deliberately free of any SAM3 import so the whole seam is testable on a toy
nn.Module without a GPU or the licence-gated checkpoint.
"""

from __future__ import annotations

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
# Adapted-module counts per flag AFTER the SplitMultiheadAttention pass
# (re-measured 2026-09-04 on a live build; fused torch MHAs now contribute
# four wrapped projections each -- 24 MHA in the language backbone, 12 in the
# DETR decoder):
#   adapt_vision_encoder             128
#   adapt_text_encoder               144   (was 73 Linears; MHA out_proj no
#                                          longer skipped, q/k/v now exist)
#   adapt_detr_encoder                12
#   adapt_detr_decoder                60   (was 42 Linears with 12 skipped)
#   adapt_geometry_encoder             6
#   adapt_mask_decoder                 0
#   dot_prod_scoring                   4 Linears <- covered by NO flag, deliberately
#
# `dot_prod_scoring` (the text/vision similarity head) is named by no adapt_*
# flag and is therefore never adapted. Add a flag if a future experiment wants
# it; do not fold it into another.
SUBMODULE_PREFIXES: dict[str, tuple[str, ...]] = {
    "adapt_vision_encoder": ("backbone.vision_backbone",),
    "adapt_text_encoder": ("backbone.language_backbone",),
    "adapt_geometry_encoder": ("geometry_encoder",),
    "adapt_detr_encoder": ("transformer.encoder",),
    "adapt_detr_decoder": ("transformer.decoder",),
    "adapt_mask_decoder": ("segmentation_head",),
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
    """
    enabled = [f for f in SUBMODULE_PREFIXES if getattr(params, f)]
    if not enabled:
        raise ValueError("at least one SAM3 LoRA adapter scope must be enabled")
    include = tuple(pref for flag in enabled for pref in SUBMODULE_PREFIXES[flag])
    return LoraConfig(
        rank=params.rank,
        alpha=params.alpha,
        dropout=params.dropout,
        target_suffixes=TARGET_SUFFIXES,
        include_prefixes=include,
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


def inject_adapters(model: nn.Module, cfg: LoraConfig) -> int:
    """Attach LoRA to every in-scope Linear the model actually calls.

    Two passes. First, every in-scope torch ``nn.MultiheadAttention`` is
    replaced with a :class:`SplitMultiheadAttention` so its fused q/k/v/out
    projections become real ``nn.Linear`` call sites (SAM3's own
    ``model_misc`` MHA clone is NOT an ``nn.MultiheadAttention`` instance and
    is deliberately left alone -- it is outside the empirically validated
    adaptation surface).  Second, every ``nn.Linear`` whose dotted path ends
    in a target suffix is wrapped in :class:`LoraLinear`.  Returns the number
    of wrapped Linears.
    """

    def _scoped(name: str) -> bool:
        if cfg.exclude_prefixes and name.startswith(cfg.exclude_prefixes):
            return False
        return not cfg.include_prefixes or name.startswith(cfg.include_prefixes)

    # Pass 1: split in-scope fused attention. Uses exact isinstance so
    # subclasses/clones with different forwards can never be silently
    # reinterpreted through torch-MHA semantics.
    fused = [
        (name, mod)
        for name, mod in model.named_modules()
        if type(mod) is nn.MultiheadAttention and _scoped(name)
    ]
    for name, mod in fused:
        *parent_path, attr = name.split(".")
        parent = model
        for part in parent_path:
            parent = getattr(parent, part)
        setattr(parent, attr, SplitMultiheadAttention.from_torch_mha(mod))

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

        After the SplitMultiheadAttention pass, in-scope torch MHAs no
        longer trip this guard (they have been replaced).  It still protects
        OUT-of-scope torch MHAs whose ``out_proj`` Linear would otherwise
        match a target suffix, and SAM3's ``model_misc`` clone, which is
        never replaced.
        """
        parent_path = name.rsplit(".", 1)[0] if "." in name else ""
        parent = by_name.get(parent_path)
        return parent is not None and hasattr(parent, "in_proj_weight")

    targets = [
        (name, mod)
        for name, mod in model.named_modules()
        if isinstance(mod, nn.Linear)
        and name.split(".")[-1] in cfg.target_suffixes
        and _scoped(name)
        and not _parent_uses_weights_directly(name)
    ]
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
