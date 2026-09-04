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
# Linear counts per prefix, so a mapping regression is detectable:
#   backbone.vision_backbone.trunk   128
#   backbone.language_backbone        73   (encoder 72 + resizer 1)
#   transformer.decoder               42
#   transformer.encoder.layers        24
#   geometry_encoder                  18
#   segmentation_head                  4
#   dot_prod_scoring                   4   <- covered by NO flag, deliberately
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


def inject_adapters(model: nn.Module, cfg: LoraConfig) -> int:
    """Wrap every nn.Linear whose dotted path ends in a target suffix."""

    def _scoped(name: str) -> bool:
        if cfg.exclude_prefixes and name.startswith(cfg.exclude_prefixes):
            return False
        return not cfg.include_prefixes or name.startswith(cfg.include_prefixes)

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
    """Fold every adapter into the base state dict, in the base's key layout.

    Adapters are trained against Meta's un-prefixed model; the published
    checkpoint is `detector.`-prefixed. An adapter that resolves to no base key
    is a HARD ERROR: skipping it silently produces a checkpoint that differs
    from base in bytes but not in behaviour, which is indistinguishable from a
    successful merge.
    """
    merged = {k: v.clone() for k, v in base.items()}
    paths = sorted({k.rsplit(".", 1)[0] for k in adapters})
    for path in paths:
        key = f"{prefix}{path}.weight"
        if key not in merged:
            raise KeyError(
                f"adapter {path!r} resolves to {key!r}, which is not in the "
                f"base checkpoint; refusing a partial merge"
            )
        a = adapters[f"{path}.lora_A"]
        b = adapters[f"{path}.lora_B"]
        delta = (b @ a) * cfg.scaling
        merged[key] = merged[key] + delta.to(merged[key].dtype)
    return merged


def adapter_touched_keys(
    adapters: dict[str, torch.Tensor], *, prefix: str = "detector."
) -> set[str]:
    """Return the base-checkpoint keys `merge_adapters` would modify.

    Single source of truth for the `{prefix}{path}.weight` formula, so a
    caller reporting "N keys carried across untouched" (publish.py) cannot
    silently disagree with the key set `merge_adapters` itself touches.
    """
    paths = sorted({k.rsplit(".", 1)[0] for k in adapters})
    return {f"{prefix}{path}.weight" for path in paths}
