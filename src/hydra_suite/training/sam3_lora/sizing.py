"""Torch-free SAM3 LoRA sizing invariants shared by admission and runtime."""

from __future__ import annotations

from typing import Any

# Measured trainable parameter coefficient per requested rank for each
# production SAM3 adapter scope.  Runtime refuses if the actual injected model
# drifts from this estimator contract.
#
# Re-measured 2026-09-04 against a live build_sam3_image_model after
# inject_adapters gained the SplitMultiheadAttention pass (fused torch
# nn.MultiheadAttention replaced with q/k/v/out Linears, each LoRA-wrapped):
#   adapt_text_encoder  245_760 -> 442_368  (+24 MHA @ E=1024: 24*8*1024/rank)
#   adapt_detr_decoder   27_648 ->  52_224  (+12 MHA @ E=256:  12*8*256/rank)
# The other scopes contain no torch MHA and are unchanged.  SAM3's own
# model_misc MHA clone is never adapted, so it contributes nothing here.
LORA_PARAMS_PER_RANK: dict[str, int] = {
    "adapt_vision_encoder": 565_248,
    "adapt_text_encoder": 442_368,
    "adapt_geometry_encoder": 13_824,
    "adapt_detr_encoder": 27_648,
    "adapt_detr_decoder": 52_224,
    "adapt_mask_decoder": 0,
}
MAX_LORA_RANK = 256
MAX_LORA_TRAINABLE_PARAMS = 128_000_000


def expected_lora_trainable_params(params: Any) -> int:
    """Return the exact trainable count admitted for one LoRA configuration."""

    rank = int(params.rank)
    params_per_rank = sum(
        coefficient
        for flag, coefficient in LORA_PARAMS_PER_RANK.items()
        if bool(getattr(params, flag, False))
    )
    return rank * params_per_rank
