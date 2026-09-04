"""Torch-free SAM3 LoRA sizing invariants shared by admission and runtime."""

from __future__ import annotations

from typing import Any

# Measured trainable parameter coefficient per requested rank for each
# production SAM3 adapter scope.  Runtime refuses if the actual injected model
# drifts from this estimator contract.
LORA_PARAMS_PER_RANK: dict[str, int] = {
    "adapt_vision_encoder": 565_248,
    "adapt_text_encoder": 245_760,
    "adapt_geometry_encoder": 13_824,
    "adapt_detr_encoder": 27_648,
    "adapt_detr_decoder": 27_648,
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
