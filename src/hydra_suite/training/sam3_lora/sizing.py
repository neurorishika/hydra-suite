"""Torch-free SAM3 LoRA sizing invariants shared by admission and runtime."""

from __future__ import annotations

from typing import Any

# Measured trainable parameter coefficient per requested rank for each
# production SAM3 adapter scope.  Runtime refuses if the actual injected model
# drifts from this estimator contract.
#
# Re-measured 2026-09-05 on a live `build_sam3_image_model` (mehek, sam3-lora
# env) after `inject_adapters` gained the SAM3-clone splitting pass and the
# geometry-encoder path allowlist -- measured by injecting each scope alone at
# rank 1 and summing `requires_grad` parameters, so the value IS the per-rank
# coefficient. Every clone the pass replaces contributes 4 wrapped E x E
# Linears, i.e. 4 * 2 * E params per rank (E = 256 everywhere outside the
# language backbone):
#   adapt_detr_encoder    27_648 ->  52_224  (+12 clones)
#   adapt_detr_decoder    52_224 ->  64_512  (+6 clones)
#   adapt_geometry_encoder 13_824 ->  28_680  (+6 clones = 12_288, plus the six
#                                             plain projections = 2_568)
#   adapt_mask_decoder         0 ->   2_048  (+1 clone: cross_attend_prompt)
# `adapt_vision_encoder` and `adapt_text_encoder` contain no SAM3 clone and are
# unchanged. Cross-checked against the spike checkpoint's own module
# decomposition (128 / 60 / 84 / 36 / 4 / 2) -- see
# docs/superpowers/specs/2026-09-05-sam3-finetune-quality-audit.md.
#
# `adapt_scoring_head` still has NO entry here, so `cli.py` keeps refusing the
# flag by name. Its coefficient was measured on the same live model and is
# 1_024 (2 x 256-in/256-out Linears); it is recorded here rather than landed
# because adding the key would silently un-refuse a scope whose paired retrain
# has not run.
LORA_PARAMS_PER_RANK: dict[str, int] = {
    "adapt_vision_encoder": 565_248,
    "adapt_text_encoder": 442_368,
    "adapt_geometry_encoder": 28_680,
    "adapt_detr_encoder": 52_224,
    "adapt_detr_decoder": 64_512,
    "adapt_mask_decoder": 2_048,
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
