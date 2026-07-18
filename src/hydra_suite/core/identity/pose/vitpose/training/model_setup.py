from __future__ import annotations

from pathlib import Path

import torch

from ..config import VARIANTS
from ..heads import build_head
from ..model import ViT
from ..vitpose import ViTPose
from ..weights import CheckpointKeyError

_EXPECTED_MISSING = {
    "keypoint_head.final_layer.weight",
    "keypoint_head.final_layer.bias",
}


def build_finetune_model(variant: str, num_keypoints: int, drop_path: float) -> ViTPose:
    if variant not in VARIANTS:
        raise ValueError(
            f"unknown variant {variant!r} (expected one of {sorted(VARIANTS)})"
        )
    v = VARIANTS[variant]
    backbone = ViT(
        embed_dim=v.embed_dim,
        depth=v.depth,
        num_heads=v.num_heads,
        drop_path_rate=drop_path,
    )
    head = build_head("classic", v.embed_dim, num_keypoints)
    return ViTPose(backbone, head)


def load_finetune_init(model: ViTPose, ckpt_path: Path) -> None:
    """Load a pretrained ViTPose checkpoint for fine-tuning: backbone (and head
    deconv) load strict; `keypoint_head.final_layer` is left freshly initialised
    so K can differ. Raises unless the ONLY missing keys are final_layer.*."""
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state = (
        blob["state_dict"] if isinstance(blob, dict) and "state_dict" in blob else blob
    )
    cleaned = {
        key: val
        for key, val in state.items()
        if not key.startswith("keypoint_head.final_layer.")
        and not key.startswith("associate_keypoint_heads.")
    }
    try:
        missing, unexpected = model.load_state_dict(cleaned, strict=False)
    except RuntimeError as exc:
        raise CheckpointKeyError(
            f"fine-tune load failed for {Path(ckpt_path).name}: {exc}"
        ) from exc
    missing, unexpected = set(missing), set(unexpected)
    if missing != _EXPECTED_MISSING or unexpected:
        raise CheckpointKeyError(
            f"fine-tune load mismatch for {Path(ckpt_path).name}\n"
            f"  unexpected missing: {sorted(missing - _EXPECTED_MISSING)}\n"
            f"  not-actually-missing: {sorted(_EXPECTED_MISSING - missing)}\n"
            f"  unexpected keys: {sorted(unexpected)[:10]}"
        )
