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


def _layer_id_for(name: str, num_layers: int) -> int:
    if name.startswith("backbone.patch_embed") or name == "backbone.pos_embed":
        return 0
    if name.startswith("backbone.blocks."):
        return int(name.split(".")[2]) + 1
    return num_layers - 1  # head, last_norm, everything downstream


def _no_decay(name: str, param) -> bool:
    return param.ndim <= 1 or name.endswith("pos_embed")


def build_param_groups(
    model, base_lr: float, layer_decay: float, weight_decay: float
) -> list[dict]:
    depth = len(model.backbone.blocks)
    num_layers = depth + 2
    buckets: dict[tuple[int, bool], dict] = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        lid = _layer_id_for(name, num_layers)
        scale = layer_decay ** (num_layers - lid - 1)
        decayed = not _no_decay(name, param)
        key = (lid, decayed)
        if key not in buckets:
            buckets[key] = {
                "params": [],
                "lr": base_lr * scale,
                "weight_decay": weight_decay if decayed else 0.0,
            }
        buckets[key]["params"].append(param)
    return list(buckets.values())
