# src/hydra_suite/core/identity/pose/vitpose/adapter.py
"""Load a fine-tuned or user-supplied ViTPose checkpoint, recovering the
variant/head/num_keypoints needed to rebuild the module.

PoseKit-free leaf module: imports nothing from hydra_suite app layers.

Bridges the training payload's checkpoint format
(``{"model_state", "variant", "num_keypoints", ...}``) into a live ``ViTPose``.
The leaf ``load_checkpoint`` expects a ``"state_dict"`` key, which the training
format does not have -- hence this adapter. Head type is not stored by the
trainer (it always builds ``"classic"``), so we infer it from parameter shapes,
which also lets arbitrary user checkpoints load.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import torch

from .config import VARIANTS
from .vitpose import ViTPose, build_vitpose
from .weights import CheckpointKeyError


@dataclass(frozen=True)
class FinetuneMeta:
    variant: str
    head: str
    num_keypoints: int


def _unwrap_state(blob: object) -> Dict[str, torch.Tensor]:
    if isinstance(blob, dict) and "model_state" in blob:
        return blob["model_state"]
    if isinstance(blob, dict) and "state_dict" in blob:
        return blob["state_dict"]
    if isinstance(blob, dict):
        return blob  # bare state_dict
    raise CheckpointKeyError(f"Unrecognized checkpoint object: {type(blob)!r}")


def infer_head_from_state(state: Dict[str, torch.Tensor]) -> str:
    """classic head has deconv layers; simple head does not."""
    has_deconv = any(k.startswith("keypoint_head.deconv_layers.") for k in state)
    return "classic" if has_deconv else "simple"


def _infer_num_keypoints(state: Dict[str, torch.Tensor]) -> int:
    w = state.get("keypoint_head.final_layer.weight")
    if w is None:
        raise CheckpointKeyError(
            "checkpoint has no keypoint_head.final_layer.weight; cannot infer K"
        )
    return int(w.shape[0])


def _infer_variant(state: Dict[str, torch.Tensor]) -> str:
    # embed_dim is the pos_embed last dim: backbone.pos_embed (1, N+1, embed_dim)
    pe = state.get("backbone.pos_embed")
    if pe is None:
        raise CheckpointKeyError(
            "checkpoint has no backbone.pos_embed; cannot infer variant"
        )
    embed_dim = int(pe.shape[-1])
    for name, spec in VARIANTS.items():
        if spec.embed_dim == embed_dim:
            return name
    raise CheckpointKeyError(f"no known variant with embed_dim={embed_dim}")


def load_finetuned_checkpoint(path: Path) -> tuple[ViTPose, FinetuneMeta]:
    path = Path(path)
    blob = torch.load(path, map_location="cpu", weights_only=True)
    state = _unwrap_state(blob)
    head = infer_head_from_state(state)
    if isinstance(blob, dict) and "variant" in blob and "num_keypoints" in blob:
        variant = str(blob["variant"])
        num_keypoints = int(blob["num_keypoints"])
    else:
        variant = _infer_variant(state)
        num_keypoints = _infer_num_keypoints(state)
    model = build_vitpose(variant, head, num_keypoints=num_keypoints)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise CheckpointKeyError(
            f"checkpoint load mismatch: {len(missing)} missing "
            f"{sorted(missing)[:6]}, {len(unexpected)} unexpected "
            f"{sorted(unexpected)[:6]}"
        )
    model.eval()
    return model, FinetuneMeta(variant=variant, head=head, num_keypoints=num_keypoints)
