"""Persistent custom backbone registry for ClassKit training."""

from __future__ import annotations

import json
from typing import Iterable

from hydra_suite.paths import get_classkit_timm_backbones_path
from hydra_suite.training.torchvision_model import BACKBONE_DISPLAY_NAMES

DEFAULT_CUSTOM_BACKBONES: list[str] = [
    "tinyclassifier",
    "mobilenet_v3_small",
    "shufflenet_v2_x1_0",
    "resnet18",
    "efficientnet_b0",
    "convnext_tiny",
    "vit_b_16",
]


def _normalize_backbone_name(name: object) -> str:
    text = str(name or "").strip()
    if not text:
        return ""
    if text.startswith("timm/"):
        return text
    return text


def load_user_timm_backbones() -> list[str]:
    """Load user-added timm backbones from persistent config storage."""
    path = get_classkit_timm_backbones_path()
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(data, list):
        return []
    ordered: list[str] = []
    for item in data:
        value = _normalize_backbone_name(item)
        if value.startswith("timm/") and value not in ordered:
            ordered.append(value)
    return ordered


def save_user_timm_backbones(backbones: Iterable[str]) -> None:
    """Persist the user-added timm backbones."""
    ordered: list[str] = []
    for item in backbones:
        value = _normalize_backbone_name(item)
        if value.startswith("timm/") and value not in ordered:
            ordered.append(value)
    get_classkit_timm_backbones_path().write_text(
        json.dumps(ordered, indent=2), encoding="utf-8"
    )


def register_user_timm_backbones(backbones: Iterable[str]) -> list[str]:
    """Add one or more timm backbones to the persistent registry and return all registered entries."""
    ordered = load_user_timm_backbones()
    for item in backbones:
        value = _normalize_backbone_name(item)
        if value.startswith("timm/") and value not in ordered:
            ordered.append(value)
    save_user_timm_backbones(ordered)
    return ordered


def get_custom_backbone_choices() -> list[str]:
    """Return the curated default backbones followed by user-added timm models."""
    ordered: list[str] = []
    for item in DEFAULT_CUSTOM_BACKBONES + load_user_timm_backbones():
        value = _normalize_backbone_name(item)
        if value and value not in ordered:
            ordered.append(value)
    return ordered


def custom_backbone_display_name(backbone: object) -> str:
    """Return a user-facing label for a custom CNN backbone key."""
    value = _normalize_backbone_name(backbone)
    if value.startswith("timm/"):
        return f"TIMM: {value.split('/', 1)[1]}"
    return BACKBONE_DISPLAY_NAMES.get(value, value)
