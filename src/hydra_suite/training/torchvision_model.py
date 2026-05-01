"""Torchvision-based classifier: model factory, freezing, ONNX export, checkpoint I/O.

This module is the sole owner of all torchvision backbone construction and
layer-freezing logic for ClassKit's Custom CNN training mode.
All functions are pure Python / PyTorch — no Qt dependency.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torchvision.models as tvm

# ---------------------------------------------------------------------------
# Backbone registry
# ---------------------------------------------------------------------------

TORCHVISION_BACKBONES: list[str] = [
    "tinyclassifier",
    "mobilenet_v3_small",
    "shufflenet_v2_x1_0",
    "convnext_tiny",
    "convnext_small",
    "efficientnet_b0",
    "resnet18",
    "resnet50",
    "vit_b_16",
]

# Human-readable labels for the GUI
BACKBONE_DISPLAY_NAMES: dict[str, str] = {
    "tinyclassifier": "TinyClassifier",
    "mobilenet_v3_small": "MobileNet-V3-Small",
    "shufflenet_v2_x1_0": "ShuffleNet-V2 x1.0",
    "convnext_tiny": "ConvNeXt-Tiny",
    "convnext_small": "ConvNeXt-Small",
    "efficientnet_b0": "EfficientNet-B0",
    "resnet18": "ResNet-18",
    "resnet50": "ResNet-50",
    "vit_b_16": "ViT-B/16",
}

HEAD_MODULE_NAMES = {
    "classifier",
    "fc",
    "head",
    "heads",
    "head_drop",
    "fc_norm",
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_classifier_normalization_stats(
    monochrome: bool = False,
) -> tuple[list[float], list[float]]:
    """Return normalization stats for ClassKit torchvision classifiers.

    Monochrome mode should preserve identical channels after normalization,
    so grayscale inputs use the same mean/std in all three channels.
    """
    if not monochrome:
        return list(IMAGENET_MEAN), list(IMAGENET_STD)

    mono_mean = float(sum(IMAGENET_MEAN) / len(IMAGENET_MEAN))
    mono_std = float(sum(IMAGENET_STD) / len(IMAGENET_STD))
    return [mono_mean, mono_mean, mono_mean], [mono_std, mono_std, mono_std]


def is_timm_backbone(backbone: str) -> bool:
    """Return True when the backbone key refers to a TIMM model."""
    return str(backbone).startswith("timm/")


def _strip_timm_prefix(backbone: str) -> str:
    return str(backbone).split("/", 1)[1] if is_timm_backbone(backbone) else backbone


def _normalize_timm_img_size(input_size: Any) -> int | tuple[int, int] | None:
    """Return a TIMM-compatible img_size value or None when not specified."""
    if input_size is None:
        return None
    if isinstance(input_size, int):
        return int(input_size) if int(input_size) > 0 else None
    if isinstance(input_size, (list, tuple)) and len(input_size) == 2:
        try:
            height = int(input_size[0])
            width = int(input_size[1])
        except (TypeError, ValueError):
            return None
        if height <= 0 or width <= 0:
            return None
        return height if height == width else (height, width)
    return None


def _load_pretrained(
    backbone: str,
    *,
    input_size: Any = None,
) -> nn.Module:
    """Load a pretrained torchvision model by backbone key."""
    if is_timm_backbone(backbone):
        try:
            import timm
        except Exception as exc:
            raise RuntimeError(
                "TIMM backbones require the 'timm' package to be installed."
            ) from exc

        model_name = _strip_timm_prefix(backbone)
        img_size = _normalize_timm_img_size(input_size)
        if img_size is not None:
            try:
                return timm.create_model(model_name, pretrained=True, img_size=img_size)
            except TypeError:
                pass
        return timm.create_model(model_name, pretrained=True)

    # ViT-B/16 positional embeddings are baked for exactly 224×224 (14×14 patch
    # grid).  Feeding any other resolution causes a shape mismatch in the
    # transformer — raise early with a clear message rather than a cryptic
    # RuntimeError inside the attention block.
    if backbone == "vit_b_16" and input_size is not None:
        if isinstance(input_size, (list, tuple)):
            sizes = [int(s) for s in input_size]
        else:
            sizes = [int(input_size)]
        if any(s != 224 for s in sizes):
            raise ValueError(
                f"vit_b_16 requires input_size=224 — its positional embeddings "
                f"are fixed to a 14×14 patch grid.  Got {input_size!r}.  "
                "Use a CNN backbone (resnet18, convnext_tiny, etc.) for arbitrary "
                "input sizes, or a timm ViT variant which supports resolution "
                "interpolation (e.g. timm/vit_base_patch16_224)."
            )

    weights_map = {
        "mobilenet_v3_small": tvm.MobileNet_V3_Small_Weights.IMAGENET1K_V1,
        "shufflenet_v2_x1_0": tvm.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1,
        "convnext_tiny": tvm.ConvNeXt_Tiny_Weights.IMAGENET1K_V1,
        "convnext_small": tvm.ConvNeXt_Small_Weights.IMAGENET1K_V1,
        "efficientnet_b0": tvm.EfficientNet_B0_Weights.IMAGENET1K_V1,
        "resnet18": tvm.ResNet18_Weights.IMAGENET1K_V1,
        "resnet50": tvm.ResNet50_Weights.IMAGENET1K_V1,
        "vit_b_16": tvm.ViT_B_16_Weights.IMAGENET1K_V1,
    }
    factory = getattr(tvm, backbone)
    return factory(weights=weights_map[backbone])


def _replace_head(model: nn.Module, backbone: str, num_classes: int) -> nn.Module:
    """Replace the final classifier head with a new linear layer."""
    if is_timm_backbone(backbone):
        if hasattr(model, "reset_classifier"):
            model.reset_classifier(num_classes=num_classes)
            return model
        raise ValueError(
            f"Unsupported TIMM backbone for head replacement: {backbone!r}"
        )

    if backbone.startswith("convnext"):
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    elif backbone.startswith("mobilenet"):
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    elif backbone.startswith("efficientnet"):
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    elif backbone.startswith("shufflenet"):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif backbone.startswith("resnet"):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif backbone == "vit_b_16":
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unsupported backbone for head replacement: {backbone!r}")
    return model


def get_layer_groups(model: nn.Module, backbone: str) -> list[nn.Module]:
    """Return backbone layer groups in shallow-to-deep order.

    The caller can index from the end to unfreeze the last N groups.
    ConvNeXt and ResNet return exactly 4 groups.
    EfficientNet returns individual feature blocks.
    ViT-B/16 returns individual encoder layers.
    """
    if is_timm_backbone(backbone):
        if hasattr(model, "blocks") and len(model.blocks) > 0:
            return list(model.blocks)
        if hasattr(model, "stages") and len(model.stages) > 0:
            return list(model.stages)
        if hasattr(model, "layer1") and hasattr(model, "layer4"):
            return [model.layer1, model.layer2, model.layer3, model.layer4]
        if hasattr(model, "features") and isinstance(model.features, nn.Sequential):
            return list(model.features)

        groups: list[nn.Module] = []
        for name, module in model.named_children():
            if name in HEAD_MODULE_NAMES or name in {"global_pool", "flatten"}:
                continue
            if isinstance(module, (nn.Sequential, nn.ModuleList)) and len(module) > 0:
                groups.extend(list(module))
            else:
                groups.append(module)
        if groups:
            return groups
        raise ValueError(f"Unsupported TIMM backbone for layer groups: {backbone!r}")

    if backbone.startswith("convnext"):
        # ConvNeXt's features Sequential interleaves LayerNorm downsampling
        # transitions at even indices (0, 2, 4, 6) with the four main ConvNeXt
        # stage blocks at odd indices (1, 3, 5, 7).  We expose only the four
        # stage blocks so that trainable_layers=N unfreezes the last N stages,
        # skipping the lightweight transition layers.
        return [model.features[i] for i in [1, 3, 5, 7]]
    elif backbone.startswith("mobilenet"):
        return list(model.features)
    elif backbone.startswith("resnet"):
        return [model.layer1, model.layer2, model.layer3, model.layer4]
    elif backbone.startswith("shufflenet"):
        return [model.stage2, model.stage3, model.stage4, model.conv5]
    elif backbone.startswith("efficientnet"):
        # features is a Sequential; expose as individual blocks
        return list(model.features)
    elif backbone == "vit_b_16":
        return list(model.encoder.layers)
    else:
        raise ValueError(f"Unsupported backbone for layer groups: {backbone!r}")


def _get_head_module(model: nn.Module, backbone: str) -> nn.Module | None:
    """Return the classifier head module for a given backbone."""
    if is_timm_backbone(backbone):
        for name in ("head", "classifier", "fc", "heads"):
            module = getattr(model, name, None)
            if isinstance(module, nn.Module):
                return module
        return None

    if backbone.startswith("convnext") or backbone.startswith("efficientnet"):
        return model.classifier
    if backbone.startswith("mobilenet"):
        return model.classifier
    if backbone.startswith("resnet") or backbone.startswith("shufflenet"):
        return model.fc
    if backbone == "vit_b_16":
        return model.heads
    return None


def freeze_backbone(model: nn.Module, backbone: str, trainable_layers: int) -> None:
    """Freeze/unfreeze backbone parameters according to trainable_layers.

    Args:
        model: Model whose backbone parameters to freeze.
        backbone: Backbone key (used to determine head parameter names).
        trainable_layers: 0=frozen, -1=all, N>0=unfreeze last N groups.
    """
    # Step 1: freeze everything
    for p in model.parameters():
        p.requires_grad = False

    # Step 2: always unfreeze the head
    head = _get_head_module(model, backbone)
    if head is not None:
        for p in head.parameters():
            p.requires_grad = True

    # Step 3: apply backbone unfreezing
    if trainable_layers == -1:
        for p in model.parameters():
            p.requires_grad = True
    elif trainable_layers > 0:
        groups = get_layer_groups(model, backbone)
        for group in groups[-trainable_layers:]:
            for p in group.parameters():
                p.requires_grad = True


def build_torchvision_classifier(
    backbone: str,
    num_classes: int,
    trainable_layers: int,
    *,
    tiny_preset: str = "medium",
    hidden_layers: int = 1,
    hidden_dim: int = 96,
    dropout: float = 0.1,
    input_width: int = 128,
    input_height: int = 64,
    input_size: int | tuple[int, int] | None = None,
) -> nn.Module:
    """Build a pretrained torchvision classifier with a new head.

    Args:
        backbone: One of the keys in TORCHVISION_BACKBONES, or "tinyclassifier".
        num_classes: Number of output classes.
        trainable_layers: Backbone freezing mode: 0=frozen, -1=all trainable, N=last N groups.
            Ignored when backbone='tinyclassifier' (always fully trainable).
        hidden_layers: Number of hidden MLP layers (TinyClassifier only).
        hidden_dim: Hidden MLP dimension (TinyClassifier only).
        dropout: Dropout rate for the MLP head (TinyClassifier only).
        input_width: Input image width in pixels (TinyClassifier only).
        input_height: Input image height in pixels (TinyClassifier only).

    Returns:
        nn.Module in train mode with head replaced and freezing applied.
    """
    if (
        backbone != "tinyclassifier"
        and backbone not in TORCHVISION_BACKBONES
        and not is_timm_backbone(backbone)
    ):
        raise ValueError(
            f"Unknown backbone {backbone!r}. Must be one of {TORCHVISION_BACKBONES} or a timm/<model> entry"
        )
    if backbone == "tinyclassifier":
        from hydra_suite.training.tiny_model import _build_tiny_classifier_class

        TinyClassifier = _build_tiny_classifier_class()
        return TinyClassifier(
            n_classes=num_classes,
            tiny_preset=tiny_preset,
            hidden_layers=hidden_layers,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
    model = _load_pretrained(backbone, input_size=input_size)
    model = _replace_head(model, backbone, num_classes)
    freeze_backbone(model, backbone, trainable_layers)
    return model


def save_torchvision_checkpoint(
    *,
    model: nn.Module,
    backbone: str,
    class_names: list[str],
    factor_names: list[str],
    input_size: tuple[int, int],
    best_val_acc: float | None,
    history: dict[str, Any],
    trainable_layers: int,
    backbone_lr_scale: float,
    monochrome: bool,
    class_names_per_factor: list[list[str]] | None = None,
    extra_meta: dict[str, Any] | None = None,
    path: str | Path,
) -> Path:
    """Save a torchvision model checkpoint in the v2 classifier-artifact format.

    Flat checkpoints pass ``class_names`` and may omit ``class_names_per_factor``
    (it will be synthesised as ``[class_names]``). Multi-head checkpoints must
    pass ``class_names_per_factor`` explicitly and leave ``class_names`` empty;
    ``class_names`` is then omitted from the saved checkpoint.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    factor_names = list(factor_names or ["flat"])
    is_multihead = len(factor_names) > 1

    if class_names_per_factor is None:
        if is_multihead:
            raise ValueError("multi-head checkpoint requires class_names_per_factor")
        class_names_per_factor = [list(class_names)]
    else:
        class_names_per_factor = [list(inner) for inner in class_names_per_factor]

    if len(class_names_per_factor) != len(factor_names):
        raise ValueError(
            "factor_names and class_names_per_factor must have the same length"
        )

    h, w = int(input_size[0]), int(input_size[1])
    num_classes = sum(len(inner) for inner in class_names_per_factor)

    ckpt: dict[str, Any] = {
        "schema_version": 2,
        "arch": backbone,
        "factor_names": factor_names,
        "class_names_per_factor": class_names_per_factor,
        "input_size": [h, w],
        "num_classes": num_classes,
        "monochrome": bool(monochrome),
        "model_state_dict": model.state_dict(),
        "best_val_acc": best_val_acc,
        "history": history,
        "trainable_layers": trainable_layers,
        "backbone_lr_scale": backbone_lr_scale,
    }
    if backbone == "tinyclassifier":
        from hydra_suite.training.tiny_model import tiny_model_checkpoint_metadata

        ckpt.update(tiny_model_checkpoint_metadata(model))
    if not is_multihead:
        ckpt["class_names"] = list(class_names_per_factor[0])
    # Head-shape fields are managed here so they always have consistent types,
    # not propagated raw via the trailing extra_meta copy.
    _HEAD_FIELDS = ("head_kind", "head_hidden_dim", "head_dropout")
    head_kind = None
    if isinstance(extra_meta, dict):
        head_kind = extra_meta.get("head_kind")
    if head_kind:
        ckpt["head_kind"] = str(head_kind)
        if "head_hidden_dim" in extra_meta:
            ckpt["head_hidden_dim"] = int(extra_meta["head_hidden_dim"])
        if "head_dropout" in extra_meta:
            ckpt["head_dropout"] = float(extra_meta["head_dropout"])
    else:
        ckpt["head_kind"] = "flat"
    if isinstance(extra_meta, dict) and extra_meta:
        for k, v in extra_meta.items():
            if k in _HEAD_FIELDS or k in ckpt:
                continue
            ckpt[k] = v
    torch.save(ckpt, str(path))
    return path


def load_torchvision_classifier(
    path: str | Path, device: str = "cpu"
) -> tuple[nn.Module, dict[str, Any]]:
    """Load a ClassKit torchvision .pth checkpoint as a ready-to-eval model.

    Dispatches on ``head_kind``:
      * ``"flat"`` (default): single linear classifier — same as before.
      * ``"multihead_shared_trunk"``: the new shared-trunk wrapper.

    Returns:
        (model_in_eval_mode_on_device, full_ckpt_dict)
    """
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    arch = ckpt["arch"]
    head_kind = str(ckpt.get("head_kind") or "flat")

    if head_kind == "multihead_shared_trunk":
        from hydra_suite.training.multihead_torchvision_model import (
            build_multihead_torchvision_classifier,
        )

        cnpf = ckpt.get("class_names_per_factor") or []
        if not cnpf:
            raise ValueError(
                f"{path!r}: shared-trunk checkpoint missing class_names_per_factor"
            )
        model = build_multihead_torchvision_classifier(
            backbone=arch,
            class_names_per_factor=cnpf,
            trainable_layers=-1,
            head_hidden_dim=int(ckpt.get("head_hidden_dim", 256)),
            head_dropout=float(ckpt.get("head_dropout", 0.1)),
            input_size=ckpt.get("input_size"),
        )
    else:
        num_classes = ckpt["num_classes"]
        model = build_torchvision_classifier(
            arch,
            num_classes,
            trainable_layers=-1,
            input_size=ckpt.get("input_size"),
        )

    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.train(False)
    return model, ckpt


def export_torchvision_to_onnx(
    model: nn.Module, ckpt: dict[str, Any], onnx_path: str | Path
) -> Path:
    """Export a torchvision classifier to ONNX format.

    Args:
        model: Model in eval mode.
        ckpt: Checkpoint dict (used for input_size).
        onnx_path: Output path for the .onnx file.

    Returns:
        Path to the exported ONNX file.
    """
    onnx_path = Path(onnx_path)
    h, w = ckpt.get("input_size", (224, 224))
    dummy = torch.zeros(1, 3, h, w)
    model.eval()
    export_kwargs = {
        "opset_version": 17,
        "input_names": ["input"],
        "output_names": ["logits"],
        "dynamic_axes": {"input": {0: "batch"}, "logits": {0: "batch"}},
    }
    try:
        torch.onnx.export(
            model,
            dummy,
            str(onnx_path),
            dynamo=False,
            **export_kwargs,
        )
    except TypeError:
        torch.onnx.export(
            model,
            dummy,
            str(onnx_path),
            **export_kwargs,
        )
    return onnx_path
