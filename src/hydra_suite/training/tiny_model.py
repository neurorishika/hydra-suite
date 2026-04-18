"""Shared tiny CNN classifier for training (runner.py) and inference (task_workers.py)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

_LEGACY_TINY_ARCH_VERSION = 1
_DEFAULT_TINY_ARCH_VERSION = 2
_LEGACY_FEATURE_CHANNELS = (16, 32, 64, 64)
_DEFAULT_FEATURE_CHANNELS = (24, 48, 96, 160)
_DEFAULT_STAGE_DEPTHS = (1, 1, 2, 1)
_DEFAULT_STAGE_STRIDES = (1, 2, 2, 2)
_DEFAULT_EXPANSION_RATIO = 2
_DEFAULT_TINY_PRESET = "medium"

_TINY_SIZE_PRESETS: dict[str, dict[str, Any]] = {
    "small": {
        "display_name": "Tiny-S",
        "summary": "16/32/64/96 channels, depths 1/1/1/1",
        "feature_channels": (16, 32, 64, 96),
        "stage_depths": (1, 1, 1, 1),
        "stage_strides": _DEFAULT_STAGE_STRIDES,
        "expansion_ratio": 2,
        "use_squeeze_excite": True,
        "dual_pool": True,
        "recommended_hidden_dim": 64,
        "recommended_dropout": 0.10,
    },
    "medium": {
        "display_name": "Tiny-M",
        "summary": "24/48/96/160 channels, depths 1/1/2/1",
        "feature_channels": _DEFAULT_FEATURE_CHANNELS,
        "stage_depths": _DEFAULT_STAGE_DEPTHS,
        "stage_strides": _DEFAULT_STAGE_STRIDES,
        "expansion_ratio": _DEFAULT_EXPANSION_RATIO,
        "use_squeeze_excite": True,
        "dual_pool": True,
        "recommended_hidden_dim": 96,
        "recommended_dropout": 0.10,
    },
    "large": {
        "display_name": "Tiny-L",
        "summary": "32/64/128/224 channels, depths 1/2/2/1",
        "feature_channels": (32, 64, 128, 224),
        "stage_depths": (1, 2, 2, 1),
        "stage_strides": _DEFAULT_STAGE_STRIDES,
        "expansion_ratio": 2,
        "use_squeeze_excite": True,
        "dual_pool": True,
        "recommended_hidden_dim": 128,
        "recommended_dropout": 0.05,
    },
}


def normalize_tiny_size_preset(value: Any) -> str:
    """Normalize a tiny size preset key from UI or checkpoint metadata."""
    token = str(value or "").strip().lower()
    return token if token in _TINY_SIZE_PRESETS else _DEFAULT_TINY_PRESET


def get_tiny_size_preset_choices() -> list[tuple[str, str]]:
    """Return UI-friendly ``(key, display_name)`` pairs for tiny presets."""
    return [
        (key, str(spec["display_name"])) for key, spec in _TINY_SIZE_PRESETS.items()
    ]


def get_tiny_size_preset_config(preset: Any) -> dict[str, Any]:
    """Return a copy of the architecture config for a tiny preset."""
    normalized = normalize_tiny_size_preset(preset)
    config = dict(_TINY_SIZE_PRESETS[normalized])
    config["key"] = normalized
    return config


def describe_tiny_size_preset(preset: Any) -> str:
    """Return a compact human-readable description for a tiny preset."""
    config = get_tiny_size_preset_config(preset)
    return f"{config['display_name']} - {config['summary']}"


def _normalize_stage_tuple(values: Any, fallback: tuple[int, ...]) -> tuple[int, ...]:
    if not values:
        return tuple(int(value) for value in fallback)
    normalized = [int(value) for value in values]
    return tuple(normalized) if normalized else tuple(int(value) for value in fallback)


def _infer_classifier_dims(
    state: dict[str, Any], ckpt: dict[str, Any]
) -> tuple[int, int, int]:
    """Infer ``(n_classes, hidden_layers, hidden_dim)`` from the classifier branch."""
    linear_keys = sorted(
        [k for k in state if k.startswith("classifier.") and k.endswith(".weight")],
        key=lambda k: int(k.split(".")[1]),
    )
    if not linear_keys:
        raise ValueError("No Linear weight keys found in checkpoint classifier branch.")

    n_classes = int(state[linear_keys[-1]].shape[0])
    hidden_count = len(linear_keys) - 1
    if hidden_count > 0:
        hidden_dim = int(state[linear_keys[0]].shape[0])
    else:
        hidden_dim = int(ckpt.get("hidden_dim", 96) or 96)
    return n_classes, hidden_count, hidden_dim


def _infer_stage_config_from_v2_state(
    state: dict[str, Any],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Infer stage widths/depths from v2 TinyClassifier state dict keys."""
    pattern = re.compile(r"^stages\.(\d+)\.(\d+)\.project\.1\.weight$")
    channels_by_stage: dict[int, int] = {}
    depth_by_stage: dict[int, int] = {}
    for key, value in state.items():
        match = pattern.match(str(key))
        if match is None:
            continue
        stage_idx = int(match.group(1))
        block_idx = int(match.group(2))
        channels_by_stage[stage_idx] = int(value.shape[0])
        depth_by_stage[stage_idx] = max(depth_by_stage.get(stage_idx, 0), block_idx + 1)
    if not channels_by_stage:
        return _DEFAULT_FEATURE_CHANNELS, _DEFAULT_STAGE_DEPTHS
    ordered_stage_indices = sorted(channels_by_stage)
    return (
        tuple(channels_by_stage[index] for index in ordered_stage_indices),
        tuple(depth_by_stage.get(index, 1) for index in ordered_stage_indices),
    )


def _infer_tiny_architecture_config(
    ckpt: dict[str, Any],
    state: dict[str, Any],
) -> dict[str, Any]:
    """Resolve the TinyClassifier architecture config for a checkpoint."""
    if any(
        str(key).startswith("stages.") or str(key).startswith("stem.") for key in state
    ):
        tiny_preset = normalize_tiny_size_preset(ckpt.get("tiny_preset"))
        preset_config = get_tiny_size_preset_config(tiny_preset)
        inferred_channels, inferred_depths = _infer_stage_config_from_v2_state(state)
        feature_channels = _normalize_stage_tuple(
            ckpt.get("feature_channels"),
            inferred_channels,
        )
        stage_depths = _normalize_stage_tuple(
            ckpt.get("stage_depths"),
            inferred_depths,
        )
        stage_strides = _normalize_stage_tuple(
            ckpt.get("stage_strides"),
            tuple(preset_config["stage_strides"])[: len(stage_depths)],
        )
        return {
            "architecture_version": int(
                ckpt.get("tiny_arch_version", _DEFAULT_TINY_ARCH_VERSION)
            ),
            "tiny_preset": tiny_preset,
            "feature_channels": feature_channels,
            "stage_depths": stage_depths,
            "stage_strides": stage_strides,
            "expansion_ratio": int(
                ckpt.get("expansion_ratio", preset_config["expansion_ratio"])
            ),
            "use_squeeze_excite": bool(
                ckpt.get("use_squeeze_excite", preset_config["use_squeeze_excite"])
            ),
            "dual_pool": bool(ckpt.get("dual_pool", preset_config["dual_pool"])),
        }

    return {
        "architecture_version": _LEGACY_TINY_ARCH_VERSION,
        "tiny_preset": "legacy",
        "feature_channels": _LEGACY_FEATURE_CHANNELS,
        "stage_depths": (),
        "stage_strides": (),
        "expansion_ratio": 1,
        "use_squeeze_excite": False,
        "dual_pool": False,
    }


def tiny_model_checkpoint_metadata(model: Any) -> dict[str, Any]:
    """Extract the persisted TinyClassifier architecture metadata from a model."""
    return {
        "tiny_arch_version": int(getattr(model, "architecture_version", 1) or 1),
        "tiny_preset": str(getattr(model, "tiny_preset", "legacy") or "legacy"),
        "feature_channels": [
            int(value) for value in getattr(model, "feature_channels", ()) or ()
        ],
        "stage_depths": [
            int(value) for value in getattr(model, "stage_depths", ()) or ()
        ],
        "stage_strides": [
            int(value) for value in getattr(model, "stage_strides", ()) or ()
        ],
        "expansion_ratio": int(getattr(model, "expansion_ratio", 1) or 1),
        "use_squeeze_excite": bool(getattr(model, "use_squeeze_excite", False)),
        "dual_pool": bool(getattr(model, "dual_pool", False)),
    }


def _build_tiny_classifier_class():
    """Return TinyClassifier class (deferred torch import)."""
    import torch
    import torch.nn as nn

    class SqueezeExcite(nn.Module):
        """Cheap channel attention for lightweight residual blocks."""

        def __init__(self, channels: int, ratio: float = 0.25):
            super().__init__()
            reduced = max(8, int(channels * ratio))
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.reduce = nn.Conv2d(channels, reduced, kernel_size=1)
            self.act = nn.SiLU(inplace=True)
            self.expand = nn.Conv2d(reduced, channels, kernel_size=1)
            self.gate = nn.Sigmoid()

        def forward(self, x):
            scale = self.pool(x)
            scale = self.reduce(scale)
            scale = self.act(scale)
            scale = self.expand(scale)
            return x * self.gate(scale)

    class TinyResidualBlock(nn.Module):
        """Depthwise-separable residual block used by the modern tiny backbone."""

        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            *,
            stride: int,
            expansion_ratio: int,
            use_squeeze_excite: bool,
        ):
            super().__init__()
            hidden_channels = max(out_channels, in_channels * max(1, expansion_ratio))
            self.expand = nn.Sequential(
                nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden_channels),
                nn.SiLU(inplace=True),
            )
            self.depthwise = nn.Sequential(
                nn.Conv2d(
                    hidden_channels,
                    hidden_channels,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    groups=hidden_channels,
                    bias=False,
                ),
                nn.BatchNorm2d(hidden_channels),
                nn.SiLU(inplace=True),
            )
            self.se = (
                SqueezeExcite(hidden_channels) if use_squeeze_excite else nn.Identity()
            )
            self.project = nn.Sequential(
                nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
            self.use_residual = stride == 1 and in_channels == out_channels
            self.out_act = nn.SiLU(inplace=True)

        def forward(self, x):
            identity = x
            x = self.expand(x)
            x = self.depthwise(x)
            x = self.se(x)
            x = self.project(x)
            if self.use_residual:
                x = x + identity
            return self.out_act(x)

    class TinyClassifier(nn.Module):
        """Lightweight image classifier with backward-compatible tiny checkpoint support."""

        def __init__(
            self,
            n_classes: int,
            hidden_layers: int = 1,
            hidden_dim: int = 96,
            dropout: float = 0.1,
            *,
            tiny_preset: str = _DEFAULT_TINY_PRESET,
            architecture_version: int = _DEFAULT_TINY_ARCH_VERSION,
            feature_channels: tuple[int, ...] | None = None,
            stage_depths: tuple[int, ...] | None = None,
            stage_strides: tuple[int, ...] | None = None,
            expansion_ratio: int = _DEFAULT_EXPANSION_RATIO,
            use_squeeze_excite: bool = True,
            dual_pool: bool = True,
        ):
            super().__init__()
            self.n_classes = n_classes
            self.hidden_layers = int(hidden_layers)
            self.hidden_dim = int(hidden_dim)
            self.dropout = float(dropout)
            self.architecture_version = int(architecture_version)

            if self.architecture_version <= _LEGACY_TINY_ARCH_VERSION:
                self.tiny_preset = "legacy"
                self.feature_channels = _normalize_stage_tuple(
                    feature_channels,
                    _LEGACY_FEATURE_CHANNELS,
                )
                self.stage_depths = ()
                self.stage_strides = ()
                self.expansion_ratio = 1
                self.use_squeeze_excite = False
                self.dual_pool = False
                channels = self.feature_channels
                self.features = nn.Sequential(
                    nn.Conv2d(3, channels[0], 3, stride=2, padding=1),
                    nn.BatchNorm2d(channels[0]),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channels[0], channels[1], 3, stride=2, padding=1),
                    nn.BatchNorm2d(channels[1]),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channels[1], channels[2], 3, stride=2, padding=1),
                    nn.BatchNorm2d(channels[2]),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channels[2], channels[3], 3, stride=2, padding=1),
                    nn.BatchNorm2d(channels[3]),
                    nn.ReLU(inplace=True),
                    nn.AdaptiveAvgPool2d(1),
                )
                pooled_dim = channels[-1]
            else:
                preset_config = get_tiny_size_preset_config(tiny_preset)
                self.tiny_preset = str(preset_config["key"])
                self.feature_channels = _normalize_stage_tuple(
                    feature_channels,
                    tuple(preset_config["feature_channels"]),
                )
                default_depths = tuple(preset_config["stage_depths"])[
                    : len(self.feature_channels)
                ]
                default_strides = tuple(preset_config["stage_strides"])[
                    : len(self.feature_channels)
                ]
                self.stage_depths = _normalize_stage_tuple(stage_depths, default_depths)
                self.stage_strides = _normalize_stage_tuple(
                    stage_strides,
                    default_strides,
                )
                self.expansion_ratio = int(expansion_ratio)
                self.use_squeeze_excite = bool(use_squeeze_excite)
                self.dual_pool = bool(dual_pool)

                stem_channels = self.feature_channels[0]
                self.stem = nn.Sequential(
                    nn.Conv2d(
                        3, stem_channels, kernel_size=3, stride=2, padding=1, bias=False
                    ),
                    nn.BatchNorm2d(stem_channels),
                    nn.SiLU(inplace=True),
                )
                in_channels = stem_channels
                stages: list[nn.Module] = []
                for out_channels, depth, stride in zip(
                    self.feature_channels,
                    self.stage_depths,
                    self.stage_strides,
                ):
                    blocks: list[nn.Module] = []
                    for block_idx in range(max(1, int(depth))):
                        blocks.append(
                            TinyResidualBlock(
                                in_channels,
                                out_channels,
                                stride=(stride if block_idx == 0 else 1),
                                expansion_ratio=self.expansion_ratio,
                                use_squeeze_excite=self.use_squeeze_excite,
                            )
                        )
                        in_channels = out_channels
                    stages.append(nn.Sequential(*blocks))
                self.stages = nn.Sequential(*stages)
                self.avg_pool = nn.AdaptiveAvgPool2d(1)
                self.max_pool = nn.AdaptiveMaxPool2d(1) if self.dual_pool else None
                pooled_dim = in_channels * (2 if self.dual_pool else 1)

            layers: list = []
            in_d = pooled_dim
            for _ in range(hidden_layers):
                layers.extend(
                    [
                        nn.Linear(in_d, hidden_dim),
                        nn.SiLU(inplace=True),
                        nn.Dropout(dropout),
                    ]
                )
                in_d = hidden_dim
            layers.append(nn.Linear(in_d, n_classes))
            self.classifier = nn.Sequential(nn.Flatten(), *layers)

        def forward_features(self, x):
            """Run the convolutional trunk and return pooled feature maps."""
            if self.architecture_version <= _LEGACY_TINY_ARCH_VERSION:
                return self.features(x)
            x = self.stem(x)
            for stage in self.stages:
                x = stage(x)
            pooled = [self.avg_pool(x)]
            if self.max_pool is not None:
                pooled.append(self.max_pool(x))
            return torch.cat(pooled, dim=1) if len(pooled) > 1 else pooled[0]

        def forward(self, x):
            """Pass input through the feature backbone and classification head."""
            x = self.forward_features(x)
            return self.classifier(x)

    return TinyClassifier


def rebuild_from_checkpoint(ckpt: dict[str, Any]):
    """Reconstruct and load a TinyClassifier from a saved checkpoint dict.

    Supports both legacy v1 checkpoints and the stronger v2 tiny backbone.

    Returns the model in eval mode on CPU.
    """
    state = ckpt["model_state_dict"]
    n_classes, hidden_count, hidden_dim = _infer_classifier_dims(state, ckpt)
    arch_config = _infer_tiny_architecture_config(ckpt, state)
    dropout = float(ckpt.get("dropout", 0.1))

    TinyClassifier = _build_tiny_classifier_class()
    model = TinyClassifier(
        n_classes=n_classes,
        hidden_layers=hidden_count,
        hidden_dim=hidden_dim,
        dropout=dropout,
        tiny_preset=arch_config["tiny_preset"],
        architecture_version=arch_config["architecture_version"],
        feature_channels=arch_config["feature_channels"],
        stage_depths=arch_config["stage_depths"],
        stage_strides=arch_config["stage_strides"],
        expansion_ratio=arch_config["expansion_ratio"],
        use_squeeze_excite=arch_config["use_squeeze_excite"],
        dual_pool=arch_config["dual_pool"],
    )
    model.load_state_dict(state)
    model.eval()
    return model


def load_tiny_classifier(path: str | Path, device: str = "cpu"):
    """Load a TinyClassifier from a .pth checkpoint file.

    Returns ``(model_in_eval_mode_on_device, full_ckpt_dict)``.
    """
    import torch

    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    model = rebuild_from_checkpoint(ckpt)
    model.to(device)
    return model, ckpt


def export_tiny_to_onnx(
    model: Any, ckpt: dict[str, Any], onnx_path: str | Path
) -> Path:
    """Export a TinyClassifier to ONNX format.

    Uses ``input_size`` from *ckpt* to build the dummy input (``[H, W]``,
    default ``[64, 128]``). Axes 0 (batch) are dynamic so any batch size
    works at runtime.

    Returns the path of the exported ONNX file.
    """
    import torch

    onnx_path = Path(onnx_path)
    input_h, input_w = ckpt.get("input_size", [64, 128])
    dummy = torch.zeros(1, 3, int(input_h), int(input_w))
    model.eval()
    # Prefer legacy TorchScript-based exporter (dynamo=False) which does not
    # require the optional onnxscript package. Fall back to positional call for
    # older PyTorch versions that do not accept the dynamo keyword argument.
    try:
        torch.onnx.export(
            model,
            dummy,
            str(onnx_path),
            dynamo=False,
            input_names=["images"],
            output_names=["logits"],
            dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=12,
        )
    except TypeError:
        torch.onnx.export(
            model,
            dummy,
            str(onnx_path),
            input_names=["images"],
            output_names=["logits"],
            dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=12,
        )
    return onnx_path


def load_tiny_onnx(onnx_path: str | Path, compute_runtime: str = "onnx_cpu"):
    """Load a TinyClassifier ONNX model as an ``onnxruntime.InferenceSession``.

    *compute_runtime* must be one of the canonical runtimes:
    ``onnx_coreml``, ``onnx_cpu``, ``onnx_cuda``, ``onnx_rocm``, or ``tensorrt``.
    """
    import onnxruntime as ort

    from hydra_suite.runtime.compute_runtime import derive_onnx_execution_providers

    providers = derive_onnx_execution_providers(compute_runtime)
    return ort.InferenceSession(str(onnx_path), providers=providers)


def run_tiny_onnx(session: Any, batch_np: Any) -> Any:
    """Run batch inference with an ONNX session.

    *batch_np*: float32 numpy array ``[N, 3, H, W]``

    Returns softmax probabilities as a float32 numpy array ``[N, n_classes]``.
    """
    import numpy as np

    input_name = session.get_inputs()[0].name
    logits = session.run(None, {input_name: batch_np.astype(np.float32)})[0]
    # Numerically stable softmax
    logits_shifted = logits - logits.max(axis=1, keepdims=True)
    exp_l = np.exp(logits_shifted)
    return (exp_l / exp_l.sum(axis=1, keepdims=True)).astype(np.float32)
