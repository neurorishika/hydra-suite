"""Shared tiny CNN classifier for training (runner.py) and inference (task_workers.py)."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _build_tiny_classifier_class():
    """Return TinyClassifier class (deferred torch import)."""
    import torch.nn as nn

    class TinyClassifier(nn.Module):
        """4-block Conv backbone + flexible MLP head for N-class image classification."""

        def __init__(
            self,
            n_classes: int,
            hidden_layers: int = 1,
            hidden_dim: int = 64,
            dropout: float = 0.2,
        ):
            super().__init__()
            self.n_classes = n_classes
            self.hidden_layers = hidden_layers
            self.hidden_dim = hidden_dim
            self.features = nn.Sequential(
                nn.Conv2d(3, 16, 3, stride=2, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 32, 3, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, 3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, 3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1),
            )
            layers: list = []
            in_d = 64
            for _ in range(hidden_layers):
                layers.extend(
                    [
                        nn.Linear(in_d, hidden_dim),
                        nn.ReLU(inplace=True),
                        nn.Dropout(dropout),
                    ]
                )
                in_d = hidden_dim
            layers.append(nn.Linear(in_d, n_classes))
            self.classifier = nn.Sequential(nn.Flatten(), *layers)

        def forward(self, x):
            """Pass input through the feature backbone and classification head."""
            x = self.features(x)
            return self.classifier(x)

    return TinyClassifier


def rebuild_from_checkpoint(ckpt: dict[str, Any]):
    """Reconstruct and load a TinyClassifier from a saved checkpoint dict.

    Infers ``n_classes``, ``hidden_layers``, and ``hidden_dim`` from the
    state dict weight shapes — so checkpoints without explicit hyper-params work.

    Returns the model in eval mode on CPU.
    """
    state = ckpt["model_state_dict"]

    # Collect Linear weight keys in the classifier branch (index-sorted)
    linear_keys = sorted(
        [k for k in state if k.startswith("classifier.") and k.endswith(".weight")],
        key=lambda k: int(k.split(".")[1]),
    )
    if not linear_keys:
        raise ValueError("No Linear weight keys found in checkpoint classifier branch.")

    n_classes = int(state[linear_keys[-1]].shape[0])
    hidden_count = len(linear_keys) - 1
    hidden_dim = int(state[linear_keys[0]].shape[0]) if hidden_count > 0 else 64

    TinyClassifier = _build_tiny_classifier_class()
    model = TinyClassifier(
        n_classes=n_classes, hidden_layers=hidden_count, hidden_dim=hidden_dim
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
