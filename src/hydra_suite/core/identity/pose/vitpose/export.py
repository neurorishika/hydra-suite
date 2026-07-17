"""ViTPose export recipe: a live torch model -> a deployable artifact.

This is the RECIPE, not the runtime. It is the piece ultralytics' model.export()
supplies for the YOLO backend and SLEAP's exporter supplies for SLEAP -- nobody supplies
it for ViTPose, so we do, and it lives beside the model whose quirks it encodes.

It deliberately knows nothing about PoseRuntimeConfig, artifact caching, signatures, or
where checkpoints live on disk. That is auto_export_vitpose_model's job in
backends/vitpose.py (Spec 3), mirroring auto_export_yolo_model (yolo.py:38) and
auto_export_sleap_model (sleap.py:1353), which use pose/artifacts.py's shared helpers.
Putting any of it here would break this package's leaf constraint.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from .config import IMAGE_SIZE_WH


class ExportError(RuntimeError):
    """Raised when a model cannot be exported safely."""


#: mmpose's exporter asserts opset_version == 11, but that is an mmpose-era
#: constraint, not one this model imposes. We require 14+ because the graph
#: this exporter produces (scaled-dot-product attention decomposition,
#: GELU, and the interpolation-free pos_embed slice-add baked in by
#: constant folding) relies on opset-14 operator semantics.
_MIN_OPSET = 14


def export_onnx(
    model: nn.Module,
    path: Path,
    *,
    opset: int = 17,
    dynamic_batch: bool = True,
    dataset_index: int | None = None,
) -> Path:
    """Export ViTPose to ONNX at a fixed 256x192 input.

    Fixed resolution is not a limitation we chose: pos_embed is a (1, 193, D)
    parameter with no interpolation path, and constant-folding bakes the
    [:, 1:] + [:, :1] slice-add into a constant. 256x192 is the only shape
    the checkpoints target.

    opset 17, not 11: mmpose's exporter asserts opset_version == 11, but
    that is an mmpose-era constraint, not a model one.
    """
    if model.training:
        raise ExportError(
            "model must be in eval() mode before export: the classic head's "
            "BatchNorm2d layers would otherwise emit training-mode "
            "BatchNormalization and silently produce garbage (and DropPath "
            "would trace to a random node)"
        )

    if opset < _MIN_OPSET:
        raise ExportError(
            f"opset={opset} is below the minimum this exporter supports "
            f"({_MIN_OPSET}). mmpose's exporter asserts opset_version == 11, "
            "but that is an mmpose-era constraint, not one this model's graph "
            f"tolerates -- request opset={_MIN_OPSET} or higher."
        )

    w, h = IMAGE_SIZE_WH
    dummy = torch.zeros(1, 3, h, w)
    path.parent.mkdir(parents=True, exist_ok=True)

    # MoE takes dataset_index; classic does not. Wrap so the exported graph has a single
    # tensor input either way, and so a concrete index bakes in ONE expert per block
    # rather than upstream's 6-expert masked sum.
    if dataset_index is not None:

        class _Fixed(nn.Module):
            def __init__(self, inner: nn.Module, idx: int) -> None:
                super().__init__()
                self.inner = inner
                self.idx = idx

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.inner(x, dataset_index=self.idx)

        model = _Fixed(model, dataset_index).eval()

    dynamic_axes = (
        {"input": {0: "batch"}, "output": {0: "batch"}} if dynamic_batch else None
    )
    # dynamo=False: torch >= 2.5 defaults torch.onnx.export to the dynamo-based
    # exporter, which requires the optional `onnxscript` package. This environment
    # only ships `onnx`/`onnxruntime`, and the legacy TorchScript-based exporter is
    # sufficient (and traces this model's control flow the same way regardless).
    try:
        torch.onnx.export(
            model,
            dummy,
            str(path),
            input_names=["input"],
            output_names=["output"],
            opset_version=opset,
            do_constant_folding=True,
            dynamic_axes=dynamic_axes,
            dynamo=False,
        )
    except (TypeError, ImportError) as e:
        # TypeError: a future torch drops the `dynamo` kwarg entirely.
        # ImportError/ModuleNotFoundError: the kwarg survives but the legacy
        # TorchScript-based exporter module it selects has been removed.
        raise ExportError(
            f"torch.onnx.export rejected the legacy exporter on torch "
            f"{torch.__version__}: {e}. Either install `onnxscript` and switch to "
            f"dynamo=True, or pin an older torch. dynamo=False is used here because "
            f"torch 2.11 defaults to the dynamo exporter, which requires onnxscript."
        ) from e
    return path
