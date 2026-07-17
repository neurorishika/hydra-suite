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


def _require_eval_mode(model: nn.Module) -> None:
    """Shared guard for every export target: a train()-mode model would silently
    emit training-mode BatchNorm2d (classic head) and a random DropPath node,
    producing a garbage artifact rather than a loud failure."""
    if model.training:
        raise ExportError(
            "model must be in eval() mode before export: the classic head's "
            "BatchNorm2d layers would otherwise emit training-mode "
            "BatchNormalization and silently produce garbage (and DropPath "
            "would trace to a random node)"
        )


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
    _require_eval_mode(model)

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


def build_tensorrt_engine(
    onnx_path: Path,
    engine_path: Path,
    *,
    fp16: bool = False,
    workspace_gb: float = 4.0,
    max_batch: int = 64,
) -> Path:
    """Build a native TensorRT engine from an exported ONNX file.

    Structural precedent: sleap.py's ``_build_trt_engine_from_onnx``
    (sleap.py:374-465) -- builder + explicit-batch network + ``OnnxParser``
    from an existing ONNX file, never ``trtexec`` (this repo's convention is
    the Python API). Unlike that helper, this one raises instead of returning
    False: it is the export recipe, not a runtime with an ORT-TRT-EP fallback
    to drop to, so a build failure must surface as an ExportError.

    fp16 defaults to False -- a deliberate decision, not an oversight. SLEAP
    keeps FP32 "to preserve keypoint precision" (sleap.py:420-421) and
    compute_runtime.py:141-142 states the same rule. ViTPose's entire value is
    sub-pixel keypoint accuracy, so the OBB path's half=True is the wrong
    analog here.
    """
    import tensorrt as trt

    trt_logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(trt_logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, trt_logger)
    onnx_bytes = onnx_path.read_bytes()
    if not parser.parse(onnx_bytes):
        msgs = [parser.get_error(i).desc() for i in range(parser.num_errors)]
        raise ExportError(
            f"TensorRT ONNX parser failed for {onnx_path}: {'; '.join(msgs)}"
        )

    config = builder.create_builder_config()
    workspace_bytes = int(workspace_gb * (1 << 30))
    # Version-tolerant workspace-limit call: TRT >= 8.4 exposes
    # set_memory_pool_limit; TRT 10 removed max_workspace_size entirely, so
    # hardcoding either alone is brittle. Mirrors sleap.py:410-413.
    if hasattr(config, "set_memory_pool_limit"):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    elif hasattr(config, "max_workspace_size"):  # pragma: no cover (TRT < 8.4)
        config.max_workspace_size = workspace_bytes

    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    # export_onnx's graph has a single dynamic leading (batch) dim. TensorRT
    # refuses to build a network with dynamic inputs unless an optimization
    # profile pins the min/opt/max shapes. Mirror sleap.py:420-444's 1 /
    # min(64, max_batch) / max_batch convention.
    profile = builder.create_optimization_profile()
    has_dynamic = False
    for idx in range(network.num_inputs):
        inp = network.get_input(idx)
        shape = list(inp.shape)
        if not any(int(d) < 0 for d in shape):
            continue
        # Only the leading batch dim may be dynamic; a dynamic H/W/C would
        # need a concrete size we can't infer, so refuse rather than build a
        # silently-wrong engine (sleap.py:429-439's guard).
        if any(int(d) < 0 for d in shape[1:]):
            raise ExportError(
                f"input {inp.name!r} has dynamic non-batch dims {shape}: this "
                "exporter only supports a dynamic leading batch dim, refusing "
                "to build an engine that would silently pin the wrong shape."
            )
        static = [int(d) for d in shape[1:]]
        min_shape = (1, *static)
        opt_shape = (min(64, max_batch), *static)
        max_shape = (max_batch, *static)
        profile.set_shape(inp.name, min_shape, opt_shape, max_shape)
        has_dynamic = True
    if has_dynamic:
        config.add_optimization_profile(profile)

    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise ExportError(
            f"TensorRT build_serialized_network returned None for {onnx_path}"
        )

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(bytes(plan))
    return engine_path


def export_coreml(
    model: nn.Module,
    path: Path,
    *,
    compute_units: str = "ALL",
) -> Path:
    """Export ViTPose to a CoreML .mlpackage at a fixed (1, 3, 256, 192) input.

    Batch stays 1, not dynamic: the OBB CoreML path pins batch=1 for a documented
    reason (core/inference/runtime_artifacts.py:293-299) -- dynamic batch combined
    with spatial dims crashes the CoreML compiler. This model has the same fixed-
    resolution constraint as export_onnx (pos_embed has no interpolation path), so
    a fully-static input shape costs nothing extra here.

    Uses torch.jit.trace + coremltools' mlprogram conversion, mirroring the OBB
    export path's use of ultralytics' underlying trace-based CoreML conversion.
    """
    import coremltools as ct

    _require_eval_mode(model)

    w, h = IMAGE_SIZE_WH
    dummy = torch.zeros(1, 3, h, w)
    path.parent.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        traced = torch.jit.trace(model, dummy)

    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="input", shape=dummy.shape)],
        outputs=[ct.TensorType(name="output")],
        convert_to="mlprogram",
        compute_units=getattr(ct.ComputeUnit, compute_units),
        # mlprogram defaults to FLOAT16 compute precision, which alone
        # accounts for ~1e-3 max-abs drift against the FP32 torch reference
        # (Gate D's bound). FLOAT32 keeps this export numerically comparable
        # to export_onnx/build_tensorrt_engine rather than silently trading
        # keypoint precision for speed -- the same "no half for keypoints"
        # rule build_tensorrt_engine documents (sleap.py:420-421).
        compute_precision=ct.precision.FLOAT32,
    )
    mlmodel.save(str(path))
    return path
