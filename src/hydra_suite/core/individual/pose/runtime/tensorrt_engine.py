"""Native TensorRT engine runner and ONNX-to-engine builder for pose artifacts.

Moved verbatim from ``backends/sleap.py`` (``_build_trt_engine_from_onnx`` and
``_DirectTensorRTEngine``) so pose backends can share the native TensorRT path.
Behavior is unchanged; only the public names were renamed
``_build_trt_engine_from_onnx`` -> ``build_trt_engine_from_onnx`` and
``_DirectTensorRTEngine`` -> ``TensorRTEngineRunner``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# Max batch the native SLEAP TRT optimization profile supports. Matches the
# classifier backend's ORT-TRT-EP profile ceiling so both paths accept the same
# crop-batch range.
_TRT_PROFILE_MAX_BATCH = 512


def build_trt_engine_from_onnx(
    onnx_path: Path,
    engine_path: Path,
    workspace_gb: float = 4.0,
    fixed_hw: Optional[Tuple[int, int]] = None,
) -> bool:
    """Build a native TensorRT engine from an ONNX file and serialize it.

    Returns True when the engine was built and written to *engine_path*; False
    when building is not feasible in the current environment (CUDA / TensorRT not
    available or import fails).  Never raises — callers fall back to ORT-TRT-EP.

    This function requires CUDA and the ``tensorrt`` package.  On non-CUDA
    platforms (e.g. macOS/MPS) it returns False immediately without attempting
    any imports so the code path is safe to unit-test with mocks.

    *fixed_hw* is the (height, width) every crop is actually letterboxed to
    before inference (see ``_prepare_export_crop``/``SleapExportedBackend._input_hw``).
    SLEAP's sleap-nn exporter emits a fully-convolutional graph with symbolic
    H/W dims (it can technically run at any resolution), but our own
    pre-processing always resizes to one fixed size — so any dynamic H/W dims
    can be safely pinned to *fixed_hw* in the optimization profile, leaving
    only the batch dim actually dynamic.
    """
    try:
        import tensorrt as trt  # noqa: PLC0415
    except Exception:
        return False

    trt_logger = trt.Logger(trt.Logger.WARNING)
    try:
        builder = trt.Builder(trt_logger)
        network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(network_flags)
        parser = trt.OnnxParser(network, trt_logger)
        onnx_bytes = onnx_path.read_bytes()
        if not parser.parse(onnx_bytes):
            msgs = [parser.get_error(i).desc() for i in range(parser.num_errors)]
            logger.warning(
                "TRT ONNX parser failed for %s: %s", onnx_path, "; ".join(msgs)
            )
            return False

        config = builder.create_builder_config()
        workspace_bytes = int(workspace_gb * (1 << 30))
        if hasattr(config, "set_memory_pool_limit"):
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
        elif hasattr(config, "max_workspace_size"):
            config.max_workspace_size = workspace_bytes

        # The SLEAP UNet ONNX exports a dynamic leading (batch) dim. TensorRT
        # refuses to build a network with dynamic inputs unless an optimization
        # profile pins the min/opt/max shapes — without it,
        # build_serialized_network fails with "no optimization profile has been
        # defined" and we drop to the slow ORT-TRT-EP fallback every session.
        # Mirror the classifier backend's 1 / 64 / max convention (fp32 kept —
        # fp16 is deferred to preserve keypoint precision).
        profile = builder.create_optimization_profile()
        has_dynamic = False
        for idx in range(network.num_inputs):
            inp = network.get_input(idx)
            shape = list(inp.shape)
            if not any(int(d) < 0 for d in shape):
                continue
            non_batch = [int(d) for d in shape[1:]]
            dynamic_non_batch = [d for d in non_batch if d < 0]
            if dynamic_non_batch:
                # A dynamic H/W (or other non-batch dim) needs a concrete size
                # we can't infer from the graph alone. If the caller told us
                # what fixed size every crop is actually resized to, pin the
                # dynamic dims to that (in encounter order); otherwise bail to
                # ORT-EP rather than building a wrong/unusable engine.
                if fixed_hw is None or len(dynamic_non_batch) != len(fixed_hw):
                    logger.warning(
                        "TRT build: input %s has %d dynamic non-batch dim(s) %s "
                        "with no matching fixed-size hint (got %s); cannot build "
                        "a native engine — using ORT-TRT-EP.",
                        inp.name,
                        len(dynamic_non_batch),
                        shape,
                        fixed_hw,
                    )
                    return False
                fill = iter(fixed_hw)
                static = [int(next(fill)) if d < 0 else d for d in non_batch]
            else:
                static = non_batch
            min_shape = (1, *static)
            opt_shape = (min(64, _TRT_PROFILE_MAX_BATCH), *static)
            max_shape = (_TRT_PROFILE_MAX_BATCH, *static)
            profile.set_shape(inp.name, min_shape, opt_shape, max_shape)
            has_dynamic = True
        if has_dynamic:
            config.add_optimization_profile(profile)

        plan = builder.build_serialized_network(network, config)
        if plan is None:
            logger.warning(
                "TRT build_serialized_network returned None for %s", onnx_path
            )
            return False

        engine_path.write_bytes(bytes(plan))
        logger.info(
            "Built native TensorRT engine from ONNX: %s -> %s",
            onnx_path,
            engine_path,
        )
        return True
    except Exception as exc:
        logger.warning("Failed to build TensorRT engine from %s: %s", onnx_path, exc)
        return False


class TensorRTEngineRunner:
    def __init__(self, model_path: Path) -> None:
        import tensorrt as trt
        import torch

        self._trt = trt
        self._torch = torch
        self._logger = trt.Logger(trt.Logger.WARNING)
        self._runtime = trt.Runtime(self._logger)
        self._engine_bytes = model_path.read_bytes()
        self._engine = self._runtime.deserialize_cuda_engine(self._engine_bytes)
        if self._engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {model_path}")
        self._context = self._engine.create_execution_context()
        if self._context is None:
            raise RuntimeError(f"Failed to create TensorRT context: {model_path}")

        self.input_name = self._first_tensor_name(self._trt.TensorIOMode.INPUT)
        self.output_names = self._tensor_names(self._trt.TensorIOMode.OUTPUT)
        self.input_hw, self.input_channels = self._detect_input_spec()
        self.input_format = self._detect_input_format()
        self.model_min_batch = self._detect_min_batch()

    def _tensor_names(self, mode: Any) -> List[str]:
        names: List[str] = []
        if hasattr(self._engine, "num_io_tensors"):
            for idx in range(int(self._engine.num_io_tensors)):
                name = self._engine.get_tensor_name(idx)
                if self._engine.get_tensor_mode(name) == mode:
                    names.append(str(name))
            return names
        if hasattr(self._engine, "num_bindings"):
            for idx in range(int(self._engine.num_bindings)):
                is_input = bool(self._engine.binding_is_input(idx))
                if is_input == (mode == self._trt.TensorIOMode.INPUT):
                    names.append(str(self._engine.get_binding_name(idx)))
        return names

    def _first_tensor_name(self, mode: Any) -> str:
        names = self._tensor_names(mode)
        if not names:
            raise RuntimeError("TensorRT engine is missing an input tensor.")
        return names[0]

    def _tensor_shape(self, name: str) -> Tuple[int, ...]:
        if hasattr(self._engine, "get_tensor_shape"):
            return tuple(int(v) for v in self._engine.get_tensor_shape(name))
        if hasattr(self._engine, "get_binding_shape"):
            index = self._engine.get_binding_index(name)
            return tuple(int(v) for v in self._engine.get_binding_shape(index))
        raise RuntimeError("TensorRT engine does not expose tensor shapes.")

    def _tensor_dtype(self, name: str):
        if hasattr(self._engine, "get_tensor_dtype"):
            return self._engine.get_tensor_dtype(name)
        if hasattr(self._engine, "get_binding_dtype"):
            index = self._engine.get_binding_index(name)
            return self._engine.get_binding_dtype(index)
        raise RuntimeError("TensorRT engine does not expose tensor dtypes.")

    def _detect_input_spec(self) -> Tuple[Optional[Tuple[int, int]], Optional[int]]:
        shape = list(self._tensor_shape(self.input_name))
        dims: List[int] = []
        for dim in shape:
            dims.append(int(dim) if int(dim) > 0 else -1)
        input_hw = None
        input_channels = None
        if len(dims) >= 4:
            if dims[1] in (1, 3):
                input_channels = int(dims[1])
                if dims[-2] > 0 and dims[-1] > 0:
                    input_hw = (int(dims[-2]), int(dims[-1]))
            elif dims[-1] in (1, 3):
                input_channels = int(dims[-1])
                if dims[-3] > 0 and dims[-2] > 0:
                    input_hw = (int(dims[-3]), int(dims[-2]))
        return input_hw, input_channels

    def _detect_input_format(self) -> Dict[str, Any]:
        shape = list(self._tensor_shape(self.input_name))
        layout = "nhwc"
        if len(shape) >= 4:
            if int(shape[1]) in (1, 3):
                layout = "nchw"
            elif int(shape[-1]) in (1, 3):
                layout = "nhwc"
        dtype_name = str(self._tensor_dtype(self.input_name)).lower()
        return {"layout": layout, "is_float": "float" in dtype_name}

    def _detect_min_batch(self) -> Optional[int]:
        if hasattr(self._engine, "get_tensor_profile_shape"):
            try:
                min_shape, _opt_shape, _max_shape = (
                    self._engine.get_tensor_profile_shape(
                        self.input_name,
                        0,
                    )
                )
                batch = int(min_shape[0])
                return batch if batch > 0 else None
            except Exception:
                pass
        if hasattr(self._engine, "get_profile_shape"):
            try:
                index = self._engine.get_binding_index(self.input_name)
                min_shape, _opt_shape, _max_shape = self._engine.get_profile_shape(
                    0, index
                )
                batch = int(min_shape[0])
                return batch if batch > 0 else None
            except Exception:
                pass
        shape = self._tensor_shape(self.input_name)
        if shape and int(shape[0]) > 0:
            return int(shape[0])
        return None

    def _torch_dtype(self, trt_dtype: Any):
        np_dtype = np.dtype(self._trt.nptype(trt_dtype))
        mapping = {
            np.dtype(np.float32): self._torch.float32,
            np.dtype(np.float16): self._torch.float16,
            np.dtype(np.int32): self._torch.int32,
            np.dtype(np.int8): self._torch.int8,
            np.dtype(np.uint8): self._torch.uint8,
            np.dtype(np.bool_): self._torch.bool,
        }
        if np_dtype not in mapping:
            raise RuntimeError(f"Unsupported TensorRT dtype: {np_dtype}")
        return mapping[np_dtype]

    def run(self, batch: np.ndarray) -> Dict[str, np.ndarray]:
        input_shape = tuple(int(v) for v in batch.shape)
        if hasattr(self._context, "set_input_shape"):
            self._context.set_input_shape(self.input_name, input_shape)
        elif hasattr(self._context, "set_binding_shape"):
            index = self._engine.get_binding_index(self.input_name)
            self._context.set_binding_shape(index, input_shape)

        input_dtype = self._torch_dtype(self._tensor_dtype(self.input_name))
        input_tensor = self._torch.as_tensor(
            np.ascontiguousarray(batch),
            device="cuda",
            dtype=input_dtype,
        )
        output_tensors: Dict[str, Any] = {}
        if hasattr(self._context, "set_tensor_address"):
            self._context.set_tensor_address(
                self.input_name, int(input_tensor.data_ptr())
            )
            for name in self.output_names:
                out_shape = tuple(int(v) for v in self._context.get_tensor_shape(name))
                out_tensor = self._torch.empty(
                    out_shape,
                    device="cuda",
                    dtype=self._torch_dtype(self._tensor_dtype(name)),
                )
                self._context.set_tensor_address(name, int(out_tensor.data_ptr()))
                output_tensors[name] = out_tensor
            stream = self._torch.cuda.current_stream().cuda_stream
            ok = self._context.execute_async_v3(stream_handle=stream)
        else:
            bindings: List[int] = [0] * int(self._engine.num_bindings)
            in_index = self._engine.get_binding_index(self.input_name)
            bindings[in_index] = int(input_tensor.data_ptr())
            for name in self.output_names:
                out_index = self._engine.get_binding_index(name)
                out_shape = tuple(
                    int(v) for v in self._context.get_binding_shape(out_index)
                )
                out_tensor = self._torch.empty(
                    out_shape,
                    device="cuda",
                    dtype=self._torch_dtype(self._tensor_dtype(name)),
                )
                bindings[out_index] = int(out_tensor.data_ptr())
                output_tensors[name] = out_tensor
            stream = self._torch.cuda.current_stream().cuda_stream
            ok = self._context.execute_async_v2(bindings=bindings, stream_handle=stream)
        if not ok:
            raise RuntimeError("TensorRT inference failed.")
        self._torch.cuda.current_stream().synchronize()
        return {
            name: tensor.detach().cpu().numpy()
            for name, tensor in output_tensors.items()
        }

    def run_cuda(self, batch_cuda: Any) -> Dict[str, np.ndarray]:
        """Like :meth:`run` but accepts a device-resident CUDA tensor.

        Eliminates the ``np.ascontiguousarray`` → ``torch.as_tensor(device='cuda')``
        host-to-device copy when the batch is already on the GPU.  The output
        keypoint tensors are small so the final GPU→CPU transfer is negligible.
        """
        input_shape = tuple(int(v) for v in batch_cuda.shape)
        if hasattr(self._context, "set_input_shape"):
            self._context.set_input_shape(self.input_name, input_shape)
        elif hasattr(self._context, "set_binding_shape"):
            index = self._engine.get_binding_index(self.input_name)
            self._context.set_binding_shape(index, input_shape)

        input_dtype = self._torch_dtype(self._tensor_dtype(self.input_name))
        input_tensor = batch_cuda.contiguous().to(dtype=input_dtype)

        output_tensors: Dict[str, Any] = {}
        if hasattr(self._context, "set_tensor_address"):
            self._context.set_tensor_address(
                self.input_name, int(input_tensor.data_ptr())
            )
            for name in self.output_names:
                out_shape = tuple(int(v) for v in self._context.get_tensor_shape(name))
                out_tensor = self._torch.empty(
                    out_shape,
                    device="cuda",
                    dtype=self._torch_dtype(self._tensor_dtype(name)),
                )
                self._context.set_tensor_address(name, int(out_tensor.data_ptr()))
                output_tensors[name] = out_tensor
            stream = self._torch.cuda.current_stream().cuda_stream
            ok = self._context.execute_async_v3(stream_handle=stream)
        else:
            bindings: List[int] = [0] * int(self._engine.num_bindings)
            in_index = self._engine.get_binding_index(self.input_name)
            bindings[in_index] = int(input_tensor.data_ptr())
            for name in self.output_names:
                out_index = self._engine.get_binding_index(name)
                out_shape = tuple(
                    int(v) for v in self._context.get_binding_shape(out_index)
                )
                out_tensor = self._torch.empty(
                    out_shape,
                    device="cuda",
                    dtype=self._torch_dtype(self._tensor_dtype(name)),
                )
                bindings[out_index] = int(out_tensor.data_ptr())
                output_tensors[name] = out_tensor
            stream = self._torch.cuda.current_stream().cuda_stream
            ok = self._context.execute_async_v2(bindings=bindings, stream_handle=stream)
        if not ok:
            raise RuntimeError("TensorRT inference failed.")
        self._torch.cuda.current_stream().synchronize()
        return {
            name: tensor.detach().cpu().numpy()
            for name, tensor in output_tensors.items()
        }

    def close(self) -> None:
        self._context = None
        self._engine = None
        self._runtime = None
