"""
SLEAP pose backend implementation.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
import tempfile
import time
from multiprocessing import shared_memory
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from hydra_suite.core.identity.pose.artifacts import (
    artifact_meta_matches,
    path_fingerprint_token,
    write_artifact_meta,
)
from hydra_suite.core.identity.pose.backends.sleap_utils import (
    derive_sleap_export_input_hw,
    looks_like_sleap_export_path,
    run_cli_command,
)
from hydra_suite.core.identity.pose.types import PoseResult, PoseRuntimeConfig
from hydra_suite.core.identity.pose.utils import (
    coerce_prediction_batch,
    empty_pose_result,
    summarize_keypoints,
)
from hydra_suite.runtime.compute_runtime import derive_onnx_execution_providers

logger = logging.getLogger(__name__)

ExportTransform = Tuple[float, float, float, int, int]


def _resolve_export_model_path(
    exported_model_path: str,
    runtime_flavor: str,
) -> Path:
    path = Path(str(exported_model_path or "")).expanduser().resolve()
    runtime = str(runtime_flavor or "onnx").strip().lower()
    if path.is_dir():
        if runtime == "onnx":
            matches = sorted(path.rglob("*.onnx"))
            if not matches:
                raise RuntimeError(
                    f"No ONNX artifact found in export directory: {path}"
                )
            return matches[0]
        matches = sorted(list(path.rglob("*.engine")) + list(path.rglob("*.trt")))
        if not matches:
            onnx_matches = sorted(path.rglob("*.onnx"))
            if onnx_matches:
                return onnx_matches[0]
            raise RuntimeError(
                f"No TensorRT artifact found in export directory: {path}"
            )
        return matches[0]
    return path


def _extract_meta_value(meta: Any, keys: Sequence[str], default: Any = None) -> Any:
    for key in keys:
        if isinstance(meta, dict) and key in meta:
            return meta.get(key)
        if hasattr(meta, key):
            return getattr(meta, key)
    return default


def _detect_export_input_spec(
    exported_model_path: str,
) -> Tuple[Optional[Tuple[int, int]], Optional[int]]:
    input_hw: Optional[Tuple[int, int]] = None
    input_channels: Optional[int] = None
    try:
        meta_path = Path(str(exported_model_path)).expanduser().resolve()
        if meta_path.is_file():
            meta_path = meta_path.parent
        metadata_file = None
        for candidate_name in (
            "metadata.json",
            "export_metadata.json",
            ".runtime_meta.json",
        ):
            candidate = meta_path / candidate_name
            if candidate.exists() and candidate.is_file():
                metadata_file = candidate
                break
        if metadata_file is None:
            return None, None
        import json

        meta = json.loads(metadata_file.read_text(encoding="utf-8"))
    except Exception:
        return None, None

    hw = _extract_meta_value(
        meta,
        ("crop_size", "crop_hw", "input_hw", "image_hw"),
        None,
    )
    if isinstance(hw, (list, tuple)) and len(hw) >= 2:
        try:
            input_hw = (int(hw[0]), int(hw[1]))
        except Exception:
            input_hw = None

    input_shape = _extract_meta_value(meta, ("input_image_shape", "input_shape"), None)
    if input_shape is not None:
        try:
            arr = np.asarray(input_shape).reshape((-1,))
            if arr.size >= 4:
                if int(arr[-1]) in (1, 3):
                    input_channels = int(arr[-1])
                    input_hw = (int(arr[-3]), int(arr[-2]))
                elif int(arr[1]) in (1, 3):
                    input_channels = int(arr[1])
                    input_hw = (int(arr[-2]), int(arr[-1]))
        except Exception:
            pass

    channels = _extract_meta_value(
        meta, ("input_channels", "channels", "num_channels"), None
    )
    if channels is not None:
        try:
            input_channels = int(channels)
        except Exception:
            pass
    return input_hw, input_channels


def _detect_onnx_input_spec(
    session: Any,
) -> Tuple[Optional[Tuple[int, int]], Optional[int]]:
    if session is None or not hasattr(session, "get_inputs"):
        return None, None
    try:
        inputs = session.get_inputs()
    except Exception:
        return None, None
    if not inputs:
        return None, None

    try:
        shape = list(getattr(inputs[0], "shape", []) or [])
    except Exception:
        shape = []
    dims: List[int] = []
    for dim in shape:
        try:
            dims.append(int(dim))
        except Exception:
            dims.append(-1)

    input_hw = None
    input_channels = None
    if len(dims) >= 4:
        if dims[-1] in (1, 3):
            input_channels = int(dims[-1])
            if dims[-3] > 0 and dims[-2] > 0:
                input_hw = (int(dims[-3]), int(dims[-2]))
        elif dims[1] in (1, 3):
            input_channels = int(dims[1])
            if dims[-2] > 0 and dims[-1] > 0:
                input_hw = (int(dims[-2]), int(dims[-1]))
    return input_hw, input_channels


def _detect_onnx_input_format(session: Any) -> Optional[Dict[str, Any]]:
    if session is None or not hasattr(session, "get_inputs"):
        return None
    try:
        inputs = session.get_inputs()
        if not inputs:
            return None
        inp = inputs[0]
        raw_type = str(getattr(inp, "type", "")).lower()
        shape = list(getattr(inp, "shape", []) or [])
    except Exception:
        return None

    dims: List[int] = []
    for dim in shape:
        try:
            dims.append(int(dim))
        except Exception:
            dims.append(-1)

    layout = "nhwc"
    if len(dims) >= 4:
        if dims[1] in (1, 3):
            layout = "nchw"
        elif dims[-1] in (1, 3):
            layout = "nhwc"
    return {"layout": layout, "is_float": "float" in raw_type}


def _detect_onnx_min_batch(session: Any) -> Optional[int]:
    if session is None or not hasattr(session, "get_inputs"):
        return None
    try:
        inputs = session.get_inputs()
        if not inputs:
            return None
        shape = getattr(inputs[0], "shape", [])
        if not shape:
            return None
        batch = int(shape[0])
        return batch if batch > 0 else None
    except Exception:
        return None


def _prepare_export_crop(
    crop: np.ndarray,
    input_hw: Tuple[int, int],
    input_channels: Optional[int],
) -> Tuple[np.ndarray, ExportTransform]:
    # The pipeline's canonical crops are float32 [0, 1]; casting them straight to
    # uint8 floors every pixel to 0 (all-black image → SLEAP returns zero-conf
    # keypoints). Scale [0, 1] floats to [0, 255] before the cast; pass uint8 /
    # [0, 255] inputs through. Mirrors SleapServiceBackend._to_uint8_image; this
    # is the shared choke point for the onnx_cuda/tensorrt exported CPU path.
    arr = np.asarray(crop)
    if arr.dtype != np.uint8:
        a = arr.astype(np.float32)
        finite = a[np.isfinite(a)]
        if finite.size and float(finite.max()) <= 1.0 + 1e-3:
            a = a * 255.0
        arr = np.clip(np.nan_to_num(a), 0.0, 255.0).astype(np.uint8)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    if arr.ndim != 3:
        raise RuntimeError(
            f"Invalid crop shape for SLEAP exported inference: {arr.shape}"
        )
    if arr.shape[2] > 3:
        arr = arr[:, :, :3]

    orig_h, orig_w = int(arr.shape[0]), int(arr.shape[1])
    if input_channels == 1 and arr.shape[2] != 1:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)[:, :, None]
    elif input_channels is None or input_channels == 3:
        if arr.shape[2] == 1:
            arr = np.repeat(arr, 3, axis=2)
        elif arr.shape[2] == 3:
            arr = arr[:, :, ::-1].copy()

    transform: ExportTransform = (1.0, 0.0, 0.0, orig_w, orig_h)

    if input_hw is not None:
        out_h, out_w = int(input_hw[0]), int(input_hw[1])
        if out_h > 0 and out_w > 0 and (orig_h != out_h or orig_w != out_w):
            scale = min(float(out_w) / float(orig_w), float(out_h) / float(orig_h))
            new_w = max(1, int(round(float(orig_w) * scale)))
            new_h = max(1, int(round(float(orig_h) * scale)))
            pad_x = int((out_w - new_w) // 2)
            pad_y = int((out_h - new_h) // 2)
            if arr.shape[2] == 1:
                resized = cv2.resize(
                    arr[:, :, 0],
                    (new_w, new_h),
                    interpolation=cv2.INTER_LINEAR,
                )
                canvas = np.zeros((out_h, out_w, 1), dtype=np.uint8)
                canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w, 0] = resized
                arr = canvas
            else:
                resized = cv2.resize(
                    arr, (new_w, new_h), interpolation=cv2.INTER_LINEAR
                )
                canvas = np.zeros((out_h, out_w, arr.shape[2]), dtype=np.uint8)
                canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized
                arr = canvas
            transform = (float(scale), float(pad_x), float(pad_y), orig_w, orig_h)

    return np.asarray(arr, dtype=np.uint8), transform


def _restore_export_keypoints(
    keypoints: np.ndarray,
    transform: ExportTransform,
) -> np.ndarray:
    arr = np.asarray(keypoints, dtype=np.float32).copy()
    if arr.ndim != 2 or arr.shape[1] < 2:
        return arr
    scale, pad_x, pad_y, orig_w, orig_h = transform
    scale = float(scale) if float(scale) > 0.0 else 1.0
    arr[:, 0] = (arr[:, 0] - float(pad_x)) / scale
    arr[:, 1] = (arr[:, 1] - float(pad_y)) / scale
    arr[:, 0] = np.clip(arr[:, 0], 0.0, float(max(0, int(orig_w) - 1)))
    arr[:, 1] = np.clip(arr[:, 1], 0.0, float(max(0, int(orig_h) - 1)))
    return arr


def _coerce_export_output(
    outputs: Any,
    output_names: Sequence[str],
    batch_size: int,
) -> List[Optional[np.ndarray]]:
    if isinstance(outputs, dict):
        raw = outputs
    elif isinstance(outputs, (list, tuple)):
        raw = {
            str(name): value for name, value in zip(list(output_names), list(outputs))
        }
    else:
        raw = outputs
    return coerce_prediction_batch(raw, batch_size)


def _canonical_export_runtime(
    runtime_request: str,
    runtime_flavor: str,
    device: str,
) -> str:
    req = str(runtime_request or "").strip().lower().replace("-", "_")
    rt = str(runtime_flavor or "onnx").strip().lower()
    dev = str(device or "cpu").strip().lower()
    if req in {"onnx_coreml", "onnx_mps"}:
        return "onnx_coreml"
    if req == "onnx_cuda":
        return "onnx_cuda"
    if req == "onnx_cpu":
        return "onnx_cpu"
    if req.startswith("tensorrt") or rt == "tensorrt":
        return "tensorrt"
    if dev == "mps":
        return "onnx_coreml"
    if dev.startswith("cuda"):
        return "onnx_cuda"
    return "onnx_cpu"


class _DirectOnnxSession:
    def __init__(
        self,
        model_path: Path,
        compute_runtime: str,
    ) -> None:
        import onnxruntime as ort

        self._session = ort.InferenceSession(
            str(model_path),
            providers=derive_onnx_execution_providers(compute_runtime),
        )
        self.input_name = self._session.get_inputs()[0].name
        self.output_names = [output.name for output in self._session.get_outputs()]
        self.input_hw, self.input_channels = _detect_onnx_input_spec(self._session)
        self.input_format = _detect_onnx_input_format(self._session)
        self.model_min_batch = _detect_onnx_min_batch(self._session)

    def run(self, batch: np.ndarray) -> Any:
        return self._session.run(None, {self.input_name: batch})

    def close(self) -> None:
        self._session = None


def _build_trt_engine_from_onnx(
    onnx_path: Path,
    engine_path: Path,
    workspace_gb: float = 4.0,
) -> bool:
    """Build a native TensorRT engine from an ONNX file and serialize it.

    Returns True when the engine was built and written to *engine_path*; False
    when building is not feasible in the current environment (CUDA / TensorRT not
    available or import fails).  Never raises — callers fall back to ORT-TRT-EP.

    This function requires CUDA and the ``tensorrt`` package.  On non-CUDA
    platforms (e.g. macOS/MPS) it returns False immediately without attempting
    any imports so the code path is safe to unit-test with mocks.
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


class _DirectTensorRTEngine:
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


class SleapExportedBackend:
    """Direct HYDRA-side execution for exported SLEAP ONNX and TensorRT artifacts."""

    def __init__(
        self,
        exported_model_path: str,
        runtime_flavor: str,
        runtime_request: str,
        device: str,
        keypoint_names: Sequence[str],
        min_valid_conf: float = 0.2,
        batch_size: int = 4,
        export_input_hw: Optional[Tuple[int, int]] = None,
    ) -> None:
        self.runtime_flavor = str(runtime_flavor or "onnx").strip().lower()
        self.runtime_request = (
            str(runtime_request or self.runtime_flavor).strip().lower()
        )
        self.device = str(device or "cpu").strip() or "cpu"
        self.output_keypoint_names = [str(value) for value in keypoint_names]
        self.min_valid_conf = float(min_valid_conf)
        self.batch_size = max(1, int(batch_size))
        self.model_path = _resolve_export_model_path(
            exported_model_path, self.runtime_flavor
        )
        self._last_profile: Dict[str, float] = {}

        metadata_hw, metadata_channels = _detect_export_input_spec(str(self.model_path))
        derived_hw = None
        try:
            derived_hw = derive_sleap_export_input_hw(str(self.model_path.parent))
        except Exception:
            derived_hw = None
        self._input_hw = (
            (int(export_input_hw[0]), int(export_input_hw[1]))
            if isinstance(export_input_hw, (tuple, list))
            and len(export_input_hw) >= 2
            and int(export_input_hw[0]) > 0
            and int(export_input_hw[1]) > 0
            else metadata_hw or derived_hw
        )
        self._input_channels = metadata_channels
        self._output_names: List[str] = []

        if self.runtime_flavor == "tensorrt":
            self._runner = self._init_tensorrt_runner(self.model_path)
        else:
            canonical_runtime = _canonical_export_runtime(
                self.runtime_request,
                self.runtime_flavor,
                self.device,
            )
            self._runner = _DirectOnnxSession(self.model_path, canonical_runtime)

        self._output_names = list(getattr(self._runner, "output_names", []) or [])
        self._input_hw = getattr(self._runner, "input_hw", None) or self._input_hw
        self._input_channels = (
            getattr(self._runner, "input_channels", None) or self._input_channels
        )
        self._input_format = getattr(self._runner, "input_format", None) or {
            "layout": "nchw",
            "is_float": False,
        }
        self._model_min_batch = getattr(self._runner, "model_min_batch", None)

        if self._input_hw is None:
            raise RuntimeError(
                "SLEAP exported runtime requires fixed input shape metadata or explicit export_input_hw."
            )

    def _init_tensorrt_runner(
        self, model_path: Path
    ) -> "_DirectTensorRTEngine | _DirectOnnxSession":
        """Select the best backend when runtime_flavor == 'tensorrt'.

        Decision tree
        -------------
        1. If *model_path* is a native ``.engine``/``.trt`` file → attempt to
           deserialize directly.  On version-mismatch / stale-engine the
           ``deserialize_cuda_engine`` call returns ``None`` and
           ``_DirectTensorRTEngine`` raises; in that case fall through to (2).
        2. If *model_path* is an ``.onnx`` file (fallback from
           ``_resolve_export_model_path`` when no ``.trt`` was found) OR after a
           stale-engine failure → attempt to build a fresh native TRT engine from
           the ONNX via :func:`_build_trt_engine_from_onnx` and deserialize it.
        3. If building is not feasible (CUDA absent, TRT import fails, builder
           error) → fall back to ``_DirectOnnxSession`` with the ORT TensorRT-EP.
           Emit a WARNING so operators know which slow path is active.

        The ONNX path from step 2/3 is stored beside the resolved ``model_path``
        directory with a ``.trt`` suffix so the next startup skips the build.
        """
        is_native_engine = model_path.suffix.lower() in {".engine", ".trt"}
        is_onnx = model_path.suffix.lower() == ".onnx"

        if is_native_engine:
            try:
                return _DirectTensorRTEngine(model_path)
            except RuntimeError as exc:
                logger.warning(
                    "TensorRT engine at %s failed to deserialize (stale/version "
                    "mismatch: %s). Attempting to rebuild from ONNX.",
                    model_path,
                    exc,
                )
                # Locate a sibling .onnx to rebuild from
                onnx_siblings = sorted(model_path.parent.rglob("*.onnx"))
                if not onnx_siblings:
                    logger.warning(
                        "No sibling ONNX file found in %s — cannot rebuild TRT "
                        "engine. Falling back to ORT TensorRT-EP (slow: ~8 s init "
                        "overhead per session).",
                        model_path.parent,
                    )
                    return self._ort_trt_ep_fallback()
                onnx_path = onnx_siblings[0]
                rebuilt = model_path  # rebuild in-place
                if _build_trt_engine_from_onnx(onnx_path, rebuilt):
                    try:
                        return _DirectTensorRTEngine(rebuilt)
                    except RuntimeError as build_exc:
                        logger.warning(
                            "Rebuilt TRT engine at %s still fails: %s. "
                            "Falling back to ORT TensorRT-EP.",
                            rebuilt,
                            build_exc,
                        )
                        return self._ort_trt_ep_fallback()
                return self._ort_trt_ep_fallback()

        if is_onnx:
            # No .trt engine exists yet — attempt to build one beside the .onnx
            engine_path = model_path.with_suffix(".trt")
            if _build_trt_engine_from_onnx(model_path, engine_path):
                try:
                    engine = _DirectTensorRTEngine(engine_path)
                    # Update model_path so the next warmup / profile reflects it
                    self.model_path = engine_path
                    return engine
                except RuntimeError as exc:
                    logger.warning(
                        "Freshly built TRT engine at %s fails to deserialize: %s. "
                        "Falling back to ORT TensorRT-EP.",
                        engine_path,
                        exc,
                    )
            return self._ort_trt_ep_fallback()

        # Unknown suffix — best-effort ONNX session
        return self._ort_trt_ep_fallback()

    def _ort_trt_ep_fallback(self) -> "_DirectOnnxSession":
        """Return an ORT session using the TensorRT ExecutionProvider.

        This is the ~8 s per-session initialisation path.  Callers emit a
        WARNING before calling this so the cost is visible in logs.
        """
        logger.warning(
            "SLEAP pose backend falling back to ORT TensorRT-EP (ONNX Runtime "
            "TensorRT ExecutionProvider). This incurs ~8 s of plan-cache "
            "initialisation per session. To avoid this cost ensure a valid "
            "native TRT engine is present beside the ONNX export, or that CUDA "
            "and TensorRT are available for on-the-fly engine build."
        )
        # The model path may still be .onnx if we got here from the missing-engine
        # branch; _resolve_export_model_path already returned it.  If we arrived
        # from a stale .trt rebuild-fail, look for a sibling .onnx.
        onnx_path = self.model_path
        if onnx_path.suffix.lower() not in {".onnx"}:
            siblings = sorted(self.model_path.parent.rglob("*.onnx"))
            if siblings:
                onnx_path = siblings[0]
        return _DirectOnnxSession(onnx_path, "tensorrt")

    @property
    def preferred_input_size(self) -> int:
        return int(max(self._input_hw or (0, 0))) if self._input_hw else 0

    def warmup(self) -> None:
        try:
            dummy = np.zeros((32, 32, 3), dtype=np.uint8)
            self.predict_batch([dummy])
        except Exception:
            logger.debug("SLEAP exported warmup skipped.", exc_info=True)

    def consume_last_profile(self) -> Dict[str, float]:
        profile = dict(self._last_profile)
        self._last_profile = {}
        return profile

    def _prepare_batch(
        self, crops: Sequence[np.ndarray]
    ) -> Tuple[np.ndarray, List[ExportTransform]]:
        prepared: List[np.ndarray] = []
        transforms: List[ExportTransform] = []
        for crop in crops:
            arr, transform = _prepare_export_crop(
                crop,
                self._input_hw,
                self._input_channels,
            )
            prepared.append(arr)
            transforms.append(transform)

        batch = np.stack(prepared, axis=0)
        if self._input_format.get("is_float", True):
            batch = batch.astype(np.float32) / 255.0
        else:
            batch = batch.astype(np.uint8, copy=False)
        if self._input_format.get("layout") == "nchw" and batch.ndim == 4:
            batch = np.transpose(batch, (0, 3, 1, 2))
        return np.ascontiguousarray(batch), transforms

    def _prepare_batch_cuda(
        self, crops_chw: Sequence[Any]
    ) -> Tuple[Any, List[ExportTransform]]:
        """GPU-native counterpart of :meth:`_prepare_batch`.

        Accepts ``C×H×W`` CUDA float32 tensors (BGR channel order, pixel values
        in ``[0, 255]``).  Returns a batched CUDA tensor ready for TensorRT
        inference and the corresponding per-crop :class:`ExportTransform` tuples
        (computed on CPU from integer shape arithmetic — negligible overhead).
        """
        import torch
        import torch.nn.functional as F

        out_h, out_w = int(self._input_hw[0]), int(self._input_hw[1])
        n_out_ch = int(self._input_channels or 3)
        is_float = bool(self._input_format.get("is_float", True))
        is_nchw = self._input_format.get("layout") == "nchw"

        transforms: List[ExportTransform] = []
        padded: List[Any] = []

        for crop_chw in crops_chw:
            C_in = int(crop_chw.shape[0])
            orig_h = int(crop_chw.shape[1])
            orig_w = int(crop_chw.shape[2])

            # ---- channel normalisation (mirrors _prepare_export_crop) ----
            if n_out_ch == 1 and C_in != 1:
                # Approximate BGR→Gray: same coefficients as cv2.COLOR_BGR2GRAY
                t = crop_chw.float()
                crop_chw = 0.114 * t[0:1] + 0.587 * t[1:2] + 0.299 * t[2:3]
            elif n_out_ch != 1:
                if C_in == 1:
                    crop_chw = crop_chw.expand(3, -1, -1)
                elif C_in >= 3:
                    # Input is BGR (from NVDec/gpu_canonical_crop); flip to RGB
                    crop_chw = crop_chw[:3].flip(0)

            # ---- letterbox: scale/pad computed on CPU ----
            scale = min(float(out_w) / float(orig_w), float(out_h) / float(orig_h))
            new_w = max(1, int(round(float(orig_w) * scale)))
            new_h = max(1, int(round(float(orig_h) * scale)))
            pad_x = int((out_w - new_w) // 2)
            pad_y = int((out_h - new_h) // 2)
            transforms.append(
                (float(scale), float(pad_x), float(pad_y), orig_w, orig_h)
            )

            # ---- GPU resize + letterbox pad ----
            out_ch = int(crop_chw.shape[0])
            t = F.interpolate(
                crop_chw.float().unsqueeze(0),
                size=(new_h, new_w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            canvas = torch.zeros(out_ch, out_h, out_w, device=t.device, dtype=t.dtype)
            canvas[:, pad_y : pad_y + new_h, pad_x : pad_x + new_w] = t
            padded.append(canvas)

        batch = torch.stack(padded, dim=0)  # (N, C, H, W)

        if is_float:
            batch = batch / 255.0

        if not is_nchw:
            batch = batch.permute(0, 2, 3, 1).contiguous()  # → (N, H, W, C)

        return batch.contiguous(), transforms

    def predict_batch(self, crops: Sequence[np.ndarray]) -> List[PoseResult]:
        self._last_profile = {}
        if not crops:
            return []

        effective_batch = max(self.batch_size, int(self._model_min_batch or 1))
        outputs: List[PoseResult] = []
        total_transport_s = 0.0
        total_inference_s = 0.0
        total_postprocess_s = 0.0

        for start in range(0, len(crops), effective_batch):
            chunk = list(crops[start : start + effective_batch])
            if not chunk:
                continue
            raw_count = len(chunk)
            if raw_count < effective_batch:
                chunk.extend([chunk[-1]] * (effective_batch - raw_count))

            prep_start = time.perf_counter()
            batch, transforms = self._prepare_batch(chunk)
            total_transport_s += time.perf_counter() - prep_start

            infer_start = time.perf_counter()
            raw_outputs = self._runner.run(batch)
            total_inference_s += time.perf_counter() - infer_start

            postprocess_start = time.perf_counter()
            parsed = _coerce_export_output(raw_outputs, self._output_names, len(chunk))
            for index in range(raw_count):
                arr = parsed[index] if index < len(parsed) else None
                if arr is None:
                    outputs.append(empty_pose_result())
                    continue
                arr = np.asarray(arr, dtype=np.float32)
                if arr.ndim != 2 or arr.shape[1] < 2:
                    outputs.append(empty_pose_result())
                    continue
                arr = _restore_export_keypoints(arr, transforms[index])
                if arr.shape[1] == 2:
                    arr = np.column_stack(
                        (arr, np.zeros((arr.shape[0],), dtype=np.float32))
                    )
                outputs.append(summarize_keypoints(arr[:, :3], self.min_valid_conf))
            total_postprocess_s += time.perf_counter() - postprocess_start

        self._last_profile = {
            "pose_transport_s": total_transport_s,
            "pose_inference_s": total_inference_s,
            "pose_postprocess_s": total_postprocess_s,
        }
        return outputs

    def predict_batch_cuda(self, crops: Sequence[Any]) -> List[PoseResult]:
        """GPU-native counterpart of :meth:`predict_batch`.

        Accepts a sequence of ``C×H×W`` CUDA float32 tensors (BGR, ``[0, 1]``
        range) produced by the NVDec→:func:`~hydra_suite.core.canonicalization.crop.gpu_canonical_crop`
        pipeline.

        Only the TensorRT backend benefits from the zero-copy GPU path.  For
        ONNX backends the crops are converted to ``H×W×C`` uint8 numpy arrays
        and forwarded to :meth:`predict_batch` transparently.
        """
        if not crops:
            return []

        if not isinstance(self._runner, _DirectTensorRTEngine):
            # ONNX has a hard numpy boundary — convert to uint8 [0,255] numpy and
            # forward to predict_batch (whose _prepare_export_crop casts to uint8
            # and then divides by 255). CRITICAL: the canonical crops are float32
            # [0, 1], so a direct uint8 cast floors every pixel to 0 (all-black
            # image → SLEAP returns zero-confidence keypoints). Scale by 255 first
            # (only when in [0,1]; pass [0,255] through) — mirrors _to_uint8_image.
            def _crop_to_u8(c):
                t = c.float()
                if float(t.max()) <= 1.0 + 1e-3:
                    t = t * 255.0
                return t.permute(1, 2, 0).clamp(0, 255).byte().cpu().numpy()

            cpu_crops = [_crop_to_u8(c) for c in crops]
            return self.predict_batch(cpu_crops)

        self._last_profile = {}
        effective_batch = max(self.batch_size, int(self._model_min_batch or 1))
        outputs: List[PoseResult] = []
        total_inference_s = 0.0
        total_postprocess_s = 0.0

        for start in range(0, len(crops), effective_batch):
            chunk = list(crops[start : start + effective_batch])
            if not chunk:
                continue
            raw_count = len(chunk)
            if raw_count < effective_batch:
                chunk.extend([chunk[-1]] * (effective_batch - raw_count))

            batch_cuda, transforms = self._prepare_batch_cuda(chunk)

            infer_start = time.perf_counter()
            raw_outputs = self._runner.run_cuda(batch_cuda)
            total_inference_s += time.perf_counter() - infer_start

            postprocess_start = time.perf_counter()
            parsed = _coerce_export_output(raw_outputs, self._output_names, len(chunk))
            for index in range(raw_count):
                arr = parsed[index] if index < len(parsed) else None
                if arr is None:
                    outputs.append(empty_pose_result())
                    continue
                arr = np.asarray(arr, dtype=np.float32)
                if arr.ndim != 2 or arr.shape[1] < 2:
                    outputs.append(empty_pose_result())
                    continue
                arr = _restore_export_keypoints(arr, transforms[index])
                if arr.shape[1] == 2:
                    arr = np.column_stack(
                        (arr, np.zeros((arr.shape[0],), dtype=np.float32))
                    )
                outputs.append(summarize_keypoints(arr[:, :3], self.min_valid_conf))
            total_postprocess_s += time.perf_counter() - postprocess_start

        self._last_profile = {
            "pose_inference_s": total_inference_s,
            "pose_postprocess_s": total_postprocess_s,
        }
        return outputs

    def close(self) -> None:
        closer = getattr(self._runner, "close", None)
        if callable(closer):
            closer()


def _build_sleap_export_kwargs(
    params: Any,
    model_dir: Path,
    export_dir: Path,
    runtime: str,
    batch_size: int,
    max_instances: int,
) -> Dict[str, Any]:
    """Build keyword arguments for a SLEAP export function based on its signature."""
    kwargs: Dict[str, Any] = {}

    # Model input path (try several parameter names)
    _MODEL_PARAM_NAMES = ["model_dir", "model_path", "trained_model_path"]
    for name in _MODEL_PARAM_NAMES:
        if name in params:
            kwargs[name] = str(model_dir)
            break

    # Output directory
    _OUTPUT_PARAM_NAMES = ["output_dir", "export_dir", "save_dir"]
    for name in _OUTPUT_PARAM_NAMES:
        if name in params:
            kwargs[name] = str(export_dir)
            break

    # Runtime / format hints
    for name in ("runtime", "model_type", "format"):
        if name in params:
            kwargs[name] = runtime

    # Numeric options
    if "batch_size" in params:
        kwargs["batch_size"] = int(max(1, batch_size))
    if "max_instances" in params:
        kwargs["max_instances"] = int(max(1, max_instances))

    return kwargs


def _try_sleap_export_function(
    fn: Any,
    model_dir: Path,
    export_dir: Path,
    runtime: str,
    batch_size: int,
    max_instances: int,
) -> Tuple[bool, str]:
    """Attempt to call a single SLEAP export function and check the result."""
    import inspect

    sig = inspect.signature(fn)
    kwargs = _build_sleap_export_kwargs(
        sig.parameters, model_dir, export_dir, runtime, batch_size, max_instances
    )
    if kwargs:
        fn(**kwargs)
    else:
        fn(str(model_dir), str(export_dir))
    if looks_like_sleap_export_path(str(export_dir), runtime):
        return True, ""
    return False, "Export completed but output not found."


def _attempt_sleap_python_export(
    model_dir: Path,
    export_dir: Path,
    runtime_flavor: str,
    batch_size: int,
    max_instances: int,
) -> Tuple[bool, str]:
    try:
        import importlib
    except Exception as exc:
        return False, str(exc)

    runtime = str(runtime_flavor).strip().lower()
    module_candidates = ["sleap_nn.export.exporters", "sleap_nn.export"]
    func_candidates = ["export_model", "export"]
    if runtime == "onnx":
        func_candidates.insert(0, "export_to_onnx")
    elif runtime == "tensorrt":
        func_candidates.insert(0, "export_to_tensorrt")

    last_err = "No compatible SLEAP Python export API found."
    for module_name in module_candidates:
        try:
            mod = importlib.import_module(module_name)
        except Exception as exc:
            last_err = str(exc)
            continue
        for fn_name in func_candidates:
            fn = getattr(mod, fn_name, None)
            if not callable(fn):
                continue
            try:
                ok, err = _try_sleap_export_function(
                    fn, model_dir, export_dir, runtime, batch_size, max_instances
                )
                if ok:
                    return True, ""
            except Exception as exc:
                last_err = str(exc)
                continue
    return False, last_err


def _attempt_sleap_cli_export(
    model_dir: Path,
    export_dir: Path,
    runtime_flavor: str,
    sleap_env: str,
    input_hw: Optional[Tuple[int, int]] = None,
    batch_size: int = 1,
) -> Tuple[bool, str]:
    runtime = str(runtime_flavor).strip().lower()
    sleap_env = str(sleap_env or "").strip()
    batch_size = int(max(1, int(batch_size)))
    runtime_tokens = [runtime]
    if runtime == "tensorrt":
        runtime_tokens.append("trt")

    size_candidates: List[Tuple[int, int]] = []
    if (
        isinstance(input_hw, (tuple, list))
        and len(input_hw) >= 2
        and int(input_hw[0]) > 0
        and int(input_hw[1]) > 0
    ):
        size_candidates.append((int(input_hw[0]), int(input_hw[1])))
    size_candidates.extend([(224, 224), (256, 256)])
    deduped_sizes: List[Tuple[int, int]] = []
    for s in size_candidates:
        if s not in deduped_sizes:
            deduped_sizes.append(s)
    size_candidates = deduped_sizes

    command_variants: List[List[str]] = []
    for token in runtime_tokens:
        profile_variants = [
            ["--batch-size", str(batch_size)],
            ["--batch", str(batch_size)],
            [],
        ]
        base_variants = [
            [
                "sleap-nn",
                "export",
                str(model_dir),
                "--output",
                str(export_dir),
                "--format",
                token,
            ],
            [
                "sleap-nn",
                "export",
                "--output",
                str(export_dir),
                "--format",
                token,
                str(model_dir),
            ],
            [
                "python",
                "-m",
                "sleap_nn.export.cli",
                str(model_dir),
                "--output",
                str(export_dir),
                "--format",
                token,
            ],
            [
                "python",
                "-m",
                "sleap_nn.export.cli",
                "--output",
                str(export_dir),
                "--format",
                token,
                str(model_dir),
            ],
        ]
        for h, w in size_candidates:
            for base in base_variants:
                for prof in profile_variants:
                    command_variants.append(
                        [
                            *base,
                            "--input-height",
                            str(int(h)),
                            "--input-width",
                            str(int(w)),
                            *prof,
                        ]
                    )
        for base in base_variants:
            for prof in profile_variants:
                command_variants.append([*base, *prof])

    if shutil.which("conda") and sleap_env:
        conda_wrapped = []
        for cmd in command_variants:
            conda_wrapped.append(["conda", "run", "-n", sleap_env, *cmd])
        command_variants = conda_wrapped

    last_err = "No SLEAP export CLI command succeeded."
    for cmd in command_variants:
        ok, err = run_cli_command(cmd)
        if ok and looks_like_sleap_export_path(str(export_dir), runtime):
            return True, ""
        if err:
            last_err = err
    return False, last_err


def auto_export_sleap_model(config: PoseRuntimeConfig, runtime_flavor: str) -> str:
    runtime = str(runtime_flavor or "native").strip().lower()
    if runtime not in {"onnx", "tensorrt"}:
        raise RuntimeError(f"Unsupported SLEAP auto-export runtime: {runtime}")

    model_path = Path(str(config.model_path or "")).expanduser().resolve()
    if looks_like_sleap_export_path(str(model_path), runtime):
        return str(model_path)
    if not model_path.exists() or not model_path.is_dir():
        raise RuntimeError(
            f"SLEAP model path does not exist or is not a directory: {model_path}"
        )

    input_hw = (
        tuple(int(v) for v in config.sleap_export_input_hw)
        if config.sleap_export_input_hw is not None
        else None
    )
    sig_blob = (
        f"{path_fingerprint_token(str(model_path))}|runtime={runtime}|"
        f"batch={int(config.sleap_batch)}|max_instances={int(config.sleap_max_instances)}|"
        f"input_hw={input_hw}"
    ).encode("utf-8")
    sig = hashlib.sha1(sig_blob).hexdigest()[:16]
    export_dir = model_path.parent / f"{model_path.name}.{runtime}"
    if looks_like_sleap_export_path(str(export_dir), runtime) and artifact_meta_matches(
        export_dir, sig
    ):
        return str(export_dir.resolve())
    if export_dir.exists():
        shutil.rmtree(export_dir, ignore_errors=True)
    export_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Exporting SLEAP model for %s runtime: %s -> %s",
        runtime,
        model_path,
        export_dir,
    )
    if input_hw is not None:
        logger.info(
            "SLEAP export input size hint: %dx%d",
            int(input_hw[0]),
            int(input_hw[1]),
        )
    sleap_env = str(config.sleap_env or "").strip()
    ok, err = _attempt_sleap_cli_export(
        model_dir=model_path,
        export_dir=export_dir,
        runtime_flavor=runtime,
        sleap_env=sleap_env,
        input_hw=input_hw,
        batch_size=int(max(1, config.sleap_batch)),
    )
    if not ok and not sleap_env:
        ok, err = _attempt_sleap_python_export(
            model_dir=model_path,
            export_dir=export_dir,
            runtime_flavor=runtime,
            batch_size=int(max(1, config.sleap_batch)),
            max_instances=int(max(1, config.sleap_max_instances)),
        )
    if not ok or not looks_like_sleap_export_path(str(export_dir), runtime):
        raise RuntimeError(f"SLEAP auto-export failed for runtime '{runtime}'. {err}")
    write_artifact_meta(export_dir, sig)
    return str(export_dir.resolve())


class SleapServiceBackend:
    """SLEAP runtime adapter via PoseInferenceService HTTP service."""

    def __init__(
        self,
        model_dir: str,
        out_root: str,
        keypoint_names: Sequence[str],
        min_valid_conf: float = 0.2,
        sleap_env: str = "sleap",
        sleap_device: str = "auto",
        sleap_batch: int = 4,
        sleap_max_instances: int = 1,
        skeleton_edges: Optional[Sequence[Sequence[int]]] = None,
        runtime_flavor: str = "native",
        exported_model_path: str = "",
        export_input_hw: Optional[Tuple[int, int]] = None,
    ):
        from hydra_suite.integrations.sleap.service import PoseInferenceService

        self.model_dir = Path(model_dir).expanduser().resolve()
        self.out_root = Path(out_root).expanduser().resolve()
        self.output_keypoint_names = [str(v) for v in keypoint_names]
        self.min_valid_conf = float(min_valid_conf)
        self.sleap_env = str(sleap_env or "sleap").strip() or "sleap"
        self.sleap_device = str(sleap_device or "auto")
        self.sleap_batch = max(1, int(sleap_batch))
        self.sleap_max_instances = max(1, int(sleap_max_instances))
        self.runtime_flavor = str(runtime_flavor or "native").strip().lower()
        self.exported_model_path = str(exported_model_path or "").strip()
        self.export_input_hw = (
            (int(export_input_hw[0]), int(export_input_hw[1]))
            if isinstance(export_input_hw, (tuple, list))
            and len(export_input_hw) >= 2
            and int(export_input_hw[0]) > 0
            and int(export_input_hw[1]) > 0
            else None
        )
        self.skeleton_edges = (
            [tuple(int(v) for v in e[:2]) for e in (skeleton_edges or [])]
            if skeleton_edges
            else []
        )
        self._infer = PoseInferenceService(
            self.out_root, self.output_keypoint_names, self.skeleton_edges
        )
        self._service_started_here = False
        self._native_in_memory_supported: Optional[bool] = None
        self._tmp_dir = tempfile.TemporaryDirectory(prefix="hydra_posekit_")
        self._tmp_root = Path(self._tmp_dir.name)
        self._last_profile: Dict[str, float] = {}

    @property
    def preferred_input_size(self) -> int:
        """SLEAP models have no fixed input size; return 0 (no preference)."""
        return 0

    def warmup(self) -> None:
        try:
            was_running = self._infer.sleap_service_running()
            ok, err, _ = self._infer.start_sleap_service(self.sleap_env, self.out_root)
            if not ok:
                raise RuntimeError(err or "Failed to start SLEAP service.")
            self._service_started_here = (
                not was_running
            ) and self._infer.sleap_service_running()
            if self.runtime_flavor == "native":
                self._native_in_memory_supported = bool(
                    self._infer.sleap_native_array_video_supported()
                )
        except Exception as exc:
            logger.warning("SLEAP service warmup failed: %s", exc)

    def _native_in_memory_enabled(self) -> bool:
        if self.runtime_flavor in {"onnx", "tensorrt"}:
            return True
        if self._native_in_memory_supported is None:
            try:
                self._native_in_memory_supported = bool(
                    self._infer.sleap_native_array_video_supported()
                )
            except Exception:
                self._native_in_memory_supported = False
        return bool(self._native_in_memory_supported)

    @staticmethod
    def _to_uint8_image(crop: np.ndarray) -> np.ndarray:
        """Coerce a crop to uint8 [0, 255] for SLEAP.

        The inference pipeline's canonical crops are float32 in [0, 1] (the crop
        builder does ``.float() / 255.0`` so the classifier/CNN backends get
        normalized input). SLEAP — both the shared-memory transport and the
        temp-file PNG path — expects 8-bit images; casting a [0, 1] float crop
        straight to uint8 floors every pixel to 0, producing a blank image with
        no detectable instances (all-zero keypoints). Scale [0, 1] floats back to
        [0, 255] before the uint8 cast; pass uint8 / [0, 255] inputs through.
        """
        a = np.asarray(crop)
        if a.dtype == np.uint8:
            return a
        a = a.astype(np.float32, copy=False)
        finite = a[np.isfinite(a)]
        if finite.size and float(finite.max()) <= 1.0 + 1e-3:
            a = a * 255.0
        return np.clip(np.nan_to_num(a), 0.0, 255.0).astype(np.uint8)

    def predict_batch(self, crops: Sequence[np.ndarray]) -> List[PoseResult]:
        self._last_profile = {}
        if not crops:
            return []
        # Normalize every crop to uint8 [0, 255] up front so both transports
        # (shared-memory and temp-file) feed SLEAP 8-bit images regardless of the
        # incoming dtype/range (see _to_uint8_image).
        crops = [self._to_uint8_image(c) for c in crops]
        # Prefer zero-copy shared-memory transport: raw uint8 arrays straight to
        # the service's warm in-process predictor — no PNG encode, no disk write/
        # read, no decode. (The old sio array-video gate is obsolete now that the
        # service builds inference tensors from the arrays directly.) Fall back to
        # temp-file PNGs only if shared memory genuinely fails.
        try:
            return self._predict_batch_via_shared_memory(crops)
        except Exception:
            # Shared-memory transport failed; fall back to temp-file PNGs. Logged
            # once at WARNING since the fallback is markedly slower (disk I/O).
            if not getattr(SleapServiceBackend, "_shm_fallback_logged", False):
                SleapServiceBackend._shm_fallback_logged = True
                logger.warning(
                    "SLEAP shared-memory transport failed; falling back to "
                    "temp-file transport (slower).",
                    exc_info=True,
                )
            return self._predict_batch_via_temp_files(crops)

    def consume_last_profile(self) -> Dict[str, float]:
        """Return and clear the most recent SLEAP batch timing profile."""
        profile = dict(self._last_profile)
        self._last_profile = {}
        return profile

    def _consume_service_metrics(self) -> Dict[str, float]:
        getter = getattr(self._infer, "get_last_sleap_service_metrics", None)
        if not callable(getter):
            return {}
        try:
            return dict(getter() or {})
        except Exception:
            return {}

    def _finalize_profile(
        self,
        transport_s: float,
        service_call_s: float,
        service_metrics: Dict[str, float],
        local_postprocess_s: float,
    ) -> None:
        decode_s = float(service_metrics.get("service_decode_s", 0.0) or 0.0)
        materialize_s = float(service_metrics.get("service_materialize_s", 0.0) or 0.0)
        extract_s = float(service_metrics.get("service_extract_s", 0.0) or 0.0)
        inference_s = float(service_metrics.get("service_inference_s", 0.0) or 0.0)
        if inference_s <= 0.0:
            inference_s = max(
                0.0, float(service_call_s) - decode_s - materialize_s - extract_s
            )
        self._last_profile = {
            "pose_transport_s": max(0.0, float(transport_s) + decode_s + materialize_s),
            "pose_inference_s": max(0.0, inference_s),
            "pose_postprocess_s": max(0.0, float(local_postprocess_s) + extract_s),
        }

    def _predict_batch_via_shared_memory(
        self, crops: Sequence[np.ndarray]
    ) -> List[PoseResult]:
        if not crops:
            return []

        def _share_crop(i: int, crop: np.ndarray):
            image_id = f"inmem_crop_{i:06d}"
            try:
                arr = np.ascontiguousarray(np.asarray(crop, dtype=np.uint8))
            except Exception:
                return image_id, None, None
            if arr.ndim not in (2, 3) or any(int(v) <= 0 for v in arr.shape):
                return image_id, None, None
            shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
            np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)[:] = arr
            payload = {
                "id": image_id,
                "shape": [int(v) for v in arr.shape],
                "dtype": "uint8",
                "shm_name": shm.name,
                "nbytes": int(arr.nbytes),
            }
            return image_id, payload, shm

        transport_start = time.perf_counter()
        encoded = [_share_crop(i, crop) for i, crop in enumerate(crops)]
        transport_s = time.perf_counter() - transport_start

        image_payloads = [payload for _, payload, _ in encoded if payload is not None]
        valid_ids = [
            image_id for image_id, payload, _ in encoded if payload is not None
        ]
        preds: Dict[str, List[Any]] = {}
        try:
            if valid_ids:
                service_call_start = time.perf_counter()
                pred_map, err = self._infer.predict(
                    model_path=self.model_dir,
                    image_paths=valid_ids,
                    device="auto",
                    imgsz=640,
                    conf=1e-4,
                    batch=min(self.sleap_batch, max(1, len(valid_ids))),
                    backend="sleap",
                    sleap_env=self.sleap_env,
                    sleap_device=self.sleap_device,
                    sleap_batch=min(self.sleap_batch, max(1, len(valid_ids))),
                    sleap_max_instances=self.sleap_max_instances,
                    sleap_runtime_flavor=self.runtime_flavor,
                    sleap_exported_model_path=self.exported_model_path,
                    sleap_export_input_hw=self.export_input_hw,
                    image_payloads=image_payloads,
                    cache_predictions=False,
                )
                service_call_s = time.perf_counter() - service_call_start
                if pred_map is None:
                    raise RuntimeError(err or "SLEAP inference failed.")
                preds = pred_map
            else:
                service_call_s = 0.0
        finally:
            for _, _, shm in encoded:
                if shm is None:
                    continue
                try:
                    shm.close()
                except Exception:
                    pass
                try:
                    shm.unlink()
                except FileNotFoundError:
                    pass
                except Exception:
                    logger.debug(
                        "Failed to unlink temporary SLEAP shared-memory payload.",
                        exc_info=True,
                    )

        outputs: List[PoseResult] = []
        postprocess_start = time.perf_counter()
        for image_id, payload, _ in encoded:
            if payload is None:
                outputs.append(empty_pose_result())
                continue
            pred = preds.get(image_id)
            if not pred:
                outputs.append(empty_pose_result())
                continue
            arr = np.asarray(pred, dtype=np.float32)
            if arr.ndim != 2 or arr.shape[1] != 3:
                outputs.append(empty_pose_result())
                continue
            outputs.append(summarize_keypoints(arr, self.min_valid_conf))

        self._finalize_profile(
            transport_s,
            service_call_s,
            self._consume_service_metrics(),
            time.perf_counter() - postprocess_start,
        )

        return outputs

    def _predict_batch_via_temp_files(
        self, crops: Sequence[np.ndarray]
    ) -> List[PoseResult]:
        from concurrent.futures import ThreadPoolExecutor

        def _write_crop(args):
            i, crop = args
            p = self._tmp_root / f"crop_{i:06d}.png"
            ok = cv2.imwrite(str(p), crop)
            return p if ok else Path("__invalid__")

        transport_start = time.perf_counter()
        with ThreadPoolExecutor(
            max_workers=min(4, len(crops)),
            thread_name_prefix="sleap-crop-write",
        ) as pool:
            paths: List[Path] = list(pool.map(_write_crop, enumerate(crops)))
        transport_s = time.perf_counter() - transport_start

        valid_paths = [p for p in paths if p.exists()]
        preds: Dict[str, List[Any]] = {}
        if valid_paths:
            service_call_start = time.perf_counter()
            pred_map, err = self._infer.predict(
                model_path=self.model_dir,
                image_paths=valid_paths,
                device="auto",
                imgsz=640,
                conf=1e-4,
                batch=min(self.sleap_batch, max(1, len(valid_paths))),
                backend="sleap",
                sleap_env=self.sleap_env,
                sleap_device=self.sleap_device,
                sleap_batch=min(self.sleap_batch, max(1, len(valid_paths))),
                sleap_max_instances=self.sleap_max_instances,
                sleap_runtime_flavor=self.runtime_flavor,
                sleap_exported_model_path=self.exported_model_path,
                sleap_export_input_hw=self.export_input_hw,
                cache_predictions=False,
            )
            service_call_s = time.perf_counter() - service_call_start
            if pred_map is None:
                raise RuntimeError(err or "SLEAP inference failed.")
            preds = pred_map
        else:
            service_call_s = 0.0

        outputs: List[PoseResult] = []
        postprocess_start = time.perf_counter()
        for p in paths:
            if not p.exists():
                outputs.append(empty_pose_result())
                continue
            pred = preds.get(str(p))
            if pred is None:
                pred = preds.get(str(p.resolve()))
            if not pred:
                outputs.append(empty_pose_result())
                continue
            arr = np.asarray(pred, dtype=np.float32)
            if arr.ndim != 2 or arr.shape[1] != 3:
                outputs.append(empty_pose_result())
                continue
            outputs.append(summarize_keypoints(arr, self.min_valid_conf))

        self._finalize_profile(
            transport_s,
            service_call_s,
            self._consume_service_metrics(),
            time.perf_counter() - postprocess_start,
        )

        return outputs

    def close(self) -> None:
        if self._service_started_here:
            try:
                self._infer.shutdown_sleap_service()
            except Exception:
                logger.debug(
                    "Failed to stop SLEAP service from backend close.", exc_info=True
                )
            self._service_started_here = False
        try:
            self._tmp_dir.cleanup()
        except Exception:
            pass
