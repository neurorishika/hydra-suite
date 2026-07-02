"""Direct ONNX/TensorRT execution helpers for YOLO OBB runtime artifacts.

These helpers bypass ``ultralytics.YOLO(...).predict(...)`` for deployed
runtime artifacts while still reusing Ultralytics preprocessing and OBB NMS
utilities so the detector keeps the same output contract.

Only CUDA-backed runtime artifacts are supported here. Other runtimes keep the
existing Ultralytics wrapper path.
"""

from __future__ import annotations

import ast
import json
import struct
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    import torch

import numpy as np


def _parse_meta_bool(value, default: bool = False) -> bool:
    """Parse a metadata boolean that may be stored as a Python string.

    Ultralytics serialises ONNX ``custom_metadata_map`` values with ``str()``,
    so ``False`` becomes the string ``"False"``.  ``bool("False")`` evaluates
    to ``True`` (non-empty string), which would incorrectly set ``_end2end=True``
    for raw-head CBC artifacts exported with ``end2end=False``.
    """
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() not in ("false", "0", "no", "none", "")
    return bool(value)


class _BaseDirectOBBExecutor:
    def __init__(
        self,
        artifact_path: str,
        imgsz: int,
        class_names: dict[int, str] | None = None,
        class_count: int | None = None,
    ) -> None:
        self.artifact_path = str(Path(artifact_path).expanduser().resolve())
        self.imgsz = max(32, int(imgsz or 640))
        self.names = {
            int(key): str(value) for key, value in (class_names or {}).items()
        }
        self.nc = max(1, int(class_count or len(self.names) or 1))
        self.stride = 32
        # Cache the LetterBox transform so it is not re-instantiated on every
        # inference call.
        from ultralytics.data.augment import LetterBox

        self._letterbox = LetterBox(
            (self.imgsz, self.imgsz), auto=False, stride=self.stride
        )
        # Pre-allocate a page-locked (pinned) host buffer for the single-frame
        # fast path.  Pinned memory enables async DMA to the CUDA device without
        # going through the CUDA caching allocator on every call, which eliminates
        # the occasional allocator-induced latency spikes.
        import torch

        self._pinned_input = torch.empty(
            (1, 3, self.imgsz, self.imgsz), dtype=torch.uint8, pin_memory=True
        )
        # Pre-allocated CUDA float32 tensor for the fast path — reused every
        # frame so no per-call GPU allocation is needed.
        self._gpu_input = torch.empty(
            (1, 3, self.imgsz, self.imgsz), dtype=torch.float32, device="cuda:0"
        )

    def _preprocess(self, frames: Sequence[np.ndarray]):
        import torch

        if not frames:
            raise ValueError("direct OBB executor received no frames")

        # Keeping uint8 until the GPU transfer reduces the CPU→GPU copy by 4×
        # (3 MB vs 12 MB for 1024-px inputs).
        #
        # Combined BGR→RGB + HWC→CHW in one ascontiguousarray call matches
        # Ultralytics' BasePredictor.preprocess pattern:
        #   transpose(2,0,1)  — non-contiguous HWC→CHW view
        #   [::-1]            — non-contiguous BGR→RGB reverse view
        #   ascontiguousarray — single contiguous CHW RGB uint8 copy
        # This avoids a separate cv2.cvtColor allocation (one copy instead of two).

        if len(frames) == 1:
            # Fast path (inference batch size 1, the common production case):
            # Write into the pre-allocated pinned host buffer and transfer to the
            # pre-allocated CUDA float tensor with a non-blocking async DMA copy.
            # This avoids per-call CUDA memory allocations and the associated
            # allocator-induced latency spikes.
            lb_frame = self._letterbox(image=frames[0])
            if lb_frame.ndim != 3 or lb_frame.shape[2] != 3:
                raise ValueError("direct OBB executor expects HxWx3 BGR frames")
            # np.copyto into pinned memory — fast CPU memcpy, zero allocation.
            np.copyto(self._pinned_input.numpy()[0], lb_frame.transpose(2, 0, 1)[::-1])
            # Non-blocking async pinned→GPU copy with on-the-fly uint8→float32
            # conversion.  These ops are queued on PyTorch's default CUDA stream.
            # The caller is responsible for ensuring they complete before inference:
            #   ONNX executor:         _run_inference calls _preprocess_event.synchronize()
            #   PyTorch-CUDA executor: _run_inference runs on the same default stream
            self._gpu_input.copy_(self._pinned_input, non_blocking=True)
            self._gpu_input.mul_(1.0 / 255.0)
            return self._gpu_input

        # Multi-frame path: build a batched uint8 array and transfer in one shot.
        batch = []
        for frame in frames:
            lb_frame = self._letterbox(image=frame)
            if lb_frame.ndim != 3 or lb_frame.shape[2] != 3:
                raise ValueError("direct OBB executor expects HxWx3 BGR frames")
            batch.append(np.ascontiguousarray(lb_frame.transpose(2, 0, 1)[::-1]))
        batch_uint8 = np.stack(batch, axis=0)
        return (
            torch.from_numpy(batch_uint8).to(device="cuda:0").float().mul_(1.0 / 255.0)
        )

    def _preprocess_cuda(self, cuda_rgb_hwc: "torch.Tensor") -> "torch.Tensor":
        """GPU-only preprocessing for frames decoded directly to CUDA memory.

        Accepts a CUDA ``(H, W, 3)`` uint8 RGB tensor (as output by
        PyNvVideoCodec with ``OutputColorType.RGB``) and populates
        ``self._gpu_input`` in-place with the letterboxed, normalised
        ``1×C×H×W`` float32 tensor ready for inference.  No CPU↔GPU copy
        is performed; the entire pipeline runs on the device.

        Returns ``self._gpu_input``.
        """
        import torch
        import torch.nn.functional as F

        H, W = int(cuda_rgb_hwc.shape[0]), int(cuda_rgb_hwc.shape[1])
        r = min(self.imgsz / H, self.imgsz / W)
        new_h = int(H * r)
        new_w = int(W * r)

        # permute HWC→CHW, add batch dim, cast uint8→float32 in one kernel call.
        t = cuda_rgb_hwc.permute(2, 0, 1).unsqueeze(0).to(dtype=torch.float32)

        # Bilinear resize, skipped when already the target size.
        if new_h != H or new_w != W:
            t = F.interpolate(
                t, size=(new_h, new_w), mode="bilinear", align_corners=False
            )

        # Symmetric letterbox padding — YOLO uses 114/255 grey.
        pad_top = (self.imgsz - new_h) // 2
        pad_left = (self.imgsz - new_w) // 2
        pad_bot = self.imgsz - new_h - pad_top
        pad_right = self.imgsz - new_w - pad_left
        if pad_top or pad_bot or pad_left or pad_right:
            t = F.pad(t, (pad_left, pad_right, pad_top, pad_bot), value=114.0)

        # Normalise into the pre-allocated GPU buffer (avoids a new allocation).
        self._gpu_input.copy_(t.mul_(1.0 / 255.0), non_blocking=True)
        return self._gpu_input

    def _preprocess_cuda_batch(self, cuda_frames: list) -> "torch.Tensor":
        """GPU-only preprocessing for a list of CUDA RGB HWC uint8 tensors.

        Each frame must already be cloned from the decoder buffer before being
        passed here — NVDec frames share decoder memory that is invalidated by
        the next ``get_batch_frames()`` call.

        Returns a ``[N, 3, imgsz, imgsz]`` float32 CUDA tensor, letterboxed
        and normalised to ``[0, 1]``.
        """
        import torch
        import torch.nn.functional as F

        processed = []
        for cuda_rgb_hwc in cuda_frames:
            H, W = int(cuda_rgb_hwc.shape[0]), int(cuda_rgb_hwc.shape[1])
            r = min(self.imgsz / H, self.imgsz / W)
            new_h = int(H * r)
            new_w = int(W * r)
            # HWC uint8 → NCHW float32 (0–255 scale, normalised after padding)
            t = cuda_rgb_hwc.permute(2, 0, 1).unsqueeze(0).to(dtype=torch.float32)
            if new_h != H or new_w != W:
                t = F.interpolate(
                    t, size=(new_h, new_w), mode="bilinear", align_corners=False
                )
            pad_top = (self.imgsz - new_h) // 2
            pad_left = (self.imgsz - new_w) // 2
            pad_bot = self.imgsz - new_h - pad_top
            pad_right = self.imgsz - new_w - pad_left
            if pad_top or pad_bot or pad_left or pad_right:
                t = F.pad(t, (pad_left, pad_right, pad_top, pad_bot), value=114.0)
            processed.append(t.squeeze(0).mul_(1.0 / 255.0))  # [3, H, W]
        return torch.stack(processed, dim=0)  # [N, 3, H, W]

    def predict_from_cuda_frame(
        self,
        cuda_rgb_hwc: "torch.Tensor",
        orig_hw: "tuple[int, int]",
        *,
        conf_thres: float,
        classes,
        max_det: int,
    ):
        """Run inference on a GPU-decoded frame, eliminating all CPU↔GPU copies.

        Parameters
        ----------
        cuda_rgb_hwc:
            A CUDA ``(H, W, 3)`` uint8 RGB tensor as output by PyNvVideoCodec
            with ``OutputColorType.RGB``.
        orig_hw:
            ``(height, width)`` of the original decoded frame, used for
            output coordinate scaling in the returned Results objects.
        """
        img_tensor = self._preprocess_cuda(cuda_rgb_hwc)
        raw_preds = self._run_inference(img_tensor)

        orig_h, orig_w = int(orig_hw[0]), int(orig_hw[1])
        # _postprocess reads orig_img.shape for scaling; pixel values are unused.
        # Cache the placeholder to avoid a large numpy allocation every call.
        if getattr(self, "_dummy_orig", None) is None or self._dummy_orig.shape[:2] != (
            orig_h,
            orig_w,
        ):
            self._dummy_orig = np.empty((orig_h, orig_w, 3), dtype=np.uint8)
        return self._postprocess(
            raw_preds,
            img_tensor,
            [self._dummy_orig],
            conf_thres=conf_thres,
            classes=classes,
            max_det=max_det,
        )

    def _postprocess(
        self,
        raw_preds,
        img_tensor,
        orig_frames: Sequence[np.ndarray],
        conf_thres: float,
        classes,
        max_det: int,
    ):
        import torch
        from ultralytics.engine.results import Results
        from ultralytics.utils import nms, ops

        preds = raw_preds[0] if isinstance(raw_preds, (tuple, list)) else raw_preds
        if not isinstance(preds, torch.Tensor):
            preds = torch.as_tensor(preds, device=img_tensor.device)

        is_end2end = getattr(self, "_end2end", False)
        # For end-to-end backends (e.g. PyTorch CUDA) the model already applied
        # NMS internally, so we skip IOU filtering here.  For raw-CBC backends
        # (TRT, ONNX with end2end=False) the output contains many overlapping
        # anchors for each object.  Applying NMS before the max_det cap ensures
        # that the top-N slots are filled by *distinct* objects rather than by
        # multiple high-confidence duplicate boxes for the same object.
        iou_thres = 1.0 if is_end2end else 0.5
        filtered = nms.non_max_suppression(
            preds,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            classes=classes,
            max_det=max_det,
            nc=self.nc,
            rotated=True,
            end2end=is_end2end,
        )

        results = []
        for pred, orig_img in zip(filtered, orig_frames):
            if pred is None or len(pred) == 0:
                empty = torch.zeros((0, 7), device=img_tensor.device)
                results.append(Results(orig_img, path="", names=self.names, obb=empty))
                continue

            rboxes = torch.cat([pred[:, :4], pred[:, -1:]], dim=-1)
            rboxes[:, :4] = ops.scale_boxes(
                img_tensor.shape[2:], rboxes[:, :4], orig_img.shape, xywh=True
            )
            obb = torch.cat([rboxes, pred[:, 4:6]], dim=-1)
            results.append(Results(orig_img, path="", names=self.names, obb=obb))
        return results

    def predict(
        self,
        frames: Sequence[np.ndarray],
        *,
        conf_thres: float,
        classes,
        max_det: int,
    ):
        """Run inference on a list of frames.

        Accepts either numpy BGR frames (standard path) or CUDA RGB uint8
        tensors decoded directly to device memory (NVDec path).  When CUDA
        tensors are supplied, each must already be cloned from the decoder
        buffer before this call.

        Chunks input to the model's fixed batch size automatically so that
        ONNX/TRT models exported with static shapes are never fed an
        unexpected batch dimension.
        """
        import torch

        cuda_input = bool(
            frames and isinstance(frames[0], torch.Tensor) and frames[0].is_cuda
        )

        # ONNX/TRT models exported with dynamic=False have a fixed input batch
        # dimension.  Chunk and pad so the executor is never called with a
        # mismatched batch size — this covers BOTH over-sized batches (chunk)
        # and under-sized ones like the last partial batch (pad).
        #
        # _static_batch=True on ONNX executors when the model has a *numeric*
        # (fixed) batch dimension.  This is set for ONNX artifacts exported with
        # dynamic=False (batch=1 for realtime OBB).  Dynamic-axis ONNX models
        # report _static_batch=False and the condition below is skipped so all N
        # frames are passed in a single run_with_iobinding() call.
        model_bs = getattr(self, "_model_batch_size", 1)
        static_batch = getattr(self, "_static_batch", False)
        if (model_bs > 1 or static_batch) and len(frames) != model_bs:
            all_results: list = []
            for i in range(0, len(frames), model_bs):
                chunk = list(frames[i : i + model_bs])
                actual = len(chunk)
                if actual < model_bs:
                    chunk = chunk + [chunk[0]] * (model_bs - actual)
                all_results.extend(
                    self._predict_chunk(
                        chunk,
                        cuda_input=cuda_input,
                        conf_thres=conf_thres,
                        classes=classes,
                        max_det=max_det,
                    )[:actual]
                )
            return all_results

        return self._predict_chunk(
            list(frames),
            cuda_input=cuda_input,
            conf_thres=conf_thres,
            classes=classes,
            max_det=max_det,
        )

    def _predict_chunk(
        self,
        frames: list,
        *,
        cuda_input: bool,
        conf_thres: float,
        classes,
        max_det: int,
    ):
        """Run inference on a single chunk already sized to the model batch."""
        if cuda_input:
            img_tensor = self._preprocess_cuda_batch(frames)
            orig_h = int(frames[0].shape[0])
            orig_w = int(frames[0].shape[1])
            # _postprocess only reads orig_img.shape; pixel values are unused.
            orig_frames: list = [np.empty((orig_h, orig_w, 3), dtype=np.uint8)] * len(
                frames
            )
        else:
            img_tensor = self._preprocess(frames)
            orig_frames = list(frames)
        raw_preds = self._run_inference(img_tensor)
        return self._postprocess(
            raw_preds,
            img_tensor,
            orig_frames,
            conf_thres=conf_thres,
            classes=classes,
            max_det=max_det,
        )

    def _run_inference(self, img_tensor):
        raise NotImplementedError


class DirectONNXOBBExecutor(_BaseDirectOBBExecutor):
    def __init__(
        self,
        artifact_path: str,
        imgsz: int,
        class_names: dict[int, str] | None = None,
        class_count: int | None = None,
    ) -> None:
        super().__init__(artifact_path, imgsz, class_names, class_count)

        import onnxruntime as ort

        providers = [
            ("CUDAExecutionProvider", {"device_id": 0}),
            "CPUExecutionProvider",
        ]
        self.session = ort.InferenceSession(self.artifact_path, providers=providers)
        if "CUDAExecutionProvider" not in self.session.get_providers():
            raise RuntimeError(
                "onnxruntime CUDAExecutionProvider is unavailable for direct OBB execution"
            )

        inp = self.session.get_inputs()[0]
        out = self.session.get_outputs()[0]
        self._input_name = inp.name
        self._output_name = out.name
        self._fp16 = inp.type == "tensor(float16)"
        # Cache the model's fixed input batch size for chunking in predict().
        # Dynamic-axis models return a symbolic name (str) — treat as 1.
        self._model_batch_size = (
            int(inp.shape[0])
            if isinstance(inp.shape[0], int) and inp.shape[0] > 0
            else 1
        )
        # True when the model has a *numeric* (static) batch dimension.
        # Dynamic-axis models have a symbolic string here (e.g. "batch_size").
        # Used by predict() to decide whether to chunk per-frame for batch-1
        # realtime ONNX artifacts (which fail with ORT shape errors for N>1).
        self._static_batch: bool = isinstance(inp.shape[0], int) and inp.shape[0] > 0
        # Determine output numpy dtype and cache static output shape so _run_inference
        # can pre-allocate the output directly on CUDA, avoiding a GPU→CPU→GPU roundtrip.
        import numpy as np

        self._out_np_dtype = np.float16 if out.type == "tensor(float16)" else np.float32
        self._out_torch_dtype = (
            "float16" if out.type == "tensor(float16)" else "float32"
        )
        self._static_out_shape: list[int] = [
            int(d) if (isinstance(d, int) and d > 0) else 1 for d in out.shape
        ]

        meta = self.session.get_modelmeta().custom_metadata_map or {}
        names = meta.get("names")
        if names and not self.names:
            try:
                parsed = ast.literal_eval(names)
            except (ValueError, SyntaxError):
                parsed = {}
            self.names = {
                int(key): str(value) for key, value in dict(parsed or {}).items()
            }
            self.nc = max(1, len(self.names) or self.nc)
        # Read the end2end flag from ONNX metadata so _postprocess uses the
        # correct NMS mode (iou_thres=1.0 for BNC end2end vs 0.5 for raw CBC
        # head).  _yolo_runtime_export_profile always forces end2end=False for
        # hydra-exported artifacts; this defensive read handles user-supplied
        # ONNX files that may have been exported with end2end=True.
        # NOTE: Ultralytics stores ONNX custom_metadata_map values as strings via
        # str(), so bool(False) becomes the string "False".  Use _parse_meta_bool
        # instead of bool() to avoid bool("False") == True.
        self._end2end = _parse_meta_bool(meta.get("end2end", False))

        # Pre-allocate the output tensor and create a persistent IO binding with
        # both input and output pre-bound from fixed CUDA buffers so that
        # _run_inference needs zero rebind per frame (for fp32 models).
        import torch

        out_torch_dtype = getattr(torch, self._out_torch_dtype)
        self._output_tensor = torch.empty(
            self._static_out_shape, dtype=out_torch_dtype, device="cuda:0"
        )
        # Pre-allocate a fp16 input staging tensor for fp16 OBB models.
        if self._fp16:
            self._gpu_input_fp16 = self._gpu_input.half()
            in_np_dtype = np.float16
            in_buf_ptr = self._gpu_input_fp16.data_ptr()
        else:
            self._gpu_input_fp16 = None
            in_np_dtype = np.float32
            in_buf_ptr = self._gpu_input.data_ptr()

        self._io_binding = self.session.io_binding()
        self._io_binding.bind_input(
            name=self._input_name,
            device_type="cuda",
            device_id=0,
            element_type=in_np_dtype,
            shape=(
                tuple(self._gpu_input.shape)
                if not self._fp16
                else tuple(self._gpu_input_fp16.shape)
            ),
            buffer_ptr=in_buf_ptr,
        )
        # Track the last bound input pointer so _run_inference can detect when
        # _preprocess_cuda_batch returns a new tensor (different address) and
        # needs to update the io_binding before the ORT session run.
        self._last_bound_input_ptr: int = in_buf_ptr
        # CUDA event used to synchronize preprocessing (default stream) →
        # ORT CUDA EP inference without stalling the CPU more than necessary.
        # _preprocess writes to _gpu_input via non-blocking copy_ + mul_; without
        # this event the ORT session could start reading the buffer on ORT's
        # internal CUDA stream before those PyTorch ops complete, producing
        # stale-frame results.
        self._preprocess_event = torch.cuda.Event()
        self._io_binding.bind_output(
            name=self._output_name,
            device_type="cuda",
            device_id=0,
            element_type=self._out_np_dtype,
            shape=tuple(self._output_tensor.shape),
            buffer_ptr=self._output_tensor.data_ptr(),
        )

    def _run_inference(self, img_tensor):
        import numpy as np

        # For fp16 models, convert the float32 GPU input to fp16 in the
        # pre-allocated staging buffer and rebind if necessary.
        if self._fp16:
            self._gpu_input_fp16.copy_(img_tensor)  # float32 -> fp16, in-place
            x = self._gpu_input_fp16
        else:
            x = img_tensor  # already float32, same buffer as bound in __init__

        # Rebuild output tensor only if the batch dimension changed at runtime
        # (not expected during normal single-frame inference).
        if x.shape[0] != self._output_tensor.shape[0]:
            import torch

            out_shape = list(self._static_out_shape)
            out_shape[0] = int(x.shape[0])
            out_dtype = getattr(torch, self._out_torch_dtype)
            self._output_tensor = torch.empty(
                out_shape, dtype=out_dtype, device=x.device
            )
            self._io_binding.clear_binding_outputs()
            self._io_binding.bind_output(
                name=self._output_name,
                device_type="cuda",
                device_id=0,
                element_type=self._out_np_dtype,
                shape=tuple(self._output_tensor.shape),
                buffer_ptr=self._output_tensor.data_ptr(),
            )

        # Rebind the input whenever the data pointer changes.
        # _preprocess_cuda_batch creates a new tensor on every call so its
        # address differs from the pre-allocated self._gpu_input that was bound
        # in __init__.  For the numpy single-frame path x IS self._gpu_input so
        # the pointer is stable and this check is a no-op after the first call.
        # If the CUDA allocator happens to reuse the same address (common for
        # same-size tensors) the check also correctly skips the redundant rebind.
        cur_ptr = x.data_ptr()
        if cur_ptr != self._last_bound_input_ptr:
            self._io_binding.clear_binding_inputs()
            self._io_binding.bind_input(
                name=self._input_name,
                device_type="cuda",
                device_id=0,
                element_type=np.float16 if self._fp16 else np.float32,
                shape=tuple(x.shape),
                buffer_ptr=cur_ptr,
            )
            self._last_bound_input_ptr = cur_ptr

        # Synchronize PyTorch's default CUDA stream before submitting the ORT
        # session.  _preprocess writes into _gpu_input via non-blocking copy_
        # and mul_ operations queued on PyTorch's stream; ORT's CUDA EP uses
        # its own internal stream, so without this sync ORT would read the
        # buffer before PyTorch's writes are visible, returning the previous
        # frame's data (stale-frame bug).
        self._preprocess_event.record()
        self._preprocess_event.synchronize()
        self.session.run_with_iobinding(self._io_binding)
        return self._output_tensor


class DirectTensorRTOBBExecutor(_BaseDirectOBBExecutor):
    def __init__(
        self,
        artifact_path: str,
        imgsz: int,
        class_names: dict[int, str] | None = None,
        class_count: int | None = None,
    ) -> None:
        super().__init__(artifact_path, imgsz, class_names, class_count)

        import tensorrt as trt  # type: ignore[import-not-found]

        with open(self.artifact_path, "rb") as handle:
            meta_len = struct.unpack("<I", handle.read(4))[0]
            meta_json = handle.read(meta_len).decode("utf-8")
            engine_data = handle.read()

        meta = json.loads(meta_json)
        if not self.names:
            self.names = {
                int(key): str(value)
                for key, value in dict(meta.get("names") or {}).items()
            }
            self.nc = max(1, len(self.names) or self.nc)
        # Read the end2end flag from the engine metadata so _postprocess uses
        # the correct NMS mode (iou_thres=1.0 for BNC end2end vs. 0.5 for CBC).
        # TRT metadata is JSON-parsed so values are native Python booleans, but
        # use _parse_meta_bool for consistency with the ONNX path.
        self._end2end = _parse_meta_bool(meta.get("end2end", False))

        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        self.engine = runtime.deserialize_cuda_engine(engine_data)
        if self.engine is None:
            raise RuntimeError("TensorRT failed to deserialize OBB engine")
        self.context = self.engine.create_execution_context()

        tensor_names = [
            self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)
        ]
        input_names = [
            name
            for name in tensor_names
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
        ]
        output_names = [
            name
            for name in tensor_names
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT
        ]
        if len(input_names) != 1 or len(output_names) != 1:
            raise RuntimeError(
                "TensorRT OBB engine must expose exactly one input and one output tensor"
            )
        self._input_name = input_names[0]
        self._output_name = output_names[0]
        # Store the model's static batch size for chunking in predict().
        batch_dim = self.engine.get_tensor_shape(self._input_name)[0]
        self._model_batch_size = int(batch_dim) if batch_dim > 0 else 1
        # A numeric (fixed) batch dim -- including exactly 1, the common case
        # for these OBB/detect engines exported with batch=1 -- means predict()
        # MUST chunk down to it. Without this, `model_bs > 1` alone misses the
        # batch=1 case and predict() feeds the whole multi-frame batch straight
        # to the engine, which TensorRT rejects outright
        # ("IExecutionContext::setInputShape: ... Static dimension mismatch").
        # Dynamic-axis engines report -1 here and should NOT be chunked.
        self._static_batch: bool = batch_dim > 0
        # Use a dedicated non-default CUDA stream so TensorRT does not inject extra
        # cudaStreamSynchronize() calls that it adds when running on the default stream.
        import torch

        self._cuda_stream = torch.cuda.Stream()
        # CUDA event used to synchronize preprocessing (default stream) →
        # TRT inference (dedicated stream) without blocking the CPU.
        self._sync_event = torch.cuda.Event()

        # Warm up the TRT engine immediately after construction to trigger JIT kernel
        # compilation on Ada/Hopper GPUs ("compiler backend").  Without this, the
        # first N real inference batches stall ~30-45 s each while the driver
        # compiles CUDA kernels.  Two warmup runs cover both JIT passes.
        try:
            _warmup = torch.zeros(
                (self._model_batch_size, 3, self.imgsz, self.imgsz),
                dtype=torch.float32,
                device="cuda",
            )
            self._run_inference(_warmup)  # JIT pass 1
            self._run_inference(_warmup)  # JIT pass 2
            torch.cuda.synchronize()
            del _warmup
        except Exception:
            pass

    def _run_inference(self, img_tensor):
        import torch

        x = img_tensor.float().contiguous()
        self.context.set_input_shape(self._input_name, tuple(x.shape))
        out_shape = tuple(self.context.get_tensor_shape(self._output_name))
        output = torch.empty(out_shape, dtype=torch.float32, device=x.device)
        self.context.set_tensor_address(self._input_name, x.data_ptr())
        self.context.set_tensor_address(self._output_name, output.data_ptr())
        # Record a CUDA event on the current (default) stream so that any
        # preprocessing work submitted there (copy_, mul_, etc.) is guaranteed
        # to complete before the TRT dedicated stream begins reading the input.
        # Using an event instead of a full stream.synchronize() avoids stalling
        # the CPU while still creating the necessary GPU-side ordering dependency.
        self._sync_event.record(torch.cuda.current_stream())
        self._cuda_stream.wait_event(self._sync_event)
        self.context.execute_async_v3(self._cuda_stream.cuda_stream)
        self._cuda_stream.synchronize()
        return output


class DirectPyTorchCUDAOBBExecutor(_BaseDirectOBBExecutor):
    """Direct PyTorch CUDA OBB executor with ``auto=False`` square preprocessing.

    Runs inference through the underlying PyTorch ``nn.Module`` directly,
    bypassing the Ultralytics predictor.  This guarantees identical
    preprocessing (square ``LetterBox(auto=False)``) as the TRT and ONNX direct
    executors, eliminating the input-shape discrepancy that arises when the
    Ultralytics predictor applies rectangular letterboxing (``auto=True``) for
    widescreen video frames.

    For a 1920×1080 input the Ultralytics predictor produces a 576×1024 model
    input while the TRT/ONNX executors produce 1024×1024.  The different FPN
    activations cause detection score differences even though both paths map
    boxes back to the original frame correctly via ``scale_boxes``.
    """

    def __init__(
        self,
        pt_model,
        imgsz: int,
        class_names: dict[int, str] | None = None,
        class_count: int | None = None,
    ) -> None:
        # _BaseDirectOBBExecutor.__init__ expects an artifact_path string for
        # file-based executors; pass an empty placeholder — this executor does
        # not read any file.
        super().__init__("", imgsz, class_names, class_count)

        # Retrieve the underlying nn.Module from a YOLO wrapper if needed.
        self._nn_module = getattr(pt_model, "model", pt_model)
        self._nn_module.eval()
        # Ensure model weights are on CUDA.
        if not any(p.is_cuda for p in self._nn_module.parameters()):
            self._nn_module.to("cuda:0")
        # Populate class names from the model if not supplied by caller.
        if not self.names:
            model_names = getattr(pt_model, "names", None) or getattr(
                self._nn_module, "names", None
            )
            if model_names:
                self.names = {int(k): str(v) for k, v in dict(model_names).items()}
                self.nc = max(1, len(self.names))
        # Fixed batch size of 1 — the PT executor is always single-frame.
        self._model_batch_size = 1
        # Always run inference in raw-head (CBC / one2many) mode so that the
        # CUDA path is consistent with TRT and ONNX artifacts, which are
        # exported with end2end=False by _yolo_runtime_export_profile.
        # If the model head has end2end=True (e.g. OBB26) we store a reference
        # and temporarily disable it for each forward call, then restore it.
        self._end2end = False
        self._e2e_head_ref = None  # head to patch per forward call
        try:
            _inner_model = getattr(self._nn_module, "model", None)
            if _inner_model is not None:
                _head = list(_inner_model.children())[-1]
                if bool(getattr(_head, "end2end", False)):
                    self._e2e_head_ref = _head
                    import logging as _logging

                    _logging.getLogger(__name__).info(
                        "DirectPyTorchCUDAOBBExecutor: end2end head detected — "
                        "running in raw CBC mode (matching TRT/ONNX export profile)."
                    )
        except Exception:
            pass

    def _run_inference(self, img_tensor):
        import torch

        # Temporarily disable end2end on the head so forward() returns raw CBC
        # output from the one2many head, consistent with how TRT/ONNX artifacts
        # are exported (end2end=False via _yolo_runtime_export_profile).
        _head = self._e2e_head_ref
        if _head is not None:
            _head.end2end = False
        try:
            with torch.no_grad():
                output = self._nn_module(img_tensor)
        finally:
            if _head is not None:
                _head.end2end = True  # restore original state
        # Ultralytics OBB model forward() returns (decoded_preds, raw_features)
        # in eval mode; we only need the first element.
        if isinstance(output, (list, tuple)):
            return output[0]
        return output


def create_direct_obb_executor(
    *,
    runtime: str,
    artifact_path: str,
    imgsz: int,
    class_names: dict[int, str] | None = None,
    class_count: int | None = None,
    pt_model=None,
):
    """Instantiate an ONNX, TensorRT, or PyTorch CUDA OBB executor.

    Parameters
    ----------
    runtime:
        ``"onnx"``, ``"tensorrt"``, or ``"cuda"`` / ``"pytorch_cuda"``.
    artifact_path:
        Path to the ``.onnx`` or ``.engine`` file (ignored for ``"cuda"``
        runtime).
    imgsz:
        Square model input size (pixels).
    class_names:
        Optional mapping of class index → class name.
    class_count:
        Number of output classes for the NMS ``nc`` parameter.
    pt_model:
        Required for ``"cuda"`` runtime: a loaded Ultralytics ``YOLO`` object or
        a bare ``nn.Module``.  Ignored for ONNX and TensorRT runtimes.
    """
    runtime_name = str(runtime or "").strip().lower()
    if runtime_name == "onnx":
        return DirectONNXOBBExecutor(
            artifact_path,
            imgsz,
            class_names=class_names,
            class_count=class_count,
        )
    if runtime_name == "tensorrt":
        return DirectTensorRTOBBExecutor(
            artifact_path,
            imgsz,
            class_names=class_names,
            class_count=class_count,
        )
    if runtime_name in {"cuda", "pytorch_cuda"}:
        if pt_model is None:
            raise ValueError(
                "create_direct_obb_executor: 'pt_model' is required for runtime='cuda'"
            )
        return DirectPyTorchCUDAOBBExecutor(
            pt_model,
            imgsz,
            class_names=class_names,
            class_count=class_count,
        )
    raise ValueError(f"Unsupported direct OBB runtime: {runtime}")


# ---------------------------------------------------------------------------
# Direct YOLO detect executors (sequential stage-1)
# ---------------------------------------------------------------------------
# These subclass the OBB executors to reuse the entire IO-binding / TensorRT
# pipeline — they only override ``_postprocess`` to return ``Results(boxes=…)``
# instead of ``Results(obb=…)``, matching the detect task's NMS output contract.


class DirectONNXDetectExecutor(DirectONNXOBBExecutor):
    """ONNX-backed direct executor for the YOLO *detect* task.

    Used as stage-1 of the sequential pipeline so that the entire
    NVDec → stage-1 detect → GPU crop → stage-2 OBB chain never
    leaves device memory.
    """

    def _postprocess(
        self,
        raw_preds,
        img_tensor,
        orig_frames,
        conf_thres: float,
        classes,
        max_det: int,
    ):
        import torch
        from ultralytics.engine.results import Results
        from ultralytics.utils import nms, ops

        preds = raw_preds[0] if isinstance(raw_preds, (tuple, list)) else raw_preds
        if not isinstance(preds, torch.Tensor):
            preds = torch.as_tensor(preds, device=img_tensor.device)

        is_end2end = getattr(self, "_end2end", False)
        # Match the OBB _postprocess convention: raw-CBC head exports
        # (end2end=False) require actual NMS suppression with iou_thres=0.5
        # because the model outputs many overlapping anchors per object.
        # End-to-end models already have NMS baked in, so iou_thres=1.0
        # is used to pass all slots through without further suppression.
        iou_thres = 1.0 if is_end2end else 0.5

        # rotated=False → standard NMS for YOLO detect.
        # Output per image: [N, 6] = [x1, y1, x2, y2, conf, cls] in letterbox space.
        filtered = nms.non_max_suppression(
            preds,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            classes=classes,
            max_det=max_det,
            nc=self.nc,
            rotated=False,
            end2end=is_end2end,
        )

        results = []
        for pred, orig_img in zip(filtered, orig_frames):
            if pred is None or len(pred) == 0:
                empty = torch.zeros((0, 6), device=img_tensor.device)
                results.append(
                    Results(orig_img, path="", names=self.names, boxes=empty)
                )
                continue
            # Scale xyxy from letterbox space → original-image pixel space.
            pred[:, :4] = ops.scale_boxes(
                img_tensor.shape[2:], pred[:, :4], orig_img.shape
            )
            results.append(Results(orig_img, path="", names=self.names, boxes=pred))
        return results


class DirectTensorRTDetectExecutor(DirectTensorRTOBBExecutor):
    """TensorRT-backed direct executor for the YOLO *detect* task.

    Mirrors ``DirectONNXDetectExecutor``; only ``_postprocess`` differs from the
    OBB equivalent.
    """

    def _postprocess(
        self,
        raw_preds,
        img_tensor,
        orig_frames,
        conf_thres: float,
        classes,
        max_det: int,
    ):
        import torch
        from ultralytics.engine.results import Results
        from ultralytics.utils import nms, ops

        preds = raw_preds[0] if isinstance(raw_preds, (tuple, list)) else raw_preds
        if not isinstance(preds, torch.Tensor):
            preds = torch.as_tensor(preds, device=img_tensor.device)

        is_end2end = getattr(self, "_end2end", False)
        iou_thres = 1.0 if is_end2end else 0.5

        filtered = nms.non_max_suppression(
            preds,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            classes=classes,
            max_det=max_det,
            nc=self.nc,
            rotated=False,
            end2end=is_end2end,
        )

        results = []
        for pred, orig_img in zip(filtered, orig_frames):
            if pred is None or len(pred) == 0:
                empty = torch.zeros((0, 6), device=img_tensor.device)
                results.append(
                    Results(orig_img, path="", names=self.names, boxes=empty)
                )
                continue
            pred[:, :4] = ops.scale_boxes(
                img_tensor.shape[2:], pred[:, :4], orig_img.shape
            )
            results.append(Results(orig_img, path="", names=self.names, boxes=pred))
        return results


def create_direct_detect_executor(
    *,
    runtime: str,
    artifact_path: str,
    imgsz: int,
    class_names: dict[int, str] | None = None,
    class_count: int | None = None,
):
    """Instantiate an ONNX or TensorRT detect executor for sequential stage-1.

    Parameters mirror :func:`create_direct_obb_executor`; the only difference is
    that the returned executor produces ``Results(boxes=…)`` objects (xyxy
    coordinates) rather than OBB results.
    """
    runtime_name = str(runtime or "").strip().lower()
    if runtime_name == "onnx":
        return DirectONNXDetectExecutor(
            artifact_path,
            imgsz,
            class_names=class_names,
            class_count=class_count,
        )
    if runtime_name == "tensorrt":
        return DirectTensorRTDetectExecutor(
            artifact_path,
            imgsz,
            class_names=class_names,
            class_count=class_count,
        )
    raise ValueError(f"Unsupported direct detect runtime: {runtime}")
