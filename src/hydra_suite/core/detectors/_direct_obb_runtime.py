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
from typing import Sequence

import numpy as np


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

    def _preprocess(self, frames: Sequence[np.ndarray]):
        import torch
        from ultralytics.data.augment import LetterBox

        if not frames:
            raise ValueError("direct OBB executor received no frames")

        letterbox = LetterBox((self.imgsz, self.imgsz), auto=False, stride=self.stride)
        batch = []
        for frame in frames:
            transformed = letterbox(image=frame)
            if transformed.ndim != 3 or transformed.shape[2] != 3:
                raise ValueError("direct OBB executor expects HxWx3 BGR frames")
            transformed = transformed[..., ::-1]
            transformed = transformed.transpose((2, 0, 1))
            batch.append(np.ascontiguousarray(transformed, dtype=np.float32))

        batch_np = np.stack(batch, axis=0) * (1.0 / 255.0)
        return torch.from_numpy(batch_np).to(device="cuda:0", dtype=torch.float32)

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

        filtered = nms.non_max_suppression(
            preds,
            conf_thres=conf_thres,
            iou_thres=1.0,
            classes=classes,
            max_det=max_det,
            nc=self.nc,
            rotated=True,
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
        img_tensor = self._preprocess(frames)
        raw_preds = self._run_inference(img_tensor)
        return self._postprocess(
            raw_preds,
            img_tensor,
            frames,
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

    def _run_inference(self, img_tensor):
        import numpy as np
        import torch

        x = img_tensor.half() if self._fp16 else img_tensor.float()
        x = x.contiguous()

        io = self.session.io_binding()
        io.bind_input(
            name=self._input_name,
            device_type="cuda",
            device_id=0,
            element_type=np.float16 if self._fp16 else np.float32,
            shape=tuple(x.shape),
            buffer_ptr=x.data_ptr(),
        )
        io.bind_output(self._output_name, device_type="cuda", device_id=0)
        self.session.run_with_iobinding(io)
        out_np = io.get_outputs()[0].numpy()
        return torch.from_numpy(out_np).to(x.device)


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

    def _run_inference(self, img_tensor):
        import torch

        x = img_tensor.float().contiguous()
        self.context.set_input_shape(self._input_name, tuple(x.shape))
        out_shape = tuple(self.context.get_tensor_shape(self._output_name))
        output = torch.empty(out_shape, dtype=torch.float32, device=x.device)
        self.context.set_tensor_address(self._input_name, x.data_ptr())
        self.context.set_tensor_address(self._output_name, output.data_ptr())
        stream = torch.cuda.current_stream(x.device)
        self.context.execute_async_v3(stream.cuda_stream)
        stream.synchronize()
        return output


def create_direct_obb_executor(
    *,
    runtime: str,
    artifact_path: str,
    imgsz: int,
    class_names: dict[int, str] | None = None,
    class_count: int | None = None,
):
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
    raise ValueError(f"Unsupported direct OBB runtime: {runtime}")
