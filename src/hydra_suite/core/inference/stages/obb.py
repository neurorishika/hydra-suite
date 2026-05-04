from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np
import torch
from ultralytics import YOLO

from ..config import ComputeRuntime, OBBConfig
from ..result import OBBResult
from ..runtime import RuntimeContext


class _RawOBBTensors(NamedTuple):
    """CUDA tensors from OBB model — no .cpu() call until filter_from_tensors()."""

    frame_idx: int
    xywhr: torch.Tensor  # (N, 5): cx, cy, w, h, angle_rad on device
    corners: torch.Tensor  # (N, 4, 2): corner coords on device
    conf: torch.Tensor  # (N,): confidence on device


@dataclass
class OBBModels:
    mode: str  # "direct" or "sequential"
    direct_model: Any | None = None
    detect_model: Any | None = None  # sequential stage-1
    obb_model: Any | None = None  # sequential stage-2

    def close(self) -> None:
        pass  # ultralytics models don't need explicit cleanup


def load_obb_models(config: OBBConfig, runtime: RuntimeContext) -> OBBModels:
    if config.mode == "direct":
        assert config.direct is not None
        m = _load_yolo(config.direct.model_path, config.direct.compute_runtime)
        return OBBModels(mode="direct", direct_model=m)
    assert config.sequential is not None
    detect = _load_yolo(
        config.sequential.detect_model_path,
        config.sequential.detect_compute_runtime,
    )
    obb = _load_yolo(
        config.sequential.obb_model_path, config.sequential.obb_compute_runtime
    )
    return OBBModels(mode="sequential", detect_model=detect, obb_model=obb)


def run_obb(
    frames: list[np.ndarray | torch.Tensor],
    models: OBBModels,
    config: OBBConfig,
    runtime: RuntimeContext,
) -> list[OBBResult | _RawOBBTensors]:
    """Run OBB detection on a batch of frames.

    Native CUDA path (tensor_on_cuda=True): returns _RawOBBTensors per frame.
    CPU/MPS or ONNX/TRT path (tensor_on_cuda=False): returns OBBResult per frame.
    iou=1.0 disables YOLO's internal NMS — filtering stage handles it.
    """
    if models.mode == "direct":
        return _run_direct(frames, models.direct_model, config, runtime)
    return _run_sequential(frames, models, config, runtime)


def _run_direct(
    frames: list,
    model: Any,
    config: OBBConfig,
    runtime: RuntimeContext,
) -> list[OBBResult | _RawOBBTensors]:
    conf_floor = config.direct.confidence_floor if config.direct else 1e-3
    results = model.predict(
        frames,
        conf=conf_floor,
        iou=1.0,
        verbose=False,
        device=runtime.device,
    )
    # Only native PyTorch "cuda" runtime leaves tensors on device.
    # onnx_cuda and tensorrt: predict() returns CPU numpy regardless of GPU use.
    if runtime.tensor_on_cuda:
        return [
            _extract_raw_tensors(r, idx, runtime.device)
            for idx, r in enumerate(results)
        ]
    return [_extract_obb_result(r, idx) for idx, r in enumerate(results)]


def _run_sequential(
    frames: list,
    models: OBBModels,
    config: OBBConfig,
    runtime: RuntimeContext,
) -> list[OBBResult]:
    seq = config.sequential
    stage1 = models.detect_model.predict(
        frames,
        conf=seq.detect_confidence_threshold,
        iou=1.0,
        verbose=False,
        device=runtime.device,
        imgsz=seq.detect_image_size if seq.detect_image_size > 0 else None,
    )
    results: list[OBBResult] = []
    for frame_idx, (frame, s1) in enumerate(zip(frames, stage1)):
        boxes = s1.boxes
        if boxes is None or len(boxes) == 0:
            results.append(_empty_obb_result(frame_idx))
            continue
        crops, offsets = _build_crops(frame, boxes, seq, runtime)
        if not crops:
            results.append(_empty_obb_result(frame_idx))
            continue
        batch_size = seq.stage2_batch_size or len(crops)
        sub: list[OBBResult] = []
        for i in range(0, len(crops), batch_size):
            batch = crops[i : i + batch_size]
            s2 = models.obb_model.predict(
                batch,
                conf=seq.obb_confidence_threshold,
                iou=1.0,
                verbose=False,
                device=runtime.device,
                imgsz=seq.stage2_image_size,
            )
            for j, r in enumerate(s2):
                sub.append(_extract_obb_result(r, frame_idx, offset=offsets[i + j]))
        results.append(_merge_obb_results(frame_idx, sub))
    return results


def _build_crops(
    frame: np.ndarray | torch.Tensor,
    boxes: Any,
    seq: Any,
    runtime: RuntimeContext,
) -> tuple[list[np.ndarray], list[tuple[float, float]]]:
    if isinstance(frame, torch.Tensor):
        arr = frame.cpu().numpy()
        if arr.ndim == 3 and arr.shape[0] == 3:
            arr = arr.transpose(1, 2, 0)
    else:
        arr = frame
    h, w = arr.shape[:2]
    crops: list[np.ndarray] = []
    offsets: list[tuple[float, float]] = []
    for x1, y1, x2, y2 in boxes.xyxy.cpu().numpy():
        bw, bh = x2 - x1, y2 - y1
        pad = seq.crop_pad_ratio * max(bw, bh)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        half = max(bw, bh) / 2 + pad
        if seq.enforce_square_crop:
            half = max(half, seq.min_crop_size_px / 2)
        ox1 = max(0, int(cx - half))
        oy1 = max(0, int(cy - half))
        ox2 = min(w, int(cx + half))
        oy2 = min(h, int(cy + half))
        crop = arr[oy1:oy2, ox1:ox2]
        if crop.size == 0:
            continue
        crops.append(crop)
        offsets.append((float(ox1), float(oy1)))
    return crops, offsets


def _extract_raw_tensors(result: Any, frame_idx: int, device: str) -> _RawOBBTensors:
    """Keep OBB tensors on the compute device — no .cpu() call."""
    obb = result.obb
    if obb is None or len(obb) == 0:
        dev = torch.device(device)
        return _RawOBBTensors(
            frame_idx=frame_idx,
            xywhr=torch.zeros((0, 5), dtype=torch.float32, device=dev),
            corners=torch.zeros((0, 4, 2), dtype=torch.float32, device=dev),
            conf=torch.zeros(0, dtype=torch.float32, device=dev),
        )
    return _RawOBBTensors(
        frame_idx=frame_idx,
        xywhr=obb.xywhr,
        corners=obb.xyxyxyxy,
        conf=obb.conf,
    )


def _extract_obb_result(
    result: Any,
    frame_idx: int,
    offset: tuple[float, float] = (0.0, 0.0),
) -> OBBResult:
    obb = result.obb
    if obb is None or len(obb) == 0:
        return _empty_obb_result(frame_idx)
    xywhr = obb.xywhr.cpu().numpy()  # (N, 5): cx,cy,w,h,angle
    corners = obb.xyxyxyxy.cpu().numpy()  # (N, 4, 2)
    conf = obb.conf.cpu().numpy()  # (N,)
    n = int(len(conf))
    ox, oy = offset
    centroids = xywhr[:, :2].copy()
    centroids[:, 0] += ox
    centroids[:, 1] += oy
    corners = corners.copy()
    corners[:, :, 0] += ox
    corners[:, :, 1] += oy
    w_arr, h_arr = xywhr[:, 2], xywhr[:, 3]
    sizes = w_arr * h_arr
    safe_h = np.where(h_arr > 0, h_arr, 1.0)
    aspect = np.where(h_arr > 0, w_arr / safe_h, 1.0)
    return OBBResult(
        frame_idx=frame_idx,
        centroids=centroids.astype(np.float32),
        angles=xywhr[:, 4].astype(np.float32),
        sizes=sizes.astype(np.float32),
        shapes=np.stack([sizes, aspect], axis=1).astype(np.float32),
        confidences=conf.astype(np.float32),
        corners=corners.astype(np.float32),
        detection_ids=OBBResult.make_detection_ids(frame_idx, n),
    )


def _empty_obb_result(frame_idx: int) -> OBBResult:
    return OBBResult(
        frame_idx=frame_idx,
        centroids=np.zeros((0, 2), dtype=np.float32),
        angles=np.zeros(0, dtype=np.float32),
        sizes=np.zeros(0, dtype=np.float32),
        shapes=np.zeros((0, 2), dtype=np.float32),
        confidences=np.zeros(0, dtype=np.float32),
        corners=np.zeros((0, 4, 2), dtype=np.float32),
        detection_ids=OBBResult.make_detection_ids(frame_idx, 0),
    )


def _merge_obb_results(frame_idx: int, parts: list[OBBResult]) -> OBBResult:
    non_empty = [r for r in parts if r.num_detections > 0]
    if not non_empty:
        return _empty_obb_result(frame_idx)
    total = sum(r.num_detections for r in non_empty)
    return OBBResult(
        frame_idx=frame_idx,
        centroids=np.concatenate([r.centroids for r in non_empty], axis=0),
        angles=np.concatenate([r.angles for r in non_empty]),
        sizes=np.concatenate([r.sizes for r in non_empty]),
        shapes=np.concatenate([r.shapes for r in non_empty], axis=0),
        confidences=np.concatenate([r.confidences for r in non_empty]),
        corners=np.concatenate([r.corners for r in non_empty], axis=0),
        # Regenerate IDs across the merged frame so they remain contiguous
        detection_ids=OBBResult.make_detection_ids(frame_idx, total),
    )


def _load_yolo(model_path: str, compute_runtime: ComputeRuntime) -> Any:
    model = YOLO(model_path)
    # Only native PyTorch models support .to(). ONNX and TensorRT artifacts
    # (.onnx, .engine) ignore .to() — their execution provider is set at
    # predict() time via device=. CoreML provider activates when device="mps".
    if compute_runtime == "cuda":
        model.to("cuda:0")
    elif compute_runtime == "mps":
        model.to("mps")
    return model
