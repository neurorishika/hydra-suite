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


def _normalize_obb_geometry(
    w_arr: np.ndarray, h_arr: np.ndarray, angle_arr: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Canonicalize OBB axis angle and major/minor geometry.

    Mirrors legacy ``_obb_geometry._extract_raw_detections``:

    1. Fold raw angle to [0, pi) — the OBB axis is undirected, so any value
       outside this half-circle is equivalent to ``angle % pi``.
    2. When ``w < h``, the YOLO export reports the angle of the *minor*
       axis. Add 90 degrees so ``angles`` always describes the major axis.
    3. Compute ``major = max(w, h)``, ``minor = min(w, h)``; size = major*minor
       (==w*h, since multiplication is commutative) and aspect = major/minor.

    Returns (angles_rad in [0, pi), sizes, aspect_ratio) all as float32.
    """
    if angle_arr.size == 0:
        return (
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=np.float32),
        )
    # YOLO ultralytics exports report theta in radians; some non-standard
    # exports report degrees. Mirror legacy parity guard.
    if np.nanmax(np.abs(angle_arr)) > (2.0 * np.pi + 1e-3):
        angle_rad = np.deg2rad(angle_arr)
    else:
        angle_rad = angle_arr
    angle_deg = np.rad2deg(angle_rad) % 180.0
    swap_mask = w_arr < h_arr
    angle_deg = np.where(swap_mask, (angle_deg + 90.0) % 180.0, angle_deg)
    angles_fixed = np.deg2rad(angle_deg).astype(np.float32)
    major = np.where(swap_mask, h_arr, w_arr)
    minor = np.where(swap_mask, w_arr, h_arr)
    sizes = (major * minor).astype(np.float32)
    safe_minor = np.where(minor > 0, minor, 1.0)
    aspect = np.where(minor > 0, major / safe_minor, 1.0).astype(np.float32)
    return angles_fixed, sizes, aspect

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
    angles_fixed, sizes, aspect = _normalize_obb_geometry(
        xywhr[:, 2], xywhr[:, 3], xywhr[:, 4]
    )
    return OBBResult(
        frame_idx=frame_idx,
        centroids=centroids.astype(np.float32),
        angles=angles_fixed,
        sizes=sizes,
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


def materialize_tensors(raw: _RawOBBTensors) -> OBBResult:
    """Pull all device tensors to CPU as OBBResult with no filtering gates applied.

    Used before caching the GPU-path raw tensors. Generates fresh detection_ids
    via OBBResult.make_detection_ids since _RawOBBTensors does not carry them.
    The aspect-ratio computation uses a safe-h guard to avoid divide-by-zero
    warnings on degenerate detections.
    """
    if raw.xywhr.shape[0] == 0:
        return _empty_obb_result(raw.frame_idx)
    xywhr_np = raw.xywhr.cpu().numpy()
    corners_np = raw.corners.cpu().numpy()
    conf_np = raw.conf.cpu().numpy()
    angles_fixed, sizes, aspect = _normalize_obb_geometry(
        xywhr_np[:, 2], xywhr_np[:, 3], xywhr_np[:, 4]
    )
    n = int(len(conf_np))
    return OBBResult(
        frame_idx=raw.frame_idx,
        centroids=xywhr_np[:, :2].astype(np.float32),
        angles=angles_fixed,
        sizes=sizes,
        shapes=np.stack([sizes, aspect], axis=1),
        confidences=conf_np.astype(np.float32),
        corners=corners_np.astype(np.float32),
        detection_ids=OBBResult.make_detection_ids(raw.frame_idx, n),
    )
