from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np
import torch

from ..config import ComputeRuntime, OBBConfig
from ..result import OBBResult
from ..runtime import RuntimeContext, runtime_to_compute_runtime
from ..runtime_artifacts import load_obb_executor

logger = logging.getLogger(__name__)


def _valid_detection_mask(
    cx: np.ndarray,
    cy: np.ndarray,
    w_arr: np.ndarray,
    h_arr: np.ndarray,
    angle_fixed: np.ndarray,
    conf: np.ndarray,
) -> np.ndarray:
    """Finite + positive-geometry validity mask.

    Mirrors legacy ``_obb_geometry._extract_raw_detections`` (lines 303-312):
    drop detections whose centroid/size/angle/confidence is non-finite, or whose
    geometry is non-positive (``w<=0`` or ``h<=0`` ⟺ ``major<=0`` or
    ``minor<=0``). This prevents NaN/Inf centroids or angles from propagating
    into assignment costs and the Kalman filter (and from being cached).
    """
    return (
        np.isfinite(cx)
        & np.isfinite(cy)
        & np.isfinite(w_arr)
        & np.isfinite(h_arr)
        & np.isfinite(angle_fixed)
        & np.isfinite(conf)
        & (w_arr > 0)
        & (h_arr > 0)
    )


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


def _corners_from_xywhr(
    cx: np.ndarray,
    cy: np.ndarray,
    w_arr: np.ndarray,
    h_arr: np.ndarray,
    angle_fixed: np.ndarray,
) -> np.ndarray:
    """Build OBB corners from xywhr in legacy's ordering.

    Mirrors ``_obb_geometry._extract_raw_detections``: corners are constructed in
    the *major*-axis frame (``major = max(w, h)``, ``minor = min(w, h)``) rotated
    by ``angle_fixed``, ordered [TL, TR, BR, BL].

    Ultralytics' raw ``xyxyxyxy`` corner order differs from this, which makes
    ``compute_alignment_affine`` build the canonical crop mirrored/180-rotated vs
    legacy. SLEAP then predicts keypoints ~86px off (vs ~2.7px with this order),
    and the head-tail classifier sees a differently-posed crop. Reconstructing
    here makes every canonical crop (head-tail / CNN / pose) and its alignment
    affine byte-identical to legacy (validated element-wise across 25k dets).
    """
    if cx.size == 0:
        return np.zeros((0, 4, 2), dtype=np.float32)
    major = np.maximum(w_arr, h_arr)
    minor = np.minimum(w_arr, h_arr)
    half_w = major / 2.0
    half_h = minor / 2.0
    x_offsets = np.stack((-half_w, half_w, half_w, -half_w), axis=1)
    y_offsets = np.stack((-half_h, -half_h, half_h, half_h), axis=1)
    cos_t = np.cos(angle_fixed)
    sin_t = np.sin(angle_fixed)
    x = cx[:, None] + x_offsets * cos_t[:, None] - y_offsets * sin_t[:, None]
    y = cy[:, None] + x_offsets * sin_t[:, None] + y_offsets * cos_t[:, None]
    return np.stack((x, y), axis=2).astype(np.float32)


def load_obb_models(config: OBBConfig, runtime: RuntimeContext) -> OBBModels:
    # Derive backend from the RuntimeContext (which reflects runtime_tier via
    # from_config). Per-stage compute_runtime fields are deprecated in favor of
    # runtime_tier; they are kept in place for serialization only.
    compute_runtime = runtime_to_compute_runtime(runtime)
    if compute_runtime == "tensorrt":
        logger.warning(
            "Runtime fallback may apply for OBB stage: "
            "gpu_fast (TensorRT) requested — artifact availability governs actual backend."
        )
    if config.mode == "direct":
        assert config.direct is not None
        auto_export = config.direct.auto_export
        m = _load_yolo(
            config.direct.model_path,
            compute_runtime,
            auto_export=auto_export,
            max_det=config.max_detections,
        )
        return OBBModels(mode="direct", direct_model=m)
    assert config.sequential is not None
    auto_export = config.sequential.auto_export
    detect = _load_yolo(
        config.sequential.detect_model_path,
        compute_runtime,
        auto_export=auto_export,
        max_det=config.max_detections,
    )
    obb = _load_yolo(
        config.sequential.obb_model_path,
        compute_runtime,
        auto_export=auto_export,
        max_det=config.max_detections,
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
        classes=config.target_classes or None,
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
    return [
        _apply_raw_detection_cap(_extract_obb_result(r, idx), config.raw_detection_cap)
        for idx, r in enumerate(results)
    ]


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
        classes=config.target_classes or None,
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
        results.append(
            _apply_raw_detection_cap(
                _merge_obb_results(frame_idx, sub), config.raw_detection_cap
            )
        )
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
    conf = obb.conf.cpu().numpy()  # (N,)
    ox, oy = offset
    centroids = xywhr[:, :2].copy()
    centroids[:, 0] += ox
    centroids[:, 1] += oy
    angles_fixed, sizes, aspect = _normalize_obb_geometry(
        xywhr[:, 2], xywhr[:, 3], xywhr[:, 4]
    )
    # H6 parity: drop non-finite / non-positive-geometry detections before they
    # can be cached or fed to assignment/Kalman (legacy _obb_geometry:303-312).
    mask = _valid_detection_mask(
        centroids[:, 0], centroids[:, 1], xywhr[:, 2], xywhr[:, 3], angles_fixed, conf
    )
    if not mask.all():
        dropped = int(mask.size - int(mask.sum()))
        if dropped > 0:
            logger.warning(
                "Dropping %d invalid OBB detections with non-finite or "
                "non-positive geometry.",
                dropped,
            )
        xywhr = xywhr[mask]
        conf = conf[mask]
        centroids = centroids[mask]
        angles_fixed = angles_fixed[mask]
        sizes = sizes[mask]
        aspect = aspect[mask]
    n = int(len(conf))
    # Rebuild corners from xywhr in legacy ordering (see _corners_from_xywhr)
    # instead of ultralytics' xyxyxyxy, so canonical crops match legacy.
    corners = _corners_from_xywhr(
        centroids[:, 0], centroids[:, 1], xywhr[:, 2], xywhr[:, 3], angles_fixed
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


def _apply_raw_detection_cap(r: OBBResult, cap: int) -> OBBResult:
    """Sort detections by confidence descending and keep the top ``cap``.

    Replicates legacy ``_obb_geometry._extract_raw_detections`` (cap = 2 *
    MAX_TARGETS): the raw cap is applied at extraction, BEFORE size/aspect/IoU
    filtering, so the new pipeline feeds the identical detection set to the
    filtering + assignment stages. Detection IDs are regenerated in the new
    (confidence-sorted) slot order, matching legacy's post-sort id assignment.
    ``cap <= 0`` disables (no sort, no truncation).
    """
    if cap <= 0 or r.num_detections == 0:
        return r
    order = np.argsort(r.confidences)[::-1]
    if len(order) > cap:
        order = order[:cap]
    n = int(len(order))
    return OBBResult(
        frame_idx=r.frame_idx,
        centroids=np.ascontiguousarray(r.centroids[order]),
        angles=np.ascontiguousarray(r.angles[order]),
        sizes=np.ascontiguousarray(r.sizes[order]),
        shapes=np.ascontiguousarray(r.shapes[order]),
        confidences=np.ascontiguousarray(r.confidences[order]),
        corners=np.ascontiguousarray(r.corners[order]),
        detection_ids=OBBResult.make_detection_ids(r.frame_idx, n),
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


def _load_yolo(
    model_path: str,
    compute_runtime: ComputeRuntime,
    *,
    auto_export: bool = True,
    max_det: int = 20,
) -> Any:
    """Load the OBB executor for ``model_path`` under ``compute_runtime``.

    Thin delegator to :func:`load_obb_executor`:
      * cpu/mps/cuda → a plain ultralytics ``YOLO`` model (``.to()``-moved as
        before; CPU does not call ``.to()`` so CPU byte-parity is preserved).
      * onnx_*/tensorrt → a direct ONNX/TRT executor (auto-exporting ``.onnx``/
        ``.engine`` from ``.pt`` on first load when ``auto_export``); when no
        artifact exists and ``auto_export`` is False, a clear error is raised
        instead of silently running PyTorch (parity finding H4).
    """
    return load_obb_executor(
        model_path,
        compute_runtime,
        auto_export=auto_export,
        max_det=max_det,
    )


def materialize_tensors(raw: _RawOBBTensors, raw_detection_cap: int = 0) -> OBBResult:
    """Pull all device tensors to CPU as OBBResult with no filtering gates applied.

    Used before caching the GPU-path raw tensors. Generates fresh detection_ids
    via OBBResult.make_detection_ids since _RawOBBTensors does not carry them.
    The aspect-ratio computation uses a safe-h guard to avoid divide-by-zero
    warnings on degenerate detections.
    """
    if raw.xywhr.shape[0] == 0:
        return _empty_obb_result(raw.frame_idx)
    xywhr_np = raw.xywhr.cpu().numpy()
    conf_np = raw.conf.cpu().numpy()
    angles_fixed, sizes, aspect = _normalize_obb_geometry(
        xywhr_np[:, 2], xywhr_np[:, 3], xywhr_np[:, 4]
    )
    # H6 parity: drop non-finite / non-positive-geometry detections before
    # caching the GPU-path raw tensors (legacy _obb_geometry:303-312).
    mask = _valid_detection_mask(
        xywhr_np[:, 0],
        xywhr_np[:, 1],
        xywhr_np[:, 2],
        xywhr_np[:, 3],
        angles_fixed,
        conf_np,
    )
    if not mask.all():
        dropped = int(mask.size - int(mask.sum()))
        if dropped > 0:
            logger.warning(
                "Dropping %d invalid OBB detections with non-finite or "
                "non-positive geometry.",
                dropped,
            )
        xywhr_np = xywhr_np[mask]
        conf_np = conf_np[mask]
        angles_fixed = angles_fixed[mask]
        sizes = sizes[mask]
        aspect = aspect[mask]
    n = int(len(conf_np))
    # Rebuild corners in legacy ordering from xywhr (see _corners_from_xywhr).
    corners_np = _corners_from_xywhr(
        xywhr_np[:, 0], xywhr_np[:, 1], xywhr_np[:, 2], xywhr_np[:, 3], angles_fixed
    )
    result = OBBResult(
        frame_idx=raw.frame_idx,
        centroids=xywhr_np[:, :2].astype(np.float32),
        angles=angles_fixed,
        sizes=sizes,
        shapes=np.stack([sizes, aspect], axis=1),
        confidences=conf_np.astype(np.float32),
        corners=corners_np.astype(np.float32),
        detection_ids=OBBResult.make_detection_ids(raw.frame_idx, n),
    )
    return _apply_raw_detection_cap(result, raw_detection_cap)
