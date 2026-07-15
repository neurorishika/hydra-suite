from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, NamedTuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from ....utils.obb_from_mask import letterbox_gain_pad, rotated_rect_from_masks
from ..config import ComputeRuntime, OBBConfig
from ..result import OBBResult
from ..runtime import RuntimeContext, runtime_to_compute_runtime
from ..runtime_artifacts import DirectExecutorAdapter, load_obb_executor

logger = logging.getLogger(__name__)

_FALLBACK_IMGSZ = 1024


def _resolve_imgsz(model: Any) -> int:
    """Read the input image size from a loaded ultralytics YOLO model.

    Tries the documented attributes in order of preference:
      1. ``model.imgsz`` — set by ``DirectExecutorAdapter`` for direct
         ONNX/TRT executors (gpu_fast), which have no ``.overrides``/
         ``.model.args`` for the checks below to duck-type against.
      2. ``model.overrides["imgsz"]`` — set from the .pt checkpoint args at
         load time and patched by predict() kwargs.
      3. ``model.model.args["imgsz"]`` — the Ultralytics Trainer-style dict
         stored in the inner nn.Module after load.

    Falls back to ``_FALLBACK_IMGSZ`` (1024) with a warning if none of these
    attributes resolve to a positive integer.
    """
    try:
        v = getattr(model, "imgsz", 0)
        if isinstance(v, (int, float)) and int(v) > 0:
            return int(v)
    except Exception:
        pass
    try:
        v = getattr(model, "overrides", {}).get("imgsz", 0)
        if isinstance(v, (int, float)) and int(v) > 0:
            return int(v)
        # overrides may store a list when different h/w are used
        if isinstance(v, (list, tuple)) and len(v) > 0 and int(v[0]) > 0:
            return int(v[0])
    except Exception:
        pass
    try:
        v = model.model.args["imgsz"]
        if isinstance(v, (int, float)) and int(v) > 0:
            return int(v)
        if isinstance(v, (list, tuple)) and len(v) > 0 and int(v[0]) > 0:
            return int(v[0])
    except Exception:
        pass
    logger.warning(
        "Could not read imgsz from ultralytics model (checked overrides and "
        "model.args); falling back to %d. Detections will be correct only if "
        "the actual model input size is %d.",
        _FALLBACK_IMGSZ,
        _FALLBACK_IMGSZ,
    )
    return _FALLBACK_IMGSZ


def _gpu_letterbox_batch(
    cuda_frames: list,
    imgsz: int,
) -> tuple[torch.Tensor, list[tuple[float, float, float]]]:
    """Letterbox a list of CUDA HWC uint8 RGB tensors into a single batched tensor.

    Mirrors ``_DirectOBBRuntime._preprocess_cuda_batch`` (detectors package)
    but is kept local to avoid a cross-module import that would violate the
    layer boundary (inference → detectors is not permitted).

    Each frame is scaled so that ``max(H, W)`` fits within ``imgsz`` (aspect-
    ratio-preserving), then symmetrically padded to ``(imgsz, imgsz)`` with
    grey (114/255).  The result is normalised to float32 ``[0, 1]``.

    Returns
    -------
    batched : torch.Tensor
        Shape ``(B, 3, imgsz, imgsz)``, float32, on the same CUDA device as the
        input frames.
    params : list of (r, pad_left, pad_top)
        Per-frame letterbox parameters needed to invert the transform on the
        model outputs.  ``r`` is the scale factor; ``pad_left`` and ``pad_top``
        are the integer pixel offsets of the content region inside ``imgsz``.
    """
    processed: list[torch.Tensor] = []
    params: list[tuple[float, float, float]] = []
    for frame in cuda_frames:
        H, W = int(frame.shape[0]), int(frame.shape[1])
        r = min(imgsz / H, imgsz / W)
        new_h = int(H * r)
        new_w = int(W * r)
        # HWC uint8 → NCHW float32
        t = frame.permute(2, 0, 1).unsqueeze(0).to(dtype=torch.float32)
        if new_h != H or new_w != W:
            t = F.interpolate(
                t, size=(new_h, new_w), mode="bilinear", align_corners=False
            )
        pad_top = (imgsz - new_h) // 2
        pad_left = (imgsz - new_w) // 2
        pad_bot = imgsz - new_h - pad_top
        pad_right = imgsz - new_w - pad_left
        if pad_top or pad_bot or pad_left or pad_right:
            t = F.pad(t, (pad_left, pad_right, pad_top, pad_bot), value=114.0)
        processed.append(t.squeeze(0).mul_(1.0 / 255.0))
        params.append((float(r), float(pad_left), float(pad_top)))
    return torch.stack(processed, dim=0), params


def _frames_are_cuda_tensors(frames: list) -> bool:
    """True when the frame list is the NVDEC CUDA-tensor kind (HWC uint8 RGB)."""
    return bool(frames) and isinstance(frames[0], torch.Tensor) and frames[0].is_cuda


def _invert_letterbox_on_result(
    result: Any,
    r: float,
    pad_left: float,
    pad_top: float,
    orig_shape: tuple[int, int] | None = None,
) -> None:
    """Invert the letterbox transform on a single ultralytics Results object in-place.

    When ``model.predict`` receives a pre-letterboxed ``(B,3,imgsz,imgsz)``
    float32 tensor, ultralytics treats ``orig_shape == tensor_shape == imgsz``
    and therefore does NOT rescale boxes back to original-frame coordinates.
    This function applies the inverse letterbox so that downstream extract
    functions (``_extract_raw_tensors``, ``_extract_obb_result``) always see
    original-frame coordinates, exactly as they would on the numpy list path.

    Inverse formula (all on-device tensor ops, no .cpu() call):
        x_orig = (x_lb - pad_left) / r
        y_orig = (y_lb - pad_top)  / r
        w_orig = w_lb / r
        h_orig = h_lb / r
        angle  = unchanged

    We mutate the backing ``result.obb.data`` tensor directly rather than the
    ``xywhr`` / ``xyxyxyxy`` properties. In ultralytics ``OBB.xywhr`` is a slice
    of ``data`` (a view — mutation would persist) but ``xyxyxyxy`` is RECOMPUTED
    from ``data`` on every access (mutating the returned tensor is discarded).
    Writing through ``data`` (columns 0-3 = cx, cy, w, h; col 4 = angle) is the
    single source of truth: both ``xywhr`` and the recomputed corners then
    reflect original-frame coordinates, independent of ultralytics version.

    ``result.obb`` is ``None`` for a **detect** or **segment** checkpoint, so
    the same inversion is applied to ``result.boxes`` (whose ``data`` columns
    0-3 are x1, y1, x2, y2) in that case -- without it, detect geometry stays
    in letterbox coordinates (every centroid/size scaled by ``r`` and shifted
    by the pad) with no error raised.

    ``orig_shape``, when given, also restores ``result.orig_shape`` to the TRUE
    frame shape. ultralytics sets it to ``(imgsz, imgsz)`` when predicting on a
    pre-letterboxed tensor, which would make ``letterbox_gain_pad`` degenerate
    to ``gain=1, pad=0`` and turn the segment path's mask<->frame conversion
    into a no-op.
    """
    if orig_shape is not None:
        result.orig_shape = (int(orig_shape[0]), int(orig_shape[1]))
    obb = getattr(result, "obb", None)
    # ultralytics runs predict() under torch.inference_mode(), so `data` is an
    # "inference tensor" that cannot be mutated in-place outside that context
    # ("Inplace update to inference tensor outside InferenceMode is not
    # allowed"). Re-enter inference_mode to perform the in-place coord inversion.
    if obb is not None and len(obb) > 0:
        data = obb.data  # (N, >=5): cx, cy, w, h, angle, [conf, cls]
        with torch.inference_mode():
            data[:, 0] = (data[:, 0] - pad_left) / r  # cx
            data[:, 1] = (data[:, 1] - pad_top) / r  # cy
            data[:, 2] = data[:, 2] / r  # w
            data[:, 3] = data[:, 3] / r  # h
            # angle (col 4) and conf/cls (cols 5+) are unchanged
        return
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return
    data = boxes.data  # (N, >=6): x1, y1, x2, y2, [track_id], conf, cls
    with torch.inference_mode():
        data[:, 0] = (data[:, 0] - pad_left) / r  # x1
        data[:, 1] = (data[:, 1] - pad_top) / r  # y1
        data[:, 2] = (data[:, 2] - pad_left) / r  # x2
        data[:, 3] = (data[:, 3] - pad_top) / r  # y2
        # conf/cls (and any track id) are unchanged.
    # result.masks (segment) is deliberately NOT touched: it stays in its own
    # letterbox-space canvas, and _extract_obb_from_masks maps the (now
    # original-frame) boxes into that canvas via letterbox_gain_pad against the
    # restored result.orig_shape -- which is exactly what the numpy list path
    # produces.


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


def _assert_direct_task_matches_checkpoint(
    model: Any, model_task: str, model_path: str
) -> None:
    """Fail loudly when the checkpoint's own task disagrees with ``model_task``.

    On the torch runtimes (cpu/mps/cuda) ultralytics infers the task from the
    checkpoint and IGNORES the ``task=`` argument, so a mismatch is silent: a
    segment checkpoint used with ``model_task="obb"`` makes ``result.obb`` None
    (and an OBB checkpoint used with ``model_task="detect"`` makes
    ``result.boxes`` None) on EVERY frame -- the user just sees zero detections
    for the whole video with no error. The direct executors (tensorrt/coreml)
    expose no ``.task``; they are built for the requested task explicitly, so
    they are skipped here.
    """
    ckpt_task = getattr(model, "task", None)
    if not isinstance(ckpt_task, str) or ckpt_task == model_task:
        return
    raise ValueError(
        f"Direct-mode OBB model task mismatch: the checkpoint "
        f"'{model_path}' is a '{ckpt_task}' model, but the configured "
        f"direct model task is '{model_task}'. Detection would silently "
        f"return zero results on every frame. Set the direct model task to "
        f"'{ckpt_task}', or select a '{model_task}' checkpoint."
    )


def load_obb_models(
    config: OBBConfig, runtime: RuntimeContext, *, batch_size: int = 1
) -> OBBModels:
    # Derive backend from the RuntimeContext (which reflects runtime_tier via
    # from_config). Per-stage compute_runtime fields are deprecated in favor of
    # runtime_tier; they are kept in place for serialization only.
    compute_runtime = runtime_to_compute_runtime(runtime)
    if compute_runtime in ("tensorrt", "coreml"):
        logger.warning(
            "Runtime fallback may apply for OBB stage: "
            "gpu_fast (%s) requested — artifact availability governs actual backend.",
            compute_runtime,
        )
    if config.mode == "direct":
        assert config.direct is not None
        auto_export = config.direct.auto_export
        m = _load_yolo(
            config.direct.model_path,
            compute_runtime,
            auto_export=auto_export,
            max_det=config.max_detections,
            batch_size=batch_size,
            task=config.direct.model_task,
        )
        _assert_direct_task_matches_checkpoint(
            m, config.direct.model_task, config.direct.model_path
        )
        return OBBModels(mode="direct", direct_model=m)
    assert config.sequential is not None
    if batch_size > 1:
        logger.warning(
            "Sequential-mode OBB with detection_batch_size=%d: dynamic-batch "
            "TensorRT engines showed a much larger cross-run detection "
            "discrepancy for sequential mode (~18%%) than direct mode (~1%%) "
            "in real-hardware verification. Root-caused: stage-1's raw "
            "detections are as clean as direct mode (~0.16%% divergence), "
            "but small stage-1 coordinate differences shift crop boundaries "
            "fed to stage-2, and stage-2's tiny-crop OBB estimation is "
            "highly sensitive to that shift -- an architectural property of "
            "the sequential/crop-based pipeline, not a batching bug. See "
            "docs/superpowers/specs/2026-07-03-tensorrt-coreml-"
            "cross-frame-batching-design.md. Consider batch_size=1 for "
            "sequential OBB unless this cross-run divergence is acceptable "
            "for your use case.",
            batch_size,
        )
    auto_export = config.sequential.auto_export
    detect_imgsz = config.sequential.detect_image_size
    detect = _load_yolo(
        config.sequential.detect_model_path,
        compute_runtime,
        auto_export=auto_export,
        max_det=config.max_detections,
        imgsz_override=detect_imgsz if detect_imgsz > 0 else None,
        # Stage-1 is a plain detector (no angle head) -- must be parsed as
        # Results(boxes=...), not Results(obb=...), under tensorrt/onnx.
        task="detect",
        batch_size=batch_size,
    )
    # stage2_image_size is always the effective input size (the pipeline
    # pre-resizes every crop to it in _resize_crops_for_stage2), so the
    # artifact must be built at that size, not the checkpoint's own default.
    # stage2_batch_size, when set, is the number of crops stage-2 is called
    # with per chunk (see _run_sequential's `batch_size = seq.stage2_batch_size
    # or len(crops)`); falls back to the frame-window batch_size when unset so
    # the exported artifact still gets a dynamic profile sized reasonably.
    obb = _load_yolo(
        config.sequential.obb_model_path,
        compute_runtime,
        auto_export=auto_export,
        max_det=config.max_detections,
        imgsz_override=config.sequential.stage2_image_size,
        batch_size=config.sequential.stage2_batch_size or batch_size,
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

    # Detect the CUDA-tensor frame case: NvdecFrameReader yields a list of
    # CUDA torch.Tensors (HWC uint8 RGB). A plain ultralytics YOLO model (the
    # torch cpu/mps/cuda runtimes) does NOT accept a list of tensors as a
    # prediction source (raises TypeError inside check_source / autocast_list),
    # so for that case only we GPU-letterbox the list into a single batched
    # tensor, run predict on that, then invert the letterbox on each result so
    # downstream extract functions see original-frame coordinates — identical
    # to what the numpy list path produces.
    #
    # DirectExecutorAdapter (gpu_fast direct executors) is NOT an ultralytics
    # model — its own predict() already accepts a raw list of CUDA HWC frames
    # and does its own correct letterbox + original-frame coordinate scaling
    # (_BaseDirectOBBExecutor._preprocess_cuda_batch / _postprocess). Routing
    # it through the manual pre-batch above double-preprocesses: the adapter
    # splits the already-letterboxed (B,3,imgsz,imgsz) tensor back into a list
    # of (3,imgsz,imgsz) slices and re-letterboxes each as if it were a raw
    # (H,W,3) frame, corrupting the shape fed to TensorRT ("Static dimension
    # mismatch" in setInputShape) whenever imgsz != 3. So it must take the
    # plain frames-list path below, same as the non-CUDA-tensor case.
    if _frames_are_cuda_tensors(frames) and not isinstance(
        model, DirectExecutorAdapter
    ):
        imgsz = _resolve_imgsz(model)
        batched, lb_params = _gpu_letterbox_batch(frames, imgsz)
        results = model.predict(
            batched,
            conf=conf_floor,
            iou=1.0,
            classes=config.target_classes or None,
            verbose=False,
            device=runtime.device,
        )
        # Invert letterbox so coordinates are in original-frame space before
        # the extract functions read them. This covers obb (result.obb), detect
        # (result.boxes) AND segment (result.boxes + restored result.orig_shape,
        # which the mask<->frame conversion derives its gain from).
        for frame, result, (r, pad_left, pad_top) in zip(frames, results, lb_params):
            _invert_letterbox_on_result(
                result,
                r,
                pad_left,
                pad_top,
                orig_shape=(int(frame.shape[0]), int(frame.shape[1])),
            )
    else:
        results = model.predict(
            frames,
            conf=conf_floor,
            iou=1.0,
            classes=config.target_classes or None,
            verbose=False,
            device=runtime.device,
        )

    model_task = config.direct.model_task if config.direct else "obb"

    if model_task == "detect":
        fixed_angle_rad = math.radians(
            config.direct.fixed_angle_deg if config.direct else 0.0
        )
        # Zero-CPU-sync fast path under the native cuda runtime, mirroring
        # "obb"'s own tensor_on_cuda branch below -- normalize/corners/
        # finite-filtering is deferred to the shared materialize_tensors().
        if runtime.tensor_on_cuda:
            return [
                _extract_raw_tensors_from_boxes(r, idx, fixed_angle_rad, runtime.device)
                for idx, r in enumerate(results)
            ]
        return [
            _apply_raw_detection_cap(
                _extract_obb_from_boxes(r, idx, fixed_angle_rad),
                config.raw_detection_cap,
            )
            for idx, r in enumerate(results)
        ]

    if model_task == "segment":
        # rotated_rect_from_masks does all the heavy per-pixel/per-angle work
        # on-device with no internal .cpu() calls, so under the native cuda
        # runtime segment gets the exact same zero-CPU-sync _RawOBBTensors
        # fast path as "obb"/"detect" -- the sync is deferred to
        # materialize_tensors(), same as every other detection source.
        seg_num_angles = config.direct.seg_num_angles if config.direct else 24
        seg_crop_size = config.direct.seg_crop_size if config.direct else 64
        seg_pad_ratio = config.direct.seg_pad_ratio if config.direct else 0.15
        seg_mask_threshold = config.direct.seg_mask_threshold if config.direct else 0.5
        if runtime.tensor_on_cuda:
            return [
                _extract_raw_tensors_from_masks(
                    r,
                    idx,
                    runtime.device,
                    config.raw_detection_cap,
                    num_angles=seg_num_angles,
                    crop_size=seg_crop_size,
                    pad_ratio=seg_pad_ratio,
                    mask_threshold=seg_mask_threshold,
                )
                for idx, r in enumerate(results)
            ]
        return [
            _apply_raw_detection_cap(
                _extract_obb_from_masks(
                    r,
                    idx,
                    config.raw_detection_cap,
                    num_angles=seg_num_angles,
                    crop_size=seg_crop_size,
                    pad_ratio=seg_pad_ratio,
                    mask_threshold=seg_mask_threshold,
                ),
                config.raw_detection_cap,
            )
            for idx, r in enumerate(results)
        ]

    # model_task == "obb": existing native-OBB behaviour, unchanged.
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
    stage1_kwargs: dict[str, Any] = {}
    if seq.detect_image_size > 0:
        stage1_kwargs["imgsz"] = seq.detect_image_size
    stage1 = models.detect_model.predict(
        frames,
        conf=seq.detect_confidence_threshold,
        iou=1.0,
        classes=config.target_classes or None,
        verbose=False,
        device=runtime.device,
        **stage1_kwargs,
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
        orig_sizes = [(c.shape[1], c.shape[0]) for c in crops]  # (w, h)
        # Mirror legacy yolo_detector._seq_resize_crops_for_stage2: pre-resize
        # each crop to the exact stage-2 input size with cv2 INTER_LINEAR,
        # rather than letting Ultralytics' internal letterbox resize it (which
        # can pick a different interpolation/stride-padded shape and shift
        # borderline detections across the confidence threshold).
        crops = _resize_crops_for_stage2(crops, seq.stage2_image_size)
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
                orig_w, orig_h = orig_sizes[i + j]
                scale = (
                    (orig_w / seq.stage2_image_size, orig_h / seq.stage2_image_size)
                    if seq.stage2_image_size > 0
                    else (1.0, 1.0)
                )
                sub.append(
                    _extract_obb_result(
                        r, frame_idx, offset=offsets[i + j], scale=scale
                    )
                )
        results.append(
            _apply_raw_detection_cap(
                _merge_obb_results(frame_idx, sub), config.raw_detection_cap
            )
        )
    return results


def _resize_crops_for_stage2(
    crops: list[np.ndarray], stage2_image_size: int
) -> list[np.ndarray]:
    if stage2_image_size <= 0:
        return crops
    out = []
    for crop in crops:
        h_c, w_c = crop.shape[:2]
        if h_c != stage2_image_size or w_c != stage2_image_size:
            out.append(
                cv2.resize(
                    crop,
                    (stage2_image_size, stage2_image_size),
                    interpolation=cv2.INTER_LINEAR,
                )
            )
        else:
            out.append(crop)
    return out


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
        # Mirrors legacy yolo_detector._build_sequential_crop exactly (padded
        # square box centered on the stage-1 bbox, floor/ceil-clipped to the
        # frame) so stage-2 sees byte-identical crop content to legacy.
        bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        half = max(bw, bh) / 2 + seq.crop_pad_ratio * max(bw, bh)
        if seq.enforce_square_crop:
            half = max(half, seq.min_crop_size_px / 2)
        ox1 = int(np.floor(max(0.0, cx - half)))
        oy1 = int(np.floor(max(0.0, cy - half)))
        ox2 = int(np.ceil(min(float(w), cx + half)))
        oy2 = int(np.ceil(min(float(h), cy + half)))
        if ox2 <= ox1 or oy2 <= oy1:
            continue
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
    scale: tuple[float, float] = (1.0, 1.0),
) -> OBBResult:
    obb = result.obb
    if obb is None or len(obb) == 0:
        return _empty_obb_result(frame_idx)
    xywhr = obb.xywhr.cpu().numpy().copy()  # (N, 5): cx,cy,w,h,angle
    conf = obb.conf.cpu().numpy()  # (N,)
    ox, oy = offset
    sx, sy = scale
    # Stage-2 predicts on a crop resized to a fixed square (stage2_image_size);
    # rescale cx/w by sx and cy/h by sy back to the crop's own pixel space
    # before offsetting into frame coordinates (mirrors legacy
    # yolo_detector._seq_accumulate_crop_detections).
    xywhr[:, 0] *= sx
    xywhr[:, 2] *= sx
    xywhr[:, 1] *= sy
    xywhr[:, 3] *= sy
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


def _extract_obb_from_boxes(
    result: Any,
    frame_idx: int,
    fixed_angle_rad: float,
) -> OBBResult:
    """Build an OBBResult from a plain (axis-aligned) detect model's boxes.

    Every detection is assigned ``fixed_angle_rad`` before being folded through
    the same ``_normalize_obb_geometry`` / ``_corners_from_xywhr`` pipeline used
    for native-OBB output, so downstream consumers (filtering, assignment,
    canonical crops) cannot tell the geometry did not come from an OBB head.
    """
    boxes = result.boxes
    if boxes is None or boxes.xyxy.shape[0] == 0:
        return _empty_obb_result(frame_idx)
    xyxy = boxes.xyxy.cpu().numpy().copy()  # (N, 4): x1,y1,x2,y2
    conf = boxes.conf.cpu().numpy()  # (N,)
    cx = (xyxy[:, 0] + xyxy[:, 2]) / 2.0
    cy = (xyxy[:, 1] + xyxy[:, 3]) / 2.0
    w_arr = xyxy[:, 2] - xyxy[:, 0]
    h_arr = xyxy[:, 3] - xyxy[:, 1]
    angle_arr = np.full(cx.shape, float(fixed_angle_rad), dtype=np.float32)
    angles_fixed, sizes, aspect = _normalize_obb_geometry(w_arr, h_arr, angle_arr)
    mask = _valid_detection_mask(cx, cy, w_arr, h_arr, angles_fixed, conf)
    if not mask.all():
        dropped = int(mask.size - int(mask.sum()))
        if dropped > 0:
            logger.warning(
                "Dropping %d invalid detect-as-OBB detections with non-finite "
                "or non-positive geometry.",
                dropped,
            )
        cx, cy, w_arr, h_arr = cx[mask], cy[mask], w_arr[mask], h_arr[mask]
        conf, angles_fixed, sizes, aspect = (
            conf[mask],
            angles_fixed[mask],
            sizes[mask],
            aspect[mask],
        )
    n = int(len(conf))
    corners = _corners_from_xywhr(cx, cy, w_arr, h_arr, angles_fixed)
    return OBBResult(
        frame_idx=frame_idx,
        centroids=np.stack([cx, cy], axis=1).astype(np.float32),
        angles=angles_fixed,
        sizes=sizes,
        shapes=np.stack([sizes, aspect], axis=1).astype(np.float32),
        confidences=conf.astype(np.float32),
        corners=corners.astype(np.float32),
        detection_ids=OBBResult.make_detection_ids(frame_idx, n),
    )


def _extract_obb_from_masks(
    result: Any,
    frame_idx: int,
    raw_detection_cap: int = 0,
    *,
    num_angles: int = 24,
    crop_size: int = 64,
    pad_ratio: float = 0.15,
    mask_threshold: float = 0.5,
) -> OBBResult:
    """Build an OBBResult from a segmentation model's predicted masks.

    Angle/size come from ``rotated_rect_from_masks`` (GPU-native, no cv2) run
    on the mask tensor's own square coordinate space; ``letterbox_gain_pad``
    converts the caller's original-frame boxes into that space beforehand and
    the resulting (cx, cy, w, h) back afterwards -- a single uniform gain plus
    translation, which (unlike independent per-axis ratios) never distorts
    the recovered angle. The result is then folded through the same
    ``_normalize_obb_geometry`` / ``_corners_from_xywhr`` pipeline as every
    other OBB source for output-contract consistency.
    """
    masks = result.masks
    if masks is None or masks.data is None or masks.data.shape[0] == 0:
        return _empty_obb_result(frame_idx)
    mask_tensor = masks.data
    boxes = result.boxes
    conf_all = boxes.conf if boxes is not None else None
    if conf_all is None or len(conf_all) == 0:
        return _empty_obb_result(frame_idx)
    boxes_orig = boxes.xyxy

    # Optimization: the downstream cap keeps only the top-`raw_detection_cap`
    # detections by confidence (see _apply_raw_detection_cap). Select that same
    # top-k HERE, before rotated_rect_from_masks -- whose cost is
    # O(N . num_angles . crop^2) -- so the kernel never processes rows the cap
    # would discard. Ordering mirrors _apply_raw_detection_cap exactly
    # (confidence descending) and the caller re-applies the cap afterwards, so
    # the final result is unchanged.
    if raw_detection_cap > 0 and int(conf_all.shape[0]) > raw_detection_cap:
        order = np.argsort(conf_all.detach().cpu().numpy())[::-1][:raw_detection_cap]
        keep = torch.as_tensor(
            np.ascontiguousarray(order),
            device=mask_tensor.device,
            dtype=torch.long,
        )
        mask_tensor = mask_tensor[keep]
        boxes_orig = boxes_orig[keep]
        conf_all = conf_all[keep]

    gain, pad_x, pad_y = letterbox_gain_pad(
        tuple(mask_tensor.shape[-2:]), tuple(result.orig_shape)
    )
    pad = torch.tensor(
        [pad_x, pad_y, pad_x, pad_y], device=boxes_orig.device, dtype=boxes_orig.dtype
    )
    boxes_mask_space = boxes_orig * gain + pad

    rect_mask_space = rotated_rect_from_masks(
        mask_tensor,
        boxes_mask_space,
        num_angles=num_angles,
        crop_size=crop_size,
        pad_ratio=pad_ratio,
        mask_threshold=mask_threshold,
    )
    cx_m, cy_m, w_m, h_m, angle_rad = rect_mask_space.unbind(-1)
    cx = ((cx_m - pad_x) / gain).cpu().numpy()
    cy = ((cy_m - pad_y) / gain).cpu().numpy()
    w_arr = (w_m / gain).cpu().numpy()
    h_arr = (h_m / gain).cpu().numpy()
    angle_arr = angle_rad.cpu().numpy()
    conf = conf_all.cpu().numpy()

    angles_fixed, sizes, aspect = _normalize_obb_geometry(w_arr, h_arr, angle_arr)
    mask_valid = _valid_detection_mask(cx, cy, w_arr, h_arr, angles_fixed, conf)
    if not mask_valid.all():
        dropped = int(mask_valid.size - int(mask_valid.sum()))
        if dropped > 0:
            logger.warning(
                "Dropping %d invalid segment-as-OBB detections (non-finite "
                "geometry or empty mask crop).",
                dropped,
            )
        cx, cy, w_arr, h_arr = (
            cx[mask_valid],
            cy[mask_valid],
            w_arr[mask_valid],
            h_arr[mask_valid],
        )
        conf, angles_fixed, sizes, aspect = (
            conf[mask_valid],
            angles_fixed[mask_valid],
            sizes[mask_valid],
            aspect[mask_valid],
        )
    n = int(len(conf))
    corners = _corners_from_xywhr(cx, cy, w_arr, h_arr, angles_fixed)
    return OBBResult(
        frame_idx=frame_idx,
        centroids=np.stack([cx, cy], axis=1).astype(np.float32),
        angles=angles_fixed,
        sizes=sizes,
        shapes=np.stack([sizes, aspect], axis=1).astype(np.float32),
        confidences=conf.astype(np.float32),
        corners=corners.astype(np.float32),
        detection_ids=OBBResult.make_detection_ids(frame_idx, n),
    )


def _extract_raw_tensors_from_masks(
    result: Any,
    frame_idx: int,
    device: str,
    raw_detection_cap: int = 0,
    *,
    num_angles: int = 24,
    crop_size: int = 64,
    pad_ratio: float = 0.15,
    mask_threshold: float = 0.5,
) -> _RawOBBTensors:
    """Keep segment-as-OBB tensors on the compute device -- no .cpu() call.

    Mirrors _extract_raw_tensors_from_boxes's contract: the gain/pad
    conversion is plain tensor arithmetic (not a sync), and
    rotated_rect_from_masks already returns a device tensor with no internal
    .cpu() calls, so this function never leaves the accelerator.
    normalize/corners/finite-value filtering is deferred to
    materialize_tensors().
    """
    masks = result.masks
    boxes = result.boxes
    conf_all = boxes.conf if boxes is not None else None
    if (
        masks is None
        or masks.data is None
        or masks.data.shape[0] == 0
        or conf_all is None
        or len(conf_all) == 0
    ):
        dev = torch.device(device)
        return _RawOBBTensors(
            frame_idx=frame_idx,
            xywhr=torch.zeros((0, 5), dtype=torch.float32, device=dev),
            corners=torch.zeros((0, 4, 2), dtype=torch.float32, device=dev),
            conf=torch.zeros(0, dtype=torch.float32, device=dev),
        )
    mask_tensor = masks.data
    boxes_orig = boxes.xyxy
    # Optimization mirroring _extract_obb_from_masks: pre-cap to the top-k
    # detections by confidence BEFORE the O(N . num_angles . crop^2) kernel, so
    # it never processes rows materialize_tensors()'s own cap would discard.
    # torch.topk keeps the selection fully on-device (no .cpu()/.item()) --
    # required by this raw fast path's zero-host-sync contract -- and returns
    # indices in descending-confidence order, matching _apply_raw_detection_cap.
    if raw_detection_cap > 0 and int(conf_all.shape[0]) > raw_detection_cap:
        keep = torch.topk(conf_all, raw_detection_cap).indices
        mask_tensor = mask_tensor[keep]
        boxes_orig = boxes_orig[keep]
        conf_all = conf_all[keep]
    gain, pad_x, pad_y = letterbox_gain_pad(
        tuple(mask_tensor.shape[-2:]), tuple(result.orig_shape)
    )
    pad = torch.tensor(
        [pad_x, pad_y, pad_x, pad_y], device=boxes_orig.device, dtype=boxes_orig.dtype
    )
    boxes_mask_space = boxes_orig * gain + pad
    rect_mask_space = rotated_rect_from_masks(
        mask_tensor,
        boxes_mask_space,
        num_angles=num_angles,
        crop_size=crop_size,
        pad_ratio=pad_ratio,
        mask_threshold=mask_threshold,
    )
    cx_m, cy_m, w_m, h_m, angle = rect_mask_space.unbind(-1)
    cx, cy = (cx_m - pad_x) / gain, (cy_m - pad_y) / gain
    w_arr, h_arr = w_m / gain, h_m / gain
    xywhr = torch.stack([cx, cy, w_arr, h_arr, angle], dim=1)
    # NaN rows from rotated_rect_from_masks (empty mask crops) are dropped
    # later by materialize_tensors()'s existing isfinite-based valid-mask
    # check -- no special-casing needed here.
    corners = torch.zeros(
        (xywhr.shape[0], 4, 2), dtype=torch.float32, device=xywhr.device
    )
    return _RawOBBTensors(
        frame_idx=frame_idx, xywhr=xywhr, corners=corners, conf=conf_all
    )


def _extract_raw_tensors_from_boxes(
    result: Any, frame_idx: int, fixed_angle_rad: float, device: str
) -> _RawOBBTensors:
    """Keep detect-as-OBB tensors on the compute device -- no .cpu() call.

    Mirrors _extract_raw_tensors's contract exactly: raw, unfiltered geometry.
    normalize/corners/finite-value filtering is deferred to
    materialize_tensors(), which already works generically for ANY
    (xywhr, conf) device-tensor pair regardless of detection source.
    """
    boxes = result.boxes
    if boxes is None or boxes.xyxy.shape[0] == 0:
        dev = torch.device(device)
        return _RawOBBTensors(
            frame_idx=frame_idx,
            xywhr=torch.zeros((0, 5), dtype=torch.float32, device=dev),
            corners=torch.zeros((0, 4, 2), dtype=torch.float32, device=dev),
            conf=torch.zeros(0, dtype=torch.float32, device=dev),
        )
    xyxy = boxes.xyxy  # (N, 4), stays on whatever device it already is
    cx = (xyxy[:, 0] + xyxy[:, 2]) / 2.0
    cy = (xyxy[:, 1] + xyxy[:, 3]) / 2.0
    w_arr = xyxy[:, 2] - xyxy[:, 0]
    h_arr = xyxy[:, 3] - xyxy[:, 1]
    angle = torch.full_like(cx, float(fixed_angle_rad))
    xywhr = torch.stack([cx, cy, w_arr, h_arr, angle], dim=1)
    # materialize_tensors() ignores raw.corners and rebuilds corners fresh
    # from xywhr (see its existing implementation) -- this field is a
    # placeholder, exactly like _extract_raw_tensors's own corners field is
    # for the "obb" fast path today.
    corners = torch.zeros(
        (xywhr.shape[0], 4, 2), dtype=torch.float32, device=xywhr.device
    )
    return _RawOBBTensors(
        frame_idx=frame_idx, xywhr=xywhr, corners=corners, conf=boxes.conf
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
    imgsz_override: int | None = None,
    task: str = "obb",
    batch_size: int = 1,
) -> Any:
    """Load the OBB executor for ``model_path`` under ``compute_runtime``.

    Thin delegator to :func:`load_obb_executor`:
      * cpu/mps/cuda → a plain ultralytics ``YOLO`` model (``.to()``-moved as
        before; CPU does not call ``.to()`` so CPU byte-parity is preserved).
      * tensorrt → a direct TensorRT executor (auto-exporting the ``.engine``
        from ``.pt`` on first load when ``auto_export``); coreml → the
        ``.mlpackage``. When no artifact exists and ``auto_export`` is False, a
        clear error is raised instead of silently running PyTorch (finding H4).

    ``imgsz_override``, when set, forces the ONNX/TRT export/load size instead
    of the checkpoint's own embedded default -- needed for the sequential-OBB
    stage-2 (crop) model, whose ``stage2_image_size`` config value may differ
    from the checkpoint's default (see :func:`load_obb_executor`). Ignored for
    the torch runtimes (cpu/mps/cuda), which take crops pre-resized by the
    caller and never re-export an artifact.

    ``task="detect"`` must be passed for the sequential pipeline's stage-1
    model under tensorrt/onnx (see :func:`load_obb_executor`) -- it is a plain
    detector, not an OBB model.

    ``batch_size`` is forwarded to :func:`load_obb_executor` and governs
    whether a TensorRT export uses a static batch=1 engine or a dynamic-batch
    engine (see Task 1) -- ignored for the torch runtimes and for coreml.

    For ``compute_runtime="tensorrt"`` (gpu_fast tier), if the TRT artifact is
    unavailable or the build fails, falls back to native ``"cuda"`` and logs a
    WARNING.  Never falls back to CPU — stays on GPU device.
    """
    try:
        return load_obb_executor(
            model_path,
            compute_runtime,
            auto_export=auto_export,
            max_det=max_det,
            imgsz_override=imgsz_override,
            task=task,
            batch_size=batch_size,
        )
    except (
        Exception
    ) as exc:  # best-effort GPU-Fast fallback (spec §3: never a hard crash)
        if str(compute_runtime) == "coreml":
            logger.warning(
                "GPU-Fast OBB CoreML load/build failed (%s); falling back to native MPS",
                exc,
            )
            return load_obb_executor(
                model_path,
                "mps",
                auto_export=auto_export,
                max_det=max_det,
            )
        if str(compute_runtime) != "tensorrt":
            raise
        logger.warning(
            "GPU-Fast OBB TensorRT load/build failed (%s); falling back to native CUDA",
            exc,
        )
        return load_obb_executor(
            model_path,
            "cuda",
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
