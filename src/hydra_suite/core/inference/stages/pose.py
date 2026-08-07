from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from hydra_suite.core.canonicalization.fit import (
    apply_fit,
    fit_affine,
    fit_to_model_input,
)
from hydra_suite.core.canonicalization.geometry import (
    CanonicalGeometry,
    canonical_affine,
)
from hydra_suite.core.individual.pose.crop_dtype import to_uint8_image

from ..config import PoseConfig
from ..result import CropBatch, OBBResult, PoseResult
from ..runtime import RuntimeContext, resolved_backend_for

logger = logging.getLogger(__name__)


def model_input_wh(model: "PoseModel", geometry: CanonicalGeometry) -> tuple[int, int]:
    """The pose backend's fixed (W, H) input, or ``geometry``'s own canvas.

    Backends that perform their own Layer-2 fit inside their preprocessing
    (``does_own_letterbox``, e.g. ViTPose's box2cs/top_down_affine) get an
    IDENTITY fit here: canvas -> canvas, scale 1. Doing the fit again at this
    layer (to ``preferred_input_wh``) would be a second, redundant resample
    that diverges from what the backend was trained on.

    Backends with a true non-square input that do NOT do their own letterbox
    (e.g. SLEAP-exported fixed HxW) expose ``preferred_input_wh`` and that is
    used directly -- no collapsing to a square, which would otherwise force a
    second, redundant letterbox inside the backend's own preprocessing.

    Backends with no fixed input size (fully convolutional, e.g. native SLEAP's
    ``preferred_input_size == 0``) get an identity fit: feed the canonical crop
    unchanged and let the backend's own preprocessing handle it, matching the
    legacy "native-extent crop, backend resizes" behaviour for those backends.
    """
    if getattr(model.backend, "does_own_letterbox", False):
        return geometry.canvas_wh
    wh = getattr(model.backend, "preferred_input_wh", None)
    if wh is not None:
        try:
            w, h = int(wh[0]), int(wh[1])
        except (TypeError, ValueError, IndexError):
            w, h = 0, 0
        if w > 0 and h > 0:
            return (w, h)
    try:
        dim = int(getattr(model.backend, "preferred_input_size", 0) or 0)
    except (TypeError, ValueError):
        dim = 0
    if dim > 0:
        return (dim, dim)
    return geometry.canvas_wh


def compose_affine(m2: np.ndarray, m1: np.ndarray) -> np.ndarray:
    """Compose two 2x3 affines: result = m2 . m1 (apply m1 first, then m2)."""
    a = np.eye(3, dtype=np.float64)
    a[:2, :] = np.asarray(m2, dtype=np.float64)
    b = np.eye(3, dtype=np.float64)
    b[:2, :] = np.asarray(m1, dtype=np.float64)
    return (a @ b)[:2, :].astype(np.float64)


def _warmup_backend(backend: Any) -> None:
    """Start the backend's persistent service / warm caches once at load time.

    Critical for the SLEAP service backend: without warmup, the per-frame
    predict_batch falls back to a temp-file sleap-track subprocess that reloads
    the model on every call (~seconds/frame). Legacy warms the pose backend the
    same way (see core/tracking/worker.py). No-op for backends without warmup().
    """
    warm = getattr(backend, "warmup", None)
    if not callable(warm):
        logger.info(
            "Pose backend %s has no warmup(); skipping.", type(backend).__name__
        )
        return
    logger.info("Warming up pose backend: %s", type(backend).__name__)
    try:
        warm()
    except Exception:
        logger.warning("Pose backend warmup FAILED (non-fatal)", exc_info=True)
        return
    # For the SLEAP service backend, log whether the warm in-memory (shared-memory)
    # transport is enabled. If False, per-frame predict falls back to a temp-file
    # sleap-track subprocess (model reload each call → ~seconds/frame).
    if type(backend).__name__ == "SleapServiceBackend":
        logger.info(
            "Pose backend warmed: SLEAP service ready; crops stream via shared "
            "memory to the warm in-process predictor (no per-frame CLI reload)."
        )


@dataclass
class PoseModel:
    backend: Any  # YoloNativeBackend or SleapExportedBackend
    n_keypoints: int
    keypoint_names: list[str]

    def close(self) -> None:
        pass


def load_pose_model(
    config: PoseConfig,
    runtime: RuntimeContext,
    *,
    keypoint_names: "list[str] | None" = None,
    skeleton_edges: "list | None" = None,
    out_root: str = ".",
    exported_model_path: str = "",
) -> PoseModel:
    from hydra_suite.core.individual.pose.utils import load_skeleton_from_json

    # Reuse the canonical skeleton loader so both legacy and new pipelines accept
    # the same JSON formats ("keypoint_names"/"skeleton_edges" and the legacy
    # "keypoints"/"edges" aliases) and resolve/validate the path identically.
    #
    # GUI pose workers already hold the resolved keypoint names/edges (from the
    # project or a loaded skeleton) and pass them directly via the shared shim;
    # batch/config callers instead provide only config.skeleton_file and we
    # derive them here. An explicit override wins so PoseKit -- which has no
    # skeleton_file path -- can still supply the names SLEAP requires.
    if keypoint_names is not None:
        keypoint_names = list(keypoint_names)
        skeleton_edges = [tuple(e) for e in (skeleton_edges or [])]
    elif config.skeleton_file:
        names, edges = load_skeleton_from_json(config.skeleton_file)
        keypoint_names = list(names)
        skeleton_edges = [tuple(e) for e in edges]
    else:
        keypoint_names = []
        skeleton_edges = []
    n_kpts = len(keypoint_names)

    # Derive the resolved backend from the RuntimeContext (reflects runtime_tier
    # via the Gen-2 resolver). Per-stage compute_runtime fields no longer exist;
    # runtime_tier is the sole source of truth.
    resolved = resolved_backend_for(runtime)

    if config.backend == "yolo":
        assert config.yolo is not None
        from hydra_suite.core.individual.pose.backends.yolo import YoloNativeBackend

        # The resolver never emits an ONNX-on-CUDA device, so device=="cuda"
        # covers native torch CUDA and TensorRT alike; coreml resolves to mps.
        device = (
            "cuda:0"
            if resolved.device == "cuda"
            else ("mps" if resolved.device == "mps" else "cpu")
        )
        backend = YoloNativeBackend(
            model_path=config.yolo.model_path,
            device=device,
            min_valid_conf=config.min_keypoint_confidence,
            keypoint_names=keypoint_names if keypoint_names else None,
            conf=config.yolo.confidence_threshold,
            iou=config.yolo.iou_threshold,
            max_det=config.yolo.max_detections_per_crop,
            batch_size=config.yolo.batch_size,
        )
        _warmup_backend(backend)
        return PoseModel(
            backend=backend, n_keypoints=n_kpts, keypoint_names=keypoint_names
        )

    if config.backend == "vitpose":
        assert config.vitpose is not None
        from hydra_suite.core.individual.pose.api import create_pose_backend_from_config
        from hydra_suite.core.individual.pose.types import PoseRuntimeConfig

        if resolved.backend == "tensorrt":
            vp_flavor, vp_device = "tensorrt", "cuda"
        elif resolved.backend == "coreml":
            vp_flavor, vp_device = "coreml", "mps"
        else:  # torch
            vp_flavor, vp_device = "native", resolved.device
        runtime_cfg = PoseRuntimeConfig(
            backend_family="vitpose",
            runtime_flavor=vp_flavor,
            device=vp_device,
            model_path=str(config.vitpose.model_path),
            min_valid_conf=float(config.min_keypoint_confidence),
            keypoint_names=list(keypoint_names),
            vitpose_batch=int(config.vitpose.batch_size),
            vitpose_variant=str(config.vitpose.variant),
            vitpose_num_keypoints=int(config.vitpose.num_keypoints),
            vitpose_auto_export=bool(config.vitpose.auto_export),
        )
        backend = create_pose_backend_from_config(runtime_cfg)
        _warmup_backend(backend)
        return PoseModel(
            backend=backend, n_keypoints=n_kpts, keypoint_names=keypoint_names
        )

    assert config.sleap is not None
    from hydra_suite.core.individual.pose.api import create_pose_backend_from_config
    from hydra_suite.core.individual.pose.types import PoseRuntimeConfig

    sleap_cfg = config.sleap
    # Debug/A-B override: force the SLEAP runtime flavor independent of the tier
    # (e.g. HYDRA_SLEAP_FLAVOR=native|onnx_cuda|tensorrt|onnx_cpu). Lets us run
    # the full pipeline with identical crops across flavors to verify the
    # exported models reproduce native SLEAP keypoints. Unset in normal use.
    import os as _os

    _flavor_override = _os.environ.get("HYDRA_SLEAP_FLAVOR", "").strip().lower()
    if _flavor_override:
        runtime_flavor = _flavor_override
        device = "cpu" if _flavor_override == "onnx_cpu" else "cuda"
    elif resolved.backend == "tensorrt":
        # gpu_fast CUDA tier: exported TensorRT SLEAP backend.
        runtime_flavor = "tensorrt"
        device = "cuda"
    elif resolved.device == "cuda":
        # gpu tier: native torch CUDA everywhere else in the pipeline, so SLEAP
        # runs its native (non-exported) model on CUDA too, via the service
        # backend, instead of silently using the gpu_fast (onnx) path.
        runtime_flavor = "native"
        device = "cuda"
    elif resolved.device == "mps":
        # On Apple Silicon, ONNX Runtime has no MPS provider and its CoreML
        # provider fails on SLEAP's UNet (dynamic-shape "ios18.max_pool" /
        # "unbounded dimension" errors). Use SLEAP's native TensorFlow runtime
        # instead (Metal-accelerated via the sleap conda env) rather than CoreML.
        # Covers both the gpu (torch/mps) and gpu_fast (coreml/mps) resolutions.
        runtime_flavor = "native"
        device = "mps"
    else:
        # cpu tier: SLEAP runs its native (non-exported) sleap-nn model on
        # torch-CPU via the service backend -- consistent with the cuda/mps
        # native path, no ONNX export. Slower than exported ONNX-CPU, but keeps
        # a single service backend across tiers (pose runtime golden rule).
        runtime_flavor = "native"
        device = "cpu"

    runtime_cfg = PoseRuntimeConfig(
        backend_family="sleap",
        runtime_flavor=runtime_flavor,
        device=device,
        batch_size=int(sleap_cfg.batch_size),
        model_path=str(sleap_cfg.model_path),
        exported_model_path=str(exported_model_path or ""),
        out_root=str(out_root or "."),
        min_valid_conf=float(config.min_keypoint_confidence),
        sleap_env=str(sleap_cfg.conda_env),
        sleap_device=device,
        sleap_batch=int(sleap_cfg.batch_size),
        sleap_max_instances=int(sleap_cfg.max_instances),
        keypoint_names=list(keypoint_names),
        skeleton_edges=skeleton_edges,
    )
    backend = create_pose_backend_from_config(runtime_cfg)
    _warmup_backend(backend)
    return PoseModel(backend=backend, n_keypoints=n_kpts, keypoint_names=keypoint_names)


def run_pose(
    crops: torch.Tensor,
    obb_result: OBBResult,
    model: PoseModel,
    config: PoseConfig,
    runtime: RuntimeContext,
    geometry: CanonicalGeometry,
) -> PoseResult:
    """Run pose estimation on canonical crops. Returns (D, K, 3) keypoints + valid_mask.

    ``crops`` is on ``geometry``'s fixed canvas (Layer 1: rotation + translation
    only). Each crop is fit to the model's expected input via Layer 2
    (``fit_to_model_input`` / ``apply_fit``); the inverse affine that maps
    predicted keypoints back to image coordinates is the composite of both
    transforms: ``m_total = fit_affine(fit) . m_align``. With a fixed canvas
    there is no per-detection native extent to slice back to (unlike the old
    batch-max-padded crops this replaces) -- every crop is already the shared
    canvas size, so it is used whole.

    Keypoints are returned in IMAGE coordinates (not crop coordinates) by
    inverting ``m_total`` — matching legacy, whose pose cache stores
    image-space keypoints. Downstream heading/identity then work in the global
    frame.
    """
    import cv2

    from hydra_suite.core.canonicalization.crop import invert_keypoints

    n = obb_result.num_detections
    empty = PoseResult(
        keypoints=np.zeros((0, model.n_keypoints, 3), dtype=np.float32),
        valid_mask=np.zeros(0, dtype=bool),
    )
    if crops.shape[0] == 0 or n == 0:
        return empty

    model_wh = model_input_wh(model, geometry)
    fit = fit_to_model_input(geometry.canvas_wh, model_wh)
    fit_m = fit_affine(fit)

    np_crops: list[np.ndarray] = []
    affines: list[np.ndarray | None] = []
    for i in range(crops.shape[0]):
        hwc = crops[i].permute(1, 2, 0).cpu().numpy()
        hwc_u8 = to_uint8_image(hwc)
        corners = obb_result.corners[i] if i < n else None
        m_inv = None
        if corners is not None:
            try:
                m_align, _theta, _clipped = canonical_affine(corners, geometry)
                m_total = compose_affine(fit_m, m_align)
                m_inv = cv2.invertAffineTransform(m_total)
            except Exception:
                m_inv = None
        try:
            hwc_u8 = apply_fit(hwc_u8, fit)
        except Exception:
            pass
        np_crops.append(np.ascontiguousarray(hwc_u8))
        affines.append(m_inv)

    raw_results = model.backend.predict_batch(np_crops)

    return _assemble_pose_result(
        raw_results, affines, n, model, config, invert_keypoints
    )


def _assemble_pose_result(
    raw_results: list,
    affines: "list[np.ndarray | None]",
    n: int,
    model: "PoseModel",
    config: "PoseConfig",
    invert_keypoints_fn: "Any",
) -> PoseResult:
    """Assemble PoseResult from raw backend predictions.

    Shared by run_pose and run_pose_batch so per-detection logic is never duplicated.
    affines[i] is the inverse affine mapping crop coords back to image coords (or None).
    """
    kpts_out = np.zeros((n, model.n_keypoints, 3), dtype=np.float32)
    valid = np.zeros(n, dtype=bool)
    min_kpt_conf = config.min_keypoint_confidence
    min_valid = config.min_valid_keypoints

    for i, r in enumerate(raw_results):
        if i >= n:
            break
        # Both YOLO and SLEAP backends return the canonical pose.types.PoseResult,
        # whose `.keypoints` is already a numpy (K, 3) array (x, y, conf) or None.
        kpts = getattr(r, "keypoints", None)
        if kpts is None:
            continue
        kpts = np.asarray(kpts, dtype=np.float32)
        if kpts.ndim == 3:  # tolerate a leading (1, K, 3) batch axis
            if kpts.shape[0] == 0:
                continue
            kpts = kpts[0]
        if kpts.ndim != 2 or kpts.shape[0] == 0:
            continue
        k = min(kpts.shape[0], model.n_keypoints)
        kpts = kpts[:k].copy()
        m_inv = affines[i] if i < len(affines) else None
        if m_inv is not None:
            kpts[:, :2] = invert_keypoints_fn(kpts[:, :2].astype(np.float32), m_inv)
        kpts_out[i, :k] = kpts
        n_confident = int(np.sum(kpts[:, 2] >= min_kpt_conf))
        valid[i] = n_confident >= min_valid

    return PoseResult(keypoints=kpts_out, valid_mask=valid)


def run_pose_batch(
    batch: CropBatch,
    model: PoseModel,
    config: PoseConfig,
    runtime: RuntimeContext,
    geometry: CanonicalGeometry,
) -> "dict[int, PoseResult]":
    """Run pose estimation over a CropBatch; return one PoseResult per frame.

    ``batch.crops`` is already on ``geometry``'s fixed canvas -- unlike the old
    batch-max-padded crops, there is no per-detection native extent to slice
    back to, so every crop is used whole (CPU path fit to the model input via
    Layer 2, same as run_pose). Runs the backend ONCE over all crops in batch
    (cross-frame perf win), then splits results per frame via
    batch.select_frame. Delegates per-detection assembly to
    _assemble_pose_result (DRY with run_pose).

    Calls predict_batch_cuda when batch.crops.is_cuda and backend supports it;
    that branch feeds the canonical-canvas crop straight through (no Layer 2
    fit), matching the no-host-round-trip on-device crop paths elsewhere.

    For backends that own their Layer-2 fit (``does_own_letterbox``, e.g.
    ViTPose), ``model_wh == geometry.canvas_wh`` and ``fit_m`` is the identity
    affine, so composing it into back-projection on both branches is a no-op
    -- it is done unconditionally for those backends so the CUDA and non-CUDA
    paths compute keypoints via the exact same formula. Backends that do NOT
    own their letterbox (SLEAP-exported, YOLO) keep the pre-existing
    behaviour: the CUDA branch inverts ``m_align`` alone (their own
    ``predict_batch_cuda`` resizes internally in that frame), unchanged.
    """
    import cv2

    from hydra_suite.core.canonicalization.crop import invert_keypoints

    n_total = batch.crops.shape[0]
    on_cuda = batch.crops.is_cuda and hasattr(model.backend, "predict_batch_cuda")
    does_own_letterbox = getattr(model.backend, "does_own_letterbox", False)

    model_wh = model_input_wh(model, geometry)
    fit = fit_to_model_input(geometry.canvas_wh, model_wh)
    fit_m = fit_affine(fit)

    np_crops: list[np.ndarray] = []
    cuda_crops: list[Any] = []
    affines_all: list[np.ndarray | None] = []

    for i in range(n_total):
        if on_cuda:
            cuda_crops.append(batch.crops[i])
        else:
            hwc = batch.crops[i].permute(1, 2, 0).cpu().numpy()

        # Compute inverse affine for this crop using its OBB corners
        frame_idx = int(batch.frame_index[i])
        obb = batch.obb_by_frame.get(frame_idx)
        m_inv = None
        if obb is not None:
            rows = batch.select_frame(frame_idx)
            # local index of crop i within its frame
            local_idx = int(np.searchsorted(rows, i))
            # rows is sorted from select_frame; rows[local_idx] == i confirms exact hit
            if (
                local_idx < len(rows)
                and rows[local_idx] == i
                and local_idx < obb.num_detections
            ):
                corners = obb.corners[local_idx]
                try:
                    m_align, _theta, _clipped = canonical_affine(corners, geometry)
                    if on_cuda and not does_own_letterbox:
                        m_inv = cv2.invertAffineTransform(m_align)
                    else:
                        m_total = compose_affine(fit_m, m_align)
                        m_inv = cv2.invertAffineTransform(m_total)
                except Exception:
                    m_inv = None

        if not on_cuda:
            hwc_u8 = to_uint8_image(hwc)
            try:
                hwc_u8 = apply_fit(hwc_u8, fit)
            except Exception:
                pass
            np_crops.append(np.ascontiguousarray(hwc_u8))
        affines_all.append(m_inv)

    if on_cuda:
        raw_results = model.backend.predict_batch_cuda(cuda_crops)
    else:
        raw_results = model.backend.predict_batch(np_crops) if np_crops else []

    results: dict[int, PoseResult] = {}
    prob_offset = 0
    for frame_idx in sorted(batch.obb_by_frame):
        obb = batch.obb_by_frame[frame_idx]
        rows = batch.select_frame(frame_idx)
        n = len(rows)
        if n == 0:
            results[frame_idx] = PoseResult(
                keypoints=np.zeros(
                    (obb.num_detections, model.n_keypoints, 3), dtype=np.float32
                ),
                valid_mask=np.zeros(obb.num_detections, dtype=bool),
            )
            continue
        frame_raw = raw_results[prob_offset : prob_offset + n]
        frame_affines = affines_all[prob_offset : prob_offset + n]
        prob_offset += n
        results[frame_idx] = _assemble_pose_result(
            frame_raw, frame_affines, n, model, config, invert_keypoints
        )
    return results
