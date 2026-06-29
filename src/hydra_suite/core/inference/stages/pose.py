from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from ..config import PoseConfig
from ..result import OBBResult, PoseResult
from ..runtime import RuntimeContext

logger = logging.getLogger(__name__)

# Crop geometry used to build the shared canonical crops (mirrors the runner's
# headtail aspect/margin defaults). Used to recover each detection's native crop
# extent and the affine that maps crop coords back to the frame.
_CANONICAL_ASPECT_RATIO = 2.0
_CANONICAL_MARGIN = 1.3


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


def load_pose_model(config: PoseConfig, runtime: RuntimeContext) -> PoseModel:
    from hydra_suite.core.identity.pose.utils import load_skeleton_from_json

    # Reuse the canonical skeleton loader so both legacy and new pipelines accept
    # the same JSON formats ("keypoint_names"/"skeleton_edges" and the legacy
    # "keypoints"/"edges" aliases) and resolve/validate the path identically.
    if config.skeleton_file:
        names, edges = load_skeleton_from_json(config.skeleton_file)
        keypoint_names: list[str] = list(names)
        skeleton_edges = [tuple(e) for e in edges]
    else:
        keypoint_names = []
        skeleton_edges = []
    n_kpts = len(keypoint_names)

    if config.backend == "yolo":
        assert config.yolo is not None
        from hydra_suite.core.identity.pose.backends.yolo import YoloNativeBackend

        device = (
            "cuda:0"
            if config.yolo.compute_runtime in ("cuda", "onnx_cuda", "tensorrt")
            else ("mps" if config.yolo.compute_runtime == "mps" else "cpu")
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

    assert config.sleap is not None
    from hydra_suite.core.identity.pose.api import create_pose_backend_from_config
    from hydra_suite.core.identity.pose.types import PoseRuntimeConfig

    sleap_cfg = config.sleap
    compute_runtime = str(sleap_cfg.compute_runtime or "cpu").lower()
    if compute_runtime in ("cuda", "onnx_cuda"):
        runtime_flavor = "onnx_cuda"
        device = "cuda"
    elif compute_runtime == "mps":
        # On Apple Silicon, ONNX Runtime has no MPS provider and its CoreML
        # provider fails on SLEAP's UNet (dynamic-shape "ios18.max_pool" /
        # "unbounded dimension" errors). Use SLEAP's native TensorFlow runtime
        # instead (Metal-accelerated via the sleap conda env) rather than CoreML.
        runtime_flavor = "native"
        device = "mps"
    elif compute_runtime == "tensorrt":
        runtime_flavor = "tensorrt"
        device = "cuda"
    else:
        runtime_flavor = "onnx_cpu"
        device = "cpu"

    runtime_cfg = PoseRuntimeConfig(
        backend_family="sleap",
        runtime_flavor=runtime_flavor,
        device=device,
        batch_size=int(sleap_cfg.batch_size),
        model_path=str(sleap_cfg.model_path),
        out_root=".",
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
    aspect_ratio: float = _CANONICAL_ASPECT_RATIO,
    margin: float = _CANONICAL_MARGIN,
) -> PoseResult:
    """Run pose estimation on canonical crops. Returns (D, K, 3) keypoints + valid_mask.

    Two corrections vs the naive path, both verified against the legacy pipeline:

    1. The shared ``crops`` tensor is padded to a single per-batch max size so it
       can stack as one tensor (head-tail / CNN need that). Feeding those padded
       crops to SLEAP put each ant in a corner of an oversized canvas; the
       sizematcher then shrank it and the single-instance model produced
       scattered / missing keypoints. We recover each detection's NATIVE crop
       (padding is bottom/right zeros) so SLEAP sees the ant filling the frame.

    2. Keypoints are returned in IMAGE coordinates (not crop coordinates) by
       inverting the per-detection canonical affine — matching legacy, whose
       pose cache stores image-space keypoints. Downstream heading/identity then
       work in the global frame.
    """
    import cv2

    from hydra_suite.core.canonicalization.crop import (
        compute_alignment_affine,
        compute_native_crop_dimensions,
        invert_keypoints,
    )

    n = obb_result.num_detections
    empty = PoseResult(
        keypoints=np.zeros((0, model.n_keypoints, 3), dtype=np.float32),
        valid_mask=np.zeros(0, dtype=bool),
    )
    if crops.shape[0] == 0 or n == 0:
        return empty

    pad = max(0.0, float(margin) - 1.0)
    np_crops: list[np.ndarray] = []
    affines: list[np.ndarray | None] = []
    for i in range(crops.shape[0]):
        hwc = crops[i].permute(1, 2, 0).cpu().numpy()
        corners = obb_result.corners[i] if i < n else None
        m_inv = None
        if corners is not None:
            try:
                cw, ch = compute_native_crop_dimensions(corners, aspect_ratio, pad)
                # Recover the native crop region (padding is bottom/right).
                hwc = hwc[: int(ch), : int(cw)]
                m_align, _ = compute_alignment_affine(corners, int(cw), int(ch), pad)
                m_inv = cv2.invertAffineTransform(m_align)
            except Exception:
                m_inv = None
        np_crops.append(np.ascontiguousarray(hwc))
        affines.append(m_inv)

    raw_results = model.backend.predict_batch(np_crops)

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
        # Map crop-space (x, y) -> image coordinates; keep confidence column.
        m_inv = affines[i] if i < len(affines) else None
        if m_inv is not None:
            kpts[:, :2] = invert_keypoints(kpts[:, :2].astype(np.float32), m_inv)
        kpts_out[i, :k] = kpts
        n_confident = int(np.sum(kpts[:, 2] >= min_kpt_conf))
        valid[i] = n_confident >= min_valid

    return PoseResult(keypoints=kpts_out, valid_mask=valid)
