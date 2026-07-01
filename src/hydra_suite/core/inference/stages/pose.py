from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from ..config import PoseConfig
from ..result import CropBatch, OBBResult, PoseResult
from ..runtime import RuntimeContext, runtime_to_compute_runtime

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

    # Derive compute_runtime from RuntimeContext (reflects runtime_tier).
    # Per-stage compute_runtime fields are deprecated in favor of runtime_tier;
    # kept in place for serialization only.
    compute_runtime = runtime_to_compute_runtime(runtime)

    if config.backend == "yolo":
        assert config.yolo is not None
        from hydra_suite.core.identity.pose.backends.yolo import YoloNativeBackend

        device = (
            "cuda:0"
            if compute_runtime in ("cuda", "onnx_cuda", "tensorrt")
            else ("mps" if compute_runtime in ("mps", "coreml") else "cpu")
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
    # Debug/A-B override: force the SLEAP runtime flavor independent of the tier
    # (e.g. HYDRA_SLEAP_FLAVOR=native|onnx_cuda|tensorrt|onnx_cpu). Lets us run
    # the full pipeline with identical crops across flavors to verify the
    # exported models reproduce native SLEAP keypoints. Unset in normal use.
    import os as _os

    _flavor_override = _os.environ.get("HYDRA_SLEAP_FLAVOR", "").strip().lower()
    if _flavor_override:
        runtime_flavor = _flavor_override
        device = "cpu" if _flavor_override == "onnx_cpu" else "cuda"
    elif compute_runtime in ("cuda", "onnx_cuda"):
        runtime_flavor = "onnx_cuda"
        device = "cuda"
    elif compute_runtime in ("mps", "coreml"):
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
    aspect_ratio: float = _CANONICAL_ASPECT_RATIO,
    margin: float = _CANONICAL_MARGIN,
) -> "dict[int, PoseResult]":
    """Run pose estimation over a CropBatch; return one PoseResult per frame.

    Runs the backend ONCE over all crops in batch (cross-frame perf win), then
    splits results per frame via batch.select_frame. Uses batch.native_sizes to
    undo per-crop padding exactly as run_pose does. Delegates per-detection
    assembly to _assemble_pose_result (DRY with run_pose).

    Calls predict_batch_cuda when batch.crops.is_cuda and backend supports it.
    """
    import cv2

    from hydra_suite.core.canonicalization.crop import (
        compute_alignment_affine,
        compute_native_crop_dimensions,
        invert_keypoints,
    )

    pad = max(0.0, float(margin) - 1.0)
    n_total = batch.crops.shape[0]
    on_cuda = batch.crops.is_cuda and hasattr(model.backend, "predict_batch_cuda")

    np_crops: list[np.ndarray] = []
    # CUDA: must slice to native extent (see run_pose docstring) — validate on mehek.
    # Mirror the CPU native-slice but keep tensors on-device (C×H×W) so the model
    # sees the ant filling the frame, consistent with the native-computed m_inv.
    cuda_crops: list[Any] = []
    affines_all: list[np.ndarray | None] = []

    for i in range(n_total):
        # Native extent for this crop (padding is bottom/right zeros).
        if i < len(batch.native_sizes):
            native_h, native_w = int(batch.native_sizes[i, 0]), int(
                batch.native_sizes[i, 1]
            )
        else:
            native_h, native_w = (
                int(batch.crops.shape[2]),
                int(batch.crops.shape[3]),
            )

        if on_cuda:
            # Slice the device tensor (C, H, W) to its native extent — no
            # host round-trip. predict_batch_cuda accepts a list of C×H×W
            # CUDA tensors (see SleapExportedBackend.predict_batch_cuda).
            cuda_crops.append(batch.crops[i, :, :native_h, :native_w])
        else:
            hwc = batch.crops[i].permute(1, 2, 0).cpu().numpy()
            hwc = hwc[:native_h, :native_w]

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
                    cw, ch = compute_native_crop_dimensions(corners, aspect_ratio, pad)
                    m_align, _ = compute_alignment_affine(
                        corners, int(cw), int(ch), pad
                    )
                    m_inv = cv2.invertAffineTransform(m_align)
                except Exception:
                    m_inv = None

        if not on_cuda:
            np_crops.append(np.ascontiguousarray(hwc))
        affines_all.append(m_inv)

    if on_cuda:
        # Feed NATIVE-sized device crops (not the window-padded batch.crops) so
        # the model input matches the native-extent m_inv back-projection.
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
