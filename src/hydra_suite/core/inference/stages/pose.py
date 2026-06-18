from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from ..config import PoseConfig
from ..result import OBBResult, PoseResult
from ..runtime import RuntimeContext


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
        runtime_flavor = "onnx_mps"
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
    return PoseModel(backend=backend, n_keypoints=n_kpts, keypoint_names=keypoint_names)


def run_pose(
    crops: torch.Tensor,
    obb_result: OBBResult,
    model: PoseModel,
    config: PoseConfig,
    runtime: RuntimeContext,
) -> PoseResult:
    """Run pose estimation on canonical crops. Returns (D, K, 3) keypoints + valid_mask.

    Keypoints are in CROP coordinates (the canonical crop space). Mapping back to
    image coordinates is the responsibility of the caller (it has the OBB+affine).
    """
    n = obb_result.num_detections
    empty = PoseResult(
        keypoints=np.zeros((0, model.n_keypoints, 3), dtype=np.float32),
        valid_mask=np.zeros(0, dtype=bool),
    )
    if crops.shape[0] == 0 or n == 0:
        return empty

    np_crops = [crops[i].permute(1, 2, 0).cpu().numpy() for i in range(crops.shape[0])]
    raw_results = model.backend.predict_batch(np_crops)

    kpts_out = np.zeros((n, model.n_keypoints, 3), dtype=np.float32)
    valid = np.zeros(n, dtype=bool)

    min_kpt_conf = config.min_keypoint_confidence
    min_valid = config.min_valid_keypoints

    for i, r in enumerate(raw_results):
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
        kpts_out[i, :k] = kpts[:k]
        n_confident = int(np.sum(kpts[:k, 2] >= min_kpt_conf))
        valid[i] = n_confident >= min_valid

    return PoseResult(keypoints=kpts_out, valid_mask=valid)
