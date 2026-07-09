"""Public helpers for callers outside core/inference/.

Keep this surface minimal: each helper exists to support a specific kept consumer
that cannot directly depend on the internal stages module.

Correction 21: apply_detection_filter shim for optimizer.py and optimizer_workers.py
Correction 22: predict_pose_for_image helper and create_pose_backend_from_config
  shim for posekit/gui/workers.py.
  create_pose_backend_from_config re-exports from core/identity/pose/api.py
  while it exists; once that module is deleted the implementation will move
  here. (Task 8: build_runtime_config was deleted from pose/api.py — it had
  zero real callers left — so its shim here was removed too.)
"""

from __future__ import annotations

from .config import OBBConfig
from .result import OBBResult
from .stages.filtering import filter_detections

# Correction 22: stable re-export so posekit/gui/workers.py does not need to
# import from the soon-to-be-deleted core/identity/pose/api module.
try:
    from hydra_suite.core.identity.pose.api import (  # noqa: F401
        create_pose_backend_from_config,
    )
except ImportError:
    create_pose_backend_from_config = None  # type: ignore[assignment]


def apply_detection_filter(raw: OBBResult, config: OBBConfig) -> OBBResult:
    """Filter raw OBB detections using the same logic the runner uses internally.

    Used by core/tracking/optimization/optimizer.py and optimizer_workers.py to score
    parameter configurations against cached detections. Pure function — no I/O,
    no model loading.
    """
    return filter_detections(raw, config, roi_mask=None)


def predict_pose_for_image(image, pose_config) -> "PoseResult":  # noqa: F821
    """One-shot pose prediction on a single image, used by PoseKit labeling UI.

    Loads a pose model, runs inference once, and discards the model. NOT for
    batch use — call InferenceRunner.run_realtime if you need persistent state.

    Correction 22: replaces the lazy import of build_runtime_config /
    create_pose_backend_from_config from (eventually) deleted
    core/identity/pose/api.py.
    """
    from .config import InferenceConfig, OBBConfig, OBBDirectConfig
    from .runner import _load_pose_model
    from .runtime import RuntimeContext
    from .stages.pose import run_pose

    compute_runtime = "cpu"
    if pose_config is not None:
        if hasattr(pose_config, "yolo") and pose_config.yolo is not None:
            compute_runtime = getattr(pose_config.yolo, "compute_runtime", "cpu")
        elif hasattr(pose_config, "sleap") and pose_config.sleap is not None:
            compute_runtime = getattr(pose_config.sleap, "compute_runtime", "cpu")

    # Build a minimal InferenceConfig so RuntimeContext.from_config() works.
    _min_cfg = InferenceConfig(
        obb=OBBConfig(
            mode="direct",
            direct=OBBDirectConfig(
                model_path="",
                compute_runtime=compute_runtime,
            ),
        ),
        pose=pose_config,
    )
    try:
        runtime = RuntimeContext.from_config(_min_cfg)
    except Exception:
        # Fall back to CPU if device unavailable
        _min_cfg.obb.direct.compute_runtime = "cpu"

        runtime = RuntimeContext(
            cuda_mode=False,
            device="cpu",
            use_nvdec=False,
            default_runtime="cpu",
            tensor_on_cuda=False,
        )

    model = _load_pose_model(pose_config, runtime)
    try:
        # Single-frame, full-image: synthetic OBBResult covering the whole image.
        import numpy as np

        h, w = image.shape[:2] if hasattr(image, "shape") else (1, 1)
        synthetic_obb = OBBResult(
            frame_idx=0,
            centroids=np.array([[w / 2, h / 2]], dtype=np.float32),
            angles=np.zeros(1, dtype=np.float32),
            sizes=np.array([float(w * h)], dtype=np.float32),
            shapes=np.array(
                [[float(w * h), float(w) / float(h + 1e-6)]], dtype=np.float32
            ),
            confidences=np.ones(1, dtype=np.float32),
            corners=np.array([[[0, 0], [w, 0], [w, h], [0, h]]], dtype=np.float32),
            detection_ids=OBBResult.make_detection_ids(0, 1),
        )
        results = run_pose([image], synthetic_obb, model, pose_config, runtime)
        return results[0] if results else None
    finally:
        del model
