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


def load_pose_backend(
    *,
    backend_family,
    model_path,
    compute_runtime,
    keypoint_names=None,
    confidence_threshold=1e-4,
    batch_size=64,
    min_valid_confidence=0.2,
):
    """Build a pose backend (with predict_batch) via the canonical stages/pose loader.

    Single source of the tier->backend golden rule; returns the backend, not the
    PoseModel wrapper. GUI pose workers should migrate onto this instead of
    hand-rolling backend construction so CPU/GPU/GPU-Fast tiers all resolve
    through `stages.pose.load_pose_model`.
    """
    from .config import (
        InferenceConfig,
        OBBConfig,
        OBBDirectConfig,
        PoseConfig,
        PoseSLEAPConfig,
        PoseYOLOConfig,
        migrate_runtime_to_tier,
    )
    from .runtime import RuntimeContext
    from .stages.pose import load_pose_model

    family = (backend_family or "").strip().lower()
    if family == "yolo":
        pose_cfg = PoseConfig(
            backend="yolo",
            yolo=PoseYOLOConfig(
                model_path=model_path,
                compute_runtime=compute_runtime,
                confidence_threshold=confidence_threshold,
                batch_size=batch_size,
            ),
            min_keypoint_confidence=min_valid_confidence,
        )
    else:
        pose_cfg = PoseConfig(
            backend="sleap",
            sleap=PoseSLEAPConfig(
                model_path=model_path,
                compute_runtime=compute_runtime,
                batch_size=batch_size,
            ),
            min_keypoint_confidence=min_valid_confidence,
        )

    # RuntimeContext.from_config derives its tier from cfg.runtime_tier, NOT
    # from the deprecated per-stage OBBDirectConfig.compute_runtime (see
    # runtime.py's from_config / runtime_to_compute_runtime). Without this,
    # the minimal InferenceConfig keeps the default "gpu" runtime_tier and
    # the requested compute_runtime is silently ignored.
    _min_cfg = InferenceConfig(
        obb=OBBConfig(
            mode="direct",
            direct=OBBDirectConfig(model_path="", compute_runtime=compute_runtime),
        ),
        pose=pose_cfg,
        runtime_tier=migrate_runtime_to_tier({compute_runtime}),
    )
    runtime = RuntimeContext.from_config(_min_cfg)
    model = load_pose_model(pose_cfg, runtime)
    return model.backend  # PoseModel.backend is the predict_batch-capable object


def predict_pose_for_image(image, pose_config) -> "PoseResult":  # noqa: F821
    """One-shot pose prediction on a single image, used by PoseKit labeling UI.

    Loads a pose model, builds a whole-image canonical crop, runs pose once,
    and discards the model. NOT for batch use — call InferenceRunner.run_realtime
    if you need persistent state.
    """
    import numpy as np

    from .config import (
        InferenceConfig,
        OBBConfig,
        OBBDirectConfig,
        migrate_runtime_to_tier,
    )
    from .result import OBBResult
    from .runtime import RuntimeContext
    from .stages.crops import extract_canonical_crops
    from .stages.pose import load_pose_model, run_pose

    compute_runtime = "cpu"
    if pose_config is not None:
        if getattr(pose_config, "yolo", None) is not None:
            compute_runtime = getattr(pose_config.yolo, "compute_runtime", "cpu")
        elif getattr(pose_config, "sleap", None) is not None:
            compute_runtime = getattr(pose_config.sleap, "compute_runtime", "cpu")

    # See load_pose_backend above: RuntimeContext.from_config reads
    # cfg.runtime_tier, not the deprecated per-stage compute_runtime fields,
    # so runtime_tier must be derived from the resolved compute_runtime here.
    _min_cfg = InferenceConfig(
        obb=OBBConfig(
            mode="direct",
            direct=OBBDirectConfig(model_path="", compute_runtime=compute_runtime),
        ),
        pose=pose_config,
        runtime_tier=migrate_runtime_to_tier({compute_runtime}),
    )
    try:
        runtime = RuntimeContext.from_config(_min_cfg)
    except Exception:
        _min_cfg.obb.direct.compute_runtime = "cpu"
        runtime = RuntimeContext(
            cuda_mode=False,
            device="cpu",
            use_nvdec=False,
            default_runtime="cpu",
            tensor_on_cuda=False,
            requested_gpu=False,
        )

    h, w = image.shape[:2] if hasattr(image, "shape") else (1, 1)
    synthetic_obb = OBBResult(
        frame_idx=0,
        centroids=np.array([[w / 2, h / 2]], dtype=np.float32),
        angles=np.zeros(1, dtype=np.float32),
        sizes=np.array([float(w * h)], dtype=np.float32),
        shapes=np.array([[float(w * h), float(w) / float(h + 1e-6)]], dtype=np.float32),
        confidences=np.ones(1, dtype=np.float32),
        corners=np.array([[[0, 0], [w, 0], [w, h], [0, h]]], dtype=np.float32),
        detection_ids=OBBResult.make_detection_ids(0, 1),
    )

    ar = 2.0
    mg = 1.3
    model = load_pose_model(pose_config, runtime)
    try:
        crops = extract_canonical_crops(image, synthetic_obb, ar, mg, runtime)
        return run_pose(crops, synthetic_obb, model, pose_config, runtime, ar, mg)
    finally:
        del model
