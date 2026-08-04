"""Regression test: pose-batch crop geometry must match the geometry used to
build those crops.

``Pipeline._process_obb_results`` (in ``core/inference/pipeline.py``) builds
pose crops via ``extract_canonical_crops_batch(..., ar, mg, ...)`` where
``ar``/``mg`` come from ``cfg.headtail.canonical_aspect_ratio`` /
``canonical_margin``. It must pass the SAME ``ar``/``mg`` to
``run_pose_batch(...)`` so the native-crop-dimension recovery and the inverse
affine used to map keypoints back to frame coordinates agree with how the
crops were built. If the two calls disagree (e.g. ``run_pose_batch`` silently
falls back to its own defaults), keypoints are decoded against the wrong
crop geometry and are displaced from their true frame position.

This test drives ``Pipeline._process_window`` end-to-end (no real models —
OBB/crop/pose stage functions are monkeypatched in the ``pipeline`` module,
following the same pattern as ``tests/test_inference_depth_invariance.py`` /
``tests/helpers/tiny_clip.py``) with a configured ``canonical_aspect_ratio``
of 2.45 and ``canonical_margin`` of 1.5 — deliberately NOT the module-level
defaults (2.0 / 1.3) baked into ``run_pose_batch``'s signature — and asserts
the aspect ratio / margin actually observed by ``extract_canonical_crops_batch``
equals the aspect ratio / margin actually observed by ``run_pose_batch``.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import torch

from hydra_suite.core.inference.cache.writer import CacheWriter
from hydra_suite.core.inference.config import (
    HeadTailConfig,
    InferenceConfig,
    OBBConfig,
    OBBDirectConfig,
    PoseConfig,
    PoseYOLOConfig,
)
from hydra_suite.core.inference.pipeline import BatchWindow, Pipeline, PipelineStages
from hydra_suite.core.inference.result import CropBatch, OBBResult, PoseResult
from hydra_suite.core.inference.runtime import RuntimeContext
from hydra_suite.core.inference.stages.pose import PoseModel

_CONFIGURED_AR = 2.45
_CONFIGURED_MG = 1.5


def _make_obb(frame_idx: int, n: int = 1) -> OBBResult:
    rng = np.random.default_rng(frame_idx)
    return OBBResult(
        frame_idx=frame_idx,
        centroids=rng.uniform(10, 50, (n, 2)).astype(np.float32),
        angles=rng.uniform(0, np.pi, n).astype(np.float32),
        sizes=np.full(n, 100.0, dtype=np.float32),
        shapes=np.ones((n, 2), dtype=np.float32),
        confidences=np.full(n, 0.95, dtype=np.float32),
        corners=rng.uniform(5, 55, (n, 4, 2)).astype(np.float32),
        detection_ids=OBBResult.make_detection_ids(frame_idx, n),
    )


def _fake_run_obb(frames, models, obb_config, runtime, roi_mask=None):
    # frame_idx is re-stamped by the pipeline's materialize+re-stamp loop, so
    # the stub value here is irrelevant.
    return [_make_obb(frame_idx=0) for _ in frames]


def _build_pipeline(captured: dict) -> tuple[Pipeline, BatchWindow]:
    cfg = InferenceConfig(
        obb=OBBConfig(
            mode="direct",
            direct=OBBDirectConfig(model_path="/stub.pt"),
            confidence_threshold=0.1,
            iou_threshold=1.0,
        ),
        # headtail_model stays None (below) so run_headtail_batch never runs,
        # but cfg.headtail is still consulted by pipeline.py to derive ar/mg.
        headtail=HeadTailConfig(
            model_path="/stub_ht.pt",
            canonical_aspect_ratio=_CONFIGURED_AR,
            canonical_margin=_CONFIGURED_MG,
        ),
        cnn_phases=[],
        pose=PoseConfig(
            backend="yolo",
            yolo=PoseYOLOConfig(model_path="/stub_pose.pt"),
            skeleton_file="",
        ),
        detection_batch_size=2,
        runtime_tier="cpu",
    )

    stages = PipelineStages(
        config=cfg,
        obb_models=MagicMock(),
        headtail_model=None,  # skip HT stage entirely; only cfg.headtail matters here
        cnn_models=[],
        pose_model=PoseModel(backend=MagicMock(), n_keypoints=1, keypoint_names=["a"]),
        apriltag_model=None,
    )
    runtime = RuntimeContext(
        cuda_mode=False, device="cpu", use_nvdec=False, tensor_on_cuda=False
    )
    writer = CacheWriter({}, [], async_mode=False)

    pipe = Pipeline(stages, runtime, writer, depth=1)
    window = BatchWindow(frames=[np.zeros((4, 4, 3), np.uint8)], frame_indices=[0])
    return pipe, window


def _capturing_extract_canonical_crops_batch(captured):
    def _fn(frames, obb_results, aspect_ratio, margin, runtime, **kwargs):
        captured["crop_ar"] = aspect_ratio
        captured["crop_mg"] = margin
        n_total = sum(o.num_detections for o in obb_results)
        det_ids = (
            np.concatenate([o.detection_ids for o in obb_results])
            if obb_results
            else np.zeros(0, np.int64)
        )
        frame_index = (
            np.concatenate(
                [np.full(o.num_detections, o.frame_idx, np.int64) for o in obb_results]
            )
            if obb_results
            else np.zeros(0, np.int64)
        )
        obb_by_frame = {o.frame_idx: o for o in obb_results}
        native_sizes = np.zeros((n_total, 2), np.int64)
        crops = torch.zeros((n_total, 3, 1, 1))
        return CropBatch(
            crops=crops,
            detection_ids=det_ids,
            frame_index=frame_index,
            obb_by_frame=obb_by_frame,
            native_sizes=native_sizes,
        )

    return _fn


def _capturing_run_pose_batch(captured):
    def _fn(crop_batch, model, config, runtime, aspect_ratio=2.0, margin=1.3, **kwargs):
        captured["pose_ar"] = aspect_ratio
        captured["pose_mg"] = margin
        results: dict[int, PoseResult] = {}
        for frame_idx, obb in crop_batch.obb_by_frame.items():
            n = obb.num_detections
            results[frame_idx] = PoseResult(
                keypoints=np.zeros((n, 1, 3), np.float32),
                valid_mask=np.ones(n, dtype=bool),
            )
        return results

    return _fn


def test_pose_batch_crop_geometry_matches_build_geometry():
    """The ar/mg used to BUILD pose crops must equal the ar/mg used to RECOVER them.

    Before the fix, ``run_pose_batch`` is called without forwarding ``ar``/``mg``,
    so it silently falls back to its own defaults (2.0 / 1.3) even though the
    crops were built at the configured 2.45 / 1.5 — this assertion catches that
    mismatch directly.
    """
    captured: dict = {}
    pipe, window = _build_pipeline(captured)

    with (
        patch("hydra_suite.core.inference.pipeline.run_obb", side_effect=_fake_run_obb),
        patch(
            "hydra_suite.core.inference.pipeline.extract_canonical_crops_batch",
            side_effect=_capturing_extract_canonical_crops_batch(captured),
        ),
        patch(
            "hydra_suite.core.inference.pipeline.run_pose_batch",
            side_effect=_capturing_run_pose_batch(captured),
        ),
    ):
        pipe._process_window(window)

    assert "crop_ar" in captured and "pose_ar" in captured, (
        "extract_canonical_crops_batch / run_pose_batch were not both called "
        f"(captured={captured})"
    )
    assert captured["crop_ar"] == _CONFIGURED_AR, (
        "sanity: crop-build aspect ratio should reflect the configured "
        f"canonical_aspect_ratio; got {captured['crop_ar']}"
    )
    assert captured["crop_ar"] == captured["pose_ar"], (
        "pose-batch recovery aspect ratio must match the crop-build aspect "
        f"ratio: built with {captured['crop_ar']}, recovered with "
        f"{captured['pose_ar']}"
    )
    assert captured["crop_mg"] == captured["pose_mg"], (
        "pose-batch recovery margin must match the crop-build margin: "
        f"built with {captured['crop_mg']}, recovered with {captured['pose_mg']}"
    )
