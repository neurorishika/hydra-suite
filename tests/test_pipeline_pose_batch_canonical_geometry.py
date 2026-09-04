"""Regression test: pose-batch crop geometry must match the geometry used to
build those crops.

``Pipeline._process_obb_results`` (in ``core/inference/pipeline.py``) builds
pose crops via ``extract_canonical_crops_batch(..., geometry, ...)`` where
``geometry`` comes from ``cfg.canonical`` -- the single project-wide canonical
geometry shared by head-tail, CNN, and pose. It must pass that SAME geometry
object to ``run_pose_batch(...)`` so the inverse affine used to map keypoints
back to frame coordinates agrees with how the crops were built. If the two
calls disagree (e.g. ``run_pose_batch`` silently falls back to its own
default geometry), keypoints are decoded against the wrong crop geometry and
are displaced from their true frame position.

This test drives ``Pipeline._process_window`` end-to-end (no real models —
OBB/crop/pose stage functions are monkeypatched in the ``pipeline`` module,
following the same pattern as ``tests/test_inference_depth_invariance.py`` /
``tests/helpers/tiny_clip.py``) with a configured ``canonical`` geometry of
aspect ratio 2.45 / margin 1.5 — deliberately NOT the default aspect/margin
(2.0 / 1.3) — and asserts the geometry object actually observed by
``extract_canonical_crops_batch`` is the same object observed by
``run_pose_batch``.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import torch

from hydra_suite.core.canonicalization.geometry import CanonicalGeometry
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
from hydra_suite.core.inference.result import (
    CropBatch,
    HeadTailResult,
    OBBResult,
    PoseResult,
)
from hydra_suite.core.inference.runtime import RuntimeContext
from hydra_suite.core.inference.stages.pose import PoseModel

_CONFIGURED_GEOMETRY = CanonicalGeometry.from_reference(20.0, 2.45, 1.5)


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
        # headtail_model stays None (below) so run_headtail_batch never runs.
        headtail=HeadTailConfig(model_path="/stub_ht.pt"),
        cnn_phases=[],
        pose=PoseConfig(
            backend="yolo",
            yolo=PoseYOLOConfig(model_path="/stub_pose.pt"),
            skeleton_file="",
            suppress_foreign_regions=False,
        ),
        canonical=_CONFIGURED_GEOMETRY,
        detection_batch_size=2,
        runtime_tier="cpu",
    )

    stages = PipelineStages(
        config=cfg,
        obb_models=MagicMock(),
        headtail_model=None,  # skip HT stage entirely
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
    def _fn(frames, obb_results, geometry, runtime, **kwargs):
        captured["crop_geometry"] = geometry
        captured["crop_calls"] = captured.get("crop_calls", 0) + 1
        n_total = sum(o.num_detections for o in obb_results)
        captured.setdefault("crop_batch_sizes", []).append(n_total)
        captured.setdefault("crop_frame_counts", []).append(len(frames))
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
        batch = CropBatch(
            crops=crops,
            detection_ids=det_ids,
            frame_index=frame_index,
            obb_by_frame=obb_by_frame,
            native_sizes=native_sizes,
        )
        captured["extracted_batch"] = batch
        return batch

    return _fn


def _capturing_run_pose_batch(captured):
    def _fn(crop_batch, model, config, runtime, geometry=None, **kwargs):
        captured["pose_geometry"] = geometry
        captured["pose_batch"] = crop_batch
        results: dict[int, PoseResult] = {}
        for frame_idx, obb in crop_batch.obb_by_frame.items():
            n = obb.num_detections
            results[frame_idx] = PoseResult(
                keypoints=np.zeros((n, 1, 3), np.float32),
                valid_mask=np.ones(n, dtype=bool),
            )
        return results

    return _fn


def _capturing_run_headtail_batch(captured):
    def _fn(
        frames,
        obb_results,
        model,
        config,
        runtime,
        geometry=None,
        canonical_batch=None,
    ):
        captured["headtail_batch"] = canonical_batch
        return {
            obb.frame_idx: HeadTailResult(
                heading_hints=np.full(obb.num_detections, np.nan, np.float32),
                heading_confidences=np.zeros(obb.num_detections, np.float32),
                directed_mask=np.zeros(obb.num_detections, np.uint8),
                canonical_affines=None,
            )
            for obb in obb_results
        }

    return _fn


def test_pose_batch_crop_geometry_matches_build_geometry():
    """The geometry used to BUILD pose crops must equal the geometry used to
    RECOVER them (invert keypoints back to frame coordinates).

    Before the fix, ``run_pose_batch`` is called without forwarding
    ``geometry``, so it silently falls back to its own default even though the
    crops were built at the configured aspect 2.45 / margin 1.5 -- this
    assertion catches that mismatch directly.
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

    assert "crop_geometry" in captured and "pose_geometry" in captured, (
        "extract_canonical_crops_batch / run_pose_batch were not both called "
        f"(captured={captured})"
    )
    assert captured["crop_geometry"] is _CONFIGURED_GEOMETRY, (
        "sanity: crop-build geometry should be the configured cfg.canonical; "
        f"got {captured['crop_geometry']}"
    )
    assert captured["crop_geometry"] is captured["pose_geometry"], (
        "pose-batch recovery geometry must be the SAME object as the "
        f"crop-build geometry: built with {captured['crop_geometry']}, "
        f"recovered with {captured['pose_geometry']}"
    )


def test_headtail_and_pose_share_one_canonical_extraction():
    captured: dict = {}
    pipe, window = _build_pipeline(captured)
    pipe.stages.headtail_model = MagicMock()

    with (
        patch("hydra_suite.core.inference.pipeline.run_obb", side_effect=_fake_run_obb),
        patch(
            "hydra_suite.core.inference.pipeline.extract_canonical_crops_batch",
            side_effect=_capturing_extract_canonical_crops_batch(captured),
        ),
        patch(
            "hydra_suite.core.inference.pipeline.run_headtail_batch",
            side_effect=_capturing_run_headtail_batch(captured),
        ),
        patch(
            "hydra_suite.core.inference.pipeline.run_pose_batch",
            side_effect=_capturing_run_pose_batch(captured),
        ),
    ):
        pipe._process_window(window)

    assert captured["crop_calls"] == 1
    assert captured["headtail_batch"] is captured["extracted_batch"]
    assert captured["pose_batch"] is captured["extracted_batch"]


def test_downstream_crop_extraction_is_frame_local_not_window_sized():
    captured: dict = {}
    pipe, _window = _build_pipeline(captured)
    window = BatchWindow(
        frames=[np.zeros((4, 4, 3), np.uint8) for _ in range(3)],
        frame_indices=[0, 1, 2],
    )

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

    assert captured["crop_frame_counts"] == [1, 1, 1]
    assert captured["crop_batch_sizes"] == [1, 1, 1]
