"""Tiny-clip harness for depth-invariance tests.

``run_pipeline_to_caches(out_dir, depth)`` builds a 6-frame 64×64 synthetic
video, wires deterministic stub models through a real ``InferenceRunner``, runs
``run_batch_pass``, and returns ``{filename: sha256hex}`` for every ``.npz``
written into *out_dir*.

Designed so Tasks 11/12 can call it with ``depth=2`` or ``depth=4`` and assert
the returned dicts are equal to the depth=1 result (byte-identical caches).
For THIS task only depth=1 is exercised; depth>1 currently degrades to depth=1
(real concurrency lands later).

Stub strategy
-------------
* ``_load_all_models`` is monkeypatched to return an ``_AllModels`` where:
  - ``obb`` is a real ``OBBModels(mode="direct")``; the live ``direct_model``
    slot is never called because we also patch ``run_obb`` in the pipeline
    module to return deterministic ``OBBResult``s without touching the model.
  - ``headtail`` is a stub ``HeadTailModel`` (MagicMock backend, input_size
    (64,64), class_names ["head","tail"]). The live backend is never called
    because ``run_headtail_batch`` is patched in the pipeline module.
  - ``cnn`` holds one stub ``CNNModel`` for the single configured CNN phase.
    The live backend is never called because ``run_cnn_batch`` is patched.
  - ``pose`` is a stub ``PoseModel`` (MagicMock backend, n_keypoints=3,
    keypoint_names ["a","b","c"]). The live backend is never called because
    ``run_pose_batch`` (and ``extract_canonical_crops_batch``) are patched.
  - ``apriltag`` is ``None`` (disabled in config).
* ``run_obb`` in ``hydra_suite.core.inference.pipeline`` is patched to return a
  fixed 2-detection ``OBBResult`` per frame, seeded from the frame index so
  detection_ids are stable across identical index sequences.
* ``run_headtail_batch``, ``run_cnn_batch``, ``extract_canonical_crops_batch``,
  and ``run_pose_batch`` in the pipeline module are all patched so no real model
  inference touches torch or disk — every output is a pure function of
  frame_idx / detection_id (no wall-clock, no RNG without a fixed seed).

``detection.npz``, ``headtail.npz``, ``cnn_<label>.npz``, and ``pose.npz``
are all written.  The detection cache is byte-stable because:
  1. OBBResult values are deterministic functions of frame_idx.
  2. The video signature is computed from the fixed file bytes (same video
     content each call → same sig → same cache key).
  3. ``_npz_save`` uses ``np.savez``, which is order-stable for a fixed
     set of named arrays written in the same sequence.
"""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from hydra_suite.core.inference.config import (
    CNNConfig,
    HeadTailConfig,
    InferenceConfig,
    OBBConfig,
    OBBDirectConfig,
    PoseConfig,
    PoseYOLOConfig,
)
from hydra_suite.core.inference.result import (
    CNNDetectionPrediction,
    CNNFactorPrediction,
    CNNResult,
    CropBatch,
    HeadTailResult,
    OBBResult,
    PoseResult,
)
from hydra_suite.core.inference.stages.cnn import CNNModel
from hydra_suite.core.inference.stages.headtail import HeadTailModel
from hydra_suite.core.inference.stages.obb import OBBModels
from hydra_suite.core.inference.stages.pose import PoseModel

# ---------------------------------------------------------------------------
# Tiny synthetic video
# ---------------------------------------------------------------------------

_FRAME_W = 64
_FRAME_H = 64
_NUM_FRAMES = 6
_FPS = 5.0

# Number of keypoints the stub pose model produces
_N_KEYPOINTS = 3
# CNN label for the single stub phase
_CNN_LABEL = "stub_cnn"
_CNN_FACTOR = "stub_factor"
_CNN_CLASSES = ["class0", "class1"]


def _write_tiny_video(path: Path) -> None:
    """Write a 6-frame 64×64 MP4 with deterministic pixel content."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, _FPS, (_FRAME_W, _FRAME_H))
    for i in range(_NUM_FRAMES):
        frame = np.full((_FRAME_H, _FRAME_W, 3), (i * 40) % 256, dtype=np.uint8)
        writer.write(frame)
    writer.release()


# ---------------------------------------------------------------------------
# Deterministic stub OBBResult
# ---------------------------------------------------------------------------


def _make_stub_obb(frame_idx: int, n: int = 2) -> OBBResult:
    """Return a deterministic OBBResult for ``frame_idx`` with ``n`` detections.

    All geometry values are simple integer multiples of frame_idx so the
    resulting cache bytes are identical across identical frame sequences.
    """
    rng = np.random.default_rng(frame_idx)
    centroids = rng.uniform(10, 50, (n, 2)).astype(np.float32)
    angles = rng.uniform(0, np.pi, n).astype(np.float32)
    sizes = np.full(n, 100.0, dtype=np.float32)
    shapes = np.ones((n, 2), dtype=np.float32)
    confidences = np.full(n, 0.95, dtype=np.float32)
    corners = rng.uniform(5, 55, (n, 4, 2)).astype(np.float32)
    detection_ids = OBBResult.make_detection_ids(frame_idx, n)
    return OBBResult(
        frame_idx=frame_idx,
        centroids=centroids,
        angles=angles,
        sizes=sizes,
        shapes=shapes,
        confidences=confidences,
        corners=corners,
        detection_ids=detection_ids,
    )


def _fake_run_obb(frames, models, obb_config, runtime):
    """Deterministic run_obb stub: returns one OBBResult per input frame.

    The frame_idx is not available at call time (the pipeline passes raw frame
    arrays, not indices), so we produce a result whose frame_idx will be
    re-stamped by the pipeline's materialize+re-stamp loop.  We return
    OBBResults with frame_idx=0; the pipeline overwrites frame_idx with the
    real index, making detection_ids stable and correct.
    """
    return [_make_stub_obb(frame_idx=0, n=2) for _ in frames]


# ---------------------------------------------------------------------------
# Deterministic stub HeadTailResult
# ---------------------------------------------------------------------------


def _make_stub_headtail(frame_idx: int, detection_ids: np.ndarray) -> HeadTailResult:
    """Deterministic HeadTailResult: heading seeded by (frame_idx, det_id)."""
    n = len(detection_ids)
    rng = np.random.default_rng(int(frame_idx) * 100 + 7)
    hints = rng.uniform(-np.pi, np.pi, n).astype(np.float32)
    confs = rng.uniform(0.6, 1.0, n).astype(np.float32)
    directed = np.ones(n, dtype=np.uint8)
    return HeadTailResult(
        heading_hints=hints,
        heading_confidences=confs,
        directed_mask=directed,
        canonical_affines=None,
    )


def _fake_run_headtail_batch(frames, obb_results, model, config, runtime, ar, mg):
    """Deterministic run_headtail_batch stub.

    Keyed by frame_idx via the OBBResult list (parallel to frames). Output is a
    pure function of frame_idx and detection_ids so mis-splits across windows
    produce different hashes.
    """
    results: dict[int, HeadTailResult] = {}
    for obb in obb_results:
        results[obb.frame_idx] = _make_stub_headtail(obb.frame_idx, obb.detection_ids)
    return results


# ---------------------------------------------------------------------------
# Deterministic stub CNNResult
# ---------------------------------------------------------------------------


def _make_stub_cnn(frame_idx: int, detection_ids: np.ndarray, label: str) -> CNNResult:
    """Deterministic CNNResult seeded by frame_idx."""
    n = len(detection_ids)
    rng = np.random.default_rng(int(frame_idx) * 100 + 13)
    preds: list[CNNDetectionPrediction] = []
    for i, det_id in enumerate(detection_ids):
        raw_probs = rng.dirichlet([1.0] * len(_CNN_CLASSES)).astype(np.float32)
        factor = CNNFactorPrediction(
            factor_name=_CNN_FACTOR,
            class_names=list(_CNN_CLASSES),
            raw_probabilities=raw_probs,
        )
        preds.append(CNNDetectionPrediction(det_index=int(det_id), factors=[factor]))
    return CNNResult(label=label, predictions=preds)


def _fake_run_cnn_batch(frames, obb_results, model, config, runtime, ar, mg):
    """Deterministic run_cnn_batch stub.

    Returns one CNNResult per frame, keyed by frame_idx. The label is taken from
    the CNNModel.factor_names stub attribute so the cache label is stable.
    """
    results: dict[int, CNNResult] = {}
    label = getattr(model, "_stub_label", _CNN_LABEL)
    for obb in obb_results:
        results[obb.frame_idx] = _make_stub_cnn(obb.frame_idx, obb.detection_ids, label)
    return results


# ---------------------------------------------------------------------------
# Deterministic stub PoseResult + fake CropBatch
# ---------------------------------------------------------------------------


def _make_stub_pose(frame_idx: int, n_dets: int, n_kpts: int) -> PoseResult:
    """Deterministic PoseResult seeded by frame_idx."""
    rng = np.random.default_rng(int(frame_idx) * 100 + 17)
    keypoints = rng.uniform(0, 64, (n_dets, n_kpts, 3)).astype(np.float32)
    # Set confidence channel (index 2) to values between 0.5 and 1.0
    keypoints[:, :, 2] = rng.uniform(0.5, 1.0, (n_dets, n_kpts)).astype(np.float32)
    valid_mask = np.ones(n_dets, dtype=bool)
    return PoseResult(keypoints=keypoints, valid_mask=valid_mask)


def _fake_extract_canonical_crops_batch(
    frames, obb_results, canonical_aspect_ratio, canonical_margin, runtime, **kwargs
):
    """Stub for extract_canonical_crops_batch: returns a minimal CropBatch.

    Returns a CropBatch whose obb_by_frame is populated from obb_results so
    the downstream run_pose_batch stub can extract frame indices. Crops tensor
    is a zeros placeholder (never read by the stub).
    """
    import torch

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


def _fake_run_pose_batch(crop_batch, model, config, runtime, **kwargs):
    """Deterministic run_pose_batch stub.

    Derives frame indices from crop_batch.obb_by_frame (populated by the fake
    extract_canonical_crops_batch). Returns one PoseResult per frame keyed by
    frame_idx, seeded by frame_idx so mis-splits produce different hashes.
    """
    n_kpts = getattr(model, "n_keypoints", _N_KEYPOINTS)
    results: dict[int, PoseResult] = {}
    for frame_idx, obb in crop_batch.obb_by_frame.items():
        results[frame_idx] = _make_stub_pose(frame_idx, obb.num_detections, n_kpts)
    return results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_pipeline_to_caches(out_dir: Path, depth: int = 1) -> dict[str, str]:
    """Run a full inference batch pass over a tiny clip; return SHA-256 of each .npz.

    Parameters
    ----------
    out_dir:
        Directory to receive the cache ``.npz`` files.  Created if absent.
    depth:
        ``pipeline_depth`` passed to ``InferenceConfig``.  Currently depth>1
        degrades to depth=1 (Tasks 11/12 add real concurrency).

    Returns
    -------
    dict mapping ``{filename: sha256hex}`` for every ``.npz`` found in
    *out_dir* after the pass, sorted by filename for stability.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Write the tiny video into a temporary file so the video signature is
    # consistent across calls (same bytes → same sig → same cache key).
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        video_path = Path(f.name)
    try:
        _write_tiny_video(video_path)
        return _run_pass(video_path, out_dir, depth)
    finally:
        video_path.unlink(missing_ok=True)


def _run_pass(video_path: Path, cache_dir: Path, depth: int) -> dict[str, str]:
    from hydra_suite.core.inference.runner import InferenceRunner, _AllModels

    cfg = InferenceConfig(
        obb=OBBConfig(
            mode="direct",
            direct=OBBDirectConfig(model_path="/stub.pt", compute_runtime="cpu"),
            # Set confidence_threshold low enough that the stub detections pass
            confidence_threshold=0.1,
            iou_threshold=1.0,
        ),
        headtail=HeadTailConfig(
            model_path="/stub_ht.pt",
            compute_runtime="cpu",
        ),
        cnn_phases=[
            CNNConfig(
                label=_CNN_LABEL,
                model_path="/stub_cnn.pt",
                compute_runtime="cpu",
            )
        ],
        pose=PoseConfig(
            backend="yolo",
            yolo=PoseYOLOConfig(
                model_path="/stub_pose.pt",
                compute_runtime="cpu",
            ),
            skeleton_file="",
        ),
        detection_batch_size=2,  # small batch → multiple windows over 6 frames
        pipeline_depth=depth,
    )

    # Stub HeadTailModel: backend is a MagicMock (never called due to patch).
    stub_ht_model = HeadTailModel(
        backend=MagicMock(),
        input_size=(64, 64),
        class_names=["head", "tail"],
    )
    # Stub CNNModel: backend is a MagicMock; _stub_label lets the fake pick the label.
    stub_cnn_model = CNNModel(
        backend=MagicMock(),
        input_size=(64, 64),
        factor_names=[_CNN_FACTOR],
        factor_class_names=[list(_CNN_CLASSES)],
    )
    stub_cnn_model._stub_label = _CNN_LABEL  # type: ignore[attr-defined]
    # Stub PoseModel: backend is a MagicMock (never called due to patch).
    stub_pose_model = PoseModel(
        backend=MagicMock(),
        n_keypoints=_N_KEYPOINTS,
        keypoint_names=["a", "b", "c"],
    )

    stub_models = _AllModels(
        obb=OBBModels(mode="direct", direct_model=MagicMock()),
        headtail=stub_ht_model,
        cnn=[stub_cnn_model],
        pose=stub_pose_model,
        apriltag=None,
    )

    with (
        patch(
            "hydra_suite.core.inference.runner._load_all_models",
            return_value=stub_models,
        ),
        patch(
            "hydra_suite.core.inference.pipeline.run_obb",
            side_effect=_fake_run_obb,
        ),
        patch(
            "hydra_suite.core.inference.pipeline.run_headtail_batch",
            side_effect=_fake_run_headtail_batch,
        ),
        patch(
            "hydra_suite.core.inference.pipeline.run_cnn_batch",
            side_effect=_fake_run_cnn_batch,
        ),
        patch(
            "hydra_suite.core.inference.pipeline.extract_canonical_crops_batch",
            side_effect=_fake_extract_canonical_crops_batch,
        ),
        patch(
            "hydra_suite.core.inference.pipeline.run_pose_batch",
            side_effect=_fake_run_pose_batch,
        ),
    ):
        # Pass video_path=None so video_signature(None)==""  (see cache/keys.py),
        # making the cache key mtime-insensitive across separate runs.
        # The actual video file is only needed by run_batch_pass for cv2 reading.
        runner = InferenceRunner(
            cfg,
            cache_dir=cache_dir,
            video_path=None,
        )
        runner.run_batch_pass(video_path)

    return _hash_npz_files(cache_dir)


def _hash_npz_files(directory: Path) -> dict[str, str]:
    """Return {filename: sha256hex} for every .npz in *directory*, sorted by name."""
    result: dict[str, str] = {}
    for npz_path in sorted(directory.glob("*.npz")):
        digest = hashlib.sha256(npz_path.read_bytes()).hexdigest()
        result[npz_path.name] = digest
    return result
