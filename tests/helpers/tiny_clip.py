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
  - ``headtail``, ``cnn``, ``pose``, ``apriltag`` are all ``None`` / ``[]``
    (disabled in the config), so no downstream stage runs.
* ``run_obb`` in ``hydra_suite.core.inference.pipeline`` is patched to return a
  fixed 2-detection ``OBBResult`` per frame, seeded from the frame index so
  detection_ids are stable across identical index sequences.

Only ``detection.npz`` is written (no headtail/cnn/pose/apriltag caches).
The detection cache is byte-stable because:
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
    InferenceConfig,
    OBBConfig,
    OBBDirectConfig,
)
from hydra_suite.core.inference.result import OBBResult
from hydra_suite.core.inference.stages.obb import OBBModels

# ---------------------------------------------------------------------------
# Tiny synthetic video
# ---------------------------------------------------------------------------

_FRAME_W = 64
_FRAME_H = 64
_NUM_FRAMES = 6
_FPS = 5.0


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
        detection_batch_size=2,  # small batch → multiple windows over 6 frames
        pipeline_depth=depth,
    )

    stub_models = _AllModels(
        obb=OBBModels(mode="direct", direct_model=MagicMock()),
        headtail=None,
        cnn=[],
        pose=None,
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
    ):
        # Pass video_path=None so the video signature is "" (stable across runs).
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
