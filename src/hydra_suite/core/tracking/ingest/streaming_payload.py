"""Shared filtered-detection analysis payload.

Streaming Phase 1: single handoff object produced once after the sequence:
  detect → filter → head-tail → canonical reorientation

Consumed by pose analysis, CNN analysis, and identity evidence emission.
Detection slot ordering is stable from construction until all downstream caches
are emitted, so (frame_idx, detection_ids[i]) is a safe primary index key for
the identity evidence sidecar.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class StreamingAnalysisPayload:
    """Post-filter, post-head-tail handoff for individual analysis.

    All mutable array fields are owned by this object and must not be modified
    by consumers.  If a downstream step needs to alter an array it must copy it.

    Parameters
    ----------
    frame_idx:
        Absolute frame index in the video.
    detection_ids:
        Shape (D,) int64.  Stable detection slot IDs within this frame.  These
        match the indices stored in the detection cache and the identity evidence
        sidecar.
    obb_corners:
        Shape (D, 4, 2) float32.  OBB corner coordinates for each filtered
        detection, in pixel space.
    headtail_heading:
        Shape (D,) float32.  Heading angle in radians (canonical head-to-tail
        direction), aligned with the head-tail model output.
    headtail_confidence:
        Shape (D,) float32.  Head-tail model confidence per detection.
    headtail_directed:
        Shape (D,) bool.  True when the head-tail model assigned a directed
        orientation (not just symmetric).
    canonical_affines:
        Shape (D, 2, 3) float32.  Affine matrix warping each detection to its
        canonical head-forward orientation.  None entries signal that no
        canonical affine is available for that detection slot.
    input_is_bgr:
        Whether frame imagery uses BGR channel order (OpenCV convention).
    runtime_family:
        One of ``'cpu'``, ``'mps'``, ``'cuda'``, ``'onnx_cpu'``,
        ``'onnx_cuda'``, ``'tensorrt'``.  Informs downstream backends about
        safe device placement and inference paths.
    canonical_crops_cpu:
        Optional list of pre-extracted canonical crops as CPU uint8 arrays, one
        per detection, shape (H, W, 3).  Present when crops were extracted
        upstream and can be reused without re-extracting.
    canonical_crops_cuda:
        Optional stacked CUDA tensor (B, C, H, W) of canonical crops.  Present
        only when ``runtime_family`` is ``'cuda'`` or ``'tensorrt'`` and the
        upstream crop path was GPU-native.
    """

    frame_idx: int
    detection_ids: np.ndarray
    obb_corners: np.ndarray
    headtail_heading: np.ndarray
    headtail_confidence: np.ndarray
    headtail_directed: np.ndarray
    canonical_affines: np.ndarray
    input_is_bgr: bool = True
    runtime_family: str = "cpu"
    canonical_crops_cpu: Optional[list] = None
    canonical_crops_cuda: Optional[object] = None

    @property
    def num_detections(self) -> int:
        """Number of filtered detections in this payload."""
        return len(self.detection_ids)

    def has_cuda_crops(self) -> bool:
        """True if GPU-resident canonical crops are available."""
        return self.canonical_crops_cuda is not None

    def has_cpu_crops(self) -> bool:
        """True if CPU canonical crops are available."""
        return (
            self.canonical_crops_cpu is not None and len(self.canonical_crops_cpu) > 0
        )

    def best_crops(self) -> Optional[list]:
        """Return CPU crops, falling back to None.

        GPU crops (``canonical_crops_cuda``) are not unwrapped here; callers
        that can consume CUDA tensors should check ``has_cuda_crops()`` first.
        """
        if self.has_cpu_crops():
            return self.canonical_crops_cpu
        return None

    @classmethod
    def from_frame_result(
        cls,
        frame_result: "FrameResult",  # noqa: F821
        runtime_family: str = "cpu",
        input_is_bgr: bool = True,
    ) -> "StreamingAnalysisPayload":
        """Construct a StreamingAnalysisPayload from a new-pipeline FrameResult.

        Task 17g / Correction 23: bridges the new InferenceRunner output to the
        legacy streaming-payload contract consumed by identity analysis workers.
        """
        obb = frame_result.obb
        n = obb.num_detections

        ids = obb.detection_ids.copy()
        corners = obb.corners.copy()

        ht = frame_result.headtail
        if ht is not None:
            heading = ht.heading_hints.copy()
            conf = ht.heading_confidences.copy()
            directed = ht.directed_mask.astype(bool).copy()
            if ht.canonical_affines is not None:
                affines = ht.canonical_affines.copy()
            else:
                affines = np.zeros((n, 2, 3), dtype=np.float32)
        else:
            heading = np.zeros(n, dtype=np.float32)
            conf = np.zeros(n, dtype=np.float32)
            directed = np.zeros(n, dtype=bool)
            affines = np.zeros((n, 2, 3), dtype=np.float32)

        return cls(
            frame_idx=int(obb.frame_idx),
            detection_ids=ids,
            obb_corners=corners,
            headtail_heading=heading,
            headtail_confidence=conf,
            headtail_directed=directed,
            canonical_affines=affines,
            input_is_bgr=input_is_bgr,
            runtime_family=runtime_family,
        )


def build_streaming_payload(
    frame_idx: int,
    raw_meas: list,
    raw_obb_corners: list,
    raw_heading_hints: list,
    raw_heading_confidences: list,
    raw_directed_mask: list,
    raw_canonical_affines: list,
    detection_ids: list,
    input_is_bgr: bool = True,
    runtime_family: str = "cpu",
    canonical_crops_cpu: Optional[list] = None,
    canonical_crops_cuda: Optional[object] = None,
) -> StreamingAnalysisPayload:
    """Construct a ``StreamingAnalysisPayload`` from raw detector output arrays.

    All lists may be empty (no detections).  Empty payloads are valid and will
    propagate gracefully through downstream consumer code.

    Parameters mirror the per-frame raw return contract of
    ``YOLOOBBDetector.detect_objects_batched(return_raw=True)``.
    """
    n = len(detection_ids)

    ids = np.asarray(detection_ids, dtype=np.int64)

    if n == 0:
        corners = np.empty((0, 4, 2), dtype=np.float32)
        heading = np.empty(0, dtype=np.float32)
        conf = np.empty(0, dtype=np.float32)
        directed = np.empty(0, dtype=bool)
        affines = np.empty((0, 2, 3), dtype=np.float32)
    else:
        corners = np.asarray(raw_obb_corners, dtype=np.float32)
        if corners.ndim == 2:
            corners = corners.reshape(n, 4, 2)

        heading = np.asarray(
            raw_heading_hints if raw_heading_hints else [0.0] * n, dtype=np.float32
        )
        conf = np.asarray(
            raw_heading_confidences if raw_heading_confidences else [0.0] * n,
            dtype=np.float32,
        )
        directed = np.asarray(
            raw_directed_mask if raw_directed_mask else [False] * n, dtype=bool
        )

        # canonical_affines: list of (2,3) arrays or None
        aff_rows = []
        for a in raw_canonical_affines if raw_canonical_affines else [None] * n:
            if a is not None:
                aff_rows.append(np.asarray(a, dtype=np.float32).reshape(2, 3))
            else:
                aff_rows.append(np.zeros((2, 3), dtype=np.float32))
        affines = np.stack(aff_rows, axis=0)

    return StreamingAnalysisPayload(
        frame_idx=frame_idx,
        detection_ids=ids,
        obb_corners=corners,
        headtail_heading=heading,
        headtail_confidence=conf,
        headtail_directed=directed,
        canonical_affines=affines,
        input_is_bgr=input_is_bgr,
        runtime_family=runtime_family,
        canonical_crops_cpu=canonical_crops_cpu,
        canonical_crops_cuda=canonical_crops_cuda,
    )
