from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch

# Max detections per frame; matches the legacy stride used to encode
# detection IDs as `frame_idx * STRIDE + slot`. Downstream consumers
# (CSV writer, identity evidence, pose keypoint maps, AprilTag association)
# treat this integer as the primary key for joins across caches.
DETECTION_ID_STRIDE = 10000


@dataclass
class OBBResult:
    frame_idx: int
    centroids: np.ndarray  # (D, 2)  cx, cy
    angles: np.ndarray  # (D,)    radians
    sizes: np.ndarray  # (D,)    area px²
    shapes: np.ndarray  # (D, 2)  ellipse_area, aspect_ratio
    confidences: np.ndarray  # (D,)    raw detection confidence
    corners: np.ndarray  # (D, 4, 2) OBB corners
    detection_ids: np.ndarray  # (D,) int64 primary key for downstream consumers
    class_ids: np.ndarray | None = (
        None  # (D,) int64 model class id per detection; None => all class 0
    )

    @property
    def num_detections(self) -> int:
        return int(len(self.confidences))

    @property
    def class_ids_or_zeros(self) -> np.ndarray:
        """Safe (D,) int64 class-id array, defaulting to all-zeros when unset."""
        if self.class_ids is None:
            return np.zeros(self.num_detections, dtype=np.int64)
        return self.class_ids

    @staticmethod
    def make_detection_ids(frame_idx: int, num_detections: int) -> np.ndarray:
        """Generate legacy-compatible primary keys: frame_idx * STRIDE + slot."""
        return np.arange(num_detections, dtype=np.int64) + np.int64(
            frame_idx
        ) * np.int64(DETECTION_ID_STRIDE)


@dataclass
class CropBatch:
    """Cross-frame canonical crops shared read-only by head-tail / CNN / pose.

    Row order is detection-id order (frame-index-derived), so batch membership
    is a pure function of frame index — the reproducibility invariant.
    """

    crops: "torch.Tensor"  # (N, C, H, W) device-resident or CPU
    detection_ids: np.ndarray  # (N,) int64
    frame_index: np.ndarray  # (N,) int64
    obb_by_frame: dict  # frame_idx -> OBBResult
    native_sizes: np.ndarray  # (N, 2) int64 — pre-pad crop h,w

    def frames(self) -> list:
        return sorted({int(f) for f in self.frame_index.tolist()})

    def select_frame(self, frame_idx: int) -> np.ndarray:
        return np.nonzero(self.frame_index == int(frame_idx))[0]


@dataclass
class HeadTailResult:
    heading_hints: np.ndarray  # (D,) radians; nan = no confident direction
    heading_confidences: np.ndarray  # (D,)
    directed_mask: np.ndarray  # (D,) uint8; 1 = heading trusted
    # (D, 2, 3) affine matrices, or None when loaded from cache (caches store
    # outputs only; affines are recomputable from OBBResult + headtail config).
    canonical_affines: np.ndarray | None


@dataclass
class CNNFactorPrediction:
    factor_name: str
    class_names: list[str]
    raw_probabilities: np.ndarray  # (num_classes,) pre-calibration


@dataclass
class CNNDetectionPrediction:
    det_index: int
    factors: list[CNNFactorPrediction]  # len=1 flat; len=K multi-head


@dataclass
class CNNResult:
    label: str  # from CNNConfig.label
    predictions: list[CNNDetectionPrediction]  # one per detection


@dataclass
class PoseResult:
    keypoints: np.ndarray  # (D, K, 3): [x, y, confidence] per keypoint
    valid_mask: np.ndarray  # (D,) bool: meets min_kpt_conf + min_valid_kpts


@dataclass
class AprilTagResult:
    tag_ids: list[int]
    det_indices: list[int]  # which OBB detection each tag maps to
    centers: np.ndarray  # (T, 2)
    corners: np.ndarray  # (T, 4, 2)


@dataclass
class FrameResult:
    frame_idx: int
    obb: OBBResult
    filtered_indices: list[int]  # detections that survived filtering
    headtail: HeadTailResult | None
    cnn: list[CNNResult]  # one per CNN phase
    pose: PoseResult | None
    apriltag: AprilTagResult | None
    resolved_headings: np.ndarray  # (D,) final merged heading per detection
    # Task 17g: populated by InferenceRunner.run_realtime() for legacy-API
    # consumers that expect a StreamingAnalysisPayload (identity evidence,
    # live feature pre-compute workers).  None in batch-pass results.
    streaming_payload: "StreamingAnalysisPayload | None" = None  # noqa: F821
    # Task 10b: bg-sub preview overlays (SHOW_FG / SHOW_BG). Populated by
    # InferenceRunner.run_realtime() on the bgsub detection source only; None in
    # batch-pass results and under any other detection source, since carrying
    # full-frame masks through a cached batch pass would be pure waste (they are
    # only ever drawn on a live preview). Both are in the RESIZE_FACTOR-scaled
    # coordinate space the stage detected in.
    fg_mask: np.ndarray | None = None  # (H, W) uint8, the mask detection ran on
    bg_u8: np.ndarray | None = None  # (H, W) uint8, the background it ran against


def assemble_resolved_headings(
    obb: OBBResult,
    headtail: HeadTailResult | None,
    pose_headings: np.ndarray | None,  # (D,) nan where pose unavailable
    pose_valid: np.ndarray | None,  # (D,) bool
    overrides_headtail: bool = True,
) -> np.ndarray:
    """Merge headings with priority: pose -> headtail -> OBB axis.

    When overrides_headtail=False the priority is: headtail -> pose -> OBB axis.
    """
    result = obb.angles.copy()

    if headtail is not None:
        for i in range(obb.num_detections):
            if headtail.directed_mask[i] and not math.isnan(
                float(headtail.heading_hints[i])
            ):
                result[i] = headtail.heading_hints[i]

    if pose_headings is not None and pose_valid is not None:
        for i in range(obb.num_detections):
            if not pose_valid[i]:
                continue
            if math.isnan(float(pose_headings[i])):
                continue
            if overrides_headtail:
                result[i] = pose_headings[i]
            elif headtail is None or not headtail.directed_mask[i]:
                result[i] = pose_headings[i]

    return result
