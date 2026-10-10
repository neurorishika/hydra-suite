"""The one declared per-frame limit of the inference pipeline.

``MAX_DETECTIONS_PER_FRAME`` bounds (1) detections stored per frame in the
detection cache, (2) per-animal analyses per frame and (3) the number of
animals N itself. It is a LOUD limit: N above it is rejected, and a frame
exceeding it is truncated with a WARNING plus an end-of-run summary.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

MAX_DETECTIONS_PER_FRAME = 1024
# Raw OBB extraction keeps every candidate at or above this confidence. It is
# N-independent and also the lowest confidence threshold replay can use
# without re-running inference.
EXTRACTION_CONFIDENCE_FLOOR = 0.01
# Per-animal crops are materialised and run in chunks of this many detections
# so a 1024-detection frame does not spike memory.
DOWNSTREAM_CHUNK_SIZE = 256
# MEMORY-ESTIMATE ASSUMPTION, NOT A DETECTION CAP: it never drops a detection.
# Batch admission (stages/slicing.py ``estimated_prediction_job_bytes``) sizes
# a segment model's dense masks for at most this many masks per predicted item
# (tile / crop / frame). The compact term still counts every candidate up to
# the per-frame limit. Masks are emitted at the model's letterbox resolution
# (ultralytics ``process_mask(..., upsample=True)``; measured: 24 masks at
# imgsz 1024 = 25.2 MB uint8), so budgeting all 1025 candidate slots at full
# resolution (~1 GiB per 1024 px item) collapsed every segment batch to 1.
# 64 full-resolution masks is the byte equivalent of 1025 masks at prototype
# resolution ((imgsz/4)^2), and keeps a 26-animal sliced segment run at
# imgsz 1024 / 256 MiB admitting >= the pre-N-free 3 tiles per call.
DENSE_MASK_ESTIMATE_CANDIDATES = 64


class DetectionLimitError(ValueError):
    """N (number of animals) exceeds MAX_DETECTIONS_PER_FRAME."""


def require_target_count_within_limit(n: int) -> int:
    n = int(n)
    if n > MAX_DETECTIONS_PER_FRAME:
        raise DetectionLimitError(
            f"Number of animals N={n} exceeds the hard limit of "
            f"{MAX_DETECTIONS_PER_FRAME} detections per frame. Reduce the "
            "number of animals (or arenas x animals per arena)."
        )
    return n


@dataclass
class DetectionLimitStats:
    """Run-scoped record of frames that hit MAX_DETECTIONS_PER_FRAME."""

    frames: list[tuple[int, int]] = field(default_factory=list)

    def record(
        self, frame_idx: int, candidate_count: int, criterion: str = "confidence"
    ) -> None:
        self.frames.append((int(frame_idx), int(candidate_count)))
        logger.warning(
            "Frame %d produced %d detection candidates; the hard limit is %d "
            "per frame -- kept the top %d by %s, dropped the rest.",
            frame_idx,
            candidate_count,
            MAX_DETECTIONS_PER_FRAME,
            MAX_DETECTIONS_PER_FRAME,
            criterion,
        )

    def summary(self) -> str | None:
        if not self.frames:
            return None
        worst = max(c for _, c in self.frames)
        first = ", ".join(str(f) for f, _ in self.frames[:10])
        more = "" if len(self.frames) <= 10 else f" (+{len(self.frames) - 10} more)"
        return (
            f"{len(self.frames)} frame(s) hit the {MAX_DETECTIONS_PER_FRAME}-"
            f"detection-per-frame limit (worst: {worst} candidates); frames: "
            f"{first}{more}. Detections beyond the limit were dropped."
        )
