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

    def record(self, frame_idx: int, candidate_count: int) -> None:
        self.frames.append((int(frame_idx), int(candidate_count)))
        logger.warning(
            "Frame %d produced %d detection candidates; the hard limit is %d "
            "per frame -- kept the top %d by confidence, dropped the rest.",
            frame_idx,
            candidate_count,
            MAX_DETECTIONS_PER_FRAME,
            MAX_DETECTIONS_PER_FRAME,
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
