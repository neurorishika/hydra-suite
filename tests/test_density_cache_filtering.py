from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from hydra_suite.core.inference.result import OBBResult
from hydra_suite.core.tracking.ingest.frame_result_bridge import (
    build_density_cache_dict,
)


def _obb(frame_idx: int, confidences: list[float]) -> OBBResult:
    count = len(confidences)
    return OBBResult(
        frame_idx=frame_idx,
        centroids=np.column_stack(
            [np.arange(count, dtype=np.float32), np.zeros(count, dtype=np.float32)]
        ),
        angles=np.zeros(count, dtype=np.float32),
        sizes=np.ones(count, dtype=np.float32),
        shapes=np.ones((count, 2), dtype=np.float32),
        confidences=np.asarray(confidences, dtype=np.float32),
        corners=np.zeros((count, 4, 2), dtype=np.float32),
        detection_ids=np.arange(count, dtype=np.int64),
    )


def test_density_cache_uses_runner_filtered_candidate_detections() -> None:
    """Density regions must see the same candidate filter as cached replay."""

    filtered = _obb(4, [0.9])

    class _DetectionCache:
        def is_valid(self) -> bool:
            return True

        def iter_covered_frames(self, start_frame: int, end_frame: int):
            assert (start_frame, end_frame) == (4, 4)
            return [4]

        def read_frame(self, _frame_idx: int):
            raise AssertionError("density builder must not consume unfiltered OBBs")

    class _Runner:
        cache_dir = "cache"
        _caches = SimpleNamespace(detection=_DetectionCache())

        def __init__(self) -> None:
            self.filtered_frames: list[int] = []

        def load_filtered_obb(self, frame_idx: int):
            self.filtered_frames.append(frame_idx)
            return filtered, np.array([1], dtype=np.int64)

    runner = _Runner()

    density_cache = build_density_cache_dict(runner, 4, 4)

    assert runner.filtered_frames == [4]
    measurements, confidences, sizes = density_cache[4]
    assert measurements.shape == (1, 3)
    np.testing.assert_allclose(confidences, [0.9])
    assert sizes.tolist() == [1.0]
