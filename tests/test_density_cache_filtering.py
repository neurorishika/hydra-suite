from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from hydra_suite.core.inference.result import OBBResult
from hydra_suite.core.tracking.confidence.confidence_density import (
    ConfidenceDensityCancelled,
)
from hydra_suite.core.tracking.ingest.frame_result_bridge import (
    build_density_cache_dict,
    density_cache_covered_frame_count,
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


def test_density_coverage_count_uses_metadata_without_loading_filtered_obbs() -> None:
    """Admission must inspect intervals, not decompress every candidate frame."""

    class _DetectionCache:
        def is_valid(self) -> bool:
            return True

        def coverage_ranges(self):
            return ((100, 104), (1_000_000, 1_000_004))

        def iter_covered_frames(self, *_args):
            raise AssertionError("coverage metadata should avoid frame iteration")

    class _Runner:
        cache_dir = "cache"
        _caches = SimpleNamespace(detection=_DetectionCache())

        def load_filtered_obb(self, _frame_idx: int):
            raise AssertionError("coverage preflight must not load filtered OBBs")

    assert density_cache_covered_frame_count(_Runner(), 100, 104) == 5


def test_density_legacy_coverage_preflight_uses_requested_span_without_payload() -> (
    None
):
    """Legacy NPZs have no range manifest, so admission stays conservative."""

    class _LegacyDetectionCache:
        is_legacy = True

        def is_valid(self) -> bool:
            return True

        def coverage_ranges(self):
            raise AssertionError("legacy coverage payload must not be materialized")

        def iter_covered_frames(self, *_args):
            raise AssertionError("legacy frame IDs must not be materialized")

    class _Runner:
        cache_dir = "cache"
        _caches = SimpleNamespace(detection=_LegacyDetectionCache())

    assert density_cache_covered_frame_count(_Runner(), 100, 104) == 5


def test_density_cache_collection_stops_before_loading_the_next_frame() -> None:
    """Cancellation must not return a partially materialized evidence cache."""

    stopped = {"value": False}

    class _DetectionCache:
        def is_valid(self) -> bool:
            return True

        def iter_covered_frames(self, start_frame: int, end_frame: int):
            assert (start_frame, end_frame) == (4, 5)
            return [4, 5]

    class _Runner:
        cache_dir = "cache"
        _caches = SimpleNamespace(detection=_DetectionCache())

        def __init__(self) -> None:
            self.loaded: list[int] = []

        def load_filtered_obb(self, frame_idx: int):
            self.loaded.append(frame_idx)
            stopped["value"] = True
            return _obb(frame_idx, [0.9]), np.array([1], dtype=np.int64)

    runner = _Runner()
    with pytest.raises(ConfidenceDensityCancelled):
        build_density_cache_dict(
            runner,
            4,
            5,
            should_stop=lambda: stopped["value"],
        )
    assert runner.loaded == [4]


def test_density_cache_collection_stops_after_the_final_payload_load() -> None:
    """A stop raised by the final filter call cannot leak partial evidence."""

    stopped = {"value": False}

    class _DetectionCache:
        def is_valid(self) -> bool:
            return True

        def iter_covered_frames(self, start_frame: int, end_frame: int):
            assert (start_frame, end_frame) == (4, 4)
            return [4]

    class _Runner:
        cache_dir = "cache"
        _caches = SimpleNamespace(detection=_DetectionCache())

        def __init__(self) -> None:
            self.loaded: list[int] = []

        def load_filtered_obb(self, frame_idx: int):
            self.loaded.append(frame_idx)
            stopped["value"] = True
            return _obb(frame_idx, [0.9]), np.array([1], dtype=np.int64)

    runner = _Runner()
    with pytest.raises(ConfidenceDensityCancelled):
        build_density_cache_dict(
            runner,
            4,
            4,
            should_stop=lambda: stopped["value"],
        )
    assert runner.loaded == [4]
