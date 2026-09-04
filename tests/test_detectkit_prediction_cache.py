from __future__ import annotations

import pytest


def _key():
    from hydra_suite.core.inference.cache.base import CacheKey

    return CacheKey(4, "/model.pt", 1.0, "settings")


def _detection(confidence: float, vertices: int = 4) -> dict:
    return {
        "class_id": 2,
        "confidence": confidence,
        "polygon_px": [(float(i), float(i + 1)) for i in range(vertices)],
    }


def test_prediction_cache_round_trips_polygons_incrementally(tmp_path):
    from hydra_suite.detectkit.jobs.prediction_cache import (
        DatasetPredictionCache,
        DatasetPredictionWriter,
    )

    path = tmp_path / "predictions.npz"
    writer = DatasetPredictionWriter(path, _key(), chunk_size=1)
    writer.write_frame(0, [_detection(0.75, vertices=6)])
    assert path.is_file(), "first frame must be visible before the run completes"
    writer.write_frame(1, [])
    writer.close()

    cache = DatasetPredictionCache(path, _key())
    assert cache.is_valid()
    assert cache.read_frame(0) == [
        {
            "class_id": 2,
            "confidence": pytest.approx(0.75),
            "polygon_px": [(float(i), float(i + 1)) for i in range(6)],
        }
    ]
    assert cache.read_frame(1) == []
    assert cache.read_frame(2) is None


def test_prediction_cache_lru_is_fixed_as_source_grows(tmp_path):
    from hydra_suite.detectkit.jobs.prediction_cache import (
        DatasetPredictionCache,
        DatasetPredictionWriter,
    )

    path = tmp_path / "predictions.npz"
    writer = DatasetPredictionWriter(path, _key(), chunk_size=3)
    for frame in range(30):
        writer.write_frame(frame, [_detection(frame / 100)])
    writer.close()
    cache = DatasetPredictionCache(path, _key(), lru_frames=3)
    for frame in range(30):
        assert cache.read_frame(frame) is not None
        assert cache.retained_frame_count <= 3


def test_prediction_path_index_uses_bounded_random_reads(tmp_path):
    from hydra_suite.detectkit.jobs.prediction_cache import (
        PredictionPathIndex,
        write_path_index,
    )

    cache_path = tmp_path / "predictions.npz"
    paths = [tmp_path / f"images/frame-{index:05d}.jpg" for index in range(100)]
    write_path_index(cache_path, paths)
    index = PredictionPathIndex(cache_path)
    assert len(index) == 100
    assert index.path_at(72) == str(paths[72].resolve())
    assert index.index_of(paths[72]) == 72
    assert index.index_of(tmp_path / "missing.jpg") is None


def test_prediction_statistics_stream_chunks_without_retaining_all_frames(tmp_path):
    from hydra_suite.detectkit.jobs.prediction_cache import (
        DatasetPredictionCache,
        DatasetPredictionWriter,
    )

    path = tmp_path / "predictions.npz"
    writer = DatasetPredictionWriter(path, _key(), chunk_size=2)
    for frame in range(12):
        writer.write_frame(frame, [_detection(0.1 if frame % 2 else 0.9)])
    writer.close()
    cache = DatasetPredictionCache(path, _key(), lru_frames=2)
    stats = cache.statistics(0.5)
    assert stats["image_count"] == 12
    assert stats["detection_count"] == 6
    assert stats["class_counts"] == {2: 6}
    assert cache.retained_frame_count <= 2
