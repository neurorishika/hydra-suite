from __future__ import annotations

import pytest


def _key():
    from hydra_suite.core.inference.cache.base import CACHE_SCHEMA_VERSION, CacheKey

    return CacheKey(CACHE_SCHEMA_VERSION, "sha256:deadbeef", "settings")


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


def test_prediction_path_index_consumes_a_single_pass_iterator(tmp_path):
    from hydra_suite.detectkit.jobs.prediction_cache import (
        PredictionPathIndex,
        write_path_index,
    )

    cache_path = tmp_path / "predictions.npz"
    consumed = []

    def paths():
        for index in range(25):
            consumed.append(index)
            yield tmp_path / f"images/frame-{index:05d}.jpg"

    write_path_index(cache_path, paths())

    assert consumed == list(range(25))
    assert len(PredictionPathIndex(cache_path)) == 25


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


def test_prediction_writer_rejects_nonfinite_or_unbounded_frame_payload(tmp_path):
    from hydra_suite.detectkit.jobs.prediction_cache import (
        MAX_DETECTIONS_PER_FRAME,
        MAX_PREDICTION_CLASSES,
        DatasetPredictionWriter,
    )

    writer = DatasetPredictionWriter(tmp_path / "predictions.npz", _key())
    with pytest.raises(ValueError, match="non-finite"):
        writer.write_frame(0, [{**_detection(0.5), "confidence": float("nan")}])
    with pytest.raises(ValueError, match="count exceeds"):
        writer.write_frame(
            1, (_detection(0.5) for _ in range(MAX_DETECTIONS_PER_FRAME + 1))
        )
    with pytest.raises(ValueError, match="class id"):
        writer.write_frame(2, [{**_detection(0.5), "class_id": MAX_PREDICTION_CLASSES}])


def test_two_different_dataset_directories_get_different_source_ids(tmp_path):
    """Fix M6: source_path is a DATASET DIRECTORY (dataset_panel.py:394 does
    Path(source_path)/"images"), not a video file. video_signature(dir) hits
    IsADirectoryError -> swallowed to "" by its except OSError, so every
    source in a project would collapse onto the same identity. Prove the
    dispatch keeps them distinct."""
    from hydra_suite.detectkit.jobs.prediction_cache import prediction_cache_key

    a = tmp_path / "dataset_a" / "images"
    b = tmp_path / "dataset_b" / "images"
    a.mkdir(parents=True)
    b.mkdir(parents=True)
    (a / "1.png").write_bytes(b"aaa")
    (b / "1.png").write_bytes(b"bbbbbbb")
    # Fix V-minor: DIFFERENT-LENGTH content, not just different bytes at the
    # same length -- b"aaa" vs b"bbb" (both 3 bytes) makes distinctness ride
    # entirely on mtime_ns (both files are written back-to-back in the same
    # test, so their mtimes differ only by whatever the filesystem's mtime
    # resolution happens to catch), which is FLAKY on coarse-mtime
    # filesystems even though the docstring claims "different bytes" is what
    # is being tested. A different length changes size_bytes too, which
    # every fingerprint in this plan hashes unconditionally (no mtime
    # coalescing needed to see the difference).
    #
    # Fix B13: the real entry point is `prediction_cache_key(source_path,
    # model_paths, settings)` -- three POSITIONAL parameters
    # (`detectkit/jobs/prediction_cache.py:31-34`). There is no
    # `build_cache_key` anywhere in the repo.
    key_a = prediction_cache_key(str(tmp_path / "dataset_a"), [], {})
    key_b = prediction_cache_key(str(tmp_path / "dataset_b"), [], {})
    assert key_a.as_string() != key_b.as_string()


def test_editing_a_label_does_not_change_the_prediction_cache_key(tmp_path):
    """Fix A4: source_path is the DATASET ROOT, and labels/ (sibling of
    images/, edited by the reviewer on every save via dataset_panel.py:394)
    must NOT be part of the source identity. If it were, cache_path_for
    (prediction_cache.py:63-66) would derive a NEW on-disk filename from
    key.as_string() on every label save, orphaning the previous prediction
    cache file and growing artifacts/inference_cache/ without bound while the
    reviewer works -- with no test anywhere catching it before this one."""
    from hydra_suite.detectkit.jobs.prediction_cache import prediction_cache_key

    root = tmp_path / "dataset"
    (root / "images").mkdir(parents=True)
    (root / "images" / "1.png").write_bytes(b"aaa")
    (root / "labels").mkdir(parents=True)
    (root / "labels" / "1.json").write_text('{"boxes": []}')

    before = prediction_cache_key(str(root), [], {})
    (root / "labels" / "1.json").write_text('{"boxes": [[1, 2, 3, 4]]}')
    (root / "labels" / "2.json").write_text('{"boxes": []}')
    after = prediction_cache_key(str(root), [], {})

    assert before.as_string() == after.as_string()


def test_adding_an_image_does_change_the_prediction_cache_key(tmp_path):
    """The images/ subtree LISTING is still the real invalidation signal --
    fix A4 narrows the hash scope, it does not remove cache invalidation for
    the thing that actually should invalidate it."""
    from hydra_suite.detectkit.jobs.prediction_cache import prediction_cache_key

    root = tmp_path / "dataset"
    (root / "images").mkdir(parents=True)
    (root / "images" / "1.png").write_bytes(b"aaa")

    before = prediction_cache_key(str(root), [], {})
    (root / "images" / "2.png").write_bytes(b"bbb")
    after = prediction_cache_key(str(root), [], {})

    assert before.as_string() != after.as_string()
