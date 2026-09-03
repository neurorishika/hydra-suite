"""Regression tests for bounded, crash-resumable inference cache storage."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hydra_suite.core.inference.cache import chunked
from hydra_suite.core.inference.cache.base import CACHE_SCHEMA_VERSION, CacheKey
from hydra_suite.core.inference.cache.store import (
    AprilTagCacheHandle,
    CNNCacheHandle,
    DetectionCacheHandle,
    HeadTailCacheHandle,
    PoseCacheHandle,
    _npz_save,
)
from hydra_suite.core.inference.result import (
    AprilTagResult,
    CNNDetectionPrediction,
    CNNFactorPrediction,
    OBBResult,
)


def _key() -> CacheKey:
    return CacheKey(CACHE_SCHEMA_VERSION, "/model.pt", 1.25, "config")


def _obb(frame: int, count: int = 2) -> OBBResult:
    return OBBResult(
        frame_idx=frame,
        centroids=np.full((count, 2), frame, np.float32),
        angles=np.arange(count, dtype=np.float32),
        sizes=np.full(count, 20.0, np.float32),
        shapes=np.full((count, 2), 2.0, np.float32),
        confidences=np.full(count, 0.8, np.float32),
        corners=np.full((count, 4, 2), frame, np.float32),
        detection_ids=OBBResult.make_detection_ids(frame, count),
        class_ids=np.full(count, frame % 3, np.int64),
    )


def _cnn(frame: int) -> list[CNNDetectionPrediction]:
    return [
        CNNDetectionPrediction(
            det_index=frame,
            factors=[
                CNNFactorPrediction(
                    factor_name="color",
                    class_names=["red", "blue"],
                    raw_probabilities=np.asarray([0.25, 0.75], np.float32),
                )
            ],
        )
    ]


def test_detection_round_trip_across_chunks_and_explicit_empty(tmp_path):
    path = tmp_path / "detection.npz"
    writer = DetectionCacheHandle(path, _key(), chunk_size=2)
    for frame, count in enumerate((2, 1, 0, 3, 1)):
        writer.write_frame(frame, result=_obb(frame, count))
        assert len(writer._buffer) < 2
    writer.close()

    with np.load(path, allow_pickle=False) as manifest:
        assert (
            int(manifest["chunked_format_version"][0]) == chunked.CHUNK_FORMAT_VERSION
        )
    assert len(list((tmp_path / "detection.npz.chunks").rglob("chunk-*.npz"))) == 3

    reader = DetectionCacheHandle(path, _key())
    assert reader.covers_frame_range(0, 4)
    assert reader.read_frame(2) is not None
    assert reader.read_frame(2).num_detections == 0
    assert reader.read_frame(8) is None
    assert reader.read_frame(3).class_ids.tolist() == [0, 0, 0]


def test_every_downstream_cache_round_trips_multiple_chunks_and_empty(tmp_path):
    key = _key()
    headtail = HeadTailCacheHandle(tmp_path / "headtail.npz", key, chunk_size=1)
    cnn = CNNCacheHandle(tmp_path / "cnn_id.npz", key, "id", chunk_size=1)
    pose = PoseCacheHandle(tmp_path / "pose.npz", key, chunk_size=1)
    tags = AprilTagCacheHandle(tmp_path / "apriltag.npz", key, chunk_size=1)

    for frame, count in ((0, 1), (1, 0), (2, 1)):
        det = np.arange(count, dtype=np.int32)
        headtail.write_frame(
            frame,
            det_indices=det,
            heading_hints=np.full(count, frame + 0.5, np.float32),
            heading_confidences=np.full(count, 0.9, np.float32),
            directed_mask=np.ones(count, np.uint8),
        )
        cnn.write_frame(frame, predictions=_cnn(frame) if count else [])
        pose.write_frame(
            frame,
            det_indices=det,
            keypoints=np.full((count, 2, 3), frame, np.float32),
            valid_mask=np.ones(count, bool),
        )
        tags.write_frame(
            frame,
            result=AprilTagResult(
                tag_ids=np.arange(count, dtype=np.int32),
                det_indices=det,
                centers=np.full((count, 2), frame, np.float32),
                corners=np.full((count, 4, 2), frame, np.float32),
            ),
        )
    for handle in (headtail, cnn, pose, tags):
        handle.close()

    ht_reader = HeadTailCacheHandle(tmp_path / "headtail.npz", key)
    cnn_reader = CNNCacheHandle(tmp_path / "cnn_id.npz", key, "id")
    pose_reader = PoseCacheHandle(tmp_path / "pose.npz", key)
    tag_reader = AprilTagCacheHandle(tmp_path / "apriltag.npz", key)
    assert ht_reader.read_frame(1)[0].size == 0
    assert cnn_reader.read_frame(1) == []
    assert pose_reader.read_frame(1)[0].shape == (0, 2, 3)
    assert len(tag_reader.read_frame(1).tag_ids) == 0
    assert ht_reader.read_frame(99) is None
    assert cnn_reader.read_frame(2)[0].det_index == 2
    assert pose_reader.read_frame(2)[2].tolist() == [True]
    assert tag_reader.read_frame(2).centers.tolist() == [[2.0, 2.0]]


def test_random_indexed_read_opens_only_requested_payload_chunks(tmp_path, monkeypatch):
    path = tmp_path / "detection.npz"
    writer = DetectionCacheHandle(path, _key(), chunk_size=2)
    for frame in range(6):
        writer.write_frame(frame, result=_obb(frame, 1))
    writer.close()

    original_load = chunked.np.load
    opened: list[Path] = []

    def recording_load(path_arg, *args, **kwargs):
        candidate = Path(path_arg)
        if candidate.parent.name.startswith(tuple("0123456789abcdef")):
            opened.append(candidate)
        return original_load(path_arg, *args, **kwargs)

    monkeypatch.setattr(chunked.np, "load", recording_load)
    reader = DetectionCacheHandle(path, _key())
    assert reader.read_frame(0).frame_idx == 0
    assert reader.read_frame(1).frame_idx == 1  # same chunk, no reopen
    assert reader.read_frame(5).frame_idx == 5
    assert [item.name.split("-", 2)[:2] for item in opened] == [
        ["chunk", "00000000"],
        ["chunk", "00000002"],
    ]


def test_resume_appends_only_new_chunk_and_keeps_covered_empty_frames(tmp_path):
    path = tmp_path / "detection.npz"
    first = DetectionCacheHandle(path, _key(), chunk_size=2)
    first.write_frame(0, result=_obb(0, 1))
    first.write_frame(1, result=_obb(1, 0))
    first.close()

    resumed = DetectionCacheHandle(path, _key(), chunk_size=2)
    assert resumed.get_missing_frames(0, 3) == [2, 3]
    resumed.write_frame(2, result=_obb(2, 1))
    resumed.write_frame(3, result=_obb(3, 1))
    resumed.close()

    reader = DetectionCacheHandle(path, _key())
    assert reader.covers_frame_range(0, 3)
    assert reader.read_frame(1).num_detections == 0
    assert len(list((tmp_path / "detection.npz.chunks").rglob("chunk-*.npz"))) == 2


def test_renamed_but_unpublished_chunk_is_not_claimed_complete(tmp_path, monkeypatch):
    path = tmp_path / "detection.npz"
    writer = DetectionCacheHandle(path, _key(), chunk_size=1)

    def fail_manifest(_entries):
        raise RuntimeError("simulated crash before manifest promotion")

    monkeypatch.setattr(writer._store, "_publish_manifest", fail_manifest)
    with pytest.raises(RuntimeError, match="manifest promotion"):
        writer.write_frame(0, result=_obb(0, 1))

    assert len(list((tmp_path / "detection.npz.chunks").rglob("chunk-*.npz"))) == 1
    recovered = DetectionCacheHandle(path, _key())
    assert not recovered.is_valid()
    assert recovered.read_frame(0) is None


def test_failed_new_generation_keeps_previous_manifest_honest(tmp_path, monkeypatch):
    path = tmp_path / "detection.npz"
    first = DetectionCacheHandle(path, _key(), chunk_size=1)
    first.write_frame(0, result=_obb(0, 1))
    first.close()
    published = path.read_bytes()

    replacement = DetectionCacheHandle(path, _key(), chunk_size=1)

    def fail_manifest(_entries):
        raise RuntimeError("simulated crash after replacement chunk rename")

    monkeypatch.setattr(replacement._store, "_publish_manifest", fail_manifest)
    replacement.write_frame(0, result=_obb(0, 2))
    with pytest.raises(RuntimeError, match="replacement chunk rename"):
        replacement.close()

    assert path.read_bytes() == published
    recovered = DetectionCacheHandle(path, _key())
    assert recovered.is_valid()
    assert recovered.read_frame(0).num_detections == 1


def test_deliberate_recompute_starts_new_generation_not_duplicate_append(tmp_path):
    path = tmp_path / "detection.npz"
    first = DetectionCacheHandle(path, _key(), chunk_size=1)
    first.write_frame(0, result=_obb(0, 1))
    first.close()

    replacement = DetectionCacheHandle(path, _key(), chunk_size=1)
    replacement.write_frame(0, result=_obb(0, 3))
    replacement.close()

    reader = DetectionCacheHandle(path, _key())
    assert reader.read_frame(0).num_detections == 3
    assert len(reader._store._entries) == 1
    # Old generation remains recoverable on disk until an explicit cache
    # cleanup; the current manifest references only the replacement.
    assert len(list((tmp_path / "detection.npz.chunks").glob("*"))) == 2


def test_failed_chunk_rename_leaves_no_manifest_or_complete_chunk(
    tmp_path, monkeypatch
):
    path = tmp_path / "detection.npz"
    monkeypatch.setattr(
        chunked,
        "_exclusive_npz_save",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            OSError("simulated crash before chunk publish")
        ),
    )
    writer = DetectionCacheHandle(path, _key(), chunk_size=1)
    with pytest.raises(OSError, match="chunk publish"):
        writer.write_frame(0, result=_obb(0, 1))
    assert not path.exists()
    assert not list(tmp_path.rglob("chunk-*.npz"))


def test_truncated_referenced_chunk_invalidates_cache(tmp_path):
    path = tmp_path / "detection.npz"
    writer = DetectionCacheHandle(path, _key(), chunk_size=1)
    writer.write_frame(0, result=_obb(0, 1))
    writer.close()
    payload = next((tmp_path / "detection.npz.chunks").rglob("chunk-*.npz"))
    payload.write_bytes(payload.read_bytes()[:32])

    reader = DetectionCacheHandle(path, _key())
    assert not reader.is_valid()
    assert reader.read_frame(0) is None


def test_legacy_detection_npz_retains_read_parity(tmp_path):
    path = tmp_path / "detection.npz"
    result = _obb(4, 2)
    _npz_save(
        path,
        _key(),
        frame_indices=np.asarray([4, 4], np.int32),
        written_frames=np.asarray([4, 5], np.int32),
        centroids=result.centroids,
        angles=result.angles,
        sizes=result.sizes,
        shapes=result.shapes,
        confidences=result.confidences,
        corners=result.corners,
        detection_ids=result.detection_ids,
        class_ids=result.class_ids,
    )
    reader = DetectionCacheHandle(path, _key())
    assert reader.is_valid()
    assert reader._store.is_legacy
    assert reader.read_frame(4).detection_ids.tolist() == [40000, 40001]
    assert reader.read_frame(5).num_detections == 0
    assert reader.read_frame(6) is None


def test_all_legacy_downstream_npz_types_retain_read_parity(tmp_path):
    key = _key()
    common = {
        "frame_indices": np.asarray([3], np.int32),
        "written_frames": np.asarray([3, 4], np.int32),
    }
    _npz_save(
        tmp_path / "headtail.npz",
        key,
        **common,
        det_indices=np.asarray([0], np.int32),
        heading_hints=np.asarray([1.25], np.float32),
        heading_confidences=np.asarray([0.8], np.float32),
        directed_mask=np.asarray([1], np.uint8),
    )
    _npz_save(
        tmp_path / "cnn.npz",
        key,
        **common,
        det_indices=np.asarray([0], np.int32),
        factor_names_json=np.asarray(['["color"]']),
        class_names_json=np.asarray(['[["red", "blue"]]']),
        class_counts=np.asarray([2], np.int32),
        probabilities=np.asarray([[[0.2, 0.8]]], np.float32),
    )
    _npz_save(
        tmp_path / "pose.npz",
        key,
        **common,
        det_indices=np.asarray([0], np.int32),
        keypoints=np.ones((1, 2, 3), np.float32),
        valid_mask=np.asarray([1], np.uint8),
    )
    _npz_save(
        tmp_path / "apriltag.npz",
        key,
        **common,
        tag_ids=np.asarray([7], np.int32),
        det_indices=np.asarray([0], np.int32),
        centers=np.ones((1, 2), np.float32),
        corners=np.ones((1, 4, 2), np.float32),
    )

    assert HeadTailCacheHandle(tmp_path / "headtail.npz", key).read_frame(3)[1][
        0
    ] == pytest.approx(1.25)
    assert CNNCacheHandle(tmp_path / "cnn.npz", key, "id").read_frame(3)[0].factors[
        0
    ].class_names == ["red", "blue"]
    assert PoseCacheHandle(tmp_path / "pose.npz", key).read_frame(3)[0].shape == (
        1,
        2,
        3,
    )
    assert AprilTagCacheHandle(tmp_path / "apriltag.npz", key).read_frame(
        3
    ).tag_ids.tolist() == [7]
    assert (
        HeadTailCacheHandle(tmp_path / "headtail.npz", key).read_frame(4)[0].size == 0
    )


def test_write_concatenation_never_exceeds_configured_chunk(tmp_path, monkeypatch):
    import hydra_suite.core.inference.cache.store as store

    real_concatenate = store.np.concatenate
    largest_input_list = 0

    def bounded_concatenate(parts, *args, **kwargs):
        nonlocal largest_input_list
        largest_input_list = max(largest_input_list, len(parts))
        return real_concatenate(parts, *args, **kwargs)

    monkeypatch.setattr(store.np, "concatenate", bounded_concatenate)
    writer = DetectionCacheHandle(tmp_path / "detection.npz", _key(), chunk_size=8)
    for frame in range(1000):
        writer.write_frame(frame, result=_obb(frame, 2))
        assert len(writer._buffer) < 8
    writer.close()

    assert largest_input_list <= 8
    assert len(writer._store._entries) == 125


def test_chunked_cnn_export_consumer_reads_all_chunks(tmp_path):
    from hydra_suite.core.individual.properties.export import (
        build_detected_cnn_lookup_dataframe_from_cache,
    )

    path = tmp_path / "cnn_id.npz"
    writer = CNNCacheHandle(path, _key(), "id", chunk_size=1)
    writer.write_frame(2, predictions=_cnn(2))
    writer.write_frame(5, predictions=_cnn(5))
    writer.close()

    frame = build_detected_cnn_lookup_dataframe_from_cache(str(path), "id")
    assert frame["_cnn_frame_id"].tolist() == [2, 5]
    assert frame["CNN_id_Class"].tolist() == ["blue", "blue"]


def test_cache_key_mismatch_does_not_return_empty_result(tmp_path):
    path = tmp_path / "detection.npz"
    writer = DetectionCacheHandle(path, _key(), chunk_size=1)
    writer.write_frame(0, result=_obb(0, 0))
    writer.close()
    wrong = CacheKey(CACHE_SCHEMA_VERSION, "/other.pt", 1.25, "config")
    reader = DetectionCacheHandle(path, wrong)
    assert not reader.is_valid()
    assert reader.read_frame(0) is None


def _write_manifest(path: Path, *, session_id: str, entries: list[dict], **overrides):
    arrays = {
        "chunked_format_version": np.asarray([chunked.CHUNK_FORMAT_VERSION], np.int32),
        "cache_kind": np.asarray(["detection"]),
        "cache_key": np.asarray([_key().as_string()]),
        "session_id": np.asarray([session_id]),
        "generation_id": np.asarray(["test-generation"]),
        "chunks_json": np.asarray([json.dumps(entries)]),
    }
    arrays.update(overrides)
    chunked._atomic_npz_save(path, **arrays)


def test_manifest_rejects_empty_scalar_arrays_instead_of_raising(tmp_path):
    path = tmp_path / "detection.npz"
    _write_manifest(
        path,
        session_id="safe-session",
        entries=[],
        chunked_format_version=np.asarray([], np.int32),
    )

    reader = DetectionCacheHandle(path, _key())
    assert reader.is_valid() is False


@pytest.mark.parametrize("session_id", [".", "..", "a/b", "a\\b", ""])
def test_manifest_rejects_unsafe_session_ids(tmp_path, monkeypatch, session_id):
    path = tmp_path / "detection.npz"
    checked_components = []
    real_safe_component = chunked._safe_component

    def recording_safe_component(value):
        checked_components.append(value)
        return real_safe_component(value)

    monkeypatch.setattr(chunked, "_safe_component", recording_safe_component)
    _write_manifest(path, session_id=session_id, entries=[])
    assert DetectionCacheHandle(path, _key()).is_valid() is False
    assert session_id in checked_components


@pytest.mark.parametrize(
    ("ranges", "expected_error"),
    [
        ([[-1, 0]], "out of bounds"),
        ([[4, 3]], "out of bounds"),
        ([[0, 2], [2, 4]], "ordered and disjoint"),
        ([[0, 10**12]], "out of bounds"),
        ([], "nonempty list"),
    ],
)
def test_manifest_rejects_invalid_or_attacker_sized_ranges(
    tmp_path, monkeypatch, ranges, expected_error
):
    path = tmp_path / "detection.npz"
    session = "safe-session"
    chunk_dir = tmp_path / "detection.npz.chunks" / session
    payload_path = chunk_dir / f"chunk-00000000-{'0' * 32}.npz"
    chunked._atomic_npz_save(
        payload_path,
        written_frames=np.asarray([0], np.int64),
        frame_indices=np.asarray([0], np.int64),
        marker=np.asarray([1], np.int64),
    )
    entry = {
        "name": payload_path.name,
        "ranges": ranges,
        "byte_size": payload_path.stat().st_size,
        "sha256": chunked._sha256_file(payload_path),
    }
    observed_errors = []
    real_from_dict = chunked.ChunkEntry.from_dict.__func__

    def recording_from_dict(cls, raw):
        try:
            return real_from_dict(cls, raw)
        except ValueError as exc:
            observed_errors.append(str(exc))
            raise

    monkeypatch.setattr(
        chunked.ChunkEntry, "from_dict", classmethod(recording_from_dict)
    )
    _write_manifest(
        path,
        session_id=session,
        entries=[entry],
    )

    reader = DetectionCacheHandle(path, _key())
    assert reader.is_valid() is False
    assert reader.contains_frame(0) is False
    assert any(expected_error in error for error in observed_errors)


def test_contains_frame_uses_index_without_expanding_coverage(tmp_path):
    path = tmp_path / "detection.npz"
    writer = DetectionCacheHandle(path, _key(), chunk_size=2)
    writer.write_frame(10, result=_obb(10, 0))
    writer.write_frame(11, result=_obb(11, 1))
    writer.close()

    reader = DetectionCacheHandle(path, _key(), read_only=True)
    assert reader.contains_frame(10)
    assert reader.contains_frame(11)
    assert not reader.contains_frame(12)


def test_iter_arrays_emits_only_winning_rows_from_overlapping_chunks(tmp_path):
    store = chunked.ChunkedArrayStore(tmp_path / "detection.npz", _key(), "detection")

    def payload(frames, markers):
        count = len(frames)
        return {
            "frame_indices": np.asarray(frames),
            "centroids": np.column_stack([markers, markers]).astype(np.float32),
            "angles": np.zeros(count, np.float32),
            "sizes": np.ones(count, np.float32),
            "shapes": np.ones((count, 2), np.float32),
            "confidences": np.ones(count, np.float32),
            "corners": np.zeros((count, 4, 2), np.float32),
            "detection_ids": np.asarray(markers, np.int64),
            "class_ids": np.zeros(count, np.int64),
        }

    store.append_chunk([0], payload([0], [0]))
    store.append_chunk([2], payload([2], [2]))
    store.append_chunk([1, 2], payload([1, 2], [11, 12]))

    rows = [
        int(v) for arrays in store.iter_chunk_arrays() for v in arrays["detection_ids"]
    ]
    assert rows == [0, 11, 12]


def test_same_size_chunk_corruption_fails_deep_reuse_validation(tmp_path):
    path = tmp_path / "detection.npz"
    writer = DetectionCacheHandle(path, _key(), chunk_size=1)
    writer.write_frame(0, result=_obb(0, 1))
    writer.close()
    payload = next((tmp_path / "detection.npz.chunks").rglob("chunk-*.npz"))
    damaged = bytearray(payload.read_bytes())
    damaged[len(damaged) // 2] ^= 0x01
    payload.write_bytes(damaged)

    reader = DetectionCacheHandle(path, _key(), read_only=True)
    assert reader.is_valid() is True
    assert reader.is_reusable() is False


def test_deep_validation_rejects_wrong_kind_specific_shape_with_valid_checksum(
    tmp_path,
):
    path = tmp_path / "detection.npz"
    writer = DetectionCacheHandle(path, _key(), chunk_size=1)
    writer.write_frame(0, result=_obb(0, 1))
    writer.close()
    payload_path = next((tmp_path / "detection.npz.chunks").rglob("chunk-*.npz"))
    with np.load(payload_path, allow_pickle=False) as raw:
        arrays = {name: raw[name] for name in raw.files}
    arrays["corners"] = np.zeros((1, 3, 2), np.float32)
    chunked._atomic_npz_save(payload_path, **arrays)
    with np.load(path, allow_pickle=False) as raw:
        manifest = {name: raw[name] for name in raw.files}
    entries = json.loads(str(manifest["chunks_json"][0]))
    entries[0]["byte_size"] = payload_path.stat().st_size
    entries[0]["sha256"] = chunked._sha256_file(payload_path)
    manifest["chunks_json"] = np.asarray([json.dumps(entries)])
    chunked._atomic_npz_save(path, **manifest)

    reader = DetectionCacheHandle(path, _key(), read_only=True)
    assert reader.is_valid()
    assert not reader.is_reusable()
    assert reader.read_frame(0) is None


def test_zip_bomb_metadata_is_rejected_before_np_load(tmp_path, monkeypatch):
    path = tmp_path / "detection.npz"
    # This member declares >1 MiB but compresses to a few KiB, crossing the
    # expansion-ratio guard before NumPy can allocate it.
    np.savez_compressed(path, cache_key=np.zeros(3_000_000, np.uint8))

    def forbidden_load(*args, **kwargs):
        raise AssertionError("np.load must not run for rejected ZIP metadata")

    monkeypatch.setattr(chunked.np, "load", forbidden_load)
    assert DetectionCacheHandle(path, _key()).is_valid() is False


def test_manifest_rejects_duplicate_or_mispositioned_chunk_names(tmp_path):
    path = tmp_path / "detection.npz"
    writer = DetectionCacheHandle(path, _key(), chunk_size=1)
    writer.write_frame(0, result=_obb(0, 1))
    writer.write_frame(1, result=_obb(1, 1))
    writer.close()
    with np.load(path, allow_pickle=False) as raw:
        manifest = {name: raw[name] for name in raw.files}
    entries = json.loads(str(manifest["chunks_json"][0]))
    entries[1]["name"] = entries[0]["name"]
    manifest["chunks_json"] = np.asarray([json.dumps(entries)])
    chunked._atomic_npz_save(path, **manifest)

    assert DetectionCacheHandle(path, _key()).is_valid() is False


def test_chunk_name_collision_retries_without_overwriting(tmp_path, monkeypatch):
    real_save = chunked._exclusive_npz_save
    attempts: list[Path] = []

    def collide_once(path, **arrays):
        attempts.append(Path(path))
        if len(attempts) == 1:
            raise FileExistsError("simulated concurrent orphan")
        real_save(path, **arrays)

    monkeypatch.setattr(chunked, "_exclusive_npz_save", collide_once)
    writer = DetectionCacheHandle(tmp_path / "detection.npz", _key(), chunk_size=1)
    writer.write_frame(0, result=_obb(0, 1))
    writer.close()

    assert len(attempts) == 2
    assert attempts[0].name != attempts[1].name
    assert DetectionCacheHandle(tmp_path / "detection.npz", _key()).is_reusable()


def test_checksumming_does_not_make_numeric_cnn_class_names_reusable(tmp_path):
    path = tmp_path / "cnn_id.npz"
    writer = CNNCacheHandle(path, _key(), "id", chunk_size=1)
    writer.write_frame(0, predictions=_cnn(0))
    writer.close()
    payload_path = next((tmp_path / "cnn_id.npz.chunks").rglob("chunk-*.npz"))
    with np.load(payload_path, allow_pickle=False) as raw:
        arrays = {name: raw[name] for name in raw.files}
    arrays["class_names_json"] = np.asarray(["[[1, 2]]"])
    chunked._atomic_npz_save(payload_path, **arrays)
    with np.load(path, allow_pickle=False) as raw:
        manifest = {name: raw[name] for name in raw.files}
    entries = json.loads(str(manifest["chunks_json"][0]))
    entries[0]["byte_size"] = payload_path.stat().st_size
    entries[0]["sha256"] = chunked._sha256_file(payload_path)
    manifest["chunks_json"] = np.asarray([json.dumps(entries)])
    chunked._atomic_npz_save(path, **manifest)

    assert not CNNCacheHandle(path, _key(), "id", read_only=True).is_reusable()


def test_pose_writer_rejects_keypoint_dimension_change_across_chunks(tmp_path):
    writer = PoseCacheHandle(tmp_path / "pose.npz", _key(), chunk_size=1)
    for frame, keypoint_count in ((0, 2), (1, 3)):
        if frame == 1:
            with pytest.raises(ValueError, match="keypoint dimensions"):
                writer.write_frame(
                    frame,
                    det_indices=np.asarray([0], np.int32),
                    keypoints=np.zeros((1, keypoint_count, 3), np.float32),
                    valid_mask=np.ones(1, np.uint8),
                )
            break
        writer.write_frame(
            frame,
            det_indices=np.asarray([0], np.int32),
            keypoints=np.zeros((1, keypoint_count, 3), np.float32),
            valid_mask=np.ones(1, np.uint8),
        )


def test_pose_cache_rejects_keypoint_dimension_change_across_chunks(tmp_path):
    path = tmp_path / "pose.npz"
    store = chunked.ChunkedArrayStore(path, _key(), "pose")
    for frame, keypoint_count in ((0, 2), (1, 3)):
        store.append_chunk(
            [frame],
            {
                "frame_indices": np.asarray([frame], np.int64),
                "det_indices": np.asarray([0], np.int32),
                "keypoints": np.zeros((1, keypoint_count, 3), np.float32),
                "valid_mask": np.ones(1, np.uint8),
            },
        )

    assert not PoseCacheHandle(path, _key(), read_only=True).is_reusable()


def test_legacy_manifest_remains_visible_until_full_replacement_close(tmp_path):
    path = tmp_path / "detection.npz"
    legacy = _obb(9, 1)
    _npz_save(
        path,
        _key(),
        written_frames=np.asarray([9], np.int64),
        frame_indices=np.asarray([9], np.int64),
        centroids=legacy.centroids,
        angles=legacy.angles,
        sizes=legacy.sizes,
        shapes=legacy.shapes,
        confidences=legacy.confidences,
        corners=legacy.corners,
        detection_ids=legacy.detection_ids,
        class_ids=legacy.class_ids,
    )
    replacement = DetectionCacheHandle(path, _key(), chunk_size=1, write_mode="fresh")
    replacement.write_frame(0, result=_obb(0, 1))

    during = DetectionCacheHandle(path, _key(), read_only=True)
    assert during.is_legacy
    assert during.read_frame(9).num_detections == 1

    replacement.write_frame(1, result=_obb(1, 1))
    replacement.close()
    after = DetectionCacheHandle(path, _key(), read_only=True)
    assert not after.is_legacy
    assert after.covers_frame_range(0, 1)


def test_aborted_replacement_close_does_not_promote_partial_generation(tmp_path):
    path = tmp_path / "detection.npz"
    first = DetectionCacheHandle(path, _key(), chunk_size=1)
    first.write_frame(9, result=_obb(9, 1))
    first.close()

    replacement = DetectionCacheHandle(path, _key(), chunk_size=1, write_mode="fresh")
    replacement.write_frame(0, result=_obb(0, 1))
    replacement.close(commit_generation=False)

    reader = DetectionCacheHandle(path, _key(), read_only=True)
    assert reader.contains_frame(9)
    assert not reader.contains_frame(0)


def test_resume_mode_keeps_generation_when_first_frame_is_already_covered(tmp_path):
    path = tmp_path / "detection.npz"
    first = DetectionCacheHandle(path, _key(), chunk_size=1)
    first.write_frame(0, result=_obb(0, 1))
    first.close()
    session = DetectionCacheHandle(path, _key())._store._session_id

    resumed = DetectionCacheHandle(path, _key(), chunk_size=1, write_mode="resume")
    resumed.write_frame(0, result=_obb(0, 2))
    resumed.write_frame(1, result=_obb(1, 1))
    resumed.close()

    reader = DetectionCacheHandle(path, _key())
    assert reader._store._session_id == session
    assert reader.read_frame(0).num_detections == 1
    assert reader.read_frame(1).num_detections == 1


def test_handle_flushes_on_bytes_before_frame_count(tmp_path):
    writer = DetectionCacheHandle(
        tmp_path / "detection.npz",
        _key(),
        chunk_size=100,
        max_buffer_bytes=3000,
    )
    writer.write_frame(0, result=_obb(0, 2))
    writer.write_frame(1, result=_obb(1, 2))
    writer.close()
    assert len(writer._store._entries) >= 2


def test_handle_rejects_one_payload_larger_than_buffer_budget(tmp_path):
    writer = DetectionCacheHandle(
        tmp_path / "detection.npz",
        _key(),
        max_buffer_bytes=64,
    )
    with pytest.raises(ValueError, match="cache frame payload exceeds"):
        writer.write_frame(0, result=_obb(0, 10))


def test_detection_write_rejects_misaligned_result(tmp_path):
    writer = DetectionCacheHandle(tmp_path / "detection.npz", _key())
    with pytest.raises(ValueError, match="frame_idx"):
        writer.write_frame(3, result=_obb(4, 1))


def test_detection_write_rejects_misaligned_array_lengths(tmp_path):
    writer = DetectionCacheHandle(tmp_path / "detection.npz", _key())
    result = _obb(3, 2)
    result.angles = result.angles[:1]
    with pytest.raises(ValueError, match="aligned lengths"):
        writer.write_frame(3, result=result)
