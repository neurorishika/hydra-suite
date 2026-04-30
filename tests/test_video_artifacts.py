from __future__ import annotations

import os
from pathlib import Path

from tests.helpers.module_loader import load_src_module

mod = load_src_module(
    "hydra_suite/utils/video_artifacts.py",
    "video_artifacts_under_test",
)


def test_detection_cache_path_uses_video_cache_subdirectory(tmp_path: Path) -> None:
    video_path = tmp_path / "clip.mp4"
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()

    cache_path = mod.build_detection_cache_path(
        str(video_path),
        "model123",
        artifact_base_dir=artifact_root,
    )

    assert cache_path == (
        artifact_root / "clip_caches" / "clip_detection_cache_model123.npz"
    )


def test_tracking_log_path_uses_video_log_subdirectory(tmp_path: Path) -> None:
    video_path = tmp_path / "clip.mp4"
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()

    log_path = mod.build_tracking_session_log_path(
        str(video_path),
        "20260308_120000",
        artifact_base_dir=artifact_root,
    )

    assert log_path == (
        artifact_root / "clip_logs" / "clip_tracking_20260308_120000.log"
    )


def test_iter_detection_cache_candidates_scans_new_and_legacy_locations(
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "clip.mp4"
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()

    legacy_path = mod.build_legacy_detection_cache_path(
        str(video_path),
        "legacy",
        artifact_base_dir=artifact_root,
    )
    legacy_path.write_bytes(b"legacy")

    new_path = mod.build_detection_cache_path(
        str(video_path),
        "newer",
        artifact_base_dir=artifact_root,
        create_dir=True,
    )
    new_path.write_bytes(b"newer")
    os.utime(
        new_path, (legacy_path.stat().st_atime + 1, legacy_path.stat().st_mtime + 1)
    )

    candidates = mod.iter_detection_cache_candidates(
        str(video_path),
        artifact_base_dirs=[artifact_root],
    )

    assert candidates[0] == new_path
    assert legacy_path in candidates


def test_find_existing_detection_cache_path_prefers_new_layout(tmp_path: Path) -> None:
    video_path = tmp_path / "clip.mp4"
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()

    legacy_path = mod.build_legacy_detection_cache_path(
        str(video_path),
        "model123",
        artifact_base_dir=artifact_root,
    )
    legacy_path.write_bytes(b"legacy")

    new_path = mod.build_detection_cache_path(
        str(video_path),
        "model123",
        artifact_base_dir=artifact_root,
        create_dir=True,
    )
    new_path.write_bytes(b"new")

    resolved = mod.find_existing_detection_cache_path(
        str(video_path),
        "model123",
        artifact_base_dirs=[artifact_root],
    )

    assert resolved == new_path


# ---- AprilTag cache path ----


def test_apriltag_cache_path_uses_video_cache_subdirectory(tmp_path: Path) -> None:
    video_path = tmp_path / "clip.mp4"
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()

    cache_path = mod.build_apriltag_cache_path(
        str(video_path),
        "at_hash123",
        0,
        100,
        artifact_base_dir=artifact_root,
    )

    assert cache_path == (
        artifact_root / "clip_caches" / "clip_apriltag_cache_at_hash123_0_100.npz"
    )


def test_find_existing_apriltag_cache_path(tmp_path: Path) -> None:
    video_path = tmp_path / "clip.mp4"
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()

    cache_path = mod.build_apriltag_cache_path(
        str(video_path),
        "hash_abc",
        10,
        200,
        artifact_base_dir=artifact_root,
        create_dir=True,
    )
    cache_path.write_bytes(b"data")

    found = mod.find_existing_apriltag_cache_path(
        str(video_path),
        "hash_abc",
        10,
        200,
        artifact_base_dirs=[artifact_root],
    )
    assert found == cache_path

    not_found = mod.find_existing_apriltag_cache_path(
        str(video_path),
        "hash_xyz",
        10,
        200,
        artifact_base_dirs=[artifact_root],
    )
    assert not_found is None


# ---- Classify cache path ----


def test_classify_cache_path_uses_video_cache_subdirectory(tmp_path: Path) -> None:
    video_path = tmp_path / "clip.mp4"
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()

    cache_path = mod.build_classify_cache_path(
        str(video_path),
        "cl_hash456",
        "individual_id",
        0,
        100,
        artifact_base_dir=artifact_root,
    )

    assert cache_path == (
        artifact_root
        / "clip_caches"
        / "clip_classify_cache_individual_id_cl_hash456_0_100.npz"
    )


def test_classify_cache_path_does_not_nest_when_base_is_existing_cache_dir(
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "clip.mp4"
    cache_dir = tmp_path / "clip_caches"
    cache_dir.mkdir()

    cache_path = mod.build_classify_cache_path(
        str(video_path),
        "cl_hash456",
        "individual_id",
        0,
        100,
        artifact_base_dir=cache_dir,
    )

    assert cache_path == (
        cache_dir / "clip_classify_cache_individual_id_cl_hash456_0_100.npz"
    )


def test_find_existing_classify_cache_path(tmp_path: Path) -> None:
    video_path = tmp_path / "clip.mp4"
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()

    cache_path = mod.build_classify_cache_path(
        str(video_path),
        "hash_def",
        "age",
        5,
        50,
        artifact_base_dir=artifact_root,
        create_dir=True,
    )
    cache_path.write_bytes(b"data")

    found = mod.find_existing_classify_cache_path(
        str(video_path),
        "hash_def",
        "age",
        5,
        50,
        artifact_base_dirs=[artifact_root],
    )
    assert found == cache_path

    not_found = mod.find_existing_classify_cache_path(
        str(video_path),
        "hash_other",
        "age",
        5,
        50,
        artifact_base_dirs=[artifact_root],
    )
    assert not_found is None
