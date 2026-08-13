from __future__ import annotations

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

    assert cache_path == (artifact_root / ".inference_cache_clip" / "detection.npz")


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


def test_iter_detection_cache_candidates_scans_modern_opt_dir_only(
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "clip.mp4"
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()

    # Legacy `<stem>_caches/` layout must no longer be scanned.
    legacy_cache_dir = mod.build_video_cache_dir(
        str(video_path), artifact_base_dir=artifact_root, create=True
    )
    legacy_path = (
        legacy_cache_dir / f"{mod._video_stem(video_path)}_detection_cache_legacy.npz"
    )
    legacy_path.write_bytes(b"legacy")

    # Modern `.inference_cache_<stem>/opt` directory must be scanned.
    opt_dir = mod.build_optimizer_detection_cache_path(
        str(video_path),
        "modelA",
        100,
        artifact_base_dir=artifact_root,
        create_dir=True,
    )
    (opt_dir / "detection.npz").write_bytes(b"newer")

    candidates = mod.iter_detection_cache_candidates(
        str(video_path),
        artifact_base_dirs=[artifact_root],
    )

    assert candidates == [opt_dir]


def test_individual_properties_cache_path_without_detection_cache_uses_modern_dir(
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "clip.mp4"
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()

    cache_path = mod.build_individual_properties_cache_path(
        str(video_path),
        "props1",
        0,
        10,
        detection_cache_path=None,
        artifact_base_dir=artifact_root,
    )

    assert cache_path.parent == artifact_root / ".inference_cache_clip"
    assert "clip_caches" not in cache_path.parts


def test_detected_properties_cache_path_without_detection_cache_uses_modern_dir(
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "clip.mp4"
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()

    cache_path = mod.build_detected_properties_cache_path(
        str(video_path),
        "props1",
        0,
        10,
        detection_cache_path=None,
        artifact_base_dir=artifact_root,
    )

    assert cache_path.parent == artifact_root / ".inference_cache_clip"
    assert "clip_caches" not in cache_path.parts


def test_find_existing_detection_cache_path_prefers_new_layout(tmp_path: Path) -> None:
    video_path = tmp_path / "clip.mp4"
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()

    legacy_path = (
        artifact_root / f"{mod._video_stem(video_path)}_detection_cache_model123.npz"
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
