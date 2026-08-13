"""Tests for the "Clear All Caches" scan helpers.

``TrackingOrchestrator._iter_cache_artifact_paths`` yields legacy cache
*files* (globbed, deleted via ``unlink``); ``_iter_inference_cache_dirs``
yields cache *directories* (deleted via ``shutil.rmtree``). Together they
must cover:

- the modern ``.inference_cache_<stem>/`` dir (where live caches actually
  live now), via ``_iter_inference_cache_dirs``
- the stale legacy ``<stem>_caches/`` dir, retained strictly so users can
  still purge leftovers, via ``_iter_cache_artifact_paths``

Both helpers are ``@staticmethod`` on the orchestrator class, so they are
callable directly without constructing a live GUI / QApplication.
"""

from __future__ import annotations

from pathlib import Path

from hydra_suite.trackerkit.gui.orchestrators.tracking import TrackingOrchestrator


def test_iter_inference_cache_dirs_finds_modern_dir(tmp_path):
    video_path = tmp_path / "myvideo.mp4"
    video_path.write_bytes(b"")

    modern_dir = tmp_path / ".inference_cache_myvideo"
    modern_dir.mkdir()
    (modern_dir / "detection.npz").write_bytes(b"x")

    found = TrackingOrchestrator._iter_inference_cache_dirs(
        str(video_path), [str(tmp_path)]
    )

    assert modern_dir.resolve() in {p.resolve() for p in found}


def test_iter_cache_artifact_paths_finds_stale_legacy_dir_files():
    """Legacy ``<stem>_caches/`` dir must still be scanned for deletion."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        video_path = tmp_path / "myvideo.mp4"
        video_path.write_bytes(b"")

        legacy_dir = tmp_path / "myvideo_caches"
        legacy_dir.mkdir()
        legacy_cache_file = legacy_dir / "myvideo_cache_abc.npz"
        legacy_cache_file.write_bytes(b"x")

        found = TrackingOrchestrator._iter_cache_artifact_paths(
            str(video_path), [str(tmp_path)]
        )

        assert legacy_cache_file.resolve() in {p.resolve() for p in found}


def test_clear_all_caches_scan_covers_modern_and_legacy(tmp_path):
    """The combined scan (as used by clear_detection_caches) must surface
    both the modern per-video cache dir AND the stale legacy cache dir.
    """
    video_path = tmp_path / "myvideo.mp4"
    video_path.write_bytes(b"")

    modern_dir = tmp_path / ".inference_cache_myvideo"
    modern_dir.mkdir()
    (modern_dir / "detection.npz").write_bytes(b"x")

    legacy_dir = tmp_path / "myvideo_caches"
    legacy_dir.mkdir()
    legacy_cache_file = legacy_dir / "myvideo_cache_abc.npz"
    legacy_cache_file.write_bytes(b"x")

    # A stray <stem>_logs/ dir should NOT be swept up by either cache scan —
    # these two helpers are strictly for *caches*, not logs.
    logs_dir = tmp_path / "myvideo_logs"
    logs_dir.mkdir()
    log_file = logs_dir / "myvideo_tracking_20260101.log"
    log_file.write_bytes(b"log")

    base_dirs = [str(tmp_path)]
    cache_files = TrackingOrchestrator._iter_cache_artifact_paths(
        str(video_path), base_dirs
    )
    cache_dirs = TrackingOrchestrator._iter_inference_cache_dirs(
        str(video_path), base_dirs
    )

    resolved_files = {p.resolve() for p in cache_files}
    resolved_dirs = {p.resolve() for p in cache_dirs}

    assert modern_dir.resolve() in resolved_dirs
    assert legacy_cache_file.resolve() in resolved_files
    assert log_file.resolve() not in resolved_files
    assert logs_dir.resolve() not in resolved_dirs
