"""Identity Phase 5: locating the evidence sidecar from just the video path.

``find_identity_evidence_cache_path`` is the seam that lets post-processing
(``core.tracking.session.TrackingSessionCore``) thread the Phase-3 evidence
cache into the offline solver without needing the tracking worker's live
``InferenceRunner`` instance (whose sidecar filename embeds an internal
content-hash signature this seam intentionally does not try to
recompute -- see the function's docstring).
"""

from __future__ import annotations

import time

from hydra_suite.core.individual.identity.cache import find_identity_evidence_cache_path


def test_returns_none_when_cache_dir_missing(tmp_path):
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"")
    assert find_identity_evidence_cache_path(str(video_path)) is None


def test_finds_batch_sidecar(tmp_path):
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"")
    cache_dir = tmp_path / ".inference_cache_clip"
    cache_dir.mkdir()
    sidecar = cache_dir / "detection_identity_evidence_batch_abc123.npz"
    sidecar.write_bytes(b"")

    found = find_identity_evidence_cache_path(str(video_path))
    assert found == sidecar


def test_falls_back_to_live_when_no_batch_sidecar(tmp_path):
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"")
    cache_dir = tmp_path / ".inference_cache_clip"
    cache_dir.mkdir()
    sidecar = cache_dir / "detection_identity_evidence_live_xyz789.npz"
    sidecar.write_bytes(b"")

    found = find_identity_evidence_cache_path(str(video_path))
    assert found == sidecar


def test_prefers_batch_over_live_when_both_present(tmp_path):
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"")
    cache_dir = tmp_path / ".inference_cache_clip"
    cache_dir.mkdir()
    (cache_dir / "detection_identity_evidence_live_xyz789.npz").write_bytes(b"")
    batch_sidecar = cache_dir / "detection_identity_evidence_batch_abc123.npz"
    batch_sidecar.write_bytes(b"")

    found = find_identity_evidence_cache_path(str(video_path))
    assert found == batch_sidecar


def test_returns_most_recently_modified_match(tmp_path):
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"")
    cache_dir = tmp_path / ".inference_cache_clip"
    cache_dir.mkdir()
    older = cache_dir / "detection_identity_evidence_batch_old.npz"
    older.write_bytes(b"")
    time.sleep(0.01)
    newer = cache_dir / "detection_identity_evidence_batch_new.npz"
    newer.write_bytes(b"")

    found = find_identity_evidence_cache_path(str(video_path))
    assert found == newer


def test_returns_none_when_no_sidecar_matches(tmp_path):
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"")
    cache_dir = tmp_path / ".inference_cache_clip"
    cache_dir.mkdir()
    (cache_dir / "detection.npz").write_bytes(b"")

    assert find_identity_evidence_cache_path(str(video_path)) is None
