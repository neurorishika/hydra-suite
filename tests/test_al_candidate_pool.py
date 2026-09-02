"""Tests for hydra_suite.data.al.candidate_pool."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from hydra_suite.data.al.candidate_pool import CandidatePoolConfig, build_candidate_pool
from hydra_suite.data.al.frame_source import ImageFolderFrameSource, VideoFrameSource

_VIDEO_SIZE = (
    64,
    48,
)  # (width, height), matches tests/test_al_frame_source.py convention


def _write_video(path: Path, frames: list[np.ndarray]) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 10.0, _VIDEO_SIZE)
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()


def _make_dataset(tmp_path, n_unique: int, n_dupes: int) -> ImageFolderFrameSource:
    rng = np.random.default_rng(0)
    idx = 0
    for _ in range(n_unique):
        img = rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)
        cv2.imwrite(str(tmp_path / f"img_{idx:04d}.png"), img)
        idx += 1
        for _ in range(n_dupes):
            cv2.imwrite(str(tmp_path / f"img_{idx:04d}.png"), img)
            idx += 1
    return ImageFolderFrameSource(str(tmp_path))


def test_candidate_pool_drops_perceptual_duplicates(tmp_path):
    src = _make_dataset(tmp_path, n_unique=4, n_dupes=2)
    cfg = CandidatePoolConfig(dedup_method="phash", dedup_threshold=4)

    refs = build_candidate_pool(src, cfg)

    assert 4 <= len(refs) <= 6  # 4 unique kept; minor dedup bleed allowed
    assert len(refs) < src.length()


def test_candidate_pool_respects_max_candidates(tmp_path):
    src = _make_dataset(tmp_path, n_unique=10, n_dupes=0)
    cfg = CandidatePoolConfig(dedup_method="none", max_candidates=3)

    refs = build_candidate_pool(src, cfg)
    assert len(refs) == 3


def test_candidate_pool_stops_when_cancellation_is_requested(tmp_path):
    src = _make_dataset(tmp_path, n_unique=10, n_dupes=0)
    cfg = CandidatePoolConfig(dedup_method="none", max_candidates=None)
    checks = 0

    def should_stop() -> bool:
        nonlocal checks
        checks += 1
        return checks > 3

    refs = build_candidate_pool(src, cfg, should_stop=should_stop)

    assert len(refs) == 3
    assert checks == 4


def test_candidate_pool_no_dedup_passthrough(tmp_path):
    src = _make_dataset(tmp_path, n_unique=5, n_dupes=0)
    cfg = CandidatePoolConfig(dedup_method="none")

    refs = build_candidate_pool(src, cfg)
    assert len(refs) == 5


def test_candidate_pool_defaults_enable_bounded_dedup_and_motion_prefilter():
    cfg = CandidatePoolConfig()
    assert cfg.dedup_window is not None and cfg.dedup_window > 0
    assert cfg.motion_threshold > 0.0


@pytest.fixture
def synthetic_video_with_repeats(tmp_path) -> Path:
    """10 distinct frames, then a near-dup of frame 0 (far outside a small
    window), then a near-dup of frame 9 (inside a small window)."""
    rng = np.random.default_rng(42)
    w, h = _VIDEO_SIZE
    frames = [rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8) for _ in range(10)]
    frames.append(frames[0].copy())  # frame 10: near-dup of frame 0
    frames.append(frames[9].copy())  # frame 11: near-dup of frame 9

    video = tmp_path / "repeats.mp4"
    _write_video(video, frames)
    return video


@pytest.fixture
def synthetic_video_static_and_moving(tmp_path) -> Path:
    """20 identical (static) frames, followed by 5 frames with real motion."""
    rng = np.random.default_rng(7)
    w, h = _VIDEO_SIZE
    static_frame = rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)
    frames = [static_frame.copy() for _ in range(20)]
    frames.extend(
        rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8) for _ in range(5)
    )

    video = tmp_path / "static_and_moving.mp4"
    _write_video(video, frames)
    return video


def test_windowed_dedup_only_compares_against_recent_window(
    synthetic_video_with_repeats,
):
    source = VideoFrameSource(str(synthetic_video_with_repeats))
    # dedup_threshold=10 is calibrated to this fixture: lossy mp4v re-encoding
    # of the repeated random-noise frames yields a phash Hamming distance of
    # 10 between each repeat and its source frame, while frames 0-9 remain
    # mutually distinct well above this threshold (verified separately).
    pool_config = CandidatePoolConfig(dedup_window=3, dedup_threshold=10)
    candidates = build_candidate_pool(source, pool_config)
    ids = {c.frame_id for c in candidates}
    assert 10 in ids  # far-apart near-duplicate of frame 0 is NOT deduped away
    assert 11 not in ids  # near-duplicate within the window IS deduped away


def test_frame_difference_prefilter_skips_static_frames(
    synthetic_video_static_and_moving,
):
    source = VideoFrameSource(str(synthetic_video_static_and_moving))
    # dedup_threshold=-1 deliberately makes the plain perceptual-hash dedup
    # step incapable of ever flagging a duplicate on its own (a Hamming/
    # Bhattacharyya distance is always >= 0, so `distance <= -1` is never
    # true). Without this, the pre-existing dedup step alone -- with its
    # default threshold=8 -- already collapses these 20 identical static
    # frames down to 1 kept frame (verified separately: phash distance
    # between the static frames is exactly 0.0), so the assertion below would
    # pass even with a broken/absent motion prefilter. Disabling dedup here
    # forces the "at most 1 static frame kept" assertion to depend solely on
    # the motion prefilter actually firing.
    pool_config = CandidatePoolConfig(
        dedup_threshold=-1, motion_threshold=5.0, periodic_sample_every=50
    )
    candidates = build_candidate_pool(source, pool_config)
    ids = {c.frame_id for c in candidates}
    assert any(20 <= i <= 24 for i in ids)  # motion frames survive the prefilter
    assert (
        sum(1 for i in ids if i < 20) <= 1
    )  # static run: at most the periodic-floor sample
