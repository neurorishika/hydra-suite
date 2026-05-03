"""Tests for hydra_suite.data.al.candidate_pool."""

from __future__ import annotations

import cv2
import numpy as np

from hydra_suite.data.al.candidate_pool import CandidatePoolConfig, build_candidate_pool
from hydra_suite.data.al.frame_source import ImageFolderFrameSource


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


def test_candidate_pool_no_dedup_passthrough(tmp_path):
    src = _make_dataset(tmp_path, n_unique=5, n_dupes=0)
    cfg = CandidatePoolConfig(dedup_method="none")

    refs = build_candidate_pool(src, cfg)
    assert len(refs) == 5
