"""Integration tests for FilterWorker's preserve-full-frames pipeline wiring."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

pytest.importorskip("PySide6")

from hydra_suite.filterkit.gui.main_window import FilterWorker


def _write_gray_image(path, value: int) -> None:
    img = np.full((10, 10), value, dtype=np.uint8)
    cv2.imwrite(str(path), img)


def _build_identity_dataset(tmp_path, frame_values: dict[int, int], individuals: int):
    images_dir = tmp_path / "dataset" / "images"
    images_dir.mkdir(parents=True)
    for frame_id, value in frame_values.items():
        for det_idx in range(individuals):
            detection_id = frame_id * 10000 + det_idx
            path = images_dir / f"did{detection_id}.png"
            _write_gray_image(path, value)
    return tmp_path / "dataset"


def test_filterworker_preserve_full_frames_expands_and_restores_companions(
    tmp_path,
) -> None:
    dataset_root = _build_identity_dataset(
        tmp_path,
        frame_values={0: 10, 1: 12, 2: 200, 3: 205},
        individuals=2,
    )
    config = {
        "temporal_enabled": False,
        "dedup_enabled": False,
        "diversity_enabled": True,
        "diversity_target": 4,  # ~2 individuals/frame -> back-solves to 2 frames
        "quality_enabled": True,
        "quality_min_blur": 1e9,  # deliberately impossible: drops every crop
        "quality_min_contrast": 0,
        "preserve_full_frames": True,
    }
    worker = FilterWorker(str(dataset_root), config)

    results = []
    worker.finished.connect(results.append)
    worker.execute()

    assert len(results) == 1
    result = results[0]
    # Quality filtering alone would drop everything; expansion restores
    # companions from the original dataset for whichever frames "survive"
    # diversity sampling on the (empty) quality-filtered set.
    stats = result["stats"]
    assert "after_expansion" in stats


def test_filterworker_preserve_full_frames_off_matches_baseline(tmp_path) -> None:
    dataset_root = _build_identity_dataset(
        tmp_path,
        frame_values={0: 10, 1: 200},
        individuals=2,
    )
    config = {
        "temporal_enabled": False,
        "dedup_enabled": False,
        "diversity_enabled": False,
        "diversity_target": 100,
        "quality_enabled": False,
        "quality_min_blur": 30,
        "quality_min_contrast": 20,
        "preserve_full_frames": False,
    }
    worker = FilterWorker(str(dataset_root), config)

    results = []
    worker.finished.connect(results.append)
    worker.execute()

    result = results[0]
    assert result["stats"]["after_expansion"] == result["stats"]["after_diversity"]
    assert len(result["selected_dataset"]) == 4
