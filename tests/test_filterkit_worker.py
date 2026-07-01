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
    rng = np.random.default_rng(0)
    images_dir = tmp_path / "dataset" / "images"
    images_dir.mkdir(parents=True)
    for frame_id in range(4):
        # det_idx=0: textured noise, passes quality filtering.
        detection_id_good = frame_id * 10000 + 0
        good_img = rng.integers(0, 256, size=(20, 20), dtype=np.uint8)
        cv2.imwrite(str(images_dir / f"did{detection_id_good}.png"), good_img)

        # det_idx=1: flat constant image, always fails quality filtering
        # (zero blur variance, zero contrast).
        detection_id_bad = frame_id * 10000 + 1
        bad_img = np.full((20, 20), 128, dtype=np.uint8)
        cv2.imwrite(str(images_dir / f"did{detection_id_bad}.png"), bad_img)

    dataset_root = tmp_path / "dataset"
    config = {
        "temporal_enabled": False,
        "dedup_enabled": False,
        "diversity_enabled": True,
        "diversity_target": 4,  # avg 2 individuals/frame -> back-solves to 2 frames
        "quality_enabled": True,
        "quality_min_blur": 30,
        "quality_min_contrast": 20,
        "preserve_full_frames": True,
    }
    worker = FilterWorker(str(dataset_root), config)

    results = []
    worker.finished.connect(results.append)
    worker.execute()

    assert len(results) == 1
    result = results[0]
    selected = result["selected_dataset"]

    # Quality filtering drops every det_idx=1 crop (4 dropped), leaving 4
    # det_idx=0 crops to diversity-sample from. Back-solving picks 2 frames.
    # Expansion must restore each selected frame's dropped det_idx=1 companion.
    selected_frames = {item["frame_idx"] for item in selected}
    assert len(selected_frames) == 2
    assert len(selected) == 4  # 2 frames x 2 individuals each, companions restored
    for frame_id in selected_frames:
        det_idxs = {
            item["det_idx"] for item in selected if item["frame_idx"] == frame_id
        }
        assert det_idxs == {
            0,
            1,
        }, f"frame {frame_id} missing a companion after expansion: {det_idxs}"


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
