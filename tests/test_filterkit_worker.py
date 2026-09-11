"""Integration tests for FilterWorker's preserve-full-frames pipeline wiring."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

pytest.importorskip("PySide6")

from hydra_suite.filterkit.gui.main_window import FilterKitMediaReader, FilterWorker


def _write_gray_image(path, value: int) -> None:
    img = np.full((10, 10), value, dtype=np.uint8)
    cv2.imwrite(str(path), img)


def _write_video(path, values: list[int]) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (16, 16))
    assert writer.isOpened()
    try:
        for value in values:
            writer.write(np.full((16, 16, 3), value, dtype=np.uint8))
    finally:
        writer.release()


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

    selected_paths = {item["path"] for item in selected}
    removed_paths = {item["path"] for item in result["removed_examples"]}
    assert not (
        selected_paths & removed_paths
    ), "restored companions must not remain listed as removed"


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


def test_filterworker_video_temporal_selection_retains_frame_provenance(
    tmp_path,
) -> None:
    video_path = tmp_path / "recording.avi"
    _write_video(video_path, [20, 60, 120, 200])
    worker = FilterWorker(
        str(video_path),
        {
            "temporal_enabled": True,
            "temporal_interval": 2,
            "dedup_enabled": False,
            "diversity_enabled": False,
            "quality_enabled": False,
            "preserve_full_frames": False,
        },
    )

    results = []
    worker.finished.connect(results.append)
    worker.execute()

    selected = results[0]["selected_dataset"]
    assert [item["frame_idx"] for item in selected] == [0, 2]
    assert all(item["video_path"] == str(video_path.resolve()) for item in selected)


def test_filterworker_video_decodes_temporal_candidates_once(
    tmp_path, monkeypatch
) -> None:
    video_path = tmp_path / "recording.avi"
    _write_video(video_path, [20, 60, 120, 200, 80, 160])
    reads = 0
    original_read = FilterKitMediaReader.read

    def count_reads(self, item):
        nonlocal reads
        reads += 1
        return original_read(self, item)

    monkeypatch.setattr(FilterKitMediaReader, "read", count_reads)
    worker = FilterWorker(
        str(video_path),
        {
            "temporal_enabled": True,
            "temporal_interval": 2,
            "dedup_enabled": True,
            "dedup_method": "phash",
            "dedup_threshold": 0,
            "diversity_enabled": True,
            "diversity_target": 2,
            "quality_enabled": False,
            "preserve_full_frames": False,
        },
    )

    results = []
    worker.finished.connect(results.append)
    worker.execute()

    assert len(results) == 1
    assert results[0]["stats"]["after_temporal"] == 3
    assert reads == 3


def test_filterworker_video_skips_decode_when_diversity_is_a_noop(
    tmp_path, monkeypatch
) -> None:
    video_path = tmp_path / "recording.avi"
    _write_video(video_path, [20, 60, 120, 200])
    reads = 0
    original_read = FilterKitMediaReader.read

    def count_reads(self, item):
        nonlocal reads
        reads += 1
        return original_read(self, item)

    monkeypatch.setattr(FilterKitMediaReader, "read", count_reads)
    worker = FilterWorker(
        str(video_path),
        {
            "temporal_enabled": False,
            "dedup_enabled": False,
            "diversity_enabled": True,
            "diversity_target": 10,
            "quality_enabled": False,
            "preserve_full_frames": False,
        },
    )
    results = []
    worker.finished.connect(results.append)
    worker.execute()

    assert len(results) == 1
    assert len(results[0]["selected_dataset"]) == 4
    assert reads == 0
