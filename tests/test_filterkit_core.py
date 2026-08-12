"""Tests for FilterKit core dataset loading and deduplication behavior."""

from __future__ import annotations

import json

import cv2
import numpy as np

from hydra_suite.filterkit.core import FilterKitCore


def _make_item(path: str, det_id: int, signature, color_signature=None):
    item = {
        "path": path,
        "filename": path.rsplit("/", 1)[-1],
        "det_id": det_id,
        "frame_idx": det_id // 10000,
        "det_idx": det_id % 10000,
        "dedup_signature": signature,
    }
    if color_signature is not None:
        item["color_signature"] = color_signature
    return item


def _one_hot(size: int, index: int) -> np.ndarray:
    signature = np.zeros((size,), dtype=np.float32)
    signature[index] = 1.0
    return signature


def test_filterkit_hash_dedup_keeps_first_nonduplicate_match() -> None:
    core = FilterKitCore()
    dataset = [
        _make_item("/tmp/a.png", 1, 0b0000),
        _make_item("/tmp/b.png", 2, 0b0001),
        _make_item("/tmp/c.png", 3, 0b0011),
        _make_item("/tmp/d.png", 4, 0b1000),
    ]

    kept, groups = core.deduplicate_by_hash(
        dataset,
        threshold=1,
        method="phash",
        return_groups=True,
    )

    assert [item["path"] for item in kept] == ["/tmp/a.png", "/tmp/c.png"]
    assert groups == [
        {
            "hash": "0",
            "count": 3,
            "paths": ["/tmp/a.png", "/tmp/b.png", "/tmp/d.png"],
            "method": "phash",
        }
    ]


def test_filterkit_hash_dedup_preserves_distinct_colors() -> None:
    core = FilterKitCore()
    color_a = _one_hot(8 * 8 * 8, 0)
    color_b = _one_hot(8 * 8 * 8, 1)
    dataset = [
        _make_item("/tmp/a.png", 1, 123, color_signature=color_a),
        _make_item("/tmp/b.png", 2, 123, color_signature=color_b),
        _make_item("/tmp/c.png", 3, 123, color_signature=color_a),
    ]

    kept, groups = core.deduplicate_by_hash(
        dataset,
        threshold=0,
        method="phash",
        return_groups=True,
        color_threshold=0.2,
    )

    assert [item["path"] for item in kept] == ["/tmp/a.png", "/tmp/b.png"]
    assert groups == [
        {
            "hash": "123",
            "count": 2,
            "paths": ["/tmp/a.png", "/tmp/c.png"],
            "method": "phash",
        }
    ]


def test_filterkit_histogram_dedup_matches_existing_behavior() -> None:
    core = FilterKitCore()
    dataset = [
        _make_item("/tmp/a.png", 1, _one_hot(32, 0)),
        _make_item("/tmp/b.png", 2, _one_hot(32, 0)),
        _make_item("/tmp/c.png", 3, _one_hot(32, 5)),
    ]

    kept, groups = core.deduplicate_by_hash(
        dataset,
        threshold=0.1,
        method="histogram",
        return_groups=True,
    )

    assert [item["path"] for item in kept] == ["/tmp/a.png", "/tmp/c.png"]
    assert groups == [
        {
            "hash": str(_one_hot(32, 0)),
            "count": 2,
            "paths": ["/tmp/a.png", "/tmp/b.png"],
            "method": "histogram",
        }
    ]


def test_filterkit_load_dataset_accepts_detected_and_interpolated_flat_names(
    tmp_path,
) -> None:
    dataset_root = tmp_path / "dataset"
    images_dir = dataset_root / "images"
    images_dir.mkdir(parents=True)
    (images_dir / "did101.jpg").write_bytes(b"x")
    interp_name = "interp_f000002_traj0001_seg000001-000003_p001of001.png"
    (images_dir / interp_name).write_bytes(b"y")
    (dataset_root / "metadata.json").write_text(
        json.dumps(
            {
                "images": [
                    {"filename": "did101.jpg", "source_type": "yolo_obb"},
                    {"filename": interp_name, "source_type": "interpolated"},
                ]
            }
        ),
        encoding="utf-8",
    )

    dataset = FilterKitCore().load_dataset(str(images_dir))

    assert [item["filename"] for item in dataset] == ["did101.jpg", interp_name]
    assert dataset[0]["det_id"] == 101
    assert dataset[0]["frame_idx"] == 0
    assert dataset[0]["annotations"][0]["filename"] == "did101.jpg"
    assert dataset[1]["interpolated"] is True
    assert dataset[1]["source_type"] == "interpolated"
    assert dataset[1]["frame_idx"] == 2
    assert dataset[1]["trajectory_id"] == 1
    assert dataset[1]["det_id"] < 0
    assert dataset[1]["annotations"][0]["filename"] == interp_name


def test_filterkit_load_images_from_root_parses_identity_filenames(tmp_path) -> None:
    dataset_root = tmp_path / "dataset"
    images_dir = dataset_root / "images"
    images_dir.mkdir(parents=True)
    (images_dir / "did101.jpg").write_bytes(b"x")
    (images_dir / "did202.jpg").write_bytes(b"y")
    (images_dir / "plain_frame_007.png").write_bytes(b"z")

    source_kind, items = FilterKitCore().load_images_from_root(dataset_root)

    assert source_kind == "images"
    by_name = {item["filename"]: item for item in items}
    assert by_name["did101.jpg"]["frame_idx"] == 0
    assert by_name["did101.jpg"]["det_idx"] == 101
    assert by_name["did101.jpg"]["detection_id"] == 101
    assert by_name["did202.jpg"]["frame_idx"] == 0
    assert by_name["did202.jpg"]["det_idx"] == 202
    # Non-matching filenames keep the sequential fallback.
    assert by_name["plain_frame_007.png"]["det_idx"] == 0
    assert by_name["plain_frame_007.png"]["source_type"] == "images"


def _write_gray_image(path, value: int) -> None:
    img = np.full((10, 10), value, dtype=np.uint8)
    cv2.imwrite(str(path), img)


def test_filterkit_compute_avg_individuals_per_frame() -> None:
    core = FilterKitCore()
    dataset = [
        {"frame_idx": 0},
        {"frame_idx": 0},
        {"frame_idx": 1},
        {"frame_idx": 2},
        {"frame_idx": 2},
    ]
    assert core.compute_avg_individuals_per_frame(dataset) == 5 / 3


def test_filterkit_compute_avg_individuals_per_frame_empty_dataset() -> None:
    core = FilterKitCore()
    assert core.compute_avg_individuals_per_frame([]) == 1.0


def test_filterkit_diversity_sample_by_frame_keeps_all_individuals_per_frame(
    tmp_path,
) -> None:
    core = FilterKitCore()
    # 4 frames, 2 individuals each. Frames 0/1 are visually similar (dark);
    # frames 2/3 are visually similar (bright) — two well-separated clusters.
    values = {0: 10, 1: 12, 2: 200, 3: 205}
    dataset = []
    for frame_id, value in values.items():
        for det_idx in range(2):
            path = tmp_path / f"f{frame_id}_d{det_idx}.png"
            _write_gray_image(path, value)
            dataset.append(
                {
                    "path": str(path),
                    "filename": path.name,
                    "det_id": frame_id * 10000 + det_idx,
                    "frame_idx": frame_id,
                    "det_idx": det_idx,
                }
            )

    selected = core.diversity_sample(dataset, 2, by_frame=True)

    selected_frames = {item["frame_idx"] for item in selected}
    assert len(selected_frames) == 2
    # One frame from the dark cluster, one from the bright cluster.
    assert len(selected_frames & {0, 1}) == 1
    assert len(selected_frames & {2, 3}) == 1
    # Every selected frame keeps both of its individuals.
    for frame_id in selected_frames:
        crops = [item for item in selected if item["frame_idx"] == frame_id]
        assert len(crops) == 2


def test_filterkit_diversity_sample_default_behavior_unchanged(tmp_path) -> None:
    core = FilterKitCore()
    dataset = []
    for i in range(6):
        path = tmp_path / f"img{i}.png"
        _write_gray_image(path, i * 40)
        dataset.append({"path": str(path), "filename": path.name, "frame_idx": i})

    by_frame_default = core.diversity_sample(dataset, 3)
    by_frame_explicit_false = core.diversity_sample(dataset, 3, by_frame=False)

    assert [item["path"] for item in by_frame_default] == [
        item["path"] for item in by_frame_explicit_false
    ]
    assert len(by_frame_default) <= 3


def test_filterkit_expand_to_full_frames_restores_filtered_companions() -> None:
    core = FilterKitCore()
    full_dataset = [
        {"det_id": 1, "frame_idx": 0, "det_idx": 0, "path": "/tmp/a.png"},
        {"det_id": 2, "frame_idx": 0, "det_idx": 1, "path": "/tmp/b.png"},
        {"det_id": 3, "frame_idx": 1, "det_idx": 0, "path": "/tmp/c.png"},
    ]
    # Simulate quality filtering having dropped det_id=2 (blurry companion).
    kept = [full_dataset[0]]

    expanded = core.expand_to_full_frames(kept, full_dataset)

    assert {item["det_id"] for item in expanded} == {1, 2}
    assert all(item["frame_idx"] == 0 for item in expanded)


def test_filterkit_expand_to_full_frames_empty_kept_returns_empty() -> None:
    core = FilterKitCore()
    assert core.expand_to_full_frames([], [{"det_id": 1, "frame_idx": 0}]) == []


def test_filterkit_expand_to_full_frames_deduplicates_by_det_id() -> None:
    core = FilterKitCore()
    full_dataset = [
        {"det_id": 1, "frame_idx": 0, "det_idx": 0},
        {"det_id": 2, "frame_idx": 0, "det_idx": 1},
    ]
    kept = [full_dataset[0], full_dataset[1]]  # both already present

    expanded = core.expand_to_full_frames(kept, full_dataset)

    assert len(expanded) == 2


def test_filterkit_expand_to_full_frames_handles_multiple_distinct_frames() -> None:
    """Verify expand_to_full_frames correctly handles multiple frames with distinct det_ids.

    This test locks in the invariant that det_id is globally unique across frames
    (derived from detection_id where frame_idx = detection_id // 10000,
    det_idx = detection_id % 10000), confirming that the global dedup-by-det_id
    is safe and does not cause unintended cross-frame collisions.
    """
    core = FilterKitCore()
    full_dataset = [
        {"det_id": 10000, "frame_idx": 1, "det_idx": 0, "path": "/tmp/f1_d0.png"},
        {"det_id": 10001, "frame_idx": 1, "det_idx": 1, "path": "/tmp/f1_d1.png"},
        {"det_id": 20000, "frame_idx": 2, "det_idx": 0, "path": "/tmp/f2_d0.png"},
        {"det_id": 20001, "frame_idx": 2, "det_idx": 1, "path": "/tmp/f2_d1.png"},
    ]
    # Only one individual from each frame survived filtering.
    kept = [full_dataset[0], full_dataset[2]]

    expanded = core.expand_to_full_frames(kept, full_dataset)

    assert {item["det_id"] for item in expanded} == {10000, 10001, 20000, 20001}
    assert {item["frame_idx"] for item in expanded} == {1, 2}
    assert len(expanded) == 4
