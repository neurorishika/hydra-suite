import json
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from hydra_suite.core.canonicalization.geometry import CanonicalGeometry
from hydra_suite.core.individual.dataset.naming import read_canonical_provenance
from hydra_suite.core.individual.dataset.oriented_video import (
    OrientedTrackVideoExporter,
    resolve_individual_dataset_dir,
    resolve_oriented_track_video_dir,
)
from hydra_suite.core.inference.cache.base import CacheKey
from hydra_suite.core.inference.cache.store import DetectionCacheHandle
from hydra_suite.core.inference.result import OBBResult


def _write_video(path: Path, colors: list[tuple[int, int, int]]) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        5.0,
        (64, 48),
    )
    assert writer.isOpened()
    try:
        for color in colors:
            frame = np.full((48, 64, 3), color, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()


def _square(cx: float, cy: float, half: float) -> np.ndarray:
    return np.array(
        [
            [cx - half, cy - half],
            [cx + half, cy - half],
            [cx + half, cy + half],
            [cx - half, cy + half],
        ],
        dtype=np.float32,
    )


def _write_detection_cache(path: Path, frames: list[dict]) -> None:
    """Write a modern ``detection.npz`` (``DetectionCacheHandle`` format).

    Each entry in ``frames`` is a dict with ``frame_idx`` plus optional
    ``meas`` (list of ``[cx, cy, theta]``), ``shapes`` (list of
    ``(ellipse_area, aspect_ratio)``), ``confidences``, ``obb_corners``, and
    ``detection_ids``. Missing keys default to zero detections for that
    frame (still recorded as a written frame).
    """
    key = CacheKey(
        schema_version=0, model_path="test", model_mtime=0.0, config_hash="h"
    )
    handle = DetectionCacheHandle(path=path, key=key)
    for frame in frames:
        frame_idx = int(frame["frame_idx"])
        meas = frame.get("meas", [])
        shapes = frame.get("shapes", [])
        confidences = frame.get("confidences", [])
        obb_corners = frame.get("obb_corners", [])
        detection_ids = frame.get("detection_ids", [])
        n = len(meas)
        centroids = (
            np.array([[m[0], m[1]] for m in meas], dtype=np.float32)
            if n
            else np.zeros((0, 2), dtype=np.float32)
        )
        angles = (
            np.array([m[2] for m in meas], dtype=np.float32)
            if n
            else np.zeros(0, dtype=np.float32)
        )
        sizes = (
            np.array([s[0] for s in shapes], dtype=np.float32)
            if n
            else np.zeros(0, dtype=np.float32)
        )
        shapes_arr = (
            np.array(shapes, dtype=np.float32)
            if n
            else np.zeros((0, 2), dtype=np.float32)
        )
        confidences_arr = (
            np.array(confidences, dtype=np.float32)
            if n
            else np.zeros(0, dtype=np.float32)
        )
        corners_arr = (
            np.array(obb_corners, dtype=np.float32)
            if n
            else np.zeros((0, 4, 2), dtype=np.float32)
        )
        detection_ids_arr = (
            np.array(detection_ids, dtype=np.int64)
            if n
            else np.zeros(0, dtype=np.int64)
        )
        handle.write_frame(
            frame_idx,
            result=OBBResult(
                frame_idx=frame_idx,
                centroids=centroids,
                angles=angles,
                sizes=sizes,
                shapes=shapes_arr,
                confidences=confidences_arr,
                corners=corners_arr,
                detection_ids=detection_ids_arr,
            ),
        )
    handle.close()


def test_resolve_individual_dataset_dir_uses_run_id(tmp_path: Path):
    root = tmp_path / "individual_crops"
    dataset_dir = root / "session_20260311"
    dataset_dir.mkdir(parents=True)

    resolved = resolve_individual_dataset_dir(
        root, dataset_name="session", run_id="20260311"
    )

    assert resolved == dataset_dir


def test_resolve_oriented_track_video_dir_uses_run_id(tmp_path: Path):
    root = tmp_path / "oriented_videos"
    run_dir = root / "20260311_120000"
    run_dir.mkdir(parents=True)

    resolved = resolve_oriented_track_video_dir(root, run_id="20260311_120000")

    assert resolved == run_dir


def test_oriented_track_video_export_streams_from_source_video_and_caches(
    tmp_path: Path,
):
    dataset_dir = tmp_path / "individual_crops" / "run_20260311"
    video_path = tmp_path / "source.mp4"
    cache_path = tmp_path / "detections.npz"
    interp_npz_path = tmp_path / "interpolated_rois.npz"
    final_csv_path = tmp_path / "tracks_final.csv"

    _write_video(
        video_path,
        [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)],
    )

    _write_detection_cache(
        cache_path,
        [
            {
                "frame_idx": 0,
                "meas": [[20.0, 24.0, 0.0]],
                "shapes": [(64.0, 1.0)],
                "confidences": [0.9],
                "obb_corners": [_square(20.0, 24.0, 6.0)],
                "detection_ids": [101],
            },
            {
                "frame_idx": 1,
                "meas": [[24.0, 24.0, 0.0]],
                "shapes": [(64.0, 1.0)],
                "confidences": [0.9],
                "obb_corners": [_square(24.0, 24.0, 6.0)],
                "detection_ids": [102],
            },
            {"frame_idx": 2},
            {
                "frame_idx": 3,
                "meas": [[42.0, 24.0, 0.0]],
                "shapes": [(64.0, 1.0)],
                "confidences": [0.9],
                "obb_corners": [_square(42.0, 24.0, 6.0)],
                "detection_ids": [201],
            },
        ],
    )

    np.savez_compressed(
        str(interp_npz_path),
        frame_id=np.array([2], dtype=np.int64),
        trajectory_id=np.array([1], dtype=np.int64),
        filename=np.array([""], dtype=object),
        cx=np.array([28.0], dtype=np.float32),
        cy=np.array([24.0], dtype=np.float32),
        w=np.array([12.0], dtype=np.float32),
        h=np.array([12.0], dtype=np.float32),
        theta=np.array([0.0], dtype=np.float32),
        interp_from_start=np.array([1], dtype=np.int64),
        interp_from_end=np.array([3], dtype=np.int64),
        interp_index=np.array([1], dtype=np.int64),
        interp_total=np.array([1], dtype=np.int64),
        obb_corners=np.array([_square(28.0, 24.0, 6.0)], dtype=np.float32),
    )

    pd.DataFrame(
        [
            {
                "TrajectoryID": 1,
                "FrameID": 0,
                "DetectionID": 101,
                "Theta": 0.0,
                "State": "active",
            },
            {
                "TrajectoryID": 1,
                "FrameID": 1,
                "DetectionID": 102,
                "Theta": 0.0,
                "State": "active",
            },
            {
                "TrajectoryID": 1,
                "FrameID": 2,
                "DetectionID": np.nan,
                "Theta": 0.0,
                "State": "occluded",
            },
            {
                "TrajectoryID": 2,
                "FrameID": 3,
                "DetectionID": 201,
                "Theta": 0.0,
                "State": "active",
            },
        ]
    ).to_csv(final_csv_path, index=False)

    exporter = OrientedTrackVideoExporter(
        dataset_dir,
        final_csv_path,
        video_path=video_path,
        detection_cache_path=cache_path,
        interpolated_roi_npz_path=interp_npz_path,
        fps=5.0,
    )
    result = exporter.export()

    assert result.exported_videos == 2
    assert result.exported_tracks == 2
    assert result.exported_frames == 4
    assert result.exported_images == 0
    assert result.missing_rows == 0

    track1_path = Path(result.output_dir) / "trajectory_0001.mp4"
    track2_path = Path(result.output_dir) / "trajectory_0002.mp4"
    assert track1_path.exists()
    assert track2_path.exists()

    cap1 = cv2.VideoCapture(str(track1_path))
    try:
        assert cap1.isOpened()
        assert int(cap1.get(cv2.CAP_PROP_FRAME_COUNT)) == 3
    finally:
        cap1.release()

    cap2 = cv2.VideoCapture(str(track2_path))
    try:
        assert cap2.isOpened()
        assert int(cap2.get(cv2.CAP_PROP_FRAME_COUNT)) == 1
    finally:
        cap2.release()


def test_final_canonical_media_export_writes_images_and_videos(tmp_path: Path):
    dataset_dir = tmp_path / "individual_crops" / "run_20260311"
    image_output_dir = dataset_dir / "images"
    video_path = tmp_path / "source.mp4"
    cache_path = tmp_path / "detections.npz"
    interp_npz_path = tmp_path / "interpolated_rois.npz"
    final_csv_path = tmp_path / "tracks_final.csv"

    _write_video(
        video_path,
        [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)],
    )

    _write_detection_cache(
        cache_path,
        [
            {
                "frame_idx": 0,
                "meas": [[20.0, 24.0, 0.0]],
                "shapes": [(64.0, 1.0)],
                "confidences": [0.9],
                "obb_corners": [_square(20.0, 24.0, 6.0)],
                "detection_ids": [101],
            },
            {
                "frame_idx": 1,
                "meas": [[24.0, 24.0, 0.0]],
                "shapes": [(64.0, 1.0)],
                "confidences": [0.9],
                "obb_corners": [_square(24.0, 24.0, 6.0)],
                "detection_ids": [102],
            },
            {"frame_idx": 2},
            {
                "frame_idx": 3,
                "meas": [[42.0, 24.0, 0.0]],
                "shapes": [(64.0, 1.0)],
                "confidences": [0.9],
                "obb_corners": [_square(42.0, 24.0, 6.0)],
                "detection_ids": [201],
            },
        ],
    )

    np.savez_compressed(
        str(interp_npz_path),
        frame_id=np.array([2], dtype=np.int64),
        trajectory_id=np.array([1], dtype=np.int64),
        filename=np.array([""], dtype=object),
        cx=np.array([28.0], dtype=np.float32),
        cy=np.array([24.0], dtype=np.float32),
        w=np.array([12.0], dtype=np.float32),
        h=np.array([12.0], dtype=np.float32),
        theta=np.array([0.0], dtype=np.float32),
        interp_from_start=np.array([1], dtype=np.int64),
        interp_from_end=np.array([3], dtype=np.int64),
        interp_index=np.array([1], dtype=np.int64),
        interp_total=np.array([1], dtype=np.int64),
        obb_corners=np.array([_square(28.0, 24.0, 6.0)], dtype=np.float32),
    )

    pd.DataFrame(
        [
            {
                "TrajectoryID": 1,
                "FrameID": 0,
                "DetectionID": 101,
                "Theta": 0.0,
                "State": "active",
            },
            {
                "TrajectoryID": 1,
                "FrameID": 1,
                "DetectionID": 102,
                "Theta": 0.0,
                "State": "active",
            },
            {
                "TrajectoryID": 1,
                "FrameID": 2,
                "DetectionID": np.nan,
                "Theta": 0.0,
                "State": "occluded",
            },
            {
                "TrajectoryID": 2,
                "FrameID": 3,
                "DetectionID": 201,
                "Theta": 0.0,
                "State": "active",
            },
        ]
    ).to_csv(final_csv_path, index=False)

    exporter = OrientedTrackVideoExporter(
        dataset_dir,
        final_csv_path,
        video_path=video_path,
        detection_cache_path=cache_path,
        interpolated_roi_npz_path=interp_npz_path,
        fps=5.0,
        export_images=True,
        image_output_dir=image_output_dir,
        export_videos=True,
        output_subdir="oriented_videos",
    )

    result = exporter.export()

    assert result.exported_videos == 2
    assert result.exported_images == 4
    assert Path(result.image_output_dir) == image_output_dir
    assert (image_output_dir / "did101.png").exists()
    assert (image_output_dir / "did102.png").exists()
    assert (
        image_output_dir / "interp_f000002_traj0001_seg000001-000003_p001of001.png"
    ).exists()
    assert (image_output_dir / "did201.png").exists()


def test_oriented_track_video_preserves_branch_across_interpolated_rows(
    tmp_path: Path,
):
    """Actual-row heading now comes from the final CSV's ``Theta`` column
    (upstream head-tail correction already bakes the resolved/directed
    heading into it) -- the modern per-frame detection cache
    (``DetectionCacheHandle``/``OBBResult``) has no slot for cache-level
    heading hints or a directed mask; those live in the separate head-tail
    cache, which this dataset exporter does not read. This test now checks
    that an interpolated row's stored heading (on the opposite pi-branch)
    still collapses onto the resolved reference branch from the real
    detection, rather than exercising a cache-supplied heading override."""
    dataset_dir = tmp_path / "individual_crops" / "run_20260311"
    cache_path = tmp_path / "detections.npz"
    interp_npz_path = tmp_path / "interpolated_rois.npz"
    final_csv_path = tmp_path / "tracks_final.csv"
    video_path = tmp_path / "source.mp4"

    _write_video(video_path, [(0, 0, 255), (0, 255, 0)])

    _write_detection_cache(
        cache_path,
        [
            {
                "frame_idx": 0,
                "meas": [[20.0, 24.0, 0.0]],
                "shapes": [(64.0, 1.0)],
                "confidences": [0.9],
                "obb_corners": [_square(20.0, 24.0, 6.0)],
                "detection_ids": [101],
            },
            {"frame_idx": 1},
        ],
    )

    np.savez_compressed(
        str(interp_npz_path),
        frame_id=np.array([1], dtype=np.int64),
        trajectory_id=np.array([1], dtype=np.int64),
        filename=np.array([""], dtype=object),
        cx=np.array([28.0], dtype=np.float32),
        cy=np.array([24.0], dtype=np.float32),
        w=np.array([12.0], dtype=np.float32),
        h=np.array([12.0], dtype=np.float32),
        theta=np.array([math.pi], dtype=np.float32),
        interp_from_start=np.array([0], dtype=np.int64),
        interp_from_end=np.array([0], dtype=np.int64),
        interp_index=np.array([1], dtype=np.int64),
        interp_total=np.array([1], dtype=np.int64),
        obb_corners=np.array([_square(28.0, 24.0, 6.0)], dtype=np.float32),
    )

    pd.DataFrame(
        [
            {
                "TrajectoryID": 1,
                "FrameID": 0,
                "DetectionID": 101,
                "Theta": 0.0,
                "State": "active",
            },
            {
                "TrajectoryID": 1,
                "FrameID": 1,
                "DetectionID": np.nan,
                "Theta": math.pi,
                "State": "occluded",
            },
        ]
    ).to_csv(final_csv_path, index=False)

    exporter = OrientedTrackVideoExporter(
        dataset_dir,
        final_csv_path,
        video_path=video_path,
        detection_cache_path=cache_path,
        interpolated_roi_npz_path=interp_npz_path,
        fps=5.0,
    )

    trajectories_df = exporter._load_final_dataframe()
    interp_lookup = exporter._load_interpolated_roi_lookup()
    frame_bundles, _track_sizes, missing_rows = exporter._build_frame_bundles(
        trajectories_df,
        interp_lookup,
    )

    assert missing_rows == 0
    task0 = frame_bundles[0].tasks[0]
    task1 = frame_bundles[1].tasks[0]
    expected_affine, expected_w, expected_h = exporter._canonical_affine_for_task(
        20.0,
        24.0,
        12.0,
        12.0,
        0.0,
    )
    interp_expected_affine, interp_expected_w, interp_expected_h = (
        exporter._canonical_affine_for_task(
            28.0,
            24.0,
            12.0,
            12.0,
            0.0,
        )
    )

    assert np.allclose(task0.affine, expected_affine)
    assert task0.out_w == expected_w
    assert task0.out_h == expected_h
    assert np.allclose(task1.affine, interp_expected_affine)
    assert task1.out_w == interp_expected_w
    assert task1.out_h == interp_expected_h


def test_oriented_track_video_can_fix_short_heading_flip_bursts(tmp_path: Path):
    dataset_dir = tmp_path / "oriented_videos" / "run_20260311"
    cache_path = tmp_path / "detections.npz"
    final_csv_path = tmp_path / "tracks_final.csv"
    video_path = tmp_path / "source.mp4"

    _write_video(video_path, [(0, 0, 255), (0, 255, 0), (255, 0, 0)])

    _write_detection_cache(
        cache_path,
        [
            {
                "frame_idx": 0,
                "meas": [[20.0, 24.0, 0.0]],
                "shapes": [(64.0, 1.0)],
                "confidences": [0.9],
                "obb_corners": [_square(20.0, 24.0, 6.0)],
                "detection_ids": [101],
            },
            {
                "frame_idx": 1,
                "meas": [[24.0, 24.0, math.pi]],
                "shapes": [(64.0, 1.0)],
                "confidences": [0.9],
                "obb_corners": [_square(24.0, 24.0, 6.0)],
                "detection_ids": [102],
            },
            {
                "frame_idx": 2,
                "meas": [[28.0, 24.0, 0.0]],
                "shapes": [(64.0, 1.0)],
                "confidences": [0.9],
                "obb_corners": [_square(28.0, 24.0, 6.0)],
                "detection_ids": [103],
            },
        ],
    )

    pd.DataFrame(
        [
            {
                "TrajectoryID": 1,
                "FrameID": 0,
                "DetectionID": 101,
                "Theta": 0.0,
                "State": "active",
            },
            {
                "TrajectoryID": 1,
                "FrameID": 1,
                "DetectionID": 102,
                "Theta": math.pi,
                "State": "active",
            },
            {
                "TrajectoryID": 1,
                "FrameID": 2,
                "DetectionID": 103,
                "Theta": 0.0,
                "State": "active",
            },
        ]
    ).to_csv(final_csv_path, index=False)

    exporter = OrientedTrackVideoExporter(
        dataset_dir,
        final_csv_path,
        video_path=video_path,
        detection_cache_path=cache_path,
        fps=5.0,
        fix_direction_flips=True,
        heading_flip_max_burst=1,
    )

    trajectories_df = exporter._load_final_dataframe()
    frame_bundles, _track_sizes, missing_rows = exporter._build_frame_bundles(
        trajectories_df,
        {},
    )

    assert missing_rows == 0
    corrected_task = frame_bundles[1].tasks[0]
    expected_affine, expected_w, expected_h = exporter._canonical_affine_for_task(
        24.0,
        24.0,
        12.0,
        12.0,
        0.0,
    )

    assert np.allclose(corrected_task.affine, expected_affine)
    assert corrected_task.out_w == expected_w
    assert corrected_task.out_h == expected_h


def test_oriented_track_video_affine_stabilization_smooths_jitter(tmp_path: Path):
    dataset_dir = tmp_path / "oriented_videos" / "run_20260311"
    cache_path = tmp_path / "detections.npz"
    final_csv_path = tmp_path / "tracks_final.csv"
    video_path = tmp_path / "source.mp4"

    _write_video(video_path, [(0, 0, 255), (0, 255, 0), (255, 0, 0)])

    _write_detection_cache(
        cache_path,
        [
            {
                "frame_idx": 0,
                "meas": [[20.0, 24.0, 0.0]],
                "shapes": [(64.0, 1.0)],
                "confidences": [0.9],
                "obb_corners": [_square(20.0, 24.0, 6.0)],
                "detection_ids": [101],
            },
            {
                "frame_idx": 1,
                "meas": [[28.0, 24.0, 0.0]],
                "shapes": [(256.0, 1.0)],
                "confidences": [0.9],
                "obb_corners": [_square(28.0, 24.0, 12.0)],
                "detection_ids": [102],
            },
            {
                "frame_idx": 2,
                "meas": [[20.0, 24.0, 0.0]],
                "shapes": [(64.0, 1.0)],
                "confidences": [0.9],
                "obb_corners": [_square(20.0, 24.0, 6.0)],
                "detection_ids": [103],
            },
        ],
    )

    pd.DataFrame(
        [
            {
                "TrajectoryID": 1,
                "FrameID": 0,
                "DetectionID": 101,
                "Theta": 0.0,
                "State": "active",
            },
            {
                "TrajectoryID": 1,
                "FrameID": 1,
                "DetectionID": 102,
                "Theta": 0.0,
                "State": "active",
            },
            {
                "TrajectoryID": 1,
                "FrameID": 2,
                "DetectionID": 103,
                "Theta": 0.0,
                "State": "active",
            },
        ]
    ).to_csv(final_csv_path, index=False)

    exporter = OrientedTrackVideoExporter(
        dataset_dir,
        final_csv_path,
        video_path=video_path,
        detection_cache_path=cache_path,
        fps=5.0,
        enable_affine_stabilization=True,
        stabilization_window=3,
    )

    trajectories_df = exporter._load_final_dataframe()
    frame_bundles, _track_sizes, missing_rows = exporter._build_frame_bundles(
        trajectories_df,
        {},
    )

    assert missing_rows == 0
    stabilized_task = frame_bundles[1].tasks[0]
    expected_affine, expected_w, expected_h = exporter._canonical_affine_for_task(
        20.0,
        24.0,
        12.0,
        12.0,
        0.0,
    )

    assert np.allclose(stabilized_task.affine, expected_affine)
    assert stabilized_task.out_w == expected_w
    assert stabilized_task.out_h == expected_h


def test_final_media_export_reports_missing_geometry_breakdown(tmp_path: Path):
    dataset_dir = tmp_path / "oriented_videos" / "run_20260311"
    cache_path = tmp_path / "detections.npz"
    interp_npz_path = tmp_path / "interpolated_rois.npz"
    final_csv_path = tmp_path / "tracks_final.csv"
    video_path = tmp_path / "source.mp4"

    _write_video(
        video_path,
        [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)],
    )

    _write_detection_cache(
        cache_path,
        [
            {
                "frame_idx": 0,
                "meas": [[20.0, 24.0, 0.0]],
                "shapes": [(64.0, 1.0)],
                "confidences": [0.9],
                "obb_corners": [_square(20.0, 24.0, 6.0)],
                "detection_ids": [101],
            },
            {"frame_idx": 1},
            {"frame_idx": 2},
            {"frame_idx": 3},
        ],
    )

    np.savez_compressed(
        str(interp_npz_path),
        frame_id=np.array([3], dtype=np.int64),
        trajectory_id=np.array([4], dtype=np.int64),
        filename=np.array([""], dtype=object),
        cx=np.array([np.nan], dtype=np.float32),
        cy=np.array([24.0], dtype=np.float32),
        w=np.array([12.0], dtype=np.float32),
        h=np.array([12.0], dtype=np.float32),
        theta=np.array([0.0], dtype=np.float32),
        interp_from_start=np.array([2], dtype=np.int64),
        interp_from_end=np.array([2], dtype=np.int64),
        interp_index=np.array([1], dtype=np.int64),
        interp_total=np.array([1], dtype=np.int64),
        obb_corners=np.array([_square(28.0, 24.0, 6.0)], dtype=np.float32),
    )

    pd.DataFrame(
        [
            {
                "TrajectoryID": 1,
                "FrameID": 0,
                "DetectionID": 101,
                "Theta": 0.0,
                "State": "active",
            },
            {
                "TrajectoryID": 2,
                "FrameID": 1,
                "DetectionID": 999,
                "Theta": 0.0,
                "State": "active",
            },
            {
                "TrajectoryID": 3,
                "FrameID": 2,
                "DetectionID": np.nan,
                "Theta": 0.0,
                "State": "occluded",
            },
            {
                "TrajectoryID": 4,
                "FrameID": 3,
                "DetectionID": np.nan,
                "Theta": 0.0,
                "State": "occluded",
            },
        ]
    ).to_csv(final_csv_path, index=False)

    exporter = OrientedTrackVideoExporter(
        dataset_dir,
        final_csv_path,
        video_path=video_path,
        detection_cache_path=cache_path,
        interpolated_roi_npz_path=interp_npz_path,
        fps=5.0,
    )

    result = exporter.export()

    assert result.exported_videos == 1
    assert result.exported_tracks == 1
    assert result.missing_rows == 3
    assert result.missing_detected_rows == 1
    assert result.missing_interpolated_rows == 1
    assert result.invalid_geometry_rows == 1


def test_exporter_uses_supplied_geometry_not_the_fallback(tmp_path: Path):
    """A caller-supplied geometry must drive the canvas and the stamped
    provenance -- not the fallback default. This is the wired path from
    trackerkit.canonical_geometry via media_export.export_final_media()."""
    dataset_dir = tmp_path / "individual_crops" / "run_20260311"
    supplied_geometry = CanonicalGeometry.from_reference(40.0, 3.0, 1.5)

    exporter = OrientedTrackVideoExporter(
        dataset_dir,
        tmp_path / "final.csv",
        video_path=tmp_path / "source.mp4",
        detection_cache_path=tmp_path / "detections.npz",
        fps=5.0,
        geometry=supplied_geometry,
    )

    assert exporter._geometry == supplied_geometry

    affine, out_w, out_h = exporter._canonical_affine_for_task(
        50.0, 50.0, 40.0, 20.0, 0.0
    )
    assert (out_w, out_h) == supplied_geometry.canvas_wh
    assert (out_w, out_h) != CanonicalGeometry.from_reference(20.0, 2.0, 1.3).canvas_wh

    exporter._write_canonical_metadata()
    stamped = json.loads((dataset_dir / "metadata.json").read_text(encoding="utf-8"))
    assert stamped["parameters"]["canonical"]["canvas_wh"] == list(
        supplied_geometry.canvas_wh
    )
    assert read_canonical_provenance(dataset_dir) == supplied_geometry


def test_exporter_without_geometry_falls_back_and_warns(tmp_path, caplog):
    """No geometry supplied -> project-wide default fallback, but it must
    announce itself instead of silently diverging from the real geometry."""
    dataset_dir = tmp_path / "individual_crops" / "run_20260311"

    with caplog.at_level("WARNING"):
        exporter = OrientedTrackVideoExporter(
            dataset_dir,
            tmp_path / "final.csv",
            video_path=tmp_path / "source.mp4",
            detection_cache_path=tmp_path / "detections.npz",
            fps=5.0,
        )

    assert exporter._geometry == CanonicalGeometry.from_reference(20.0, 2.0, 1.3)
    assert any(
        "no project geometry supplied" in record.getMessage()
        for record in caplog.records
    )
