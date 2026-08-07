import json
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from hydra_suite.core.canonicalization.geometry import CanonicalGeometry
from hydra_suite.core.identity.dataset.naming import read_canonical_provenance
from hydra_suite.core.identity.dataset.oriented_video import (
    OrientedTrackVideoExporter,
    resolve_individual_dataset_dir,
    resolve_oriented_track_video_dir,
)
from hydra_suite.data.detection_cache import DetectionCache


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

    with DetectionCache(cache_path, mode="w", start_frame=0, end_frame=3) as cache:
        cache.add_frame(
            0,
            meas=[np.array([20.0, 24.0, 0.0], dtype=np.float32)],
            sizes=[64.0],
            shapes=[(64.0, 1.0)],
            confidences=[0.9],
            obb_corners=[_square(20.0, 24.0, 6.0)],
            detection_ids=[101],
        )
        cache.add_frame(
            1,
            meas=[np.array([24.0, 24.0, 0.0], dtype=np.float32)],
            sizes=[64.0],
            shapes=[(64.0, 1.0)],
            confidences=[0.9],
            obb_corners=[_square(24.0, 24.0, 6.0)],
            detection_ids=[102],
        )
        cache.add_frame(2, [], [], [], [])
        cache.add_frame(
            3,
            meas=[np.array([42.0, 24.0, 0.0], dtype=np.float32)],
            sizes=[64.0],
            shapes=[(64.0, 1.0)],
            confidences=[0.9],
            obb_corners=[_square(42.0, 24.0, 6.0)],
            detection_ids=[201],
        )
        cache.save()

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
        padding_fraction=0.0,
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

    with DetectionCache(cache_path, mode="w", start_frame=0, end_frame=3) as cache:
        cache.add_frame(
            0,
            meas=[np.array([20.0, 24.0, 0.0], dtype=np.float32)],
            sizes=[64.0],
            shapes=[(64.0, 1.0)],
            confidences=[0.9],
            obb_corners=[_square(20.0, 24.0, 6.0)],
            detection_ids=[101],
        )
        cache.add_frame(
            1,
            meas=[np.array([24.0, 24.0, 0.0], dtype=np.float32)],
            sizes=[64.0],
            shapes=[(64.0, 1.0)],
            confidences=[0.9],
            obb_corners=[_square(24.0, 24.0, 6.0)],
            detection_ids=[102],
        )
        cache.add_frame(2, [], [], [], [])
        cache.add_frame(
            3,
            meas=[np.array([42.0, 24.0, 0.0], dtype=np.float32)],
            sizes=[64.0],
            shapes=[(64.0, 1.0)],
            confidences=[0.9],
            obb_corners=[_square(42.0, 24.0, 6.0)],
            detection_ids=[201],
        )
        cache.save()

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
        padding_fraction=0.0,
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


def test_oriented_track_video_prefers_directed_heading_and_preserves_branch(
    tmp_path: Path,
):
    dataset_dir = tmp_path / "individual_crops" / "run_20260311"
    cache_path = tmp_path / "detections.npz"
    interp_npz_path = tmp_path / "interpolated_rois.npz"
    final_csv_path = tmp_path / "tracks_final.csv"
    video_path = tmp_path / "source.mp4"

    _write_video(video_path, [(0, 0, 255), (0, 255, 0)])

    with DetectionCache(cache_path, mode="w", start_frame=0, end_frame=1) as cache:
        cache.add_frame(
            0,
            meas=[np.array([20.0, 24.0, math.pi], dtype=np.float32)],
            sizes=[64.0],
            shapes=[(64.0, 1.0)],
            confidences=[0.9],
            obb_corners=[_square(20.0, 24.0, 6.0)],
            detection_ids=[101],
            heading_hints=[0.0],
            directed_mask=[1],
        )
        cache.add_frame(1, [], [], [], [])
        cache.save()

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
                "Theta": math.pi,
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
        padding_fraction=0.0,
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
    expected_affine, expected_w, expected_h = exporter._compute_affine(
        20.0,
        24.0,
        12.0,
        12.0,
        0.0,
    )
    interp_expected_affine, interp_expected_w, interp_expected_h = (
        exporter._compute_affine(
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

    with DetectionCache(cache_path, mode="w", start_frame=0, end_frame=2) as cache:
        cache.add_frame(
            0,
            meas=[np.array([20.0, 24.0, 0.0], dtype=np.float32)],
            sizes=[64.0],
            shapes=[(64.0, 1.0)],
            confidences=[0.9],
            obb_corners=[_square(20.0, 24.0, 6.0)],
            detection_ids=[101],
        )
        cache.add_frame(
            1,
            meas=[np.array([24.0, 24.0, math.pi], dtype=np.float32)],
            sizes=[64.0],
            shapes=[(64.0, 1.0)],
            confidences=[0.9],
            obb_corners=[_square(24.0, 24.0, 6.0)],
            detection_ids=[102],
        )
        cache.add_frame(
            2,
            meas=[np.array([28.0, 24.0, 0.0], dtype=np.float32)],
            sizes=[64.0],
            shapes=[(64.0, 1.0)],
            confidences=[0.9],
            obb_corners=[_square(28.0, 24.0, 6.0)],
            detection_ids=[103],
        )
        cache.save()

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
        padding_fraction=0.0,
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
    expected_affine, expected_w, expected_h = exporter._compute_affine(
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

    with DetectionCache(cache_path, mode="w", start_frame=0, end_frame=2) as cache:
        cache.add_frame(
            0,
            meas=[np.array([20.0, 24.0, 0.0], dtype=np.float32)],
            sizes=[64.0],
            shapes=[(64.0, 1.0)],
            confidences=[0.9],
            obb_corners=[_square(20.0, 24.0, 6.0)],
            detection_ids=[101],
        )
        cache.add_frame(
            1,
            meas=[np.array([28.0, 24.0, 0.0], dtype=np.float32)],
            sizes=[256.0],
            shapes=[(256.0, 1.0)],
            confidences=[0.9],
            obb_corners=[_square(28.0, 24.0, 12.0)],
            detection_ids=[102],
        )
        cache.add_frame(
            2,
            meas=[np.array([20.0, 24.0, 0.0], dtype=np.float32)],
            sizes=[64.0],
            shapes=[(64.0, 1.0)],
            confidences=[0.9],
            obb_corners=[_square(20.0, 24.0, 6.0)],
            detection_ids=[103],
        )
        cache.save()

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
        padding_fraction=0.0,
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
    expected_affine, expected_w, expected_h = exporter._compute_affine(
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

    with DetectionCache(cache_path, mode="w", start_frame=0, end_frame=3) as cache:
        cache.add_frame(
            0,
            meas=[np.array([20.0, 24.0, 0.0], dtype=np.float32)],
            sizes=[64.0],
            shapes=[(64.0, 1.0)],
            confidences=[0.9],
            obb_corners=[_square(20.0, 24.0, 6.0)],
            detection_ids=[101],
        )
        cache.add_frame(1, [], [], [], [])
        cache.add_frame(2, [], [], [], [])
        cache.add_frame(3, [], [], [], [])
        cache.save()

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
        padding_fraction=0.0,
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
    provenance -- not the padding-driven fallback default. This is the
    wired path from trackerkit.canonical_geometry via video_worker.py."""
    dataset_dir = tmp_path / "individual_crops" / "run_20260311"
    supplied_geometry = CanonicalGeometry.from_reference(40.0, 3.0, 1.5)

    exporter = OrientedTrackVideoExporter(
        dataset_dir,
        tmp_path / "final.csv",
        video_path=tmp_path / "source.mp4",
        detection_cache_path=tmp_path / "detections.npz",
        fps=5.0,
        padding_fraction=0.1,
        geometry=supplied_geometry,
    )

    assert exporter._geometry == supplied_geometry

    affine, out_w, out_h = exporter._canonical_affine_for_task(
        50.0, 50.0, 40.0, 20.0, 0.0
    )
    assert (out_w, out_h) == supplied_geometry.canvas_wh
    assert (out_w, out_h) != CanonicalGeometry.from_reference(20.0, 2.0, 1.1).canvas_wh

    exporter._write_canonical_metadata()
    stamped = json.loads((dataset_dir / "metadata.json").read_text(encoding="utf-8"))
    assert stamped["parameters"]["canonical"]["canvas_wh"] == list(
        supplied_geometry.canvas_wh
    )
    assert read_canonical_provenance(dataset_dir) == supplied_geometry


def test_exporter_without_geometry_falls_back_and_warns(tmp_path, caplog):
    """No geometry supplied -> padding-driven fallback, but it must announce
    itself instead of silently diverging from the project's real geometry."""
    dataset_dir = tmp_path / "individual_crops" / "run_20260311"

    with caplog.at_level("WARNING"):
        exporter = OrientedTrackVideoExporter(
            dataset_dir,
            tmp_path / "final.csv",
            video_path=tmp_path / "source.mp4",
            detection_cache_path=tmp_path / "detections.npz",
            fps=5.0,
            padding_fraction=0.1,
        )

    assert exporter._geometry == CanonicalGeometry.from_reference(20.0, 2.0, 1.1)
    assert any(
        "no project geometry supplied" in record.getMessage()
        for record in caplog.records
    )
