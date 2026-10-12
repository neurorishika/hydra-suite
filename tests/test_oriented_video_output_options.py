"""Oriented per-track video output options: render scale, pose overlay,
identity-based file names and the high-quality encoder mode."""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from hydra_suite.core.canonicalization.geometry import CanonicalGeometry
from hydra_suite.core.individual.dataset.oriented_video import (
    OrientedTrackVideoExporter,
    TrackPoseOverlay,
)
from hydra_suite.core.individual.identity import columns as C
from hydra_suite.core.post import media_export
from hydra_suite.core.post.rich_export import rich_export_path
from tests.test_oriented_track_video_export import (
    _square,
    _write_detection_cache,
    _write_video,
)

_GEOMETRY = CanonicalGeometry.from_reference(12.0, 1.0, 1.5)  # 18x18 canvas


def _two_track_fixture(tmp_path: Path):
    video_path = tmp_path / "source.mp4"
    cache_path = tmp_path / "detections.npz"
    final_csv = tmp_path / "tracks_final.csv"
    _write_video(video_path, [(40, 80, 120)] * 3)
    _write_detection_cache(
        cache_path,
        [
            {
                "frame_idx": f,
                "meas": [[20.0, 24.0, 0.0], [44.0, 24.0, 0.0]],
                "shapes": [(64.0, 1.0), (64.0, 1.0)],
                "confidences": [0.9, 0.9],
                "obb_corners": [_square(20.0, 24.0, 6.0), _square(44.0, 24.0, 6.0)],
                "detection_ids": [100 + f, 200 + f],
            }
            for f in range(3)
        ],
    )
    rows = []
    for f in range(3):
        rows.append(
            {"TrajectoryID": 1, "FrameID": f, "DetectionID": 100 + f, "Theta": 0.0}
        )
        rows.append(
            {"TrajectoryID": 2, "FrameID": f, "DetectionID": 200 + f, "Theta": 0.0}
        )
    pd.DataFrame(rows).to_csv(final_csv, index=False)
    return video_path, cache_path, final_csv


def _video_wh_and_frames(path: Path):
    cap = cv2.VideoCapture(str(path))
    try:
        assert cap.isOpened()
        return (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
    finally:
        cap.release()


def test_render_scale_and_names_apply_to_videos_only(tmp_path: Path):
    video_path, cache_path, final_csv = _two_track_fixture(tmp_path)
    out = tmp_path / "out"
    exporter = OrientedTrackVideoExporter(
        out,
        final_csv,
        video_path=video_path,
        detection_cache_path=cache_path,
        fps=5.0,
        geometry=_GEOMETRY,
        export_images=True,
        output_subdir="",
        render_scale=4.0,
        track_video_names={1: "antA_track1"},
    )
    result = exporter.export()

    assert result.exported_videos == 2
    named = Path(result.output_dir) / "antA_track1.mp4"
    plain = Path(result.output_dir) / "trajectory_0002.mp4"
    assert named.exists() and plain.exists()
    assert not (Path(result.output_dir) / "trajectory_0001.mp4").exists()
    cw, ch = _GEOMETRY.canvas_wh
    assert _video_wh_and_frames(named) == (cw * 4, ch * 4, 3)

    # Stills are training data: they stay on the project canvas.
    images = sorted(Path(result.image_output_dir).glob("*.png"))
    assert images
    assert cv2.imread(str(images[0])).shape[:2] == (ch, cw)


def test_default_render_is_source_pixel_scale(tmp_path: Path):
    video_path, cache_path, final_csv = _two_track_fixture(tmp_path)
    exporter = OrientedTrackVideoExporter(
        tmp_path / "out",
        final_csv,
        video_path=video_path,
        detection_cache_path=cache_path,
        fps=5.0,
        geometry=_GEOMETRY,
        output_subdir="",
    )
    result = exporter.export()
    clip = Path(result.output_dir) / "trajectory_0001.mp4"
    assert _video_wh_and_frames(clip)[:2] == _GEOMETRY.canvas_wh


def test_pose_keypoint_follows_the_canonical_transform(tmp_path: Path):
    exporter = OrientedTrackVideoExporter(
        tmp_path / "out",
        tmp_path / "final.csv",
        video_path=tmp_path / "source.mp4",
        detection_cache_path=tmp_path / "detections.npz",
        fps=5.0,
        geometry=_GEOMETRY,
        render_scale=2.0,
        pose_overlay=TrackPoseOverlay(
            # One keypoint 3 px right of the animal centre (source pixels).
            keypoints=np.array([[[23.0, 24.0, 0.9]]], dtype=np.float32),
            row_index={(0, 1): 0},
            row_colors=[(0, 0, 255)],
            edges=[],
            min_conf=0.2,
            point_radius=1,
            point_thickness=-1,
            line_thickness=1,
        ),
    )
    task = exporter._build_task(
        frame_id=0,
        trajectory_id=1,
        center_x=20.0,
        center_y=24.0,
        width=12.0,
        height=12.0,
        theta=0.0,
        corners=_square(20.0, 24.0, 6.0),
        polygon_index=0,
    )
    vw, vh = exporter._video_canvas_size(_GEOMETRY.canvas_wh)
    image = np.zeros((vh, vw, 3), dtype=np.uint8)
    exporter._draw_pose(image, task)

    expected = exporter._video_affine(task.affine) @ np.array([23.0, 24.0, 1.0])
    ys, xs = np.nonzero(image[:, :, 2] == 255)
    assert len(xs) > 0
    assert abs(xs.mean() - expected[0]) <= 1.0
    assert abs(ys.mean() - expected[1]) <= 1.0
    # Canvas centre is the animal centre; the keypoint sits 3 source px
    # (6 output px at 2x) along the body axis from it.
    assert abs(abs(expected[0] - vw / 2.0) - 6.0) < 1e-6


def test_identity_names_number_tracks_per_identity_by_first_frame():
    df = pd.DataFrame(
        {
            "TrajectoryID": [5, 5, 2, 2, 7, 9, 3],
            "FrameID": [10, 11, 0, 1, 0, 0, 4],
            C.FINAL_LABEL: ["antA", "antA", "antA", "antA", "unknown", np.nan, "Q/7"],
        }
    )
    names = media_export.build_identity_track_video_names(df)
    assert names == {2: "antA_track1", 5: "antA_track2", 3: "Q_7_track1"}


def _identity_fixture(tmp_path: Path):
    video_path, cache_path, final_csv = _two_track_fixture(tmp_path)
    rich = pd.read_csv(final_csv)
    rich[C.FINAL_LABEL] = np.where(rich["TrajectoryID"] == 1, "queen", "")
    rich.to_csv(rich_export_path(str(final_csv)), index=False)
    return video_path, cache_path, final_csv


def _export(tmp_path, video_path, cache_path, final_csv, config):
    return media_export.export_final_media(
        final_csv_path=str(final_csv),
        config=config,
        video_path=str(video_path),
        detection_cache_path=str(cache_path),
        interpolated_roi_npz_path=None,
        fps=5.0,
        image_root=None,
        video_root=str(tmp_path / "videos"),
        export_images=False,
        export_videos=True,
        background_color=(0, 0, 0),
        geometry=_GEOMETRY,
        params={},
    )


def test_export_final_media_names_identity_tracks_and_writes_index(tmp_path):
    video_path, cache_path, final_csv = _identity_fixture(tmp_path)
    result = _export(
        tmp_path,
        video_path,
        cache_path,
        final_csv,
        {"final_media_export_video_scale": 2.0},
    )
    out = Path(result["output_dir"])
    assert (out / "queen_track1.mp4").exists()
    assert (out / "trajectory_0002.mp4").exists()
    index = pd.read_csv(out / "track_index.csv")
    assert index.to_dict("records") == [
        {"video": "queen_track1.mp4", "TrajectoryID": 1}
    ]
    cw, ch = _GEOMETRY.canvas_wh
    assert _video_wh_and_frames(out / "queen_track1.mp4")[:2] == (cw * 2, ch * 2)


def test_export_final_media_ignores_identity_when_identity_is_off(tmp_path):
    video_path, cache_path, final_csv = _identity_fixture(tmp_path)
    result = _export(
        tmp_path,
        video_path,
        cache_path,
        final_csv,
        {"enable_identity_analysis": False},
    )
    out = Path(result["output_dir"])
    assert (out / "trajectory_0001.mp4").exists()
    assert not (out / "track_index.csv").exists()


def test_high_quality_encoder_uses_software_x264(tmp_path: Path):
    from hydra_suite.utils import video_encoder as ve

    if not ve._libx264_available():
        pytest.skip("libx264 not available")
    enc = ve.VideoEncoder(
        tmp_path / "a.mp4",
        fps=5.0,
        width=32,
        height=16,
        backend="videotoolbox",
        quality="high",
    )
    try:
        assert enc._backend == "pyav_software"
        assert enc._stream.options == ve._HIGH_QUALITY_X264_OPTS
        enc.write(np.zeros((16, 32, 3), dtype=np.uint8))
    finally:
        enc.release()


def test_pose_overlay_track_colors_follow_the_preview_palette():
    df = pd.DataFrame(
        {
            "TrajectoryID": [1, 1, 2],
            "FrameID": [0, 1, 0],
            "PoseKpt_head_X": [1.0, 2.0, 3.0],
            "PoseKpt_head_Y": [1.0, 2.0, 3.0],
            "PoseKpt_head_Conf": [0.9, 0.9, 0.9],
        }
    )
    colors = [(10, 20, 30), (40, 50, 60)]
    overlay = media_export.build_track_pose_overlay(
        df,
        {"TRAJECTORY_COLORS": colors},
        {"final_media_export_show_pose": True},
        scale=2.0,
    )
    assert overlay is not None
    assert overlay.row_index == {(0, 1): 0, (1, 1): 1, (0, 2): 2}
    assert overlay.row_colors[0] == overlay.row_colors[1]
    assert overlay.row_colors[0] != overlay.row_colors[2]
    assert overlay.point_radius == 6  # default 3 source px at 2x
    assert media_export.build_track_pose_overlay(df, {}, {}) is None
