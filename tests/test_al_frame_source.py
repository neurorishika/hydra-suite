"""Tests for hydra_suite.data.al.frame_source."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from hydra_suite.data.al.frame_source import FrameRef, VideoFrameSource


def _write_synthetic_video(
    path: Path, n_frames: int, size: tuple[int, int] = (64, 48)
) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 10.0, size)
    try:
        for i in range(n_frames):
            frame = np.full((size[1], size[0], 3), i % 255, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()


def test_video_frame_source_iterates_with_stride(tmp_path):
    video = tmp_path / "synth.mp4"
    _write_synthetic_video(video, n_frames=10)

    src = VideoFrameSource(str(video), stride=2)
    refs = list(src)

    assert all(isinstance(r, FrameRef) for r in refs)
    assert [r.frame_id for r in refs] == [0, 2, 4, 6, 8]
    assert all(r.path is None for r in refs)
    assert src.length() == 10


def test_video_frame_source_read_returns_array(tmp_path):
    video = tmp_path / "synth.mp4"
    _write_synthetic_video(video, n_frames=3)

    src = VideoFrameSource(str(video))
    ref = next(iter(src))
    img = src.read(ref)
    assert img is not None
    assert img.ndim == 3 and img.shape[2] == 3


def test_image_folder_frame_source(tmp_path):
    from hydra_suite.data.al.frame_source import ImageFolderFrameSource

    for i, color in enumerate([10, 50, 90, 130]):
        cv2.imwrite(
            str(tmp_path / f"img_{i:03d}.png"), np.full((8, 8, 3), color, np.uint8)
        )
    (tmp_path / "ignored.txt").write_text("not an image")

    src = ImageFolderFrameSource(str(tmp_path))
    refs = list(src)
    assert len(refs) == 4
    assert [r.frame_id for r in refs] == [0, 1, 2, 3]
    assert all(r.path and r.path.endswith(".png") for r in refs)
    img = src.read(refs[0])
    assert img is not None and img.shape == (8, 8, 3)
    assert src.length() == 4


def test_detectkit_project_source_skips_labeled(tmp_path):
    from hydra_suite.data.al.frame_source import DetectKitProjectSource

    src_dir = tmp_path / "src1"
    (src_dir / "images").mkdir(parents=True)
    (src_dir / "labels").mkdir(parents=True)
    for i in range(3):
        cv2.imwrite(
            str(src_dir / "images" / f"f_{i}.jpg"), np.zeros((4, 4, 3), np.uint8)
        )
    (src_dir / "labels" / "f_1.txt").write_text("0 0.5 0.5 0.6 0.5 0.6 0.6 0.5 0.6\n")

    class _SrcStub:
        def __init__(self, path, name):
            self.path = path
            self.name = name

    class _ProjStub:
        sources = [_SrcStub(str(src_dir), "src1")]

    src = DetectKitProjectSource(_ProjStub(), only_unlabeled=True)
    refs = list(src)
    names = sorted(Path(r.path).stem for r in refs if r.path)
    assert names == ["f_0", "f_2"]
