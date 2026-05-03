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
