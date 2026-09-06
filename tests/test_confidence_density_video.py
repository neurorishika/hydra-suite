import tempfile
from pathlib import Path

import numpy as np

import hydra_suite.core.tracking.confidence.confidence_density as density_module
from hydra_suite.core.tracking.confidence.confidence_density import (
    DensityRegion,
    export_diagnostic_video,
)


def _fake_frame_reader(n_frames, h=64, w=64):
    def reader(frame_idx):
        if frame_idx >= n_frames:
            return None
        return np.full((h, w, 3), 128, dtype=np.uint8)

    return reader, n_frames, h, w


def test_export_diagnostic_video_creates_file():
    """export_diagnostic_video writes an mp4 file."""
    reader, n_frames, h, w = _fake_frame_reader(10)
    grids = [np.zeros((h, w), dtype=np.float32) for _ in range(n_frames)]
    grids[3][10:20, 10:20] = 1.0
    regions = [DensityRegion("region-1", 3, 3, (10, 10, 20, 20))]

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "test_confidence_map.mp4"
        export_diagnostic_video(
            frame_reader=reader,
            n_frames=n_frames,
            frame_h=h,
            frame_w=w,
            density_grids=grids,
            regions=regions,
            output_path=out,
            fps=5,
        )
        assert out.exists()
        assert out.stat().st_size > 0


def test_export_diagnostic_video_uses_absolute_sparse_density_indices(
    monkeypatch, tmp_path
):
    """A missing cache key gets no adjacent row's diagnostic evidence."""

    writers = []

    class _Writer:
        def __init__(self, *_args, **_kwargs):
            self.frames = []
            writers.append(self)

        def write(self, frame):
            self.frames.append(frame.copy())

        def release(self):
            pass

    source_indices = []

    def _reader(frame_index):
        source_indices.append(frame_index)
        return np.zeros((4, 4, 3), dtype=np.uint8)

    monkeypatch.setattr(density_module, "VideoEncoder", _Writer)
    density_module.export_diagnostic_video(
        frame_reader=_reader,
        n_frames=3,
        frame_h=4,
        frame_w=4,
        density_grids=np.ones((2, 4, 4), dtype=np.float32),
        regions=[],
        output_path=tmp_path / "ignored.mp4",
        heatmap_alpha=1.0,
        density_frame_indices=np.array([100, 102], dtype=np.int64),
        start_frame_index=100,
    )

    assert source_indices == [100, 101, 102]
    assert len(writers) == 1
    assert writers[0].frames[0][..., 2].max() == 255
    assert writers[0].frames[1].max() == 0
    assert writers[0].frames[2][..., 2].max() == 255
