import tempfile
from pathlib import Path

import numpy as np

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
