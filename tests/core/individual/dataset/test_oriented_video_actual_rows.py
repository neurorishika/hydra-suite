"""Actual-row (cached-detection) geometry must come from the shared read-only
detection-cache reader, and a missing cache file must degrade gracefully
rather than raising."""

from pathlib import Path

import numpy as np
import pandas as pd

from hydra_suite.core.individual.dataset.oriented_video import (
    FrameBundle,
    OrientedTrackVideoExporter,
)
from hydra_suite.core.inference.cache.base import CacheKey
from hydra_suite.core.inference.cache.store import DetectionCacheHandle
from hydra_suite.core.inference.result import OBBResult


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


def _write_one_frame(path: Path) -> None:
    key = CacheKey(schema_version=3, model_path="m", model_mtime=1.0, config_hash="h")
    handle = DetectionCacheHandle(path=path, key=key)
    handle.write_frame(
        0,
        result=OBBResult(
            frame_idx=0,
            centroids=np.array([[20.0, 24.0]], np.float32),
            angles=np.array([0.0], np.float32),
            sizes=np.array([64.0], np.float32),
            shapes=np.array([[64.0, 1.0]], np.float32),
            confidences=np.array([0.9], np.float32),
            corners=np.array([_square(20.0, 24.0, 6.0)], np.float32),
            detection_ids=np.array([101], np.int64),
        ),
    )
    handle.close()


def _make_exporter(
    tmp_path: Path, detection_cache_path: Path
) -> OrientedTrackVideoExporter:
    return OrientedTrackVideoExporter(
        tmp_path / "individual_crops" / "run",
        tmp_path / "tracks_final.csv",
        video_path=tmp_path / "source.mp4",
        detection_cache_path=detection_cache_path,
        fps=5.0,
    )


def test_actual_rows_build_geometry_from_modern_detection_cache(tmp_path: Path):
    cache_path = tmp_path / "detection.npz"
    _write_one_frame(cache_path)

    exporter = _make_exporter(tmp_path, cache_path)
    row = next(
        pd.DataFrame(
            [{"TrajectoryID": 1, "FrameID": 0, "DetectionID": 101, "Theta": 0.0}]
        ).itertuples(index=False)
    )

    from hydra_suite.core.inference.cache.reader import open_detection_cache_reader

    reader = open_detection_cache_reader(cache_path)
    bundle = FrameBundle()
    try:
        missing = exporter._add_actual_tasks(
            reader,
            0,
            [row],
            bundle,
            track_sizes={},
            track_theta_state={},
        )
    finally:
        reader.close()

    assert missing == {"missing_detected_rows": 0, "invalid_geometry_rows": 0}
    assert len(bundle.tasks) == 1
    task = bundle.tasks[0]
    assert task.affine is not None
    assert task.corners.shape == (4, 2)
    assert np.isclose(task.center_x, 20.0)
    assert np.isclose(task.center_y, 24.0)


def test_missing_detection_cache_file_yields_zero_actual_tasks_without_raising(
    tmp_path: Path,
):
    missing_cache_path = tmp_path / "does_not_exist_detection.npz"
    assert not missing_cache_path.exists()

    exporter = _make_exporter(tmp_path, missing_cache_path)

    pd.DataFrame(
        [{"TrajectoryID": 1, "FrameID": 0, "DetectionID": 101, "Theta": 0.0}]
    ).to_csv(exporter.final_csv_path, index=False)

    trajectories_df = exporter._load_final_dataframe()
    frame_bundles, track_sizes, missing_rows = exporter._build_frame_bundles(
        trajectories_df,
        {},
    )

    assert frame_bundles == {}
    assert track_sizes == {}
    assert missing_rows == 1
    assert exporter._last_missing_breakdown["missing_detected_rows"] == 1
