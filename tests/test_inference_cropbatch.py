import numpy as np
import torch
from hydra_suite.core.inference.result import CropBatch, OBBResult


def _obb(frame_idx, n):
    return OBBResult(
        frame_idx=frame_idx,
        centroids=np.zeros((n, 2), np.float32),
        angles=np.zeros(n, np.float32),
        sizes=np.ones(n, np.float32),
        shapes=np.ones((n, 2), np.float32),
        confidences=np.ones(n, np.float32),
        corners=np.zeros((n, 4, 2), np.float32),
        detection_ids=np.array([frame_idx * 10000 + s for s in range(n)], np.int64),
    )


def test_cropbatch_indexes_rows_by_frame():
    batch = CropBatch(
        crops=torch.zeros(3, 3, 8, 8),
        detection_ids=np.array([0, 1, 10000], np.int64),
        frame_index=np.array([0, 0, 1], np.int64),
        obb_by_frame={0: _obb(0, 2), 1: _obb(1, 1)},
        native_sizes=np.array([[8, 8], [8, 8], [8, 8]], np.int64),
    )
    assert batch.frames() == [0, 1]
    rows = batch.select_frame(1)
    assert list(rows) == [2]
    rows0 = batch.select_frame(0)
    assert list(rows0) == [0, 1]
