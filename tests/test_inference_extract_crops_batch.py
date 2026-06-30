import numpy as np
import torch

from hydra_suite.core.inference.result import OBBResult
from hydra_suite.core.inference.runtime import RuntimeContext
from hydra_suite.core.inference.stages.crops import extract_canonical_crops_batch


def _runtime_cpu():
    return RuntimeContext(
        cuda_mode=False,
        device="cpu",
        use_nvdec=False,
        default_runtime="cpu",
        tensor_on_cuda=False,
    )


def _obb(frame_idx, n):
    cx = np.linspace(20, 40, n).astype(np.float32)
    corners = np.stack(
        [
            np.stack([cx - 5, np.full(n, 15, np.float32)], -1),
            np.stack([cx + 5, np.full(n, 15, np.float32)], -1),
            np.stack([cx + 5, np.full(n, 25, np.float32)], -1),
            np.stack([cx - 5, np.full(n, 25, np.float32)], -1),
        ],
        axis=1,
    ).astype(np.float32)
    return OBBResult(
        frame_idx=frame_idx,
        centroids=np.stack([cx, np.full(n, 20, np.float32)], -1),
        angles=np.zeros(n, np.float32),
        sizes=np.full(n, 100, np.float32),
        shapes=np.ones((n, 2), np.float32),
        confidences=np.ones(n, np.float32),
        corners=corners,
        detection_ids=np.array([frame_idx * 10000 + s for s in range(n)], np.int64),
    )


def test_extract_crops_concatenates_window_in_detection_id_order():
    frames = [np.zeros((64, 64, 3), np.uint8), np.zeros((64, 64, 3), np.uint8)]
    obbs = [_obb(0, 2), _obb(1, 1)]
    batch = extract_canonical_crops_batch(frames, obbs, 2.0, 1.3, _runtime_cpu())
    assert batch.crops.shape[0] == 3
    assert list(batch.detection_ids) == [0, 1, 10000]
    assert list(batch.frame_index) == [0, 0, 1]
    assert batch.frames() == [0, 1]
