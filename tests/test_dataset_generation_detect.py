import numpy as np

import hydra_suite.data.dataset_generation as dg
from hydra_suite.core.inference.result import OBBResult


class _FakeRunner:
    def __init__(self, per_frame):
        self._per_frame = per_frame

    def detect_batch(self, frames, frame_indices=None):
        out = []
        for n in self._per_frame[: len(frames)]:
            out.append(
                OBBResult(
                    frame_idx=0,
                    centroids=np.tile([5.0, 6.0], (n, 1)).astype(np.float32),
                    angles=np.zeros(n, np.float32),
                    sizes=np.ones(n, np.float32),
                    shapes=np.tile([100.0, 2.0], (n, 1)).astype(np.float32),
                    confidences=np.ones(n, np.float32),
                    corners=np.zeros((n, 4, 2), np.float32),
                    detection_ids=OBBResult.make_detection_ids(0, n),
                )
            )
        return out


def test_detect_batch_produces_one_dict_per_frame(monkeypatch):
    runner = _FakeRunner(per_frame=[2, 0])
    frames = [np.zeros((8, 8, 3), np.uint8), np.zeros((8, 8, 3), np.uint8)]
    params = {"RESIZE_FACTOR": 1.0}
    results = dg._detect_batch(runner, frames, [0, 1], [0, 1], params)
    assert isinstance(results, list) and len(results) == 2
    # frame 0 has 2 detections, frame 1 has 0
    assert results[1] == {} or len(results[1]) == 0
