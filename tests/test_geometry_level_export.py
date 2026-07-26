import numpy as np

from hydra_suite.core.inference.result import OBBResult


def _empty_result():
    return OBBResult(
        frame_idx=0,
        centroids=np.zeros((0, 2), np.float32),
        angles=np.zeros((0,), np.float32),
        sizes=np.zeros((0,), np.float32),
        shapes=np.zeros((0, 2), np.float32),
        confidences=np.zeros((0,), np.float32),
        corners=np.zeros((0, 4, 2), np.float32),
        detection_ids=np.zeros((0,), np.int64),
    )


def test_polygons_defaults_none():
    assert _empty_result().polygons is None


def test_polygons_settable():
    r = _empty_result()
    r.polygons = [np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]], np.float32)]
    assert len(r.polygons) == 1
