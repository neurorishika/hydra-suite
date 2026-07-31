import numpy as np

from hydra_suite.core.identity.pose.vitpose.training.validate import pck_from_preds


def test_pck_perfect_and_thresholded():
    gt = np.array([[10, 10, 2], [20, 20, 2]], np.float32)
    bbox = np.array([0, 0, 40, 40], np.float32)  # norm = sqrt(1600) = 40
    perfect = gt[:, :2].copy()
    assert pck_from_preds(perfect, gt, bbox, (0.05, 0.1))[0.05] == 1.0
    # move one joint 3 px: 3/40 = 0.075 -> fails @0.05, passes @0.1
    off = gt[:, :2].copy()
    off[0, 0] += 3.0
    r = pck_from_preds(off, gt, bbox, (0.05, 0.1))
    assert r[0.05] == 0.5 and r[0.1] == 1.0


def test_pck_ignores_invisible():
    gt = np.array([[10, 10, 2], [20, 20, 0]], np.float32)  # joint 1 unlabelled
    bbox = np.array([0, 0, 40, 40], np.float32)
    pred = np.array([[10, 10], [999, 999]], np.float32)
    assert pck_from_preds(pred, gt, bbox, (0.05,))[0.05] == 1.0
