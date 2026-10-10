import numpy as np
import pytest

from hydra_suite.core.inference.config import OBBConfig, OBBDirectConfig
from hydra_suite.core.inference.result import OBBResult
from hydra_suite.core.inference.stages.filtering import filter_with_indices


def _ranked(confs, xs=None, half=5.0):
    confs = np.asarray(confs, np.float32)
    order = np.lexsort((np.arange(len(confs)), -confs))  # cache invariant
    confs = confs[order]
    n = len(confs)
    xs = (
        np.arange(n, dtype=np.float32) * 40.0
        if xs is None
        else np.asarray(xs, np.float32)[order]
    )
    c = np.stack([xs, np.zeros(n, np.float32)], 1)
    corners = np.stack(
        [c + d for d in ([-half, -half], [half, -half], [half, half], [-half, half])],
        1,
    )
    return OBBResult(
        frame_idx=0,
        centroids=c,
        angles=np.zeros(n, np.float32),
        sizes=np.full(n, 4 * half * half, np.float32),
        shapes=np.tile(np.array([[1.0, 1.0]], np.float32), (n, 1)),
        confidences=confs,
        corners=corners.astype(np.float32),
        detection_ids=OBBResult.make_detection_ids(0, n),
    )


def _cfg(n, conf=0.0, iou=0.5):
    return OBBConfig(
        mode="direct",
        direct=OBBDirectConfig(model_path="/m.pt"),
        max_detections=n,
        confidence_threshold=conf,
        iou_threshold=iou,
    )


def test_window_is_prefix_of_2n():
    raw = _ranked([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
    _, idx = filter_with_indices(raw, _cfg(1, iou=1.0))
    assert idx.tolist() == [0]
    # a later row can never enter, even when earlier rows are filtered
    _, idx = filter_with_indices(raw, _cfg(1, conf=0.85, iou=1.0))
    assert idx.tolist() == [0]
    _, idx = filter_with_indices(raw, _cfg(1, conf=0.95, iou=1.0))
    assert idx.tolist() == []


def test_superset_has_no_window_or_n_cut():
    raw = _ranked([0.9, 0.8, 0.7, 0.6])
    _, idx = filter_with_indices(raw, _cfg(1, iou=1.0), apply_max_detections=False)
    assert idx.tolist() == [0, 1, 2, 3]


@pytest.mark.parametrize("seed", range(25))
def test_final_set_is_subset_of_superset_with_ties(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(5, 60))
    confs = rng.choice([0.3, 0.5, 0.5, 0.7, 0.9], size=n)  # heavy ties
    xs = rng.uniform(0, 300, size=n)  # overlaps -> NMS active
    raw = _ranked(confs, xs, half=8.0)
    _, sup = filter_with_indices(raw, _cfg(1, conf=0.4), apply_max_detections=False)
    for N in (1, 2, 3, 7, 20, 100):
        _, fin = filter_with_indices(raw, _cfg(N, conf=0.4))
        assert set(fin.tolist()) <= set(sup.tolist()), (seed, N)
        again = filter_with_indices(raw, _cfg(N, conf=0.4))[1]
        assert again.tolist() == fin.tolist()


def test_window_excludes_rows_beyond_2n_even_if_earlier_rows_are_gated():
    raw = _ranked([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])  # x = 0,40,80,...
    roi = np.zeros((10, 400), np.uint8)
    roi[:, 70:] = 1  # rows 0 and 1 (x=0,40) fall outside the ROI
    _, idx = filter_with_indices(raw, _cfg(1, iou=1.0), roi)
    assert idx.tolist() == []  # window = rows 0..1, both gated
    _, idx = filter_with_indices(raw, _cfg(2, iou=1.0), roi)
    assert idx.tolist() == [2, 3]  # window = rows 0..3
    _, idx = filter_with_indices(raw, _cfg(1, iou=1.0), roi, apply_max_detections=False)
    assert idx.tolist() == [2, 3, 4, 5]
