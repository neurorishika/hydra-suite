import numpy as np

from hydra_suite.core.inference.cache.keys import bgsub_detection_cache_key
from hydra_suite.core.inference.config import BgSubConfig, InferenceConfig
from hydra_suite.core.inference.result import OBBResult
from hydra_suite.core.inference.stages.filtering import filter_for_source


def _bg(sizes):
    n = len(sizes)
    return OBBResult(
        frame_idx=0,
        centroids=np.zeros((n, 2), np.float32),
        angles=np.zeros(n, np.float32),
        sizes=np.asarray(sizes, np.float32),
        shapes=np.ones((n, 2), np.float32),
        confidences=np.full(n, np.nan, np.float32),
        corners=np.zeros((n, 4, 2), np.float32),
        detection_ids=OBBResult.make_detection_ids(0, n),
    )


def _cfg(n, mult=20):
    return InferenceConfig(
        bgsub=BgSubConfig(max_targets=n, max_contour_multiplier=mult),
    )


def test_bgsub_key_ignores_n_and_multiplier():
    a = BgSubConfig(params={"MAX_TARGETS": 5, "MAX_CONTOUR_MULTIPLIER": 20})
    b = BgSubConfig(params={"MAX_TARGETS": 50, "MAX_CONTOUR_MULTIPLIER": 3})
    assert bgsub_detection_cache_key(a) == bgsub_detection_cache_key(b)


def test_bgsub_replay_keeps_top_n_by_area():
    out, idx = filter_for_source(_cfg(2), _bg([10.0, 30.0, 20.0]))
    assert sorted(out.sizes.tolist()) == [20.0, 30.0]
    assert sorted(idx.tolist()) == [1, 2]


def test_bgsub_replay_skips_frame_over_contour_budget():
    out, idx = filter_for_source(_cfg(1, mult=2), _bg([1.0, 2.0, 3.0]))
    assert out.num_detections == 0 and len(idx) == 0


def test_bgsub_superset_ignores_n():
    out, _ = filter_for_source(
        _cfg(1, mult=2), _bg([1.0, 2.0, 3.0]), apply_max_detections=False
    )
    assert out.num_detections == 3


def test_bgsub_replay_top_n_is_stable_on_ties():
    out, idx = filter_for_source(_cfg(2), _bg([5.0, 5.0, 5.0]))
    assert idx.tolist() == [0, 1]


def test_detect_objects_gates_are_optional():
    import cv2

    from hydra_suite.core.background.measure import BackgroundMeasurer

    mask = np.zeros((200, 200), np.uint8)
    for i in range(6):
        cv2.circle(mask, (20 + 30 * i, 100), 8, 255, -1)
    m = BackgroundMeasurer(
        {"MAX_TARGETS": 2, "MAX_CONTOUR_MULTIPLIER": 2, "MIN_CONTOUR_AREA": 5}
    )
    gated = m.detect_objects(mask, 0)
    free = m.detect_objects(mask, 0, apply_target_gates=False)
    assert len(gated[0]) == 0  # 6 contours > 2 * 2 budget -> frame skipped
    assert len(free[0]) == 6
