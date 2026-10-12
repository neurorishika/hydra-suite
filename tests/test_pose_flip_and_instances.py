"""Left/right flip pairs and the crop's central pose (core, numpy only)."""

from types import SimpleNamespace

import numpy as np
import pytest

from hydra_suite.core.pose_flip import (
    flip_index,
    normalize_flip_pairs,
    parse_flip_pairs,
    remap_pairs_on_move,
    remap_pairs_on_remove,
    suggest_flip_pairs,
)
from hydra_suite.core.pose_instances import central_instance_index, instance_centers

ANT = [
    "left_antenna_tip",
    "right_antenna_tip",
    "left_antenna_elbow",
    "right_antenna_elbow",
    "clypeus",
    "neck",
    "petiole_post_petiole",
    "tip_of_gaster",
]


# ── flip pairs ───────────────────────────────────────────────────────────


def test_suggest_pairs_left_right_words():
    assert suggest_flip_pairs(ANT) == [(0, 1), (2, 3)]


@pytest.mark.parametrize(
    "names, expected",
    [
        (["L_ear", "R_ear", "nose"], [(0, 1)]),
        (["ear_l", "nose", "ear_r"], [(0, 2)]),
        (["Left Eye", "right eye", "tail"], [(0, 1)]),
        (["leftwing", "rightwing"], [(0, 1)]),
        (["head", "thorax", "abdomen"], []),
        # "l" inside a word is not a side marker
        (["tail", "tair"], []),
    ],
)
def test_suggest_pairs_naming_styles(names, expected):
    assert suggest_flip_pairs(names) == expected


def test_normalize_drops_invalid_and_reused():
    pairs = [(1, 0), (2, 2), (3, 9), (0, 4), ("x", 1), (5, 6)]
    # (1,0) kept as (0,1); self-pair, out of range, reuse of 0, junk dropped
    assert normalize_flip_pairs(pairs, 8) == [(0, 1), (5, 6)]


def test_flip_index_is_a_swap_permutation():
    assert flip_index([(0, 1), (2, 3)], 6) == [1, 0, 3, 2, 4, 5]
    assert flip_index([], 3) == [0, 1, 2]


def test_remap_on_remove_and_move():
    pairs = [(0, 1), (2, 3)]
    assert remap_pairs_on_remove(pairs, 1) == [(1, 2)]
    # move row 0 to row 3: rows 1..3 shift up
    assert remap_pairs_on_move(pairs, 0, 3) == [(3, 0), (1, 2)]


def test_parse_flip_pairs_tolerates_junk():
    data = {"flip_pairs": [[0, 1], [2], "x", [3, "4"], [None, 1]]}
    assert parse_flip_pairs(data) == [(0, 1), (3, 4)]
    assert parse_flip_pairs({}) == []


# ── central instance ─────────────────────────────────────────────────────


def test_central_beats_more_confident_neighbour():
    centers = np.array([[20.0, 20.0], [96.0, 64.0]])  # crop 192x128
    scores = np.array([0.95, 0.40])
    assert central_instance_index(centers, (192, 128), scores) == 1


def test_central_tie_broken_by_score():
    centers = np.array([[96.4, 64.0], [95.8, 64.0]])
    assert central_instance_index(centers, (192, 128), np.array([0.2, 0.9])) == 1


def test_central_without_any_centre_falls_back_to_score():
    centers = np.full((2, 2), np.nan)
    assert central_instance_index(centers, (10, 10), np.array([0.1, 0.7])) == 1
    assert central_instance_index(np.zeros((0, 2)), (10, 10)) == 0


def test_instance_centers_prefers_boxes_then_confident_keypoints():
    xy = np.array(
        [
            [[0.0, 0.0], [10.0, 10.0]],
            [[100.0, 100.0], [0.0, 0.0]],
        ]
    )
    conf = np.array([[0.9, 0.9], [0.9, 0.0]])
    c = instance_centers(xy, conf)
    np.testing.assert_allclose(c, [[5.0, 5.0], [100.0, 100.0]])
    boxes = np.array([[50.0, 60.0, 4.0, 4.0], [1.0, 2.0, 4.0, 4.0]])
    np.testing.assert_allclose(instance_centers(xy, conf, boxes), [[50, 60], [1, 2]])


def test_yolo_backend_reports_the_central_pose():
    torch = pytest.importorskip("torch")
    from hydra_suite.core.individual.pose.backends.yolo import YoloNativeBackend

    xy = torch.tensor(
        [
            [[10.0, 10.0], [14.0, 12.0]],  # neighbour, very confident
            [[94.0, 62.0], [98.0, 66.0]],  # crop's own animal
        ]
    )
    conf = torch.tensor([[0.99, 0.99], [0.5, 0.5]])
    result = SimpleNamespace(
        keypoints=SimpleNamespace(xy=xy, conf=conf),
        boxes=SimpleNamespace(xywh=torch.tensor([[12, 11, 8, 6], [96, 64, 8, 6]])),
        orig_shape=(128, 192),
    )
    backend = YoloNativeBackend.__new__(YoloNativeBackend)
    backend.min_valid_conf = 0.0
    backend.output_keypoint_names = []
    backend._warned_conf_out_of_range = False
    pose = backend._parse_single_result(result)
    assert pose.keypoints is not None
    np.testing.assert_allclose(np.asarray(pose.keypoints)[:, :2], xy[1].numpy())


def test_pose_detection_limit_allows_neighbours():
    from hydra_suite.core.individual.pose.types import PoseRuntimeConfig
    from hydra_suite.core.inference.config import PoseYOLOConfig
    from hydra_suite.core.pose_instances import MAX_POSES_PER_CROP

    assert MAX_POSES_PER_CROP > 1
    assert PoseRuntimeConfig(backend_family="yolo").yolo_max_det == MAX_POSES_PER_CROP
    assert PoseYOLOConfig(model_path="m.pt").max_detections_per_crop > 1
