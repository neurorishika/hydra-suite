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


def test_vitpose_local_flip_index_matches_core():
    pytest.importorskip("torch")
    from hydra_suite.core.individual.pose.vitpose.training.dataset import (
        flip_index as vit_flip_index,
    )

    for pairs, k in (([(0, 1), (2, 3)], 6), ([], 3), ([(1, 4)], 5)):
        assert vit_flip_index(pairs, k) == flip_index(pairs, k)


def test_central_ignores_low_confidence_fragments():
    # real animal 4 px off-centre at 0.95; junk box exactly centred at 3e-4
    centers = np.array([[100.0, 64.0], [96.0, 64.0]])
    assert central_instance_index(centers, (192, 128), np.array([0.95, 3e-4])) == 0
    # a partly hidden central animal (0.4) still beats a clear neighbour
    centers = np.array([[20.0, 20.0], [96.0, 64.0]])
    assert central_instance_index(centers, (192, 128), np.array([0.95, 0.4])) == 1
    # nothing confident: the best one wins
    assert central_instance_index(centers, (192, 128), np.array([0.05, 0.02])) == 0


def test_yolo_pose_cache_key_tracks_the_selection_rule():
    from hydra_suite.core.canonicalization.geometry import CanonicalGeometry
    from hydra_suite.core.inference.cache.keys import pose_cache_key
    from hydra_suite.core.inference.config import PoseConfig, PoseYOLOConfig

    geom = CanonicalGeometry(canvas_wh=(96, 48), margin=1.3, aspect_ratio=2.0)
    one = PoseConfig(
        backend="yolo",
        yolo=PoseYOLOConfig(model_path="m.pt", max_detections_per_crop=1),
    )
    ten = PoseConfig(backend="yolo", yolo=PoseYOLOConfig(model_path="m.pt"))
    assert (
        pose_cache_key(one, geom).config_hash != pose_cache_key(ten, geom).config_hash
    )


SIDE_PAIRS = [
    ["left_antenna_tip", "right_antenna_tip"],
    ["Left Eye", "right eye"],
    ["eyeLeft", "eyeRight"],
    ["l_ear", "r_ear"],
    ["ear_l", "ear_r"],
    ["l.ant", "r.ant"],
    ["antL", "antR"],
    ["forelegL4", "forelegR4"],
    ["L1", "R1"],
    ["legL1", "legR1"],
    ["LF", "RF"],
    ["lf_leg", "rf_leg"],
    ["lfpaw", "rfpaw"],
    ["FL", "FR"],
    ["FL_paw", "FR_paw"],
    ["paw_fl", "paw_fr"],
    ["lhip", "rhip"],
    ["Leye", "Reye"],
    ["lear", "rear"],
    ["LFoot", "RFoot"],
]
MIDLINE_CAPS = [
    "HEAD",
    "THORAX",
    "ABDOMEN",
    "LEG_TIP",
    "STING",
    "RAY1",
    "BRAIN",
    "FLANK",
    "MRI_MARK",
    "LEG",
    "LIP",
    "ROOT",
    "REAR",
    "RUMP",
    "LOBE",
    "LR",
    "BL",
]
MIDLINE = [
    "head",
    "thorax",
    "abdomen_tip",
    "tail_root",
    "throat",
    "clypeus",
    "neck",
    "petiole_post_petiole",
    "tip_of_gaster",
    "rostrum",
    "rear",
    "bright",
    "tarsus",
    "front",
    "hind",
    "mid",
    "withers",
    "labrum",
    "frons",
    "rump",
]


@pytest.mark.parametrize("pair", SIDE_PAIRS, ids=lambda p: p[0])
def test_every_detected_side_style_is_suggestible(pair):
    from hydra_suite.core.pose_flip import is_side_named

    names = ["head"] + pair
    assert all(is_side_named(n, names) for n in pair)
    assert suggest_flip_pairs(names) == [(1, 2)]


def test_midline_names_are_not_sides():
    from hydra_suite.core.pose_flip import is_side_named

    assert [n for n in MIDLINE if is_side_named(n, MIDLINE)] == []
    assert [n for n in MIDLINE_CAPS if is_side_named(n, MIDLINE_CAPS)] == []


def test_flip_override_for_side_like_midline_names():
    from hydra_suite.posekit.core.training_options import plan_flips

    names = ["head", "l-r", "tail"]  # flagged, but no partner exists
    assert plan_flips(names, [], 0.5, 0.5).blocked
    plan = plan_flips(names, [], 0.5, 0.5, allow_unpaired=True)
    assert plan.enabled and plan.flip_idx == [0, 1, 2]


def test_yolo_predict_size_from_model_or_export_metadata(tmp_path):
    pytest.importorskip("torch")
    import json

    from hydra_suite.core.individual.pose.backends import yolo as Y

    assert Y._imgsz_value([128, 192]) == 192
    assert Y._imgsz_value("x") is None and Y._imgsz_value(0) is None
    pt = SimpleNamespace(overrides={"imgsz": 192})
    assert Y._trained_imgsz(pt, str(tmp_path / "m.pt")) == 192

    def engine(name, meta):
        blob = json.dumps(meta).encode()
        path = tmp_path / name
        path.write_bytes(len(blob).to_bytes(4, "little") + blob + b"\0" * 16)
        return str(path)

    none = SimpleNamespace(overrides={})
    dyn = engine("d.engine", {"imgsz": [192, 192], "args": {"dynamic": True}})
    static = engine("s.engine", {"imgsz": [320, 320], "args": {"dynamic": False}})
    assert Y._trained_imgsz(none, dyn) == 192  # would otherwise run at 640
    assert Y._trained_imgsz(none, static) is None  # its fixed input is used
    junk = tmp_path / "j.engine"
    junk.write_bytes(b"\xff\xff\xff\x7fgarbage")
    assert Y._trained_imgsz(none, str(junk)) is None


def test_prediction_cache_version_bumped_for_yolo_only(tmp_path):
    from hydra_suite.integrations.sleap.service import PoseInferenceService

    svc = PoseInferenceService(tmp_path, ["a", "b"])
    assert svc._cache_version("yolo") == PoseInferenceService.YOLO_CACHE_VERSION
    assert svc._cache_version("sleap") == svc._cache_version("vitpose") == 3
