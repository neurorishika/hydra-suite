"""Replay side of the N-free cache contract (Task 8).

Per-animal caches hold every filter survivor (the N-free superset) keyed by
RAW detection-cache index. Replay derives the final-N set (raw indices, a
subset of the superset) and looks each one up by raw index -- never by
position -- and a missing index is an error, never a silent NaN/0 fill.
Per-animal cache keys follow the replay filters, never N.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest

from hydra_suite.core.inference.cache.keys import replay_filter_hash
from hydra_suite.core.inference.config import (
    AprilTagConfig,
    CNNConfig,
    HeadTailConfig,
    InferenceConfig,
    OBBConfig,
    OBBDirectConfig,
    PoseConfig,
    PoseYOLOConfig,
)
from hydra_suite.core.inference.downstream_select import DownstreamCacheError
from hydra_suite.core.inference.result import (
    AprilTagResult,
    CNNDetectionPrediction,
    CNNFactorPrediction,
)
from tests.test_pipeline_superset import _obb


def _cfg(n=4, conf=0.25, **obb_kw):
    return InferenceConfig(
        obb=OBBConfig(
            mode="direct",
            direct=OBBDirectConfig(model_path="/m.pt"),
            max_detections=n,
            confidence_threshold=conf,
            **obb_kw,
        ),
        headtail=HeadTailConfig(model_path="/ht.pt"),
    )


def test_downstream_keys_change_with_filters_not_n(tmp_path):
    from hydra_suite.core.inference.runner import _open_caches

    def k(c, roi=None):
        return _open_caches(c, tmp_path, read_only=True, roi_mask=roi).headtail.key

    assert k(_cfg(n=4)) == k(_cfg(n=400))
    assert k(_cfg(conf=0.25)) != k(_cfg(conf=0.5))
    assert k(_cfg(min_object_size=1.0)) != k(_cfg(min_object_size=2.0))
    assert k(_cfg(iou_threshold=0.5)) != k(_cfg(iou_threshold=0.7))
    roi = np.ones((4, 4), np.uint8)
    assert k(_cfg(), roi) != k(_cfg())
    assert replay_filter_hash(_cfg(n=4), None) == replay_filter_hash(_cfg(n=9), None)


def test_detection_key_has_no_filter_hash(tmp_path):
    from hydra_suite.core.inference.runner import _open_caches

    def k(c):
        return _open_caches(c, tmp_path, read_only=True).detection.key

    assert k(_cfg(conf=0.25)) == k(_cfg(conf=0.5))
    assert k(_cfg(n=4)) == k(_cfg(n=400))


def test_cnn_loader_maps_raw_to_positions():
    from hydra_suite.core.inference.runner import _load_cnn_for_indices

    cache = MagicMock()
    cache.read_frame.return_value = [CNNDetectionPrediction(i, []) for i in (3, 5, 8)]
    cfg = MagicMock(label="id")
    (res,) = _load_cnn_for_indices([cache], [cfg], 0, np.array([8, 3]))
    assert [p.det_index for p in res.predictions] == [0, 1]


def test_cnn_loader_missing_index_raises():
    from hydra_suite.core.inference.runner import _load_cnn_for_indices

    cache = MagicMock()
    cache.read_frame.return_value = [CNNDetectionPrediction(i, []) for i in (3, 5)]
    with pytest.raises(DownstreamCacheError):
        _load_cnn_for_indices([cache], [MagicMock(label="id")], 0, np.array([4]))


def test_load_missing_index_raises():
    from hydra_suite.core.inference.runner import _load_headtail_for_indices

    cache = MagicMock()
    cache.read_frame.return_value = (
        np.array([0, 1], np.int32),
        np.zeros(2, np.float32),
        np.zeros(2, np.float32),
        np.zeros(2, np.uint8),
    )
    with pytest.raises(DownstreamCacheError):
        _load_headtail_for_indices(cache, 0, np.array([2]), None)


def test_pose_loader_missing_index_raises():
    from hydra_suite.core.inference.runner import _load_pose_for_indices

    cache = MagicMock()
    cache.read_frame.return_value = (
        np.zeros((2, 3, 3), np.float32),
        np.array([0, 4], np.int32),
        np.ones(2, bool),
    )
    with pytest.raises(DownstreamCacheError):
        _load_pose_for_indices(cache, 0, np.array([1]), None)


# --- real on-disk caches, replayed at two different N ---------------------


def _replay_cfg(n):
    # min_object_size drops the tiny raw row 2, so the superset raw indices are
    # NON-contiguous ([0, 1, 3, 4, 5, 6, 7]) and raw != positional keying.
    return InferenceConfig(
        obb=OBBConfig(
            mode="direct",
            direct=OBBDirectConfig(model_path="/m.pt"),
            max_detections=n,
            confidence_threshold=0.0,
            iou_threshold=1.0,
            min_object_size=5.0,
        ),
        headtail=HeadTailConfig(model_path="/ht.pt"),
        cnn_phases=[CNNConfig(label="id", model_path="/id.pt")],
        pose=PoseConfig(yolo=PoseYOLOConfig(model_path="/pose.pt")),
        apriltag=AprilTagConfig(enabled=True),
    )


def _write_superset_caches(tmp_path, cfg, raw):
    from hydra_suite.core.inference.runner import _open_caches
    from hydra_suite.core.inference.stages.filtering import filter_for_source

    _, sup = filter_for_source(cfg, raw, None, apply_max_detections=False)
    assert sup.tolist() == [0, 1, 3, 4, 5, 6, 7]
    caches = _open_caches(cfg, tmp_path, write_mode="fresh")
    caches.detection.write_frame(0, result=raw)
    caches.headtail.write_frame(
        0,
        det_indices=sup,
        heading_hints=sup.astype(np.float32) * 10.0,
        heading_confidences=np.ones(len(sup), np.float32),
        directed_mask=np.ones(len(sup), np.uint8),
    )
    caches.cnn[0].write_frame(
        0,
        predictions=[
            CNNDetectionPrediction(
                int(r),
                [CNNFactorPrediction("f", ["a", "b"], np.array([r, 1.0], np.float32))],
            )
            for r in sup
        ],
    )
    kp = np.zeros((len(sup), 2, 3), np.float32)
    kp[:, :, 0] = sup[:, None]
    caches.pose.write_frame(
        0, det_indices=sup, keypoints=kp, valid_mask=np.ones(len(sup), bool)
    )
    caches.apriltag.write_frame(
        0,
        result=AprilTagResult(
            tag_ids=[30, 60],
            det_indices=[3, 6],
            centers=np.array([[3, 0], [6, 0]], np.float32),
            corners=np.zeros((2, 4, 2), np.float32),
        ),
    )
    caches.close()


@pytest.mark.parametrize(
    "n, final",
    [
        # 2N window = raw rows 0..5, minus the dropped row 2, top-3 by conf.
        (3, [0, 1, 3]),
        (5, [0, 1, 3, 4, 5]),
    ],
)
def test_replay_at_other_n_reads_by_raw_index(tmp_path, n, final):
    from hydra_suite.core.inference.runner import (
        _load_apriltag,
        _load_cnn_for_indices,
        _load_headtail_for_indices,
        _load_pose_for_indices,
        _open_caches,
    )
    from hydra_suite.core.inference.stages.filtering import filter_for_source

    raw = _obb(0, 8, small=2)
    _write_superset_caches(tmp_path, _replay_cfg(10), raw)

    cfg = _replay_cfg(n)
    caches = _open_caches(cfg, tmp_path, read_only=True)
    assert all(h.is_valid() for h in caches.all_handles())
    filtered, det_idx = filter_for_source(cfg, raw, None)
    assert det_idx.tolist() == final

    ht = _load_headtail_for_indices(caches.headtail, 0, det_idx, filtered)
    assert ht.heading_hints.tolist() == [10.0 * r for r in final]
    (cnn,) = _load_cnn_for_indices(caches.cnn, cfg.cnn_phases, 0, det_idx)
    assert [p.det_index for p in cnn.predictions] == list(range(len(final)))
    assert [float(p.factors[0].raw_probabilities[0]) for p in cnn.predictions] == [
        float(r) for r in final
    ]
    pose = _load_pose_for_indices(caches.pose, 0, det_idx, filtered)
    assert pose.keypoints[:, 0, 0].tolist() == [float(r) for r in final]
    tags = _load_apriltag(caches.apriltag, 0, det_idx)
    # Raw 3 sits at final position 2; raw 6 is outside both final sets (R6:
    # an absent tag is legitimately dropped).
    assert tags.tag_ids == [30]
    assert tags.det_indices == [2]


def test_identity_sidecar_reads_cnn_by_raw_index(tmp_path):
    from hydra_suite.core.inference.runner import (
        _open_caches,
        write_identity_evidence_sidecar,
    )

    raw = _obb(0, 8, small=2)
    _write_superset_caches(tmp_path, _replay_cfg(10), raw)
    cfg = _replay_cfg(3)
    caches = _open_caches(cfg, tmp_path, read_only=True)

    stage = MagicMock(catalog_labels_by_source={})
    stage.evidences_for_frame.return_value = []
    write_identity_evidence_sidecar(
        caches, cfg, stage, range(0, 1), tmp_path / "ev.npz", ("a", "b")
    )
    (frame_idx, det_ids, cnn_reads, tag_read), _ = stage.evidences_for_frame.call_args
    assert frame_idx == 0
    assert det_ids == [int(raw.detection_ids[r]) for r in (0, 1, 3)]
    preds = cnn_reads["id"]
    assert [p.det_index for p in preds] == [0, 1, 2]
    assert [float(p.factors[0].raw_probabilities[0]) for p in preds] == [0, 1, 3]
    assert tag_read.tag_ids == [30] and tag_read.det_indices == [2]
