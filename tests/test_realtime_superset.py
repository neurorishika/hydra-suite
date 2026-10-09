"""Realtime runner: per-animal stages run on the N-free superset (Task 7).

Mirrors ``tests/test_pipeline_superset.py`` through ``run_realtime``: the
caches hold every filter survivor keyed by RAW detection-cache index; the
returned FrameResult and the identity evidence use the final-N set,
positionally aligned.
"""

from unittest.mock import MagicMock, patch

import numpy as np

from hydra_suite.core.inference.config import (
    AprilTagConfig,
    CNNConfig,
    HeadTailConfig,
    InferenceConfig,
    OBBConfig,
    OBBDirectConfig,
    PoseConfig,
)
from hydra_suite.core.inference.result import (
    AprilTagResult,
    CNNDetectionPrediction,
    CNNResult,
    HeadTailResult,
    OBBResult,
    PoseResult,
)


def _obb(frame_idx, n, small=None):
    # Confidence descends with raw index, so the ranked cache order == raw
    # order. Row ``small`` gets a tiny area so the size filter drops it:
    # superset raw indices become NON-contiguous, making raw keying
    # distinguishable from positional keying.
    xs = np.arange(n, dtype=np.float32) * 50.0
    c = np.stack([xs, np.zeros(n, np.float32)], 1)
    corners = np.stack([c + d for d in ([-4, -4], [4, -4], [4, 4], [-4, 4])], 1)
    sizes = np.full(n, 64.0, np.float32)
    if small is not None:
        sizes[small] = 1.0
    return OBBResult(
        frame_idx,
        c,
        np.zeros(n, np.float32),
        sizes,
        np.ones((n, 2), np.float32),
        np.linspace(0.9, 0.5, n).astype(np.float32),
        corners.astype(np.float32),
        OBBResult.make_detection_ids(frame_idx, n),
    )


def _fake_ht(frame, obb, model, cfg, runtime, geometry):
    return HeadTailResult(
        obb.centroids[:, 0].copy(),
        np.ones(obb.num_detections, np.float32),
        np.ones(obb.num_detections, np.uint8),
        None,
    )


def _fake_cnn(frame, obb, model, cfg, runtime, geometry):
    # chunk-local det_index, payload = the crop's centroid x
    return CNNResult(
        label=cfg.label,
        predictions=[
            CNNDetectionPrediction(det_index=i, factors=[float(obb.centroids[i, 0])])
            for i in range(obb.num_detections)
        ],
    )


def _fake_pose(crops, obb, model, cfg, runtime, geometry):
    kp = np.zeros((obb.num_detections, 1, 3), np.float32)
    kp[:, 0, 0] = obb.centroids[:, 0]
    kp[:, 0, 2] = 1.0
    return PoseResult(keypoints=kp, valid_mask=np.ones(obb.num_detections, bool))


def _fake_apriltag(aabb_crops, obb, model, cfg):
    # Sparse: a tag on every even chunk-local row, tag_id = centroid x.
    rows = list(range(0, obb.num_detections, 2))
    return AprilTagResult(
        tag_ids=[int(obb.centroids[r, 0]) for r in rows],
        det_indices=rows,
        centers=obb.centroids[rows].astype(np.float32),
        corners=obb.corners[rows].astype(np.float32),
    )


def _cfg(max_detections, **kw):
    return InferenceConfig(
        obb=OBBConfig(
            mode="direct",
            direct=OBBDirectConfig(model_path="/m.pt"),
            max_detections=max_detections,
            confidence_threshold=0.0,
            iou_threshold=1.0,
            min_object_size=10.0,
        ),
        **kw,
    )


def _run(cfg, models, caches, obb, tmp_path, frame=None, ht=_fake_ht, pose=None):
    from hydra_suite.core.inference.runner import InferenceRunner

    evidence_calls = []
    frame = np.zeros((64, 600, 3), np.uint8) if frame is None else frame
    with (
        patch(
            "hydra_suite.core.inference.runner._load_all_models", return_value=models
        ),
        patch("hydra_suite.core.inference.runner._open_caches", return_value=caches),
        patch("hydra_suite.core.inference.runner.run_obb", return_value=[obb]),
        patch("hydra_suite.core.inference.runner.run_headtail", side_effect=ht),
        patch("hydra_suite.core.inference.runner.run_cnn", side_effect=_fake_cnn),
        patch(
            "hydra_suite.core.inference.runner.run_pose",
            side_effect=pose or _fake_pose,
        ),
        patch(
            "hydra_suite.core.inference.runner.run_apriltag",
            side_effect=_fake_apriltag,
        ),
    ):
        runner = InferenceRunner(cfg, cache_dir=tmp_path)
        runner._identity_stage = MagicMock()
        runner._write_identity_evidence_realtime = (
            lambda fi, obb_, cnn, at: evidence_calls.append((fi, obb_, cnn, at))
        )
        fr = runner.run_realtime(frame, obb.frame_idx)
    return fr, evidence_calls


def test_realtime_writes_superset_and_returns_final(tmp_path):
    from hydra_suite.core.inference.runner import _AllModels, _CacheSet

    cfg = _cfg(2, headtail=HeadTailConfig(model_path="/ht.pt"))
    caches = _CacheSet(detection=MagicMock(), headtail=MagicMock())
    models = _AllModels(
        obb=MagicMock(), headtail=MagicMock(), cnn=[], pose=None, apriltag=None
    )
    # raw row 1 is dropped by the size filter -> superset raw [0, 2, 3, 4, 5]
    fr, _ = _run(cfg, models, caches, _obb(0, 6, small=1), tmp_path)

    (call,) = caches.headtail.write_frame.call_args_list
    assert call.args[0] == 0
    assert call.kwargs["det_indices"].tolist() == [0, 2, 3, 4, 5]
    assert call.kwargs["heading_hints"].tolist() == [0, 100, 150, 200, 250]
    # the returned in-memory result is the final N=2 set: raw [0, 2]
    assert fr.filtered_indices == [0, 2]
    assert fr.obb.num_detections == 2
    assert fr.obb.centroids[:, 0].tolist() == [0.0, 100.0]
    assert fr.headtail.heading_hints.tolist() == [0.0, 100.0]


def test_realtime_superset_chunked_raw_in_cache_positional_in_memory(tmp_path):
    from hydra_suite.core.inference.runner import _AllModels, _CacheSet

    cfg = _cfg(
        3,
        headtail=HeadTailConfig(model_path="/ht.pt"),
        cnn_phases=[CNNConfig(label="id", model_path="/c.pt")],
        pose=PoseConfig(suppress_foreign_regions=False),
        apriltag=AprilTagConfig(enabled=True),
    )
    caches = _CacheSet(
        detection=MagicMock(),
        headtail=MagicMock(),
        cnn=[MagicMock()],
        pose=MagicMock(),
        apriltag=MagicMock(),
    )
    chunk_sizes = []

    def _ht_record(frame, obb, *a, **k):
        chunk_sizes.append(obb.num_detections)
        return _fake_ht(frame, obb, *a, **k)

    models = _AllModels(
        obb=MagicMock(),
        headtail=MagicMock(),
        cnn=[MagicMock()],
        pose=MagicMock(),
        apriltag=MagicMock(),
    )
    # raw row 1 dropped -> superset raw [0, 2, 3, ..., 9] (9 rows); chunks of 4
    # start at raw 0, 5, 9 (non-identity positions).
    with patch("hydra_suite.core.inference.limits.DOWNSTREAM_CHUNK_SIZE", 4):
        fr, evidence = _run(
            cfg, models, caches, _obb(0, 10, small=1), tmp_path, ht=_ht_record
        )
    sup = [0, 2, 3, 4, 5, 6, 7, 8, 9]
    xs = [50.0 * r for r in sup]

    # crops are materialised a bounded chunk at a time
    assert chunk_sizes == [4, 4, 1]
    # cache: whole superset, in order, keyed by RAW index, payloads aligned
    (ht_call,) = caches.headtail.write_frame.call_args_list
    assert ht_call.kwargs["det_indices"].tolist() == sup
    assert ht_call.kwargs["heading_hints"].tolist() == xs
    (cnn_call,) = caches.cnn[0].write_frame.call_args_list
    preds = cnn_call.kwargs["predictions"]
    assert [p.det_index for p in preds] == sup
    assert [p.factors[0] for p in preds] == xs
    (pose_call,) = caches.pose.write_frame.call_args_list
    assert pose_call.kwargs["det_indices"].tolist() == sup
    assert pose_call.kwargs["keypoints"][:, 0, 0].tolist() == xs
    # AprilTag: sparse, det_indices RAW. Tags sit on chunk-local rows 0, 2 of
    # each chunk -> superset positions 0, 2, 4, 6, 8 -> raw 0, 3, 5, 7, 9.
    (at_call,) = caches.apriltag.write_frame.call_args_list
    at = at_call.kwargs["result"]
    assert list(at.det_indices) == [0, 3, 5, 7, 9]
    assert list(at.tag_ids) == [0, 150, 250, 350, 450]

    # memory: final N=3 = raw [0, 2, 3]; everything positional 0..K-1, aligned
    assert fr.filtered_indices == [0, 2, 3]
    assert fr.obb.centroids[:, 0].tolist() == [0.0, 100.0, 150.0]
    assert [p.det_index for p in fr.cnn[0].predictions] == [0, 1, 2]
    assert [p.factors[0] for p in fr.cnn[0].predictions] == [0.0, 100.0, 150.0]
    assert fr.headtail.heading_hints.tolist() == [0.0, 100.0, 150.0]
    assert fr.pose.keypoints[:, 0, 0].tolist() == [0.0, 100.0, 150.0]
    # tags on raw 0 and 3 -> final positions 0 and 2 (raw 2 has no tag)
    assert list(fr.apriltag.det_indices) == [0, 2]
    assert list(fr.apriltag.tag_ids) == [0, 150]

    # identity evidence: the final-N set, CNN/AprilTag positional over it
    ((ev_fi, ev_obb, ev_cnn, ev_at),) = evidence
    assert ev_fi == 0
    assert ev_obb.detection_ids.tolist() == fr.obb.detection_ids.tolist()
    assert ev_obb.num_detections == 3
    assert [p.det_index for p in ev_cnn[0].predictions] == [0, 1, 2]
    assert [p.factors[0] for p in ev_cnn[0].predictions] == [0.0, 100.0, 150.0]
    assert list(ev_at.det_indices) == [0, 2]


def test_realtime_nonempty_superset_empty_final_writes_superset(tmp_path):
    """A superset that survives the N-free filters but whose final set is empty
    still persists the superset caches and returns an empty frame result."""
    from hydra_suite.core.inference.runner import _AllModels, _CacheSet

    cfg = _cfg(2, headtail=HeadTailConfig(model_path="/ht.pt"))
    caches = _CacheSet(detection=MagicMock(), headtail=MagicMock())
    models = _AllModels(
        obb=MagicMock(), headtail=MagicMock(), cnn=[], pose=None, apriltag=None
    )
    final_empty = (
        _obb(0, 0),
        np.zeros(0, np.int32),
    )
    from hydra_suite.core.inference.stages import filtering

    real = filtering.filter_for_source

    def _filter(config, raw, roi_mask=None, *, apply_max_detections=True):
        if apply_max_detections:
            return final_empty
        return real(config, raw, roi_mask, apply_max_detections=False)

    with patch("hydra_suite.core.inference.runner.filter_for_source", _filter):
        fr, evidence = _run(cfg, models, caches, _obb(0, 4), tmp_path)

    (call,) = caches.headtail.write_frame.call_args_list
    assert call.kwargs["det_indices"].tolist() == [0, 1, 2, 3]
    assert fr.obb.num_detections == 0
    assert fr.filtered_indices == []


# --- R7: pose foreign-region masking uses the FULL superset, not the chunk ---


def _crowded_obb(frame_idx, n):
    # Overlapping 12x12 boxes 6 px apart: every crop sees several neighbours,
    # including ones that land in a different chunk.
    xs = 30.0 + np.arange(n, dtype=np.float32) * 6.0
    c = np.stack([xs, np.full(n, 30.0, np.float32)], 1)
    corners = np.stack([c + d for d in ([-6, -6], [6, -6], [6, 6], [-6, 6])], 1)
    return OBBResult(
        frame_idx,
        c,
        np.zeros(n, np.float32),
        np.full(n, 144.0, np.float32),
        np.ones((n, 2), np.float32),
        np.linspace(0.9, 0.5, n).astype(np.float32),
        corners.astype(np.float32),
        OBBResult.make_detection_ids(frame_idx, n),
    )


def _realtime_pose_crops(chunk_size, tmp_path):
    from hydra_suite.core.inference.runner import _AllModels, _CacheSet

    cfg = _cfg(3, pose=PoseConfig(suppress_foreign_regions=True))
    seen = []

    def _pose(crops, obb, *a, **k):
        seen.append(crops.detach().cpu().clone())
        return _fake_pose(crops, obb, *a, **k)

    rng = np.random.default_rng(7)
    frame = rng.integers(1, 255, (64, 160, 3), dtype=np.uint8)
    models = _AllModels(
        obb=MagicMock(), headtail=None, cnn=[], pose=MagicMock(), apriltag=None
    )
    caches = _CacheSet(detection=MagicMock(), pose=MagicMock())
    with patch("hydra_suite.core.inference.limits.DOWNSTREAM_CHUNK_SIZE", chunk_size):
        _run(cfg, models, caches, _crowded_obb(0, 7), tmp_path, frame=frame, pose=_pose)
    return seen


def test_realtime_pose_foreign_mask_is_independent_of_chunking(tmp_path):
    import torch

    whole = _realtime_pose_crops(256, tmp_path)
    chunked = _realtime_pose_crops(3, tmp_path)
    assert [t.shape[0] for t in whole] == [7]
    assert [t.shape[0] for t in chunked] == [3, 3, 1]
    assert torch.equal(torch.cat(chunked), whole[0])


def test_realtime_superset_round_trips_through_real_caches(tmp_path):
    """On disk: the head-tail cache for frame 0 holds the whole superset."""
    from hydra_suite.core.inference.runner import (
        InferenceRunner,
        _AllModels,
        _open_caches,
    )

    cfg = _cfg(2, headtail=HeadTailConfig(model_path="/ht.pt"))
    models = _AllModels(
        obb=MagicMock(), headtail=MagicMock(), cnn=[], pose=None, apriltag=None
    )
    with (
        patch(
            "hydra_suite.core.inference.runner._load_all_models", return_value=models
        ),
        patch("hydra_suite.core.inference.runner.run_obb", return_value=[_obb(0, 6)]),
        patch("hydra_suite.core.inference.runner.run_headtail", side_effect=_fake_ht),
    ):
        runner = InferenceRunner(cfg, cache_dir=tmp_path)
        fr = runner.run_realtime(np.zeros((64, 600, 3), np.uint8), 0)
        runner.close()
        caches = _open_caches(cfg, tmp_path, runner._video_sig, None, read_only=True)
    assert fr.obb.num_detections == 2
    det_indices, hints, _confs, _directed = caches.headtail.read_frame(0)
    assert det_indices.tolist() == [0, 1, 2, 3, 4, 5]
    assert hints.tolist() == [0, 50, 100, 150, 200, 250]
