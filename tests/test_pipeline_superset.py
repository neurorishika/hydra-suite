"""Batch pipeline: per-animal stages run on the N-free superset (Task 6).

The cache holds every filter survivor keyed by RAW detection-cache index; the
in-memory FrameResult is narrowed to the final-N set, positionally aligned.
"""

from unittest.mock import MagicMock, patch

import numpy as np

from hydra_suite.core.inference.config import (
    CNNConfig,
    HeadTailConfig,
    InferenceConfig,
    OBBConfig,
    OBBDirectConfig,
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
    # order. Row ``small`` gets a tiny area so the size filter (applied after
    # rank_and_bound) drops it: superset raw indices become NON-contiguous,
    # making raw keying distinguishable from positional keying.
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


def _fake_ht(frames, obbs, model, cfg, runtime, geometry, canonical_batch=None):
    return {
        o.frame_idx: HeadTailResult(
            o.centroids[:, 0].copy(),
            np.ones(o.num_detections, np.float32),
            np.ones(o.num_detections, np.uint8),
            None,
        )
        for o in obbs
    }


def _fake_cnn(frames, obbs, model, cfg, runtime, geometry, headtail_by_frame=None):
    # Tag each prediction with its chunk-local position AND the crop's x so the
    # test can check both index spaces (raw in the cache, positional in memory).
    return {
        o.frame_idx: CNNResult(
            label=cfg.label,
            predictions=[
                CNNDetectionPrediction(det_index=i, factors=[float(o.centroids[i, 0])])
                for i in range(o.num_detections)
            ],
        )
        for o in obbs
    }


def _fake_pose(crop_batch, model, pcfg, runtime, geometry):
    # keypoint x = the detection's centroid x (payload identifies the detection)
    out = {}
    for fi, o in crop_batch.obb_by_frame.items():
        kp = np.zeros((o.num_detections, 1, 3), np.float32)
        kp[:, 0, 0] = o.centroids[:, 0]
        kp[:, 0, 2] = 1.0
        out[fi] = PoseResult(keypoints=kp, valid_mask=np.ones(o.num_detections, bool))
    return out


def _fake_apriltag(aabb_crops, obb, model, cfg):
    # Sparse: a tag on every even chunk-local row, tag_id = centroid x.
    rows = list(range(0, obb.num_detections, 2))
    return AprilTagResult(
        tag_ids=[int(obb.centroids[r, 0]) for r in rows],
        det_indices=rows,
        centers=obb.centroids[rows].astype(np.float32),
        corners=obb.corners[rows].astype(np.float32),
    )


def _run_window(cfg, n_dets, models, caches, small=None, **extra_patches):
    from hydra_suite.core.inference.pipeline import BatchWindow
    from hydra_suite.core.inference.runner import InferenceRunner

    downstream = []
    with (
        patch("hydra_suite.core.inference.runner._load_all_models") as ml,
        patch(
            "hydra_suite.core.inference.pipeline.run_obb",
            side_effect=lambda frames, *a, **k: [_obb(0, n_dets, small)],
        ),
        patch(
            "hydra_suite.core.inference.pipeline.run_headtail_batch",
            side_effect=extra_patches.get("ht", _fake_ht),
        ),
        patch(
            "hydra_suite.core.inference.pipeline.run_cnn_batch",
            side_effect=extra_patches.get("cnn", _fake_cnn),
        ),
        patch(
            "hydra_suite.core.inference.pipeline.run_pose_batch",
            side_effect=_fake_pose,
        ),
        patch(
            "hydra_suite.core.inference.pipeline.run_apriltag",
            side_effect=_fake_apriltag,
        ),
    ):
        ml.return_value = models
        runner = InferenceRunner(cfg, cache_dir=None)
        pipeline = runner._build_pipeline(caches)
        real_write = pipeline.cache_writer.write_downstream

        def _spy(frame_idx, **kw):
            downstream.append(kw)
            return real_write(frame_idx, **kw)

        pipeline.cache_writer.write_downstream = _spy
        results = pipeline._process_window(
            BatchWindow(frames=[np.zeros((64, 600, 3), np.uint8)], frame_indices=[0])
        )
        pipeline.cache_writer.flush()
    return results, downstream


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


def test_batch_writes_superset_and_returns_final():
    from hydra_suite.core.inference.runner import _CacheSet

    cfg = _cfg(2, headtail=HeadTailConfig(model_path="/ht.pt"))
    caches = _CacheSet(detection=MagicMock(), headtail=MagicMock())
    writer_calls = []
    caches.headtail.write_frame.side_effect = lambda fi, **kw: writer_calls.append(kw)
    models = MagicMock(
        obb=MagicMock(), headtail=MagicMock(), cnn=[], pose=None, apriltag=None
    )
    # raw row 1 is dropped by the size filter -> superset raw [0, 2, 3, 4, 5]
    results, _ = _run_window(cfg, 6, models, caches, small=1)

    # cache holds the N-free superset, keyed by RAW index, payloads aligned
    assert writer_calls[0]["det_indices"].tolist() == [0, 2, 3, 4, 5]
    assert writer_calls[0]["heading_hints"].tolist() == [0, 100, 150, 200, 250]
    # the returned in-memory result is the final N=2 set: raw [0, 2]
    (fr,) = [r for r in results if r.frame_idx == 0]
    assert fr.filtered_indices == [0, 2]
    assert fr.obb.num_detections == 2
    assert fr.obb.centroids[:, 0].tolist() == [0.0, 100.0]
    assert fr.headtail.heading_hints.tolist() == [0.0, 100.0]


def test_superset_is_chunked_and_cnn_is_raw_in_cache_positional_in_memory():
    from hydra_suite.core.inference.config import AprilTagConfig, PoseConfig
    from hydra_suite.core.inference.runner import _CacheSet

    cfg = _cfg(
        3,
        headtail=HeadTailConfig(model_path="/ht.pt"),
        cnn_phases=[CNNConfig(label="id", model_path="/c.pt")],
        pose=PoseConfig(suppress_foreign_regions=False),
        apriltag=AprilTagConfig(enabled=True),
    )
    caches = _CacheSet(detection=MagicMock(), headtail=MagicMock(), cnn=[MagicMock()])
    ht_calls, cnn_calls, chunk_sizes = [], [], []
    caches.headtail.write_frame.side_effect = lambda fi, **kw: ht_calls.append(kw)
    caches.cnn[0].write_frame.side_effect = lambda fi, **kw: cnn_calls.append(kw)

    def _ht_record(frames, obbs, *a, **k):
        chunk_sizes.extend(o.num_detections for o in obbs)
        return _fake_ht(frames, obbs, *a, **k)

    models = MagicMock(
        obb=MagicMock(),
        headtail=MagicMock(),
        cnn=[MagicMock()],
        pose=MagicMock(),
        apriltag=MagicMock(),
    )
    # raw row 1 dropped -> superset raw [0, 2, 3, ..., 9] (9 rows); chunks of 4
    # start at raw 0, 5, 9 (non-identity positions).
    with patch("hydra_suite.core.inference.limits.DOWNSTREAM_CHUNK_SIZE", 4):
        results, downstream = _run_window(
            cfg, 10, models, caches, small=1, ht=_ht_record
        )
    sup = [0, 2, 3, 4, 5, 6, 7, 8, 9]
    xs = [50.0 * r for r in sup]

    # crops are materialised a bounded chunk at a time
    assert chunk_sizes == [4, 4, 1]
    # cache: whole superset, in order, keyed by RAW index, payloads aligned
    assert ht_calls[0]["det_indices"].tolist() == sup
    assert ht_calls[0]["heading_hints"].tolist() == xs
    preds = cnn_calls[0]["predictions"]
    assert [p.det_index for p in preds] == sup
    assert [p.factors[0] for p in preds] == xs
    (dw,) = downstream
    assert dw["det_indices"].tolist() == sup
    assert dw["pose"].keypoints[:, 0, 0].tolist() == xs
    # AprilTag: sparse, det_indices RAW. Tags sit on chunk-local rows 0, 2 of
    # each chunk -> superset positions 0, 2, 4, 6, 8 -> raw 0, 3, 5, 7, 9.
    at = dw["apriltag"]
    assert list(at.det_indices) == [0, 3, 5, 7, 9]
    assert list(at.tag_ids) == [0, 150, 250, 350, 450]

    # memory: final N=3 = raw [0, 2, 3]; everything positional 0..K-1, aligned
    (fr,) = results
    assert fr.filtered_indices == [0, 2, 3]
    assert fr.obb.centroids[:, 0].tolist() == [0.0, 100.0, 150.0]
    assert [p.det_index for p in fr.cnn[0].predictions] == [0, 1, 2]
    assert [p.factors[0] for p in fr.cnn[0].predictions] == [0.0, 100.0, 150.0]
    assert fr.headtail.heading_hints.tolist() == [0.0, 100.0, 150.0]
    assert fr.pose.keypoints[:, 0, 0].tolist() == [0.0, 100.0, 150.0]
    # tags on raw 0 and 3 -> final positions 0 and 2 (raw 2 has no tag)
    assert list(fr.apriltag.det_indices) == [0, 2]
    assert list(fr.apriltag.tag_ids) == [0, 150]


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


def _pose_crops(chunk_size, with_headtail):
    from hydra_suite.core.inference.config import PoseConfig
    from hydra_suite.core.inference.pipeline import BatchWindow
    from hydra_suite.core.inference.result import PoseResult
    from hydra_suite.core.inference.runner import InferenceRunner, _CacheSet

    cfg = _cfg(
        3,
        headtail=HeadTailConfig(model_path="/ht.pt") if with_headtail else None,
        pose=PoseConfig(suppress_foreign_regions=True),
    )
    seen = []

    def _fake_pose(crop_batch, model, pcfg, runtime, geometry):
        seen.append(crop_batch.crops.detach().cpu().clone())
        n = crop_batch.crops.shape[0]
        fi = int(crop_batch.frame_index[0])
        return {
            fi: PoseResult(
                keypoints=np.zeros((n, 1, 3), np.float32),
                valid_mask=np.ones(n, bool),
            )
        }

    rng = np.random.default_rng(7)
    frame = rng.integers(1, 255, (64, 160, 3), dtype=np.uint8)
    models = MagicMock(
        obb=MagicMock(),
        headtail=MagicMock() if with_headtail else None,
        cnn=[],
        pose=MagicMock(),
        apriltag=None,
    )
    with (
        patch("hydra_suite.core.inference.runner._load_all_models") as ml,
        patch(
            "hydra_suite.core.inference.pipeline.run_obb",
            side_effect=lambda frames, *a, **k: [_crowded_obb(0, 7)],
        ),
        patch(
            "hydra_suite.core.inference.pipeline.run_headtail_batch",
            side_effect=_fake_ht,
        ),
        patch("hydra_suite.core.inference.pipeline.run_pose_batch", _fake_pose),
        patch("hydra_suite.core.inference.limits.DOWNSTREAM_CHUNK_SIZE", chunk_size),
    ):
        ml.return_value = models
        runner = InferenceRunner(cfg, cache_dir=None)
        caches = _CacheSet(detection=MagicMock(), pose=MagicMock())
        pipeline = runner._build_pipeline(caches)
        pipeline._process_window(BatchWindow(frames=[frame], frame_indices=[0]))
    return seen


def test_pose_foreign_mask_is_independent_of_chunking():
    import torch

    for with_headtail in (False, True):
        whole = _pose_crops(256, with_headtail)
        chunked = _pose_crops(3, with_headtail)
        assert [t.shape[0] for t in whole] == [7]
        assert [t.shape[0] for t in chunked] == [3, 3, 1]
        # Every detection's masked pose crop is identical whether or not its
        # neighbours landed in the same chunk (foreign set = full superset).
        assert torch.equal(torch.cat(chunked), whole[0]), with_headtail


def test_batch_cnn_phase_without_result_stays_phase_aligned():
    """A CNN phase with no result must not shift later phases onto its cache:
    phase "a" gets explicit empty coverage, phase "b" its own predictions."""
    from hydra_suite.core.inference.runner import _CacheSet

    cfg = _cfg(
        2,
        cnn_phases=[
            CNNConfig(label="a", model_path="/a.pt"),
            CNNConfig(label="b", model_path="/b.pt"),
        ],
    )
    caches = _CacheSet(detection=MagicMock(), cnn=[MagicMock(), MagicMock()])
    caches.cnn[0].label = "a"
    caches.cnn[1].label = "b"
    models = MagicMock(
        obb=MagicMock(),
        headtail=None,
        cnn=[MagicMock(), MagicMock()],
        pose=None,
        apriltag=None,
    )

    def _cnn_a_empty(frames, obbs, model, cfg, *a, **k):
        if cfg.label == "a":
            return {}
        return _fake_cnn(frames, obbs, model, cfg, *a, **k)

    results, downstream = _run_window(cfg, 4, models, caches, small=1, cnn=_cnn_a_empty)

    (dw,) = downstream
    assert dw["cnn_results"][0] is None
    assert dw["cnn_results"][1].label == "b"
    assert [p.det_index for p in dw["cnn_results"][1].predictions] == [0, 2, 3]
    (fr,) = results
    assert [r.label for r in fr.cnn] == ["b"]
    assert [p.det_index for p in fr.cnn[0].predictions] == [0, 1]


def _run_multi_window(cfg, n_by_frame, models, caches):
    """One window over several frames; ``n_by_frame[i]`` detections in frame i."""
    from hydra_suite.core.inference.pipeline import BatchWindow
    from hydra_suite.core.inference.runner import InferenceRunner

    with (
        patch("hydra_suite.core.inference.runner._load_all_models") as ml,
        patch(
            "hydra_suite.core.inference.pipeline.run_obb",
            side_effect=lambda frames, *a, **k: [
                _obb(i, n) for i, n in enumerate(n_by_frame)
            ],
        ),
        patch(
            "hydra_suite.core.inference.pipeline.run_headtail_batch",
            side_effect=_fake_ht,
        ),
        patch(
            "hydra_suite.core.inference.pipeline.run_cnn_batch",
            side_effect=_fake_cnn,
        ),
    ):
        ml.return_value = models
        pipeline = InferenceRunner(cfg, cache_dir=None)._build_pipeline(caches)
        pipeline._process_window(
            BatchWindow(
                frames=[np.zeros((64, 600, 3), np.uint8) for _ in n_by_frame],
                frame_indices=list(range(len(n_by_frame))),
            )
        )
        pipeline.cache_writer.flush()


@__import__("pytest").mark.parametrize(
    "n_by_frame",
    [(3, 0, 3), (3, 3, 0), (0, 3, 0, 3, 0), (0, 0, 0)],
)
def test_downstream_rows_written_in_frame_order_with_empty_frames(n_by_frame):
    from hydra_suite.core.inference.runner import _CacheSet

    cfg = _cfg(
        4,
        headtail=HeadTailConfig(model_path="/ht.pt"),
        cnn_phases=[CNNConfig(label="a", model_path="/c.pt")],
    )
    caches = _CacheSet(detection=MagicMock(), headtail=MagicMock(), cnn=[MagicMock()])
    ht_frames, cnn_frames = [], []
    caches.headtail.write_frame.side_effect = lambda fi, **kw: ht_frames.append(
        (fi, len(kw["det_indices"]))
    )
    caches.cnn[0].write_frame.side_effect = lambda fi, **kw: cnn_frames.append(
        (fi, len(kw["predictions"]))
    )
    models = MagicMock(
        obb=MagicMock(),
        headtail=MagicMock(),
        cnn=[MagicMock()],
        pose=None,
        apriltag=None,
    )
    _run_multi_window(cfg, n_by_frame, models, caches)

    expected = [(i, n) for i, n in enumerate(n_by_frame)]
    # every frame present (empty ones as explicit empty rows), strictly increasing
    assert ht_frames == expected
    assert [f for f, _ in cnn_frames] == list(range(len(n_by_frame)))
    # empty frames carry an explicit empty CNN phase (predictions=[])
    assert all(n == 0 for (f, n) in cnn_frames if n_by_frame[f] == 0)
