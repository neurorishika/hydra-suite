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
    CNNDetectionPrediction,
    CNNResult,
    HeadTailResult,
    OBBResult,
)


def _obb(frame_idx, n):
    xs = np.arange(n, dtype=np.float32) * 50.0
    c = np.stack([xs, np.zeros(n, np.float32)], 1)
    corners = np.stack([c + d for d in ([-4, -4], [4, -4], [4, 4], [-4, 4])], 1)
    return OBBResult(
        frame_idx,
        c,
        np.zeros(n, np.float32),
        np.full(n, 64.0, np.float32),
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


def _run_window(cfg, n_dets, models, caches, **extra_patches):
    from hydra_suite.core.inference.pipeline import BatchWindow
    from hydra_suite.core.inference.runner import InferenceRunner

    with (
        patch("hydra_suite.core.inference.runner._load_all_models") as ml,
        patch(
            "hydra_suite.core.inference.pipeline.run_obb",
            side_effect=lambda frames, *a, **k: [_obb(0, n_dets)],
        ),
        patch(
            "hydra_suite.core.inference.pipeline.run_headtail_batch",
            side_effect=extra_patches.get("ht", _fake_ht),
        ),
        patch(
            "hydra_suite.core.inference.pipeline.run_cnn_batch",
            side_effect=_fake_cnn,
        ),
    ):
        ml.return_value = models
        runner = InferenceRunner(cfg, cache_dir=None)
        pipeline = runner._build_pipeline(caches)
        results = pipeline._process_window(
            BatchWindow(frames=[np.zeros((64, 400, 3), np.uint8)], frame_indices=[0])
        )
        pipeline.cache_writer.flush()
    return results


def _cfg(max_detections, **kw):
    return InferenceConfig(
        obb=OBBConfig(
            mode="direct",
            direct=OBBDirectConfig(model_path="/m.pt"),
            max_detections=max_detections,
            confidence_threshold=0.0,
            iou_threshold=1.0,
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
    results = _run_window(cfg, 6, models, caches)

    # cache holds the N-free superset (all 6), keyed by raw index
    assert writer_calls[0]["det_indices"].tolist() == [0, 1, 2, 3, 4, 5]
    assert writer_calls[0]["heading_hints"].tolist() == [0, 50, 100, 150, 200, 250]
    # the returned in-memory result is the final N=2 set, aligned
    (fr,) = [r for r in results if r.frame_idx == 0]
    assert fr.filtered_indices == [0, 1]
    assert fr.obb.num_detections == 2
    assert fr.headtail.heading_hints.tolist() == [0.0, 50.0]


def test_superset_is_chunked_and_cnn_is_raw_in_cache_positional_in_memory():
    from hydra_suite.core.inference.runner import _CacheSet

    cfg = _cfg(
        3,
        headtail=HeadTailConfig(model_path="/ht.pt"),
        cnn_phases=[CNNConfig(label="id", model_path="/c.pt")],
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
        pose=None,
        apriltag=None,
    )
    with patch("hydra_suite.core.inference.limits.DOWNSTREAM_CHUNK_SIZE", 4):
        results = _run_window(cfg, 10, models, caches, ht=_ht_record)

    # crops are materialised a bounded chunk at a time
    assert chunk_sizes == [4, 4, 2]
    # cache: whole superset, in order, CNN keyed by RAW index
    assert ht_calls[0]["det_indices"].tolist() == list(range(10))
    assert ht_calls[0]["heading_hints"].tolist() == [50.0 * i for i in range(10)]
    preds = cnn_calls[0]["predictions"]
    assert [p.det_index for p in preds] == list(range(10))
    assert [p.factors[0] for p in preds] == [50.0 * i for i in range(10)]
    # memory: final N=3, CNN positional 0..K-1 and aligned with the OBB rows
    (fr,) = results
    assert fr.obb.num_detections == 3
    assert [p.det_index for p in fr.cnn[0].predictions] == [0, 1, 2]
    assert [p.factors[0] for p in fr.cnn[0].predictions] == [0.0, 50.0, 100.0]
    assert fr.headtail.heading_hints.tolist() == [0.0, 50.0, 100.0]


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
