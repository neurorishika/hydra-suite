"""I1: a filter change reuses detections and recomputes only per-animal stages.

Caches are built once through the real batch pass (fixture of
``tests/test_n_independent_replay.py``). A second batch pass at a changed
replay filter, with detection-cache reuse allowed, must:

* never run the detector (``run_obb`` 0 calls) nor rewrite the detection cache,
* rerun the per-animal stages whose cache keys the filter change invalidated,
* produce caches whose replay equals a fresh full run at the new filter.

Changing only N must rerun nothing at all.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from tests.test_n_independent_replay import (
    _EMPTY_FRAMES,
    _NUM_FRAMES,
    _build,
    _cfg,
    _fake_apriltag,
    _fake_cnn,
    _fake_ht,
    _fake_pose,
    _fake_run_obb,
    _models,
    _view,
    _write_video,
)

_P = "hydra_suite.core.inference.pipeline."


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "clip.mp4"
    _write_video(path)
    return path


def _replay_cfg(cache_dir, cfg, video):
    from hydra_suite.core.inference.runner import InferenceRunner

    with patch("hydra_suite.core.inference.runner._load_all_models") as ml:
        ml.return_value = _models()
        runner = InferenceRunner(
            cfg, cache_dir=cache_dir, video_path=video, cache_only=True
        )
    assert runner.caches_all_valid()
    assert runner.detection_cache_covers_range(0, _NUM_FRAMES - 1)
    return {i: runner.load_frame(i) for i in range(_NUM_FRAMES)}


def _fresh_build(cache_dir, cfg, video):
    from hydra_suite.core.inference.runner import InferenceRunner

    with (
        patch("hydra_suite.core.inference.runner._load_all_models") as ml,
        patch(_P + "run_obb", side_effect=_fake_run_obb),
        patch(_P + "run_headtail_batch", side_effect=_fake_ht),
        patch(_P + "run_cnn_batch", side_effect=_fake_cnn),
        patch(_P + "run_pose_batch", side_effect=_fake_pose),
        patch(_P + "run_apriltag", side_effect=_fake_apriltag),
    ):
        ml.return_value = _models()
        runner = InferenceRunner(cfg, cache_dir=cache_dir, video_path=video)
        runner.run_batch_pass(video)
        runner.close()


def _reuse_pass(cache_dir, cfg, video):
    """Batch pass with detection reuse allowed; returns call counts."""
    from hydra_suite.core.inference.cache.chunked import ChunkedArrayStore
    from hydra_suite.core.inference.runner import InferenceRunner

    written_kinds: list[str] = []
    real_append = ChunkedArrayStore.append_chunk

    def _append(self, frames, arrays):
        written_kinds.append(self.kind)
        return real_append(self, frames, arrays)

    with (
        patch("hydra_suite.core.inference.runner._load_all_models") as ml,
        patch.object(ChunkedArrayStore, "append_chunk", _append),
        patch(_P + "run_obb", side_effect=_fake_run_obb) as obb,
        patch(_P + "run_headtail_batch", side_effect=_fake_ht) as ht,
        patch(_P + "run_cnn_batch", side_effect=_fake_cnn) as cnn,
        patch(_P + "run_pose_batch", side_effect=_fake_pose) as pose,
        patch(_P + "run_apriltag", side_effect=_fake_apriltag) as tag,
    ):
        ml.return_value = _models()
        runner = InferenceRunner(cfg, cache_dir=cache_dir, video_path=video)
        runner.run_batch_pass(video, reuse_detection_cache=True)
        runner.close()
    return {
        "run_obb": obb.call_count,
        "headtail": ht.call_count,
        "cnn": cnn.call_count,
        "pose": pose.call_count,
        "apriltag": tag.call_count,
        "written_kinds": written_kinds,
    }


def _stricter_confidence():
    cfg = _cfg(10)
    cfg.obb.confidence_threshold = 0.7  # linspace(0.9, 0.5, 30): drops the tail
    return cfg


def _stricter_size():
    cfg = _cfg(10)
    cfg.obb.min_object_size = 100.0  # also drops raw _MEDIUM (size 60)
    return cfg


@pytest.mark.parametrize("depth", [1, 2])  # sync and double-buffered writer
@pytest.mark.parametrize("changed", [_stricter_confidence, _stricter_size])
def test_filter_change_reuses_detections_and_recomputes_per_animal(
    tmp_path, video, changed, depth
):
    built = tmp_path / "built"
    _build(built, 10, video)
    new_cfg = changed()
    new_cfg.pipeline_depth = depth

    calls = _reuse_pass(built, new_cfg, video)

    assert calls["run_obb"] == 0, "a filter change must not rerun the detector"
    assert "detection" not in calls["written_kinds"], "detections not rewritten"
    for stage in ("headtail", "cnn", "pose", "apriltag"):
        assert calls[stage] > 0, f"{stage} keyed by the filter must recompute"
    assert {"headtail", "cnn", "pose", "apriltag"} <= set(calls["written_kinds"])

    fresh = tmp_path / "fresh"
    _fresh_build(fresh, new_cfg, video)
    got = _replay_cfg(built, new_cfg, video)
    want = _replay_cfg(fresh, new_cfg, video)
    for i in range(_NUM_FRAMES):
        assert _view(got[i]) == _view(want[i]), f"frame {i}"
        if i not in _EMPTY_FRAMES:
            assert got[i].filtered_indices  # non-trivial comparison


def test_n_change_only_reruns_nothing(tmp_path, video):
    built = tmp_path / "built"
    _build(built, 10, video)

    calls = _reuse_pass(built, _cfg(20), video)

    assert calls["run_obb"] == 0
    for stage in ("headtail", "cnn", "pose", "apriltag"):
        assert calls[stage] == 0, f"{stage} must be reused at a new N"
    assert calls["written_kinds"] == []


def test_reuse_not_requested_still_reruns_the_detector(tmp_path, video):
    """Default (user disabled reuse): a filter change is a full fresh run."""
    from hydra_suite.core.inference.runner import InferenceRunner

    built = tmp_path / "built"
    _build(built, 10, video)
    with (
        patch("hydra_suite.core.inference.runner._load_all_models") as ml,
        patch(_P + "run_obb", side_effect=_fake_run_obb) as obb,
        patch(_P + "run_headtail_batch", side_effect=_fake_ht),
        patch(_P + "run_cnn_batch", side_effect=_fake_cnn),
        patch(_P + "run_pose_batch", side_effect=_fake_pose),
        patch(_P + "run_apriltag", side_effect=_fake_apriltag),
    ):
        ml.return_value = _models()
        runner = InferenceRunner(_stricter_size(), cache_dir=built, video_path=video)
        runner.run_batch_pass(video)
        runner.close()
    assert obb.call_count > 0


def test_detection_cache_covering_a_different_range_is_not_partially_reused(
    tmp_path, video
):
    """Partial reuse needs detection coverage == the requested range; a
    shorter request falls back to running the detector (promotion would
    otherwise mix coverages)."""
    from hydra_suite.core.inference.runner import InferenceRunner

    built = tmp_path / "built"
    _build(built, 10, video)
    with (
        patch("hydra_suite.core.inference.runner._load_all_models") as ml,
        patch(_P + "run_obb", side_effect=_fake_run_obb) as obb,
        patch(_P + "run_headtail_batch", side_effect=_fake_ht),
        patch(_P + "run_cnn_batch", side_effect=_fake_cnn),
        patch(_P + "run_pose_batch", side_effect=_fake_pose),
        patch(_P + "run_apriltag", side_effect=_fake_apriltag),
    ):
        ml.return_value = _models()
        runner = InferenceRunner(_stricter_size(), cache_dir=built, video_path=video)
        runner.run_batch_pass(video, end_frame=2, reuse_detection_cache=True)
        runner.close()
    assert obb.call_count > 0


def test_only_the_stale_stage_is_recomputed_and_written(tmp_path, video):
    """A pose-only key change reruns pose (plus the head-tail it shares crops
    with, unwritten) and reuses CNN / AprilTag / head-tail caches untouched."""
    built = tmp_path / "built"
    _build(built, 10, video)
    cfg = _cfg(10)
    cfg.pose.suppress_foreign_regions = True  # pose cache key only

    calls = _reuse_pass(built, cfg, video)

    assert calls["run_obb"] == 0
    assert calls["pose"] > 0
    assert calls["cnn"] == 0 and calls["apriltag"] == 0
    assert set(calls["written_kinds"]) == {"pose"}
    got = _replay_cfg(built, cfg, video)
    assert got[0].pose is not None and got[0].filtered_indices


def test_worker_filter_change_with_reuse_skips_the_detector(
    monkeypatch, tmp_path, video
):
    """End to end through TrackingEngineCore: the rerun at a stricter size
    filter runs the batch pass (per-animal caches re-keyed) but never the
    detector, and recomputes the CNN."""
    import tests.test_n_independent_replay as rp

    shared = tmp_path / "cache"
    ok, fwd = rp._run_worker(monkeypatch, tmp_path, video, shared, 10, reuse=False)
    assert ok is True and fwd["batch_pass"] == 1

    real_cfg = rp._worker_cfg

    def _stricter(n):
        cfg = real_cfg(n)
        cfg.obb.min_object_size = 100.0
        return cfg

    obb_calls = []

    def _counting_obb(*a, **k):
        obb_calls.append(1)
        return _fake_run_obb(*a, **k)

    monkeypatch.setattr(rp, "_worker_cfg", _stricter)
    monkeypatch.setattr(rp, "_fake_run_obb", _counting_obb)
    ok, rerun = rp._run_worker(monkeypatch, tmp_path, video, shared, 10, reuse=True)
    assert ok is True
    assert rerun["batch_pass"] == 1, "re-keyed per-animal caches need a pass"
    assert obb_calls == [], "the detector must not run on a filter change"
    assert rerun["cnn_calls"] > 0


def test_tied_confidences_keep_raw_index_alignment(tmp_path, video, monkeypatch):
    """Cached frames are already ranked; re-ranking them would reverse tie
    groups (extraction breaks ties later-first) and re-key every per-animal
    row. Partial reuse must consume the stored order unchanged."""
    import numpy as np

    import tests.test_n_independent_replay as rp

    real_obb = rp._obb

    def _tied(frame_idx, n):
        r = real_obb(frame_idx, n)
        r.confidences[:] = np.where(np.arange(n) % 3 == 0, 0.9, 0.6).astype(np.float32)
        return r

    monkeypatch.setattr(rp, "_obb", _tied)
    built = tmp_path / "built"
    _build(built, 10, video)
    new_cfg = _stricter_size()

    calls = _reuse_pass(built, new_cfg, video)
    assert calls["run_obb"] == 0

    fresh = tmp_path / "fresh"
    _fresh_build(fresh, new_cfg, video)
    got = _replay_cfg(built, new_cfg, video)
    want = _replay_cfg(fresh, new_cfg, video)
    for i in range(_NUM_FRAMES):
        assert _view(got[i]) == _view(want[i]), f"frame {i}"
