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


def test_window_inside_the_cached_range_is_partially_reused(tmp_path, video):
    """Partial reuse needs detection to COVER the requested window, not equal
    it: a window inside the cached range reads detections back."""
    built = tmp_path / "built"
    _build(built, 10, video)

    calls = _window_pass(built, _stricter_size(), video, start=0, end=2, reuse=True)

    assert calls["run_obb"] == 0


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


def test_corrupt_key_valid_member_falls_back_to_a_full_fresh_run(tmp_path, video):
    """A per-animal member whose key is valid but whose chunk fails its
    checksum cannot be repaired in resume mode (resume skips every manifest
    frame). The reuse pass must fall back to the full fresh run, which
    rebuilds and heals the cache."""
    from hydra_suite.core.inference.runner import InferenceRunner

    built = tmp_path / "built"
    _build(built, 10, video)
    chunk = sorted(
        p for p in built.rglob("chunk-*.npz") if "apriltag.npz.chunks" in str(p)
    )[0]
    data = bytearray(chunk.read_bytes())
    data[len(data) // 2] ^= 0xFF
    chunk.unlink()
    chunk.write_bytes(bytes(data))

    calls = _reuse_pass(built, _cfg(10), video)
    assert calls["run_obb"] > 0, "corrupt member: fall back to the fresh run"
    assert "apriltag" in calls["written_kinds"]

    with patch("hydra_suite.core.inference.runner._load_all_models") as ml:
        ml.return_value = _models()
        runner = InferenceRunner(
            _cfg(10), cache_dir=built, video_path=video, cache_only=True
        )
    assert runner.caches_all_valid()
    for i in range(_NUM_FRAMES):
        assert runner.load_frame(i).apriltag is not None, f"frame {i}"


def test_reuse_run_says_limit_hits_were_reported_at_build_time(tmp_path, video, caplog):
    import logging

    built = tmp_path / "built"
    _build(built, 10, video)
    with caplog.at_level(logging.INFO, logger="hydra_suite.core.inference.runner"):
        _reuse_pass(built, _stricter_size(), video)
    msgs = [r.getMessage() for r in caplog.records]
    hits = [m for m in msgs if "reused from the cache" in m]
    assert len(hits) == 1
    assert "detection-limit hits were reported when the cache was built" in hits[0]


def _window_pass(
    cache_dir, cfg, video, *, start, end, reuse, should_stop=None, obb=_fake_run_obb
):
    """Batch pass over ``[start, end]``; returns call counts and written kinds."""
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
        patch(_P + "run_obb", side_effect=obb) as obb_mock,
        patch(_P + "run_headtail_batch", side_effect=_fake_ht),
        patch(_P + "run_cnn_batch", side_effect=_fake_cnn),
        patch(_P + "run_pose_batch", side_effect=_fake_pose),
        patch(_P + "run_apriltag", side_effect=_fake_apriltag),
    ):
        ml.return_value = _models()
        runner = InferenceRunner(cfg, cache_dir=cache_dir, video_path=video)
        try:
            runner.run_batch_pass(
                video,
                start_frame=start,
                end_frame=end,
                reuse_detection_cache=reuse,
                should_stop=should_stop,
            )
        finally:
            runner.close()
    return {"run_obb": obb_mock.call_count, "written_kinds": written_kinds}


def _replay_runner(cache_dir, cfg, video):
    from hydra_suite.core.inference.runner import InferenceRunner

    with patch("hydra_suite.core.inference.runner._load_all_models") as ml:
        ml.return_value = _models()
        return InferenceRunner(
            cfg, cache_dir=cache_dir, video_path=video, cache_only=True
        )


def _cache_listing(cache_dir):
    generations = cache_dir / ".cache-generations"
    return (
        sorted(str(p.relative_to(cache_dir)) for p in generations.glob("*/*")),
        (cache_dir / "cache_set.json").read_bytes(),
    )


def test_window_after_filter_change_recomputes_only_the_window(tmp_path, video):
    """The reported failure: a window inside the cached range after a
    replay-filter change. Detections are reused, the stale per-animal stages
    run over the window only, and the result is promoted with per-animal
    coverage a subset of detection's."""
    built = tmp_path / "built"
    _build(built, 10, video)
    new_cfg = _stricter_size()

    calls = _window_pass(built, new_cfg, video, start=2, end=4, reuse=True)

    assert calls["run_obb"] == 0
    assert "detection" not in calls["written_kinds"]
    runner = _replay_runner(built, new_cfg, video)
    assert runner.caches_all_valid()
    assert runner.detection_cache_covers_range(2, 4)
    assert not runner.detection_cache_covers_range(0, _NUM_FRAMES - 1)
    missing = runner.detection_cache_missing_frames(0, _NUM_FRAMES - 1)
    assert missing and set(missing) == {0, 1, 5}

    fresh = tmp_path / "fresh"
    _fresh_build(fresh, new_cfg, video)
    want = _replay_cfg(fresh, new_cfg, video)
    for i in range(2, 5):
        assert _view(runner.load_frame(i)) == _view(want[i]), f"frame {i}"


def test_full_run_after_a_window_fills_the_rest_without_the_detector(tmp_path, video):
    built = tmp_path / "built"
    _build(built, 10, video)
    new_cfg = _stricter_size()
    _window_pass(built, new_cfg, video, start=2, end=4, reuse=True)

    calls = _window_pass(built, new_cfg, video, start=0, end=None, reuse=True)

    assert calls["run_obb"] == 0
    assert "detection" not in calls["written_kinds"]
    fresh = tmp_path / "fresh"
    _fresh_build(fresh, new_cfg, video)
    got = _replay_cfg(built, new_cfg, video)
    want = _replay_cfg(fresh, new_cfg, video)
    for i in range(_NUM_FRAMES):
        assert _view(got[i]) == _view(want[i]), f"frame {i}"


@pytest.mark.parametrize("reuse", [True, False])
def test_window_starting_at_zero_keeps_the_full_cache(tmp_path, video, reuse):
    """A window from frame 0 that ends early must not replace a valid
    full-range cache with a window-only generation."""
    built = tmp_path / "built"
    _build(built, 10, video)

    _window_pass(built, _cfg(10), video, start=0, end=2, reuse=reuse)

    runner = _replay_runner(built, _cfg(10), video)
    assert runner.caches_all_valid()
    assert runner.detection_cache_covers_range(0, _NUM_FRAMES - 1)


def test_stale_detection_never_resumes_over_valid_per_animal_members(tmp_path, video):
    """Per-animal keys do not fold in the detector's identity: a detection
    re-key with otherwise-valid per-animal members must start a new
    generation, so no per-animal row can point at superseded detections."""
    built = tmp_path / "built"
    _build(built, 10, video)
    before = _cache_listing(built)[1]
    cfg = _cfg(10)
    cfg.detection_batch_size = 2  # detection cache key only

    calls = _window_pass(built, cfg, video, start=2, end=4, reuse=True)

    assert calls["run_obb"] > 0
    assert {"headtail", "cnn", "pose", "apriltag"} <= set(calls["written_kinds"])
    assert _cache_listing(built)[1] != before
    runner = _replay_runner(built, cfg, video)
    assert runner.caches_all_valid()
    assert runner.detection_cache_covers_range(2, 4)
    assert not runner.detection_cache_covers_range(0, 1)


@pytest.mark.parametrize("start", [0, 2])  # fresh and resume staging
def test_cancelled_pass_leaves_no_staging_behind(tmp_path, video, start):
    from hydra_suite.core.inference.cancellation import InferenceCancelled

    built = tmp_path / "built"
    _build(built, 10, video)
    before = _cache_listing(built)

    with pytest.raises(InferenceCancelled):
        _window_pass(
            built,
            _stricter_size(),
            video,
            start=start,
            end=None,
            reuse=False,
            should_stop=lambda: True,
        )

    assert _cache_listing(built) == before


def _shifted_run_obb(frames, *a, **k):
    """A detector whose rows differ from the cached ones (x + 1)."""
    import dataclasses

    out = []
    for obb in _fake_run_obb(frames, *a, **k):
        centroids = obb.centroids.copy()
        centroids[:, 0] += 1.0
        corners = obb.corners.copy()
        corners[:, :, 0] += 1.0
        out.append(dataclasses.replace(obb, centroids=centroids, corners=corners))
    return out


def _assert_per_animal_rows_follow_detections(cache_dir, cfg, video, frames):
    runner = _replay_runner(cache_dir, cfg, video)
    for i in frames:
        if i in _EMPTY_FRAMES:
            continue
        frame = runner.load_frame(i)
        assert frame.headtail.heading_hints.tolist() == (
            frame.obb.centroids[:, 0].tolist()
        ), f"frame {i}"


@pytest.mark.parametrize(
    "built_end, start, end, reuse", [(None, 2, 4, False), (3, 2, 5, True)]
)
def test_resumed_window_keys_per_animal_rows_to_the_kept_detections(
    tmp_path, video, built_end, start, end, reuse
):
    """Resume keeps the cached detection rows; per-animal stages must be
    computed from them, not from a live detector output that differs."""
    built = tmp_path / "built"
    _window_pass(built, _cfg(10), video, start=0, end=built_end, reuse=False)

    _window_pass(
        built,
        _stricter_size(),
        video,
        start=start,
        end=end,
        reuse=reuse,
        obb=_shifted_run_obb,
    )

    _assert_per_animal_rows_follow_detections(
        built, _stricter_size(), video, range(start, end + 1)
    )


def test_resumed_realtime_keys_per_animal_rows_to_the_kept_detections(tmp_path, video):
    import cv2

    from hydra_suite.core.inference.runner import InferenceRunner
    from tests.test_n_independent_replay import _rt_fakes

    built = tmp_path / "built"
    _build(built, 10, video)
    cfg = _stricter_size()
    ht, cnn, pose, tag = _rt_fakes()
    cap = cv2.VideoCapture(str(video))
    frames = [cap.read()[1] for _ in range(_NUM_FRAMES)]
    cap.release()
    r = "hydra_suite.core.inference.runner."
    with (
        patch(r + "_load_all_models") as ml,
        patch(r + "run_obb", side_effect=_shifted_run_obb),
        patch(r + "run_headtail", side_effect=ht),
        patch(r + "run_cnn", side_effect=cnn),
        patch(r + "run_pose", side_effect=pose),
        patch(r + "run_apriltag", side_effect=tag),
    ):
        ml.return_value = _models()
        runner = InferenceRunner(cfg, cache_dir=built, video_path=video)
        for i, frame in enumerate(frames):
            runner.run_realtime(frame, i)
        runner.close()

    _assert_per_animal_rows_follow_detections(built, cfg, video, range(_NUM_FRAMES))
