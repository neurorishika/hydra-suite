"""Task 9: N (MAX_TARGETS) is a replay-time knob, end to end.

Caches are built ONCE at N=10 through the real batch pass (real video, real
Pipeline windowing, real on-disk cache store) and replayed at N=5, 10, 20.
Every replay must equal a fresh run at that N -- both the fresh run's
in-memory tracking output and a replay of the fresh run's own caches.
Raising N must rerun no per-animal model and yield no NaN.

Fixture properties that make the proof discriminating:

* NON-contiguous raw indices: raw row ``_SMALL`` is dropped by the size
  filter, so final sets look like ``[0, 1, 3, 4, ...]``; any raw-vs-positional
  index mixup in the per-animal caches shifts payloads and fails.
* ``detection_batch_size=3`` with an EMPTY frame in the middle of each window
  (frames 1 and 4 of 0..5): the per-animal rows of a window must reach the
  store in frame order, or its "unique and increasing" rule raises.
* every per-animal stage is on (head-tail, CNN, pose, AprilTag) and each
  payload is a function of the detection's centroid x, so a row read for the
  wrong detection is visible.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

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
from hydra_suite.core.inference.result import (
    AprilTagResult,
    CNNDetectionPrediction,
    CNNFactorPrediction,
    CNNResult,
    HeadTailResult,
    OBBResult,
    PoseResult,
)

_N_DETS = 30  # detections per non-empty frame (> 2N for N=10, < 2N for N=20)
_SMALL = 2  # raw row size-filtered out -> non-contiguous raw indices
_MEDIUM = 1  # raw row kept by the written filters, dropped by a stricter one
_MEDIUM_SIZE = 60.0
_EMPTY_FRAMES = (1, 4)  # middle frame of each 3-frame window
_NUM_FRAMES = 6
_BATCH = 3
_FRAME_W, _FRAME_H = 1600, 64
_CLASSES = ["a", "b", "c"]
_STAGES = (
    "run_obb",
    "run_headtail_batch",
    "run_cnn_batch",
    "run_pose_batch",
    "run_apriltag",
)


# --- fixture -----------------------------------------------------------------


def _write_video(path):
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (_FRAME_W, _FRAME_H)
    )
    for i in range(_NUM_FRAMES):
        writer.write(np.full((_FRAME_H, _FRAME_W, 3), i * 40, np.uint8))
    writer.release()


def _frame_index(frame):
    # The pipeline hands run_obb pixels, not indices; the clip encodes its
    # frame index as a flat grey level (i * 40), robust to mp4v rounding.
    return int(round(float(frame.mean()) / 40.0))


def _obb(frame_idx, n):
    # Confidence descends with raw index (ranked order == raw order); boxes
    # sit 50 px apart so they never overlap; row _SMALL is tiny.
    xs = 20.0 + np.arange(n, dtype=np.float32) * 50.0
    c = np.stack([xs, np.full(n, 32.0, np.float32)], 1)
    corners = np.stack([c + d for d in ([-8, -8], [8, -8], [8, 8], [-8, 8])], 1)
    sizes = np.full(n, 256.0, np.float32)
    if n > _SMALL:
        sizes[_SMALL] = 1.0
        sizes[_MEDIUM] = _MEDIUM_SIZE
    return OBBResult(
        frame_idx,
        c.astype(np.float32),
        np.zeros(n, np.float32),
        sizes,
        np.ones((n, 2), np.float32),
        np.linspace(0.9, 0.5, n).astype(np.float32),
        corners.astype(np.float32),
        OBBResult.make_detection_ids(frame_idx, n),
    )


def _fake_run_obb(frames, *a, **k):
    out = []
    for frame in frames:
        fi = _frame_index(frame)
        out.append(_obb(fi, 0 if fi in _EMPTY_FRAMES else _N_DETS))
    return out


def _probs(x):
    p = np.array([1.0 + (x % 7), 1.0 + (x % 5), 1.0 + (x % 3)], np.float32)
    return p / p.sum()


def _fake_ht(frames, obbs, model, cfg, runtime, geometry, canonical_batch=None):
    return {
        o.frame_idx: HeadTailResult(
            o.centroids[:, 0].copy(),
            np.full(o.num_detections, 0.75, np.float32),
            np.ones(o.num_detections, np.uint8),
            None,
        )
        for o in obbs
    }


def _fake_cnn(frames, obbs, model, cfg, runtime, geometry, headtail_by_frame=None):
    # det_index is chunk-LOCAL (the run_cnn_batch contract); the probabilities
    # identify the detection by its centroid x.
    return {
        o.frame_idx: CNNResult(
            label=cfg.label,
            predictions=[
                CNNDetectionPrediction(
                    det_index=i,
                    factors=[
                        CNNFactorPrediction(
                            "f0", list(_CLASSES), _probs(float(o.centroids[i, 0]))
                        )
                    ],
                )
                for i in range(o.num_detections)
            ],
        )
        for o in obbs
    }


def _fake_pose(crop_batch, model, pcfg, runtime, geometry=None, **kw):
    out = {}
    for fi, o in crop_batch.obb_by_frame.items():
        kp = np.zeros((o.num_detections, 2, 3), np.float32)
        kp[:, 0, 0] = o.centroids[:, 0]
        kp[:, 1, 1] = o.centroids[:, 0] + 1.0
        kp[:, :, 2] = 1.0
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


def _cfg(n):
    return InferenceConfig(
        obb=OBBConfig(
            mode="direct",
            direct=OBBDirectConfig(model_path="/m.pt"),
            max_detections=n,
            confidence_threshold=0.0,
            iou_threshold=1.0,
            min_object_size=10.0,
        ),
        headtail=HeadTailConfig(model_path="/ht.pt"),
        cnn_phases=[CNNConfig(label="id", model_path="/c.pt")],
        pose=PoseConfig(
            backend="yolo",
            yolo=PoseYOLOConfig(model_path="/pose.pt"),
            skeleton_file="",
            suppress_foreign_regions=False,
        ),
        apriltag=AprilTagConfig(enabled=True),
        detection_batch_size=_BATCH,
        runtime_tier="cpu",
    )


def _models():
    return MagicMock(
        obb=MagicMock(),
        bgsub=None,
        headtail=MagicMock(),
        cnn=[MagicMock()],
        pose=MagicMock(),
        apriltag=MagicMock(),
    )


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "clip.mp4"
    _write_video(path)
    return path


def _build(cache_dir, n, video):
    """Full batch pass at N; return the fresh run's in-memory FrameResults."""
    from hydra_suite.core.inference.pipeline import Pipeline
    from hydra_suite.core.inference.runner import InferenceRunner

    fresh = {}
    # Every depth (sync or double-buffered) funnels a window's results
    # through _process_obb_results.
    real_process = Pipeline._process_obb_results

    def _spy(self, window, *a, **k):
        results = real_process(self, window, *a, **k)
        fresh.update({r.frame_idx: r for r in results})
        return results

    p = "hydra_suite.core.inference.pipeline."
    with (
        patch("hydra_suite.core.inference.runner._load_all_models") as ml,
        patch.object(Pipeline, "_process_obb_results", _spy),
        patch(p + "run_obb", side_effect=_fake_run_obb),
        patch(p + "run_headtail_batch", side_effect=_fake_ht),
        patch(p + "run_cnn_batch", side_effect=_fake_cnn),
        patch(p + "run_pose_batch", side_effect=_fake_pose),
        patch(p + "run_apriltag", side_effect=_fake_apriltag),
    ):
        ml.return_value = _models()
        runner = InferenceRunner(_cfg(n), cache_dir=cache_dir, video_path=video)
        runner.run_batch_pass(video)
        runner.close()
    # The batch pass assembles in-memory results for non-empty frames only
    # (empty frames exist solely as explicit empty cache rows).
    assert sorted(fresh) == [i for i in range(_NUM_FRAMES) if i not in _EMPTY_FRAMES]
    return fresh


def _replay(cache_dir, n, video):
    from hydra_suite.core.inference.runner import InferenceRunner

    with patch("hydra_suite.core.inference.runner._load_all_models") as ml:
        ml.return_value = _models()
        runner = InferenceRunner(
            _cfg(n), cache_dir=cache_dir, video_path=video, cache_only=True
        )
    assert runner.caches_all_valid()
    assert runner.detection_cache_covers_range(0, _NUM_FRAMES - 1)
    return {i: runner.load_frame(i) for i in range(_NUM_FRAMES)}


def _view(fr):
    """Every replayed field of a FrameResult, as comparable plain values."""
    return {
        "filtered_indices": list(fr.filtered_indices),
        "centroids": fr.obb.centroids.tolist(),
        "confidences": fr.obb.confidences.tolist(),
        "detection_ids": np.asarray(fr.obb.detection_ids).tolist(),
        "ht": fr.headtail
        and (
            fr.headtail.heading_hints.tolist(),
            fr.headtail.heading_confidences.tolist(),
            fr.headtail.directed_mask.tolist(),
        ),
        "cnn": [
            (
                r.label,
                [
                    (
                        p.det_index,
                        [
                            (f.factor_name, list(f.class_names))
                            + tuple(np.round(f.raw_probabilities, 6).tolist())
                            for f in p.factors
                        ],
                    )
                    for p in r.predictions
                ],
            )
            for r in fr.cnn
        ],
        "pose": fr.pose and (fr.pose.keypoints.tolist(), fr.pose.valid_mask.tolist()),
        "tags": fr.apriltag
        and (list(fr.apriltag.det_indices), list(fr.apriltag.tag_ids)),
        "headings": fr.resolved_headings.tolist(),
    }


def _expected_final(n):
    survivors = [i for i in range(_N_DETS) if i != _SMALL]
    return survivors[:n]


# --- the proof ---------------------------------------------------------------


@pytest.mark.parametrize("n", [5, 10, 20])
def test_replay_at_any_n_equals_fresh_run(tmp_path, video, n):
    built = tmp_path / "n10"
    _build(built, 10, video)
    fresh_dir = tmp_path / f"fresh{n}"
    fresh = _build(fresh_dir, n, video)

    got = _replay(built, n, video)
    replay_of_fresh = _replay(fresh_dir, n, video)
    for i in range(_NUM_FRAMES):
        got_i = _view(got[i])
        assert got_i == _view(replay_of_fresh[i]), f"frame {i}: vs replay of fresh"
        if i in fresh:  # the batch pass assembles non-empty frames only
            assert got_i == _view(fresh[i]), f"frame {i}: vs fresh in-memory"

    # The fixture really is non-contiguous and has empty frames mid-window.
    for i in range(_NUM_FRAMES):
        expected = [] if i in _EMPTY_FRAMES else _expected_final(n)
        assert got[i].filtered_indices == expected
    assert _SMALL not in got[0].filtered_indices
    # Payloads belong to the raw rows they are keyed by (x = 20 + 50 * raw).
    xs = [20.0 + 50.0 * r for r in _expected_final(n)]
    assert got[0].headtail.heading_hints.tolist() == xs
    assert got[0].pose.keypoints[:, 0, 0].tolist() == xs


def test_replay_at_larger_n_reuses_superset_without_nan(tmp_path, video):
    _build(tmp_path, 10, video)
    p = "hydra_suite.core.inference.pipeline."
    r = "hydra_suite.core.inference.runner."
    patches = [patch(p + s) for s in _STAGES] + [
        patch(r + s)
        for s in ("run_obb", "run_headtail", "run_cnn", "run_pose", "run_apriltag")
    ]
    mocks = [pt.start() for pt in patches]
    try:
        frames = _replay(tmp_path, 20, video)
    finally:
        for pt in patches:
            pt.stop()
    for m in mocks:
        m.assert_not_called()

    for i, f in frames.items():
        if i in _EMPTY_FRAMES:
            assert f.filtered_indices == []
            continue
        assert len(f.filtered_indices) == 20
        assert f.filtered_indices == _expected_final(20)
        assert np.isfinite(f.headtail.heading_hints).all()
        assert np.isfinite(f.headtail.heading_confidences).all()
        assert np.isfinite(f.pose.keypoints).all()
        assert np.isfinite(f.resolved_headings).all()
        (cnn,) = f.cnn
        assert [p.det_index for p in cnn.predictions] == list(range(20))
        for p in cnn.predictions:
            assert np.isfinite(p.factors[0].raw_probabilities).all()


def test_replay_beyond_the_written_superset_raises(tmp_path, video):
    """A looser filter than the caches were written with is loud, not NaN."""
    from hydra_suite.core.inference.downstream_select import DownstreamCacheError
    from hydra_suite.core.inference.runner import InferenceRunner

    _build(tmp_path, 10, video)
    written = _cfg(10)
    looser = _cfg(10)
    looser.obb.min_object_size = 0.0  # admits raw row _SMALL
    with patch("hydra_suite.core.inference.runner._load_all_models") as ml:
        ml.return_value = _models()
        runner = InferenceRunner(
            looser,
            cache_dir=tmp_path,
            video_path=video,
            cache_only=True,
            cache_filter_config=written,
        )
    assert runner.caches_all_valid()
    with pytest.raises(DownstreamCacheError, match="recomputing"):
        runner.load_frame(0)
    assert runner.load_frame(1).filtered_indices == []  # empty frame: nothing to miss


def test_stricter_candidate_drops_a_mid_ranked_row_and_reads_by_raw_index(
    tmp_path, video
):
    """Read-only candidate replay at a stricter SIZE filter than the caches
    were written with: the final set skips a mid-ranked row that IS in the
    written superset, so final positions in the superset are not a prefix
    (positions_in != arange) and payloads must still follow the raw rows."""
    from hydra_suite.core.inference.runner import InferenceRunner

    _build(tmp_path, 10, video)
    candidate = _cfg(10)
    candidate.obb.min_object_size = 100.0  # drops raw _MEDIUM (60) and _SMALL
    with patch("hydra_suite.core.inference.runner._load_all_models") as ml:
        ml.return_value = _models()
        runner = InferenceRunner(
            candidate,
            cache_dir=tmp_path,
            video_path=video,
            cache_only=True,
            cache_filter_config=_cfg(10),
        )
    assert runner.caches_all_valid()

    final = [i for i in range(_N_DETS) if i not in (_MEDIUM, _SMALL)][:10]
    assert final[:3] == [0, 3, 4]
    xs = [20.0 + 50.0 * r for r in final]
    for i in range(_NUM_FRAMES):
        fr = runner.load_frame(i)
        if i in _EMPTY_FRAMES:
            assert fr.filtered_indices == []
            continue
        assert fr.filtered_indices == final
        assert fr.obb.centroids[:, 0].tolist() == xs
        assert fr.headtail.heading_hints.tolist() == xs
        assert fr.pose.keypoints[:, 0, 0].tolist() == xs
        (cnn,) = fr.cnn
        assert [p.det_index for p in cnn.predictions] == list(range(len(final)))
        for p, x in zip(cnn.predictions, xs):
            np.testing.assert_allclose(p.factors[0].raw_probabilities, _probs(x))
        tags = fr.apriltag
        assert len(tags.det_indices) > 0
        for pos, tag_id in zip(tags.det_indices, tags.tag_ids):
            assert tag_id == int(xs[pos])


# --- realtime-written caches replay the same way (Task 7 open check) ---------


def _rt_fakes():
    from tests.test_realtime_superset import _fake_apriltag as rt_tag

    def ht(frame, obb, *a, **k):
        return _fake_ht([frame], [obb], None, None, None, None)[obb.frame_idx]

    def cnn(frame, obb, model, cfg, *a, **k):
        return _fake_cnn([frame], [obb], model, cfg, None, None)[obb.frame_idx]

    def pose(crops, obb, *a, **k):
        batch = MagicMock(obb_by_frame={obb.frame_idx: obb})
        return _fake_pose(batch, None, None, None)[obb.frame_idx]

    return ht, cnn, pose, rt_tag


def _build_realtime(cache_dir, n, video):
    from hydra_suite.core.inference.runner import InferenceRunner

    ht, cnn, pose, tag = _rt_fakes()
    cap = cv2.VideoCapture(str(video))
    frames = [cap.read()[1] for _ in range(_NUM_FRAMES)]
    cap.release()
    r = "hydra_suite.core.inference.runner."
    out = {}
    with (
        patch(r + "_load_all_models") as ml,
        patch(r + "run_obb", side_effect=_fake_run_obb),
        patch(r + "run_headtail", side_effect=ht),
        patch(r + "run_cnn", side_effect=cnn),
        patch(r + "run_pose", side_effect=pose),
        patch(r + "run_apriltag", side_effect=tag),
    ):
        ml.return_value = _models()
        runner = InferenceRunner(_cfg(n), cache_dir=cache_dir, video_path=video)
        for i, frame in enumerate(frames):
            out[i] = runner.run_realtime(frame, i)
        runner.close()
    return out


@pytest.mark.parametrize("n", [5, 20])
def test_realtime_caches_replay_at_other_n(tmp_path, video, n):
    built = tmp_path / "rt10"
    _build_realtime(built, 10, video)
    fresh = _build_realtime(tmp_path / f"rt{n}", n, video)
    got = _replay(built, n, video)
    for i in range(_NUM_FRAMES):
        g, w = got[i], fresh[i]
        assert g.filtered_indices == w.filtered_indices
        expected = [] if i in _EMPTY_FRAMES else _expected_final(n)
        assert g.filtered_indices == expected
        if i in _EMPTY_FRAMES:
            continue
        np.testing.assert_array_equal(g.obb.centroids, w.obb.centroids)
        np.testing.assert_array_equal(
            g.headtail.heading_hints, w.headtail.heading_hints
        )
        np.testing.assert_array_equal(g.pose.keypoints, w.pose.keypoints)
        assert [p.det_index for p in g.cnn[0].predictions] == [
            p.det_index for p in w.cnn[0].predictions
        ]
        for gp, wp in zip(g.cnn[0].predictions, w.cnn[0].predictions):
            np.testing.assert_allclose(
                gp.factors[0].raw_probabilities, wp.factors[0].raw_probabilities
            )
        assert list(g.apriltag.det_indices) == list(w.apriltag.det_indices)
        assert list(g.apriltag.tag_ids) == list(w.apriltag.tag_ids)
        np.testing.assert_allclose(g.resolved_headings, w.resolved_headings)


# --- worker level: identity sidecar reuse at a new N (R9) --------------------
#
# A real TrackingEngineCore drives a real InferenceRunner over the real clip.
# Test seams: the stage functions/models are faked (as above), the identity
# run config is injected (no classifier artifact on disk), and the runner's
# InferenceConfig is the fixture config at the worker's N (the worker builds
# the config from params; the fixture's fake model paths would not resolve).


def _worker_cfg(n):
    cfg = _cfg(n)
    cfg.pose = None
    cfg.apriltag = AprilTagConfig(enabled=False)
    return cfg


def _worker_identity():
    from hydra_suite.core.individual.identity.resolve import resolve_catalog_spec
    from hydra_suite.core.inference.identity_evidence_config import (
        IdentityEvidenceCNNPhaseConfig,
        IdentityEvidenceRunConfig,
    )

    spec = resolve_catalog_spec(
        [
            {
                "label": "id",
                "unique_identifier": True,
                "class_names_per_factor": [_CLASSES],
            }
        ],
        [],
    )
    return IdentityEvidenceRunConfig(
        catalog_spec=spec,
        cnn_phases=(
            IdentityEvidenceCNNPhaseConfig(
                label="id", class_names_per_factor=[_CLASSES]
            ),
        ),
    )


def _worker_params(tmp_path, n):
    from hydra_suite.trackerkit.engine_params import RuntimeContext, build_engine_params
    from tests.test_worker_real_inference_integration import _a_real_model_file

    params = build_engine_params(
        {
            "frame_width": _FRAME_W,
            "frame_height": _FRAME_H,
            "detection_method": "yolo_obb",
            "animals_per_arena": n,
        },
        runtime=RuntimeContext(
            fps=5.0,
            total_frames=_NUM_FRAMES,
            frame_width=_FRAME_W,
            frame_height=_FRAME_H,
        ),
    )
    params.update(
        {
            "START_FRAME": 0,
            "END_FRAME": _NUM_FRAMES - 1,
            "TRACKING_WORKFLOW_MODE": "non_realtime",
            "ENABLE_INDIVIDUAL_PIPELINE": True,
            "YOLO_OBB_MODEL_PATH": _a_real_model_file(tmp_path),
            "CNN_CLASSIFIERS": [],
            "USE_APRILTAGS": False,
            "ENABLE_POSE_EXTRACTOR": False,
            "ENABLE_CONFIDENCE_DENSITY_MAP": False,
            "ENABLE_FRAME_PREFETCH": False,
            "VISUALIZATION_FREE_MODE": True,
            "COMPUTE_RUNTIME": "cpu",
        }
    )
    assert params["MAX_TARGETS"] == n
    return params


def _run_worker(monkeypatch, tmp_path, video, cache_dir, n, *, reuse):
    """One forward worker run at N; returns (success, spies)."""
    import hydra_suite.core.tracking.worker as worker_mod
    from hydra_suite.core.inference.runner import InferenceRunner
    from tests.test_worker_real_inference_integration import _FakeProfiler

    spies = {"batch_pass": 0, "ensure": [], "runners": [], "evidence_reads": []}

    class _Runner(InferenceRunner):
        def __init__(self, config, *a, **k):
            assert config.obb.max_detections == n  # the worker's N reaches us
            assert k.get("identity_evidence") is not None
            super().__init__(_worker_cfg(n), *a, **k)
            spies["runners"].append(self)

        def run_batch_pass(self, *a, **k):
            spies["batch_pass"] += 1
            return super().run_batch_pass(*a, **k)

        def ensure_identity_evidence_sidecar(self, *a, **k):
            spies["ensure"].append((a, k))
            return super().ensure_identity_evidence_sidecar(*a, **k)

    from hydra_suite.core.individual.identity import cache as ev_cache_mod

    real_ev_init = ev_cache_mod.IdentityEvidenceCache.__init__

    def _ev_init(self, path, *a, **k):
        real_ev_init(self, path, *a, **k)
        if self._mode == "r":
            spies["evidence_reads"].append(str(path))

    monkeypatch.setattr(worker_mod, "TrackingProfiler", _FakeProfiler)
    monkeypatch.setattr(worker_mod, "InferenceRunner", _Runner)
    monkeypatch.setattr(
        worker_mod.TrackingEngineCore,
        "_resolve_identity_evidence_run_config",
        lambda self, p: _worker_identity(),
    )
    monkeypatch.setattr(ev_cache_mod.IdentityEvidenceCache, "__init__", _ev_init)

    models = _models()
    models.pose = None
    models.apriltag = None
    p = "hydra_suite.core.inference.pipeline."
    r = "hydra_suite.core.inference.runner."
    captured = {}
    with (
        patch(r + "_load_all_models", return_value=models),
        patch(p + "run_obb", side_effect=_fake_run_obb),
        patch(p + "run_headtail_batch", side_effect=_fake_ht),
        patch(p + "run_cnn_batch", side_effect=_fake_cnn) as cnn_batch,
        patch(r + "run_cnn") as cnn_rt,
    ):
        worker = worker_mod.TrackingEngineCore(
            str(video),
            on_finished=lambda ok, _fps, _traj: captured.__setitem__("ok", ok),
            use_cached_detections=reuse,
            inference_cache_dir=str(cache_dir),
        )
        worker.set_parameters(_worker_params(tmp_path, n))
        worker.run_tracking()
    spies["cnn_calls"] = cnn_batch.call_count + cnn_rt.call_count
    return captured.get("ok"), spies


def _sidecar_evidence(path):
    from hydra_suite.core.individual.identity.cache import IdentityEvidenceCache

    cache = IdentityEvidenceCache(str(path), mode="r")
    return {
        f: sorted(
            (e.detection_id, e.source_name, tuple(np.round(e.log_probs, 6)))
            for e in cache.load_frame(f)
        )
        for f in range(_NUM_FRAMES)
    }


def test_worker_reuse_at_new_n_rebuilds_identity_sidecar_without_cnn(
    monkeypatch, tmp_path, video
):
    shared = tmp_path / "cache"
    ok, fwd = _run_worker(monkeypatch, tmp_path, video, shared, 10, reuse=False)
    assert ok is True
    assert fwd["batch_pass"] == 1 and fwd["cnn_calls"] > 0
    sidecar10 = fwd["runners"][0].identity_evidence_sidecar_path("batch")
    assert sidecar10.exists()

    ok, rerun = _run_worker(monkeypatch, tmp_path, video, shared, 3, reuse=True)
    assert ok is True
    assert rerun["batch_pass"] == 0, "caches written at N=10 must serve N=3"
    assert rerun["cnn_calls"] == 0, "reuse must not run the CNN"
    assert len(rerun["ensure"]) == 1
    sidecar3 = rerun["runners"][0].identity_evidence_sidecar_path("batch")
    assert sidecar3 != sidecar10
    assert sidecar3.exists()
    assert str(sidecar3) in rerun["evidence_reads"], "worker must load the N=3 sidecar"

    # The rebuilt sidecar equals a fresh forward run's at N=3.
    fresh_dir = tmp_path / "fresh3"
    ok, fresh = _run_worker(monkeypatch, tmp_path, video, fresh_dir, 3, reuse=False)
    assert ok is True and fresh["batch_pass"] == 1
    want = _sidecar_evidence(
        fresh["runners"][0].identity_evidence_sidecar_path("batch")
    )
    got = _sidecar_evidence(sidecar3)
    assert got == want
    nonempty = [f for f in range(_NUM_FRAMES) if f not in _EMPTY_FRAMES]
    assert all(len(got[f]) == 3 for f in nonempty)  # final N=3 set has evidence
    assert all(got[f] == [] for f in _EMPTY_FRAMES)
