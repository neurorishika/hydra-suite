"""Task 12 / Fix 2: the parameter optimizer must simulate the run it tunes.

`TrackingOptimizerCore._run_tracking_loop` and `run_tracking_preview` both
build a `TrackAssigner` and drive the real assignment loop. Before this fix
neither called `set_track_arena`, and neither passed `meas_arena` to
`compute_cost_matrix`/`assign_tracks` -- so both simulated UNRESTRICTED
cross-arena tracking with one joint Hungarian, the exact defect Task 11 removed
from the live path, and then handed the parameters they tuned to a properly
gated run.

These tests drive the REAL loops with a real (tiny) video and a faked detection
cache, and assert on what the assigner actually RECEIVES, in the coordinate
space the detections are actually in.
"""

from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

import hydra_suite.core.tracking.optimization.optimizer as opt_mod  # noqa: E402
from hydra_suite.core.inference.result import OBBResult  # noqa: E402
from hydra_suite.trackerkit.engine_params import (  # noqa: E402
    RuntimeContext,
    build_engine_params,
)

# Native video is 100x100 with the arena wall at native x=50. RESIZE_FACTOR
# 0.5 means detections live in a 50x50 space whose wall is at x=25.
NATIVE = 100
RESIZE = 0.5
# Right of the RESIZED wall (25) but left of the NATIVE wall (50): resolving
# this point in the wrong coordinate space flips its arena from 1 to 0.
DET_XY = (40.0, 25.0)


def _video(tmp_path):
    path = tmp_path / "arena_opt.mp4"
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (NATIVE, NATIVE)
    )
    for _ in range(3):
        writer.write(np.zeros((NATIVE, NATIVE, 3), dtype=np.uint8))
    writer.release()
    assert path.exists()
    return path


def _params(single_arena: bool = False):
    cfg = {
        "frame_width": NATIVE,
        "frame_height": NATIVE,
        "detection_method": "background_subtraction",
        "resize_factor": RESIZE,
        "animals_per_arena": 1,
        "reference_body_size": 10.0,
    }
    if not single_arena:
        cfg["roi_shapes"] = [
            {
                "type": "polygon",
                "mode": "include",
                "arena_id": 0,
                "params": [[0, 0], [50, 0], [50, 100], [0, 100]],
            },
            {
                "type": "polygon",
                "mode": "include",
                "arena_id": 1,
                "params": [[50, 0], [100, 0], [100, 100], [50, 100]],
            },
        ]
    params = build_engine_params(
        cfg,
        runtime=RuntimeContext(
            fps=30.0, total_frames=3, frame_width=NATIVE, frame_height=NATIVE
        ),
    )
    params.update(
        {
            "MAX_DISTANCE_THRESHOLD": 1000.0,
            "LOST_THRESHOLD_FRAMES": 1,
            "ENABLE_POSE_EXTRACTOR": False,
        }
    )
    return params


class _OneDetectionCache:
    def read_frame(self, frame_idx):
        cx, cy = DET_XY
        return OBBResult(
            frame_idx=frame_idx,
            centroids=np.array([[cx, cy]], dtype=np.float32),
            angles=np.array([0.0], dtype=np.float32),
            sizes=np.array([50.0], dtype=np.float32),
            shapes=np.array([[10.0, 1.2]], dtype=np.float32),
            confidences=np.array([0.9], dtype=np.float32),
            corners=np.zeros((1, 4, 2), dtype=np.float32),
            detection_ids=OBBResult.make_detection_ids(frame_idx, 1),
        )


class _ArenaProbeAssigner:
    """Wraps the REAL TrackAssigner and records what the optimizer gives it."""

    calls: dict = {}

    def __init__(self, params):
        from hydra_suite.core.assigners.hungarian import TrackAssigner

        self._inner = TrackAssigner(params)
        self.params = params
        _ArenaProbeAssigner.calls = {
            "set_track_arena": [],
            "cost_meas_arena": [],
            "assign_meas_arena": [],
        }

    def set_track_arena(self, track_arena):
        _ArenaProbeAssigner.calls["set_track_arena"].append(
            None if track_arena is None else list(map(int, track_arena))
        )
        return self._inner.set_track_arena(track_arena)

    @property
    def track_arena(self):
        return self._inner.track_arena

    def compute_cost_matrix(self, *args, **kwargs):
        _ArenaProbeAssigner.calls["cost_meas_arena"].append(
            _norm(kwargs.get("meas_arena", "MISSING"))
        )
        return self._inner.compute_cost_matrix(*args, **kwargs)

    def assign_tracks(self, *args, **kwargs):
        _ArenaProbeAssigner.calls["assign_meas_arena"].append(
            _norm(kwargs.get("meas_arena", "MISSING"))
        )
        return self._inner.assign_tracks(*args, **kwargs)


def _norm(value):
    if isinstance(value, str) or value is None:
        return value
    return list(map(int, value))


def _run_optimizer_loop(monkeypatch, tmp_path, *, single_arena):
    monkeypatch.setattr(opt_mod, "TrackAssigner", _ArenaProbeAssigner)
    optimizer = opt_mod.TrackingOptimizerCore(
        video_path=str(_video(tmp_path)),
        detection_cache_path=str(tmp_path / "cache"),
        start_frame=0,
        end_frame=1,
        base_params={},
        tuning_config={},
    )
    optimizer.cache = _OneDetectionCache()
    optimizer._pose_run_context = (None, [], [], [], False)
    optimizer._pose_frame_cache = {}
    optimizer._stop_requested = False
    score, _positions, _extra = optimizer._run_tracking_loop(_params(single_arena))
    assert np.isfinite(score)
    return _ArenaProbeAssigner.calls


# ---------------------------------------------------------------------------
# optimizer.py -- the scoring loop that actually chooses the parameters
# ---------------------------------------------------------------------------


def test_optimizer_installs_the_slot_arena_mapping(monkeypatch, tmp_path):
    calls = _run_optimizer_loop(monkeypatch, tmp_path, single_arena=False)
    assert calls["set_track_arena"] == [[0, 1]], (
        "the optimizer must label each track slot with its arena "
        f"(2 arenas x 1 animal -> [0, 1]); got {calls['set_track_arena']}"
    )


def test_optimizer_threads_meas_arena_in_the_detections_coordinate_space(
    monkeypatch, tmp_path
):
    """The detection at resized (40, 25) is right of the RESIZED wall (25), so
    it belongs to arena 1. Resolving it in the label image's native space
    (wall at 50) would call it arena 0 -- this asserts on the resolved id, so
    a wrong coordinate space fails, not just a missing kwarg."""
    calls = _run_optimizer_loop(monkeypatch, tmp_path, single_arena=False)
    assert calls["cost_meas_arena"], "the cost matrix was never built"
    assert all(c == [1] for c in calls["cost_meas_arena"]), calls["cost_meas_arena"]
    assert all(c == [1] for c in calls["assign_meas_arena"]), calls["assign_meas_arena"]


def test_single_arena_optimizer_run_stays_structurally_ungated(monkeypatch, tmp_path):
    """`None` everywhere -- the assigner's pre-arena path, not an arena gate
    that happens to allow everything."""
    calls = _run_optimizer_loop(monkeypatch, tmp_path, single_arena=True)
    assert calls["set_track_arena"] == [None]
    assert calls["cost_meas_arena"] and all(
        c is None for c in calls["cost_meas_arena"]
    ), calls["cost_meas_arena"]
    assert all(c is None for c in calls["assign_meas_arena"]), calls[
        "assign_meas_arena"
    ]


def test_optimizer_refuses_a_layout_that_does_not_cover_every_slot(
    monkeypatch, tmp_path
):
    """Half-gated cost matrices must be loud, not silent (see
    `check_slot_arena_covers_all_slots`)."""
    monkeypatch.setattr(opt_mod, "TrackAssigner", _ArenaProbeAssigner)
    optimizer = opt_mod.TrackingOptimizerCore(
        video_path=str(_video(tmp_path)),
        detection_cache_path=str(tmp_path / "cache"),
        start_frame=0,
        end_frame=0,
        base_params={},
        tuning_config={},
    )
    optimizer.cache = _OneDetectionCache()
    optimizer._pose_run_context = (None, [], [], [], False)
    optimizer._pose_frame_cache = {}
    optimizer._stop_requested = False
    params = _params(single_arena=False)
    params["MAX_TARGETS"] = 5  # no longer n_arenas * animals_per_arena
    with pytest.raises(RuntimeError, match="exactly one entry per"):
        optimizer._run_tracking_loop(params)


# ---------------------------------------------------------------------------
# optimizer_workers.py -- the preview loop the user watches
# ---------------------------------------------------------------------------


def _run_preview(monkeypatch, tmp_path, *, single_arena):
    import types

    import hydra_suite.core.tracking.optimization.optimizer_workers as ow

    monkeypatch.setattr(ow, "TrackAssigner", _ArenaProbeAssigner)

    class _Handle(_OneDetectionCache):
        def is_valid(self):
            return True

    monkeypatch.setattr(
        ow,
        "_open_caches",
        lambda *_a, **_k: types.SimpleNamespace(
            detection=_Handle(), set_manifest_valid=True
        ),
    )
    monkeypatch.setattr(ow, "video_signature", lambda *_a, **_k: "sig")
    monkeypatch.setattr(
        ow, "build_inference_config_from_params", lambda *_a, **_k: None
    )

    frames = []
    ow.run_tracking_preview(
        video_path=str(_video(tmp_path)),
        detection_cache_path=str(tmp_path / "cache"),
        start_frame=0,
        end_frame=1,
        params=_params(single_arena),
        frame_cb=frames.append,
        stop_check=lambda: False,
    )
    assert frames, "preview rendered no frames -- the test proves nothing"
    return _ArenaProbeAssigner.calls


def test_preview_installs_the_slot_arena_mapping(monkeypatch, tmp_path):
    calls = _run_preview(monkeypatch, tmp_path, single_arena=False)
    assert calls["set_track_arena"] == [[0, 1]]
    assert calls["cost_meas_arena"] and all(
        c == [1] for c in calls["cost_meas_arena"]
    ), calls["cost_meas_arena"]
    assert all(c == [1] for c in calls["assign_meas_arena"]), calls["assign_meas_arena"]


def test_single_arena_preview_stays_structurally_ungated(monkeypatch, tmp_path):
    calls = _run_preview(monkeypatch, tmp_path, single_arena=True)
    assert calls["set_track_arena"] == [None]
    assert all(c is None for c in calls["cost_meas_arena"]), calls["cost_meas_arena"]
    assert all(c is None for c in calls["assign_meas_arena"]), calls[
        "assign_meas_arena"
    ]
