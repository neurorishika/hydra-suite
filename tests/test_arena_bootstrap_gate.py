"""Fixture-free regression test for the free-detection bootstrap arena gate.

`worker.py`'s Phase-3 respawn (`TrackAssigner._assign_respawn`) has its own
arena gate and is already covered by
`test_arena_worker_wiring.py::test_worker_wires_arena_layout_and_gates_assignment_by_arena`
-- but that test deliberately inflates `MAX_DISTANCE_THRESHOLD` to 1000.0 so
the proximity check on the zero-initialised Kalman state trivially succeeds,
which routes the detection through `_assign_respawn` rather than through
worker.py's SEPARATE, un-gated `free_dets` bootstrap loop (see the comment at
`tests/test_arena_worker_wiring.py:325` and the loop itself in
`core/tracking/worker.py` near the `for d_idx in free_dets:` block, just
below the "Arena gate for the bootstrap loop" comment).

At a REALISTIC `MAX_DISTANCE_THRESHOLD` (the engine's own default -- 2.5x
scaled body size), a genuine cold-start detection is always far from the
zero-initialised KF position `(0, 0)`, so `_assign_respawn`'s proximity gate
correctly rejects it (best distance >> MAX_DIST) and the detection falls
through to `free_dets` -- exercising the ACTUAL bootstrap loop this test
targets. That loop takes the FIRST "lost" slot in GLOBAL slot order unless it
is itself arena-gated -- which, prior to the gate this test pins, it was
not. This is "the branch's largest cross-arena leak" per the final
whole-branch review: a detection born in arena 1 could bootstrap into arena
0's slot, and the arena-blocked cost matrix then refuses to ever rematch that
slot to its own arena's detections, so the slot churns every frame.

This test drives the REAL `TrackingEngineCore.run_tracking()` bg-sub loop
end-to-end (same technique as `test_arena_worker_wiring.py`) with two tiny
in-memory fakes (a fake `VideoCapture` and a fake `InferenceRunner`) -- no
video file, no equivalence clip, no gitignored fixture of any kind -- so it
runs on a bare fresh clone and cannot be skipped for missing fixtures the way
the fixture-gated tiling oracle can.
"""

from __future__ import annotations

import numpy as np

from tests.test_arena_worker_wiring import (
    _bgsub_arena_params,
    _CapturingCSVWriter,
    _ClippingStatsStub,
    _FakeBgsubResult,
    _FakeProfiler,
    _FakeVideoCapture,
    _matched_rows,
)


class _SteadyArena1Runner:
    """Call 1 (frame 0): warmup (no background model yet, `obb=None`).
    Calls 2-3 (frames 1-2): the SAME detection at resized-frame (40, 25) --
    right of the resized midpoint (25), i.e. arena 1's territory.

    Two consecutive detection frames (not just one) matter here: the
    bootstrap loop updates internal track state (KF init, `track_states`,
    `trajectory_ids`) for THIS frame, but the CSV row for THIS frame is
    written earlier in the loop body, before the bootstrap runs -- so the
    bootstrap frame's own row still shows the pre-bootstrap 'lost'/NaN
    values. The SECOND detection frame is matched normally (phase 1/2,
    not respawn) against the now-initialised track, and its row carries the
    real coordinates -- which is what lets this test observe which slot
    (i.e. which arena) the cold start actually landed in.
    """

    DETECTION_XY_RESIZED = (40.0, 25.0)

    def __init__(self, *_a, **_k):
        self.clipping_stats = _ClippingStatsStub()
        self._call = 0

    def run_realtime(self, frame, frame_idx, roi_mask=None):
        from hydra_suite.core.inference.result import OBBResult

        self._call += 1
        h, w = frame.shape[0], frame.shape[1]
        if self._call == 1:
            return _FakeBgsubResult(fg_mask=None, bg_u8=None, obb=None)
        cx, cy = self.DETECTION_XY_RESIZED
        obb = OBBResult(
            frame_idx=frame_idx,
            centroids=np.array([[cx, cy]], dtype=np.float32),
            angles=np.array([0.0], dtype=np.float32),
            sizes=np.array([50.0], dtype=np.float32),
            shapes=np.array([[10.0, 1.2]], dtype=np.float32),
            confidences=np.array([float("nan")], dtype=np.float32),
            corners=np.zeros((1, 4, 2), dtype=np.float32),
            detection_ids=OBBResult.make_detection_ids(frame_idx, 1),
        )
        return _FakeBgsubResult(
            fg_mask=np.zeros((h, w), dtype=np.uint8),
            bg_u8=np.zeros((h, w), dtype=np.uint8),
            obb=obb,
        )

    def close(self):
        pass


def _run_cold_start_bootstrap_case(monkeypatch, tmp_path):
    """Same driving technique as `_run_bgsub_arena_case`, but with a REALISTIC
    (non-inflated) `MAX_DISTANCE_THRESHOLD` so the cold-start detection cannot
    be claimed by `_assign_respawn`'s proximity check and must instead be
    resolved by the free-detection bootstrap loop.

    `_bgsub_arena_params` hardcodes `MAX_DISTANCE_THRESHOLD=1000.0` internally
    (see its docstring) specifically to route AROUND this loop; passing the
    engine's own real default (`2.5 * REFERENCE_BODY_SIZE * RESIZE_FACTOR` =
    `2.5 * 10.0 * 0.5` = `12.5`) as an override restores the realistic gate.
    The fake detection sits at resized-space `(40, 25)`, distance
    `sqrt(40**2 + 25**2)` ~= 47.2 from the zero-initialised KF origin --
    far outside a 12.5-px gate, so `_assign_respawn` MUST reject it.
    """
    import hydra_suite.core.tracking.worker as worker_mod

    monkeypatch.setattr(worker_mod, "TrackingProfiler", _FakeProfiler)
    monkeypatch.setattr(worker_mod.cv2, "VideoCapture", _FakeVideoCapture)
    monkeypatch.setattr(worker_mod, "InferenceRunner", _SteadyArena1Runner)

    csv_writer = _CapturingCSVWriter()
    captured = {}

    def _on_finished(success, _fps, _traj):
        captured["success"] = success

    worker = worker_mod.TrackingEngineCore(
        str(tmp_path / "video.mp4"),
        on_finished=_on_finished,
        preview_mode=True,
        csv_writer_thread=csv_writer,
    )
    params = _bgsub_arena_params(
        single_arena=False,
        MAX_DISTANCE_THRESHOLD=12.5,
    )
    worker.set_parameters(params)
    worker.run_tracking()
    assert captured.get("success") is not False
    return csv_writer.rows


def test_cold_start_bootstrap_is_arena_gated(monkeypatch, tmp_path):
    """A cold-start detection in arena 1 must bootstrap arena 1's slot
    (TrackID 1), never arena 0's slot (TrackID 0), even though slot 0 is the
    first "lost" slot in global order and every track starts "lost".

    Without the arena gate on the `free_dets` bootstrap loop, this detection
    -- unreachable by `_assign_respawn`'s proximity gate at this realistic
    `MAX_DISTANCE_THRESHOLD` -- would be handed to slot 0 (arena 0) purely
    because it is first in global slot order, regardless of which arena the
    detection is actually in. That is the leak: a detection born in one
    arena silently bootstraps a DIFFERENT arena's slot, and the arena-blocked
    cost matrix then refuses to ever rematch that slot to its own arena's
    real detections again.
    """
    rows = _run_cold_start_bootstrap_case(monkeypatch, tmp_path)
    matched = _matched_rows(rows)
    assert matched, "expected at least one row with a real (non-NaN) detection"
    track_ids = {int(r[0]) for r in matched}
    assert track_ids == {1}, (
        "cold-start bootstrap must land in arena-1's slot (TrackID 1) -- "
        f"got TrackID(s) {track_ids}. This is the un-gated free_dets "
        "bootstrap loop handing an arena-1 detection to arena 0's slot."
    )
    for r in matched:
        x = float(r[3])
        assert not np.isnan(x)
