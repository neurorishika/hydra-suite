"""Task 6: worker wiring — layout, per-frame arena lookup, and the arena column.

Every prior arena task's code (`build_arena_labels`, `ArenaLayout`,
`TrackAssigner.set_track_arena`/`meas_arena`, `ArenaDecoderRegistry`) is dormant
until `engine_params.py`/`worker.py` actually connect it. These tests prove the
connections, not just that the interfaces exist:

* `build_engine_params` derives `MAX_TARGETS` from `N_ARENAS * ANIMALS_PER_ARENA`
  and emits `ARENA_LABELS`/`N_ARENAS`/`ANIMALS_PER_ARENA`.
* A legacy config (no `roi_shapes`/`arena_id`) reproduces today's single-arena
  `MAX_TARGETS` exactly.
* The raw tracking CSV header gains `arena_id` ONLY when `n_arenas > 1`.
* `worker.py` resolves the per-frame arena lookup in the SAME coordinate space
  the detector's centroids are actually in -- the resized tracking frame, not
  the arena label image's native resolution -- driving the REAL
  `TrackingEngineCore.run_tracking()` loop end to end with a faked bg-sub
  runner (background subtraction is the one detection method RESIZE_FACTOR
  is not clamped to 1.0 for).
"""

from __future__ import annotations

import numpy as np

from hydra_suite.trackerkit.engine_params import RuntimeContext, build_engine_params


def _runtime(width: int = 100, height: int = 100) -> RuntimeContext:
    return RuntimeContext(
        fps=30.0, total_frames=100, frame_width=width, frame_height=height
    )


def _cfg(**over):
    cfg = {
        "roi_shapes": [
            {
                "type": "circle",
                "params": [25, 25, 15],
                "mode": "include",
                "arena_id": 0,
            },
            {
                "type": "circle",
                "params": [75, 25, 15],
                "mode": "include",
                "arena_id": 1,
            },
        ],
        "animals_per_arena": 3,
        "frame_width": 100,
        "frame_height": 100,
    }
    cfg.update(over)
    return cfg


def _params(cfg):
    return build_engine_params(
        cfg, runtime=_runtime(cfg.get("frame_width", 100), cfg.get("frame_height", 100))
    )


# ---------------------------------------------------------------------------
# engine_params.py: MAX_TARGETS derivation + ARENA_LABELS emission
# ---------------------------------------------------------------------------


def test_max_targets_is_derived_from_arenas_and_animals():
    params = _params(_cfg())
    assert params["N_ARENAS"] == 2
    assert params["ANIMALS_PER_ARENA"] == 3
    assert params["MAX_TARGETS"] == 6


def test_arena_labels_are_emitted_and_agree_with_roi_mask():
    params = _params(_cfg())
    labels = params["ARENA_LABELS"]
    assert labels.dtype == np.uint16
    np.testing.assert_array_equal(labels > 0, params["ROI_MASK"] > 0)


def test_legacy_config_without_arena_ids_is_single_arena():
    cfg = _cfg(
        roi_shapes=[{"type": "circle", "params": [50, 50, 30], "mode": "include"}],
        animals_per_arena=4,
    )
    params = _params(cfg)
    assert params["N_ARENAS"] == 1
    assert params["MAX_TARGETS"] == 4


def test_no_roi_shapes_at_all_is_single_arena_and_matches_legacy_max_targets():
    """Mutation target: deleting the `max_targets = n_arenas * animals_per_arena`
    override (leaving the old `max_targets` line untouched) would silently pass
    here too IF `animals_per_arena` defaulted to something other than the
    legacy `max_targets` knob -- this pins the default chain."""
    cfg = {"frame_width": 100, "frame_height": 100, "max_targets": 7}
    params = _params(cfg)
    assert params["N_ARENAS"] == 1
    assert params["ARENA_LABELS"] is None
    # No `animals_per_arena` override -> falls back to the legacy `max_targets`
    # knob, so single-arena configs keep today's exact MAX_TARGETS semantics.
    assert params["MAX_TARGETS"] == 7


# ---------------------------------------------------------------------------
# headless_tracking.py: `arena_id` column gated on n_arenas > 1
# ---------------------------------------------------------------------------


def test_csv_header_omits_arena_id_for_single_arena():
    from hydra_suite.trackerkit.headless_tracking import build_tracking_csv_header

    header = build_tracking_csv_header(n_arenas=1)
    assert "arena_id" not in header


def test_csv_header_includes_arena_id_for_multi_arena():
    from hydra_suite.trackerkit.headless_tracking import build_tracking_csv_header

    header = build_tracking_csv_header(n_arenas=2)
    assert header[-1] == "arena_id"


# ---------------------------------------------------------------------------
# worker.py: ArenaLayout wiring + coordinate-space pin (end-to-end)
#
# Drives the REAL TrackingEngineCore.run_tracking() loop -- a faked
# InferenceRunner/bgsub runner at the module boundary (same pattern as
# tests/test_worker_real_inference_integration.py), a real synthetic video,
# and a real CSV row capture -- so a wiring regression (e.g. reverting to the
# label image's native resolution, or never calling set_track_arena/threading
# meas_arena into assign_tracks) fails on ACTUAL engine behavior, not a mock
# assertion.
# ---------------------------------------------------------------------------


class _FakeProfiler:
    def __init__(self, enabled: bool = False):
        self.enabled = enabled

    def __getattr__(self, _name):
        return lambda *a, **k: None


class _FakeVideoCapture:
    """Two 100x100 BGR frames -- big enough to carry a real left/right split."""

    WIDTH = 100
    HEIGHT = 100

    def __init__(self, *_args, **_kwargs):
        self._frames = [
            np.zeros((self.HEIGHT, self.WIDTH, 3), dtype=np.uint8) for _ in range(2)
        ]
        self._idx = 0
        self._opened = True

    def isOpened(self):
        return self._opened

    def read(self):
        if self._idx >= len(self._frames):
            return False, None
        frame = self._frames[self._idx]
        self._idx += 1
        return True, frame.copy()

    def get(self, prop_id):
        import hydra_suite.core.tracking.worker as _wm

        if prop_id == _wm.cv2.CAP_PROP_FRAME_COUNT:
            return len(self._frames)
        if prop_id == _wm.cv2.CAP_PROP_FPS:
            return 30.0
        if prop_id == _wm.cv2.CAP_PROP_FRAME_WIDTH:
            return self.WIDTH
        if prop_id == _wm.cv2.CAP_PROP_FRAME_HEIGHT:
            return self.HEIGHT
        if prop_id == _wm.cv2.CAP_PROP_POS_FRAMES:
            return self._idx
        return 0

    def set(self, prop_id, value):
        import hydra_suite.core.tracking.worker as _wm

        if prop_id == _wm.cv2.CAP_PROP_POS_FRAMES:
            self._idx = int(value)
        return True

    def release(self):
        self._opened = False


class _ClippingStatsStub:
    def summary(self):
        return ""


class _FakeBgsubResult:
    def __init__(self, fg_mask, bg_u8, obb):
        self.fg_mask = fg_mask
        self.bg_u8 = bg_u8
        self.obb = obb


class _FakeBgsubRunner:
    """Frame 0: warmup (no background model yet). Frame 1: one detection at
    resized-frame coordinate (40, 25) -- i.e. RIGHT of the resized frame's
    midpoint (25 = 50/2), which must resolve to arena 1."""

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


class _CapturingCSVWriter:
    def __init__(self):
        self.rows = []

    def enqueue(self, row):
        self.rows.append(list(row))


def _bgsub_arena_params(*, single_arena: bool = False, **overrides):
    """Real engine params (via `build_engine_params`) for a bg-sub run with two
    left/right arenas at NATIVE 100x100 resolution -- the arena boundary sits
    at native x=50. `RESIZE_FACTOR=0.5` means the tracking frame (and hence
    detection coordinates) is 50x50, boundary at x=25.

    Using the real builder (rather than a hand-assembled params dict) means
    `ARENA_LABELS`/`N_ARENAS`/`ANIMALS_PER_ARENA`/`MAX_TARGETS` and every other
    assigner-required default (`W_POSITION`, `USE_MAHALANOBIS`, ...) come from
    the same code path production configs go through.
    """
    cfg = {
        "frame_width": 100,
        "frame_height": 100,
        "detection_method": "background_subtraction",
        "resize_factor": 0.5,
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
    params = build_engine_params(cfg, runtime=_runtime(100, 100))
    params.update(
        {
            "START_FRAME": 0,
            "END_FRAME": 1,
            "MIN_DETECTIONS_TO_START": 1,
            "MIN_DETECTION_COUNTS": 2,
            "LOST_THRESHOLD_FRAMES": 1,
            # Large enough that Phase-3 respawn's proximity check (comparing the
            # resized-space detection at ~(40, 25) against each lost track's
            # zero-initialised KF position) actually succeeds for the
            # ARENA-CORRECT candidate on the very first detection frame -- this
            # is what makes the test exercise `_assign_respawn`'s arena gate
            # (`TrackAssigner.set_track_arena`/`meas_arena`) rather than falling
            # through to worker.py's un-gated `free_dets` bootstrap loop, which
            # would pass regardless of whether the arena lookup is correct.
            "MAX_DISTANCE_THRESHOLD": 1000.0,
            "ENABLE_POSE_EXTRACTOR": False,
            "USE_APRILTAGS": False,
            "CNN_CLASSIFIERS": [],
            "ENABLE_CONFIDENCE_DENSITY_MAP": False,
            "ENABLE_FRAME_PREFETCH": False,
            "COMPUTE_RUNTIME": "cpu",
        }
    )
    params.update(overrides)
    return params


def _run_bgsub_arena_case(monkeypatch, tmp_path, single_arena: bool = False):
    """Drive the real run_tracking() bg-sub loop; return captured CSV rows."""
    import hydra_suite.core.tracking.worker as worker_mod

    monkeypatch.setattr(worker_mod, "TrackingProfiler", _FakeProfiler)
    monkeypatch.setattr(worker_mod.cv2, "VideoCapture", _FakeVideoCapture)
    monkeypatch.setattr(worker_mod, "InferenceRunner", _FakeBgsubRunner)

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
    worker.set_parameters(_bgsub_arena_params(single_arena=single_arena))
    worker.run_tracking()
    assert captured.get("success") is not False
    return csv_writer.rows


def _matched_rows(rows):
    """Rows with a real (non-NaN) X value -- i.e. an actual detection was
    written for that track slot this frame."""
    return [r for r in rows if not np.isnan(float(r[3]))]


def test_worker_wires_arena_layout_and_gates_assignment_by_arena(monkeypatch, tmp_path):
    """End-to-end pin: a detection at RESIZED-frame (40, 25) -- right of the
    resized midpoint (25) -- must initialize the arena-1 track slot (slot 1,
    since ANIMALS_PER_ARENA=1 lays slot 0 -> arena 0, slot 1 -> arena 1).

    Mutation coverage: reverting the worker.py `frame_size=(target_w, target_h)`
    argument to `frame_size=None` (native-resolution lookup) looks up (40, 25)
    in the NATIVE 100x100 label image instead, where the arena boundary is at
    x=50 -- landing in arena 0 instead of arena 1 -- and this assertion fails.
    Likewise, removing `assigner.set_track_arena(...)` or the `meas_arena=`
    threading into `assign_tracks`/`compute_cost_matrix` un-gates the
    assignment entirely, and BOTH slots become eligible -- Hungarian/respawn
    tie-breaking would then pick slot 0 first (lowest index), which also fails
    this assertion.
    """
    rows = _run_bgsub_arena_case(monkeypatch, tmp_path)
    matched = _matched_rows(rows)
    assert matched, "expected at least one row with a real detection"
    track_ids = {int(r[0]) for r in matched}
    assert track_ids == {1}, (
        f"expected the detection to initialize arena-1's slot (TrackID 1), "
        f"got TrackID(s) {track_ids} -- coordinate space or arena gating regressed"
    )
    # arena_id is the last column (n_arenas=2 > 1 -> column emitted).
    assert all(int(r[-1]) == 1 for r in matched)


def test_single_arena_run_never_touches_arena_gating(monkeypatch, tmp_path):
    """With N_ARENAS=1, `arena_id` must not be emitted (single-arena byte-
    identity contract) and the sole track slot must be free to take the
    detection regardless of its position -- there is no arena to gate on."""
    rows = _run_bgsub_arena_case(monkeypatch, tmp_path, single_arena=True)
    matched = _matched_rows(rows)
    assert matched, "expected at least one row with a real detection"
    assert all(int(r[0]) == 0 for r in matched)
    # Header-column contract: no arena_id column, so the last column here is
    # whatever the identity/tag block ends with -- not an int arena id. The
    # row length for a single-arena run must equal exactly the multi-arena
    # row length minus one (the arena_id column dropped).
    multi_rows = _run_bgsub_arena_case(monkeypatch, tmp_path)
    assert len(rows[0]) == len(multi_rows[0]) - 1
