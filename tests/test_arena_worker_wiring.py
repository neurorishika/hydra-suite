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
    """Three 100x100 BGR frames -- a left/right split, plus a third frame to
    reach the "no detection this frame" bulk-NaN-row CSV branch after
    `detection_initialized` has already gone True."""

    WIDTH = 100
    HEIGHT = 100
    NUM_FRAMES = 3

    def __init__(self, *_args, **_kwargs):
        self._frames = [
            np.zeros((self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
            for _ in range(self.NUM_FRAMES)
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
    """Call 1 (frame 0): warmup (no background model yet). Call 2 (frame 1):
    one detection at resized-frame coordinate (40, 25) -- i.e. RIGHT of the
    resized frame's midpoint (25 = 50/2), which must resolve to arena 1.
    Call 3 (frame 2): no detections -- reaches the "no detection this frame"
    bulk-NaN-row CSV branch (the third `csv_writer_thread.enqueue` site)."""

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
        if self._call == 3:
            obb = OBBResult(
                frame_idx=frame_idx,
                centroids=np.zeros((0, 2), dtype=np.float32),
                angles=np.zeros(0, dtype=np.float32),
                sizes=np.zeros(0, dtype=np.float32),
                shapes=np.zeros((0, 2), dtype=np.float32),
                confidences=np.zeros(0, dtype=np.float32),
                corners=np.zeros((0, 4, 2), dtype=np.float32),
                detection_ids=OBBResult.make_detection_ids(frame_idx, 0),
            )
            return _FakeBgsubResult(
                fg_mask=np.zeros((h, w), dtype=np.uint8),
                bg_u8=np.zeros((h, w), dtype=np.uint8),
                obb=obb,
            )
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
            "END_FRAME": 2,
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


def _run_bgsub_density_case(monkeypatch, tmp_path, regions, single_arena=False):
    """Same real bg-sub loop, but with density regions pre-loaded and the
    density flag helper spied on, so the `meas_arena=` threading at worker.py's
    `get_density_region_flags` call site is observed on live engine data."""
    import hydra_suite.core.tracking.worker as worker_mod

    monkeypatch.setattr(worker_mod, "TrackingProfiler", _FakeProfiler)
    monkeypatch.setattr(worker_mod.cv2, "VideoCapture", _FakeVideoCapture)
    monkeypatch.setattr(worker_mod, "InferenceRunner", _FakeBgsubRunner)

    calls = []
    _real = worker_mod.get_density_region_flags

    def _spy(meas, regions_, frame_idx, meas_arena=None):
        out = _real(meas, regions_, frame_idx, meas_arena=meas_arena)
        calls.append(
            {
                "meas": [list(m)[:2] for m in meas],
                "meas_arena": (
                    None if meas_arena is None else list(map(int, meas_arena))
                ),
                "flags": out.tolist(),
            }
        )
        return out

    monkeypatch.setattr(worker_mod, "get_density_region_flags", _spy)

    csv_writer = _CapturingCSVWriter()
    worker = worker_mod.TrackingEngineCore(
        str(tmp_path / "video.mp4"),
        on_finished=lambda *_a: None,
        preview_mode=True,
        csv_writer_thread=csv_writer,
    )
    worker.set_parameters(
        _bgsub_arena_params(
            single_arena=single_arena, ENABLE_CONFIDENCE_DENSITY_MAP=True
        )
    )
    worker._density_regions = list(regions)
    worker.run_tracking()
    return calls


def _density_region_everywhere(arena):
    from hydra_suite.core.tracking.confidence.confidence_density import DensityRegion

    return DensityRegion(
        label="region-1",
        frame_start=0,
        frame_end=99,
        pixel_bbox=(0, 0, 1000, 1000),
        arena=arena,
    )


def test_worker_threads_meas_arena_into_the_density_gate(monkeypatch, tmp_path):
    """The density gate must see each detection's arena id.

    The bg-sub fixture's only detection sits in arena 1 (resized (40, 25),
    boundary at 25). A whole-frame region tagged `arena=0` must therefore NOT
    flag it, while the same region tagged `arena=1` must. Asserting on the
    derived FLAG (not on the config key, and not merely that a kwarg was
    passed) is what makes this fail if `meas_arena=` is dropped at the call
    site: without it the tag is ignored and the arena-0 region flags an
    arena-1 detection.
    """
    calls0 = _run_bgsub_density_case(
        monkeypatch, tmp_path, [_density_region_everywhere(0)]
    )
    assert calls0, "density gate never ran -- the test proves nothing"
    assert any(c["meas"] for c in calls0)
    assert all(c["meas_arena"] == [1] for c in calls0 if c["meas"]), calls0
    assert all(c["flags"] == [False] for c in calls0 if c["meas"]), calls0

    calls1 = _run_bgsub_density_case(
        monkeypatch, tmp_path, [_density_region_everywhere(1)]
    )
    assert any(c["flags"] == [True] for c in calls1 if c["meas"]), calls1


def test_single_arena_density_gate_gets_no_meas_arena(monkeypatch, tmp_path):
    """Single-arena inertness at the call site: `meas_arena` stays `None`, so
    the tag branch inside `get_density_region_flags` is never entered."""
    calls = _run_bgsub_density_case(
        monkeypatch, tmp_path, [_density_region_everywhere(None)], single_arena=True
    )
    assert calls, "density gate never ran -- the test proves nothing"
    assert all(c["meas_arena"] is None for c in calls), calls


class _StopAfterDensity(RuntimeError):
    """Sentinel raised from the probed density builder to end run_tracking()."""


class _CachedOBBRunner:
    """Valid, covering caches -> the forward pass takes the cached-replay
    branch, which is the one that computes the confidence density map."""

    def __init__(self, *_a, **_k):
        self.clipping_stats = _ClippingStatsStub()

    def caches_all_valid(self):
        return True

    def detection_cache_covers_range(self, *_a, **_k):
        return True

    def detection_cache_missing_frames(self, *_a, **_k):
        return []

    def run_batch_pass(self, *_a, **_k):
        raise AssertionError("run_batch_pass must NOT run when caches are valid")

    def load_frame(self, *_a, **_k):
        return None

    def close(self):
        pass


def _capture_density_builder_kwargs(monkeypatch, tmp_path, *, single_arena):
    """Drive the REAL cached-replay forward pass and capture the kwargs the
    worker hands to `compute_density_map_from_cache`."""
    import hydra_suite.core.tracking.confidence.confidence_density as cd_mod
    import hydra_suite.core.tracking.worker as worker_mod

    monkeypatch.setattr(worker_mod, "TrackingProfiler", _FakeProfiler)
    monkeypatch.setattr(worker_mod.cv2, "VideoCapture", _FakeVideoCapture)
    monkeypatch.setattr(worker_mod, "InferenceRunner", _CachedOBBRunner)
    monkeypatch.setattr(
        worker_mod, "build_density_cache_dict", lambda *_a, **_k: {0: None}
    )

    captured = {}

    def _probe(**kwargs):
        captured.update(kwargs)
        raise _StopAfterDensity("density builder reached")

    monkeypatch.setattr(cd_mod, "compute_density_map_from_cache", _probe)

    worker = worker_mod.TrackingEngineCore(
        str(tmp_path / "video.mp4"),
        on_finished=lambda *_a: None,
        preview_mode=True,
        use_cached_detections=True,
        csv_writer_thread=_CapturingCSVWriter(),
    )
    params = _bgsub_arena_params(
        single_arena=single_arena, ENABLE_CONFIDENCE_DENSITY_MAP=True
    )
    params["DETECTION_METHOD"] = "yolo_obb"
    params["TRACKING_WORKFLOW_MODE"] = "non_realtime"
    params["RESIZE_FACTOR"] = 1.0
    worker.set_parameters(params)
    worker.run_tracking()
    return worker, captured


def test_worker_passes_the_arena_layout_to_the_density_map_builder(
    monkeypatch, tmp_path
):
    """`compute_density_map_from_cache` must receive the layout, else regions
    are computed whole-frame no matter how good the per-arena code is.

    Asserts on the LAYOUT's derived properties (arena count and a label image
    that actually splits the frame), not merely that some kwarg was present.
    """
    worker, kwargs = _capture_density_builder_kwargs(
        monkeypatch, tmp_path, single_arena=False
    )
    layout = kwargs.get("arena_layout")
    assert layout is not None, f"arena_layout not forwarded; got {sorted(kwargs)}"
    assert layout.n_arenas == 2 and not layout.is_single_arena
    assert layout.label_image is not None
    # The layout really partitions the frame the way the config asked for.
    probes = np.array([[10.0, 50.0], [90.0, 50.0]], dtype=np.float32)
    np.testing.assert_array_equal(
        layout.arena_of_points(probes, frame_size=(100, 100)), [0, 1]
    )


def test_single_arena_density_builder_gets_an_inert_layout(monkeypatch, tmp_path):
    """A single-arena run must reach `compute_density_map_from_cache`'s ORIGINAL
    whole-frame branch: the layout it receives has to be one the dispatch
    rejects (single arena and/or no label image)."""
    _worker, kwargs = _capture_density_builder_kwargs(
        monkeypatch, tmp_path, single_arena=True
    )
    layout = kwargs.get("arena_layout")
    assert layout is not None
    assert layout.is_single_arena or layout.label_image is None


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

    Also asserts `arena_id` is present on EVERY captured row -- matched
    (frame 1's respawned track), unmatched (frame 1's still-lost track), AND
    the bulk "no detection this frame" rows (frame 2, all N tracks) -- the
    three separate `csv_writer_thread.enqueue` call sites in worker.py. A
    regression at any ONE of those sites produces a ragged row (one column
    short of the other two), caught by the uniform-length assertion below.
    """
    rows = _run_bgsub_arena_case(monkeypatch, tmp_path)
    assert len(rows) >= 4, f"expected rows from >=2 frames (2 tracks each), got {rows}"
    row_lengths = {len(r) for r in rows}
    assert len(row_lengths) == 1, (
        f"ragged CSV rows -- every row must have the same column count "
        f"regardless of which enqueue site wrote it; lengths seen: {row_lengths}"
    )
    matched = _matched_rows(rows)
    assert matched, "expected at least one row with a real detection"
    track_ids = {int(r[0]) for r in matched}
    assert track_ids == {1}, (
        f"expected the detection to initialize arena-1's slot (TrackID 1), "
        f"got TrackID(s) {track_ids} -- coordinate space or arena gating regressed"
    )
    # arena_id is the last column (n_arenas=2 > 1 -> column emitted) on EVERY
    # row, not just the matched ones.
    for r in rows:
        assert int(r[-1]) in (0, 1), f"row missing/invalid arena_id: {r}"


def test_single_arena_run_never_touches_arena_gating(monkeypatch, tmp_path):
    """With N_ARENAS=1, `arena_id` must not be emitted (single-arena byte-
    identity contract) and the sole track slot must be free to take the
    detection regardless of its position -- there is no arena to gate on."""
    rows = _run_bgsub_arena_case(monkeypatch, tmp_path, single_arena=True)
    assert len(rows) >= 2
    matched = _matched_rows(rows)
    assert matched, "expected at least one row with a real detection"
    assert all(int(r[0]) == 0 for r in matched)
    # Header-column contract: no arena_id column on ANY row (matched,
    # unmatched, or bulk no-detection) -- the row length for a single-arena
    # run must equal exactly the multi-arena row length minus one (the
    # arena_id column dropped), uniformly across every row, not just row 0.
    single_lengths = {len(r) for r in rows}
    assert len(single_lengths) == 1, f"ragged single-arena rows: {single_lengths}"
    multi_rows = _run_bgsub_arena_case(monkeypatch, tmp_path)
    multi_lengths = {len(r) for r in multi_rows}
    assert len(multi_lengths) == 1, f"ragged multi-arena rows: {multi_lengths}"
    assert single_lengths == {next(iter(multi_lengths)) - 1}


# ---------------------------------------------------------------------------
# Coordinate-space pin for the OTHER three `arena_of_points` call sites
# (cached YOLO OBB, cached bg-sub replay [backward], realtime YOLO OBB) --
# the end-to-end test above only exercises the realtime bg-sub site. These
# use a spy on `ArenaLayout.arena_of_points` that records BOTH the
# `frame_size` argument and the RETURNED arena id, so a wrong `frame_size`
# (e.g. reverted to `None`) is caught by the returned value diverging, not
# merely by an argument being present. This sidesteps needing
# Hungarian/respawn assignment to succeed (unlike the end-to-end test above),
# so it is deliberately independent of `MAX_DISTANCE_THRESHOLD` tuning.
#
# The mismatch trick: `ARENA_LABELS` is rasterized at 200x100 (the
# "configured" resolution `roi_shapes` were authored against), but the video
# actually being tracked is 100x100 (`_FakeVideoCapture`) -- HALF that width.
# `RESIZE_FACTOR` is irrelevant here (forced to 1.0 for every non-bg-sub
# method, and the cached paths don't even see a live frame), so this
# discriminates purely on whether the label image gets resized to the
# CURRENT frame's dimensions before lookup:
#   - correct:  200x100 label resized to 100x100 -> boundary shifts 100->50;
#               MISMATCH_XY=(60, 25) is right of 50 -> arena 1.
#   - wrong:    raw 200-wide label indexed directly at x=60 -> still left of
#               its own boundary at 100 -> arena 0. MISMATCH.
# ---------------------------------------------------------------------------

_LABEL_WIDTH = 200
_LABEL_HEIGHT = 100
_MISMATCH_DETECTION_XY = (60.0, 25.0)


def _spy_arena_of_points(monkeypatch):
    """Wrap `ArenaLayout.arena_of_points` to record (frame_size, xy, result)
    for every call, while still running the real implementation."""
    import hydra_suite.core.tracking.arenas as arenas_mod

    original = arenas_mod.ArenaLayout.arena_of_points
    calls: list[dict] = []

    def _wrapper(self, xy, frame_size=None):
        result = original(self, xy, frame_size=frame_size)
        calls.append(
            {
                "frame_size": frame_size,
                "xy": np.asarray(xy).tolist(),
                "result": np.asarray(result).tolist(),
            }
        )
        return result

    monkeypatch.setattr(arenas_mod.ArenaLayout, "arena_of_points", _wrapper)
    return calls


def _arena_mismatch_params(
    *, detection_method: str, workflow_mode: str = "non_realtime"
):
    """Engine params with `ARENA_LABELS` rasterized at 200x100 against a
    100x100 video -- see module-level comment above for why this
    discriminates the `frame_size=` wiring regardless of `RESIZE_FACTOR`."""
    cfg = {
        "frame_width": _LABEL_WIDTH,
        "frame_height": _LABEL_HEIGHT,
        "detection_method": detection_method,
        "animals_per_arena": 1,
        "roi_shapes": [
            {
                "type": "polygon",
                "mode": "include",
                "arena_id": 0,
                "params": [
                    [0, 0],
                    [_LABEL_WIDTH // 2, 0],
                    [_LABEL_WIDTH // 2, _LABEL_HEIGHT],
                    [0, _LABEL_HEIGHT],
                ],
            },
            {
                "type": "polygon",
                "mode": "include",
                "arena_id": 1,
                "params": [
                    [_LABEL_WIDTH // 2, 0],
                    [_LABEL_WIDTH, 0],
                    [_LABEL_WIDTH, _LABEL_HEIGHT],
                    [_LABEL_WIDTH // 2, _LABEL_HEIGHT],
                ],
            },
        ],
    }
    params = build_engine_params(cfg, runtime=_runtime(_LABEL_WIDTH, _LABEL_HEIGHT))
    params.update(
        {
            "START_FRAME": 0,
            "END_FRAME": 1,
            "TRACKING_WORKFLOW_MODE": workflow_mode,
            "MIN_DETECTIONS_TO_START": 1,
            "MIN_DETECTION_COUNTS": 2,
            "LOST_THRESHOLD_FRAMES": 1,
            "ENABLE_POSE_EXTRACTOR": False,
            "USE_APRILTAGS": False,
            "CNN_CLASSIFIERS": [],
            "ENABLE_CONFIDENCE_DENSITY_MAP": False,
            "ENABLE_FRAME_PREFETCH": False,
            "COMPUTE_RUNTIME": "cpu",
        }
    )
    return params


def _make_mismatch_frame_result(frame_idx: int):
    from hydra_suite.core.inference.result import FrameResult, OBBResult

    cx, cy = _MISMATCH_DETECTION_XY
    obb = OBBResult(
        frame_idx=frame_idx,
        centroids=np.array([[cx, cy]], dtype=np.float32),
        angles=np.array([0.0], dtype=np.float32),
        sizes=np.array([50.0], dtype=np.float32),
        shapes=np.array([[10.0, 1.2]], dtype=np.float32),
        confidences=np.array([0.9], dtype=np.float32),
        corners=np.zeros((1, 4, 2), dtype=np.float32),
        detection_ids=OBBResult.make_detection_ids(frame_idx, 1),
    )
    return FrameResult(
        frame_idx=frame_idx,
        obb=obb,
        filtered_indices=[0],
        headtail=None,
        cnn=[],
        pose=None,
        apriltag=None,
        resolved_headings=np.array([0.0], dtype=np.float32),
    )


class _FakeYoloRunner:
    """Serves both `inference_runner` (yolo_obb) and `bgsub_runner`
    (background_subtraction, via `.load_frame` only) call surfaces used by
    worker.py's cached-replay and realtime-YOLO dispatch branches."""

    def __init__(self, *_a, **_k):
        self.clipping_stats = _ClippingStatsStub()

    def caches_all_valid(self):
        return True

    def detection_cache_covers_range(self, *_a, **_k):
        return True

    def detection_cache_missing_frames(self, *_a, **_k):
        return []

    def run_batch_pass(self, *_a, **_k):
        pass

    def run_realtime(self, frame, frame_idx, roi_mask=None):
        return _make_mismatch_frame_result(frame_idx)

    def load_frame(self, frame_idx):
        return _make_mismatch_frame_result(frame_idx)

    def close(self):
        pass


def _make_traced_yolo_runner_cls():
    """A fresh `_FakeYoloRunner` subclass with its own `call_log` (a class
    attribute, so it survives across the single instance worker.py
    constructs) recording which dispatch methods actually fired. Existing to
    catch exactly the failure mode a review found: a test asserting on
    `arena_of_points` output without ever confirming which worker.py BRANCH
    produced it -- `run_realtime`/`load_frame`/`run_batch_pass` are
    dispatch-exclusive in every scenario these tests drive, so asserting the
    log names the intended site (and only that site) fired."""

    class _TracedYoloRunner(_FakeYoloRunner):
        call_log: list[str] = []

        def run_realtime(self, frame, frame_idx, roi_mask=None):
            self.call_log.append("run_realtime")
            return super().run_realtime(frame, frame_idx, roi_mask=roi_mask)

        def load_frame(self, frame_idx):
            self.call_log.append("load_frame")
            return super().load_frame(frame_idx)

        def run_batch_pass(self, *_a, **_k):
            self.call_log.append("run_batch_pass")
            return super().run_batch_pass(*_a, **_k)

    return _TracedYoloRunner


def _assert_mismatch_calls_resolve_arena_one(calls):
    """At least one recorded `arena_of_points` call must have used the
    CURRENT (100x100) frame size -- not `None`/the label image's own 200x100
    native shape -- to correctly resolve `_MISMATCH_DETECTION_XY` to arena 1.
    """
    hits = [c for c in calls if c["xy"] and len(c["xy"]) == 1]
    assert hits, f"arena_of_points was never called with a real detection: {calls}"
    resolved = [c["result"][0] for c in hits]
    assert 1 in resolved, (
        f"expected at least one call to resolve arena 1 (i.e. frame_size was "
        f"the CURRENT 100x100 frame, not None/the label's native 200x100); "
        f"got {calls}"
    )


def test_cached_yolo_replay_meas_arena_uses_current_frame_size(monkeypatch, tmp_path):
    """Backward-mode cached YOLO OBB replay (`inference_runner.load_frame`):
    `frame` is always `None` here, so this also pins the `frame is None`
    fallback (`target_w = cap.get(CAP_PROP_FRAME_WIDTH) * resize_f`) that
    only cached-detection modes exercise.
    """
    import hydra_suite.core.tracking.worker as worker_mod

    monkeypatch.setattr(worker_mod, "TrackingProfiler", _FakeProfiler)
    monkeypatch.setattr(worker_mod.cv2, "VideoCapture", _FakeVideoCapture)
    traced_cls = _make_traced_yolo_runner_cls()
    monkeypatch.setattr(worker_mod, "InferenceRunner", traced_cls)
    calls = _spy_arena_of_points(monkeypatch)

    captured = {}
    worker = worker_mod.TrackingEngineCore(
        str(tmp_path / "video.mp4"),
        on_finished=lambda success, _fps, _traj: captured.__setitem__(
            "success", success
        ),
        backward_mode=True,
        detection_cache_path=str(tmp_path / "fwd_cache"),
        preview_mode=True,
    )
    worker.set_parameters(_arena_mismatch_params(detection_method="yolo_obb"))
    worker.run_tracking()
    assert captured.get("success") is not False
    assert (
        "load_frame" in traced_cls.call_log
    ), f"expected the cached-replay dispatch (load_frame); call_log={traced_cls.call_log}"
    assert (
        "run_realtime" not in traced_cls.call_log
    ), f"backward-mode replay must never call run_realtime; call_log={traced_cls.call_log}"
    _assert_mismatch_calls_resolve_arena_one(calls)


def test_cached_bgsub_replay_meas_arena_uses_current_frame_size(monkeypatch, tmp_path):
    """Backward-mode cached bg-sub replay (`bgsub_runner.load_frame`) -- the
    SECOND path (besides realtime bg-sub) where `RESIZE_FACTOR != 1` is
    possible in real usage, and the one the brief specifically called out as
    unpinned. `frame` is always `None` here too (frame-is-None fallback)."""
    import hydra_suite.core.tracking.worker as worker_mod

    monkeypatch.setattr(worker_mod, "TrackingProfiler", _FakeProfiler)
    monkeypatch.setattr(worker_mod.cv2, "VideoCapture", _FakeVideoCapture)
    traced_cls = _make_traced_yolo_runner_cls()
    monkeypatch.setattr(worker_mod, "InferenceRunner", traced_cls)
    calls = _spy_arena_of_points(monkeypatch)

    captured = {}
    worker = worker_mod.TrackingEngineCore(
        str(tmp_path / "video.mp4"),
        on_finished=lambda success, _fps, _traj: captured.__setitem__(
            "success", success
        ),
        backward_mode=True,
        detection_cache_path=str(tmp_path / "fwd_cache"),
        preview_mode=True,
    )
    worker.set_parameters(
        _arena_mismatch_params(detection_method="background_subtraction")
    )
    worker.run_tracking()
    assert captured.get("success") is not False
    assert (
        "load_frame" in traced_cls.call_log
    ), f"expected the cached-replay dispatch (load_frame); call_log={traced_cls.call_log}"
    assert (
        "run_realtime" not in traced_cls.call_log
    ), f"backward-mode replay must never call run_realtime; call_log={traced_cls.call_log}"
    _assert_mismatch_calls_resolve_arena_one(calls)


def test_realtime_yolo_meas_arena_uses_current_frame_size(monkeypatch, tmp_path):
    """Forward realtime YOLO OBB (`inference_runner.run_realtime`) -- `frame`
    IS populated here (unlike the cached sites above).

    Regression note: an earlier version of this test set `preview_mode=True`
    and only `TRACKING_WORKFLOW_MODE="realtime"`. Neither is sufficient --
    `effective_realtime_tracking_mode` (worker.py) additionally requires
    `TRACKING_REALTIME_MODE=True` (the key `build_engine_params` ALWAYS
    emits, hardcoded `False`, so the workflow-mode fallback that reads it
    never fires once the key already exists in `p`) AND `not preview_mode`.
    Missing either silently routed the whole run through
    `run_batch_pass` + cached `load_frame` instead -- an exact duplicate of
    the cached-YOLO test above, never touching `run_realtime` at all, and
    the `arena_of_points` assertion below passed anyway (same underlying
    dispatch site, coincidentally). The `call_log` assertions make that
    silent-wrong-branch failure mode impossible to reintroduce.
    """
    import hydra_suite.core.tracking.worker as worker_mod

    monkeypatch.setattr(worker_mod, "TrackingProfiler", _FakeProfiler)
    monkeypatch.setattr(worker_mod.cv2, "VideoCapture", _FakeVideoCapture)
    traced_cls = _make_traced_yolo_runner_cls()
    monkeypatch.setattr(worker_mod, "InferenceRunner", traced_cls)
    calls = _spy_arena_of_points(monkeypatch)

    captured = {}
    worker = worker_mod.TrackingEngineCore(
        str(tmp_path / "video.mp4"),
        on_finished=lambda success, _fps, _traj: captured.__setitem__(
            "success", success
        ),
        preview_mode=False,
    )
    params = _arena_mismatch_params(
        detection_method="yolo_obb", workflow_mode="realtime"
    )
    params["TRACKING_REALTIME_MODE"] = True
    worker.set_parameters(params)
    worker.run_tracking()
    assert captured.get("success") is not False
    assert (
        "run_realtime" in traced_cls.call_log
    ), f"expected the realtime dispatch (run_realtime); call_log={traced_cls.call_log}"
    assert not {"load_frame", "run_batch_pass"} & set(traced_cls.call_log), (
        f"expected a pure realtime run (no cached/batch dispatch); "
        f"call_log={traced_cls.call_log}"
    )
    _assert_mismatch_calls_resolve_arena_one(calls)


# ---------------------------------------------------------------------------
# ArenaDecoderRegistry wiring: multi-arena runs must construct the registry
# (one OnlineIdentityDecoder per arena, uniqueness scoped per arena);
# single-arena runs must construct the literal bare OnlineIdentityDecoder
# (not a one-decoder registry), so single-arena byte-identity stays
# structural. Without this wiring the registry is dead code and multi-arena
# identity decoding degrades to one global uniqueness constraint across every
# arena -- the exact failure the registry exists to prevent.
# ---------------------------------------------------------------------------


def _identity_decoder_wiring_params(*, n_arenas: int):
    """Engine params that make `individual_pipeline_enabled` and a non-empty
    identity catalog both true, via `TAG_IDENTITY_LABELS` (no CNN model file
    needed to resolve a catalog)."""
    cfg = {
        "frame_width": 100,
        "frame_height": 100,
        "detection_method": "yolo_obb",
        "animals_per_arena": 2,
    }
    if n_arenas > 1:
        cfg["roi_shapes"] = [
            {
                "type": "polygon",
                "mode": "include",
                "arena_id": arena_id,
                "params": [[0, 0], [100, 0], [100, 100], [0, 100]],
            }
            for arena_id in range(n_arenas)
        ]
    params = build_engine_params(cfg, runtime=_runtime(100, 100))
    params.update(
        {
            "START_FRAME": 0,
            "END_FRAME": 1,
            "TRACKING_WORKFLOW_MODE": "realtime",
            "ENABLE_INDIVIDUAL_PIPELINE": True,
            "TAG_IDENTITY_LABELS": ["antA", "antB"],
            "CNN_CLASSIFIERS": [],
            "USE_APRILTAGS": False,
            "ENABLE_POSE_EXTRACTOR": False,
            "MIN_DETECTIONS_TO_START": 1,
            "MIN_DETECTION_COUNTS": 2,
            "LOST_THRESHOLD_FRAMES": 1,
            "ENABLE_CONFIDENCE_DENSITY_MAP": False,
            "ENABLE_FRAME_PREFETCH": False,
            "COMPUTE_RUNTIME": "cpu",
        }
    )
    return params


def _run_identity_decoder_wiring_case(monkeypatch, tmp_path, *, n_arenas: int):
    import hydra_suite.core.individual.identity.online as online_mod
    import hydra_suite.core.tracking.identity.decoder_registry as registry_mod
    import hydra_suite.core.tracking.worker as worker_mod

    constructed = {"online": 0, "registry": 0, "registry_slot_arena": None}

    orig_online_init = online_mod.OnlineIdentityDecoder.__init__

    def _spy_online_init(self, *a, **k):
        constructed["online"] += 1
        return orig_online_init(self, *a, **k)

    orig_registry_init = registry_mod.ArenaDecoderRegistry.__init__

    def _spy_registry_init(self, catalog, params, slot_arena, *a, **k):
        constructed["registry"] += 1
        constructed["registry_slot_arena"] = np.asarray(slot_arena).tolist()
        return orig_registry_init(self, catalog, params, slot_arena, *a, **k)

    monkeypatch.setattr(online_mod.OnlineIdentityDecoder, "__init__", _spy_online_init)
    monkeypatch.setattr(
        registry_mod.ArenaDecoderRegistry, "__init__", _spy_registry_init
    )
    monkeypatch.setattr(worker_mod, "TrackingProfiler", _FakeProfiler)
    monkeypatch.setattr(worker_mod.cv2, "VideoCapture", _FakeVideoCapture)
    monkeypatch.setattr(worker_mod, "InferenceRunner", _FakeYoloRunner)

    captured = {}
    worker = worker_mod.TrackingEngineCore(
        str(tmp_path / "video.mp4"),
        on_finished=lambda success, _fps, _traj: captured.__setitem__(
            "success", success
        ),
        preview_mode=True,
    )
    worker.set_parameters(_identity_decoder_wiring_params(n_arenas=n_arenas))
    worker.run_tracking()
    assert captured.get("success") is not False
    return constructed


def test_multi_arena_run_constructs_arena_decoder_registry(monkeypatch, tmp_path):
    constructed = _run_identity_decoder_wiring_case(monkeypatch, tmp_path, n_arenas=3)
    assert constructed["registry"] == 1, "expected exactly one ArenaDecoderRegistry"
    # `ArenaDecoderRegistry.__init__` itself constructs one bare
    # `OnlineIdentityDecoder` per arena internally (that IS the mechanism --
    # per-arena uniqueness), so `online` > 0 here is expected and correct.
    # 3 arenas -> exactly 3 internal decoders, one per arena.
    assert constructed["online"] == 3, (
        f"expected the registry to construct one internal decoder per arena "
        f"(3), got {constructed['online']}"
    )
    # animals_per_arena=2, 3 arenas -> slot_arena = [0,0,1,1,2,2].
    assert constructed["registry_slot_arena"] == [0, 0, 1, 1, 2, 2]


def test_single_arena_run_constructs_bare_online_decoder(monkeypatch, tmp_path):
    constructed = _run_identity_decoder_wiring_case(monkeypatch, tmp_path, n_arenas=1)
    assert constructed["online"] == 1, "expected exactly one bare OnlineIdentityDecoder"
    assert constructed["registry"] == 0, (
        "single-arena must keep the literal bare decoder object (not a "
        "one-decoder registry), so single-arena byte-identity stays "
        "structural, not arithmetic"
    )


# ---------------------------------------------------------------------------
# ArenaDecoderRegistry method routing, built the way worker.py builds it.
#
# `get_belief`/`clear_slot`/`decay_absent_slot_beliefs` are called from
# worker.py ONLY inside the `try:` block at the per-frame identity-decoder
# update site (guarded by a bare `except Exception: logger.debug(...)`) --
# so a routing bug in any of the three would degrade the identity path
# silently, with nothing reported. Driving all three through a genuine
# run_tracking() pass would require synthesizing real per-detection identity
# EVIDENCE (tag or CNN posteriors) across several frames with a specific
# track-state sequence (commit -> respawn -> still-absent-while-committed)
# to reach every method naturally -- disproportionate for this follow-up.
# Per the review's explicit fallback, this instead builds the registry via
# the EXACT construction worker.py:1886-1892 uses (the real
# `resolve_catalog_spec` -> `IdentityCatalog.from_spec` catalog, and a real
# `ArenaLayout.slot_arena`, not the simplified `IdentityCatalog.from_labels`
# + hand-rolled array the registry's own unit tests
# (`tests/test_arena_identity_decoders.py`) already use) and calls each
# method directly, asserting routing to the decoder OWNING the slot.
# ---------------------------------------------------------------------------


def test_registry_methods_route_correctly_when_built_the_way_worker_builds_it():
    from hydra_suite.core.individual.identity.catalog import IdentityCatalog
    from hydra_suite.core.individual.identity.resolve import resolve_catalog_spec
    from hydra_suite.core.tracking.arenas import ArenaLayout
    from hydra_suite.core.tracking.identity.decoder_registry import ArenaDecoderRegistry

    # Exactly worker.py's catalog construction (resolve_catalog_spec ->
    # IdentityCatalog.from_spec), driven off TAG_IDENTITY_LABELS the same way
    # `_identity_decoder_wiring_params` above does.
    catalog_spec = resolve_catalog_spec([], ["antA", "antB"])
    catalog = IdentityCatalog.from_spec(catalog_spec)
    # Exactly worker.py's slot_arena source: ArenaLayout.slot_arena, not a
    # hand-rolled array. 2 arenas x 2 animals -> slot_arena = [0, 0, 1, 1].
    layout = ArenaLayout(n_arenas=2, animals_per_arena=2)
    params = {"IDENTITY_ONLINE_COMMIT_THRESHOLD": 0.9}
    reg = ArenaDecoderRegistry(catalog, params, layout.slot_arena)
    dec0, dec1 = reg.decoders[0], reg.decoders[1]
    assert dec0 is not dec1

    # get_belief: slot 1 belongs to arena 0 (dec0); slot 2 belongs to arena 1
    # (dec1). Stamp a sentinel belief directly on ONE decoder's internal
    # state and confirm the registry reads it back only through the correct
    # decoder.
    from hydra_suite.core.individual.identity.online import TrackIdentityBelief

    sentinel_belief = TrackIdentityBelief(
        slot_index=1,
        log_posterior=np.zeros(catalog.size, dtype=np.float64),
    )
    # `_beliefs` is OnlineIdentityDecoder's private internal state -- poked
    # directly here (rather than routed through the real evidence-fusion
    # pipeline) is exactly what makes this a routing test: it isolates
    # ArenaDecoderRegistry's slot->decoder dispatch from decoder-internal
    # belief-update logic (already covered elsewhere).
    dec0._beliefs[1] = sentinel_belief
    assert reg.get_belief(1) is sentinel_belief
    assert reg.get_belief(2) is None, "slot 2 (arena 1) must not see arena 0's belief"

    # clear_slot: must clear the OWNING decoder's belief, not the other
    # arena's.
    dec1._beliefs[2] = TrackIdentityBelief(
        slot_index=2,
        log_posterior=np.zeros(catalog.size, dtype=np.float64),
    )
    reg.clear_slot(2, reason="respawn at frame 5", respawn_frame_idx=5)
    assert 2 not in dec1._beliefs, "clear_slot must remove the owning decoder's belief"
    assert 1 in dec0._beliefs, "clear_slot(2, ...) must not touch arena 0's slot 1"

    # decay_absent_slot_beliefs: mixed-arena input must be partitioned and
    # each sub-list routed to its own decoder (not one decoder seeing both
    # arenas' slots).
    seen: dict[int, list[int]] = {}

    def _make_decay_spy(arena_id):
        def _spy(slots):
            seen[arena_id] = list(slots)

        return _spy

    dec0.decay_absent_slot_beliefs = _make_decay_spy(0)
    dec1.decay_absent_slot_beliefs = _make_decay_spy(1)
    reg.decay_absent_slot_beliefs([0, 1, 2, 3])
    assert seen == {0: [0, 1], 1: [2, 3]}, (
        f"decay_absent_slot_beliefs must partition by arena and route each "
        f"partition to its owning decoder; got {seen}"
    )
