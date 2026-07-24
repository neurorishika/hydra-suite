import types

import numpy as np
import pytest
import torch

from hydra_suite.core.inference.config import OBBConfig, OBBDirectConfig, SliceConfig
from hydra_suite.core.inference.stages.obb import OBBModels, run_obb
from hydra_suite.core.inference.stages.slicing import (
    get_slice_bboxes,
    plan_slices,
    run_direct_sliced,
    tiles_overlap,
)


def test_grid_covers_frame_and_flushes_to_edge():
    # 1000x1000 frame, 640 tiles, 0.2 overlap -> step 512.
    boxes = get_slice_bboxes(1000, 1000, 640, 640, 0.2, 0.2)
    # every pixel covered: min corner at 0, max corner at frame edge.
    xs0 = sorted({b[0] for b in boxes})
    assert xs0[0] == 0
    # last tile flush to right edge (no runt): some tile ends exactly at 1000.
    assert max(b[2] for b in boxes) == 1000
    assert max(b[3] for b in boxes) == 1000
    # no tile exceeds the frame.
    assert all(b[2] <= 1000 and b[3] <= 1000 for b in boxes)
    # tile size preserved (last tile shifted back, not shrunk).
    assert all((b[2] - b[0]) == 640 and (b[3] - b[1]) == 640 for b in boxes)


def test_zero_overlap_tiles_are_disjoint_step_equals_size():
    boxes = get_slice_bboxes(1280, 1280, 640, 640, 0.0, 0.0)
    assert len(boxes) == 4
    assert (0, 0, 640, 640) in boxes


def test_auto_model_uses_imgsz():
    plan = plan_slices(
        (2000, 2000),
        SliceConfig(enabled=True, geometry_mode="auto_model"),
        imgsz=1024,
        roi_mask=None,
    )
    assert plan.slice_wh == (1024, 1024)


def test_custom_uses_explicit_size():
    cfg = SliceConfig(
        enabled=True, geometry_mode="custom", slice_height=512, slice_width=768
    )
    plan = plan_slices((2000, 2000), cfg, imgsz=1024, roi_mask=None)
    assert plan.slice_wh == (768, 512)  # (w, h)


def test_auto_object_sizes_from_reference():
    # ref object 64px, want it to span 0.1 of the tile -> tile ~640px.
    cfg = SliceConfig(
        enabled=True, geometry_mode="auto_object", object_tile_fraction=0.1
    )
    plan = plan_slices((4000, 4000), cfg, imgsz=1024, roi_mask=None, ref_object_px=64.0)
    assert 600 <= plan.slice_wh[0] <= 680


def test_roi_gating_drops_tiles_outside_mask():
    # ROI only in the top-left quadrant.
    mask = np.zeros((1000, 1000), dtype=np.uint8)
    mask[:400, :400] = 1
    cfg = SliceConfig(
        enabled=True, geometry_mode="custom", slice_height=256, slice_width=256
    )
    full = plan_slices(
        (1000, 1000),
        SliceConfig(
            enabled=True, geometry_mode="custom", slice_height=256, slice_width=256
        ),
        imgsz=256,
        roi_mask=None,
    )
    gated = plan_slices((1000, 1000), cfg, imgsz=256, roi_mask=mask)
    assert len(gated.tiles) < len(full.tiles)
    # every kept tile intersects the ROI.
    for x0, y0, x1, y1 in gated.tiles:
        assert mask[y0:y1, x0:x1].any()


def test_perform_standard_pred_flag():
    cfg = SliceConfig(
        enabled=True,
        geometry_mode="custom",
        slice_height=256,
        slice_width=256,
        perform_standard_pred=True,
    )
    plan = plan_slices((512, 512), cfg, imgsz=256, roi_mask=None)
    assert plan.full_frame is True


def test_roi_eliminates_every_tile_falls_back_to_full_grid():
    """When ROI mask would drop every tile, fallback to full grid."""
    # All-zero ROI mask eliminates all tiles.
    mask = np.zeros((1000, 1000), dtype=np.uint8)
    cfg = SliceConfig(
        enabled=True, geometry_mode="custom", slice_height=256, slice_width=256
    )

    # Plan with no ROI.
    full = plan_slices((1000, 1000), cfg, imgsz=256, roi_mask=None)

    # Plan with all-zero ROI (eliminates everything).
    gated = plan_slices((1000, 1000), cfg, imgsz=256, roi_mask=mask)

    # Should fall back to full grid: same tiles, same length.
    assert len(gated.tiles) == len(full.tiles)
    assert gated.tiles == full.tiles


def test_tile_larger_than_frame():
    """Tile size larger than frame produces single tile without exceeding bounds."""
    cfg = SliceConfig(
        enabled=True, geometry_mode="custom", slice_height=2000, slice_width=2000
    )
    plan = plan_slices((500, 500), cfg, imgsz=2000, roi_mask=None)

    # Should produce exactly one tile.
    assert len(plan.tiles) == 1
    x0, y0, x1, y1 = plan.tiles[0]

    # Tile should not exceed frame bounds.
    assert x0 >= 0 and y0 >= 0 and x1 <= 500 and y1 <= 500
    # Tile size is clamped to frame size.
    assert (x1 - x0) == 500 and (y1 - y0) == 500


def test_auto_object_with_zero_ref_object_px_falls_back():
    """auto_object with zero ref_object_px falls back to auto_model sizing."""
    cfg = SliceConfig(
        enabled=True, geometry_mode="auto_object", object_tile_fraction=0.1
    )
    plan = plan_slices((4000, 4000), cfg, imgsz=1024, roi_mask=None, ref_object_px=0.0)
    # Should fall back to auto_model: tile size equals imgsz.
    assert plan.slice_wh == (1024, 1024)


def test_auto_object_with_negative_ref_object_px_falls_back():
    """auto_object with negative ref_object_px falls back to auto_model sizing."""
    cfg = SliceConfig(
        enabled=True, geometry_mode="auto_object", object_tile_fraction=0.1
    )
    plan = plan_slices(
        (4000, 4000), cfg, imgsz=1024, roi_mask=None, ref_object_px=-10.0
    )
    # Should fall back to auto_model: tile size equals imgsz.
    assert plan.slice_wh == (1024, 1024)


def test_overlap_approaching_one():
    """Extreme overlap (0.99) drives step toward zero; max(1, step) floor engages.

    For slice=64, overlap=0.99: int(64 * (1 - 0.99)) = int(0.64) = 0.
    Without the max(1, ...) guard, range(0, N, 0) would raise ValueError.
    With the guard, step=max(1, 0)=1.

    Asserted on the ``get_slice_bboxes`` primitive, not ``plan_slices``: the
    latter now refuses a pathological tile count outright (finding I5, see
    ``test_pathological_tile_count_raises_instead_of_hanging``), while the
    primitive stays a pure, unguarded geometry function.
    """
    tiles = get_slice_bboxes(160, 160, 64, 64, 0.99, 0.99)

    # Should not crash (max(1, ...) prevents ValueError from range(0, N, 0)).
    assert len(tiles) > 0
    # With step=1 (floor engaged), tile count = (160-64+1)^2.
    assert len(tiles) == (160 - 64 + 1) ** 2

    # Every tile should be full size (not shrunk).
    for x0, y0, x1, y1 in tiles:
        assert (x1 - x0) == 64 and (y1 - y0) == 64


def test_asymmetric_frame_and_overlap():
    """Non-square frame with different height/width overlap ratios tiles correctly."""
    cfg = SliceConfig(
        enabled=True,
        geometry_mode="custom",
        slice_height=256,
        slice_width=384,
        overlap_height_ratio=0.3,
        overlap_width_ratio=0.5,
    )
    plan = plan_slices((900, 1600), cfg, imgsz=384, roi_mask=None)

    # Should cover the entire frame: some tile reaches right edge, some reaches bottom.
    xs = sorted({b[0] for b in plan.tiles})
    ys = sorted({b[1] for b in plan.tiles})
    assert xs[0] == 0
    assert ys[0] == 0
    assert max(b[2] for b in plan.tiles) == 1600  # right edge
    assert max(b[3] for b in plan.tiles) == 900  # bottom edge

    # Every tile should be full configured size (not shrunk).
    for x0, y0, x1, y1 in plan.tiles:
        assert (x1 - x0) == 384 and (y1 - y0) == 256


class _FakeRuntime:
    device = "cpu"
    tensor_on_cuda = False


class _FakeOBB:
    def __init__(self, cx, cy, w, h):
        self._xywhr = np.array([[cx, cy, w, h, 0.0]], np.float32)
        self._conf = np.array([0.9], np.float32)

    def __len__(self):
        return 1

    @property
    def xywhr(self):
        import torch

        return torch.from_numpy(self._xywhr)

    @property
    def conf(self):
        import torch

        return torch.from_numpy(self._conf)

    @property
    def cls(self):
        import torch

        return torch.zeros(1)

    @property
    def xyxyxyxy(self):
        # Axis-aligned corners (angle is always 0 in this fake): TL,TR,BR,BL.
        cx, cy, w, h = (
            self._xywhr[0, 0],
            self._xywhr[0, 1],
            self._xywhr[0, 2],
            self._xywhr[0, 3],
        )
        hw, hh = w / 2.0, h / 2.0
        corners = np.array(
            [
                [
                    [cx - hw, cy - hh],
                    [cx + hw, cy - hh],
                    [cx + hw, cy + hh],
                    [cx - hw, cy + hh],
                ]
            ],
            np.float32,
        )
        return torch.from_numpy(corners)


class _FakeYOLO:
    """Stub: returns one obb detection at a fixed frame-space point per tile that
    covers (200,200). imgsz reported = 256 so slice_size==imgsz exact path."""

    imgsz = 256
    overrides = {"imgsz": 256}

    def predict(self, source, **kw):
        # `source` is a list of tile images; emit a detection only when the tile
        # is the one containing (200,200) -- detect straddle via image content.
        results = []
        for img in source:
            r = types.SimpleNamespace()
            # tile is 256x256; put a detection at local (60,60) always.
            r.obb = _FakeOBB(cx=60, cy=60, w=30, h=30)
            results.append(r)
        return results


class _FakeOBBN:
    """Fake ultralytics ``.obb`` result carrying an arbitrary (possibly zero)
    number of xywhr rows, all at a fixed angle of 0."""

    def __init__(self, dets):
        n = len(dets)
        if n:
            arr = np.array([[cx, cy, w, h, 0.0] for cx, cy, w, h in dets], np.float32)
        else:
            arr = np.zeros((0, 5), np.float32)
        self._xywhr = arr
        self._conf = np.full(n, 0.9, np.float32)
        self._n = n

    def __len__(self):
        return self._n

    @property
    def xywhr(self):
        import torch

        return torch.from_numpy(self._xywhr)

    @property
    def conf(self):
        import torch

        return torch.from_numpy(self._conf)

    @property
    def cls(self):
        import torch

        return torch.zeros(self._n)

    @property
    def xyxyxyxy(self):
        # Axis-aligned corners (angle is always 0 in this fake): TL,TR,BR,BL.
        if self._n == 0:
            return torch.zeros((0, 4, 2), dtype=torch.float32)
        cx, cy, w, h = (
            self._xywhr[:, 0],
            self._xywhr[:, 1],
            self._xywhr[:, 2],
            self._xywhr[:, 3],
        )
        hw, hh = w / 2.0, h / 2.0
        corners = np.stack(
            [
                np.stack([cx - hw, cy - hh], axis=1),
                np.stack([cx + hw, cy - hh], axis=1),
                np.stack([cx + hw, cy + hh], axis=1),
                np.stack([cx - hw, cy + hh], axis=1),
            ],
            axis=1,
        ).astype(np.float32)
        return torch.from_numpy(corners)


class _FakeYOLOGlobalPoint:
    """Stub whose detections are keyed on tile GEOMETRY, not local coordinates.

    Given the flattened tile job list, it emits a detection at a single fixed
    GLOBAL point -- expressed in each tile's own local coordinates -- for
    every tile whose bounding box actually contains that point, and an empty
    result for every tile that doesn't. Two tiles that genuinely overlap the
    global point therefore both "see" the same real object, which
    ``_FakeYOLO`` (fixed local (60,60) in every tile) can never simulate.

    Relies on ``predict()`` being called once per frame with tiles in the same
    order as ``plan.tiles`` (true here: single frame, no full-frame job).
    """

    imgsz = 256
    overrides = {"imgsz": 256}

    def __init__(self, tiles, global_point, size=(20.0, 20.0)):
        self.tiles = tiles
        self.gx, self.gy = global_point
        self.w, self.h = size
        self._call = 0

    def predict(self, source, **kw):
        results = []
        for _ in source:
            x0, y0, x1, y1 = self.tiles[self._call % len(self.tiles)]
            self._call += 1
            r = types.SimpleNamespace()
            if x0 <= self.gx < x1 and y0 <= self.gy < y1:
                r.obb = _FakeOBBN([(self.gx - x0, self.gy - y0, self.w, self.h)])
            else:
                r.obb = _FakeOBBN([])
            results.append(r)
        return results


class _FakeYOLOEmpty:
    """Stub: every tile yields zero detections."""

    imgsz = 256
    overrides = {"imgsz": 256}

    def predict(self, source, **kw):
        results = []
        for _ in source:
            r = types.SimpleNamespace()
            r.obb = _FakeOBBN([])
            results.append(r)
        return results


class _FakeYOLOFirstTileOnly:
    """Stub: only the first tile job yields a detection; the rest are empty."""

    imgsz = 256
    overrides = {"imgsz": 256}

    def predict(self, source, **kw):
        results = []
        for i, _ in enumerate(source):
            r = types.SimpleNamespace()
            if i == 0:
                r.obb = _FakeOBBN([(60.0, 60.0, 30.0, 30.0)])
            else:
                r.obb = _FakeOBBN([])
            results.append(r)
        return results


def _direct_cfg(enabled, **slice_kw):
    return OBBConfig(
        mode="direct",
        direct=OBBDirectConfig(
            model_path="m.pt",
            model_task="obb",
            slice=SliceConfig(enabled=enabled, geometry_mode="auto_model", **slice_kw),
        ),
        confidence_threshold=0.0,
        raw_detection_cap=0,
        max_detections=100,
    )


def test_sliced_cpu_obb_remaps_into_frame_space():
    # Asymmetric frame (h=300, w=500) so tiling differs on each axis: an X/Y-swap
    # bug in `_offset_result` would produce a different (and thus detectable)
    # result instead of silently passing on a symmetric grid.
    frame = np.zeros((300, 500, 3), np.uint8)
    cfg = _direct_cfg(True, overlap_height_ratio=0.0, overlap_width_ratio=0.0)
    out = run_direct_sliced([frame], _FakeYOLO(), cfg, _FakeRuntime())
    assert len(out) == 1
    res = out[0]

    # Independently derive the expected tile grid + exact global centroids:
    # get_slice_bboxes(frame_h=300, frame_w=500, slice=256, overlap=0) ->
    # xs=[0, 244], ys=[0, 44] (edge-flush last tile on each axis) -> 4 tiles,
    # none of which overlap each other (256px tiles, steps of 244/244 leave the
    # exclusive interiors of each tile disjoint at the (60, 60) detection point),
    # so no merge collapses these -- one detection per tile is expected.
    tiles = get_slice_bboxes(300, 500, 256, 256, 0.0, 0.0)
    assert tiles == [
        (0, 0, 256, 256),
        (244, 0, 500, 256),
        (0, 44, 256, 300),
        (244, 44, 500, 300),
    ]
    expected = sorted((x0 + 60, y0 + 60) for x0, y0, _, _ in tiles)

    # Exact detection count: one per tile, no drops/dupes from a job/result desync.
    assert res.num_detections == len(tiles)
    actual = sorted(map(tuple, res.centroids.tolist()))
    for (ex, ey), (ax, ay) in zip(expected, actual):
        assert ax == pytest.approx(ex, abs=1e-3)
        assert ay == pytest.approx(ey, abs=1e-3)


def test_sliced_cpu_obb_merges_cross_tile_duplicate_at_zero_configured_overlap():
    """Finding 1/2 regression test: at configured overlap 0.0, edge-flush still
    produces REAL geometric overlap between the last two tiles on each axis, so
    an object detected independently by both tiles must collapse to ONE
    detection in the merged frame result -- not silently double-count it.

    Frame 300x300, tile 256x256, overlap 0.0 -> tiles (via get_slice_bboxes):
      (0,0,256,256), (44,0,300,256), (0,44,256,300), (44,44,300,300)
    Global point (250, 20) lies inside BOTH tile0 [0,256)x[0,256) and tile1
    [44,300)x[0,256) -- a genuine 212px cross-tile overlap in x, at y=20 which
    is outside the y-overlap band, isolating this to an x-axis duplicate.
    """
    frame = np.zeros((300, 300, 3), np.uint8)
    cfg = _direct_cfg(True, overlap_height_ratio=0.0, overlap_width_ratio=0.0)
    tiles = get_slice_bboxes(300, 300, 256, 256, 0.0, 0.0)
    assert tiles == [
        (0, 0, 256, 256),
        (44, 0, 300, 256),
        (0, 44, 256, 300),
        (44, 44, 300, 300),
    ]
    global_point = (250.0, 20.0)
    model = _FakeYOLOGlobalPoint(tiles, global_point)

    out = run_direct_sliced([frame], model, cfg, _FakeRuntime())
    res = out[0]

    # Collapsed to exactly one detection at (approximately) the shared global point.
    assert res.num_detections == 1
    assert res.centroids[0, 0] == pytest.approx(global_point[0], abs=1.0)
    assert res.centroids[0, 1] == pytest.approx(global_point[1], abs=1.0)


def test_sliced_cpu_obb_one_tile_empty_others_not():
    """Finding 4: a tile with zero detections must not crash the merge/concat
    path, and must not contribute any spurious detections."""
    frame = np.zeros((300, 300, 3), np.uint8)
    cfg = _direct_cfg(True, overlap_height_ratio=0.0, overlap_width_ratio=0.0)
    out = run_direct_sliced([frame], _FakeYOLOFirstTileOnly(), cfg, _FakeRuntime())
    assert len(out) == 1
    res = out[0]
    assert res.num_detections == 1
    assert res.centroids[0, 0] == pytest.approx(60.0, abs=1e-3)
    assert res.centroids[0, 1] == pytest.approx(60.0, abs=1e-3)


def test_sliced_cpu_obb_all_tiles_empty_yields_empty_result():
    """Finding 4: every tile yielding zero detections must produce a valid,
    empty OBBResult (num_detections == 0), not a crash."""
    frame = np.zeros((300, 300, 3), np.uint8)
    cfg = _direct_cfg(True, overlap_height_ratio=0.0, overlap_width_ratio=0.0)
    out = run_direct_sliced([frame], _FakeYOLOEmpty(), cfg, _FakeRuntime())
    assert len(out) == 1
    res = out[0]
    assert res.num_detections == 0
    assert res.centroids.shape == (0, 2)
    assert res.corners.shape[0] == 0


def test_enabled_false_dispatch_uses_plain_run_direct(monkeypatch):
    frame = np.zeros((300, 300, 3), np.uint8)
    cfg = _direct_cfg(False)
    models = OBBModels(mode="direct", direct_model=_FakeYOLO())
    called = {"sliced": False}
    monkeypatch.setattr(
        "hydra_suite.core.inference.stages.slicing.run_direct_sliced",
        lambda *a, **k: called.__setitem__("sliced", True) or [],
    )
    run_obb([frame], models, cfg, _FakeRuntime())
    assert called["sliced"] is False  # disabled -> never dispatched


def test_disabled_slice_is_identical_to_plain_run_direct():
    """enabled=False must be byte-identical to `_run_direct` (structural bypass):
    `run_obb` dispatches straight to `_run_direct` when slicing is off, so the
    disabled path must never diverge from the pre-feature pipeline."""
    from hydra_suite.core.inference.stages.obb import OBBModels, _run_direct, run_obb

    frame = np.zeros((300, 300, 3), np.uint8)
    model = _FakeYOLO()
    models = OBBModels(mode="direct", direct_model=model)
    cfg_off = _direct_cfg(False)

    got = run_obb([frame], models, cfg_off, _FakeRuntime())
    expected = _run_direct([frame], model, cfg_off, _FakeRuntime())
    assert len(got) == len(expected)
    for g, e in zip(got, expected):
        assert g.num_detections == e.num_detections
        np.testing.assert_array_equal(g.centroids, e.centroids)
        np.testing.assert_array_equal(g.corners, e.corners)
        np.testing.assert_array_equal(g.confidences, e.confidences)
        np.testing.assert_array_equal(g.angles, e.angles)
        np.testing.assert_array_equal(g.sizes, e.sizes)


def test_enabled_true_dispatches_to_sliced(monkeypatch):
    frame = np.zeros((300, 300, 3), np.uint8)
    cfg = _direct_cfg(True)
    models = OBBModels(mode="direct", direct_model=_FakeYOLO())
    marker = object()
    monkeypatch.setattr(
        "hydra_suite.core.inference.stages.slicing.run_direct_sliced",
        lambda *a, **k: [marker],
    )
    out = run_obb([frame], models, cfg, _FakeRuntime())
    assert out == [marker]


# --- Task 7: tiles_overlap geometry predicate ---------------------------------


def test_tiles_overlap_true_for_edge_flush_case_at_zero_configured_ratio():
    """The exact bug example from the task brief: a 300px frame with 256px
    tiles at a CONFIGURED ratio of 0.0 still edge-flushes the last tile to
    the frame border, producing 212px of REAL overlap. ``tiles_overlap`` must
    report True here even though the config ratio is 0 -- this is precisely
    why callers must gate on tile geometry, never on the config ratio."""
    tiles = get_slice_bboxes(300, 300, 256, 256, 0.0, 0.0)
    assert tiles == [
        (0, 0, 256, 256),
        (44, 0, 300, 256),
        (0, 44, 256, 300),
        (44, 44, 300, 300),
    ]
    assert tiles_overlap(tiles) is True


def test_tiles_overlap_false_for_grid_that_divides_evenly():
    """A 512px frame with 256px tiles at ratio 0.0 divides evenly (2x2 grid,
    no edge-flush needed) -- tiles genuinely do not overlap."""
    tiles = get_slice_bboxes(512, 512, 256, 256, 0.0, 0.0)
    assert tiles == [
        (0, 0, 256, 256),
        (256, 0, 512, 256),
        (0, 256, 256, 512),
        (256, 256, 512, 512),
    ]
    assert tiles_overlap(tiles) is False


# --- Task 7: native-cuda sliced path (_RawOBBTensors preservation) -------------
#
# NOTE on the fixture sizes below: the task brief's own draft tests reused the
# 300x300/256px-tile geometry (the SAME frame/tile size used above to
# demonstrate the edge-flush overlap bug) for BOTH a "no cross-tile overlap"
# scenario and a "real overlap" scenario, varying only the CONFIGURED overlap
# ratio (0.0 vs 0.2). But as proven above, that frame/tile combination
# produces REAL tile overlap at BOTH ratios (edge-flush forces it regardless
# of the ratio) -- so a test asserting "ratio 0.0 -> _RawOBBTensors preserved"
# on that geometry would be asserting something arithmetically false, exactly
# the numeric-fixture bug this task explicitly warns about. The two cases
# below instead use genuinely different tile geometries: a 512x512 frame
# (divides evenly, tiles_overlap literally False) for the "no dedup needed"
# path, and the 300x300 frame (genuine tile overlap) WITH a fake model that
# emits an actual duplicate detection at a shared point for the "merge
# required" path.


class _FakeCudaRuntime:
    """Extraction yields device tensors (``tensor_on_cuda``), device faked to cpu.

    This is the tier ``gpu`` runtime shape: ``tensor_on_cuda=True`` while frames
    are plain numpy (NVDEC, and therefore CUDA-tensor frames, is confined to
    ``gpu_fast`` -- see ``runtime._should_use_nvdec``). Tests that additionally
    need CUDA-tensor FRAMES must say so explicitly via ``_simulate_cuda_frames``.
    """

    device = "cpu"  # simulate: real cuda uses "cuda", tensors stay torch
    tensor_on_cuda = True


def _simulate_cuda_frames(monkeypatch):
    """Make the production device-frame predicate accept CPU torch tensors.

    ``obb._frames_are_cuda_tensors`` requires ``tensor.is_cuda``, which cannot
    be satisfied on a CPU/MPS box. Patching the ONE predicate the sliced path
    consults (rather than inventing a parallel test-only predicate) keeps the
    production dispatch rule under test: everything downstream -- device-side
    tiling, ``_gpu_letterbox_batch``, the ``DirectExecutorAdapter`` exemption --
    runs unmodified against real torch tensors.
    """
    import hydra_suite.core.inference.stages.obb as obb_mod

    monkeypatch.setattr(
        obb_mod,
        "_frames_are_cuda_tensors",
        lambda frames: bool(frames) and isinstance(frames[0], torch.Tensor),
    )


class _FakeYOLOCudaTensorsFixed:
    """Predict returns one fixed-local-point obb detection per tile.

    ``source`` is the (B,3,imgsz,imgsz) GPU-letterboxed batch built by
    ``run_direct_sliced`` (``slicing.py``) for CUDA-tensor frames, then
    extracted device-side via ``slicing_cuda.assemble_raw_frames`` -- content
    is irrelevant to this fake, only the batch size matters (one result per
    tile job).
    """

    imgsz = 256
    overrides = {"imgsz": 256}

    def predict(self, source, **kw):
        b = source.shape[0] if hasattr(source, "shape") else len(source)
        results = []
        for _ in range(b):
            r = types.SimpleNamespace()
            r.obb = _FakeOBB(cx=60, cy=60, w=30, h=30)
            results.append(r)
        return results


class _FakeYOLOCudaGlobalPoint:
    """Predict emits a detection at a single fixed GLOBAL point -- expressed in
    each tile's own local coordinates -- for every tile whose bounding box
    actually contains that point (mirrors ``_FakeYOLOGlobalPoint``, adapted
    for the batched-tensor calling convention of the cuda path). Relies on
    ``predict()`` being called once per frame with tiles in ``plan.tiles``
    order (true here: single frame, no full-frame job).
    """

    imgsz = 256
    overrides = {"imgsz": 256}

    def __init__(self, tiles, global_point, size=(20.0, 20.0)):
        self.tiles = tiles
        self.gx, self.gy = global_point
        self.w, self.h = size
        self._call = 0

    def predict(self, source, **kw):
        b = source.shape[0] if hasattr(source, "shape") else len(source)
        results = []
        for _ in range(b):
            x0, y0, x1, y1 = self.tiles[self._call % len(self.tiles)]
            self._call += 1
            r = types.SimpleNamespace()
            if x0 <= self.gx < x1 and y0 <= self.gy < y1:
                r.obb = _FakeOBBN([(self.gx - x0, self.gy - y0, self.w, self.h)])
            else:
                r.obb = _FakeOBBN([])
            results.append(r)
        return results


class _FakeOBBDataView:
    """OBB fake whose ``.data`` is a REAL backing tensor: ``.xywhr`` is a VIEW
    of ``data[:, :5]`` (not a copy), and ``.xyxyxyxy`` is recomputed from
    ``data`` on every access -- matching the real ultralytics ``OBB`` contract
    that ``_invert_letterbox_on_result`` depends on (it mutates ``result.obb
    .data`` in place under ``torch.inference_mode()``, and downstream extract
    functions must see that mutation through both properties)."""

    def __init__(self, cx, cy, w, h, angle=0.0, conf=0.9, cls=0.0):
        self.data = torch.tensor(
            [[cx, cy, w, h, angle, conf, cls]], dtype=torch.float32
        )

    def __len__(self):
        return self.data.shape[0]

    @property
    def xywhr(self):
        return self.data[:, :5]

    @property
    def conf(self):
        return self.data[:, 5]

    @property
    def cls(self):
        return self.data[:, 6]

    @property
    def xyxyxyxy(self):
        cx, cy, w, h = (
            self.data[:, 0],
            self.data[:, 1],
            self.data[:, 2],
            self.data[:, 3],
        )
        hw, hh = w / 2.0, h / 2.0
        return torch.stack(
            [
                torch.stack([cx - hw, cy - hh], dim=1),
                torch.stack([cx + hw, cy - hh], dim=1),
                torch.stack([cx + hw, cy + hh], dim=1),
                torch.stack([cx - hw, cy + hh], dim=1),
            ],
            dim=1,
        )


class _FakeYOLOCudaLetterboxPoint:
    """Predict emits ONE obb detection expressed in LETTERBOXED-tile
    coordinates (not tile-local pixel coordinates) -- the coordinate space
    ``_invert_letterbox_on_result`` must transform back to tile-local pixels
    BEFORE ``_remap_raw`` adds the tile origin. Every other cuda-path fake in
    this file uses tiles exactly == imgsz (r=1, pad=0), so this is the only
    fake that can exercise the letterbox-invert guard in
    ``run_direct_sliced``'s ``_predict_tiles`` helper (``slicing.py``)."""

    imgsz = 256
    overrides = {"imgsz": 256}

    def __init__(self, cx_lb, cy_lb, w_lb, h_lb):
        self.cx_lb, self.cy_lb, self.w_lb, self.h_lb = cx_lb, cy_lb, w_lb, h_lb

    def predict(self, source, **kw):
        b = source.shape[0] if hasattr(source, "shape") else len(source)
        results = []
        for _ in range(b):
            r = types.SimpleNamespace()
            r.obb = _FakeOBBDataView(self.cx_lb, self.cy_lb, self.w_lb, self.h_lb)
            results.append(r)
        return results


def test_cuda_sliced_letterbox_invert_applies_real_scale_and_pad(monkeypatch):
    """Task 7 coverage gap: both existing cuda-path tests use tiles exactly
    == imgsz (256), so the letterbox-invert guard
    ``if r != 1.0 or pad_left != 0.0 or pad_top != 0.0:`` in
    ``slicing.py``'s ``_predict_tiles`` never executes -- r==1.0, pad==0.0
    always (``slicing_cuda.py`` only handles device-tensor EXTRACTION, not
    letterboxing). A
    non-square 100(h) x 200(w) tile against imgsz=256 forces
    ``_gpu_letterbox_batch`` to compute a REAL scale (min(256/100, 256/200))
    and a REAL vertical pad, covering BOTH halves of the guard.

    Per the task brief: derive (r, pad_left, pad_top) from the real function,
    do not hand-compute them.
    """
    from hydra_suite.core.inference.stages.obb import _gpu_letterbox_batch

    # Combination 4 of the frame/extraction matrix: CUDA-tensor frames AND
    # device-tensor extraction. Letterboxing only ever happens on the
    # CUDA-tensor FRAME path, so this test must declare that frame kind.
    _simulate_cuda_frames(monkeypatch)
    frame = torch.zeros((100, 200, 3), dtype=torch.uint8)  # H=100, W=200
    cfg = OBBConfig(
        mode="direct",
        direct=OBBDirectConfig(
            model_path="m.pt",
            model_task="obb",
            slice=SliceConfig(
                enabled=True,
                geometry_mode="custom",
                slice_height=100,
                slice_width=200,
                overlap_height_ratio=0.0,
                overlap_width_ratio=0.0,
            ),
        ),
        confidence_threshold=0.0,
        raw_detection_cap=0,
        max_detections=100,
    )

    # Ground truth #1: the real tile plan. Frame size == tile size -> exactly
    # one tile, flush at the origin (no offset to worry about here -- the
    # tile-origin remap is already covered by the other cuda-path tests).
    plan = plan_slices(
        (100, 200),
        cfg.direct.slice,
        imgsz=256,
        roi_mask=None,
        ref_object_px=cfg.direct.slice.reference_body_px,
    )
    assert plan.tiles == [(0, 0, 200, 100)]
    x0, y0, x1, y1 = plan.tiles[0]

    # Ground truth #2: the REAL letterbox params for this exact tile, from
    # the real function that production code calls -- not hand arithmetic.
    tile = frame[y0:y1, x0:x1]
    _, lb_params = _gpu_letterbox_batch([tile], imgsz=256)
    r, pad_left, pad_top = lb_params[0]

    # Prove the fixture actually reaches the guarded branch (non-vacuity of
    # the geometry, independent of the disable/re-enable proof below).
    assert r != 1.0
    assert pad_top != 0.0

    # Pick an arbitrary detection point in LETTERBOXED-tile space, then
    # derive the expected inverted + remapped point from the REAL (r,
    # pad_left, pad_top) via the documented inverse formula:
    #   x_orig = (x_lb - pad_left) / r ;  y_orig = (y_lb - pad_top) / r
    cx_lb, cy_lb, w_lb, h_lb = 130.0, 100.0, 51.2, 25.6
    expected_x = (cx_lb - pad_left) / r + x0
    expected_y = (cy_lb - pad_top) / r + y0
    expected_w = w_lb / r
    expected_h = h_lb / r

    model = _FakeYOLOCudaLetterboxPoint(cx_lb, cy_lb, w_lb, h_lb)
    out = run_direct_sliced([frame], model, cfg, _FakeCudaRuntime())

    assert len(out) == 1
    raw = out[0]
    assert raw.xywhr.shape[0] == 1
    assert raw.xywhr[0, 0].item() == pytest.approx(expected_x, abs=1e-3)
    assert raw.xywhr[0, 1].item() == pytest.approx(expected_y, abs=1e-3)
    assert raw.xywhr[0, 2].item() == pytest.approx(expected_w, abs=1e-3)
    assert raw.xywhr[0, 3].item() == pytest.approx(expected_h, abs=1e-3)


def test_cuda_no_tile_overlap_preserves_raw_tensors():
    """Combination 1: numpy frames + ``tensor_on_cuda=True`` (tier ``gpu``).

    This is the shape ``RuntimeContext.from_config`` emits for tier ``gpu`` on a
    CUDA box: torch backend -> device-tensor extraction, but CPU decode -> numpy
    frames (NVDEC is gpu_fast-only). Before finding C1 was fixed, this exact
    combination crashed with ``'numpy.ndarray' object has no attribute
    'permute'`` because the tiling path was selected by ``tensor_on_cuda``.
    """
    from hydra_suite.core.inference.stages.obb import _RawOBBTensors

    frame = np.zeros((512, 512, 3), np.uint8)
    cfg = _direct_cfg(True, overlap_height_ratio=0.0, overlap_width_ratio=0.0)
    out = run_direct_sliced(
        [frame], _FakeYOLOCudaTensorsFixed(), cfg, _FakeCudaRuntime()
    )
    assert len(out) == 1
    raw = out[0]
    assert isinstance(raw, _RawOBBTensors)  # no tile overlap -> zero sync, stays raw

    # Correctness, not just type: 4 disjoint tiles each report local (60, 60),
    # which remaps to 4 distinct global centroids -- verify the translation.
    tiles = get_slice_bboxes(512, 512, 256, 256, 0.0, 0.0)
    expected = sorted((x0 + 60, y0 + 60) for x0, y0, _, _ in tiles)
    actual = sorted(map(tuple, raw.xywhr[:, :2].tolist()))
    assert len(actual) == 4
    for (ex, ey), (ax, ay) in zip(expected, actual):
        assert ax == pytest.approx(ex, abs=1e-3)
        assert ay == pytest.approx(ey, abs=1e-3)
    # w, h, angle untouched by the remap (pure translation only).
    assert torch.allclose(raw.xywhr[:, 2], torch.full((4,), 30.0))
    assert torch.allclose(raw.xywhr[:, 3], torch.full((4,), 30.0))
    assert torch.allclose(raw.xywhr[:, 4], torch.zeros(4))


def test_cuda_tile_overlap_with_real_duplicate_materializes_and_merges():
    from hydra_suite.core.inference.result import OBBResult
    from hydra_suite.core.inference.stages.obb import _RawOBBTensors

    # Combination 1 again (numpy frames + device-tensor extraction), this time
    # with genuinely overlapping tiles so the merge/materialize branch runs.
    frame = np.zeros((300, 300, 3), np.uint8)
    cfg = _direct_cfg(True, overlap_height_ratio=0.2, overlap_width_ratio=0.2)
    tiles = get_slice_bboxes(300, 300, 256, 256, 0.2, 0.2)
    assert tiles_overlap(tiles) is True
    # (200, 200) sits inside all 4 tiles' genuine overlap band -> every tile
    # independently "detects" the same real object.
    global_point = (200.0, 200.0)
    model = _FakeYOLOCudaGlobalPoint(tiles, global_point)

    out = run_direct_sliced([frame], model, cfg, _FakeCudaRuntime())
    assert len(out) == 1
    assert not isinstance(out[0], _RawOBBTensors)  # overlap -> materialized + merged
    assert isinstance(out[0], OBBResult)
    # 4 identical cross-tile duplicates of the same object collapse to 1.
    assert out[0].num_detections == 1
    assert out[0].centroids[0, 0] == pytest.approx(global_point[0], abs=1.0)
    assert out[0].centroids[0, 1] == pytest.approx(global_point[1], abs=1.0)


# --- C1: the four frame-kind x extraction-path combinations -------------------
#
# The two decisions are ORTHOGONAL and must be dispatched separately:
#
#   tiling/preprocess  <- frames are CUDA tensors AND model is not a
#                         DirectExecutorAdapter  (device tiling + GPU letterbox)
#   extraction         <- runtime.tensor_on_cuda  (raw device tensors vs OBBResult)
#
#   | # | frames | tensor_on_cuda | producible as                        |
#   |---|--------|----------------|--------------------------------------|
#   | 1 | numpy  | True           | tier gpu (torch+cuda, CPU decode)    |
#   | 2 | numpy  | False          | cpu / mps / gpu_fast-CoreML          |
#   | 3 | cuda   | False          | gpu_fast + NVDEC + TRT adapter       |
#   | 4 | cuda   | True           | not producible today (defensive)     |
#
# Combination 1 is covered by test_cuda_no_tile_overlap_preserves_raw_tensors
# and test_cuda_tile_overlap_with_real_duplicate_materializes_and_merges;
# combination 2 by the test_sliced_cpu_* tests; combination 4 by
# test_cuda_sliced_letterbox_invert_applies_real_scale_and_pad. Combination 3
# is covered below.


class _FakeDirectExecutor:
    """Stands in for a TRT/ONNX direct executor behind ``DirectExecutorAdapter``.

    Asserts it is handed the RAW tile list (CUDA tensors), never a
    pre-letterboxed ``(B,3,imgsz,imgsz)`` batch -- the adapter does its own
    letterbox + original-frame rescale, so pre-batching double-preprocesses
    (the exact hazard ``_run_direct`` documents at its dispatch site).
    """

    imgsz = 256
    names = None

    def __init__(self):
        self.calls = []

    def predict(self, frames, *, conf_thres, classes, max_det):
        assert isinstance(frames, list)
        for f in frames:
            assert isinstance(f, torch.Tensor), f"adapter got {type(f).__name__}"
        self.calls.append(len(frames))
        results = []
        for _ in frames:
            r = types.SimpleNamespace()
            r.obb = _FakeOBBN([(60.0, 60.0, 30.0, 30.0)])
            results.append(r)
        return results


def test_cuda_frames_with_direct_executor_adapter_keeps_tensor_tiles(monkeypatch):
    """Combination 3: CUDA-tensor frames + ``tensor_on_cuda=False``.

    tier gpu_fast on CUDA: NVDEC yields CUDA-tensor frames, but the OBB stage
    resolves to the TensorRT backend, so extraction is the OBBResult kind.
    Before C1 was fixed this combination hit ``np.ascontiguousarray`` on a CUDA
    tensor. The adapter must receive tile TENSORS (it letterboxes internally),
    and the result must be a plain ``OBBResult``.
    """
    from hydra_suite.core.inference.result import OBBResult
    from hydra_suite.core.inference.runtime_artifacts import DirectExecutorAdapter

    _simulate_cuda_frames(monkeypatch)
    frame = torch.zeros((512, 512, 3), dtype=torch.uint8)
    executor = _FakeDirectExecutor()
    model = DirectExecutorAdapter(executor)
    cfg = _direct_cfg(True, overlap_height_ratio=0.0, overlap_width_ratio=0.0)

    out = run_direct_sliced([frame], model, cfg, _FakeRuntime())

    assert len(out) == 1
    assert isinstance(out[0], OBBResult)
    tiles = get_slice_bboxes(512, 512, 256, 256, 0.0, 0.0)
    assert sum(executor.calls) == len(tiles)  # one job per tile, nothing dropped
    expected = sorted((x0 + 60, y0 + 60) for x0, y0, _, _ in tiles)
    actual = sorted(map(tuple, out[0].centroids.tolist()))
    for (ex, ey), (ax, ay) in zip(expected, actual):
        assert ax == pytest.approx(ex, abs=1e-3)
        assert ay == pytest.approx(ey, abs=1e-3)


def test_cuda_frames_with_plain_torch_model_letterboxes_device_tiles(monkeypatch):
    """Combination 3/4 preprocess half: CUDA-tensor frames + a plain ultralytics
    model must be GPU-letterboxed into a single batched tensor (a list of
    tensors is not a valid ultralytics predict source)."""
    _simulate_cuda_frames(monkeypatch)
    seen = {}

    class _Model:
        imgsz = 256
        overrides = {"imgsz": 256}

        def predict(self, source, **kw):
            seen["type"] = type(source)
            seen["shape"] = tuple(source.shape)
            return [
                types.SimpleNamespace(obb=_FakeOBBN([(60.0, 60.0, 30.0, 30.0)]))
                for _ in range(source.shape[0])
            ]

    frame = torch.zeros((512, 512, 3), dtype=torch.uint8)
    cfg = _direct_cfg(True, overlap_height_ratio=0.0, overlap_width_ratio=0.0)
    out = run_direct_sliced([frame], _Model(), cfg, _FakeRuntime())
    assert seen["type"] is torch.Tensor
    assert seen["shape"][1:] == (3, 256, 256)
    assert out[0].num_detections == 4


def test_fixture_runtime_flag_combinations_are_producible(monkeypatch):
    """Guard against the class of error that hid C1: a fixture encoding an
    IMPOSSIBLE ``RuntimeContext``.

    Derives the (tensor_on_cuda, use_nvdec) pair that
    ``RuntimeContext.from_config`` really emits per tier on a CUDA host, and
    pins the two facts the sliced dispatch depends on:
      * tier ``gpu``   -> tensor_on_cuda=True  AND no NVDEC (numpy frames)
      * tier gpu_fast  -> NVDEC frames         AND tensor_on_cuda=False
    i.e. "device-tensor extraction" and "CUDA-tensor frames" are mutually
    exclusive in production, so neither may be inferred from the other.
    """
    from hydra_suite.core.inference import runtime as rt_mod
    from hydra_suite.core.inference.config import InferenceConfig
    from hydra_suite.runtime import resolver as resolver_mod

    monkeypatch.setattr(
        resolver_mod,
        "detect_platform",
        lambda: resolver_mod.PlatformInfo(has_cuda=True, has_mps=False),
    )
    monkeypatch.setattr(rt_mod, "_cuda_device_available", lambda: "cuda:0")
    monkeypatch.setattr(rt_mod, "_nvdec_available", lambda: True)

    produced = {}
    for tier in ("cpu", "gpu", "gpu_fast"):
        ctx = rt_mod.RuntimeContext.from_config(
            InferenceConfig(obb=_direct_cfg(True), runtime_tier=tier)
        )
        produced[tier] = (bool(ctx.tensor_on_cuda), bool(ctx.use_nvdec))

    assert produced["gpu"] == (True, False)  # combination 1 fixture shape
    assert produced["gpu_fast"] == (False, True)  # combination 3 fixture shape
    assert produced["cpu"] == (False, False)  # combination 2 fixture shape
    # No producible tier yields CUDA-tensor frames AND device-tensor extraction.
    assert not any(t and n for t, n in produced.values())

    # The fixtures used in this module must each match a producible shape.
    assert (_FakeCudaRuntime.tensor_on_cuda, False) == produced["gpu"]
    assert (_FakeRuntime.tensor_on_cuda, False) == produced["cpu"]


# --- I2: bounded predict batch ------------------------------------------------


def test_predict_is_chunked_to_a_bounded_tile_count():
    """A window of frames must NOT be issued as one frames x tiles predict call:
    with 25 tiles that is 25x the peak activation memory the user configured."""
    from hydra_suite.core.inference.stages.slicing import MAX_TILE_CHUNK

    sizes = []

    class _Model:
        imgsz = 256
        overrides = {"imgsz": 256}

        def predict(self, source, **kw):
            n = source.shape[0] if hasattr(source, "shape") else len(source)
            sizes.append(n)
            return [types.SimpleNamespace(obb=_FakeOBBN([])) for _ in range(n)]

    frames = [np.zeros((512, 512, 3), np.uint8) for _ in range(8)]
    cfg = _direct_cfg(True, overlap_height_ratio=0.0, overlap_width_ratio=0.0)
    out = run_direct_sliced(frames, _Model(), cfg, _FakeRuntime())
    assert len(out) == 8
    assert sum(sizes) == 8 * 4  # 4 tiles/frame, every job issued exactly once
    assert len(sizes) > 1  # actually chunked, not one giant call
    assert max(sizes) <= MAX_TILE_CHUNK
    assert max(sizes) <= 4  # bounded by tiles-per-frame (the TRT engine profile)


# --- I3: cap applied before the merge on every path ---------------------------


def test_raw_detection_cap_is_applied_before_the_merge(monkeypatch):
    """The cap must bound the O(n^2) merge input, not just the output -- and
    both extraction paths must keep the SAME detections."""
    from hydra_suite.core.inference.stages import merge as merge_mod

    seen = {}
    real_merge = merge_mod.merge_obb_detections

    def _spy(result, **kw):
        seen["n_in"] = result.num_detections
        return real_merge(result, **kw)

    monkeypatch.setattr(merge_mod, "merge_obb_detections", _spy)

    class _ManyDetModel:
        imgsz = 256
        overrides = {"imgsz": 256}

        def predict(self, source, **kw):
            n = source.shape[0] if hasattr(source, "shape") else len(source)
            return [
                types.SimpleNamespace(
                    obb=_FakeOBBN([(20.0 + 5 * i, 20.0, 8.0, 8.0) for i in range(10)])
                )
                for _ in range(n)
            ]

    frame = np.zeros((300, 300, 3), np.uint8)
    cfg = _direct_cfg(True, overlap_height_ratio=0.2, overlap_width_ratio=0.2)
    cfg.raw_detection_cap = 5
    out = run_direct_sliced([frame], _ManyDetModel(), cfg, _FakeRuntime())
    assert seen["n_in"] == 5  # 4 tiles x 10 dets capped to 5 BEFORE merging
    assert out[0].num_detections <= 5


# --- I5: analytic overlap predicate + tile-count guard ------------------------


@pytest.mark.parametrize(
    "frame_h,frame_w,sh,sw,oh,ow",
    [
        (300, 300, 256, 256, 0.0, 0.0),
        (512, 512, 256, 256, 0.0, 0.0),
        (900, 1600, 256, 384, 0.3, 0.5),
        (1000, 1000, 256, 256, 0.2, 0.2),
        (500, 500, 2000, 2000, 0.2, 0.2),
        (1080, 1920, 640, 640, 0.0, 0.0),
    ],
)
def test_tiles_overlap_matches_the_pairwise_reference(frame_h, frame_w, sh, sw, oh, ow):
    """The O(1) analytic predicate must agree with the O(T^2) definition."""
    tiles = get_slice_bboxes(frame_h, frame_w, sh, sw, oh, ow)

    def _reference(ts):
        for i in range(len(ts)):
            ax0, ay0, ax1, ay1 = ts[i]
            for j in range(i + 1, len(ts)):
                bx0, by0, bx1, by1 = ts[j]
                if ax0 < bx1 and bx0 < ax1 and ay0 < by1 and by0 < ay1:
                    return True
        return False

    assert tiles_overlap(tiles) is _reference(tiles)


def test_tiles_overlap_is_not_quadratic_on_a_huge_grid():
    """53k tiles must be answered instantly; the O(T^2) scan was ~2.8e9 pure
    Python iterations -- an unkillable-looking hang with no log line."""
    import time

    tiles = get_slice_bboxes(1080, 1920, 64, 64, 0.9, 0.9)
    assert len(tiles) > 20000
    t0 = time.perf_counter()
    assert tiles_overlap(tiles) is True
    assert time.perf_counter() - t0 < 1.0


def test_pathological_tile_count_raises_instead_of_hanging():
    """slice=64 + overlap=0.9 on 1080p is reachable via advanced_config.json and
    yields ~53k tiles (53k forward passes). Fail loudly, do not silently spin."""
    from hydra_suite.core.inference.stages.slicing import MAX_TILES_PER_FRAME

    cfg = SliceConfig(
        enabled=True,
        geometry_mode="custom",
        slice_height=64,
        slice_width=64,
        overlap_height_ratio=0.9,
        overlap_width_ratio=0.9,
    )
    with pytest.raises(ValueError, match="tile"):
        plan_slices((1080, 1920), cfg, imgsz=64, roi_mask=None)
    # A sane plan of the same shape is unaffected.
    ok = plan_slices(
        (1080, 1920),
        SliceConfig(
            enabled=True,
            geometry_mode="custom",
            slice_height=640,
            slice_width=640,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2,
        ),
        imgsz=640,
        roi_mask=None,
    )
    assert 0 < len(ok.tiles) <= MAX_TILES_PER_FRAME


# ---- ROI tile-gating wired end-to-end through run_direct_sliced ----


class _CountingBlobYOLO:
    """CPU fake that (a) counts every tile image handed to ``predict`` and (b)
    emits a detection at the centroid of any non-zero ("white blob") pixels in
    each tile crop.

    Content-based detection (not tile-index-based) makes it robust to ROI
    gating: a gated-out tile is simply never passed to ``predict``, and a tile
    that IS passed produces the same detection whether or not other tiles were
    dropped. ``n_tiles_predicted`` is the observable that proves gating dropped
    real forward passes.
    """

    imgsz = 256
    overrides = {"imgsz": 256}

    def __init__(self):
        self.n_tiles_predicted = 0

    def predict(self, source, **kw):
        results = []
        for img in source:
            self.n_tiles_predicted += 1
            r = types.SimpleNamespace()
            ys, xs = np.where(np.asarray(img)[..., 0] > 0)
            if xs.size > 0:
                cx, cy = float(xs.mean()), float(ys.mean())
                w = float(xs.max() - xs.min() + 1)
                h = float(ys.max() - ys.min() + 1)
                r.obb = _FakeOBBN([(cx, cy, w, h)])
            else:
                r.obb = _FakeOBBN([])
            results.append(r)
        return results


def _gating_cfg():
    # auto_model + imgsz 256 => 256px tiles; zero overlap keeps the grid simple.
    return _direct_cfg(True, overlap_height_ratio=0.0, overlap_width_ratio=0.0)


def test_roi_gating_drops_predict_tiles_end_to_end():
    """Test 1: with a real ROI mask that empties a corner region big enough to
    contain whole tiles, the sliced run issues FEWER model.predict tiles than
    the ungated run -- proving tiles are actually dropped, not merely 'ran'."""
    frame = np.zeros((1000, 1000, 3), np.uint8)
    cfg = _gating_cfg()

    # Independently confirm the ungated grid is 4x4 = 16 tiles (256px, step 256,
    # edge-flush last tile): xs=ys=[0,256,512,744].
    full_plan = plan_slices((1000, 1000), cfg.direct.slice, imgsz=256, roi_mask=None)
    assert len(full_plan.tiles) == 16

    # Zero the entire top-left 512x512 corner -> the 4 tiles fully inside it
    # ((0,0),(256,0),(0,256),(256,256)) contain no live ROI pixel and must drop.
    mask = np.ones((1000, 1000), np.uint8)
    mask[:512, :512] = 0
    gated_plan = plan_slices((1000, 1000), cfg.direct.slice, imgsz=256, roi_mask=mask)
    assert len(gated_plan.tiles) == 12  # 16 - 4 corner tiles

    ungated_model = _CountingBlobYOLO()
    run_direct_sliced([frame], ungated_model, cfg, _FakeRuntime(), roi_mask=None)

    gated_model = _CountingBlobYOLO()
    run_direct_sliced([frame], gated_model, cfg, _FakeRuntime(), roi_mask=mask)

    assert ungated_model.n_tiles_predicted == 16
    assert gated_model.n_tiles_predicted == 12
    assert gated_model.n_tiles_predicted < ungated_model.n_tiles_predicted


def test_roi_gating_leaves_final_detections_unchanged():
    """Test 2: 'results identical, only compute saved'. A detection placed well
    inside the ROI (and inside a KEPT tile) survives identically whether or not
    the empty corner tiles are gated away."""
    frame = np.zeros((1000, 1000, 3), np.uint8)
    # White blob at ~(600, 600): lands in tile (512,512)-(768,768), which is NOT
    # in the zeroed corner, so gating never removes its tile.
    frame[595:606, 595:606, :] = 255
    cfg = _gating_cfg()

    mask = np.ones((1000, 1000), np.uint8)
    mask[:512, :512] = 0  # gated corner contains no blob -> no in-ROI detection

    ungated = run_direct_sliced(
        [frame], _CountingBlobYOLO(), cfg, _FakeRuntime(), roi_mask=None
    )[0]
    gated = run_direct_sliced(
        [frame], _CountingBlobYOLO(), cfg, _FakeRuntime(), roi_mask=mask
    )[0]

    assert ungated.num_detections == gated.num_detections
    assert ungated.num_detections >= 1
    ung = sorted(map(tuple, np.round(ungated.centroids, 3).tolist()))
    gat = sorted(map(tuple, np.round(gated.centroids, 3).tolist()))
    assert ung == gat


def test_plan_slices_coordinate_space_guard_treats_wrong_shape_as_none(caplog):
    """Test 4: a mask whose shape != frame_hw must NOT mis-gate; it degrades to
    the full grid (== roi_mask None) and logs a warning."""
    cfg = SliceConfig(
        enabled=True, geometry_mode="custom", slice_height=256, slice_width=256
    )
    full = plan_slices((1000, 1000), cfg, imgsz=256, roi_mask=None)
    # Mask sized for a DIFFERENT frame (e.g. a resized-space mask) -- wrong space.
    wrong = np.ones((500, 500), np.uint8)
    wrong[:250, :250] = 0
    with caplog.at_level("WARNING"):
        guarded = plan_slices((1000, 1000), cfg, imgsz=256, roi_mask=wrong)
    assert guarded.tiles == full.tiles  # no gating applied
    assert any("does not match frame" in rec.message for rec in caplog.records)
