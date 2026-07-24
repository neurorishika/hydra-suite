import types

import numpy as np
import pytest

from hydra_suite.core.inference.config import OBBConfig, OBBDirectConfig, SliceConfig
from hydra_suite.core.inference.stages.obb import OBBModels, run_obb
from hydra_suite.core.inference.stages.slicing import (
    get_slice_bboxes,
    plan_slices,
    run_direct_sliced,
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
    With the guard, step=max(1, 0)=1, yielding (512-64+1)^2 ≈ 201,601 tiles.
    """
    cfg = SliceConfig(
        enabled=True,
        geometry_mode="custom",
        slice_height=64,
        slice_width=64,
        overlap_height_ratio=0.99,
        overlap_width_ratio=0.99,
    )
    plan = plan_slices((512, 512), cfg, imgsz=64, roi_mask=None)

    # Should not crash (max(1, ...) prevents ValueError from range(0, N, 0)).
    assert len(plan.tiles) > 0
    # With step=1 (floor engaged), tile count = (512-64+1)^2 ≈ 201,601.
    # Upper bound accounts for the math: (frame - slice + 1)^2.
    assert len(plan.tiles) <= (512 - 64 + 1) ** 2

    # Every tile should be full size (not shrunk).
    for x0, y0, x1, y1 in plan.tiles:
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
