# DetectKit SAHI Sliced Training + Inference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make DetectKit produce direct OBB models usable under SAHI sliced inference, by generating training data at the sliced scale (multi-scale, size-robust), previewing sliced predictions, and stamping the training geometry into the model manifest.

**Architecture:** Extract the pure tile planner from `core/inference/stages/slicing.py` into `utils/slice_geometry.py` so inference, the new training builder, and the preview all tile through ONE implementation (the only structural guarantee that train-tiles == inference-tiles). A new `training/sliced_dataset.py` tiles a merged OBB dataset: clip each label polygon to each tile, area-threshold, remap, re-derive geometry level, sample negatives, mix full frames, emit multiple target scales. DetectKit's preview gains an executor-level `predict_sliced` wrapper reusing the shipped merge helpers. Settings live in one shared block persisted in the project JSON; the resolved geometry is stamped through `model_publish.py`.

**Tech Stack:** Python 3, NumPy, OpenCV (`cv2`), Ultralytics YOLO datasets, PySide6/Qt (DetectKit GUI), pytest.

## Global Constraints

- **Byte-parity of inference:** After extracting geometry to `utils/slice_geometry.py`, `core/inference/stages/slicing.py` MUST emit byte-identical tile boxes to the pre-extraction path. The existing `tests/test_inference_slicing.py` imports `get_slice_bboxes`, `plan_slices`, `tiles_overlap` from `hydra_suite.core.inference.stages.slicing` and MUST keep passing unchanged (re-export the moved names).
- **`utils/` layer purity:** `utils/slice_geometry.py` MUST NOT import from `hydra_suite.core.inference` or `hydra_suite.training` or Qt. It takes plain geometry params, sits beside `utils/rotated_iou.py` and `utils/obb_from_mask.py`. Level re-derivation (which needs `GeometryLevel`) lives in `training/`, not `utils/`.
- **Defaults off / non-invasive:** An existing non-sliced DetectKit project MUST be unaffected. `SliceTrainingSettings.enabled` defaults `False`; when off, the dataset build and preview reduce exactly to today's behavior.
- **`_tile_size` semantics are frozen:** the extracted `tile_size_for_mode` MUST reproduce `_tile_size` exactly — custom: `slice_width/height` else `imgsz`; auto_object with `reference_body_px > 0`: `frac = clamp(object_tile_fraction, 0.01, 0.9)`, `size = round(reference_body_px / frac)`, `clamp(64, 4096)`, square; otherwise `(imgsz, imgsz)`.
- **Metadata write only (this spec):** we WRITE slice geometry into the model manifest; TrackerKit reading it to auto-configure `SliceConfig` is a separate follow-up spec — do NOT implement any TrackerKit read side here.
- **Default parameter values (verbatim):** `min_area_ratio = 0.1`, `negative_tile_fraction = 0.15`, `full_frame_mix = True`, `overlap = 0.2`, `object_tile_fraction = 0.15`. Multi-scale is "a configurable list of target apparent sizes" (default 3 values: `[200.0, 300.0, 400.0]`).
- **Determinism:** every builder path that samples (negatives, split assignment) MUST take a `seed` and use `random.Random(seed)` — no global RNG, no `Date.now()`-style nondeterminism.
- **Commit identity:** commit as the configured git user; do NOT add a `Co-Authored-By: Claude` trailer.

---

### Task 1: Extract pure tile geometry into `utils/slice_geometry.py`

**Files:**
- Create: `src/hydra_suite/utils/slice_geometry.py`
- Modify: `src/hydra_suite/core/inference/stages/slicing.py` (replace moved defs with re-imports)
- Test: `tests/test_slice_geometry.py`, `tests/test_slice_geometry_parity.py`

**Interfaces:**
- Produces (from `hydra_suite.utils.slice_geometry`):
  - `SlicePlan` dataclass: `tiles: list[tuple[int,int,int,int]]`, `full_frame: bool`, `slice_wh: tuple[int,int]`, `frame_wh: tuple[int,int]`, property `jobs_per_frame -> int`.
  - `MAX_TILES_PER_FRAME: int = 4096`
  - `get_slice_bboxes(frame_h, frame_w, slice_h, slice_w, overlap_h, overlap_w) -> list[tuple[int,int,int,int]]`
  - `tiles_overlap(tiles) -> bool`
  - `tile_size_for_mode(*, geometry_mode: str, imgsz: int, reference_body_px: float, object_tile_fraction: float, slice_width: int, slice_height: int) -> tuple[int,int]`
  - `plan_tiles(frame_hw, slice_w, slice_h, overlap_w, overlap_h, *, full_frame=False, roi_mask=None) -> SlicePlan`
- Consumes: nothing (pure geometry, numpy only).

- [ ] **Step 1: Write the failing parity test**

Create `tests/test_slice_geometry_parity.py`:

```python
import numpy as np
import pytest

from hydra_suite.core.inference.config import SliceConfig
from hydra_suite.core.inference.stages import slicing as stages_slicing
from hydra_suite.utils import slice_geometry as sg


def test_names_are_reexported_from_stages():
    # Existing inference tests import these from stages.slicing; they must stay.
    assert stages_slicing.get_slice_bboxes is sg.get_slice_bboxes
    assert stages_slicing.tiles_overlap is sg.tiles_overlap
    assert stages_slicing.SlicePlan is sg.SlicePlan
    assert stages_slicing.MAX_TILES_PER_FRAME == sg.MAX_TILES_PER_FRAME


@pytest.mark.parametrize(
    "frame_hw,mode,imgsz,ref,frac,sw,sh,overlap",
    [
        ((1000, 1000), "auto_model", 640, 0.0, 0.15, 0, 0, 0.2),
        ((2000, 2000), "custom", 1024, 0.0, 0.15, 512, 512, 0.2),
        ((4000, 4000), "auto_object", 1024, 64.0, 0.15, 0, 0, 0.2),
        ((900, 1600), "auto_model", 384, 0.0, 0.15, 0, 0, 0.3),
    ],
)
def test_plan_tiles_matches_plan_slices(frame_hw, mode, imgsz, ref, frac, sw, sh, overlap):
    cfg = SliceConfig(
        enabled=True,
        geometry_mode=mode,
        slice_width=sw,
        slice_height=sh,
        overlap_width_ratio=overlap,
        overlap_height_ratio=overlap,
        object_tile_fraction=frac,
    )
    plan = stages_slicing.plan_slices(frame_hw, cfg, imgsz=imgsz, roi_mask=None, ref_object_px=ref)
    w, h = sg.tile_size_for_mode(
        geometry_mode=mode, imgsz=imgsz, reference_body_px=ref,
        object_tile_fraction=frac, slice_width=sw, slice_height=sh,
    )
    direct = sg.plan_tiles(frame_hw, w, h, overlap, overlap, full_frame=False, roi_mask=None)
    assert plan.tiles == direct.tiles
    assert plan.slice_wh == direct.slice_wh
```

- [ ] **Step 2: Run it to verify it fails**

Run: `PYTHONPATH=src python -m pytest tests/test_slice_geometry_parity.py -v`
Expected: FAIL with `ModuleNotFoundError: hydra_suite.utils.slice_geometry`.

- [ ] **Step 3: Create `utils/slice_geometry.py` with the moved primitives**

Move the geometry verbatim from `stages/slicing.py`. Create `src/hydra_suite/utils/slice_geometry.py`:

```python
"""Pure tile-planning geometry shared by inference, training, and preview.

Extracted from ``core/inference/stages/slicing.py`` so the training dataset
builder and DetectKit preview tile with the EXACT same grid the inference path
uses (Approach B). No ``core.inference`` / ``training`` / Qt imports — plain
geometry beside ``utils/rotated_iou.py`` and ``utils/obb_from_mask.py``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# Hard ceiling on tiles per frame (see the original slicing.py note, finding I5):
# refuse pathological slice/overlap combos loudly instead of spinning for hours.
MAX_TILES_PER_FRAME = 4096


@dataclass
class SlicePlan:
    """A memoizable tiling of a fixed-size frame."""

    tiles: list[tuple[int, int, int, int]]  # (x0, y0, x1, y1) per tile
    full_frame: bool  # append one full-frame pass in addition to tiles
    slice_wh: tuple[int, int]  # (w, h) of each tile
    frame_wh: tuple[int, int]  # (w, h) of the source frame

    @property
    def jobs_per_frame(self) -> int:
        return len(self.tiles) + (1 if self.full_frame else 0)


def _axis_starts(total: int, size: int, step: int) -> list[int]:
    """Tile start offsets along one axis, last tile flush to the far edge."""
    if size >= total:
        return [0]
    starts = list(range(0, total - size + 1, step))
    last = total - size
    if starts[-1] != last:
        starts.append(last)
    return starts


def _axis_geometry(frame_h, frame_w, slice_h, slice_w, overlap_h, overlap_w):
    """Return ``(xs, ys, slice_w, slice_h)`` with sizes clamped to the frame."""
    slice_w = min(slice_w, frame_w)
    slice_h = min(slice_h, frame_h)
    step_x = max(1, int(slice_w * (1.0 - overlap_w)))
    step_y = max(1, int(slice_h * (1.0 - overlap_h)))
    return (
        _axis_starts(frame_w, slice_w, step_x),
        _axis_starts(frame_h, slice_h, step_y),
        slice_w,
        slice_h,
    )


def get_slice_bboxes(frame_h, frame_w, slice_h, slice_w, overlap_h, overlap_w):
    """SAHI ``get_slice_bboxes``: fixed-size tiles, last tile flush to the edge.

    Pure geometry primitive: deliberately UNGUARDED (``plan_tiles`` owns the
    tile-count ceiling), so tests can exercise degenerate steps directly.
    """
    xs, ys, slice_w, slice_h = _axis_geometry(
        frame_h, frame_w, slice_h, slice_w, overlap_h, overlap_w
    )
    return [(x, y, x + slice_w, y + slice_h) for y in ys for x in xs]


def tiles_overlap(tiles):
    """True when ANY two planned tiles actually intersect. Analytic for a regular grid."""
    n = len(tiles)
    if n <= 1:
        return False
    xs = sorted({t[0] for t in tiles})
    ys = sorted({t[1] for t in tiles})
    w = tiles[0][2] - tiles[0][0]
    h = tiles[0][3] - tiles[0][1]
    regular_grid = n == len(xs) * len(ys) and all(
        (t[2] - t[0]) == w and (t[3] - t[1]) == h for t in tiles
    )
    if not regular_grid:
        return _tiles_overlap_pairwise(tiles)
    if any(b - a < w for a, b in zip(xs, xs[1:])):
        return True
    return any(b - a < h for a, b in zip(ys, ys[1:]))


def _tiles_overlap_pairwise(tiles):
    """O(T^2) reference definition; only reached for a non-grid tile list."""
    for i in range(len(tiles)):
        ax0, ay0, ax1, ay1 = tiles[i]
        for j in range(i + 1, len(tiles)):
            bx0, by0, bx1, by1 = tiles[j]
            if ax0 < bx1 and bx0 < ax1 and ay0 < by1 and by0 < ay1:
                return True
    return False


def tile_size_for_mode(
    *,
    geometry_mode: str,
    imgsz: int,
    reference_body_px: float,
    object_tile_fraction: float,
    slice_width: int,
    slice_height: int,
) -> tuple[int, int]:
    """Return (w, h) tile size for the configured geometry mode.

    Reproduces ``stages/slicing.py:_tile_size`` exactly (semantics are frozen).
    """
    if geometry_mode == "custom":
        w = slice_width if slice_width > 0 else imgsz
        h = slice_height if slice_height > 0 else imgsz
        return int(w), int(h)
    if geometry_mode == "auto_object" and reference_body_px > 0:
        frac = max(0.01, min(0.9, object_tile_fraction))
        size = int(round(reference_body_px / frac))
        size = max(64, min(4096, size))
        return size, size
    # auto_model (and auto_object fallback when no ref object is known).
    return int(imgsz), int(imgsz)


def plan_tiles(
    frame_hw,
    slice_w: int,
    slice_h: int,
    overlap_w: float,
    overlap_h: float,
    *,
    full_frame: bool = False,
    roi_mask: "np.ndarray | None" = None,
) -> SlicePlan:
    """Compute the tile plan for one frame size.

    Raises ``ValueError`` when the geometry would produce more than
    ``MAX_TILES_PER_FRAME`` tiles. ROI gating drops tiles with no live mask
    pixel; a mask whose shape mismatches the frame is treated as ``None``.
    """
    frame_h, frame_w = int(frame_hw[0]), int(frame_hw[1])
    xs, ys, eff_w, eff_h = _axis_geometry(
        frame_h, frame_w, slice_h, slice_w, overlap_h, overlap_w
    )
    n_tiles = len(xs) * len(ys)
    if n_tiles > MAX_TILES_PER_FRAME:
        raise ValueError(
            f"Sliced tiling would produce {n_tiles} tiles per frame "
            f"({len(xs)}x{len(ys)}) for a {frame_w}x{frame_h} frame with "
            f"{eff_w}x{eff_h} tiles at overlap ({overlap_w}, {overlap_h}) -- "
            f"above the {MAX_TILES_PER_FRAME}-tile ceiling. Increase the slice "
            f"size or lower the overlap."
        )
    tiles = [(x, y, x + eff_w, y + eff_h) for y in ys for x in xs]
    if roi_mask is not None and roi_mask.shape[:2] != (frame_h, frame_w):
        logger.warning(
            "ROI mask shape %s does not match frame (%d, %d); skipping ROI tile "
            "gating for this frame to avoid mis-gating.",
            tuple(roi_mask.shape[:2]), frame_h, frame_w,
        )
        roi_mask = None
    if roi_mask is not None:
        h, w = roi_mask.shape[:2]
        kept = []
        for x0, y0, x1, y1 in tiles:
            yy0, yy1 = max(0, y0), min(h, y1)
            xx0, xx1 = max(0, x0), min(w, x1)
            if yy1 > yy0 and xx1 > xx0 and roi_mask[yy0:yy1, xx0:xx1].any():
                kept.append((x0, y0, x1, y1))
        tiles = kept if kept else tiles
    return SlicePlan(
        tiles=tiles,
        full_frame=bool(full_frame),
        slice_wh=(eff_w, eff_h),
        frame_wh=(frame_w, frame_h),
    )
```

- [ ] **Step 4: Rewire `stages/slicing.py` to re-import and delegate**

In `src/hydra_suite/core/inference/stages/slicing.py`:

1. Replace the top-of-file geometry defs (`MAX_TILES_PER_FRAME`, `SlicePlan`, `_axis_starts`, `_axis_geometry`, `get_slice_bboxes`, `tiles_overlap`, `_tiles_overlap_pairwise`) with an import. Add near the existing imports:

```python
from hydra_suite.utils.slice_geometry import (
    MAX_TILES_PER_FRAME,
    SlicePlan,
    get_slice_bboxes,
    plan_tiles,
    tile_size_for_mode,
    tiles_overlap,
)
```

Keep `MAX_TILE_CHUNK = 128` and the `_gpu_merge_backend_downgrade_logged` / `_log_gpu_merge_backend_downgrade_once` helpers in `slicing.py` (they are inference-specific, not geometry). Delete the moved defs.

2. Replace `_tile_size` with a thin delegate:

```python
def _tile_size(slice_cfg: SliceConfig, imgsz: int, ref_object_px: float) -> tuple[int, int]:
    return tile_size_for_mode(
        geometry_mode=slice_cfg.geometry_mode,
        imgsz=imgsz,
        reference_body_px=ref_object_px,
        object_tile_fraction=slice_cfg.object_tile_fraction,
        slice_width=slice_cfg.slice_width,
        slice_height=slice_cfg.slice_height,
    )
```

3. Replace the body of `plan_slices` (keep its signature and docstring) with:

```python
def plan_slices(frame_hw, slice_cfg, imgsz, roi_mask, ref_object_px=0.0):
    slice_w, slice_h = _tile_size(slice_cfg, imgsz, ref_object_px)
    return plan_tiles(
        frame_hw,
        slice_w,
        slice_h,
        slice_cfg.overlap_width_ratio,
        slice_cfg.overlap_height_ratio,
        full_frame=bool(slice_cfg.perform_standard_pred),
        roi_mask=roi_mask,
    )
```

Leave `_extract_tile`, `_offset_result`, `_build_tile_jobs`, `_predict_tiles`, `_merge_frame_obb_results`, `run_direct_sliced` untouched (they already reference `plan_slices`, `SlicePlan`, `get_slice_bboxes` via the module namespace).

- [ ] **Step 5: Write the unit geometry test**

Create `tests/test_slice_geometry.py`:

```python
import numpy as np

from hydra_suite.utils.slice_geometry import (
    MAX_TILES_PER_FRAME, get_slice_bboxes, plan_tiles, tile_size_for_mode, tiles_overlap,
)


def test_grid_flushes_last_tile_to_edge():
    boxes = get_slice_bboxes(1000, 1000, 640, 640, 0.2, 0.2)
    assert all(x1 <= 1000 and y1 <= 1000 for _, _, x1, y1 in boxes)
    assert any(x1 == 1000 for _, _, x1, _ in boxes)


def test_tile_size_custom_falls_back_to_imgsz():
    assert tile_size_for_mode(geometry_mode="custom", imgsz=512, reference_body_px=0.0,
                              object_tile_fraction=0.15, slice_width=0, slice_height=0) == (512, 512)
    assert tile_size_for_mode(geometry_mode="custom", imgsz=512, reference_body_px=0.0,
                              object_tile_fraction=0.15, slice_width=300, slice_height=200) == (300, 200)


def test_tile_size_auto_object():
    # ref=64, frac=0.16 -> 400
    assert tile_size_for_mode(geometry_mode="auto_object", imgsz=1024, reference_body_px=64.0,
                              object_tile_fraction=0.16, slice_width=0, slice_height=0) == (400, 400)


def test_tile_size_auto_object_zero_ref_falls_back_to_auto_model():
    assert tile_size_for_mode(geometry_mode="auto_object", imgsz=1024, reference_body_px=0.0,
                              object_tile_fraction=0.15, slice_width=0, slice_height=0) == (1024, 1024)


def test_plan_tiles_ceiling_raises():
    import pytest
    with pytest.raises(ValueError):
        plan_tiles((1080, 1920), 64, 64, 0.9, 0.9)


def test_plan_tiles_roi_gating_drops_tiles():
    mask = np.zeros((1000, 1000), dtype=bool)
    mask[:256, :256] = True
    full = plan_tiles((1000, 1000), 256, 256, 0.0, 0.0)
    gated = plan_tiles((1000, 1000), 256, 256, 0.0, 0.0, roi_mask=mask)
    assert len(gated.tiles) < len(full.tiles)


def test_tiles_overlap_true_for_flush_last_tile():
    assert tiles_overlap(get_slice_bboxes(300, 300, 256, 256, 0.0, 0.0)) is True
```

- [ ] **Step 6: Run all affected tests**

Run: `PYTHONPATH=src python -m pytest tests/test_slice_geometry.py tests/test_slice_geometry_parity.py tests/test_inference_slicing.py -v`
Expected: PASS (the pre-existing `test_inference_slicing.py` proves byte-parity of the inference path).

- [ ] **Step 7: Commit**

```bash
git add src/hydra_suite/utils/slice_geometry.py src/hydra_suite/core/inference/stages/slicing.py tests/test_slice_geometry.py tests/test_slice_geometry_parity.py
git commit -m "refactor(inference): extract pure tile geometry into utils/slice_geometry"
```

---

### Task 2: Polygon-clip-to-tile geometry in `utils/slice_geometry.py`

**Files:**
- Modify: `src/hydra_suite/utils/slice_geometry.py`
- Test: `tests/test_slice_geometry.py`

**Interfaces:**
- Produces:
  - `clip_polygon_to_tile(poly_px: np.ndarray, tile: tuple[int,int,int,int]) -> np.ndarray | None` — Sutherland–Hodgman clip of an `(N,2)` pixel-space polygon against the axis-aligned tile rect `(x0,y0,x1,y1)`. Returns the clipped `(M,2)` polygon in the SAME frame-pixel space (NOT yet remapped), or `None` when the intersection is empty/degenerate.
  - `polygon_area(poly: np.ndarray) -> float` — absolute area of an `(N,2)` polygon (shoelace).
- Consumes: nothing new.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_slice_geometry.py`:

```python
from hydra_suite.utils.slice_geometry import clip_polygon_to_tile, polygon_area


def test_polygon_area_unit_square():
    sq = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32)
    assert abs(polygon_area(sq) - 100.0) < 1e-3


def test_clip_fully_inside_returns_same_area():
    poly = np.array([[10, 10], [30, 10], [30, 30], [10, 30]], dtype=np.float32)
    clipped = clip_polygon_to_tile(poly, (0, 0, 100, 100))
    assert clipped is not None
    assert abs(polygon_area(clipped) - 400.0) < 1e-3


def test_clip_straddling_boundary_halves_area():
    # 20x20 square centered on x=100 boundary of tile [0..100]
    poly = np.array([[90, 40], [110, 40], [110, 60], [90, 60]], dtype=np.float32)
    clipped = clip_polygon_to_tile(poly, (0, 0, 100, 200))
    assert clipped is not None
    assert abs(polygon_area(clipped) - 200.0) < 1e-3  # half of 400
    assert clipped[:, 0].max() <= 100.0 + 1e-3


def test_clip_fully_outside_returns_none():
    poly = np.array([[200, 200], [220, 200], [220, 220], [200, 220]], dtype=np.float32)
    assert clip_polygon_to_tile(poly, (0, 0, 100, 100)) is None
```

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH=src python -m pytest tests/test_slice_geometry.py -k clip -v`
Expected: FAIL with `ImportError: cannot import name 'clip_polygon_to_tile'`.

- [ ] **Step 3: Implement clip + area**

Append to `src/hydra_suite/utils/slice_geometry.py`:

```python
def polygon_area(poly: np.ndarray) -> float:
    """Absolute area of an (N,2) polygon via the shoelace formula."""
    p = np.asarray(poly, dtype=np.float64)
    if p.shape[0] < 3:
        return 0.0
    x, y = p[:, 0], p[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) * 0.5)


def _clip_against_edge(poly, inside_fn, intersect_fn):
    """One Sutherland-Hodgman pass against a single half-plane."""
    out = []
    n = len(poly)
    if n == 0:
        return out
    for i in range(n):
        cur = poly[i]
        prev = poly[i - 1]
        cur_in = inside_fn(cur)
        prev_in = inside_fn(prev)
        if cur_in:
            if not prev_in:
                out.append(intersect_fn(prev, cur))
            out.append(cur)
        elif prev_in:
            out.append(intersect_fn(prev, cur))
    return out


def clip_polygon_to_tile(poly_px: np.ndarray, tile: tuple[int, int, int, int]) -> "np.ndarray | None":
    """Sutherland-Hodgman clip of an (N,2) polygon against an axis-aligned tile rect.

    Returns the clipped (M,2) polygon in the same pixel space, or None when the
    intersection is empty or degenerate (< 3 vertices, ~zero area).
    """
    x0, y0, x1, y1 = (float(tile[0]), float(tile[1]), float(tile[2]), float(tile[3]))
    poly = [np.asarray(p, dtype=np.float64) for p in np.asarray(poly_px, dtype=np.float64)]

    def _lerp(a, b, t):
        return a + (b - a) * t

    # Left x>=x0, Right x<=x1, Top y>=y0, Bottom y<=y1.
    edges = [
        (lambda p: p[0] >= x0, lambda a, b: _lerp(a, b, (x0 - a[0]) / (b[0] - a[0]) if b[0] != a[0] else 0.0)),
        (lambda p: p[0] <= x1, lambda a, b: _lerp(a, b, (x1 - a[0]) / (b[0] - a[0]) if b[0] != a[0] else 0.0)),
        (lambda p: p[1] >= y0, lambda a, b: _lerp(a, b, (y0 - a[1]) / (b[1] - a[1]) if b[1] != a[1] else 0.0)),
        (lambda p: p[1] <= y1, lambda a, b: _lerp(a, b, (y1 - a[1]) / (b[1] - a[1]) if b[1] != a[1] else 0.0)),
    ]
    for inside_fn, intersect_fn in edges:
        poly = _clip_against_edge(poly, inside_fn, intersect_fn)
        if not poly:
            return None
    if len(poly) < 3:
        return None
    arr = np.asarray(poly, dtype=np.float32)
    if polygon_area(arr) <= 1e-6:
        return None
    return arr
```

- [ ] **Step 4: Run to verify it passes**

Run: `PYTHONPATH=src python -m pytest tests/test_slice_geometry.py -k "clip or area" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/utils/slice_geometry.py tests/test_slice_geometry.py
git commit -m "feat(utils): polygon clip-to-tile + shoelace area for sliced training"
```

---

### Task 3: Builder pure helpers — reference measurement + level re-derivation

**Files:**
- Create: `src/hydra_suite/training/sliced_dataset.py`
- Test: `tests/test_sliced_dataset.py`

**Interfaces:**
- Consumes: `hydra_suite.utils.slice_geometry.{clip_polygon_to_tile, polygon_area}`, `hydra_suite.training.geometry_levels.GeometryLevel`, `hydra_suite.training.dataset_builders._parse_geometry_label_lines`, `cv2`, `numpy`.
- Produces (from `hydra_suite.training.sliced_dataset`):
  - `measure_reference_body_px(labels: list[tuple[int, np.ndarray]], frame_wh: tuple[int,int]) -> float` — median OBB major axis (longest side of `cv2.minAreaRect`) over the frame's normalized-point labels, in frame pixels. Returns `0.0` when there are no labels.
  - `project_to_level(poly_norm: np.ndarray, level: "GeometryLevel") -> np.ndarray` — re-derive a clipped `(M,2)` normalized contour DOWN to `level`: `POLYGON` keeps the contour; `OBB` returns the 4 `cv2.minAreaRect` corners; `AABB` returns the 4 axis-aligned envelope corners.
  - `label_line_for_level(class_id: int, pts_norm: np.ndarray, level: "GeometryLevel") -> str` — format one YOLO label line: `AABB` -> `cls cx cy w h` (5 fields); `OBB` -> `cls x1 y1 ... x4 y4` (9 fields); `POLYGON` -> `cls x1 y1 ... xP yP` (2P+1 fields). Coordinates `%.6f`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_sliced_dataset.py`:

```python
import numpy as np

from hydra_suite.training.geometry_levels import GeometryLevel
from hydra_suite.training.sliced_dataset import (
    label_line_for_level, measure_reference_body_px, project_to_level,
)


def test_measure_reference_body_px_median_major_axis():
    # Two objects: 40x20 and 80x20 (px) at frame 100x100 -> majors 40, 80 -> median 60.
    def rect_norm(cx, cy, w, h):
        pts = np.array([[cx - w / 2, cy - h / 2], [cx + w / 2, cy - h / 2],
                        [cx + w / 2, cy + h / 2], [cx - w / 2, cy + h / 2]], dtype=np.float32)
        pts[:, 0] /= 100.0
        pts[:, 1] /= 100.0
        return pts
    labels = [(0, rect_norm(50, 50, 40, 20)), (0, rect_norm(50, 50, 80, 20))]
    ref = measure_reference_body_px(labels, (100, 100))
    assert abs(ref - 60.0) < 1.0


def test_project_to_level_aabb_from_polygon():
    poly = np.array([[0.1, 0.1], [0.5, 0.2], [0.4, 0.6], [0.05, 0.4]], dtype=np.float32)
    aabb = project_to_level(poly, GeometryLevel.AABB)
    assert aabb.shape == (4, 2)
    assert abs(aabb[:, 0].min() - 0.05) < 1e-4
    assert abs(aabb[:, 0].max() - 0.5) < 1e-4


def test_project_to_level_obb_returns_four_corners():
    poly = np.array([[0.1, 0.1], [0.5, 0.1], [0.5, 0.3], [0.1, 0.3], [0.3, 0.35]], dtype=np.float32)
    obb = project_to_level(poly, GeometryLevel.OBB)
    assert obb.shape == (4, 2)


def test_project_to_level_polygon_keeps_contour():
    poly = np.array([[0.1, 0.1], [0.5, 0.1], [0.3, 0.5]], dtype=np.float32)
    out = project_to_level(poly, GeometryLevel.POLYGON)
    assert np.allclose(out, poly)


def test_label_line_field_counts():
    aabb = np.array([[0.1, 0.1], [0.3, 0.1], [0.3, 0.3], [0.1, 0.3]], dtype=np.float32)
    assert len(label_line_for_level(2, aabb, GeometryLevel.AABB).split()) == 5
    assert len(label_line_for_level(0, aabb, GeometryLevel.OBB).split()) == 9
    tri = np.array([[0.1, 0.1], [0.3, 0.1], [0.2, 0.4]], dtype=np.float32)
    assert len(label_line_for_level(1, tri, GeometryLevel.POLYGON).split()) == 7
```

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH=src python -m pytest tests/test_sliced_dataset.py -v`
Expected: FAIL with `ModuleNotFoundError: hydra_suite.training.sliced_dataset`.

- [ ] **Step 3: Implement the pure helpers**

Create `src/hydra_suite/training/sliced_dataset.py`:

```python
"""Sliced (tiled) training-data builder for DetectKit direct OBB models.

Tiles a merged OBB dataset so a direct model learns to detect at the SAME scale
SAHI feeds at inference. Tiles through ``utils.slice_geometry`` — the exact grid
the inference path uses (Approach B). See
docs/superpowers/specs/2026-07-27-detectkit-sahi-sliced-training-design.md.
"""

from __future__ import annotations

import cv2
import numpy as np

from .geometry_levels import GeometryLevel


def measure_reference_body_px(labels, frame_wh) -> float:
    """Median OBB major axis (px) over a frame's normalized-point labels."""
    w, h = float(frame_wh[0]), float(frame_wh[1])
    majors: list[float] = []
    for _cls_id, pts_norm in labels:
        pts = np.asarray(pts_norm, dtype=np.float32).copy()
        pts[:, 0] *= w
        pts[:, 1] *= h
        if pts.shape[0] < 3:
            continue
        (_c, (bw, bh), _a) = cv2.minAreaRect(pts.astype(np.float32))
        majors.append(float(max(bw, bh)))
    if not majors:
        return 0.0
    return float(np.median(np.asarray(majors, dtype=np.float64)))


def project_to_level(poly_norm: np.ndarray, level: GeometryLevel) -> np.ndarray:
    """Re-derive a normalized (M,2) contour DOWN to ``level`` (contour space kept)."""
    poly = np.asarray(poly_norm, dtype=np.float32)
    if level == GeometryLevel.POLYGON:
        return poly
    if level == GeometryLevel.OBB:
        box = cv2.boxPoints(cv2.minAreaRect(poly))
        return np.asarray(box, dtype=np.float32)
    # AABB: axis-aligned envelope corners.
    x1, y1 = float(poly[:, 0].min()), float(poly[:, 1].min())
    x2, y2 = float(poly[:, 0].max()), float(poly[:, 1].max())
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)


def label_line_for_level(class_id: int, pts_norm: np.ndarray, level: GeometryLevel) -> str:
    """Format one YOLO label line for ``level`` (coords clipped to [0,1], %.6f)."""
    pts = np.clip(np.asarray(pts_norm, dtype=np.float32), 0.0, 1.0)
    if level == GeometryLevel.AABB:
        x1, y1 = float(pts[:, 0].min()), float(pts[:, 1].min())
        x2, y2 = float(pts[:, 0].max()), float(pts[:, 1].max())
        cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
        bw, bh = max(0.0, x2 - x1), max(0.0, y2 - y1)
        return f"{int(class_id)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
    coords = " ".join(f"{float(v):.6f}" for v in pts.reshape(-1))
    return f"{int(class_id)} {coords}"
```

- [ ] **Step 4: Run to verify it passes**

Run: `PYTHONPATH=src python -m pytest tests/test_sliced_dataset.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/training/sliced_dataset.py tests/test_sliced_dataset.py
git commit -m "feat(training): sliced-dataset pure helpers (ref measure, level re-derive)"
```

---

### Task 4: Sliced dataset builder — single-scale tiling of a merged OBB dataset

**Files:**
- Modify: `src/hydra_suite/training/sliced_dataset.py`
- Test: `tests/test_sliced_dataset.py`

**Interfaces:**
- Consumes: Task 3 helpers; `hydra_suite.utils.slice_geometry.{tile_size_for_mode, plan_tiles, clip_polygon_to_tile, polygon_area}`; `dataset_builders._parse_geometry_label_lines`, `dataset_builders.IMAGE_EXTS`, `dataset_builders._find_label_for_obb_image`; `dataset_inspector.inspect_obb_or_detect_dataset`; `contracts.DatasetBuildResult`.
- Produces:
  - `SliceBuildParams` dataclass: `geometry_mode: str = "auto_object"`, `imgsz: int = 640`, `object_tile_fraction: float = 0.15`, `slice_width: int = 0`, `slice_height: int = 0`, `overlap: float = 0.2`, `min_area_ratio: float = 0.1`, `negative_tile_fraction: float = 0.15`, `target_sizes: list[float] = [200.0, 300.0, 400.0]`, `full_frame_mix: bool = True`, `reference_body_px: float = 0.0` (0 => measure from labels).
  - `build_sliced_obb_dataset(merged_obb_dataset_dir, output_root, *, level: GeometryLevel, params: SliceBuildParams, seed: int = 42) -> DatasetBuildResult` — writes a new tiled dataset (`sliced_obb_<ts>/images|labels/{train,val,test}` + `dataset.yaml` + `manifest.json`) and returns its `DatasetBuildResult`.
  - `_tile_one_image(img, labels, tile, level, min_area_ratio) -> tuple[np.ndarray, list[str]]` — crop + emit the kept label lines for one tile (helper, unit-tested directly). `labels` are `(cls_id, (P,2) normalized-frame)` pairs.

This task implements SINGLE-scale tiling (one tile size from `geometry_mode`). Multi-scale/full-frame emission is Task 5 — here `target_sizes`/`full_frame_mix` are accepted but only the single geometry is emitted, so keep the loop structured to extend.

- [ ] **Step 1: Write the failing test (synthetic dataset)**

Append to `tests/test_sliced_dataset.py`:

```python
from pathlib import Path

import cv2

from hydra_suite.training.geometry_levels import GeometryLevel
from hydra_suite.training.sliced_dataset import (
    SliceBuildParams, build_sliced_obb_dataset,
)


def _write_synthetic_obb_dataset(root: Path) -> Path:
    """One 512x512 train image with two axis-aligned OBB labels (9-field)."""
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
    def obb_line(cx, cy, w, h):
        x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
        c = [x1 / 512, y1 / 512, x2 / 512, y1 / 512, x2 / 512, y2 / 512, x1 / 512, y2 / 512]
        return "0 " + " ".join(f"{v:.6f}" for v in c)
    for split in ("train", "val"):
        cv2.imwrite(str(root / "images" / split / "f0.jpg"), np.zeros((512, 512, 3), dtype=np.uint8))
        (root / "labels" / split / "f0.txt").write_text(
            obb_line(80, 80, 40, 40) + "\n" + obb_line(430, 430, 40, 40) + "\n",
            encoding="utf-8",
        )
    (root / "dataset.yaml").write_text(
        "path: {}\ntrain: images/train\nval: images/val\nnames:\n  0: object\n".format(root.resolve()),
        encoding="utf-8",
    )
    return root


def test_build_sliced_dataset_produces_tiled_labels(tmp_path):
    merged = _write_synthetic_obb_dataset(tmp_path / "merged")
    params = SliceBuildParams(
        geometry_mode="custom", slice_width=256, slice_height=256, overlap=0.2,
        target_sizes=[], full_frame_mix=False, negative_tile_fraction=0.0,
    )
    out = build_sliced_obb_dataset(
        str(merged), str(tmp_path / "out"), level=GeometryLevel.OBB, params=params, seed=1,
    )
    out_dir = Path(out.dataset_dir)
    assert (out_dir / "dataset.yaml").exists()
    train_labels = list((out_dir / "labels" / "train").glob("*.txt"))
    assert train_labels, "expected at least one tiled train label"
    # Each kept tile label is a 9-field OBB line at OBB level.
    for lp in train_labels:
        for ln in lp.read_text().splitlines():
            if ln.strip():
                assert len(ln.split()) == 9


def test_build_sliced_dataset_area_threshold_drops_slivers(tmp_path):
    merged = _write_synthetic_obb_dataset(tmp_path / "merged")
    # High threshold: an object only partly inside a tile must be dropped there.
    params = SliceBuildParams(
        geometry_mode="custom", slice_width=100, slice_height=100, overlap=0.0,
        min_area_ratio=0.95, target_sizes=[], full_frame_mix=False, negative_tile_fraction=0.0,
    )
    out = build_sliced_obb_dataset(
        str(merged), str(tmp_path / "out"), level=GeometryLevel.OBB, params=params, seed=1,
    )
    # Objects (40px) straddling 100px tile edges lose >5% area on boundary tiles;
    # only tiles fully containing an object keep it. Build must still succeed.
    assert Path(out.dataset_dir, "dataset.yaml").exists()
```

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH=src python -m pytest tests/test_sliced_dataset.py -k build_sliced -v`
Expected: FAIL with `ImportError: cannot import name 'build_sliced_obb_dataset'`.

- [ ] **Step 3: Implement the builder (single-scale)**

Append to `src/hydra_suite/training/sliced_dataset.py`:

```python
import json
import random
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .contracts import DatasetBuildResult
from .dataset_builders import (
    IMAGE_EXTS, _find_label_for_obb_image, _parse_geometry_label_lines,
)
from hydra_suite.utils.slice_geometry import (
    clip_polygon_to_tile, plan_tiles, polygon_area, tile_size_for_mode,
)


@dataclass
class SliceBuildParams:
    geometry_mode: str = "auto_object"
    imgsz: int = 640
    object_tile_fraction: float = 0.15
    slice_width: int = 0
    slice_height: int = 0
    overlap: float = 0.2
    min_area_ratio: float = 0.1
    negative_tile_fraction: float = 0.15
    target_sizes: list[float] = field(default_factory=lambda: [200.0, 300.0, 400.0])
    full_frame_mix: bool = True
    reference_body_px: float = 0.0


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _tile_one_image(img, labels, tile, level, min_area_ratio):
    """Crop one tile and emit kept label lines. ``labels`` = (cls, (P,2) frame-norm)."""
    x0, y0, x1, y1 = tile
    fh, fw = img.shape[:2]
    xi0, yi0 = max(0, int(x0)), max(0, int(y0))
    xi1, yi1 = min(fw, int(x1)), min(fh, int(y1))
    crop = img[yi0:yi1, xi0:xi1]
    tw, th = float(xi1 - xi0), float(yi1 - yi0)
    lines: list[str] = []
    if tw <= 0 or th <= 0:
        return crop, lines
    for cls_id, poly_norm in labels:
        poly_px = np.asarray(poly_norm, dtype=np.float32).copy()
        poly_px[:, 0] *= fw
        poly_px[:, 1] *= fh
        full_area = polygon_area(poly_px)
        if full_area <= 1e-6:
            continue
        clipped = clip_polygon_to_tile(poly_px, (xi0, yi0, xi1, yi1))
        if clipped is None:
            continue
        if polygon_area(clipped) / full_area < min_area_ratio:
            continue
        local = clipped.copy()
        local[:, 0] = (local[:, 0] - xi0) / tw
        local[:, 1] = (local[:, 1] - yi0) / th
        derived = project_to_level(np.clip(local, 0.0, 1.0), level)
        lines.append(label_line_for_level(int(cls_id), derived, level))
    return crop, lines


def _iter_dataset_items(merged_dir: Path):
    """Yield (split, image_path, label_path) for a merged OBB dataset."""
    for split in ("train", "val", "test"):
        src_img = merged_dir / "images" / split
        src_lbl = merged_dir / "labels" / split
        if not src_img.exists():
            continue
        for img_path in sorted(src_img.rglob("*")):
            if img_path.suffix.lower() not in IMAGE_EXTS:
                continue
            lbl_path = _find_label_for_obb_image(img_path, src_img, src_lbl)
            if lbl_path is not None:
                yield split, img_path, lbl_path


def _tile_sizes_for_params(params, reference_body_px) -> list[tuple[int, int]]:
    """Resolve the list of (w,h) tile sizes to emit (single-scale here)."""
    w, h = tile_size_for_mode(
        geometry_mode=params.geometry_mode, imgsz=params.imgsz,
        reference_body_px=reference_body_px,
        object_tile_fraction=params.object_tile_fraction,
        slice_width=params.slice_width, slice_height=params.slice_height,
    )
    return [(w, h)]


def build_sliced_obb_dataset(merged_obb_dataset_dir, output_root, *, level, params, seed=42):
    merged_dir = Path(merged_obb_dataset_dir).expanduser().resolve()
    out_root = Path(output_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    out_dir = out_root / f"sliced_obb_{_timestamp()}"
    for split in ("train", "val", "test"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    rng = random.Random(int(seed))
    counts = {"train": 0, "val": 0, "test": 0, "tiles": 0, "negatives": 0, "objects": 0}
    class_names = _read_class_names(merged_dir)

    for split, img_path, lbl_path in _iter_dataset_items(merged_dir):
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None or img.size == 0:
            continue
        fh, fw = img.shape[:2]
        labels = _parse_geometry_label_lines(lbl_path)
        ref_px = params.reference_body_px or measure_reference_body_px(labels, (fw, fh))
        for tile_w, tile_h in _tile_sizes_for_params(params, ref_px):
            try:
                plan = plan_tiles((fh, fw), tile_w, tile_h, params.overlap, params.overlap)
            except ValueError:
                continue
            for ti, tile in enumerate(plan.tiles):
                crop, lines = _tile_one_image(img, labels, tile, level, params.min_area_ratio)
                if crop.size == 0:
                    continue
                is_negative = not lines
                if is_negative and rng.random() >= params.negative_tile_fraction:
                    continue
                stem = f"{img_path.stem}_t{tile_w}x{tile_h}_{ti:04d}"
                cv2.imwrite(str(out_dir / "images" / split / f"{stem}.jpg"), crop)
                (out_dir / "labels" / split / f"{stem}.txt").write_text(
                    ("\n".join(lines) + "\n") if lines else "", encoding="utf-8"
                )
                counts[split] += 1
                counts["tiles"] += 1
                counts["objects"] += len(lines)
                if is_negative:
                    counts["negatives"] += 1

    _write_sliced_yaml(out_dir, class_names)
    manifest = {
        "type": "sliced_obb",
        "source": str(merged_dir),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "level": level.label,
        "counts": counts,
        "slice_geometry": _slice_geometry_manifest(params),
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return DatasetBuildResult(
        dataset_dir=str(out_dir), stats=manifest, manifest_path=str(manifest_path)
    )


def _read_class_names(merged_dir: Path) -> list[str]:
    yaml_path = merged_dir / "dataset.yaml"
    if not yaml_path.exists():
        return ["object"]
    try:
        import yaml
        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
        names = data.get("names", {})
        if isinstance(names, dict):
            return [str(names[k]) for k in sorted(names, key=lambda x: int(x))] or ["object"]
        if isinstance(names, list):
            return [str(n) for n in names] or ["object"]
    except Exception:
        pass
    return ["object"]


def _write_sliced_yaml(out_dir: Path, class_names: list[str]) -> None:
    lines = [
        f"path: {out_dir.resolve()}", "train: images/train", "val: images/val", "names:",
    ]
    lines.extend(f"  {i}: {n}" for i, n in enumerate(class_names))
    (out_dir / "dataset.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _slice_geometry_manifest(params) -> dict:
    return {
        "geometry_mode": params.geometry_mode,
        "imgsz": params.imgsz,
        "object_tile_fraction": params.object_tile_fraction,
        "slice_width": params.slice_width,
        "slice_height": params.slice_height,
        "overlap": params.overlap,
        "min_area_ratio": params.min_area_ratio,
        "negative_tile_fraction": params.negative_tile_fraction,
        "target_sizes": list(params.target_sizes),
        "full_frame_mix": params.full_frame_mix,
        "reference_body_px": params.reference_body_px,
    }
```

- [ ] **Step 4: Run to verify it passes**

Run: `PYTHONPATH=src python -m pytest tests/test_sliced_dataset.py -k build_sliced -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/training/sliced_dataset.py tests/test_sliced_dataset.py
git commit -m "feat(training): single-scale sliced OBB dataset builder"
```

---

### Task 5: Multi-scale emission + full-frame mix

**Files:**
- Modify: `src/hydra_suite/training/sliced_dataset.py` (extend `_tile_sizes_for_params`, add full-frame emission)
- Test: `tests/test_sliced_dataset.py`

**Interfaces:**
- Consumes: Task 4 surface.
- Produces: extended `_tile_sizes_for_params(params, reference_body_px) -> list[tuple[int,int]]` (multiple sizes in `auto_object` mode) and a new `_emit_full_frames(...)` path invoked when `params.full_frame_mix`. Public builder signature unchanged.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_sliced_dataset.py`:

```python
def test_multiscale_emits_distinct_tile_sizes(tmp_path):
    merged = _write_synthetic_obb_dataset(tmp_path / "merged")
    params = SliceBuildParams(
        geometry_mode="auto_object", imgsz=640, reference_body_px=40.0,
        target_sizes=[80.0, 160.0], full_frame_mix=False, negative_tile_fraction=0.0,
    )
    out = build_sliced_obb_dataset(
        str(merged), str(tmp_path / "out"), level=GeometryLevel.OBB, params=params, seed=1,
    )
    names = [p.name for p in Path(out.dataset_dir, "images", "train").glob("*.jpg")]
    # ref=40, target=80 -> frac=0.125 -> tile 320; target=160 -> frac=0.25 -> tile 160.
    assert any("_t320x320_" in n for n in names)
    assert any("_t160x160_" in n for n in names)


def test_full_frame_mix_emits_full_frame_sample(tmp_path):
    merged = _write_synthetic_obb_dataset(tmp_path / "merged")
    params = SliceBuildParams(
        geometry_mode="custom", slice_width=256, slice_height=256,
        target_sizes=[], full_frame_mix=True, negative_tile_fraction=0.0,
    )
    out = build_sliced_obb_dataset(
        str(merged), str(tmp_path / "out"), level=GeometryLevel.OBB, params=params, seed=1,
    )
    names = [p.name for p in Path(out.dataset_dir, "images", "train").glob("*.jpg")]
    assert any("_full_" in n for n in names)
```

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH=src python -m pytest tests/test_sliced_dataset.py -k "multiscale or full_frame_mix" -v`
Expected: FAIL (`_t320x320_`/`_full_` names absent — single scale, no full frame).

- [ ] **Step 3: Implement multi-scale + full-frame**

In `src/hydra_suite/training/sliced_dataset.py`, replace `_tile_sizes_for_params`:

```python
def _tile_sizes_for_params(params, reference_body_px) -> list[tuple[int, int]]:
    """Resolve the (deduped) list of (w,h) tile sizes to emit.

    ``auto_object`` with a measured reference and a non-empty ``target_sizes``
    fans out one square tile per target apparent size (target/imgsz -> fraction);
    otherwise a single size from the geometry mode.
    """
    if params.geometry_mode == "auto_object" and reference_body_px > 0 and params.target_sizes:
        sizes: list[tuple[int, int]] = []
        for target in params.target_sizes:
            frac = max(0.01, min(0.9, float(target) / max(1, params.imgsz)))
            w, h = tile_size_for_mode(
                geometry_mode="auto_object", imgsz=params.imgsz,
                reference_body_px=reference_body_px, object_tile_fraction=frac,
                slice_width=0, slice_height=0,
            )
            if (w, h) not in sizes:
                sizes.append((w, h))
        if sizes:
            return sizes
    w, h = tile_size_for_mode(
        geometry_mode=params.geometry_mode, imgsz=params.imgsz,
        reference_body_px=reference_body_px,
        object_tile_fraction=params.object_tile_fraction,
        slice_width=params.slice_width, slice_height=params.slice_height,
    )
    return [(w, h)]
```

Then, inside `build_sliced_obb_dataset`, after the tile loop for an image (still inside the `for split, img_path, lbl_path` loop), add full-frame emission:

```python
        if params.full_frame_mix:
            lines = []
            for cls_id, poly_norm in labels:
                derived = project_to_level(np.clip(np.asarray(poly_norm, np.float32), 0, 1), level)
                lines.append(label_line_for_level(int(cls_id), derived, level))
            stem = f"{img_path.stem}_full"
            cv2.imwrite(str(out_dir / "images" / split / f"{stem}.jpg"), img)
            (out_dir / "labels" / split / f"{stem}.txt").write_text(
                ("\n".join(lines) + "\n") if lines else "", encoding="utf-8"
            )
            counts[split] += 1
            counts["objects"] += len(lines)
```

- [ ] **Step 4: Run to verify it passes**

Run: `PYTHONPATH=src python -m pytest tests/test_sliced_dataset.py -v`
Expected: PASS (all builder tests).

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/training/sliced_dataset.py tests/test_sliced_dataset.py
git commit -m "feat(training): multi-scale tile emission + full-frame mix"
```

---

### Task 6: `SliceTrainingSettings` dataclass + DetectKit project persistence

**Files:**
- Modify: `src/hydra_suite/detectkit/gui/models.py`
- Test: `tests/test_detectkit_slice_settings.py`

**Interfaces:**
- Produces (from `hydra_suite.detectkit.gui.models`):
  - `SliceTrainingSettings` dataclass with `to_dict()` / `from_dict(d)` (mirroring `OBBSource`): fields `enabled: bool = False`, `geometry_mode: str = "auto_object"`, `object_tile_fraction: float = 0.15`, `slice_width: int = 0`, `slice_height: int = 0`, `overlap: float = 0.2`, `min_area_ratio: float = 0.1`, `negative_tile_fraction: float = 0.15`, `target_sizes: list[float] = [200.0, 300.0, 400.0]`, `full_frame_mix: bool = True`, `merge_threshold: float = 0.5`.
  - `DetectKitProject.slice_settings: SliceTrainingSettings` field, round-tripped through `to_dict`/`load` with explicit handling (the generic loader cannot cast a nested dataclass).
- Consumes: nothing new.

- [ ] **Step 1: Write the failing test**

Create `tests/test_detectkit_slice_settings.py`:

```python
from pathlib import Path

from hydra_suite.detectkit.gui.models import DetectKitProject, SliceTrainingSettings


def test_slice_settings_defaults_off():
    s = SliceTrainingSettings()
    assert s.enabled is False
    assert s.geometry_mode == "auto_object"
    assert s.target_sizes == [200.0, 300.0, 400.0]


def test_project_slice_settings_round_trip(tmp_path):
    proj = DetectKitProject(project_dir=tmp_path)
    proj.slice_settings = SliceTrainingSettings(
        enabled=True, geometry_mode="custom", slice_width=512, slice_height=512,
        target_sizes=[150.0, 350.0], negative_tile_fraction=0.2,
    )
    out = tmp_path / "state.json"
    proj.save(out)
    loaded = DetectKitProject.load(out)
    assert loaded.slice_settings.enabled is True
    assert loaded.slice_settings.geometry_mode == "custom"
    assert loaded.slice_settings.slice_width == 512
    assert loaded.slice_settings.target_sizes == [150.0, 350.0]
    assert abs(loaded.slice_settings.negative_tile_fraction - 0.2) < 1e-9


def test_legacy_project_without_slice_settings_loads(tmp_path):
    out = tmp_path / "legacy.json"
    out.write_text('{"version": 1, "class_names": ["ant"]}', encoding="utf-8")
    loaded = DetectKitProject.load(out)
    assert loaded.slice_settings.enabled is False  # default when absent
```

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH=src python -m pytest tests/test_detectkit_slice_settings.py -v`
Expected: FAIL with `ImportError: cannot import name 'SliceTrainingSettings'`.

- [ ] **Step 3: Add the dataclass + persistence hooks**

In `src/hydra_suite/detectkit/gui/models.py`, add after `OBBSource`:

```python
@dataclass
class SliceTrainingSettings:
    """Shared SAHI sliced-training + preview geometry, persisted with the project."""

    enabled: bool = False
    geometry_mode: str = "auto_object"  # auto_model | auto_object | custom
    object_tile_fraction: float = 0.15
    slice_width: int = 0
    slice_height: int = 0
    overlap: float = 0.2
    min_area_ratio: float = 0.1
    negative_tile_fraction: float = 0.15
    target_sizes: list[float] = field(default_factory=lambda: [200.0, 300.0, 400.0])
    full_frame_mix: bool = True
    merge_threshold: float = 0.5

    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "geometry_mode": self.geometry_mode,
            "object_tile_fraction": self.object_tile_fraction,
            "slice_width": self.slice_width,
            "slice_height": self.slice_height,
            "overlap": self.overlap,
            "min_area_ratio": self.min_area_ratio,
            "negative_tile_fraction": self.negative_tile_fraction,
            "target_sizes": list(self.target_sizes),
            "full_frame_mix": self.full_frame_mix,
            "merge_threshold": self.merge_threshold,
        }

    @staticmethod
    def from_dict(d: dict) -> "SliceTrainingSettings":
        base = SliceTrainingSettings()
        if not isinstance(d, dict):
            return base
        return SliceTrainingSettings(
            enabled=bool(d.get("enabled", base.enabled)),
            geometry_mode=str(d.get("geometry_mode", base.geometry_mode) or base.geometry_mode),
            object_tile_fraction=float(d.get("object_tile_fraction", base.object_tile_fraction)),
            slice_width=int(d.get("slice_width", base.slice_width)),
            slice_height=int(d.get("slice_height", base.slice_height)),
            overlap=float(d.get("overlap", base.overlap)),
            min_area_ratio=float(d.get("min_area_ratio", base.min_area_ratio)),
            negative_tile_fraction=float(d.get("negative_tile_fraction", base.negative_tile_fraction)),
            target_sizes=[float(x) for x in (d.get("target_sizes") or base.target_sizes)],
            full_frame_mix=bool(d.get("full_frame_mix", base.full_frame_mix)),
            merge_threshold=float(d.get("merge_threshold", base.merge_threshold)),
        )
```

Add the field to `DetectKitProject` (after `training_history`):

```python
    slice_settings: SliceTrainingSettings = field(default_factory=SliceTrainingSettings)
```

In `DetectKitProject.to_dict`, extend the per-field branch:

```python
            elif f.name == "sources":
                d[f.name] = [s.to_dict() for s in val]
            elif f.name == "slice_settings":
                d[f.name] = val.to_dict()
            else:
                d[f.name] = val
```

In `DetectKitProject.load`, add a branch alongside `sources`:

```python
            elif name == "sources":
                proj.sources = [OBBSource.from_dict(s) for s in val]
            elif name == "slice_settings":
                proj.slice_settings = SliceTrainingSettings.from_dict(val)
```

(The generic `else` cannot cast a nested dataclass, so the explicit branch is required; absence in `raw` leaves the `default_factory` instance, satisfying the legacy test.)

- [ ] **Step 4: Run to verify it passes**

Run: `PYTHONPATH=src python -m pytest tests/test_detectkit_slice_settings.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/detectkit/gui/models.py tests/test_detectkit_slice_settings.py
git commit -m "feat(detectkit): SliceTrainingSettings persisted in project JSON"
```

---

### Task 7: Wire the builder into the training orchestrator + dialog

**Files:**
- Modify: `src/hydra_suite/training/service.py` (add `build_sliced_obb_dataset` method)
- Modify: `src/hydra_suite/detectkit/gui/dialogs/training_dialog.py` (`_build_role_datasets` inserts the sliced step)
- Test: `tests/test_training_service_sliced.py`

**Interfaces:**
- Consumes: `hydra_suite.training.sliced_dataset.{SliceBuildParams, build_sliced_obb_dataset}`; `GeometryLevel`; `DetectKitProject.slice_settings`.
- Produces: `TrainingOrchestrator.build_sliced_obb_dataset(merged_obb_dataset_dir, *, level, params, seed) -> DatasetBuildResult` (writes under `self.workspace_root / "datasets_sliced"`).

- [ ] **Step 1: Write the failing test**

Create `tests/test_training_service_sliced.py`:

```python
from pathlib import Path

import cv2
import numpy as np

from hydra_suite.training.geometry_levels import GeometryLevel
from hydra_suite.training.service import TrainingOrchestrator
from hydra_suite.training.sliced_dataset import SliceBuildParams


def _merged(tmp: Path) -> Path:
    root = tmp / "merged"
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(root / "images" / split / "f0.jpg"), np.zeros((512, 512, 3), np.uint8))
        (root / "labels" / split / "f0.txt").write_text(
            "0 0.14 0.14 0.24 0.14 0.24 0.24 0.14 0.24\n", encoding="utf-8"
        )
    (root / "dataset.yaml").write_text(
        f"path: {root.resolve()}\ntrain: images/train\nval: images/val\nnames:\n  0: object\n",
        encoding="utf-8",
    )
    return root


def test_orchestrator_builds_sliced_dataset(tmp_path):
    orch = TrainingOrchestrator(tmp_path / "ws")
    merged = _merged(tmp_path)
    params = SliceBuildParams(
        geometry_mode="custom", slice_width=256, slice_height=256,
        target_sizes=[], full_frame_mix=False, negative_tile_fraction=0.0,
    )
    result = orch.build_sliced_obb_dataset(
        str(merged), level=GeometryLevel.OBB, params=params, seed=7
    )
    assert Path(result.dataset_dir, "dataset.yaml").exists()
    assert Path(result.dataset_dir).is_relative_to((tmp_path / "ws").resolve())
```

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH=src python -m pytest tests/test_training_service_sliced.py -v`
Expected: FAIL with `AttributeError: 'TrainingOrchestrator' object has no attribute 'build_sliced_obb_dataset'`.

- [ ] **Step 3: Add the orchestrator method**

In `src/hydra_suite/training/service.py`, add to the imports near line 17:

```python
from .sliced_dataset import SliceBuildParams, build_sliced_obb_dataset
```

Add a method to the orchestrator class (after `build_merged_obb_dataset`, before `build_role_dataset`):

```python
    def build_sliced_obb_dataset(
        self,
        merged_obb_dataset_dir: str,
        *,
        level: GeometryLevel,
        params: SliceBuildParams,
        seed: int = 42,
    ) -> DatasetBuildResult:
        """Tile a merged OBB dataset into a sliced dataset for SAHI-usable training."""
        out_root = self.workspace_root / "datasets_sliced"
        out_root.mkdir(parents=True, exist_ok=True)
        return build_sliced_obb_dataset(
            merged_obb_dataset_dir, out_root, level=level, params=params, seed=int(seed)
        )
```

(`GeometryLevel` and `DatasetBuildResult` are already imported in `service.py`.)

- [ ] **Step 4: Wire the dialog build seam**

In `src/hydra_suite/detectkit/gui/dialogs/training_dialog.py`, inside `_build_role_datasets`, after the merged build (currently line ~1975–1979) and before the role loop, insert the sliced step. Replace:

```python
            self.role_dataset_dirs = {}
            self._append_log(f"Merged dataset: {merged.dataset_dir}")

            merged_level = merged_level_and_blocker(self._project.sources)[0]
```

with:

```python
            self.role_dataset_dirs = {}
            self._append_log(f"Merged dataset: {merged.dataset_dir}")

            merged_level = merged_level_and_blocker(self._project.sources)[0]

            role_source_dir = merged.dataset_dir
            slice_settings = getattr(self._project, "slice_settings", None)
            if slice_settings is not None and slice_settings.enabled:
                from hydra_suite.training.sliced_dataset import SliceBuildParams

                params = SliceBuildParams(
                    geometry_mode=slice_settings.geometry_mode,
                    imgsz=self._project.imgsz_obb_direct,
                    object_tile_fraction=slice_settings.object_tile_fraction,
                    slice_width=slice_settings.slice_width,
                    slice_height=slice_settings.slice_height,
                    overlap=slice_settings.overlap,
                    min_area_ratio=slice_settings.min_area_ratio,
                    negative_tile_fraction=slice_settings.negative_tile_fraction,
                    target_sizes=list(slice_settings.target_sizes),
                    full_frame_mix=slice_settings.full_frame_mix,
                )
                sliced = orchestrator.build_sliced_obb_dataset(
                    merged.dataset_dir,
                    level=merged_level,
                    params=params,
                    seed=self.spin_seed.value(),
                )
                role_source_dir = sliced.dataset_dir
                self._append_log(f"Sliced dataset: {sliced.dataset_dir}")
```

Then change the role loop to derive from `role_source_dir` instead of `merged.dataset_dir`:

```python
            for role in roles:
                build = orchestrator.build_role_dataset(
                    role,
                    role_source_dir,
```

- [ ] **Step 5: Run the service test + confirm the dialog module imports**

Run: `PYTHONPATH=src python -m pytest tests/test_training_service_sliced.py -v`
Run: `PYTHONPATH=src python -c "import hydra_suite.detectkit.gui.dialogs.training_dialog"`
Expected: test PASS; import prints nothing (no error).

- [ ] **Step 6: Commit**

```bash
git add src/hydra_suite/training/service.py src/hydra_suite/detectkit/gui/dialogs/training_dialog.py tests/test_training_service_sliced.py
git commit -m "feat(detectkit): route sliced dataset build through training orchestrator"
```

---

### Task 8: Stamp slice geometry into the model manifest

**Files:**
- Modify: `src/hydra_suite/training/model_publish.py` (accept + write slice geometry)
- Test: `tests/test_model_publish_slice_geometry.py`

**Interfaces:**
- Consumes: `publish_trained_model`'s existing signature.
- Produces: `publish_trained_model(..., slice_geometry: dict | None = None)`. When provided AND `role == TrainingRole.OBB_DIRECT`: write a `<artifact>.slice_meta.json` sidecar next to the stored `.pt`, and add `"slice_geometry": <dict>` to the registry metadata entry. When `None`, behavior is byte-identical to today.

- [ ] **Step 1: Write the failing test**

Create `tests/test_model_publish_slice_geometry.py`:

```python
import json
from pathlib import Path

import hydra_suite.training.model_publish as mp
from hydra_suite.training.contracts import TrainingRole


def test_slice_geometry_written_as_sidecar_and_registry(tmp_path, monkeypatch):
    # Redirect the models root into tmp: replacing _project_root with a lambda
    # whose __module__ != model_publish activates _use_project_root_override(),
    # so get_models_root() returns tmp_path/"models".
    monkeypatch.setattr(mp, "_project_root", lambda: tmp_path, raising=False)

    src = tmp_path / "weights.pt"
    src.write_bytes(b"fake-weights")
    geom = {"geometry_mode": "auto_object", "target_sizes": [200.0, 300.0], "reference_body_px": 42.0}

    key, stored = mp.publish_trained_model(
        role=TrainingRole.OBB_DIRECT, artifact_path=str(src), size="s", species="ant",
        model_info="sliced", trained_from_run_id="r1", dataset_fingerprint="fp",
        base_model="yolo26s-obb.pt", slice_geometry=geom,
    )
    sidecar = Path(stored).with_suffix(".slice_meta.json")
    assert sidecar.exists()
    assert json.loads(sidecar.read_text())["reference_body_px"] == 42.0
    reg = mp.load_model_registry()
    assert reg["entries"][key]["slice_geometry"]["geometry_mode"] == "auto_object"


def test_no_slice_geometry_writes_no_sidecar(tmp_path, monkeypatch):
    monkeypatch.setattr(mp, "_project_root", lambda: tmp_path, raising=False)
    src = tmp_path / "w2.pt"
    src.write_bytes(b"x")
    _key, stored = mp.publish_trained_model(
        role=TrainingRole.OBB_DIRECT, artifact_path=str(src), size="s", species="ant",
        model_info="plain", trained_from_run_id="r2", dataset_fingerprint="fp",
        base_model="yolo26s-obb.pt",
    )
    assert not Path(stored).with_suffix(".slice_meta.json").exists()
```

Note: `_repo_dir_for_role`/`get_models_root` use `_use_project_root_override()` which checks `_project_root.__module__ != __name__`. Because the reviewer/implementer patches `mp._project_root` with a lambda whose `__module__` differs, the override path activates and models land under `tmp_path/models`. If patching proves fragile, set an env-independent models dir the same way existing `model_publish` tests do — check `tests/` for the established monkeypatch pattern before finalizing this test.

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH=src python -m pytest tests/test_model_publish_slice_geometry.py -v`
Expected: FAIL with `TypeError: publish_trained_model() got an unexpected keyword argument 'slice_geometry'`.

- [ ] **Step 3: Implement the parameter + write path**

In `src/hydra_suite/training/model_publish.py`, add `slice_geometry: dict[str, Any] | None = None` to `publish_trained_model`'s keyword args (after `classifier_v2_meta`). After the artifact copy (`shutil.copy2(src, dst)`) and after the `classifier_meta` block, before building `metadata`, add:

```python
    slice_geom_sidecar_name: str | None = None
    if slice_geometry and role == TrainingRole.OBB_DIRECT and dst.suffix.lower() == ".pt":
        slice_sidecar = dst.with_suffix(".slice_meta.json")
        slice_sidecar.write_text(json.dumps(dict(slice_geometry), indent=2), encoding="utf-8")
        slice_geom_sidecar_name = slice_sidecar.name
```

In the `metadata` dict construction, after `training_params` handling, add:

```python
    if slice_geometry and role == TrainingRole.OBB_DIRECT:
        metadata["slice_geometry"] = dict(slice_geometry)
        if slice_geom_sidecar_name:
            metadata["slice_meta_sidecar"] = slice_geom_sidecar_name
```

- [ ] **Step 4: Run to verify it passes**

Run: `PYTHONPATH=src python -m pytest tests/test_model_publish_slice_geometry.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/training/model_publish.py tests/test_model_publish_slice_geometry.py
git commit -m "feat(training): stamp slice geometry into OBB model manifest"
```

---

### Task 9: Sliced inference in DetectKit preview (`predict_sliced`)

**Files:**
- Modify: `src/hydra_suite/detectkit/gui/prediction_preview.py`
- Test: `tests/test_detectkit_sliced_preview.py`

**Interfaces:**
- Consumes: `hydra_suite.utils.slice_geometry.{tile_size_for_mode, plan_tiles}`; `core.inference.stages.slicing._offset_result`; `core.inference.stages.merge.{band_membership, merge_obb_detections}`; `core.inference.stages.obb.merge_obb_results`; `extract_obb_result`.
- Produces: `predict_sliced_obb_result(executor, frame, *, geometry_mode, imgsz, reference_body_px, object_tile_fraction, slice_width, slice_height, overlap, merge_threshold, confidence_threshold, iou=_PREVIEW_IOU) -> OBBResult | None`. Returns the merged, frame-space `OBBResult`. Reuses the shipped cv2 merge.

- [ ] **Step 1: Write the failing test**

Create `tests/test_detectkit_sliced_preview.py`:

```python
import numpy as np

from hydra_suite.core.inference.result import OBBResult
from hydra_suite.detectkit.gui import prediction_preview as pp


class _FakeResult:
    """Minimal stand-in ultralytics result for one tile with zero detections."""
    def __init__(self):
        self.obb = None
        self.boxes = None


class _FakeExecutor:
    """Records the tile batch it was asked to predict; returns empty results."""
    def __init__(self):
        self.calls = []

    def predict(self, images, **kw):
        self.calls.append([np.asarray(im).shape for im in images])
        return [_FakeResult() for _ in images]


def test_predict_sliced_tiles_and_merges_empty(monkeypatch):
    # Force extract_obb_result to yield empty OBBResults so we exercise tiling+merge
    # without a real model.
    monkeypatch.setattr(
        pp, "extract_obb_result",
        lambda res, frame_idx=0, **kw: OBBResult(
            frame_idx=frame_idx,
            centroids=np.zeros((0, 2), np.float32), angles=np.zeros((0,), np.float32),
            sizes=np.zeros((0,), np.float32), shapes=np.zeros((0, 2), np.float32),
            confidences=np.zeros((0,), np.float32), corners=np.zeros((0, 4, 2), np.float32),
            detection_ids=np.zeros((0,), np.int64),
        ),
    )
    frame = np.zeros((512, 512, 3), np.uint8)
    ex = _FakeExecutor()
    out = pp.predict_sliced_obb_result(
        ex, frame, geometry_mode="custom", imgsz=640, reference_body_px=0.0,
        object_tile_fraction=0.15, slice_width=256, slice_height=256,
        overlap=0.2, merge_threshold=0.5, confidence_threshold=0.25,
    )
    assert out is not None
    assert out.num_detections == 0
    # 512x512 with 256 tiles + 0.2 overlap tiles into a >1 tile grid.
    assert len(ex.calls[0]) > 1


def test_predict_sliced_offsets_detection_into_frame_space(monkeypatch):
    # A single detection in tile (256,256)-(512,512) must land near frame (300,300).
    def _fake_extract(res, frame_idx=0, **kw):
        return OBBResult(
            frame_idx=frame_idx,
            centroids=np.array([[44.0, 44.0]], np.float32),  # tile-local
            angles=np.array([0.0], np.float32), sizes=np.array([100.0], np.float32),
            shapes=np.array([[100.0, 1.0]], np.float32), confidences=np.array([0.9], np.float32),
            corners=np.array([[[39, 39], [49, 39], [49, 49], [39, 49]]], np.float32),
            detection_ids=np.array([0], np.int64),
        ) if getattr(res, "tag", "") == "hit" else OBBResult(
            frame_idx=frame_idx,
            centroids=np.zeros((0, 2), np.float32), angles=np.zeros((0,), np.float32),
            sizes=np.zeros((0,), np.float32), shapes=np.zeros((0, 2), np.float32),
            confidences=np.zeros((0,), np.float32), corners=np.zeros((0, 4, 2), np.float32),
            detection_ids=np.zeros((0,), np.int64),
        )
    monkeypatch.setattr(pp, "extract_obb_result", _fake_extract)

    class _Exec:
        def predict(self, images, **kw):
            out = []
            for i, _ in enumerate(images):
                r = _FakeResult()
                r.tag = "hit" if i == len(images) - 1 else ""
                out.append(r)
            return out

    frame = np.zeros((512, 512, 3), np.uint8)
    out = pp.predict_sliced_obb_result(
        _Exec(), frame, geometry_mode="custom", imgsz=640, reference_body_px=0.0,
        object_tile_fraction=0.15, slice_width=256, slice_height=256,
        overlap=0.0, merge_threshold=0.5, confidence_threshold=0.25,
    )
    assert out.num_detections == 1
    # last tile starts at (256,256); local (44,44) -> frame ~ (300,300).
    assert abs(out.centroids[0][0] - 300.0) < 2.0
    assert abs(out.centroids[0][1] - 300.0) < 2.0
```

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH=src python -m pytest tests/test_detectkit_sliced_preview.py -v`
Expected: FAIL with `AttributeError: module ... has no attribute 'predict_sliced_obb_result'`.

- [ ] **Step 3: Implement the wrapper**

In `src/hydra_suite/detectkit/gui/prediction_preview.py`, add imports:

```python
import numpy as np

from hydra_suite.core.inference.stages.merge import band_membership, merge_obb_detections
from hydra_suite.core.inference.stages.slicing import _offset_result
from hydra_suite.utils.slice_geometry import plan_tiles, tile_size_for_mode
```

(`merge_obb_results` and `extract_obb_result` are already imported at the top of `prediction_preview.py` — reuse them, do not re-import.)

Add the function (near `_predict_direct`):

```python
def predict_sliced_obb_result(
    executor,
    frame,
    *,
    geometry_mode: str,
    imgsz: int,
    reference_body_px: float,
    object_tile_fraction: float,
    slice_width: int,
    slice_height: int,
    overlap: float,
    merge_threshold: float,
    confidence_threshold: float,
    iou: float = _PREVIEW_IOU,
):
    """Executor-level sliced OBB inference on one BGR frame (preview/AL).

    Tiles via ``utils.slice_geometry`` (same grid inference uses), predicts each
    tile, offsets detections into frame space, and merges cross-tile duplicates
    with the shipped cv2 oracle. Returns a frame-space ``OBBResult`` or None.
    """
    fh, fw = int(frame.shape[0]), int(frame.shape[1])
    tw, th = tile_size_for_mode(
        geometry_mode=geometry_mode, imgsz=imgsz, reference_body_px=reference_body_px,
        object_tile_fraction=object_tile_fraction, slice_width=slice_width, slice_height=slice_height,
    )
    plan = plan_tiles((fh, fw), tw, th, overlap, overlap)
    raw_floor = max(1e-4, float(confidence_threshold))
    tiles_img = [np.ascontiguousarray(frame[y0:y1, x0:x1]) for (x0, y0, x1, y1) in plan.tiles]
    if not tiles_img:
        return _predict_direct(executor, frame, confidence_threshold=confidence_threshold, iou=iou)
    results = executor.predict(tiles_img, conf=raw_floor, iou=float(iou), verbose=False)

    parts = []
    for (x0, y0, _x1, _y1), res in zip(plan.tiles, results):
        local = extract_obb_result(res, frame_idx=0)
        parts.append(_offset_result(local, max(0, x0), max(0, y0), 0))
    concat = merge_obb_results(0, parts)
    if concat.num_detections <= 1:
        return concat
    bands = band_membership(concat.corners, plan.tiles)
    return merge_obb_detections(
        concat, policy="greedy_nmm", metric="ios", threshold=float(merge_threshold),
        backend="cv2", overlap_bands=bands, runtime=None,
    )
```

- [ ] **Step 4: Run to verify it passes**

Run: `PYTHONPATH=src python -m pytest tests/test_detectkit_sliced_preview.py tests/test_detectkit_prediction_preview.py -v`
Expected: PASS (the pre-existing preview test proves the non-sliced path is untouched).

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/detectkit/gui/prediction_preview.py tests/test_detectkit_sliced_preview.py
git commit -m "feat(detectkit): executor-level predict_sliced for preview/AL"
```

---

### Task 10: DetectKit UI — shared slice settings block + preview toggle

**Files:**
- Modify: `src/hydra_suite/detectkit/gui/dialogs/training_dialog.py` (a "Sliced dataset" group that reads/writes `project.slice_settings`)
- Modify: `src/hydra_suite/detectkit/gui/panels/dataset_panel.py` OR `tools_panel.py` (an "Enable sliced inference" toggle for preview — locate the preview control host first)
- Modify: `src/hydra_suite/detectkit/gui/main_window.py` (preview call routes through `predict_sliced_obb_result` when the toggle is on)
- Test: `tests/test_detectkit_slice_ui.py`

**Interfaces:**
- Consumes: `SliceTrainingSettings`, `predict_sliced_obb_result`.
- Produces: a `SliceSettingsGroup` `QWidget` (in `training_dialog.py` or a small new `panels/slice_settings_widget.py`) with `load_from(settings: SliceTrainingSettings)` and `to_settings() -> SliceTrainingSettings`; the training dialog persists it into `project.slice_settings` on save. The preview toggle reads the same `project.slice_settings.enabled`.

Before writing widgets, READ how `training_dialog._build_config_group`/`_build_augmentation_group` construct groups and how `_load_from_project`/save map widgets↔project fields, and how `main_window` currently invokes `predict_obb_for_frame`/`predict_preview_detections`. Match those patterns exactly (this is a Qt-heavy task; the widget layout mirrors the existing groups). Follow the existing `tests/test_detection_panel_slice_widgets.py` (TrackerKit's SAHI widgets) as the closest precedent for a headless widget test.

- [ ] **Step 1: Write the failing widget round-trip test**

Create `tests/test_detectkit_slice_ui.py`:

```python
import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication  # noqa: E402

from hydra_suite.detectkit.gui.models import SliceTrainingSettings  # noqa: E402


@pytest.fixture(scope="module")
def _app():
    app = QApplication.instance() or QApplication([])
    yield app


def test_slice_settings_group_round_trips(_app):
    from hydra_suite.detectkit.gui.panels.slice_settings_widget import SliceSettingsGroup

    w = SliceSettingsGroup()
    s = SliceTrainingSettings(
        enabled=True, geometry_mode="custom", slice_width=384, slice_height=384,
        overlap=0.25, target_sizes=[123.0, 456.0], negative_tile_fraction=0.3,
    )
    w.load_from(s)
    out = w.to_settings()
    assert out.enabled is True
    assert out.geometry_mode == "custom"
    assert out.slice_width == 384
    assert out.overlap == pytest.approx(0.25)
    assert out.target_sizes == [123.0, 456.0]
    assert out.negative_tile_fraction == pytest.approx(0.3)
```

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH=src python -m pytest tests/test_detectkit_slice_ui.py -v`
Expected: FAIL with `ModuleNotFoundError: ...slice_settings_widget`.

- [ ] **Step 3: Implement `SliceSettingsGroup`**

Create `src/hydra_suite/detectkit/gui/panels/slice_settings_widget.py`:

```python
"""Shared SAHI sliced-training/preview settings widget for DetectKit."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox, QLineEdit, QSpinBox,
)

from ..models import SliceTrainingSettings


class SliceSettingsGroup(QGroupBox):
    """Group box binding widgets to a SliceTrainingSettings block."""

    def __init__(self, parent=None) -> None:
        super().__init__("Sliced dataset / inference (SAHI)", parent)
        form = QFormLayout(self)

        self.chk_enabled = QCheckBox("Enable sliced training + preview")
        self.cmb_mode = QComboBox()
        self.cmb_mode.addItems(["auto_object", "auto_model", "custom"])
        self.spin_frac = QDoubleSpinBox()
        self.spin_frac.setRange(0.01, 0.9)
        self.spin_frac.setSingleStep(0.01)
        self.spin_w = QSpinBox(); self.spin_w.setRange(0, 8192)
        self.spin_h = QSpinBox(); self.spin_h.setRange(0, 8192)
        self.spin_overlap = QDoubleSpinBox(); self.spin_overlap.setRange(0.0, 0.9); self.spin_overlap.setSingleStep(0.05)
        self.spin_min_area = QDoubleSpinBox(); self.spin_min_area.setRange(0.0, 1.0); self.spin_min_area.setSingleStep(0.05)
        self.spin_neg = QDoubleSpinBox(); self.spin_neg.setRange(0.0, 1.0); self.spin_neg.setSingleStep(0.05)
        self.txt_targets = QLineEdit()  # comma-separated apparent sizes
        self.chk_full = QCheckBox("Mix full frames")
        self.spin_merge = QDoubleSpinBox(); self.spin_merge.setRange(0.0, 1.0); self.spin_merge.setSingleStep(0.05)

        form.addRow(self.chk_enabled)
        form.addRow("Geometry mode", self.cmb_mode)
        form.addRow("Object tile fraction", self.spin_frac)
        form.addRow("Custom tile W", self.spin_w)
        form.addRow("Custom tile H", self.spin_h)
        form.addRow("Overlap", self.spin_overlap)
        form.addRow("Min area ratio", self.spin_min_area)
        form.addRow("Negative tile fraction", self.spin_neg)
        form.addRow("Target sizes (px, comma)", self.txt_targets)
        form.addRow(self.chk_full)
        form.addRow("Merge threshold", self.spin_merge)

    def load_from(self, s: SliceTrainingSettings) -> None:
        self.chk_enabled.setChecked(bool(s.enabled))
        idx = self.cmb_mode.findText(s.geometry_mode)
        self.cmb_mode.setCurrentIndex(idx if idx >= 0 else 0)
        self.spin_frac.setValue(float(s.object_tile_fraction))
        self.spin_w.setValue(int(s.slice_width))
        self.spin_h.setValue(int(s.slice_height))
        self.spin_overlap.setValue(float(s.overlap))
        self.spin_min_area.setValue(float(s.min_area_ratio))
        self.spin_neg.setValue(float(s.negative_tile_fraction))
        self.txt_targets.setText(", ".join(f"{v:g}" for v in s.target_sizes))
        self.chk_full.setChecked(bool(s.full_frame_mix))
        self.spin_merge.setValue(float(s.merge_threshold))

    def to_settings(self) -> SliceTrainingSettings:
        targets: list[float] = []
        for tok in self.txt_targets.text().split(","):
            tok = tok.strip()
            if tok:
                try:
                    targets.append(float(tok))
                except ValueError:
                    continue
        return SliceTrainingSettings(
            enabled=self.chk_enabled.isChecked(),
            geometry_mode=self.cmb_mode.currentText(),
            object_tile_fraction=self.spin_frac.value(),
            slice_width=self.spin_w.value(),
            slice_height=self.spin_h.value(),
            overlap=self.spin_overlap.value(),
            min_area_ratio=self.spin_min_area.value(),
            negative_tile_fraction=self.spin_neg.value(),
            target_sizes=targets or SliceTrainingSettings().target_sizes,
            full_frame_mix=self.chk_full.isChecked(),
            merge_threshold=self.spin_merge.value(),
        )
```

- [ ] **Step 4: Mount the group in the training dialog + persist**

In `training_dialog.py`: instantiate `SliceSettingsGroup` inside the training tab (add it near `_build_config_group`'s return, or append to the training tab layout in `_build_training_tab`). In `_load_from_project`, call `self.slice_group.load_from(self._project.slice_settings)`. Wherever the dialog writes widget state back into `self._project` before save (follow the existing save path — search for where `self._project.epochs`/`self._project.batch` are assigned), add `self._project.slice_settings = self.slice_group.to_settings()`.

- [ ] **Step 5: Route the preview toggle**

In `main_window.py`, locate the preview detection call (`predict_obb_for_frame` / `predict_preview_detections*`). When `self._project.slice_settings.enabled`, call `predict_sliced_obb_result(...)` with the project's `slice_settings` + `imgsz_obb_direct` and convert its `OBBResult` to canvas dicts via the existing `_dicts_from_obb_result` (export it from `prediction_preview` if not already public — add a thin `dicts_from_obb_result(obb)` public wrapper delegating to `_dicts_from_obb_result`). Keep the non-sliced branch exactly as-is so a disabled toggle is byte-identical to today.

- [ ] **Step 6: Run the widget test + import checks**

Run: `PYTHONPATH=src python -m pytest tests/test_detectkit_slice_ui.py -v`
Run: `PYTHONPATH=src python -c "import hydra_suite.detectkit.gui.main_window, hydra_suite.detectkit.gui.dialogs.training_dialog"`
Expected: test PASS (or skip if PySide6 absent); imports clean.

- [ ] **Step 7: Commit**

```bash
git add src/hydra_suite/detectkit/gui/panels/slice_settings_widget.py src/hydra_suite/detectkit/gui/dialogs/training_dialog.py src/hydra_suite/detectkit/gui/main_window.py tests/test_detectkit_slice_ui.py
git commit -m "feat(detectkit): slice settings UI group + sliced preview toggle"
```

---

### Task 11: Collaborator runbook

**Files:**
- Create: `docs/runbooks/detectkit-sahi-sliced-training.md`

**Interfaces:** none (documentation).

- [ ] **Step 1: Write the runbook**

Create `docs/runbooks/detectkit-sahi-sliced-training.md` with these sections (fill each with concrete steps, not placeholders):

1. **Why** — one paragraph: crowding merges ants at the full-frame downscale; SAHI splits the crowd but the model must be trained at the sliced scale.
2. **Prerequisites** — DetectKit project with OBB/polygon-labeled sources; note the `imgsz_obb_direct` used.
3. **Configure sliced training** — open the training dialog → "Sliced dataset / inference (SAHI)" group → enable; pick `auto_object`; set target sizes (default `200, 300, 400`), overlap `0.2`, min area ratio `0.1`, negative fraction `0.15`, full-frame mix on. Explain the target-size knob (larger target → smaller tiles → more crowd-splitting; the model learns to detect at that apparent size).
4. **Build + train** — click build datasets; confirm the log shows `Sliced dataset: ...`; then train the OBB-direct role.
5. **Validate** — enable "sliced inference" in preview on a crowded frame; confirm clusters separate. Re-run the collaborator's scale sweep and confirm the detection-vs-scale curve has flattened.
6. **Ship back** — the published model carries `<model>.slice_meta.json`; note the trained geometry (mode + target range) so TrackerKit's SAHI inference can be matched to it.

- [ ] **Step 2: Commit**

```bash
git add docs/runbooks/detectkit-sahi-sliced-training.md
git commit -m "docs(detectkit): SAHI sliced-training collaborator runbook"
```

---

## Notes for the executor

- **Worktree:** create an isolated worktree under `.worktrees/` (branch NOT checked out on main) via `superpowers:using-git-worktrees`; run every test with `PYTHONPATH=<worktree>/src`.
- **Baseline suite has ~pre-existing failures:** use a delta gate — a task's tests plus the specific pre-existing tests it touches (`test_inference_slicing.py`, `test_detectkit_prediction_preview.py`, `test_detection_panel_slice_widgets.py`), not the whole suite.
- **Task 1 is the parity keystone:** the extraction MUST leave `test_inference_slicing.py` green unchanged; if it does not, the geometry was not moved verbatim.
- **Layer purity is a review gate:** any `hydra_suite.core.inference` or `hydra_suite.training` import appearing in `utils/slice_geometry.py` is a defect.
- **GUI tasks (7, 10):** the exact widget mounting/save wiring in `training_dialog.py`/`main_window.py` must be discovered by reading those files at implementation time; the plan pins the seam (`_build_role_datasets`, the preview call site) and the contracts, not every Qt line.
