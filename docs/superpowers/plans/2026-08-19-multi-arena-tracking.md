# Multi-Arena Tracking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Track 10–100 physically separate arenas in one camera view completely independently — slots, assignment, identity, tracklet matching — from a single shared inference pass.

**Architecture:** Arena is a *static label*, not a control-flow change. Each ROI shape carries an `arena_id`; those rasterize into a `uint16` label image alongside the unchanged `ROI_MASK`. Track slots are laid out in contiguous per-arena blocks with a static `slot_arena` array. Assignment blocks cross-arena pairs with the existing `1e6` hard-reject sentinel, which makes Hungarian decompose exactly into per-arena problems. Identity gets one decoder per arena; post-processing groups by arena. `run_tracking`'s structure is untouched.

**Tech Stack:** Python 3, NumPy, Numba (`@njit` assignment kernels), OpenCV (rasterization), pandas (post-processing), scipy (`linear_sum_assignment`), PyQt (TrackerKit GUI), pytest.

## Global Constraints

- **Single-arena runs MUST be byte-identical to current `main`.** With one arena, `slot_arena` is uniform, no pair is arena-blocked, and there is one identity decoder — every new code path must degenerate to today's exactly. This is the primary regression gate.
- **Arena gating conditions on `track_arena[i] != meas_arena[j]`, never on `cost >= 1e6`.** Gating on the sentinel value would also skip *distance*-gated cells that exist today, changing debug confidence metrics. Arena-mismatch is the only legal predicate.
- **Animal count is one shared value across all arenas.** `MAX_TARGETS = n_arenas × animals_per_arena` is derived, never entered directly.
- **Legacy `roi_shapes` without `arena_id` all map to arena 0.** Multi-arena is opt-in, never inferred from shape count.
- **Track IDs and trajectory IDs stay globally unique** across arenas. Only the new `arena_id` CSV column distinguishes arenas.
- **Label images resize with `cv2.INTER_NEAREST` only.** Any interpolating resize invents arena ids at boundaries.
- **Core may not import app layers.** `core/tracking/arenas.py` imports nothing from `trackerkit`.
- Run `make format` before each commit; the repo has pre-commit hooks (black, isort, flake8).

## Deviation from the spec (deliberate, please note)

The spec's Touch point 1 put arena labeling inside `core/inference/stages/filtering.py` and flowed `arena_ids` through the detection cache. **This plan instead derives arena at the tracking layer** from measurement centroids and the static label image.

Rationale discovered while reading the code: `OBBResult` (`core/inference/result.py:20`) is reconstructed field-by-field by `_select()`, and its arrays are serialized into the `.npz` detection cache. Adding a field means touching the result dataclass, `_select`, three filter paths including the CUDA tensor path, the cache writer/reader, and potentially cache keys — which would invalidate every existing user cache. Arena is a pure function of `(centroid, label_image)`, both of which the tracking layer already holds, so deriving it there is mathematically identical and costs one vectorized gather per frame. Everything downstream of this decision is unchanged.

## File Structure

**New files:**

| File | Responsibility |
|---|---|
| `src/hydra_suite/core/tracking/arenas.py` | `ArenaLayout` — slot↔arena mapping, detection→arena lookup. Qt-free, no app-layer imports. |
| `src/hydra_suite/trackerkit/gui/dialogs/arena_grid_dialog.py` | Grid generator dialog (`BaseDialog` subclass) producing arena shapes. |
| `tests/test_arena_layout.py` | `ArenaLayout` unit tests. |
| `tests/test_arena_mask_build.py` | Label-image rasterization tests. |
| `tests/test_arena_blocked_assignment.py` | Blocked assignment ≡ per-arena Hungarian. |
| `tests/test_arena_identity_decoders.py` | Per-arena decoder isolation. |
| `tests/test_arena_postproc_grouping.py` | Post-processing arena grouping. |
| `tests/test_arena_grid_dialog.py` | Grid generator geometry. |
| `tests/test_arena_tiling_oracle.py` | End-to-end independence oracle. |

**Modified files:**

| File | Change |
|---|---|
| `src/hydra_suite/trackerkit/engine_params.py:193` | `build_arena_labels()` next to `build_roi_mask()`; emit `ARENA_LABELS`, `N_ARENAS`, `ANIMALS_PER_ARENA`; derive `MAX_TARGETS`. |
| `src/hydra_suite/core/assigners/hungarian.py` | Arena arrays through `compute_cost_matrix`, the numba kernel, the Python fallback, `_apply_bayesian_identity_cost`, `_assign_respawn`. |
| `src/hydra_suite/core/tracking/worker.py` | Build `ArenaLayout`; per-frame detection→arena gather; per-arena decoder registry; `arena_id` on emitted rows. |
| `src/hydra_suite/core/post/processing.py` | `resolve_trajectories` / `process_trajectories_from_csv` arena grouping. |
| `src/hydra_suite/core/individual/identity/offline.py` | Per-arena uniqueness solve. |
| `src/hydra_suite/trackerkit/config/schemas.py:25` | `animals_per_arena` field. |
| `src/hydra_suite/trackerkit/gui/orchestrators/session.py:2112` | Arena selector on shape creation. |

---

### Task 1: Arena label rasterization

**Files:**
- Modify: `src/hydra_suite/trackerkit/engine_params.py` (after `build_roi_mask`, ends line 240)
- Test: `tests/test_arena_mask_build.py`

**Interfaces:**
- Consumes: existing `build_roi_mask(roi_shapes, width, height)`.
- Produces: `build_arena_labels(roi_shapes, width, height) -> tuple[np.ndarray | None, int]` returning `(label_image_uint16, n_arenas)`. Pixel value is `arena_id + 1`; 0 means outside every arena. Returns `(None, 1)` when there are no shapes.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_arena_mask_build.py
import numpy as np
import pytest

from hydra_suite.trackerkit.engine_params import build_arena_labels, build_roi_mask


def _circle(cx, cy, r, arena_id=None, mode="include"):
    shape = {"type": "circle", "params": [cx, cy, r], "mode": mode}
    if arena_id is not None:
        shape["arena_id"] = arena_id
    return shape


def test_legacy_shapes_without_arena_id_collapse_to_one_arena():
    """Back-compat: three shapes, no arena_id -> a single arena 0."""
    shapes = [_circle(20, 20, 10), _circle(60, 20, 10), _circle(20, 60, 10)]
    labels, n_arenas = build_arena_labels(shapes, 100, 100)
    assert n_arenas == 1
    assert set(np.unique(labels)) == {0, 1}


def test_distinct_arena_ids_produce_distinct_labels():
    shapes = [_circle(20, 20, 10, arena_id=0), _circle(60, 20, 10, arena_id=1)]
    labels, n_arenas = build_arena_labels(shapes, 100, 100)
    assert n_arenas == 2
    assert labels[20, 20] == 1
    assert labels[20, 60] == 2
    assert labels[50, 50] == 0


def test_exclusion_hole_is_outside_every_arena():
    shapes = [_circle(50, 50, 30, arena_id=0), _circle(50, 50, 10, mode="exclude")]
    labels, _ = build_arena_labels(shapes, 100, 100)
    assert labels[50, 50] == 0       # inside the hole
    assert labels[50, 75] == 1       # in the annulus


def test_label_union_matches_roi_mask_exactly():
    """Invariant: (labels > 0) is pixel-identical to the existing ROI mask."""
    shapes = [
        _circle(30, 30, 15, arena_id=0),
        _circle(70, 70, 15, arena_id=1),
        _circle(30, 30, 5, mode="exclude"),
    ]
    labels, _ = build_arena_labels(shapes, 100, 100)
    roi = build_roi_mask(shapes, 100, 100)
    np.testing.assert_array_equal(labels > 0, roi > 0)


def test_no_shapes_returns_none():
    assert build_arena_labels([], 100, 100) == (None, 1)
    assert build_arena_labels(None, 100, 100) == (None, 1)


def test_arena_ids_are_densified():
    """Sparse ids (0, 5, 9) become contiguous labels 1, 2, 3."""
    shapes = [
        _circle(20, 20, 8, arena_id=0),
        _circle(50, 20, 8, arena_id=5),
        _circle(80, 20, 8, arena_id=9),
    ]
    labels, n_arenas = build_arena_labels(shapes, 100, 100)
    assert n_arenas == 3
    assert sorted(np.unique(labels).tolist()) == [0, 1, 2, 3]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_arena_mask_build.py -v`
Expected: FAIL — `ImportError: cannot import name 'build_arena_labels'`

- [ ] **Step 3: Write minimal implementation**

Insert after `build_roi_mask` in `src/hydra_suite/trackerkit/engine_params.py`:

```python
def build_arena_labels(
    roi_shapes: list[dict[str, Any]] | None,
    width: int | None,
    height: int | None,
) -> tuple[np.ndarray | None, int]:
    """Rasterize ROI shapes into a uint16 arena-label image.

    Pixel value is ``arena_id + 1``; 0 means outside every arena. The set
    ``labels > 0`` is pixel-identical to ``build_roi_mask`` on the same shapes,
    so detection gating semantics are unchanged.

    Shapes without an ``arena_id`` key map to arena 0 -- a legacy project that
    drew several shapes as one region keeps single-arena behavior exactly.
    Sparse ids are densified to a contiguous 0..n-1 range.
    """
    if not roi_shapes or not width or not height:
        return None, 1

    includes = [s for s in roi_shapes if s.get("mode", "include") != "exclude"]
    raw_ids = sorted({int(s.get("arena_id", 0)) for s in includes}) or [0]
    dense = {raw: i for i, raw in enumerate(raw_ids)}

    labels = np.zeros((height, width), np.uint16)
    for shape in includes:
        value = dense[int(shape.get("arena_id", 0))] + 1
        _fill_shape(labels, shape, value)
    for shape in roi_shapes:
        if shape.get("mode", "include") == "exclude":
            _fill_shape(labels, shape, 0)
    return labels, len(raw_ids)


def _fill_shape(canvas: np.ndarray, shape: dict[str, Any], value: int) -> None:
    """Rasterize one ROI shape onto *canvas* with the given fill value."""
    if shape.get("type") == "circle":
        center_x, center_y, radius = shape.get("params", [0, 0, 0])
        cv2.circle(canvas, (int(center_x), int(center_y)), int(radius), value, -1)
    elif shape.get("type") == "polygon":
        points = np.array(shape.get("params", []), dtype=np.int32)
        if len(points) > 0:
            cv2.fillPoly(canvas, [points], value)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_arena_mask_build.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
make format
git add src/hydra_suite/trackerkit/engine_params.py tests/test_arena_mask_build.py
git commit -m "feat(arena): rasterize ROI shapes into a uint16 arena-label image"
```

---

### Task 2: ArenaLayout — slot mapping and detection lookup

**Files:**
- Create: `src/hydra_suite/core/tracking/arenas.py`
- Test: `tests/test_arena_layout.py`

**Interfaces:**
- Consumes: the label image from Task 1.
- Produces:
  - `ArenaLayout(n_arenas: int, animals_per_arena: int, label_image: np.ndarray | None)`
  - `.max_targets -> int` = `n_arenas * animals_per_arena`
  - `.slot_arena -> np.ndarray` int32, shape `(max_targets,)`, contiguous blocks
  - `.arena_of_points(xy: np.ndarray) -> np.ndarray` int32, shape `(M,)`, `-1` outside
  - `.is_single_arena -> bool`
  - `.label_image_for_size(w, h) -> np.ndarray | None` — cached `INTER_NEAREST` resize

- [ ] **Step 1: Write the failing test**

```python
# tests/test_arena_layout.py
import numpy as np
import pytest

from hydra_suite.core.tracking.arenas import ArenaLayout


def _labels():
    """100x100: arena 0 = left half rows 0-49, arena 1 = right half rows 0-49."""
    labels = np.zeros((100, 100), np.uint16)
    labels[0:50, 0:50] = 1
    labels[0:50, 50:100] = 2
    return labels


def test_slot_arena_is_contiguous_blocks():
    layout = ArenaLayout(n_arenas=3, animals_per_arena=2, label_image=None)
    assert layout.max_targets == 6
    np.testing.assert_array_equal(layout.slot_arena, [0, 0, 1, 1, 2, 2])


def test_single_arena_layout_is_flagged():
    layout = ArenaLayout(n_arenas=1, animals_per_arena=4, label_image=None)
    assert layout.is_single_arena
    assert layout.max_targets == 4
    np.testing.assert_array_equal(layout.slot_arena, [0, 0, 0, 0])


def test_arena_of_points_maps_centroids():
    layout = ArenaLayout(n_arenas=2, animals_per_arena=1, label_image=_labels())
    xy = np.array([[10.0, 10.0], [80.0, 10.0], [10.0, 80.0]], dtype=np.float32)
    np.testing.assert_array_equal(layout.arena_of_points(xy), [0, 1, -1])


def test_arena_of_points_clips_out_of_frame_coordinates():
    """Mirrors filter_with_indices:300 -- coordinates are clipped, never wrapped."""
    layout = ArenaLayout(n_arenas=2, animals_per_arena=1, label_image=_labels())
    xy = np.array([[-5.0, 10.0], [500.0, 10.0]], dtype=np.float32)
    np.testing.assert_array_equal(layout.arena_of_points(xy), [0, 1])


def test_arena_of_points_without_label_image_is_all_zero():
    layout = ArenaLayout(n_arenas=1, animals_per_arena=4, label_image=None)
    xy = np.array([[10.0, 10.0], [90.0, 90.0]], dtype=np.float32)
    np.testing.assert_array_equal(layout.arena_of_points(xy), [0, 0])


def test_empty_detection_array_returns_empty():
    layout = ArenaLayout(n_arenas=2, animals_per_arena=1, label_image=_labels())
    out = layout.arena_of_points(np.zeros((0, 2), dtype=np.float32))
    assert out.shape == (0,)


def test_resize_uses_nearest_and_invents_no_labels():
    layout = ArenaLayout(n_arenas=2, animals_per_arena=1, label_image=_labels())
    small = layout.label_image_for_size(50, 50)
    assert small.shape == (50, 50)
    assert set(np.unique(small).tolist()) <= {0, 1, 2}


def test_resize_result_is_cached():
    layout = ArenaLayout(n_arenas=2, animals_per_arena=1, label_image=_labels())
    assert layout.label_image_for_size(50, 50) is layout.label_image_for_size(50, 50)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_arena_layout.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'hydra_suite.core.tracking.arenas'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/hydra_suite/core/tracking/arenas.py
"""Arena layout: the static slot<->arena mapping and detection->arena lookup.

An arena is a labelled ROI region. Arena membership is a *static* property --
of a track slot for its whole life, and of a detection via its centroid -- so
independent per-arena tracking needs no control-flow change, only this label.

Qt-free and app-layer-free by the Core dependency rule.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class ArenaLayout:
    """Slot<->arena mapping plus the frame-space arena label image.

    Slots are laid out in contiguous per-arena blocks: with 3 arenas of 2
    animals, slots 0-1 belong to arena 0, 2-3 to arena 1, 4-5 to arena 2.
    """

    n_arenas: int
    animals_per_arena: int
    label_image: np.ndarray | None = None
    _resize_cache: dict = field(default_factory=dict, repr=False, compare=False)

    @property
    def max_targets(self) -> int:
        return int(self.n_arenas) * int(self.animals_per_arena)

    @property
    def is_single_arena(self) -> bool:
        return int(self.n_arenas) <= 1

    @property
    def slot_arena(self) -> np.ndarray:
        """(max_targets,) int32 arena id per track slot."""
        return np.repeat(
            np.arange(self.n_arenas, dtype=np.int32), self.animals_per_arena
        )

    def label_image_for_size(self, width: int, height: int) -> np.ndarray | None:
        """Label image at (width, height), nearest-neighbour resized and cached.

        INTER_NEAREST is mandatory: any interpolating resize would blend
        neighbouring arena ids and invent labels at arena boundaries.
        """
        if self.label_image is None:
            return None
        if self.label_image.shape[:2] == (height, width):
            return self.label_image
        key = (width, height)
        cached = self._resize_cache.get(key)
        if cached is None:
            cached = cv2.resize(
                self.label_image, (width, height), interpolation=cv2.INTER_NEAREST
            )
            self._resize_cache[key] = cached
        return cached

    def arena_of_points(self, xy: np.ndarray) -> np.ndarray:
        """Arena id per point; -1 for points outside every arena.

        Without a label image every point is arena 0, so single-arena runs take
        an identical path to today's.
        """
        xy = np.asarray(xy, dtype=np.float32).reshape(-1, 2)
        if xy.shape[0] == 0:
            return np.zeros(0, dtype=np.int32)
        if self.label_image is None:
            return np.zeros(xy.shape[0], dtype=np.int32)
        h, w = self.label_image.shape[:2]
        labels = self.label_image_for_size(w, h)
        cx = np.clip(xy[:, 0].astype(np.int32), 0, w - 1)
        cy = np.clip(xy[:, 1].astype(np.int32), 0, h - 1)
        return labels[cy, cx].astype(np.int32) - 1
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_arena_layout.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
make format
git add src/hydra_suite/core/tracking/arenas.py tests/test_arena_layout.py
git commit -m "feat(arena): ArenaLayout slot mapping and detection->arena lookup"
```

---

### Task 3: Arena-blocked cost matrix

**Files:**
- Modify: `src/hydra_suite/core/assigners/hungarian.py:32` (`_compute_cost_matrix_numba`), `:462` (`compute_cost_matrix`), `:1085` (`_compute_cost_python_fallback`)
- Test: `tests/test_arena_blocked_assignment.py`

**Interfaces:**
- Consumes: `ArenaLayout.slot_arena` (Task 2).
- Produces: `TrackAssigner` accepts `track_arena: np.ndarray | None` and `meas_arena: np.ndarray | None`. Both `None` (the default) means single-arena — no gating, byte-identical to today. `_compute_cost_matrix_numba` gains two trailing `int32` array parameters.

**Why this works:** the kernel already writes `1e6` for distance-gated pairs (`hungarian.py:86`) and `_assign_established_hungarian` *rejects* any solved pair with `cost >= 1e6` (`hungarian.py:749`). The sentinel is a hard reject, not a large finite cost, so blocking off-block cells makes the global Hungarian solution exactly equal to independent per-arena solutions.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_arena_blocked_assignment.py
import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment

from hydra_suite.core.assigners.hungarian import _compute_cost_matrix_numba

BLOCKED = 1e6


def _kernel_args(n, m, rng):
    return dict(
        meas_pos=rng.uniform(0, 500, (m, 2)).astype(np.float32),
        meas_ori=rng.uniform(-np.pi, np.pi, m).astype(np.float32),
        pred_pos=rng.uniform(0, 500, (n, 2)).astype(np.float32),
        pred_ori=rng.uniform(-np.pi, np.pi, n).astype(np.float32),
        shapes_area=rng.uniform(10, 50, m).astype(np.float32),
        shapes_asp=rng.uniform(1, 3, m).astype(np.float32),
        prev_areas=rng.uniform(10, 50, n).astype(np.float32),
        prev_asps=rng.uniform(1, 3, n).astype(np.float32),
        S_inv_batch=np.tile(np.eye(3, dtype=np.float32), (n, 1, 1)),
        use_maha=False,
        Wp=1.0,
        Wo=0.1,
        Wa=0.01,
        Wasp=0.01,
        per_track_gates=np.full(n, 1e9, dtype=np.float32),
        meas_ori_directed=np.ones(m, dtype=np.int32),
    )


def _cost(n, m, track_arena, meas_arena, seed=0):
    rng = np.random.default_rng(seed)
    return _compute_cost_matrix_numba(
        n, m, track_arena=track_arena, meas_arena=meas_arena, **_kernel_args(n, m, rng)
    )


def test_cross_arena_pairs_are_blocked():
    track_arena = np.array([0, 0, 1, 1], dtype=np.int32)
    meas_arena = np.array([0, 1, 1, 0], dtype=np.int32)
    cost = _cost(4, 4, track_arena, meas_arena)
    for i in range(4):
        for j in range(4):
            if track_arena[i] != meas_arena[j]:
                assert cost[i, j] >= BLOCKED, f"({i},{j}) should be blocked"


def test_same_arena_pairs_are_unchanged_by_blocking():
    """Blocking must not perturb the cost of any within-arena pair."""
    track_arena = np.array([0, 0, 1, 1], dtype=np.int32)
    meas_arena = np.array([0, 1, 1, 0], dtype=np.int32)
    blocked = _cost(4, 4, track_arena, meas_arena)
    ungated = _cost(4, 4, None, None)
    same = track_arena[:, None] == meas_arena[None, :]
    np.testing.assert_array_equal(blocked[same], ungated[same])


def test_none_arena_arrays_reproduce_current_behaviour_exactly():
    """The single-arena path must be bit-identical to no gating at all."""
    uniform = np.zeros(4, dtype=np.int32)
    np.testing.assert_array_equal(
        _cost(4, 4, None, None), _cost(4, 4, uniform, np.zeros(4, dtype=np.int32))
    )


def test_detections_outside_every_arena_are_blocked_from_all_tracks():
    track_arena = np.array([0, 1], dtype=np.int32)
    meas_arena = np.array([-1, 0], dtype=np.int32)  # -1 == outside
    cost = _cost(2, 2, track_arena, meas_arena)
    assert cost[0, 0] >= BLOCKED and cost[1, 0] >= BLOCKED


@pytest.mark.parametrize("seed", range(10))
def test_blocked_hungarian_equals_independent_per_arena_hungarian(seed):
    """The core correctness claim: block-diagonal solve == per-arena solves."""
    n_arenas, per_arena, dets_per_arena = 4, 3, 3
    n, m = n_arenas * per_arena, n_arenas * dets_per_arena
    track_arena = np.repeat(np.arange(n_arenas), per_arena).astype(np.int32)
    meas_arena = np.repeat(np.arange(n_arenas), dets_per_arena).astype(np.int32)

    cost = _cost(n, m, track_arena, meas_arena, seed=seed)
    rows, cols = linear_sum_assignment(cost)
    joint = {int(r): int(c) for r, c in zip(rows, cols) if cost[r, c] < BLOCKED}

    separate = {}
    for a in range(n_arenas):
        tr = np.flatnonzero(track_arena == a)
        dt = np.flatnonzero(meas_arena == a)
        sub = cost[np.ix_(tr, dt)]
        r_sub, c_sub = linear_sum_assignment(sub)
        for r, c in zip(r_sub, c_sub):
            if sub[r, c] < BLOCKED:
                separate[int(tr[r])] = int(dt[c])

    assert joint == separate
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_arena_blocked_assignment.py -v`
Expected: FAIL — `TypeError: _compute_cost_matrix_numba() got an unexpected keyword argument 'track_arena'`

- [ ] **Step 3: Write minimal implementation**

In `_compute_cost_matrix_numba` (`hungarian.py:32`), add two parameters at the end of the signature and make the arena test the **first** statement of the inner loop:

```python
def _compute_cost_matrix_numba(
    N,
    M,
    meas_pos,
    meas_ori,
    pred_pos,
    pred_ori,
    shapes_area,
    shapes_asp,
    prev_areas,
    prev_asps,
    S_inv_batch,
    use_maha,
    Wp,
    Wo,
    Wa,
    Wasp,
    per_track_gates,
    meas_ori_directed,
    track_arena,
    meas_arena,
):
    """...existing docstring...

    ``track_arena``/``meas_arena`` are int32 arrays of length N and M. When
    both are length-0 sentinels, no arena gating is applied and the result is
    bit-identical to the pre-multi-arena kernel. Cross-arena pairs get the same
    ``1e6`` hard-reject sentinel used for distance-gated pairs, which makes the
    downstream Hungarian solve decompose exactly into per-arena problems.
    """
    cost = np.zeros((N, M), dtype=np.float32)
    gate_arenas = track_arena.shape[0] == N and meas_arena.shape[0] == M

    for i in range(N):
        inv_S_pos = S_inv_batch[i, :2, :2]
        gate_i = per_track_gates[i]
        arena_i = track_arena[i] if gate_arenas else 0

        for j in range(M):
            # Arena gating first: skips all downstream work for blocked pairs,
            # which is what keeps 100 arenas tractable.
            if gate_arenas and meas_arena[j] != arena_i:
                cost[i, j] = 1e6
                continue

            diff = meas_pos[j] - pred_pos[i]
            # ... rest of the loop body UNCHANGED ...
```

Add a module-level sentinel and a normalizer:

```python
_NO_ARENA = np.zeros(0, dtype=np.int32)


def _arena_arrays(track_arena, meas_arena, N, M):
    """Normalize optional arena arrays to numba-safe int32 arrays.

    Returns the length-0 sentinel pair when gating is off, which the kernel
    detects by shape and skips entirely.
    """
    if track_arena is None or meas_arena is None:
        return _NO_ARENA, _NO_ARENA
    ta = np.asarray(track_arena, dtype=np.int32)
    ma = np.asarray(meas_arena, dtype=np.int32)
    if ta.shape[0] != N or ma.shape[0] != M:
        return _NO_ARENA, _NO_ARENA
    return ta, ma
```

In `TrackAssigner.__init__` (`hungarian.py:154`) store the slot mapping:

```python
        self.track_arena = None   # set by the worker via set_track_arena()
```

```python
    def set_track_arena(self, track_arena) -> None:
        """Install the static per-slot arena mapping (None disables gating)."""
        self.track_arena = (
            None if track_arena is None else np.asarray(track_arena, dtype=np.int32)
        )
```

In `compute_cost_matrix` (`hungarian.py:462`), accept `meas_arena=None`, resolve the pair via `_arena_arrays(self.track_arena, meas_arena, N, M)`, and pass both to the kernel and to `_compute_cost_python_fallback`. In the fallback (`hungarian.py:1085`), skip blocked pairs inside the candidate loop:

```python
        for r, det_indices in candidates.items():
            inv_S = S_inv[r, :2, :2]
            arena_r = ta[r] if ta.shape[0] == N else None
            for c in det_indices:
                if arena_r is not None and ma[c] != arena_r:
                    continue    # stays at the 1e6 initialization value
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_arena_blocked_assignment.py tests/test_track_assigner.py tests/test_density_aware_assignment.py -v`
Expected: all pass — the two existing suites confirm the ungated path is unchanged.

- [ ] **Step 5: Clear the numba cache and re-run**

```bash
find . -name "__pycache__" -path "*assigners*" -exec rm -rf {} + 2>/dev/null; true
python -m pytest tests/test_arena_blocked_assignment.py -v
```

A stale `@jit(cache=True)` entry has previously masked an algorithm change in this repo and faked an equivalence pass. Never trust an assignment-kernel result without clearing the cache after a signature change.

- [ ] **Step 6: Commit**

```bash
make format
git add src/hydra_suite/core/assigners/hungarian.py tests/test_arena_blocked_assignment.py
git commit -m "feat(arena): arena-blocked cost matrix in the assignment kernel"
```

---

### Task 4: Arena gating in the identity-cost and respawn paths

**Files:**
- Modify: `src/hydra_suite/core/assigners/hungarian.py:266` (`_apply_bayesian_identity_cost`), `:811` (`_assign_respawn`), `:966` (`assign_tracks`)
- Test: `tests/test_arena_blocked_assignment.py` (extend)

**Interfaces:**
- Consumes: `self.track_arena` (Task 3), `meas_arena` per frame.
- Produces: `assign_tracks(..., meas_arena=None)` — the respawn phases honour arena boundaries.

**Why this task exists:** the cost matrix alone is not sufficient. `_assign_respawn` has two paths that bypass `cost` and compute distances directly from `meas`: the proximity respawn (`hungarian.py:952`, `np.linalg.norm(meas[c][:2] - last_pos)`) and the identity-rejoin budget check (`_within_budget`, `hungarian.py:877`). Without explicit gating, a lost track in arena 3 could respawn onto a detection in arena 7.

**Critical constraint:** in `_apply_bayesian_identity_cost` the skip predicate MUST be `track_arena[i] != meas_arena[j]`, **not** `cost[i, j] >= 1e6`. The latter would also skip distance-gated cells that exist on `main` today, changing the values feeding `compute_assignment_confidence` (`hungarian.py:697`) and breaking byte-identity of the debug confidence columns.

- [ ] **Step 1: Write the failing test (append to `tests/test_arena_blocked_assignment.py`)**

```python
def test_bayesian_identity_cost_skips_only_cross_arena_pairs():
    """Distance-gated within-arena cells must still receive the identity addon,
    or debug confidence metrics drift from main."""
    from hydra_suite.core.assigners.hungarian import TrackAssigner

    params = {
        "ENABLE_IDENTITY_ONLINE_DECODER": True,
        "ASSOCIATION_IDENTITY_HINT_SCALE": 0.5,
        "MAX_DISTANCE_THRESHOLD": 100.0,
    }
    assigner = TrackAssigner(params)
    assigner.set_track_arena(np.array([0, 1], dtype=np.int32))

    log_post = np.log(np.array([0.7, 0.3]))
    log_like = np.log(np.array([0.6, 0.4]))
    association = {
        "identity_track_log_posteriors": {0: log_post, 1: log_post},
        "identity_detection_log_likelihoods": [log_like, log_like],
    }

    # Detections: 0 and 1 in arena 0, detection 2 in arena 1.
    # Cell (0,0) is distance-gated but WITHIN arena 0 -> must still get the addon.
    # Cell (0,2) is cross-arena -> must be left untouched.
    assigner.set_track_arena(np.array([0, 1], dtype=np.int32))
    cost = np.array([[1e6, 5.0, 1e6], [5.0, 5.0, 5.0]], dtype=np.float32)
    meas_arena = np.array([0, 0, 1], dtype=np.int32)
    association["identity_detection_log_likelihoods"] = [log_like] * 3
    assigner._apply_bayesian_identity_cost(cost, association, meas_arena=meas_arena)

    assert cost[0, 0] > 1e6, "within-arena gated cell must still get the addon"
    assert cost[0, 2] == pytest.approx(1e6), "cross-arena cell must be left alone"


def test_respawn_never_crosses_arenas():
    from hydra_suite.core.assigners.hungarian import TrackAssigner

    class _KF:
        # slot 0 in arena 0 at (10,10); slot 1 in arena 1 at (410,10)
        X = np.array([[10.0, 10.0, 0.0, 0.0, 0.0], [410.0, 10.0, 0.0, 0.0, 0.0]])

    params = {
        "MAX_DISTANCE_THRESHOLD": 1000.0,
        "KALMAN_MATURITY_AGE": 10,
        "W_POSITION": 1.0,
        "W_ORIENTATION": 0.1,
        "W_AREA": 0.01,
        "W_ASPECT": 0.01,
        "USE_MAHALANOBIS": False,
    }
    assigner = TrackAssigner(params)
    assigner.set_track_arena(np.array([0, 1], dtype=np.int32))

    # One detection, sitting in arena 1, near slot 1 but within MAX_DIST of slot 0.
    meas = [np.array([400.0, 10.0, 0.0])]
    cost = np.full((2, 1), 1e6, dtype=np.float32)
    rows, cols, _ = assigner._assign_respawn(
        cost=cost,
        N=2,
        meas=meas,
        track_states=["lost", "lost"],
        tracking_continuity=[0, 0],
        kf_manager=_KF(),
        spatial_candidates=None,
        association_data=None,
        committed_slot_identities=None,
        missed_frames=[5, 5],
        _lost=[0, 1],
        _M=1,
        _MAX_DIST=1000.0,
        _assigned_dets=set(),
        meas_arena=np.array([1], dtype=np.int32),
    )
    assert 0 not in rows, "arena-0 slot must not respawn on an arena-1 detection"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_arena_blocked_assignment.py -k "bayesian or respawn" -v`
Expected: FAIL — `TypeError: _apply_bayesian_identity_cost() got an unexpected keyword argument 'meas_arena'`

- [ ] **Step 3: Write minimal implementation**

`_apply_bayesian_identity_cost` — add `meas_arena=None` and skip on arena mismatch only:

```python
    def _apply_bayesian_identity_cost(
        self,
        cost: np.ndarray,
        association_data: Dict[str, Any] | None,
        meas_arena: np.ndarray | None = None,
    ) -> None:
        ...
        n_tracks, n_dets = cost.shape
        ta = self.track_arena
        gate = (
            ta is not None
            and meas_arena is not None
            and len(ta) == n_tracks
            and len(meas_arena) == n_dets
        )
        for i in range(n_tracks):
            log_post_i = track_log_posts.get(i)
            if log_post_i is None:
                continue
            arena_i = ta[i] if gate else None
            for j in range(min(n_dets, len(det_log_likes))):
                # Arena mismatch is the ONLY legal skip predicate here. Skipping
                # on `cost >= 1e6` would also skip distance-gated cells that
                # exist today and change compute_assignment_confidence output.
                if gate and meas_arena[j] != arena_i:
                    continue
                ...  # rest UNCHANGED
```

`_assign_respawn` — add `meas_arena=None` and gate both direct-distance paths. Gate at the two loop sites, where the detection *index* is in hand; do not try to gate inside `_within_budget`, which receives only a coordinate pair and cannot recover the index:

```python
        # identity-rejoin scoring loop
                    for dj in np.flatnonzero(row > log_threshold):
                        j = cand_dets[dj]
                        if gate and meas_arena[j] != ta[slot]:
                            continue
                        det_xy = np.asarray(meas[j][:2], dtype=np.float64)
```

```python
        # proximity respawn loop
        for c in unassigned:
            if not remaining_uncommitted:
                break
            best_r, best_c_val = None, 1e6
            for r in remaining_uncommitted:
                if gate and meas_arena[c] != ta[r]:
                    continue
                last_pos = kf_manager.X[r, :2]
```

with `gate` computed once at the top of `_assign_respawn`:

```python
        ta = self.track_arena
        gate = (
            ta is not None
            and meas_arena is not None
            and len(ta) >= N
            and len(meas_arena) >= _M
        )
```

`assign_tracks` — accept `meas_arena=None` and thread it to `_apply_bayesian_identity_cost` and `_assign_respawn`. Also thread it from `compute_cost_matrix`'s caller.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_arena_blocked_assignment.py tests/test_track_assigner.py tests/test_bayesian_identity_vectorization.py -v`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
make format
git add src/hydra_suite/core/assigners/hungarian.py tests/test_arena_blocked_assignment.py
git commit -m "feat(arena): gate identity-cost and respawn paths by arena"
```

---

### Task 5: Per-arena online identity decoders

**Files:**
- Modify: `src/hydra_suite/core/tracking/worker.py:1849` (decoder construction) and its ~12 call sites
- Create: `src/hydra_suite/core/tracking/identity/decoder_registry.py`
- Test: `tests/test_arena_identity_decoders.py`

**Interfaces:**
- Consumes: `ArenaLayout.slot_arena` (Task 2), `IdentityCatalog`.
- Produces: `ArenaDecoderRegistry(catalog, params, slot_arena)` with the *same* method surface the worker already calls on a bare decoder — `get_belief(slot)`, `clear_slot(...)`, `decay_absent_slot_beliefs(slots)`, `get_slot_log_posteriors(slots)`, `all_active_slots()`, `update_frame(...)` — each routing by `slot_arena[slot]`. This keeps `worker.py`'s ~12 call sites textually unchanged apart from the constructor.

**Design note:** `OnlineIdentityDecoder` is already self-contained and slot-keyed, so per-arena isolation needs no change *inside* the decoder. One decoder per arena over the shared catalog is exactly the "labels repeat per arena" semantic: each enforces "one ant A in *this* arena".

- [ ] **Step 1: Write the failing test**

```python
# tests/test_arena_identity_decoders.py
import numpy as np
import pytest

from hydra_suite.core.individual.identity.catalog import IdentityCatalog
from hydra_suite.core.tracking.identity.decoder_registry import ArenaDecoderRegistry


@pytest.fixture
def catalog():
    return IdentityCatalog.from_labels(["antA", "antB"])


@pytest.fixture
def params():
    return {"IDENTITY_ONLINE_COMMIT_THRESHOLD": 0.9}


def test_single_arena_registry_creates_one_decoder(catalog, params):
    reg = ArenaDecoderRegistry(catalog, params, np.zeros(4, dtype=np.int32))
    assert reg.n_decoders == 1


def test_registry_creates_one_decoder_per_arena(catalog, params):
    slot_arena = np.repeat([0, 1, 2], 2).astype(np.int32)
    reg = ArenaDecoderRegistry(catalog, params, slot_arena)
    assert reg.n_decoders == 3


def test_slot_routes_to_its_arena_decoder(catalog, params):
    slot_arena = np.array([0, 0, 1, 1], dtype=np.int32)
    reg = ArenaDecoderRegistry(catalog, params, slot_arena)
    assert reg.decoder_for_slot(0) is reg.decoder_for_slot(1)
    assert reg.decoder_for_slot(2) is reg.decoder_for_slot(3)
    assert reg.decoder_for_slot(0) is not reg.decoder_for_slot(2)


def test_update_frame_partitions_evidence_by_arena(catalog, params):
    """The whole point: each decoder enforces uniqueness over ITS arena only,
    so 'antA' can be assigned once in arena 0 and again in arena 1."""
    slot_arena = np.array([0, 0, 1, 1], dtype=np.int32)
    reg = ArenaDecoderRegistry(catalog, params, slot_arena)
    seen = {}

    for arena, dec in reg.decoders.items():
        dec.update_frame = (
            lambda frame_idx, ev, a=arena: (seen.__setitem__(a, sorted(ev)), {})[1]
        )

    reg.update_frame(0, {0: "e0", 1: "e1", 2: "e2", 3: "e3"})
    assert seen[0] == [0, 1], "arena 0 decoder must see only arena 0 slots"
    assert seen[1] == [2, 3], "arena 1 decoder must see only arena 1 slots"


def test_decay_absent_slots_partitions_by_arena(catalog, params):
    slot_arena = np.array([0, 0, 1, 1], dtype=np.int32)
    reg = ArenaDecoderRegistry(catalog, params, slot_arena)
    calls = {}

    for arena, dec in reg.decoders.items():
        dec.decay_absent_slot_beliefs = (
            lambda slots, a=arena: calls.__setitem__(a, list(slots))
        )

    reg.decay_absent_slot_beliefs([0, 2, 3])
    assert calls[0] == [0]
    assert calls[1] == [2, 3]


def test_get_slot_log_posteriors_merges_across_arenas(catalog, params):
    slot_arena = np.array([0, 0, 1, 1], dtype=np.int32)
    reg = ArenaDecoderRegistry(catalog, params, slot_arena)
    for arena, dec in reg.decoders.items():
        dec.get_slot_log_posteriors = lambda slots, a=arena: {s: a for s in slots}
    assert reg.get_slot_log_posteriors([0, 1, 2, 3]) == {0: 0, 1: 0, 2: 1, 3: 1}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_arena_identity_decoders.py -v`
Expected: FAIL — `ModuleNotFoundError: ...decoder_registry`

- [ ] **Step 3: Write minimal implementation**

```python
# src/hydra_suite/core/tracking/identity/decoder_registry.py
"""One OnlineIdentityDecoder per arena, behind the single-decoder call surface.

Identity labels repeat per arena -- arena 1 and arena 2 may each contain an
"antA" -- so the decoder's one-individual-one-track uniqueness constraint must
be scoped per arena. OnlineIdentityDecoder is already self-contained and
slot-keyed, so scoping needs no change inside it: one instance per arena over
the shared catalog is exactly the required semantic.

This registry exposes the same methods worker.py already calls on a bare
decoder, routing each by slot -> arena.
"""

from __future__ import annotations

import numpy as np

from hydra_suite.core.individual.identity.catalog import IdentityCatalog
from hydra_suite.core.individual.identity.online import OnlineIdentityDecoder


class ArenaDecoderRegistry:
    """Slot-routed collection of per-arena identity decoders."""

    def __init__(
        self,
        catalog: IdentityCatalog,
        params: dict,
        slot_arena: np.ndarray,
    ) -> None:
        self._catalog = catalog
        self.slot_arena = np.asarray(slot_arena, dtype=np.int32)
        self.decoders = {
            int(a): OnlineIdentityDecoder(catalog, params)
            for a in np.unique(self.slot_arena)
        }

    @property
    def n_decoders(self) -> int:
        return len(self.decoders)

    def decoder_for_slot(self, slot_index: int) -> OnlineIdentityDecoder:
        return self.decoders[int(self.slot_arena[slot_index])]

    def _group_by_arena(self, slots) -> dict[int, list[int]]:
        grouped: dict[int, list[int]] = {}
        for slot in slots:
            grouped.setdefault(int(self.slot_arena[slot]), []).append(int(slot))
        return grouped

    # --- single-decoder call surface -------------------------------------

    def get_belief(self, slot_index: int):
        return self.decoder_for_slot(slot_index).get_belief(slot_index)

    def clear_slot(self, slot_index: int, *args, **kwargs):
        return self.decoder_for_slot(slot_index).clear_slot(
            slot_index, *args, **kwargs
        )

    def decay_absent_slot_beliefs(self, absent_slots) -> None:
        for arena, slots in self._group_by_arena(absent_slots).items():
            self.decoders[arena].decay_absent_slot_beliefs(slots)

    def get_slot_log_posteriors(self, slots) -> dict:
        merged: dict = {}
        for arena, arena_slots in self._group_by_arena(slots).items():
            merged.update(self.decoders[arena].get_slot_log_posteriors(arena_slots))
        return merged

    def all_active_slots(self) -> list[int]:
        return sorted(s for d in self.decoders.values() for s in d.all_active_slots())

    def update_frame(self, frame_idx, slot_evidence, *args, **kwargs) -> dict:
        """Run each arena's decoder over only that arena's slots.

        ``slot_evidence`` is the worker's {slot_index: evidence} mapping; each
        decoder sees a partition of it, so uniqueness is enforced per arena.
        """
        merged: dict = {}
        for arena, slots in self._group_by_arena(slot_evidence.keys()).items():
            subset = {s: slot_evidence[s] for s in slots}
            result = self.decoders[arena].update_frame(
                frame_idx, subset, *args, **kwargs
            )
            if result:
                merged.update(result)
        return merged
```

Then in `worker.py:1849` replace the bare construction:

```python
                    _identity_online_decoder = ArenaDecoderRegistry(
                        _identity_catalog,
                        p,
                        _arena_layout.slot_arena,
                    )
```

The `_catalog` attribute accessed at `worker.py:2703`, `:2993`, `:3056` is preserved by the registry, so those sites need no edit.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_arena_identity_decoders.py tests/identity -v`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
make format
git add src/hydra_suite/core/tracking/identity/decoder_registry.py src/hydra_suite/core/tracking/worker.py tests/test_arena_identity_decoders.py
git commit -m "feat(arena): one online identity decoder per arena"
```

---

### Task 6: Worker wiring — layout, per-frame lookup, arena column

**Files:**
- Modify: `src/hydra_suite/core/tracking/worker.py:873` (kf construction), `:2134`/`:2210`/`:2270`/`:2308` (meas construction sites), `:3533` (row emission)
- Modify: `src/hydra_suite/trackerkit/engine_params.py:1138` (params dict)
- Test: `tests/test_arena_worker_wiring.py`

**Interfaces:**
- Consumes: `build_arena_labels` (Task 1), `ArenaLayout` (Task 2), `set_track_arena` (Task 3), `meas_arena` (Task 4), `ArenaDecoderRegistry` (Task 5).
- Produces: params keys `ARENA_LABELS` (`np.ndarray | None`), `N_ARENAS` (`int`), `ANIMALS_PER_ARENA` (`int`); `MAX_TARGETS` derived as their product. Tracking rows gain an `arena_id` integer column.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_arena_worker_wiring.py
import numpy as np
import pytest

from hydra_suite.trackerkit.engine_params import build_engine_params


def _cfg(**over):
    cfg = {
        "roi_shapes": [
            {"type": "circle", "params": [25, 25, 15], "mode": "include", "arena_id": 0},
            {"type": "circle", "params": [75, 25, 15], "mode": "include", "arena_id": 1},
        ],
        "animals_per_arena": 3,
        "frame_width": 100,
        "frame_height": 100,
    }
    cfg.update(over)
    return cfg


def test_max_targets_is_derived_from_arenas_and_animals():
    params = build_engine_params(_cfg())
    assert params["N_ARENAS"] == 2
    assert params["ANIMALS_PER_ARENA"] == 3
    assert params["MAX_TARGETS"] == 6


def test_arena_labels_are_emitted_and_agree_with_roi_mask():
    params = build_engine_params(_cfg())
    labels = params["ARENA_LABELS"]
    assert labels.dtype == np.uint16
    np.testing.assert_array_equal(labels > 0, params["ROI_MASK"] > 0)


def test_legacy_config_without_arena_ids_is_single_arena():
    cfg = _cfg(
        roi_shapes=[{"type": "circle", "params": [50, 50, 30], "mode": "include"}],
        animals_per_arena=4,
    )
    params = build_engine_params(cfg)
    assert params["N_ARENAS"] == 1
    assert params["MAX_TARGETS"] == 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_arena_worker_wiring.py -v`
Expected: FAIL — `KeyError: 'N_ARENAS'`

- [ ] **Step 3: Write minimal implementation**

In `engine_params.py`, next to the existing `"ROI_MASK": roi_mask` entry (line 1138):

```python
    arena_labels, n_arenas = build_arena_labels(cfg.get("roi_shapes"), width, height)
    animals_per_arena = int(cfg.get("animals_per_arena", cfg.get("MAX_TARGETS", 1)))
```

```python
        "ROI_MASK": roi_mask,
        "ARENA_LABELS": arena_labels,
        "N_ARENAS": n_arenas,
        "ANIMALS_PER_ARENA": animals_per_arena,
        "MAX_TARGETS": n_arenas * animals_per_arena,
```

In `worker.py`, build the layout beside the Kalman manager (line 873):

```python
        self.arena_layout = ArenaLayout(
            n_arenas=int(p.get("N_ARENAS", 1)),
            animals_per_arena=int(p.get("ANIMALS_PER_ARENA", p["MAX_TARGETS"])),
            label_image=p.get("ARENA_LABELS"),
        )
        self.kf_manager = KalmanFilterManager(p["MAX_TARGETS"], p)
        self.assigner.set_track_arena(
            None if self.arena_layout.is_single_arena else self.arena_layout.slot_arena
        )
```

Passing `None` for the single-arena case is deliberate: it takes the exact ungated path, protecting byte-identity.

At each of the four `meas = frame_result_to_meas(...)` sites (lines 2134, 2210, 2270, 2308), derive the per-frame arena vector immediately after:

```python
                    meas_arena = self.arena_layout.arena_of_points(_obb.centroids)
```

and for the empty-detection branches (lines 2180, 2352):

```python
                    meas_arena = np.zeros(0, dtype=np.int32)
```

Thread `meas_arena=meas_arena` into the `assign_tracks(...)` call. At row emission (near line 3533), add the arena of each track slot:

```python
                    "arena_id": int(self.arena_layout.slot_arena[track_idx]),
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_arena_worker_wiring.py tests/test_roi_mask_unification.py -v`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
make format
git add src/hydra_suite/core/tracking/worker.py src/hydra_suite/trackerkit/engine_params.py tests/test_arena_worker_wiring.py
git commit -m "feat(arena): wire arena layout through the tracking worker"
```

---

### Task 7: Post-processing grouped by arena

**Files:**
- Modify: `src/hydra_suite/core/post/processing.py:1144` (`resolve_trajectories`), `:757` (`process_trajectories_from_csv`)
- Modify: `src/hydra_suite/core/individual/identity/offline.py:397` (`_base_assignment_via_substrate`)
- Test: `tests/test_arena_postproc_grouping.py`

**Interfaces:**
- Consumes: `arena_id` column on trajectory DataFrames (Task 6).
- Produces: `resolve_trajectories(forward_trajs, backward_trajs, params, *, should_stop=None, slot_arena=None)`. When `slot_arena` is `None` or uniform, behavior is unchanged. Otherwise the trajectory lists are partitioned by arena, resolved independently, concatenated, and renumbered globally.

**Design note:** this wraps the existing 200-line function rather than editing its internals — merge candidates simply never span arenas because the partitions never meet.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_arena_postproc_grouping.py
import numpy as np
import pandas as pd
import pytest

from hydra_suite.core.post.processing import resolve_trajectories


def _traj(traj_id, x0, n=30, arena_id=0):
    return pd.DataFrame(
        {
            "frame": np.arange(n),
            "x": np.full(n, float(x0)),
            "y": np.full(n, 10.0),
            "theta": np.zeros(n),
            "trajectory_id": traj_id,
            "arena_id": arena_id,
        }
    )


PARAMS = {"AGREEMENT_DISTANCE": 15.0, "MIN_OVERLAP_FRAMES": 5, "MIN_TRAJECTORY_LENGTH": 5}


def test_uniform_slot_arena_matches_ungrouped_result():
    """Single-arena parity: grouping must be a no-op when there is one arena."""
    fwd = [_traj(0, 10.0), _traj(1, 200.0)]
    bwd = [_traj(0, 10.0), _traj(1, 200.0)]
    ungrouped = resolve_trajectories(fwd, bwd, PARAMS)
    grouped = resolve_trajectories(
        fwd, bwd, PARAMS, slot_arena=np.zeros(2, dtype=np.int32)
    )
    assert len(ungrouped) == len(grouped)
    for a, b in zip(ungrouped, grouped):
        pd.testing.assert_frame_equal(
            a.reset_index(drop=True), b.reset_index(drop=True)
        )


def test_spatially_coincident_trajectories_in_different_arenas_never_merge():
    """Two arenas whose tracks sit at identical coordinates must stay separate.

    Without arena grouping these are perfect merge candidates -- this is the
    exact cross-arena contamination the feature must prevent.
    """
    fwd = [_traj(0, 50.0, arena_id=0), _traj(1, 50.0, arena_id=1)]
    bwd = [_traj(0, 50.0, arena_id=0), _traj(1, 50.0, arena_id=1)]
    out = resolve_trajectories(
        fwd, bwd, PARAMS, slot_arena=np.array([0, 1], dtype=np.int32)
    )
    arenas = [int(df["arena_id"].iloc[0]) for df in out]
    assert sorted(arenas) == [0, 1], "each arena must retain its own trajectory"


def test_trajectory_ids_are_globally_unique_after_grouping():
    fwd = [_traj(0, 10.0, arena_id=0), _traj(1, 10.0, arena_id=1)]
    bwd = [_traj(0, 10.0, arena_id=0), _traj(1, 10.0, arena_id=1)]
    out = resolve_trajectories(
        fwd, bwd, PARAMS, slot_arena=np.array([0, 1], dtype=np.int32)
    )
    ids = [int(df["trajectory_id"].iloc[0]) for df in out]
    assert len(ids) == len(set(ids))


def test_arena_column_survives_resolution():
    fwd = [_traj(0, 10.0, arena_id=3)]
    bwd = [_traj(0, 10.0, arena_id=3)]
    out = resolve_trajectories(
        fwd, bwd, PARAMS, slot_arena=np.array([3], dtype=np.int32)
    )
    assert all("arena_id" in df.columns for df in out)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_arena_postproc_grouping.py -v`
Expected: FAIL — `TypeError: resolve_trajectories() got an unexpected keyword argument 'slot_arena'`

- [ ] **Step 3: Write minimal implementation**

Rename the existing body to `_resolve_trajectories_single_arena` and add the dispatcher:

```python
def resolve_trajectories(
    forward_trajs: object,
    backward_trajs: object,
    params: object = None,
    *,
    should_stop=None,
    slot_arena=None,
) -> object:
    """Resolve forward/backward trajectories, independently per arena.

    ``slot_arena`` maps trajectory-list index -> arena id. When it is None or
    names a single arena, this delegates straight to the single-arena
    implementation, so existing behavior is bit-for-bit preserved.

    Otherwise each arena's trajectories are resolved in isolation: merge
    candidates can never span arenas because the partitions never meet. Results
    are concatenated and trajectory ids renumbered so they stay globally unique.
    """
    if slot_arena is None:
        return _resolve_trajectories_single_arena(
            forward_trajs, backward_trajs, params, should_stop=should_stop
        )
    slot_arena = np.asarray(slot_arena, dtype=np.int32)
    arenas = np.unique(slot_arena)
    if len(arenas) <= 1:
        return _resolve_trajectories_single_arena(
            forward_trajs, backward_trajs, params, should_stop=should_stop
        )

    resolved: list = []
    for arena in arenas:
        idx = np.flatnonzero(slot_arena == arena)
        fwd = [forward_trajs[i] for i in idx if i < len(forward_trajs)]
        bwd = [backward_trajs[i] for i in idx if i < len(backward_trajs)]
        if not fwd and not bwd:
            continue
        resolved.extend(
            _resolve_trajectories_single_arena(
                fwd, bwd, params, should_stop=should_stop
            )
        )
    for new_id, df in enumerate(resolved):
        df["trajectory_id"] = new_id
    return resolved
```

In `process_trajectories_from_csv` (`processing.py:757`), when the CSV has an `arena_id` column with more than one distinct value, run the existing body per arena group and concatenate:

```python
    if "arena_id" in df.columns and df["arena_id"].nunique() > 1:
        parts = [
            _process_trajectories_single_arena(group, params)
            for _, group in df.groupby("arena_id", sort=True)
        ]
        return _renumber_concatenated(parts)
```

In `offline.py:397`, partition segments by `arena_id` before the uniqueness solve and merge the per-arena assignments, so "one antA" is enforced per arena.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_arena_postproc_grouping.py tests/test_postproc_equivalence.py tests/test_postproc_invariants.py tests/test_core_post_merge.py -v`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
make format
git add src/hydra_suite/core/post/processing.py src/hydra_suite/core/individual/identity/offline.py tests/test_arena_postproc_grouping.py
git commit -m "feat(arena): group trajectory resolution and offline identity by arena"
```

---

### Task 8: GUI — arena assignment on ROI shapes

**Files:**
- Modify: `src/hydra_suite/trackerkit/config/schemas.py:25`
- Modify: `src/hydra_suite/trackerkit/gui/orchestrators/session.py:2112` and `:2133` (shape append), `:2167` (status label)
- Modify: `src/hydra_suite/trackerkit/gui/main_window.py:577` (ROI toolbar)
- Test: `tests/test_arena_shape_assignment.py`

**Interfaces:**
- Consumes: `build_arena_labels` (Task 1).
- Produces: `TrackerConfig.animals_per_arena: int = 1`; `roi_shapes` entries carry `arena_id`; `SessionOrchestrator.current_arena_id: int` and `.start_new_arena() -> int`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_arena_shape_assignment.py
import pytest

from hydra_suite.trackerkit.config.schemas import TrackerConfig


def test_animals_per_arena_defaults_to_one():
    assert TrackerConfig().animals_per_arena == 1


def test_animals_per_arena_round_trips():
    cfg = TrackerConfig(animals_per_arena=6)
    assert TrackerConfig.from_dict(cfg.to_dict()).animals_per_arena == 6


def test_legacy_config_without_the_key_loads():
    cfg = TrackerConfig.from_dict({"roi_shapes": [], "current_video_path": ""})
    assert cfg.animals_per_arena == 1


def test_shapes_round_trip_their_arena_id():
    shapes = [
        {"type": "circle", "params": [10, 10, 5], "mode": "include", "arena_id": 0},
        {"type": "circle", "params": [40, 10, 5], "mode": "include", "arena_id": 1},
    ]
    cfg = TrackerConfig(roi_shapes=shapes)
    loaded = TrackerConfig.from_dict(cfg.to_dict())
    assert [s["arena_id"] for s in loaded.roi_shapes] == [0, 1]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_arena_shape_assignment.py -v`
Expected: FAIL — `TypeError: TrackerConfig.__init__() got an unexpected keyword argument 'animals_per_arena'`

- [ ] **Step 3: Write minimal implementation**

In `schemas.py`, add to the ROI block and to both `to_dict` and `from_dict`:

```python
    # --- Arenas ---
    animals_per_arena: int = 1
```

```python
            "animals_per_arena": int(self.animals_per_arena),
```

```python
            animals_per_arena=int(data.get("animals_per_arena", 1)),
```

In `session.py`, track the active arena and stamp it on every appended shape:

```python
        # SessionOrchestrator.__init__
        self.current_arena_id = 0
```

```python
    def start_new_arena(self) -> int:
        """Begin a new arena; subsequent include-shapes join it."""
        used = [
            int(s.get("arena_id", 0))
            for s in self._mw.roi_shapes
            if s.get("mode", "include") != "exclude"
        ]
        self.current_arena_id = (max(used) + 1) if used else 0
        return self.current_arena_id
```

At both `roi_shapes.append(...)` sites (lines 2112, 2133) include `"arena_id": self.current_arena_id`. Update the status label (line 2167) to report arena count alongside shape counts.

In `main_window.py`, add to the ROI toolbar next to `btn_start_roi` (line 598) a `QPushButton("New Arena")` wired to `start_new_arena`, plus a `QSpinBox` for animals per arena bound to `config.animals_per_arena`. The distinction matters: one arena is often several shapes (an include circle plus an exclude hole), so a new shape joins the current arena unless "New Arena" is pressed.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_arena_shape_assignment.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
make format
git add src/hydra_suite/trackerkit/config/schemas.py src/hydra_suite/trackerkit/gui/orchestrators/session.py src/hydra_suite/trackerkit/gui/main_window.py tests/test_arena_shape_assignment.py
git commit -m "feat(arena): assign arena ids to ROI shapes in the TrackerKit GUI"
```

---

### Task 9: GUI — arena grid generator

**Files:**
- Create: `src/hydra_suite/trackerkit/gui/dialogs/arena_grid_dialog.py`
- Modify: `src/hydra_suite/trackerkit/gui/main_window.py:577` (toolbar button)
- Test: `tests/test_arena_grid_dialog.py`

**Interfaces:**
- Consumes: nothing from earlier tasks — pure geometry.
- Produces: `generate_grid_shapes(rows, cols, origin_x, origin_y, pitch_x, pitch_y, size, shape_type="circle", first_arena_id=0) -> list[dict]` — a module-level pure function, plus `ArenaGridDialog(BaseDialog)` wrapping it with a live preview.

**Design note:** the pure function is separately testable without Qt, which matters because this repo has GUI tests that crash the interpreter (`project_main_suite_blockers`). Only the thin dialog needs Qt.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_arena_grid_dialog.py
import pytest

from hydra_suite.trackerkit.gui.dialogs.arena_grid_dialog import generate_grid_shapes


def test_grid_produces_rows_times_cols_shapes():
    shapes = generate_grid_shapes(2, 3, 50, 50, 100, 100, 40)
    assert len(shapes) == 6


def test_arena_ids_are_sequential_and_unique():
    shapes = generate_grid_shapes(2, 3, 50, 50, 100, 100, 40)
    assert [s["arena_id"] for s in shapes] == [0, 1, 2, 3, 4, 5]


def test_shapes_are_row_major():
    """Ids increase across a row before moving down -- matches well-plate naming."""
    shapes = generate_grid_shapes(2, 2, 50, 50, 100, 100, 40)
    centers = [(s["params"][0], s["params"][1]) for s in shapes]
    assert centers == [(50, 50), (150, 50), (50, 150), (150, 150)]


def test_circle_geometry_uses_radius_half_of_size():
    shapes = generate_grid_shapes(1, 1, 50, 50, 100, 100, 40)
    assert shapes[0]["type"] == "circle"
    assert shapes[0]["params"] == [50, 50, 20]


def test_polygon_grid_emits_four_corner_squares():
    shapes = generate_grid_shapes(1, 1, 50, 50, 100, 100, 40, shape_type="polygon")
    assert shapes[0]["type"] == "polygon"
    assert shapes[0]["params"] == [[30, 30], [70, 30], [70, 70], [30, 70]]


def test_first_arena_id_offsets_the_numbering():
    shapes = generate_grid_shapes(1, 2, 50, 50, 100, 100, 40, first_arena_id=7)
    assert [s["arena_id"] for s in shapes] == [7, 8]


def test_all_shapes_are_include_mode():
    shapes = generate_grid_shapes(2, 2, 50, 50, 100, 100, 40)
    assert all(s["mode"] == "include" for s in shapes)


def test_ninety_six_well_layout_is_supported():
    shapes = generate_grid_shapes(8, 12, 30, 30, 25, 25, 20)
    assert len(shapes) == 96
    assert len({s["arena_id"] for s in shapes}) == 96
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_arena_grid_dialog.py -v`
Expected: FAIL — `ModuleNotFoundError: ...arena_grid_dialog`

- [ ] **Step 3: Write minimal implementation**

```python
# src/hydra_suite/trackerkit/gui/dialogs/arena_grid_dialog.py
"""Bulk arena entry: lay down an R x C grid of arena shapes.

Hand-drawing 96 wells is impractical, so this generates the shapes in one go.
The output is ordinary ``roi_shapes`` entries with sequential ``arena_id``s that
stay individually editable afterwards -- this is a bulk-entry convenience, not
a separate mode.

``generate_grid_shapes`` is a pure function with no Qt dependency so it can be
tested without a display.
"""

from __future__ import annotations

from typing import Any


def generate_grid_shapes(
    rows: int,
    cols: int,
    origin_x: int,
    origin_y: int,
    pitch_x: int,
    pitch_y: int,
    size: int,
    shape_type: str = "circle",
    first_arena_id: int = 0,
) -> list[dict[str, Any]]:
    """Build a row-major grid of arena shapes.

    ``origin_x``/``origin_y`` is the centre of the top-left arena, ``pitch_*``
    the centre-to-centre spacing, and ``size`` the full width of one arena
    (diameter for circles, edge length for squares).
    """
    half = int(size) // 2
    shapes: list[dict[str, Any]] = []
    arena_id = int(first_arena_id)
    for row in range(int(rows)):
        for col in range(int(cols)):
            cx = int(origin_x) + col * int(pitch_x)
            cy = int(origin_y) + row * int(pitch_y)
            if shape_type == "polygon":
                params: Any = [
                    [cx - half, cy - half],
                    [cx + half, cy - half],
                    [cx + half, cy + half],
                    [cx - half, cy + half],
                ]
            else:
                params = [cx, cy, half]
            shapes.append(
                {
                    "type": "circle" if shape_type != "polygon" else "polygon",
                    "params": params,
                    "mode": "include",
                    "arena_id": arena_id,
                }
            )
            arena_id += 1
    return shapes
```

Then add `ArenaGridDialog(BaseDialog)` in the same file: spin boxes for rows, cols, origin, pitch, size, a shape-type combo, a preview label rendering `generate_grid_shapes` output over the current reference frame on every value change, and an `accepted_shapes()` accessor. Wire a "Generate Grid" button in the ROI toolbar (`main_window.py:598`) that opens it and extends `roi_shapes` with the result.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_arena_grid_dialog.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
make format
git add src/hydra_suite/trackerkit/gui/dialogs/arena_grid_dialog.py src/hydra_suite/trackerkit/gui/main_window.py tests/test_arena_grid_dialog.py
git commit -m "feat(arena): grid generator dialog for bulk arena entry"
```

---

### Task 10: Tiling oracle and the byte-identity gate

**Files:**
- Create: `tests/test_arena_tiling_oracle.py`
- Test: the full equivalence matrix

**Interfaces:**
- Consumes: everything from Tasks 1–9.
- Produces: no production code. This is the task that *proves* independence rather than asserting it.

**Why this is the decisive test:** Tasks 3–7 each gate one coupling point. Only an end-to-end run can show that no *other* coupling exists — `run_tracking` is ~3300 lines with all per-frame state in locals, and a shared local that silently aggregates across arenas would pass every unit test above. Tiling the same clip means the expected answer is known exactly.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_arena_tiling_oracle.py
"""End-to-end proof that arenas are tracked independently.

Tile one fixture clip into a 2x2 grid, declare each tile an arena, and require
each arena's trajectories to reproduce the single-clip run exactly, modulo the
tile's coordinate offset. Any cross-arena leak -- a shared per-frame local, a
global identity constraint, an ungrouped post-processing step -- shows up here
as a mismatch.
"""

import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

FIXTURES = Path("tools/equivalence/fixtures/clips")
CLIP = FIXTURES / "fly_obb.mp4"

pytestmark = pytest.mark.skipif(
    not CLIP.exists(), reason="equivalence fixtures not fetched"
)


def _tile_2x2(src: Path, dst: Path) -> tuple[int, int]:
    """Write a 2x2 tiling of *src*; return single-tile (width, height)."""
    cap = cv2.VideoCapture(str(src))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    writer = cv2.VideoWriter(
        str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w * 2, h * 2)
    )
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(np.tile(frame, (2, 2, 1)))
    cap.release()
    writer.release()
    return w, h


def _run_tracking(video: Path, config: dict, out_dir: Path) -> pd.DataFrame:
    """Run the headless tracker and return the final trajectories."""
    from hydra_suite.trackerkit.cli_config import run_tracking_headless

    return run_tracking_headless(video=str(video), config=config, out_dir=str(out_dir))


def _match_within_tolerance(a: pd.DataFrame, b: pd.DataFrame) -> None:
    """Positions must agree to the determinism floor (exact for these clips)."""
    assert len(a) == len(b), f"row count differs: {len(a)} vs {len(b)}"
    for col in ("x", "y"):
        np.testing.assert_allclose(
            np.sort(a[col].to_numpy()), np.sort(b[col].to_numpy()), atol=1e-6
        )


def test_each_arena_reproduces_the_single_clip_run(tmp_path):
    w, h = _tile_2x2(CLIP, tmp_path / "tiled.mp4")

    single = _run_tracking(CLIP, {"animals_per_arena": 4}, tmp_path / "single")

    arenas = []
    for idx, (col, row) in enumerate([(0, 0), (1, 0), (0, 1), (1, 1)]):
        arenas.append(
            {
                "type": "polygon",
                "mode": "include",
                "arena_id": idx,
                "params": [
                    [col * w, row * h],
                    [(col + 1) * w, row * h],
                    [(col + 1) * w, (row + 1) * h],
                    [col * w, (row + 1) * h],
                ],
            }
        )

    tiled = _run_tracking(
        tmp_path / "tiled.mp4",
        {"animals_per_arena": 4, "roi_shapes": arenas},
        tmp_path / "tiled",
    )

    assert set(tiled["arena_id"].unique()) == {0, 1, 2, 3}

    for idx, (col, row) in enumerate([(0, 0), (1, 0), (0, 1), (1, 1)]):
        got = tiled[tiled["arena_id"] == idx].copy()
        got["x"] -= col * w
        got["y"] -= row * h
        _match_within_tolerance(got, single)


def test_trajectory_ids_never_span_two_arenas(tmp_path):
    """A single trajectory id appearing in two arenas is a cross-arena leak."""
    w, h = _tile_2x2(CLIP, tmp_path / "tiled.mp4")
    arenas = [
        {
            "type": "polygon",
            "mode": "include",
            "arena_id": idx,
            "params": [
                [col * w, row * h],
                [(col + 1) * w, row * h],
                [(col + 1) * w, (row + 1) * h],
                [col * w, (row + 1) * h],
            ],
        }
        for idx, (col, row) in enumerate([(0, 0), (1, 0), (0, 1), (1, 1)])
    ]
    tiled = _run_tracking(
        tmp_path / "tiled.mp4",
        {"animals_per_arena": 4, "roi_shapes": arenas},
        tmp_path / "tiled",
    )
    spans = tiled.groupby("trajectory_id")["arena_id"].nunique()
    assert (spans == 1).all(), f"trajectories spanning arenas: {spans[spans > 1]}"
```

- [ ] **Step 2: Fetch fixtures if absent, then run to verify it fails or skips**

```bash
conda activate hydra-mps
bash tools/equivalence/fixtures/fetch_fixtures.sh
python -m pytest tests/test_arena_tiling_oracle.py -v
```

Expected: FAIL on the arena assertions (not SKIP). If it skips, the fixtures did not download — a skipping oracle proves nothing.

- [ ] **Step 3: Fix whatever the oracle exposes**

Expected classes of failure, in likelihood order: a per-frame local in `run_tracking` aggregating across arenas; `MAX_TARGETS`-wide statistics (density, confidence) computed globally; a post-processing path not yet arena-grouped. Fix in the owning module — never by loosening the oracle's tolerance.

- [ ] **Step 4: Run the byte-identity gate on MPS**

Kill stale sleap/hydra processes first, then:

```bash
conda activate hydra-mps
git worktree add --detach .worktrees/equiv-legacy legacy/main
REPO=$PWD WT=$PWD \
  MAIN_SRC=$PWD/.worktrees/equiv-legacy/src WT_SRC=$PWD/src \
  OUT=/tmp/equiv_arena RUNTIME=mps \
  bash tools/equivalence/run_matrix.sh
```

Expected: every clip EQUIVALENT at its determinism floor, for both `_forward.csv` and `_tracking_final.csv`. **Verify `wc -l` > 1 on the CSVs before trusting any EQUIVALENT** — empty CSVs compare equal and falsely pass. Known acceptable noise: bistable head/tail π-flips on head/tail clips.

- [ ] **Step 5: Run the byte-identity gate on CUDA**

```bash
ssh rutalab@mehek.taild08eb9.ts.net
cd ~/hydra-suite && git fetch origin && git checkout <branch-sha>
source ~/mambaforge/etc/profile.d/conda.sh && conda activate hydra-cuda
git worktree add --detach .worktrees/equiv-legacy legacy/main
REPO=$PWD WT=$PWD MAIN_SRC=$PWD/.worktrees/equiv-legacy/src WT_SRC=$PWD/src \
  OUT=/tmp/equiv_arena RUNTIME=cuda nohup bash tools/equivalence/run_matrix.sh > /tmp/equiv_cuda.log 2>&1 &
```

- [ ] **Step 6: Profile at target scale**

```bash
python -m pytest tests/test_arena_tiling_oracle.py -v --durations=10
```

Then run a 100-arena synthetic (10×10 tiling, 1 animal each) and confirm wall-clock is within ~1.25× of the single-arena run. The likely regression sites are the Python-level per-frame loops — `_apply_bayesian_identity_cost` and `_get_spatial_candidates`. If either dominates, the fix is to iterate per-arena blocks rather than over the dense N×M range; do not raise the tolerance.

- [ ] **Step 7: Commit**

```bash
make format
git add tests/test_arena_tiling_oracle.py
git commit -m "test(arena): end-to-end tiling oracle proving per-arena independence"
```

---

## Self-Review

**Spec coverage:**

| Spec section | Task |
|---|---|
| Data model (`arena_id`, `ARENA_LABELS`, union == `ROI_MASK`) | 1 |
| Slot layout, contiguous blocks, derived `MAX_TARGETS` | 2, 6 |
| Touch point 1 — detection labeling | 2, 6 *(relocated to the tracking layer — see "Deviation from the spec")* |
| Touch point 2 — arena-blocked assignment | 3, 4 |
| Touch point 3 — per-arena identity decoding | 5 |
| Touch point 4 — post-processing grouping | 7 |
| Touch point 5 — `arena_id` output column | 6 |
| GUI manual arena assignment | 8 |
| GUI grid generator | 9 |
| Config back-compat (no `arena_id` → arena 0) | 1, 8 |
| Single-arena byte-identity gate | 10 |
| Synthetic tiling oracle | 10 |
| Unit tests (blocked ≡ per-arena, label lookup, legacy, grid) | 3, 1, 8, 9 |
| Scale profiling at 100 arenas | 10 |

No spec requirement is unassigned. Out-of-scope items (per-arena overrides, arena crossing, per-arena files, auto-detection, global catalogs) have no tasks, as intended.

**Naming consistency check:** `build_arena_labels` (Tasks 1, 6), `ArenaLayout` with `.slot_arena`/`.max_targets`/`.arena_of_points`/`.is_single_arena`/`.label_image_for_size` (Tasks 2, 6), `set_track_arena` (Tasks 3, 4, 6), `meas_arena` (Tasks 3, 4, 6), `ArenaDecoderRegistry` (Task 5), `slot_arena=` keyword on `resolve_trajectories` (Task 7), `animals_per_arena` (Tasks 6, 8), `generate_grid_shapes` (Task 9) — each used identically wherever it appears.

**Ordering note:** Tasks 3 and 4 both modify `hungarian.py` and 4 depends on 3's `set_track_arena`; run them in order. Task 6 depends on 1, 2, 3, 4, 5. Task 10 depends on everything.
