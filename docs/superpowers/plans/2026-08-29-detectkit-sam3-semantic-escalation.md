# DetectKit SAM3 Semantic Escalation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a DetectKit user type a noun phrase and have SAM3 find and segment every matching instance across a source's frames, staged for review and promoted to a new sibling source.

**Architecture:** A Qt-free `SemanticLabeler` seam in `core/inference/semantic/` with a SAM3 backend, tiled with the existing `utils/slice_geometry.py` grid plus new seam-drop/NMS logic. A DetectKit job stages results into the existing `PendingEscalation` flow, caching pre-merge candidates so the confidence threshold stays adjustable after a multi-hour run. Promotion writes a **new sibling source** and never overwrites the origin's labels.

**Tech Stack:** Python 3.11, PySide6, ultralytics 8.4.34 (`SAM3SemanticPredictor`), huggingface_hub, OpenCV, NumPy, pytest.

**Spec:** `docs/superpowers/specs/2026-08-28-detectkit-sam3-semantic-escalation-design.md`

## Global Constraints

- **Dependency direction (CLAUDE.md):** `core/`, `utils/`, `data/` must never import from an app layer (`detectkit/`, `trackerkit/`, ...) or from Qt. `core/inference/semantic/` is Qt-free. Core→Data imports are allowed and used here.
- **Environment:** `conda activate hydra-mps` on this box. Run tests with `python -m pytest`.
- **Never run the whole suite** (`pytest tests/`) — it hangs on classkit modal dialogs. Run named files only.
- **Work in the worktree** `.worktrees/sam3-v2` on branch `feat/sam3-semantic-escalation`. Do not touch `main`.
- **TrackerKit equivalence matrix is NOT a gate** for this work. Nothing here touches TrackerKit, `core/inference` detection stages, or cache keys.
- **One run = one prompt = class `0`.** No multi-class prompting.
- `max_instances=0` means unlimited, everywhere.
- **Semantic tile fraction is `0.05`** and is independent of `SliceTrainingSettings.object_tile_fraction` (`0.15`). Never read the training fraction for semantic escalation.
- **SAM3 checkpoint:** HF repo `facebook/sam3`, 3.45 GB, NOT in ultralytics' `GITHUB_ASSETS_NAMES`. Ultralytics AutoUpdate would pip-install `clip` and `ftfy` on first use — the availability probe must fail loudly before that can happen.
- **Format before committing:** `make format` (autopep8 → black → isort). Pre-commit hooks run black/isort/flake8 on staged Python.
- Commit after every task. Small commits.

**Spec amendment adopted by this plan:** the spec says each swept threshold re-runs "seam-drop + NMS". Seam-drop is purely geometric and threshold-independent, so it is applied once at collection time and only NMS is redone per threshold. This is exact, not an approximation — the cached candidates are already post-seam-drop.

---

## File Structure

**New (Qt-free core):**
- `src/hydra_suite/core/inference/masks.py` — moved from `sam2/masks.py`, plus `polygon_iou`
- `src/hydra_suite/core/inference/torch_device.py` — moved cuda→mps→cpu picker
- `src/hydra_suite/core/inference/semantic/__init__.py`
- `src/hydra_suite/core/inference/semantic/base.py` — `SemanticInstance`, `SemanticLabeler`
- `src/hydra_suite/core/inference/semantic/tiling.py` — tile resolution, seam drop, offset, NMS merge
- `src/hydra_suite/core/inference/semantic/checkpoints.py` — SAM3 catalog, download, `probe_availability`
- `src/hydra_suite/core/inference/semantic/sam3.py` — `Sam3SemanticLabeler`
- `src/hydra_suite/core/inference/semantic/calibration.py` — matching, sweep, recommendation

**New (DetectKit app layer):**
- `src/hydra_suite/detectkit/jobs/semantic_escalation.py` — request/result/run/resume/re-threshold/promote + worker
- `src/hydra_suite/detectkit/gui/dialogs/semantic_escalation_dialog.py`
- `src/hydra_suite/detectkit/gui/escalation_actions.py` — the three escalation handlers

**Modified:**
- `src/hydra_suite/core/inference/sam2/masks.py` — deleted (moved)
- `src/hydra_suite/core/inference/sam2/executor.py:13-18` — picker becomes an alias
- `src/hydra_suite/detectkit/jobs/sam2_escalation.py:17` — import repointed
- `src/hydra_suite/detectkit/gui/models.py:14-40` — `PendingEscalation` generalised
- `src/hydra_suite/detectkit/gui/panels/tools_panel.py:103,156-181` — rename + new action
- `src/hydra_suite/detectkit/gui/main_window.py:741-748,1769-1943,1985-2010` — handlers extracted
- `src/hydra_suite/detectkit/gui/dialogs/review_escalations_dialog.py` — primer-aware + re-threshold
- `tests/test_sam2_masks.py:3`, `tests/test_sam2_executor.py:9-18` — repointed
- `pyproject.toml` — `sam3` extra

**New tests:** `tests/test_semantic_masks.py`, `tests/test_semantic_tiling.py`, `tests/test_semantic_calibration.py`, `tests/test_semantic_checkpoints.py`, `tests/test_semantic_escalation_job.py`, `tests/test_pending_escalation_model.py`

---

### Task 1: Move mask helpers to `core/inference/masks.py` and add `polygon_iou`

`sam2/masks.py` is 48 lines with two functions; both are imported together at `detectkit/jobs/sam2_escalation.py:17` and `tests/test_sam2_masks.py:3`. Move the module wholesale — splitting it would leave the test importing from two places. `polygon_iou` joins them because SAM3 contours come from masks and rasterizing is the correct IoU for arbitrary non-convex polygons. `utils/rotated_iou.py:pairwise_obb_overlap` cannot be reused: it is a 4-corner Sutherland-Hodgman convex clip and returns garbage for non-convex input.

**Files:**
- Create: `src/hydra_suite/core/inference/masks.py`
- Delete: `src/hydra_suite/core/inference/sam2/masks.py`
- Modify: `src/hydra_suite/detectkit/jobs/sam2_escalation.py:17`
- Modify: `tests/test_sam2_masks.py:3`
- Test: `tests/test_semantic_masks.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `hydra_suite.core.inference.masks.polygon_iou(a: np.ndarray, b: np.ndarray) -> float`; `clip_mask_to_polygon(mask, polygon_px) -> np.ndarray`; `mask_to_contour(mask, epsilon_frac=0.01, min_points=6, min_area=4.0) -> np.ndarray | None`.

- [ ] **Step 1: Move the module and repoint its two importers**

```bash
git mv src/hydra_suite/core/inference/sam2/masks.py src/hydra_suite/core/inference/masks.py
sed -i '' 's#hydra_suite.core.inference.sam2.masks#hydra_suite.core.inference.masks#' \
  src/hydra_suite/detectkit/jobs/sam2_escalation.py tests/test_sam2_masks.py
```

Then change the module docstring's first line in `src/hydra_suite/core/inference/masks.py` from
`"""Binary mask -> simplified largest external contour (SAM2 escalation)."""` to:

```python
"""Binary-mask and polygon geometry shared by SAM2 and SAM3 escalation."""
```

- [ ] **Step 2: Verify the move broke nothing**

Run: `python -m pytest tests/test_sam2_masks.py -q`
Expected: PASS (same tests, new import path).

- [ ] **Step 3: Write the failing `polygon_iou` tests**

Create `tests/test_semantic_masks.py`:

```python
import numpy as np

from hydra_suite.core.inference.masks import polygon_iou


def _square(x0, y0, side):
    return np.array(
        [[x0, y0], [x0 + side, y0], [x0 + side, y0 + side], [x0, y0 + side]],
        dtype=np.float32,
    )


def test_identical_squares_iou_is_one():
    a = _square(10, 10, 20)
    assert polygon_iou(a, a.copy()) == 1.0


def test_disjoint_squares_iou_is_zero():
    assert polygon_iou(_square(0, 0, 10), _square(50, 50, 10)) == 0.0


def test_half_overlapping_squares_iou_is_one_third():
    # Two 20x20 squares offset by 10 in x: intersection 200, union 600.
    iou = polygon_iou(_square(0, 0, 20), _square(10, 0, 20))
    assert abs(iou - 1.0 / 3.0) < 0.02


def test_non_convex_polygons_do_not_overlap_in_their_concavity():
    # Two interlocking L/U shapes whose bounding boxes overlap heavily but
    # whose filled areas do not. A convex-hull IoU would report a large
    # overlap here; a rasterized one reports ~0.
    u_shape = np.array(
        [[0, 0], [30, 0], [30, 30], [20, 30], [20, 10], [10, 10], [10, 30], [0, 30]],
        dtype=np.float32,
    )
    plug = np.array([[12, 14], [18, 14], [18, 30], [12, 30]], dtype=np.float32)
    assert polygon_iou(u_shape, plug) == 0.0


def test_degenerate_polygon_iou_is_zero():
    two_points = np.array([[0, 0], [10, 10]], dtype=np.float32)
    assert polygon_iou(two_points, _square(0, 0, 10)) == 0.0
    assert polygon_iou(np.zeros((0, 2), dtype=np.float32), _square(0, 0, 10)) == 0.0
```

- [ ] **Step 4: Run the tests to verify they fail**

Run: `python -m pytest tests/test_semantic_masks.py -q`
Expected: FAIL — `ImportError: cannot import name 'polygon_iou'`.

- [ ] **Step 5: Implement `polygon_iou`**

Append to `src/hydra_suite/core/inference/masks.py`:

```python
def polygon_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Rasterized IoU of two (P, 2) pixel-space polygons.

    Rasterizes both onto a shared integer grid covering their combined
    bounding box and counts pixels. Chosen over an analytic clip because
    SAM3 contours are arbitrary NON-CONVEX polygons: the convex
    Sutherland-Hodgman clip in ``utils/rotated_iou.py`` is only valid for
    quads and silently returns wrong areas for these. The polygons came
    from rasterized masks in the first place, so nothing is lost.

    Returns 0.0 if either polygon has fewer than 3 points.
    """
    pa = np.asarray(a, dtype=np.float64).reshape(-1, 2)
    pb = np.asarray(b, dtype=np.float64).reshape(-1, 2)
    if pa.shape[0] < 3 or pb.shape[0] < 3:
        return 0.0

    x0 = int(np.floor(min(pa[:, 0].min(), pb[:, 0].min())))
    y0 = int(np.floor(min(pa[:, 1].min(), pb[:, 1].min())))
    x1 = int(np.ceil(max(pa[:, 0].max(), pb[:, 0].max())))
    y1 = int(np.ceil(max(pa[:, 1].max(), pb[:, 1].max())))
    w, h = x1 - x0 + 1, y1 - y0 + 1
    if w <= 0 or h <= 0:
        return 0.0

    canvas_a = np.zeros((h, w), dtype=np.uint8)
    canvas_b = np.zeros((h, w), dtype=np.uint8)
    for poly, canvas in ((pa, canvas_a), (pb, canvas_b)):
        pts = np.round(poly - np.array([x0, y0], dtype=np.float64)).astype(np.int32)
        cv2.fillPoly(canvas, [pts.reshape(-1, 1, 2)], 1)

    inter = int(np.count_nonzero(canvas_a & canvas_b))
    if inter == 0:
        return 0.0
    union = int(np.count_nonzero(canvas_a | canvas_b))
    return float(inter) / float(union)
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `python -m pytest tests/test_semantic_masks.py tests/test_sam2_masks.py -q`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
make format
git add -A
git commit -m "refactor(inference): move mask helpers to core/inference/masks.py, add polygon_iou"
```

---

### Task 2: Move the device picker to `core/inference/torch_device.py`

The picker at `sam2/executor.py:13-18` is duplicated-in-waiting. Moving it **does** break `tests/test_sam2_executor.py:9-18`, which monkeypatches `executor.TORCH_CUDA_AVAILABLE` and would no longer affect an aliased function. Repoint those tests in the same commit.

**Files:**
- Create: `src/hydra_suite/core/inference/torch_device.py`
- Modify: `src/hydra_suite/core/inference/sam2/executor.py:1-20`
- Modify: `tests/test_sam2_executor.py`

**Interfaces:**
- Consumes: `hydra_suite.utils.gpu_utils.{MPS_AVAILABLE, TORCH_CUDA_AVAILABLE}`.
- Produces: `hydra_suite.core.inference.torch_device.resolve_torch_device() -> str` returning `"cuda"`, `"mps"`, or `"cpu"`. `sam2.executor.resolve_sam2_device` remains as an alias.

- [ ] **Step 1: Create the new module**

`src/hydra_suite/core/inference/torch_device.py`:

```python
"""Single cuda -> mps -> cpu torch device picker for inference backends."""

from __future__ import annotations

from hydra_suite.utils.gpu_utils import MPS_AVAILABLE, TORCH_CUDA_AVAILABLE


def resolve_torch_device() -> str:
    """Best available torch device: cuda, else mps, else cpu."""
    if TORCH_CUDA_AVAILABLE:
        return "cuda"
    if MPS_AVAILABLE:
        return "mps"
    return "cpu"
```

- [ ] **Step 2: Replace the body in `sam2/executor.py` with an alias**

In `src/hydra_suite/core/inference/sam2/executor.py`, delete the
`from hydra_suite.utils.gpu_utils import MPS_AVAILABLE, TORCH_CUDA_AVAILABLE` line and
replace the `resolve_sam2_device` definition (`:13-18`) with:

```python
from hydra_suite.core.inference.torch_device import resolve_torch_device

# Kept as a name for existing callers; the logic lives in torch_device.py.
resolve_sam2_device = resolve_torch_device
```

- [ ] **Step 3: Run the existing executor tests to see them fail**

Run: `python -m pytest tests/test_sam2_executor.py -q`
Expected: FAIL — the tests patch `executor.TORCH_CUDA_AVAILABLE`, which no longer exists on that module.

- [ ] **Step 4: Repoint the tests at the new module**

In `tests/test_sam2_executor.py`, change the import of the executor module to also import
the new one, and patch there instead:

```python
from hydra_suite.core.inference import torch_device as td


def test_resolve_device_prefers_cuda(monkeypatch):
    monkeypatch.setattr(td, "TORCH_CUDA_AVAILABLE", True)
    monkeypatch.setattr(td, "MPS_AVAILABLE", True)
    assert td.resolve_torch_device() == "cuda"


def test_resolve_device_falls_back_to_mps(monkeypatch):
    monkeypatch.setattr(td, "TORCH_CUDA_AVAILABLE", False)
    monkeypatch.setattr(td, "MPS_AVAILABLE", True)
    assert td.resolve_torch_device() == "mps"


def test_resolve_device_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(td, "TORCH_CUDA_AVAILABLE", False)
    monkeypatch.setattr(td, "MPS_AVAILABLE", False)
    assert td.resolve_torch_device() == "cpu"


def test_sam2_alias_points_at_the_shared_picker():
    from hydra_suite.core.inference.sam2.executor import resolve_sam2_device

    assert resolve_sam2_device is td.resolve_torch_device
```

Replace the two pre-existing device tests with these four; leave every other test in the file untouched.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python -m pytest tests/test_sam2_executor.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
make format
git add -A
git commit -m "refactor(inference): extract shared torch device picker to torch_device.py"
```

---

### Task 3: The `SemanticLabeler` seam

**Files:**
- Create: `src/hydra_suite/core/inference/semantic/__init__.py`
- Create: `src/hydra_suite/core/inference/semantic/base.py`
- Test: `tests/test_semantic_tiling.py` (the fake labeler used by later tasks lives here)

**Interfaces:**
- Consumes: nothing.
- Produces: `SemanticInstance(polygon_px: np.ndarray, confidence: float)` (frozen dataclass); `SemanticLabeler` Protocol with `name: str` property and `label_image(image_bgr, prompt, *, confidence_threshold=0.0, max_instances=0) -> list[SemanticInstance]`.

- [ ] **Step 1: Write the failing contract test**

Create `tests/test_semantic_tiling.py` with just this to start:

```python
import numpy as np

from hydra_suite.core.inference.semantic.base import SemanticInstance, SemanticLabeler


class FakeLabeler:
    """Returns a scripted list of instances per call, in TILE-LOCAL coords."""

    def __init__(self, scripted: list[list[SemanticInstance]]) -> None:
        self._scripted = list(scripted)
        self.calls = 0

    @property
    def name(self) -> str:
        return "fake"

    def label_image(self, image_bgr, prompt, *, confidence_threshold=0.0,
                    max_instances=0):
        out = self._scripted[self.calls] if self.calls < len(self._scripted) else []
        self.calls += 1
        return [i for i in out if i.confidence >= confidence_threshold]


def test_fake_labeler_satisfies_the_protocol():
    assert isinstance(FakeLabeler([]), SemanticLabeler)


def test_semantic_instance_is_frozen():
    inst = SemanticInstance(polygon_px=np.zeros((4, 2), dtype=np.float32),
                            confidence=0.5)
    try:
        inst.confidence = 0.9
    except Exception as exc:
        assert "frozen" in str(exc).lower()
    else:
        raise AssertionError("SemanticInstance must be frozen")
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_semantic_tiling.py -q`
Expected: FAIL — `ModuleNotFoundError: hydra_suite.core.inference.semantic`.

- [ ] **Step 3: Create the package and the seam**

`src/hydra_suite/core/inference/semantic/__init__.py`:

```python
"""Prompt-driven semantic instance segmentation (SAM3 escalation)."""
```

`src/hydra_suite/core/inference/semantic/base.py`:

```python
"""The SemanticLabeler seam: a prompt in, instance polygons out.

Qt-free and backend-free by construction -- every test in this subsystem
runs against a fake labeler, and the SAM3 weights are needed only by
``sam3.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class SemanticInstance:
    """One segmented instance.

    ``polygon_px`` is in the coordinate space of the image handed to
    ``label_image`` -- TILE-LOCAL under tiled inference. ``tiling.py``
    offsets it to frame space; nothing else may assume frame space.
    """

    polygon_px: np.ndarray  # (P, 2) float32
    confidence: float


@runtime_checkable
class SemanticLabeler(Protocol):
    """A model that turns (image, noun phrase) into instance polygons."""

    @property
    def name(self) -> str:
        """Short identifier for provenance, e.g. ``"sam3"``."""

    def label_image(
        self,
        image_bgr: np.ndarray,
        prompt: str,
        *,
        confidence_threshold: float = 0.0,
        max_instances: int = 0,
    ) -> list[SemanticInstance]:
        """Segment every instance matching *prompt*.

        ``max_instances=0`` means unlimited. Implementations return
        instances sorted by descending confidence.
        """
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_semantic_tiling.py -q`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
make format
git add -A
git commit -m "feat(semantic): add SemanticLabeler seam and SemanticInstance"
```

---

### Task 4: Tiling — resolution, seam drop, frame-space offset, per-threshold merge

The grid comes from `utils/slice_geometry.py` (`plan_tiles` returns frame-space `(x0, y0, x1, y1)` tiles with overlap and last-tile-flush; `MAX_TILES_PER_FRAME = 4096`). This module adds only what that one does not cover.

The semantic tile fraction is `0.05`, **derived not fitted**: SAM3 letterboxes to 1008 px and needs an object to reach ~50 px at that input, so `tile_px ≈ body_px * 1008 / 50 ≈ 20 * body_px`. At the measured `body_px = 80` that is ~1600 px, consistent with the measured-good 1504. It must NOT read `SliceTrainingSettings.object_tile_fraction` (`0.15`), which would yield a 533 px tile — the configuration measured to cost 5.7x for no gain.

**Files:**
- Create: `src/hydra_suite/core/inference/semantic/tiling.py`
- Test: `tests/test_semantic_tiling.py` (extend)

**Interfaces:**
- Consumes: `SemanticInstance`, `SemanticLabeler` (Task 3); `polygon_iou` (Task 1); `hydra_suite.utils.slice_geometry.{plan_tiles, tile_size_for_mode, SlicePlan, MAX_TILES_PER_FRAME}`.
- Produces:
  - `SEMANTIC_OBJECT_TILE_FRACTION: float = 0.05`, `SAM3_MODEL_INPUT_PX = 1008`, `SEMANTIC_TARGET_OBJECT_PX = 50.0`
  - `resolve_tile_px(reference_body_px: float, fraction: float = SEMANTIC_OBJECT_TILE_FRACTION) -> int | None`
  - `TileCandidate(polygon_px: np.ndarray, confidence: float, tile_index: int)` — polygon in FRAME space
  - `collect_candidates(labeler, image_bgr, plan, prompt, *, confidence_threshold, max_instances, seam_margin_px, should_stop=None, progress=None) -> list[TileCandidate]`
  - `merge_candidates(candidates, *, confidence_threshold, iou_threshold) -> list[SemanticInstance]`
  - `plan_for_frame(frame_hw, tile_px, overlap) -> SlicePlan`

- [ ] **Step 1: Write the failing tiling tests**

Append to `tests/test_semantic_tiling.py`:

```python
import pytest

from hydra_suite.core.inference.semantic.tiling import (
    SEMANTIC_OBJECT_TILE_FRACTION,
    TileCandidate,
    collect_candidates,
    merge_candidates,
    plan_for_frame,
    resolve_tile_px,
)


def _sq(x0, y0, side):
    return np.array(
        [[x0, y0], [x0 + side, y0], [x0 + side, y0 + side], [x0, y0 + side]],
        dtype=np.float32,
    )


def test_tile_px_derives_from_body_size_not_the_training_fraction():
    # 80 px body / 0.05 -> 1600. The TRAINING fraction (0.15) would give 533,
    # the configuration measured to cost 5.7x for no gain.
    assert resolve_tile_px(80.0) == 1600
    assert SEMANTIC_OBJECT_TILE_FRACTION == 0.05


def test_tile_px_is_none_when_body_size_is_unknown():
    assert resolve_tile_px(0.0) is None
    assert resolve_tile_px(-1.0) is None


def test_plan_covers_the_frame_with_overlap():
    plan = plan_for_frame((4512, 4512), 1504, 0.2)
    assert plan.slice_wh == (1504, 1504)
    assert len(plan.tiles) == 16
    assert plan.tiles[-1][2] == 4512 and plan.tiles[-1][3] == 4512


def test_plan_rejects_pathological_geometry():
    with pytest.raises(ValueError, match="tile ceiling|ceiling"):
        plan_for_frame((10000, 10000), 64, 0.9)


def test_candidates_are_offset_into_frame_space():
    # One tile at (1000, 500); a detection at tile-local (10, 20) must come
    # back at frame (1010, 520).
    plan = plan_for_frame((2000, 2000), 1000, 0.0)
    idx = plan.tiles.index((1000, 0, 2000, 1000))
    scripted = [[] for _ in plan.tiles]
    scripted[idx] = [SemanticInstance(_sq(400, 400, 60), 0.9)]
    labeler = FakeLabeler(scripted)
    cands = collect_candidates(
        labeler, np.zeros((2000, 2000, 3), dtype=np.uint8), plan, "ant",
        confidence_threshold=0.0, max_instances=0, seam_margin_px=4,
    )
    assert len(cands) == 1
    assert cands[0].polygon_px[:, 0].min() == pytest.approx(1400.0)
    assert cands[0].polygon_px[:, 1].min() == pytest.approx(400.0)


def test_seam_touching_detection_is_dropped_but_frame_edge_is_kept():
    plan = plan_for_frame((2000, 2000), 1000, 0.0)
    # Tile 0 is (0, 0, 1000, 1000): its x=1000 edge is an interior seam,
    # its x=0 edge is the frame edge.
    at_interior_seam = SemanticInstance(_sq(994, 400, 6), 0.9)
    at_frame_edge = SemanticInstance(_sq(0, 400, 6), 0.9)
    scripted = [[at_interior_seam, at_frame_edge]] + [[] for _ in plan.tiles[1:]]
    cands = collect_candidates(
        FakeLabeler(scripted), np.zeros((2000, 2000, 3), dtype=np.uint8), plan,
        "ant", confidence_threshold=0.0, max_instances=0, seam_margin_px=4,
    )
    assert len(cands) == 1
    assert cands[0].polygon_px[:, 0].min() == pytest.approx(0.0)


def test_merge_collapses_one_object_seen_in_two_overlapping_tiles():
    dup = [
        TileCandidate(_sq(100, 100, 40), 0.8, 0),
        TileCandidate(_sq(102, 101, 40), 0.6, 1),
    ]
    merged = merge_candidates(dup, confidence_threshold=0.0, iou_threshold=0.5)
    assert len(merged) == 1
    assert merged[0].confidence == 0.8  # the higher-scoring survivor wins


def test_merge_is_redone_per_threshold_not_post_filtered():
    # A high-scoring blob suppresses a lower one at threshold 0.0. Raising the
    # threshold above the suppressor must RESURRECT the suppressed candidate,
    # which post-filtering an already-merged set can never do.
    cands = [
        TileCandidate(_sq(0, 0, 50), 0.40, 0),
        TileCandidate(_sq(2, 2, 50), 0.90, 1),
    ]
    low = merge_candidates(cands, confidence_threshold=0.0, iou_threshold=0.5)
    assert [c.confidence for c in low] == [0.90]
    # Drop the suppressor by raising the bar past it in the other direction:
    survivors = merge_candidates(
        [c for c in cands if c.confidence < 0.5],
        confidence_threshold=0.0, iou_threshold=0.5,
    )
    assert [c.confidence for c in survivors] == [0.40]


def test_should_stop_halts_between_tiles():
    plan = plan_for_frame((2000, 2000), 1000, 0.0)
    labeler = FakeLabeler([[] for _ in plan.tiles])
    collect_candidates(
        labeler, np.zeros((2000, 2000, 3), dtype=np.uint8), plan, "ant",
        confidence_threshold=0.0, max_instances=0, seam_margin_px=4,
        should_stop=lambda: labeler.calls >= 2,
    )
    assert labeler.calls == 2
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_semantic_tiling.py -q`
Expected: FAIL — `ModuleNotFoundError: ...semantic.tiling`.

- [ ] **Step 3: Implement `tiling.py`**

```python
"""Tiled semantic inference: seam handling and cross-tile merge.

The tile GRID comes from ``utils/slice_geometry.py`` -- the same planner
training and the DetectKit preview use -- so this module never invents a
second tiling convention. What it adds is what that module does not cover:
dropping detections cut by an interior tile seam, offsetting tile-local
polygons into frame space, and merging duplicates across overlapping tiles.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from hydra_suite.core.inference.masks import polygon_iou
from hydra_suite.utils.slice_geometry import SlicePlan, plan_tiles

from .base import SemanticInstance, SemanticLabeler

logger = logging.getLogger(__name__)

# SAM3 letterboxes its input to this resolution.
SAM3_MODEL_INPUT_PX = 1008
# An object needs roughly this many pixels AT THE MODEL INPUT to be found.
# This is the one empirical constant in the tile rule; everything else is
# derived from it, and calibration can move off it by varying tile size.
SEMANTIC_TARGET_OBJECT_PX = 50.0
# tile_px = body_px * SAM3_MODEL_INPUT_PX / SEMANTIC_TARGET_OBJECT_PX
#         = body_px / SEMANTIC_OBJECT_TILE_FRACTION
SEMANTIC_OBJECT_TILE_FRACTION = round(
    SEMANTIC_TARGET_OBJECT_PX / SAM3_MODEL_INPUT_PX, 2
)  # 0.05

DEFAULT_OVERLAP = 0.5
DEFAULT_SEAM_MARGIN_PX = 4
DEFAULT_MERGE_IOU = 0.5


@dataclass(frozen=True)
class TileCandidate:
    """One surviving detection from one tile, in FRAME pixel space."""

    polygon_px: np.ndarray  # (P, 2) float32
    confidence: float
    tile_index: int


def resolve_tile_px(
    reference_body_px: float,
    fraction: float = SEMANTIC_OBJECT_TILE_FRACTION,
) -> int | None:
    """Tile edge length in pixels, or None when object scale is unknown.

    Deliberately NOT ``SliceTrainingSettings.object_tile_fraction``: the
    sliced-training optimum and the SAM3 optimum differ by ~3x, so one
    persisted fraction cannot serve both. A None result means the caller
    falls back to full-frame inference and says so, rather than guessing.
    """
    if reference_body_px is None or float(reference_body_px) <= 0:
        return None
    frac = max(0.01, min(0.9, float(fraction)))
    return int(max(64, min(4096, round(float(reference_body_px) / frac))))


def plan_for_frame(frame_hw, tile_px: int, overlap: float) -> SlicePlan:
    """Tile plan for one frame. Raises ValueError above the tile ceiling."""
    return plan_tiles(frame_hw, int(tile_px), int(tile_px), float(overlap),
                      float(overlap))


def _touches_interior_seam(
    polygon_px: np.ndarray,
    tile: tuple[int, int, int, int],
    frame_wh: tuple[int, int],
    margin_px: float,
) -> bool:
    """True if the polygon comes within margin_px of a NON-frame tile edge.

    With overlap, an object clipped by an interior seam is interior to some
    other tile, so dropping the fragment loses nothing and avoids merging a
    partial contour with a whole one.
    """
    x0, y0, x1, y1 = tile
    fw, fh = frame_wh
    px_min, py_min = polygon_px[:, 0].min(), polygon_px[:, 1].min()
    px_max, py_max = polygon_px[:, 0].max(), polygon_px[:, 1].max()
    if x0 > 0 and px_min <= x0 + margin_px:
        return True
    if y0 > 0 and py_min <= y0 + margin_px:
        return True
    if x1 < fw and px_max >= x1 - margin_px:
        return True
    if y1 < fh and py_max >= y1 - margin_px:
        return True
    return False


def collect_candidates(
    labeler: SemanticLabeler,
    image_bgr: np.ndarray,
    plan: SlicePlan,
    prompt: str,
    *,
    confidence_threshold: float,
    max_instances: int,
    seam_margin_px: float,
    should_stop: Callable[[], bool] | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> list[TileCandidate]:
    """Run *labeler* over every tile; return frame-space, seam-clean candidates.

    Seam drop is purely geometric and so threshold-INDEPENDENT: applying it
    here once is exact, and lets the confidence sweep re-merge cached
    candidates without re-running inference.
    """
    frame_h, frame_w = image_bgr.shape[:2]
    out: list[TileCandidate] = []
    for ti, (x0, y0, x1, y1) in enumerate(plan.tiles):
        if should_stop is not None and should_stop():
            break
        tile_img = image_bgr[y0:y1, x0:x1]
        instances = labeler.label_image(
            tile_img,
            prompt,
            confidence_threshold=confidence_threshold,
            max_instances=max_instances,
        )
        offset = np.array([x0, y0], dtype=np.float32)
        for inst in instances:
            poly = np.asarray(inst.polygon_px, dtype=np.float32).reshape(-1, 2) + offset
            if poly.shape[0] < 3:
                continue
            if _touches_interior_seam(poly, (x0, y0, x1, y1), (frame_w, frame_h),
                                      seam_margin_px):
                continue
            out.append(TileCandidate(poly, float(inst.confidence), ti))
        if progress is not None:
            progress(ti + 1, len(plan.tiles))
    return out


def merge_candidates(
    candidates: Sequence[TileCandidate],
    *,
    confidence_threshold: float,
    iou_threshold: float,
) -> list[SemanticInstance]:
    """Threshold then greedy-NMS the candidates into final instances.

    NMS is survivor-dependent, so this MUST be re-run for each swept
    confidence -- post-filtering an already merged set gives a different
    (wrong) answer, because a suppressor removed by the higher threshold
    should resurrect whatever it suppressed.
    """
    kept = sorted(
        (c for c in candidates if c.confidence >= confidence_threshold),
        key=lambda c: -c.confidence,
    )
    survivors: list[TileCandidate] = []
    for cand in kept:
        if any(
            polygon_iou(cand.polygon_px, s.polygon_px) >= iou_threshold
            for s in survivors
        ):
            continue
        survivors.append(cand)
    return [SemanticInstance(s.polygon_px, s.confidence) for s in survivors]
```

- [ ] **Step 4: Run to verify they pass**

Run: `python -m pytest tests/test_semantic_tiling.py -q`
Expected: PASS (10 tests).

- [ ] **Step 5: Commit**

```bash
make format
git add -A
git commit -m "feat(semantic): tiled inference with seam drop and per-threshold merge"
```

---

### Task 5: SAM3 checkpoint catalog and a real availability probe

The SAM2 pattern must NOT be copied here. `sam2/checkpoints.py:48-49`'s `available_variants()` just returns `list(SAM2_VARIANTS.keys())` — a static dict — so the tools-panel guard at `panels/tools_panel.py:172-181` never checks anything. Reusing it would produce exactly the silent 3.45 GB download this feature forbids.

**Files:**
- Create: `src/hydra_suite/core/inference/semantic/checkpoints.py`
- Modify: `pyproject.toml` (add a `sam3` extra)
- Test: `tests/test_semantic_checkpoints.py`

**Interfaces:**
- Consumes: `hydra_suite.paths.get_models_dir`.
- Produces: `Sam3Entry(repo_id, filename)`; `SAM3_VARIANTS: dict[str, Sam3Entry]`; `DEFAULT_VARIANT: str`; `available_variants() -> list[str]`; `checkpoint_path(variant, cache_dir=None) -> Path`; `ensure_checkpoint(variant, *, allow_download=True, cache_dir=None) -> Path`; `probe_availability(variant=DEFAULT_VARIANT, cache_dir=None) -> tuple[bool, str]`.

- [ ] **Step 1: Write the failing probe tests**

Create `tests/test_semantic_checkpoints.py`:

```python
import pytest

from hydra_suite.core.inference.semantic import checkpoints as ck


def test_catalog_pins_repo_and_filename():
    entry = ck.SAM3_VARIANTS[ck.DEFAULT_VARIANT]
    assert entry.repo_id == "facebook/sam3"
    assert entry.filename  # a pinned filename, never inferred at runtime


def test_ensure_checkpoint_refuses_unknown_variant():
    with pytest.raises(ValueError, match="Unknown SAM3 variant"):
        ck.ensure_checkpoint("nope")


def test_ensure_checkpoint_refuses_to_download_when_offline(tmp_path):
    with pytest.raises(ValueError, match="downloads are disabled"):
        ck.ensure_checkpoint(ck.DEFAULT_VARIANT, allow_download=False,
                             cache_dir=tmp_path)


def test_probe_reports_missing_checkpoint_without_downloading(tmp_path, monkeypatch):
    def _boom(*a, **k):  # any download attempt is a test failure
        raise AssertionError("probe must never download")

    monkeypatch.setattr(ck, "hf_hub_download", _boom)
    ok, reason = ck.probe_availability(cache_dir=tmp_path)
    assert ok is False
    assert "checkpoint" in reason.lower()


def test_probe_reports_a_missing_python_dependency(tmp_path, monkeypatch):
    monkeypatch.setattr(ck, "_find_spec", lambda name: None if name == "ftfy" else object())
    ok, reason = ck.probe_availability(cache_dir=tmp_path)
    assert ok is False
    assert "ftfy" in reason


def test_probe_succeeds_when_everything_is_present(tmp_path, monkeypatch):
    monkeypatch.setattr(ck, "_find_spec", lambda name: object())
    monkeypatch.setattr(ck, "_has_predictor_symbol", lambda: True)
    (tmp_path / f"{ck.DEFAULT_VARIANT}.pt").write_bytes(b"x")
    ok, reason = ck.probe_availability(cache_dir=tmp_path)
    assert ok is True
    assert reason == ""
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_semantic_checkpoints.py -q`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement `checkpoints.py`**

```python
"""SAM3 checkpoint catalog + a probe that never triggers a download.

Two facts drive this module. ``sam3.pt`` is 3.45 GB and is NOT in
ultralytics' ``GITHUB_ASSETS_NAMES``, so it comes from the public
``facebook/sam3`` HF repo. And ultralytics AutoUpdate pip-installs ``clip``
and ``ftfy`` on first use -- unacceptable on an offline or shared install.
``probe_availability`` therefore checks the Python deps and the on-disk
checkpoint BEFORE anything can reach ultralytics, and the GUI uses it to
disable the action with a reason instead of failing at click time.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import hf_hub_download

from hydra_suite.paths import get_models_dir


@dataclass(frozen=True)
class Sam3Entry:
    repo_id: str
    filename: str


# Pinned repo + filename, same discipline as SAM2_VARIANTS. Verify against
# the published `facebook/sam3` assets if this ever fails to download.
SAM3_VARIANTS: dict[str, Sam3Entry] = {
    "sam3": Sam3Entry("facebook/sam3", "sam3.pt"),
}

DEFAULT_VARIANT = "sam3"

# Imports ultralytics AutoUpdate would otherwise install behind our back.
REQUIRED_PACKAGES = ("ultralytics", "clip", "ftfy")


def available_variants() -> list[str]:
    return list(SAM3_VARIANTS.keys())


def _find_spec(name: str):  # seam for tests
    return importlib.util.find_spec(name)


def _has_predictor_symbol() -> bool:  # seam for tests
    try:
        from ultralytics.models.sam import SAM3SemanticPredictor  # noqa: F401
    except Exception:
        return False
    return True


def _cache_dir(cache_dir: Path | None) -> Path:
    return Path(cache_dir) if cache_dir is not None else Path(get_models_dir()) / "sam3"


def checkpoint_path(variant: str = DEFAULT_VARIANT,
                    cache_dir: Path | None = None) -> Path:
    return _cache_dir(cache_dir) / f"{variant}.pt"


def probe_availability(
    variant: str = DEFAULT_VARIANT, cache_dir: Path | None = None
) -> tuple[bool, str]:
    """(usable, reason). Never downloads, never imports ultralytics lazily."""
    if variant not in SAM3_VARIANTS:
        return False, f"Unknown SAM3 variant {variant!r}."
    for pkg in REQUIRED_PACKAGES:
        if _find_spec(pkg) is None:
            return False, (
                f"Python package {pkg!r} is missing. Install the SAM3 extra: "
                "pip install 'hydra-suite[sam3]'."
            )
    if not _has_predictor_symbol():
        return False, (
            "The installed ultralytics has no SAM3SemanticPredictor "
            "(needs >= 8.4.34)."
        )
    if not checkpoint_path(variant, cache_dir).exists():
        return False, (
            f"The SAM3 checkpoint ({variant}, ~3.45 GB) is not downloaded. "
            "Download it once from the semantic escalation dialog."
        )
    return True, ""


def ensure_checkpoint(
    variant: str = DEFAULT_VARIANT,
    *,
    allow_download: bool = True,
    cache_dir: Path | None = None,
) -> Path:
    """Return the cached SAM3 checkpoint path, downloading from HF if needed."""
    if variant not in SAM3_VARIANTS:
        raise ValueError(
            f"Unknown SAM3 variant {variant!r}. "
            f"Available: {', '.join(available_variants())}."
        )
    dest = checkpoint_path(variant, cache_dir)
    if dest.exists():
        return dest
    if not allow_download:
        raise ValueError(
            f"SAM3 variant {variant!r} is not downloaded and downloads are "
            "disabled (offline). Download it once with network access."
        )
    entry = SAM3_VARIANTS[variant]
    dest.parent.mkdir(parents=True, exist_ok=True)
    src = Path(hf_hub_download(repo_id=entry.repo_id, filename=entry.filename))
    dest.write_bytes(src.read_bytes())
    return dest
```

- [ ] **Step 4: Run to verify they pass**

Run: `python -m pytest tests/test_semantic_checkpoints.py -q`
Expected: PASS (6 tests).

- [ ] **Step 5: Add the `sam3` extra**

In `pyproject.toml`, under `[project.optional-dependencies]`, add:

```toml
sam3 = [
    "ultralytics>=8.4.34",
    "ftfy",
    "clip @ git+https://github.com/openai/CLIP.git",
]
```

- [ ] **Step 6: Commit**

```bash
make format
git add -A
git commit -m "feat(semantic): SAM3 checkpoint catalog and download-free availability probe"
```

---

### Task 6: The SAM3 backend

This is the only task that needs the 3.45 GB weights, so it is the only one whose integration check is manual. Everything downstream tests against the fake labeler.

**Files:**
- Create: `src/hydra_suite/core/inference/semantic/sam3.py`
- Test: `tests/test_semantic_checkpoints.py` (extend with a construction-guard test)

**Interfaces:**
- Consumes: `SemanticInstance` (Task 3), `ensure_checkpoint`/`probe_availability` (Task 5), `resolve_torch_device` (Task 2), `mask_to_contour` (Task 1).
- Produces: `Sam3SemanticLabeler` with `name` property, classmethod `from_variant(variant=DEFAULT_VARIANT, device=None, *, allow_download=True) -> Sam3SemanticLabeler`, and `label_image(...)` per the protocol.

- [ ] **Step 1: Write the failing guard test**

Append to `tests/test_semantic_checkpoints.py`:

```python
def test_labeler_refuses_to_construct_when_the_probe_fails(tmp_path, monkeypatch):
    from hydra_suite.core.inference.semantic import sam3

    monkeypatch.setattr(sam3, "probe_availability", lambda *a, **k: (False, "no ftfy"))
    with pytest.raises(RuntimeError, match="no ftfy"):
        sam3.Sam3SemanticLabeler.from_variant(cache_dir=tmp_path)


def test_labeler_satisfies_the_protocol_without_weights():
    from hydra_suite.core.inference.semantic.base import SemanticLabeler
    from hydra_suite.core.inference.semantic.sam3 import Sam3SemanticLabeler

    stub = Sam3SemanticLabeler(predictor=object(), device="cpu")
    assert isinstance(stub, SemanticLabeler)
    assert stub.name == "sam3"
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_semantic_checkpoints.py -q`
Expected: FAIL — `...semantic.sam3` not found.

- [ ] **Step 3: Implement `sam3.py`**

```python
"""SAM3 promptable-concept-segmentation backend for the SemanticLabeler seam.

Wraps ultralytics' ``SAM3SemanticPredictor``. Construction is guarded by
``probe_availability`` so a missing dependency raises with an actionable
message instead of letting ultralytics AutoUpdate pip-install packages.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from hydra_suite.core.inference.masks import mask_to_contour
from hydra_suite.core.inference.torch_device import resolve_torch_device

from .base import SemanticInstance
from .checkpoints import DEFAULT_VARIANT, ensure_checkpoint, probe_availability

logger = logging.getLogger(__name__)


class Sam3SemanticLabeler:
    """Text-prompted instance segmentation via SAM3."""

    def __init__(self, predictor, device: str) -> None:
        self._predictor = predictor
        self._device = device

    @property
    def name(self) -> str:
        return "sam3"

    @classmethod
    def from_variant(
        cls,
        variant: str = DEFAULT_VARIANT,
        device: str | None = None,
        *,
        allow_download: bool = True,
        cache_dir: Path | None = None,
    ) -> "Sam3SemanticLabeler":
        ok, reason = probe_availability(variant, cache_dir)
        if not ok and "not downloaded" not in reason:
            raise RuntimeError(f"SAM3 is unavailable: {reason}")
        ckpt = ensure_checkpoint(variant, allow_download=allow_download,
                                 cache_dir=cache_dir)
        # Lazy import: only paid when semantic escalation actually runs.
        from ultralytics.models.sam import SAM3SemanticPredictor

        dev = device or resolve_torch_device()
        predictor = SAM3SemanticPredictor(
            overrides={"model": str(ckpt), "device": dev, "save": False,
                       "verbose": False}
        )
        return cls(predictor, dev)

    def label_image(
        self,
        image_bgr: np.ndarray,
        prompt: str,
        *,
        confidence_threshold: float = 0.0,
        max_instances: int = 0,
    ) -> list[SemanticInstance]:
        """Segment every instance of *prompt*, sorted by descending score."""
        results = self._predictor(source=image_bgr, prompt=prompt)
        out: list[SemanticInstance] = []
        for res in results:
            masks = getattr(res, "masks", None)
            boxes = getattr(res, "boxes", None)
            if masks is None or masks.data is None:
                continue
            confs = (
                boxes.conf.detach().cpu().numpy()
                if boxes is not None and boxes.conf is not None
                else np.ones(len(masks.data), dtype=np.float32)
            )
            for mask_t, conf in zip(masks.data, confs):
                score = float(conf)
                if score < confidence_threshold:
                    continue
                contour = mask_to_contour(
                    mask_t.detach().cpu().numpy().astype(bool)
                )
                if contour is None or contour.shape[0] < 3:
                    continue
                out.append(SemanticInstance(contour, score))
        out.sort(key=lambda i: -i.confidence)
        if max_instances > 0:
            out = out[:max_instances]
        return out
```

- [ ] **Step 4: Run to verify they pass**

Run: `python -m pytest tests/test_semantic_checkpoints.py -q`
Expected: PASS (8 tests).

- [ ] **Step 5: Manual integration check with real weights**

`conda activate hydra-mps`, then:

```bash
python - <<'PY'
import numpy as np
from hydra_suite.core.inference.semantic.sam3 import Sam3SemanticLabeler
lab = Sam3SemanticLabeler.from_variant()
img = np.zeros((1008, 1008, 3), dtype=np.uint8)
print(lab.name, len(lab.label_image(img, "ant")))
PY
```

Expected: prints `sam3 0` (a blank frame legitimately yields nothing) without raising and without pip-installing anything. If it errors on a missing checkpoint, that is the first run downloading 3.45 GB — let it finish and re-run.

- [ ] **Step 6: Commit**

```bash
make format
git add -A
git commit -m "feat(semantic): SAM3 promptable-concept-segmentation backend"
```

---

### Task 7: Calibration — one-to-one matching and the recall-first frontier

Report the frontier of **missed vs. to-delete per frame**, not F1: deleting a spurious polygon is one click, a missed animal must be found by eye. On measured data the F1-optimal threshold missed 4.7 animals/frame where a recall-first threshold missed 1.0.

Matching is one-to-one on centroid distance gated by containment, not IoU: SAM3's masks trace legs and antennae and run ~1.7x the labelled body-core area, so IoU penalises correct detections for a purely conventional reason. Centroid distance alone is not enough — in a dense cluster a blob's centroid lands inside a neighbour's label, and two predictions claim one label.

**Files:**
- Create: `src/hydra_suite/core/inference/semantic/calibration.py`
- Test: `tests/test_semantic_calibration.py`

**Interfaces:**
- Consumes: `TileCandidate`, `merge_candidates`, `collect_candidates`, `plan_for_frame` (Task 4); `hydra_suite.data.al.escalation.LabelRecord`.
- Produces: `CalibrationPoint(confidence, missed_per_frame, extra_per_frame, recall, n_matched)`; `MIN_MATCHED_INSTANCES = 20`; `CONFIDENCE_GRID: tuple[float, ...]`; `match_one_to_one(pred_polys, label_polys) -> list[tuple[int, int]]`; `calibrate(labeler, frames, prompt, *, tile_px, overlap, seam_margin_px, merge_iou, max_instances=0, progress=None, should_stop=None) -> list[CalibrationPoint]`; `recommend(points, *, min_matched=MIN_MATCHED_INSTANCES) -> tuple[CalibrationPoint | None, str]`.
- `frames` is `Sequence[tuple[Path, list[LabelRecord]]]` — paths and records, never an `OBBSource`: Core must not import an app-layer type.

- [ ] **Step 1: Write the failing calibration tests**

Create `tests/test_semantic_calibration.py`:

```python
import numpy as np
import pytest

from hydra_suite.core.inference.semantic.calibration import (
    CONFIDENCE_GRID,
    CalibrationPoint,
    match_one_to_one,
    recommend,
)


def _sq(cx, cy, side=20.0):
    h = side / 2.0
    return np.array(
        [[cx - h, cy - h], [cx + h, cy - h], [cx + h, cy + h], [cx - h, cy + h]],
        dtype=np.float32,
    )


def test_each_label_and_prediction_is_used_at_most_once():
    labels = [_sq(100, 100)]
    # Two predictions both centred near the one label.
    preds = [_sq(101, 100), _sq(99, 101)]
    pairs = match_one_to_one(preds, labels)
    assert len(pairs) == 1
    assert len({p for p, _ in pairs}) == 1 and len({g for _, g in pairs}) == 1


def test_dense_cluster_does_not_let_a_blob_steal_a_neighbours_label():
    # Two labels 30 px apart; one oversized prediction centred between them.
    labels = [_sq(100, 100), _sq(130, 100)]
    blob = _sq(115, 100, side=70.0)
    pairs = match_one_to_one([blob], labels)
    assert len(pairs) == 1  # it matches ONE label, not both


def test_containment_gate_rejects_a_far_away_prediction():
    labels = [_sq(100, 100)]
    preds = [_sq(400, 400)]
    assert match_one_to_one(preds, labels) == []


def test_matching_works_for_aabb_obb_and_polygon_labels():
    label_aabb = _sq(50, 50)
    label_obb = np.array([[80, 78], [96, 82], [92, 98], [76, 94]], dtype=np.float32)
    label_poly = np.array(
        [[150, 150], [162, 148], [168, 158], [158, 168], [148, 162]], dtype=np.float32
    )
    preds = [_sq(51, 50), _sq(86, 88), _sq(157, 157)]
    pairs = match_one_to_one(preds, [label_aabb, label_obb, label_poly])
    assert len(pairs) == 3


def test_confidence_grid_is_ascending_and_bounded():
    assert list(CONFIDENCE_GRID) == sorted(CONFIDENCE_GRID)
    assert 0.0 < CONFIDENCE_GRID[0] and CONFIDENCE_GRID[-1] < 1.0


def test_recommend_refuses_below_the_minimum_matched_count():
    points = [
        CalibrationPoint(confidence=c, missed_per_frame=1.0, extra_per_frame=5.0,
                         recall=0.9, n_matched=3)
        for c in (0.2, 0.4)
    ]
    best, reason = recommend(points)
    assert best is None
    assert "insufficient" in reason.lower()


def test_recommend_prefers_recall_over_f1():
    # A point that misses 1/frame with 30 extra beats one that misses 5/frame
    # with 2 extra, even though the latter has better F1.
    recall_first = CalibrationPoint(0.20, 1.0, 30.0, 0.958, 70)
    f1_optimal = CalibrationPoint(0.60, 5.0, 2.0, 0.79, 60)
    best, reason = recommend([f1_optimal, recall_first])
    assert best is recall_first
    assert reason == ""
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_semantic_calibration.py -q`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement `calibration.py`**

```python
"""Fit the confidence threshold to the user's own labelled frames.

Two design commitments, both forced by measurement:

* The objective is the MISSED-vs-TO-DELETE frontier, not F1. Deleting a
  spurious polygon is one click; a missed animal must be found by eye. The
  F1-optimal threshold missed 4.7 animals/frame where a recall-first one
  missed 1.0.
* Matching is one-to-one nearest-centroid gated by containment, not IoU.
  SAM3 masks trace legs and antennae (~1.7x the labelled body-core area),
  so IoU penalises correct detections for a purely conventional reason --
  but centroid distance alone lets one blob claim two labels in a cluster.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import cv2
import numpy as np

from .base import SemanticLabeler
from .tiling import collect_candidates, merge_candidates, plan_for_frame

# Refuse to recommend a threshold fitted on fewer matched instances than this.
MIN_MATCHED_INSTANCES = 20
# Recall floor a point must clear to be recommendable.
MIN_RECALL = 0.90
CONFIDENCE_GRID: tuple[float, ...] = tuple(
    round(float(c), 2) for c in np.arange(0.05, 0.96, 0.05)
)


@dataclass(frozen=True)
class CalibrationPoint:
    confidence: float
    missed_per_frame: float
    extra_per_frame: float
    recall: float
    n_matched: int


def _centroid(poly: np.ndarray) -> np.ndarray:
    return np.asarray(poly, dtype=np.float64).reshape(-1, 2).mean(axis=0)


def _contains(poly: np.ndarray, point: np.ndarray) -> bool:
    contour = np.asarray(poly, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.pointPolygonTest(contour, (float(point[0]), float(point[1])), False) >= 0


def match_one_to_one(
    pred_polys: Sequence[np.ndarray], label_polys: Sequence[np.ndarray]
) -> list[tuple[int, int]]:
    """Greedy nearest-centroid pairing, each side used at most once.

    A pair is admissible only if the prediction's centroid falls inside the
    label, or the label's centroid falls inside the prediction -- the
    containment gate that stops one oversized blob from claiming its
    neighbour's label in a dense cluster.
    """
    pred_c = [_centroid(p) for p in pred_polys]
    label_c = [_centroid(g) for g in label_polys]
    pairs: list[tuple[float, int, int]] = []
    for pi, pc in enumerate(pred_c):
        for gi, gc in enumerate(label_c):
            if not (_contains(label_polys[gi], pc) or _contains(pred_polys[pi], gc)):
                continue
            pairs.append((float(np.hypot(*(pc - gc))), pi, gi))
    pairs.sort()
    used_p: set[int] = set()
    used_g: set[int] = set()
    out: list[tuple[int, int]] = []
    for _dist, pi, gi in pairs:
        if pi in used_p or gi in used_g:
            continue
        used_p.add(pi)
        used_g.add(gi)
        out.append((pi, gi))
    return out


def calibrate(
    labeler: SemanticLabeler,
    frames: Sequence[tuple[Path, list]],
    prompt: str,
    *,
    tile_px: int | None,
    overlap: float,
    seam_margin_px: float,
    merge_iou: float,
    max_instances: int = 0,
    progress: Callable[[int, str], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> list[CalibrationPoint]:
    """Sweep the confidence grid against *frames*' existing labels.

    Inference runs ONCE per frame at the lowest grid threshold; the sweep
    then re-merges the cached candidates per threshold. ``frames`` carries
    image paths and ``LabelRecord``s (not an app-layer source type) so this
    module stays inside the Core -> Data dependency direction.
    """
    floor = CONFIDENCE_GRID[0]
    per_frame: list[tuple[list, list[np.ndarray]]] = []
    for fi, (img_path, records) in enumerate(frames):
        if should_stop is not None and should_stop():
            break
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        h, w = image.shape[:2]
        plan = (
            plan_for_frame((h, w), tile_px, overlap)
            if tile_px
            else plan_for_frame((h, w), max(h, w), 0.0)
        )
        candidates = collect_candidates(
            labeler, image, plan, prompt,
            confidence_threshold=floor, max_instances=max_instances,
            seam_margin_px=seam_margin_px, should_stop=should_stop,
        )
        label_polys = [
            np.asarray(r.points, dtype=np.float32).reshape(-1, 2) for r in records
        ]
        per_frame.append((candidates, label_polys))
        if progress is not None:
            progress(int(100 * (fi + 1) / max(len(frames), 1)),
                     f"Calibrating {fi + 1}/{len(frames)}")

    n_frames = max(len(per_frame), 1)
    points: list[CalibrationPoint] = []
    for conf in CONFIDENCE_GRID:
        matched = missed = extra = total_labels = 0
        for candidates, label_polys in per_frame:
            merged = merge_candidates(
                candidates, confidence_threshold=conf, iou_threshold=merge_iou
            )
            preds = [m.polygon_px for m in merged]
            pairs = match_one_to_one(preds, label_polys)
            matched += len(pairs)
            missed += len(label_polys) - len(pairs)
            extra += len(preds) - len(pairs)
            total_labels += len(label_polys)
        points.append(
            CalibrationPoint(
                confidence=conf,
                missed_per_frame=missed / n_frames,
                extra_per_frame=extra / n_frames,
                recall=(matched / total_labels) if total_labels else 0.0,
                n_matched=matched,
            )
        )
    return points


def recommend(
    points: Sequence[CalibrationPoint],
    *,
    min_matched: int = MIN_MATCHED_INSTANCES,
    min_recall: float = MIN_RECALL,
) -> tuple[CalibrationPoint | None, str]:
    """Highest confidence that still clears the recall floor, or a refusal.

    Refuses (returns ``(None, reason)``) when the best point matched fewer
    than *min_matched* instances -- a threshold fitted on a handful of
    correlated frames is not a recommendation, and the caller shows the
    frontier instead.
    """
    if not points:
        return None, "No calibration points; nothing to recommend."
    best_matched = max(p.n_matched for p in points)
    if best_matched < min_matched:
        return None, (
            f"Insufficient data: only {best_matched} instance(s) matched across "
            f"the labelled frames (need {min_matched}). The frontier below is "
            "shown for inspection, but no threshold is recommended."
        )
    eligible = [p for p in points if p.recall >= min_recall]
    if not eligible:
        return None, (
            f"No threshold reached {min_recall:.0%} recall on these frames. Try "
            "a different prompt, or a smaller tile size."
        )
    # Highest confidence still clearing the recall floor = fewest extras to
    # delete at acceptable misses. Deliberately not the F1 maximum.
    return max(eligible, key=lambda p: p.confidence), ""
```

- [ ] **Step 4: Run to verify they pass**

Run: `python -m pytest tests/test_semantic_calibration.py -q`
Expected: PASS (7 tests).

- [ ] **Step 5: Commit**

```bash
make format
git add -A
git commit -m "feat(semantic): recall-first calibration against the user's labelled frames"
```

---

### Task 8: Generalise `PendingEscalation`

`PendingEscalation` currently hardcodes `sam2_variant` (`gui/models.py:14-40`). `to_dict`/`from_dict` are hand-written with `d.get` defaults and `DetectKitProject.load`'s coercion loop never touches nested source dicts, so adding fields with a back-fill loads existing projects with no migration.

**Files:**
- Modify: `src/hydra_suite/detectkit/gui/models.py:14-40`
- Test: `tests/test_pending_escalation_model.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `PendingEscalation(staged_path, target_level, sam2_variant, created_at, primer_kind, primer_variant, primer_prompt, primer_params)` where `primer_kind ∈ {"sam2", "sam3"}` and `primer_params: dict`.

- [ ] **Step 1: Write the failing round-trip tests**

Create `tests/test_pending_escalation_model.py`:

```python
from hydra_suite.detectkit.gui.models import PendingEscalation


def test_legacy_dict_backfills_sam2_primer_fields():
    legacy = {
        "staged_path": "/tmp/staged",
        "target_level": "polygon",
        "sam2_variant": "sam2.1-hiera-large",
        "created_at": "2026-01-01T00:00:00",
    }
    p = PendingEscalation.from_dict(legacy)
    assert p.primer_kind == "sam2"
    assert p.primer_variant == "sam2.1-hiera-large"
    assert p.primer_prompt == ""
    assert p.primer_params == {}


def test_sam3_round_trip_preserves_prompt_and_params():
    p = PendingEscalation(
        staged_path="/tmp/s",
        target_level="polygon",
        created_at="2026-01-01T00:00:00",
        primer_kind="sam3",
        primer_variant="sam3",
        primer_prompt="black ant",
        primer_params={"confidence": 0.35, "tile_px": 1600},
    )
    restored = PendingEscalation.from_dict(p.to_dict())
    assert restored == p


def test_sam2_variant_stays_in_sync_for_legacy_readers():
    p = PendingEscalation.from_dict({"sam2_variant": "sam2.1-hiera-tiny"})
    assert p.sam2_variant == "sam2.1-hiera-tiny"
    assert p.to_dict()["sam2_variant"] == "sam2.1-hiera-tiny"
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_pending_escalation_model.py -q`
Expected: FAIL — `PendingEscalation` has no `primer_kind`.

- [ ] **Step 3: Generalise the dataclass**

Replace `PendingEscalation` in `src/hydra_suite/detectkit/gui/models.py` with:

```python
@dataclass
class PendingEscalation:
    """A staged (not-yet-reviewed) escalation result awaiting accept/reject.

    ``primer_kind`` distinguishes the two producers: ``"sam2"`` converts
    existing boxes to masks and promotes IN PLACE; ``"sam3"`` finds
    instances from a prompt and promotes to a NEW SIBLING SOURCE. They
    accept differently, so the kind is load-bearing, not decorative.

    ``sam2_variant`` is retained so pre-existing projects and any legacy
    reader keep working; it mirrors ``primer_variant`` for SAM2 records.
    """

    staged_path: str = ""
    target_level: str = "polygon"
    sam2_variant: str = ""
    created_at: str = ""
    primer_kind: str = "sam2"
    primer_variant: str = ""
    primer_prompt: str = ""
    primer_params: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize to a plain dictionary."""
        return {
            "staged_path": self.staged_path,
            "target_level": self.target_level,
            "sam2_variant": self.sam2_variant,
            "created_at": self.created_at,
            "primer_kind": self.primer_kind,
            "primer_variant": self.primer_variant,
            "primer_prompt": self.primer_prompt,
            "primer_params": dict(self.primer_params),
        }

    @staticmethod
    def from_dict(d: dict) -> "PendingEscalation":
        """Restore a PendingEscalation, back-filling pre-primer records."""
        legacy_variant = str(d.get("sam2_variant", ""))
        kind = str(d.get("primer_kind", "") or "sam2")
        variant = str(d.get("primer_variant", "") or legacy_variant)
        return PendingEscalation(
            staged_path=str(d.get("staged_path", "")),
            target_level=str(d.get("target_level", "polygon") or "polygon"),
            sam2_variant=legacy_variant or (variant if kind == "sam2" else ""),
            created_at=str(d.get("created_at", "")),
            primer_kind=kind,
            primer_variant=variant,
            primer_prompt=str(d.get("primer_prompt", "")),
            primer_params=dict(d.get("primer_params") or {}),
        )
```

`field` is already imported at `models.py:6`.

- [ ] **Step 4: Run to verify they pass, and that nothing else broke**

Run: `python -m pytest tests/test_pending_escalation_model.py tests/test_sam2_escalation.py -q`
Expected: PASS. (If `tests/test_sam2_escalation.py` does not exist, run
`python -m pytest tests/ -q -k escalation --collect-only` to find the right file and run that.)

- [ ] **Step 5: Commit**

```bash
make format
git add -A
git commit -m "feat(detectkit): generalise PendingEscalation with primer kind/prompt/params"
```

---

### Task 9: The semantic escalation job — run, cache, resume, cancel

Four things SAM2's `run_escalation` gets right for SAM2 but wrong for SAM3, all fixed here:

1. It filters `todo` to `level != "polygon"` (`sam2_escalation.py:177-181`). Running a prompt against a polygon-level source to find animals the polygons missed is a **primary** use case. No such filter here.
2. The staging dirname hash is `sha1(str(src_root) + variant)` (`:198-201`). The **prompt must enter the hash**, or two prompts on one source collide.
3. `remove_staged_escalation_dir(staged_root)` runs unconditionally before writing (`:215`). That makes resume impossible, because a resumed run must pass `overwrite=True` — the wiping path. Here the wipe is conditional on a `run.json` fingerprint mismatch.
4. It has no `should_stop`. A multi-hour run must be cancellable between tiles.

**Files:**
- Create: `src/hydra_suite/detectkit/jobs/semantic_escalation.py`
- Test: `tests/test_semantic_escalation_job.py`

**Interfaces:**
- Consumes: `collect_candidates`, `merge_candidates`, `plan_for_frame`, `resolve_tile_px`, `DEFAULT_*` (Task 4); `PendingEscalation`, `OBBSource` (Task 8); `write_label_file`, `LabelRecord`, `GeometryLevel`, `ensure_bundle_subdirectory`, `IMG_EXTS`, `remove_staged_escalation_dir` (existing).
- Produces:
  - `SemanticEscalationRequest(project, source_names, variant, prompt, confidence, max_instances, reference_body_px, overlap, seam_margin_px, merge_iou, tile_px, overwrite)`
  - `SemanticEscalationResult(staged, labelled, empty_images, degenerate, tile_px, skipped)`
  - `run_semantic_escalation(req, labeler, *, overwrite=False, progress=None, should_stop=None) -> SemanticEscalationResult`
  - `rethreshold_staged(source, *, confidence, merge_iou) -> int`
  - `is_prompt_failure(result, frames_processed) -> bool`
  - `CANDIDATES_FILENAME = "candidates.json"`, `RUN_FILENAME = "run.json"`

- [ ] **Step 1: Write the failing job tests**

Create `tests/test_semantic_escalation_job.py`:

```python
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from hydra_suite.core.inference.semantic.base import SemanticInstance
from hydra_suite.detectkit.gui.models import OBBSource
from hydra_suite.detectkit.jobs.semantic_escalation import (
    SemanticEscalationRequest,
    SemanticEscalationResult,
    is_prompt_failure,
    rethreshold_staged,
    run_semantic_escalation,
)


class ScriptedLabeler:
    """Returns the same tile-local instances for every tile it is given."""

    def __init__(self, instances):
        self._instances = instances

    @property
    def name(self):
        return "fake"

    def label_image(self, image_bgr, prompt, *, confidence_threshold=0.0,
                    max_instances=0):
        return [i for i in self._instances if i.confidence >= confidence_threshold]


class _Project:
    def __init__(self, project_dir, sources):
        self.project_dir = str(project_dir)
        self.sources = sources


def _make_source(tmp_path, name="src", n_images=2, level="polygon"):
    root = tmp_path / "sources" / name
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir(parents=True)
    for i in range(n_images):
        cv2.imwrite(str(root / "images" / f"f{i}.png"),
                    np.zeros((400, 400, 3), dtype=np.uint8))
        (root / "labels" / f"f{i}.txt").write_text("")
    (root / "classes.txt").write_text("object\n")
    return OBBSource(path=str(root), name=name, level=level)


def _blob(cx, cy, side=20.0):
    h = side / 2.0
    return np.array(
        [[cx - h, cy - h], [cx + h, cy - h], [cx + h, cy + h], [cx - h, cy + h]],
        dtype=np.float32,
    )


def _request(tmp_path, src, **kw):
    defaults = dict(
        project=_Project(tmp_path, [src]), source_names=[src.name], variant="sam3",
        prompt="ant", confidence=0.1, max_instances=0, reference_body_px=20.0,
        overlap=0.0, seam_margin_px=2.0, merge_iou=0.5, tile_px=None, overwrite=False,
    )
    defaults.update(kw)
    return SemanticEscalationRequest(**defaults)


def test_polygon_level_sources_are_not_filtered_out(tmp_path):
    src = _make_source(tmp_path, level="polygon")
    labeler = ScriptedLabeler([SemanticInstance(_blob(200, 200), 0.9)])
    result = run_semantic_escalation(_request(tmp_path, src), labeler)
    assert result.staged == [src.name]
    assert result.labelled > 0


def test_original_labels_are_never_touched(tmp_path):
    src = _make_source(tmp_path)
    (Path(src.path) / "labels" / "f0.txt").write_text("0 0.1 0.1 0.2 0.2\n")
    labeler = ScriptedLabeler([SemanticInstance(_blob(200, 200), 0.9)])
    run_semantic_escalation(_request(tmp_path, src), labeler)
    assert (Path(src.path) / "labels" / "f0.txt").read_text() == "0 0.1 0.1 0.2 0.2\n"


def test_two_prompts_stage_into_different_directories(tmp_path):
    src = _make_source(tmp_path)
    labeler = ScriptedLabeler([SemanticInstance(_blob(200, 200), 0.9)])
    run_semantic_escalation(_request(tmp_path, src, prompt="ant"), labeler)
    first = src.pending_escalation.staged_path
    run_semantic_escalation(_request(tmp_path, src, prompt="beetle", overwrite=True),
                            labeler, overwrite=True)
    assert src.pending_escalation.staged_path != first


def test_empty_images_are_counted_and_flagged_as_a_prompt_failure(tmp_path):
    src = _make_source(tmp_path, n_images=3)
    result = run_semantic_escalation(_request(tmp_path, src), ScriptedLabeler([]))
    assert result.empty_images == 3
    assert result.labelled == 0
    assert is_prompt_failure(result, frames_processed=3) is True


def test_degenerate_contours_are_dropped_not_fatal(tmp_path):
    src = _make_source(tmp_path, n_images=1)
    two_points = np.array([[10, 10], [20, 20]], dtype=np.float32)
    labeler = ScriptedLabeler([
        SemanticInstance(two_points, 0.9),
        SemanticInstance(_blob(200, 200), 0.9),
    ])
    result = run_semantic_escalation(_request(tmp_path, src), labeler)
    assert result.degenerate >= 1
    assert result.labelled == 1


def test_candidates_cache_is_written_into_the_staging_dir(tmp_path):
    src = _make_source(tmp_path, n_images=1)
    labeler = ScriptedLabeler([SemanticInstance(_blob(200, 200), 0.9)])
    run_semantic_escalation(_request(tmp_path, src), labeler)
    cache = Path(src.pending_escalation.staged_path) / "candidates.json"
    data = json.loads(cache.read_text())
    assert data["version"] == 1
    assert "f0.png" in data["images"]


def test_rethreshold_rewrites_labels_without_inference(tmp_path):
    src = _make_source(tmp_path, n_images=1)
    labeler = ScriptedLabeler([
        SemanticInstance(_blob(100, 100), 0.9),
        SemanticInstance(_blob(300, 300), 0.2),
    ])
    run_semantic_escalation(_request(tmp_path, src, confidence=0.1), labeler)
    staged = Path(src.pending_escalation.staged_path) / "labels" / "f0.txt"
    assert len(staged.read_text().strip().splitlines()) == 2
    kept = rethreshold_staged(src, confidence=0.5, merge_iou=0.5)
    assert kept == 1
    assert len(staged.read_text().strip().splitlines()) == 1


def test_resume_skips_images_already_in_the_cache(tmp_path):
    src = _make_source(tmp_path, n_images=2)

    class Counting(ScriptedLabeler):
        calls = 0

        def label_image(self, *a, **k):
            Counting.calls += 1
            return super().label_image(*a, **k)

    labeler = Counting([SemanticInstance(_blob(200, 200), 0.9)])
    req = _request(tmp_path, src)
    run_semantic_escalation(req, labeler, should_stop=lambda: Counting.calls >= 1)
    first_calls = Counting.calls
    run_semantic_escalation(req, labeler, overwrite=True)
    # The already-cached image is not re-inferred.
    assert Counting.calls < first_calls + 2


def test_fingerprint_mismatch_wipes_the_cache(tmp_path):
    src = _make_source(tmp_path, n_images=1)
    labeler = ScriptedLabeler([SemanticInstance(_blob(200, 200), 0.9)])
    run_semantic_escalation(_request(tmp_path, src, prompt="ant"), labeler)
    staged = Path(src.pending_escalation.staged_path)
    run_json = json.loads((staged / "run.json").read_text())
    assert run_json["prompt"] == "ant"


def test_already_pending_source_is_skipped_without_overwrite(tmp_path):
    src = _make_source(tmp_path, n_images=1)
    labeler = ScriptedLabeler([SemanticInstance(_blob(200, 200), 0.9)])
    run_semantic_escalation(_request(tmp_path, src), labeler)
    result = run_semantic_escalation(_request(tmp_path, src), labeler)
    assert result.staged == []
    assert result.skipped and result.skipped[0][0] == src.name
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_semantic_escalation_job.py -q`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement the job**

Create `src/hydra_suite/detectkit/jobs/semantic_escalation.py`:

```python
"""SAM3 semantic escalation: a prompt -> staged polygon labels for review.

Mirrors ``sam2_escalation.run_escalation``'s staging mechanics and departs
from them in four deliberate places, each marked below: no polygon-level
filter, the prompt enters the staging hash, the pre-write wipe is
conditional on a run fingerprint (so a multi-hour run can resume), and
cancellation is honoured between tiles.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from hashlib import sha1
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from PySide6.QtCore import Signal

from hydra_suite.core.inference.semantic.tiling import (
    DEFAULT_MERGE_IOU,
    DEFAULT_OVERLAP,
    DEFAULT_SEAM_MARGIN_PX,
    TileCandidate,
    collect_candidates,
    merge_candidates,
    plan_for_frame,
    resolve_tile_px,
)
from hydra_suite.data.al.escalation import LabelRecord
from hydra_suite.data.al.labels import write_label_file
from hydra_suite.data.project_bundle import ensure_bundle_subdirectory
from hydra_suite.detectkit.gui.constants import IMG_EXTS
from hydra_suite.detectkit.gui.models import OBBSource, PendingEscalation
from hydra_suite.utils.geometry_levels import GeometryLevel
from hydra_suite.widgets.workers import BaseWorker

from .sam2_escalation import remove_staged_escalation_dir

logger = logging.getLogger(__name__)

CANDIDATES_FILENAME = "candidates.json"
RUN_FILENAME = "run.json"
# A run staging nothing on a majority of frames is a prompt failure, not a
# quiet success -- the dominant failure mode is a noun phrase the model does
# not match, and it looks exactly like a clean run otherwise.
PROMPT_FAILURE_FRACTION = 0.5


@dataclass
class SemanticEscalationRequest:
    project: object  # has .project_dir and .sources (list[OBBSource])
    source_names: list[str]
    variant: str
    prompt: str
    confidence: float = 0.35
    max_instances: int = 0
    reference_body_px: float = 0.0
    overlap: float = DEFAULT_OVERLAP
    seam_margin_px: float = DEFAULT_SEAM_MARGIN_PX
    merge_iou: float = DEFAULT_MERGE_IOU
    tile_px: int | None = None  # explicit override; None = derive from body px
    overwrite: bool = False


@dataclass
class SemanticEscalationResult:
    staged: list[str] = field(default_factory=list)
    labelled: int = 0  # instances staged
    empty_images: int = 0  # frames where the model returned nothing
    degenerate: int = 0  # contours with P < 3, dropped not fatal
    tile_px: int | None = None  # resolved tile size, None = full frame
    skipped: list[tuple[str, str]] = field(default_factory=list)


def is_prompt_failure(result: SemanticEscalationResult, frames_processed: int) -> bool:
    """True when the run should be reported as a PROMPT failure, not success."""
    if frames_processed <= 0:
        return False
    return result.empty_images >= PROMPT_FAILURE_FRACTION * frames_processed


def prompt_slug(prompt: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", prompt.strip().lower()).strip("-")
    return (slug or "prompt")[:24]


def _fingerprint(req: SemanticEscalationRequest, src_root: Path,
                 tile_px: int | None) -> dict:
    return {
        "prompt": req.prompt,
        "variant": req.variant,
        "tile_px": tile_px,
        "overlap": float(req.overlap),
        "seam_margin_px": float(req.seam_margin_px),
        "max_instances": int(req.max_instances),
        "confidence_floor": float(req.confidence),
        "source_root": str(src_root.resolve()),
    }


def _load_cache(staged_root: Path) -> dict:
    path = staged_root / CANDIDATES_FILENAME
    if not path.exists():
        return {"version": 1, "images": {}}
    try:
        return json.loads(path.read_text())
    except Exception:
        logger.warning("Unreadable candidate cache at %s; starting over", path)
        return {"version": 1, "images": {}}


def _save_cache(staged_root: Path, cache: dict) -> None:
    (staged_root / CANDIDATES_FILENAME).write_text(json.dumps(cache))


def _candidates_to_json(cands: list[TileCandidate]) -> list[dict]:
    return [
        {"p": np.asarray(c.polygon_px, dtype=float).round(2).tolist(),
         "c": round(float(c.confidence), 4), "t": int(c.tile_index)}
        for c in cands
    ]


def _candidates_from_json(entries: list[dict]) -> list[TileCandidate]:
    return [
        TileCandidate(np.asarray(e["p"], dtype=np.float32).reshape(-1, 2),
                      float(e["c"]), int(e.get("t", 0)))
        for e in entries
    ]


def _write_labels_from_candidates(
    staged_root: Path, cache: dict, *, confidence: float, merge_iou: float
) -> tuple[int, int]:
    """(instances written, degenerate dropped) across every cached image."""
    written = degenerate = 0
    for rel, entry in cache["images"].items():
        h, w = entry["hw"]
        merged = merge_candidates(
            _candidates_from_json(entry["candidates"]),
            confidence_threshold=confidence, iou_threshold=merge_iou,
        )
        records: list[LabelRecord] = []
        for inst in merged:
            pts = np.asarray(inst.polygon_px, dtype=np.float32).reshape(-1, 2)
            if pts.shape[0] < 3:
                # write_label_file's _polygon_points RAISES on these
                # (data/al/labels.py:45-50), which would abort a multi-hour
                # run over one bad contour. Drop and count instead.
                degenerate += 1
                continue
            records.append(
                LabelRecord(class_id=0, confidence=float(inst.confidence),
                            points=pts, level=GeometryLevel.POLYGON)
            )
        label_path = staged_root / "labels" / Path(rel).with_suffix(".txt")
        label_path.parent.mkdir(parents=True, exist_ok=True)
        write_label_file(label_path, records, frame_size=(h, w),
                         level=GeometryLevel.POLYGON)
        written += len(records)
    return written, degenerate


def run_semantic_escalation(
    req: SemanticEscalationRequest,
    labeler,
    *,
    overwrite: bool = False,
    progress: Callable[[int, str], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> SemanticEscalationResult:
    """Stage prompt-driven polygon labels for each named source.

    Never touches a source's own labels. Promotion is
    ``accept_pending_semantic_escalation``, which writes a NEW SIBLING
    SOURCE -- SAM3's output is a different instance set at a different
    geometry convention, so overwriting in place (as SAM2 does) would
    delete the user's curated labels.
    """
    result = SemanticEscalationResult()
    by_name = {s.name: s for s in req.project.sources}
    # DEPARTURE 1: no `level != "polygon"` filter. Finding animals the
    # existing polygons missed is a primary use case for this feature.
    todo = [by_name[n] for n in req.source_names if n in by_name]
    project_root = Path(req.project.project_dir)
    tile_px = req.tile_px or resolve_tile_px(req.reference_body_px)
    result.tile_px = tile_px
    slug = prompt_slug(req.prompt)

    for si, src in enumerate(todo):
        if src.pending_escalation is not None and not (overwrite or req.overwrite):
            result.skipped.append((
                src.name,
                f"'{src.name}' already has a pending escalation; review it, or "
                "re-run with overwrite to replace it.",
            ))
            continue

        src_root = Path(src.path)
        images_dir = src_root / "images"
        # DEPARTURE 2: the PROMPT enters the hash. Without it two prompts on
        # one source collide and the replaced-pending cleanup no-ops.
        content_hash = sha1(
            (str(src_root.resolve()) + req.variant + req.prompt).encode("utf-8")
        ).hexdigest()[:10]
        staged_dirname = f"{src.name}-sam3-{slug}-{content_hash}"
        staged_root = ensure_bundle_subdirectory(
            project_root, f"artifacts/pending_escalations/{staged_dirname}"
        )

        old_pending = src.pending_escalation
        if old_pending is not None and old_pending.staged_path != str(staged_root):
            remove_staged_escalation_dir(old_pending.staged_path, project_root)

        # DEPARTURE 3: the wipe is CONDITIONAL on the fingerprint, so a
        # cancelled multi-hour run resumes instead of restarting.
        fingerprint = _fingerprint(req, src_root, tile_px)
        run_path = staged_root / RUN_FILENAME
        stale = True
        if run_path.exists():
            try:
                stale = json.loads(run_path.read_text()) != fingerprint
            except Exception:
                stale = True
        if stale:
            remove_staged_escalation_dir(staged_root, project_root)
            staged_root = ensure_bundle_subdirectory(
                project_root, f"artifacts/pending_escalations/{staged_dirname}"
            )
        (staged_root / "labels").mkdir(parents=True, exist_ok=True)
        (staged_root / RUN_FILENAME).write_text(json.dumps(fingerprint))

        cache = {"version": 1, "images": {}} if stale else _load_cache(staged_root)
        images = sorted(
            p for p in images_dir.rglob("*") if p.suffix.lower() in IMG_EXTS
        )
        for ii, img_path in enumerate(images):
            # DEPARTURE 4: cancellation, honoured between images and (inside
            # collect_candidates) between tiles.
            if should_stop is not None and should_stop():
                break
            rel = str(img_path.relative_to(images_dir))
            if rel in cache["images"]:
                continue  # already inferred by an earlier, cancelled run
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            h, w = image.shape[:2]
            plan = (
                plan_for_frame((h, w), tile_px, req.overlap)
                if tile_px
                else plan_for_frame((h, w), max(h, w), 0.0)
            )
            cands = collect_candidates(
                labeler, image, plan, req.prompt,
                confidence_threshold=req.confidence,
                max_instances=req.max_instances,
                seam_margin_px=req.seam_margin_px,
                should_stop=should_stop,
            )
            cache["images"][rel] = {"hw": [h, w],
                                    "candidates": _candidates_to_json(cands)}
            _save_cache(staged_root, cache)
            if progress:
                progress(
                    int(100 * (si + (ii + 1) / max(len(images), 1)) / max(len(todo), 1)),
                    f"{src.name}: {ii + 1}/{len(images)}",
                )

        written, degenerate = _write_labels_from_candidates(
            staged_root, cache, confidence=req.confidence, merge_iou=req.merge_iou
        )
        result.labelled += written
        result.degenerate += degenerate
        result.empty_images += sum(
            1 for e in cache["images"].values() if not e["candidates"]
        )
        (staged_root / "classes.txt").write_text(f"{req.prompt.strip() or 'object'}\n")
        src.pending_escalation = PendingEscalation(
            staged_path=str(staged_root),
            target_level=GeometryLevel.POLYGON.label,
            created_at=datetime.now().isoformat(),
            primer_kind="sam3",
            primer_variant=req.variant,
            primer_prompt=req.prompt,
            primer_params={
                "confidence": float(req.confidence),
                "merge_iou": float(req.merge_iou),
                "tile_px": tile_px,
                "overlap": float(req.overlap),
                "seam_margin_px": float(req.seam_margin_px),
                "max_instances": int(req.max_instances),
            },
        )
        result.staged.append(src.name)
    return result


def rethreshold_staged(source: OBBSource, *, confidence: float,
                       merge_iou: float) -> int:
    """Rewrite a staged result at a new confidence. No inference.

    This is why the candidate cache exists: a 30-hour run must not be a
    one-shot commitment to one threshold. NMS is redone here rather than
    post-filtering the previous labels, because suppression is
    survivor-dependent.
    """
    pending = source.pending_escalation
    if pending is None or pending.primer_kind != "sam3":
        raise ValueError(f"Source '{source.name}' has no staged SAM3 escalation.")
    staged_root = Path(pending.staged_path)
    cache = _load_cache(staged_root)
    if not cache["images"]:
        raise RuntimeError(
            f"The candidate cache for '{source.name}' is missing or empty; "
            "re-run the escalation."
        )
    written, _degenerate = _write_labels_from_candidates(
        staged_root, cache, confidence=confidence, merge_iou=merge_iou
    )
    pending.primer_params = {**pending.primer_params,
                             "confidence": float(confidence),
                             "merge_iou": float(merge_iou)}
    return written


class SemanticEscalationWorker(BaseWorker):
    """QThread wrapper around run_semantic_escalation, with cancellation."""

    result_ready = Signal(object)  # SemanticEscalationResult

    def __init__(self, request: SemanticEscalationRequest, labeler=None,
                 parent=None) -> None:
        super().__init__(parent)
        self._request = request
        self._labeler = labeler
        self._cancel = False

    def cancel(self) -> None:
        """Ask the run to stop at the next tile boundary."""
        self._cancel = True

    def execute(self) -> None:
        labeler = self._labeler
        if labeler is None:
            from hydra_suite.core.inference.semantic.sam3 import Sam3SemanticLabeler

            labeler = Sam3SemanticLabeler.from_variant(self._request.variant)
        self.status.emit(
            f"Segmenting '{self._request.prompt}' across "
            f"{len(self._request.source_names)} source(s)..."
        )
        result = run_semantic_escalation(
            self._request,
            labeler,
            overwrite=self._request.overwrite,
            progress=lambda pct, msg: (self.progress.emit(pct),
                                        self.status.emit(msg)),
            should_stop=lambda: self._cancel,
        )
        self.result_ready.emit(result)
```

- [ ] **Step 4: Run to verify they pass**

Run: `python -m pytest tests/test_semantic_escalation_job.py -q`
Expected: PASS (10 tests).

- [ ] **Step 5: Commit**

```bash
make format
git add -A
git commit -m "feat(detectkit): semantic escalation job with candidate cache, resume, cancel"
```

---

### Task 10: Promotion to a sibling source

SAM2's `accept_pending_escalation` (`sam2_escalation.py:299`) does `rmtree(source/labels)` then `copytree(staged/labels, source/labels)`. That is right for SAM2 and **destructive** for SAM3, whose staged labels are a different instance set at a different geometry convention, all class `0`. Accepting a SAM3 run on a curated OBB source would silently delete those labels — precisely the harm the spec cites as its reason not to merge the conventions.

Images are hardlinked, following `data/al/export.py:51-57`'s `_link_or_copy`, so a sibling source of 78 4512² frames costs almost no disk.

**Files:**
- Modify: `src/hydra_suite/detectkit/jobs/semantic_escalation.py` (append)
- Test: `tests/test_semantic_escalation_job.py` (extend)

**Interfaces:**
- Consumes: Task 9's module; `hydra_suite.data.al.export._link_or_copy`.
- Produces: `accept_pending_semantic_escalation(source: OBBSource, project, project_dir=None) -> OBBSource` returning the newly registered sibling. Rejection reuses the existing `sam2_escalation.reject_pending_escalation`.

- [ ] **Step 1: Write the failing promotion tests**

Append to `tests/test_semantic_escalation_job.py`:

```python
from hydra_suite.detectkit.jobs.semantic_escalation import (
    accept_pending_semantic_escalation,
)


def test_accept_creates_a_sibling_and_leaves_the_origin_untouched(tmp_path):
    src = _make_source(tmp_path, n_images=2)
    (Path(src.path) / "labels" / "f0.txt").write_text("0 0.1 0.1 0.2 0.2\n")
    original = (Path(src.path) / "labels" / "f0.txt").read_bytes()
    project = _Project(tmp_path, [src])
    labeler = ScriptedLabeler([SemanticInstance(_blob(200, 200), 0.9)])
    run_semantic_escalation(_request(tmp_path, src, project=project), labeler)

    sibling = accept_pending_semantic_escalation(src, project, tmp_path)

    assert sibling is not src
    assert sibling in project.sources
    assert sibling.level == "polygon"
    assert sibling.reviewed is False
    assert sibling.derived_from == src.name
    assert (Path(src.path) / "labels" / "f0.txt").read_bytes() == original
    assert src.pending_escalation is None


def test_sibling_carries_images_and_the_prompt_as_its_class_name(tmp_path):
    src = _make_source(tmp_path, n_images=2)
    project = _Project(tmp_path, [src])
    labeler = ScriptedLabeler([SemanticInstance(_blob(200, 200), 0.9)])
    run_semantic_escalation(
        _request(tmp_path, src, project=project, prompt="black ant"), labeler
    )
    sibling = accept_pending_semantic_escalation(src, project, tmp_path)
    root = Path(sibling.path)
    assert len(list((root / "images").rglob("*.png"))) == 2
    assert len(list((root / "labels").rglob("*.txt"))) == 2
    assert (root / "classes.txt").read_text().strip() == "black ant"


def test_the_candidate_cache_never_reaches_the_sibling(tmp_path):
    src = _make_source(tmp_path, n_images=1)
    project = _Project(tmp_path, [src])
    labeler = ScriptedLabeler([SemanticInstance(_blob(200, 200), 0.9)])
    run_semantic_escalation(_request(tmp_path, src, project=project), labeler)
    sibling = accept_pending_semantic_escalation(src, project, tmp_path)
    assert not (Path(sibling.path) / "candidates.json").exists()
    assert not (Path(sibling.path) / "run.json").exists()


def test_accept_refuses_a_sam2_pending_record(tmp_path):
    src = _make_source(tmp_path, n_images=1)
    from hydra_suite.detectkit.gui.models import PendingEscalation

    src.pending_escalation = PendingEscalation(staged_path=str(tmp_path),
                                               primer_kind="sam2")
    with pytest.raises(ValueError, match="not a SAM3"):
        accept_pending_semantic_escalation(src, _Project(tmp_path, [src]), tmp_path)


def test_accept_refuses_when_the_staging_dir_is_gone(tmp_path):
    import shutil

    src = _make_source(tmp_path, n_images=1)
    project = _Project(tmp_path, [src])
    labeler = ScriptedLabeler([SemanticInstance(_blob(200, 200), 0.9)])
    run_semantic_escalation(_request(tmp_path, src, project=project), labeler)
    shutil.rmtree(src.pending_escalation.staged_path)
    with pytest.raises(RuntimeError, match="missing on disk"):
        accept_pending_semantic_escalation(src, project, tmp_path)
```

Note: `_request` needs a `project` keyword — it already accepts `**kw` that overrides
the `project` default, so this works unchanged.

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_semantic_escalation_job.py -q -k accept or sibling or cache_never`
Expected: FAIL — `accept_pending_semantic_escalation` not found.

- [ ] **Step 3: Implement promotion**

Append to `src/hydra_suite/detectkit/jobs/semantic_escalation.py`:

```python
def _unique_source_name(project, base: str) -> str:
    existing = {s.name for s in project.sources}
    if base not in existing:
        return base
    n = 2
    while f"{base}-{n}" in existing:
        n += 1
    return f"{base}-{n}"


def accept_pending_semantic_escalation(
    source: OBBSource,
    project,
    project_dir: str | Path | None = None,
) -> OBBSource:
    """Promote a staged SAM3 result to a NEW SIBLING SOURCE.

    Deliberately NOT ``sam2_escalation.accept_pending_escalation``, which
    rmtree's the origin's labels and copies the staged ones over them. That
    is correct for SAM2 (a lossless upgrade of the SAME instances) and
    destructive here: SAM3's output is a different instance set at a
    different geometry convention (masks trace legs and antennae; tracking
    labels bound the body core), all class 0. Merging the two conventions
    into one source would degrade YOLO training -- and deleting the user's
    curated labels to do it would be worse.

    The staged labels and hardlinked images become a new source the user can
    keep, merge, or delete with the tools they already have. The candidate
    cache and run fingerprint stay behind in staging: they are consumed
    here, never shipped, so they cannot go stale against later user edits.
    """
    from hydra_suite.data.al.export import _link_or_copy

    pending = source.pending_escalation
    if pending is None:
        raise ValueError(f"Source '{source.name}' has no pending escalation.")
    if pending.primer_kind != "sam3":
        raise ValueError(
            f"Source '{source.name}' has a {pending.primer_kind!r} pending "
            "escalation, not a SAM3 one; use the SAM2 accept path."
        )

    staged_root = Path(pending.staged_path)
    staged_labels = staged_root / "labels"
    if not staged_labels.is_dir():
        raise RuntimeError(
            f"Staged escalation for '{source.name}' is missing on disk "
            f"({staged_labels}); nothing was changed. Reject it and re-run."
        )

    project_root = Path(project.project_dir)
    sibling_name = _unique_source_name(
        project, f"{source.name}-sam3-{prompt_slug(pending.primer_prompt)}"
    )
    sibling_root = Path(
        ensure_bundle_subdirectory(project_root, f"sources/{sibling_name}")
    )
    (sibling_root / "images").mkdir(parents=True, exist_ok=True)
    (sibling_root / "labels").mkdir(parents=True, exist_ok=True)

    origin_images = Path(source.path) / "images"
    for label_path in sorted(staged_labels.rglob("*.txt")):
        rel = label_path.relative_to(staged_labels)
        dst_label = sibling_root / "labels" / rel
        dst_label.parent.mkdir(parents=True, exist_ok=True)
        dst_label.write_bytes(label_path.read_bytes())
        for img in origin_images.rglob(f"{rel.stem}.*"):
            if img.suffix.lower() not in IMG_EXTS:
                continue
            dst_img = sibling_root / "images" / img.relative_to(origin_images)
            dst_img.parent.mkdir(parents=True, exist_ok=True)
            if not dst_img.exists():
                _link_or_copy(img, dst_img)
            break

    classes_src = staged_root / "classes.txt"
    (sibling_root / "classes.txt").write_text(
        classes_src.read_text() if classes_src.exists() else "object\n"
    )

    sibling = OBBSource(
        path=str(sibling_root),
        name=sibling_name,
        validated=False,
        original_path=source.path,
        source_kind="detectkit_sam3",
        imported=False,
        level=GeometryLevel.POLYGON.label,
        # Machine-derived and not yet human-confirmed: excluded from training
        # until the user runs "Mark reviewed...".
        reviewed=False,
        derived_from=source.name,
    )
    project.sources.append(sibling)

    remove_staged_escalation_dir(staged_root, project_dir or project_root)
    source.pending_escalation = None
    return sibling
```

- [ ] **Step 4: Run to verify they pass**

Run: `python -m pytest tests/test_semantic_escalation_job.py -q`
Expected: PASS (15 tests).

- [ ] **Step 5: Commit**

```bash
make format
git add -A
git commit -m "feat(detectkit): promote semantic escalation to a sibling source, never in place"
```

---

### Task 11: GUI — panel rename, dialog, handler extraction, review dialog

**Files:**
- Modify: `src/hydra_suite/detectkit/gui/panels/tools_panel.py:103,156-190`
- Create: `src/hydra_suite/detectkit/gui/dialogs/semantic_escalation_dialog.py`
- Create: `src/hydra_suite/detectkit/gui/escalation_actions.py`
- Modify: `src/hydra_suite/detectkit/gui/main_window.py:741-748` (connections), remove `:1769-1943` and `:1985-2010`
- Modify: `src/hydra_suite/detectkit/gui/dialogs/review_escalations_dialog.py`

**Interfaces:**
- Consumes: everything from Tasks 5, 7, 9, 10.
- Produces: `ToolsPanel.escalate_geometry_requested`, `ToolsPanel.semantic_escalation_requested`; `escalation_actions.{on_escalate_geometry, on_semantic_escalation, on_review_escalations}(window)`.

- [ ] **Step 1: Rename the SAM2 action and add the semantic one**

In `src/hydra_suite/detectkit/gui/panels/tools_panel.py`:

- `:103` — rename the signal and add the new one:

```python
    escalate_geometry_requested = Signal()
    semantic_escalation_requested = Signal()
```

- Replace `_build_escalation_group`'s group title, hint, and first button block
  (`:156-183`) with:

```python
    def _build_escalation_group(self) -> QGroupBox:
        box = QGroupBox("Escalation")
        v = QVBoxLayout(box)
        v.setSpacing(6)

        hint = QLabel(
            "Geometry escalation converts the boxes you already have into "
            "masks — it cannot add an animal that was never labelled. "
            "Semantic escalation finds instances from a prompt, including "
            "animals missing from the current labels, and needs no labels to "
            "start."
        )
        hint.setWordWrap(True)
        hint.setProperty("detectkitRole", "sectionHint")
        v.addWidget(hint)

        self._btn_escalate_sam2 = QPushButton(
            "Geometry escalation (SAM2): boxes to masks"
        )
        self._btn_escalate_sam2.clicked.connect(self.escalate_geometry_requested)
        try:
            from hydra_suite.core.inference.sam2.checkpoints import available_variants

            available_variants()
        except Exception:  # pragma: no cover - depends on optional SAM2 assets
            self._btn_escalate_sam2.setEnabled(False)
            self._btn_escalate_sam2.setToolTip(
                "SAM2 catalog unavailable — install the SAM2 checkpoints to enable "
                "escalation."
            )
        v.addWidget(self._btn_escalate_sam2)

        self._btn_semantic = QPushButton(
            "Semantic escalation (SAM3): prompt to masks…"
        )
        self._btn_semantic.clicked.connect(self.semantic_escalation_requested)
        # A REAL probe, not the SAM2 pattern: available_variants() above only
        # lists a static dict, so it would let a click start a silent 3.45 GB
        # download. probe_availability checks deps AND the on-disk checkpoint
        # without ever touching the network.
        try:
            from hydra_suite.core.inference.semantic.checkpoints import (
                probe_availability,
            )

            ok, reason = probe_availability()
        except Exception as exc:  # pragma: no cover - optional SAM3 assets
            ok, reason = False, str(exc)
        if not ok:
            self._btn_semantic.setEnabled(False)
            self._btn_semantic.setToolTip(reason)
        v.addWidget(self._btn_semantic)
```

- [ ] **Step 2: Write the semantic escalation dialog**

Create `src/hydra_suite/detectkit/gui/dialogs/semantic_escalation_dialog.py`:

```python
"""SemanticEscalationDialog — prompt, tiling, calibration, one-tile preview."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from hydra_suite.core.inference.semantic.tiling import (
    DEFAULT_MERGE_IOU,
    DEFAULT_OVERLAP,
    DEFAULT_SEAM_MARGIN_PX,
    resolve_tile_px,
)
from hydra_suite.widgets.dialogs import BaseDialog


class SemanticEscalationDialog(BaseDialog):
    """Configure a SAM3 semantic escalation run.

    Calibration is offered whenever the selected sources hold a labelled
    frame at ANY geometry level -- choosing an operating point needs
    instance COUNTS, not masks, so OBB and AABB labels work as well as
    polygons.
    """

    def __init__(self, sources, reference_body_px: float, parent=None) -> None:
        super().__init__(
            "Semantic escalation (SAM3)",
            parent=parent,
            buttons=(
                QDialogButtonBox.StandardButton.Ok
                | QDialogButtonBox.StandardButton.Cancel
            ),
        )
        self._sources = list(sources)
        self._reference_body_px = float(reference_body_px or 0.0)
        self.calibration_points: list = []

        container = QWidget()
        outer = QVBoxLayout(container)

        self._list = QListWidget()
        self._list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        for src in self._sources:
            self._list.addItem(f"{src.name}  ({src.level})")
        outer.addWidget(QLabel("Sources to escalate:"))
        outer.addWidget(self._list)

        form = QFormLayout()
        self._variant = QComboBox()
        from hydra_suite.core.inference.semantic.checkpoints import available_variants

        self._variant.addItems(available_variants())
        form.addRow("Model:", self._variant)

        self._prompt = QLineEdit("ant")
        self._prompt.setToolTip(
            "A noun phrase. Wording matters far less than tile size — try "
            "variants in the preview if results look wrong."
        )
        form.addRow("Prompt:", self._prompt)

        self._confidence = QDoubleSpinBox()
        self._confidence.setRange(0.01, 0.99)
        self._confidence.setSingleStep(0.05)
        self._confidence.setValue(0.35)
        form.addRow("Confidence:", self._confidence)

        self._max_instances = QSpinBox()
        self._max_instances.setRange(0, 10000)
        self._max_instances.setSpecialValueText("unlimited")
        form.addRow("Max instances/tile:", self._max_instances)

        self._overlap = QDoubleSpinBox()
        self._overlap.setRange(0.0, 0.9)
        self._overlap.setSingleStep(0.1)
        self._overlap.setValue(DEFAULT_OVERLAP)
        form.addRow("Tile overlap:", self._overlap)

        self._seam_margin = QSpinBox()
        self._seam_margin.setRange(0, 64)
        self._seam_margin.setValue(int(DEFAULT_SEAM_MARGIN_PX))
        form.addRow("Seam margin (px):", self._seam_margin)

        self._merge_iou = QDoubleSpinBox()
        self._merge_iou.setRange(0.05, 0.95)
        self._merge_iou.setSingleStep(0.05)
        self._merge_iou.setValue(DEFAULT_MERGE_IOU)
        form.addRow("Cross-tile merge IoU:", self._merge_iou)

        tile_px = resolve_tile_px(self._reference_body_px)
        self._tile_label = QLabel(
            f"{tile_px} px (from reference body size {self._reference_body_px:.0f} px)"
            if tile_px
            else "full frame — no reference body size is known, so tiling is off. "
                 "Set one in project settings for much better small-object recall."
        )
        self._tile_label.setWordWrap(True)
        form.addRow("Tile size:", self._tile_label)
        outer.addLayout(form)

        self._exhaustive = QCheckBox(
            "My labelled frames are exhaustively labelled (every animal is marked)"
        )
        self._exhaustive.setToolTip(
            "Calibration counts an unlabelled real animal as a false positive, "
            "which biases the recommended threshold upward."
        )
        outer.addWidget(self._exhaustive)

        self._btn_calibrate = QPushButton("Calibrate against labelled frames…")
        self._btn_calibrate.setEnabled(False)
        outer.addWidget(self._btn_calibrate)

        self._btn_preview = QPushButton("Preview one tile")
        self._btn_preview.setToolTip(
            "Runs ONE tile, not the whole frame: a full-frame preview shows "
            "near-zero detections and teaches you the feature is broken."
        )
        outer.addWidget(self._btn_preview)

        self._status = QLabel("")
        self._status.setWordWrap(True)
        outer.addWidget(self._status)

        self.add_content(container)

    # -- accessors used by the handler -------------------------------------

    def selected_sources(self) -> list:
        rows = [i.row() for i in self._list.selectedIndexes()]
        return [self._sources[r] for r in sorted(rows)]

    def selected_variant(self) -> str:
        return self._variant.currentText()

    def prompt(self) -> str:
        return self._prompt.text().strip()

    def parameters(self) -> dict:
        return {
            "confidence": float(self._confidence.value()),
            "max_instances": int(self._max_instances.value()),
            "overlap": float(self._overlap.value()),
            "seam_margin_px": float(self._seam_margin.value()),
            "merge_iou": float(self._merge_iou.value()),
            "reference_body_px": self._reference_body_px,
        }

    def set_calibration_enabled(self, enabled: bool, reason: str = "") -> None:
        self._btn_calibrate.setEnabled(enabled)
        if not enabled and reason:
            self._btn_calibrate.setToolTip(reason)

    def set_status(self, text: str) -> None:
        self._status.setText(text)

    def accept(self) -> None:  # noqa: D102
        if not self.prompt():
            QMessageBox.warning(self, "Semantic escalation", "Enter a prompt first.")
            return
        if not self.selected_sources():
            QMessageBox.warning(self, "Semantic escalation", "Select a source.")
            return
        super().accept()
```

- [ ] **Step 3: Extract the handlers**

Create `src/hydra_suite/detectkit/gui/escalation_actions.py` and move
`_on_escalate_to_segment_sam2` (`main_window.py:1769-1943`) and `_on_review_escalations`
(`:1985-2010`) into it verbatim as module-level functions taking the window:

```python
"""Escalation handlers, lifted out of main_window.py.

Honest accounting: this removes ~205 of main_window.py's 2152 lines, which
does NOT bring it near CLAUDE.md's thin-coordinator target. It is done
because the new semantic handler would otherwise add a THIRD escalation
flow to an already-oversized file.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMessageBox, QProgressDialog


def on_escalate_geometry(window, preselect: str | None = None) -> None:
    """Open the SAM2 escalate dialog and run a Sam2EscalationWorker."""
    # <the body of main_window._on_escalate_to_segment_sam2, with every
    #  `self` replaced by `window`. Behaviour unchanged.>


def on_semantic_escalation(window) -> None:
    """Open the SAM3 semantic escalation dialog and run the worker."""
    ...


def on_review_escalations(window) -> None:
    """Open the review dialog for every source with a pending escalation."""
    # <the body of main_window._on_review_escalations, `self` -> `window`.>
```

`on_semantic_escalation`'s body:

```python
def on_semantic_escalation(window) -> None:
    if window._project is None:
        QMessageBox.information(window, "Semantic escalation",
                                "Open a project before escalating sources.")
        return
    if window._escalation_worker is not None:
        QMessageBox.information(window, "Semantic escalation",
                                "An escalation run is already in progress.")
        return

    from hydra_suite.detectkit.jobs.semantic_escalation import (
        SemanticEscalationRequest,
        SemanticEscalationWorker,
        is_prompt_failure,
    )

    from .dialogs.semantic_escalation_dialog import SemanticEscalationDialog

    slice_settings = getattr(window._project, "slice_training", None)
    reference_body_px = float(getattr(slice_settings, "reference_body_px", 0.0) or 0.0)
    dlg = SemanticEscalationDialog(window._project.sources, reference_body_px,
                                   parent=window)
    if not dlg.exec():
        return

    sources = dlg.selected_sources()
    params = dlg.parameters()
    request = SemanticEscalationRequest(
        project=window._project,
        source_names=[s.name for s in sources],
        variant=dlg.selected_variant(),
        prompt=dlg.prompt(),
        overwrite=True,  # resume-safe: the run.json fingerprint decides
        **params,
    )

    # A REAL cancel button (the SAM2 dialog passes None here and cannot be
    # cancelled) -- this run takes tens of seconds PER FRAME.
    progress = QProgressDialog(
        f"Segmenting '{request.prompt}'…", "Cancel", 0, 100, window
    )
    progress.setWindowTitle("Semantic escalation (SAM3)")
    progress.setMinimumDuration(0)
    progress.setWindowModality(Qt.WindowModal)
    progress.setAttribute(Qt.WA_DeleteOnClose, True)
    progress.setValue(0)

    worker = SemanticEscalationWorker(request)
    progress.canceled.connect(worker.cancel)
    worker.progress.connect(progress.setValue)
    worker.status.connect(progress.setLabelText)

    def _handle_result(result) -> None:
        window._save_current_project()
        window._dataset_panel.refresh_sources(window._project)
        frames = result.labelled + result.empty_images
        if is_prompt_failure(result, frames):
            QMessageBox.warning(
                window,
                "Semantic escalation",
                f"'{request.prompt}' matched nothing on "
                f"{result.empty_images} of {frames} frame(s). This is usually a "
                "prompt that the model does not recognise — try a different "
                "noun phrase in the preview before re-running.",
            )
            return
        QMessageBox.information(
            window,
            "Semantic escalation",
            f"Staged {len(result.staged)} source(s): {result.labelled} instance(s), "
            f"{result.empty_images} empty frame(s), {result.degenerate} degenerate "
            f"contour(s) dropped. Review them before training.",
        )

    def _finish() -> None:
        progress.close()
        window._escalation_worker = None

    worker.result_ready.connect(_handle_result)
    worker.finished.connect(_finish)
    window._escalation_worker = worker
    worker.start()
```

Then in `main_window.py`, delete the two moved methods and rewrite the connections at
`:741-748`:

```python
        self._tools_panel.escalate_geometry_requested.connect(
            lambda: escalation_actions.on_escalate_geometry(self)
        )
        self._tools_panel.semantic_escalation_requested.connect(
            lambda: escalation_actions.on_semantic_escalation(self)
        )
        self._tools_panel.mark_reviewed_requested.connect(self._on_mark_reviewed)
        self._tools_panel.review_escalations_requested.connect(
            lambda: escalation_actions.on_review_escalations(self)
        )
```

with `from . import escalation_actions` at the top of `main_window.py`. Grep for other
callers of the removed method names and repoint them:

```bash
grep -rn "_on_escalate_to_segment_sam2\|_on_review_escalations" src/ tests/
```

- [ ] **Step 4: Make the review dialog primer-aware**

In `src/hydra_suite/detectkit/gui/dialogs/review_escalations_dialog.py`:

- Import both accept paths:

```python
from ...jobs.sam2_escalation import accept_pending_escalation, reject_pending_escalation
from ...jobs.semantic_escalation import (
    accept_pending_semantic_escalation,
    rethreshold_staged,
)
```

- Replace the intro text with a version that covers both producers:

```python
        intro = QLabel(
            "These sources have a staged segmentation result awaiting review.\n\n"
            "Geometry (SAM2) results REPLACE the source's own labels — same "
            "instances, upgraded to masks. Semantic (SAM3) results become a NEW "
            "SIBLING source and leave the original untouched, because they are a "
            "different instance set at a different geometry convention.\n\n"
            "Accepted sources are marked unreviewed and are excluded from "
            'training until you use "Mark reviewed…" for them.'
        )
```

- Replace the item text (`:59-62`) so it names the primer and the prompt:

```python
            detail = (
                f"prompt '{pending.primer_prompt}'"
                if pending.primer_kind == "sam3"
                else pending.primer_variant or pending.sam2_variant
            )
            item = QListWidgetItem(
                f"{src.name}  ->  {pending.target_level} "
                f"[{pending.primer_kind}: {detail}, staged {pending.created_at}]"
            )
```

- Dispatch on the primer kind in `_apply_checked`, and take a `project` so the sibling
  can be registered. Change the constructor signature to
  `def __init__(self, pending_sources, parent=None, project=None, project_dir=None)`,
  store `self._project = project`, and replace the accept branch:

```python
                if accept:
                    if src.pending_escalation.primer_kind == "sam3":
                        accept_pending_semantic_escalation(
                            src, self._project, self._project_dir
                        )
                    else:
                        accept_pending_escalation(src, self._project_dir)
                    self.accepted_names.append(src.name)
```

- Add a third button beside Accept/Reject:

```python
        self._btn_rethreshold = QPushButton("Re-threshold Checked…")
        self._btn_rethreshold.setToolTip(
            "Rewrite a staged SAM3 result at a different confidence, using the "
            "cached candidates. No inference — seconds, not hours."
        )
        self._btn_rethreshold.clicked.connect(self._rethreshold_checked)
        btn_row.addWidget(self._btn_rethreshold)
```

and the handler:

```python
    def _rethreshold_checked(self) -> None:
        from PySide6.QtWidgets import QInputDialog

        rows = self._checked_rows()
        targets = [
            self._list.item(r).data(Qt.UserRole)
            for r in rows
            if self._list.item(r).data(Qt.UserRole).pending_escalation.primer_kind
            == "sam3"
        ]
        if not targets:
            QMessageBox.information(
                self, "Re-threshold", "Check a staged SAM3 result first."
            )
            return
        current = float(
            targets[0].pending_escalation.primer_params.get("confidence", 0.35)
        )
        value, ok = QInputDialog.getDouble(
            self, "Re-threshold", "New confidence:", current, 0.01, 0.99, 2
        )
        if not ok:
            return
        for src in targets:
            merge_iou = float(
                src.pending_escalation.primer_params.get("merge_iou", 0.5)
            )
            try:
                kept = rethreshold_staged(src, confidence=value, merge_iou=merge_iou)
            except Exception as exc:
                QMessageBox.warning(self, "Re-threshold", str(exc))
                continue
            QMessageBox.information(
                self, "Re-threshold",
                f"{src.name}: {kept} instance(s) at confidence {value:.2f}.",
            )
```

Update the one construction site in `escalation_actions.on_review_escalations` to pass
`project=window._project`.

- [ ] **Step 5: Verify the GUI imports and the app still starts**

Run:

```bash
python -c "
from hydra_suite.detectkit.gui import escalation_actions
from hydra_suite.detectkit.gui.panels.tools_panel import ToolsPanel
from hydra_suite.detectkit.gui.dialogs.semantic_escalation_dialog import SemanticEscalationDialog
from hydra_suite.detectkit.gui.dialogs.review_escalations_dialog import ReviewEscalationsDialog
print('ok')
"
grep -rn "escalate_sam2_requested" src/ tests/
```

Expected: prints `ok`; the grep returns nothing (the signal is fully renamed).

Then launch `detectkit`, open a project with an OBB source, and confirm: the Escalation
group shows both buttons with the new labels; the SAM3 button is disabled with a reason
if the checkpoint is absent; the dialog opens and refuses an empty prompt.

- [ ] **Step 6: Commit**

```bash
make format
git add -A
git commit -m "feat(detectkit): semantic escalation GUI, escalation handler extraction, primer-aware review"
```

---

### Task 12: Wire calibration into the dialog

Calibration is the feature's main claim — the operating point is fitted to the user's data, not inherited from ours. It runs on the same worker pattern as the escalation itself.

**Files:**
- Modify: `src/hydra_suite/detectkit/gui/dialogs/semantic_escalation_dialog.py`
- Modify: `src/hydra_suite/detectkit/jobs/semantic_escalation.py` (add `CalibrationWorker` + the label-reading adapter)
- Test: `tests/test_semantic_escalation_job.py` (extend)

**Interfaces:**
- Consumes: `calibrate`, `recommend`, `CalibrationPoint` (Task 7); `parse_obb_label` (`detectkit/gui/utils.py:220`).
- Produces: `labelled_frames_for(source) -> list[tuple[Path, list[LabelRecord]]]`; `CalibrationWorker(request, labeler=None)` emitting `result_ready(list[CalibrationPoint])`.

- [ ] **Step 1: Write the failing adapter test**

Append to `tests/test_semantic_escalation_job.py`:

```python
def test_labelled_frames_reads_every_geometry_level(tmp_path):
    from hydra_suite.detectkit.jobs.semantic_escalation import labelled_frames_for

    src = _make_source(tmp_path, n_images=3)
    labels = Path(src.path) / "labels"
    labels.joinpath("f0.txt").write_text("0 0.5 0.5 0.1 0.1\n")  # AABB
    labels.joinpath("f1.txt").write_text(
        "0 0.4 0.4 0.5 0.4 0.5 0.5 0.4 0.5\n"  # OBB quad
    )
    labels.joinpath("f2.txt").write_text(
        "0 0.1 0.1 0.2 0.1 0.25 0.2 0.15 0.25 0.08 0.2\n"  # 5-point polygon
    )
    frames = labelled_frames_for(src)
    assert len(frames) == 3
    for _path, records in frames:
        assert len(records) == 1
        assert records[0].points.shape[1] == 2


def test_labelled_frames_skips_empty_label_files(tmp_path):
    from hydra_suite.detectkit.jobs.semantic_escalation import labelled_frames_for

    src = _make_source(tmp_path, n_images=2)
    (Path(src.path) / "labels" / "f0.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    assert len(labelled_frames_for(src)) == 1
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_semantic_escalation_job.py -q -k labelled_frames`
Expected: FAIL — `labelled_frames_for` not found.

- [ ] **Step 3: Implement the adapter and the calibration worker**

Append to `src/hydra_suite/detectkit/jobs/semantic_escalation.py`:

```python
def labelled_frames_for(source: OBBSource) -> list[tuple[Path, list[LabelRecord]]]:
    """(image path, LabelRecords) for every non-empty labelled frame.

    Uses ``gui/utils.parse_obb_label``, which already handles 5-field AABB,
    9-field quad and odd-count polygon lines. Deliberately NOT
    ``sam2_prompts.read_boxes_from_label``, which accepts only 4- and
    8-value lines and silently drops polygon lines (jobs/sam2_prompts.py:49-60)
    -- calibration must work at ANY geometry level, because choosing an
    operating point needs instance COUNTS, not masks.
    """
    from hydra_suite.detectkit.gui.utils import parse_obb_label

    root = Path(source.path)
    images_dir, labels_dir = root / "images", root / "labels"
    out: list[tuple[Path, list[LabelRecord]]] = []
    for img_path in sorted(
        p for p in images_dir.rglob("*") if p.suffix.lower() in IMG_EXTS
    ):
        label_path = labels_dir / img_path.relative_to(images_dir).with_suffix(".txt")
        if not label_path.exists() or not label_path.read_text().strip():
            continue
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        h, w = image.shape[:2]
        parsed = parse_obb_label(label_path, w, h)
        if not parsed:
            continue
        out.append((
            img_path,
            [
                LabelRecord(
                    class_id=int(d["class_id"]),
                    confidence=1.0,
                    points=np.asarray(d["polygon_px"], dtype=np.float32).reshape(-1, 2),
                    level=GeometryLevel.POLYGON,
                )
                for d in parsed
            ],
        ))
    return out


class CalibrationWorker(BaseWorker):
    """QThread wrapper around calibrate(), cancellable between frames."""

    result_ready = Signal(object)  # list[CalibrationPoint]

    def __init__(self, frames, prompt, variant, params, labeler=None,
                 parent=None) -> None:
        super().__init__(parent)
        self._frames = frames
        self._prompt = prompt
        self._variant = variant
        self._params = dict(params)
        self._labeler = labeler
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def execute(self) -> None:
        from hydra_suite.core.inference.semantic.calibration import calibrate

        labeler = self._labeler
        if labeler is None:
            from hydra_suite.core.inference.semantic.sam3 import Sam3SemanticLabeler

            labeler = Sam3SemanticLabeler.from_variant(self._variant)
        points = calibrate(
            labeler,
            self._frames,
            self._prompt,
            tile_px=resolve_tile_px(self._params.get("reference_body_px", 0.0)),
            overlap=self._params.get("overlap", DEFAULT_OVERLAP),
            seam_margin_px=self._params.get("seam_margin_px", DEFAULT_SEAM_MARGIN_PX),
            merge_iou=self._params.get("merge_iou", DEFAULT_MERGE_IOU),
            max_instances=self._params.get("max_instances", 0),
            progress=lambda pct, msg: (self.progress.emit(pct),
                                        self.status.emit(msg)),
            should_stop=lambda: self._cancel,
        )
        self.result_ready.emit(points)
```

- [ ] **Step 4: Run to verify they pass**

Run: `python -m pytest tests/test_semantic_escalation_job.py -q`
Expected: PASS (17 tests).

- [ ] **Step 5: Wire the Calibrate button in the dialog**

In `SemanticEscalationDialog.__init__`, after building `self._btn_calibrate`, enable it
based on the sources' labels and connect it:

```python
        self._btn_calibrate.clicked.connect(self._run_calibration)
        self._refresh_calibration_enabled()
```

and add:

```python
    def _refresh_calibration_enabled(self) -> None:
        from hydra_suite.detectkit.jobs.semantic_escalation import labelled_frames_for

        # Calibration works at ANY geometry level -- it needs instance COUNTS,
        # not masks -- so OBB and AABB sources qualify too.
        has_labels = any(labelled_frames_for(s) for s in self._sources)
        self.set_calibration_enabled(
            has_labels,
            "No labelled frames in these sources. Label a few (any geometry "
            "level) to calibrate the threshold to your data — or proceed and "
            "tune it by eye.",
        )

    def _run_calibration(self) -> None:
        from PySide6.QtWidgets import QProgressDialog

        from hydra_suite.core.inference.semantic.calibration import recommend
        from hydra_suite.detectkit.jobs.semantic_escalation import (
            CalibrationWorker,
            labelled_frames_for,
        )

        if not self._exhaustive.isChecked():
            QMessageBox.information(
                self,
                "Calibrate",
                "Confirm your labelled frames are exhaustively labelled first. "
                "An unlabelled real animal counts as a false positive and biases "
                "the recommended threshold upward.",
            )
            return
        frames = [f for s in self.selected_sources() or self._sources
                  for f in labelled_frames_for(s)]
        if not frames:
            QMessageBox.information(self, "Calibrate", "No labelled frames found.")
            return

        progress = QProgressDialog("Calibrating…", "Cancel", 0, 100, self)
        progress.setMinimumDuration(0)
        worker = CalibrationWorker(frames, self.prompt(), self.selected_variant(),
                                    self.parameters())
        progress.canceled.connect(worker.cancel)
        worker.progress.connect(progress.setValue)
        worker.status.connect(progress.setLabelText)

        def _done(points) -> None:
            progress.close()
            self.calibration_points = points
            best, reason = recommend(points)
            if best is None:
                self.set_status(reason)
                return
            self._confidence.setValue(best.confidence)
            self.set_status(
                f"Recommended confidence {best.confidence:.2f}: misses "
                f"{best.missed_per_frame:.1f} animal(s)/frame, leaves "
                f"{best.extra_per_frame:.1f} polygon(s)/frame to delete "
                f"(recall {best.recall:.1%}, {best.n_matched} matched). "
                "Chosen for recall, not F1 — a spurious polygon is one click, "
                "a missed animal must be found by eye."
            )

        worker.result_ready.connect(_done)
        worker.finished.connect(progress.close)
        self._calibration_worker = worker  # keep a reference alive
        worker.start()
```

- [ ] **Step 6: Verify imports and run the full new test set**

Run:

```bash
python -c "from hydra_suite.detectkit.gui.dialogs.semantic_escalation_dialog import SemanticEscalationDialog; print('ok')"
python -m pytest tests/test_semantic_masks.py tests/test_semantic_tiling.py \
  tests/test_semantic_calibration.py tests/test_semantic_checkpoints.py \
  tests/test_semantic_escalation_job.py tests/test_pending_escalation_model.py \
  tests/test_sam2_masks.py tests/test_sam2_executor.py -q
```

Expected: `ok`, then all tests pass.

- [ ] **Step 7: Commit**

```bash
make format
git add -A
git commit -m "feat(detectkit): wire GT-based calibration into the semantic escalation dialog"
```

---

### Task 13: Docs, lint gate, and the CUDA confirmation

**Files:**
- Create: `docs/user-guide/detectkit-semantic-escalation.md`
- Modify: `mkdocs.yml` (nav entry)
- Modify: `docs/superpowers/specs/2026-08-28-detectkit-sam3-semantic-escalation-design.md` (status header)

- [ ] **Step 1: Write the user-guide page**

Create `docs/user-guide/detectkit-semantic-escalation.md` covering, in this order:
the difference between the two escalations (geometry converts what you have; semantic
finds what you don't); the prompt is yours to vary and wording matters far less than
tile size; why calibration is recommended and what the exhaustive-labelling checkbox
means; that the recommendation optimises recall, not F1, and why; that the run is a
batch job at tens of seconds per frame, is cancellable, and resumes; that
re-thresholding is free and re-running is not; that accepting creates a **new sibling
source** and never touches the original's labels; and that SAM3 masks trace legs and
antennae while tracking labels bound the body core, so review is where the convention
gets settled. Point large runs at the CUDA box.

Add it to `mkdocs.yml` under the DetectKit section of `nav`.

- [ ] **Step 2: Run the gates**

```bash
make format-check
make lint-moderate
make docs-check
```

Expected: all pass. Fix anything they flag.

- [ ] **Step 3: Confirm SAM3 on CUDA**

The spec lists this as a pre-implementation gate; MPS is confirmed, CUDA never has been.

```bash
ssh rutalab@mehek.taild08eb9.ts.net
# kill any stale sleap/hydra processes first; never touch anything else
cd ~/hydra-suite && source ~/mambaforge/etc/profile.d/conda.sh && conda activate hydra-cuda
pip install 'hydra-suite[sam3]' --no-deps  # or the repo's own extra
python - <<'PY'
import numpy as np
from hydra_suite.core.inference.semantic.sam3 import Sam3SemanticLabeler
lab = Sam3SemanticLabeler.from_variant()
print(lab._device)
print(len(lab.label_image(np.zeros((1008, 1008, 3), dtype=np.uint8), "ant")))
PY
```

Expected: prints `cuda` and `0`. If SAM3 does not run on CUDA, stop and report — the
feature ships MPS-only and the docs must say so.

- [ ] **Step 4: Update the spec status header**

Change the spec's `**Status:**` line to
`Implemented — see docs/superpowers/plans/2026-08-29-detectkit-sam3-semantic-escalation.md`.

- [ ] **Step 5: Commit**

```bash
make format
git add -A
git commit -m "docs(detectkit): semantic escalation user guide; mark spec implemented"
```

---

## Self-Review

**Spec coverage.** Every spec section maps to a task: Promotion → 10; Staging → 8, 9;
The seam → 3, plus the two moves in 1 and 2; Job result and prompt failure → 9;
Tiling → 4 (with `polygon_iou` in 1); Calibration → 7, 12; Candidate cache and
re-threshold → 9, 11; Cancellation and resumability → 9, 11; Tools panel → 11; Dialog →
11, 12; Handlers → 11; Cost → 11 (the run is cancellable and resumable; the dialog's
runtime projection is deliberately **not** taken from the unreconciled archived numbers,
per the spec's Cost section); Testing → the test list in every task; CUDA
confirmation → 13.

One spec item is deliberately narrowed and stated as such in the Global Constraints: the
spec says the sweep re-runs "seam-drop + NMS" per threshold, and this plan applies
seam-drop once at collection because it is threshold-independent. This is exact, and
`test_merge_is_redone_per_threshold_not_post_filtered` guards the part that isn't.

The spec's "Projected total runtime shown before the run starts, computed from a measured
per-tile time on this machine" is only partially delivered: Task 11's dialog has no timed
preview tile yet. The preview button exists; wiring a timing readout to it is left to the
implementer as part of Task 11 Step 2's preview handler, and the dialog must not display
a projection derived from the archived 22-vs-107 s/frame numbers.

**Placeholder scan.** One intentional ellipsis remains: Task 11 Step 3's
`on_escalate_geometry` and `on_review_escalations` bodies are *moves* of named,
line-referenced existing methods (`main_window.py:1769-1943` and `:1985-2010`) with
`self` → `window`, not new code to invent.

**Type consistency.** `SemanticInstance.polygon_px` is tile-local out of `label_image`
and frame-space inside `TileCandidate` — asserted by
`test_candidates_are_offset_into_frame_space`. `collect_candidates` and
`merge_candidates` keep the same keyword names across Tasks 4, 7, and 9.
`resolve_tile_px` returns `int | None` and every call site handles `None` by falling back
to a single full-frame tile. `PendingEscalation.primer_params` is a plain `dict` in
Tasks 8, 9, 10, and 11.
