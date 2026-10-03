# Geometry-Escalation SAHI + Split Calibration Sources — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** SAM2 geometry escalation gains SAHI-style per-box (owner-tile) execution with a tile fraction calibrated against polygon ground truth, and both escalation dialogs split "Calibrate on" (polygon sources) from "Escalate" (all sources).

**Architecture:** Tile sizing/planning is shared with SAM3 (`core/inference/semantic/tiling.py`). SAM2 gets its own Qt-free owner-tile executor (`core/inference/sam2/tiling.py`) and calibration (`core/inference/sam2/calibration.py`). Calibration-frame helpers move to a neutral `detectkit/jobs/calibration_frames.py` with a polygon-only criterion used by both calibrations. One shared Qt widget (`CalibrationSourceSelector`) gives both dialogs the two-list layout.

**Tech Stack:** Python 3, numpy, OpenCV, PySide6, pytest; SAM2 (`sam2` package) on CUDA for the measurement task.

**Spec:** `docs/superpowers/specs/2026-10-03-geometry-escalation-sahi-calibration-design.md`

## Global Constraints

- Uncalibrated SAM2 escalation (`tile_fraction=None`) must be byte-identical to `main`: same staged label bytes, same executor call sequence.
- SAM2 must NOT default to `SEMANTIC_TILE_FRACTION_SEED`; its default is full frame.
- Calibration (both SAM3 and SAM2) uses polygon ground truth only. A frame qualifies only if EVERY label line in it is a polygon (coordinate count even and not 4 or 8 — DetectKit pads 4-point polygons so a polygon line never has 8 coords). Mixed frames are excluded (their box-labelled animals would score as false positives / unscored).
- Direct/YOLO calibration (`jobs/direct_calibration.py`) behaviour is unchanged: `polygon_only` defaults to `False`.
- Core/Runtime/Data/Utils must not import from app layers; new `core/inference/sam2/*` modules are Qt-free and import nothing from `detectkit`.
- Existing importers of moved helpers keep working via re-exports from `jobs/semantic_escalation.py`.
- `IOU_FLOOR` / `FALLBACK_CEIL` start as provisional constants marked `PROVISIONAL`; Task 11 replaces them with measured values. Do not merge while they are provisional.
- Run tests with `conda activate hydra-mps` and `PYTHONPATH=$PWD/src` from the worktree.
- Commits as the configured git user, no Claude co-author trailer.

## Review Focus

1. A box that straddles every tile seam (no tile fully contains it) — expect today's full-frame segmentation for that box, never a dropped instance.
2. A frame where some boxes are owned and some are not — expect exactly one extra full-frame `set_image` for the frame, after the tile passes, and all instances staged in label-file order.
3. Calibration sources at a very different scale from escalation targets — expect a visible warning, never a silent transfer.
4. A project with only box labels opening the SAM3 dialog — expect Calibrate disabled with an explanatory message, not a crash or empty frontier.
5. A dialog reopened on a project saved before this change (legacy `source_names` only) — expect those sources selected in the Escalate list and the Calibrate list defaulting to all polygon sources.

---

### Task 1: Neutral calibration-frame helpers + polygon criterion

**Files:**
- Create: `src/hydra_suite/detectkit/jobs/calibration_frames.py`
- Modify: `src/hydra_suite/detectkit/jobs/semantic_escalation.py` (delete moved bodies at ~1049-1200; add re-exports)
- Test: `tests/test_calibration_frames.py`

**Interfaces:**
- Produces:
  - `is_polygon_line(n_coords: int) -> bool`
  - `has_polygon_frames(source) -> bool` (label-file scan, no decode)
  - `labelled_frames_for(source, *, limit: int = 0, polygon_only: bool = False) -> list[tuple[Path, list[LabelRecord]]]`
  - `stratified_calibration_frames(sources, *, budget: int = CALIBRATION_SAMPLE_FRAMES, polygon_only: bool = False)`
  - `measure_median_body_px(sources, *, sample_frames=MEDIAN_BODY_SAMPLE_FRAMES, max_total_frames=MEDIAN_BODY_TOTAL_FRAMES) -> tuple[float, int, bool]`
  - `median_body_px_for`, `has_labelled_frames`, `_label_path_for`, constants `MEDIAN_BODY_SAMPLE_FRAMES`, `MEDIAN_BODY_TOTAL_FRAMES`, `CALIBRATION_SAMPLE_FRAMES`
  - `SCALE_MISMATCH_RATIO = 1.5`; `scale_mismatch(a_px: float, b_px: float, ratio: float = SCALE_MISMATCH_RATIO) -> bool` (False if either ≤ 0)

- [ ] **Step 1: Write failing tests** (`tests/test_calibration_frames.py`)

```python
from pathlib import Path

import cv2
import numpy as np

from hydra_suite.detectkit.gui.models import OBBSource
from hydra_suite.detectkit.jobs import calibration_frames as cf

POLY = "0 0.1 0.1 0.3 0.1 0.35 0.2 0.3 0.3 0.1 0.3\n"  # 5 points
OBB = "0 0.5 0.5 0.7 0.5 0.7 0.7 0.5 0.7\n"
AABB = "0 0.5 0.5 0.1 0.1\n"


def _src(tmp_path, name, frames):
    root = tmp_path / name
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir(parents=True)
    for stem, text in frames.items():
        cv2.imwrite(str(root / "images" / f"{stem}.jpg"), np.zeros((100, 100, 3), np.uint8))
        (root / "labels" / f"{stem}.txt").write_text(text)
    return OBBSource(path=str(root), name=name, level="polygon")


def test_is_polygon_line():
    assert cf.is_polygon_line(6) and cf.is_polygon_line(10)
    assert not cf.is_polygon_line(4) and not cf.is_polygon_line(8)
    assert not cf.is_polygon_line(7)


def test_has_polygon_frames_needs_an_all_polygon_frame(tmp_path):
    assert cf.has_polygon_frames(_src(tmp_path, "p", {"a": POLY}))
    assert not cf.has_polygon_frames(_src(tmp_path, "b", {"a": OBB + AABB}))
    assert not cf.has_polygon_frames(_src(tmp_path, "m", {"a": POLY + OBB}))


def test_has_polygon_frames_never_decodes(tmp_path, monkeypatch):
    src = _src(tmp_path, "p", {"a": POLY})
    monkeypatch.setattr(cf.cv2, "imread", lambda *_a, **_k: (_ for _ in ()).throw(AssertionError))
    assert cf.has_polygon_frames(src)


def test_polygon_only_skips_box_and_mixed_frames(tmp_path):
    src = _src(tmp_path, "s", {"a": POLY, "b": OBB, "c": POLY + OBB})
    frames = cf.labelled_frames_for(src, polygon_only=True)
    assert [p.stem for p, _ in frames] == ["a"]
    assert len(frames[0][1]) == 1 and frames[0][1][0].points.shape == (5, 2)
    assert len(cf.labelled_frames_for(src)) == 3  # default unchanged


def test_stratified_polygon_only(tmp_path):
    a = _src(tmp_path, "a", {"x": POLY, "y": OBB})
    b = _src(tmp_path, "b", {"z": OBB})
    out = cf.stratified_calibration_frames([a, b], budget=4, polygon_only=True)
    assert [p.stem for p, _ in out] == ["x"]


def test_scale_mismatch():
    assert cf.scale_mismatch(50.0, 100.0)
    assert not cf.scale_mismatch(50.0, 70.0)
    assert not cf.scale_mismatch(0.0, 70.0)


def test_semantic_module_reexports_same_objects():
    from hydra_suite.detectkit.jobs import semantic_escalation as se

    for name in (
        "labelled_frames_for",
        "stratified_calibration_frames",
        "measure_median_body_px",
        "median_body_px_for",
        "has_labelled_frames",
        "_label_path_for",
        "CALIBRATION_SAMPLE_FRAMES",
    ):
        assert getattr(se, name) is getattr(cf, name)
```

- [ ] **Step 2: Run, verify FAIL** — `python -m pytest tests/test_calibration_frames.py -q` → ImportError (module missing).

- [ ] **Step 3: Implement.** Create `calibration_frames.py` by MOVING (cut, not copy) `_label_path_for`, `has_labelled_frames`, `MEDIAN_BODY_SAMPLE_FRAMES`, `MEDIAN_BODY_TOTAL_FRAMES`, `CALIBRATION_SAMPLE_FRAMES`, `measure_median_body_px`, `median_body_px_for`, `labelled_frames_for`, `stratified_calibration_frames` verbatim (keep their docstrings/comments) from `semantic_escalation.py`. Module imports: `cv2`, `numpy as np`, `Path`, `LabelRecord` from `hydra_suite.data.al.escalation`, `IMG_EXTS`, `OBBSource`, `GeometryLevel`. Then add:

```python
SCALE_MISMATCH_RATIO = 1.5


def is_polygon_line(n_coords: int) -> bool:
    """A label line's coordinate count denotes a polygon.

    Even, >= 6, and not 8: AABB lines carry 4 values and OBB/quad lines 8.
    The AL label writer pads 4-point polygons (data/al/labels.py) precisely
    so a polygon line never carries 8, which is what makes this decidable.
    """
    return n_coords >= 6 and n_coords % 2 == 0 and n_coords != 8


def _is_polygon_label_text(text: str) -> bool:
    """True when every non-empty line is a polygon (and there is one)."""
    counts = [len(line.split()) - 1 for line in text.splitlines() if line.strip()]
    return bool(counts) and all(is_polygon_line(n) for n in counts)


def has_polygon_frames(source: OBBSource) -> bool:
    """True if *source* has a frame whose labels are ALL polygons.

    Calibration needs ground truth for masks. A box is not one, and a frame
    mixing boxes and polygons cannot be scored either: its box-labelled
    animals would be unmatched. Label-FILE scan only, like
    ``has_labelled_frames``.
    """
    root = Path(source.path)
    images_dir, labels_dir = root / "images", root / "labels"
    if not images_dir.is_dir() or not labels_dir.is_dir():
        return False
    for img_path in images_dir.rglob("*"):
        if img_path.suffix.lower() not in IMG_EXTS:
            continue
        label_path = _label_path_for(images_dir, labels_dir, img_path)
        try:
            if label_path.exists() and _is_polygon_label_text(label_path.read_text()):
                return True
        except OSError:  # pragma: no cover - unreadable label file
            continue
    return False


def scale_mismatch(a_px: float, b_px: float, ratio: float = SCALE_MISMATCH_RATIO) -> bool:
    """Median body sizes differ by more than *ratio* (unknown sizes never warn)."""
    if a_px <= 0 or b_px <= 0:
        return False
    return max(a_px, b_px) / min(a_px, b_px) > ratio
```

Change `labelled_frames_for` signature to `(source, *, limit: int = 0, polygon_only: bool = False)` and, right after the `label_path.exists()/strip()` check, add:

```python
        text = label_path.read_text()
        if polygon_only and not _is_polygon_label_text(text):
            continue
```

(reuse `text` for the emptiness check). Change `stratified_calibration_frames(sources, *, budget=CALIBRATION_SAMPLE_FRAMES, polygon_only=False)` to pass `polygon_only` to `labelled_frames_for`. In `semantic_escalation.py` replace the moved code with:

```python
from .calibration_frames import (  # noqa: F401  (re-exported for importers)
    CALIBRATION_SAMPLE_FRAMES,
    MEDIAN_BODY_SAMPLE_FRAMES,
    MEDIAN_BODY_TOTAL_FRAMES,
    _label_path_for,
    has_labelled_frames,
    labelled_frames_for,
    measure_median_body_px,
    median_body_px_for,
    stratified_calibration_frames,
)
```

Keep any other in-module use of `_label_path_for` working (it is used by `preview_random_frame` and others).

- [ ] **Step 4: Run** `python -m pytest tests/test_calibration_frames.py tests/test_semantic_escalation_job.py tests/test_detectkit_direct_calibration_job.py tests/test_detectkit_sam2_escalation_wiring.py -q` → all pass. If a source-text test (e.g. one asserting `"labelled_frames_for" not in source`) inspects a file you changed, read the test and keep its intent.

- [ ] **Step 5: Commit** `git commit -m "Move calibration-frame helpers to calibration_frames and add polygon-only criterion"`

---

### Task 2: Semantic calibration is polygon-only

**Files:**
- Modify: `src/hydra_suite/detectkit/sidecars/operations.py:341-352`
- Test: `tests/test_calibration_frames.py` (append)

**Interfaces:** Consumes `stratified_calibration_frames(..., polygon_only=True)`.

- [ ] **Step 1: Failing test** (append)

```python
def test_semantic_calibration_sidecar_samples_polygon_frames_only(tmp_path, monkeypatch):
    from hydra_suite.detectkit.sidecars import operations as ops

    seen = {}

    def fake_stratified(sources, *, budget, polygon_only=False):
        seen["polygon_only"] = polygon_only
        return []

    monkeypatch.setattr(
        "hydra_suite.detectkit.jobs.semantic_escalation.stratified_calibration_frames",
        fake_stratified,
    )
    monkeypatch.setattr(ops, "_semantic_sources", lambda payload: [])
    out = ops.run_semantic_calibration_sidecar({"sample_budget": 4}, lambda *_: None)
    assert seen["polygon_only"] is True and out["points"] == []
```

(Check the real function name around operations.py:324 — it is the `def` immediately above line 325; use that name.)

- [ ] **Step 2: Run, verify FAIL** (`polygon_only` False).
- [ ] **Step 3: Implement** — change the call to `stratified_calibration_frames(sources, budget=budget, polygon_only=True)` with a one-line comment: `# Calibration needs real ground truth: polygon frames only (spec 2026-10-03).`
- [ ] **Step 4: Run** the test + `tests/test_semantic_calibration*.py` → pass.
- [ ] **Step 5: Commit** `"Restrict SAM3 calibration to polygon ground-truth frames"`

---

### Task 3: Shared `TilingSettings` + SAM2 request fields

**Files:**
- Modify: `src/hydra_suite/core/inference/semantic/tiling.py` (add dataclass)
- Modify: `src/hydra_suite/detectkit/jobs/sam2_escalation.py` (`EscalationRequest`, `EscalationResult`)
- Modify: `src/hydra_suite/detectkit/jobs/semantic_escalation.py` (`SemanticEscalationRequest.tiling` property)
- Test: `tests/test_sam2_tiling.py` (new)

**Interfaces:**
- Produces:
  - `TilingSettings(reference_body_px: float = 0.0, tile_fraction: float | None = None, tile_px: int | None = None, overlap: float = DEFAULT_OVERLAP)` frozen; method `resolved_tile_px() -> int | None` = `self.tile_px or resolve_tile_px(self.reference_body_px, self.tile_fraction)`; method `plan_for(frame_hw) -> SlicePlan` = `plan_for_frame(hw, px, overlap)` if px and px < min(h, w) else `full_frame_plan(hw)`.
  - `EscalationRequest` new trailing fields: `reference_body_px: float = 0.0`, `tile_fraction: float | None = None`, `tile_px: int | None = None`, `overlap: float = DEFAULT_OVERLAP`; property `tiling -> TilingSettings`.
  - `EscalationResult.seam_fallbacks: int = 0`, `EscalationResult.tile_px: int | None = None`.
  - `SemanticEscalationRequest.tiling` property (no field changes).

- [ ] **Step 1: Failing test** (`tests/test_sam2_tiling.py`)

```python
from hydra_suite.core.inference.semantic.tiling import TilingSettings


def test_tiling_settings_default_is_full_frame():
    t = TilingSettings()
    assert t.resolved_tile_px() is None
    plan = t.plan_for((400, 600))
    assert plan.tiles == [(0, 0, 600, 400)]


def test_tiling_settings_resolves_and_plans():
    t = TilingSettings(reference_body_px=50.0, tile_fraction=0.25)
    assert t.resolved_tile_px() == 200
    assert len(t.plan_for((400, 600)).tiles) > 1


def test_tile_larger_than_frame_falls_back_to_full_frame():
    t = TilingSettings(tile_px=1000)
    assert t.plan_for((400, 600)).tiles == [(0, 0, 600, 400)]


def test_escalation_request_tiling_defaults_preserve_positional_args():
    import types

    from hydra_suite.detectkit.jobs.sam2_escalation import EscalationRequest

    req = EscalationRequest(types.SimpleNamespace(), ["a"], "v", True)
    assert req.overwrite is True and req.tiling == TilingSettings()
```

Verify the expected `resolved_tile_px()` value by calling `resolve_tile_px(50.0, 0.25)` once in a REPL first and use what it returns (it rounds via `tile_size_for_mode`).

- [ ] **Step 2: Run, verify FAIL.**
- [ ] **Step 3: Implement** in `tiling.py` (after `full_frame_plan`):

```python
@dataclass(frozen=True)
class TilingSettings:
    """Tile sizing shared by SAM3 (semantic) and SAM2 (geometry) escalation.

    Only the PLAN is shared: what runs per tile differs (SAM3 detects and
    merges, SAM2 segments known boxes). ``tile_fraction=None`` = full frame.
    """

    reference_body_px: float = 0.0
    tile_fraction: float | None = None
    tile_px: int | None = None  # explicit override; wins over tile_fraction
    overlap: float = DEFAULT_OVERLAP

    def resolved_tile_px(self) -> int | None:
        return self.tile_px or resolve_tile_px(self.reference_body_px, self.tile_fraction)

    def plan_for(self, frame_hw) -> SlicePlan:
        h, w = int(frame_hw[0]), int(frame_hw[1])
        px = self.resolved_tile_px()
        if px is None or px >= min(h, w):
            return full_frame_plan((h, w))
        return plan_for_frame((h, w), px, self.overlap)
```

(`dataclass` import may already exist; `SlicePlan` is already imported for annotations — check.) In `sam2_escalation.py` add the four trailing fields to `EscalationRequest` (import `DEFAULT_OVERLAP, TilingSettings` from `core.inference.semantic.tiling`) and:

```python
    @property
    def tiling(self) -> TilingSettings:
        return TilingSettings(
            reference_body_px=float(self.reference_body_px or 0.0),
            tile_fraction=self.tile_fraction,
            tile_px=self.tile_px,
            overlap=float(self.overlap),
        )
```

Add `seam_fallbacks: int = 0` and `tile_px: int | None = None` to `EscalationResult`. Add the same `tiling` property to `SemanticEscalationRequest` (built from its existing fields).

- [ ] **Step 4: Run** `tests/test_sam2_tiling.py tests/test_sam2_escalation.py tests/test_semantic_tiling.py` → pass.
- [ ] **Step 5: Commit** `"Add shared TilingSettings and SAM2 escalation tiling fields"`

---

### Task 4: Characterize current SAM2 escalation (byte-identity golden)

**Files:**
- Test: `tests/test_sam2_escalation_byte_identity.py` (new)

This task runs against the UNMODIFIED `run_escalation` and records its behaviour; Task 6 must keep it green.

- [ ] **Step 1: Write the characterization test**

```python
"""run_escalation with tiling off is byte-identical to pre-SAHI main.

The expected call log and label bytes were recorded against the unmodified
run_escalation (commit before Task 6) and are asserted verbatim.
"""
import types
from pathlib import Path

import cv2
import numpy as np

from hydra_suite.detectkit.gui.models import OBBSource
from hydra_suite.detectkit.jobs.sam2_escalation import EscalationRequest, run_escalation


class RecordingExec:
    def __init__(self):
        self.log = []
        self._shape = None

    def set_image(self, img):
        self._shape = img.shape[:2]
        self.log.append(("set_image", img.shape, int(img.sum()) % 1000003))

    def segment(self, box, pos, neg):
        self.log.append(
            ("segment", tuple(round(float(v), 3) for v in box),
             tuple(tuple(round(float(c), 3) for c in p) for p in pos),
             tuple(tuple(round(float(c), 3) for c in p) for p in neg))
        )
        h, w = self._shape
        m = np.zeros((h, w), bool)
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        m[y1 + 2 : y2 - 2, x1 + 2 : x2 - 2] = True  # inset box mask
        return m, 0.8


def _source(tmp_path):
    root = tmp_path / "src"
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir(parents=True)
    rng = np.random.default_rng(0)
    cv2.imwrite(str(root / "images" / "f1.png"), rng.integers(0, 255, (300, 400, 3), np.uint8))
    (root / "labels" / "f1.txt").write_text(
        "0 0.10 0.10 0.20 0.10 0.20 0.25 0.10 0.25\n"
        "1 0.15 0.20 0.30 0.20 0.30 0.35 0.15 0.35\n"
        "0 0.5 0.5 0.2 0.2\n"
    )
    (root / "classes.txt").write_text("ant\nqueen\n")
    return OBBSource(path=str(root), name="src", level="obb")


def test_untiled_escalation_matches_recorded_main(tmp_path):
    src = _source(tmp_path)
    project = types.SimpleNamespace(project_dir=str(tmp_path), sources=[src])
    ex = RecordingExec()
    run_escalation(EscalationRequest(project, ["src"], "v"), ex)
    staged = (Path(src.staged_review.staged_path) / "labels" / "f1.txt").read_text()
    assert ex.log == EXPECTED_LOG
    assert staged == EXPECTED_LABEL


EXPECTED_LOG = None  # filled in Step 2
EXPECTED_LABEL = None  # filled in Step 2
```

- [ ] **Step 2: Record the golden.** Run once with a temporary `print(repr(ex.log)); print(repr(staged))`, paste the printed values as literals into `EXPECTED_LOG`/`EXPECTED_LABEL`, remove the prints.
- [ ] **Step 3: Run** → PASS against unmodified code.
- [ ] **Step 4: Commit** `"Characterize untiled SAM2 escalation call sequence and staged bytes"`

---

### Task 5: SAM2 owner-tile executor (`core/inference/sam2/tiling.py`)

**Files:**
- Create: `src/hydra_suite/core/inference/sam2/tiling.py`
- Test: `tests/test_sam2_tiling.py` (append)

**Interfaces:**
- Consumes: `SlicePlan` (`tiles: list[(x0, y0, x1, y1)]`), `Prompt` shape (`box_xyxy`, `positive_points`, `negative_points`) — passed in as plain tuples/lists, NOT imported from detectkit.
- Produces:
  - `assign_owner_tiles(boxes_xyxy: Sequence[tuple[float,float,float,float]], tiles: Sequence[tuple[int,int,int,int]]) -> tuple[dict[int, list[int]], list[int]]`
  - `@dataclass SegmentOutcome: mask: np.ndarray | None (frame-sized bool), iou: float, owner_tile: int | None` (None = full-frame fallback)
  - `segment_boxes(executor, image, prompts, tiles) -> list[SegmentOutcome]` where `prompts` is a sequence of objects with `box_xyxy`, `positive_points`, `negative_points`; output in prompt order.

Semantics: if `tiles` has exactly one tile covering the whole frame, behave exactly as the legacy loop: one `set_image(image)` then `segment(...)` per prompt in order with UNMODIFIED coordinates, returning the executor's mask as-is (no copy into a new array). Otherwise: owned tiles processed in ascending tile index; for each, `set_image(image[y0:y1, x0:x1])`, then for each owned box in ascending box index call `segment` with box/points shifted by `(-x0, -y0)` and negative points outside the tile dropped; paste mask into `np.zeros((H, W), bool)` at `[y0:y1, x0:x1]`. Then, if any unowned boxes, ONE `set_image(image)` and segment them in ascending index with frame coords.

- [ ] **Step 1: Failing tests** (append to `tests/test_sam2_tiling.py`)

```python
import numpy as np
from types import SimpleNamespace as NS

from hydra_suite.core.inference.sam2.tiling import assign_owner_tiles, segment_boxes


def test_owner_is_tile_with_largest_margin():
    tiles = [(0, 0, 100, 100), (50, 0, 150, 100)]
    owned, unowned = assign_owner_tiles([(60, 40, 80, 60)], tiles)
    # margins: tile0 -> min(60,40,20,40)=20 ; tile1 -> min(10,40,70,40)=10
    assert owned == {0: [0]} and unowned == []


def test_tie_breaks_to_lowest_tile_index():
    tiles = [(0, 0, 100, 100), (0, 0, 100, 100)]
    owned, _ = assign_owner_tiles([(40, 40, 60, 60)], tiles)
    assert owned == {0: [0]}


def test_box_not_fully_inside_any_tile_is_unowned():
    tiles = [(0, 0, 100, 100), (100, 0, 200, 100)]
    owned, unowned = assign_owner_tiles([(90, 10, 110, 20)], tiles)
    assert owned == {} and unowned == [0]


class Exec:
    def __init__(self):
        self.log = []
        self.shape = None

    def set_image(self, img):
        self.shape = img.shape[:2]
        self.log.append(("set", img.shape[:2]))

    def segment(self, box, pos, neg):
        self.log.append(("seg", tuple(box), tuple(map(tuple, pos)), tuple(map(tuple, neg))))
        m = np.zeros(self.shape, bool)
        x1, y1, x2, y2 = (int(v) for v in box)
        m[y1:y2, x1:x2] = True
        return m, 0.5


def _p(box, neg=()):
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return NS(box_xyxy=box, positive_points=[(cx, cy)], negative_points=list(neg))


def test_tiled_segment_round_trips_coordinates_and_skips_empty_tiles():
    img = np.zeros((100, 300, 3), np.uint8)
    tiles = [(0, 0, 100, 100), (100, 0, 200, 100), (200, 0, 300, 100)]
    ex = Exec()
    out = segment_boxes(ex, img, [_p((210, 10, 230, 30))], tiles)
    assert ex.log[0] == ("set", (100, 100))
    assert ex.log[1][1] == (10, 10, 30, 30)  # shifted into tile 2
    assert len([e for e in ex.log if e[0] == "set"]) == 1  # empty tiles skipped
    m = out[0].mask
    assert m.shape == (100, 300) and m[10:30, 210:230].all() and m.sum() == 400
    assert out[0].owner_tile == 2


def test_negative_points_outside_owner_tile_are_dropped():
    img = np.zeros((100, 200, 3), np.uint8)
    tiles = [(0, 0, 100, 100), (100, 0, 200, 100)]
    ex = Exec()
    segment_boxes(ex, img, [_p((10, 10, 30, 30), neg=[(20, 20), (150, 50)])], tiles)
    assert ex.log[1][3] == ((20, 20),)


def test_unowned_boxes_get_one_full_frame_pass_after_tiles():
    img = np.zeros((100, 200, 3), np.uint8)
    tiles = [(0, 0, 100, 100), (100, 0, 200, 100)]
    prompts = [_p((90, 10, 110, 30)), _p((10, 10, 30, 30)), _p((95, 50, 105, 60))]
    ex = Exec()
    out = segment_boxes(ex, img, prompts, tiles)
    sets = [e for e in ex.log if e[0] == "set"]
    assert sets == [("set", (100, 100)), ("set", (100, 200))]
    assert [o.owner_tile for o in out] == [None, 0, None]
    assert out[0].mask[10:30, 90:110].all()


def test_single_full_frame_tile_is_the_legacy_call_sequence():
    img = np.zeros((100, 200, 3), np.uint8)
    ex = Exec()
    prompts = [_p((10, 10, 30, 30)), _p((50, 50, 70, 70))]
    segment_boxes(ex, img, prompts, [(0, 0, 200, 100)])
    assert ex.log == [
        ("set", (100, 200)),
        ("seg", (10, 10, 30, 30), ((20.0, 20.0),), ()),
        ("seg", (50, 50, 70, 70), ((60.0, 60.0),), ()),
    ]
```

- [ ] **Step 2: Run, verify FAIL** (module missing).
- [ ] **Step 3: Implement** `core/inference/sam2/tiling.py`:

```python
"""Owner-tile SAHI for SAM2 box-prompted segmentation (Qt-free, torch-free).

SAM2 resizes whatever it is given to 1024 px, so on a large frame a small
animal's box prompt collapses to a few pixels. Segmenting inside a tile keeps
the animal at a usable scale. Unlike SAM3 there is nothing to detect or
merge: each known box is segmented ONCE, inside the tile that contains it
with the most margin, and empty tiles are never encoded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class SegmentOutcome:
    mask: np.ndarray | None  # frame-sized bool mask
    iou: float
    owner_tile: int | None  # None = segmented on the full frame


def _margin(box, tile) -> float:
    x1, y1, x2, y2 = box
    tx0, ty0, tx1, ty1 = tile
    return min(x1 - tx0, y1 - ty0, tx1 - x2, ty1 - y2)


def assign_owner_tiles(boxes_xyxy, tiles):
    """(tile index -> box indices, unowned box indices); deterministic."""
    owned: dict[int, list[int]] = {}
    unowned: list[int] = []
    for bi, box in enumerate(boxes_xyxy):
        best, best_margin = None, -1.0
        for ti, tile in enumerate(tiles):
            m = _margin(box, tile)
            if m >= 0 and m > best_margin:
                best, best_margin = ti, m
        if best is None:
            unowned.append(bi)
        else:
            owned.setdefault(best, []).append(bi)
    return owned, unowned


def _is_full_frame(tiles, frame_hw) -> bool:
    h, w = frame_hw
    return len(tiles) == 1 and tuple(tiles[0]) == (0, 0, w, h)


def _shift(points, dx, dy):
    return [(float(x) - dx, float(y) - dy) for x, y in points]


def segment_boxes(executor, image, prompts, tiles) -> list[SegmentOutcome]:
    h, w = image.shape[:2]
    out: list[SegmentOutcome | None] = [None] * len(prompts)
    if _is_full_frame(tiles, (h, w)):
        # The legacy path, call for call: tiling off must stay byte-identical.
        executor.set_image(image)
        for i, p in enumerate(prompts):
            mask, iou = executor.segment(p.box_xyxy, p.positive_points, p.negative_points)
            out[i] = SegmentOutcome(mask, float(iou), 0)
        return out  # type: ignore[return-value]

    owned, unowned = assign_owner_tiles([p.box_xyxy for p in prompts], tiles)
    for ti in sorted(owned):
        x0, y0, x1, y1 = tiles[ti]
        executor.set_image(image[y0:y1, x0:x1])
        for bi in owned[ti]:
            p = prompts[bi]
            bx1, by1, bx2, by2 = p.box_xyxy
            negatives = [
                (x, y) for x, y in p.negative_points if x0 <= x < x1 and y0 <= y < y1
            ]
            mask, iou = executor.segment(
                (bx1 - x0, by1 - y0, bx2 - x0, by2 - y0),
                _shift(p.positive_points, x0, y0),
                _shift(negatives, x0, y0),
            )
            full = np.zeros((h, w), dtype=bool)
            if mask is not None:
                full[y0:y1, x0:x1] = np.asarray(mask, dtype=bool)
            out[bi] = SegmentOutcome(full, float(iou), ti)
    if unowned:
        # A box straddling every seam: one full-frame pass, today's behaviour.
        executor.set_image(image)
        for bi in unowned:
            p = prompts[bi]
            mask, iou = executor.segment(p.box_xyxy, p.positive_points, p.negative_points)
            out[bi] = SegmentOutcome(mask, float(iou), None)
    return out  # type: ignore[return-value]
```

Note the legacy test expects positive points as given (floats from `_p`) — the full-frame branch passes them through untouched.

- [ ] **Step 4: Run** `python -m pytest tests/test_sam2_tiling.py -q` → pass.
- [ ] **Step 5: Commit** `"Add SAM2 owner-tile segmentation"`

---

### Task 6: `run_escalation` uses owner tiles

**Files:**
- Modify: `src/hydra_suite/detectkit/jobs/sam2_escalation.py` (inner per-image loop, ~lines 276-310)
- Test: `tests/test_sam2_escalation.py` (append)

**Interfaces:** Consumes `segment_boxes`, `TilingSettings.plan_for`, `EscalationRequest.tiling`.

- [ ] **Step 1: Failing test** (append)

```python
def test_tiled_escalation_encodes_owner_tiles_and_counts_seam_fallbacks(tmp_path):
    root = tmp_path / "sources" / "big"
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir(parents=True)
    cv2.imwrite(str(root / "images" / "a.png"), np.zeros((400, 400, 3), np.uint8))
    # one box well inside the top-left quadrant, one across the centre seam
    (root / "labels" / "a.txt").write_text(
        "0 0.05 0.05 0.15 0.05 0.15 0.15 0.05 0.15\n"
        "0 0.45 0.45 0.55 0.45 0.55 0.55 0.45 0.55\n"
    )
    src = OBBSource(path=str(root), name="big", level="obb")
    project = types.SimpleNamespace(project_dir=str(tmp_path), sources=[src])

    class Ex:
        def __init__(self):
            self.shapes = []

        def set_image(self, img):
            self.shapes.append(img.shape[:2])
            self.shape = img.shape[:2]

        def segment(self, box, pos, neg):
            m = np.zeros(self.shape, bool)
            x1, y1, x2, y2 = (int(v) for v in box)
            m[y1:y2, x1:x2] = True
            return m, 0.9

    ex = Ex()
    req = EscalationRequest(project, ["big"], "v", tile_px=200, overlap=0.0)
    result = run_escalation(req, ex)
    assert ex.shapes == [(200, 200), (400, 400)]
    assert result.seam_fallbacks == 1 and result.primed == 2 and result.tile_px == 200
```

- [ ] **Step 2: Run, verify FAIL.**
- [ ] **Step 3: Implement.** In `run_escalation`, before the source loop: `tiling = req.tiling; result.tile_px = tiling.resolved_tile_px()`. Replace the `executor.set_image(img)` + per-box `segment` loop with:

```python
            if boxes:
                prompts = build_prompts(boxes)
                plan = tiling.plan_for((h, w))
                outcomes = segment_boxes(executor, img, prompts, plan.tiles)
                for box, outcome in zip(boxes, outcomes):
                    if outcome.owner_tile is None and len(plan.tiles) > 1:
                        result.seam_fallbacks += 1
                    mask = clip_mask_to_polygon(outcome.mask, box.polygon_px)
                    contour = mask_to_contour(mask)
                    ...  # unchanged primed/fallback + LabelRecord append
```

Keep every existing comment that still applies (SAM2 box is soft guidance; class-id preservation). Import `segment_boxes` from `hydra_suite.core.inference.sam2.tiling`.

- [ ] **Step 4: Run** `tests/test_sam2_escalation.py tests/test_sam2_escalation_byte_identity.py tests/test_sam2_tiling.py` → all pass (byte-identity unchanged).
- [ ] **Step 5: Commit** `"Run SAM2 escalation through owner-tile segmentation"`

---

### Task 7: SAM2 geometry calibration core (`core/inference/sam2/calibration.py`)

**Files:**
- Create: `src/hydra_suite/core/inference/sam2/calibration.py`
- Test: `tests/test_sam2_calibration.py`

**Interfaces:**
- Consumes: `candidate_tile_plans`, `TILE_FRACTION_GRID`, `DEFAULT_OVERLAP` (semantic tiling), `segment_boxes`, `clip_mask_to_polygon`, `mask_to_contour`, `polygon_iou` (`core/inference/masks`), `derive_down` + `GeometryLevel` (data layer — allowed: Core→Data under TYPE_CHECKING only? NO: derive_down is needed at runtime. Avoid the data import by computing the OBB with `cv2.minAreaRect` + `cv2.boxPoints` locally, mirroring `data/al/escalation._to_obb`; read `_to_obb` and replicate exactly).
- Produces:
  - `IOU_FLOOR = 0.5  # PROVISIONAL`, `FALLBACK_CEIL = 0.10  # PROVISIONAL`, `MIN_INSTANCES = 20`
  - frozen `GeometryCalibrationPoint(tile_fraction, tile_px, owned_tiles_per_frame: float, seconds_per_frame: float, median_iou: float, p10_iou: float, fallback_rate: float, seam_fallback_rate: float, n_instances: int)`
  - `calibrate_geometry(executor, frames: Sequence[tuple[Path, Sequence[np.ndarray]]], *, reference_body_px: float, tile_fractions=TILE_FRACTION_GRID, overlap=DEFAULT_OVERLAP, progress=None, should_stop=None) -> list[GeometryCalibrationPoint]` — frames are (image path, list of GT polygons in px). Only fractions that resolved on EVERY completed frame are reported (same rule as semantic `calibrate`).
  - `recommend_geometry(points, *, min_instances=MIN_INSTANCES, iou_floor=IOU_FLOOR, fallback_ceil=FALLBACK_CEIL) -> tuple[GeometryCalibrationPoint | None, str]`

Per frame and fraction: prompt per GT polygon = AABB of its OBB as `box_xyxy`, positive = polygon vertex mean, negatives = centres of other prompts whose AABBs overlap (same rule as `sam2_prompts.build_prompts`, reimplemented locally since core cannot import detectkit). Run `segment_boxes`; for each outcome: `contour = mask_to_contour(clip_mask_to_polygon(mask, obb))`; contour None → fallback (IoU 0 counted in the IoU distribution); else IoU = `polygon_iou(contour, gt)`. `owned_tiles_per_frame` = number of distinct owner tiles + (1 if any unowned and tiled). `seconds_per_frame` measured with `time.perf_counter` around `segment_boxes`. Cancellation: `should_stop()` checked between (frame, fraction) passes; a cancelled pass is not counted.

- [ ] **Step 1: Failing tests** (`tests/test_sam2_calibration.py`)

```python
from pathlib import Path

import cv2
import numpy as np

from hydra_suite.core.inference.sam2 import calibration as gc


class PerfectExec:
    """Returns the GT disc for whatever box it is given (tile coords)."""

    def set_image(self, img):
        self.shape = img.shape[:2]

    def segment(self, box, pos, neg):
        m = np.zeros(self.shape, np.uint8)
        cx, cy = pos[0]
        cv2.circle(m, (int(round(cx)), int(round(cy))), 10, 1, -1)
        return m.astype(bool), 0.9


def _frame(tmp_path, centers, size=(200, 200)):
    p = tmp_path / "f.png"
    cv2.imwrite(str(p), np.zeros((*size, 3), np.uint8))
    polys = []
    for cx, cy in centers:
        ang = np.linspace(0, 2 * np.pi, 12, endpoint=False)
        polys.append(np.stack([cx + 10 * np.cos(ang), cy + 10 * np.sin(ang)], 1).astype(np.float32))
    return p, polys


def test_perfect_masks_score_high_iou_for_every_fraction(tmp_path):
    frame = _frame(tmp_path, [(50, 50), (150, 150)])
    pts = gc.calibrate_geometry(
        PerfectExec(), [frame], reference_body_px=20.0, tile_fractions=(0.2, None)
    )
    assert {p.tile_fraction for p in pts} == {0.2, None}
    for p in pts:
        assert p.median_iou > 0.85 and p.fallback_rate == 0.0 and p.n_instances == 2
    full = next(p for p in pts if p.tile_fraction is None)
    assert full.owned_tiles_per_frame == 1


def test_recommend_prefers_fewest_tiles_clearing_floor():
    mk = lambda frac, tiles, iou, fb=0.0, n=50: gc.GeometryCalibrationPoint(
        frac, None, tiles, 1.0, iou, iou, fb, 0.0, n
    )
    best, why = gc.recommend_geometry([mk(None, 1, 0.3), mk(0.1, 4, 0.8), mk(0.05, 9, 0.85)])
    assert best.tile_fraction == 0.1 and why == ""


def test_recommend_refuses_with_reason():
    mk = lambda iou, fb, n: gc.GeometryCalibrationPoint(None, None, 1, 1.0, iou, iou, fb, 0.0, n)
    assert gc.recommend_geometry([])[0] is None
    assert "IoU" in gc.recommend_geometry([mk(0.1, 0.0, 50)])[1]
    assert "fell back" in gc.recommend_geometry([mk(0.9, 0.9, 50)])[1]
    assert "instances" in gc.recommend_geometry([mk(0.9, 0.0, 3)])[1]


def test_cancel_returns_no_partial_points(tmp_path):
    frame = _frame(tmp_path, [(50, 50)])
    assert gc.calibrate_geometry(
        PerfectExec(), [frame], reference_body_px=20.0, tile_fractions=(None,),
        should_stop=lambda: True,
    ) == []
```

- [ ] **Step 2: Run, verify FAIL.**
- [ ] **Step 3: Implement** `calibration.py` following the Interfaces block. Module docstring: state that it scores SAM2 masks against user-reviewed polygon labels, that pairing is one-to-one by construction, that the sweep is one-dimensional (tile fraction), and that `IOU_FLOOR`/`FALLBACK_CEIL` are PROVISIONAL until Task 11. `recommend_geometry` order: empty → "No calibration points"; filter `median_iou >= iou_floor` else reason containing "IoU"; filter `fallback_rate <= fallback_ceil` else reason containing "fell back"; filter `n_instances >= min_instances` else reason containing "instances"; winner `min(key=(owned_tiles_per_frame, -median_iou))`.
- [ ] **Step 4: Run** → pass. Also `python -c "import hydra_suite.core.inference.sam2.calibration"` must not import PySide6 (`python -X importtime ... 2>&1 | grep -c PySide6` → 0).
- [ ] **Step 5: Commit** `"Add SAM2 geometry calibration against polygon ground truth"`

---

### Task 8: Calibration worker + project persistence

**Files:**
- Modify: `src/hydra_suite/detectkit/jobs/sam2_escalation.py` (add `Sam2CalibrationWorker`)
- Modify: `src/hydra_suite/detectkit/gui/models.py` (`DetectKitProject.geometry_calibration`, `geometry_escalation_settings`; round trip at ~476)
- Test: `tests/test_sam2_calibration.py` (append), `tests/test_sam2_escalation.py` (append)

**Interfaces:**
- Produces:
  - `Sam2CalibrationWorker(BaseWorker)(sources, variant: str, reference_body_px: float, overlap: float, executor=None, budget: int = CALIBRATION_SAMPLE_FRAMES)`; signal `result_ready(object)` emitting `list[GeometryCalibrationPoint]`; `cancel()`; `cancelled` property; attribute `sampled_frames: list[str]`.
  - `DetectKitProject.geometry_calibration: dict[str, Any]` (keyed by SAM2 variant) and `geometry_escalation_settings: dict[str, Any]`, both default `{}` and round-tripped like `semantic_calibration`.

- [ ] **Step 1: Failing tests**

```python
# tests/test_sam2_escalation.py
def test_project_round_trips_geometry_calibration():
    from hydra_suite.detectkit.gui.models import DetectKitProject

    p = DetectKitProject(project_dir="/tmp/x")
    p.geometry_calibration = {"sam2.1-hiera-tiny": {"recommended_index": 0}}
    p.geometry_escalation_settings = {"tile_fraction": 0.1}
    q = DetectKitProject.from_dict(p.to_dict())
    assert q.geometry_calibration == p.geometry_calibration
    assert q.geometry_escalation_settings == p.geometry_escalation_settings
```

(Check the real constructor/serialisation names in `models.py` around lines 335-480 — use whatever `semantic_calibration` uses.)

```python
# tests/test_sam2_calibration.py
def test_worker_samples_polygon_frames_and_emits_points(tmp_path, qtbot=None):
    from hydra_suite.detectkit.gui.models import OBBSource
    from hydra_suite.detectkit.jobs.sam2_escalation import Sam2CalibrationWorker

    root = tmp_path / "s"
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir(parents=True)
    cv2.imwrite(str(root / "images" / "a.png"), np.zeros((200, 200, 3), np.uint8))
    ang = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    pts = np.stack([0.25 + 0.05 * np.cos(ang), 0.25 + 0.05 * np.sin(ang)], 1).ravel()
    (root / "labels" / "a.txt").write_text("0 " + " ".join(f"{v:.5f}" for v in pts) + "\n")
    src = OBBSource(path=str(root), name="s", level="polygon")
    out = []
    w = Sam2CalibrationWorker([src], "v", reference_body_px=20.0, overlap=0.5, executor=PerfectExec())
    w.result_ready.connect(out.append)
    w.execute()
    assert out and out[0] and all(p.n_instances == 1 for p in out[0])
    assert w.sampled_frames == [str(root / "images" / "a.png")]
```

- [ ] **Step 2: Run, verify FAIL.**
- [ ] **Step 3: Implement.** Worker `execute()`: `frames = stratified_calibration_frames(sources, budget=budget, polygon_only=True)`; if empty raise `RuntimeError("No polygon ground-truth frames in the calibration sources.")`; `self.sampled_frames = [str(p) for p, _ in frames]`; executor = injected or `Sam2SegmentExecutor.from_variant(variant)`; call `calibrate_geometry(executor, [(p, [r.points for r in recs]) for p, recs in frames], reference_body_px=..., overlap=..., progress=lambda pct, msg: (self.progress.emit(pct), self.status.emit(msg)), should_stop=lambda: self._cancel)`; emit `result_ready`. `calibrate_geometry`'s `progress` signature is `(pct: int, message: str)`. Add the two project fields next to `semantic_calibration` and to the dict-field name set at ~476.
- [ ] **Step 4: Run** both test files → pass.
- [ ] **Step 5: Commit** `"Add SAM2 calibration worker and project persistence"`

---

### Task 9: `CalibrationSourceSelector` widget

**Files:**
- Create: `src/hydra_suite/detectkit/gui/widgets/__init__.py` (empty), `src/hydra_suite/detectkit/gui/widgets/calibration_source_selector.py`
- Test: `tests/test_calibration_source_selector.py`

**Interfaces:**
- Produces `CalibrationSourceSelector(QWidget)(sources, *, escalation_eligible: Callable[[source], bool] = lambda s: True, ineligible_reason: str = "", parent=None)`:
  - attributes `calibration_list`, `escalation_list` (`QListWidget`, MultiSelection, no checkboxes — same interaction as today's dialogs)
  - `calibration_sources() -> list`, `escalation_sources() -> list` (source objects, list order)
  - `restore(calibration_paths: list[str] | None, escalation_names: list[str] | None, escalation_paths: list[str] | None)` — calibration default (None) = all polygon sources selected; escalation selects by path first, else by name; ineligible rows never selected
  - `select_escalation_source(name: str)` — select only that (eligible) row
  - `state() -> dict` with keys `calibration_source_paths`, `escalation_source_paths`, `escalation_source_names`
  - signals `calibration_changed()`, `escalation_changed()`
  - `has_calibration_sources() -> bool`
  - an empty-state `QLabel` (`self.calibration_empty_label`) visible iff no polygon sources: "No polygon ground truth in this project. Label polygons on a few frames to calibrate."
  - Calibrate list is populated with sources where `has_polygon_frames(s)`; group titles "Calibrate on (polygon ground truth)" and "Escalate".

- [ ] **Step 1: Failing tests**

```python
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication

from hydra_suite.detectkit.gui.widgets.calibration_source_selector import (
    CalibrationSourceSelector,
)

_app = QApplication.instance() or QApplication([])


def _sources(tmp_path):
    from tests.test_calibration_frames import OBB, POLY, _src

    return [_src(tmp_path, "poly", {"a": POLY}), _src(tmp_path, "box", {"a": OBB})]


def test_calibrate_lists_only_polygon_sources_all_selected(tmp_path):
    poly, box = _sources(tmp_path)
    sel = CalibrationSourceSelector([poly, box])
    assert sel.calibration_list.count() == 1
    assert sel.calibration_sources() == [poly]
    assert sel.escalation_list.count() == 2 and sel.escalation_sources() == []


def test_restore_legacy_names_and_state_round_trip(tmp_path):
    poly, box = _sources(tmp_path)
    sel = CalibrationSourceSelector([poly, box])
    sel.restore(None, ["box"], None)
    assert sel.escalation_sources() == [box]
    st = sel.state()
    sel2 = CalibrationSourceSelector([poly, box])
    sel2.restore(st["calibration_source_paths"], None, st["escalation_source_paths"])
    assert sel2.escalation_sources() == [box] and sel2.calibration_sources() == [poly]


def test_ineligible_escalation_rows_are_disabled(tmp_path):
    poly, box = _sources(tmp_path)
    sel = CalibrationSourceSelector(
        [poly, box], escalation_eligible=lambda s: s.name != "poly", ineligible_reason="Already polygon"
    )
    sel.restore(None, ["poly", "box"], None)
    assert sel.escalation_sources() == [box]


def test_empty_state_when_no_polygon_sources(tmp_path):
    _poly, box = _sources(tmp_path)
    sel = CalibrationSourceSelector([box])
    assert not sel.has_calibration_sources()
    assert not sel.calibration_empty_label.isHidden()
```

(If `tests` is not importable as a package, copy the `_src` helper and constants into this file instead.)

- [ ] **Step 2: Run, verify FAIL.**
- [ ] **Step 3: Implement** with `QVBoxLayout` holding two `QGroupBox`es. Store source objects with `item.setData(Qt.ItemDataRole.UserRole, index)`; disabled rows via clearing `ItemIsEnabled` and setting the tooltip; connect each list's `itemSelectionChanged` to its signal. Read `escalate_sam2_dialog.py` for the exact flag/role idiom to copy.
- [ ] **Step 4: Run** → pass.
- [ ] **Step 5: Commit** `"Add CalibrationSourceSelector for split calibrate/escalate source lists"`

---

### Task 10a: Wire the selector into the SAM3 dialog

**Files:**
- Modify: `src/hydra_suite/detectkit/gui/dialogs/semantic_escalation_dialog.py`
- Modify: `src/hydra_suite/detectkit/gui/escalation_actions.py` (`resolve_reference_body_px` — measure polygon calibration sources first)
- Test: `tests/test_semantic_escalation_dialog_persistence.py` (append; follow its `qapp`/`available_checkpoint` fixtures)

Changes:
1. Replace the "Sources to escalate" group with `self._selector = CalibrationSourceSelector(self._sources)`; keep `self._list = self._selector.escalation_list` so existing tests/handlers keep working. `self._selector.restore(saved.get("calibration_source_paths"), saved.get("source_names"), saved.get("escalation_source_paths"))`.
2. `selected_sources()` → `self._selector.escalation_sources()`. New `calibration_sources()` → `self._selector.calibration_sources()`.
3. `_refresh_calibration_enabled`: enabled iff `self._selector.has_calibration_sources()` and at least one calibration source selected; reason text "Calibration needs polygon ground truth. Label polygons on a few frames (every animal in the frame) to calibrate." Update the class docstring (drop the "ANY geometry level" claim). Connect `calibration_changed` to `_refresh_calibration_enabled` and `_refresh_scale_warning`.
4. `_run_calibration` uses `self.calibration_sources()` (replaces `self.selected_sources() or self._sources`).
5. `_settings_payload` / `_store_calibration` write `**self._selector.state()` plus legacy `"source_names"` = escalation names.
6. Scale warning: `self._scale_note = QLabel("")` under the actions; `_refresh_scale_warning()` computes `measure_median_body_px(...)[0]` for calibration and escalation sources (cache per tuple of source paths in a dict to avoid re-decoding) and, if `scale_mismatch(a, b)`, shows "⚠ Calibration frames' animals (~{a:.0f} px) differ in size from the escalation targets' (~{b:.0f} px) by more than 1.5×; a calibrated tile fraction may not transfer." Connected to both selector signals; also called once at construction.
7. `resolve_reference_body_px`: measure over `[s for s in sources if has_polygon_frames(s)] or sources` (polygon frames give the scale the calibration ran at).

Tests to append:

```python
def test_semantic_dialog_calibrates_on_polygon_sources_and_escalates_selection(
    qapp, available_checkpoint, tmp_path
):
    from tests.test_calibration_frames import OBB, POLY, _src
    from hydra_suite.detectkit.gui.dialogs.semantic_escalation_dialog import (
        SemanticEscalationDialog,
    )

    poly, box = _src(tmp_path, "poly", {"a": POLY}), _src(tmp_path, "box", {"a": OBB})
    project = _project(tmp_path, [poly, box])  # reuse the file's project helper
    project.semantic_escalation_settings = {"source_names": ["box"]}
    dlg = SemanticEscalationDialog([poly, box], 20.0, project=project)
    assert dlg.calibration_sources() == [poly]
    assert dlg.selected_sources() == [box]
    assert dlg._btn_calibrate.isEnabled()


def test_semantic_dialog_box_only_project_cannot_calibrate(qapp, available_checkpoint, tmp_path):
    from tests.test_calibration_frames import OBB, _src
    from hydra_suite.detectkit.gui.dialogs.semantic_escalation_dialog import (
        SemanticEscalationDialog,
    )

    box = _src(tmp_path, "box", {"a": OBB})
    dlg = SemanticEscalationDialog([box], 20.0, project=_project(tmp_path, [box]))
    assert not dlg._btn_calibrate.isEnabled()
```

Adjust the helper names to what the test file actually provides. Run the whole `tests/test_semantic_*dialog*.py`, `tests/test_detectkit_sam2_escalation_wiring.py`, `tests/test_semantic_escalation_job.py` set; fix any test that asserted "any geometry level qualifies" by updating it to the new polygon-only contract (that is the intended behaviour change — say so in the commit message).

Commit: `"Split SAM3 dialog into calibrate-on and escalate source lists; polygon-only calibration"`

---

### Task 10b: SAM2 dialog — selector, tiling, calibration, results

**Files:**
- Modify: `src/hydra_suite/detectkit/gui/dialogs/escalate_sam2_dialog.py`
- Create: `src/hydra_suite/detectkit/gui/dialogs/geometry_calibration_results.py` (small results table widget)
- Modify: `src/hydra_suite/detectkit/gui/escalation_actions.py` (`on_escalate_geometry`)
- Test: `tests/test_escalate_sam2_dialog.py` (append)

New dialog signature: `EscalateSam2Dialog(sources, parent=None, *, project=None, reference_body_px: float = 0.0, persist_callback=None)` (existing positional calls keep working).

Layout: left = `CalibrationSourceSelector(sources, escalation_eligible=lambda s: s.level != "polygon", ineligible_reason="This source already contains segmentation polygons.")`; keep `self._list = self._selector.escalation_list` and existing methods (`selectable_source_names` returns enabled escalation rows' display text, `preselect_source` → `select_escalation_source`, `selected_sources` → names, `selected_source_paths` → paths). Default escalation selection = all eligible (current behaviour) unless saved settings exist. Right = variant combo; "Body size (px)" `QDoubleSpinBox` (prefill `reference_body_px`); "Tile fraction" `QDoubleSpinBox` (0 = "full frame (no tiling)", 3 decimals, range 0–0.9, default from `project.geometry_calibration[variant]` chosen point else 0.0); resolved tile label (reuse `resolve_tile_px`); "Calibrate against polygon frames…" button (enabled iff selector has calibration sources); `GeometryCalibrationResults` table; status label; scale-warning label (same helper/semantics as 10a).

Accessors: `tiling_parameters() -> dict` = `{"reference_body_px", "tile_fraction" (None if 0), "overlap": DEFAULT_OVERLAP}`.

Calibrate flow: `Sam2CalibrationWorker(self._selector.calibration_sources(), variant, reference_body_px, overlap)` with a window-modal `QProgressDialog` (Cancel → `worker.cancel`), on result: `best, reason = recommend_geometry(points)`; table shows rows (fraction or "full frame", tiles/frame, s/frame, median IoU, p10 IoU, fallback %), recommended row highlighted; selecting a row sets the tile-fraction control; when not cancelled persist `project.geometry_calibration[variant] = {"created_at", "points": [asdict], "recommended_index", "reason", "calibration_source_paths", "median_body_px", "sampled_frames"}` and call `persist_callback`. On dialog `accept`, persist `project.geometry_escalation_settings = {**selector.state(), "tile_fraction", "reference_body_px", "variant"}`.

`GeometryCalibrationResults(QTableWidget)`: `set_points(points, recommended)`, signal `point_chosen(object)`.

`on_escalate_geometry`: build the dialog with `project=window._project`, `reference_body_px=resolve_reference_body_px(window._project)[0]`, `persist_callback=window._save_current_project`; pass `**dlg.tiling_parameters()` into `EscalationRequest`; include `result.seam_fallbacks` and the tile size in the completion message ("…, N segmented on the full frame because their box crossed a tile seam").

Tests to append:

```python
def test_sam2_dialog_tiling_defaults_to_full_frame():
    dlg = EscalateSam2Dialog([OBBSource(name="a", level="obb")])
    assert dlg.tiling_parameters()["tile_fraction"] is None


def test_sam2_dialog_restores_calibrated_fraction_and_split_lists(tmp_path):
    from types import SimpleNamespace
    from tests.test_calibration_frames import OBB, POLY, _src

    poly, box = _src(tmp_path, "poly", {"a": POLY}), _src(tmp_path, "box", {"a": OBB})
    project = SimpleNamespace(
        project_dir=str(tmp_path),
        geometry_calibration={DEFAULT_VARIANT: {
            "recommended_index": 0,
            "points": [dict(tile_fraction=0.1, tile_px=200, owned_tiles_per_frame=4.0,
                            seconds_per_frame=1.0, median_iou=0.8, p10_iou=0.6,
                            fallback_rate=0.0, seam_fallback_rate=0.0, n_instances=40)],
        }},
        geometry_escalation_settings={},
    )
    dlg = EscalateSam2Dialog([poly, box], project=project, reference_body_px=20.0)
    assert dlg.tiling_parameters()["tile_fraction"] == pytest.approx(0.1)
    assert dlg._selector.calibration_sources() == [poly]
    assert dlg.selected_sources() == ["box"]
    assert dlg._btn_calibrate.isEnabled()
```

Run `tests/test_escalate_sam2_dialog.py tests/test_detectkit_sam2_escalation_wiring.py` → pass. Commit `"Add tiling, calibration and split source lists to the SAM2 escalation dialog"`.

---

### Task 11: GPU measurement on mehek — set IOU_FLOOR / FALLBACK_CEIL

**Files:**
- Create: `tools/sam2_geometry_calibration/measure.py`
- Modify: `src/hydra_suite/core/inference/sam2/calibration.py` (replace PROVISIONAL constants)
- Modify: this plan + spec (record numbers)

Data: courtship `~/detectkit-training/improved_ant_detection/merged_source_20260914` (117 all-polygon frames, 2,794 instances) — copy to mehek `~/geom_sahi_data/merged_source_20260914` with `ssh courtship 'tar -C ... -cf - merged_source_20260914' | ssh mehek 'tar -C ~/geom_sahi_data -xf -'` (via this Mac).

`measure.py` (headless, Qt-free): args `--source DIR --variant V --budget N --fractions 0.05,0.1,0.2,0.3,full --device cuda`; builds an `OBBSource`, `stratified_calibration_frames(polygon_only=True, budget=N)`, `measure_median_body_px`, runs `calibrate_geometry` with `Sam2SegmentExecutor.from_variant(V, device)`; prints a table and writes JSON with every `GeometryCalibrationPoint` plus `median_body_px`, frame count, git sha, variant, GPU name.

Steps:
- [ ] Bundle the branch to mehek (`git bundle create /tmp/geom.bundle main..feat/geometry-escalation-sahi`, scp, `git fetch /tmp/geom.bundle feat/geometry-escalation-sahi:feat/geometry-escalation-sahi`, worktree `~/hydra-suite/.worktrees/geom-sahi`); kill stale sleap/hydra procs first (`pgrep -u rutalab -f "sleap|hydra"`).
- [ ] Run with `conda activate hydra-cuda`, `PYTHONPATH=<wt>/src`, variants `sam2.1-hiera-tiny` and `sam2.1-hiera-base_plus` (the dialog default), budget 30 frames.
- [ ] Decide constants from the data: `IOU_FLOOR` = the full-frame-vs-best gap informed floor — set it so the best measured configuration clears it with margin and the clearly-bad ones fail; `FALLBACK_CEIL` likewise. Record the rationale and the full table in the spec under a new "Measurement (2026-10-03)" section and here.
- [ ] Answer explicitly: does per-box tiling beat full frame on these small animals (median IoU / fallback, and at what s/frame cost)?
- [ ] Commit `"Set SAM2 calibration floors from mehek measurement"`.

If mehek or the data is unavailable: STOP before Task 12; report thresholds still PROVISIONAL.

---

### Task 12: Gate, adversarial review, merge

- [ ] `make format` (or black+isort on touched files); `make lint` clean for touched files.
- [ ] Full suite delta on `hydra-mps`: run the suite on the branch base (`main` @ the worktree's base commit, in a separate detached worktree) and on the branch; compare the SETS of failing test ids (not counts). Any new failure → fix.
- [ ] Adversarial review by a different model (dispatch a review agent with `model: "fable"` or `"sonnet"`) over `git diff main...HEAD` with the spec; fix confirmed findings, re-run affected tests.
- [ ] Docs lifecycle: `git mv` the spec into `docs/superpowers/specs/done/` and this plan into `docs/superpowers/plans/done/`, fixing the spec's `**Status:**` to `Shipped — merged to main (<sha>)`.
- [ ] Merge `--no-ff` into local `main` (not pushed); remove worktree + branch.
