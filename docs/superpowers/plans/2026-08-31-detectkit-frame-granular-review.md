# DetectKit Frame-Granular Review Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace DetectKit's per-source, all-or-nothing escalation review with one
producer-agnostic `StagedReview` reviewed frame by frame, applied immediately to the
source it ran on, with an explicit revert.

**Architecture:** A new merge primitive (`data/al/merge.py`) plus a label reader make
"add these instances to what is already there" expressible for the first time. A
`StagedReview` dataclass generalises `PendingEscalation`, and a producer-agnostic
accept path (`detectkit/jobs/staged_review.py`) applies one frame at a time, keyed by
the staged label's relative path, recording outcomes in `decisions.json` inside the
staging directory and snapshotting the source's prior state into `labels_before/` on
first accept. A review bar above the canvas drives it; the per-source checkbox dialog
is retired; SAM3's sibling-source path is deleted; dataset inference becomes a third
producer that stages into the same contract.

**Tech Stack:** Python 3.11+, NumPy, OpenCV, PySide6 (Qt6), pytest, pytest-qt.

**Spec:** `docs/superpowers/specs/2026-08-31-detectkit-frame-granular-review-design.md`

## Global Constraints

- **This plan executes on the post-overlay-registry tree.** Its integration surface is
  the registry API already built there: `PROVIDERS`, `provider.build(ctx) ->
  OverlayLayer | None`, `OBBCanvas.set_layer(layer)` (**one argument** — the key comes
  from `layer.key`) / `remove_layer(key)` / `set_layer_visible(key, visible)`, and
  `DetectKitMainWindow._refresh_overlays(keys=(...))`. The registry landed FIRST,
  inverting the sequencing the design spec originally assumed; the spec has **already
  been amended to record this** (commit `32c5e8ed` plus its "Relationship to the
  overlay registry spec" section). Do not re-amend that section — it is correct and
  more detailed than any summary. No task may add a fourth overlay layer or any
  per-instance interaction.
- **Dependency direction:** `data/` and `utils/` must never import from
  `hydra_suite.detectkit` or `hydra_suite.core.inference`. `detectkit/jobs/` may import
  from `data/`, `utils/`, `core/`.
- **Merge invariant (assert in tests, never merely trust):** `MergeMode.ADD_NEW` may
  only *append*. It must never modify, reorder, or drop an existing record.
- **Level vocabulary:** `GeometryLevel` from `hydra_suite.utils.geometry_levels`;
  ordering `AABB < OBB < POLYGON`. Upward derivation is refused everywhere except the
  one documented promotion path (Task 7), which lifts an OBB quad to a 4-point polygon
  by re-encoding, not by inventing points.
- **No equivalence gate applies** — nothing here touches the tracking pipeline. The
  gate is `python -m pytest` on the named test files.
- **Every task ends with `make format` clean and a commit.** Run
  `make format && make lint-moderate` before each commit; commit messages use the
  repo's `feat(detectkit):` / `refactor(detectkit):` / `test(detectkit):` prefixes.
- **Commit as the configured git user.** Do not add a `Co-Authored-By: Claude` trailer.

---

## File Structure

**Created**

| Path | Responsibility |
|---|---|
| `src/hydra_suite/utils/polygon_iou.py` | The rasterised polygon IoU, moved down from `core/inference/masks.py` so `data/` can use it without a lateral import. |
| `src/hydra_suite/data/al/merge.py` | `MergeMode`, `merge_records` — the missing merge primitive. |
| `src/hydra_suite/detectkit/jobs/staged_review.py` | Producer-agnostic frame-granular accept/reject/revert; `decisions.json`; the `labels_before/` snapshot; staged→source class-id resolution. |
| `src/hydra_suite/detectkit/jobs/inference_stager.py` | Writes dataset-inference predictions into the staging contract. |
| `src/hydra_suite/detectkit/gui/panels/review_bar.py` | The review bar widget above the canvas. |
| `tests/test_al_merge.py` | Merge-rule tests. |
| `tests/test_al_label_reader.py` | `read_label_file` round-trip tests. |
| `tests/test_detectkit_staged_review.py` | Accept/reject/revert/promotion/producer-agnosticism. |
| `tests/test_detectkit_review_bar.py` | Review-bar widget behaviour. |
| `tests/test_detectkit_inference_stager.py` | Inference producer. |

**Modified**

| Path | Change |
|---|---|
| `src/hydra_suite/core/inference/masks.py` | `polygon_iou` becomes a re-export of the moved function. |
| `src/hydra_suite/data/al/labels.py` | Gains `read_label_file`. |
| `src/hydra_suite/detectkit/gui/models.py` | `PendingEscalation` → `StagedReview`; `OBBSource.pending_escalation` → `staged_review`. |
| `src/hydra_suite/detectkit/gui/overlays/providers.py` | `StagedEscalationProvider` → `StagedReviewProvider`; skips decided frames. |
| `src/hydra_suite/detectkit/gui/main_window.py` | Hosts the review bar; stages predictions. |
| `src/hydra_suite/detectkit/gui/escalation_actions.py` | `on_review_escalations` retired. |
| `src/hydra_suite/detectkit/jobs/sam2_escalation.py` | Stages a `StagedReview`; wholesale accept deleted. |
| `src/hydra_suite/detectkit/jobs/semantic_escalation.py` | Stages a `StagedReview`; sibling-source accept deleted. |

**Deleted**

- `src/hydra_suite/detectkit/gui/dialogs/review_escalations_dialog.py`
- `tests/test_detectkit_review_escalations_dialog.py`
- `accept_pending_semantic_escalation`, `_unique_source_name` (`jobs/semantic_escalation.py`)
- `accept_pending_escalation` (`jobs/sam2_escalation.py`)

---

# Phase 1 — The library primitives (Tasks 1-3)

Pure library work, no UI, independently mergeable.

### Task 1: Move `polygon_iou` to `utils/`

`data/al/merge.py` needs polygon IoU. It lives in `core/inference/masks.py:51`, and
`data/` importing `core/inference` is a lateral dependency the layering rules forbid.
`utils/rotated_iou.py` is **not** a substitute — it is a convex Sutherland-Hodgman quad
clip and is silently wrong for the non-convex contours SAM3 produces (its own docstring
says so).

This is a **move, not a rewrite**. The rasterisation behaviour — including the 4x
supersampling correction and the disjoint-bbox short-circuit — must not change.

**Files:**
- Create: `src/hydra_suite/utils/polygon_iou.py`
- Modify: `src/hydra_suite/core/inference/masks.py` (remove the body, re-export)
- Test: `tests/test_semantic_masks.py` (must pass **untouched**)

**Interfaces:**
- Consumes: nothing.
- Produces: `hydra_suite.utils.polygon_iou.polygon_iou(a: np.ndarray, b: np.ndarray) -> float`,
  re-exported unchanged as `hydra_suite.core.inference.masks.polygon_iou`.

- [ ] **Step 1: Confirm the current tests pass before touching anything**

```bash
python -m pytest tests/test_semantic_masks.py -q
```

Expected: PASS. This is the baseline the move must preserve.

- [ ] **Step 2: Create the new module by moving the function verbatim**

Cut `polygon_iou` (the whole `def polygon_iou` through its `return`) out of
`src/hydra_suite/core/inference/masks.py` and paste it into the new file, keeping the
docstring word for word:

```python
"""Rasterised polygon IoU.

Lives in ``utils`` (the bottom layer) so ``data.al.merge`` and
``core.inference.semantic`` can both use it without a lateral dependency.
Moved here from ``core/inference/masks.py``; the behaviour -- including the
4x supersampling correction and the disjoint-bbox short-circuit -- is
unchanged, and ``tests/test_semantic_masks.py`` pins that.
"""

from __future__ import annotations

import cv2
import numpy as np


def polygon_iou(a: np.ndarray, b: np.ndarray) -> float:
    ...  # the moved body, verbatim, docstring included
```

- [ ] **Step 3: Re-export from the old location**

In `src/hydra_suite/core/inference/masks.py`, where the function was, put:

```python
# polygon_iou moved to utils/ so data/al/merge.py can use it without importing
# core.inference (a lateral dependency the layering rules forbid). Re-exported
# here because three modules under core/inference/semantic/ import it from this
# module, as does tests/test_semantic_masks.py.
from hydra_suite.utils.polygon_iou import polygon_iou  # noqa: F401
```

Move that import to the top of the file with the other imports, and add
`"polygon_iou"` to `__all__` if the module defines one.

- [ ] **Step 4: Verify nothing broke**

```bash
python -m pytest tests/test_semantic_masks.py -q
python -c "from hydra_suite.core.inference.masks import polygon_iou; print(polygon_iou.__module__)"
grep -rn "polygon_iou" src/hydra_suite | sort
```

Expected: tests PASS untouched; `__module__` prints
`hydra_suite.utils.polygon_iou`; the three `core/inference/semantic/*` importers are
unchanged.

- [ ] **Step 5: Guard the layering rule with a test**

Append to `tests/test_semantic_masks.py`:

```python
def test_polygon_iou_lives_in_utils_so_data_can_import_it():
    """data/al/merge.py needs this; data/ may not import core.inference."""
    from hydra_suite.utils.polygon_iou import polygon_iou as moved
    from hydra_suite.core.inference.masks import polygon_iou as reexported

    assert moved is reexported
    assert moved.__module__ == "hydra_suite.utils.polygon_iou"
```

- [ ] **Step 6: Run and commit**

```bash
python -m pytest tests/test_semantic_masks.py -q
make format && make lint-moderate
git add src/hydra_suite/utils/polygon_iou.py src/hydra_suite/core/inference/masks.py tests/test_semantic_masks.py
git commit -m "refactor(utils): move polygon_iou down from core/inference to utils"
```

---

### Task 2: `read_label_file` — the missing reader

`write_label_file` exists; nothing reads a label file back into `LabelRecord`s.
`parse_obb_label` (`detectkit/gui/utils.py:272`) returns GUI dicts and lives in the app
layer, so `data/al/merge.py` cannot use it.

**Per-line level classification, not a caller-supplied level.** A 9-field line is
ambiguous in principle (`classify_label_line` calls it `four_point`), but not in this
codebase: `_polygon_points` (`data/al/labels.py:30`) repeats the final vertex precisely
so a polygon-level file never contains a 4-point line. So the mapping is total and
unambiguous:

| Fields | Level |
|---|---|
| 5 | `AABB` |
| 9 | `OBB` |
| odd >= 7 (and not 9) | `POLYGON` |
| anything else | skipped |

**Confidence** is not stored on disk. The reader fills `confidence=1.0`, meaning
"asserted", and nothing downstream reads it — `write_label_file` ignores it entirely.

**Files:**
- Modify: `src/hydra_suite/data/al/labels.py`
- Test: `tests/test_al_label_reader.py`

**Interfaces:**
- Consumes: `LabelRecord` from `data/al/escalation.py`; `classify_label_line`,
  `GeometryLevel` from `utils/geometry_levels.py`.
- Produces: `read_label_file(path: str | Path, frame_size: tuple[int, int]) -> list[LabelRecord]`
  — pixel-space records, `frame_size` is `(height, width)` to match `write_label_file`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_al_label_reader.py`:

```python
import numpy as np
import pytest

from hydra_suite.data.al.escalation import LabelRecord
from hydra_suite.data.al.labels import read_label_file, write_label_file
from hydra_suite.utils.geometry_levels import GeometryLevel

FRAME = (100, 200)  # (height, width)


def _rec(points, level, class_id=0):
    return LabelRecord(
        class_id=class_id,
        confidence=1.0,
        points=np.asarray(points, dtype=np.float32),
        level=level,
    )


def test_reads_an_obb_line_as_obb(tmp_path):
    path = tmp_path / "a.txt"
    quad = [[10, 10], [50, 10], [50, 40], [10, 40]]
    write_label_file(path, [_rec(quad, GeometryLevel.OBB)], FRAME, GeometryLevel.OBB)

    out = read_label_file(path, FRAME)

    assert len(out) == 1
    assert out[0].level is GeometryLevel.OBB
    assert out[0].class_id == 0
    assert out[0].confidence == 1.0
    np.testing.assert_allclose(out[0].points, np.array(quad, dtype=np.float32), atol=0.05)


def test_reads_an_aabb_line_as_aabb(tmp_path):
    path = tmp_path / "a.txt"
    quad = [[10, 10], [50, 10], [50, 40], [10, 40]]
    write_label_file(path, [_rec(quad, GeometryLevel.AABB)], FRAME, GeometryLevel.AABB)

    out = read_label_file(path, FRAME)

    assert out[0].level is GeometryLevel.AABB
    np.testing.assert_allclose(out[0].points, np.array(quad, dtype=np.float32), atol=0.05)


def test_reads_a_five_point_polygon_as_polygon(tmp_path):
    path = tmp_path / "a.txt"
    poly = [[10, 10], [50, 12], [60, 40], [30, 55], [12, 38]]
    write_label_file(path, [_rec(poly, GeometryLevel.POLYGON)], FRAME, GeometryLevel.POLYGON)

    out = read_label_file(path, FRAME)

    assert out[0].level is GeometryLevel.POLYGON
    assert out[0].points.shape == (5, 2)


def test_a_promoted_quad_round_trips_as_polygon_not_obb(tmp_path):
    """_polygon_points repeats the last vertex; the reader must see 5 points."""
    path = tmp_path / "a.txt"
    quad = [[10, 10], [50, 10], [50, 40], [10, 40]]
    write_label_file(path, [_rec(quad, GeometryLevel.OBB)], FRAME, GeometryLevel.POLYGON)

    out = read_label_file(path, FRAME)

    assert out[0].level is GeometryLevel.POLYGON
    assert out[0].points.shape == (5, 2)


def test_class_ids_are_preserved(tmp_path):
    path = tmp_path / "a.txt"
    quad = [[10, 10], [50, 10], [50, 40], [10, 40]]
    write_label_file(path, [_rec(quad, GeometryLevel.OBB, class_id=3)], FRAME, GeometryLevel.OBB)

    assert read_label_file(path, FRAME)[0].class_id == 3


def test_malformed_and_empty_lines_are_skipped(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("\n0 0.1 0.2\n\nnot a label\n0 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2\n")

    out = read_label_file(path, FRAME)

    assert len(out) == 1


def test_a_missing_file_reads_as_empty(tmp_path):
    assert read_label_file(tmp_path / "nope.txt", FRAME) == []
```

- [ ] **Step 2: Run to verify they fail**

```bash
python -m pytest tests/test_al_label_reader.py -q
```

Expected: FAIL — `ImportError: cannot import name 'read_label_file'`.

- [ ] **Step 3: Implement the reader**

Append to `src/hydra_suite/data/al/labels.py` (and add
`from hydra_suite.utils.geometry_levels import GeometryLevel, classify_label_line` to
the existing import):

```python
_LEVEL_BY_KIND = {
    "aabb": GeometryLevel.AABB,
    "four_point": GeometryLevel.OBB,
    "polygon": GeometryLevel.POLYGON,
}


def read_label_file(
    path: str | Path,
    frame_size: tuple[int, int],
) -> list[LabelRecord]:
    """Read one YOLO label file back into pixel-space LabelRecords.

    The inverse of `write_label_file`. `frame_size` is (height, width), the
    same convention, because the file stores normalised coordinates.

    Each line's level comes from its own field count via
    `classify_label_line`, not from a caller-supplied level. The `four_point`
    case (9 fields) is ambiguous in principle -- an OBB or a 4-point quad
    polygon -- but not here: `_polygon_points` repeats the final vertex
    precisely so a polygon-level file never contains a 4-point line. A
    9-field line is therefore always an OBB, including inside a source whose
    own `level` says polygon (an unpromoted leftover), which is exactly what
    a caller merging into that source needs to know.

    Confidence is not stored on disk. Records read back carry
    ``confidence=1.0`` ("asserted"); nothing downstream reads it, and
    `write_label_file` ignores it.

    Unparseable lines are skipped, matching `parse_obb_label`'s tolerance for
    files a user may have hand-edited. A missing file reads as empty -- a
    frame with no label file has no labels, which is not an error.
    """
    height, width = int(frame_size[0]), int(frame_size[1])
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        return []

    records: list[LabelRecord] = []
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        kind = classify_label_line(len(parts))
        level = _LEVEL_BY_KIND.get(kind)
        if level is None:
            continue
        try:
            class_id = int(parts[0])
            coords = [float(v) for v in parts[1:]]
        except ValueError:
            continue

        if level is GeometryLevel.AABB:
            cx, cy, w, h = coords
            x1, y1 = cx - w / 2.0, cy - h / 2.0
            x2, y2 = cx + w / 2.0, cy + h / 2.0
            flat = [x1, y1, x2, y1, x2, y2, x1, y2]
        else:
            flat = coords

        pts = np.asarray(flat, dtype=np.float32).reshape(-1, 2)
        pts[:, 0] *= float(width)
        pts[:, 1] *= float(height)
        records.append(
            LabelRecord(
                class_id=class_id,
                confidence=1.0,
                points=pts,
                level=level,
            )
        )
    return records
```

- [ ] **Step 4: Run the tests**

```bash
python -m pytest tests/test_al_label_reader.py tests/test_al_labels.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
make format && make lint-moderate
git add src/hydra_suite/data/al/labels.py tests/test_al_label_reader.py
git commit -m "feat(data): add read_label_file, the inverse of write_label_file"
```

---

### Task 3: `merge_records` — the merge primitive

Every writer in the codebase truncates. `write_label_file` opens `"w"`; accept does
`rmtree` + `copytree`; the X-AnyLabeling sync-back does `rmtree` + copy. "Add these
instances to what is already there" is not expressible. This task makes it so.

**Files:**
- Create: `src/hydra_suite/data/al/merge.py`
- Test: `tests/test_al_merge.py`

**Interfaces:**
- Consumes: `LabelRecord`, `derive_down` from `data/al/escalation.py`;
  `polygon_iou` from `utils/polygon_iou.py` (Task 1).
- Produces:
  - `MergeMode.OVERWRITE`, `MergeMode.ADD_NEW`
  - `merge_records(existing, staged, *, mode, iou_threshold, level) -> list[LabelRecord]`
  - **Positional invariant relied on by Task 7:** under `ADD_NEW` the result is exactly
    `list(existing) + survivors`, in that order, with the existing objects returned by
    identity. Task 7 slices `result[len(existing):]` to get the survivors.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_al_merge.py`:

```python
import numpy as np
import pytest

from hydra_suite.data.al.escalation import LabelRecord
from hydra_suite.data.al.merge import MergeMode, merge_records
from hydra_suite.utils.geometry_levels import GeometryLevel


def _quad(x, y, size, class_id=0, level=GeometryLevel.OBB):
    return LabelRecord(
        class_id=class_id,
        confidence=1.0,
        points=np.array(
            [[x, y], [x + size, y], [x + size, y + size], [x, y + size]],
            dtype=np.float32,
        ),
        level=level,
    )


def test_overwrite_returns_only_staged():
    existing = [_quad(0, 0, 10), _quad(100, 100, 10)]
    staged = [_quad(50, 50, 10)]

    out = merge_records(
        existing, staged, mode=MergeMode.OVERWRITE, iou_threshold=0.5,
        level=GeometryLevel.OBB,
    )

    assert len(out) == 1
    np.testing.assert_allclose(out[0].points, staged[0].points)


def test_add_new_appends_a_non_overlapping_staged_record():
    existing = [_quad(0, 0, 10)]
    staged = [_quad(100, 100, 10)]

    out = merge_records(
        existing, staged, mode=MergeMode.ADD_NEW, iou_threshold=0.5,
        level=GeometryLevel.OBB,
    )

    assert len(out) == 2
    assert out[0] is existing[0]


def test_add_new_drops_a_staged_record_that_overlaps():
    existing = [_quad(0, 0, 20)]
    staged = [_quad(1, 1, 20)]  # IoU well above 0.5

    out = merge_records(
        existing, staged, mode=MergeMode.ADD_NEW, iou_threshold=0.5,
        level=GeometryLevel.OBB,
    )

    assert len(out) == 1
    assert out[0] is existing[0]


def test_add_new_iou_boundary_in_both_directions():
    """A staged record is dropped at IoU >= threshold and kept just below."""
    existing = [_quad(0, 0, 20)]
    overlapping = _quad(2, 0, 20)  # 18/22 columns shared -> IoU ~= 0.818

    dropped = merge_records(
        existing, [overlapping], mode=MergeMode.ADD_NEW, iou_threshold=0.80,
        level=GeometryLevel.OBB,
    )
    kept = merge_records(
        existing, [overlapping], mode=MergeMode.ADD_NEW, iou_threshold=0.85,
        level=GeometryLevel.OBB,
    )

    assert len(dropped) == 1
    assert len(kept) == 2


def test_add_new_compares_against_every_existing_record_not_just_the_first():
    existing = [_quad(0, 0, 10), _quad(100, 100, 20)]
    staged = [_quad(101, 101, 20)]

    out = merge_records(
        existing, staged, mode=MergeMode.ADD_NEW, iou_threshold=0.5,
        level=GeometryLevel.OBB,
    )

    assert len(out) == 2


def test_add_new_with_empty_existing_keeps_all_staged():
    staged = [_quad(0, 0, 10), _quad(100, 100, 10)]

    out = merge_records(
        [], staged, mode=MergeMode.ADD_NEW, iou_threshold=0.5,
        level=GeometryLevel.OBB,
    )

    assert len(out) == 2


def test_add_new_with_empty_staged_returns_existing_unchanged():
    existing = [_quad(0, 0, 10)]

    out = merge_records(
        existing, [], mode=MergeMode.ADD_NEW, iou_threshold=0.5,
        level=GeometryLevel.OBB,
    )

    assert out == existing


def test_add_new_never_modifies_reorders_or_drops_an_existing_record():
    """The invariant that makes immediate application safe."""
    existing = [_quad(0, 0, 20, class_id=1), _quad(60, 60, 20, class_id=2),
                _quad(200, 200, 20, class_id=3)]
    before = [r.points.copy() for r in existing]
    staged = [_quad(1, 1, 20), _quad(400, 400, 20), _quad(61, 61, 20)]

    out = merge_records(
        existing, staged, mode=MergeMode.ADD_NEW, iou_threshold=0.5,
        level=GeometryLevel.OBB,
    )

    # id(), not ==: LabelRecord.__eq__ compares ndarrays elementwise and
    # raises on bool(). List __eq__ happens to short-circuit on identity
    # here, but relying on that is a trap the next edit would spring.
    assert [id(r) for r in out[: len(existing)]] == [id(r) for r in existing]
    for rec, pts in zip(existing, before):
        np.testing.assert_array_equal(rec.points, pts)  # unmutated
    assert len(out) == len(existing) + 1  # only the disjoint one survived


def test_survivors_are_the_tail_slice():
    """The positional contract the file-level accept path relies on.

    Identity, via id(), NOT `in`/`==`. LabelRecord is a plain dataclass
    holding an ndarray, so its generated __eq__ compares `points`
    elementwise and bool() on the resulting array raises "truth value of an
    array is ambiguous" the moment class_id and confidence happen to match.
    """
    existing = [_quad(0, 0, 20)]
    staged = [_quad(1, 1, 20), _quad(400, 400, 20)]

    out = merge_records(
        existing, staged, mode=MergeMode.ADD_NEW, iou_threshold=0.5,
        level=GeometryLevel.OBB,
    )

    head, tail = out[: len(existing)], out[len(existing):]
    assert [id(r) for r in head] == [id(r) for r in existing]
    assert all(not any(r is e for e in existing) for r in tail)
    assert len(tail) == 1


def test_staged_above_the_target_level_is_derived_down():
    existing = [_quad(0, 0, 10, level=GeometryLevel.OBB)]
    poly = LabelRecord(
        class_id=0,
        confidence=1.0,
        points=np.array([[100, 100], [120, 102], [125, 120], [110, 130], [98, 118]],
                        dtype=np.float32),
        level=GeometryLevel.POLYGON,
    )

    out = merge_records(
        existing, [poly], mode=MergeMode.ADD_NEW, iou_threshold=0.5,
        level=GeometryLevel.OBB,
    )

    assert len(out) == 2
    assert out[1].level is GeometryLevel.OBB
    assert out[1].points.shape == (4, 2)


def test_records_below_the_target_level_are_refused_not_invented():
    """The primitive stays strict; lifting is the CALLER's explicit choice.

    `staged_review.accept_frame` re-tags records before calling this when a
    lift is genuinely wanted (a quad encoded as a 4-point polygon), so the
    primitive never has to guess.
    """
    existing = [_quad(0, 0, 10, level=GeometryLevel.POLYGON)]
    staged = [_quad(100, 100, 10, level=GeometryLevel.OBB)]

    with pytest.raises(ValueError, match="upward"):
        merge_records(
            existing, staged, mode=MergeMode.ADD_NEW, iou_threshold=0.5,
            level=GeometryLevel.POLYGON,
        )
```

- [ ] **Step 2: Run to verify they fail**

```bash
python -m pytest tests/test_al_merge.py -q
```

Expected: FAIL — `ModuleNotFoundError: hydra_suite.data.al.merge`.

- [ ] **Step 3: Implement the primitive**

Create `src/hydra_suite/data/al/merge.py`:

```python
"""Merging staged label records into a frame's existing ones.

Every writer in this codebase truncates -- `write_label_file` opens "w",
escalation accept did rmtree+copytree, the X-AnyLabeling sync-back does
rmtree+copy. "Add these instances to what is already there" had no
expression at all, which is why review was all-or-nothing. This is that
missing primitive.
"""

from __future__ import annotations

from enum import Enum, auto
from typing import Sequence

from hydra_suite.utils.geometry_levels import GeometryLevel
from hydra_suite.utils.polygon_iou import polygon_iou

from .escalation import LabelRecord, derive_down


class MergeMode(Enum):
    """How a staged frame's records combine with the existing ones."""

    OVERWRITE = auto()  # staged replaces existing for this frame
    ADD_NEW = auto()  # existing kept verbatim; non-overlapping staged appended


def merge_records(
    existing: Sequence[LabelRecord],
    staged: Sequence[LabelRecord],
    *,
    mode: MergeMode,
    iou_threshold: float,
    level: GeometryLevel,
) -> list[LabelRecord]:
    """Combine `staged` into `existing` at `level`.

    OVERWRITE returns the staged records alone, derived to `level`.

    ADD_NEW keeps every existing record -- by identity, in order, unmutated
    -- and appends each staged record whose IoU against EVERY existing
    record is below `iou_threshold`. The result is exactly
    ``list(existing) + survivors``; callers rely on that positional
    contract to know which records are new.

    That "a merge can only add" invariant is what makes applying a merge
    immediately (rather than accumulating a pending set) safe: no accept
    can silently degrade labels the user already curated. It is asserted in
    tests rather than trusted.

    IoU uses the rasterised `utils.polygon_iou`, not the convex quad clip in
    `utils/rotated_iou.py`, because staged contours are arbitrary non-convex
    polygons and the convex clip returns wrong areas for them silently.
    Comparison happens at `level`, after derivation, so an OBB source
    compares quads against quads.

    Raises ValueError (via `derive_down`) if a staged record is BELOW
    `level`: deriving upward would invent information.
    """
    staged_at_level = derive_down(list(staged), level)
    if mode is MergeMode.OVERWRITE:
        return staged_at_level

    existing_at_level = derive_down(list(existing), level)
    out = list(existing)
    for candidate in staged_at_level:
        if any(
            polygon_iou(candidate.points, prior.points) >= iou_threshold
            for prior in existing_at_level
        ):
            continue
        out.append(candidate)
    return out
```

Note `out` starts from `existing` (the ORIGINAL records, by identity), not
from `existing_at_level`: the derived copies exist only to compare like with
like. Returning the derived ones would silently rewrite the caller's
existing geometry, which is exactly what the "a merge can only add"
invariant forbids.

- [ ] **Step 4: Run the tests**

```bash
python -m pytest tests/test_al_merge.py -q
```

Expected: PASS (all 12).

- [ ] **Step 5: Commit**

```bash
make format && make lint-moderate
git add src/hydra_suite/data/al/merge.py tests/test_al_merge.py
git commit -m "feat(data): add merge_records, the ADD_NEW/OVERWRITE merge primitive"
```

---

# Phase 2 — `StagedReview`, decisions, and revert (Tasks 4-7)

The model and the core accept path, still driven by the existing per-source dialog
(which calls accept-all under the hood). The app keeps working throughout.

### Task 4: `StagedReview` replaces `PendingEscalation`

`PendingEscalation.primer_kind` is currently load-bearing for accept dispatch
(`review_escalations_dialog.py:107`). After this refactor it is load-bearing for
**nothing** — all producers accept identically. That is the point.

**Backwards compatibility is required, not optional:** a project holding a staged SAM2
or SAM3 escalation must review correctly with no migration step. The directory layout is
unchanged; only the JSON key names move. `from_dict` accepts the old
`pending_escalation` / `primer_kind` / `primer_variant` / `primer_prompt` /
`primer_params` / `sam2_variant` names; `to_dict` writes only the new ones. A project
saved by this version is **not readable by an older one** — the same one-way step the
`runtime_tier` migration took, stated here rather than discovered.

**Files:**
- Modify: `src/hydra_suite/detectkit/gui/models.py`
- Modify: `src/hydra_suite/detectkit/gui/overlays/providers.py` (attribute rename)
- Modify: `src/hydra_suite/detectkit/jobs/sam2_escalation.py`,
  `src/hydra_suite/detectkit/jobs/semantic_escalation.py`,
  `src/hydra_suite/detectkit/gui/escalation_actions.py`,
  `src/hydra_suite/detectkit/gui/dialogs/review_escalations_dialog.py` (call sites)
- Test: `tests/test_detectkit_models.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `StagedReview(staged_path: str, target_level: str, producer: str, producer_variant: str, prompt: str, params: dict, created_at: str)`
    with `to_dict()` / `from_dict(d)`.
  - `OBBSource.staged_review: StagedReview | None`.
  - `PendingEscalation = StagedReview` alias retained for one cycle so out-of-tree
    imports do not break mid-plan; removed in Task 14.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_detectkit_models.py`:

```python
from hydra_suite.detectkit.gui.models import OBBSource, StagedReview


def test_staged_review_round_trips():
    review = StagedReview(
        staged_path="/tmp/staging",
        target_level="polygon",
        producer="sam3",
        producer_variant="sam3-large",
        prompt="ant",
        params={"confidence": 0.35},
        created_at="2026-08-31T10:00:00",
    )

    restored = StagedReview.from_dict(review.to_dict())

    assert restored == review


def test_to_dict_writes_only_the_new_key_names():
    d = StagedReview(producer="sam2", producer_variant="sam2.1_hiera_large").to_dict()

    assert set(d) == {
        "staged_path", "target_level", "producer", "producer_variant",
        "prompt", "params", "created_at",
    }


def test_from_dict_accepts_a_legacy_sam2_record():
    legacy = {
        "staged_path": "/tmp/s",
        "target_level": "polygon",
        "sam2_variant": "sam2.1_hiera_large",
        "created_at": "2026-08-01T00:00:00",
    }

    review = StagedReview.from_dict(legacy)

    assert review.producer == "sam2"
    assert review.producer_variant == "sam2.1_hiera_large"
    assert review.prompt == ""


def test_from_dict_accepts_a_legacy_sam3_record():
    legacy = {
        "staged_path": "/tmp/s",
        "target_level": "polygon",
        "primer_kind": "sam3",
        "primer_variant": "sam3-large",
        "primer_prompt": "ant",
        "primer_params": {"confidence": 0.35},
        "created_at": "2026-08-01T00:00:00",
    }

    review = StagedReview.from_dict(legacy)

    assert review.producer == "sam3"
    assert review.producer_variant == "sam3-large"
    assert review.prompt == "ant"
    assert review.params == {"confidence": 0.35}


def test_source_loads_a_legacy_pending_escalation_key():
    src = OBBSource.from_dict(
        {
            "path": "/tmp/src",
            "name": "src",
            "pending_escalation": {
                "staged_path": "/tmp/s",
                "target_level": "polygon",
                "primer_kind": "sam3",
                "primer_prompt": "ant",
            },
        }
    )

    assert src.staged_review is not None
    assert src.staged_review.producer == "sam3"


def test_source_writes_the_new_key_name():
    src = OBBSource(path="/tmp/src", name="src", staged_review=StagedReview())

    d = src.to_dict()

    assert "staged_review" in d
    assert "pending_escalation" not in d
```

- [ ] **Step 2: Run to verify they fail**

```bash
python -m pytest tests/test_detectkit_models.py -q
```

Expected: FAIL — `ImportError: cannot import name 'StagedReview'`.

- [ ] **Step 3: Replace `PendingEscalation` with `StagedReview`**

In `src/hydra_suite/detectkit/gui/models.py`, replace the whole `PendingEscalation`
class with:

```python
@dataclass
class StagedReview:
    """A staged, not-yet-reviewed set of proposed labels for a source.

    Generalises the old `PendingEscalation`. ``producer`` is one of
    ``"sam2"``, ``"sam3"``, ``"inference"`` and is **provenance only**: all
    three stage into the same contract and accept through the same code
    path. It used to dispatch the accept (SAM2 overwrote in place, SAM3
    built a sibling source), and making it load-bearing for nothing is the
    entire point of the refactor. A test in
    tests/test_detectkit_staged_review.py fails if it ever becomes
    load-bearing again.

    The staging directory it points at holds::

        labels/          one .txt per frame, mirroring the source's images/
        classes.txt
        run.json         producer, params, fingerprint
        decisions.json   per-frame outcome (written during review)
        labels_before/   the source's prior labels, snapshotted on first accept
    """

    staged_path: str = ""
    target_level: str = "polygon"
    producer: str = "sam2"
    producer_variant: str = ""
    prompt: str = ""
    params: dict = field(default_factory=dict)
    created_at: str = ""

    def to_dict(self) -> dict:
        """Serialize to a plain dictionary, using only the new key names."""
        return {
            "staged_path": self.staged_path,
            "target_level": self.target_level,
            "producer": self.producer,
            "producer_variant": self.producer_variant,
            "prompt": self.prompt,
            "params": dict(self.params),
            "created_at": self.created_at,
        }

    @staticmethod
    def from_dict(d: dict) -> "StagedReview":
        """Restore, accepting both the new names and every legacy spelling.

        A project staged before this change must review with no migration
        step: the directory layout never changed, only these key names.
        Legacy records carry ``primer_kind``/``primer_variant``/
        ``primer_prompt``/``primer_params``; the oldest (pre-primer, SAM2
        only) carry just ``sam2_variant``.

        Note that `to_dict` writes only the new names, so a project saved by
        this version is NOT readable by an older one -- the same one-way
        step the runtime_tier migration took.
        """
        legacy_sam2_variant = str(d.get("sam2_variant", "") or "")
        producer = str(d.get("producer") or d.get("primer_kind") or "sam2")
        variant = str(
            d.get("producer_variant")
            or d.get("primer_variant")
            or legacy_sam2_variant
        )
        return StagedReview(
            staged_path=str(d.get("staged_path", "")),
            target_level=str(d.get("target_level", "polygon") or "polygon"),
            producer=producer,
            producer_variant=variant,
            prompt=str(d.get("prompt") or d.get("primer_prompt") or ""),
            params=dict(d.get("params") or d.get("primer_params") or {}),
            created_at=str(d.get("created_at", "")),
        )


# Retained for one cycle so any out-of-tree import keeps resolving while the
# rest of this plan lands. Deleted in the cleanup task.
PendingEscalation = StagedReview
```

- [ ] **Step 4: Rename the `OBBSource` field**

In `OBBSource`, replace the `pending_escalation` field, its `to_dict` entry, and its
`from_dict` entry with:

```python
    staged_review: StagedReview | None = None  # staged, unreviewed proposals
```

```python
            "staged_review": (
                self.staged_review.to_dict()
                if self.staged_review is not None
                else None
            ),
```

```python
            # The legacy key is read but never written: a project staged
            # before the rename must review without a migration step.
            staged_review=(
                StagedReview.from_dict(
                    d.get("staged_review") or d["pending_escalation"]
                )
                if (d.get("staged_review") or d.get("pending_escalation"))
                else None
            ),
```

- [ ] **Step 5: Update every call site**

```bash
grep -rn "pending_escalation\|primer_kind\|primer_variant\|primer_prompt\|primer_params" src/hydra_suite tests
```

Rename mechanically across the hits: `src.pending_escalation` → `src.staged_review`,
`pending.primer_kind` → `review.producer`, `.primer_variant` → `.producer_variant`,
`.primer_prompt` → `.prompt`, `.primer_params` → `.params`. In
`jobs/sam2_escalation.py` the construction becomes:

```python
        src.staged_review = StagedReview(
            staged_path=str(staged_root),
            target_level=GeometryLevel.POLYGON.label,
            producer="sam2",
            producer_variant=req.variant,
            created_at=datetime.now().isoformat(),
        )
```

and in `jobs/semantic_escalation.py` the SAM3 construction sets `producer="sam3"`,
`prompt=<the noun phrase>`, `params=<the run params>`. In
`overlays/providers.py`, `getattr(source, "pending_escalation", None)` becomes
`getattr(source, "staged_review", None)`, and the class is renamed
`StagedEscalationProvider` → `StagedReviewProvider` with `key = "staged"` kept as
`key = "escalation"` **for now** (the canvas key is a rendering identity; changing it
in the same commit as the model rename makes a bisect harder). Update
`overlays/__init__.py`'s imports and `__all__`, and
`tests/test_detectkit_overlay_providers.py` accordingly.

- [ ] **Step 6: Rewrite `tests/test_pending_escalation_model.py`**

That file constructs `PendingEscalation(primer_kind=..., sam2_variant=...)` and asserts
`to_dict()["sam2_variant"]` (lines 4-35). Those kwargs no longer exist and that key is
deliberately no longer written, so the file cannot merely be renamed — it pins the exact
round-trip semantics this task changes. Its coverage now lives in the six new tests in
`tests/test_detectkit_models.py` (Step 1), so delete it:

```bash
git rm tests/test_pending_escalation_model.py
```

Before deleting, check each of its three tests has a counterpart in Step 1's set:
legacy SAM2 back-fill → `test_from_dict_accepts_a_legacy_sam2_record`; SAM3 round trip
→ `test_from_dict_accepts_a_legacy_sam3_record`; `sam2_variant` sync for legacy readers
→ **intentionally dropped**, because `to_dict` no longer writes that key (the one-way
step stated in this task's preamble). If any other test is uncovered, port it rather
than dropping it.

- [ ] **Step 7: Run the full DetectKit test surface**

```bash
python -m pytest tests/test_detectkit_models.py tests/test_detectkit_overlay_providers.py \
  tests/test_detectkit_overlay_layer.py tests/test_detectkit_overlay_golden.py \
  tests/test_detectkit_project.py tests/test_detectkit_review_escalations_dialog.py \
  tests/test_detectkit_sam2_escalation_wiring.py tests/test_detectkit_show_image_multi_level.py \
  tests/test_sam2_escalation.py tests/test_semantic_escalation_job.py \
  tests/test_obbsource_reviewed.py tests/test_detectkit_staged_escalation_overlay.py -q
```

Expected: PASS. The last four are the files that reference the renamed attribute
outside the obvious DetectKit set — they are easy to miss and each fails at *collection*
(module-level imports), which aborts a whole file rather than one test. Any remaining
failure is an unrenamed call site — fix it, do not skip the test.

- [ ] **Step 8: Commit**

```bash
make format && make lint-moderate
git add -A src/hydra_suite/detectkit tests
git commit -m "refactor(detectkit): PendingEscalation -> StagedReview, producer is provenance only"
```

---

### Task 5: `decisions.json` and the revert snapshot

Per-frame decisions live in the **staging directory, not the project JSON**: a
10k-frame source would otherwise add 10k entries to every project save, and the staging
directory is already the object whose lifetime matches the review's.

Because accepts apply immediately (§4 of the spec), undo cannot be "discard pending
decisions". It is a snapshot restore instead. **The snapshot must capture more than
`labels/`**: a promoting accept (Task 7) also rewrites `source.level` and can extend
`classes.txt`, so restoring labels alone would leave the source claiming a level its
labels no longer have.

**Files:**
- Create: `src/hydra_suite/detectkit/jobs/staged_review.py`
- Test: `tests/test_detectkit_staged_review.py`

**Interfaces:**
- Consumes: `StagedReview`, `OBBSource` (Task 4).
- Produces:
  - `Decision` string constants: `ACCEPTED_OVERWRITE = "accepted_overwrite"`,
    `ACCEPTED_ADD_NEW = "accepted_add_new"`, `REJECTED = "rejected"`
  - `read_decisions(staged_root) -> dict[str, str]` (relative label path -> decision)
  - `write_decisions(staged_root, decisions) -> None`
  - `staged_frames(staged_root) -> list[str]` (sorted relative label paths, POSIX)
  - `review_key_for_image(source_path, image_path) -> str | None` — the single
    definition of a frame's review key
  - `review_progress(staged_root) -> tuple[int, int]` (decided, total)
  - `ensure_snapshot(source, staged_root) -> None` (idempotent; first call only)
  - `revert_review(source, staged_root) -> None`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_detectkit_staged_review.py`:

```python
import json
from pathlib import Path

import pytest

from hydra_suite.detectkit.gui.models import OBBSource, StagedReview
from hydra_suite.detectkit.jobs import staged_review as sr


def _source(tmp_path, labels: dict[str, str], level="obb", classes="object\n"):
    root = tmp_path / "src"
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir(parents=True)
    for rel, text in labels.items():
        p = root / "labels" / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        (root / "images" / rel).with_suffix(".png").parent.mkdir(
            parents=True, exist_ok=True
        )
    (root / "classes.txt").write_text(classes)
    return OBBSource(path=str(root), name="src", level=level)


def _staging(tmp_path, labels: dict[str, str], classes="object\n"):
    # MUST live under artifacts/pending_escalations/: with project_dir=None,
    # `_is_safe_to_delete` (sam2_escalation.py:35) accepts a path only if its
    # parent is "pending_escalations" and its grandparent is "artifacts".
    # A staging dir anywhere else is silently NOT deleted, and
    # test_finish_review_removes_staging would fail with nothing to show why.
    root = tmp_path / "artifacts" / "pending_escalations" / "staging"
    (root / "labels").mkdir(parents=True)
    for rel, text in labels.items():
        p = root / "labels" / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    (root / "classes.txt").write_text(classes)
    return root


def test_decisions_round_trip(tmp_path):
    staged = _staging(tmp_path, {"a.txt": "", "b.txt": ""})

    sr.write_decisions(staged, {"a.txt": sr.ACCEPTED_ADD_NEW})

    assert sr.read_decisions(staged) == {"a.txt": sr.ACCEPTED_ADD_NEW}


def test_decisions_read_as_empty_when_absent(tmp_path):
    assert sr.read_decisions(_staging(tmp_path, {})) == {}


def test_staged_frames_are_sorted_posix_relative_paths(tmp_path):
    staged = _staging(tmp_path, {"b.txt": "", "sub/a.txt": ""})

    assert sr.staged_frames(staged) == ["b.txt", "sub/a.txt"]


def test_review_progress_counts_decided_over_total(tmp_path):
    staged = _staging(tmp_path, {"a.txt": "", "b.txt": "", "c.txt": ""})
    sr.write_decisions(staged, {"a.txt": sr.REJECTED})

    assert sr.review_progress(staged) == (1, 3)


def test_snapshot_captures_labels_level_and_classes(tmp_path):
    source = _source(tmp_path, {"a.txt": "0 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2\n"})
    staged = _staging(tmp_path, {"a.txt": ""})

    sr.ensure_snapshot(source, staged)

    assert (staged / "labels_before" / "a.txt").read_text().startswith("0 ")
    state = json.loads((staged / "state_before.json").read_text())
    assert state["level"] == "obb"
    assert state["classes_txt"] == "object\n"


def test_snapshot_is_taken_once_and_never_overwritten(tmp_path):
    source = _source(tmp_path, {"a.txt": "original\n"})
    staged = _staging(tmp_path, {"a.txt": ""})

    sr.ensure_snapshot(source, staged)
    (Path(source.path) / "labels" / "a.txt").write_text("changed\n")
    sr.ensure_snapshot(source, staged)

    assert (staged / "labels_before" / "a.txt").read_text() == "original\n"


def test_revert_restores_labels_level_and_classes_and_clears_decisions(tmp_path):
    source = _source(tmp_path, {"a.txt": "original\n"}, level="obb")
    staged = _staging(tmp_path, {"a.txt": ""})
    sr.ensure_snapshot(source, staged)

    (Path(source.path) / "labels" / "a.txt").write_text("accepted\n")
    (Path(source.path) / "labels" / "new.txt").write_text("appeared\n")
    (Path(source.path) / "classes.txt").write_text("object\nant\n")
    source.level = "polygon"
    sr.write_decisions(staged, {"a.txt": sr.ACCEPTED_OVERWRITE})

    sr.revert_review(source, staged)

    assert (Path(source.path) / "labels" / "a.txt").read_text() == "original\n"
    assert not (Path(source.path) / "labels" / "new.txt").exists()
    assert (Path(source.path) / "classes.txt").read_text() == "object\n"
    assert source.level == "obb"
    assert sr.read_decisions(staged) == {}


def test_revert_clears_classes_appended_to_a_source_that_had_none(tmp_path):
    """Restoring the class list must mean restoring it, including to empty."""
    source = _source(tmp_path, {"a.txt": "original\n"}, classes="")
    staged = _staging(tmp_path, {"a.txt": ""})
    sr.ensure_snapshot(source, staged)
    (Path(source.path) / "classes.txt").write_text("larva\n")

    sr.revert_review(source, staged)

    assert (Path(source.path) / "classes.txt").read_text() == ""


def test_revert_without_a_snapshot_is_refused(tmp_path):
    source = _source(tmp_path, {"a.txt": "original\n"})
    staged = _staging(tmp_path, {"a.txt": ""})

    with pytest.raises(RuntimeError, match="no snapshot"):
        sr.revert_review(source, staged)
```

- [ ] **Step 2: Run to verify they fail**

```bash
python -m pytest tests/test_detectkit_staged_review.py -q
```

Expected: FAIL — `ModuleNotFoundError: ...jobs.staged_review`.

- [ ] **Step 3: Implement the module**

Create `src/hydra_suite/detectkit/jobs/staged_review.py`:

```python
"""Frame-granular review of a StagedReview, applied immediately.

One accept path for every producer. A producer's only job is to fill a
staging directory's ``labels/`` + ``classes.txt`` + ``run.json``; everything
here is producer-agnostic, which is the whole point of the StagedReview
refactor.

Decisions live in the STAGING directory, not the project JSON: a 10k-frame
source would otherwise add 10k entries to every project save, and the
staging directory is already the object whose lifetime matches the
review's.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

from hydra_suite.detectkit.gui.models import OBBSource

logger = logging.getLogger(__name__)

ACCEPTED_OVERWRITE = "accepted_overwrite"
ACCEPTED_ADD_NEW = "accepted_add_new"
REJECTED = "rejected"

DECISIONS_FILE = "decisions.json"
SNAPSHOT_DIR = "labels_before"
SNAPSHOT_STATE = "state_before.json"


def read_decisions(staged_root: str | Path) -> dict[str, str]:
    """Per-frame outcomes recorded so far. Absent or corrupt reads as empty."""
    path = Path(staged_root) / DECISIONS_FILE
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(k): str(v) for k, v in data.items()}


def write_decisions(staged_root: str | Path, decisions: dict[str, str]) -> None:
    """Persist per-frame outcomes, overwriting the file."""
    path = Path(staged_root) / DECISIONS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(decisions, indent=2, sort_keys=True), encoding="utf-8")


def staged_frames(staged_root: str | Path) -> list[str]:
    """Every staged frame, as sorted POSIX paths relative to ``labels/``.

    POSIX and relative because they are the review's keys: they index
    decisions.json, which round-trips through JSON on every platform, and
    they mirror the source's images/ tree exactly (the same key
    `_origin_image_for` and `find_staged_label_for_image` already rely on).
    """
    labels = Path(staged_root) / "labels"
    if not labels.is_dir():
        return []
    return sorted(p.relative_to(labels).as_posix() for p in labels.rglob("*.txt"))


def review_key_for_image(source_path: str | Path, image_path: str | Path) -> str | None:
    """The review key for a frame: its path under images/, suffixed .txt.

    THE one definition. The staged label mirrors the image's images-relative
    path, so this string indexes decisions.json, names the staged label, and
    names the source label -- all three. Computing it anywhere else from a
    label path instead would drift the moment
    `find_staged_label_for_image`'s stem or recursive fallback fires.

    Returns None when the image is not under the source's images/ at all,
    which the callers treat as "this frame is not part of the review".
    """
    try:
        rel = Path(image_path).relative_to(Path(source_path) / "images")
    except ValueError:
        return None
    return rel.with_suffix(".txt").as_posix()


def review_progress(staged_root: str | Path) -> tuple[int, int]:
    """(decided, total) for the progress counter."""
    frames = staged_frames(staged_root)
    decided = read_decisions(staged_root)
    return sum(1 for f in frames if f in decided), len(frames)


def ensure_snapshot(source: OBBSource, staged_root: str | Path) -> None:
    """Snapshot the source's pre-review state, once, before the first accept.

    Captures ``labels/`` AND the two other things an accept can change: the
    source's ``level`` (a promoting accept rewrites it) and ``classes.txt``
    (accepting staged classes can extend it). Restoring labels alone would
    leave a reverted source claiming a level its labels no longer have.

    Idempotent by the existence of the snapshot directory: the second and
    later accepts must NOT re-snapshot, or the snapshot would drift forward
    to whatever the last accept produced and revert would be a no-op.
    """
    root = Path(staged_root)
    snapshot = root / SNAPSHOT_DIR
    if snapshot.exists():
        return

    source_root = Path(source.path)
    source_labels = source_root / "labels"
    if source_labels.is_dir():
        shutil.copytree(source_labels, snapshot)
    else:
        snapshot.mkdir(parents=True)

    classes = source_root / "classes.txt"
    (root / SNAPSHOT_STATE).write_text(
        json.dumps(
            {
                "level": source.level,
                "classes_txt": classes.read_text() if classes.is_file() else "",
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def revert_review(source: OBBSource, staged_root: str | Path) -> None:
    """Restore the source to its pre-review state and clear every decision.

    Available only while the review is open: completing a review deletes the
    staging directory, and the snapshot with it. The review bar says so.
    """
    root = Path(staged_root)
    snapshot = root / SNAPSHOT_DIR
    if not snapshot.is_dir():
        raise RuntimeError(
            "There is no snapshot to revert to -- no frame of this review has "
            "been accepted yet, so the source is already in its original state."
        )

    source_root = Path(source.path)
    source_labels = source_root / "labels"
    # rmtree BEFORE copytree, not ignore_errors: a half-deleted labels/ makes
    # copytree raise FileExistsError and wedges the source. Raising here
    # leaves it untouched instead.
    if source_labels.exists():
        shutil.rmtree(source_labels)
    shutil.copytree(snapshot, source_labels)

    try:
        state = json.loads((root / SNAPSHOT_STATE).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        state = {}
    if state.get("level"):
        source.level = str(state["level"])
    # `in`, not truthiness: a source whose classes.txt was absent or empty
    # before the review must NOT keep the names resolve_staged_class_ids
    # appended. "Restore the class list" has to mean restore, including to
    # nothing.
    if "classes_txt" in state:
        (source_root / "classes.txt").write_text(str(state["classes_txt"]))

    write_decisions(root, {})
```

- [ ] **Step 4: Run the tests**

```bash
python -m pytest tests/test_detectkit_staged_review.py -q
```

Expected: PASS (8).

- [ ] **Step 5: Commit**

```bash
make format && make lint-moderate
git add src/hydra_suite/detectkit/jobs/staged_review.py tests/test_detectkit_staged_review.py
git commit -m "feat(detectkit): add decisions.json and the pre-review snapshot"
```

---

### Task 6: Staged-to-source class-id resolution

Frame-granular `ADD_NEW` mixes two class-id spaces in one file for the first time.
Staged ids index the **staging dir's** `classes.txt` (SAM3's ids are its prompt's, all
class 0); source ids index the **source's**. SAM2's wholesale overwrite dodged this by
copying `classes.txt` over the source's; that is no longer available when only some
frames are accepted.

**Rule:** match by name. A staged class name already in the source's `classes.txt` maps
to that id; a name not present is **appended** to the source's `classes.txt` and takes
the new id. Appending never renumbers an existing id, so labels already on disk stay
valid — that is why appending is safe and re-sorting would not be.

**Files:**
- Modify: `src/hydra_suite/detectkit/jobs/staged_review.py`
- Test: `tests/test_detectkit_staged_review.py`

**Interfaces:**
- Consumes: nothing new. (It deliberately does NOT use
  `training.class_mapping.read_classes_txt` — that helper raises when `classes.txt` is
  absent, and both a source and a staging dir may legitimately lack one here. The
  module's own `_read_names` degrades to `[]` instead.)
- Produces: `resolve_staged_class_ids(source: OBBSource, staged_root) -> dict[int, int]`
  — staged class id -> source class id, extending the source's `classes.txt` in place
  when a staged name is new.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_detectkit_staged_review.py`:

```python
def test_matching_class_names_map_by_name(tmp_path):
    source = _source(tmp_path, {}, classes="ant\nbeetle\n")
    staged = _staging(tmp_path, {}, classes="beetle\n")

    assert sr.resolve_staged_class_ids(source, staged) == {0: 1}


def test_an_unknown_staged_class_is_appended_to_the_source(tmp_path):
    source = _source(tmp_path, {}, classes="ant\n")
    staged = _staging(tmp_path, {}, classes="ant\nlarva\n")

    mapping = sr.resolve_staged_class_ids(source, staged)

    assert mapping == {0: 0, 1: 1}
    assert (Path(source.path) / "classes.txt").read_text() == "ant\nlarva\n"


def test_appending_never_renumbers_an_existing_class(tmp_path):
    source = _source(tmp_path, {}, classes="ant\nbeetle\n")
    staged = _staging(tmp_path, {}, classes="larva\n")

    mapping = sr.resolve_staged_class_ids(source, staged)

    assert mapping == {0: 2}
    assert (Path(source.path) / "classes.txt").read_text().splitlines()[:2] == [
        "ant", "beetle"
    ]


def test_resolution_is_idempotent(tmp_path):
    source = _source(tmp_path, {}, classes="ant\n")
    staged = _staging(tmp_path, {}, classes="larva\n")

    first = sr.resolve_staged_class_ids(source, staged)
    second = sr.resolve_staged_class_ids(source, staged)

    assert first == second
    assert (Path(source.path) / "classes.txt").read_text() == "ant\nlarva\n"
```

- [ ] **Step 2: Run to verify they fail**

```bash
python -m pytest tests/test_detectkit_staged_review.py -k class -q
```

Expected: FAIL — `AttributeError: module ... has no attribute 'resolve_staged_class_ids'`.

- [ ] **Step 3: Implement**

Append to `src/hydra_suite/detectkit/jobs/staged_review.py`:

```python
def _read_names(path: Path) -> list[str]:
    try:
        return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    except OSError:
        return []


def resolve_staged_class_ids(
    source: OBBSource,
    staged_root: str | Path,
) -> dict[int, int]:
    """Map staged class ids onto the source's, extending it when needed.

    Frame-granular ADD_NEW puts two class-id spaces in one label file for
    the first time: staged ids index the STAGING dir's classes.txt (SAM3's
    are its prompt's, all class 0), source ids index the source's. The old
    wholesale accept dodged this by copying classes.txt over the source's,
    which is not available when only some frames are accepted.

    Matching is BY NAME. A staged name the source does not have is APPENDED
    to the source's classes.txt and takes the new id. Appending -- rather
    than merging and re-sorting -- is what keeps every label already on disk
    valid: no existing id is ever renumbered.

    Idempotent: running it twice appends nothing the second time.
    """
    source_root = Path(source.path)
    classes_path = source_root / "classes.txt"
    source_names = _read_names(classes_path)
    staged_names = _read_names(Path(staged_root) / "classes.txt") or ["object"]

    mapping: dict[int, int] = {}
    appended = False
    for staged_id, name in enumerate(staged_names):
        if name not in source_names:
            source_names.append(name)
            appended = True
        mapping[staged_id] = source_names.index(name)

    if appended:
        classes_path.write_text("\n".join(source_names) + "\n", encoding="utf-8")
    return mapping
```

- [ ] **Step 4: Run the tests**

```bash
python -m pytest tests/test_detectkit_staged_review.py -q
```

Expected: PASS (12).

- [ ] **Step 5: Commit**

```bash
make format && make lint-moderate
git add src/hydra_suite/detectkit/jobs/staged_review.py tests/test_detectkit_staged_review.py
git commit -m "feat(detectkit): map staged class ids onto the source by name"
```

---

### Task 7: `accept_frame` / `reject_frame` — the producer-agnostic accept

This is the heart of the design. Four operations keyed by the frame's relative path,
applied **immediately** so the result appears on the ground-truth layer as the user
works.

**`merge_records` refuses upward derivation, so `accept_frame` must lift first.**
`derive_down` raises on any record below the target level, and two ordinary cases hit
that: a promoting accept (existing OBB records, target POLYGON) and a staged result
below the source's level (staged OBB into a POLYGON source). Neither is an error — an
OBB quad *is* a valid 4-point polygon, and `_polygon_points` encodes it as one by
repeating the final vertex. The lift is therefore a **re-tag**, done explicitly by this
module before `merge_records` is called, with the points untouched. The primitive stays
strict, which is what keeps a genuine "the model never produced this information" case
from being silently invented. (The spec's §3 says staged records below the source's
level are handled by `derive_down`; that is backwards — downward derivation is what
`derive_down` does, and this case is upward. Task 15 amends the sentence.)

Two further decisions the spec leaves implicit, settled here:

1. **"Existing labels are kept verbatim" is meant literally.** Under `ADD_NEW` without
   promotion, the existing file's lines are copied through **byte for byte** and only
   the surviving staged records are appended as newly formatted lines. Re-encoding
   existing records through `write_label_file` would round-trip
   denormalise→normalise→`%.6f` and shift bytes on labels the user never touched.
2. **Promotion is the one case that re-encodes everything.** Lifting an OBB source's
   quads to 4-point polygons necessarily rewrites every line (`_polygon_points` adds the
   repeated vertex). Byte drift there is intended, and the snapshot from Task 5 is what
   makes it reversible.

**Files:**
- Modify: `src/hydra_suite/detectkit/jobs/staged_review.py`
- Modify: `src/hydra_suite/detectkit/gui/dialogs/review_escalations_dialog.py`
  (accept-all through the new path, so the app keeps working)
- Test: `tests/test_detectkit_staged_review.py`

**Interfaces:**
- Consumes: `merge_records`, `MergeMode` (Task 3); `read_label_file`,
  `write_label_file` (Task 2); `derive_down` (`data/al/escalation.py`);
  `ensure_snapshot`, `resolve_staged_class_ids`, decision constants (Tasks 5-6).
- Produces:
  - `_lift(records, target) -> list[LabelRecord]` — re-tags a record's level without
    moving a point (the encoder does the rest)
  - `accept_frame(source, rel, *, mode: MergeMode, iou_threshold: float = 0.5) -> None`
  - `reject_frame(source, rel) -> None`
  - `accept_all(source, *, mode, iou_threshold=0.5) -> int` (undecided frames only)
  - `reject_all(source) -> int`
  - `is_complete(source) -> bool`
  - `finish_review(source, project_dir=None) -> None` (removes the staging dir, clears
    `source.staged_review`)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_detectkit_staged_review.py`:

```python
import numpy as np
from PIL import Image

from hydra_suite.data.al.merge import MergeMode
from hydra_suite.data.al.labels import read_label_file


def _image(source: OBBSource, rel_stem: str, size=(100, 200)):
    """A real PNG, because accept must read the frame size off the image."""
    path = Path(source.path) / "images" / f"{rel_stem}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros((size[0], size[1], 3), dtype=np.uint8)).save(path)
    return path


def _obb_line(x1, y1, x2, y2, w=200, h=100, class_id=0):
    xs = [x1, x2, x2, x1]
    ys = [y1, y1, y2, y2]
    coords = " ".join(f"{x / w:.6f} {y / h:.6f}" for x, y in zip(xs, ys))
    return f"{class_id} {coords}\n"


def _wired(tmp_path, source_labels, staged_labels, level="obb",
           target_level="obb", producer="sam2"):
    source = _source(tmp_path, source_labels, level=level)
    staged = _staging(tmp_path, staged_labels)
    for rel in set(source_labels) | set(staged_labels):
        _image(source, rel[:-4])
    source.staged_review = StagedReview(
        staged_path=str(staged), target_level=target_level, producer=producer
    )
    return source, staged


def test_accept_overwrite_replaces_the_frames_labels(tmp_path):
    source, staged = _wired(
        tmp_path, {"a.txt": _obb_line(0, 0, 20, 20)}, {"a.txt": _obb_line(50, 50, 70, 70)}
    )

    sr.accept_frame(source, "a.txt", mode=MergeMode.OVERWRITE)

    out = read_label_file(Path(source.path) / "labels" / "a.txt", (100, 200))
    assert len(out) == 1
    assert out[0].points[:, 0].min() > 40
    assert sr.read_decisions(staged)["a.txt"] == sr.ACCEPTED_OVERWRITE


def test_accept_add_new_appends_only_the_non_overlapping_staged(tmp_path):
    source, staged = _wired(
        tmp_path,
        {"a.txt": _obb_line(0, 0, 20, 20)},
        {"a.txt": _obb_line(0, 0, 20, 20) + _obb_line(60, 60, 80, 80)},
    )

    sr.accept_frame(source, "a.txt", mode=MergeMode.ADD_NEW, iou_threshold=0.5)

    out = read_label_file(Path(source.path) / "labels" / "a.txt", (100, 200))
    assert len(out) == 2
    assert sr.read_decisions(staged)["a.txt"] == sr.ACCEPTED_ADD_NEW


def test_add_new_keeps_the_existing_lines_byte_for_byte(tmp_path):
    original = _obb_line(0, 0, 20, 20)
    source, _ = _wired(tmp_path, {"a.txt": original}, {"a.txt": _obb_line(60, 60, 80, 80)})

    sr.accept_frame(source, "a.txt", mode=MergeMode.ADD_NEW)

    text = (Path(source.path) / "labels" / "a.txt").read_text()
    assert text.startswith(original)


def test_reject_changes_nothing_but_records_the_decision(tmp_path):
    original = _obb_line(0, 0, 20, 20)
    source, staged = _wired(tmp_path, {"a.txt": original}, {"a.txt": _obb_line(60, 60, 80, 80)})

    sr.reject_frame(source, "a.txt")

    assert (Path(source.path) / "labels" / "a.txt").read_text() == original
    assert sr.read_decisions(staged)["a.txt"] == sr.REJECTED


def test_accepting_a_frame_with_no_existing_label_creates_one(tmp_path):
    source, _ = _wired(tmp_path, {}, {"a.txt": _obb_line(60, 60, 80, 80)})

    sr.accept_frame(source, "a.txt", mode=MergeMode.ADD_NEW)

    assert (Path(source.path) / "labels" / "a.txt").is_file()


def test_the_first_accept_snapshots_and_later_ones_do_not(tmp_path):
    source, staged = _wired(
        tmp_path,
        {"a.txt": "0 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2\n", "b.txt": "0 0.3 0.3 0.4 0.3 0.4 0.4 0.3 0.4\n"},
        {"a.txt": _obb_line(60, 60, 80, 80), "b.txt": _obb_line(60, 60, 80, 80)},
    )
    before_a = (Path(source.path) / "labels" / "a.txt").read_text()

    sr.accept_frame(source, "a.txt", mode=MergeMode.OVERWRITE)
    sr.accept_frame(source, "b.txt", mode=MergeMode.OVERWRITE)

    assert (staged / "labels_before" / "a.txt").read_text() == before_a


def test_accepting_polygons_into_an_obb_source_promotes_it(tmp_path):
    poly = "0 " + " ".join(
        f"{x / 200:.6f} {y / 100:.6f}"
        for x, y in [(60, 60), (80, 62), (85, 80), (70, 88), (58, 78)]
    ) + "\n"
    source, _ = _wired(
        tmp_path,
        {"a.txt": _obb_line(0, 0, 20, 20)},
        {"a.txt": poly},
        level="obb",
        target_level="polygon",
    )

    sr.accept_frame(source, "a.txt", mode=MergeMode.ADD_NEW)

    assert source.level == "polygon"
    out = read_label_file(Path(source.path) / "labels" / "a.txt", (100, 200))
    assert all(r.level.name == "POLYGON" for r in out)


def test_a_promoted_quad_does_not_read_back_as_an_obb(tmp_path):
    poly = "0 " + " ".join(
        f"{x / 200:.6f} {y / 100:.6f}"
        for x, y in [(60, 60), (80, 62), (85, 80), (70, 88), (58, 78)]
    ) + "\n"
    source, _ = _wired(
        tmp_path, {"a.txt": _obb_line(0, 0, 20, 20)}, {"a.txt": poly},
        level="obb", target_level="polygon",
    )

    sr.accept_frame(source, "a.txt", mode=MergeMode.ADD_NEW)

    lifted = read_label_file(Path(source.path) / "labels" / "a.txt", (100, 200))[0]
    assert lifted.points.shape == (5, 2)


def test_promotion_does_not_drift_the_lifted_coordinates(tmp_path):
    poly = "0 " + " ".join(
        f"{x / 200:.6f} {y / 100:.6f}"
        for x, y in [(60, 60), (80, 62), (85, 80), (70, 88), (58, 78)]
    ) + "\n"
    source, _ = _wired(
        tmp_path, {"a.txt": _obb_line(10, 10, 30, 30)}, {"a.txt": poly},
        level="obb", target_level="polygon",
    )
    before = read_label_file(Path(source.path) / "labels" / "a.txt", (100, 200))[0]

    sr.accept_frame(source, "a.txt", mode=MergeMode.ADD_NEW)

    after = read_label_file(Path(source.path) / "labels" / "a.txt", (100, 200))[0]
    np.testing.assert_allclose(after.points[:4], before.points, atol=0.05)


def test_staged_below_the_source_level_is_lifted_to_it(tmp_path):
    source, _ = _wired(
        tmp_path,
        {"a.txt": "0 " + " ".join(f"{v:.6f}" for v in [0.1, 0.1, 0.2, 0.1, 0.2, 0.2, 0.15, 0.25, 0.1, 0.2]) + "\n"},
        {"a.txt": _obb_line(120, 60, 160, 90)},
        level="polygon",
        target_level="obb",
    )

    sr.accept_frame(source, "a.txt", mode=MergeMode.ADD_NEW)

    assert source.level == "polygon"
    out = read_label_file(Path(source.path) / "labels" / "a.txt", (100, 200))
    assert len(out) == 2


def test_accept_all_only_touches_undecided_frames(tmp_path):
    source, staged = _wired(
        tmp_path,
        {"a.txt": _obb_line(0, 0, 20, 20), "b.txt": _obb_line(0, 0, 20, 20)},
        {"a.txt": _obb_line(60, 60, 80, 80), "b.txt": _obb_line(60, 60, 80, 80)},
    )
    sr.reject_frame(source, "a.txt")

    changed = sr.accept_all(source, mode=MergeMode.OVERWRITE)

    assert changed == 1
    assert sr.read_decisions(staged)["a.txt"] == sr.REJECTED
    assert sr.read_decisions(staged)["b.txt"] == sr.ACCEPTED_OVERWRITE


def test_a_review_is_complete_when_every_frame_is_decided(tmp_path):
    source, _ = _wired(
        tmp_path, {}, {"a.txt": _obb_line(60, 60, 80, 80), "b.txt": _obb_line(60, 60, 80, 80)}
    )

    sr.reject_frame(source, "a.txt")
    assert not sr.is_complete(source)

    sr.reject_frame(source, "b.txt")
    assert sr.is_complete(source)


def test_finish_review_removes_staging_and_clears_the_source(tmp_path):
    source, staged = _wired(tmp_path, {}, {"a.txt": _obb_line(60, 60, 80, 80)})
    sr.reject_frame(source, "a.txt")

    sr.finish_review(source, project_dir=None)

    assert source.staged_review is None
    assert not staged.exists()


def test_rejecting_everything_leaves_the_source_reviewed(tmp_path):
    """Nothing machine-derived landed, so nothing was un-confirmed."""
    source, _ = _wired(tmp_path, {}, {"a.txt": _obb_line(60, 60, 80, 80)})
    source.reviewed = True
    sr.reject_all(source)

    sr.finish_review(source, project_dir=None)

    assert source.reviewed is True


def test_accepting_any_frame_marks_the_source_unreviewed(tmp_path):
    source, _ = _wired(
        tmp_path, {"a.txt": _obb_line(0, 0, 20, 20)}, {"a.txt": _obb_line(60, 60, 80, 80)}
    )
    source.reviewed = True
    sr.accept_frame(source, "a.txt", mode=MergeMode.ADD_NEW)

    sr.finish_review(source, project_dir=None)

    assert source.reviewed is False


def test_revert_after_a_mixed_review_restores_byte_identical_labels(tmp_path):
    """Spec test 4: accept a mix of frames in BOTH modes, then revert."""
    source, staged = _wired(
        tmp_path,
        {"a.txt": _obb_line(0, 0, 20, 20), "b.txt": _obb_line(5, 5, 25, 25),
         "c.txt": _obb_line(9, 9, 29, 29)},
        {"a.txt": _obb_line(60, 60, 80, 80), "b.txt": _obb_line(70, 70, 90, 90),
         "c.txt": _obb_line(80, 80, 95, 95)},
    )
    before = {
        p.name: p.read_bytes() for p in (Path(source.path) / "labels").rglob("*.txt")
    }

    sr.accept_frame(source, "a.txt", mode=MergeMode.OVERWRITE)
    sr.accept_frame(source, "b.txt", mode=MergeMode.ADD_NEW)
    sr.reject_frame(source, "c.txt")
    sr.revert_review(source, staged)

    after = {
        p.name: p.read_bytes() for p in (Path(source.path) / "labels").rglob("*.txt")
    }
    assert after == before


def test_revert_after_a_promoting_accept_restores_the_level_too(tmp_path):
    poly = "0 " + " ".join(
        f"{x / 200:.6f} {y / 100:.6f}"
        for x, y in [(60, 60), (80, 62), (85, 80), (70, 88), (58, 78)]
    ) + "\n"
    source, staged = _wired(
        tmp_path, {"a.txt": _obb_line(0, 0, 20, 20)}, {"a.txt": poly},
        level="obb", target_level="polygon",
    )
    before = (Path(source.path) / "labels" / "a.txt").read_bytes()

    sr.accept_frame(source, "a.txt", mode=MergeMode.ADD_NEW)
    sr.revert_review(source, staged)

    assert source.level == "obb"
    assert (Path(source.path) / "labels" / "a.txt").read_bytes() == before


@pytest.mark.parametrize("producer", ["sam2", "sam3", "inference"])
def test_accept_is_producer_agnostic(tmp_path, producer):
    """Spec test 5: identical staged content -> identical outcome, always.

    This is the test that fails if `producer` ever becomes load-bearing again.
    """
    source, _ = _wired(
        tmp_path,
        {"a.txt": _obb_line(0, 0, 20, 20)},
        {"a.txt": _obb_line(60, 60, 80, 80)},
        producer=producer,
    )

    sr.accept_frame(source, "a.txt", mode=MergeMode.ADD_NEW)

    assert (Path(source.path) / "labels" / "a.txt").read_text() == (
        _obb_line(0, 0, 20, 20) + _obb_line(60, 60, 80, 80)
    )
```

- [ ] **Step 2: Run to verify they fail**

```bash
python -m pytest tests/test_detectkit_staged_review.py -q
```

Expected: FAIL — `AttributeError: ... has no attribute 'accept_frame'`.

- [ ] **Step 3: Implement the accept path**

Append to `src/hydra_suite/detectkit/jobs/staged_review.py` (adding the imports at the
top of the file):

```python
import cv2

from hydra_suite.data.al.escalation import LabelRecord, derive_down
from hydra_suite.data.al.labels import read_label_file, write_label_file
from hydra_suite.data.al.merge import MergeMode, merge_records
from hydra_suite.detectkit.gui.constants import IMG_EXTS
from hydra_suite.utils.geometry_levels import GeometryLevel

DEFAULT_MERGE_IOU = 0.5


def _lift(records: list[LabelRecord], target: GeometryLevel) -> list[LabelRecord]:
    """Re-tag records to `target` WITHOUT moving a point.

    An OBB quad is a valid 4-point polygon; `_polygon_points` encodes it as
    one by repeating the final vertex, so lifting is purely a change of
    declared level. `derive_down` cannot express this -- it refuses upward
    derivation, correctly, because for a genuine level gap upward derivation
    would invent information. Here there is no gap to invent across: the
    points are already there.

    Records at or above `target` are returned unchanged.
    """
    out: list[LabelRecord] = []
    for rec in records:
        if rec.level >= target:
            out.append(rec)
            continue
        out.append(
            LabelRecord(
                class_id=rec.class_id,
                confidence=rec.confidence,
                points=rec.points,
                level=target,
            )
        )
    return out


def _review_of(source: OBBSource):
    review = source.staged_review
    if review is None:
        raise ValueError(f"Source '{source.name}' has no staged review.")
    return review


def _level_of(value: str, fallback: GeometryLevel) -> GeometryLevel:
    """Parse a level string from project JSON, degrading rather than raising.

    Both `OBBSource.level` and `StagedReview.target_level` are unvalidated
    strings loaded from disk, exactly as `resolve_pending_level` in the
    overlay providers already treats them.
    """
    try:
        return GeometryLevel.from_str(value)
    except ValueError:
        logger.warning("Unknown geometry level %r; treating as %s", value, fallback.label)
        return fallback


def _image_for(source: OBBSource, rel: str) -> Path | None:
    """The source image a staged label at *rel* came from.

    The staged label's relative path mirrors the image's under `images/` --
    that is the review's key -- so this is a direct sibling lookup. The
    extension loop matches `_origin_image_for` in semantic_escalation,
    including its case handling: on the case-sensitive Linux lab shares this
    is deployed to, trying only the lowercase extension silently orphans
    `a.Jpg`.
    """
    stem = Path(source.path) / "images" / Path(rel).with_suffix("")
    for ext in sorted(IMG_EXTS):
        for candidate in (
            stem.with_name(stem.name + ext),
            stem.with_name(stem.name + ext.upper()),
        ):
            if candidate.is_file():
                return candidate
    return None


def _frame_size(source: OBBSource, rel: str) -> tuple[int, int]:
    """(height, width) of the frame, read from the image on disk."""
    image = _image_for(source, rel)
    if image is None:
        raise RuntimeError(
            f"No image found for staged frame '{rel}' in source '{source.name}'; "
            "the staged label has nothing to apply to."
        )
    frame = cv2.imread(str(image))
    if frame is None:
        raise RuntimeError(f"Could not read image {image} for staged frame '{rel}'.")
    return int(frame.shape[0]), int(frame.shape[1])


def _record_decision(staged_root: Path, rel: str, decision: str) -> None:
    decisions = read_decisions(staged_root)
    decisions[rel] = decision
    write_decisions(staged_root, decisions)


def accept_frame(
    source: OBBSource,
    rel: str,
    *,
    mode: MergeMode,
    iou_threshold: float = DEFAULT_MERGE_IOU,
) -> None:
    """Apply one staged frame to the source, immediately.

    Immediately rather than into a pending set, because the entire point of
    reviewing on the frame is seeing the result land on the ground-truth
    layer as you work. `merge_records`' "a merge can only add" invariant is
    what makes that safe; the Task-5 snapshot is what makes it reversible.

    LEVEL PROMOTION. If the staged level is ABOVE the source's, the source
    is promoted: `source.level` is set and its existing labels are lifted
    (an OBB quad becomes a 4-point polygon, which `_polygon_points` encodes
    with a repeated final vertex precisely so it never reads back as an
    OBB). Promotion is a property of the SOURCE, so the first promoting
    accept sets the level and the rest of the review proceeds at the new
    one. If the staged level is BELOW the source's, the staged records are
    derived down instead and the source's level is untouched.

    VERBATIM EXISTING LINES. Under ADD_NEW without promotion the existing
    file's lines are copied through byte for byte and only the surviving
    staged records are appended. Re-encoding them through
    `write_label_file` would round-trip denormalise -> normalise -> %.6f and
    shift bytes on labels the user never touched. Promotion is the one case
    that necessarily rewrites every line, and the snapshot covers it.
    """
    review = _review_of(source)
    staged_root = Path(review.staged_path)
    staged_label = staged_root / "labels" / rel
    if not staged_label.is_file():
        raise RuntimeError(
            f"Staged label '{rel}' is missing from {staged_root / 'labels'}; "
            "nothing was changed."
        )

    ensure_snapshot(source, staged_root)

    height, width = _frame_size(source, rel)
    source_level = _level_of(source.level, GeometryLevel.OBB)
    staged_level = _level_of(review.target_level, GeometryLevel.POLYGON)
    promoting = staged_level > source_level
    target_level = staged_level if promoting else source_level

    class_map = resolve_staged_class_ids(source, staged_root)
    staged_records = [
        LabelRecord(
            class_id=class_map.get(rec.class_id, rec.class_id),
            confidence=rec.confidence,
            points=rec.points,
            level=rec.level,
        )
        for rec in read_label_file(staged_label, (height, width))
    ]

    source_label = Path(source.path) / "labels" / rel
    source_label.parent.mkdir(parents=True, exist_ok=True)
    existing = read_label_file(source_label, (height, width))

    # Lift BEFORE merging: merge_records refuses upward derivation, and both
    # a promoting accept (existing below target) and a staged result below
    # the source's level (staged below target) would otherwise raise. The
    # lift moves no points; see `_lift`.
    existing = _lift(existing, target_level)
    staged_records = _lift(staged_records, target_level)

    if mode is MergeMode.OVERWRITE:
        write_label_file(
            source_label,
            derive_down(staged_records, target_level),
            (height, width),
            target_level,
        )
    else:
        merged = merge_records(
            existing,
            staged_records,
            mode=MergeMode.ADD_NEW,
            iou_threshold=iou_threshold,
            level=target_level,
        )
        survivors = merged[len(existing):]
        if promoting or not source_label.is_file():
            # Promotion rewrites every line by necessity; a frame with no
            # prior label has nothing verbatim to preserve.
            write_label_file(
                source_label,
                derive_down(merged, target_level),
                (height, width),
                target_level,
            )
        else:
            prior_text = source_label.read_text(encoding="utf-8")
            if prior_text and not prior_text.endswith("\n"):
                prior_text += "\n"
            with source_label.open("w", encoding="utf-8") as fp:
                fp.write(prior_text)
            # Append only. write_label_file truncates, so the survivors are
            # formatted into a temp buffer and appended.
            if survivors:
                buffer = staged_root / ".append.tmp"
                write_label_file(
                    buffer,
                    derive_down(survivors, target_level),
                    (height, width),
                    target_level,
                )
                with source_label.open("a", encoding="utf-8") as fp:
                    fp.write(buffer.read_text(encoding="utf-8"))
                buffer.unlink(missing_ok=True)

    if promoting:
        _promote_source(source, target_level, skip=rel, frame_size=(height, width))
        source.level = target_level.label

    _record_decision(staged_root, rel, ACCEPTED_OVERWRITE if mode is MergeMode.OVERWRITE else ACCEPTED_ADD_NEW)


def _promote_source(
    source: OBBSource,
    target_level: GeometryLevel,
    *,
    skip: str,
    frame_size: tuple[int, int],
) -> None:
    """Lift every OTHER label file in the source to `target_level`.

    Only the encoding changes: `_polygon_points` repeats an OBB quad's final
    vertex so it reads back as a polygon, without moving a coordinate. The
    frame that triggered promotion is skipped because `accept_frame` has
    already written it.

    Each file is re-read at ITS OWN frame size, not the triggering frame's:
    a source's images need not all be the same resolution, and normalising
    with the wrong size would silently move every point.
    """
    labels_dir = Path(source.path) / "labels"
    for path in sorted(labels_dir.rglob("*.txt")):
        rel = path.relative_to(labels_dir).as_posix()
        if rel == skip:
            continue
        try:
            size = _frame_size(source, rel)
        except RuntimeError:
            size = frame_size
        records = read_label_file(path, size)
        if not records:
            continue
        lifted = [
            LabelRecord(
                class_id=rec.class_id,
                confidence=rec.confidence,
                points=rec.points,
                level=target_level,
            )
            for rec in records
        ]
        write_label_file(path, lifted, size, target_level)


def reject_frame(source: OBBSource, rel: str) -> None:
    """Record that a staged frame is not wanted. Changes nothing on disk."""
    review = _review_of(source)
    _record_decision(Path(review.staged_path), rel, REJECTED)


def accept_all(
    source: OBBSource,
    *,
    mode: MergeMode,
    iou_threshold: float = DEFAULT_MERGE_IOU,
) -> int:
    """Accept every frame not yet decided. Returns how many were accepted."""
    review = _review_of(source)
    staged_root = Path(review.staged_path)
    decided = read_decisions(staged_root)
    count = 0
    for rel in staged_frames(staged_root):
        if rel in decided:
            continue
        accept_frame(source, rel, mode=mode, iou_threshold=iou_threshold)
        count += 1
    return count


def reject_all(source: OBBSource) -> int:
    """Reject every frame not yet decided. Returns how many were rejected."""
    review = _review_of(source)
    staged_root = Path(review.staged_path)
    decided = read_decisions(staged_root)
    count = 0
    for rel in staged_frames(staged_root):
        if rel in decided:
            continue
        reject_frame(source, rel)
        count += 1
    return count


def is_complete(source: OBBSource) -> bool:
    """True when every staged frame has a decision.

    getattr rather than a bare attribute access, matching the overlay
    providers' style: this is called on whatever the window currently
    considers "the source", which is not guaranteed to be an OBBSource.
    """
    review = getattr(source, "staged_review", None)
    if review is None:
        return False
    decided, total = review_progress(review.staged_path)
    return total > 0 and decided >= total


def finish_review(source: OBBSource, project_dir: str | Path | None = None) -> None:
    """Close the review: remove the staging dir and clear the source's field.

    This DELETES the snapshot along with the staging dir, so revert is only
    available while a review is open. The review bar says so before calling
    this.

    `reviewed` drops to False only if at least one frame was ACCEPTED --
    the same meaning it has everywhere else, "machine-derived and not yet
    human-confirmed". Flipping it unconditionally would exclude a source
    from training because the user rejected every proposal, which is the
    opposite of what rejecting everything means.

    The delete goes through `remove_staged_escalation_dir`, which bounds it
    to the project's artifacts/pending_escalations/ -- `staged_path`
    round-trips through the saved project file, so it is untrusted input
    from disk and every delete here is a recursive rmtree.
    """
    from .sam2_escalation import remove_staged_escalation_dir

    review = _review_of(source)
    decisions = read_decisions(review.staged_path)
    if any(outcome != REJECTED for outcome in decisions.values()):
        source.reviewed = False
    remove_staged_escalation_dir(review.staged_path, project_dir)
    source.staged_review = None
```

- [ ] **Step 4: Run the tests**

```bash
python -m pytest tests/test_detectkit_staged_review.py -q
```

Expected: PASS. If `test_accept_is_producer_agnostic` fails for one producer only, a
branch on `producer` has crept in — remove it, that is the test's job.

- [ ] **Step 5: Route the existing dialog through the new path**

So the app keeps working before the review bar exists, replace the body of
`ReviewEscalationsDialog._apply_checked`'s per-source branch with the producer-agnostic
call:

```python
                if accept:
                    accept_all(src, mode=MergeMode.OVERWRITE)
                    finish_review(src, self._project_dir)
                    self.accepted_names.append(src.name)
                else:
                    reject_all(src)
                    finish_review(src, self._project_dir)
                    self.rejected_names.append(src.name)
```

Update the imports at the top of the dialog to
`from ...jobs.staged_review import accept_all, finish_review, reject_all` and
`from hydra_suite.data.al.merge import MergeMode`, and delete the
`accept_pending_semantic_escalation` / `accept_pending_escalation` /
`reject_pending_escalation` imports and the `primer_kind` branch. Update the intro
`QLabel` text: SAM3 results no longer become a sibling source.

- [ ] **Step 6: Run the dialog and wiring tests**

```bash
python -m pytest tests/test_detectkit_review_escalations_dialog.py \
  tests/test_detectkit_sam2_escalation_wiring.py tests/test_detectkit_staged_review.py -q
```

Expected: PASS. Tests asserting the sibling-source outcome must be updated to assert
in-place acceptance — that behaviour change is the point of the design.

- [ ] **Step 7: Commit**

```bash
make format && make lint-moderate
git add -A src/hydra_suite/detectkit tests/test_detectkit_staged_review.py tests/test_detectkit_review_escalations_dialog.py tests/test_detectkit_sam2_escalation_wiring.py
git commit -m "feat(detectkit): producer-agnostic frame-granular accept with promotion"
```

---

# Phase 3 — The review bar (Tasks 8-10)

### Task 8: The staged overlay stops drawing decided frames

Once a frame is decided, its proposal is resolved: accepted geometry now lives on the
ground-truth layer, and rejected geometry is not wanted. Continuing to draw it in
magenta would make a reviewed frame indistinguishable from an unreviewed one.

**Files:**
- Modify: `src/hydra_suite/detectkit/gui/overlays/providers.py`
- Test: `tests/test_detectkit_overlay_providers.py`

**One key, computed one way.** The review's per-frame key is the image's path relative
to the source's `images/`, and every producer stages its label at exactly that path
under `labels/` (`sam2_escalation.py:270`). But
`find_staged_label_for_image` (`gui/utils.py:159-166`) has stem and recursive
*fallbacks*, so a staging dir that does not mirror cleanly can return a path whose
`relative_to(staged_labels)` is something else entirely — and then the provider would
look up a key that `staged_frames`/`accept_frame` never write. The provider must
therefore derive the key from the **image**, exactly as `_current_staged_rel` does, and
never from the label path it happened to find.

**Interfaces:**
- Consumes: `read_decisions` (Task 5).
- Produces: `StagedReviewProvider.build` returns `None` for a frame whose
  images-relative key carries any decision; module-level
  `staged_review.review_key_for_image(source_path, image_path) -> str | None` — the one
  place that key is computed.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_detectkit_overlay_providers.py` (reusing that file's existing
`source_tree`, `_project`, `_staged`, `_ctx` fixtures/helpers):

```python
def test_staged_layer_disappears_once_the_frame_is_decided(source_tree, tmp_path):
    from hydra_suite.detectkit.jobs.staged_review import (
        ACCEPTED_ADD_NEW,
        review_key_for_image,
        write_decisions,
    )
    from hydra_suite.detectkit.gui.overlays.providers import StagedReviewProvider

    staged = _staged(tmp_path, "obb")
    project = _project(source_tree, staged_review=staged)
    ctx = _ctx(project, source_tree)
    assert StagedReviewProvider().build(ctx) is not None

    rel = review_key_for_image(source_tree.root, ctx.image_path)
    assert rel == "f0001.txt"
    write_decisions(staged.staged_path, {rel: ACCEPTED_ADD_NEW})

    assert StagedReviewProvider().build(ctx) is None


def test_a_decision_on_another_frame_does_not_hide_this_one(source_tree, tmp_path):
    from hydra_suite.detectkit.jobs.staged_review import REJECTED, write_decisions
    from hydra_suite.detectkit.gui.overlays.providers import StagedReviewProvider

    staged = _staged(tmp_path, "obb")
    project = _project(source_tree, staged_review=staged)
    write_decisions(staged.staged_path, {"some/other/frame.txt": REJECTED})

    assert StagedReviewProvider().build(_ctx(project, source_tree)) is not None
```

- [ ] **Step 2: Run to verify they fail**

```bash
python -m pytest tests/test_detectkit_overlay_providers.py -k decided -q
```

Expected: FAIL — the layer is still built.

- [ ] **Step 3: Implement**

In `StagedReviewProvider.build`, after locating `label_path` and before parsing:

```python
        # A decided frame's proposal is resolved: an accepted one now lives on
        # the ground-truth layer, a rejected one is not wanted. Keeping it in
        # magenta would make a reviewed frame look identical to an unreviewed
        # one, which is the single thing frame-granular review has to show.
        from ...jobs.staged_review import read_decisions, review_key_for_image

        # `review` here is the local that Task 4's mechanical rename left
        # named `pending`; rename that local to `review` throughout this
        # method while you are in it, so the provider reads in the new
        # vocabulary.
        #
        # The key comes from the IMAGE, not from `label_path`.
        # find_staged_label_for_image has stem and recursive fallbacks, so
        # label_path.relative_to(staged_labels) is not guaranteed to be the
        # same string staged_frames/accept_frame use as their key. Deriving
        # it from the image is the only way the three agree.
        rel = review_key_for_image(ctx.source_path, ctx.image_path)
        if rel is not None and rel in read_decisions(review.staged_path):
            return None
```

The existing `_staged` helper in `tests/test_detectkit_overlay_providers.py:121` writes
its label at `staged/labels/images/f0001.txt`, one level deeper than any real producer
stages (`sam2_escalation.py:270` mirrors the images-relative path directly under
`labels/`). It only works today because of the recursive fallback. Fix the fixture to
match the real contract as part of this task:

```python
def _staged(tmp_path, target_level):
    staged = tmp_path / "staged"
    # labels/f0001.txt, NOT labels/images/f0001.txt: the staged tree mirrors
    # the source's images/ tree, and that mirroring IS the review key. The
    # old nesting only resolved via find_staged_label_for_image's recursive
    # fallback, which no longer agrees with the decisions key.
    _write(
        staged / "labels" / "f0001.txt",
        ["0 0.2 0.2 0.4 0.2 0.4 0.4 0.2 0.4"],
    )
    (staged / "classes.txt").write_text("prompt_a\n")
    return SimpleNamespace(staged_path=str(staged), target_level=target_level)
```

- [ ] **Step 4: Run the provider tests**

```bash
python -m pytest tests/test_detectkit_overlay_providers.py tests/test_detectkit_overlay_golden.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
make format && make lint-moderate
git add src/hydra_suite/detectkit/gui/overlays/providers.py tests/test_detectkit_overlay_providers.py
git commit -m "feat(detectkit): hide the staged overlay on frames already decided"
```

---

### Task 9: The `ReviewBar` widget

A bar above the canvas, visible only when the current source has a staged review.
Widget-only in this task: it emits signals and renders state, and knows nothing about
the project or the filesystem.

**Files:**
- Create: `src/hydra_suite/detectkit/gui/panels/review_bar.py`
- Test: `tests/test_detectkit_review_bar.py`

**Interfaces:**
- Consumes: nothing from earlier tasks (deliberately — it takes plain values).
- Produces:
  - `ReviewBar(QWidget)` with signals
    `accept_overwrite_requested`, `accept_add_new_requested`, `reject_requested`,
    `accept_all_requested`, `reject_all_requested`, `next_undecided_requested`,
    `revert_requested`, `rethreshold_requested` (all `Signal()`, no args)
  - `set_review_state(producer: str, detail: str, decided: int, total: int, can_rethreshold: bool) -> None`
  - `clear_review_state() -> None` (hides the bar)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_detectkit_review_bar.py`. **The `QT_QPA_PLATFORM` line is not
optional** — every existing DetectKit GUI test file sets it before importing PySide6
(e.g. `tests/test_detectkit_canvas.py:11`), and without it these tests try to open a
real window. Note also that Tasks 10 and 12 are the **first tests in the whole suite to
construct `DetectKitMainWindow`** (`grep -rn "DetectKitMainWindow()" tests/` finds
none today). Given this repo's history of native-crash GUI tests, run each new GUI test
file **on its own** the first time and confirm it exits cleanly rather than aborting;
if a construction crashes the interpreter, drop back to testing the handlers on a
`SimpleNamespace` stub rather than a real window, and say so in the commit message.

```python
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from types import SimpleNamespace  # noqa: E402

from hydra_suite.detectkit.gui.panels.review_bar import ReviewBar  # noqa: E402


def test_the_bar_is_hidden_until_a_review_is_set(qtbot):
    bar = ReviewBar()
    qtbot.addWidget(bar)

    assert not bar.isVisibleTo(bar.parentWidget()) or bar.isHidden()


def test_setting_a_review_shows_the_bar_and_the_counter(qtbot):
    bar = ReviewBar()
    qtbot.addWidget(bar)

    bar.set_review_state("sam3", "prompt 'ant'", decided=23, total=140, can_rethreshold=True)

    assert not bar.isHidden()
    assert "23/140" in bar.progress_text()
    assert "sam3" in bar.summary_text()
    assert "ant" in bar.summary_text()


def test_clearing_hides_the_bar(qtbot):
    bar = ReviewBar()
    qtbot.addWidget(bar)
    bar.set_review_state("sam2", "sam2.1_hiera_large", 0, 10, can_rethreshold=False)

    bar.clear_review_state()

    assert bar.isHidden()


def test_rethreshold_is_offered_only_when_the_producer_supports_it(qtbot):
    bar = ReviewBar()
    qtbot.addWidget(bar)

    bar.set_review_state("sam2", "v", 0, 10, can_rethreshold=False)
    assert not bar.rethreshold_button().isVisible() or not bar.rethreshold_button().isEnabled()

    bar.set_review_state("sam3", "prompt 'ant'", 0, 10, can_rethreshold=True)
    assert bar.rethreshold_button().isEnabled()


@pytest.mark.parametrize(
    "button_name,signal_name",
    [
        ("accept_overwrite_button", "accept_overwrite_requested"),
        ("accept_add_new_button", "accept_add_new_requested"),
        ("reject_button", "reject_requested"),
        ("accept_all_button", "accept_all_requested"),
        ("reject_all_button", "reject_all_requested"),
        ("next_undecided_button", "next_undecided_requested"),
        ("revert_button", "revert_requested"),
    ],
)
def test_each_button_emits_its_signal(qtbot, button_name, signal_name):
    bar = ReviewBar()
    qtbot.addWidget(bar)
    bar.set_review_state("sam2", "v", 0, 10, can_rethreshold=False)

    with qtbot.waitSignal(getattr(bar, signal_name), timeout=500):
        getattr(bar, button_name)().click()


def test_a_complete_review_says_so(qtbot):
    bar = ReviewBar()
    qtbot.addWidget(bar)

    bar.set_review_state("sam2", "v", 10, 10, can_rethreshold=False)

    assert "complete" in bar.progress_text().lower()
```

- [ ] **Step 2: Run to verify they fail**

```bash
python -m pytest tests/test_detectkit_review_bar.py -q
```

Expected: FAIL — `ModuleNotFoundError: ...panels.review_bar`.

- [ ] **Step 3: Implement the widget**

Create `src/hydra_suite/detectkit/gui/panels/review_bar.py`:

```python
"""The frame-granular review bar, shown above the canvas.

Visible only when the current source has a staged review. It renders state
and emits intent; it touches neither the project nor the filesystem, which
is what keeps it testable without a project on disk.
"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QWidget


class ReviewBar(QWidget):
    """Accept/reject the frame on screen, or the whole review."""

    accept_overwrite_requested = Signal()
    accept_add_new_requested = Signal()
    reject_requested = Signal()
    accept_all_requested = Signal()
    reject_all_requested = Signal()
    next_undecided_requested = Signal()
    revert_requested = Signal()
    rethreshold_requested = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setProperty("detectkitRole", "reviewBar")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(8)

        self._summary = QLabel("")
        self._summary.setProperty("detectkitRole", "sectionHint")
        layout.addWidget(self._summary, 1)

        self._btn_overwrite = QPushButton("Replace")
        self._btn_overwrite.setToolTip(
            "Replace this frame's labels with the staged ones."
        )
        self._btn_add_new = QPushButton("Add New")
        self._btn_add_new.setToolTip(
            "Keep this frame's labels and add only the staged instances that do "
            "not overlap one already there."
        )
        self._btn_reject = QPushButton("Reject")
        self._btn_reject.setToolTip("Discard the staged labels for this frame.")
        self._btn_next = QPushButton("Next Undecided")
        self._btn_accept_all = QPushButton("Accept All…")
        self._btn_reject_all = QPushButton("Reject All…")
        self._btn_revert = QPushButton("Revert Review…")
        self._btn_revert.setToolTip(
            "Restore this source to its state before the review started. "
            "Available only while the review is open -- finishing it deletes "
            "the snapshot."
        )
        self._btn_rethreshold = QPushButton("Re-threshold…")
        self._btn_rethreshold.setToolTip(
            "Rewrite the staged result at a different confidence, using the "
            "cached candidates. No inference -- seconds, not hours."
        )

        for button in (
            self._btn_overwrite,
            self._btn_add_new,
            self._btn_reject,
            self._btn_next,
            self._btn_accept_all,
            self._btn_reject_all,
            self._btn_revert,
            self._btn_rethreshold,
        ):
            layout.addWidget(button)

        self._progress = QLabel("")
        layout.addWidget(self._progress)

        self._btn_overwrite.clicked.connect(self.accept_overwrite_requested)
        self._btn_add_new.clicked.connect(self.accept_add_new_requested)
        self._btn_reject.clicked.connect(self.reject_requested)
        self._btn_next.clicked.connect(self.next_undecided_requested)
        self._btn_accept_all.clicked.connect(self.accept_all_requested)
        self._btn_reject_all.clicked.connect(self.reject_all_requested)
        self._btn_revert.clicked.connect(self.revert_requested)
        self._btn_rethreshold.clicked.connect(self.rethreshold_requested)

        self.hide()

    # -- state ---------------------------------------------------------

    def set_review_state(
        self,
        producer: str,
        detail: str,
        decided: int,
        total: int,
        can_rethreshold: bool,
    ) -> None:
        """Show the bar for a staged review and render its progress."""
        self._summary.setText(f"Staged review — {producer}: {detail}")
        self._progress.setText(
            f"{decided}/{total} decided — review complete"
            if total and decided >= total
            else f"{decided}/{total} decided"
        )
        self._btn_rethreshold.setEnabled(bool(can_rethreshold))
        self._btn_rethreshold.setVisible(bool(can_rethreshold))
        self.show()

    def clear_review_state(self) -> None:
        """Hide the bar; the current source has no staged review."""
        self._summary.setText("")
        self._progress.setText("")
        self.hide()

    # -- accessors used by MainWindow and by tests ----------------------

    def summary_text(self) -> str:
        return self._summary.text()

    def progress_text(self) -> str:
        return self._progress.text()

    def accept_overwrite_button(self) -> QPushButton:
        return self._btn_overwrite

    def accept_add_new_button(self) -> QPushButton:
        return self._btn_add_new

    def reject_button(self) -> QPushButton:
        return self._btn_reject

    def accept_all_button(self) -> QPushButton:
        return self._btn_accept_all

    def reject_all_button(self) -> QPushButton:
        return self._btn_reject_all

    def next_undecided_button(self) -> QPushButton:
        return self._btn_next

    def revert_button(self) -> QPushButton:
        return self._btn_revert

    def rethreshold_button(self) -> QPushButton:
        return self._btn_rethreshold
```

- [ ] **Step 4: Run the tests**

```bash
python -m pytest tests/test_detectkit_review_bar.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
make format && make lint-moderate
git add src/hydra_suite/detectkit/gui/panels/review_bar.py tests/test_detectkit_review_bar.py
git commit -m "feat(detectkit): add the frame-granular review bar widget"
```

---

### Task 10: Wire the review bar into `MainWindow`; retire the dialog

**Files:**
- Modify: `src/hydra_suite/detectkit/gui/main_window.py`
- Modify: `src/hydra_suite/detectkit/gui/escalation_actions.py`
- Delete: `src/hydra_suite/detectkit/gui/dialogs/review_escalations_dialog.py`
- Delete: `tests/test_detectkit_review_escalations_dialog.py`
- Test: `tests/test_detectkit_review_bar.py`

**Interfaces:**
- Consumes: `ReviewBar` (Task 9); `accept_frame`, `reject_frame`, `accept_all`,
  `reject_all`, `is_complete`, `finish_review`, `revert_review`, `review_progress`,
  `review_key_for_image`, `staged_frames`, `read_decisions` (Tasks 5-7); `rethreshold_staged`,
  `rethreshold_floor_for` (`jobs/semantic_escalation.py`);
  `_refresh_overlays` (registry).
- Produces: `DetectKitMainWindow._review_bar`, `_sync_review_bar()`,
  `_current_staged_rel()`, `_on_review_accept(mode)`, `_on_review_reject()`,
  `_on_review_next_undecided()`, `_on_review_revert()`, `_on_review_rethreshold()`.

- [ ] **Step 1: Write the failing integration tests**

Append to `tests/test_detectkit_review_bar.py`:

```python
def test_accepting_a_frame_refreshes_both_layers(qtbot, monkeypatch, tmp_path):
    """Accept must redraw GT (the change landed) and staged (it is decided).

    Directly, not incidentally: a selection-preserving refresh would
    otherwise leave the accepted proposal on screen with nothing asking for
    a redraw.
    """
    from hydra_suite.data.al.merge import MergeMode
    from hydra_suite.detectkit.gui import main_window as mw

    refreshed: list = []
    window = mw.DetectKitMainWindow()
    qtbot.addWidget(window)
    monkeypatch.setattr(window, "_refresh_overlays", lambda keys=None: refreshed.append(keys))
    monkeypatch.setattr(window, "_current_staged_rel", lambda: "a.txt")
    # SimpleNamespace, not object(): _on_review_accept ends by calling
    # _offer_finish_if_complete -> is_complete, which reads staged_review.
    monkeypatch.setattr(
        window, "_current_source_obj", lambda: SimpleNamespace(staged_review=None)
    )
    monkeypatch.setattr(window, "_save_current_project", lambda: None)
    monkeypatch.setattr(window, "_sync_review_bar", lambda: None)
    monkeypatch.setattr(mw, "accept_frame", lambda *a, **k: None)

    window._on_review_accept(MergeMode.ADD_NEW)

    assert refreshed and set(refreshed[-1]) == {"gt", "escalation"}


def test_next_undecided_selects_the_first_frame_without_a_decision(qtbot, monkeypatch):
    from hydra_suite.detectkit.gui import main_window as mw

    window = mw.DetectKitMainWindow()
    qtbot.addWidget(window)
    monkeypatch.setattr(mw, "staged_frames", lambda root: ["a.txt", "b.txt", "c.txt"])
    monkeypatch.setattr(mw, "read_decisions", lambda root: {"a.txt": "rejected"})
    selected: list = []
    monkeypatch.setattr(window._dataset_panel, "select_image_by_relative_label", lambda rel: selected.append(rel))
    monkeypatch.setattr(window, "_current_staged_root", lambda: "/tmp/staging")

    window._on_review_next_undecided()

    assert selected == ["b.txt"]
```

- [ ] **Step 2: Run to verify they fail**

```bash
python -m pytest tests/test_detectkit_review_bar.py -q
```

Expected: FAIL — `AttributeError: ... has no attribute '_on_review_accept'`.

- [ ] **Step 3: Add the bar to the workspace layout**

Construct it ONCE, in `__init__` next to `self._canvas = OBBCanvas()` (line ~726), so
the handlers can reach it before the workspace page is built:

```python
        self._review_bar = ReviewBar()
```

Then, in `_build_workspace_page`, add it immediately before
`canvas_layout.addWidget(self._canvas, 1)`:

```python
        canvas_layout.addWidget(self._review_bar)
```

Import it with the other panels: `from .panels.review_bar import ReviewBar`.

Connect the signals in the same place the other panels are wired:

```python
        self._review_bar.accept_overwrite_requested.connect(
            lambda: self._on_review_accept(MergeMode.OVERWRITE)
        )
        self._review_bar.accept_add_new_requested.connect(
            lambda: self._on_review_accept(MergeMode.ADD_NEW)
        )
        self._review_bar.reject_requested.connect(self._on_review_reject)
        self._review_bar.accept_all_requested.connect(
            lambda: self._on_review_bulk(accept=True)
        )
        self._review_bar.reject_all_requested.connect(
            lambda: self._on_review_bulk(accept=False)
        )
        self._review_bar.next_undecided_requested.connect(self._on_review_next_undecided)
        self._review_bar.revert_requested.connect(self._on_review_revert)
        self._review_bar.rethreshold_requested.connect(self._on_review_rethreshold)
```

- [ ] **Step 4: Implement the handlers**

Add to `DetectKitMainWindow` (with
`from ..jobs.staged_review import accept_all, accept_frame, is_complete, finish_review, read_decisions, reject_all, reject_frame, revert_review, review_key_for_image, review_progress, staged_frames`
and `from hydra_suite.data.al.merge import MergeMode` at module scope, so the tests can
monkeypatch them on the module):

```python
    # ------------------------------------------------------------------
    # Frame-granular review
    # ------------------------------------------------------------------

    def _current_source_obj(self):
        if self._project is None or not self._current_source_path:
            return None
        return next(
            (s for s in self._project.sources if str(s.path) == self._current_source_path),
            None,
        )

    def _current_staged_root(self) -> "str | None":
        source = self._current_source_obj()
        review = getattr(source, "staged_review", None) if source else None
        if review is None or not str(review.staged_path).strip():
            return None
        return str(review.staged_path)

    def _current_staged_rel(self) -> "str | None":
        """The current frame's key into the review. One definition, reused."""
        source = self._current_source_obj()
        if source is None or not self._current_image_path:
            return None
        return review_key_for_image(source.path, self._current_image_path)

    def _sync_review_bar(self) -> None:
        """Show or hide the bar for the source on screen, and refresh its counter."""
        source = self._current_source_obj()
        review = getattr(source, "staged_review", None) if source else None
        if review is None:
            self._review_bar.clear_review_state()
            return
        decided, total = review_progress(review.staged_path)
        detail = (
            f"prompt '{review.prompt}'" if review.prompt else review.producer_variant
        )
        self._review_bar.set_review_state(
            review.producer, detail, decided, total, can_rethreshold=review.producer == "sam3"
        )

    def _after_review_change(self) -> None:
        """Everything a decision must trigger, in one place.

        Both layers, directly: the ground truth changed (that is the point of
        reviewing on the frame) and the staged proposal is now decided, so it
        must stop drawing. Neither refresh happens incidentally.
        """
        self._refresh_overlays(keys=("gt", "escalation"))
        self._sync_review_bar()
        self._save_current_project()

    def _on_review_accept(self, mode) -> None:
        source = self._current_source_obj()
        rel = self._current_staged_rel()
        if source is None or rel is None:
            return
        try:
            accept_frame(source, rel, mode=mode)
        except Exception as exc:
            QMessageBox.warning(self, "Review", str(exc))
            return
        self._after_review_change()
        self._offer_finish_if_complete(source)

    def _on_review_reject(self) -> None:
        source = self._current_source_obj()
        rel = self._current_staged_rel()
        if source is None or rel is None:
            return
        try:
            reject_frame(source, rel)
        except Exception as exc:
            QMessageBox.warning(self, "Review", str(exc))
            return
        self._after_review_change()
        self._offer_finish_if_complete(source)

    def _on_review_bulk(self, *, accept: bool) -> None:
        source = self._current_source_obj()
        if source is None:
            return
        verb = "Accept" if accept else "Reject"
        mode = MergeMode.ADD_NEW  # unused on the reject path; bound so it always exists
        if accept:
            choice = QMessageBox.question(
                self,
                "Accept All",
                "Replace each undecided frame's labels with the staged ones, "
                "or add only the non-overlapping staged instances?\n\n"
                "Yes = Replace, No = Add New.",
                QMessageBox.StandardButton.Yes
                | QMessageBox.StandardButton.No
                | QMessageBox.StandardButton.Cancel,
            )
            if choice == QMessageBox.StandardButton.Cancel:
                return
            mode = (
                MergeMode.OVERWRITE
                if choice == QMessageBox.StandardButton.Yes
                else MergeMode.ADD_NEW
            )
        try:
            count = accept_all(source, mode=mode) if accept else reject_all(source)
        except Exception as exc:
            QMessageBox.warning(self, verb + " All", str(exc))
            return
        self._after_review_change()
        self.statusBar().showMessage(f"{verb}ed {count} frame(s).", 4000)
        self._offer_finish_if_complete(source)

    def _on_review_next_undecided(self) -> None:
        staged_root = self._current_staged_root()
        if staged_root is None:
            return
        decided = read_decisions(staged_root)
        for rel in staged_frames(staged_root):
            if rel not in decided:
                self._dataset_panel.select_image_by_relative_label(rel)
                return
        self.statusBar().showMessage("Every staged frame has been decided.", 4000)

    def _on_review_revert(self) -> None:
        source = self._current_source_obj()
        staged_root = self._current_staged_root()
        if source is None or staged_root is None:
            return
        if QMessageBox.question(
            self,
            "Revert Review",
            "Restore this source's labels, geometry level and class list to "
            "their state before this review started? Every decision made so "
            "far is cleared.",
        ) != QMessageBox.StandardButton.Yes:
            return
        try:
            revert_review(source, staged_root)
        except Exception as exc:
            QMessageBox.warning(self, "Revert Review", str(exc))
            return
        self._after_review_change()

    def _on_review_rethreshold(self) -> None:
        """The one irreplaceable feature of the retired dialog, per-review."""
        from PySide6.QtWidgets import QInputDialog

        from ..jobs.semantic_escalation import rethreshold_floor_for, rethreshold_staged

        source = self._current_source_obj()
        review = getattr(source, "staged_review", None) if source else None
        if review is None or review.producer != "sam3":
            return
        # Re-thresholding rewrites the staged labels underneath any decision
        # already recorded against them, so those decisions would describe
        # geometry that no longer exists. Refuse rather than silently
        # invalidate them; reverting first is the documented way through.
        if read_decisions(review.staged_path):
            QMessageBox.information(
                self,
                "Re-threshold",
                "Frames in this review have already been decided, and "
                "re-thresholding would rewrite the staged labels those "
                "decisions refer to. Revert the review first if you want to "
                "re-threshold it.",
            )
            return
        current = float(review.params.get("confidence", 0.35))
        minimum = rethreshold_floor_for([source])
        value, ok = QInputDialog.getDouble(
            self,
            "Re-threshold",
            f"New confidence (cache floor {minimum:.2f}):",
            max(current, minimum),
            minimum,
            0.99,
            2,
        )
        if not ok:
            return
        try:
            kept = rethreshold_staged(
                source,
                confidence=value,
                merge_iou=float(review.params.get("merge_iou", 0.5)),
            )
        except Exception as exc:
            QMessageBox.warning(self, "Re-threshold", str(exc))
            return
        self._after_review_change()
        self.statusBar().showMessage(
            f"{source.name}: {kept} instance(s) at confidence {value:.2f}.", 5000
        )

    def _offer_finish_if_complete(self, source) -> None:
        if not is_complete(source):
            return
        if QMessageBox.question(
            self,
            "Review Complete",
            "Every staged frame has been decided. Close this review?\n\n"
            "Closing deletes the staged proposals AND the snapshot, so "
            "\"Revert Review\" is no longer available afterwards.",
        ) != QMessageBox.StandardButton.Yes:
            return
        finish_review(source, self._project.project_dir if self._project else None)
        self._save_current_project()
        self._dataset_panel.refresh_sources(self._project)
        self._tools_panel.refresh_overview()
        self._refresh_overlays(keys=("gt", "escalation"))
        self._sync_review_bar()
```

Call `self._sync_review_bar()` at the end of `show_image` (after
`self._refresh_overlays()`), so the bar tracks the source on screen.

- [ ] **Step 5: Add the panel's frame-selection helper**

In `src/hydra_suite/detectkit/gui/panels/dataset_panel.py`:

```python
    def select_image_by_relative_label(self, rel: str) -> bool:
        """Select the image whose label path under labels/ is *rel*.

        The review's key is the label's relative path; the list holds image
        paths. Matching on the stem-plus-parent rather than the extension is
        what makes it work for the mixed-extension sources DetectKit imports.
        """
        source_path = self._selected_source_path()
        if source_path is None:
            return False
        target = Path(rel).with_suffix("")
        for row in range(self.image_list.count()):
            item = self.image_list.item(row)
            image_path = Path(str(item.data(Qt.UserRole)))
            try:
                candidate = image_path.relative_to(Path(source_path) / "images")
            except ValueError:
                candidate = Path(image_path.name)
            if candidate.with_suffix("") == target:
                self.image_list.setCurrentRow(row)
                return True
        return False
```

- [ ] **Step 6: Delete the dialog**

```bash
git rm src/hydra_suite/detectkit/gui/dialogs/review_escalations_dialog.py
git rm tests/test_detectkit_review_escalations_dialog.py
```

In `escalation_actions.py`, delete `on_review_escalations` entirely. Remove its menu
action and every reference:

```bash
grep -rn "on_review_escalations\|ReviewEscalationsDialog\|Review Escalations" src/hydra_suite tests docs
```

Replace the menu entry with one that selects the first source having a staged review
and lets the review bar take over — add to `main_window.py`:

```python
    def _on_go_to_staged_review(self) -> None:
        """Jump to a source with a staged review; the review bar drives it.

        Kept as a menu entry because a staged review is otherwise only
        discoverable by browsing to the right source.
        """
        if self._project is None:
            QMessageBox.information(self, "Staged Reviews", "Open a project first.")
            return
        pending = [s for s in self._project.sources if s.staged_review is not None]
        if not pending:
            QMessageBox.information(
                self, "Staged Reviews", "There are no staged reviews."
            )
            return
        self._dataset_panel.select_source_by_path(str(pending[0].path))
```

If `select_source_by_path` does not exist on `DatasetPanel`, add it alongside
`select_image_by_relative_label`, matching on `self.source_combo.itemData(i)`.

- [ ] **Step 7: Run the DetectKit GUI test surface**

```bash
python -m pytest tests/test_detectkit_review_bar.py tests/test_detectkit_dataset_panel.py \
  tests/test_detectkit_dataset_panel_widget.py tests/test_detectkit_canvas.py \
  tests/test_detectkit_canvas_dual_layer.py tests/test_detectkit_show_image_multi_level.py \
  tests/test_detectkit_overlay_providers.py tests/test_detectkit_overlay_golden.py \
  tests/test_detectkit_staged_review.py -q
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
make format && make lint-moderate
git add -A src/hydra_suite/detectkit tests
git commit -m "feat(detectkit): review bar replaces the per-source escalation dialog"
```

---

# Phase 4 — Inference as a producer (Tasks 11-12)

### Task 11: The inference stager

`_dataset_predictions` is an in-memory dict cleared on source switch
(`main_window.py:712,2040`); nothing writes it to disk. Inference and escalation are the
same kind of proposal to a user and are completely different objects in the code. This
makes inference the third producer of the same staging contract.

The in-memory preview path is **unaffected**: staging is a separate, explicit action, so
running inference merely to look at it never creates reviewable state.

**Files:**
- Create: `src/hydra_suite/detectkit/jobs/inference_stager.py`
- Test: `tests/test_detectkit_inference_stager.py`

**Interfaces:**
- Consumes: `write_label_file` (`data/al/labels.py`); `LabelRecord`
  (`data/al/escalation.py`); `StagedReview` (Task 4);
  `PENDING_ESCALATIONS_RELDIR`, `ensure_bundle_subdirectory`
  (`jobs/sam2_escalation.py`, `data/project_bundle.py`).
- Produces:
  `stage_predictions(source, project_dir, per_image, *, model_path, inference_kind, confidence, device) -> StagedReview`
  where `per_image` is `{image_path: [ {class_id, polygon_px, confidence}, ... ]}` —
  exactly `_DetectKitDatasetInferenceWorker`'s `per_image` payload.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_detectkit_inference_stager.py`. The `QT_QPA_PLATFORM` guard is
required here too: Task 12 appends tests that import `main_window`.

```python
import json
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from hydra_suite.detectkit.gui.models import OBBSource
from hydra_suite.detectkit.jobs.inference_stager import stage_predictions


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "sources" / "src"
    (root / "images" / "sub").mkdir(parents=True)
    (root / "labels").mkdir(parents=True)
    (root / "classes.txt").write_text("ant\n")
    for rel in ("a.png", "sub/b.png"):
        Image.fromarray(np.zeros((100, 200, 3), dtype=np.uint8)).save(root / "images" / rel)
    return OBBSource(path=str(root), name="src", level="obb")


def _dets():
    return [
        {"class_id": 0, "polygon_px": [(10, 10), (50, 10), (50, 40), (10, 40)], "confidence": 0.9}
    ]


def test_a_label_file_is_written_per_predicted_frame(tmp_path, source):
    per_image = {
        str(Path(source.path) / "images" / "a.png"): _dets(),
        str(Path(source.path) / "images" / "sub" / "b.png"): _dets(),
    }

    review = stage_predictions(
        source, tmp_path, per_image, model_path="/models/best.pt",
        inference_kind="obb_direct", confidence=0.4, device="mps",
    )

    labels = Path(review.staged_path) / "labels"
    assert (labels / "a.txt").is_file()
    assert (labels / "sub" / "b.txt").is_file()


def test_staged_paths_mirror_the_images_tree(tmp_path, source):
    per_image = {str(Path(source.path) / "images" / "sub" / "b.png"): _dets()}

    review = stage_predictions(
        source, tmp_path, per_image, model_path="/m.pt",
        inference_kind="obb_direct", confidence=0.4, device="cpu",
    )

    staged = sorted(
        p.relative_to(Path(review.staged_path) / "labels").as_posix()
        for p in (Path(review.staged_path) / "labels").rglob("*.txt")
    )
    assert staged == ["sub/b.txt"]


def test_the_producer_is_inference_and_the_level_follows_the_kind(tmp_path, source):
    per_image = {str(Path(source.path) / "images" / "a.png"): _dets()}

    obb = stage_predictions(
        source, tmp_path, per_image, model_path="/m.pt",
        inference_kind="obb_direct", confidence=0.4, device="cpu",
    )
    assert obb.producer == "inference"
    assert obb.target_level == "obb"

    source.staged_review = None
    seg = stage_predictions(
        source, tmp_path, per_image, model_path="/m.pt",
        inference_kind="sequential_segment", confidence=0.4, device="cpu",
    )
    assert seg.target_level == "polygon"


def test_run_json_records_the_model_confidence_and_device(tmp_path, source):
    per_image = {str(Path(source.path) / "images" / "a.png"): _dets()}

    review = stage_predictions(
        source, tmp_path, per_image, model_path="/models/best.pt",
        inference_kind="obb_direct", confidence=0.42, device="mps",
    )

    run = json.loads((Path(review.staged_path) / "run.json").read_text())
    assert run["producer"] == "inference"
    assert run["params"]["model_path"] == "/models/best.pt"
    assert run["params"]["confidence"] == 0.42
    assert run["params"]["device"] == "mps"


def test_classes_txt_is_copied_from_the_source(tmp_path, source):
    per_image = {str(Path(source.path) / "images" / "a.png"): _dets()}

    review = stage_predictions(
        source, tmp_path, per_image, model_path="/m.pt",
        inference_kind="obb_direct", confidence=0.4, device="cpu",
    )

    assert (Path(review.staged_path) / "classes.txt").read_text() == "ant\n"


def test_staging_lands_inside_the_projects_pending_escalations_dir(tmp_path, source):
    per_image = {str(Path(source.path) / "images" / "a.png"): _dets()}

    review = stage_predictions(
        source, tmp_path, per_image, model_path="/m.pt",
        inference_kind="obb_direct", confidence=0.4, device="cpu",
    )

    assert "pending_escalations" in Path(review.staged_path).parts


def test_frames_with_no_detections_are_not_staged(tmp_path, source):
    per_image = {
        str(Path(source.path) / "images" / "a.png"): [],
        str(Path(source.path) / "images" / "sub" / "b.png"): _dets(),
    }

    review = stage_predictions(
        source, tmp_path, per_image, model_path="/m.pt",
        inference_kind="obb_direct", confidence=0.4, device="cpu",
    )

    staged = list((Path(review.staged_path) / "labels").rglob("*.txt"))
    assert [p.name for p in staged] == ["b.txt"]


def test_staging_nothing_at_all_is_refused_rather_than_creating_a_dead_review(
    tmp_path, source
):
    """A zero-frame review would be unfinishable.

    `is_complete` needs total > 0, reject-all rejects nothing, and revert has
    no snapshot -- the user would be stuck with a review only a hand-edit of
    the project JSON could clear.
    """
    per_image = {str(Path(source.path) / "images" / "a.png"): []}

    with pytest.raises(RuntimeError, match="no detections"):
        stage_predictions(
            source, tmp_path, per_image, model_path="/m.pt",
            inference_kind="obb_direct", confidence=0.4, device="cpu",
        )

    assert source.staged_review is None


def test_staging_over_an_open_review_is_refused(tmp_path, source):
    per_image = {str(Path(source.path) / "images" / "a.png"): _dets()}
    stage_predictions(
        source, tmp_path, per_image, model_path="/m.pt",
        inference_kind="obb_direct", confidence=0.4, device="cpu",
    )

    with pytest.raises(RuntimeError, match="already has a staged review"):
        stage_predictions(
            source, tmp_path, per_image, model_path="/m.pt",
            inference_kind="obb_direct", confidence=0.4, device="cpu",
        )
```

- [ ] **Step 2: Run to verify they fail**

```bash
python -m pytest tests/test_detectkit_inference_stager.py -q
```

Expected: FAIL — `ModuleNotFoundError: ...jobs.inference_stager`.

- [ ] **Step 3: Implement**

Create `src/hydra_suite/detectkit/jobs/inference_stager.py`:

```python
"""Dataset inference as the third producer of the staging contract.

Inference and escalation are the same kind of thing -- a machine proposal a
human must accept or reject -- and used to be completely different objects
in the code: predictions lived only in an in-memory dict that was cleared on
source switch and never written anywhere. Staging them makes them
reviewable by exactly the code that reviews SAM2 and SAM3 output.

The in-memory preview path is untouched. Staging is a separate, explicit
action, so running inference merely to LOOK at it never creates reviewable
state.
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from hydra_suite.data.al.escalation import LabelRecord
from hydra_suite.data.al.labels import write_label_file
from hydra_suite.data.project_bundle import ensure_bundle_subdirectory
from hydra_suite.detectkit.gui.models import OBBSource, StagedReview
from hydra_suite.utils.geometry_levels import GeometryLevel

from .sam2_escalation import PENDING_ESCALATIONS_RELDIR

logger = logging.getLogger(__name__)

# The geometry a kind natively produces. All five kinds
# `detectkit_resolve_inference_models` can return are listed: the segment
# kinds emit real contours, detect_direct emits axis-aligned boxes, the OBB
# kinds emit quads. Omitting a kind would silently stage its labels at the
# wrong declared level, which `read_label_file` would then disagree with.
_LEVEL_BY_KIND = {
    "obb_direct": GeometryLevel.OBB,
    "sequential": GeometryLevel.OBB,
    "detect_direct": GeometryLevel.AABB,
    "segment_direct": GeometryLevel.POLYGON,
    "sequential_segment": GeometryLevel.POLYGON,
}


def stage_predictions(
    source: OBBSource,
    project_dir: str | Path,
    per_image: dict[str, list[dict]],
    *,
    model_path: str,
    inference_kind: str,
    confidence: float,
    device: str,
) -> StagedReview:
    """Write dataset-inference predictions into the staging contract.

    `per_image` is exactly `_DetectKitDatasetInferenceWorker`'s payload:
    image path -> list of ``{class_id, polygon_px, confidence}`` dicts in
    PIXEL space.

    The staged label's relative path mirrors the image's under ``images/``
    -- that mirroring IS the review's per-frame key, relied on by
    `find_staged_label_for_image` and by `staged_review.accept_frame`.

    Staging lands under the project's ``artifacts/pending_escalations/`` so
    that `_is_safe_to_delete` keeps bounding the recursive delete that
    finishing or rejecting a review performs.

    Frames with no detections are not staged at all: an empty staged label
    would mean "accept this to delete the frame's labels", which is not what
    running inference asks for. If that leaves NO frames staged, the whole
    call is refused rather than creating a review that cannot be finished.
    """
    if source.staged_review is not None:
        raise RuntimeError(
            f"Source '{source.name}' already has a staged review. Finish or "
            "revert it before staging predictions."
        )

    level = _LEVEL_BY_KIND.get(str(inference_kind), GeometryLevel.OBB)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    staged_root = Path(
        ensure_bundle_subdirectory(
            Path(project_dir),
            str(PENDING_ESCALATIONS_RELDIR / f"{source.name}-inference-{stamp}"),
        )
    )
    (staged_root / "labels").mkdir(parents=True, exist_ok=True)

    images_dir = Path(source.path) / "images"
    staged_frames = 0
    for image_path, detections in sorted(per_image.items()):
        if not detections:
            continue
        image = Path(image_path)
        try:
            rel = image.relative_to(images_dir)
        except ValueError:
            rel = Path(image.name)

        frame = cv2.imread(str(image))
        if frame is None:
            logger.warning("Skipping unreadable image while staging: %s", image)
            continue
        height, width = int(frame.shape[0]), int(frame.shape[1])

        records = []
        for det in detections:
            pts = np.asarray(det.get("polygon_px") or [], dtype=np.float32).reshape(-1, 2)
            if pts.shape[0] < 3:
                continue
            records.append(
                LabelRecord(
                    class_id=int(det.get("class_id", 0)),
                    confidence=float(det.get("confidence", 0.0)),
                    points=pts,
                    level=level,
                )
            )
        if not records:
            continue

        out = staged_root / "labels" / rel.with_suffix(".txt")
        out.parent.mkdir(parents=True, exist_ok=True)
        write_label_file(out, records, (height, width), level)
        staged_frames += 1

    if staged_frames == 0:
        # A zero-frame review is unfinishable: is_complete needs total > 0,
        # reject-all rejects nothing, and revert has no snapshot. Refuse
        # before the source's field is set, and clean the empty dir up.
        shutil.rmtree(staged_root, ignore_errors=True)
        raise RuntimeError(
            "There were no detections to stage. Lower the confidence "
            "threshold or re-run inference before staging."
        )

    classes = Path(source.path) / "classes.txt"
    (staged_root / "classes.txt").write_text(
        classes.read_text() if classes.is_file() else "object\n"
    )

    params = {
        "model_path": str(model_path),
        "inference_kind": str(inference_kind),
        "confidence": float(confidence),
        "device": str(device),
    }
    (staged_root / "run.json").write_text(
        json.dumps(
            {
                "producer": "inference",
                "params": params,
                "staged_frames": staged_frames,
                "created_at": datetime.now().isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    review = StagedReview(
        staged_path=str(staged_root),
        target_level=level.label,
        producer="inference",
        producer_variant=Path(model_path).name,
        prompt="",
        params=params,
        created_at=datetime.now().isoformat(),
    )
    source.staged_review = review
    return review
```

- [ ] **Step 4: Run the tests**

```bash
python -m pytest tests/test_detectkit_inference_stager.py -q
```

Expected: PASS (8).

- [ ] **Step 5: Commit**

```bash
make format && make lint-moderate
git add src/hydra_suite/detectkit/jobs/inference_stager.py tests/test_detectkit_inference_stager.py
git commit -m "feat(detectkit): stage dataset-inference predictions for review"
```

---

### Task 12: "Stage predictions for review" in the UI

**Files:**
- Modify: `src/hydra_suite/detectkit/gui/main_window.py`
- Modify: `src/hydra_suite/detectkit/gui/panels/tools_panel.py`
- Test: `tests/test_detectkit_inference_stager.py`

**Interfaces:**
- Consumes: `stage_predictions` (Task 11); `_dataset_predictions`,
  `_dataset_prediction_signature`, `_dataset_signature`,
  `_filter_detections_by_confidence`, `_effective_inference_settings`,
  `detectkit_resolve_inference_models` (all existing in `main_window.py`).
- Produces: `ToolsPanel.stage_predictions_requested` (`Signal()`),
  `DetectKitMainWindow._on_stage_predictions()`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_detectkit_inference_stager.py`:

```python
def _wired_window(qtbot, monkeypatch, tmp_path, predictions):
    """A DetectKitMainWindow with just enough stubbed to run the handler.

    Every modal is patched out: an unpatched QMessageBox in a GUI test hangs
    the suite rather than failing it. `_project` MUST be set -- the handler's
    first guard reads it, and a fresh window has it as None.
    """
    from types import SimpleNamespace

    from hydra_suite.detectkit.gui import main_window as mw
    from hydra_suite.detectkit.gui.models import OBBSource

    window = mw.DetectKitMainWindow()
    qtbot.addWidget(window)
    source = OBBSource(path=str(tmp_path / "src"), name="src")
    window._project = SimpleNamespace(
        project_dir=str(tmp_path), active_model_path="m.pt", sources=[source]
    )
    window._dataset_predictions = dict(predictions)
    window._dataset_prediction_signature = ("sig", "m.pt")

    monkeypatch.setattr(window, "_current_source_obj", lambda: source)
    monkeypatch.setattr(window, "_dataset_signature", lambda settings: ("sig", "m.pt"))
    monkeypatch.setattr(
        window, "_effective_inference_settings", lambda settings: SimpleNamespace(device="mps")
    )
    monkeypatch.setattr(
        window._tools_panel,
        "get_overlay_settings",
        lambda: SimpleNamespace(confidence_threshold=0.40),
    )
    monkeypatch.setattr(window, "_save_current_project", lambda: None)
    monkeypatch.setattr(window, "_sync_review_bar", lambda: None)
    monkeypatch.setattr(window, "_refresh_overlays", lambda keys=None: None)
    monkeypatch.setattr(
        mw, "detectkit_resolve_inference_models",
        lambda project, model_path: ("sequential_segment", "p.pt", "s.pt"),
    )
    monkeypatch.setattr(mw.QMessageBox, "information", staticmethod(lambda *a, **k: None))
    monkeypatch.setattr(mw.QMessageBox, "warning", staticmethod(lambda *a, **k: None))
    return mw, window


def _det(conf):
    return {"class_id": 0, "polygon_px": [(0, 0), (10, 0), (10, 10), (0, 10)], "confidence": conf}


def test_staging_action_refuses_when_no_predictions_are_held(qtbot, monkeypatch, tmp_path):
    mw, window = _wired_window(qtbot, monkeypatch, tmp_path, {})
    called: list = []
    monkeypatch.setattr(mw, "stage_predictions", lambda *a, **k: called.append(a))

    window._on_stage_predictions()

    assert called == []


def test_staging_action_stages_only_what_is_visible_at_the_slider(
    qtbot, monkeypatch, tmp_path
):
    """The floor-retained candidates the user never saw must not be staged.

    _dataset_predictions is held at INFERENCE_CONFIDENCE_FLOOR (0.01) so the
    slider stays useful without re-running the model. Staging the raw dict
    would stage candidates the user never reviewed while run.json claimed
    the slider value.
    """
    mw, window = _wired_window(
        qtbot, monkeypatch, tmp_path, {"/img/a.png": [_det(0.9), _det(0.02)]}
    )
    seen: dict = {}
    monkeypatch.setattr(
        mw, "stage_predictions",
        lambda src, project_dir, per_image, **kw: seen.update(per_image=per_image, kw=kw),
    )

    window._on_stage_predictions()

    assert [d["confidence"] for d in seen["per_image"]["/img/a.png"]] == [0.9]
    assert seen["kw"]["confidence"] == 0.40


def test_staging_action_records_the_real_kind_and_device(qtbot, monkeypatch, tmp_path):
    """OverlaySettings carries neither field; they come from the run's own
    resolution. A sequential_segment run staged as obb_direct would declare
    polygon labels at OBB level."""
    mw, window = _wired_window(qtbot, monkeypatch, tmp_path, {"/img/a.png": [_det(0.9)]})
    seen: dict = {}
    monkeypatch.setattr(
        mw, "stage_predictions",
        lambda src, project_dir, per_image, **kw: seen.update(kw),
    )

    window._on_stage_predictions()

    assert seen["inference_kind"] == "sequential_segment"
    assert seen["device"] == "mps"
```

- [ ] **Step 2: Run to verify they fail**

```bash
python -m pytest tests/test_detectkit_inference_stager.py -k staging_action -q
```

Expected: FAIL — `AttributeError: ... '_on_stage_predictions'`.

- [ ] **Step 3: Add the action**

In `tools_panel.py`, next to the existing Run Inference button:

```python
        self.btn_stage_predictions = QPushButton("Stage Predictions for Review")
        self.btn_stage_predictions.setToolTip(
            "Write the current predictions into a staged review, so they can be "
            "accepted or rejected frame by frame like an escalation result. "
            "Looking at predictions never creates reviewable state; this does."
        )
        self.btn_stage_predictions.clicked.connect(self.stage_predictions_requested)
```

with `stage_predictions_requested = Signal()` on the class, added next to the panel's
existing signals.

In `main_window.py` (importing `from ..jobs.inference_stager import stage_predictions`
at module scope so tests can monkeypatch it):

```python
    def _on_stage_predictions(self) -> None:
        """Turn the predictions currently on screen into a staged review.

        ON SCREEN, not in memory. `_dataset_predictions` is deliberately
        retained at INFERENCE_CONFIDENCE_FLOOR (0.01, models.py:12-14) so
        that moving the slider is useful without re-running the model; the
        slider is applied at DISPLAY time by
        `_filter_detections_by_confidence`. Staging the raw dict would stage
        hundreds of 0.01 candidates the user never saw, while run.json
        recorded the slider value -- staging something other than what was
        reviewed. Filter first, at exactly the displayed threshold.

        The kind and device likewise come from the same resolution the
        inference RUN used (`detectkit_resolve_inference_models` +
        `_effective_inference_settings`), not from OverlaySettings, which
        carries neither field.
        """
        source = self._current_source_obj()
        if source is None or self._project is None:
            QMessageBox.information(
                self, "Stage Predictions", "Open a project and select a source first."
            )
            return

        settings = self._tools_panel.get_overlay_settings()
        signature = self._dataset_signature(settings)
        if (
            not self._dataset_predictions
            or signature is None
            or signature != self._dataset_prediction_signature
        ):
            QMessageBox.information(
                self,
                "Stage Predictions",
                "There are no predictions for this source and model. Run "
                "inference across the source first.",
            )
            return

        threshold = float(settings.confidence_threshold)
        visible = {
            image_path: _filter_detections_by_confidence(dets, threshold)
            for image_path, dets in self._dataset_predictions.items()
        }

        model_path = signature[1]
        try:
            kind, _primary, _secondary = detectkit_resolve_inference_models(
                self._project, model_path
            )
            device = self._effective_inference_settings(settings).device
        except RuntimeError as exc:
            QMessageBox.information(self, "Stage Predictions", str(exc))
            return

        try:
            stage_predictions(
                source,
                self._project.project_dir,
                visible,
                model_path=str(model_path),
                inference_kind=str(kind),
                confidence=threshold,
                device=str(device),
            )
        except Exception as exc:
            QMessageBox.warning(self, "Stage Predictions", str(exc))
            return
        self._save_current_project()
        self._sync_review_bar()
        self._refresh_overlays(keys=("staged",))
```

Connect it where the other tools-panel signals are wired:

```python
        self._tools_panel.stage_predictions_requested.connect(self._on_stage_predictions)
```

- [ ] **Step 4: Run the tests**

```bash
python -m pytest tests/test_detectkit_inference_stager.py tests/test_detectkit_review_bar.py \
  tests/test_detectkit_prediction_preview.py tests/test_detectkit_inference_cancel.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
make format && make lint-moderate
git add -A src/hydra_suite/detectkit tests/test_detectkit_inference_stager.py
git commit -m "feat(detectkit): add Stage Predictions for Review"
```

---

# Phase 5 — Deletions and closeout (Tasks 13-15)

Last, so the new path is proven before the old one goes.

### Task 13: Delete the sibling-source path

`accept_pending_semantic_escalation` builds a whole new source from a staged SAM3 run.
That is the originally reported symptom: accepting a reviewed run produces a new source
when the run was a pass over labels that already exist. It is now dead — nothing calls
it — and deleting it is what makes SAM3 accept into the source it ran on.

`derived_from` and `original_path` on `OBBSource` are **retained**: the bundle exporter
reads them (`gui/project.py:356-362`) independently of escalation.

**Files:**
- Modify: `src/hydra_suite/detectkit/jobs/semantic_escalation.py`
- Modify: `src/hydra_suite/detectkit/jobs/sam2_escalation.py`
- Test: `tests/test_detectkit_sam2_escalation_wiring.py`

**Interfaces:**
- Consumes: nothing.
- Produces: nothing. `remove_staged_escalation_dir`, `_is_safe_to_delete`,
  `reject_pending_escalation` and `PENDING_ESCALATIONS_RELDIR` all stay —
  `finish_review` uses the delete helpers.

- [ ] **Step 1: Prove they are dead**

```bash
grep -rn "accept_pending_semantic_escalation\|_unique_source_name\|accept_pending_escalation" src/hydra_suite tests docs
```

Expected: only their definitions, plus any test that still imports them. If a live call
site remains, Task 7 or 10 was left incomplete — finish it before deleting.

- [ ] **Step 2: Delete**

Remove from `jobs/semantic_escalation.py`: `accept_pending_semantic_escalation` and
`_unique_source_name`, plus the now-unused `_link_or_copy` import inside the former.
Remove `accept_pending_escalation` from `jobs/sam2_escalation.py`.

`reject_pending_escalation` is **also dead** after Task 10: `finish_review` calls
`remove_staged_escalation_dir` directly, and the dialog that was its only other caller
is gone. Confirm with the Step 1 grep, then delete it too. **Keep**
`remove_staged_escalation_dir` and `_is_safe_to_delete` — `finish_review` depends on
them, and they are the only thing bounding a recursive rmtree whose path came from a
project file on disk. `tests/test_sam2_escalation.py`'s two out-of-bounds tests
(`test_reject_refuses_to_delete_out_of_bounds_staged_path`,
`test_reject_refuses_to_delete_filesystem_root`) must be re-pointed at
`remove_staged_escalation_dir` rather than deleted: they pin that bound.

- [ ] **Step 3: Port or delete the tests of the deleted functions**

Two existing files import the deleted functions at MODULE level, so after Step 2 they
fail at collection — the whole file, not one test:

- `tests/test_sam2_escalation.py:11` imports `accept_pending_escalation`. Six tests use
  it: `test_accept_pending_escalation_promotes_labels_and_resets_reviewed` (152),
  `test_accept_refuses_when_staged_labels_missing_files` (212),
  `test_accept_fails_loudly_if_clearing_source_labels_fails` (348),
  `test_accept_works_when_source_has_no_labels_dir` (381),
  `test_accept_without_pending_raises` (459), and the staging tests that assert a
  pending record afterwards.
- `tests/test_semantic_escalation_job.py:12` imports
  `accept_pending_semantic_escalation`; nine references, including
  `test_accept_creates_a_sibling_and_leaves_the_origin_untouched` (302),
  `test_accept_refuses_a_sam2_pending_record` (345),
  `test_accept_refuses_when_the_staging_dir_is_gone` (356).

Port each to the `staged_review` path rather than deleting wholesale — most assert
behaviour the new path must also have:

| Old test | Disposition |
|---|---|
| `..._promotes_labels_and_resets_reviewed` | Port to `accept_all(src, mode=MergeMode.OVERWRITE)` + `finish_review`; `reviewed` still goes False (an accept happened). |
| `..._refuses_when_staged_labels_missing_files` | Port: `accept_frame` raises when the staged label is missing (Task 7's guard). The old whole-source pre-check is gone by design — frame-granular accept cannot delete a label it has nothing staged for, because it only touches the frame it is given. Say so in the ported test's docstring. |
| `..._fails_loudly_if_clearing_source_labels_fails` | Port to `revert_review`, which is now the only rmtree-then-copytree on a source's labels. |
| `..._works_when_source_has_no_labels_dir` | Port to `accept_frame` on a source with no `labels/`. |
| `..._without_pending_raises` | Port: `accept_frame` raises `ValueError` via `_review_of`. |
| `..._creates_a_sibling_and_leaves_the_origin_untouched` | **Delete.** Its asserted behaviour is exactly what this task removes; Step 4's new test replaces it. |
| `..._refuses_a_sam2_pending_record` | **Delete.** Producer is no longer load-bearing — refusing by producer is the bug the refactor fixes. |
| `..._refuses_when_the_staging_dir_is_gone` | Port to `accept_frame`. |

Every "delete" above must be justified in the commit message, not silently dropped.

- [ ] **Step 4: Add a regression test for the behaviour change**

Append to `tests/test_detectkit_sam2_escalation_wiring.py`:

```python
def test_accepting_a_sam3_review_does_not_create_a_sibling_source(tmp_path):
    """The originally reported symptom: accept used to spawn a new source."""
    import hydra_suite.detectkit.jobs.semantic_escalation as se

    assert not hasattr(se, "accept_pending_semantic_escalation")
    assert not hasattr(se, "_unique_source_name")
```

- [ ] **Step 5: Run the full DetectKit surface**

```bash
python -m pytest tests/test_detectkit_sam2_escalation_wiring.py tests/test_detectkit_staged_review.py \
  tests/test_detectkit_review_bar.py tests/test_detectkit_project.py \
  tests/test_detectkit_source_manager_dialog.py \
  tests/test_sam2_escalation.py tests/test_semantic_escalation_job.py -q
```

Expected: PASS, with no collection errors. A collection error here means Step 3 was
skipped.

- [ ] **Step 6: Commit**

```bash
make format && make lint-moderate
git add -A src/hydra_suite/detectkit tests
git commit -m "refactor(detectkit): delete the SAM3 sibling-source accept path"
```

---

### Task 14: Remove the transitional alias and rename the overlay key

Task 4 kept `PendingEscalation = StagedReview` so nothing broke mid-plan, and Task 4
kept the canvas layer key as `"escalation"` so a bisect through the model rename stayed
readable. Both are now free to go — the key becomes `"staged"`, which is what the layer
actually holds now that inference stages into it too.

**Files:**
- Modify: `src/hydra_suite/detectkit/gui/models.py`
- Modify: `src/hydra_suite/detectkit/gui/overlays/providers.py`
- Modify: `src/hydra_suite/detectkit/gui/main_window.py`
- Test: `tests/test_detectkit_overlay_providers.py`, `tests/test_detectkit_overlay_golden.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `StagedReviewProvider.key == "staged"`; no `PendingEscalation` name.

- [ ] **Step 1: Confirm the alias is unused**

```bash
grep -rn "PendingEscalation" src/hydra_suite tests docs
```

Expected: only the alias line in `models.py`. Fix any remaining hit first.

- [ ] **Step 2: Delete the alias and rename the key**

Delete the two `PendingEscalation = StagedReview` lines (the alias and its comment).
In `providers.py`, change `key = "escalation"` to `key = "staged"`. Then:

```bash
grep -rn '"escalation"' src/hydra_suite/detectkit tests
```

and update every `_refresh_overlays(keys=("gt", "escalation"))` /
`keys=("escalation",)` call in `main_window.py` to `"staged"`, plus the golden test's
expected key set.

- [ ] **Step 3: Rename `_refresh_escalation_overlay`**

```python
    def _refresh_staged_overlay(self) -> None:
        self._refresh_overlays(keys=("staged",))
```

and update its call sites (`grep -rn "_refresh_escalation_overlay" src tests`).

- [ ] **Step 4: Run the overlay tests**

```bash
python -m pytest tests/test_detectkit_overlay_providers.py tests/test_detectkit_overlay_golden.py \
  tests/test_detectkit_overlay_layer.py tests/test_detectkit_canvas.py \
  tests/test_detectkit_canvas_dual_layer.py tests/test_detectkit_show_image_multi_level.py \
  tests/test_detectkit_review_bar.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
make format && make lint-moderate
git add -A src/hydra_suite/detectkit tests
git commit -m "refactor(detectkit): drop the PendingEscalation alias, key the layer 'staged'"
```

---

### Task 15: Documentation and spec closeout

**Files:**
- Modify: `docs/superpowers/specs/2026-08-31-detectkit-frame-granular-review-design.md`
- Modify: `docs/superpowers/specs/2026-08-31-detectkit-overlay-layer-registry-design.md`
- Modify: `docs/user-guide/` DetectKit review documentation (find it with the grep below)
- Move: this plan + the design spec into their `done/` subfolders

**Interfaces:** none.

- [ ] **Step 1: Find every doc that describes the old review flow**

```bash
grep -rln "Review Escalations\|sibling source\|pending escalation" docs
```

- [ ] **Step 2: Update the user-facing docs**

In each hit, replace the per-source accept/reject description with the frame-granular
one: the review bar's four operations, "next undecided", the progress counter, that
accepts apply immediately and appear on the ground-truth layer, that **Revert Review**
exists but only while the review is open, that accepting polygons into an OBB source
promotes it, and that SAM3 now accepts into the source it ran on rather than creating a
sibling. Mention that dataset predictions can be staged for the same review flow.

- [ ] **Step 3: Correct the design spec's §3, and only that**

The design spec's "Relationship to the overlay registry spec" section is **already
correct** — it was amended in `32c5e8ed` after the registry shipped and is more
detailed than any replacement summary (it covers the `"pred"` provider staying
independent, `set_layer(layer)`'s single argument, and why the "layer count shrinks"
framing does not hold post-registry). **Leave it alone.** Add only a one-line note that
`StagedEscalationProvider` is now `StagedReviewProvider` with key `"staged"`, serving
all three producers.

The correction that IS needed is §3. It says a staged level *below* the source's is
handled by `derive_down`. That case is an upward re-tag (an OBB quad encoded as a
4-point polygon), which `derive_down` refuses by design. Replace the sentence with a
reference to `staged_review._lift`, and note that `merge_records` stays strict on
purpose.

In `2026-08-31-detectkit-overlay-layer-registry-design.md`, add the same one-line
provider rename note.

- [ ] **Step 4: Set the status headers and move both docs**

Set the design spec's `**Status:**` line to
`Shipped — merged to main (<merge-sha>)` and move both files:

```bash
git mv docs/superpowers/specs/2026-08-31-detectkit-frame-granular-review-design.md \
       docs/superpowers/specs/done/
git mv docs/superpowers/plans/2026-08-31-detectkit-frame-granular-review.md \
       docs/superpowers/plans/done/
```

Per the repo's docs lifecycle rule, do this in the merge commit/PR, and only once every
checkbox above is ticked.

- [ ] **Step 5: Verify the docs build**

```bash
make docs-check
```

Expected: strict mkdocs build passes; terminology check clean.

- [ ] **Step 6: Run the whole DetectKit + AL test surface one last time**

```bash
python -m pytest tests/test_al_merge.py tests/test_al_label_reader.py tests/test_al_labels.py \
  tests/test_al_escalation.py tests/test_semantic_masks.py \
  tests/test_detectkit_staged_review.py tests/test_detectkit_inference_stager.py \
  tests/test_detectkit_review_bar.py tests/test_detectkit_models.py \
  tests/test_detectkit_overlay_providers.py tests/test_detectkit_overlay_golden.py \
  tests/test_detectkit_overlay_layer.py tests/test_detectkit_canvas.py \
  tests/test_detectkit_canvas_dual_layer.py tests/test_detectkit_show_image_multi_level.py \
  tests/test_detectkit_dataset_panel.py tests/test_detectkit_dataset_panel_widget.py \
  tests/test_detectkit_project.py tests/test_detectkit_sam2_escalation_wiring.py \
  tests/test_sam2_escalation.py tests/test_semantic_escalation_job.py \
  tests/test_obbsource_reviewed.py tests/test_detectkit_staged_escalation_overlay.py -q
```

The last four are the files that reference the renamed/deleted surfaces from outside
the obvious DetectKit set. Each fails at COLLECTION if missed, so a green run of the
earlier list alone proves nothing about them.

Expected: PASS. Note the batching gotcha: run this list as one invocation and compare
the failure SET, not the count, against the same list run on the base commit.

- [ ] **Step 7: Commit**

```bash
git add -A docs
git commit -m "docs(detectkit): document frame-granular review; close out both specs"
```

---

## Spec Coverage Check

| Spec section | Task(s) |
|---|---|
| §1 `StagedReview` replacing three concepts | 4 |
| §1 staging layout as contract (`decisions.json`, `labels_before/`) | 5 |
| §1 `pending_escalation` → `staged_review` rename | 4 |
| §1 backwards compatibility (old keys, old `primer_*` names) | 4 |
| §2 `merge_records`, `MergeMode` | 3 |
| §2 `polygon_iou` move, behaviour unchanged | 1 |
| §3 level promotion, `_polygon_points` reuse | 7 |
| §3 staged-below-source handling (spec says `derive_down`; it is a LIFT — `_lift`, spec amended in Task 15) | 7, 15 |
| §4 four operations, keyed by relative path, applied immediately | 7 |
| §4 first-accept snapshot + Revert | 5, 7, 10 |
| §4 `decisions.json` in the staging dir | 5 |
| §4 review completion removes staging, clears the field | 7, 10 |
| §5 inference as a producer, preview path unaffected | 11, 12 |
| §6 review bar, four operations, next-undecided, counter | 9, 10 |
| §6 both layers refreshed on accept | 8, 10 |
| §6 dialog retired, re-threshold moved to the bar | 10 |
| "What gets deleted" | 10, 13, 14 |
| Testing 1 (merge rule + invariant) | 3 |
| Testing 2 (`polygon_iou` move, tests untouched) | 1 |
| Testing 3 (promotion, no drift, no OBB read-back) | 7 |
| Testing 4 (revert byte-identical after a mixed review) | 7 |
| Testing 5 (producer-agnosticism) | 7 |
| Testing 6 (backwards compatibility) | 4 |
| Non-goal: no per-instance identity introduced | — (nothing in this plan adds one) |
| Gaps the spec left open: label reader, class-id spaces, verbatim vs re-encode, snapshot completeness, decided-frame rendering | 2, 6, 7, 5, 8 |
