# AL Escalated Multi-Format Export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make TrackerKit's active-learning export emit every geometry level the detection model can support (polygon, obb, aabb) as directly-ingestible DetectKit sources, and converge TrackerKit's AL implementation onto DetectKit's shared core.

**Architecture:** One AL round writes up to three sibling *source roots*, each a valid DetectKit source with hardlinked images and a stamped `GeometryLevel`. A new shared `data/al/{escalation,labels,export}.py` becomes the single authority for geometry escalation, label writing, and dataset layout; both TrackerKit's `export_dataset` and DetectKit's `run_active_learning` collapse onto it. Acquisition scoring moves from within-run min-max normalization to absolute per-channel floors, after three signals orphaned by an earlier refactor are ported forward.

**Tech Stack:** Python 3.11+, numpy, OpenCV (`cv2`), pandas, PySide6 (Qt, UI tasks only), pytest, ultralytics YOLO.

**Spec:** `docs/superpowers/specs/2026-08-17-al-escalated-multi-format-export-design.md`

## Global Constraints

- **Layer direction is absolute.** `data/`, `core/`, `utils/`, `training/`, `runtime/` must never import from an app package (`trackerkit`, `detectkit`, `posekit`, `classkit`, `refinekit`, `filterkit`). App packages may import downward. The one existing carve-out (`data/al/candidate_pool.py` importing `FilterKitCore`) stays as documented; do not add new ones.
- **The tracking hot path must stay byte-identical.** Every change is export-only or behind an opt-in flag defaulting to off. `emit_native_geometry` defaults `False`; the bgsub contour return is opt-in.
- **Never claim a geometry level the model did not produce.** A rotated quad is not a polygon. See the level table in Task 8.
- **All new config knobs flow through the shared `build_engine_params`** (`src/hydra_suite/trackerkit/engine_params.py`), never around it, so CLI and GUI cannot diverge.
- **Work in a git worktree branched from local HEAD**, never fresh-from-origin: `git worktree add .worktrees/al-escalation -b feat/al-escalated-export HEAD`.
- **Commit as the repo's configured git user.** Do not add a `Co-Authored-By: Claude` trailer.
- **Before any heavy run** (equivalence harness, inference): kill stale `sleap`/`hydra` processes first; never touch other processes.
- Run tests with `python -m pytest`. Format with `make format` before committing. The base suite has ~24 pre-existing failures — use a delta gate (compare failures before and after), not an absolute zero.

---

## File Structure

**Created:**
- `src/hydra_suite/utils/geometry_levels.py` — `GeometryLevel` enum + `classify_label_line`, at the bottom layer so Data and Training can both import them.
- `src/hydra_suite/data/al/escalation.py` — `LabelRecord`, `records_from_obb_result`, `derive_down`. The only place a level conversion is written.
- `src/hydra_suite/data/al/labels.py` — `write_label_file`. The only place a label line is formatted.
- `src/hydra_suite/data/al/export.py` — `export_al_dataset`. The only place the three-root layout is written.
- `tests/test_al_escalation.py`, `tests/test_al_labels.py`, `tests/test_al_export.py`, `tests/test_al_absolute_scoring.py`, `tests/test_bgsub_contours.py`, `tests/test_al_strict_labels.py`.

**Modified:**
- `src/hydra_suite/training/geometry_levels.py` — re-export moved symbols.
- `src/hydra_suite/detectkit/jobs/al_worker.py` — drop `_write_geometry_label`, call shared modules.
- `src/hydra_suite/core/inference/config.py:1054` — `build_obb_only_config` forwards task + native geometry.
- `src/hydra_suite/core/background/measure.py:183` — opt-in contour return.
- `src/hydra_suite/core/inference/stages/bgsub.py:133` — populate `OBBResult.polygons`.
- `src/hydra_suite/data/dataset_generation.py` — exporter collapses onto shared core; legacy scorer retired.
- `src/hydra_suite/data/al/{signals,acquisition}.py` — fragmentation channel, absolute floors.
- `src/hydra_suite/core/post/dataset_export.py` — frame shape, manifest passthrough.
- `src/hydra_suite/trackerkit/config/schemas.py`, `engine_params.py`, `gui/panels/dataset_panel.py`, `gui/workers/dataset_worker.py`.
- `src/hydra_suite/detectkit/gui/dialogs/active_learning.py`.

---

## Phase 1 — Shared escalation core

### Task 1: Relocate `GeometryLevel` to the utils layer

`data/al/` needs `GeometryLevel`, but it currently lives in `training/geometry_levels.py`. Data importing Training is a lateral move the layering rules do not sanction. Move the two pure symbols down to `utils/` and re-export them so every existing import keeps working.

**Files:**
- Create: `src/hydra_suite/utils/geometry_levels.py`
- Modify: `src/hydra_suite/training/geometry_levels.py:1-60`
- Test: `tests/test_geometry_levels.py` (exists — add to it)

**Interfaces:**
- Consumes: nothing.
- Produces: `hydra_suite.utils.geometry_levels.GeometryLevel` (IntEnum: `AABB=0`, `OBB=1`, `POLYGON=2`, with `.label` property and `.from_str(value)` staticmethod) and `classify_label_line(field_count: int) -> str` returning one of `"aabb"`, `"four_point"`, `"polygon"`, `"invalid"`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_geometry_levels.py`:

```python
def test_geometry_level_importable_from_utils_and_training():
    from hydra_suite.training.geometry_levels import GeometryLevel as TrainingLevel
    from hydra_suite.utils.geometry_levels import GeometryLevel as UtilsLevel

    # Same object, not a copy -- otherwise IntEnum identity checks break
    # across the two import paths.
    assert TrainingLevel is UtilsLevel
    assert UtilsLevel.POLYGON > UtilsLevel.OBB > UtilsLevel.AABB
    assert UtilsLevel.from_str("polygon") is UtilsLevel.POLYGON
    assert UtilsLevel.POLYGON.label == "polygon"


def test_classify_label_line_importable_from_utils():
    from hydra_suite.utils.geometry_levels import classify_label_line

    assert classify_label_line(5) == "aabb"
    assert classify_label_line(9) == "four_point"
    assert classify_label_line(11) == "polygon"
    assert classify_label_line(6) == "invalid"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_geometry_levels.py::test_geometry_level_importable_from_utils_and_training -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'hydra_suite.utils.geometry_levels'`

- [ ] **Step 3: Create the utils module**

Create `src/hydra_suite/utils/geometry_levels.py` by moving the `GeometryLevel` class and `classify_label_line` function verbatim out of `src/hydra_suite/training/geometry_levels.py`:

```python
"""Geometry-level vocabulary for polygon-first labels.

A label line stays ``class_id`` followed by a normalized point list. The
information content of a source is captured by a totally-ordered level:

    aabb  <  obb  <  polygon

Downward derivation (polygon -> minAreaRect -> obb -> aabb) is lossless to the
target; upward derivation needs new information.

This lives in ``utils`` (the bottom layer) so both ``data.al`` and ``training``
can import it without a lateral dependency.
"""

from __future__ import annotations

from enum import IntEnum


class GeometryLevel(IntEnum):
    """Information content of a source's geometry, totally ordered."""

    AABB = 0
    OBB = 1
    POLYGON = 2

    @property
    def label(self) -> str:
        return self.name.lower()

    @staticmethod
    def from_str(value: str) -> "GeometryLevel":
        key = str(value).strip().lower()
        for level in GeometryLevel:
            if level.name.lower() == key:
                return level
        raise ValueError(f"Unknown geometry level: {value!r}")


def classify_label_line(field_count: int) -> str:
    """Classify one label line by its whitespace field count.

    Returns:
        - "aabb"       for 5 fields (class + cx cy w h),
        - "four_point" for 9 fields (class + 8 coords: OBB or quad polygon),
        - "polygon"    for an odd field count >= 7 encoding 3 or >=5 points,
        - "invalid"    otherwise.
    """
    if field_count == 5:
        return "aabb"
    if field_count == 9:
        return "four_point"
    coords = field_count - 1
    if field_count >= 7 and coords % 2 == 0:
        points = coords // 2
        if points >= 3 and points != 4:
            return "polygon"
    return "invalid"
```

- [ ] **Step 4: Re-export from the training module**

In `src/hydra_suite/training/geometry_levels.py`, delete the `GeometryLevel` class body and the `classify_label_line` function, and replace the `from enum import IntEnum` import with a re-export. The file keeps `SourceLevelScan`, `_classify_file`, and `scan_source_levels` unchanged:

```python
from hydra_suite.utils.geometry_levels import (  # noqa: F401  (re-exported)
    GeometryLevel,
    classify_label_line,
)
```

- [ ] **Step 5: Run the full geometry-level and DetectKit test files**

Run: `python -m pytest tests/test_geometry_levels.py tests/test_detectkit_al_worker.py -v`
Expected: PASS (all pre-existing tests still pass through the re-export)

- [ ] **Step 6: Commit**

```bash
make format
git add src/hydra_suite/utils/geometry_levels.py src/hydra_suite/training/geometry_levels.py tests/test_geometry_levels.py
git commit -m "refactor: move GeometryLevel to utils so data/ can import it"
```

---

### Task 2: `data/al/escalation.py` — the geometry escalation authority

**Files:**
- Create: `src/hydra_suite/data/al/escalation.py`
- Test: `tests/test_al_escalation.py`

**Interfaces:**
- Consumes: `GeometryLevel` from Task 1; `OBBResult` from `hydra_suite.core.inference.result`.
- Produces:
  - `@dataclass LabelRecord` with fields `class_id: int`, `confidence: float`, `points: np.ndarray` (`(P, 2)` float32, pixel space), `level: GeometryLevel`.
  - `records_from_obb_result(obb, native_level: GeometryLevel, keep: Sequence[int] | None = None) -> list[LabelRecord]`
  - `derive_down(records: Sequence[LabelRecord], target: GeometryLevel) -> list[LabelRecord]`
  - `achievable_levels(native_level: GeometryLevel) -> list[GeometryLevel]` returning the native level and everything below it, highest first.

- [ ] **Step 1: Write the failing test**

Create `tests/test_al_escalation.py`:

```python
import numpy as np
import pytest

from hydra_suite.core.inference.result import OBBResult
from hydra_suite.data.al.escalation import (
    LabelRecord,
    achievable_levels,
    derive_down,
    records_from_obb_result,
)
from hydra_suite.utils.geometry_levels import GeometryLevel


def _obb_result(polygons=None):
    """One detection: a 40x20 box centred at (100, 50), unrotated."""
    corners = np.array(
        [[[80.0, 40.0], [120.0, 40.0], [120.0, 60.0], [80.0, 60.0]]],
        dtype=np.float32,
    )
    return OBBResult(
        frame_idx=0,
        centroids=np.array([[100.0, 50.0]], dtype=np.float32),
        angles=np.array([0.0], dtype=np.float32),
        sizes=np.array([800.0], dtype=np.float32),
        shapes=np.array([[800.0, 2.0]], dtype=np.float32),
        confidences=np.array([0.9], dtype=np.float32),
        corners=corners,
        detection_ids=OBBResult.make_detection_ids(0, 1),
        class_ids=np.array([3], dtype=np.int64),
        polygons=polygons,
    )


def test_records_from_obb_uses_corners_at_obb_level():
    records = records_from_obb_result(_obb_result(), GeometryLevel.OBB)
    assert len(records) == 1
    assert records[0].level is GeometryLevel.OBB
    assert records[0].class_id == 3
    assert records[0].confidence == pytest.approx(0.9)
    assert records[0].points.shape == (4, 2)


def test_records_from_obb_uses_native_polygons_at_polygon_level():
    poly = [np.array([[80.0, 40.0], [120.0, 45.0], [110.0, 60.0]], dtype=np.float32)]
    records = records_from_obb_result(_obb_result(polygons=poly), GeometryLevel.POLYGON)
    assert records[0].level is GeometryLevel.POLYGON
    assert records[0].points.shape == (3, 2)


def test_records_from_obb_rejects_polygon_level_without_polygons():
    with pytest.raises(ValueError, match="native polygons"):
        records_from_obb_result(_obb_result(), GeometryLevel.POLYGON)


def test_records_from_obb_honours_keep_indices():
    obb = _obb_result()
    assert records_from_obb_result(obb, GeometryLevel.OBB, keep=[]) == []


def test_derive_down_polygon_to_obb_gives_four_points():
    poly = [
        np.array(
            [[80.0, 40.0], [120.0, 40.0], [120.0, 60.0], [80.0, 60.0], [100.0, 62.0]],
            dtype=np.float32,
        )
    ]
    records = records_from_obb_result(_obb_result(polygons=poly), GeometryLevel.POLYGON)
    derived = derive_down(records, GeometryLevel.OBB)
    assert derived[0].level is GeometryLevel.OBB
    assert derived[0].points.shape == (4, 2)
    # minAreaRect must enclose every source point.
    assert derived[0].points[:, 0].min() <= 80.0
    assert derived[0].points[:, 1].max() >= 62.0


def test_derive_down_to_aabb_is_axis_aligned():
    poly = [np.array([[80.0, 40.0], [125.0, 45.0], [110.0, 65.0]], dtype=np.float32)]
    records = records_from_obb_result(_obb_result(polygons=poly), GeometryLevel.POLYGON)
    derived = derive_down(records, GeometryLevel.AABB)
    pts = derived[0].points
    assert derived[0].level is GeometryLevel.AABB
    assert pts.shape == (4, 2)
    assert sorted(set(np.round(pts[:, 0], 4))) == [80.0, 125.0]
    assert sorted(set(np.round(pts[:, 1], 4))) == [40.0, 65.0]


def test_derive_down_refuses_to_escalate_upward():
    records = records_from_obb_result(_obb_result(), GeometryLevel.OBB)
    with pytest.raises(ValueError, match="upward"):
        derive_down(records, GeometryLevel.POLYGON)


def test_derive_down_to_same_level_is_identity():
    records = records_from_obb_result(_obb_result(), GeometryLevel.OBB)
    derived = derive_down(records, GeometryLevel.OBB)
    np.testing.assert_array_equal(derived[0].points, records[0].points)


def test_achievable_levels_is_highest_first():
    assert achievable_levels(GeometryLevel.POLYGON) == [
        GeometryLevel.POLYGON,
        GeometryLevel.OBB,
        GeometryLevel.AABB,
    ]
    assert achievable_levels(GeometryLevel.AABB) == [GeometryLevel.AABB]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_al_escalation.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'hydra_suite.data.al.escalation'`

- [ ] **Step 3: Write the implementation**

Create `src/hydra_suite/data/al/escalation.py`:

```python
"""Geometry escalation for active-learning labels.

The single authority for converting one detection's geometry between levels.
Downward derivation only: polygon -> minAreaRect -> obb -> aabb. Upward
derivation would invent information and is refused.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np

from hydra_suite.utils.geometry_levels import GeometryLevel


@dataclass
class LabelRecord:
    """One detection's exportable geometry, in frame pixel space."""

    class_id: int
    confidence: float
    points: np.ndarray  # (P, 2) float32, pixel space
    level: GeometryLevel


def achievable_levels(native_level: GeometryLevel) -> list[GeometryLevel]:
    """Levels derivable from `native_level`, highest first."""
    return [lvl for lvl in sorted(GeometryLevel, reverse=True) if lvl <= native_level]


def records_from_obb_result(
    obb,
    native_level: GeometryLevel,
    keep: Sequence[int] | None = None,
) -> list[LabelRecord]:
    """Build LabelRecords from an OBBResult at its native geometry level.

    `keep` optionally restricts output to those detection indices (used by the
    strict-label filter). Passing an empty sequence yields no records.
    """
    indices = range(obb.num_detections) if keep is None else [int(i) for i in keep]

    if native_level is GeometryLevel.POLYGON and obb.polygons is None:
        raise ValueError(
            "native polygons requested but OBBResult.polygons is None; the "
            "detection stage was not run with emit_native_geometry=True"
        )

    class_ids = obb.class_ids_or_zeros
    records: list[LabelRecord] = []
    for i in indices:
        if native_level is GeometryLevel.POLYGON:
            pts = np.asarray(obb.polygons[i], dtype=np.float32).reshape(-1, 2)
        else:
            pts = np.asarray(obb.corners[i], dtype=np.float32).reshape(-1, 2)
        records.append(
            LabelRecord(
                class_id=int(class_ids[i]),
                confidence=float(obb.confidences[i]),
                points=pts.copy(),
                level=native_level,
            )
        )
    return records


def _to_obb(points: np.ndarray) -> np.ndarray:
    """Minimum-area rotated rectangle enclosing `points`, as (4, 2) float32."""
    rect = cv2.minAreaRect(np.asarray(points, dtype=np.float32).reshape(-1, 1, 2))
    return cv2.boxPoints(rect).astype(np.float32)


def _to_aabb(points: np.ndarray) -> np.ndarray:
    """Axis-aligned bounding quad of `points`, as (4, 2) float32."""
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    x1, y1 = float(pts[:, 0].min()), float(pts[:, 1].min())
    x2, y2 = float(pts[:, 0].max()), float(pts[:, 1].max())
    return np.array(
        [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
        dtype=np.float32,
    )


def derive_down(
    records: Sequence[LabelRecord],
    target: GeometryLevel,
) -> list[LabelRecord]:
    """Derive `records` down to `target`. Refuses upward derivation."""
    out: list[LabelRecord] = []
    for rec in records:
        if target > rec.level:
            raise ValueError(
                f"cannot derive upward from {rec.level.label} to {target.label}: "
                "upward derivation requires information the model did not produce"
            )
        if target is rec.level:
            out.append(rec)
            continue
        if target is GeometryLevel.OBB:
            pts = _to_obb(rec.points)
        else:
            pts = _to_aabb(rec.points)
        out.append(
            LabelRecord(
                class_id=rec.class_id,
                confidence=rec.confidence,
                points=pts,
                level=target,
            )
        )
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_al_escalation.py -v`
Expected: PASS (10 tests)

- [ ] **Step 5: Commit**

```bash
make format
git add src/hydra_suite/data/al/escalation.py tests/test_al_escalation.py
git commit -m "feat(al): add shared geometry escalation authority"
```

---

### Task 3: `data/al/labels.py` — the shared label writer

Moves `_write_geometry_label` down out of `detectkit/jobs/al_worker.py` and teaches it to emit AABB in YOLO-detect's 5-field form (which `classify_label_line` recognizes as `"aabb"`).

**Files:**
- Create: `src/hydra_suite/data/al/labels.py`
- Modify: `src/hydra_suite/detectkit/jobs/al_worker.py:118-142`
- Test: `tests/test_al_labels.py`

**Interfaces:**
- Consumes: `LabelRecord`, `GeometryLevel` from Task 2 / Task 1.
- Produces: `write_label_file(path: Path, records: Sequence[LabelRecord], frame_size: tuple[int, int], level: GeometryLevel) -> None`, where `frame_size` is `(height, width)` matching `img.shape[:2]`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_al_labels.py`:

```python
import numpy as np

from hydra_suite.data.al.escalation import LabelRecord
from hydra_suite.data.al.labels import write_label_file
from hydra_suite.utils.geometry_levels import GeometryLevel, classify_label_line


def _record(points, level, class_id=0):
    return LabelRecord(
        class_id=class_id,
        confidence=0.9,
        points=np.asarray(points, dtype=np.float32),
        level=level,
    )


def test_obb_level_writes_nine_fields(tmp_path):
    path = tmp_path / "f.txt"
    rec = _record([[10, 20], [30, 20], [30, 40], [10, 40]], GeometryLevel.OBB)
    write_label_file(path, [rec], frame_size=(100, 200), level=GeometryLevel.OBB)

    fields = path.read_text().strip().split()
    assert len(fields) == 9
    assert classify_label_line(len(fields)) == "four_point"
    assert fields[0] == "0"
    # x normalized by width=200, y by height=100
    assert float(fields[1]) == 0.05
    assert float(fields[2]) == 0.20


def test_polygon_level_writes_point_list(tmp_path):
    path = tmp_path / "f.txt"
    rec = _record([[10, 20], [30, 20], [30, 40], [20, 50], [10, 40]], GeometryLevel.POLYGON)
    write_label_file(path, [rec], frame_size=(100, 200), level=GeometryLevel.POLYGON)

    fields = path.read_text().strip().split()
    assert len(fields) == 11
    assert classify_label_line(len(fields)) == "polygon"


def test_aabb_level_writes_five_field_yolo_detect(tmp_path):
    path = tmp_path / "f.txt"
    rec = _record([[10, 20], [30, 20], [30, 40], [10, 40]], GeometryLevel.AABB)
    write_label_file(path, [rec], frame_size=(100, 200), level=GeometryLevel.AABB)

    fields = path.read_text().strip().split()
    assert len(fields) == 5
    assert classify_label_line(len(fields)) == "aabb"
    assert float(fields[1]) == 0.10   # cx = 20/200
    assert float(fields[2]) == 0.30   # cy = 30/100
    assert float(fields[3]) == 0.10   # w  = 20/200
    assert float(fields[4]) == 0.20   # h  = 20/100


def test_class_id_is_preserved(tmp_path):
    path = tmp_path / "f.txt"
    rec = _record([[10, 20], [30, 20], [30, 40], [10, 40]], GeometryLevel.OBB, class_id=7)
    write_label_file(path, [rec], frame_size=(100, 200), level=GeometryLevel.OBB)
    assert path.read_text().split()[0] == "7"


def test_coordinates_are_clamped_to_unit_range(tmp_path):
    path = tmp_path / "f.txt"
    rec = _record([[-50, -50], [400, -50], [400, 400], [-50, 400]], GeometryLevel.OBB)
    write_label_file(path, [rec], frame_size=(100, 200), level=GeometryLevel.OBB)
    values = [float(v) for v in path.read_text().split()[1:]]
    assert all(0.0 <= v <= 1.0 for v in values)


def test_empty_records_writes_empty_file(tmp_path):
    path = tmp_path / "f.txt"
    write_label_file(path, [], frame_size=(100, 200), level=GeometryLevel.OBB)
    assert path.exists()
    assert path.read_text() == ""
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_al_labels.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'hydra_suite.data.al.labels'`

- [ ] **Step 3: Write the implementation**

Create `src/hydra_suite/data/al/labels.py`:

```python
"""YOLO label-file writing for active-learning datasets.

One label line is ``class_id`` followed by normalized coordinates. The encoding
depends on the level:

    aabb     -> 5 fields:  class cx cy w h        (YOLO detect)
    obb      -> 9 fields:  class x1 y1 ... x4 y4  (YOLO OBB)
    polygon  -> odd >= 7:  class x1 y1 ... xP yP  (YOLO segment)
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from hydra_suite.utils.geometry_levels import GeometryLevel

from .escalation import LabelRecord


def _normalized_points(points: np.ndarray, height: int, width: int) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2).copy()
    pts[:, 0] = np.clip(pts[:, 0] / float(width), 0.0, 1.0)
    pts[:, 1] = np.clip(pts[:, 1] / float(height), 0.0, 1.0)
    return pts


def _format_line(rec: LabelRecord, height: int, width: int, level: GeometryLevel) -> str:
    pts = _normalized_points(rec.points, height, width)
    if level is GeometryLevel.AABB:
        x1, y1 = float(pts[:, 0].min()), float(pts[:, 1].min())
        x2, y2 = float(pts[:, 0].max()), float(pts[:, 1].max())
        w, h = max(0.0, x2 - x1), max(0.0, y2 - y1)
        coords = [x1 + w * 0.5, y1 + h * 0.5, w, h]
    else:
        coords = list(pts.reshape(-1))
    body = " ".join(f"{float(v):.6f}" for v in coords)
    return f"{int(rec.class_id)} {body}\n"


def write_label_file(
    path: str | Path,
    records: Sequence[LabelRecord],
    frame_size: tuple[int, int],
    level: GeometryLevel,
) -> None:
    """Write one YOLO label file. `frame_size` is (height, width)."""
    height, width = int(frame_size[0]), int(frame_size[1])
    with Path(path).open("w") as fp:
        for rec in records:
            fp.write(_format_line(rec, height, width, level))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_al_labels.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Repoint DetectKit's AL worker at the shared writer**

In `src/hydra_suite/detectkit/jobs/al_worker.py`, delete `_write_geometry_label` and `_write_yolo_obb_label` (lines 118-142) and replace their single call site inside `run_active_learning` (the `_write_geometry_label(labels_dir / f"f_{fid:06d}.txt", dets, frame_size=img.shape[:2])` call) with a conversion to `LabelRecord` plus the shared writer. Add these imports at the top of the file:

```python
from hydra_suite.data.al.escalation import LabelRecord
from hydra_suite.data.al.labels import write_label_file
from hydra_suite.utils.geometry_levels import GeometryLevel
```

Add this module-level helper, which converts the worker's detection tuples
(6-tuple `(cx, cy, w, h, theta, conf)`, or 7-tuple with a trailing native
polygon) into records:

```python
def _records_from_detections(detections: list) -> list[LabelRecord]:
    """Convert detector tuples into LabelRecords, polygon-first."""
    records: list[LabelRecord] = []
    for rec in detections:
        cx, cy, ww, hh, theta, conf = rec[:6]
        polygon = rec[6] if len(rec) > 6 else None
        if polygon is not None:
            pts = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
            level = GeometryLevel.POLYGON
        else:
            pts = _detection_corners(cx, cy, ww, hh, theta)
            level = GeometryLevel.OBB
        records.append(
            LabelRecord(
                class_id=0,
                confidence=float(conf),
                points=pts,
                level=level,
            )
        )
    return records
```

Replace the call site with:

```python
        records = _records_from_detections(dets)
        write_label_file(
            labels_dir / f"f_{fid:06d}.txt",
            records,
            frame_size=img.shape[:2],
            level=GeometryLevel.from_str(req.export_level),
        )
```

- [ ] **Step 6: Run the DetectKit AL tests**

Run: `python -m pytest tests/test_detectkit_al_worker.py -v`
Expected: PASS. If a test imports `_write_geometry_label` directly, update it to import `write_label_file` from `hydra_suite.data.al.labels` and build `LabelRecord`s, matching the test style in `tests/test_al_labels.py`.

- [ ] **Step 7: Commit**

```bash
make format
git add src/hydra_suite/data/al/labels.py src/hydra_suite/detectkit/jobs/al_worker.py tests/test_al_labels.py tests/test_detectkit_al_worker.py
git commit -m "feat(al): extract shared label writer; add AABB 5-field encoding"
```

---

### Task 4: `data/al/export.py` — the three-root layout writer

**Files:**
- Create: `src/hydra_suite/data/al/export.py`
- Test: `tests/test_al_export.py`

**Interfaces:**
- Consumes: `LabelRecord`, `derive_down`, `achievable_levels` (Task 2); `write_label_file` (Task 3).
- Produces:
  - `@dataclass ExportedFrame` with `frame_id: int`, `image_name: str`, `records: list[LabelRecord]`, `is_context: bool`, `drops: dict[str, int]`.
  - `export_al_dataset(*, round_dir, frames, images, native_level, levels, class_names, provenance) -> dict` returning a manifest dict. `images` maps `frame_id -> np.ndarray` (BGR). `levels` is the requested subset of `achievable_levels(native_level)`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_al_export.py`:

```python
import json

import numpy as np
import pytest

from hydra_suite.data.al.escalation import LabelRecord
from hydra_suite.data.al.export import ExportedFrame, export_al_dataset
from hydra_suite.utils.geometry_levels import GeometryLevel


def _frame(frame_id=0, is_context=False, drops=None):
    poly = np.array([[10, 20], [30, 20], [30, 40], [10, 45]], dtype=np.float32)
    return ExportedFrame(
        frame_id=frame_id,
        image_name=f"f{frame_id:06d}.jpg",
        records=[
            LabelRecord(
                class_id=0,
                confidence=0.9,
                points=poly,
                level=GeometryLevel.POLYGON,
            )
        ],
        is_context=is_context,
        drops=drops or {},
    )


def _images(frame_ids):
    return {fid: np.zeros((100, 200, 3), dtype=np.uint8) for fid in frame_ids}


def _provenance():
    return {"model_path": "seg.pt", "model_task": "segment", "preset": "tracker_default"}


def test_writes_one_root_per_requested_level(tmp_path):
    manifest = export_al_dataset(
        round_dir=tmp_path / "al_round",
        frames=[_frame(0)],
        images=_images([0]),
        native_level=GeometryLevel.POLYGON,
        levels=[GeometryLevel.POLYGON, GeometryLevel.OBB, GeometryLevel.AABB],
        class_names=["ant"],
        provenance=_provenance(),
    )
    root = tmp_path / "al_round"
    for name in ("polygon", "obb", "aabb"):
        assert (root / name / "images" / "f000000.jpg").is_file()
        assert (root / name / "labels" / "f000000.txt").is_file()
        assert (root / name / "classes.txt").read_text() == "ant\n"
        assert (root / name / "source.json").is_file()
    assert {r["level"] for r in manifest["roots"]} == {"polygon", "obb", "aabb"}


def test_authoritative_root_is_the_native_level(tmp_path):
    export_al_dataset(
        round_dir=tmp_path / "al_round",
        frames=[_frame(0)],
        images=_images([0]),
        native_level=GeometryLevel.POLYGON,
        levels=[GeometryLevel.POLYGON, GeometryLevel.OBB],
        class_names=["ant"],
        provenance=_provenance(),
    )
    root = tmp_path / "al_round"
    poly_meta = json.loads((root / "polygon" / "source.json").read_text())
    obb_meta = json.loads((root / "obb" / "source.json").read_text())
    assert poly_meta["authoritative"] is True
    assert poly_meta["derived_from"] is None
    assert poly_meta["reviewed"] is True
    assert obb_meta["authoritative"] is False
    assert obb_meta["derived_from"] == "polygon"
    assert obb_meta["reviewed"] is False


def test_derived_root_images_are_hardlinks(tmp_path):
    export_al_dataset(
        round_dir=tmp_path / "al_round",
        frames=[_frame(0)],
        images=_images([0]),
        native_level=GeometryLevel.POLYGON,
        levels=[GeometryLevel.POLYGON, GeometryLevel.OBB],
        class_names=["ant"],
        provenance=_provenance(),
    )
    root = tmp_path / "al_round"
    a = (root / "polygon" / "images" / "f000000.jpg").stat()
    b = (root / "obb" / "images" / "f000000.jpg").stat()
    assert a.st_ino == b.st_ino


def test_refuses_level_above_native(tmp_path):
    with pytest.raises(ValueError, match="upward|not achievable"):
        export_al_dataset(
            round_dir=tmp_path / "al_round",
            frames=[_frame(0)],
            images=_images([0]),
            native_level=GeometryLevel.OBB,
            levels=[GeometryLevel.POLYGON],
            class_names=["ant"],
            provenance=_provenance(),
        )


def test_manifest_records_drop_accounting_and_context(tmp_path):
    manifest = export_al_dataset(
        round_dir=tmp_path / "al_round",
        frames=[
            _frame(0, drops={"lost": 2, "unmatched": 1}),
            _frame(5, is_context=True, drops={"lost": 0, "unmatched": 3}),
        ],
        images=_images([0, 5]),
        native_level=GeometryLevel.POLYGON,
        levels=[GeometryLevel.POLYGON],
        class_names=["ant"],
        provenance=_provenance(),
    )
    assert manifest["totals"]["dropped_lost"] == 2
    assert manifest["totals"]["dropped_unmatched"] == 4
    assert manifest["totals"]["frames_exported"] == 2
    assert manifest["selected_frame_ids"] == [0]
    assert manifest["context_frame_ids"] == [5]


def test_provenance_is_stamped_into_source_json(tmp_path):
    export_al_dataset(
        round_dir=tmp_path / "al_round",
        frames=[_frame(0)],
        images=_images([0]),
        native_level=GeometryLevel.POLYGON,
        levels=[GeometryLevel.POLYGON],
        class_names=["ant"],
        provenance=_provenance(),
    )
    meta = json.loads((tmp_path / "al_round" / "polygon" / "source.json").read_text())
    assert meta["provenance"]["model_task"] == "segment"
    assert meta["source_kind"] == "trackerkit_al"


def test_partial_write_is_not_left_behind_on_failure(tmp_path, monkeypatch):
    import hydra_suite.data.al.export as export_mod

    def boom(*args, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(export_mod, "write_label_file", boom)
    with pytest.raises(RuntimeError, match="disk full"):
        export_al_dataset(
            round_dir=tmp_path / "al_round",
            frames=[_frame(0)],
            images=_images([0]),
            native_level=GeometryLevel.POLYGON,
            levels=[GeometryLevel.POLYGON],
            class_names=["ant"],
            provenance=_provenance(),
        )
    assert not (tmp_path / "al_round").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_al_export.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'hydra_suite.data.al.export'`

- [ ] **Step 3: Write the implementation**

Create `src/hydra_suite/data/al/export.py`:

```python
"""Three-root active-learning dataset layout.

One AL round writes up to three sibling source roots -- one per geometry level
the model can support -- each independently a valid DetectKit source:

    <round_dir>/
      polygon/  images/ labels/ classes.txt source.json   (authoritative)
      obb/      images/ labels/ classes.txt source.json   (derived)
      aabb/     images/ labels/ classes.txt source.json   (derived)

Images in derived roots are hardlinks to the authoritative root's images, so
disk cost stays at roughly 1x regardless of how many levels are written.

The whole round is staged in a sibling temporary directory and moved into place
only on success, so a failure never registers a half-written source.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from hydra_suite.utils.geometry_levels import GeometryLevel

from .escalation import LabelRecord, achievable_levels, derive_down
from .labels import write_label_file

logger = logging.getLogger(__name__)

SOURCE_KIND = "trackerkit_al"
MANIFEST_SCHEMA_VERSION = 2


@dataclass
class ExportedFrame:
    """One frame's exportable content plus its strict-label drop accounting."""

    frame_id: int
    image_name: str
    records: list[LabelRecord]
    is_context: bool = False
    drops: dict[str, int] = field(default_factory=dict)


def _link_or_copy(src: Path, dst: Path) -> None:
    """Hardlink `src` to `dst`, falling back to a copy across devices."""
    try:
        dst.hardlink_to(src)
    except (OSError, NotImplementedError) as exc:
        logger.warning("Hardlink failed (%s); copying %s -> %s", exc, src, dst)
        shutil.copy2(src, dst)


def _write_root(
    root: Path,
    frames: Sequence[ExportedFrame],
    images: dict,
    level: GeometryLevel,
    class_names: Sequence[str],
    provenance: dict,
    authoritative_root: Path | None,
    native_level: GeometryLevel,
) -> dict:
    images_dir = root / "images"
    labels_dir = root / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    for frame in frames:
        img_dst = images_dir / frame.image_name
        if authoritative_root is None:
            cv2.imwrite(str(img_dst), images[frame.frame_id])
        else:
            _link_or_copy(authoritative_root / "images" / frame.image_name, img_dst)

        records = derive_down(frame.records, level)
        height, width = images[frame.frame_id].shape[:2]
        write_label_file(
            labels_dir / (Path(frame.image_name).stem + ".txt"),
            records,
            frame_size=(height, width),
            level=level,
        )

    (root / "classes.txt").write_text("\n".join(class_names) + "\n")

    meta = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "level": level.label,
        "native_level": native_level.label,
        "authoritative": authoritative_root is None,
        "derived_from": None if authoritative_root is None else native_level.label,
        "reviewed": authoritative_root is None,
        "source_kind": SOURCE_KIND,
        "class_names": list(class_names),
        "provenance": dict(provenance),
    }
    (root / "source.json").write_text(json.dumps(meta, indent=2))
    return meta


def export_al_dataset(
    *,
    round_dir: str | Path,
    frames: Sequence[ExportedFrame],
    images: dict,
    native_level: GeometryLevel,
    levels: Sequence[GeometryLevel],
    class_names: Sequence[str],
    provenance: dict,
) -> dict:
    """Write one AL round as up to three sibling source roots.

    Returns a manifest dict describing every root written plus round-level
    totals. Raises ValueError if any requested level exceeds `native_level`.
    """
    allowed = achievable_levels(native_level)
    for lvl in levels:
        if lvl not in allowed:
            raise ValueError(
                f"level {lvl.label!r} is not achievable from native level "
                f"{native_level.label!r}: upward derivation is refused"
            )

    round_path = Path(round_dir)
    staging = round_path.parent / (round_path.name + ".partial")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    # Highest level first, so the authoritative root exists before any derived
    # root tries to hardlink its images.
    ordered = sorted(set(levels), reverse=True)
    roots: list[dict] = []
    try:
        authoritative_root: Path | None = None
        for lvl in ordered:
            root = staging / lvl.label
            meta = _write_root(
                root,
                frames,
                images,
                lvl,
                class_names,
                provenance,
                authoritative_root,
                native_level,
            )
            meta["path"] = str(round_path / lvl.label)
            roots.append(meta)
            if authoritative_root is None:
                authoritative_root = root

        totals = {
            "frames_exported": len(frames),
            "dropped_lost": sum(int(f.drops.get("lost", 0)) for f in frames),
            "dropped_unmatched": sum(
                int(f.drops.get("unmatched", 0)) for f in frames
            ),
            "objects": sum(len(f.records) for f in frames),
        }
        manifest = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "round_dir": str(round_path),
            "native_level": native_level.label,
            "roots": roots,
            "totals": totals,
            "selected_frame_ids": [f.frame_id for f in frames if not f.is_context],
            "context_frame_ids": [f.frame_id for f in frames if f.is_context],
            "provenance": dict(provenance),
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2))
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    staging.rename(round_path)
    return manifest
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_al_export.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
make format
git add src/hydra_suite/data/al/export.py tests/test_al_export.py
git commit -m "feat(al): add three-root export layout with hardlinked images"
```

---

## Phase 2 — Inference plumbing (opt-in, byte-identical)

### Task 5: `build_obb_only_config` forwards task and native geometry

Today `build_obb_only_config` hardcodes `model_task="obb"`, so a segmentation checkpoint reaching the export runner trips the loud task-mismatch check at `core/inference/stages/obb.py:358`. Without this fix, polygon-level export is unreachable.

**Files:**
- Modify: `src/hydra_suite/core/inference/config.py:1054-1082`
- Test: `tests/test_inference_config.py` (exists — add to it; if absent, create `tests/test_obb_only_config.py` with the same tests)

**Interfaces:**
- Consumes: nothing new.
- Produces: `build_obb_only_config(model_path, *, compute_runtime="cpu", runtime_tier=None, confidence_threshold=0.25, iou_threshold=0.7, max_targets=8, mode="direct", model_task="obb", emit_native_geometry=False) -> InferenceConfig`.

- [ ] **Step 1: Write the failing test**

```python
def test_build_obb_only_config_forwards_model_task_and_native_geometry():
    from hydra_suite.core.inference.config import build_obb_only_config

    cfg = build_obb_only_config(
        "seg.pt",
        runtime_tier="cpu",
        model_task="segment",
        emit_native_geometry=True,
    )
    assert cfg.obb.direct.model_task == "segment"
    assert cfg.obb.emit_native_geometry is True


def test_build_obb_only_config_defaults_are_unchanged():
    from hydra_suite.core.inference.config import build_obb_only_config

    cfg = build_obb_only_config("m.pt", runtime_tier="cpu")
    assert cfg.obb.direct.model_task == "obb"
    assert cfg.obb.emit_native_geometry is False


def test_build_obb_only_config_rejects_unknown_task():
    import pytest

    from hydra_suite.core.inference.config import build_obb_only_config

    with pytest.raises(ValueError, match="model_task"):
        build_obb_only_config("m.pt", runtime_tier="cpu", model_task="pose")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_inference_config.py -k obb_only -v`
Expected: FAIL with `TypeError: build_obb_only_config() got an unexpected keyword argument 'model_task'`

- [ ] **Step 3: Write the implementation**

In `src/hydra_suite/core/inference/config.py`, change the `build_obb_only_config` signature and body:

```python
def build_obb_only_config(
    model_path: str,
    *,
    compute_runtime: str = "cpu",
    runtime_tier: str | None = None,
    confidence_threshold: float = 0.25,
    iou_threshold: float = 0.7,
    max_targets: int = 8,
    mode: str = "direct",
    model_task: str = "obb",
    emit_native_geometry: bool = False,
) -> InferenceConfig:
    """Detection-only InferenceConfig for one-shot / dataset OBB detection.

    ``model_task`` selects the checkpoint's head ("obb", "detect", "segment");
    it MUST match the checkpoint, which ``stages/obb.py`` verifies loudly.
    ``emit_native_geometry`` is the export-only opt-in that populates
    ``OBBResult.polygons`` with native contours (segment task only).
    """
    task = str(model_task).strip().lower()
    if task not in {"obb", "detect", "segment"}:
        raise ValueError(
            f"model_task must be one of 'obb', 'detect', 'segment'; got {model_task!r}"
        )
    params: dict = {
        "DETECTION_METHOD": "yolo_obb",
        "YOLO_OBB_MODE": mode,
        "YOLO_OBB_DIRECT_MODEL_PATH": model_path,
        "YOLO_OBB_DIRECT_TASK": task,
        "COMPUTE_RUNTIME": compute_runtime,
        "YOLO_CONFIDENCE_THRESHOLD": confidence_threshold,
        "YOLO_IOU_THRESHOLD": iou_threshold,
        "MAX_TARGETS": max_targets,
    }
    if runtime_tier:
        params["RUNTIME_TIER"] = runtime_tier
    cfg = build_inference_config_from_params(params)
    if emit_native_geometry:
        cfg.obb.emit_native_geometry = True
    return cfg
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_inference_config.py -k obb_only -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
make format
git add src/hydra_suite/core/inference/config.py tests/test_inference_config.py
git commit -m "fix(inference): forward model_task and emit_native_geometry from build_obb_only_config"
```

---

### Task 6: bgsub emits native contours

`core/background/measure.py:196` already runs `cv2.findContours` and discards the contours. Return them behind an opt-in so bgsub reaches polygon level, and so bgsub AL export stops being 100% fabricated boxes.

**Files:**
- Modify: `src/hydra_suite/core/background/measure.py:183-262`
- Modify: `src/hydra_suite/core/inference/stages/bgsub.py:133-222`
- Test: `tests/test_bgsub_contours.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `BackgroundMeasurer.detect_objects(fg_mask, frame_count, *, return_contours: bool = False)` returning the existing 4-tuple when `return_contours=False`, and a 5-tuple `(meas, sizes, shapes, confidences, contours)` when `True`, where `contours` is a `list[np.ndarray]` of `(P, 2)` float32 pixel-space points aligned row-for-row with `meas`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_bgsub_contours.py`:

```python
import numpy as np

from hydra_suite.core.background.measure import BackgroundMeasurer


def _params():
    return {
        "MAX_TARGETS": 4,
        "MIN_CONTOUR_AREA": 10,
        "MAX_CONTOUR_MULTIPLIER": 20,
        "ENABLE_SIZE_FILTERING": False,
    }


def _mask_with_two_blobs():
    mask = np.zeros((200, 200), dtype=np.uint8)
    mask[20:60, 30:90] = 255     # wide blob
    mask[120:170, 130:160] = 255  # tall blob
    return mask


def test_detect_objects_default_return_shape_is_unchanged():
    engine = BackgroundMeasurer(_params())
    result = engine.detect_objects(_mask_with_two_blobs(), 0)
    assert len(result) == 4


def test_detect_objects_returns_contours_when_requested():
    engine = BackgroundMeasurer(_params())
    meas, sizes, shapes, confs, contours = engine.detect_objects(
        _mask_with_two_blobs(), 0, return_contours=True
    )
    assert len(contours) == len(meas) == 2
    for contour in contours:
        assert contour.ndim == 2 and contour.shape[1] == 2
        assert contour.dtype == np.float32


def test_contours_stay_aligned_after_size_filtering():
    params = _params()
    params["ENABLE_SIZE_FILTERING"] = True
    params["MIN_OBJECT_SIZE"] = 1000
    params["MAX_OBJECT_SIZE"] = float("inf")
    engine = BackgroundMeasurer(params)
    meas, _sizes, _shapes, _confs, contours = engine.detect_objects(
        _mask_with_two_blobs(), 0, return_contours=True
    )
    assert len(contours) == len(meas)
    # Each surviving contour's centroid must sit near its own measurement.
    for m, contour in zip(meas, contours):
        assert abs(float(contour[:, 0].mean()) - float(m[0])) < 20.0


def test_contours_stay_aligned_after_max_targets_cap():
    params = _params()
    params["MAX_TARGETS"] = 1
    engine = BackgroundMeasurer(params)
    meas, _sizes, _shapes, _confs, contours = engine.detect_objects(
        _mask_with_two_blobs(), 0, return_contours=True
    )
    assert len(meas) == 1
    assert len(contours) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_bgsub_contours.py -v`
Expected: FAIL with `TypeError: detect_objects() got an unexpected keyword argument 'return_contours'`

- [ ] **Step 3: Thread contours through `detect_objects`**

In `src/hydra_suite/core/background/measure.py`, change the signature to accept the keyword and carry a parallel `contours` list through every subselection. The critical correctness point: the size filter and the `MAX_TARGETS` cap must reorder/filter `contours` in lockstep with `meas`.

```python
    def detect_objects(
        self,
        fg_mask: np.ndarray,
        frame_count: int,
        *,
        return_contours: bool = False,
    ) -> tuple:
        """Detect and measure objects from the final foreground mask.

        Returns (meas, sizes, shapes, confidences), or that tuple plus a
        parallel list of (P, 2) float32 pixel-space contours when
        `return_contours=True`. The contour list is filtered and reordered in
        lockstep with `meas`, so row i always describes detection i.
        """
        p = self.params
        cnts, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        N = p["MAX_TARGETS"]
        max_allowed_contours = N * p.get("MAX_CONTOUR_MULTIPLIER", 20)

        if len(cnts) > max_allowed_contours:
            logger.debug(
                f"Frame {frame_count}: Too many contours ({len(cnts)}), skipping."
            )
            return ([], [], [], [], []) if return_contours else ([], [], [], [])

        meas, sizes, shapes, confidences = [], [], [], []
        contours: list = []
        for c in cnts:
            area = cv2.contourArea(c)
            if area < p["MIN_CONTOUR_AREA"] or len(c) < 5:
                continue

            (cx, cy), (ax1, ax2), ang = cv2.fitEllipse(c)

            if ax1 < ax2:
                ax1, ax2 = ax2, ax1
                ang = (ang + 90) % 180

            confidence = np.nan
            ellipse_area = np.pi * (ax1 / 2.0) * (ax2 / 2.0)
            meas.append(np.array([cx, cy, np.deg2rad(ang)], np.float32))
            sizes.append(ellipse_area)
            shapes.append((ellipse_area, ax1 / ax2 if ax2 > 0 else 0))
            confidences.append(confidence)
            contours.append(np.asarray(c, dtype=np.float32).reshape(-1, 2))

        if meas and p.get("ENABLE_SIZE_FILTERING", False):
            min_size = p.get("MIN_OBJECT_SIZE", 0)
            max_size = p.get("MAX_OBJECT_SIZE", float("inf"))

            original_count = len(meas)
            keep = [i for i, s in enumerate(sizes) if min_size <= s <= max_size]
            meas = [meas[i] for i in keep]
            sizes = [sizes[i] for i in keep]
            shapes = [shapes[i] for i in keep]
            confidences = [confidences[i] for i in keep]
            contours = [contours[i] for i in keep]

            if len(meas) != original_count:
                logger.debug(
                    f"Size filtering: {original_count} -> {len(meas)} detections"
                )

        if len(meas) > N:
            idxs = np.argsort(sizes)[::-1][:N]
            meas = [meas[i] for i in idxs]
            sizes = [sizes[i] for i in idxs]
            shapes = [shapes[i] for i in idxs]
            confidences = [confidences[i] for i in idxs]
            contours = [contours[i] for i in idxs]

        if return_contours:
            return meas, sizes, shapes, confidences, contours
        return meas, sizes, shapes, confidences
```

Note the size-filter rewrite from `zip(*filtered)` to index selection: the old
form collapsed to `[], [], [], []` when nothing survived and needed a special
case. Index selection handles the empty case naturally and extends to
`contours` without another branch.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_bgsub_contours.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Populate `OBBResult.polygons` in `run_bgsub`**

In `src/hydra_suite/core/inference/stages/bgsub.py`, inside `run_bgsub`, replace the `detect_objects` call and add the polygon assignment before the `return OBBResult(...)`:

```python
    emit_polygons = bool(getattr(config, "emit_native_geometry", False))
    if emit_polygons:
        meas, sizes, shapes, confidences, contours = model.measurer.detect_objects(
            fg_mask, frame_idx, return_contours=True
        )
    else:
        meas, sizes, shapes, confidences = model.measurer.detect_objects(
            fg_mask, frame_idx
        )
        contours = None
    if not meas:
        return _empty_result(frame_idx)
```

Then, immediately before the existing `return OBBResult(...)`, build the result
into a local and attach polygons:

```python
    result = OBBResult(
        frame_idx=frame_idx,
        centroids=centroids,
        angles=angles,
        sizes=sizes_arr,
        shapes=shapes_arr,
        confidences=np.array(confidences, np.float32),
        corners=corners,
        detection_ids=detection_ids,
        # bg-sub has no class head -- all detections are the single generic
        # "object" class (0).
        class_ids=np.zeros(len(meas), dtype=np.int64),
    )
    if contours is not None:
        # Fall back to the fitted-ellipse corners for any degenerate contour,
        # mirroring the segment extractor's behaviour at stages/obb.py:1046.
        result.polygons = [
            contours[i] if contours[i].shape[0] >= 3 else corners[i].copy()
            for i in range(len(meas))
        ]
    return result
```

- [ ] **Step 6: Add the stage-level test**

Append to `tests/test_bgsub_contours.py`:

```python
def test_run_bgsub_leaves_polygons_none_by_default():
    """The tracking hot path must not compute or carry contours."""
    from hydra_suite.core.inference.config import BgSubConfig

    assert BgSubConfig.__dataclass_fields__ is not None
    cfg = BgSubConfig()
    assert getattr(cfg, "emit_native_geometry", False) is False
```

If `BgSubConfig` has no `emit_native_geometry` field, add it as
`emit_native_geometry: bool = False` with a comment matching `OBBConfig`'s:
export-only, default off to keep the hot path byte-identical.

- [ ] **Step 7: Run the bgsub test suite**

Run: `python -m pytest tests/test_bgsub_contours.py tests/ -k bgsub -v`
Expected: PASS; no new failures versus the pre-task baseline.

- [ ] **Step 8: Commit**

```bash
make format
git add src/hydra_suite/core/background/measure.py src/hydra_suite/core/inference/stages/bgsub.py src/hydra_suite/core/inference/config.py tests/test_bgsub_contours.py
git commit -m "feat(bgsub): emit native contours behind an export-only opt-in"
```

---

### Task 7: Export runner resolves the real detection source

**Files:**
- Modify: `src/hydra_suite/data/dataset_generation.py:413-450` (`_init_detection_runner`)
- Test: `tests/test_dataset_generation.py` (add)

**Interfaces:**
- Consumes: `build_obb_only_config` (Task 5).
- Produces:
  - `resolve_native_level(params: dict) -> GeometryLevel` — the level the configured detection source can produce.
  - `_init_detection_runner(params)` — unchanged name, now returns a runner for bgsub and for every YOLO task, and configures `emit_native_geometry` when the native level is POLYGON.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_dataset_generation.py`:

```python
import pytest

from hydra_suite.utils.geometry_levels import GeometryLevel


@pytest.mark.parametrize(
    "params,expected",
    [
        (
            {"DETECTION_METHOD": "yolo_obb", "YOLO_OBB_DIRECT_TASK": "segment"},
            GeometryLevel.POLYGON,
        ),
        (
            {"DETECTION_METHOD": "yolo_obb", "YOLO_OBB_DIRECT_TASK": "obb"},
            GeometryLevel.OBB,
        ),
        (
            {"DETECTION_METHOD": "yolo_obb", "YOLO_OBB_DIRECT_TASK": "detect"},
            GeometryLevel.AABB,
        ),
        ({"DETECTION_METHOD": "background_subtraction"}, GeometryLevel.POLYGON),
    ],
)
def test_resolve_native_level(params, expected):
    from hydra_suite.data.dataset_generation import resolve_native_level

    assert resolve_native_level(params) is expected


def test_resolve_native_level_uses_stage2_task_in_sequential_mode():
    from hydra_suite.data.dataset_generation import resolve_native_level

    params = {
        "DETECTION_METHOD": "yolo_obb",
        "YOLO_OBB_MODE": "sequential",
        "YOLO_OBB_DIRECT_TASK": "detect",
        "YOLO_OBB_STAGE2_TASK": "segment",
    }
    assert resolve_native_level(params) is GeometryLevel.POLYGON


def test_resolve_native_level_defaults_to_obb_for_unknown_method():
    from hydra_suite.data.dataset_generation import resolve_native_level

    assert resolve_native_level({"DETECTION_METHOD": "mystery"}) is GeometryLevel.OBB
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_dataset_generation.py -k resolve_native_level -v`
Expected: FAIL with `ImportError: cannot import name 'resolve_native_level'`

- [ ] **Step 3: Write the implementation**

In `src/hydra_suite/data/dataset_generation.py`, add the resolver and rewrite `_init_detection_runner`:

```python
from hydra_suite.utils.geometry_levels import GeometryLevel

_TASK_LEVELS = {
    "segment": GeometryLevel.POLYGON,
    "obb": GeometryLevel.OBB,
    "detect": GeometryLevel.AABB,
}


def resolve_native_level(params) -> GeometryLevel:
    """The geometry level the configured detection source can actually produce.

    Never claims a level the model did not compute: a rotated quad is OBB, not
    a polygon. bg-sub produces true foreground contours, so it reaches POLYGON.
    """
    method = str(params.get("DETECTION_METHOD", "background_subtraction")).lower()
    if method == "background_subtraction":
        return GeometryLevel.POLYGON
    if method != "yolo_obb":
        return GeometryLevel.OBB

    mode = str(params.get("YOLO_OBB_MODE", "direct")).strip().lower()
    if mode == "sequential":
        task = str(params.get("YOLO_OBB_STAGE2_TASK", "obb")).strip().lower()
    else:
        task = str(params.get("YOLO_OBB_DIRECT_TASK", "obb")).strip().lower()
    return _TASK_LEVELS.get(task, GeometryLevel.OBB)


def _init_detection_runner(params):
    """Build a detection-only InferenceRunner for dataset label extraction.

    Unlike the legacy version this supports every detection source, not just
    `yolo_obb`: returning None for bg-sub meant every exported label was a
    fabricated reference-size box.
    """
    method = str(params.get("DETECTION_METHOD", "background_subtraction")).lower()
    native_level = resolve_native_level(params)
    try:
        from ..core.inference.runner import InferenceRunner

        if method == "background_subtraction":
            from ..core.inference.config import build_inference_config_from_params

            cfg = build_inference_config_from_params(dict(params))
            if native_level is GeometryLevel.POLYGON and cfg.bgsub is not None:
                cfg.bgsub.emit_native_geometry = True
        else:
            from ..core.inference.config import build_obb_only_config

            mode = str(params.get("YOLO_OBB_MODE", "direct")).strip().lower()
            task = (
                str(params.get("YOLO_OBB_STAGE2_TASK", "obb"))
                if mode == "sequential"
                else str(params.get("YOLO_OBB_DIRECT_TASK", "obb"))
            ).strip().lower()
            model_path = str(
                params.get(
                    "YOLO_OBB_DIRECT_MODEL_PATH",
                    params.get("YOLO_MODEL_PATH", "yolo26s-obb.pt"),
                )
                or "yolo26s-obb.pt"
            )
            cfg = build_obb_only_config(
                model_path,
                runtime_tier=str(params.get("RUNTIME_TIER", "") or "") or None,
                confidence_threshold=float(
                    params.get("DATASET_YOLO_CONFIDENCE_THRESHOLD", 0.05)
                ),
                iou_threshold=float(params.get("DATASET_YOLO_IOU_THRESHOLD", 0.5)),
                max_targets=max(1, int(params.get("MAX_TARGETS", 8))),
                mode=mode,
                model_task=task,
                emit_native_geometry=(native_level is GeometryLevel.POLYGON),
            )

        runner = InferenceRunner(cfg)
        logger.info(
            "Detection runner initialized for dataset export (method=%s, level=%s)",
            method,
            native_level.label,
        )
        return runner
    except Exception as e:
        logger.warning(
            "Could not initialize detection runner: %s. "
            "Labels will fall back to reference-size approximation.",
            e,
        )
        return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_dataset_generation.py -k resolve_native_level -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
make format
git add src/hydra_suite/data/dataset_generation.py tests/test_dataset_generation.py
git commit -m "feat(al): resolve native geometry level; support bgsub and all YOLO tasks in export"
```

---

## Phase 3 — TrackerKit escalated export

### Task 8: `export_dataset` collapses onto the shared exporter

**Files:**
- Modify: `src/hydra_suite/data/dataset_generation.py:764-894` (`export_dataset`), deleting `_format_obb_corners`, `_compute_obb_corners`, and `_write_dataset_files`
- Modify: `src/hydra_suite/core/post/dataset_export.py:130-160`
- Test: `tests/test_dataset_generation.py`, `tests/test_dataset_export.py`

**Interfaces:**
- Consumes: `export_al_dataset`, `ExportedFrame` (Task 4); `records_from_obb_result`, `LabelRecord` (Task 2); `resolve_native_level` (Task 7).
- Produces: `export_dataset(..., export_levels: Sequence[GeometryLevel] | None = None) -> dict` — **return type changes from `str` to the manifest dict**. `generate_active_learning_dataset` returns `{"success", "num_frames", "dir", "manifest"}`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_dataset_generation.py`:

```python
def test_export_dataset_writes_three_roots_for_segmentation(tmp_path, monkeypatch):
    """A segmentation source yields polygon + obb + aabb roots."""
    import numpy as np
    import pandas as pd

    import hydra_suite.data.dataset_generation as dg
    from hydra_suite.utils.geometry_levels import GeometryLevel

    video = tmp_path / "clip.mp4"
    video.write_bytes(b"")
    csv_path = tmp_path / "tracks.csv"
    pd.DataFrame(
        {
            "FrameID": [0, 0],
            "TrackID": [1, 2],
            "X": [100.0, 150.0],
            "Y": [50.0, 60.0],
            "Theta": [0.0, 0.5],
            "State": ["tracked", "tracked"],
        }
    ).to_csv(csv_path, index=False)

    frame = np.zeros((200, 400, 3), dtype=np.uint8)
    monkeypatch.setattr(dg, "_open_video", lambda p: _FakeCap(frame, total=3))
    monkeypatch.setattr(dg, "_init_detection_runner", lambda params: None)
    monkeypatch.setattr(
        dg,
        "_detect_records_for_frames",
        lambda runner, frames, params, level: {
            0: dg.records_from_obb_result(_seg_obb_result(), level)
        },
    )

    manifest = dg.export_dataset(
        video_path=str(video),
        csv_path=str(csv_path),
        frame_ids=[0],
        output_dir=str(tmp_path / "out"),
        dataset_name="round",
        class_name="ant",
        params={"DETECTION_METHOD": "yolo_obb", "YOLO_OBB_DIRECT_TASK": "segment"},
        include_context=False,
    )

    assert manifest["native_level"] == "polygon"
    assert {r["level"] for r in manifest["roots"]} == {"polygon", "obb", "aabb"}
    round_dir = Path(manifest["round_dir"])
    assert (round_dir / "polygon" / "labels").is_dir()
    assert (round_dir / "aabb" / "labels").is_dir()


def test_export_dataset_writes_two_roots_for_obb_model(tmp_path, monkeypatch):
    """An OBB model must never produce a polygon root."""
    ...  # same fixture as above with YOLO_OBB_DIRECT_TASK="obb"
    # assert {r["level"] for r in manifest["roots"]} == {"obb", "aabb"}
```

Define the shared helpers once at the top of the test file:

```python
class _FakeCap:
    """Minimal cv2.VideoCapture stand-in returning one constant frame."""

    def __init__(self, frame, total):
        self._frame = frame
        self._total = total

    def isOpened(self):
        return True

    def get(self, prop):
        import cv2

        return {
            cv2.CAP_PROP_FRAME_COUNT: self._total,
            cv2.CAP_PROP_FRAME_WIDTH: self._frame.shape[1],
            cv2.CAP_PROP_FRAME_HEIGHT: self._frame.shape[0],
        }.get(prop, 0)

    def set(self, prop, value):
        return True

    def read(self):
        return True, self._frame.copy()

    def release(self):
        return None


def _seg_obb_result():
    import numpy as np

    from hydra_suite.core.inference.result import OBBResult

    corners = np.array(
        [[[90.0, 40.0], [110.0, 40.0], [110.0, 60.0], [90.0, 60.0]]], dtype=np.float32
    )
    result = OBBResult(
        frame_idx=0,
        centroids=np.array([[100.0, 50.0]], dtype=np.float32),
        angles=np.array([0.0], dtype=np.float32),
        sizes=np.array([400.0], dtype=np.float32),
        shapes=np.array([[400.0, 1.0]], dtype=np.float32),
        confidences=np.array([0.8], dtype=np.float32),
        corners=corners,
        detection_ids=OBBResult.make_detection_ids(0, 1),
        class_ids=np.array([0], dtype=np.int64),
    )
    result.polygons = [
        np.array([[90.0, 40.0], [110.0, 42.0], [108.0, 60.0]], dtype=np.float32)
    ]
    return result
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_dataset_generation.py -k three_roots -v`
Expected: FAIL — `export_dataset` returns a string path, and `_open_video` / `_detect_records_for_frames` do not exist.

- [ ] **Step 3: Rewrite `export_dataset`**

Replace `export_dataset` in `src/hydra_suite/data/dataset_generation.py`. Delete `_format_obb_corners`, `_compute_obb_corners`, `_write_dataset_files`, and `_make_dataset_dir` (the shared exporter owns layout now). Add the two seams the test monkeypatches (`_open_video`, `_detect_records_for_frames`) so the video and detector can be faked without touching disk:

```python
def _open_video(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    return cap


def _detect_records_for_frames(runner, frames, params, native_level):
    """Run detection over `frames` and return {frame_id: [LabelRecord]}."""
    if runner is None:
        return {}
    batch_size = _get_detector_batch_size(runner)
    out: dict[int, list] = {}
    frame_ids = sorted(frames)
    for start in range(0, len(frame_ids), batch_size):
        chunk = frame_ids[start : start + batch_size]
        images = [frames[fid] for fid in chunk]
        try:
            results = runner.detect_batch(images, frame_indices=list(chunk))
        except Exception as e:
            logger.warning("Detection failed for batch starting at %s: %s", chunk[0], e)
            continue
        for fid, obb in zip(chunk, results):
            out[fid] = records_from_obb_result(obb, native_level)
    return out


def export_dataset(
    video_path,
    csv_path,
    frame_ids,
    output_dir,
    dataset_name,
    class_name,
    params,
    include_context: bool = True,
    export_levels=None,
    class_names=None,
):
    """Export selected frames and labels as an escalated AL dataset.

    Returns the manifest dict from `export_al_dataset` (previously a directory
    path string).
    """
    from datetime import datetime

    import pandas as pd

    native_level = resolve_native_level(params)
    allowed = achievable_levels(native_level)
    levels = list(export_levels) if export_levels else list(allowed)
    unsupported = [lvl for lvl in levels if lvl not in allowed]
    if unsupported:
        raise ValueError(
            f"requested levels {[lvl.label for lvl in unsupported]} exceed the "
            f"native level {native_level.label!r} of the configured detector"
        )

    resolved_class_names = list(class_names) if class_names else [class_name or "object"]

    cap = _open_video(video_path)
    runner = _init_detection_runner(params)
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        df = pd.read_csv(csv_path)
        selected = set(int(f) for f in frame_ids)
        frames_to_export = _expand_frame_ids(frame_ids, include_context, total_frames)

        images: dict[int, np.ndarray] = {}
        detection_frames: dict[int, np.ndarray] = {}
        for fid in frames_to_export:
            read = _read_and_resize_frame(cap, fid, params, None)
            if read is None:
                continue
            original, for_detection, _shape = read
            images[fid] = original
            detection_frames[fid] = for_detection

        records_by_frame = _detect_records_for_frames(
            runner, detection_frames, params, native_level
        )

        resize_factor = params.get("RESIZE_FACTOR", 1.0)
        scale_back = _csv_scale_back(df, resize_factor, frame_width, frame_height)
        rows_by_frame = {int(fid): sub for fid, sub in df.groupby("FrameID")}

        exported: list[ExportedFrame] = []
        for fid in sorted(images):
            records, drops = _select_records_for_frame(
                rows_by_frame.get(fid),
                records_by_frame.get(fid, []),
                params,
                scale_back,
            )
            exported.append(
                ExportedFrame(
                    frame_id=fid,
                    image_name=f"f{fid:06d}.jpg",
                    records=records,
                    is_context=fid not in selected,
                    drops=drops,
                )
            )
    finally:
        cap.release()
        if runner is not None:
            runner.close()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{dataset_name}_{timestamp}" if str(dataset_name).strip() else timestamp
    round_dir = Path(output_dir).resolve() / name

    provenance = {
        "source_video": str(video_path),
        "source_csv": str(csv_path),
        "detection_method": params.get("DETECTION_METHOD"),
        "model_path": params.get("YOLO_OBB_DIRECT_MODEL_PATH"),
        "model_task": params.get("YOLO_OBB_DIRECT_TASK"),
        "export_confidence_threshold": params.get(
            "DATASET_YOLO_CONFIDENCE_THRESHOLD", 0.05
        ),
        "export_iou_threshold": params.get("DATASET_YOLO_IOU_THRESHOLD", 0.5),
        "acquisition_preset": params.get("DATASET_AL_PRESET"),
        "image_width": frame_width,
        "image_height": frame_height,
        "note": (
            "Labels come from a dedicated export detection pass at lower "
            "confidence than tracking, so they may differ from the tracked "
            "detections. Review before training."
        ),
    }

    manifest = export_al_dataset(
        round_dir=round_dir,
        frames=exported,
        images=images,
        native_level=native_level,
        levels=levels,
        class_names=resolved_class_names,
        provenance=provenance,
    )
    logger.info("Dataset exported to %s (%d frames)", round_dir, len(exported))
    return manifest
```

Add the imports at the top of the module:

```python
from hydra_suite.data.al.escalation import (
    LabelRecord,
    achievable_levels,
    records_from_obb_result,
)
from hydra_suite.data.al.export import ExportedFrame, export_al_dataset
```

`_select_records_for_frame` is implemented in Task 13. For this task, add a
temporary version that keeps existing behaviour (all non-NaN rows, nearest
detection) so the task is independently testable:

```python
def _select_records_for_frame(rows, frame_records, params, scale_back):
    """Placeholder pairing: returns detector records unchanged. Task 13 replaces
    this with mutual-exclusion matching and strict drops."""
    return list(frame_records), {"lost": 0, "unmatched": 0}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_dataset_generation.py -k "three_roots or two_roots" -v`
Expected: PASS

- [ ] **Step 5: Update `generate_active_learning_dataset` for the new return type**

In `src/hydra_suite/core/post/dataset_export.py`, change the `export_dataset` call and the returns:

```python
        manifest = export_dataset(
            video_path=video_path,
            csv_path=csv_path,
            frame_ids=selected_frames,
            output_dir=output_dir,
            dataset_name=dataset_name,
            class_name=class_name,
            params=params,
            include_context=include_context,
            export_levels=export_levels,
            class_names=class_names,
        )
        dataset_dir = manifest["round_dir"]
        if _stopped(should_stop):
            return {
                "success": False,
                "cancelled": True,
                "num_frames": len(selected_frames),
                "dir": dataset_dir,
                "manifest": manifest,
            }
        _emit(progress, 100, "Dataset generation complete!")
        return {
            "success": True,
            "num_frames": len(selected_frames),
            "dir": dataset_dir,
            "manifest": manifest,
        }
```

Add `export_levels=None` and `class_names=None` to the
`generate_active_learning_dataset` keyword-only signature and pass them through.

- [ ] **Step 6: Run the export-chain tests**

Run: `python -m pytest tests/test_dataset_export.py tests/test_session_export_chain.py tests/test_dataset_generation.py -v`
Expected: PASS. Update any assertion that expected `result["dir"]` to be the old flat layout.

- [ ] **Step 7: Commit**

```bash
make format
git add src/hydra_suite/data/dataset_generation.py src/hydra_suite/core/post/dataset_export.py tests/
git commit -m "feat(al): TrackerKit export writes escalated multi-level roots"
```

---

## Phase 4 — Acquisition scoring: port before delete

### Task 9: Port the fragmentation signal into `data/al/signals.py`

The legacy `_score_fragmented_detections` detects one animal split into two detections — the most tracker-specific signal in the file, weighted 0.3 in legacy, currently computed and discarded. `crowd` (max pairwise overlap) detects animals *touching*, a different phenomenon, and is not a substitute.

**Files:**
- Modify: `src/hydra_suite/data/al/signals.py`
- Modify: `src/hydra_suite/data/al/acquisition.py:15-75`
- Test: `tests/test_al_absolute_scoring.py`

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - `score_fragmentation(obb_corners: Sequence[np.ndarray], reference_major_axis: float | None = None) -> float` in `[0, 1]`.
  - `ALSignals.fragmentation_score: float = 0.0` (new field).
  - `AcquisitionWeights.fragmentation: float = 0.0` (new field); `tracker_default` gets `fragmentation=0.30`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_al_absolute_scoring.py`:

```python
import numpy as np

from hydra_suite.utils.geometry import obb_corners_from_dims


def _box(cx, cy, w=40.0, h=16.0, theta=0.0):
    return obb_corners_from_dims(cx, cy, w, h, theta)


def test_fragmentation_is_zero_for_well_separated_normal_boxes():
    from hydra_suite.data.al.signals import score_fragmentation

    boxes = [_box(100, 100), _box(300, 300), _box(500, 100)]
    assert score_fragmentation(boxes) == 0.0


def test_fragmentation_fires_on_two_small_overlapping_boxes():
    """One animal split into two half-size detections sitting on top of it."""
    from hydra_suite.data.al.signals import score_fragmentation

    boxes = [
        _box(300, 300),                      # normal-size animal
        _box(500, 300),                      # normal-size animal
        _box(100, 100, w=18.0, h=8.0),       # fragment
        _box(108, 102, w=18.0, h=8.0),       # its twin, close + small
    ]
    assert score_fragmentation(boxes) > 0.45


def test_fragmentation_needs_at_least_two_boxes():
    from hydra_suite.data.al.signals import score_fragmentation

    assert score_fragmentation([]) == 0.0
    assert score_fragmentation([_box(100, 100)]) == 0.0


def test_fragmentation_is_bounded():
    from hydra_suite.data.al.signals import score_fragmentation

    boxes = [_box(100, 100, w=5, h=3), _box(100, 100, w=5, h=3)]
    assert 0.0 <= score_fragmentation(boxes) <= 1.0


def test_tracker_default_preset_weights_fragmentation():
    from hydra_suite.data.al.acquisition import PRESETS

    assert PRESETS["tracker_default"].fragmentation == 0.30


def test_al_signals_carries_fragmentation_field():
    from hydra_suite.data.al.signals import ALSignals

    assert ALSignals(frame_id=0).fragmentation_score == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_al_absolute_scoring.py -v`
Expected: FAIL with `ImportError: cannot import name 'score_fragmentation'`

- [ ] **Step 3: Write the implementation**

Add to `src/hydra_suite/data/al/signals.py` (the heuristic is ported from
`data/dataset_generation.py:_score_fragmented_detections`, now operating on
corner arrays rather than raw measurements):

```python
from hydra_suite.utils.geometry import clamp01 as _clamp01


def score_fragmentation(
    obb_corners: Sequence[np.ndarray],
    reference_major_axis: float | None = None,
) -> float:
    """Return [0, 1] evidence that one object was split into several detections.

    A suspicious pair is close together, overlapping, and *both* smaller than
    the frame's typical detection -- the signature of a single animal broken
    into fragments. This is distinct from `score_crowd`, which measures genuine
    overlap between full-size neighbours.

    Ported from the legacy FrameQualityScorer so the signal keeps its meaning;
    the 0.45 suspicion gate and the pair weights are unchanged.
    """
    boxes = [
        np.asarray(c, dtype=np.float32).reshape(-1, 2)
        for c in obb_corners
        if c is not None
    ]
    boxes = [b for b in boxes if b.shape[0] >= 3]
    if len(boxes) < 2:
        return 0.0

    centers = [b.mean(axis=0) for b in boxes]
    major_axes = [
        float(
            max(
                np.linalg.norm(b[1] - b[0]),
                np.linalg.norm(b[2] - b[1]),
            )
        )
        for b in boxes
    ]
    typical = float(reference_major_axis or np.median(major_axes))
    typical = max(typical, 1.0)

    suspicious = 0
    best = 0.0
    for i, j in combinations(range(len(boxes)), 2):
        distance = float(np.linalg.norm(centers[i] - centers[j]))
        proximity = _clamp01(1.0 - distance / max(typical * 0.65, 1.0))
        overlap = _polygon_overlap_ratio(boxes[i], boxes[j])
        pair_major = (major_axes[i] + major_axes[j]) / 2.0
        smallness = _clamp01(1.0 - pair_major / typical)

        pair_score = _clamp01(0.5 * proximity + 0.3 * overlap + 0.2 * smallness)
        if pair_score >= 0.45:
            suspicious += 1
        best = max(best, pair_score)

    if best < 0.45:
        return 0.0
    return _clamp01(best + min(0.1 * max(suspicious - 1, 0), 0.2))
```

Add `fragmentation_score: float = 0.0` to `ALSignals` (immediately after
`crowd_score`).

In `src/hydra_suite/data/al/acquisition.py`, add `fragmentation: float = 0.0` to
`AcquisitionWeights`, add `"fragmentation": w.fragmentation` to the `channels`
dict in `_composite_score`, add `"fragmentation": "fragmentation_score"` to the
`attr_map` in `_channel_array`, and set `fragmentation=0.30` in the
`tracker_default` preset.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_al_absolute_scoring.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
make format
git add src/hydra_suite/data/al/signals.py src/hydra_suite/data/al/acquisition.py tests/test_al_absolute_scoring.py
git commit -m "feat(al): port fragmentation signal forward as a first-class channel"
```

---

### Task 10: Absolute floors replace min-max normalization

**Files:**
- Modify: `src/hydra_suite/data/al/acquisition.py:78-135` (`_channel_array`, `_minmax`, `_composite_score`)
- Modify: `src/hydra_suite/data/al/signals.py` (`score_uncertainty`, `score_count_deviation`)
- Test: `tests/test_al_absolute_scoring.py` (append)

**Interfaces:**
- Consumes: Task 9's channels.
- Produces:
  - `score_uncertainty(confidences, conf_floor=0.5) -> float` — **returns a single absolute severity now, not a `(mean, margin)` tuple**. `ALSignals.margin` is removed.
  - `score_count_deviation(n, expected) -> float` — asymmetric.
  - `_composite_score` no longer calls `_minmax`; `_minmax` is deleted.
  - `select(..., min_score=0.0)` gains `-> list[int]` unchanged, plus a new `explain(signals, weights) -> dict[str, float]` returning per-channel maxima for the empty-selection UI message.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_al_absolute_scoring.py`:

```python
import pytest

from hydra_suite.data.al.acquisition import AcquisitionWeights, explain, select
from hydra_suite.data.al.signals import (
    ALSignals,
    score_count_deviation,
    score_uncertainty,
)


def test_uncertainty_is_zero_above_the_floor():
    assert score_uncertainty([0.9, 0.8], conf_floor=0.5) == 0.0


def test_uncertainty_rises_as_confidence_falls():
    low = score_uncertainty([0.1], conf_floor=0.5)
    mid = score_uncertainty([0.3], conf_floor=0.5)
    assert 0.0 < mid < low <= 1.0


def test_uncertainty_of_all_nan_confidences_is_zero():
    """bg-sub emits all-NaN confidences; that must not read as 'uncertain'."""
    assert score_uncertainty([float("nan"), float("nan")]) == 0.0


def test_count_deviation_is_zero_on_exact_match():
    assert score_count_deviation(4, 4) == 0.0


def test_count_deviation_penalizes_undercount_twice_as_hard():
    under = score_count_deviation(2, 4)   # missed two animals
    over = score_count_deviation(6, 4)    # two spurious boxes
    assert under == pytest.approx(2 * over)


def test_composite_score_is_zero_for_a_clean_frame():
    """A frame with nothing wrong must score exactly 0, not 'least bad'."""
    clean = ALSignals(frame_id=0, n_detections=4, mean_confidence=0.95)
    weights = AcquisitionWeights(uncertainty=1.0)
    assert select([clean], weights=weights, k=1, min_score=0.01) == []


def test_min_score_gate_is_comparable_across_runs():
    weights = AcquisitionWeights(uncertainty=1.0)
    mild = ALSignals(frame_id=0, mean_confidence=0.45)
    severe = ALSignals(frame_id=9999, mean_confidence=0.05)

    # Severe alone, and severe alongside mild, must both clear a 0.5 gate --
    # under min-max normalization the lone frame would have normalized to 0.
    assert select([severe], weights=weights, k=5, min_score=0.5, probabilistic=False)
    picked = select(
        [mild, severe], weights=weights, k=5, min_score=0.5, probabilistic=False
    )
    assert picked == [9999]


def test_explain_reports_per_channel_maxima():
    weights = AcquisitionWeights(uncertainty=0.5, count=0.5)
    signals = [
        ALSignals(frame_id=0, mean_confidence=0.4, count_deviation=0.25),
        ALSignals(frame_id=1, mean_confidence=0.2, count_deviation=0.10),
    ]
    report = explain(signals, weights)
    assert report["uncertainty"] == pytest.approx(0.6)
    assert report["count"] == pytest.approx(0.25)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_al_absolute_scoring.py -k "uncertainty or count_deviation or composite or min_score or explain" -v`
Expected: FAIL — `score_uncertainty` returns a tuple; `explain` does not exist.

- [ ] **Step 3: Make the signal functions absolute**

In `src/hydra_suite/data/al/signals.py`, replace `score_uncertainty` and
`score_count_deviation`, and delete the `margin` field from `ALSignals`:

```python
def score_uncertainty(
    confidences: Sequence[float],
    conf_floor: float = 0.5,
) -> float:
    """Return absolute detection-uncertainty severity in [0, 1].

    Exactly 0 when the frame's mean confidence sits at or above `conf_floor` --
    a confidently-detected frame is not an active-learning candidate. All-NaN
    confidences (bg-sub, which has no confidence head) also score 0; treating
    "no information" as "maximum uncertainty" would make every bg-sub frame a
    candidate.
    """
    valid = [float(c) for c in confidences if c is not None and not math.isnan(c)]
    if not valid:
        return 0.0
    mean_conf = float(np.mean(valid))
    floor = max(float(conf_floor), 1e-6)
    if mean_conf >= floor:
        return 0.0
    return float(min(1.0, (floor - mean_conf) / floor))


def score_count_deviation(n: int, expected: int) -> float:
    """Return absolute count-mismatch severity in [0, 1]. 0 if expected <= 0.

    Asymmetric by design, preserving the legacy scorer's judgement: a missed
    animal is twice as bad as a spurious box, because a false negative removes
    training signal while a false positive is easy to delete during review.
    """
    if expected <= 0:
        return 0.0
    if n == expected:
        return 0.0
    if n < expected:
        return float(min(1.0, (expected - n) / float(expected)))
    return float(min(1.0, (n - expected) / float(expected)) * 0.5)
```

`ALSignals` keeps `mean_confidence` (still used for reporting) but loses
`margin`. Add a new field `uncertainty_score: float = 0.0` holding the absolute
value, since the raw `mean_confidence` is no longer sufficient to derive it
without knowing the floor.

- [ ] **Step 4: Remove min-max from the composite and add `explain`**

In `src/hydra_suite/data/al/acquisition.py`, delete `_minmax`, change
`_channel_array`'s `uncertainty` branch to read the precomputed
`s.uncertainty_score`, drop the `_minmax` call in `_composite_score`, and add
`explain`:

```python
def _composite_score(
    signals: Sequence[ALSignals],
    weights: AcquisitionWeights,
) -> np.ndarray:
    """Weighted sum of ABSOLUTE per-channel severities, in [0, 1].

    Deliberately NOT min-max normalized: a within-run rescale makes the top
    frame score ~1 however easy the video was, which both defeats `min_score`
    as a gate and makes scores incomparable across videos.
    """
    w = weights.normalized()
    channels = {
        "uncertainty": w.uncertainty,
        "nms_instability": w.nms_instability,
        "count": w.count,
        "crowd": w.crowd,
        "fragmentation": w.fragmentation,
        "edge": w.edge,
        "assignment": w.assignment,
        "track_loss": w.track_loss,
        "position_uncertainty": w.position_uncertainty,
    }
    score = np.zeros(len(signals), dtype=np.float64)
    for name, weight in channels.items():
        if weight <= 0:
            continue
        score += weight * np.clip(_channel_array(signals, name), 0.0, 1.0)
    return score


def explain(
    signals: Sequence[ALSignals],
    weights: AcquisitionWeights,
) -> dict[str, float]:
    """Per-channel maxima observed across `signals`, for weighted channels only.

    Used to tell the user WHY a selection came back empty under absolute
    scoring, rather than surfacing a bare "no frames met the criteria".
    """
    w = weights.normalized()
    out: dict[str, float] = {}
    for name in (
        "uncertainty",
        "nms_instability",
        "count",
        "crowd",
        "fragmentation",
        "edge",
        "assignment",
        "track_loss",
        "position_uncertainty",
    ):
        if getattr(w, name) <= 0:
            continue
        arr = _channel_array(signals, name)
        out[name] = float(arr.max()) if arr.size else 0.0
    return out
```

Export `explain` from `src/hydra_suite/data/al/__init__.py`.

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_al_absolute_scoring.py -v`
Expected: PASS (14 tests)

- [ ] **Step 6: Fix the DetectKit AL worker's now-stale unpacking**

`detectkit/jobs/al_worker.py:_frame_signals` unpacks
`mean_conf, margin = score_uncertainty(...)`. Change it to:

```python
    uncertainty = score_uncertainty(confidences, conf_floor=base_conf)
    ...
    signal = ALSignals(
        frame_id=frame_id,
        n_detections=len(detections),
        mean_confidence=float(np.mean(confidences)) if confidences else float("nan"),
        uncertainty_score=uncertainty,
        nms_instability=nms,
        count_deviation=count_dev,
        crowd_score=crowd,
        fragmentation_score=score_fragmentation(obb_corners),
        edge_score=edge,
    )
```

Add `score_fragmentation` to the imports from `hydra_suite.data.al.signals`.

- [ ] **Step 7: Run the DetectKit AL tests**

Run: `python -m pytest tests/test_detectkit_al_worker.py tests/test_detectkit_al_dialog.py -v`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
make format
git add src/hydra_suite/data/al/ src/hydra_suite/detectkit/jobs/al_worker.py tests/test_al_absolute_scoring.py
git commit -m "feat(al): absolute per-channel floors replace within-run min-max scoring"
```

---

### Task 11: Real frame shape, and bgsub weight renormalization

**Files:**
- Modify: `src/hydra_suite/data/dataset_generation.py:81-135` (`FrameQualityScorer.score_frame`)
- Modify: `src/hydra_suite/core/post/dataset_export.py:27-140`
- Test: `tests/test_dataset_generation.py` (append)

**Interfaces:**
- Consumes: `explain` (Task 10).
- Produces: `FrameQualityScorer(params, frame_shape: tuple[int, int] | None = None)`; `FrameQualityScorer.explain_scores() -> dict[str, float]`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_dataset_generation.py`:

```python
def test_edge_score_uses_real_frame_shape():
    """Regression: frame_shape=(1, 1) made edge_score ~1900 instead of [0, 1]."""
    from hydra_suite.data.dataset_generation import FrameQualityScorer

    scorer = FrameQualityScorer({"MAX_TARGETS": 2}, frame_shape=(1080, 1920))
    scorer.score_frame(
        0,
        detection_data={
            "confidences": [0.9, 0.9],
            "count": 2,
            "obb_corners": [
                [[100, 100], [140, 100], [140, 120], [100, 120]],
                [[900, 500], [940, 500], [940, 520], [900, 520]],
            ],
        },
        tracking_data={},
    )
    assert 0.0 <= scorer.frame_signals[0].edge_score <= 1.0


def test_edge_score_is_high_for_a_detection_at_the_border():
    from hydra_suite.data.dataset_generation import FrameQualityScorer

    scorer = FrameQualityScorer({"MAX_TARGETS": 1}, frame_shape=(1080, 1920))
    scorer.score_frame(
        0,
        detection_data={
            "confidences": [0.9],
            "count": 1,
            "obb_corners": [[[0, 0], [40, 0], [40, 20], [0, 20]]],
        },
        tracking_data={},
    )
    assert scorer.frame_signals[0].edge_score > 0.9


def test_bgsub_zeroes_uncertainty_weight_and_renormalizes():
    """All-NaN confidences must not silently dilute the other channels."""
    from hydra_suite.data.dataset_generation import FrameQualityScorer

    scorer = FrameQualityScorer(
        {"MAX_TARGETS": 4, "DETECTION_METHOD": "background_subtraction"},
        frame_shape=(1080, 1920),
    )
    assert scorer._weights.uncertainty == 0.0
    total = sum(
        getattr(scorer._weights, f)
        for f in (
            "uncertainty",
            "nms_instability",
            "count",
            "crowd",
            "fragmentation",
            "edge",
            "assignment",
            "track_loss",
            "position_uncertainty",
        )
    )
    assert total == pytest.approx(1.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_dataset_generation.py -k "edge_score or bgsub_zeroes" -v`
Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument 'frame_shape'`

- [ ] **Step 3: Write the implementation**

In `FrameQualityScorer.__init__`, accept and store the shape, and zero the
uncertainty weight on bgsub:

```python
    def __init__(self, params, frame_shape: tuple[int, int] | None = None):
        ...
        # (H, W) of the coordinate space `obb_corners` live in. Required for a
        # meaningful edge score: passing (1, 1) with pixel-space corners made
        # score_crowd return values in the hundreds.
        self.frame_shape = tuple(frame_shape) if frame_shape else None
        ...
        method = str(params.get("DETECTION_METHOD", "")).strip().lower()
        self._confidence_available = method != "background_subtraction"
```

Build the weights with `uncertainty=... if self._confidence_available else 0.0`,
then normalize explicitly so the remaining channels are not diluted:

```python
        self._weights = AcquisitionWeights(...).normalized()
```

In `score_frame`, replace the hardcoded shape:

```python
        obb_corners = self._extract_obb_corners(detection_data)
        if obb_corners and self.frame_shape is not None:
            crowd, edge = score_crowd(obb_corners, frame_shape=self.frame_shape)
        elif obb_corners:
            # No frame shape available: crowd is shape-independent, edge is not.
            crowd, _ = score_crowd(obb_corners, frame_shape=(1, 1))
            edge = 0.0
        else:
            crowd, edge = 0.0, 0.0
        fragmentation = score_fragmentation(
            obb_corners, reference_major_axis=self.reference_body_size * 2.2
        )
```

and populate `uncertainty_score` and `fragmentation_score` on the `ALSignals`.

Add:

```python
    def explain_scores(self) -> dict:
        """Per-channel maxima, for reporting why a selection came back empty."""
        from hydra_suite.data.al.acquisition import explain

        return explain(list(self.frame_signals.values()), self._weights)
```

- [ ] **Step 4: Pass the real frame shape from the caller**

In `src/hydra_suite/core/post/dataset_export.py:generate_active_learning_dataset`,
read the video dimensions once and pass them in, and report `explain_scores()`
when nothing is selected:

```python
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        try:
            frame_shape = (
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            )
        finally:
            cap.release()

        scorer = FrameQualityScorer(params, frame_shape=frame_shape)
```

and:

```python
        if not selected_frames:
            observed = scorer.explain_scores()
            detail = ", ".join(f"{k}={v:.2f}" for k, v in sorted(observed.items()))
            return {
                "success": False,
                "error": (
                    "No frames scored above the minimum selection score. "
                    f"Highest severity observed per signal: {detail or 'none'}. "
                    "Lower 'Min selection score' to export the best available "
                    "frames, or accept that tracking found nothing difficult."
                ),
                "channel_maxima": observed,
            }
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_dataset_generation.py tests/test_dataset_export.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
make format
git add src/hydra_suite/data/dataset_generation.py src/hydra_suite/core/post/dataset_export.py tests/
git commit -m "fix(al): score edges against the real frame shape; renormalize weights on bgsub"
```

---

### Task 12: Retire the legacy scalar scoring pipeline

Only now that all three orphaned behaviours are ported forward is the old code safe to delete.

**Files:**
- Modify: `src/hydra_suite/data/dataset_generation.py:137-359` (delete `_score_confidence`, `_score_count_mismatch`, `_score_assignment_cost`, `_score_track_loss`, `_score_uncertainty`, `_score_fragmented_detections`, and the `frame_scores` attribute)
- Modify: `tests/test_dataset_generation.py:396-680`

**Interfaces:**
- Consumes: Tasks 9-11.
- Produces: `FrameQualityScorer.score_frame(frame_id, detection_data=None, tracking_data=None) -> float` — now returns the **absolute composite severity** for that frame instead of the legacy weighted sum.

- [ ] **Step 1: Repoint the eight legacy tests**

In `tests/test_dataset_generation.py`, the tests `test_score_frame_zero_count`,
`test_score_normalization`, `test_multiple_frames_independent`,
`test_empty_confidences_list`, `test_low_confidence_uses_frame_average_not_minimum`,
`test_score_frame_uses_assignment_confidence_when_costs_missing`, and
`test_score_frame_prioritizes_split_detections_over_clean_overcount` assert on
`scorer.frame_scores[fid]["metrics"]`. Rewrite each to assert on
`scorer.frame_signals[fid]` instead. For example:

```python
def test_low_confidence_uses_frame_average_not_minimum():
    """One low-confidence detection among confident ones must not dominate."""
    scorer = FrameQualityScorer({"MAX_TARGETS": 4, "DATASET_CONF_THRESHOLD": 0.5},
                                frame_shape=(1080, 1920))
    scorer.score_frame(0, {"confidences": [0.95, 0.95, 0.95, 0.2], "count": 4}, {})
    # mean = 0.7625, above the 0.5 floor -> no uncertainty severity at all.
    assert scorer.frame_signals[0].uncertainty_score == 0.0


def test_score_frame_prioritizes_split_detections_over_clean_overcount():
    """A fragmented frame must outrank a frame that merely has an extra box."""
    params = {"MAX_TARGETS": 2, "REFERENCE_BODY_SIZE": 20.0}
    scorer = FrameQualityScorer(params, frame_shape=(1080, 1920))

    fragmented_corners = [
        obb_corners_from_dims(500, 500, 44, 16, 0.0),
        obb_corners_from_dims(100, 100, 20, 7, 0.0),
        obb_corners_from_dims(108, 102, 20, 7, 0.0),
    ]
    clean_corners = [
        obb_corners_from_dims(200, 200, 44, 16, 0.0),
        obb_corners_from_dims(600, 600, 44, 16, 0.0),
        obb_corners_from_dims(900, 300, 44, 16, 0.0),
    ]
    scorer.score_frame(0, {"confidences": [0.9] * 3, "count": 3,
                           "obb_corners": fragmented_corners}, {})
    scorer.score_frame(1, {"confidences": [0.9] * 3, "count": 3,
                           "obb_corners": clean_corners}, {})

    assert scorer.frame_signals[0].fragmentation_score > 0.45
    assert scorer.frame_signals[1].fragmentation_score == 0.0
    picked = scorer.get_worst_frames(1, diversity_window=1, probabilistic=False)
    assert picked == [0]
```

Delete `test_frame_quality_scorer_uses_tracker_default_preset_after_refactor`'s
assertions about `frame_scores` while keeping its preset assertions.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_dataset_generation.py -v`
Expected: FAIL — the rewritten tests reference `uncertainty_score` /
`fragmentation_score` paths that `score_frame` does not yet fully populate,
and `score_frame` still returns the legacy sum.

- [ ] **Step 3: Delete the legacy pipeline**

In `src/hydra_suite/data/dataset_generation.py`:
- Delete the six `_score_*` methods and the `self.frame_scores` defaultdict.
- Delete the now-unused `self.use_*` boolean attributes' legacy consumers, keeping
  the `_enabled` mapping that gates weights.
- Delete the module-level `combinations`, `_clamp01`, `_detection_corners_from_dims`,
  and `_polygon_overlap_ratio` imports if nothing else uses them.
- Rewrite the tail of `score_frame` to return the absolute composite:

```python
        self.frame_signals[int(frame_id)] = signal

        from hydra_suite.data.al.acquisition import _composite_score

        return float(_composite_score([signal], self._weights)[0])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_dataset_generation.py tests/test_dataset_export.py tests/test_session_export_chain.py -v`
Expected: PASS

- [ ] **Step 5: Confirm the dead weight is gone**

Run: `python -m pytest tests/ -k "dataset or al_" -v` and
`grep -rn "frame_scores" src/ tests/`
Expected: no matches for `frame_scores`; test suite green.

- [ ] **Step 6: Commit**

```bash
make format
git add src/hydra_suite/data/dataset_generation.py tests/test_dataset_generation.py
git commit -m "refactor(al): retire legacy scalar scoring pipeline after porting its signals"
```

---

## Phase 5 — Strict labels and dedup

### Task 13: Mutual-exclusion matching and strict label drops

**Files:**
- Modify: `src/hydra_suite/data/dataset_generation.py` (`_select_records_for_frame` from Task 8; delete `_match_yolo_detection`, `_measurements_to_detections`, `_detect_batch`, `_dims_from_shape`, `_write_frame_annotations` if unused)
- Test: `tests/test_al_strict_labels.py`

**Interfaces:**
- Consumes: `LabelRecord` (Task 2).
- Produces: `_select_records_for_frame(rows, frame_records, params, scale_back) -> tuple[list[LabelRecord], dict[str, int]]`, where the dict has keys `"lost"` and `"unmatched"`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_al_strict_labels.py`:

```python
import numpy as np
import pandas as pd

from hydra_suite.data.al.escalation import LabelRecord
from hydra_suite.data.dataset_generation import _select_records_for_frame
from hydra_suite.utils.geometry import obb_corners_from_dims
from hydra_suite.utils.geometry_levels import GeometryLevel


def _detector_record(cx, cy):
    return LabelRecord(
        class_id=0,
        confidence=0.9,
        points=obb_corners_from_dims(cx, cy, 44.0, 16.0, 0.0),
        level=GeometryLevel.OBB,
    )


def _rows(entries):
    """entries: list of (x, y, state)."""
    return pd.DataFrame(
        [
            {"FrameID": 0, "TrackID": i, "X": x, "Y": y, "Theta": 0.0, "State": s}
            for i, (x, y, s) in enumerate(entries)
        ]
    )


PARAMS = {"REFERENCE_BODY_SIZE": 20.0}


def test_matched_rows_are_exported():
    rows = _rows([(100.0, 100.0, "tracked"), (300.0, 300.0, "tracked")])
    dets = [_detector_record(101.0, 99.0), _detector_record(299.0, 301.0)]
    records, drops = _select_records_for_frame(rows, dets, PARAMS, 1.0)
    assert len(records) == 2
    assert drops == {"lost": 0, "unmatched": 0}


def test_lost_rows_are_dropped_and_counted():
    rows = _rows([(100.0, 100.0, "tracked"), (300.0, 300.0, "lost")])
    dets = [_detector_record(101.0, 99.0), _detector_record(299.0, 301.0)]
    records, drops = _select_records_for_frame(rows, dets, PARAMS, 1.0)
    assert len(records) == 1
    assert drops["lost"] == 1


def test_unmatched_rows_are_dropped_not_fabricated():
    """The legacy exporter invented a ref*2.2 x ref*0.8 box here."""
    rows = _rows([(100.0, 100.0, "tracked"), (2000.0, 2000.0, "tracked")])
    dets = [_detector_record(101.0, 99.0)]
    records, drops = _select_records_for_frame(rows, dets, PARAMS, 1.0)
    assert len(records) == 1
    assert drops["unmatched"] == 1


def test_two_rows_cannot_bind_the_same_detection():
    """Greedy nearest-centre matching emitted duplicate identical boxes here."""
    rows = _rows([(100.0, 100.0, "tracked"), (104.0, 101.0, "tracked")])
    dets = [_detector_record(102.0, 100.0)]
    records, drops = _select_records_for_frame(rows, dets, PARAMS, 1.0)
    assert len(records) == 1
    assert drops["unmatched"] == 1


def test_match_radius_scales_with_reference_body_size():
    """A 120 px offset matches for a large animal and not for a small one."""
    rows = _rows([(100.0, 100.0, "tracked")])
    dets = [_detector_record(220.0, 100.0)]

    _small, small_drops = _select_records_for_frame(
        rows, dets, {"REFERENCE_BODY_SIZE": 8.0}, 1.0
    )
    large, large_drops = _select_records_for_frame(
        rows, dets, {"REFERENCE_BODY_SIZE": 90.0}, 1.0
    )
    assert small_drops["unmatched"] == 1
    assert len(large) == 1
    assert large_drops["unmatched"] == 0


def test_nan_positions_are_dropped_as_unmatched():
    rows = _rows([(float("nan"), float("nan"), "tracked")])
    records, drops = _select_records_for_frame(rows, [_detector_record(100, 100)], PARAMS, 1.0)
    assert records == []
    assert drops["unmatched"] == 1


def test_no_rows_yields_no_records():
    records, drops = _select_records_for_frame(None, [_detector_record(100, 100)], PARAMS, 1.0)
    assert records == []
    assert drops == {"lost": 0, "unmatched": 0}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_al_strict_labels.py -v`
Expected: FAIL — the placeholder from Task 8 returns all detector records and zero drops.

- [ ] **Step 3: Write the implementation**

Replace the placeholder in `src/hydra_suite/data/dataset_generation.py`:

```python
_LOST_STATES = {"lost", "interpolated", "predicted"}


def _select_records_for_frame(rows, frame_records, params, scale_back):
    """Pair tracked CSV rows with detector geometry; export only real matches.

    Strict by design. The legacy exporter wrote a fabricated
    `ref*2.2 x ref*0.8` box whenever a row had no nearby detection, and wrote
    labels for `lost`/interpolated rows the detector never saw. Since AL
    selects frames precisely where tracking struggled, both behaviours injected
    wrong boxes exactly where the model was weakest.

    Matching is mutual-exclusion (one row <-> one detection) via the Hungarian
    algorithm, gated by a radius scaled to REFERENCE_BODY_SIZE rather than the
    legacy hardcoded 50 px.
    """
    import pandas as pd
    from scipy.optimize import linear_sum_assignment

    drops = {"lost": 0, "unmatched": 0}
    if rows is None or len(rows) == 0 or not frame_records:
        if rows is not None:
            for _, row in rows.iterrows():
                state = str(row.get("State", "")).strip().lower()
                if state in _LOST_STATES:
                    drops["lost"] += 1
                else:
                    drops["unmatched"] += 1
        return [], drops

    live_rows = []
    for _, row in rows.iterrows():
        state = str(row.get("State", "")).strip().lower()
        if state in _LOST_STATES:
            drops["lost"] += 1
            continue
        if pd.isna(row["X"]) or pd.isna(row["Y"]):
            drops["unmatched"] += 1
            continue
        live_rows.append(
            (float(row["X"]) * scale_back, float(row["Y"]) * scale_back)
        )

    if not live_rows:
        return [], drops

    reference = max(float(params.get("REFERENCE_BODY_SIZE", 20.0)), 1.0)
    max_distance = reference * 2.2

    centers = np.array(
        [rec.points.mean(axis=0) for rec in frame_records], dtype=np.float64
    )
    targets = np.array(live_rows, dtype=np.float64)
    cost = np.linalg.norm(targets[:, None, :] - centers[None, :, :], axis=2)

    row_idx, col_idx = linear_sum_assignment(cost)
    matched_detections: list[int] = []
    matched_rows = set()
    for r, c in zip(row_idx, col_idx):
        if cost[r, c] <= max_distance:
            matched_detections.append(int(c))
            matched_rows.add(int(r))

    drops["unmatched"] += len(live_rows) - len(matched_rows)
    return [frame_records[i] for i in sorted(matched_detections)], drops
```

Then delete `_match_yolo_detection`, `_write_frame_annotations`,
`_measurements_to_detections`, `_dims_from_shape`, and `_detect_batch` if no
caller remains (`grep -rn "_match_yolo_detection\|_write_frame_annotations" src/ tests/`).

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_al_strict_labels.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Run the whole AL surface**

Run: `python -m pytest tests/test_dataset_generation.py tests/test_dataset_export.py tests/test_al_export.py tests/test_session_export_chain.py -v`
Expected: PASS. Remove `test_write_frame_annotations_prefers_matched_detector_corners`
and `test_measurements_to_detections_scales_back_raw_obb_corners` — the functions
they cover are deleted; their behaviour is now covered by
`tests/test_al_strict_labels.py`.

- [ ] **Step 6: Commit**

```bash
make format
git add src/hydra_suite/data/dataset_generation.py tests/
git commit -m "feat(al): strict labels via mutual-exclusion matching with drop accounting"
```

---

### Task 14: Post-selection perceptual dedup

**Files:**
- Modify: `src/hydra_suite/core/post/dataset_export.py`
- Test: `tests/test_dataset_export.py` (append)

**Interfaces:**
- Consumes: `build_candidate_pool`, `CandidatePoolConfig` from `hydra_suite.data.al.candidate_pool`; `VideoFrameSource` from `hydra_suite.data.al.frame_source`.
- Produces: `generate_active_learning_dataset(..., dedup_method: str = "phash", dedup_threshold: int = 8)`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_dataset_export.py`:

```python
def test_dedup_runs_over_selected_frames_only(monkeypatch, tmp_path):
    """pHash over a whole video is prohibitive; only the picks get deduped."""
    import hydra_suite.core.post.dataset_export as de

    seen = {}

    def fake_pool(source, cfg):
        seen["n_candidates"] = source.length()
        seen["method"] = cfg.dedup_method
        return [ref for ref in source][:1]

    monkeypatch.setattr(de, "build_candidate_pool", fake_pool)
    ...  # drive generate_active_learning_dataset with 3 selected frames
    assert seen["n_candidates"] == 3
    assert seen["method"] == "phash"


def test_dedup_none_skips_the_pool_entirely(monkeypatch):
    import hydra_suite.core.post.dataset_export as de

    def boom(*args, **kwargs):
        raise AssertionError("build_candidate_pool must not run when method='none'")

    monkeypatch.setattr(de, "build_candidate_pool", boom)
    ...  # drive with dedup_method="none"; must not raise
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_dataset_export.py -k dedup -v`
Expected: FAIL with `AttributeError: module ... has no attribute 'build_candidate_pool'`

- [ ] **Step 3: Write the implementation**

In `src/hydra_suite/core/post/dataset_export.py`, add the imports and a helper,
then call it between selection and export:

```python
from hydra_suite.data.al.candidate_pool import CandidatePoolConfig, build_candidate_pool
from hydra_suite.data.al.frame_source import FrameRef, VideoFrameSource


class _SelectedFrameSource:
    """FrameSource restricted to an explicit frame-id list.

    Dedup runs over the SELECTED frames plus their context, never the whole
    video: perceptual hashing 100k frames is prohibitive, and the near-duplicate
    problem lives entirely in the +/-1 context expansion anyway.
    """

    def __init__(self, video_path: str, frame_ids):
        self._inner = VideoFrameSource(video_path)
        self._ids = sorted(int(f) for f in frame_ids)
        self._source_id = f"selected:{len(self._ids)}"

    def __iter__(self):
        for fid in self._ids:
            yield FrameRef(source_id=self._source_id, frame_id=fid, path=None)

    def read(self, ref):
        return self._inner.read(ref)

    def length(self) -> int:
        return len(self._ids)


def _dedup_selected_frames(video_path, frame_ids, method, threshold):
    """Drop perceptually near-duplicate picks. Returns the surviving ids."""
    if str(method).strip().lower() == "none" or len(frame_ids) < 2:
        return list(frame_ids)
    source = _SelectedFrameSource(str(video_path), frame_ids)
    cfg = CandidatePoolConfig(
        dedup_method=str(method).strip().lower(),
        dedup_threshold=int(threshold),
    )
    kept = build_candidate_pool(source, cfg)
    return [ref.frame_id for ref in kept]
```

Call it immediately after `scorer.get_worst_frames(...)` and before the
`export_dataset` call, logging how many picks were dropped:

```python
        before = len(selected_frames)
        selected_frames = _dedup_selected_frames(
            video_path, selected_frames, dedup_method, dedup_threshold
        )
        if len(selected_frames) < before:
            _emit(
                progress,
                55,
                f"Perceptual dedup dropped {before - len(selected_frames)} "
                f"near-duplicate frames.",
            )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_dataset_export.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
make format
git add src/hydra_suite/core/post/dataset_export.py tests/test_dataset_export.py
git commit -m "feat(al): perceptual dedup over selected frames"
```

---

## Phase 6 — Configuration and UI

### Task 15: Config schema and engine params

**Files:**
- Modify: `src/hydra_suite/trackerkit/config/schemas.py`
- Modify: `src/hydra_suite/trackerkit/engine_params.py:1205-1250`
- Modify: `src/hydra_suite/core/tracking/session.py:374-420`
- Test: `tests/test_engine_params.py` (exists — append)

**Interfaces:**
- Consumes: nothing new.
- Produces: new `TrackerConfig` fields `dataset_export_levels: list[str]` (default `["polygon", "obb", "aabb"]`), `dataset_dedup_method: str = "phash"`, `dataset_dedup_threshold: int = 8`, `dataset_class_names: str = ""`, `dataset_detectkit_project: str = ""`; and the matching `DATASET_EXPORT_LEVELS`, `DATASET_DEDUP_METHOD`, `DATASET_DEDUP_THRESHOLD`, `DATASET_CLASS_NAMES`, `DATASET_DETECTKIT_PROJECT` engine-param keys.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_engine_params.py`:

```python
def test_dataset_export_knobs_reach_engine_params():
    from hydra_suite.trackerkit.config.schemas import TrackerConfig
    from hydra_suite.trackerkit.engine_params import build_engine_params

    cfg = TrackerConfig()
    cfg.dataset_export_levels = ["polygon", "obb"]
    cfg.dataset_dedup_method = "dhash"
    cfg.dataset_dedup_threshold = 12
    cfg.dataset_class_names = "ant, larva"

    params = build_engine_params(cfg)
    assert params["DATASET_EXPORT_LEVELS"] == ["polygon", "obb"]
    assert params["DATASET_DEDUP_METHOD"] == "dhash"
    assert params["DATASET_DEDUP_THRESHOLD"] == 12
    assert params["DATASET_CLASS_NAMES"] == ["ant", "larva"]


def test_dataset_class_names_falls_back_to_single_class_name():
    from hydra_suite.trackerkit.config.schemas import TrackerConfig
    from hydra_suite.trackerkit.engine_params import build_engine_params

    cfg = TrackerConfig()
    cfg.dataset_class_name = "bee"
    cfg.dataset_class_names = ""
    assert build_engine_params(cfg)["DATASET_CLASS_NAMES"] == ["bee"]
```

Check `tests/test_engine_params.py` for the exact `build_engine_params` call
convention used by neighbouring tests and match it; the function's real
signature is in `src/hydra_suite/trackerkit/engine_params.py`.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_engine_params.py -k dataset_export_knobs -v`
Expected: FAIL with `KeyError: 'DATASET_EXPORT_LEVELS'`

- [ ] **Step 3: Add the fields and params**

Add the five fields to `TrackerConfig` in
`src/hydra_suite/trackerkit/config/schemas.py` with the defaults listed above,
and confirm they round-trip through the dataclass's `to_dict` / `from_dict`.

In `src/hydra_suite/trackerkit/engine_params.py`, beside the existing
`DATASET_*` keys:

```python
        "DATASET_EXPORT_LEVELS": list(
            _cfg_get(cfg, "dataset_export_levels", default=["polygon", "obb", "aabb"])
        ),
        "DATASET_DEDUP_METHOD": str(
            _cfg_get(cfg, "dataset_dedup_method", default="phash")
        ),
        "DATASET_DEDUP_THRESHOLD": int(
            _cfg_get(cfg, "dataset_dedup_threshold", default=8)
        ),
        "DATASET_CLASS_NAMES": _dataset_class_names(cfg),
        "DATASET_DETECTKIT_PROJECT": str(
            _cfg_get(cfg, "dataset_detectkit_project", default="")
        ),
```

with the module-level helper:

```python
def _dataset_class_names(cfg) -> list[str]:
    """Ordered class names (index = class id), falling back to the single name."""
    raw = str(_cfg_get(cfg, "dataset_class_names", default="") or "")
    names = [part.strip() for part in raw.split(",") if part.strip()]
    if names:
        return names
    single = str(_cfg_get(cfg, "dataset_class_name", default="") or "").strip()
    return [single or "object"]
```

- [ ] **Step 4: Thread them through the session**

In `src/hydra_suite/core/tracking/session.py:_run_dataset_generation`, pass the
new values into `generate_active_learning_dataset`:

```python
        from hydra_suite.utils.geometry_levels import GeometryLevel

        levels = [
            GeometryLevel.from_str(name)
            for name in self.params.get(
                "DATASET_EXPORT_LEVELS", ["polygon", "obb", "aabb"]
            )
        ]
        return dataset_export.generate_active_learning_dataset(
            ...,
            export_levels=levels,
            class_names=self.params.get("DATASET_CLASS_NAMES", [class_name]),
            dedup_method=self.params.get("DATASET_DEDUP_METHOD", "phash"),
            dedup_threshold=int(self.params.get("DATASET_DEDUP_THRESHOLD", 8)),
        )
```

Note: requesting a level above the detector's native level is clamped here
rather than raised, since the user's stored preference should not fail a
tracking run:

```python
        from hydra_suite.data.dataset_generation import resolve_native_level
        from hydra_suite.data.al.escalation import achievable_levels

        allowed = set(achievable_levels(resolve_native_level(self.params)))
        levels = [lvl for lvl in levels if lvl in allowed] or sorted(allowed, reverse=True)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_engine_params.py tests/test_session_export_chain.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
make format
git add src/hydra_suite/trackerkit/config/schemas.py src/hydra_suite/trackerkit/engine_params.py src/hydra_suite/core/tracking/session.py tests/test_engine_params.py
git commit -m "feat(al): add export-level, dedup and multi-class knobs to the shared param builder"
```

---

### Task 16: TrackerKit `DatasetPanel` controls

**Files:**
- Modify: `src/hydra_suite/trackerkit/gui/panels/dataset_panel.py:88-330`
- Modify: `src/hydra_suite/trackerkit/gui/workers/dataset_worker.py:15-78`
- Test: `tests/test_dataset_panel.py` (create)

**Interfaces:**
- Consumes: `resolve_native_level`, `achievable_levels`.
- Produces: `DatasetPanel.refresh_export_levels()`; `DatasetGenerationWorker.finished_signal = Signal(str, int, dict)` carrying `(dataset_dir, num_frames, manifest)`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_dataset_panel.py`:

```python
import pytest

pytest.importorskip("PySide6")

from hydra_suite.utils.geometry_levels import GeometryLevel


@pytest.mark.parametrize(
    "params,expected_enabled",
    [
        (
            {"DETECTION_METHOD": "yolo_obb", "YOLO_OBB_DIRECT_TASK": "segment"},
            {"polygon", "obb", "aabb"},
        ),
        (
            {"DETECTION_METHOD": "yolo_obb", "YOLO_OBB_DIRECT_TASK": "obb"},
            {"obb", "aabb"},
        ),
        (
            {"DETECTION_METHOD": "yolo_obb", "YOLO_OBB_DIRECT_TASK": "detect"},
            {"aabb"},
        ),
    ],
)
def test_level_checkboxes_reflect_detector_capability(params, expected_enabled):
    """Pure logic: which level checkboxes a given detector config enables."""
    from hydra_suite.data.al.escalation import achievable_levels
    from hydra_suite.data.dataset_generation import resolve_native_level

    enabled = {lvl.label for lvl in achievable_levels(resolve_native_level(params))}
    assert enabled == expected_enabled


def test_level_status_text_names_the_missing_requirement():
    from hydra_suite.trackerkit.gui.panels.dataset_panel import format_level_status

    text = format_level_status(GeometryLevel.OBB)
    assert "obb" in text and "aabb" in text
    assert "segmentation" in text.lower()

    assert "polygon" in format_level_status(GeometryLevel.POLYGON)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_dataset_panel.py -v`
Expected: FAIL with `ImportError: cannot import name 'format_level_status'`

- [ ] **Step 3: Add the status formatter and the controls**

Add to `src/hydra_suite/trackerkit/gui/panels/dataset_panel.py`, at module level:

```python
def format_level_status(native_level: GeometryLevel) -> str:
    """Human-readable summary of which label levels this detector can produce."""
    labels = " + ".join(lvl.label for lvl in achievable_levels(native_level))
    if native_level is GeometryLevel.POLYGON:
        return f"Will export: {labels}"
    if native_level is GeometryLevel.OBB:
        return (
            f"Will export: {labels} — polygon labels require a segmentation model"
        )
    return (
        f"Will export: {labels} — oriented and polygon labels require an OBB or "
        "segmentation model"
    )
```

In `_build_ui`, inside the `g_dataset_config` form, after the class-name row:

```python
        # Export levels -- what the detector can actually produce.
        self.lbl_export_level_status = QLabel(format_level_status(GeometryLevel.OBB))
        self.lbl_export_level_status.setWordWrap(True)
        f_config.addRow("Label levels", self.lbl_export_level_status)

        self.chk_level_polygon = QCheckBox("polygon (segmentation masks)")
        self.chk_level_obb = QCheckBox("obb (oriented boxes)")
        self.chk_level_aabb = QCheckBox("aabb (axis-aligned boxes)")
        for chk in (self.chk_level_polygon, self.chk_level_obb, self.chk_level_aabb):
            chk.setChecked(True)
            chk.setToolTip(
                "Each enabled level is written as its own DetectKit source. "
                "Images are hardlinked, so extra levels cost almost no disk."
            )
        _levels_row = QVBoxLayout()
        _levels_row.addWidget(self.chk_level_polygon)
        _levels_row.addWidget(self.chk_level_obb)
        _levels_row.addWidget(self.chk_level_aabb)
        f_config.addRow("Export as", _levels_row)
```

Rename the class-name row's tooltip and placeholder to accept a list:

```python
        self.line_dataset_class_name.setPlaceholderText("e.g., ant  (or: ant, larva)")
        self.line_dataset_class_name.setToolTip(
            "Ordered class names, comma-separated. Position determines class id: "
            "the first name is class 0, the second class 1, and so on.\n"
            "Single-class users can just type one name."
        )
```

Add the dedup controls to the `g_frame_selection` form:

```python
        self.combo_dataset_dedup = QComboBox()
        for method in ("phash", "ahash", "dhash", "histogram", "none"):
            self.combo_dataset_dedup.addItem(method)
        self.combo_dataset_dedup.setToolTip(
            "Perceptual dedup applied to the SELECTED frames (and their context "
            "frames) after ranking. Removes near-identical picks that the "
            "diversity window cannot catch. 'none' disables it."
        )
        f_selection.addRow("Duplicate filter", self.combo_dataset_dedup)

        self.spin_dataset_dedup_threshold = QSpinBox()
        self.spin_dataset_dedup_threshold.setRange(0, 64)
        self.spin_dataset_dedup_threshold.setValue(8)
        self.spin_dataset_dedup_threshold.setToolTip(
            "Hamming distance (hash methods) or bin distance (histogram) below "
            "which two frames count as duplicates. Higher = more aggressive."
        )
        f_selection.addRow("Duplicate threshold", self.spin_dataset_dedup_threshold)
```

Rewrite the min-selection-score tooltip, which currently describes the old
relative behaviour:

```python
        self.spin_dataset_min_selection_score.setToolTip(
            "Min selection score (0.0-1.0).\n\n"
            "Scores are ABSOLUTE severities: a frame with nothing wrong scores "
            "exactly 0, and the value is comparable across videos. A cleanly "
            "tracked video can legitimately export no frames.\n\n"
            "0.0 = export the best available frames regardless of severity."
        )
```

Add `refresh_export_levels`:

```python
    def refresh_export_levels(self) -> None:
        """Sync the level status label and checkboxes to the detection config."""
        params = self._main_window.get_parameters_dict()
        native = resolve_native_level(params)
        allowed = set(achievable_levels(native))
        self.lbl_export_level_status.setText(format_level_status(native))
        for level, chk in (
            (GeometryLevel.POLYGON, self.chk_level_polygon),
            (GeometryLevel.OBB, self.chk_level_obb),
            (GeometryLevel.AABB, self.chk_level_aabb),
        ):
            available = level in allowed
            chk.setEnabled(available)
            if not available:
                chk.setChecked(False)

        is_bgsub = (
            str(params.get("DETECTION_METHOD", "")).lower() == "background_subtraction"
        )
        self.lbl_bgsub_notice.setVisible(is_bgsub)
```

with the notice label created in `_build_ui`:

```python
        self.lbl_bgsub_notice = self._main_window._create_help_label(
            "Background subtraction produces no detection confidences, so the "
            "confidence signal is disabled and the remaining frame-selection "
            "signals are reweighted to compensate.",
            attach_to_title=False,
        )
        self.lbl_bgsub_notice.setVisible(False)
        vl_content.addWidget(self.lbl_bgsub_notice)
```

Call `refresh_export_levels()` from `apply_config` and wire it to the detection
panel's model/method change signal (follow the existing pattern in
`trackerkit/gui/orchestrators/config.py` for cross-panel refresh).

- [ ] **Step 4: Widen the worker signal and report the manifest**

In `src/hydra_suite/trackerkit/gui/workers/dataset_worker.py`:

```python
    finished_signal = Signal(str, int, dict)  # dataset_dir, num_frames, manifest
```

and in `execute`:

```python
        if result.get("success"):
            self.finished_signal.emit(
                result["dir"], result["num_frames"], result.get("manifest", {})
            )
```

Update the connected slot in `trackerkit/gui/orchestrators/tracking.py` (find it
via `grep -n "dataset_worker.finished_signal" src/`) to accept the third
argument and surface the summary:

```python
    def _on_dataset_generation_finished(self, dataset_dir, num_frames, manifest):
        totals = manifest.get("totals", {})
        roots = ", ".join(r["level"] for r in manifest.get("roots", []))
        message = (
            f"Exported {num_frames} frames to {dataset_dir}\n"
            f"Label levels: {roots or 'none'}\n"
            f"Objects labelled: {totals.get('objects', 0)}\n"
            f"Dropped (lost/interpolated tracks): {totals.get('dropped_lost', 0)}\n"
            f"Dropped (no matching detection): {totals.get('dropped_unmatched', 0)}"
        )
        self._mw.statusBar().showMessage(f"Dataset exported to {dataset_dir}", 10000)
        logger.info(message)
```

- [ ] **Step 5: Retarget the stale X-AnyLabeling text**

In `dataset_panel.py`, the workflow help string already says DetectKit — verify
it reads "Run tracking → Review/correct in DetectKit → Train improved model".
The generated README lived in the deleted `_write_dataset_files`, so no README
work remains; confirm with `grep -rn "AnyLabeling" src/hydra_suite/data/ src/hydra_suite/trackerkit/`.

- [ ] **Step 6: Run test to verify it passes**

Run: `python -m pytest tests/test_dataset_panel.py -v`
Expected: PASS (4 tests)

- [ ] **Step 7: Commit**

```bash
make format
git add src/hydra_suite/trackerkit/gui/ tests/test_dataset_panel.py
git commit -m "feat(trackerkit): export-level, dedup and multi-class controls in the dataset panel"
```

---

### Task 17: DetectKit AL dialog level controls

**Files:**
- Modify: `src/hydra_suite/detectkit/gui/dialogs/active_learning.py`
- Test: `tests/test_detectkit_al_dialog.py` (append)

**Interfaces:**
- Consumes: `format_level_status` logic (reimplemented locally — the panel version lives in an app package TrackerKit owns, and DetectKit must not import from TrackerKit).
- Produces: `ALRequest.export_level` actually set from the dialog; new `ALRequest.export_levels: list[str]` defaulting to `["obb"]`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_detectkit_al_dialog.py`:

```python
def test_dialog_sets_export_level_from_the_model_task(qtbot, tmp_path):
    """Regression: ALRequest.export_level was never set, so it stayed 'obb'."""
    from hydra_suite.detectkit.gui.dialogs.active_learning import (
        ActiveLearningDialog,
    )
    from hydra_suite.detectkit.gui.models import DetectKitProject

    project = DetectKitProject(project_dir=tmp_path)
    dialog = ActiveLearningDialog(project=project)
    qtbot.addWidget(dialog)

    dialog.set_model_task("segment")
    request = dialog.build_request()
    assert request.export_level == "polygon"
    assert "polygon" in request.export_levels


def test_dialog_refuses_polygon_for_an_obb_model(qtbot, tmp_path):
    from hydra_suite.detectkit.gui.dialogs.active_learning import (
        ActiveLearningDialog,
    )
    from hydra_suite.detectkit.gui.models import DetectKitProject

    dialog = ActiveLearningDialog(project=DetectKitProject(project_dir=tmp_path))
    qtbot.addWidget(dialog)

    dialog.set_model_task("obb")
    assert dialog.chk_level_polygon.isEnabled() is False
    assert dialog.build_request().export_level == "obb"
```

Match the dialog's real constructor signature — read
`src/hydra_suite/detectkit/gui/dialogs/active_learning.py` before writing these
tests and adjust the instantiation to match.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_detectkit_al_dialog.py -k export_level -v`
Expected: FAIL with `AttributeError: 'ActiveLearningDialog' object has no attribute 'set_model_task'`

- [ ] **Step 3: Write the implementation**

Add to the dialog: a read-only level status label, three level checkboxes
mirroring Task 16, a `set_model_task(task: str)` method mapping
`segment → POLYGON`, `obb → OBB`, `detect → AABB` and enabling checkboxes
accordingly, and wiring in `build_request` (or wherever `ALRequest` is
constructed) to set both `export_level` (the highest checked level) and
`export_levels` (all checked levels, as label strings).

Add `export_levels: list[str] = field(default_factory=lambda: ["obb"])` to
`ALRequest` in `src/hydra_suite/detectkit/jobs/al_worker.py`, and in
`run_active_learning` register one `OBBSource` per requested level by calling
`export_al_dataset` instead of the hand-rolled directory writing, mirroring
Task 8's structure.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_detectkit_al_dialog.py tests/test_detectkit_al_worker.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
make format
git add src/hydra_suite/detectkit/ tests/
git commit -m "feat(detectkit): wire export level controls into the AL dialog"
```

---

### Task 18: Documentation and the equivalence gate

**Files:**
- Modify: `docs/user-guide/dataset-generation.md`
- Modify: `docs/developer-guide/confidence-metrics.md`
- Create: `tests/goldens/al_selection_characterization.json`
- Test: `tests/test_al_selection_golden.py`

**Interfaces:**
- Consumes: everything above.
- Produces: a committed characterization golden pinning frame selection.

- [ ] **Step 1: Write the characterization golden test**

Create `tests/test_al_selection_golden.py`:

```python
"""Characterization golden for AL frame selection.

Frame selection deliberately CHANGES in this work (absolute floors replace
min-max normalization, the fragmentation channel is restored, edge scoring is
fixed). A byte-identity oracle against the old behaviour would therefore be
wrong, and an oracle derived from the new code would be tautological. This
golden pins the new behaviour against a fixed synthetic signal set, so future
refactors that claim to preserve selection must actually preserve it.

To regenerate after an INTENTIONAL scoring change:
    python -m pytest tests/test_al_selection_golden.py --update-golden
and review the diff as part of the change.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from hydra_suite.data.al.acquisition import PRESETS, select
from hydra_suite.data.al.signals import ALSignals

GOLDEN = Path(__file__).parent / "goldens" / "al_selection_characterization.json"


def _fixed_signals():
    """120 deterministic frames spanning every channel's dynamic range."""
    rng = np.random.default_rng(20260817)
    signals = []
    for fid in range(120):
        signals.append(
            ALSignals(
                frame_id=fid,
                n_detections=int(rng.integers(0, 8)),
                mean_confidence=float(rng.uniform(0.1, 1.0)),
                uncertainty_score=float(rng.uniform(0.0, 1.0)),
                count_deviation=float(rng.uniform(0.0, 1.0)),
                crowd_score=float(rng.uniform(0.0, 1.0)),
                fragmentation_score=float(rng.uniform(0.0, 1.0)),
                edge_score=float(rng.uniform(0.0, 1.0)),
                extras={
                    "assignment": float(rng.uniform(0.0, 1.0)),
                    "track_loss": float(rng.uniform(0.0, 1.0)),
                    "position_uncertainty": float(rng.uniform(0.0, 1.0)),
                },
            )
        )
    return signals


def test_selection_matches_golden(request):
    picked = select(
        _fixed_signals(),
        weights=PRESETS["tracker_default"],
        k=20,
        diversity_window=5,
        probabilistic=False,       # deterministic: no rng in the golden
        min_score=0.30,
    )
    if request.config.getoption("--update-golden", default=False):
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN.write_text(json.dumps({"picked": picked}, indent=2))
        pytest.skip("golden updated")

    expected = json.loads(GOLDEN.read_text())["picked"]
    assert picked == expected
```

Register the flag in `tests/conftest.py`:

```python
def pytest_addoption(parser):
    parser.addoption(
        "--update-golden",
        action="store_true",
        default=False,
        help="Rewrite characterization goldens instead of asserting against them.",
    )
```

(If `pytest_addoption` already exists in `conftest.py`, add the option to it
rather than defining a second hook.)

- [ ] **Step 2: Generate the golden and verify it locks**

Run: `python -m pytest tests/test_al_selection_golden.py --update-golden`
Then: `python -m pytest tests/test_al_selection_golden.py -v`
Expected: first run skips with "golden updated"; second run PASSES.
Confirm `tests/goldens/al_selection_characterization.json` is not swallowed by
`.gitignore`: `git check-ignore -v tests/goldens/al_selection_characterization.json`
must print nothing.

- [ ] **Step 3: Update the user guide**

In `docs/user-guide/dataset-generation.md`, replace the description of the
output layout with the three-root layout, and document:
- which detection sources reach which levels (copy the table from the spec),
- that a cleanly tracked video can legitimately export no frames under absolute
  scoring, and what the reported per-channel maxima mean,
- that lost/interpolated tracks and unmatched rows are dropped rather than
  labelled, and where the counts appear (`source.json` and the finish summary),
- that each root is a directly-importable DetectKit source.

In `docs/developer-guide/confidence-metrics.md`, update the frame-selection
metric list to name the `fragmentation` channel separately from `crowd`, and
note that scores are absolute severities rather than within-run ranks.

- [ ] **Step 4: Run the docs gate**

Run: `make docs-check`
Expected: PASS (strict mkdocs build + terminology check)

- [ ] **Step 5: Run the delta test gate**

Run: `python -m pytest tests/ -x -q -p no:randomly 2>&1 | tail -30`

Compare the failure count against the pre-work baseline captured on the base
commit. Per the repo's known state there are roughly 24 pre-existing failures;
the delta introduced by this branch must be zero. Note that
`tests/test_classkit_*` can hang on modal dialogs — batch per-file if the full
run stalls.

- [ ] **Step 6: Run the equivalence harness as a no-op gate**

Every change in this plan is export-only or behind an opt-in flag, so tracking
output must be byte-identical. First kill stale sleap/hydra processes, then:

```bash
conda activate hydra-mps
git fetch origin --tags
git worktree add --detach .worktrees/equiv-legacy legacy/main
REPO=$PWD WT=$PWD \
  MAIN_SRC=$PWD/.worktrees/equiv-legacy/src WT_SRC=$PWD/src \
  OUT=/tmp/equiv_al RUNTIME=mps \
  bash tools/equivalence/run_matrix.sh
```

Expected: EQUIVALENCE at the DETERMINISM floor for every clip, on both
`_forward.csv` and `_tracking_final.csv`. **Verify row counts > 1** on the CSVs
before trusting any `EQUIVALENT` verdict — a bare shell yields empty CSVs that
falsely compare equal. Then repeat on the CUDA box:

```bash
ssh rutalab@mehek.taild08eb9.ts.net
# see CLAUDE.md "CUDA box (mehek)" for the full recipe
```

Clean up: `git worktree remove --force .worktrees/equiv-legacy && git worktree prune`

- [ ] **Step 7: Commit**

```bash
make format
git add docs/ tests/
git commit -m "docs: document escalated AL export; add selection characterization golden"
```

---

## Self-Review

**Spec coverage:**

| Spec section | Tasks |
|---|---|
| 1. Output layout (three roots, hardlinks) | 4 |
| 2. Level honesty | 2 (`achievable_levels`), 7 (`resolve_native_level`), 8, 16, 17 |
| 3. Shared modules | 1, 2, 3, 4 |
| 4a. Absolute floors | 10 |
| 4b. Fragmentation channel | 9 |
| 4c. Asymmetric count | 10 |
| 4 (frame shape, margin removal, bgsub renormalization) | 10, 11 |
| 4 (delete legacy pipeline) | 12 |
| 5. Strict labels + accounting | 13, 4 (`source.json` totals) |
| 6. Inference plumbing | 5, 6, 7 |
| 7. Candidate-pool dedup | 14 |
| 8. UI | 15, 16, 17 |
| Error handling (empty selection, level refusal, contour fallback, hardlink fallback, partial writes) | 11, 4, 6, 4, 4 |
| Testing (incl. equivalence no-op gate + golden) | every task; 18 |

No spec requirement is unassigned.

**Known deviations from a strict reading of the spec:**

- Task 8 introduces a temporary `_select_records_for_frame` placeholder so the
  task is independently testable; Task 13 replaces it. Tasks 8 and 13 must not
  be reordered or run in parallel.
- Task 15 clamps an over-ambitious stored level preference rather than raising,
  because a stale config should not fail a tracking run. The spec's "refused at
  config time" applies to an explicit user request, which is what Tasks 16/17
  enforce by disabling unachievable checkboxes.
- `scipy.optimize.linear_sum_assignment` is used directly in Task 13 rather than
  `core/assigners/hungarian.py`'s `TrackAssigner`, which is built around Kalman
  track state and carries tracking-specific cost terms this pairing does not
  want. Confirm `scipy` is already a hard dependency (`grep -rn "scipy" pyproject.toml`);
  it is used by the assigner module, so it is.

**Type consistency:** `LabelRecord` fields (`class_id`, `confidence`, `points`,
`level`) are used identically in Tasks 2, 3, 4, 8, 13. `ExportedFrame` fields
(`frame_id`, `image_name`, `records`, `is_context`, `drops`) match between Tasks
4 and 8. `write_label_file(path, records, frame_size, level)` is called with the
same argument names in Tasks 3, 4. `frame_size` is `(height, width)` everywhere.
`export_dataset` returns a manifest dict in Tasks 8, 15, 16 consistently.
`ALSignals.uncertainty_score` and `.fragmentation_score` are introduced in Tasks
9/10 and consumed in 11, 12, 18.
