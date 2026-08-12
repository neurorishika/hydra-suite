# DetectKit Geometry Levels — Polygon-First Label Model — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make DetectKit's canonical label model polygon-first with a recorded `GeometryLevel` (`aabb < obb < polygon`), so it can train detect-only and segmentation models, stops destroying segmentation at import, gates training roles by information content, round-trips X-AnyLabeling at the right mode, and exports the richest geometry a detector produced — without perturbing any existing OBB-only project's datasets.

**Architecture:** One new pure module (`training/geometry_levels.py`) defines the level ordering and the label/source scanners. The level is stored per source on the project JSON (`OBBSource.level`), defaulting to `obb` so existing projects are a no-op. Import records the level instead of collapsing to `minAreaRect`; validation blocks mixed-evidence sources; the dataset builders gain polygon-aware derivations and three new training roles; the merged dataset's level is `min()` across sources and roles are gated (with the blocking source named); the X-AnyLabeling launch/sync-back becomes level-aware; and AL export carries native contours through a new opt-in `OBBResult.polygons` field.

**Tech Stack:** Python 3.10+, NumPy, OpenCV (`cv2`), PySide6 (Qt), pytest. YOLO-OBB / YOLO-seg / YOLO-detect on-disk label syntax (`class_id` + normalized coords).

## Global Constraints

- **On-disk label syntax is unchanged.** Every label line stays `class_id` followed by a normalized point list. An `aabb`-level source stores axis-aligned **quads** (8 coords), not `cx cy w h`, except detect-*role output* datasets which write `cx cy w h`. (Spec §3a)
- **Byte-identical regression guarantee.** An existing `obb`-only project MUST produce byte-identical merged and derived datasets before and after this change. New code paths run only when a source is *not* `obb`-level. (Spec §10)
- **No-op migration.** A source with no recorded `level` reads as `obb`. No label files are rewritten; no re-import. (Spec §8)
- **Sources are homogeneous.** A source has exactly one level; mixed evidence blocks at validation and is resolved by the user, never guessed. (Spec §4)
- **Hot path untouched.** `OBBResult.polygons` is populated only when `emit_native_geometry` is explicitly requested; the tracking/`.npz`-cache path never sets it and never serializes it. (Spec §7a)
- **Fail loudly.** A role requested above the merged level is refused naming the blocking source; a homogeneity failure blocks; a derivation producing zero valid objects for an image reports that image. (Spec §9)
- **Level total order:** `aabb (0) < obb (1) < polygon (2)`. Downward derivation is automatic and lossless-to-target; upward derivation is out of scope (piece B). (Spec §3)
- **Commit style:** commit as the configured git user; do NOT add any `Co-Authored-By: Claude` trailer.

---

## File Structure

**New files**
- `src/hydra_suite/training/geometry_levels.py` — `GeometryLevel` enum + ordering; pure label-line classifier; source-directory scanner returning a homogeneity verdict. No Qt, no I/O beyond reading label text.
- `tests/test_geometry_levels.py` — unit tests for the classifier and scanner.
- `tests/test_geometry_level_builders.py` — derivation-path tests (poly→obb, poly→aabb, obb→aabb, poly→crop-polygon) + segment passthrough.
- `tests/test_geometry_level_import.py` — import-records-level tests (YOLO-obb/detect, COCO seg/bbox).
- `tests/test_geometry_level_export.py` — `OBBResult.polygons` + extractor emit + AL label writer + level stamping.
- `tests/test_geometry_level_regression.py` — the byte-identical obb-only regression gate.

**Modified files**
- `src/hydra_suite/detectkit/gui/models.py` — `OBBSource.level` field + (de)serialization + default.
- `src/hydra_suite/detectkit/gui/source_import.py` — import records level; COCO seg → polygon, detect → aabb quad; `MaterializedDetectKitSource.level`.
- `src/hydra_suite/detectkit/gui/dialogs/source_validation.py` — homogeneity block + confirm-override.
- `src/hydra_suite/training/contracts.py` — three new `TrainingRole` members.
- `src/hydra_suite/training/dataset_builders.py` — variable-length label parser; polygon input cases; crop-segment builder; role dispatch + min-level gating.
- `src/hydra_suite/detectkit/gui/dialogs/training_dialog.py` — new role checkboxes; min()-merge gating with named blocking source; hide `seq_crop_segment`.
- `src/hydra_suite/detectkit/gui/panels/dataset_panel.py` — level-derived launch mode; validating sync-back.
- `src/hydra_suite/core/inference/result.py` — `OBBResult.polygons` field.
- `src/hydra_suite/core/inference/config.py` — `OBBConfig.emit_native_geometry`.
- `src/hydra_suite/core/inference/stages/obb.py` — extractors populate `polygons` when asked.
- `src/hydra_suite/detectkit/gui/prediction_preview.py` — export detector returns native polygons.
- `src/hydra_suite/detectkit/jobs/al_worker.py` — label writer emits point lists; stamp source level.

---

## Task 1: `GeometryLevel` enum and label-line classifier

**Files:**
- Create: `src/hydra_suite/training/geometry_levels.py`
- Test: `tests/test_geometry_levels.py`

**Interfaces:**
- Produces:
  - `class GeometryLevel(IntEnum)` with members `AABB=0`, `OBB=1`, `POLYGON=2`; property `.label -> str` (lowercase name); `@staticmethod from_str(value: str) -> GeometryLevel`.
  - `classify_label_line(field_count: int) -> str` returning one of `"aabb"`, `"four_point"`, `"polygon"`, `"invalid"`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_geometry_levels.py
import pytest

from hydra_suite.training.geometry_levels import GeometryLevel, classify_label_line


def test_level_ordering_and_labels():
    assert GeometryLevel.AABB < GeometryLevel.OBB < GeometryLevel.POLYGON
    assert GeometryLevel.AABB.label == "aabb"
    assert GeometryLevel.OBB.label == "obb"
    assert GeometryLevel.POLYGON.label == "polygon"
    assert GeometryLevel.from_str("Polygon") is GeometryLevel.POLYGON


def test_from_str_rejects_unknown():
    with pytest.raises(ValueError):
        GeometryLevel.from_str("blob")


@pytest.mark.parametrize(
    "field_count,expected",
    [
        (5, "aabb"),        # class + cx cy w h
        (9, "four_point"),  # class + 8 coords (obb OR quad polygon)
        (7, "polygon"),     # class + 3 points
        (11, "polygon"),    # class + 5 points
        (13, "polygon"),    # class + 6 points
        (4, "invalid"),
        (8, "invalid"),     # even field count => odd coord count
        (1, "invalid"),
    ],
)
def test_classify_label_line(field_count, expected):
    assert classify_label_line(field_count) == expected
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_geometry_levels.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'hydra_suite.training.geometry_levels'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/hydra_suite/training/geometry_levels.py
"""Geometry-level model for DetectKit's polygon-first labels.

A label line stays ``class_id`` followed by a normalized point list. The
information content of a source is captured by a totally-ordered level:

    aabb  <  obb  <  polygon

Downward derivation (polygon -> minAreaRect -> obb -> aabb) is lossless to the
target; upward derivation needs new information and is out of scope here.
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
    # coords = field_count - 1 must be even and >= 6 (>=3 points), excluding 4 points (handled above).
    coords = field_count - 1
    if field_count >= 7 and coords % 2 == 0:
        points = coords // 2
        if points >= 3 and points != 4:
            return "polygon"
    return "invalid"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_geometry_levels.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/training/geometry_levels.py tests/test_geometry_levels.py
git commit -m "feat(geometry): GeometryLevel enum and label-line classifier"
```

---

## Task 2: Source-directory scanner with homogeneity verdict

**Files:**
- Modify: `src/hydra_suite/training/geometry_levels.py`
- Test: `tests/test_geometry_levels.py`

**Interfaces:**
- Consumes: `GeometryLevel`, `classify_label_line` (Task 1).
- Produces:
  - `@dataclass(frozen=True) class SourceLevelScan` with fields:
    `resolved_level: GeometryLevel`, `is_homogeneous: bool`,
    `conflict_files: list[str]`, `needs_confirmation: bool`, `reason: str`.
  - `scan_source_levels(labels_dir: str | Path, intended_level: GeometryLevel = GeometryLevel.OBB, *, confirm_quads_are_polygons: bool = False) -> SourceLevelScan`
    — scans every `*.txt` under `labels_dir`. Files with polygon evidence force `POLYGON`; files that are exactly-4-point-only are resolved by `intended_level`; `cx cy w h` files are `AABB`. A source that mixes polygon-evidence files with four-point-only files is non-homogeneous (blocks) unless `confirm_quads_are_polygons=True`, which asserts the quads are genuine contours and resolves the whole source to `POLYGON`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_geometry_levels.py  (append)
from pathlib import Path

from hydra_suite.training.geometry_levels import scan_source_levels


def _write(labels: Path, name: str, text: str) -> None:
    labels.mkdir(parents=True, exist_ok=True)
    (labels / name).write_text(text, encoding="utf-8")


def test_scan_all_polygon(tmp_path):
    labels = tmp_path / "labels"
    _write(labels, "a.txt", "0 0.1 0.1 0.5 0.1 0.5 0.5 0.1 0.5 0.3 0.7\n")  # 5 pts
    scan = scan_source_levels(labels)
    assert scan.resolved_level is GeometryLevel.POLYGON
    assert scan.is_homogeneous


def test_scan_four_point_uses_intended(tmp_path):
    labels = tmp_path / "labels"
    _write(labels, "a.txt", "0 0.1 0.1 0.5 0.1 0.5 0.5 0.1 0.5\n")  # 4 pts
    assert scan_source_levels(labels, GeometryLevel.OBB).resolved_level is GeometryLevel.OBB
    assert scan_source_levels(labels, GeometryLevel.POLYGON).resolved_level is GeometryLevel.POLYGON


def test_scan_aabb(tmp_path):
    labels = tmp_path / "labels"
    _write(labels, "a.txt", "0 0.5 0.5 0.2 0.2\n")  # cx cy w h
    scan = scan_source_levels(labels)
    assert scan.resolved_level is GeometryLevel.AABB
    assert scan.is_homogeneous


def test_scan_mixed_polygon_and_fourpoint_blocks(tmp_path):
    labels = tmp_path / "labels"
    _write(labels, "poly.txt", "0 0.1 0.1 0.5 0.1 0.5 0.5 0.1 0.5 0.3 0.7\n")
    _write(labels, "quad.txt", "0 0.1 0.1 0.5 0.1 0.5 0.5 0.1 0.5\n")
    scan = scan_source_levels(labels)
    assert not scan.is_homogeneous
    assert scan.needs_confirmation
    assert "quad.txt" in scan.conflict_files


def test_scan_mixed_resolved_by_confirm(tmp_path):
    labels = tmp_path / "labels"
    _write(labels, "poly.txt", "0 0.1 0.1 0.5 0.1 0.5 0.5 0.1 0.5 0.3 0.7\n")
    _write(labels, "quad.txt", "0 0.1 0.1 0.5 0.1 0.5 0.5 0.1 0.5\n")
    scan = scan_source_levels(labels, confirm_quads_are_polygons=True)
    assert scan.is_homogeneous
    assert scan.resolved_level is GeometryLevel.POLYGON
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_geometry_levels.py -v`
Expected: FAIL — `ImportError: cannot import name 'scan_source_levels'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/hydra_suite/training/geometry_levels.py  (append)
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class SourceLevelScan:
    """Verdict of scanning a source's label directory for its geometry level."""

    resolved_level: GeometryLevel
    is_homogeneous: bool
    conflict_files: list[str] = field(default_factory=list)
    needs_confirmation: bool = False
    reason: str = ""


def _classify_file(path: Path) -> str:
    """Return the strongest evidence in a single label file.

    One of: "polygon", "four_point", "aabb", "empty", "invalid".
    Any polygon-evidence line makes the file "polygon"; otherwise a file that
    mixes aabb and four-point lines is "invalid" (internally inconsistent).
    """
    seen: set[str] = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        kind = classify_label_line(len(line.split()))
        if kind == "invalid":
            return "invalid"
        seen.add(kind)
    if not seen:
        return "empty"
    if "polygon" in seen:
        return "polygon"
    if "aabb" in seen and "four_point" in seen:
        return "invalid"
    if "four_point" in seen:
        return "four_point"
    return "aabb"


def scan_source_levels(
    labels_dir: str | Path,
    intended_level: GeometryLevel = GeometryLevel.OBB,
    *,
    confirm_quads_are_polygons: bool = False,
) -> SourceLevelScan:
    """Scan a source's labels/ and resolve its single geometry level."""
    root = Path(labels_dir)
    poly_files: list[str] = []
    fourpt_files: list[str] = []
    aabb_files: list[str] = []
    invalid_files: list[str] = []

    for path in sorted(root.rglob("*.txt")):
        kind = _classify_file(path)
        if kind == "polygon":
            poly_files.append(path.name)
        elif kind == "four_point":
            fourpt_files.append(path.name)
        elif kind == "aabb":
            aabb_files.append(path.name)
        elif kind == "invalid":
            invalid_files.append(path.name)

    if invalid_files:
        return SourceLevelScan(
            resolved_level=intended_level,
            is_homogeneous=False,
            conflict_files=invalid_files,
            needs_confirmation=False,
            reason="Some label files contain malformed or internally mixed lines.",
        )

    has_poly = bool(poly_files)
    has_fourpt = bool(fourpt_files)
    has_aabb = bool(aabb_files)

    # aabb never coexists with obb/polygon evidence: you cannot mix axis-aligned
    # boxes with oriented/contour geometry in one homogeneous source.
    if has_aabb and (has_poly or has_fourpt):
        return SourceLevelScan(
            resolved_level=GeometryLevel.AABB,
            is_homogeneous=False,
            conflict_files=aabb_files,
            needs_confirmation=False,
            reason="Source mixes axis-aligned boxes with oriented/contour geometry.",
        )

    if has_poly and has_fourpt:
        if confirm_quads_are_polygons:
            return SourceLevelScan(
                resolved_level=GeometryLevel.POLYGON,
                is_homogeneous=True,
                reason="Quad files confirmed as genuine contours.",
            )
        return SourceLevelScan(
            resolved_level=GeometryLevel.POLYGON,
            is_homogeneous=False,
            conflict_files=fourpt_files,
            needs_confirmation=True,
            reason="Source mixes polygon files with four-point-only files.",
        )

    if has_poly:
        return SourceLevelScan(GeometryLevel.POLYGON, True)
    if has_fourpt:
        return SourceLevelScan(intended_level, True)
    if has_aabb:
        return SourceLevelScan(GeometryLevel.AABB, True)
    # No labels at all: treat as the intended level, homogeneous.
    return SourceLevelScan(intended_level, True, reason="No non-empty label files found.")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_geometry_levels.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/training/geometry_levels.py tests/test_geometry_levels.py
git commit -m "feat(geometry): source-directory level scanner with homogeneity verdict"
```

---

## Task 3: Store level on `OBBSource` (project file)

**Files:**
- Modify: `src/hydra_suite/detectkit/gui/models.py:26-58` (`OBBSource`)
- Test: `tests/test_geometry_level_import.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `OBBSource.level: str = "obb"` field, serialized in `to_dict`, restored in `from_dict` (missing key => `"obb"`, satisfying the no-op migration).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_geometry_level_import.py
from hydra_suite.detectkit.gui.models import OBBSource


def test_obbsource_level_defaults_to_obb():
    src = OBBSource(path="/x", name="s")
    assert src.level == "obb"


def test_obbsource_level_roundtrips():
    src = OBBSource(path="/x", name="s", level="polygon")
    assert OBBSource.from_dict(src.to_dict()).level == "polygon"


def test_obbsource_from_dict_missing_level_is_obb():
    # Simulates a pre-migration project JSON with no "level" key.
    legacy = {"path": "/x", "name": "s", "validated": True, "source_kind": "detectkit"}
    assert OBBSource.from_dict(legacy).level == "obb"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_geometry_level_import.py -v`
Expected: FAIL — `AttributeError: 'OBBSource' object has no attribute 'level'`

- [ ] **Step 3: Write minimal implementation**

In `src/hydra_suite/detectkit/gui/models.py`, add the field to the `OBBSource` dataclass (after `imported`):

```python
@dataclass
class OBBSource:
    """Represents one source dataset directory."""

    path: str = ""
    name: str = ""
    validated: bool = False
    original_path: str = ""
    source_kind: str = "detectkit"
    imported: bool = False
    level: str = "obb"  # GeometryLevel.label; "obb" for pre-migration sources

    def to_dict(self) -> dict:
        """Serialize to a plain dictionary."""
        return {
            "path": self.path,
            "name": self.name,
            "validated": self.validated,
            "original_path": self.original_path,
            "source_kind": self.source_kind,
            "imported": self.imported,
            "level": self.level,
        }

    @staticmethod
    def from_dict(d: dict) -> OBBSource:
        """Restore an OBBSource from a dictionary."""
        return OBBSource(
            path=str(d.get("path", "")),
            name=str(d.get("name", "")),
            validated=bool(d.get("validated", False)),
            original_path=str(d.get("original_path", "")),
            source_kind=str(d.get("source_kind", "detectkit") or "detectkit"),
            imported=bool(d.get("imported", False)),
            level=str(d.get("level", "obb") or "obb"),
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_geometry_level_import.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/detectkit/gui/models.py tests/test_geometry_level_import.py
git commit -m "feat(detectkit): store geometry level per source in the project file"
```

---

## Task 4: Import records level instead of collapsing segmentation

**Files:**
- Modify: `src/hydra_suite/detectkit/gui/source_import.py` — add `level` to `MaterializedDetectKitSource` (`:36-47`); write full COCO contours (`_coco_annotation_to_obb` `:397`, `_materialize_coco_source` `:408`); detect `aabb`/`obb`/`polygon` level per source; expose it on the result.
- Test: `tests/test_geometry_level_import.py`

**Interfaces:**
- Consumes: `GeometryLevel`, `scan_source_levels` (Tasks 1-2).
- Produces:
  - `MaterializedDetectKitSource.level: str` (a `GeometryLevel.label`).
  - New helper `_coco_annotation_to_points(annotation, width, height) -> tuple[list[float], str] | None` returning `(normalized_coords, evidence)` where evidence is `"polygon"` (from `segmentation`) or `"aabb"` (from `bbox` quad). COCO seg is written as the **full** contour (normalized point list), not `minAreaRect`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_geometry_level_import.py  (append)
import json
from pathlib import Path

import numpy as np
from PIL import Image

from hydra_suite.detectkit.gui.source_import import materialize_detectkit_source


def _img(path: Path, w=20, h=20):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros((h, w, 3), dtype=np.uint8)).save(path)


def test_import_yolo_obb_is_obb_level(tmp_path):
    root = tmp_path / "src"
    _img(root / "images" / "a.png")
    (root / "labels").mkdir(parents=True)
    (root / "labels" / "a.txt").write_text(
        "0 0.1 0.1 0.5 0.1 0.5 0.5 0.1 0.5\n", encoding="utf-8"
    )
    (root / "classes.txt").write_text("object\n", encoding="utf-8")
    mat = materialize_detectkit_source(root, tmp_path / "proj", force_import=True)
    assert mat.level == "obb"


def test_import_yolo_detect_is_aabb_level(tmp_path):
    root = tmp_path / "src"
    _img(root / "images" / "a.png")
    (root / "labels").mkdir(parents=True)
    (root / "labels" / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (root / "classes.txt").write_text("object\n", encoding="utf-8")
    mat = materialize_detectkit_source(root, tmp_path / "proj", force_import=True)
    # detect input keeps aabb information: stored as an axis-aligned quad, level aabb.
    assert mat.level == "aabb"
    lines = (Path(mat.canonical_path) / "labels" / "a.txt").read_text().strip().splitlines()
    assert len(lines[0].split()) == 9  # class + 8 coords (quad), not cx cy w h


def test_import_coco_segmentation_preserved_as_polygon(tmp_path):
    root = tmp_path / "src"
    _img(root / "images" / "a.png", 20, 20)
    payload = {
        "images": [{"id": 1, "file_name": "a.png", "width": 20, "height": 20}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1,
             "segmentation": [[2, 2, 18, 2, 18, 18, 10, 19, 2, 18]], "bbox": [2, 2, 16, 16]}
        ],
        "categories": [{"id": 1, "name": "object"}],
    }
    (root / "annotations.json").write_text(json.dumps(payload), encoding="utf-8")
    mat = materialize_detectkit_source(root, tmp_path / "proj", force_import=True)
    assert mat.level == "polygon"
    line = (Path(mat.canonical_path) / "labels" / "a.txt").read_text().strip().splitlines()[0]
    assert len(line.split()) == 11  # class + 5 points preserved (not collapsed to a quad)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_geometry_level_import.py -v -k import`
Expected: FAIL — `AttributeError: 'MaterializedDetectKitSource' object has no attribute 'level'`

- [ ] **Step 3: Write minimal implementation**

In `src/hydra_suite/detectkit/gui/source_import.py`:

Add the import near the top (after the existing `from hydra_suite.training...` imports):

```python
from hydra_suite.training.geometry_levels import GeometryLevel, scan_source_levels
```

Add `level` to the result dataclass (`MaterializedDetectKitSource`):

```python
@dataclass(slots=True, frozen=True)
class MaterializedDetectKitSource:
    """Result of resolving a selected source into DetectKit's canonical layout."""

    source_root: Path
    canonical_path: Path
    source_kind: str
    display_name: str
    images_count: int
    annotation_count: int
    discovered_labels: list[str]
    imported: bool
    level: str = "obb"
```

Replace `_coco_annotation_to_obb` (`:397-405`) with a points-preserving version:

```python
def _coco_annotation_to_points(
    annotation: dict[str, Any], width: int, height: int
) -> tuple[list[float], str] | None:
    """Return (normalized_coords, evidence). Segmentation is preserved as a full
    contour ("polygon"); a bbox-only annotation yields an axis-aligned quad ("aabb")."""
    points = _coco_segmentation_points(annotation.get("segmentation"))
    if len(points) >= 3:
        coords: list[float] = []
        for x_pos, y_pos in points:
            coords.extend([x_pos / float(width), y_pos / float(height)])
        return coords, "polygon"
    quad = _coco_bbox_to_polygon(annotation.get("bbox"), width, height)
    if quad is not None:
        return quad, "aabb"
    return None
```

In `_materialize_coco_source` (`:449-465`), replace the per-annotation body that calls `_coco_annotation_to_obb` with the points version:

```python
        lines: list[str] = []
        for annotation in annotations_by_image.get(int(image_id), []):
            category_id = annotation.get("category_id")
            if category_id is None:
                continue
            dense_id = category_to_dense.get(int(category_id))
            if dense_id is None:
                continue
            converted = _coco_annotation_to_points(annotation, width, height)
            if converted is None:
                continue
            coords, _evidence = converted
            lines.append(_format_obb_line(dense_id, coords))
```

`_points_to_min_area_rect` is now only referenced by train-time derivations; leave the function in place (used later by builders) but it is no longer called at import.

At the end of `materialize_detectkit_source` (`:528-589`), compute the level from the *materialized* labels before returning. Replace each of the three `return MaterializedDetectKitSource(...)` sites so `level` is set. Add this helper above `materialize_detectkit_source`:

```python
def _detect_source_level(canonical_root: Path, source_kind: str) -> str:
    """Resolve a materialized source's single geometry level from its labels."""
    intended = GeometryLevel.OBB
    if source_kind in {"yolo_detect"}:
        intended = GeometryLevel.AABB
    scan = scan_source_levels(canonical_root / "labels", intended_level=intended)
    return scan.resolved_level.label
```

Then in each `return MaterializedDetectKitSource(...)`, add `level=_detect_source_level(<canonical_path>, inspection.source_kind)` where `<canonical_path>` is `root` for the linked/no-import branches and `dest_root` for the imported branch. For example, the final return becomes:

```python
    return MaterializedDetectKitSource(
        source_root=root,
        canonical_path=dest_root,
        source_kind=inspection.source_kind,
        display_name=root.name,
        images_count=inspection.images_count,
        annotation_count=inspection.annotation_count,
        discovered_labels=list(inspection.discovered_labels),
        imported=True,
        level=_detect_source_level(dest_root, inspection.source_kind),
    )
```

Apply the analogous `level=_detect_source_level(root, inspection.source_kind)` to the two early returns (linked branch `:547` and the not-required-import branch `:559`).

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_geometry_level_import.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/detectkit/gui/source_import.py tests/test_geometry_level_import.py
git commit -m "feat(detectkit): preserve segmentation at import and record source level"
```

---

## Task 5: Stamp level on the source when added (source manager wiring)

**Files:**
- Modify: `src/hydra_suite/detectkit/gui/dialogs/source_manager.py` — where `OBBSource(...)` is constructed from a `MaterializedDetectKitSource`, pass `level=materialized.level`.
- Test: covered by Task 4's import tests + a manual assertion below (no separate red test needed if the construction site already builds `OBBSource`; add an assertion to the existing source-manager test module if present).

**Interfaces:**
- Consumes: `MaterializedDetectKitSource.level` (Task 4), `OBBSource.level` (Task 3).
- Produces: added sources carry their detected level into the project.

- [ ] **Step 1: Locate the construction site**

Run: `grep -n "OBBSource(" src/hydra_suite/detectkit/gui/dialogs/source_manager.py`
Expected: one or more `OBBSource(...)` constructions built from a `materialize_detectkit_source(...)` result.

- [ ] **Step 2: Write the failing test**

```python
# tests/test_geometry_level_import.py  (append)
from hydra_suite.detectkit.gui.source_import import materialize_detectkit_source
from hydra_suite.detectkit.gui.models import OBBSource


def test_added_source_carries_level(tmp_path):
    root = tmp_path / "src"
    _img(root / "images" / "a.png")
    (root / "labels").mkdir(parents=True)
    (root / "labels" / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (root / "classes.txt").write_text("object\n", encoding="utf-8")
    mat = materialize_detectkit_source(root, tmp_path / "proj", force_import=True)
    # The source-manager builds the OBBSource with the materialized level.
    src = OBBSource(
        path=str(mat.canonical_path), name=mat.display_name,
        source_kind=mat.source_kind, imported=mat.imported, level=mat.level,
    )
    assert src.level == "aabb"
```

- [ ] **Step 3: Run test to verify it fails / passes**

Run: `python -m pytest tests/test_geometry_level_import.py::test_added_source_carries_level -v`
Expected: PASS once the manager wiring in Step 4 is done (this test asserts the contract; the wiring makes it real in the app).

- [ ] **Step 4: Wire the level through**

In `source_manager.py`, at each `OBBSource(...)` built from a `materialize_detectkit_source(...)` result, add `level=materialized.level` (use the actual local variable name found in Step 1). Do not change any other field.

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/detectkit/gui/dialogs/source_manager.py tests/test_geometry_level_import.py
git commit -m "feat(detectkit): carry detected level onto sources added via the manager"
```

---

## Task 6: Homogeneity block + confirm-override in the validation dialog

**Files:**
- Modify: `src/hydra_suite/detectkit/gui/dialogs/source_validation.py` — after inspection, run `scan_source_levels` on the source's `labels/`; if non-homogeneous, show a blocking message with the conflict files and, when `needs_confirmation`, offer a "These quads are genuine contours" override that re-scans with `confirm_quads_are_polygons=True`.
- Test: `tests/test_geometry_levels.py` (logic-level; the dialog is thin over the pure scanner).

**Interfaces:**
- Consumes: `scan_source_levels`, `SourceLevelScan` (Task 2).
- Produces: a pure helper `resolve_source_level_or_block(labels_dir, intended_level, confirm) -> SourceLevelScan` importable/testable without Qt, plus the dialog calling it.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_geometry_levels.py  (append)
from hydra_suite.detectkit.gui.dialogs.source_validation import (
    resolve_source_level_or_block,
)
from hydra_suite.training.geometry_levels import GeometryLevel


def test_resolve_blocks_on_mixed(tmp_path):
    labels = tmp_path / "labels"
    _write(labels, "poly.txt", "0 0.1 0.1 0.5 0.1 0.5 0.5 0.1 0.5 0.3 0.7\n")
    _write(labels, "quad.txt", "0 0.1 0.1 0.5 0.1 0.5 0.5 0.1 0.5\n")
    scan = resolve_source_level_or_block(labels, GeometryLevel.OBB, confirm=False)
    assert not scan.is_homogeneous and scan.needs_confirmation


def test_resolve_confirm_override(tmp_path):
    labels = tmp_path / "labels"
    _write(labels, "poly.txt", "0 0.1 0.1 0.5 0.1 0.5 0.5 0.1 0.5 0.3 0.7\n")
    _write(labels, "quad.txt", "0 0.1 0.1 0.5 0.1 0.5 0.5 0.1 0.5\n")
    scan = resolve_source_level_or_block(labels, GeometryLevel.OBB, confirm=True)
    assert scan.is_homogeneous and scan.resolved_level is GeometryLevel.POLYGON
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_geometry_levels.py -v -k resolve`
Expected: FAIL — `ImportError: cannot import name 'resolve_source_level_or_block'`

- [ ] **Step 3: Write minimal implementation**

In `source_validation.py`, add near the top:

```python
from hydra_suite.training.geometry_levels import (
    GeometryLevel,
    SourceLevelScan,
    scan_source_levels,
)


def resolve_source_level_or_block(
    labels_dir,
    intended_level: GeometryLevel = GeometryLevel.OBB,
    *,
    confirm: bool = False,
) -> SourceLevelScan:
    """Resolve a source's level, honoring an explicit quads-are-contours override."""
    return scan_source_levels(
        labels_dir, intended_level=intended_level, confirm_quads_are_polygons=confirm
    )
```

Then, inside `DetectKitSourceValidationDialog`, after building the summary, add a level row and, if the scan blocks, disable the accept buttons and show the reason + conflict files. Add to `__init__` after `layout.addLayout(form)`:

```python
        scan = resolve_source_level_or_block(
            Path(source_root) / "labels",
            _intended_level_for_kind(inspection.source_kind),
        )
        self._level_scan = scan
        self._level_value = QLabel(scan.resolved_level.label)
        form.addRow("Geometry level:", self._level_value)
        if not scan.is_homogeneous:
            warn = QLabel(
                f"Mixed geometry: {scan.reason}\nConflicting files: "
                + ", ".join(scan.conflict_files[:8])
                + (" …" if len(scan.conflict_files) > 8 else "")
            )
            warn.setWordWrap(True)
            warn.setStyleSheet("color: #c0392b;")
            layout.addWidget(warn)
```

Add the module-level helper:

```python
def _intended_level_for_kind(source_kind: str) -> GeometryLevel:
    if source_kind == "yolo_detect":
        return GeometryLevel.AABB
    return GeometryLevel.OBB
```

In `_accept_choice`, refuse to accept a non-homogeneous source unless the user has confirmed the override:

```python
    def _accept_choice(self, mode: str) -> None:
        if not self._level_scan.is_homogeneous:
            if self._level_scan.needs_confirmation:
                from PySide6.QtWidgets import QMessageBox

                answer = QMessageBox.question(
                    self,
                    "Mixed Geometry",
                    "This source mixes polygon files with four-point-only files. "
                    "Confirm the four-point files are genuine contours (treat the "
                    "whole source as polygon)? Choose No to cancel and fix the source.",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )
                if answer != QMessageBox.StandardButton.Yes:
                    return
                self._level_scan = resolve_source_level_or_block(
                    Path(self._path_value.text()) / "labels",
                    GeometryLevel.OBB,
                    confirm=True,
                )
            else:
                from PySide6.QtWidgets import QMessageBox

                QMessageBox.warning(
                    self, "Mixed Geometry",
                    "This source cannot be added until its geometry is homogeneous:\n\n"
                    + self._level_scan.reason,
                )
                return
        self._selection = DetectKitSourceAdditionChoice(mode=mode)
        self.accept()
```

Expose the resolved level so the caller can stamp the source. Add a method and include it in the returned choice by adding a `level: str = "obb"` field to `DetectKitSourceAdditionChoice` and setting it in `_accept_choice` (`level=self._level_scan.resolved_level.label`). The source-manager (Task 5) then prefers `choice.level` when present, falling back to `materialized.level`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_geometry_levels.py -v -k resolve`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/detectkit/gui/dialogs/source_validation.py tests/test_geometry_levels.py
git commit -m "feat(detectkit): block mixed-geometry sources with a confirm-override at validation"
```

---

## Task 7: Variable-length label parser and new training roles

**Files:**
- Modify: `src/hydra_suite/training/contracts.py:10-24` (`TrainingRole`)
- Modify: `src/hydra_suite/training/dataset_builders.py` — add `_parse_geometry_label_lines`
- Test: `tests/test_geometry_level_builders.py`

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - `TrainingRole.DETECT_DIRECT = "detect_direct"`, `TrainingRole.SEGMENT_DIRECT = "segment_direct"`, `TrainingRole.SEQ_CROP_SEGMENT = "seq_crop_segment"`.
  - `_parse_geometry_label_lines(lbl_path: Path) -> list[tuple[int, np.ndarray]]` in `dataset_builders.py`, returning `(class_id, points)` where `points` is `(P, 2)` float32; a 5-field detect line (`cx cy w h`) expands to its 4-corner axis-aligned quad. `_parse_obb_label_lines` (strict 9-field) is left untouched for byte-identical OBB paths.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_geometry_level_builders.py
from pathlib import Path

import numpy as np
import pytest

from hydra_suite.training.contracts import TrainingRole
from hydra_suite.training.dataset_builders import _parse_geometry_label_lines


def test_new_roles_exist():
    assert TrainingRole.DETECT_DIRECT.value == "detect_direct"
    assert TrainingRole.SEGMENT_DIRECT.value == "segment_direct"
    assert TrainingRole.SEQ_CROP_SEGMENT.value == "seq_crop_segment"


def test_parse_polygon_line(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("0 0.1 0.1 0.5 0.1 0.5 0.5 0.3 0.7 0.1 0.5\n", encoding="utf-8")  # 5 pts
    parsed = _parse_geometry_label_lines(p)
    assert parsed[0][0] == 0
    assert parsed[0][1].shape == (5, 2)


def test_parse_detect_line_expands_to_quad(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("2 0.5 0.5 0.2 0.4\n", encoding="utf-8")  # cx cy w h
    cls, pts = _parse_geometry_label_lines(p)[0]
    assert cls == 2 and pts.shape == (4, 2)
    assert np.allclose(pts[0], [0.4, 0.3])  # x1,y1 = cx-w/2, cy-h/2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_geometry_level_builders.py -v`
Expected: FAIL — `AttributeError: DETECT_DIRECT` / `ImportError: _parse_geometry_label_lines`

- [ ] **Step 3: Write minimal implementation**

In `contracts.py`, extend the enum (after `SEQ_CROP_OBB`):

```python
class TrainingRole(str, Enum):
    """Canonical training roles supported by MAT."""

    OBB_DIRECT = "obb_direct"
    DETECT_DIRECT = "detect_direct"
    SEGMENT_DIRECT = "segment_direct"
    SEQ_DETECT = "seq_detect"
    SEQ_CROP_OBB = "seq_crop_obb"
    SEQ_CROP_SEGMENT = "seq_crop_segment"

    # ClassKit classification roles
    CLASSIFY_FLAT_YOLO = "classify_flat_yolo"
    # ... (unchanged)
```

In `dataset_builders.py`, add next to `_parse_obb_label_lines`:

```python
def _parse_geometry_label_lines(lbl_path: Path) -> list[tuple[int, np.ndarray]]:
    """Parse a label file of mixed geometry into (class_id, (P,2) points).

    - 5 fields (cx cy w h) expand to the 4-corner axis-aligned quad.
    - >=7 odd fields are read as a normalized point list of P = (fields-1)/2 points.
    Raises on malformed lines.
    """
    out: list[tuple[int, np.ndarray]] = []
    for raw in lbl_path.read_text(encoding="utf-8").splitlines():
        ln = raw.strip()
        if not ln:
            continue
        parts = ln.split()
        cls_id = int(float(parts[0]))
        coords = [float(v) for v in parts[1:]]
        if len(parts) == 5:
            cx, cy, w, h = coords
            x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
            pts = np.asarray(
                [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32
            )
        elif len(coords) >= 6 and len(coords) % 2 == 0:
            pts = np.asarray(coords, dtype=np.float32).reshape(-1, 2)
        else:
            raise RuntimeError(f"Invalid geometry label line in {lbl_path}: {ln}")
        out.append((cls_id, pts))
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_geometry_level_builders.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/training/contracts.py src/hydra_suite/training/dataset_builders.py tests/test_geometry_level_builders.py
git commit -m "feat(training): add detect/segment/crop-segment roles and a polygon-aware label parser"
```

---

## Task 8: Polygon-aware detect and crop derivations + segment/crop-segment builders

**Files:**
- Modify: `src/hydra_suite/training/dataset_builders.py` — `derive_detect_dataset_from_obb` (`:352`) and `_process_crop_obb_image` (`:503`) switch to `_parse_geometry_label_lines`; add `derive_segment_dataset_from_source` (passthrough) and `derive_crop_segment_dataset_from_source` (clip contour to crop).
- Test: `tests/test_geometry_level_builders.py`

**Interfaces:**
- Consumes: `_parse_geometry_label_lines`, `_convert_obb_to_aabb`, `_extract_crop_for_object` (existing), `_clip_crop` (existing).
- Produces:
  - `derive_detect_dataset_from_obb` now accepts polygon/quad inputs (AABB = min/max over all points).
  - `derive_segment_dataset_from_source(src_dataset_dir, output_root, *, class_name=None, class_names=None) -> DatasetBuildResult` — copies images + label lines verbatim (YOLO-seg passthrough), manifest type `"derived_segment"`.
  - `derive_crop_segment_dataset_from_source(src_dataset_dir, output_root, *, class_name=None, class_names=None, pad_ratio=0.15, min_crop_size_px=64, enforce_square=True) -> DatasetBuildResult` — crops around each contour's AABB and re-normalizes the **clipped** contour into crop space; drops objects whose clipped crop is degenerate; manifest type `"derived_crop_segment"`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_geometry_level_builders.py  (append)
import cv2

from hydra_suite.training.dataset_builders import (
    derive_detect_dataset_from_obb,
    derive_segment_dataset_from_source,
    derive_crop_segment_dataset_from_source,
)


def _mk_dataset(root: Path, label_line: str):
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(root / "images" / split / "a.jpg"), np.zeros((40, 40, 3), np.uint8))
        (root / "labels" / split / "a.txt").write_text(label_line, encoding="utf-8")


def test_detect_from_polygon(tmp_path):
    src = tmp_path / "poly"
    _mk_dataset(src, "0 0.1 0.1 0.9 0.1 0.9 0.9 0.5 0.95 0.1 0.9\n")  # 5-pt contour
    res = derive_detect_dataset_from_obb(src, tmp_path / "out", class_names=["object"])
    line = next((Path(res.dataset_dir) / "labels" / "train").glob("*.txt")).read_text().split()
    assert len(line) == 5  # class + cx cy w h
    assert float(line[3]) == pytest.approx(0.8, abs=1e-3)  # width = 0.9-0.1


def test_segment_passthrough_preserves_points(tmp_path):
    src = tmp_path / "poly"
    line = "0 0.1 0.1 0.9 0.1 0.9 0.9 0.5 0.95 0.1 0.9\n"
    _mk_dataset(src, line)
    res = derive_segment_dataset_from_source(src, tmp_path / "out", class_names=["object"])
    out = next((Path(res.dataset_dir) / "labels" / "train").glob("*.txt")).read_text().strip()
    assert len(out.split()) == 11  # class + 5 points preserved


def test_crop_segment_clips_and_renormalizes(tmp_path):
    src = tmp_path / "poly"
    _mk_dataset(src, "0 0.2 0.2 0.6 0.2 0.6 0.6 0.2 0.6 0.3 0.7\n")
    res = derive_crop_segment_dataset_from_source(
        src, tmp_path / "out", class_names=["object"], enforce_square=False, pad_ratio=0.0
    )
    out = next((Path(res.dataset_dir) / "labels" / "train").glob("*.txt")).read_text().split()
    pts = np.asarray([float(v) for v in out[1:]], dtype=np.float32).reshape(-1, 2)
    assert pts.min() >= 0.0 and pts.max() <= 1.0  # re-normalized into crop space
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_geometry_level_builders.py -v -k "detect_from_polygon or segment or crop_segment"`
Expected: FAIL — import errors for the new builders / AABB derived from a polygon.

- [ ] **Step 3: Write minimal implementation**

In `dataset_builders.py`:

Replace the detection loop in `derive_detect_dataset_from_obb` (`:389-394`) to use the geometry parser and min/max AABB:

```python
            detections = _parse_geometry_label_lines(lbl_path)
            out_lines = []
            for cls_id, pts in detections:
                encountered_class_ids.add(int(cls_id))
                x1, y1 = float(pts[:, 0].min()), float(pts[:, 1].min())
                x2, y2 = float(pts[:, 0].max()), float(pts[:, 1].max())
                x1, y1 = max(0.0, x1), max(0.0, y1)
                x2, y2 = min(1.0, x2), min(1.0, y2)
                bw, bh = max(0.0, x2 - x1), max(0.0, y2 - y1)
                cx, cy = x1 + bw * 0.5, y1 + bh * 0.5
                out_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
```

Add the segment passthrough builder (mirrors `derive_detect_dataset_from_obb`'s scaffolding, copying label text verbatim):

```python
def derive_segment_dataset_from_source(
    src_dataset_dir: str | Path,
    output_root: str | Path,
    class_name: str | None = None,
    *,
    class_names: list[str] | None = None,
) -> DatasetBuildResult:
    """YOLO-seg passthrough: copy images and normalized-contour labels verbatim."""
    resolved_class_names = _normalize_class_names(class_names=class_names, class_name=class_name)
    src = Path(src_dataset_dir).expanduser().resolve()
    out_root = Path(output_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    out_dir = out_root / f"derived_segment_{_timestamp()}"
    for split in ("train", "val", "test"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    counts = {"train": 0, "val": 0, "test": 0, "objects": 0}
    encountered_class_ids: set[int] = set()
    for split in ("train", "val", "test"):
        src_img = src / "images" / split
        src_lbl = src / "labels" / split
        if not src_img.exists():
            continue
        for img_path in sorted(src_img.rglob("*")):
            if img_path.suffix.lower() not in IMAGE_EXTS:
                continue
            lbl_path = _find_label_for_obb_image(img_path, src_img, src_lbl)
            if lbl_path is None:
                continue
            detections = _parse_geometry_label_lines(lbl_path)
            if not detections:
                continue
            for cls_id, _pts in detections:
                encountered_class_ids.add(int(cls_id))
            dst_img, dst_lbl = _unique_dst_pair(out_dir, split, img_path)
            shutil.copy2(img_path, dst_img)
            shutil.copy2(lbl_path, dst_lbl)
            counts[split] += 1
            counts["objects"] += len(detections)

    include_test = counts["test"] > 0
    _validate_class_name_coverage(resolved_class_names, encountered_class_ids, dataset_label="Derived segment dataset")
    _write_dataset_yaml(out_dir, class_names=resolved_class_names, include_test=include_test)
    manifest = {"type": "derived_segment", "source": str(src),
                "created_at": datetime.now().isoformat(timespec="seconds"), "counts": counts}
    manifest_path = out_dir / "manifest.json"
    _write_manifest(manifest_path, manifest)
    return DatasetBuildResult(str(out_dir), stats=manifest, manifest_path=str(manifest_path))
```

Add the crop-segment builder. It reuses the crop geometry of `_extract_crop_for_object` but must handle a variable-length contour, so add a contour-aware crop extractor and a per-image processor:

```python
def _extract_crop_for_contour(
    img: np.ndarray,
    poly_norm: np.ndarray,
    pad_ratio: float,
    min_crop_size_px: int,
    enforce_square: bool,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Crop around a contour's AABB; return (crop, contour re-normalized+clipped to crop)."""
    h, w = img.shape[:2]
    pts_px = poly_norm.astype(np.float32).copy()
    pts_px[:, 0] *= float(w)
    pts_px[:, 1] *= float(h)
    x1, x2 = float(pts_px[:, 0].min()), float(pts_px[:, 0].max())
    y1, y2 = float(pts_px[:, 1].min()), float(pts_px[:, 1].max())
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    cx, cy = x1 + bw * 0.5, y1 + bh * 0.5
    crop_w = max(float(min_crop_size_px), bw * (1.0 + 2.0 * max(0.0, pad_ratio)))
    crop_h = max(float(min_crop_size_px), bh * (1.0 + 2.0 * max(0.0, pad_ratio)))
    if enforce_square:
        crop_w = crop_h = max(crop_w, crop_h)
    c = _clip_crop(cx - crop_w * 0.5, cy - crop_h * 0.5, cx + crop_w * 0.5, cy + crop_h * 0.5, w, h)
    if c is None:
        return None
    xi1, yi1, xi2, yi2 = c
    crop = img[yi1:yi2, xi1:xi2]
    if crop is None or crop.size == 0:
        return None
    ch, cw = crop.shape[:2]
    if ch <= 0 or cw <= 0:
        return None
    contour = pts_px.copy()
    contour[:, 0] = np.clip((contour[:, 0] - float(xi1)) / float(cw), 0.0, 1.0)
    contour[:, 1] = np.clip((contour[:, 1] - float(yi1)) / float(ch), 0.0, 1.0)
    return crop, contour


def _process_crop_segment_image(
    img_path: Path, lbl_path: Path, out_dir: Path, split: str,
    pad_ratio: float, min_crop_size_px: int, enforce_square: bool,
) -> tuple[int, set[int]]:
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None or img.size == 0:
        return 0, set()
    written = 0
    class_ids: set[int] = set()
    for obj_idx, (cls_id, poly_norm) in enumerate(_parse_geometry_label_lines(lbl_path)):
        class_ids.add(int(cls_id))
        result = _extract_crop_for_contour(img, poly_norm, pad_ratio, min_crop_size_px, enforce_square)
        if result is None:
            continue
        crop, contour = result
        stem = f"{img_path.stem}__obj{obj_idx:03d}"
        dst_img, dst_lbl = _unique_crop_output_paths(out_dir, split, stem)
        cv2.imwrite(str(dst_img), crop)
        coords = " ".join(f"{float(v):.6f}" for v in contour.reshape(-1))
        dst_lbl.write_text(f"{cls_id} {coords}\n", encoding="utf-8")
        written += 1
    return written, class_ids


def derive_crop_segment_dataset_from_source(
    src_dataset_dir: str | Path,
    output_root: str | Path,
    class_name: str | None = None,
    *,
    class_names: list[str] | None = None,
    pad_ratio: float = 0.15,
    min_crop_size_px: int = 64,
    enforce_square: bool = True,
) -> DatasetBuildResult:
    """Crop-domain segmentation dataset: clip each contour to its padded crop."""
    resolved_class_names = _normalize_class_names(class_names=class_names, class_name=class_name)
    src = Path(src_dataset_dir).expanduser().resolve()
    out_root = Path(output_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    out_dir = out_root / f"derived_crop_segment_{_timestamp()}"
    for split in ("train", "val", "test"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    counts = {"train": 0, "val": 0, "test": 0, "objects": 0}
    encountered_class_ids: set[int] = set()
    for split in ("train", "val", "test"):
        src_img = src / "images" / split
        src_lbl = src / "labels" / split
        if not src_img.exists():
            continue
        for img_path in sorted(src_img.rglob("*")):
            if img_path.suffix.lower() not in IMAGE_EXTS:
                continue
            lbl_path = _find_label_for_obb_image(img_path, src_img, src_lbl)
            if lbl_path is None:
                continue
            written, class_ids = _process_crop_segment_image(
                img_path, lbl_path, out_dir, split, pad_ratio, min_crop_size_px, enforce_square
            )
            encountered_class_ids.update(class_ids)
            counts[split] += written
            counts["objects"] += written

    include_test = counts["test"] > 0
    _validate_class_name_coverage(resolved_class_names, encountered_class_ids, dataset_label="Derived crop segment dataset")
    _write_dataset_yaml(out_dir, class_names=resolved_class_names, include_test=include_test)
    manifest = {"type": "derived_crop_segment", "source": str(src),
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "pad_ratio": float(pad_ratio), "min_crop_size_px": int(min_crop_size_px),
                "enforce_square": bool(enforce_square), "counts": counts}
    manifest_path = out_dir / "manifest.json"
    _write_manifest(manifest_path, manifest)
    return DatasetBuildResult(str(out_dir), stats=manifest, manifest_path=str(manifest_path))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_geometry_level_builders.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/training/dataset_builders.py tests/test_geometry_level_builders.py
git commit -m "feat(training): polygon-aware detect/crop derivations + segment and crop-segment builders"
```

---

## Task 9: Role dispatch, min-level, and role gating in `prepare_role_dataset`

**Files:**
- Modify: `src/hydra_suite/training/dataset_builders.py` — extend `prepare_role_dataset` (`:618-659`) to dispatch the new roles; add `role_min_level(role) -> GeometryLevel` and `blocked_roles_for_level(level, roles) -> dict[TrainingRole, GeometryLevel]`.
- Test: `tests/test_geometry_level_builders.py`

**Interfaces:**
- Consumes: `GeometryLevel` (Task 1), `TrainingRole` (Task 7), the builders (Task 8).
- Produces:
  - `role_min_level(role: TrainingRole) -> GeometryLevel` per Spec §5:
    `obb_direct→OBB`, `detect_direct→AABB`, `segment_direct→POLYGON`,
    `seq_detect→AABB`, `seq_crop_obb→OBB`, `seq_crop_segment→POLYGON`.
  - `blocked_roles_for_level(merged_level, roles) -> dict[TrainingRole, GeometryLevel]` mapping each requested role whose min level exceeds `merged_level` to its required level.
  - `prepare_role_dataset` handles `DETECT_DIRECT` (→ `derive_detect_dataset_from_obb`), `SEGMENT_DIRECT` (→ `derive_segment_dataset_from_source`), `SEQ_CROP_SEGMENT` (→ `derive_crop_segment_dataset_from_source`), and refuses a role whose min level exceeds the merged level with a message naming the level.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_geometry_level_builders.py  (append)
from hydra_suite.training.dataset_builders import (
    role_min_level, blocked_roles_for_level, prepare_role_dataset,
)
from hydra_suite.training.geometry_levels import GeometryLevel


def test_role_min_levels():
    assert role_min_level(TrainingRole.DETECT_DIRECT) is GeometryLevel.AABB
    assert role_min_level(TrainingRole.OBB_DIRECT) is GeometryLevel.OBB
    assert role_min_level(TrainingRole.SEGMENT_DIRECT) is GeometryLevel.POLYGON
    assert role_min_level(TrainingRole.SEQ_CROP_SEGMENT) is GeometryLevel.POLYGON


def test_blocked_roles_for_aabb_merge():
    roles = [TrainingRole.OBB_DIRECT, TrainingRole.DETECT_DIRECT, TrainingRole.SEGMENT_DIRECT]
    blocked = blocked_roles_for_level(GeometryLevel.AABB, roles)
    assert TrainingRole.OBB_DIRECT in blocked and TrainingRole.SEGMENT_DIRECT in blocked
    assert TrainingRole.DETECT_DIRECT not in blocked


def test_prepare_segment_direct_refused_above_level(tmp_path):
    with pytest.raises(RuntimeError, match="polygon"):
        prepare_role_dataset(
            TrainingRole.SEGMENT_DIRECT, str(tmp_path), tmp_path / "out",
            class_names=["object"], merged_level=GeometryLevel.OBB,
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_geometry_level_builders.py -v -k "role_min or blocked or refused"`
Expected: FAIL — import errors / `prepare_role_dataset` has no `merged_level` kwarg.

- [ ] **Step 3: Write minimal implementation**

In `dataset_builders.py`, import `GeometryLevel` at the top:

```python
from .geometry_levels import GeometryLevel
```

Add the gating helpers above `prepare_role_dataset`:

```python
_ROLE_MIN_LEVEL = {
    TrainingRole.OBB_DIRECT: GeometryLevel.OBB,
    TrainingRole.DETECT_DIRECT: GeometryLevel.AABB,
    TrainingRole.SEGMENT_DIRECT: GeometryLevel.POLYGON,
    TrainingRole.SEQ_DETECT: GeometryLevel.AABB,
    TrainingRole.SEQ_CROP_OBB: GeometryLevel.OBB,
    TrainingRole.SEQ_CROP_SEGMENT: GeometryLevel.POLYGON,
}


def role_min_level(role: TrainingRole) -> GeometryLevel:
    """Minimum geometry level a training role requires."""
    try:
        return _ROLE_MIN_LEVEL[role]
    except KeyError as exc:
        raise RuntimeError(f"Role has no geometry-level requirement: {role}") from exc


def blocked_roles_for_level(
    merged_level: GeometryLevel, roles: list[TrainingRole]
) -> dict[TrainingRole, GeometryLevel]:
    """Roles whose minimum level exceeds the merged dataset's level."""
    blocked: dict[TrainingRole, GeometryLevel] = {}
    for role in roles:
        required = role_min_level(role)
        if required > merged_level:
            blocked[role] = required
    return blocked
```

Extend `prepare_role_dataset` — add a `merged_level: GeometryLevel = GeometryLevel.POLYGON` keyword and dispatch the new roles. Replace the body's role branches:

```python
def prepare_role_dataset(
    role: TrainingRole,
    merged_obb_dataset_dir: str,
    role_output_root: str | Path,
    class_name: str | None = None,
    *,
    class_names: list[str] | None = None,
    crop_pad_ratio: float = 0.15,
    min_crop_size_px: int = 64,
    enforce_square: bool = True,
    merged_level: GeometryLevel = GeometryLevel.POLYGON,
) -> DatasetBuildResult:
    """Prepare role-specific dataset from the merged source."""
    required = role_min_level(role)
    if required > merged_level:
        raise RuntimeError(
            f"Role {role.value} requires {required.label}-level data but the merged "
            f"dataset is {merged_level.label}-level."
        )

    out_root = Path(role_output_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    if role == TrainingRole.OBB_DIRECT:
        manifest_path = Path(merged_obb_dataset_dir) / "manifest.json"
        return DatasetBuildResult(
            dataset_dir=str(Path(merged_obb_dataset_dir).resolve()),
            stats={"type": "passthrough_obb"},
            manifest_path=str(manifest_path) if manifest_path.exists() else "",
        )
    if role in (TrainingRole.SEQ_DETECT, TrainingRole.DETECT_DIRECT):
        return derive_detect_dataset_from_obb(
            merged_obb_dataset_dir, out_root, class_name=class_name, class_names=class_names,
        )
    if role == TrainingRole.SEGMENT_DIRECT:
        return derive_segment_dataset_from_source(
            merged_obb_dataset_dir, out_root, class_name=class_name, class_names=class_names,
        )
    if role == TrainingRole.SEQ_CROP_OBB:
        return derive_crop_obb_dataset_from_obb(
            merged_obb_dataset_dir, out_root, class_name=class_name, class_names=class_names,
            pad_ratio=crop_pad_ratio, min_crop_size_px=min_crop_size_px, enforce_square=enforce_square,
        )
    if role == TrainingRole.SEQ_CROP_SEGMENT:
        return derive_crop_segment_dataset_from_source(
            merged_obb_dataset_dir, out_root, class_name=class_name, class_names=class_names,
            pad_ratio=crop_pad_ratio, min_crop_size_px=min_crop_size_px, enforce_square=enforce_square,
        )

    raise RuntimeError(f"Unsupported training role for dataset preparation: {role}")
```

Note: `merged_level` defaults to `POLYGON` so existing callers that don't pass it keep working (the gate only tightens once the training dialog passes the real merged level in Task 10).

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_geometry_level_builders.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/training/dataset_builders.py tests/test_geometry_level_builders.py
git commit -m "feat(training): dispatch new roles and gate role preparation by merged geometry level"
```

---

## Task 10: Training dialog — new role checkboxes, min()-merge gating, hidden crop-segment

**Files:**
- Modify: `src/hydra_suite/detectkit/gui/models.py` — add `role_detect_direct: bool = False`, `role_segment_direct: bool = False` project fields (persisted automatically via `fields()`).
- Modify: `src/hydra_suite/detectkit/gui/dialogs/training_dialog.py` — add `chk_role_detect_direct`, `chk_role_segment_direct` checkboxes; compute the merged level as `min()` across selected sources' levels; disable+annotate roles blocked by the merged level, naming the lowest blocking source; do NOT expose `seq_crop_segment`.
- Test: `tests/test_geometry_level_builders.py` (pure helper `merged_level_and_blocker`).

**Interfaces:**
- Consumes: `OBBSource.level` (Task 3), `GeometryLevel` (Task 1), `role_min_level`/`blocked_roles_for_level` (Task 9).
- Produces:
  - New pure helper in `training_dialog.py`: `merged_level_and_blocker(sources) -> tuple[GeometryLevel, OBBSource | None]` returning the `min()` level and the source that set it (the blocker), or `(POLYGON, None)` for an empty list.
  - Two new project booleans and their checkboxes, following the exact pattern of the three existing role checkboxes at `:656-661`, `:1077-1079`, `:1136-1138`, `:1711-1716`, `:2170-2172`, `:2212-2216`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_geometry_level_builders.py  (append)
from hydra_suite.detectkit.gui.models import OBBSource
from hydra_suite.detectkit.gui.dialogs.training_dialog import merged_level_and_blocker


def test_merged_level_min_and_blocker():
    sources = [
        OBBSource(path="/a", name="poly", level="polygon"),
        OBBSource(path="/b", name="boxes", level="obb"),
    ]
    level, blocker = merged_level_and_blocker(sources)
    assert level is GeometryLevel.OBB
    assert blocker is not None and blocker.name == "boxes"


def test_merged_level_empty_is_polygon():
    level, blocker = merged_level_and_blocker([])
    assert level is GeometryLevel.POLYGON and blocker is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_geometry_level_builders.py -v -k merged_level`
Expected: FAIL — `ImportError: cannot import name 'merged_level_and_blocker'`

- [ ] **Step 3: Write minimal implementation**

In `models.py`, add two role fields next to the existing role block (`:112-114`):

```python
    # Roles
    role_obb_direct: bool = True
    role_detect_direct: bool = False
    role_segment_direct: bool = False
    role_seq_detect: bool = True
    role_seq_crop_obb: bool = True
```

In `training_dialog.py`, add the module-level helper (near the top, after imports):

```python
from hydra_suite.training.geometry_levels import GeometryLevel


def merged_level_and_blocker(sources):
    """Return (min geometry level across sources, the source that set it)."""
    if not sources:
        return GeometryLevel.POLYGON, None
    blocker = min(sources, key=lambda s: GeometryLevel.from_str(getattr(s, "level", "obb")))
    return GeometryLevel.from_str(getattr(blocker, "level", "obb")), blocker
```

Add the two checkboxes alongside the existing ones (`:656-661`):

```python
        self.chk_role_obb_direct = QCheckBox("obb_direct")
        self.chk_role_detect_direct = QCheckBox("detect_direct")
        self.chk_role_segment_direct = QCheckBox("segment_direct")
        self.chk_role_seq_detect = QCheckBox("seq_detect")
        self.chk_role_seq_crop_obb = QCheckBox("seq_crop_obb")
        self.chk_role_obb_direct.setChecked(True)
        self.chk_role_seq_detect.setChecked(True)
        self.chk_role_seq_crop_obb.setChecked(True)
```

Add the new checkboxes to the layout row list, the toggle-connection list (`:543-545`), the `_collect_selected_roles`-style assembly (`:1711-1716`), the project load/save round-trip (`:1077-1079`, `:1136-1138`), and the JSON role dict (`:2170-2172`, `:2212-2216`) — mirroring the existing three exactly:

```python
        # role assembly (mirrors existing OBB/seq entries)
        if self.chk_role_detect_direct.isChecked():
            roles.append(TrainingRole.DETECT_DIRECT)
        if self.chk_role_segment_direct.isChecked():
            roles.append(TrainingRole.SEGMENT_DIRECT)
```

```python
        # load / save
        self.chk_role_detect_direct.setChecked(proj.role_detect_direct)
        self.chk_role_segment_direct.setChecked(proj.role_segment_direct)
        # ...
        proj.role_detect_direct = self.chk_role_detect_direct.isChecked()
        proj.role_segment_direct = self.chk_role_segment_direct.isChecked()
```

Add a role-gating refresh. Create `_refresh_role_gating(self)` and call it from `_on_role_selection_changed` and whenever the source selection changes:

```python
    def _refresh_role_gating(self) -> None:
        sources = list(self._project.sources) if self._project else []
        level, blocker = merged_level_and_blocker(sources)
        role_checks = {
            TrainingRole.OBB_DIRECT: self.chk_role_obb_direct,
            TrainingRole.DETECT_DIRECT: self.chk_role_detect_direct,
            TrainingRole.SEGMENT_DIRECT: self.chk_role_segment_direct,
            TrainingRole.SEQ_DETECT: self.chk_role_seq_detect,
            TrainingRole.SEQ_CROP_OBB: self.chk_role_seq_crop_obb,
        }
        blocked = blocked_roles_for_level(level, list(role_checks))
        for role, chk in role_checks.items():
            required = blocked.get(role)
            if required is not None:
                chk.setEnabled(False)
                chk.setChecked(False)
                who = blocker.name if blocker is not None else "a source"
                chk.setToolTip(
                    f"{role.value} unavailable: needs {required.label}-level data, but "
                    f"source '{who}' is {level.label}-level."
                )
            else:
                chk.setEnabled(True)
                chk.setToolTip("")
```

Import `blocked_roles_for_level` and `TrainingRole` where not already imported. `seq_crop_segment` gets **no checkbox** (defined in the taxonomy, hidden until piece C).

When assembling the training run, pass the merged level to `prepare_role_dataset(...)`. At the call site (`service.py:304` or the dialog's role loop that invokes preparation), thread `merged_level=merged_level_and_blocker(self._project.sources)[0]`. If preparation is invoked in `service.py`, add a `merged_level` parameter to the service entry and pass it through; the dialog computes and supplies it.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_geometry_level_builders.py -v -k merged_level`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/detectkit/gui/models.py src/hydra_suite/detectkit/gui/dialogs/training_dialog.py tests/test_geometry_level_builders.py
git commit -m "feat(detectkit): expose detect/segment roles and gate them by merged geometry level"
```

---

## Task 11: Level-aware X-AnyLabeling launch + validating sync-back

**Files:**
- Modify: `src/hydra_suite/detectkit/gui/panels/dataset_panel.py` — derive the `--mode` from the selected source's level (`:636-640`); in `_sync_xal_stage_back` (`:437-447`), re-scan the staged labels, run homogeneity, and update the source's level in the project before copying back.
- Test: `tests/test_geometry_levels.py` (pure helper `xal_mode_for_level`).

**Interfaces:**
- Consumes: `GeometryLevel` (Task 1), `scan_source_levels` (Task 2), `OBBSource.level` (Task 3).
- Produces:
  - `xal_mode_for_level(level: GeometryLevel) -> str`: `AABB→"rectangle"`, `OBB→"obb"`, `POLYGON→"polygon"`.
  - `_sync_xal_stage_back` validates before copying and refuses a mixed staged source (raising, surfaced as a Qt warning), then stamps the recomputed level onto the project's `OBBSource`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_geometry_levels.py  (append)
from hydra_suite.detectkit.gui.panels.dataset_panel import xal_mode_for_level


def test_xal_mode_for_level():
    assert xal_mode_for_level(GeometryLevel.AABB) == "rectangle"
    assert xal_mode_for_level(GeometryLevel.OBB) == "obb"
    assert xal_mode_for_level(GeometryLevel.POLYGON) == "polygon"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_geometry_levels.py -v -k xal_mode`
Expected: FAIL — `ImportError: cannot import name 'xal_mode_for_level'`

- [ ] **Step 3: Write minimal implementation**

In `dataset_panel.py`, add near the top imports:

```python
from hydra_suite.training.geometry_levels import GeometryLevel, scan_source_levels


def xal_mode_for_level(level: GeometryLevel) -> str:
    """Map a geometry level to the X-AnyLabeling convert --mode value."""
    return {
        GeometryLevel.AABB: "rectangle",
        GeometryLevel.OBB: "obb",
        GeometryLevel.POLYGON: "polygon",
    }[level]
```

Add a helper to fetch the selected source's `OBBSource`:

```python
    def _selected_source_obj(self):
        path = self._selected_source_path()
        if path is None or self._project is None:
            return None
        for src in self._project.sources:
            if src.path == path:
                return src
        return None
```

In `_open_xanylabeling` (`:636-640`), derive the mode from the source level (default `obb` for unknown, so behavior is unchanged for existing sources):

```python
        src_obj = self._selected_source_obj()
        level = GeometryLevel.from_str(getattr(src_obj, "level", "obb")) if src_obj else GeometryLevel.OBB
        mode = xal_mode_for_level(level)
        convert_cmd = (
            f"xanylabeling convert --task yolo2xlabel --mode {mode} "
            "--images ./images --labels ./labels --output ./images "
            "--classes classes.txt"
        )
```

> **Implementation-time verification (Spec §6):** confirm the exact `--mode` vocabulary (`rectangle`/`obb`/`polygon`) accepted by the installed `xanylabeling convert` CLI (`integrations/xanylabeling/cli.py:33`) before relying on `rectangle`/`polygon`; `obb` is the known-good anchor. If the CLI names differ, adjust the mapping in `xal_mode_for_level` only.

Rewrite `_sync_xal_stage_back` to validate first (launch mode = declared intent), then update the level:

```python
    def _sync_xal_stage_back(self, source_dir: Path, stage_dir: Path) -> None:
        """Validate staged labels, then copy back and update the source's level."""
        labels_src = stage_dir / "labels"
        if not labels_src.exists():
            classes_src = stage_dir / "classes.txt"
            if classes_src.exists():
                shutil.copyfile(classes_src, source_dir / "classes.txt")
            return

        src_obj = self._selected_source_obj()
        intended = GeometryLevel.from_str(getattr(src_obj, "level", "obb")) if src_obj else GeometryLevel.OBB
        scan = scan_source_levels(labels_src, intended_level=intended)
        if not scan.is_homogeneous:
            raise RuntimeError(
                "Edited labels are not homogeneous and were not copied back:\n"
                + scan.reason
                + "\nConflicting files: "
                + ", ".join(scan.conflict_files[:8])
            )

        labels_dst = source_dir / "labels"
        shutil.rmtree(labels_dst, ignore_errors=True)
        _copy_tree_without_metadata(labels_src, labels_dst)
        classes_src = stage_dir / "classes.txt"
        if classes_src.exists():
            shutil.copyfile(classes_src, source_dir / "classes.txt")

        if src_obj is not None:
            src_obj.level = scan.resolved_level.label
```

In `_refresh_labels` (`:695-712`), wrap the `_sync_xal_stage_back` call in try/except to surface the block as a Qt warning:

```python
        if convert_dir == stage_dir:
            try:
                self._sync_xal_stage_back(source_dir, stage_dir)
            except RuntimeError as exc:
                QMessageBox.warning(self, "Mixed Geometry", str(exc))
                return
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_geometry_levels.py -v -k xal_mode`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/detectkit/gui/panels/dataset_panel.py tests/test_geometry_levels.py
git commit -m "feat(detectkit): level-aware X-AnyLabeling launch and validating sync-back"
```

---

## Task 12: `OBBResult.polygons` field

**Files:**
- Modify: `src/hydra_suite/core/inference/result.py:19-31` (`OBBResult`)
- Test: `tests/test_geometry_level_export.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `OBBResult.polygons: list[np.ndarray] | None = None` — native contours in frame pixel space, export-only, defaults to `None`, never serialized.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_geometry_level_export.py
import numpy as np

from hydra_suite.core.inference.result import OBBResult


def _empty_result():
    return OBBResult(
        frame_idx=0,
        centroids=np.zeros((0, 2), np.float32),
        angles=np.zeros((0,), np.float32),
        sizes=np.zeros((0,), np.float32),
        shapes=np.zeros((0, 2), np.float32),
        confidences=np.zeros((0,), np.float32),
        corners=np.zeros((0, 4, 2), np.float32),
        detection_ids=np.zeros((0,), np.int64),
    )


def test_polygons_defaults_none():
    assert _empty_result().polygons is None


def test_polygons_settable():
    r = _empty_result()
    r.polygons = [np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]], np.float32)]
    assert len(r.polygons) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_geometry_level_export.py -v`
Expected: FAIL — `TypeError` (unexpected keyword) or `AttributeError` on `.polygons`

- [ ] **Step 3: Write minimal implementation**

In `result.py`, add the field after `class_ids` (`:29-31`):

```python
    class_ids: np.ndarray | None = (
        None  # (D,) int64 model class id per detection; None => all class 0
    )
    # Native contours per detection in frame pixel space, populated ONLY when a
    # detection stage is asked to emit native geometry (export-only). Never
    # serialized to the .npz cache; the tracking hot path leaves this None.
    polygons: list[np.ndarray] | None = None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_geometry_level_export.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/core/inference/result.py tests/test_geometry_level_export.py
git commit -m "feat(inference): add export-only OBBResult.polygons field"
```

---

## Task 13: Extractors emit native polygons on request

**Files:**
- Modify: `src/hydra_suite/core/inference/config.py` — add `OBBConfig.emit_native_geometry: bool = False`.
- Modify: `src/hydra_suite/core/inference/stages/obb.py` — `extract_obb_result` (`:784`), `_extract_obb_from_boxes` (`:851`), `_extract_obb_from_masks` (`:904`) gain `emit_native_geometry: bool = False` and populate `OBBResult.polygons` when true.
- Test: `tests/test_geometry_level_export.py`

**Interfaces:**
- Consumes: `OBBResult.polygons` (Task 12).
- Produces: each extractor, when `emit_native_geometry=True`, sets `result.polygons` to a list of `(P, 2)` float32 arrays in **frame pixel space**:
  - `extract_obb_result` (OBB): the 4 corners already computed (`corners[i]`).
  - `_extract_obb_from_boxes` (detect): the axis-aligned quad from `corners[i]`.
  - `_extract_obb_from_masks` (segment): the native mask contour (the same contour the extractor derives the OBB from).
  Default `False` keeps the hot path byte-identical.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_geometry_level_export.py  (append)
import pytest
from hydra_suite.core.inference.stages.obb import _extract_obb_from_boxes


class _FakeBoxes:
    def __init__(self, xyxy, conf):
        import torch
        self.xyxy = torch.tensor(xyxy, dtype=torch.float32)
        self.conf = torch.tensor(conf, dtype=torch.float32)


class _FakeDetectResult:
    def __init__(self, xyxy, conf):
        self.boxes = _FakeBoxes(xyxy, conf)


def test_detect_extractor_emits_quad_polygons():
    torch = pytest.importorskip("torch")
    res = _FakeDetectResult([[10.0, 20.0, 30.0, 60.0]], [0.9])
    out = _extract_obb_from_boxes(res, frame_idx=0, fixed_angle_rad=0.0, emit_native_geometry=True)
    assert out.polygons is not None
    assert out.polygons[0].shape == (4, 2)


def test_detect_extractor_default_no_polygons():
    pytest.importorskip("torch")
    res = _FakeDetectResult([[10.0, 20.0, 30.0, 60.0]], [0.9])
    out = _extract_obb_from_boxes(res, frame_idx=0, fixed_angle_rad=0.0)
    assert out.polygons is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_geometry_level_export.py -v -k detect_extractor`
Expected: FAIL — `_extract_obb_from_boxes()` got an unexpected keyword `emit_native_geometry`

- [ ] **Step 3: Write minimal implementation**

In `config.py`, add the field to `OBBConfig` (locate `class OBBConfig`, add near its other flags):

```python
    emit_native_geometry: bool = False  # export-only; populate OBBResult.polygons
```

Confirm `OBBConfig.from_dict` (used by `_dict_to_config`) tolerates the new key; if `from_dict` enumerates known keys, add `emit_native_geometry` to it defaulting to `False`.

In `obb.py`, add the parameter and population to each extractor. For `_extract_obb_from_boxes` (`:851`), change the signature and the return:

```python
def _extract_obb_from_boxes(
    result: Any,
    frame_idx: int,
    fixed_angle_rad: float,
    *,
    emit_native_geometry: bool = False,
) -> OBBResult:
```

and before `return OBBResult(...)` (`:892`):

```python
    out = OBBResult(
        frame_idx=frame_idx,
        centroids=np.stack([cx, cy], axis=1).astype(np.float32),
        angles=angles_fixed,
        sizes=sizes,
        shapes=np.stack([sizes, aspect], axis=1).astype(np.float32),
        confidences=conf.astype(np.float32),
        corners=corners.astype(np.float32),
        detection_ids=OBBResult.make_detection_ids(frame_idx, n),
    )
    if emit_native_geometry:
        out.polygons = [corners[i].astype(np.float32).copy() for i in range(n)]
    return out
```

For `extract_obb_result` (`:784`), add `*, emit_native_geometry: bool = False` to the signature and, before the final return, capture the corners as polygons:

```python
    out = OBBResult(
        frame_idx=frame_idx,
        centroids=centroids.astype(np.float32),
        angles=angles_fixed,
        sizes=sizes,
        shapes=np.stack([sizes, aspect], axis=1).astype(np.float32),
        confidences=conf.astype(np.float32),
        corners=corners.astype(np.float32),
        detection_ids=OBBResult.make_detection_ids(frame_idx, n),
        class_ids=cls,
    )
    if emit_native_geometry:
        out.polygons = [corners[i].astype(np.float32).copy() for i in range(n)]
    return out
```

For `_extract_obb_from_masks` (`:904`), add `emit_native_geometry: bool = False` to the keyword-only params and, where the extractor already computes per-detection mask contours (the polygons it reduces to an OBB), retain them into `out.polygons` when the flag is set. The contour is in the same frame pixel space as `corners`; populate `out.polygons` as the list of `(P_i, 2)` float32 contours in detection-id order, falling back to `corners[i]` for any detection whose contour was unavailable.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_geometry_level_export.py -v -k detect_extractor`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/core/inference/config.py src/hydra_suite/core/inference/stages/obb.py tests/test_geometry_level_export.py
git commit -m "feat(inference): extractors emit native contours into OBBResult.polygons on request"
```

---

## Task 14: AL export writes point lists and stamps source level

**Files:**
- Modify: `src/hydra_suite/detectkit/gui/prediction_preview.py` — add export helpers that request native geometry and return `(tuple, polygon)` per detection.
- Modify: `src/hydra_suite/detectkit/jobs/al_worker.py` — `_write_yolo_obb_label` (`:112`) generalizes to write a point list; detections may carry a native polygon; stamp the new `OBBSource.level` from the model task.
- Modify: `src/hydra_suite/detectkit/gui/main_window.py:1369-1424` — the AL detector_fn returns detections carrying native polygons, plus a resolved export level.
- Test: `tests/test_geometry_level_export.py`

**Interfaces:**
- Consumes: extractor `emit_native_geometry` (Task 13), `OBBResult.polygons` (Task 12), `OBBSource.level` (Task 3), `GeometryLevel` (Task 1).
- Produces:
  - `_write_geometry_label(path, records, frame_size)` in `al_worker.py`, where each record is `(cx, cy, w, h, theta, conf, polygon_or_none)`; a record with a polygon writes the normalized point list, otherwise the OBB corners (preserving today's output byte-for-byte when polygon is `None`).
  - The AL detector_fn contract extended: tuples may be 7-length `(cx, cy, w, h, theta, conf, polygon)` where `polygon` is an `(P, 2)` pixel-space array or `None`. `_frame_signals` and `_write_yolo_obb_label` tolerate both 6- and 7-length tuples.
  - `ALRequest.export_level: str = "obb"`; the created `OBBSource` is stamped with it.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_geometry_level_export.py  (append)
from pathlib import Path

import numpy as np

from hydra_suite.detectkit.jobs.al_worker import _write_geometry_label


def test_write_geometry_label_polygon(tmp_path):
    path = tmp_path / "a.txt"
    poly = np.array([[10, 20], [30, 20], [30, 60], [20, 70], [10, 60]], np.float32)  # 5 pts
    records = [(20.0, 40.0, 20.0, 40.0, 0.0, 0.9, poly)]
    _write_geometry_label(path, records, frame_size=(100, 100))
    fields = path.read_text().strip().split()
    assert fields[0] == "0" and len(fields) == 11  # class + 5 points
    assert 0.0 <= min(float(v) for v in fields[1:]) and max(float(v) for v in fields[1:]) <= 1.0


def test_write_geometry_label_none_matches_obb(tmp_path):
    path = tmp_path / "a.txt"
    records = [(50.0, 50.0, 20.0, 10.0, 0.0, 0.9, None)]
    _write_geometry_label(path, records, frame_size=(100, 100))
    fields = path.read_text().strip().split()
    assert fields[0] == "0" and len(fields) == 9  # class + 8 coords (OBB corners)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_geometry_level_export.py -v -k write_geometry`
Expected: FAIL — `ImportError: cannot import name '_write_geometry_label'`

- [ ] **Step 3: Write minimal implementation**

In `al_worker.py`, add the generalized writer and keep the old one delegating to it for back-compat:

```python
def _write_geometry_label(
    path: Path, records: list, frame_size: tuple[int, int]
) -> None:
    """Write YOLO labels: a native polygon when present, else OBB corners."""
    h, w = frame_size
    with path.open("w") as fp:
        for rec in records:
            cx, cy, ww, hh, theta, _conf = rec[:6]
            polygon = rec[6] if len(rec) > 6 else None
            if polygon is not None:
                pts = np.asarray(polygon, dtype=np.float32).reshape(-1, 2).copy()
            else:
                pts = _detection_corners(cx, cy, ww, hh, theta)
            pts[:, 0] = np.clip(pts[:, 0] / w, 0.0, 1.0)
            pts[:, 1] = np.clip(pts[:, 1] / h, 0.0, 1.0)
            line = "0 " + " ".join(f"{v:.6f}" for v in pts.reshape(-1)) + "\n"
            fp.write(line)


def _write_yolo_obb_label(
    path: Path, detections: list, frame_size: tuple[int, int]
) -> None:
    _write_geometry_label(path, detections, frame_size)
```

Make `_frame_signals` (`:78-109`) tolerant of the extra tuple element — it reads `d[5]` for confidence and `d[:5]` for corners, both still valid for 7-length tuples, so no change is required there; add a comment noting 7-length tuples are accepted.

Add `export_level` to `ALRequest` (`:40-56`):

```python
    base_iou: float = 0.7
    export_level: str = "obb"
```

Stamp the level on the created source (`:211-218`):

```python
    new_source = OBBSource(
        path=str(source_root),
        name=f"al_round_{timestamp}",
        validated=False,
        original_path=req.input_path,
        source_kind="detectkit_al",
        imported=True,
        level=req.export_level,
    )
```

Replace the `_write_yolo_obb_label(...)` call in the write loop (`:202-206`) with `_write_geometry_label(...)` (identical signature).

In `prediction_preview.py`, add export variants that request native geometry. Model on `predict_obb_for_frame` (`:312-329`) which currently returns `_tuples_from_obb_result(obb)`; add:

```python
def _tuples_with_polygons_from_obb_result(obb):
    """Like _tuples_from_obb_result but append each detection's native polygon (or None)."""
    base = _tuples_from_obb_result(obb)
    polys = obb.polygons if obb.polygons is not None else [None] * len(base)
    return [(*t, polys[i] if i < len(polys) else None) for i, t in enumerate(base)]


def predict_obb_for_frame_export(model, frame, *, device="auto", conf=0.25, iou=0.7):
    """Export-oriented direct inference: detections carry native polygons."""
    obb = _predict_obb_result_for_frame(model, frame, device=device, conf=conf, iou=iou,
                                         emit_native_geometry=True)
    if obb is None:
        return []
    return _tuples_with_polygons_from_obb_result(obb)
```

Thread `emit_native_geometry=True` into the direct-inference helper `_predict_obb_result_for_frame` (the function at `:188-198` that calls `extract_obb_result(results[0], frame_idx=0)`), adding an `emit_native_geometry: bool = False` parameter and passing it to `extract_obb_result`.

In `main_window.py` `_load_active_detector_fn` (`:1369-1424`), (a) use `predict_obb_for_frame_export` for the direct branch so tuples carry polygons, and (b) resolve and store the export level so `_start_al_round` can pass it. Add a small resolver:

```python
    def _resolve_export_level(self, kind: str) -> str:
        # segment stage-2 -> polygon; obb -> obb; detect-only -> aabb.
        # Direct OBB / sequential-with-OBB stage-2 keep obb.
        return "obb"
```

and in `_start_al_round` (`:1330-1342`) pass `export_level=self._resolve_export_level(kind)` — capture `kind` from `detectkit_resolve_inference_models`. (For this task the direct/obb path stamps `obb`; segment/detect export levels become meaningful once a segment/detect model is the active model, and `_resolve_export_level` is the single place to extend.)

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_geometry_level_export.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/detectkit/gui/prediction_preview.py src/hydra_suite/detectkit/jobs/al_worker.py src/hydra_suite/detectkit/gui/main_window.py tests/test_geometry_level_export.py
git commit -m "feat(detectkit): AL export carries native polygons and stamps the source level"
```

---

## Task 15: Byte-identical regression gate for OBB-only projects

**Files:**
- Test: `tests/test_geometry_level_regression.py`

**Interfaces:**
- Consumes: `merge_obb_sources`, `derive_detect_dataset_from_obb`, `derive_crop_obb_dataset_from_obb` (existing), `_parse_obb_label_lines` (unchanged).
- Produces: a regression test proving that for an `obb`-level source, the merged and derived label bytes are unchanged by this feature.

This is the guarantee named in Spec §10: a data-model change must not perturb existing training data. Because every new code path keys on a non-`obb` level (`derive_detect_dataset_from_obb` now uses `_parse_geometry_label_lines`, which for a 9-field OBB line returns the same 4 points, and `_convert_obb_to_aabb` math is unchanged), OBB-only output must be identical.

- [ ] **Step 1: Write the failing/observing test**

```python
# tests/test_geometry_level_regression.py
from pathlib import Path

import cv2
import numpy as np

from hydra_suite.training.contracts import SourceDataset, SplitConfig
from hydra_suite.training.dataset_builders import (
    merge_obb_sources, derive_detect_dataset_from_obb,
)


def _obb_source(root: Path):
    for split in ("all",):
        (root / "images").mkdir(parents=True, exist_ok=True)
        (root / "labels").mkdir(parents=True, exist_ok=True)
    for i in range(4):
        cv2.imwrite(str(root / "images" / f"f{i}.jpg"), np.zeros((32, 32, 3), np.uint8))
        (root / "labels" / f"f{i}.txt").write_text(
            "0 0.10 0.12 0.51 0.13 0.49 0.55 0.11 0.52\n", encoding="utf-8"
        )
    (root / "classes.txt").write_text("object\n", encoding="utf-8")


def test_obb_only_detect_derivation_is_stable(tmp_path):
    src = tmp_path / "src"
    _obb_source(src)
    merged = merge_obb_sources(
        [SourceDataset(path=str(src), name="s")], tmp_path / "merged",
        SplitConfig(0.75, 0.25, 0.0), class_names=["object"], seed=7,
    )
    detect = derive_detect_dataset_from_obb(merged.dataset_dir, tmp_path / "det", class_names=["object"])
    # Golden AABB for the OBB above: x in [0.10,0.51], y in [0.12,0.55].
    for lbl in (Path(detect.dataset_dir) / "labels").rglob("*.txt"):
        cx, cy, bw, bh = (float(v) for v in lbl.read_text().split()[1:])
        assert abs(bw - 0.41) < 1e-4 and abs(bh - 0.43) < 1e-4
        assert abs(cx - 0.305) < 1e-4 and abs(cy - 0.335) < 1e-4
```

- [ ] **Step 2: Run test to verify it passes**

Run: `python -m pytest tests/test_geometry_level_regression.py -v`
Expected: PASS (the derivations are numerically stable for OBB input). If it FAILS, the polygon-aware parser changed OBB behavior — fix the parser/derivation before proceeding.

- [ ] **Step 3: Run the whole new suite together**

Run: `python -m pytest tests/test_geometry_levels.py tests/test_geometry_level_builders.py tests/test_geometry_level_import.py tests/test_geometry_level_export.py tests/test_geometry_level_regression.py -v`
Expected: PASS

- [ ] **Step 4: Run the delta gate against the base suite**

Run: `python -m pytest -q` (compare failures to the ~24 known pre-existing base-suite failures; no NEW failures attributable to this change).

- [ ] **Step 5: Commit**

```bash
git add tests/test_geometry_level_regression.py
git commit -m "test(geometry): byte-identical regression gate for OBB-only derivations"
```

---

## Task 16: Format, lint, and final verification

**Files:** none (verification only)

- [ ] **Step 1: Format**

Run: `make format`
Expected: black + isort clean.

- [ ] **Step 2: Lint**

Run: `make lint-moderate`
Expected: no new moderate issues in the modified files.

- [ ] **Step 3: Full new-suite run**

Run: `python -m pytest tests/test_geometry_levels.py tests/test_geometry_level_builders.py tests/test_geometry_level_import.py tests/test_geometry_level_export.py tests/test_geometry_level_regression.py -v`
Expected: PASS

- [ ] **Step 4: Commit any formatting changes**

```bash
git add -A
git commit -m "style(geometry): apply formatter to geometry-level feature"
```

---

## Self-Review Notes (spec coverage)

- §3 GeometryLevel + ordering → Task 1. §3a syntax-unchanged (aabb=quad) → Tasks 4, 7. §3b level on project file → Task 3. §3c import stops downgrading → Task 4.
- §4 homogeneous sources + confirm-override → Tasks 2, 6.
- §5 six roles + min levels → Tasks 7, 9; §5a builders → Task 8; §5b min()-merge + named blocker + hidden crop-segment → Task 10.
- §6 level-aware X-AnyLabeling round-trip + validating sync-back → Task 11 (with the CLI-vocabulary verification note carried inline).
- §7 AL export richest geometry via opt-in `OBBResult.polygons` → Tasks 12, 13, 14.
- §8 no-op migration → Task 3 (`level` default `"obb"`, missing-key restore).
- §9 loud failures → Task 6 (validation block), Task 9 (role refusal), plus existing empty-object handling in the builders.
- §10 testing (level detection, homogeneity, each derivation, round-trip recompute, export per task, regression gate) → Tasks 1–2, 6, 8, 11, 13–15.

**Deferred within piece A (by design, not a gap):** `seq_crop_segment` is in the taxonomy and builders (Tasks 7–9) but has no training-dialog checkbox (Task 10) — it stays hidden until piece C makes sequential-mode segmentation runnable in the inference pipeline.
