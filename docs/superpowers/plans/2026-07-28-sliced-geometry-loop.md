# Close the Sliced-Geometry Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the trained slice geometry flow end-to-end — DetectKit measures & stamps a real `reference_body_px`, its preview tiles at the trained scale, and TrackerKit reads the model's `.slice_meta.json` to pre-fill the SAHI panel.

**Architecture:** Three chained components. (A) `build_sliced_obb_dataset` measures a dataset-level median object size and records the *actual* reference into the manifest (→ model sidecar); the DetectKit dialog populates `SliceTrainingSettings.reference_body_px` from it when unset. (B) DetectKit `auto_object` preview computes `object_tile_fraction = median(target_sizes)/imgsz`. (C) A new pure `core/inference/slice_meta.py` reads the sidecar and maps it to panel/config values; TrackerKit pre-fills the SAHI panel on model selection (scale-independent knobs + a new, distinctly-named `slice_trained_body_px`), leaving `REFERENCE_BODY_SIZE` untouched; `SLICE_TRAINED_BODY_PX` overrides `SliceConfig.reference_body_px` when set.

**Tech Stack:** Python 3, NumPy, OpenCV (`cv2`), Ultralytics YOLO datasets, PySide6/Qt, pytest.

## Global Constraints

- **Test env:** the base python has a broken torch import; run every test with `PYTHONPATH="$(pwd)/src" conda run -n hydra-mps python -m pytest ...`. Base suite has ~pre-existing failures → delta gate (a task's own tests + the specific guard suites it touches).
- **`REFERENCE_BODY_SIZE` is sacrosanct.** It is the ant's size in the full tracking frame (`spin_reference_body_size`, range 1–500). This work MUST NEVER read from or write to it. The trained scale is a separate, model-internal value carried by a NEW param `SLICE_TRAINED_BODY_PX` (advanced-config key `slice_trained_body_px`).
- **Defaults-off / non-invasive:** with no sidecar and `SLICE_TRAINED_BODY_PX` absent/0, TrackerKit behavior is byte-identical to today. With a user-set `SliceTrainingSettings.reference_body_px`, the DetectKit build must not overwrite it.
- **Shared median convention:** `object_tile_fraction = median(target_sizes)/imgsz` is used in BOTH the DetectKit preview (Component B) and the TrackerKit mapper (Component C). Both fall back to the stored `object_tile_fraction` when `target_sizes` is empty OR `imgsz` is absent/0, and both clamp the result to `[0.01, 0.9]`.
- **Layer purity:** `core/inference/slice_meta.py` imports only stdlib + numpy (no Qt, no training). `preview_object_tile_fraction` lives in the already-Qt-free `detectkit/gui/prediction_preview.py`.
- **Metadata WRITE only** on the DetectKit side; the TrackerKit READ side is exactly Components C — nothing forces values at inference time, the panel/advanced-config remain user-editable.
- **Commit identity:** commit as the configured git user; NO `Co-Authored-By: Claude` trailer.

---

### Task 1: Builder — dataset-level median reference + manifest stamping (Component A)

**Files:**
- Modify: `src/hydra_suite/training/sliced_dataset.py` (`measure_reference_body_px`, `build_sliced_obb_dataset`, `_slice_geometry_manifest`)
- Test: `tests/test_sliced_dataset_reference.py`

**Interfaces:**
- Consumes: existing `build_sliced_obb_dataset(merged_obb_dataset_dir, output_root, *, level, params, seed=42) -> DatasetBuildResult`; `_parse_geometry_label_lines`; `cv2.minAreaRect`.
- Produces:
  - `object_major_axes_px(labels, frame_wh) -> list[float]` — all per-object minAreaRect major axes in frame pixels (the list `measure_reference_body_px` currently medians internally).
  - `build_sliced_obb_dataset`'s returned `DatasetBuildResult.stats` (== the manifest dict) gains a top-level key `"measured_reference_body_px": float`, and `stats["slice_geometry"]["reference_body_px"]` now records the ACTUAL reference used (explicit if `params.reference_body_px > 0`, else the measured median).

- [ ] **Step 1: Write the failing test**

Create `tests/test_sliced_dataset_reference.py`:

```python
from pathlib import Path

import cv2
import numpy as np

from hydra_suite.training.geometry_levels import GeometryLevel
from hydra_suite.training.sliced_dataset import (
    SliceBuildParams, build_sliced_obb_dataset, object_major_axes_px,
)


def _rect_norm(cx, cy, w, h, W=512, H=512):
    pts = np.array([[cx - w / 2, cy - h / 2], [cx + w / 2, cy - h / 2],
                    [cx + w / 2, cy + h / 2], [cx - w / 2, cy + h / 2]], dtype=np.float32)
    pts[:, 0] /= W
    pts[:, 1] /= H
    return pts


def test_object_major_axes_px_returns_all_majors():
    labels = [(0, _rect_norm(100, 100, 40, 20)), (0, _rect_norm(200, 200, 80, 20))]
    majors = object_major_axes_px(labels, (512, 512))
    assert sorted(round(m) for m in majors) == [40, 80]


def _write_dataset(root: Path, majors_px):
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
    def obb_line(cx, cy, w, h):
        p = _rect_norm(cx, cy, w, h)
        return "0 " + " ".join(f"{v:.6f}" for v in p.reshape(-1))
    for split in ("train", "val"):
        cv2.imwrite(str(root / "images" / split / "f0.jpg"), np.zeros((512, 512, 3), np.uint8))
        # two objects of the given major sizes (square, so major == side)
        lines = [obb_line(120, 120, majors_px[0], majors_px[0]),
                 obb_line(360, 360, majors_px[1], majors_px[1])]
        (root / "labels" / split / "f0.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (root / "dataset.yaml").write_text(
        f"path: {root.resolve()}\ntrain: images/train\nval: images/val\nnames:\n  0: object\n",
        encoding="utf-8",
    )
    return root


def test_manifest_records_measured_reference_when_params_zero(tmp_path):
    merged = _write_dataset(tmp_path / "merged", majors_px=(40, 80))
    params = SliceBuildParams(geometry_mode="custom", slice_width=256, slice_height=256,
                              target_sizes=[], full_frame_mix=False, negative_tile_fraction=0.0,
                              reference_body_px=0.0)
    out = build_sliced_obb_dataset(str(merged), str(tmp_path / "out"),
                                   level=GeometryLevel.OBB, params=params, seed=1)
    # 4 objects across 2 frames: majors 40,80,40,80 -> median 60.
    assert abs(out.stats["measured_reference_body_px"] - 60.0) < 1.0
    assert abs(out.stats["slice_geometry"]["reference_body_px"] - 60.0) < 1.0


def test_manifest_records_explicit_reference_when_params_set(tmp_path):
    merged = _write_dataset(tmp_path / "merged", majors_px=(40, 80))
    params = SliceBuildParams(geometry_mode="custom", slice_width=256, slice_height=256,
                              target_sizes=[], full_frame_mix=False, negative_tile_fraction=0.0,
                              reference_body_px=123.0)
    out = build_sliced_obb_dataset(str(merged), str(tmp_path / "out"),
                                   level=GeometryLevel.OBB, params=params, seed=1)
    assert out.stats["slice_geometry"]["reference_body_px"] == 123.0
    # measured is still reported (independent of the explicit override).
    assert out.stats["measured_reference_body_px"] > 0.0
```

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH="$(pwd)/src" conda run -n hydra-mps python -m pytest tests/test_sliced_dataset_reference.py -v`
Expected: FAIL — `ImportError: cannot import name 'object_major_axes_px'`.

- [ ] **Step 3: Refactor the measurement + accumulate a dataset median**

In `src/hydra_suite/training/sliced_dataset.py`, replace `measure_reference_body_px` with a pair (extract the per-object list, median on top):

```python
def object_major_axes_px(labels, frame_wh) -> list[float]:
    """All per-object minAreaRect major axes (px) over a frame's normalized labels."""
    w, h = float(frame_wh[0]), float(frame_wh[1])
    majors: list[float] = []
    for _cls_id, pts_norm in labels:
        pts = np.asarray(pts_norm, dtype=np.float32).copy()
        pts[:, 0] *= w
        pts[:, 1] *= h
        if pts.shape[0] < 3:
            continue
        _c, (bw, bh), _a = cv2.minAreaRect(pts.astype(np.float32))
        majors.append(float(max(bw, bh)))
    return majors


def measure_reference_body_px(labels, frame_wh) -> float:
    """Median minAreaRect major axis (px) over a frame's normalized-point labels."""
    majors = object_major_axes_px(labels, frame_wh)
    if not majors:
        return 0.0
    return float(np.median(np.asarray(majors, dtype=np.float64)))
```

- [ ] **Step 4: Accumulate across the dataset walk + record in the manifest**

In `build_sliced_obb_dataset`, initialize an accumulator before the item loop:

```python
    rng = random.Random(int(seed))
    counts = {"train": 0, "val": 0, "test": 0, "tiles": 0, "negatives": 0, "objects": 0}
    class_names = _read_class_names(merged_dir)
    all_majors: list[float] = []
```

Inside the `for split, img_path, lbl_path in _iter_dataset_items(merged_dir):` loop, right after `labels = _parse_geometry_label_lines(lbl_path)`, add:

```python
        all_majors.extend(object_major_axes_px(labels, (fw, fh)))
```

After the loop, before building the manifest, compute the median and thread it in:

```python
    measured_reference_body_px = (
        float(np.median(np.asarray(all_majors, dtype=np.float64))) if all_majors else 0.0
    )
    _write_sliced_yaml(out_dir, class_names)
    manifest = {
        "type": "sliced_obb",
        "source": str(merged_dir),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "level": level.label,
        "counts": counts,
        "measured_reference_body_px": measured_reference_body_px,
        "slice_geometry": _slice_geometry_manifest(params, measured_reference_body_px),
    }
```

And update `_slice_geometry_manifest` to record the actual reference:

```python
def _slice_geometry_manifest(params, measured_reference_body_px: float = 0.0) -> dict:
    reference_body_px = (
        params.reference_body_px if params.reference_body_px > 0 else measured_reference_body_px
    )
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
        "reference_body_px": reference_body_px,
    }
```

- [ ] **Step 5: Run to verify it passes**

Run: `PYTHONPATH="$(pwd)/src" conda run -n hydra-mps python -m pytest tests/test_sliced_dataset_reference.py tests/test_sliced_dataset.py -v`
Expected: PASS (new tests + the existing builder suite unchanged).

- [ ] **Step 6: Commit**

```bash
git add src/hydra_suite/training/sliced_dataset.py tests/test_sliced_dataset_reference.py
git commit -m "feat(training): measure dataset-median reference_body_px into sliced manifest"
```

---

### Task 2: DetectKit dialog — populate `reference_body_px` from the build (Component A)

**Files:**
- Modify: `src/hydra_suite/detectkit/gui/models.py` (add a pure helper)
- Modify: `src/hydra_suite/detectkit/gui/dialogs/training_dialog.py` (`_build_role_datasets`)
- Test: `tests/test_detectkit_populate_reference.py`

**Interfaces:**
- Consumes: Task 1's `DatasetBuildResult.stats["measured_reference_body_px"]`; `SliceTrainingSettings`.
- Produces: `populate_measured_reference(settings: SliceTrainingSettings, measured: float) -> bool` (in `models.py`) — sets `settings.reference_body_px = measured` and returns True ONLY when `settings.reference_body_px == 0.0` and `measured > 0`; otherwise leaves it and returns False.

- [ ] **Step 1: Write the failing test**

Create `tests/test_detectkit_populate_reference.py`:

```python
from hydra_suite.detectkit.gui.models import SliceTrainingSettings, populate_measured_reference


def test_populate_sets_when_unset():
    s = SliceTrainingSettings(reference_body_px=0.0)
    changed = populate_measured_reference(s, 55.0)
    assert changed is True
    assert s.reference_body_px == 55.0


def test_populate_preserves_user_value():
    s = SliceTrainingSettings(reference_body_px=30.0)
    changed = populate_measured_reference(s, 55.0)
    assert changed is False
    assert s.reference_body_px == 30.0


def test_populate_ignores_zero_measured():
    s = SliceTrainingSettings(reference_body_px=0.0)
    changed = populate_measured_reference(s, 0.0)
    assert changed is False
    assert s.reference_body_px == 0.0
```

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH="$(pwd)/src" conda run -n hydra-mps python -m pytest tests/test_detectkit_populate_reference.py -v`
Expected: FAIL — `ImportError: cannot import name 'populate_measured_reference'`.

- [ ] **Step 3: Add the pure helper**

In `src/hydra_suite/detectkit/gui/models.py`, after the `SliceTrainingSettings` dataclass, add:

```python
def populate_measured_reference(settings: SliceTrainingSettings, measured: float) -> bool:
    """Set settings.reference_body_px from a measured value only when currently unset.

    Returns True iff it changed the value (settings.reference_body_px was 0.0 and
    measured > 0). A user-set value is never overwritten.
    """
    if settings.reference_body_px == 0.0 and float(measured) > 0.0:
        settings.reference_body_px = float(measured)
        return True
    return False
```

- [ ] **Step 4: Wire it into the dialog build path**

In `src/hydra_suite/detectkit/gui/dialogs/training_dialog.py`, inside `_build_role_datasets`, find the sliced-build block (added by the prior feature) that calls `orchestrator.build_sliced_obb_dataset(...)` and assigns `sliced`. Immediately after `role_source_dir = sliced.dataset_dir` (and its `self._append_log(...)`), add:

```python
                from hydra_suite.detectkit.gui.models import populate_measured_reference

                measured_ref = float(sliced.stats.get("measured_reference_body_px", 0.0))
                if populate_measured_reference(self._project.slice_settings, measured_ref):
                    from hydra_suite.detectkit.gui.project import save_project

                    save_project(self._project)
                    self._append_log(
                        f"Auto-set reference body size: {measured_ref:.1f}px (measured)"
                    )
```

(READ the actual sliced-build block first; the variable is `sliced` (a `DatasetBuildResult`) and `sliced.stats` is the manifest dict from Task 1. Match the real surrounding text.)

- [ ] **Step 5: Run tests + dialog import smoke check**

Run: `PYTHONPATH="$(pwd)/src" conda run -n hydra-mps python -m pytest tests/test_detectkit_populate_reference.py tests/test_detectkit_slice_settings.py -v`
Run: `PYTHONPATH="$(pwd)/src" conda run -n hydra-mps python -c "import hydra_suite.detectkit.gui.dialogs.training_dialog; print('dialog OK')"`
Expected: tests PASS; import prints `dialog OK`.

- [ ] **Step 6: Commit**

```bash
git add src/hydra_suite/detectkit/gui/models.py src/hydra_suite/detectkit/gui/dialogs/training_dialog.py tests/test_detectkit_populate_reference.py
git commit -m "feat(detectkit): auto-populate reference_body_px from measured build median"
```

---

### Task 3: DetectKit preview — median-target `object_tile_fraction` (Component B)

**Files:**
- Modify: `src/hydra_suite/detectkit/gui/prediction_preview.py` (new pure helper)
- Modify: `src/hydra_suite/detectkit/gui/main_window.py` (preview routing)
- Test: `tests/test_detectkit_preview_target.py`

**Interfaces:**
- Consumes: `SliceTrainingSettings.target_sizes`, `.object_tile_fraction`; `project.imgsz_obb_direct`.
- Produces: `preview_object_tile_fraction(target_sizes: list[float], object_tile_fraction: float, imgsz: int) -> float` — `median(target_sizes)/imgsz` clamped to `[0.01, 0.9]` when `target_sizes` is non-empty and `imgsz > 0`; else the passed `object_tile_fraction`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_detectkit_preview_target.py`:

```python
from hydra_suite.detectkit.gui.prediction_preview import preview_object_tile_fraction


def test_median_target_over_imgsz():
    # median([200,300,400]) = 300 -> 300/640 = 0.46875
    assert abs(preview_object_tile_fraction([200.0, 300.0, 400.0], 0.15, 640) - 0.46875) < 1e-6


def test_even_count_median():
    # median([200,400]) = 300 -> 0.46875
    assert abs(preview_object_tile_fraction([200.0, 400.0], 0.15, 640) - 0.46875) < 1e-6


def test_empty_target_sizes_falls_back():
    assert preview_object_tile_fraction([], 0.2, 640) == 0.2


def test_zero_imgsz_falls_back():
    assert preview_object_tile_fraction([300.0], 0.2, 0) == 0.2


def test_result_is_clamped():
    # 4000/640 = 6.25 -> clamped to 0.9
    assert preview_object_tile_fraction([4000.0], 0.15, 640) == 0.9
```

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH="$(pwd)/src" conda run -n hydra-mps python -m pytest tests/test_detectkit_preview_target.py -v`
Expected: FAIL — `ImportError: cannot import name 'preview_object_tile_fraction'`.

- [ ] **Step 3: Add the pure helper**

In `src/hydra_suite/detectkit/gui/prediction_preview.py` (Qt-free), add near the top-level helpers:

```python
def preview_object_tile_fraction(target_sizes, object_tile_fraction, imgsz) -> float:
    """object_tile_fraction for auto_object preview: median(target_sizes)/imgsz.

    Uses the trained apparent scale (median of the training target sizes) so the
    preview tiles at a scale the model was actually trained at. Falls back to the
    stored object_tile_fraction when target_sizes is empty or imgsz <= 0. Clamped
    to the same [0.01, 0.9] range tile_size_for_mode uses.
    """
    sizes = [float(t) for t in (target_sizes or [])]
    if not sizes or int(imgsz) <= 0:
        return float(object_tile_fraction)
    import numpy as np

    frac = float(np.median(np.asarray(sizes, dtype=np.float64))) / float(imgsz)
    return max(0.01, min(0.9, frac))
```

- [ ] **Step 4: Use it in the preview routing**

In `src/hydra_suite/detectkit/gui/main_window.py`, the `predict_sliced_obb_result(...)` call currently passes `object_tile_fraction=slice_settings.object_tile_fraction`. Add the import at the existing import site (where `predict_sliced_obb_result` is imported):

```python
    predict_sliced_obb_result,
    preview_object_tile_fraction,
```

and change that one keyword argument to:

```python
                            object_tile_fraction=preview_object_tile_fraction(
                                slice_settings.target_sizes,
                                slice_settings.object_tile_fraction,
                                self._imgsz_obb_direct,
                            ),
```

Leave every other argument and the disabled-slicing branch unchanged.

- [ ] **Step 5: Run tests + import smoke check**

Run: `PYTHONPATH="$(pwd)/src" conda run -n hydra-mps python -m pytest tests/test_detectkit_preview_target.py tests/test_detectkit_sliced_preview.py tests/test_detectkit_prediction_preview.py -v`
Run: `PYTHONPATH="$(pwd)/src" conda run -n hydra-mps python -c "import hydra_suite.detectkit.gui.main_window; print('mw OK')"`
Expected: tests PASS; import prints `mw OK`.

- [ ] **Step 6: Commit**

```bash
git add src/hydra_suite/detectkit/gui/prediction_preview.py src/hydra_suite/detectkit/gui/main_window.py tests/test_detectkit_preview_target.py
git commit -m "feat(detectkit): preview auto_object tiles at median trained target scale"
```

---

### Task 4: core — `slice_meta.py` read + map (Component C, pure)

**Files:**
- Create: `src/hydra_suite/core/inference/slice_meta.py`
- Test: `tests/test_slice_meta_read.py`

**Interfaces:**
- Consumes: nothing (stdlib `json`/`pathlib` + numpy for the median).
- Produces:
  - `read_slice_meta(model_path: str | Path) -> dict | None` — reads `<model_path>.slice_meta.json`; returns the parsed dict, or `None` on absent file / bad JSON / non-dict. Never raises.
  - `slice_meta_to_panel_values(meta: dict) -> dict` — returns `{"enabled": True, "geometry_mode": str, "overlap": float, "object_tile_fraction": float, "trained_body_px": float}`. `object_tile_fraction = median(meta["target_sizes"]) / meta["imgsz"]` clamped to `[0.01, 0.9]`; falls back to `meta.get("object_tile_fraction", 0.15)` when `target_sizes` is empty/absent or `imgsz` is absent/0. Other keys fall back to sensible defaults (`geometry_mode="auto_object"`, `overlap=0.2`, `trained_body_px=meta.get("reference_body_px", 0.0)`).

- [ ] **Step 1: Write the failing test**

Create `tests/test_slice_meta_read.py`:

```python
import json
from pathlib import Path

from hydra_suite.core.inference.slice_meta import read_slice_meta, slice_meta_to_panel_values


def test_read_absent_returns_none(tmp_path):
    assert read_slice_meta(tmp_path / "model.pt") is None


def test_read_malformed_returns_none(tmp_path):
    model = tmp_path / "model.pt"
    model.write_bytes(b"x")
    (tmp_path / "model.pt.slice_meta.json").write_text("{not json", encoding="utf-8")
    assert read_slice_meta(model) is None


def test_read_present_returns_dict(tmp_path):
    model = tmp_path / "model.pt"
    model.write_bytes(b"x")
    (tmp_path / "model.pt.slice_meta.json").write_text(
        json.dumps({"geometry_mode": "auto_object", "reference_body_px": 560.0}), encoding="utf-8"
    )
    meta = read_slice_meta(model)
    assert meta["reference_body_px"] == 560.0


def test_map_full():
    meta = {"geometry_mode": "auto_object", "overlap": 0.2,
            "reference_body_px": 560.0, "target_sizes": [200.0, 300.0, 400.0], "imgsz": 640}
    v = slice_meta_to_panel_values(meta)
    assert v["enabled"] is True
    assert v["geometry_mode"] == "auto_object"
    assert v["overlap"] == 0.2
    assert v["trained_body_px"] == 560.0
    assert abs(v["object_tile_fraction"] - 300.0 / 640.0) < 1e-6


def test_map_empty_targets_falls_back_to_object_tile_fraction():
    meta = {"geometry_mode": "auto_object", "target_sizes": [], "imgsz": 640,
            "object_tile_fraction": 0.17, "reference_body_px": 100.0}
    assert slice_meta_to_panel_values(meta)["object_tile_fraction"] == 0.17


def test_map_missing_imgsz_falls_back():
    meta = {"target_sizes": [300.0], "object_tile_fraction": 0.18, "reference_body_px": 50.0}
    assert slice_meta_to_panel_values(meta)["object_tile_fraction"] == 0.18


def test_map_missing_keys_use_defaults():
    v = slice_meta_to_panel_values({})
    assert v["geometry_mode"] == "auto_object"
    assert v["overlap"] == 0.2
    assert v["trained_body_px"] == 0.0
    assert v["object_tile_fraction"] == 0.15
```

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH="$(pwd)/src" conda run -n hydra-mps python -m pytest tests/test_slice_meta_read.py -v`
Expected: FAIL — `ModuleNotFoundError: hydra_suite.core.inference.slice_meta`.

- [ ] **Step 3: Implement the module**

Create `src/hydra_suite/core/inference/slice_meta.py`:

```python
"""Read a model's .slice_meta.json sidecar and map it to SAHI panel/config values.

Pure (stdlib + numpy). The sidecar is written by training/model_publish.py for
OBB-direct models trained with sliced data; TrackerKit reads it to pre-fill the
SAHI panel. Mirrors the <artifact>.runtime_meta.json sidecar convention in
runtime_artifacts.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def read_slice_meta(model_path) -> "dict | None":
    """Return the parsed <model_path>.slice_meta.json dict, or None on absent/bad."""
    try:
        sidecar = Path(model_path).with_suffix(Path(model_path).suffix + ".slice_meta.json")
        if not sidecar.exists():
            return None
        data = json.loads(sidecar.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def slice_meta_to_panel_values(meta: dict) -> dict:
    """Translate a slice_meta dict into SAHI panel/config values.

    object_tile_fraction = median(target_sizes)/imgsz (clamped [0.01, 0.9]);
    falls back to meta['object_tile_fraction'] (default 0.15) when target_sizes is
    empty/absent or imgsz is absent/0. reference stays model-internal: trained_body_px.
    """
    stored_fraction = float(meta.get("object_tile_fraction", 0.15) or 0.15)
    target_sizes = [float(t) for t in (meta.get("target_sizes") or [])]
    imgsz = int(meta.get("imgsz", 0) or 0)
    if target_sizes and imgsz > 0:
        frac = float(np.median(np.asarray(target_sizes, dtype=np.float64))) / float(imgsz)
        object_tile_fraction = max(0.01, min(0.9, frac))
    else:
        object_tile_fraction = stored_fraction
    return {
        "enabled": True,
        "geometry_mode": str(meta.get("geometry_mode", "auto_object") or "auto_object"),
        "overlap": float(meta.get("overlap", 0.2) or 0.2),
        "object_tile_fraction": object_tile_fraction,
        "trained_body_px": float(meta.get("reference_body_px", 0.0) or 0.0),
    }
```

- [ ] **Step 4: Run to verify it passes**

Run: `PYTHONPATH="$(pwd)/src" conda run -n hydra-mps python -m pytest tests/test_slice_meta_read.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/core/inference/slice_meta.py tests/test_slice_meta_read.py
git commit -m "feat(inference): pure slice_meta sidecar read + panel-value mapper"
```

---

### Task 5: core config — `SLICE_TRAINED_BODY_PX` overrides `reference_body_px`

**Files:**
- Modify: `src/hydra_suite/core/inference/config.py` (`build_inference_config_from_params`)
- Test: `tests/test_slice_trained_body_px.py`

**Interfaces:**
- Consumes: the existing `build_inference_config_from_params(params) -> InferenceConfig` and its `SliceConfig.reference_body_px` computation.
- Produces: when `params["SLICE_TRAINED_BODY_PX"] > 0`, `SliceConfig.reference_body_px == that value`; otherwise it stays `REFERENCE_BODY_SIZE × RESIZE_FACTOR` (unchanged).

- [ ] **Step 1: Write the failing test**

Create `tests/test_slice_trained_body_px.py`:

```python
from hydra_suite.core.inference.config import build_inference_config_from_params


def _base(**extra):
    p = {
        "RUNTIME_TIER": "cpu",
        "YOLO_OBB_MODE": "direct",
        "YOLO_OBB_DIRECT_MODEL_PATH": "m.pt",
        "REFERENCE_BODY_SIZE": 20.0,
        "RESIZE_FACTOR": 2.0,  # product -> 40.0
        "SLICE_ENABLED": True,
        "SLICE_GEOMETRY_MODE": "auto_object",
    }
    p.update(extra)
    return p


def test_trained_body_px_overrides_reference():
    cfg = build_inference_config_from_params(_base(SLICE_TRAINED_BODY_PX=560.0))
    assert cfg.obb.direct.slice.reference_body_px == 560.0


def test_absent_trained_body_px_uses_product():
    cfg = build_inference_config_from_params(_base())
    assert cfg.obb.direct.slice.reference_body_px == 40.0  # 20 * 2


def test_zero_trained_body_px_uses_product():
    cfg = build_inference_config_from_params(_base(SLICE_TRAINED_BODY_PX=0.0))
    assert cfg.obb.direct.slice.reference_body_px == 40.0
```

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH="$(pwd)/src" conda run -n hydra-mps python -m pytest tests/test_slice_trained_body_px.py -v`
Expected: FAIL — `test_trained_body_px_overrides_reference` asserts 560.0 but gets 40.0.

- [ ] **Step 3: Implement the override**

In `src/hydra_suite/core/inference/config.py`, in `build_inference_config_from_params`, the `SliceConfig(...)` construction currently sets:

```python
            reference_body_px=_clamped_float(
                float(params.get("REFERENCE_BODY_SIZE", 20.0) or 20.0)
                * float(params.get("RESIZE_FACTOR", 1.0) or 1.0),
                0.0,
                0.0,
                8192.0,
            ),
```

Replace that argument with a trained-override-aware computation. Just before the `slice_cfg = SliceConfig(` line, add:

```python
        _trained_body_px = _clamped_float(
            params.get("SLICE_TRAINED_BODY_PX", 0.0), 0.0, 0.0, 8192.0
        )
        _reference_body_px = (
            _trained_body_px
            if _trained_body_px > 0
            else _clamped_float(
                float(params.get("REFERENCE_BODY_SIZE", 20.0) or 20.0)
                * float(params.get("RESIZE_FACTOR", 1.0) or 1.0),
                0.0,
                0.0,
                8192.0,
            )
        )
```

and set the field to `reference_body_px=_reference_body_px,`.

- [ ] **Step 4: Run to verify it passes**

Run: `PYTHONPATH="$(pwd)/src" conda run -n hydra-mps python -m pytest tests/test_slice_trained_body_px.py tests/test_inference_slicing.py -v`
Expected: PASS (new tests + the existing slicing suite unchanged).

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/core/inference/config.py tests/test_slice_trained_body_px.py
git commit -m "feat(inference): SLICE_TRAINED_BODY_PX overrides SliceConfig.reference_body_px"
```

---

### Task 6: TrackerKit — pre-fill SAHI panel from the sidecar on model selection (Component C, Qt)

**Files:**
- Modify: `src/hydra_suite/trackerkit/gui/orchestrators/config.py` (emit `SLICE_TRAINED_BODY_PX`)
- Modify: `src/hydra_suite/trackerkit/gui/panels/detection_panel.py` (pre-fill on model selection + banner)
- Test: `tests/test_trackerkit_slice_meta_prefill.py`

**Interfaces:**
- Consumes: `core.inference.slice_meta.{read_slice_meta, slice_meta_to_panel_values}` (Task 4); the `SLICE_TRAINED_BODY_PX` override (Task 5).
- Produces: on OBB-model selection, if `read_slice_meta(model_path)` returns a dict, apply `slice_meta_to_panel_values` — set `chk_slice_enabled`=True, `combo_slice_geometry`=geometry_mode, and write `slice_overlap` / `slice_object_tile_fraction` / `slice_trained_body_px` into `advanced_config`; raise a dismissible "Matched trained SAHI geometry" banner; NEVER touch `spin_reference_body_size` (`REFERENCE_BODY_SIZE`). Config emit adds `"SLICE_TRAINED_BODY_PX": advanced_config.get("slice_trained_body_px", 0.0)`.

THIS TASK IS DISCOVERY-HEAVY (Qt). The brief pins the seam; the implementer must READ the actual files to find: (a) how `combo_yolo_model` selection resolves to a model FILE PATH (the existing `on_yolo_model_changed` / `_auto_apply_yolo_training_params("obb_direct")` path already resolves the selected OBB model — reuse that path resolution); (b) how `advanced_config` is stored/mutated (it is `self._main_window.advanced_config`, a dict); (c) the existing banner/InfoBar mechanism in TrackerKit (search for existing user-facing status/notification calls in the panel or main window — reuse it; if none exists, a `QMessageBox.information` or a status-label update is acceptable). Follow `tests/test_detection_panel_slice_widgets.py` as the closest headless-panel test precedent.

- [ ] **Step 1: Emit the new param from the config orchestrator**

In `src/hydra_suite/trackerkit/gui/orchestrators/config.py`, in the params-dict block that already emits `"SLICE_OBJECT_TILE_FRACTION": advanced_config.get("slice_object_tile_fraction", 0.15),`, add immediately after it:

```python
            "SLICE_TRAINED_BODY_PX": advanced_config.get("slice_trained_body_px", 0.0),
```

- [ ] **Step 2: Write the failing pre-fill test**

Create `tests/test_trackerkit_slice_meta_prefill.py`. READ `tests/test_detection_panel_slice_widgets.py` first to copy its exact panel-construction fixture (main-window stub, `advanced_config` dict, QApplication). Then:

```python
import json

import pytest

pytest.importorskip("PySide6")


def _make_panel_with_sidecar(tmp_path):
    """Construct the detection panel + a model file with a .slice_meta.json sidecar.

    Mirror tests/test_detection_panel_slice_widgets.py's panel fixture: a stub
    main_window exposing `.advanced_config` (dict) and whatever attributes the
    panel reads, and a QApplication. Return (panel, main_window_stub, model_path).
    Wire the stub so the OBB-model combo resolves to `model_path`.
    """
    ...  # copy the fixture pattern from the sibling test


def test_selecting_model_with_sidecar_prefills(tmp_path):
    panel, mw, model_path = _make_panel_with_sidecar(tmp_path)
    (model_path.parent / (model_path.name + ".slice_meta.json")).write_text(
        json.dumps({"geometry_mode": "auto_object", "overlap": 0.25,
                    "reference_body_px": 560.0, "target_sizes": [200.0, 300.0, 400.0],
                    "imgsz": 640}),
        encoding="utf-8",
    )
    ref_before = panel.spin_reference_body_size.value()

    panel.apply_slice_meta_for_model(str(model_path))  # the new method under test

    assert panel.chk_slice_enabled.isChecked() is True
    assert panel.combo_slice_geometry.currentText() == "auto_object"
    assert mw.advanced_config["slice_overlap"] == 0.25
    assert abs(mw.advanced_config["slice_object_tile_fraction"] - 300.0 / 640.0) < 1e-6
    assert mw.advanced_config["slice_trained_body_px"] == 560.0
    # REFERENCE_BODY_SIZE must be left untouched.
    assert panel.spin_reference_body_size.value() == ref_before


def test_selecting_model_without_sidecar_is_noop(tmp_path):
    panel, mw, model_path = _make_panel_with_sidecar(tmp_path)
    enabled_before = panel.chk_slice_enabled.isChecked()
    adv_before = dict(mw.advanced_config)

    panel.apply_slice_meta_for_model(str(model_path))  # no sidecar written

    assert panel.chk_slice_enabled.isChecked() == enabled_before
    assert mw.advanced_config == adv_before
```

- [ ] **Step 3: Run to verify it fails**

Run: `PYTHONPATH="$(pwd)/src" conda run -n hydra-mps python -m pytest tests/test_trackerkit_slice_meta_prefill.py -v`
Expected: FAIL — `AttributeError: ... has no attribute 'apply_slice_meta_for_model'` (or the fixture import error until Step 4 lands the method).

- [ ] **Step 4: Implement the pre-fill method + wire it to model selection**

In `src/hydra_suite/trackerkit/gui/panels/detection_panel.py`, add a method (place it near the other `_on_yolo_*`/model handlers):

```python
    def apply_slice_meta_for_model(self, model_path: str) -> None:
        """Pre-fill SAHI settings from a model's .slice_meta.json sidecar, if present.

        Scale-independent trained knobs + the model-internal slice_trained_body_px;
        REFERENCE_BODY_SIZE (spin_reference_body_size) is deliberately left untouched
        (it is the full-frame tracking body size, a different quantity from the
        training-image body scale). No-op when no sidecar exists.
        """
        from hydra_suite.core.inference.slice_meta import (
            read_slice_meta, slice_meta_to_panel_values,
        )

        meta = read_slice_meta(model_path)
        if meta is None:
            return
        values = slice_meta_to_panel_values(meta)
        self.chk_slice_enabled.setChecked(bool(values["enabled"]))
        idx = self.combo_slice_geometry.findText(values["geometry_mode"])
        if idx >= 0:
            self.combo_slice_geometry.setCurrentIndex(idx)
        adv = self._main_window.advanced_config
        adv["slice_overlap"] = float(values["overlap"])
        adv["slice_object_tile_fraction"] = float(values["object_tile_fraction"])
        adv["slice_trained_body_px"] = float(values["trained_body_px"])
        self._notify_matched_geometry()

    def _notify_matched_geometry(self) -> None:
        """Show a dismissible 'Matched trained SAHI geometry' banner.

        Reuse TrackerKit's existing status/notification mechanism located during
        implementation; fall back to a status-label update if there is no banner API.
        """
        ...  # implement using the discovered banner/status mechanism
```

Then call it from the OBB-model selection path. Find where `combo_yolo_model` selection resolves the chosen model to a file path (the existing `on_yolo_model_changed` handler / `_auto_apply_yolo_training_params("obb_direct")`). At the point the selected OBB model's absolute path is known, call `self.apply_slice_meta_for_model(<resolved_path>)`. If the path is resolved in `main_window` rather than the panel, call `self._panels.detection.apply_slice_meta_for_model(path)` from there instead — wire it wherever the resolved OBB model path first becomes available on selection.

- [ ] **Step 5: Run tests + import smoke check**

Run: `PYTHONPATH="$(pwd)/src" conda run -n hydra-mps python -m pytest tests/test_trackerkit_slice_meta_prefill.py tests/test_detection_panel_slice_widgets.py -v`
Run: `PYTHONPATH="$(pwd)/src" conda run -n hydra-mps python -c "import hydra_suite.trackerkit.gui.panels.detection_panel, hydra_suite.trackerkit.gui.orchestrators.config; print('tk OK')"`
Expected: tests PASS (or skip if PySide6 absent); import prints `tk OK`.

- [ ] **Step 6: Commit**

```bash
git add src/hydra_suite/trackerkit/gui/panels/detection_panel.py src/hydra_suite/trackerkit/gui/orchestrators/config.py tests/test_trackerkit_slice_meta_prefill.py
git commit -m "feat(trackerkit): pre-fill SAHI panel from model slice_meta sidecar on selection"
```

---

## Notes for the executor

- **Worktree:** create an isolated worktree under `.worktrees/` (branch NOT on main) via `superpowers:using-git-worktrees`; run every test with `PYTHONPATH=<wt>/src conda run -n hydra-mps python -m pytest`.
- **Delta gate:** a task's own tests + the specific guard suites it names (`test_sliced_dataset.py`, `test_detectkit_slice_settings.py`, `test_detectkit_prediction_preview.py`, `test_detectkit_sliced_preview.py`, `test_inference_slicing.py`, `test_detection_panel_slice_widgets.py`). Not the whole suite.
- **`REFERENCE_BODY_SIZE` guard is a review gate:** any read/write of `REFERENCE_BODY_SIZE` / `spin_reference_body_size` introduced by this work is a defect (Global Constraints). The trained scale lives only in `SLICE_TRAINED_BODY_PX` / `slice_trained_body_px`.
- **GUI tasks (2, 3, 6):** the exact widget/handler wiring must be discovered by reading the files at implementation time; the plan pins the seams (`_build_role_datasets` sliced block; the `predict_sliced_obb_result` call; `combo_yolo_model` selection → model-path resolution) and the contracts, not every Qt line.
