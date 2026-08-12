# ViTPose Dataset Auto-Sizing Implementation Plan (Slice 2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let PoseKit measure a project's labelled pose data and propose a ViTPose input geometry that fits the animals, instead of leaving the operator to hand-edit `run.json`.

**Architecture:** A pure measurement module in `src/hydra_suite/training/` reads the PoseKit YOLO-pose label store (the COCO dataset does not exist yet at dialog time), converts normalized keypoints to pixels via PIL header reads, and reduces per-instance bounding boxes to a suggested `[H, W]`. The PoseKit training dialog gains a size control, a Detail multiplier, and an "Auto from dataset" button; the chosen value threads through `ViTPoseTrainingWorker` into `RunConfig.input_size`, which Slice 1 already built and validated.

**Tech Stack:** Python 3.13, NumPy, Pillow, PySide6, pytest. Conda env `hydra-mps`.

Spec: `docs/superpowers/specs/2026-08-04-vitpose-dataset-auto-sizing-design.md`

## Global Constraints

- **`src/hydra_suite/training/` must not import from any app layer** (PoseKit, ClassKit, DetectKit...). The repo's dependency rule is one-directional. This is why the new module parses label files itself rather than reusing PoseKit's `load_yolo_pose_label` — which, separately, parses only the **first line** of a label file and would silently under-count every multi-animal frame.
- Suggested dimensions are multiples of **32**, clamped to **[64, 384]**. The clamp constrains the *suggestion* only; the spin boxes accept up to 1024 so an operator can type a larger value deliberately.
- `heatmap_size_wh` is derived elsewhere; this slice only ever produces an image size as `[H, W]` (height first), matching `RunConfig.input_size` from Slice 1.
- **Never fall back to a default on bad input.** A silently defaulted suggestion looks like a measurement and is not one. Raise `ValueError`.
- Do not change the ViTPose default. A project that never presses the button must train exactly as it does today.
- Do not fix the multi-source label routing (see Task 2 notes). The estimator deliberately matches the dialog's existing flat `project.labels_dir` behaviour.
- Environment for every command:
  ```
  source ~/miniforge3/etc/profile.d/conda.sh && conda activate hydra-mps
  export KMP_DUPLICATE_LIB_OK=TRUE
  export PYTHONPATH=<worktree>/src
  ```
  Run pytest from the worktree root. `tests/test_identity_postprocess.py` has a pre-existing collection error unrelated to this branch — always `--ignore` it. `make format` before each commit; its `isort` step may incidentally touch the unrelated pre-existing `src/hydra_suite/refinekit/gui/dialogs/merge_wizard.py` — revert it, never commit it.
- Baseline before this plan: `python -m pytest tests/ -k "vitpose or posekit" -q --ignore=tests/test_identity_postprocess.py` — record the number in Task 1 Step 1 and do not regress it.

## File Structure

**Created:**
- `src/hydra_suite/training/pose_geometry_measure.py` — measurement and suggestion. No Qt, no app-layer imports.
- `tests/test_pose_geometry_measure.py`
- `tests/test_vitpose_training_input_size_threading.py`

**Modified:**
- `src/hydra_suite/posekit/gui/dialogs/training.py` — the size control, the Auto button, settings persistence, worker signature, params dict, and the call site.

---

### Task 1: The estimator

**Files:**
- Create: `src/hydra_suite/training/pose_geometry_measure.py`
- Test: `tests/test_pose_geometry_measure.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `PoseSizeStats` (frozen dataclass; fields `sample_count: int`, `frames_scanned: int`, `frames_skipped: int`, `median_aspect: float`, `median_long_px: float`, `p90_long_px: float`, `suggested_hw: list[int]`, `clamped: bool`); `measure_pose_geometry(image_paths, labels_dir, num_keypoints, *, detail=1.0, max_images=500, seed=0) -> PoseSizeStats`; constants `SIZE_MULTIPLE = 32`, `MIN_SIZE = 64`, `MAX_SUGGESTED_SIZE = 384`.

- [ ] **Step 1: Record the baseline**

```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate hydra-mps
export KMP_DUPLICATE_LIB_OK=TRUE
export PYTHONPATH=$PWD/src
python -m pytest tests/ -k "vitpose or posekit" -q --ignore=tests/test_identity_postprocess.py 2>&1 | tail -3
```

Write the passed/skipped counts into your report. Every later run must match or exceed it.

- [ ] **Step 2: Write the failing tests**

Create `tests/test_pose_geometry_measure.py`:

```python
"""Measuring a PoseKit label set to suggest a ViTPose input geometry."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from hydra_suite.training.pose_geometry_measure import (
    MAX_SUGGESTED_SIZE,
    PoseSizeStats,
    measure_pose_geometry,
)

K = 4  # keypoints per instance in these fixtures


def _write_frame(tmp_path, name, img_wh, instances, k=K):
    """One image + one YOLO-pose label file.

    instances: list of (x0, y0, x1, y1, visible) in PIXELS. Each becomes one
    label line whose k keypoints are placed at the box corners, so the visible
    extent is exactly the box.
    """
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir(exist_ok=True)
    labels.mkdir(exist_ok=True)
    w_px, h_px = img_wh
    img_path = images / f"{name}.png"
    Image.new("RGB", (w_px, h_px), (0, 0, 0)).save(img_path)

    lines = []
    for x0, y0, x1, y1, visible in instances:
        corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)][:k]
        parts = ["0", "0.5", "0.5", "0.5", "0.5"]
        for cx, cy in corners:
            parts += [f"{cx / w_px:.6f}", f"{cy / h_px:.6f}", "2" if visible else "0"]
        lines.append(" ".join(parts))
    (labels / f"{name}.txt").write_text("\n".join(lines) + "\n")
    return img_path, labels


def test_square_animals_give_a_square_suggestion(tmp_path):
    paths = []
    for i in range(5):
        p, labels = _write_frame(
            tmp_path, f"f{i}", (400, 400), [(100, 100, 228, 228, True)]
        )
        paths.append(p)
    stats = measure_pose_geometry(paths, labels, K)
    assert isinstance(stats, PoseSizeStats)
    assert stats.sample_count == 5
    assert stats.median_aspect == pytest.approx(1.0)
    assert stats.median_long_px == pytest.approx(128.0)
    assert stats.suggested_hw == [128, 128]
    assert stats.clamped is False


def test_wide_animals_give_width_greater_than_height(tmp_path):
    paths = []
    for i in range(5):
        p, labels = _write_frame(
            tmp_path, f"f{i}", (400, 400), [(50, 100, 178, 164, True)]
        )
        paths.append(p)
    stats = measure_pose_geometry(paths, labels, K)
    # box is 128 wide, 64 tall -> aspect 2.0
    assert stats.median_aspect == pytest.approx(2.0)
    h, w = stats.suggested_hw
    assert w > h
    assert stats.suggested_hw == [64, 128]


def test_tall_animals_give_height_greater_than_width(tmp_path):
    paths = []
    for i in range(5):
        p, labels = _write_frame(
            tmp_path, f"f{i}", (400, 400), [(100, 50, 164, 178, True)]
        )
        paths.append(p)
    stats = measure_pose_geometry(paths, labels, K)
    # box is 64 wide, 128 tall -> aspect 0.5
    assert stats.median_aspect == pytest.approx(0.5)
    h, w = stats.suggested_hw
    assert h > w
    assert stats.suggested_hw == [128, 64]


def test_every_instance_on_a_multi_animal_frame_is_counted(tmp_path):
    # PoseKit's own reader parses only the first line; this module must not.
    p, labels = _write_frame(
        tmp_path,
        "multi",
        (400, 400),
        [
            (0, 0, 128, 128, True),
            (200, 0, 328, 128, True),
            (0, 200, 128, 328, True),
        ],
    )
    stats = measure_pose_geometry([p], labels, K)
    assert stats.sample_count == 3
    assert stats.frames_scanned == 1


def test_invisible_keypoints_are_excluded_from_the_extent(tmp_path):
    # Two visible corners 128 apart, two invisible ones far away. If the
    # invisible pair were counted the extent would be much larger.
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    Image.new("RGB", (400, 400), (0, 0, 0)).save(images / "a.png")
    parts = ["0", "0.5", "0.5", "0.5", "0.5"]
    for cx, cy, v in [(100, 100, 2), (228, 228, 2), (0, 0, 0), (399, 399, 0)]:
        parts += [f"{cx / 400:.6f}", f"{cy / 400:.6f}", str(v)]
    (labels / "a.txt").write_text(" ".join(parts) + "\n")
    stats = measure_pose_geometry([images / "a.png"], labels, K)
    assert stats.median_long_px == pytest.approx(128.0)


def test_detail_multiplier_scales_the_suggestion(tmp_path):
    paths = []
    for i in range(3):
        p, labels = _write_frame(
            tmp_path, f"f{i}", (400, 400), [(100, 100, 228, 228, True)]
        )
        paths.append(p)
    assert measure_pose_geometry(paths, labels, K, detail=2.0).suggested_hw == [256, 256]
    assert measure_pose_geometry(paths, labels, K, detail=0.5).suggested_hw == [64, 64]


def test_suggestions_are_always_multiples_of_thirty_two(tmp_path):
    paths = []
    for i in range(3):
        p, labels = _write_frame(
            tmp_path, f"f{i}", (500, 500), [(10, 10, 157, 123, True)]
        )
        paths.append(p)
    stats = measure_pose_geometry(paths, labels, K)
    assert all(v % 32 == 0 for v in stats.suggested_hw)


def test_very_large_animals_are_clamped_and_flagged(tmp_path):
    paths = []
    for i in range(3):
        p, labels = _write_frame(
            tmp_path, f"f{i}", (2000, 2000), [(100, 100, 1700, 1700, True)]
        )
        paths.append(p)
    stats = measure_pose_geometry(paths, labels, K)
    assert stats.clamped is True
    assert max(stats.suggested_hw) == MAX_SUGGESTED_SIZE


def test_p90_is_reported_and_at_least_the_median(tmp_path):
    paths = []
    for i in range(9):
        extent = 64 if i < 8 else 320  # one large outlier
        p, labels = _write_frame(
            tmp_path, f"f{i}", (600, 600), [(10, 10, 10 + extent, 10 + extent, True)]
        )
        paths.append(p)
    stats = measure_pose_geometry(paths, labels, K)
    assert stats.median_long_px == pytest.approx(64.0)
    assert stats.p90_long_px > stats.median_long_px


def test_measurement_is_deterministic(tmp_path):
    paths = []
    for i in range(30):
        p, labels = _write_frame(
            tmp_path, f"f{i}", (400, 400), [(i, i, i + 100, i + 120, True)]
        )
        paths.append(p)
    a = measure_pose_geometry(paths, labels, K, max_images=10)
    b = measure_pose_geometry(paths, labels, K, max_images=10)
    assert a == b


def test_subsampling_caps_the_frames_scanned(tmp_path):
    paths = []
    for i in range(25):
        p, labels = _write_frame(
            tmp_path, f"f{i}", (400, 400), [(10, 10, 138, 138, True)]
        )
        paths.append(p)
    stats = measure_pose_geometry(paths, labels, K, max_images=7)
    assert stats.frames_scanned == 7


def test_unreadable_image_is_skipped_not_fatal(tmp_path):
    good, labels = _write_frame(
        tmp_path, "good", (400, 400), [(100, 100, 228, 228, True)]
    )
    bad = tmp_path / "images" / "bad.png"
    bad.write_bytes(b"not an image")
    (labels / "bad.txt").write_text((labels / "good.txt").read_text())
    stats = measure_pose_geometry([good, bad], labels, K)
    assert stats.frames_scanned == 2
    assert stats.frames_skipped == 1
    assert stats.sample_count == 1


def test_short_label_line_is_skipped(tmp_path):
    good, labels = _write_frame(
        tmp_path, "good", (400, 400), [(100, 100, 228, 228, True)]
    )
    with (labels / "good.txt").open("a", encoding="utf-8") as fh:
        fh.write("0 0.5 0.5 0.5 0.5 0.1 0.1 2\n")  # only 1 keypoint, needs 4
    stats = measure_pose_geometry([good], labels, K)
    assert stats.sample_count == 1


def test_no_labelled_frames_raises(tmp_path):
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    Image.new("RGB", (64, 64), (0, 0, 0)).save(images / "a.png")
    with pytest.raises(ValueError, match="no labelled frames"):
        measure_pose_geometry([images / "a.png"], labels, K)


def test_labels_present_but_none_usable_raises(tmp_path):
    # every keypoint invisible -> no usable instance
    p, labels = _write_frame(
        tmp_path, "a", (400, 400), [(100, 100, 228, 228, False)]
    )
    with pytest.raises(ValueError, match="usable"):
        measure_pose_geometry([p], labels, K)


def test_non_positive_detail_raises(tmp_path):
    p, labels = _write_frame(
        tmp_path, "a", (400, 400), [(100, 100, 228, 228, True)]
    )
    with pytest.raises(ValueError, match="detail"):
        measure_pose_geometry([p], labels, K, detail=0.0)


def test_module_does_not_import_any_app_layer():
    # Training must not depend on PoseKit; the whole reason this module
    # re-implements label parsing.
    import hydra_suite.training.pose_geometry_measure as mod

    src = Path(mod.__file__).read_text(encoding="utf-8")
    for banned in ("posekit", "classkit", "detectkit", "trackerkit", "refinekit"):
        assert banned not in src
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m pytest tests/test_pose_geometry_measure.py -v`
Expected: collection error — `ModuleNotFoundError: No module named 'hydra_suite.training.pose_geometry_measure'`.

- [ ] **Step 4: Implement the module**

Create `src/hydra_suite/training/pose_geometry_measure.py`:

```python
"""Measure a PoseKit label set and suggest a ViTPose input geometry.

Pure measurement: no Qt, and no imports from any app layer -- Training must not
depend on PoseKit. That rule is a benefit here rather than a cost: PoseKit's own
`load_yolo_pose_label` parses only the FIRST line of a label file, so it silently
under-counts multi-animal frames. This module parses every line.

Sizing targets the bare keypoint extent on purpose. Inference already pads by
PADDING_FACTOR = 1.25 inside `box2cs` before warping, so the model sees more than
the animal; the `detail` multiplier is for operator preference, not to compensate
for that padding.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
from PIL import Image

SIZE_MULTIPLE = 32
MIN_SIZE = 64
# ViT attention cost tracks token count: 192x256 is 192 tokens, 256x256 is 256,
# 384x384 is 576, 512x512 is 1024. Cap the SUGGESTION so the tool never quietly
# proposes a model that trains several times slower; typing a larger value by
# hand stays available.
MAX_SUGGESTED_SIZE = 384


@dataclass(frozen=True)
class PoseSizeStats:
    sample_count: int  # instances measured, not files read
    frames_scanned: int
    frames_skipped: int
    median_aspect: float  # width / height of the keypoint bounding box
    median_long_px: float
    p90_long_px: float
    suggested_hw: List[int]  # [H, W]
    clamped: bool


def _snap(value: float) -> Tuple[int, bool]:
    """Round to the nearest multiple of 32 and clamp; report whether capped."""
    snapped = int(round(value / SIZE_MULTIPLE)) * SIZE_MULTIPLE
    clamped = False
    if snapped > MAX_SUGGESTED_SIZE:
        snapped, clamped = MAX_SUGGESTED_SIZE, True
    if snapped < MIN_SIZE:
        snapped = MIN_SIZE
    return snapped, clamped


def _instance_extents(
    text: str, num_keypoints: int, w_px: int, h_px: int
) -> List[Tuple[float, float]]:
    """Per-instance (width, height) in pixels of the VISIBLE keypoints' box.

    Invisible keypoints (v == 0) are excluded: they carry no position
    information, and counting them at their stored coordinates would bias every
    box toward the origin.
    """
    out: List[Tuple[float, float]] = []
    need = 5 + 3 * num_keypoints
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < need:
            continue
        try:
            vals = [float(v) for v in parts[1:need]]
        except ValueError:
            continue
        xs: List[float] = []
        ys: List[float] = []
        for i in range(num_keypoints):
            x, y, v = vals[4 + 3 * i], vals[5 + 3 * i], vals[6 + 3 * i]
            if v > 0:
                xs.append(x * w_px)
                ys.append(y * h_px)
        if len(xs) < 2:
            continue
        bw = max(xs) - min(xs)
        bh = max(ys) - min(ys)
        if bw <= 0 or bh <= 0:
            continue
        out.append((bw, bh))
    return out


def measure_pose_geometry(
    image_paths: Sequence[Path],
    labels_dir: Path,
    num_keypoints: int,
    *,
    detail: float = 1.0,
    max_images: int = 500,
    seed: int = 0,
) -> PoseSizeStats:
    """Suggest a ViTPose input geometry from a PoseKit label store."""
    if detail <= 0:
        raise ValueError(f"detail must be positive; got {detail}")
    labels_dir = Path(labels_dir)

    labelled: List[Tuple[Path, Path]] = []
    for raw in image_paths:
        img_path = Path(raw)
        label_path = labels_dir / f"{img_path.stem}.txt"
        try:
            if label_path.exists() and label_path.stat().st_size > 0:
                labelled.append((img_path, label_path))
        except OSError:
            continue
    if not labelled:
        raise ValueError(f"no labelled frames found under {labels_dir}")

    if len(labelled) > max_images:
        labelled = random.Random(seed).sample(labelled, max_images)
        labelled.sort()  # order-independent of the sample draw

    widths: List[float] = []
    heights: List[float] = []
    scanned = 0
    skipped = 0
    for img_path, label_path in labelled:
        scanned += 1
        try:
            with Image.open(img_path) as im:
                w_px, h_px = im.size
            text = label_path.read_text(encoding="utf-8")
        except Exception:
            skipped += 1
            continue
        extents = _instance_extents(text, num_keypoints, int(w_px), int(h_px))
        if not extents:
            skipped += 1
            continue
        for bw, bh in extents:
            widths.append(bw)
            heights.append(bh)

    if not widths:
        raise ValueError(
            f"found {scanned} labelled frame(s) under {labels_dir} but none held a "
            "usable instance (an instance needs at least 2 visible keypoints)"
        )

    w_arr = np.asarray(widths, dtype=np.float64)
    h_arr = np.asarray(heights, dtype=np.float64)
    long_arr = np.maximum(w_arr, h_arr)
    median_aspect = float(np.median(w_arr / h_arr))
    median_long = float(np.median(long_arr))
    p90_long = float(np.percentile(long_arr, 90))

    # Reconstruct a coherent (W, H) from one length and one aspect. Taking
    # independent medians of width and height could describe an animal that
    # does not exist in the data.
    long_side = median_long * detail
    if median_aspect >= 1.0:
        raw_w, raw_h = long_side, long_side / median_aspect
    else:
        raw_h, raw_w = long_side, long_side * median_aspect

    snapped_w, clamped_w = _snap(raw_w)
    snapped_h, clamped_h = _snap(raw_h)
    return PoseSizeStats(
        sample_count=len(widths),
        frames_scanned=scanned,
        frames_skipped=skipped,
        median_aspect=median_aspect,
        median_long_px=median_long,
        p90_long_px=p90_long,
        suggested_hw=[snapped_h, snapped_w],
        clamped=clamped_w or clamped_h,
    )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_pose_geometry_measure.py -v`
Expected: all pass.

If `test_suggestions_are_always_multiples_of_thirty_two` or a specific expected `suggested_hw` fails, do NOT adjust the expected numbers to match the output — recompute the arithmetic by hand from the fixture's pixel box and find which of `_snap`, the aspect branch, or the normalization is wrong. Expected-value churn is how a geometry bug gets locked in.

- [ ] **Step 6: Confirm nothing else regressed**

Run: `python -m pytest tests/ -k "vitpose or posekit" -q --ignore=tests/test_identity_postprocess.py`
Expected: the Step 1 baseline, unchanged.

- [ ] **Step 7: Commit**

```bash
make format
git add src/hydra_suite/training/pose_geometry_measure.py tests/test_pose_geometry_measure.py
git commit -m "feat(training): measure labelled pose data to suggest a ViTPose input size"
```

---

### Task 2: The control and the threading

**Files:**
- Modify: `src/hydra_suite/posekit/gui/dialogs/training.py` — imports (~line 24), `ViTPoseTrainingWorker.__init__` (311-341), its `params` dict (364-372), the ViTPose group (688-702), `_apply_settings` (~1102), `_save_settings` (~1365), `_start_vitpose_training` (1600-1614)
- Test: `tests/test_vitpose_training_input_size_threading.py`

**Interfaces:**
- Consumes: `measure_pose_geometry(image_paths, labels_dir, num_keypoints, *, detail=1.0, max_images=500, seed=0) -> PoseSizeStats` and `PoseSizeStats` from Task 1; `RunConfig.input_size: list[int] | None` and `validate_run_config` from Slice 1.
- Produces: `ViTPoseTrainingWorker.__init__(..., device, input_size=None)` storing `self.input_size`; `"input_size"` in the worker's `params` dict; dialog attributes `vitpose_h_spin`, `vitpose_w_spin`, `vitpose_detail_spin`, `vitpose_auto_btn`, `vitpose_size_summary`; method `_auto_size_from_dataset()`.

**Note on multi-source projects, deliberately not fixed here:** the dialog reads labels from a flat `self.project.labels_dir` (`training.py:1420`, `:1562`, `:1602`) while the main window routes per-source. The estimator is passed that same flat `labels_dir`, so the estimate describes exactly the data the run will train on. Do not "improve" this.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_vitpose_training_input_size_threading.py`:

```python
"""input_size threads from the dialog's worker into RunConfig.

Qt-free by design: this constructs the worker object directly and inspects the
params dict, rather than building the dialog. The repo has known modal-dialog
hangs that stop the full suite completing, so GUI construction stays out of tests.
"""

from __future__ import annotations

import pytest

from hydra_suite.core.individual.pose.vitpose.training.config import validate_run_config

pytest.importorskip("PySide6")


def _worker(**over):
    from hydra_suite.posekit.gui.dialogs.training import ViTPoseTrainingWorker

    kwargs = dict(
        image_paths=[],
        labels_dir="labels",
        run_dir="run",
        cache_dir="cache",
        class_names=["a"],
        keypoint_names=["k0", "k1"],
        skeleton_edges=[],
        variant="B",
        init_checkpoint="ckpt.pth",
        num_keypoints=2,
        epochs=1,
        batch=1,
        device="cpu",
    )
    kwargs.update(over)
    return ViTPoseTrainingWorker(**kwargs)


def test_worker_accepts_and_stores_input_size():
    w = _worker(input_size=[256, 256])
    assert w.input_size == [256, 256]


def test_input_size_defaults_to_none_so_existing_callers_are_unaffected():
    assert _worker().input_size is None


def test_params_dict_shape_is_accepted_by_run_config():
    # Mirrors the dict ViTPoseTrainingWorker.run() builds, with input_size added.
    params = dict(
        init_checkpoint="ckpt.pth",
        variant="B",
        num_keypoints=2,
        dataset_dir="ds",
        output_dir="out",
        device="cpu",
        epochs=1,
        batch_size=1,
        input_size=[256, 256],
    )
    cfg = validate_run_config(params)
    assert cfg.input_size == [256, 256]


def test_run_config_rejects_a_bad_input_size_from_the_dialog():
    params = dict(
        init_checkpoint="ckpt.pth",
        variant="B",
        num_keypoints=2,
        dataset_dir="ds",
        output_dir="out",
        device="cpu",
        epochs=1,
        batch_size=1,
        input_size=[250, 192],  # not a multiple of 32
    )
    with pytest.raises(ValueError, match="input_size"):
        validate_run_config(params)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_vitpose_training_input_size_threading.py -v`
Expected: `test_worker_accepts_and_stores_input_size` and `test_input_size_defaults_to_none_so_existing_callers_are_unaffected` FAIL with `TypeError: __init__() got an unexpected keyword argument 'input_size'`. The two `validate_run_config` tests should already PASS — Slice 1 built that end.

- [ ] **Step 3: Widen the worker**

In `src/hydra_suite/posekit/gui/dialogs/training.py`, add `input_size=None` as the LAST parameter of `ViTPoseTrainingWorker.__init__` (after `device`), and store it beside the other assignments:

```python
        self.device = device
        self.input_size = list(input_size) if input_size else None
```

Then add it to the `params` dict inside `run()` (currently `training.py:364-372`), after `batch_size`:

```python
            params = dict(
                init_checkpoint=self.init_checkpoint,
                variant=self.variant,
                num_keypoints=self.num_keypoints,
                dataset_dir=str(ds["dataset_dir"]),
                device=self.device,
                epochs=self.epochs,
                batch_size=self.batch,
                input_size=self.input_size,
            )
```

`prepare_run` copies `params` wholesale into `validate_run_config`, and the key must be exactly `input_size` or validation rejects it.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_vitpose_training_input_size_threading.py -v`
Expected: 4 passed.

- [ ] **Step 5: Add the control to the ViTPose group**

First add `QApplication` to the existing `PySide6.QtWidgets` import block (~line 24) — the other widgets used below are already imported.

Then, in `TrainingRunnerDialog.__init__`, insert these rows into `vitpose_layout` after the `Init checkpoint` row and BEFORE `content_layout.addWidget(self.vitpose_group)`:

```python
        # Input geometry. Spin ranges deliberately exceed the auto-suggestion
        # cap of 384: the operator may type a larger value on purpose; the
        # estimator just will not propose one.
        self.vitpose_h_spin = QSpinBox()
        self.vitpose_h_spin.setRange(64, 1024)
        self.vitpose_h_spin.setSingleStep(32)
        self.vitpose_h_spin.setValue(256)
        self.vitpose_w_spin = QSpinBox()
        self.vitpose_w_spin.setRange(64, 1024)
        self.vitpose_w_spin.setSingleStep(32)
        self.vitpose_w_spin.setValue(192)
        size_row = QHBoxLayout()
        size_row.addWidget(QLabel("H"))
        size_row.addWidget(self.vitpose_h_spin)
        size_row.addWidget(QLabel("W"))
        size_row.addWidget(self.vitpose_w_spin)
        vitpose_layout.addRow("Input size", size_row)

        self.vitpose_detail_spin = QDoubleSpinBox()
        self.vitpose_detail_spin.setRange(0.25, 4.0)
        self.vitpose_detail_spin.setSingleStep(0.25)
        self.vitpose_detail_spin.setValue(1.0)
        self.vitpose_detail_spin.setSuffix("x")
        self.vitpose_auto_btn = QPushButton("Auto from dataset")
        self.vitpose_auto_btn.clicked.connect(self._auto_size_from_dataset)
        auto_row = QHBoxLayout()
        auto_row.addWidget(QLabel("Detail"))
        auto_row.addWidget(self.vitpose_detail_spin)
        auto_row.addWidget(self.vitpose_auto_btn)
        auto_row.addStretch(1)
        vitpose_layout.addRow("", auto_row)

        self.vitpose_size_summary = QLabel("")
        self.vitpose_size_summary.setWordWrap(True)
        vitpose_layout.addRow("", self.vitpose_size_summary)
```

The group's visibility is already toggled wholesale by `_update_backend_ui`, so these rows need no extra show/hide wiring.

- [ ] **Step 6: Implement the Auto button**

Add this method to `TrainingRunnerDialog` (place it directly above `_start_vitpose_training`):

```python
    def _auto_size_from_dataset(self):
        """Measure the labelled data and propose an input geometry.

        Synchronous under a wait cursor: this is a few hundred image-header
        reads (not decodes), so a worker thread would add failure modes without
        buying responsiveness.
        """
        from hydra_suite.training.pose_geometry_measure import measure_pose_geometry

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            stats = measure_pose_geometry(
                self.image_paths,
                self.project.labels_dir,
                len(self.project.keypoint_names),
                detail=float(self.vitpose_detail_spin.value()),
            )
        except ValueError as exc:
            self.vitpose_size_summary.setText(f"Could not measure: {exc}")
            return
        except Exception as exc:  # unexpected: surface it, do not guess a size
            QMessageBox.warning(self, "Auto-size failed", str(exc))
            return
        finally:
            QApplication.restoreOverrideCursor()

        h, w = stats.suggested_hw
        self.vitpose_h_spin.setValue(h)
        self.vitpose_w_spin.setValue(w)
        note = " (clamped to the 384 cap)" if stats.clamped else ""
        self.vitpose_size_summary.setText(
            f"{stats.sample_count} instance(s) over {stats.frames_scanned} frame(s); "
            f"median aspect {stats.median_aspect:.2f}, median long side "
            f"{stats.median_long_px:.0f}px, p90 {stats.p90_long_px:.0f}px "
            f"-> {h}x{w}{note}"
        )
        self._append_log(
            f"[ViTPose] Auto-sized to {h}x{w} from {stats.sample_count} instance(s)"
            f"{note}"
        )
```

Reporting p90 beside the median is the point of the summary: it tells the operator whether the median is representative or whether the dataset has a long tail of large individuals.

- [ ] **Step 7: Persist the values and pass them to the worker**

In `_save_settings`, add to the dict:

```python
                "vitpose_input_h": int(self.vitpose_h_spin.value()),
                "vitpose_input_w": int(self.vitpose_w_spin.value()),
                "vitpose_detail": float(self.vitpose_detail_spin.value()),
```

In `_apply_settings`, add alongside the other restores:

```python
        self.vitpose_h_spin.setValue(
            int(settings.get("vitpose_input_h", self.vitpose_h_spin.value()))
        )
        self.vitpose_w_spin.setValue(
            int(settings.get("vitpose_input_w", self.vitpose_w_spin.value()))
        )
        self.vitpose_detail_spin.setValue(
            float(settings.get("vitpose_detail", self.vitpose_detail_spin.value()))
        )
```

In `_start_vitpose_training`, add the argument to the `ViTPoseTrainingWorker(...)` construction, after `device=`:

```python
            device=self.device_combo.currentText(),
            input_size=[
                int(self.vitpose_h_spin.value()),
                int(self.vitpose_w_spin.value()),
            ],
        )
```

Always pass it. When the operator has not touched the control it carries the current default `[256, 192]`, which is byte-for-byte today's behaviour.

- [ ] **Step 8: Verify**

```bash
python -m pytest tests/test_vitpose_training_input_size_threading.py tests/test_pose_geometry_measure.py -v
python -m pytest tests/ -k "vitpose or posekit" -q --ignore=tests/test_identity_postprocess.py
```
Expected: the new files pass; the broad selection matches or exceeds Task 1 Step 1's baseline.

Then confirm the module imports cleanly in isolation (catches a missing `QApplication` import, which would otherwise only fail when the button is clicked):

```bash
python -c "import hydra_suite.posekit.gui.dialogs.training as t; print('import OK', hasattr(t.ViTPoseTrainingWorker, 'run'))"
```
Expected: `import OK True`.

- [ ] **Step 9: Commit**

```bash
make format
git add src/hydra_suite/posekit/gui/dialogs/training.py tests/test_vitpose_training_input_size_threading.py
git commit -m "feat(posekit): auto-size ViTPose input geometry from the labelled dataset"
```

---

## Out of Scope

- The `measured_input_size` dataset-manifest stamp, cut during design: the manifest is written after the size is already chosen, and the chosen size is already recorded in `run.json` and in every checkpoint's `input_size`.
- Multi-source label routing in the training dialog (documented under Task 2).
- Auto-measuring on dialog open. Measurement is I/O and happens on demand.
- Any change to the ViTPose default geometry.
