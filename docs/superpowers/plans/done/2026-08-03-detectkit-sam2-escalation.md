# DetectKit Phase B — SAM2 Mask Priming (escalate-all) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a DetectKit "escalate-all" bulk action that runs SAM2 over existing OBB/box labels to prime segmentation masks, writing new reviewable `<name>_seg` sources the user refines in X-AnyLabeling.

**Architecture:** A standalone torch-only SAM2 executor (prompt-in/mask-out, outside `InferenceRunner`) + a pure escalation orchestrator + a `BaseWorker` wrapper + a `BaseDialog`. Derived sources are new directories (originals byte-untouched), gated `reviewed=False` until the user confirms. Masks write through the existing `_write_geometry_label` sink; review reuses the existing polygon→`segment` X-AnyLabeling round-trip.

**Tech Stack:** Python, PySide6 (Qt), ultralytics-style YOLO labels, OpenCV (cv2), `sam2` package, `huggingface_hub`, numpy, pytest.

**Spec:** `docs/superpowers/specs/2026-08-03-detectkit-sam2-escalation-design.md`

## Global Constraints

- SAM2 is a **hard dependency** (`sam2` in `[project.dependencies]`) but **lazy-imported** — no `import sam2` at module top level in any file that loads without escalation.
- Originals are **never mutated**: escalation only creates new `<name>_seg` source directories.
- New `OBBSource.reviewed` defaults **`True`** so all existing/legacy sources are unaffected; only escalation sets `False`.
- Masks/labels always write through the existing `_write_geometry_label` (`detectkit/jobs/al_worker.py:116`) — do not write label files by hand.
- SAM2 runs **torch-only** on the resolved device (cuda → mps → cpu); no TensorRT/CoreML/ONNX export, not via `InferenceRunner`/`load_obb_executor`.
- Commit as the configured git user; **no `Co-Authored-By: Claude` trailer**.
- Follow `make format` (black + isort) before each commit; keep files focused (<~500 lines).

---

## File Structure

**Create:**
- `src/hydra_suite/core/inference/sam2/__init__.py` — package marker.
- `src/hydra_suite/core/inference/sam2/checkpoints.py` — variant catalog + `ensure_checkpoint` (HF managed weights).
- `src/hydra_suite/core/inference/sam2/masks.py` — `mask_to_contour`.
- `src/hydra_suite/core/inference/sam2/executor.py` — `Sam2SegmentExecutor` (lazy `sam2` import).
- `src/hydra_suite/detectkit/jobs/sam2_prompts.py` — `SourceBox`, `read_boxes_from_label`, `Prompt`, `build_prompts`.
- `src/hydra_suite/detectkit/jobs/sam2_escalation.py` — `EscalationRequest`/`EscalationResult`, pure `run_escalation`, `Sam2EscalationWorker`.
- `src/hydra_suite/detectkit/gui/dialogs/escalate_sam2_dialog.py` — `EscalateSam2Dialog(BaseDialog)`.
- Tests: `tests/test_sam2_prompts.py`, `tests/test_sam2_masks.py`, `tests/test_sam2_checkpoints.py`, `tests/test_sam2_executor.py`, `tests/test_sam2_escalation.py`, `tests/test_obbsource_reviewed.py`, `tests/test_training_gating_reviewed.py`.

**Modify:**
- `src/hydra_suite/detectkit/gui/models.py:26-61` — add `reviewed`/`derived_from`/`sam2_variant`.
- `src/hydra_suite/training/dataset_builders.py` — exclude `reviewed=False` sources from builds.
- `src/hydra_suite/detectkit/gui/dialogs/training_dialog.py:~1504` — secondary "Escalate to segment (SAM2)" entry.
- `src/hydra_suite/detectkit/gui/panels/tools_panel.py` (or dataset panel) — primary "Escalate to segment (SAM2)" button + "Mark reviewed" action.
- `pyproject.toml` — add `sam2` dependency; document a `sam2_segment` pipeline key.

---

### Task 1: `OBBSource` review/provenance fields

**Files:**
- Modify: `src/hydra_suite/detectkit/gui/models.py:26-61`
- Test: `tests/test_obbsource_reviewed.py`

**Interfaces:**
- Produces: `OBBSource.reviewed: bool = True`, `OBBSource.derived_from: str | None = None`, `OBBSource.sam2_variant: str | None = None`, carried through `to_dict`/`from_dict`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_obbsource_reviewed.py
from hydra_suite.detectkit.gui.models import OBBSource


def test_defaults_keep_legacy_sources_trusted():
    s = OBBSource(name="orig")
    assert s.reviewed is True and s.derived_from is None and s.sam2_variant is None


def test_roundtrip_preserves_new_fields():
    s = OBBSource(name="orig_seg", level="polygon", reviewed=False,
                  derived_from="orig", sam2_variant="sam2.1-hiera-base_plus")
    back = OBBSource.from_dict(s.to_dict())
    assert back.reviewed is False
    assert back.derived_from == "orig"
    assert back.sam2_variant == "sam2.1-hiera-base_plus"


def test_from_dict_missing_new_fields_defaults_reviewed_true():
    back = OBBSource.from_dict({"name": "legacy", "level": "obb"})
    assert back.reviewed is True and back.derived_from is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_obbsource_reviewed.py -v`
Expected: FAIL (`TypeError: unexpected keyword argument 'reviewed'`).

- [ ] **Step 3: Add the fields + serialization**

In `models.py`, add to the `OBBSource` dataclass (after `level`):

```python
    reviewed: bool = True  # False only for un-reviewed SAM2-primed derived sources
    derived_from: str | None = None  # origin source name for derived sources
    sam2_variant: str | None = None  # SAM2 version that primed a derived source
```

Add to `to_dict` (inside the returned dict):

```python
            "reviewed": self.reviewed,
            "derived_from": self.derived_from,
            "sam2_variant": self.sam2_variant,
```

Add to `from_dict` (inside the `OBBSource(...)` call):

```python
            reviewed=bool(d.get("reviewed", True)),
            derived_from=(d.get("derived_from") or None),
            sam2_variant=(d.get("sam2_variant") or None),
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_obbsource_reviewed.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/detectkit/gui/models.py tests/test_obbsource_reviewed.py
git commit -m "feat(detectkit): add reviewed/derived_from/sam2_variant to OBBSource"
```

---

### Task 2: Prompt geometry (label reader + prompt builder)

**Files:**
- Create: `src/hydra_suite/detectkit/jobs/sam2_prompts.py`
- Test: `tests/test_sam2_prompts.py`

**Interfaces:**
- Produces:
  - `SourceBox` dataclass: `aabb: tuple[float,float,float,float]` (x1,y1,x2,y2 px), `center: tuple[float,float]`, `polygon_px: list[tuple[float,float]]`.
  - `read_boxes_from_label(label_path: Path, img_w: int, img_h: int) -> list[SourceBox]` — parses 5-field aabb (`cls cx cy w h`) and 9-field obb (`cls x1..y4`) normalized lines into pixel `SourceBox`es; skips other/blank lines.
  - `Prompt` dataclass: `box_xyxy`, `positive_points: list[tuple[float,float]]`, `negative_points: list[tuple[float,float]]`.
  - `build_prompts(boxes: list[SourceBox]) -> list[Prompt]` — box + center-positive + overlapping-neighbor-center negatives.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sam2_prompts.py
from pathlib import Path
from hydra_suite.detectkit.jobs.sam2_prompts import (
    SourceBox, read_boxes_from_label, build_prompts,
)


def test_read_obb_label_to_pixel_box(tmp_path):
    p = tmp_path / "f.txt"
    # 9-field OBB: unit square from (0.1,0.1) to (0.3,0.3) in a 100x100 image
    p.write_text("0 0.1 0.1 0.3 0.1 0.3 0.3 0.1 0.3\n")
    boxes = read_boxes_from_label(p, 100, 100)
    assert len(boxes) == 1
    assert boxes[0].aabb == (10.0, 10.0, 30.0, 30.0)
    assert boxes[0].center == (20.0, 20.0)


def test_read_aabb_label_to_pixel_box(tmp_path):
    p = tmp_path / "f.txt"
    # 5-field aabb: cx=0.5 cy=0.5 w=0.2 h=0.4 in 100x100 -> x[40,60] y[30,70]
    p.write_text("0 0.5 0.5 0.2 0.4\n")
    boxes = read_boxes_from_label(p, 100, 100)
    assert boxes[0].aabb == (40.0, 30.0, 60.0, 70.0)
    assert boxes[0].center == (50.0, 50.0)


def test_build_prompts_overlapping_neighbor_becomes_negative():
    a = SourceBox(aabb=(0, 0, 10, 10), center=(5, 5), polygon_px=[])
    b = SourceBox(aabb=(8, 8, 18, 18), center=(13, 13), polygon_px=[])   # overlaps a
    c = SourceBox(aabb=(50, 50, 60, 60), center=(55, 55), polygon_px=[])  # disjoint
    prompts = build_prompts([a, b, c])
    # prompt for a: box=a.aabb, positive=[a.center], negative=[b.center] (not c)
    assert prompts[0].box_xyxy == (0, 0, 10, 10)
    assert prompts[0].positive_points == [(5, 5)]
    assert prompts[0].negative_points == [(13, 13)]
    # prompt for c: no overlaps -> no negatives
    assert prompts[2].negative_points == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_sam2_prompts.py -v`
Expected: FAIL (`ModuleNotFoundError: sam2_prompts`).

- [ ] **Step 3: Implement `sam2_prompts.py`**

```python
"""Pure prompt geometry for SAM2 escalation (no SAM2 import)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class SourceBox:
    aabb: tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels
    center: tuple[float, float]
    polygon_px: list[tuple[float, float]]  # original label polygon (fallback)


@dataclass
class Prompt:
    box_xyxy: tuple[float, float, float, float]
    positive_points: list[tuple[float, float]]
    negative_points: list[tuple[float, float]]


def _aabb_of(points: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (min(xs), min(ys), max(xs), max(ys))


def read_boxes_from_label(label_path: Path, img_w: int, img_h: int) -> list[SourceBox]:
    """Parse normalized YOLO aabb (5-field) / obb (9-field) lines to pixel boxes."""
    out: list[SourceBox] = []
    try:
        text = Path(label_path).read_text(encoding="utf-8")
    except Exception:
        return out
    for line in text.splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        try:
            vals = [float(v) for v in parts[1:]]
        except ValueError:
            continue
        if len(vals) == 4:  # aabb: cx cy w h (normalized)
            cx, cy, w, h = vals
            poly = [
                ((cx - w / 2) * img_w, (cy - h / 2) * img_h),
                ((cx + w / 2) * img_w, (cy - h / 2) * img_h),
                ((cx + w / 2) * img_w, (cy + h / 2) * img_h),
                ((cx - w / 2) * img_w, (cy + h / 2) * img_h),
            ]
        elif len(vals) == 8:  # obb: x1 y1 .. x4 y4 (normalized)
            poly = [(vals[i] * img_w, vals[i + 1] * img_h) for i in range(0, 8, 2)]
        else:
            continue
        aabb = _aabb_of(poly)
        center = (sum(p[0] for p in poly) / len(poly),
                  sum(p[1] for p in poly) / len(poly))
        out.append(SourceBox(aabb=aabb, center=center, polygon_px=poly))
    return out


def _overlaps(a: tuple[float, float, float, float],
              b: tuple[float, float, float, float]) -> bool:
    ix = min(a[2], b[2]) - max(a[0], b[0])
    iy = min(a[3], b[3]) - max(a[1], b[1])
    return ix > 0 and iy > 0


def build_prompts(boxes: list[SourceBox]) -> list[Prompt]:
    """Box + center-positive + overlapping-neighbor-center negatives, per box."""
    prompts: list[Prompt] = []
    for i, box in enumerate(boxes):
        negatives = [
            other.center
            for j, other in enumerate(boxes)
            if j != i and _overlaps(box.aabb, other.aabb)
        ]
        prompts.append(Prompt(box_xyxy=box.aabb,
                              positive_points=[box.center],
                              negative_points=negatives))
    return prompts
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_sam2_prompts.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/detectkit/jobs/sam2_prompts.py tests/test_sam2_prompts.py
git commit -m "feat(detectkit): pure SAM2 prompt geometry (label reader + prompt builder)"
```

---

### Task 3: Mask → contour reducer

**Files:**
- Create: `src/hydra_suite/core/inference/sam2/__init__.py` (empty), `src/hydra_suite/core/inference/sam2/masks.py`
- Test: `tests/test_sam2_masks.py`

**Interfaces:**
- Produces: `mask_to_contour(mask: np.ndarray, epsilon_frac: float = 0.01, min_points: int = 6, min_area: float = 4.0) -> np.ndarray | None` — largest external contour as an `(P, 2)` float32 pixel array, simplified; `None` for empty/degenerate masks.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sam2_masks.py
import numpy as np
from hydra_suite.core.inference.sam2.masks import mask_to_contour


def test_square_mask_yields_rectangular_contour():
    mask = np.zeros((50, 50), dtype=bool)
    mask[10:40, 15:35] = True
    poly = mask_to_contour(mask)
    assert poly is not None and poly.shape[1] == 2
    xs, ys = poly[:, 0], poly[:, 1]
    assert xs.min() <= 16 and xs.max() >= 33
    assert ys.min() <= 11 and ys.max() >= 38


def test_empty_mask_returns_none():
    assert mask_to_contour(np.zeros((20, 20), dtype=bool)) is None


def test_largest_contour_selected_over_speck():
    mask = np.zeros((60, 60), dtype=bool)
    mask[5:55, 5:55] = True      # big blob
    mask[0:2, 0:2] = True         # tiny speck (separate)
    poly = mask_to_contour(mask)
    # bbox of returned contour must be the big blob, not the speck
    assert poly[:, 0].max() > 40 and poly[:, 1].max() > 40
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_sam2_masks.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement `masks.py`**

```python
"""Binary mask -> simplified largest external contour (SAM2 escalation)."""
from __future__ import annotations

import cv2
import numpy as np


def mask_to_contour(
    mask: np.ndarray,
    epsilon_frac: float = 0.01,
    min_points: int = 6,
    min_area: float = 4.0,
) -> np.ndarray | None:
    """Largest external contour of a binary mask as an (P, 2) float32 array.

    Simplified with approxPolyDP (epsilon = epsilon_frac * perimeter). Returns
    None when the mask is empty or the largest contour is degenerate. Single
    contour only (YOLO-seg has no holes).
    """
    m = np.ascontiguousarray(mask.astype(np.uint8))
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < min_area:
        return None
    eps = epsilon_frac * cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, eps, True).reshape(-1, 2).astype(np.float32)
    if approx.shape[0] < min_points:
        approx = c.reshape(-1, 2).astype(np.float32)  # keep detail if oversimplified
    return approx
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_sam2_masks.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/core/inference/sam2/__init__.py src/hydra_suite/core/inference/sam2/masks.py tests/test_sam2_masks.py
git commit -m "feat(inference): mask_to_contour reducer for SAM2 escalation"
```

---

### Task 4: SAM2 checkpoint catalog + managed download

**Files:**
- Create: `src/hydra_suite/core/inference/sam2/checkpoints.py`
- Modify: `pyproject.toml` (add `sam2` dependency)
- Test: `tests/test_sam2_checkpoints.py`

**Interfaces:**
- Produces:
  - `SAM2_VARIANTS: dict[str, Sam2Entry]` keyed by variant (`sam2.1-hiera-tiny|small|base_plus|large`), default `DEFAULT_VARIANT = "sam2.1-hiera-base_plus"`.
  - `Sam2Entry` dataclass: `repo_id: str`, `filename: str`, `config_name: str`.
  - `available_variants() -> list[str]`.
  - `ensure_checkpoint(variant: str, *, allow_download: bool = True, cache_dir: Path | None = None) -> Path` — returns cached checkpoint path; downloads via HF when missing and `allow_download`; raises `ValueError` naming the variant when uncached and `not allow_download` (offline) or variant unknown.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sam2_checkpoints.py
import pytest
from pathlib import Path
from hydra_suite.core.inference.sam2 import checkpoints as ck


def test_default_variant_is_in_catalog():
    assert ck.DEFAULT_VARIANT in ck.SAM2_VARIANTS
    assert "sam2.1-hiera-large" in ck.available_variants()


def test_unknown_variant_raises_named(tmp_path):
    with pytest.raises(ValueError, match="bogus"):
        ck.ensure_checkpoint("bogus", cache_dir=tmp_path)


def test_offline_uncached_raises_named(tmp_path):
    with pytest.raises(ValueError, match="not downloaded"):
        ck.ensure_checkpoint(ck.DEFAULT_VARIANT, allow_download=False, cache_dir=tmp_path)


def test_cached_checkpoint_returned_without_download(tmp_path, monkeypatch):
    variant = ck.DEFAULT_VARIANT
    dest = tmp_path / f"{variant}.pt"
    dest.write_bytes(b"fake")
    def _boom(*a, **k):
        raise AssertionError("should not download when cached")
    monkeypatch.setattr(ck, "hf_hub_download", _boom)
    assert ck.ensure_checkpoint(variant, cache_dir=tmp_path) == dest
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_sam2_checkpoints.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement `checkpoints.py`**

```python
"""SAM2 checkpoint catalog + HF-managed download (mirrors vitpose_checkpoints)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import hf_hub_download

from hydra_suite.paths import get_models_dir


@dataclass(frozen=True)
class Sam2Entry:
    repo_id: str
    filename: str
    config_name: str  # sam2 package config (e.g. "configs/sam2.1/sam2.1_hiera_b+.yaml")


# NOTE: repo_id/filename/config_name pinned to the `sam2` package's published
# assets at implementation time (verify against the installed sam2 version).
SAM2_VARIANTS: dict[str, Sam2Entry] = {
    "sam2.1-hiera-tiny": Sam2Entry(
        "facebook/sam2.1-hiera-tiny", "sam2.1_hiera_tiny.pt",
        "configs/sam2.1/sam2.1_hiera_t.yaml"),
    "sam2.1-hiera-small": Sam2Entry(
        "facebook/sam2.1-hiera-small", "sam2.1_hiera_small.pt",
        "configs/sam2.1/sam2.1_hiera_s.yaml"),
    "sam2.1-hiera-base_plus": Sam2Entry(
        "facebook/sam2.1-hiera-base-plus", "sam2.1_hiera_base_plus.pt",
        "configs/sam2.1/sam2.1_hiera_b+.yaml"),
    "sam2.1-hiera-large": Sam2Entry(
        "facebook/sam2.1-hiera-large", "sam2.1_hiera_large.pt",
        "configs/sam2.1/sam2.1_hiera_l.yaml"),
}

DEFAULT_VARIANT = "sam2.1-hiera-base_plus"


def available_variants() -> list[str]:
    return list(SAM2_VARIANTS.keys())


def _cache_dir(cache_dir: Path | None) -> Path:
    return Path(cache_dir) if cache_dir is not None else Path(get_models_dir()) / "sam2"


def ensure_checkpoint(
    variant: str, *, allow_download: bool = True, cache_dir: Path | None = None
) -> Path:
    """Return the cached SAM2 checkpoint path, downloading from HF if needed."""
    if variant not in SAM2_VARIANTS:
        raise ValueError(
            f"Unknown SAM2 variant {variant!r}. "
            f"Available: {', '.join(available_variants())}."
        )
    entry = SAM2_VARIANTS[variant]
    cdir = _cache_dir(cache_dir)
    dest = cdir / f"{variant}.pt"
    if dest.exists():
        return dest
    if not allow_download:
        raise ValueError(
            f"SAM2 variant {variant!r} is not downloaded and downloads are "
            f"disabled (offline). Download it once with network access."
        )
    cdir.mkdir(parents=True, exist_ok=True)
    src = Path(hf_hub_download(repo_id=entry.repo_id, filename=entry.filename))
    dest.write_bytes(src.read_bytes())
    return dest
```

Add to `pyproject.toml` `[project.dependencies]` (verify exact version pin against the installed package at implementation time):

```toml
    "sam2>=1.1",
```

- [ ] **Step 4: Run tests + verify import**

Run: `python -m pytest tests/test_sam2_checkpoints.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/core/inference/sam2/checkpoints.py tests/test_sam2_checkpoints.py pyproject.toml
git commit -m "feat(inference): SAM2 checkpoint catalog + HF-managed download; add sam2 dep"
```

---

### Task 5: `Sam2SegmentExecutor` (lazy import, device-resolved)

**Files:**
- Create: `src/hydra_suite/core/inference/sam2/executor.py`
- Test: `tests/test_sam2_executor.py`

**Interfaces:**
- Consumes: `ensure_checkpoint` (Task 4), `SAM2_VARIANTS`.
- Produces:
  - `resolve_sam2_device() -> str` — `"cuda"` if CUDA else `"mps"` if MPS else `"cpu"`.
  - `Sam2SegmentExecutor` with `set_image(image_bgr: np.ndarray) -> None` and `segment(box_xyxy, positive_points, negative_points) -> tuple[np.ndarray, float]` (best-IoU boolean mask HxW + its predicted IoU). Constructed via `Sam2SegmentExecutor.from_variant(variant, device=None)`.

- [ ] **Step 1: Write the failing test** (device resolution is pure and testable without weights; the predictor is injected)

```python
# tests/test_sam2_executor.py
import numpy as np
from hydra_suite.core.inference.sam2 import executor as ex


def test_resolve_device_prefers_cuda(monkeypatch):
    monkeypatch.setattr(ex, "TORCH_CUDA_AVAILABLE", True)
    monkeypatch.setattr(ex, "MPS_AVAILABLE", False)
    assert ex.resolve_sam2_device() == "cuda"


def test_resolve_device_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(ex, "TORCH_CUDA_AVAILABLE", False)
    monkeypatch.setattr(ex, "MPS_AVAILABLE", False)
    assert ex.resolve_sam2_device() == "cpu"


def test_segment_picks_highest_iou_mask():
    # Inject a fake SAM2 image-predictor: predict() returns 3 masks + 3 ious.
    class _FakePredictor:
        def set_image(self, rgb): self.rgb = rgb
        def predict(self, box=None, point_coords=None, point_labels=None,
                    multimask_output=True):
            masks = np.stack([
                np.zeros((4, 4), bool),
                np.ones((4, 4), bool),      # best
                np.zeros((4, 4), bool),
            ])
            ious = np.array([0.1, 0.9, 0.2])
            return masks, ious, None
    e = ex.Sam2SegmentExecutor(_FakePredictor())
    e.set_image(np.zeros((4, 4, 3), np.uint8))
    mask, iou = e.segment((0, 0, 4, 4), [(2, 2)], [(0, 0)])
    assert iou == 0.9 and mask.all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_sam2_executor.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement `executor.py`** (lazy `sam2` import lives only in `from_variant`)

```python
"""Standalone torch-only SAM2 prompt-in/mask-out executor (lazy sam2 import)."""
from __future__ import annotations

import cv2
import numpy as np

from hydra_suite.utils.gpu_utils import MPS_AVAILABLE, TORCH_CUDA_AVAILABLE

from .checkpoints import SAM2_VARIANTS, ensure_checkpoint


def resolve_sam2_device() -> str:
    if TORCH_CUDA_AVAILABLE:
        return "cuda"
    if MPS_AVAILABLE:
        return "mps"
    return "cpu"


class Sam2SegmentExecutor:
    """Wraps a SAM2 image predictor: set_image once, segment per prompt."""

    def __init__(self, predictor) -> None:
        self._predictor = predictor

    @classmethod
    def from_variant(cls, variant: str, device: str | None = None,
                     *, allow_download: bool = True) -> "Sam2SegmentExecutor":
        # Lazy import: only paid when escalation actually runs.
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        entry = SAM2_VARIANTS[variant]
        ckpt = ensure_checkpoint(variant, allow_download=allow_download)
        dev = device or resolve_sam2_device()
        model = build_sam2(entry.config_name, str(ckpt), device=dev)
        return cls(SAM2ImagePredictor(model))

    def set_image(self, image_bgr: np.ndarray) -> None:
        self._predictor.set_image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))

    def segment(self, box_xyxy, positive_points, negative_points):
        pts = list(positive_points) + list(negative_points)
        labels = [1] * len(positive_points) + [0] * len(negative_points)
        masks, ious, _ = self._predictor.predict(
            box=np.array(box_xyxy, dtype=np.float32),
            point_coords=np.array(pts, dtype=np.float32) if pts else None,
            point_labels=np.array(labels, dtype=np.int32) if pts else None,
            multimask_output=True,
        )
        best = int(np.argmax(ious))
        return masks[best].astype(bool), float(ious[best])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_sam2_executor.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/core/inference/sam2/executor.py tests/test_sam2_executor.py
git commit -m "feat(inference): Sam2SegmentExecutor (lazy import, device-resolved, best-IoU)"
```

---

### Task 6: Escalation orchestrator (pure `run_escalation`)

**Files:**
- Create: `src/hydra_suite/detectkit/jobs/sam2_escalation.py`
- Test: `tests/test_sam2_escalation.py`

**Interfaces:**
- Consumes: `read_boxes_from_label`/`build_prompts` (Task 2), `mask_to_contour` (Task 3), `_write_geometry_label` (`al_worker.py:116`), `OBBSource` (Task 1), `parse` helpers.
- Produces:
  - `EscalationRequest` dataclass: `project`, `source_names: list[str]`, `variant: str`.
  - `EscalationResult` dataclass: `derived: list[str]` (new source names), `primed: int`, `fell_back: int`.
  - `run_escalation(req, executor, *, progress=None) -> EscalationResult` — `executor` is any object with `set_image`/`segment` (injected fake in tests). Writes `<name>_seg` sources, registers them `reviewed=False`, `derived_from`, `sam2_variant`; empty-mask → OBB-corner fallback via `_write_geometry_label` (`rec[6]` = original polygon).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sam2_escalation.py
import types
import numpy as np
import cv2
from pathlib import Path
from hydra_suite.detectkit.gui.models import OBBSource
from hydra_suite.detectkit.jobs.sam2_escalation import (
    EscalationRequest, run_escalation,
)


class _FakeExec:
    """Returns a full-object mask for detection 0, empty mask for others."""
    def __init__(self): self.calls = 0
    def set_image(self, img): pass
    def segment(self, box, pos, neg):
        self.calls += 1
        if self.calls == 1:
            m = np.zeros((100, 100), bool); m[10:40, 10:40] = True
            return m, 0.9
        return np.zeros((100, 100), bool), 0.0  # -> fallback


def _make_source(tmp_path):
    root = tmp_path / "sources" / "orig"
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir(parents=True)
    cv2.imwrite(str(root / "images" / "a.jpg"), np.zeros((100, 100, 3), np.uint8))
    # two OBB detections
    (root / "labels" / "a.txt").write_text(
        "0 0.1 0.1 0.4 0.1 0.4 0.4 0.1 0.4\n"
        "0 0.6 0.6 0.9 0.6 0.9 0.9 0.6 0.9\n")
    (root / "classes.txt").write_text("ant\n")
    return OBBSource(path=str(root), name="orig", level="obb")


def test_escalation_writes_reviewed_false_derived_source(tmp_path):
    src = _make_source(tmp_path)
    project = types.SimpleNamespace(project_dir=str(tmp_path), sources=[src])
    req = EscalationRequest(project=project, source_names=["orig"],
                            variant="sam2.1-hiera-base_plus")
    result = run_escalation(req, _FakeExec())

    assert result.derived == ["orig_seg"]
    assert result.primed == 1 and result.fell_back == 1
    new = [s for s in project.sources if s.name == "orig_seg"][0]
    assert new.level == "polygon" and new.reviewed is False
    assert new.derived_from == "orig" and new.sam2_variant == "sam2.1-hiera-base_plus"
    label = Path(new.path) / "labels" / "a.txt"
    assert label.exists() and len(label.read_text().splitlines()) == 2
    assert (Path(new.path) / "images" / "a.jpg").exists()  # image copied
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_sam2_escalation.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement `sam2_escalation.py`** (worker added in Task 7; this file holds the pure function first)

```python
"""SAM2 escalation orchestrator: existing OBB/box labels -> primed seg source."""
from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2

from hydra_suite.detectkit.gui.models import OBBSource
from hydra_suite.detectkit.jobs.al_worker import _write_geometry_label
from hydra_suite.core.inference.sam2.masks import mask_to_contour
from .sam2_prompts import build_prompts, read_boxes_from_label


@dataclass
class EscalationRequest:
    project: object            # has .project_dir and .sources (list[OBBSource])
    source_names: list[str]
    variant: str


@dataclass
class EscalationResult:
    derived: list[str] = field(default_factory=list)
    primed: int = 0
    fell_back: int = 0


def _sources_by_name(project) -> dict[str, OBBSource]:
    return {s.name: s for s in project.sources}


def run_escalation(
    req: EscalationRequest,
    executor,
    *,
    progress: Callable[[int, str], None] | None = None,
) -> EscalationResult:
    """Escalate each named source to a new <name>_seg source (reviewed=False)."""
    result = EscalationResult()
    by_name = _sources_by_name(req.project)
    todo = [by_name[n] for n in req.source_names
            if n in by_name and by_name[n].level != "polygon"]
    for si, src in enumerate(todo):
        src_root = Path(src.path)
        images_dir = src_root / "images"
        labels_dir = src_root / "labels"
        out_name = f"{src.name}_seg"
        out_root = Path(req.project.project_dir) / "sources" / out_name
        (out_root / "images").mkdir(parents=True, exist_ok=True)
        (out_root / "labels").mkdir(parents=True, exist_ok=True)

        images = sorted(p for p in images_dir.glob("*")
                        if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        for ii, img_path in enumerate(images):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            label_path = labels_dir / f"{img_path.stem}.txt"
            boxes = read_boxes_from_label(label_path, w, h)
            records = []
            if boxes:
                prompts = build_prompts(boxes)
                executor.set_image(img)
                for box, prompt in zip(boxes, prompts):
                    mask, _iou = executor.segment(
                        prompt.box_xyxy, prompt.positive_points,
                        prompt.negative_points)
                    contour = mask_to_contour(mask)
                    if contour is not None:
                        result.primed += 1
                        poly = contour
                    else:  # fallback: original OBB corners as the polygon
                        result.fell_back += 1
                        poly = box.polygon_px
                    records.append((0.0, 0.0, 0.0, 0.0, 0.0, 1.0, poly))
            _write_geometry_label(
                out_root / "labels" / f"{img_path.stem}.txt", records, (h, w))
            shutil.copy2(img_path, out_root / "images" / img_path.name)
            if progress:
                progress(int(100 * (si + (ii + 1) / max(len(images), 1)) / len(todo)),
                         f"{src.name}: {ii + 1}/{len(images)}")

        (out_root / "classes.txt").write_text(
            (src_root / "classes.txt").read_text()
            if (src_root / "classes.txt").exists() else "object\n")
        req.project.sources.append(OBBSource(
            path=str(out_root), name=out_name, level="polygon",
            reviewed=False, derived_from=src.name, sam2_variant=req.variant,
            source_kind="detectkit_sam2", imported=True))
        result.derived.append(out_name)
    return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_sam2_escalation.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/detectkit/jobs/sam2_escalation.py tests/test_sam2_escalation.py
git commit -m "feat(detectkit): pure SAM2 escalation orchestrator (run_escalation)"
```

---

### Task 7: `Sam2EscalationWorker` (BaseWorker wrapper)

**Files:**
- Modify: `src/hydra_suite/detectkit/jobs/sam2_escalation.py`
- Test: `tests/test_sam2_escalation.py` (add a worker test)

**Interfaces:**
- Consumes: `run_escalation` (Task 6), `Sam2SegmentExecutor.from_variant` (Task 5), `BaseWorker` (`widgets/workers.py:6`).
- Produces: `Sam2EscalationWorker(BaseWorker)` with a `result_ready(object)` signal; `execute()` builds the executor from the request's variant, runs `run_escalation`, emits `progress`/`status`, then `result_ready`. Accepts an optional `executor` injection for tests.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_sam2_escalation.py
def test_worker_runs_with_injected_executor(tmp_path):
    from hydra_suite.detectkit.jobs.sam2_escalation import (
        EscalationRequest, Sam2EscalationWorker)
    src = _make_source(tmp_path)
    project = types.SimpleNamespace(project_dir=str(tmp_path), sources=[src])
    req = EscalationRequest(project=project, source_names=["orig"],
                            variant="sam2.1-hiera-base_plus")
    worker = Sam2EscalationWorker(req, executor=_FakeExec())
    captured = {}
    worker.result_ready.connect(lambda r: captured.update(derived=r.derived))
    worker.execute()  # call directly (no thread) — BaseWorker pattern
    assert captured["derived"] == ["orig_seg"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_sam2_escalation.py::test_worker_runs_with_injected_executor -v`
Expected: FAIL (`Sam2EscalationWorker` undefined).

- [ ] **Step 3: Add the worker to `sam2_escalation.py`**

```python
from hydra_suite.widgets.workers import BaseWorker
from PySide6.QtCore import Signal


class Sam2EscalationWorker(BaseWorker):
    """QThread wrapper around run_escalation (BaseWorker signals + result_ready)."""

    result_ready = Signal(object)  # EscalationResult

    def __init__(self, request: EscalationRequest, executor=None, parent=None) -> None:
        super().__init__(parent)
        self._request = request
        self._executor = executor

    def execute(self) -> None:
        from hydra_suite.core.inference.sam2.executor import Sam2SegmentExecutor
        executor = self._executor or Sam2SegmentExecutor.from_variant(
            self._request.variant)
        self.status.emit(f"Escalating {len(self._request.source_names)} source(s)...")
        result = run_escalation(
            self._request, executor,
            progress=lambda pct, msg: (self.progress.emit(pct), self.status.emit(msg)))
        self.status.emit(
            f"Done: {result.primed} primed, {result.fell_back} fell back "
            f"(review these first).")
        self.result_ready.emit(result)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_sam2_escalation.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/detectkit/jobs/sam2_escalation.py tests/test_sam2_escalation.py
git commit -m "feat(detectkit): Sam2EscalationWorker (BaseWorker) around run_escalation"
```

---

### Task 8: Training gating excludes unreviewed sources

**Files:**
- Modify: `src/hydra_suite/training/dataset_builders.py` (source-collection point)
- Test: `tests/test_training_gating_reviewed.py`

**Interfaces:**
- Consumes: `OBBSource.reviewed` (Task 1).
- Produces: `eligible_sources(sources: list) -> tuple[list, list[str]]` — returns `(kept, skipped_messages)`, dropping `reviewed=False` sources with a message naming each and why. Used at the dataset-build source-collection point.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_training_gating_reviewed.py
from hydra_suite.detectkit.gui.models import OBBSource
from hydra_suite.training.dataset_builders import eligible_sources


def test_unreviewed_source_excluded_with_message():
    ok = OBBSource(name="orig", level="obb", reviewed=True)
    pending = OBBSource(name="orig_seg", level="polygon", reviewed=False)
    kept, messages = eligible_sources([ok, pending])
    assert [s.name for s in kept] == ["orig"]
    assert any("orig_seg" in m and "unreviewed" in m.lower() for m in messages)


def test_all_reviewed_sources_kept():
    a = OBBSource(name="a", reviewed=True)
    b = OBBSource(name="b", reviewed=True)
    kept, messages = eligible_sources([a, b])
    assert len(kept) == 2 and messages == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_training_gating_reviewed.py -v`
Expected: FAIL (`ImportError: cannot import name 'eligible_sources'`).

- [ ] **Step 3: Implement `eligible_sources` + wire it into the build**

Add to `dataset_builders.py`:

```python
def eligible_sources(sources):
    """Drop reviewed=False sources from a training build, returning skip messages."""
    kept, messages = [], []
    for s in sources:
        if getattr(s, "reviewed", True):
            kept.append(s)
        else:
            messages.append(
                f"Source '{s.name}' is unreviewed (SAM2-primed) — review it in "
                f"X-AnyLabeling and Mark reviewed before it can be used for training."
            )
    return kept, messages
```

Then, at the point where `dataset_builders` collects sources for a build (the function that iterates `project.sources` / receives a `sources` list — locate the existing source-iteration site), replace the raw list with `kept, skipped = eligible_sources(sources)` and surface `skipped` to the caller's log/return. Keep the change minimal: filter at the single collection site so every role/level path inherits it.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_training_gating_reviewed.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/training/dataset_builders.py tests/test_training_gating_reviewed.py
git commit -m "feat(training): exclude unreviewed SAM2-primed sources from builds"
```

---

### Task 9: Escalate dialog (`EscalateSam2Dialog`)

**Files:**
- Create: `src/hydra_suite/detectkit/gui/dialogs/escalate_sam2_dialog.py`
- Test: `tests/test_escalate_sam2_dialog.py`

**Interfaces:**
- Consumes: `available_variants`/`DEFAULT_VARIANT` (Task 4), `BaseDialog` (`widgets/dialogs.py`), `EscalationRequest` (Task 6).
- Produces: `EscalateSam2Dialog(BaseDialog)` with `selected_sources() -> list[str]`, `selected_variant() -> str`; lists sources (already-`polygon` disabled), a **variant dropdown** (default `DEFAULT_VARIANT`), and OK/Cancel.

- [ ] **Step 1: Write the failing test** (headless Qt; construct + assert widgets, no exec)

```python
# tests/test_escalate_sam2_dialog.py
import pytest
pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication
from hydra_suite.detectkit.gui.models import OBBSource
from hydra_suite.detectkit.gui.dialogs.escalate_sam2_dialog import EscalateSam2Dialog
from hydra_suite.core.inference.sam2.checkpoints import DEFAULT_VARIANT, available_variants

_app = QApplication.instance() or QApplication([])


def test_dialog_lists_variants_and_eligible_sources():
    sources = [OBBSource(name="a", level="obb"),
               OBBSource(name="b_seg", level="polygon")]  # already polygon -> disabled
    dlg = EscalateSam2Dialog(sources)
    assert dlg.selected_variant() == DEFAULT_VARIANT
    assert set(dlg._variant_combo_items()) == set(available_variants())
    # 'a' selectable, 'b_seg' disabled
    assert "a" in dlg.selectable_source_names()
    assert "b_seg" not in dlg.selectable_source_names()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_escalate_sam2_dialog.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement the dialog** (mirror an existing `BaseDialog` subclass in `detectkit/gui/dialogs/` for layout conventions)

```python
"""Dialog to pick sources + SAM2 variant for escalate-all."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QComboBox, QLabel, QListWidget, QListWidgetItem,
                               QVBoxLayout)

from hydra_suite.widgets.dialogs import BaseDialog
from hydra_suite.core.inference.sam2.checkpoints import (
    DEFAULT_VARIANT, available_variants)


class EscalateSam2Dialog(BaseDialog):
    def __init__(self, sources, parent=None) -> None:
        super().__init__(parent, title="Escalate to segment (SAM2)")
        self._sources = sources
        layout = QVBoxLayout()
        layout.addWidget(QLabel("SAM2 version:"))
        self._variant = QComboBox()
        for v in available_variants():
            self._variant.addItem(v)
        self._variant.setCurrentText(DEFAULT_VARIANT)
        layout.addWidget(self._variant)
        layout.addWidget(QLabel("Sources to escalate:"))
        self._list = QListWidget()
        for s in sources:
            item = QListWidgetItem(s.name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            if s.level == "polygon":
                item.setFlags(item.flags() & ~Qt.ItemIsEnabled)  # already segment
                item.setCheckState(Qt.Unchecked)
            else:
                item.setCheckState(Qt.Checked)
            self._list.addItem(item)
        layout.addWidget(self._list)
        self.set_content_layout(layout)  # BaseDialog adds the OK/Cancel button box

    def _variant_combo_items(self) -> list[str]:
        return [self._variant.itemText(i) for i in range(self._variant.count())]

    def selectable_source_names(self) -> list[str]:
        return [self._list.item(i).text() for i in range(self._list.count())
                if self._list.item(i).flags() & Qt.ItemIsEnabled]

    def selected_variant(self) -> str:
        return self._variant.currentText()

    def selected_sources(self) -> list[str]:
        return [self._list.item(i).text() for i in range(self._list.count())
                if self._list.item(i).checkState() == Qt.Checked]
```

Note: match `BaseDialog`'s actual constructor/content API (`set_content_layout` or equivalent) to the existing subclasses in `detectkit/gui/dialogs/`; adjust method names if the base differs.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_escalate_sam2_dialog.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/detectkit/gui/dialogs/escalate_sam2_dialog.py tests/test_escalate_sam2_dialog.py
git commit -m "feat(detectkit): EscalateSam2Dialog (source picker + SAM2 variant dropdown)"
```

---

### Task 10: Wire entry points (button, tooltip action, "Mark reviewed")

**Files:**
- Modify: `src/hydra_suite/detectkit/gui/panels/tools_panel.py` (or the dataset panel) — primary button + "Mark reviewed" action.
- Modify: `src/hydra_suite/detectkit/gui/dialogs/training_dialog.py:~1504` — secondary "Escalate '{who}'…" action at the role-gating tooltip.
- Modify: `src/hydra_suite/detectkit/gui/main_window.py` — handler that opens `EscalateSam2Dialog`, launches `Sam2EscalationWorker`, refreshes the source list on `result_ready`.
- Test: manual (GUI wiring); covered by a smoke construction test if a panel test harness exists.

**Interfaces:**
- Consumes: `EscalateSam2Dialog` (Task 9), `Sam2EscalationWorker` (Task 7), `OBBSource.reviewed` (Task 1).

- [ ] **Step 1: Add the primary action handler in `main_window.py`**

Add a slot that: builds `EscalateSam2Dialog(self.project.sources)`, on accept builds `EscalationRequest(project=self.project, source_names=dlg.selected_sources(), variant=dlg.selected_variant())`, starts a `Sam2EscalationWorker`, wires `progress`/`status`/`error` to the existing status UI, and on `result_ready` appends the new sources to the project view + shows the primed/fell-back summary. Disable the action (with a tooltip) if `available_variants()` import fails.

- [ ] **Step 2: Add the "Escalate to segment (SAM2)" button** to the tools/dataset panel, connected to the slot from Step 1.

- [ ] **Step 3: Add "Mark reviewed" action** for a selected derived source: set `source.reviewed = True`, persist the project, refresh the list. Enable it only for `reviewed=False` sources.

- [ ] **Step 4: Add the secondary entry** at `training_dialog.py:~1504` — where the tooltip names the blocking source, add an "Escalate '{who}' to segment (SAM2)" button that opens the same dialog pre-selecting `{who}`.

- [ ] **Step 5: Manual verification**

Run `detectkit`, open a project with an OBB source:
- Tools panel shows "Escalate to segment (SAM2)"; clicking opens the dialog with the variant dropdown (default base_plus) and the source checked.
- Running creates `<name>_seg` (level polygon, greyed as unreviewed), leaves the original untouched.
- "Open in X-AnyLabeling" opens it in `--mode segment`; "Mark reviewed" clears the unreviewed state.
- The training dialog excludes the unreviewed source with the skip message and offers the escalate action at the blocked-role tooltip.

- [ ] **Step 6: Commit**

```bash
git add src/hydra_suite/detectkit/gui/panels/ src/hydra_suite/detectkit/gui/dialogs/training_dialog.py src/hydra_suite/detectkit/gui/main_window.py
git commit -m "feat(detectkit): wire SAM2 escalate action, Mark-reviewed, and tooltip entry"
```

---

### Task 11: Docs + guarded real-SAM2 smoke

**Files:**
- Modify: `docs/developer-guide/runtime-integration.md` — note the `sam2_segment` standalone executor (outside the tier system) + the escalate workflow.
- Modify: `CLAUDE.md` PoseKit/DetectKit section — one line on the escalate-to-segment feature (optional).
- Test: `tests/test_sam2_executor.py` — add a guarded real smoke.

- [ ] **Step 1: Add the guarded real-SAM2 smoke test**

```python
# add to tests/test_sam2_executor.py
import sys
import pytest


@pytest.mark.skipif(sys.platform not in ("darwin", "linux"), reason="torch device")
def test_real_sam2_segment_smoke(tmp_path):
    pytest.importorskip("sam2")
    import numpy as np
    from hydra_suite.core.inference.sam2.executor import Sam2SegmentExecutor
    from hydra_suite.core.inference.sam2.checkpoints import DEFAULT_VARIANT
    try:
        ex = Sam2SegmentExecutor.from_variant(DEFAULT_VARIANT)
    except Exception as e:  # weights not downloaded in CI, etc.
        pytest.skip(f"SAM2 weights unavailable: {e}")
    img = (np.random.rand(256, 256, 3) * 255).astype("uint8")
    ex.set_image(img)
    mask, iou = ex.segment((50, 50, 200, 200), [(125, 125)], [])
    assert mask.shape == (256, 256) and 0.0 <= iou <= 1.0
```

- [ ] **Step 2: Document the integration**

In `docs/developer-guide/runtime-integration.md`, add a short subsection: SAM2 escalation runs a standalone `Sam2SegmentExecutor` (torch-only, prompt-in/mask-out) that intentionally sits **outside** the tier/`InferenceRunner` system because prompt models don't fit `predict(frame)`. New pipeline key: `sam2_segment`.

- [ ] **Step 3: Run the full targeted suite**

Run: `python -m pytest tests/test_sam2_prompts.py tests/test_sam2_masks.py tests/test_sam2_checkpoints.py tests/test_sam2_executor.py tests/test_sam2_escalation.py tests/test_obbsource_reviewed.py tests/test_training_gating_reviewed.py -v`
Expected: PASS (real-SAM2 smoke skipped without weights).

- [ ] **Step 4: Commit**

```bash
git add docs/developer-guide/runtime-integration.md CLAUDE.md tests/test_sam2_executor.py
git commit -m "docs(inference): document SAM2 escalation executor; guarded real smoke"
```

---

## Self-Review Notes

- **Spec coverage:** §3 decisions → Tasks 1 (reviewed model), 4 (hard dep + managed weights + variants), 6/7 (escalate-all + derived source + trust), 2/5 (prompts + executor); §4 data flow → Task 6; §5 components → Tasks 2–10; §6 error handling → Task 6 (fallback), Task 4 (offline), Task 9 (already-polygon disabled); §7 testing → each task's tests + Task 11 smoke. All covered.
- **Variant selection** (the late-added requirement): Task 4 (catalog + per-variant download) + Task 9 (dropdown) + Task 6/7 (persisted on the derived source).
- **Deferred/verify-at-impl:** exact SAM2 HF repo ids/filenames/config names (Task 4) and `BaseDialog`'s content API (Task 9) are flagged to confirm against the installed packages during implementation.
