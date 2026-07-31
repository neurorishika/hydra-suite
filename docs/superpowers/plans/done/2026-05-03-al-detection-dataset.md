# AL Detection Dataset Generation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a shared `data/al/` core (frame source, FilterKit-backed candidate pool, per-frame signals, acquisition selector), refactor the existing TrackerKit dataset path onto it, and add a DetectKit-side AL loop (worker + dialog) that uses the trained detector itself to pick frames for labeling.

**Architecture:** New `hydra_suite/data/al/` package owns the math: `frame_source.py` (Protocol + 3 adapters), `candidate_pool.py` (wraps `FilterKitCore`), `signals.py` (`ALSignals` dataclass + scorers), `acquisition.py` (weight presets + ranked selection with diversity guard). TrackerKit's `FrameQualityScorer` and `get_worst_frames` get rewritten as thin adapters over this core. DetectKit gets a `BaseWorker` subclass and a `BaseDialog` modal that ties FilterKit subsampling + project active-model inference + acquisition.select into one workflow.

**Tech Stack:** Python 3.11, NumPy, OpenCV, PySide6 (Qt), pytest. Reuses `hydra_suite.widgets.workers.BaseWorker`, `hydra_suite.widgets.dialogs.BaseDialog`, `hydra_suite.filterkit.core.FilterKitCore`, and the existing `compute_runtime` model loader.

**Spec:** `docs/superpowers/specs/2026-05-03-al-detection-dataset-design.md`

---

## File Structure

**New:**
- `src/hydra_suite/data/al/__init__.py` — re-exports `FrameRef`, `ALSignals`, `AcquisitionWeights`, `select`.
- `src/hydra_suite/data/al/frame_source.py` — `FrameRef`, `FrameSource` Protocol, `VideoFrameSource`, `ImageFolderFrameSource`, `DetectKitProjectSource`.
- `src/hydra_suite/data/al/candidate_pool.py` — `CandidatePoolConfig`, `build_candidate_pool()`.
- `src/hydra_suite/data/al/signals.py` — `ALSignals` dataclass, `score_uncertainty`, `score_count_deviation`, `score_crowd`, `score_nms_instability`.
- `src/hydra_suite/data/al/acquisition.py` — `AcquisitionWeights`, `PRESETS`, `select`.
- `src/hydra_suite/detectkit/jobs/__init__.py` (package marker).
- `src/hydra_suite/detectkit/jobs/al_worker.py` — `ALWorker(BaseWorker)`, plus a pure `run_active_learning()` function.
- `src/hydra_suite/detectkit/gui/dialogs/active_learning.py` — `ActiveLearningDialog(BaseDialog)`.
- `tests/test_al_frame_source.py`, `tests/test_al_candidate_pool.py`, `tests/test_al_signals.py`, `tests/test_al_acquisition.py`, `tests/test_detectkit_al_worker.py`, `tests/test_detectkit_al_dialog.py`.

**Modified:**
- `src/hydra_suite/data/dataset_generation.py` — `FrameQualityScorer` and `get_worst_frames` delegate into `data/al/`.
- `src/hydra_suite/trackerkit/gui/panels/dataset_panel.py` — replace "Quality threshold" widget with "Min selection score"; add preset combo.
- `src/hydra_suite/detectkit/gui/main_window.py` — add menu entry to open the new dialog.
- `src/hydra_suite/detectkit/gui/dialogs/__init__.py` — add `ActiveLearningDialog` to package exports if module enumerates them.

---

## Task 1 — Scaffold `data/al/` package + `FrameRef`/`FrameSource` Protocol + `VideoFrameSource`

**Files:**
- Create: `src/hydra_suite/data/al/__init__.py`
- Create: `src/hydra_suite/data/al/frame_source.py`
- Create: `tests/test_al_frame_source.py`

- [ ] **Step 1: Write the failing test for VideoFrameSource**

`tests/test_al_frame_source.py`:
```python
"""Tests for hydra_suite.data.al.frame_source."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from hydra_suite.data.al.frame_source import (
    FrameRef,
    VideoFrameSource,
)


def _write_synthetic_video(path: Path, n_frames: int, size: tuple[int, int] = (64, 48)) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 10.0, size)
    try:
        for i in range(n_frames):
            frame = np.full((size[1], size[0], 3), i % 255, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()


def test_video_frame_source_iterates_with_stride(tmp_path):
    video = tmp_path / "synth.mp4"
    _write_synthetic_video(video, n_frames=10)

    src = VideoFrameSource(str(video), stride=2)
    refs = list(src)

    assert all(isinstance(r, FrameRef) for r in refs)
    assert [r.frame_id for r in refs] == [0, 2, 4, 6, 8]
    assert all(r.path is None for r in refs)
    assert src.length() == 10


def test_video_frame_source_read_returns_array(tmp_path):
    video = tmp_path / "synth.mp4"
    _write_synthetic_video(video, n_frames=3)

    src = VideoFrameSource(str(video))
    ref = next(iter(src))
    img = src.read(ref)
    assert img is not None
    assert img.ndim == 3 and img.shape[2] == 3
```

- [ ] **Step 2: Run the test and confirm import-error failure**

Run: `python -m pytest tests/test_al_frame_source.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'hydra_suite.data.al'`.

- [ ] **Step 3: Create the package and implement `frame_source.py`**

`src/hydra_suite/data/al/__init__.py`:
```python
"""Active learning core: frame sources, candidate pool, signals, acquisition."""
from .frame_source import FrameRef, FrameSource, VideoFrameSource

__all__ = ["FrameRef", "FrameSource", "VideoFrameSource"]
```

`src/hydra_suite/data/al/frame_source.py`:
```python
"""Frame-source adapters for active learning pipelines."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Protocol

import cv2
import numpy as np


@dataclass(frozen=True)
class FrameRef:
    """Reference to one candidate frame within a source."""

    source_id: str
    frame_id: int
    path: str | None = None


class FrameSource(Protocol):
    """Stream of FrameRefs with random-access read."""

    def __iter__(self) -> Iterator[FrameRef]: ...

    def read(self, ref: FrameRef) -> np.ndarray | None: ...

    def length(self) -> int: ...


class VideoFrameSource:
    """FrameSource backed by a video file."""

    def __init__(self, video_path: str, stride: int = 1) -> None:
        if stride < 1:
            raise ValueError("stride must be >= 1")
        self._video_path = video_path
        self._stride = stride
        self._source_id = f"video:{Path(video_path).name}"
        cap = cv2.VideoCapture(video_path)
        try:
            self._n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            cap.release()

    def __iter__(self) -> Iterator[FrameRef]:
        for fid in range(0, self._n_frames, self._stride):
            yield FrameRef(source_id=self._source_id, frame_id=fid, path=None)

    def read(self, ref: FrameRef) -> np.ndarray | None:
        cap = cv2.VideoCapture(self._video_path)
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, ref.frame_id)
            ok, frame = cap.read()
            return frame if ok else None
        finally:
            cap.release()

    def length(self) -> int:
        return self._n_frames
```

- [ ] **Step 4: Run the test and confirm pass**

Run: `python -m pytest tests/test_al_frame_source.py -v`
Expected: 2 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/data/al/__init__.py src/hydra_suite/data/al/frame_source.py tests/test_al_frame_source.py
git commit -m "feat(data/al): scaffold AL package with FrameSource Protocol and VideoFrameSource"
```

---

## Task 2 — `ImageFolderFrameSource` + `DetectKitProjectSource`

**Files:**
- Modify: `src/hydra_suite/data/al/frame_source.py`
- Modify: `src/hydra_suite/data/al/__init__.py`
- Modify: `tests/test_al_frame_source.py`

- [ ] **Step 1: Add failing tests for the two new sources**

Append to `tests/test_al_frame_source.py`:
```python
def test_image_folder_frame_source(tmp_path):
    from hydra_suite.data.al.frame_source import ImageFolderFrameSource

    for i, color in enumerate([10, 50, 90, 130]):
        cv2.imwrite(str(tmp_path / f"img_{i:03d}.png"), np.full((8, 8, 3), color, np.uint8))
    (tmp_path / "ignored.txt").write_text("not an image")

    src = ImageFolderFrameSource(str(tmp_path))
    refs = list(src)
    assert len(refs) == 4
    assert [r.frame_id for r in refs] == [0, 1, 2, 3]
    assert all(r.path and r.path.endswith(".png") for r in refs)
    img = src.read(refs[0])
    assert img is not None and img.shape == (8, 8, 3)
    assert src.length() == 4


def test_detectkit_project_source_skips_labeled(tmp_path):
    from hydra_suite.data.al.frame_source import DetectKitProjectSource

    src_dir = tmp_path / "src1"
    (src_dir / "images").mkdir(parents=True)
    (src_dir / "labels").mkdir(parents=True)
    for i in range(3):
        cv2.imwrite(str(src_dir / "images" / f"f_{i}.jpg"), np.zeros((4, 4, 3), np.uint8))
    # Label only image 1
    (src_dir / "labels" / "f_1.txt").write_text("0 0.5 0.5 0.6 0.5 0.6 0.6 0.5 0.6\n")

    class _SrcStub:
        def __init__(self, path, name):
            self.path = path
            self.name = name

    class _ProjStub:
        sources = [_SrcStub(str(src_dir), "src1")]

    src = DetectKitProjectSource(_ProjStub(), only_unlabeled=True)
    refs = list(src)
    names = sorted(Path(r.path).stem for r in refs if r.path)
    assert names == ["f_0", "f_2"]
```

- [ ] **Step 2: Run tests and confirm import failure**

Run: `python -m pytest tests/test_al_frame_source.py -v`
Expected: 2 new tests FAIL with ImportError.

- [ ] **Step 3: Implement both adapters**

Append to `src/hydra_suite/data/al/frame_source.py`:
```python
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


class ImageFolderFrameSource:
    """FrameSource backed by a directory of image files."""

    def __init__(self, folder: str) -> None:
        self._folder = Path(folder)
        self._paths: list[Path] = sorted(
            p for p in self._folder.iterdir()
            if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
        )
        self._source_id = f"folder:{self._folder.name}"

    def __iter__(self) -> Iterator[FrameRef]:
        for idx, p in enumerate(self._paths):
            yield FrameRef(source_id=self._source_id, frame_id=idx, path=str(p))

    def read(self, ref: FrameRef) -> np.ndarray | None:
        if ref.path is None:
            return None
        img = cv2.imread(ref.path)
        return img if img is not None else None

    def length(self) -> int:
        return len(self._paths)


class DetectKitProjectSource:
    """FrameSource backed by all sources in a DetectKitProject.

    `only_unlabeled=True` skips images that have a corresponding non-empty `.txt`
    label file in the source's `labels/` directory.
    """

    def __init__(self, project, only_unlabeled: bool = True) -> None:
        self._only_unlabeled = only_unlabeled
        self._items: list[tuple[str, Path]] = []  # (source_id, image path)
        for src in getattr(project, "sources", []):
            root = Path(src.path)
            images_dir = root / "images"
            labels_dir = root / "labels"
            if not images_dir.is_dir():
                continue
            for img_path in sorted(images_dir.iterdir()):
                if img_path.suffix.lower() not in _IMAGE_EXTS:
                    continue
                if only_unlabeled:
                    label_path = labels_dir / (img_path.stem + ".txt")
                    if label_path.is_file() and label_path.stat().st_size > 0:
                        continue
                self._items.append((f"project:{src.name}", img_path))

    def __iter__(self) -> Iterator[FrameRef]:
        for idx, (sid, p) in enumerate(self._items):
            yield FrameRef(source_id=sid, frame_id=idx, path=str(p))

    def read(self, ref: FrameRef) -> np.ndarray | None:
        if ref.path is None:
            return None
        return cv2.imread(ref.path)

    def length(self) -> int:
        return len(self._items)
```

Update `src/hydra_suite/data/al/__init__.py` to also re-export `ImageFolderFrameSource` and `DetectKitProjectSource`.

- [ ] **Step 4: Run tests, confirm pass**

Run: `python -m pytest tests/test_al_frame_source.py -v`
Expected: 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/data/al/frame_source.py src/hydra_suite/data/al/__init__.py tests/test_al_frame_source.py
git commit -m "feat(data/al): add ImageFolder and DetectKitProject frame sources"
```

---

## Task 3 — `candidate_pool.build_candidate_pool` (FilterKit-backed dedup)

**Files:**
- Create: `src/hydra_suite/data/al/candidate_pool.py`
- Create: `tests/test_al_candidate_pool.py`
- Modify: `src/hydra_suite/data/al/__init__.py`

- [ ] **Step 1: Write the failing test**

`tests/test_al_candidate_pool.py`:
```python
"""Tests for hydra_suite.data.al.candidate_pool."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from hydra_suite.data.al.candidate_pool import (
    CandidatePoolConfig,
    build_candidate_pool,
)
from hydra_suite.data.al.frame_source import ImageFolderFrameSource


def _make_dataset(tmp_path, n_unique: int, n_dupes: int) -> ImageFolderFrameSource:
    rng = np.random.default_rng(0)
    idx = 0
    for _ in range(n_unique):
        img = rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)
        cv2.imwrite(str(tmp_path / f"img_{idx:04d}.png"), img)
        idx += 1
        for _ in range(n_dupes):
            cv2.imwrite(str(tmp_path / f"img_{idx:04d}.png"), img)
            idx += 1
    return ImageFolderFrameSource(str(tmp_path))


def test_candidate_pool_drops_perceptual_duplicates(tmp_path):
    src = _make_dataset(tmp_path, n_unique=4, n_dupes=2)
    cfg = CandidatePoolConfig(dedup_method="phash", dedup_threshold=4)

    refs = build_candidate_pool(src, cfg)

    assert 4 <= len(refs) <= 6  # 4 unique kept; minor dedup bleed allowed
    assert len(refs) < src.length()


def test_candidate_pool_respects_max_candidates(tmp_path):
    src = _make_dataset(tmp_path, n_unique=10, n_dupes=0)
    cfg = CandidatePoolConfig(dedup_method="none", max_candidates=3)

    refs = build_candidate_pool(src, cfg)
    assert len(refs) == 3


def test_candidate_pool_no_dedup_passthrough(tmp_path):
    src = _make_dataset(tmp_path, n_unique=5, n_dupes=0)
    cfg = CandidatePoolConfig(dedup_method="none")

    refs = build_candidate_pool(src, cfg)
    assert len(refs) == 5
```

- [ ] **Step 2: Run tests, confirm import failure**

Run: `python -m pytest tests/test_al_candidate_pool.py -v`
Expected: FAIL with ImportError.

- [ ] **Step 3: Implement candidate pool**

`src/hydra_suite/data/al/candidate_pool.py`:
```python
"""Candidate-pool construction backed by FilterKit dedup primitives."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from hydra_suite.filterkit.core import FilterKitCore

from .frame_source import FrameRef, FrameSource

DedupMethod = Literal["phash", "ahash", "dhash", "histogram", "none"]


@dataclass
class CandidatePoolConfig:
    """Configuration for `build_candidate_pool`."""

    dedup_method: DedupMethod = "phash"
    dedup_threshold: int = 8  # Hamming for hashes; bins for histogram
    max_candidates: int | None = None


def build_candidate_pool(
    source: FrameSource,
    cfg: CandidatePoolConfig,
) -> list[FrameRef]:
    """Return a deduplicated, optionally capped list of candidate FrameRefs.

    Iterates `source`, computes the configured perceptual signature for each
    frame, and keeps only frames whose signature is sufficiently distinct from
    all previously-kept frames.
    """
    fk = FilterKitCore()
    kept: list[FrameRef] = []
    kept_signatures: list = []

    for ref in source:
        if cfg.max_candidates is not None and len(kept) >= cfg.max_candidates:
            break

        if cfg.dedup_method == "none":
            kept.append(ref)
            continue

        img = source.read(ref)
        if img is None:
            continue
        sig = fk.compute_signature(img, method=cfg.dedup_method)

        is_dup = any(
            fk.is_duplicate(sig, prev, cfg.dedup_threshold, cfg.dedup_method)
            for prev in kept_signatures
        )
        if not is_dup:
            kept.append(ref)
            kept_signatures.append(sig)

    return kept
```

Update `src/hydra_suite/data/al/__init__.py` to also export `CandidatePoolConfig`, `build_candidate_pool`.

- [ ] **Step 4: Run tests, confirm pass**

Run: `python -m pytest tests/test_al_candidate_pool.py -v`
Expected: 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/data/al/candidate_pool.py src/hydra_suite/data/al/__init__.py tests/test_al_candidate_pool.py
git commit -m "feat(data/al): add FilterKit-backed candidate pool builder"
```

---

## Task 4 — `signals.ALSignals` + `score_uncertainty` + `score_count_deviation` + `score_crowd`

**Files:**
- Create: `src/hydra_suite/data/al/signals.py`
- Create: `tests/test_al_signals.py`
- Modify: `src/hydra_suite/data/al/__init__.py`

- [ ] **Step 1: Write failing tests**

`tests/test_al_signals.py`:
```python
"""Tests for hydra_suite.data.al.signals."""
from __future__ import annotations

import math

import numpy as np
import pytest

from hydra_suite.data.al.signals import (
    ALSignals,
    score_count_deviation,
    score_crowd,
    score_uncertainty,
)


def test_alsignals_defaults():
    s = ALSignals(frame_id=7)
    assert s.frame_id == 7
    assert s.n_detections == 0
    assert math.isnan(s.mean_confidence)
    assert s.extras == {}


def test_score_uncertainty_high_confidence_yields_high_margin():
    mean_conf, margin = score_uncertainty([0.95, 0.92, 0.97], conf_floor=0.5)
    assert mean_conf > 0.9
    assert margin > 0.4  # well above the floor


def test_score_uncertainty_low_confidence_yields_low_margin():
    mean_conf, margin = score_uncertainty([0.4, 0.45, 0.55], conf_floor=0.5)
    assert mean_conf < 0.55
    assert margin <= 0.05


def test_score_uncertainty_empty_returns_nan_zero():
    mean_conf, margin = score_uncertainty([], conf_floor=0.5)
    assert math.isnan(mean_conf)
    assert margin == 0.0


def test_score_count_deviation():
    assert score_count_deviation(4, expected=4) == 0.0
    assert score_count_deviation(0, expected=4) == 1.0
    assert score_count_deviation(2, expected=4) == 0.5
    assert score_count_deviation(8, expected=4) == 1.0  # clipped
    assert score_count_deviation(3, expected=0) == 0.0  # no expected -> no signal


def test_score_crowd_no_overlap():
    boxes = [
        np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32),
        np.array([[100, 100], [110, 100], [110, 110], [100, 110]], dtype=np.float32),
    ]
    crowd, edge = score_crowd(boxes, frame_shape=(200, 200))
    assert crowd == 0.0
    assert edge == 0.0


def test_score_crowd_full_overlap():
    box = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32)
    crowd, edge = score_crowd([box, box.copy()], frame_shape=(200, 200))
    assert crowd > 0.9
```

- [ ] **Step 2: Run tests, confirm import failure**

Run: `python -m pytest tests/test_al_signals.py -v`
Expected: FAIL with ImportError.

- [ ] **Step 3: Implement signals (NMS-instability is in Task 5)**

`src/hydra_suite/data/al/signals.py`:
```python
"""Per-frame active-learning signals."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import combinations
from typing import Sequence

import cv2
import numpy as np


@dataclass
class ALSignals:
    """Per-frame signal record consumed by the acquisition selector."""

    frame_id: int
    n_detections: int = 0
    mean_confidence: float = float("nan")
    margin: float = 0.0
    nms_instability: float = 0.0
    count_deviation: float = 0.0
    crowd_score: float = 0.0
    edge_score: float = 0.0
    extras: dict[str, float] = field(default_factory=dict)


def score_uncertainty(
    confidences: Sequence[float],
    conf_floor: float = 0.5,
) -> tuple[float, float]:
    """Return (mean_confidence, margin).

    `margin` is `min(c) - conf_floor` clipped to [0, 1]. A small/zero margin
    indicates at least one detection sits at or below the floor — a good
    AL signal.
    """
    valid = [float(c) for c in confidences if c is not None and not math.isnan(c)]
    if not valid:
        return float("nan"), 0.0
    mean_conf = float(np.mean(valid))
    raw_margin = float(min(valid) - conf_floor)
    margin = float(max(0.0, min(1.0, raw_margin)))
    return mean_conf, margin


def score_count_deviation(n: int, expected: int) -> float:
    """Return |n - expected| / max(expected, 1), clipped to [0, 1]. 0 if expected<=0."""
    if expected <= 0:
        return 0.0
    return float(min(1.0, abs(n - expected) / float(expected)))


def _polygon_overlap_ratio(corners_a: np.ndarray, corners_b: np.ndarray) -> float:
    """Intersection area divided by smaller polygon area, clipped to [0, 1]."""
    poly_a = np.asarray(corners_a, dtype=np.float32).reshape(-1, 1, 2)
    poly_b = np.asarray(corners_b, dtype=np.float32).reshape(-1, 1, 2)
    if len(poly_a) < 3 or len(poly_b) < 3:
        return 0.0
    area_a = abs(float(cv2.contourArea(poly_a)))
    area_b = abs(float(cv2.contourArea(poly_b)))
    if area_a <= 0.0 or area_b <= 0.0:
        return 0.0
    try:
        inter, _ = cv2.intersectConvexConvex(poly_a, poly_b)
    except cv2.error:
        return 0.0
    if inter <= 0.0:
        return 0.0
    return float(max(0.0, min(1.0, inter / max(min(area_a, area_b), 1e-6))))


def score_crowd(
    obb_corners: Sequence[np.ndarray],
    frame_shape: tuple[int, int],
) -> tuple[float, float]:
    """Return (crowd_score, edge_score).

    `crowd_score` = max pairwise polygon-overlap ratio across all detection pairs.
    `edge_score`  = max box-corner proximity to frame border, normalized to [0, 1]
                    (0 = far inside frame, 1 = touching edge).
    """
    if len(obb_corners) < 1:
        return 0.0, 0.0
    h, w = int(frame_shape[0]), int(frame_shape[1])

    crowd = 0.0
    if len(obb_corners) >= 2:
        for a, b in combinations(obb_corners, 2):
            crowd = max(crowd, _polygon_overlap_ratio(a, b))

    edge = 0.0
    for box in obb_corners:
        arr = np.asarray(box, dtype=np.float32).reshape(-1, 2)
        if arr.size == 0:
            continue
        dx = np.minimum(arr[:, 0], w - arr[:, 0])
        dy = np.minimum(arr[:, 1], h - arr[:, 1])
        margin_px = float(np.min(np.minimum(dx, dy)))
        ref = max(min(w, h) * 0.10, 1.0)  # within 10% of min dim counts as edge-y
        edge_norm = max(0.0, 1.0 - margin_px / ref)
        edge = max(edge, edge_norm)

    return float(crowd), float(edge)
```

Update `src/hydra_suite/data/al/__init__.py` to also export `ALSignals`, `score_uncertainty`, `score_count_deviation`, `score_crowd`.

- [ ] **Step 4: Run tests, confirm pass**

Run: `python -m pytest tests/test_al_signals.py -v`
Expected: 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/data/al/signals.py src/hydra_suite/data/al/__init__.py tests/test_al_signals.py
git commit -m "feat(data/al): add ALSignals dataclass and uncertainty/count/crowd scorers"
```

---

## Task 5 — `signals.score_nms_instability`

**Files:**
- Modify: `src/hydra_suite/data/al/signals.py`
- Modify: `tests/test_al_signals.py`
- Modify: `src/hydra_suite/data/al/__init__.py`

- [ ] **Step 1: Add failing test**

Append to `tests/test_al_signals.py`:
```python
def test_nms_instability_stable_detector_returns_low_score():
    from hydra_suite.data.al.signals import score_nms_instability

    base = [(10, 10, 8, 4, 0.0, 0.95),
            (50, 50, 8, 4, 0.0, 0.93),
            (90, 90, 8, 4, 0.0, 0.97)]

    def detector(_frame, conf, iou):
        return [d for d in base if d[5] >= conf]

    score = score_nms_instability(
        frame=np.zeros((100, 100, 3), np.uint8),
        detector_fn=detector,
        base_conf=0.5,
        base_iou=0.7,
    )
    assert score < 0.05


def test_nms_instability_unstable_detector_returns_high_score():
    from hydra_suite.data.al.signals import score_nms_instability

    def detector(_frame, conf, iou):
        if conf < 0.4:
            return [(10, 10, 8, 4, 0.0, 0.45),
                    (30, 30, 8, 4, 0.0, 0.42),
                    (60, 60, 8, 4, 0.0, 0.95)]
        return [(60, 60, 8, 4, 0.0, 0.95)]

    score = score_nms_instability(
        frame=np.zeros((100, 100, 3), np.uint8),
        detector_fn=detector,
        base_conf=0.5,
        base_iou=0.7,
    )
    assert score > 0.4
```

- [ ] **Step 2: Run tests, confirm import failure**

Run: `python -m pytest tests/test_al_signals.py -v`
Expected: 2 new tests FAIL with ImportError.

- [ ] **Step 3: Implement `score_nms_instability`**

Append to `src/hydra_suite/data/al/signals.py`:
```python
from typing import Callable

Detection = tuple  # (cx, cy, w, h, theta, conf)


def _set_iou_greedy(
    set_a: Sequence[Detection],
    set_b: Sequence[Detection],
    match_distance: float = 12.0,
) -> float:
    """Approximate set IoU via greedy center-distance matching."""
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    used_b: set[int] = set()
    matched = 0
    for det_a in set_a:
        best_idx, best_dist = -1, math.inf
        for j, det_b in enumerate(set_b):
            if j in used_b:
                continue
            dist = math.hypot(det_a[0] - det_b[0], det_a[1] - det_b[1])
            if dist < best_dist:
                best_dist, best_idx = dist, j
        if best_idx >= 0 and best_dist <= match_distance:
            used_b.add(best_idx)
            matched += 1
    union = len(set_a) + len(set_b) - matched
    return matched / max(union, 1)


def score_nms_instability(
    frame: np.ndarray,
    detector_fn: Callable[[np.ndarray, float, float], Sequence[Detection]],
    base_conf: float,
    base_iou: float,
) -> float:
    """Return 1 - mean(set_IoU) across two (conf, iou) perturbations.

    Higher score = detection set changes meaningfully under small NMS-threshold
    shifts -> model is unstable on this frame -> good AL pick.
    """
    base_set = list(detector_fn(frame, base_conf, base_iou))
    perturbations = [
        (max(base_conf * 0.7, 0.01), base_iou),
        (base_conf, min(base_iou * 1.3, 0.95)),
    ]
    ious = []
    for conf, iou in perturbations:
        ious.append(_set_iou_greedy(base_set, list(detector_fn(frame, conf, iou))))
    if not ious:
        return 0.0
    return float(1.0 - sum(ious) / len(ious))
```

Update `src/hydra_suite/data/al/__init__.py` to export `score_nms_instability`.

- [ ] **Step 4: Run tests, confirm pass**

Run: `python -m pytest tests/test_al_signals.py -v`
Expected: 9 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/data/al/signals.py src/hydra_suite/data/al/__init__.py tests/test_al_signals.py
git commit -m "feat(data/al): add NMS-instability scorer with greedy set IoU"
```

---

## Task 6 — `acquisition.AcquisitionWeights` + `PRESETS` + `select`

**Files:**
- Create: `src/hydra_suite/data/al/acquisition.py`
- Create: `tests/test_al_acquisition.py`
- Modify: `src/hydra_suite/data/al/__init__.py`

- [ ] **Step 1: Write failing tests**

`tests/test_al_acquisition.py`:
```python
"""Tests for hydra_suite.data.al.acquisition."""
from __future__ import annotations

import numpy as np
import pytest

from hydra_suite.data.al.acquisition import (
    AcquisitionWeights,
    PRESETS,
    select,
)
from hydra_suite.data.al.signals import ALSignals


def _signal(frame_id: int, **kwargs) -> ALSignals:
    return ALSignals(frame_id=frame_id, **kwargs)


def test_presets_are_normalized():
    for name, w in PRESETS.items():
        total = (
            w.uncertainty + w.nms_instability + w.count + w.crowd + w.edge
            + w.assignment + w.track_loss + w.position_uncertainty
        )
        assert abs(total - 1.0) < 1e-6, f"preset {name} weights sum to {total}"


def test_select_picks_highest_score():
    signals = [
        _signal(0, mean_confidence=0.95, margin=0.4, count_deviation=0.0, crowd_score=0.0),
        _signal(100, mean_confidence=0.4, margin=0.0, count_deviation=0.5, crowd_score=0.7),
        _signal(200, mean_confidence=0.85, margin=0.3, count_deviation=0.0, crowd_score=0.2),
    ]
    picks = select(signals, weights=PRESETS["balanced"], k=1, diversity_window=0,
                   probabilistic=False)
    assert picks == [100]


def test_select_diversity_window_blocks_neighbors():
    signals = [_signal(i, mean_confidence=0.9 - 0.01 * i) for i in range(20)]
    picks = select(signals, weights=PRESETS["balanced"], k=3, diversity_window=10,
                   probabilistic=False)
    assert len(picks) == 3
    diffs = [abs(a - b) for a in picks for b in picks if a \!= b]
    assert min(diffs) >= 10


def test_select_returns_at_most_k():
    signals = [_signal(i) for i in range(5)]
    picks = select(signals, weights=PRESETS["balanced"], k=20, diversity_window=0,
                   probabilistic=False)
    assert len(picks) <= 5


def test_select_probabilistic_deterministic_with_seed():
    signals = [_signal(i, mean_confidence=0.5 + 0.01 * i) for i in range(20)]
    rng_a = np.random.default_rng(42)
    rng_b = np.random.default_rng(42)
    a = select(signals, weights=PRESETS["balanced"], k=5, diversity_window=2,
               probabilistic=True, rng=rng_a)
    b = select(signals, weights=PRESETS["balanced"], k=5, diversity_window=2,
               probabilistic=True, rng=rng_b)
    assert a == b


def test_select_min_score_filters_out_low_scoring_frames():
    signals = [_signal(i, mean_confidence=0.99 - 0.001 * i) for i in range(10)]
    picks = select(signals, weights=PRESETS["balanced"], k=10, diversity_window=0,
                   probabilistic=False, min_score=0.5)
    # All composite scores will be small for these very high-confidence signals.
    assert picks == []
```

- [ ] **Step 2: Run tests, confirm import failure**

Run: `python -m pytest tests/test_al_acquisition.py -v`
Expected: FAIL with ImportError.

- [ ] **Step 3: Implement `acquisition.py`**

`src/hydra_suite/data/al/acquisition.py`:
```python
"""Active-learning frame acquisition: weighted ranking with diversity guard."""
from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Sequence

import numpy as np

from .signals import ALSignals


@dataclass
class AcquisitionWeights:
    """Per-signal weights. Auto-normalized to sum to 1.0 at use."""

    uncertainty: float = 0.40
    nms_instability: float = 0.20
    count: float = 0.20
    crowd: float = 0.15
    edge: float = 0.05
    # Tracker-only extras (zero on detector-only paths)
    assignment: float = 0.0
    track_loss: float = 0.0
    position_uncertainty: float = 0.0

    def normalized(self) -> "AcquisitionWeights":
        total = sum(getattr(self, f.name) for f in fields(self))
        if total <= 0:
            return AcquisitionWeights()
        return AcquisitionWeights(
            **{f.name: getattr(self, f.name) / total for f in fields(self)}
        )


PRESETS: dict[str, AcquisitionWeights] = {
    "balanced": AcquisitionWeights(
        uncertainty=0.40, nms_instability=0.20, count=0.20, crowd=0.15, edge=0.05,
    ),
    "uncertainty_heavy": AcquisitionWeights(
        uncertainty=0.55, nms_instability=0.25, count=0.10, crowd=0.05, edge=0.05,
    ),
    "exploration_heavy": AcquisitionWeights(
        uncertainty=0.25, nms_instability=0.15, count=0.15, crowd=0.30, edge=0.15,
    ),
    "tracker_default": AcquisitionWeights(
        uncertainty=0.30, nms_instability=0.0, count=0.20, crowd=0.15, edge=0.05,
        assignment=0.15, track_loss=0.10, position_uncertainty=0.05,
    ),
}


def _channel_array(signals: Sequence[ALSignals], attr: str) -> np.ndarray:
    """Pull a signal channel into a numpy array, treating NaN as 0."""
    if attr in {"assignment", "track_loss", "position_uncertainty"}:
        vals = [s.extras.get(attr, 0.0) for s in signals]
    elif attr == "uncertainty":
        vals = []
        for s in signals:
            if math.isnan(s.mean_confidence):
                vals.append(0.0)
            else:
                vals.append(max(0.0, min(1.0, 1.0 - s.mean_confidence)))
    else:
        attr_name = "count_deviation" if attr == "count" else attr
        vals = [getattr(s, attr_name) for s in signals]
    arr = np.asarray(vals, dtype=np.float64)
    arr = np.where(np.isnan(arr), 0.0, arr)
    return arr


def _minmax(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return arr
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo <= 1e-12:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def _composite_score(
    signals: Sequence[ALSignals],
    weights: AcquisitionWeights,
) -> np.ndarray:
    w = weights.normalized()
    channels = {
        "uncertainty": w.uncertainty,
        "nms_instability": w.nms_instability,
        "count": w.count,
        "crowd": w.crowd,
        "edge": w.edge,
        "assignment": w.assignment,
        "track_loss": w.track_loss,
        "position_uncertainty": w.position_uncertainty,
    }
    n = len(signals)
    score = np.zeros(n, dtype=np.float64)
    for name, weight in channels.items():
        if weight <= 0:
            continue
        score += weight * _minmax(_channel_array(signals, name))
    return score


def select(
    signals: Sequence[ALSignals],
    weights: AcquisitionWeights,
    k: int,
    diversity_window: int = 30,
    probabilistic: bool = True,
    rng: np.random.Generator | None = None,
    min_score: float = 0.0,
) -> list[int]:
    """Return up to k frame_ids from `signals`, ranked by weighted composite score.

    `diversity_window` enforces minimum frame-index spacing between picks.
    `probabilistic=True` uses rank-based sampling; False is deterministic top-K.
    `min_score` drops candidates whose composite score is below this cutoff.
    """
    if not signals or k <= 0:
        return []

    score = _composite_score(signals, weights)

    keep_mask = score >= float(min_score)
    indices = [int(i) for i in np.argsort(-score) if keep_mask[i]]
    if not indices:
        return []
    sorted_ids = [int(signals[i].frame_id) for i in indices]
    rng = rng or np.random.default_rng()

    picks: list[int] = []

    def _diverse(fid: int) -> bool:
        return all(abs(fid - p) >= diversity_window for p in picks)

    if not probabilistic:
        for fid in sorted_ids:
            if len(picks) >= k:
                break
            if _diverse(fid):
                picks.append(fid)
        return picks

    candidates = sorted_ids[:]
    while len(picks) < k and candidates:
        weights_arr = np.array([1.0 / (i + 1) for i in range(len(candidates))])
        weights_arr /= weights_arr.sum()
        chosen_idx = int(rng.choice(len(candidates), p=weights_arr))
        fid = candidates[chosen_idx]
        if _diverse(fid):
            picks.append(fid)
        candidates = [c for c in candidates if abs(c - fid) >= diversity_window]
    return picks
```

Update `src/hydra_suite/data/al/__init__.py` to also export `AcquisitionWeights`, `PRESETS`, `select`.

- [ ] **Step 4: Run tests, confirm pass**

Run: `python -m pytest tests/test_al_acquisition.py -v`
Expected: 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/data/al/acquisition.py src/hydra_suite/data/al/__init__.py tests/test_al_acquisition.py
git commit -m "feat(data/al): add weighted-ranking acquisition selector with diversity guard"
```

---

## Task 7 — Tracker-side refactor: `FrameQualityScorer` delegates to `data/al/`

**Files:**
- Modify: `src/hydra_suite/data/dataset_generation.py:18-360`
- Modify: `tests/test_dataset_generation.py` (regression test)

- [ ] **Step 1: Add a regression test that locks the new tracker_default preset behavior**

Append to `tests/test_dataset_generation.py`:
```python
def test_frame_quality_scorer_uses_tracker_default_preset_after_refactor():
    """After refactor, scorer routes through data/al/acquisition with tracker_default."""
    from hydra_suite.data.dataset_generation import FrameQualityScorer

    params = {
        "MAX_TARGETS": 4,
        "DATASET_CONF_THRESHOLD": 0.5,
        "REFERENCE_BODY_SIZE": 20.0,
        "METRIC_LOW_CONFIDENCE": True,
        "METRIC_COUNT_MISMATCH": True,
        "METRIC_HIGH_ASSIGNMENT_COST": True,
        "METRIC_TRACK_LOSS": True,
        "METRIC_HIGH_UNCERTAINTY": False,
        "METRIC_FRAGMENTED_DETECTIONS": False,
    }
    scorer = FrameQualityScorer(params)

    # Frame 0: clean; frame 100: bad (low conf, low count, lost tracks).
    scorer.score_frame(0, detection_data={"confidences": [0.9, 0.9, 0.9, 0.9], "count": 4},
                          tracking_data={"lost_tracks": 0})
    scorer.score_frame(100, detection_data={"confidences": [0.2, 0.3], "count": 2},
                            tracking_data={"lost_tracks": 2,
                                           "assignment_confidences": [0.3, 0.3]})

    picks = scorer.get_worst_frames(max_frames=1, diversity_window=0, probabilistic=False)
    assert picks == [100]
```

- [ ] **Step 2: Run regression test against current code, observe baseline**

Run: `python -m pytest tests/test_dataset_generation.py -v -k tracker_default_preset`
Expected: PASS (frame 100 is the obvious worst pick under any reasonable scorer). Lock this contract before refactoring.

- [ ] **Step 3: Refactor `FrameQualityScorer` and `get_worst_frames` to delegate**

Replace the body of `src/hydra_suite/data/dataset_generation.py:18-360` (the `FrameQualityScorer` class and the helpers `_clamp01`, `_detection_corners_from_dims`, `_polygon_overlap_ratio`) with the version below. Keep the export-pipeline functions (`export_dataset` and its helpers from line ~416 onward) untouched.

Add `import math` at the top if not already present.

```python
class FrameQualityScorer:
    """Tracker-side adapter that produces ALSignals and selects worst frames.

    Public API (`score_frame`, `get_worst_frames`) is preserved for callers; the
    underlying ranking now lives in `hydra_suite.data.al.acquisition`.
    """

    def __init__(self, params):
        from hydra_suite.data.al.acquisition import PRESETS, AcquisitionWeights

        self.params = params
        self.frame_signals: dict[int, "ALSignals"] = {}
        self.max_targets = params.get("MAX_TARGETS", 4)
        self.conf_threshold = params.get("DATASET_CONF_THRESHOLD", 0.5)
        self.reference_body_size = max(
            float(params.get("REFERENCE_BODY_SIZE", 20.0)), 1.0
        )

        self._enabled = {
            "uncertainty": bool(params.get("METRIC_LOW_CONFIDENCE", True)),
            "count": bool(params.get("METRIC_COUNT_MISMATCH", True)),
            "assignment": bool(params.get("METRIC_HIGH_ASSIGNMENT_COST", True)),
            "track_loss": bool(params.get("METRIC_TRACK_LOSS", True)),
            "position_uncertainty": bool(params.get("METRIC_HIGH_UNCERTAINTY", False)),
            "crowd": bool(params.get("METRIC_FRAGMENTED_DETECTIONS", True)),
        }

        base = PRESETS["tracker_default"]
        self._weights = AcquisitionWeights(
            uncertainty=base.uncertainty if self._enabled["uncertainty"] else 0.0,
            nms_instability=0.0,
            count=base.count if self._enabled["count"] else 0.0,
            crowd=base.crowd if self._enabled["crowd"] else 0.0,
            edge=base.edge,
            assignment=base.assignment if self._enabled["assignment"] else 0.0,
            track_loss=base.track_loss if self._enabled["track_loss"] else 0.0,
            position_uncertainty=(
                base.position_uncertainty if self._enabled["position_uncertainty"] else 0.0
            ),
        )

        # Backward-compat scalar map for tests that read `frame_scores`.
        from collections import defaultdict
        self.frame_scores = defaultdict(lambda: {"score": 0.0, "metrics": {}})

    def score_frame(self, frame_id, detection_data=None, tracking_data=None):
        from hydra_suite.data.al.signals import (
            ALSignals,
            score_count_deviation,
            score_crowd,
            score_uncertainty,
        )

        detection_data = detection_data or {}
        tracking_data = tracking_data or {}

        confidences = detection_data.get("confidences") or []
        mean_conf, margin = score_uncertainty(confidences, conf_floor=self.conf_threshold)

        n_dets = int(detection_data.get("count", len(confidences)))
        count_dev = score_count_deviation(n_dets, self.max_targets)

        obb_corners = self._extract_obb_corners(detection_data)
        crowd, edge = score_crowd(obb_corners, frame_shape=(1, 1)) if obb_corners else (0.0, 0.0)

        extras: dict[str, float] = {}
        ac = tracking_data.get("assignment_confidences") or []
        if ac:
            extras["assignment"] = max(0.0, 1.0 - float(np.mean(ac)))
        elif tracking_data.get("assignment_costs"):
            costs = tracking_data["assignment_costs"]
            extras["assignment"] = float(min(np.mean(costs) / 50.0, 1.0))

        lost = int(tracking_data.get("lost_tracks", 0))
        if lost > 0:
            extras["track_loss"] = float(min(lost / max(self.max_targets, 1), 1.0))

        unc = tracking_data.get("uncertainties") or []
        if unc:
            extras["position_uncertainty"] = float(min(np.mean(unc) / 50.0, 1.0))

        signal = ALSignals(
            frame_id=int(frame_id),
            n_detections=n_dets,
            mean_confidence=mean_conf,
            margin=margin,
            count_deviation=count_dev,
            crowd_score=crowd,
            edge_score=edge,
            extras=extras,
        )
        self.frame_signals[int(frame_id)] = signal

        proxy = self._score_proxy(signal)
        self.frame_scores[int(frame_id)] = {"score": proxy, "metrics": {}}
        return proxy

    @staticmethod
    def _score_proxy(signal) -> float:
        # Monotonic proxy of "challengingness" for legacy scalar callers.
        mc = signal.mean_confidence
        return float(
            (1.0 - mc if not (isinstance(mc, float) and math.isnan(mc)) else 0.0)
            + signal.count_deviation
            + signal.crowd_score
            + signal.extras.get("assignment", 0.0)
            + signal.extras.get("track_loss", 0.0)
        )

    def get_worst_frames(self, max_frames, diversity_window=30, probabilistic=True):
        from hydra_suite.data.al.acquisition import select

        signals = list(self.frame_signals.values())
        rng = np.random.default_rng() if probabilistic else None
        return select(
            signals,
            weights=self._weights,
            k=int(max_frames),
            diversity_window=int(diversity_window),
            probabilistic=bool(probabilistic),
            rng=rng,
            min_score=float(self.params.get("DATASET_MIN_SELECTION_SCORE", 0.0)),
        )

    def _extract_obb_corners(self, detection_data):
        corners = detection_data.get("obb_corners") or []
        out: list[np.ndarray] = []
        for c in corners:
            if c is None:
                continue
            arr = np.asarray(c, dtype=np.float32).reshape(-1, 2)
            if arr.shape[0] >= 3:
                out.append(arr)
        return out
```

Delete the now-orphaned helper functions in this file: `_clamp01`, `_detection_corners_from_dims`, `_polygon_overlap_ratio`. They have moved into `signals.py` (private to that module) and the export pipeline doesn't depend on the standalone copies.

- [ ] **Step 4: Run the regression test plus the rest of the existing dataset_generation suite**

Run: `python -m pytest tests/test_dataset_generation.py -v`
Expected: All tests PASS, including the new regression test. If a sub-test failure indicates the new ranking diverges from the old hard-coded weights, document the diff in the spec migration notes — the new normalized scoring is intentionally not bit-exact with the old.

- [ ] **Step 5: Run the full data-layer test slice for safety**

Run: `python -m pytest tests/test_dataset_generation.py tests/test_al_acquisition.py tests/test_al_signals.py tests/test_al_candidate_pool.py tests/test_al_frame_source.py -v`
Expected: All PASS.

- [ ] **Step 6: Commit**

```bash
git add src/hydra_suite/data/dataset_generation.py tests/test_dataset_generation.py
git commit -m "refactor(data): route FrameQualityScorer through data/al core"
```

---

## Task 8 — DetectKit AL worker (non-GUI, integration-tested)

**Files:**
- Create: `src/hydra_suite/detectkit/jobs/__init__.py`
- Create: `src/hydra_suite/detectkit/jobs/al_worker.py`
- Create: `tests/test_detectkit_al_worker.py`

- [ ] **Step 1: Write the integration test**

`tests/test_detectkit_al_worker.py`:
```python
"""Integration test for the DetectKit AL worker."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from hydra_suite.detectkit.gui.models import DetectKitProject


def _seed_image_folder(tmp_path: Path, n: int = 6) -> Path:
    folder = tmp_path / "frames"
    folder.mkdir()
    rng = np.random.default_rng(0)
    for i in range(n):
        img = rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)
        cv2.imwrite(str(folder / f"f_{i:03d}.png"), img)
    return folder


def test_al_worker_writes_seeded_labels_and_registers_source(tmp_path):
    from hydra_suite.detectkit.jobs.al_worker import (
        ALRequest,
        run_active_learning,
    )

    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    project = DetectKitProject(project_dir=project_dir, sources=[])

    folder = _seed_image_folder(tmp_path, n=6)

    def fake_detector(frame, conf, iou):
        return [
            (10, 10, 8, 4, 0.0, 0.95),
            (30, 30, 8, 4, 0.0, 0.55),
            (50, 50, 8, 4, 0.0, 0.30),
        ]

    request = ALRequest(
        input_kind="folder",
        input_path=str(folder),
        project=project,
        budget=3,
        preset="balanced",
        expected_count=2,
        detector_fn=fake_detector,
        diversity_window=0,
        probabilistic=False,
    )

    result = run_active_learning(request)

    assert result.n_picked == 3
    new_source_dir = Path(result.source_path)
    assert (new_source_dir / "images").is_dir()
    assert (new_source_dir / "labels").is_dir()
    image_files = list((new_source_dir / "images").iterdir())
    label_files = list((new_source_dir / "labels").iterdir())
    assert len(image_files) == 3
    assert len(label_files) == 3
    for lf in label_files:
        lines = lf.read_text().strip().splitlines()
        assert len(lines) == 3  # all three model predictions seeded as YOLO OBB lines

    assert any(s.path == str(new_source_dir) for s in project.sources)
```

- [ ] **Step 2: Run the test, confirm import failure**

Run: `python -m pytest tests/test_detectkit_al_worker.py -v`
Expected: FAIL with ImportError.

- [ ] **Step 3: Implement `al_worker.py` (pure pipeline + thin QThread wrapper)**

`src/hydra_suite/detectkit/jobs/__init__.py`:
```python
"""DetectKit background workers."""
```

`src/hydra_suite/detectkit/jobs/al_worker.py`:
```python
"""Active learning worker for DetectKit projects."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Literal, Sequence

import cv2
import numpy as np
from PySide6.QtCore import Signal

from hydra_suite.data.al.acquisition import (
    PRESETS,
    AcquisitionWeights,
    select,
)
from hydra_suite.data.al.candidate_pool import (
    CandidatePoolConfig,
    build_candidate_pool,
)
from hydra_suite.data.al.frame_source import (
    DetectKitProjectSource,
    FrameSource,
    ImageFolderFrameSource,
    VideoFrameSource,
)
from hydra_suite.data.al.signals import (
    ALSignals,
    score_count_deviation,
    score_crowd,
    score_nms_instability,
    score_uncertainty,
)
from hydra_suite.detectkit.gui.models import DetectKitProject, OBBSource
from hydra_suite.widgets.workers import BaseWorker

logger = logging.getLogger(__name__)

Detection = tuple  # (cx, cy, w, h, theta, conf)
DetectorFn = Callable[[np.ndarray, float, float], Sequence[Detection]]


@dataclass
class ALRequest:
    """User input for one active-learning round."""

    input_kind: Literal["video", "folder", "project"]
    input_path: str
    project: DetectKitProject
    budget: int
    preset: str = "balanced"
    weights_override: AcquisitionWeights | None = None
    expected_count: int = 0
    detector_fn: DetectorFn | None = None
    diversity_window: int = 30
    probabilistic: bool = True
    candidate_pool: CandidatePoolConfig = field(default_factory=CandidatePoolConfig)
    base_conf: float = 0.25
    base_iou: float = 0.7


@dataclass
class ALResult:
    """Outcome of one AL round."""

    source_path: str
    n_picked: int
    selected_frames: list[int]


def _build_frame_source(req: ALRequest) -> FrameSource:
    if req.input_kind == "video":
        return VideoFrameSource(req.input_path)
    if req.input_kind == "folder":
        return ImageFolderFrameSource(req.input_path)
    if req.input_kind == "project":
        return DetectKitProjectSource(req.project, only_unlabeled=True)
    raise ValueError(f"unknown input_kind: {req.input_kind}")


def _detection_corners(cx, cy, ww, hh, theta) -> np.ndarray:
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    local = np.array([
        [-ww/2, -hh/2], [ww/2, -hh/2], [ww/2, hh/2], [-ww/2, hh/2],
    ], dtype=np.float32)
    rot = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float32)
    return local @ rot.T + np.array([cx, cy], dtype=np.float32)


def _frame_signals(
    frame: np.ndarray,
    frame_id: int,
    detector_fn: DetectorFn,
    expected_count: int,
    base_conf: float,
    base_iou: float,
) -> tuple[ALSignals, list]:
    detections = list(detector_fn(frame, base_conf, base_iou))
    confidences = [d[5] for d in detections]
    mean_conf, margin = score_uncertainty(confidences, conf_floor=base_conf)
    count_dev = score_count_deviation(len(detections), expected_count)

    h, w = frame.shape[:2]
    obb_corners = [_detection_corners(*d[:5]) for d in detections]
    crowd, edge = score_crowd(obb_corners, frame_shape=(h, w))

    nms = score_nms_instability(frame, detector_fn, base_conf=base_conf, base_iou=base_iou)

    signal = ALSignals(
        frame_id=frame_id,
        n_detections=len(detections),
        mean_confidence=mean_conf,
        margin=margin,
        nms_instability=nms,
        count_deviation=count_dev,
        crowd_score=crowd,
        edge_score=edge,
    )
    return signal, detections


def _write_yolo_obb_label(path: Path, detections: list, frame_size: tuple[int, int]) -> None:
    h, w = frame_size
    with path.open("w") as fp:
        for cx, cy, ww, hh, theta, _ in detections:
            corners = _detection_corners(cx, cy, ww, hh, theta)
            corners[:, 0] = np.clip(corners[:, 0] / w, 0.0, 1.0)
            corners[:, 1] = np.clip(corners[:, 1] / h, 0.0, 1.0)
            line = "0 " + " ".join(f"{v:.6f}" for v in corners.flatten()) + "\n"
            fp.write(line)


def run_active_learning(req: ALRequest, progress: Callable[[int, str], None] | None = None) -> ALResult:
    """Execute one AL round end-to-end. Pure function for testability."""
    if req.detector_fn is None:
        raise ValueError("ALRequest.detector_fn must be set (model must be loaded by caller)")

    weights = req.weights_override or PRESETS.get(req.preset, PRESETS["balanced"])

    if progress:
        progress(5, "Building candidate pool...")
    source = _build_frame_source(req)
    candidates = build_candidate_pool(source, req.candidate_pool)
    if not candidates:
        raise RuntimeError("0 candidates after FilterKit dedup; relax threshold or stride.")

    if progress:
        progress(20, f"Scoring {len(candidates)} candidates...")
    signals: list[ALSignals] = []
    detections_by_id: dict[int, tuple[np.ndarray, list]] = {}
    for i, ref in enumerate(candidates):
        img = source.read(ref)
        if img is None:
            continue
        sig, dets = _frame_signals(
            img, ref.frame_id, req.detector_fn,
            req.expected_count, req.base_conf, req.base_iou,
        )
        signals.append(sig)
        detections_by_id[ref.frame_id] = (img, dets)
        if progress and i % 10 == 0:
            progress(20 + int(60 * i / max(len(candidates), 1)),
                     f"Scoring {i}/{len(candidates)}")

    if progress:
        progress(85, "Selecting top-K frames...")
    rng = np.random.default_rng()
    picked_ids = select(
        signals,
        weights=weights,
        k=req.budget,
        diversity_window=req.diversity_window,
        probabilistic=req.probabilistic,
        rng=rng if req.probabilistic else None,
    )

    if progress:
        progress(95, "Writing dataset...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    proj_dir = Path(req.project.project_dir)
    source_root = proj_dir / "sources" / f"al_round_{timestamp}"
    images_dir = source_root / "images"
    labels_dir = source_root / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    for fid in picked_ids:
        img, dets = detections_by_id[fid]
        img_path = images_dir / f"f_{fid:06d}.jpg"
        cv2.imwrite(str(img_path), img)
        _write_yolo_obb_label(labels_dir / f"f_{fid:06d}.txt", dets, frame_size=img.shape[:2])

    (source_root / "classes.txt").write_text(req.project.class_name + "\n")

    new_source = OBBSource(
        path=str(source_root),
        name=f"al_round_{timestamp}",
        validated=False,
        original_path=req.input_path,
        source_kind="detectkit_al",
        imported=True,
    )
    req.project.sources.append(new_source)

    if progress:
        progress(100, "Active learning complete")

    return ALResult(
        source_path=str(source_root),
        n_picked=len(picked_ids),
        selected_frames=picked_ids,
    )


class ALWorker(BaseWorker):
    """QThread wrapper around run_active_learning."""

    progress_signal = Signal(int, str)
    finished_signal = Signal(str, int, list)
    error_signal = Signal(str)

    def __init__(self, request: ALRequest):
        super().__init__()
        self._request = request

    def execute(self):
        try:
            def cb(pct, msg):
                if not self._should_stop():
                    self.progress_signal.emit(int(pct), str(msg))

            result = run_active_learning(self._request, progress=cb)
            if not self._should_stop():
                self.finished_signal.emit(
                    result.source_path, result.n_picked, list(result.selected_frames)
                )
        except Exception as exc:
            logger.exception("AL worker failed")
            self.error_signal.emit(str(exc))

    def _should_stop(self) -> bool:
        return bool(self.isInterruptionRequested())
```

- [ ] **Step 4: Run the integration test, confirm pass**

Run: `python -m pytest tests/test_detectkit_al_worker.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/detectkit/jobs/__init__.py src/hydra_suite/detectkit/jobs/al_worker.py tests/test_detectkit_al_worker.py
git commit -m "feat(detectkit): add active-learning worker with FilterKit + acquisition"
```

---

## Task 9 — DetectKit AL dialog + main_window wiring

**Files:**
- Create: `src/hydra_suite/detectkit/gui/dialogs/active_learning.py`
- Create: `tests/test_detectkit_al_dialog.py`
- Modify: `src/hydra_suite/detectkit/gui/main_window.py` (add menu action)

- [ ] **Step 1: Write the dialog smoke test**

`tests/test_detectkit_al_dialog.py`:
```python
"""Smoke test for the DetectKit active-learning dialog."""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication

from hydra_suite.detectkit.gui.dialogs.active_learning import ActiveLearningDialog
from hydra_suite.detectkit.gui.models import DetectKitProject


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def test_dialog_constructs_with_project(qapp, tmp_path):
    project = DetectKitProject(project_dir=tmp_path)
    dlg = ActiveLearningDialog(project=project)
    assert dlg is not None
    presets = [dlg.preset_combo.itemText(i) for i in range(dlg.preset_combo.count())]
    assert "balanced" in presets
    assert "uncertainty_heavy" in presets
    assert "exploration_heavy" in presets
    dlg.close()


def test_dialog_disables_run_until_inputs_valid(qapp, tmp_path):
    project = DetectKitProject(project_dir=tmp_path)
    dlg = ActiveLearningDialog(project=project)
    assert not dlg.run_button.isEnabled()
    dlg.close()
```

- [ ] **Step 2: Run the test, confirm import failure**

Run: `python -m pytest tests/test_detectkit_al_dialog.py -v`
Expected: FAIL with ImportError.

- [ ] **Step 3: Implement the dialog**

`src/hydra_suite/detectkit/gui/dialogs/active_learning.py`:
```python
"""Modal dialog for running an active-learning round in DetectKit."""
from __future__ import annotations

from typing import Callable

from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from hydra_suite.data.al.acquisition import PRESETS
from hydra_suite.detectkit.gui.models import DetectKitProject
from hydra_suite.widgets.dialogs import BaseDialog


class ActiveLearningDialog(BaseDialog):
    """Three-section AL dialog: Input, Acquisition, Execution."""

    def __init__(self, project: DetectKitProject, parent: QWidget | None = None):
        super().__init__(parent=parent, title="Active Learning")
        self._project = project
        self._run_handler: Callable[[], None] | None = None
        self._cancel_handler: Callable[[], None] | None = None
        self._build_ui()
        self._sync_run_enabled()

    def _build_ui(self) -> None:
        root = QVBoxLayout()

        input_form = QFormLayout()
        self.rb_video = QRadioButton("Video")
        self.rb_folder = QRadioButton("Image folder")
        self.rb_project = QRadioButton("Existing project source (unlabeled)")
        self.rb_video.setChecked(True)
        for rb in (self.rb_video, self.rb_folder, self.rb_project):
            rb.toggled.connect(self._sync_run_enabled)
        rb_row = QHBoxLayout()
        rb_row.addWidget(self.rb_video)
        rb_row.addWidget(self.rb_folder)
        rb_row.addWidget(self.rb_project)
        input_form.addRow("Source kind", _wrap(rb_row))

        self.input_path_edit = QLineEdit()
        self.input_path_edit.textChanged.connect(self._sync_run_enabled)
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse)
        path_row = QHBoxLayout()
        path_row.addWidget(self.input_path_edit)
        path_row.addWidget(browse_btn)
        input_form.addRow("Path", _wrap(path_row))

        self.preset_combo = QComboBox()
        for name in PRESETS:
            if name \!= "tracker_default":
                self.preset_combo.addItem(name)

        self.expected_count_spin = QSpinBox()
        self.expected_count_spin.setRange(0, 1000)
        self.expected_count_spin.setValue(0)

        self.budget_spin = QSpinBox()
        self.budget_spin.setRange(1, 1000)
        self.budget_spin.setValue(50)

        acq_form = QFormLayout()
        acq_form.addRow("Preset", self.preset_combo)
        acq_form.addRow("Expected count per frame (0 = unknown)", self.expected_count_spin)
        acq_form.addRow("Budget (top-K)", self.budget_spin)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.status_label = QLabel("Idle.")
        self.run_button = QPushButton("Run")
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        btn_row = QHBoxLayout()
        btn_row.addWidget(self.run_button)
        btn_row.addWidget(self.cancel_button)

        self.run_button.clicked.connect(self._on_run)
        self.cancel_button.clicked.connect(self._on_cancel)

        root.addWidget(_section("Input", input_form))
        root.addWidget(_section("Acquisition", acq_form))
        root.addWidget(self.progress)
        root.addWidget(self.status_label)
        root.addLayout(btn_row)
        self.setLayout(root)

    def _browse(self) -> None:
        if self.rb_video.isChecked():
            path, _ = QFileDialog.getOpenFileName(
                self, "Select video", "", "Video files (*.mp4 *.mov *.avi)"
            )
        elif self.rb_folder.isChecked():
            path = QFileDialog.getExistingDirectory(self, "Select image folder")
        else:
            path = ""
        if path:
            self.input_path_edit.setText(path)

    def _sync_run_enabled(self, *_):
        path_ok = (
            self.rb_project.isChecked()
            or bool(self.input_path_edit.text().strip())
        )
        model_ok = bool(self._project.active_model_path)
        self.run_button.setEnabled(path_ok and model_ok)
        if not model_ok:
            self.status_label.setText("Set an active model in DetectKit before running AL.")
        elif not path_ok:
            self.status_label.setText("Pick an input source.")
        else:
            self.status_label.setText("Ready.")

    def _on_run(self) -> None:
        if self._run_handler is not None:
            self._run_handler()

    def _on_cancel(self) -> None:
        if self._cancel_handler is not None:
            self._cancel_handler()

    def set_run_handler(self, handler: Callable[[], None]) -> None:
        """Main window wires this to construct + start the AL worker."""
        self._run_handler = handler

    def set_cancel_handler(self, handler: Callable[[], None]) -> None:
        self._cancel_handler = handler


def _wrap(layout) -> QWidget:
    w = QWidget()
    w.setLayout(layout)
    return w


def _section(title: str, content) -> QWidget:
    box = QWidget()
    v = QVBoxLayout(box)
    v.addWidget(QLabel(f"<b>{title}</b>"))
    if isinstance(content, QFormLayout):
        wrapper = QWidget()
        wrapper.setLayout(content)
        v.addWidget(wrapper)
    else:
        v.addWidget(content)
    return box
```

If `src/hydra_suite/detectkit/gui/dialogs/__init__.py` enumerates dialogs explicitly, add `ActiveLearningDialog` to its exports; otherwise leave it untouched.

- [ ] **Step 4: Wire the dialog into `main_window.py`**

In `src/hydra_suite/detectkit/gui/main_window.py`, add imports near other dialog imports:

```python
from .dialogs.active_learning import ActiveLearningDialog
from hydra_suite.detectkit.jobs.al_worker import ALRequest, ALWorker
```

Add three methods to the main-window class, alongside `_open_training_dialog` / `_open_history_dialog`:

```python
def _open_active_learning_dialog(self):
    if self._project is None:
        return
    dlg = ActiveLearningDialog(project=self._project, parent=self)
    dlg.set_run_handler(lambda: self._start_al_round(dlg))
    dlg.set_cancel_handler(lambda: self._cancel_al_round())
    dlg.open()  # non-blocking; modal events handled by Qt event loop

def _start_al_round(self, dlg):
    request = ALRequest(
        input_kind=("video" if dlg.rb_video.isChecked()
                    else "folder" if dlg.rb_folder.isChecked()
                    else "project"),
        input_path=dlg.input_path_edit.text(),
        project=self._project,
        budget=dlg.budget_spin.value(),
        preset=dlg.preset_combo.currentText(),
        expected_count=dlg.expected_count_spin.value(),
        detector_fn=self._load_active_detector_fn(),
    )
    worker = ALWorker(request)
    worker.progress_signal.connect(
        lambda p, s: (dlg.progress.setValue(p), dlg.status_label.setText(s))
    )
    worker.finished_signal.connect(
        lambda path, n, _ids: dlg.status_label.setText(f"Imported {n} frames -> {path}")
    )
    worker.error_signal.connect(lambda msg: dlg.status_label.setText(f"Error: {msg}"))
    worker.start()
    self._al_worker = worker

def _cancel_al_round(self):
    worker = getattr(self, "_al_worker", None)
    if worker is not None:
        worker.requestInterruption()

def _load_active_detector_fn(self):
    """Return a detector_fn(frame, conf, iou) -> list[(cx,cy,w,h,theta,conf)].

    NOTE: extraction of the in-process model loader from this file's existing
    dataset-prediction code (around line 1237 onward) is a follow-up. Until
    that helper exists, raise NotImplementedError to surface the gap loudly.
    """
    raise NotImplementedError(
        "Active-learning detector loading is not yet wired. "
        "Extract the model loader from main_window._run_dataset_prediction "
        "into a reusable helper, then call it here."
    )
```

Add a menu entry next to existing ones (search for where "Training..." or "History..." actions are added):

```python
al_action = QAction("Active Learning...", self)
al_action.triggered.connect(self._open_active_learning_dialog)
# Append to whichever menu hosts Training / History actions.
```

- [ ] **Step 5: Run the dialog smoke test**

Run: `python -m pytest tests/test_detectkit_al_dialog.py -v`
Expected: 2 tests PASS.

- [ ] **Step 6: Run the broader DetectKit test slice for regressions**

Run: `python -m pytest tests/test_detectkit_main_window.py tests/test_detectkit_al_dialog.py tests/test_detectkit_al_worker.py -v`
Expected: All PASS. If `test_detectkit_main_window.py` enumerates menu actions, update its expected list rather than removing the new action.

- [ ] **Step 7: Commit**

```bash
git add src/hydra_suite/detectkit/gui/dialogs/active_learning.py src/hydra_suite/detectkit/gui/main_window.py tests/test_detectkit_al_dialog.py
git commit -m "feat(detectkit): add Active Learning dialog and main-window wiring"
```

---

## Task 10 — Tracker UI cleanup: replace "Quality threshold" with "Min selection score" + preset combo

**Files:**
- Modify: `src/hydra_suite/trackerkit/gui/panels/dataset_panel.py:130-145`
- Modify: `tests/test_trackerkit_panels_smoke.py` (or whichever existing test exercises `DatasetPanel`)

- [ ] **Step 1: Add a smoke-level test for the new widgets**

In the trackerkit panels smoke test file, add (adapt to whatever stub/main-window fixture the rest of the file uses):

```python
def test_dataset_panel_has_min_selection_score_and_preset(qtbot):
    from hydra_suite.trackerkit.gui.panels.dataset_panel import DatasetPanel
    from hydra_suite.trackerkit.config.schemas import TrackerConfig

    panel = DatasetPanel(main_window=_StubMainWindow(), config=TrackerConfig())
    qtbot.addWidget(panel)

    assert hasattr(panel, "spin_dataset_min_selection_score")
    spin = panel.spin_dataset_min_selection_score
    assert spin.minimum() == 0.0
    assert spin.maximum() == 1.0
    assert "Min selection score" in spin.toolTip()

    assert hasattr(panel, "combo_dataset_preset")
    items = [panel.combo_dataset_preset.itemText(i) for i in range(panel.combo_dataset_preset.count())]
    assert "tracker_default" in items
    assert "balanced" in items
```

- [ ] **Step 2: Run the test, observe failure**

Run: `python -m pytest tests/test_trackerkit_panels_smoke.py -v -k min_selection_score_and_preset`
Expected: FAIL — attributes do not yet exist.

- [ ] **Step 3: Replace the widget block in `dataset_panel.py`**

In `src/hydra_suite/trackerkit/gui/panels/dataset_panel.py:130-145` find the `self.spin_dataset_conf_threshold` block and replace it with:

```python
# Min selection score (replaces legacy "Quality threshold").
# Under the new normalized scoring, this is a 0-1 cutoff applied AFTER ranking.
self.spin_dataset_min_selection_score = QDoubleSpinBox()
self.spin_dataset_min_selection_score.setRange(0.0, 1.0)
self.spin_dataset_min_selection_score.setSingleStep(0.05)
self.spin_dataset_min_selection_score.setDecimals(2)
self.spin_dataset_min_selection_score.setValue(0.0)
self.spin_dataset_min_selection_score.setToolTip(
    "Min selection score (0.0-1.0).\n\n"
    "Frames with a normalized acquisition score below this value are\n"
    "discarded before top-K selection. 0.0 = no filter (default).\n\n"
    "Replaces the legacy 'Quality threshold' control: under the new\n"
    "normalized scoring, an unbounded threshold no longer makes sense."
)
f_selection.addRow("Min selection score", self.spin_dataset_min_selection_score)

# Acquisition preset selector.
self.combo_dataset_preset = QComboBox()
for name in ("tracker_default", "balanced", "uncertainty_heavy", "exploration_heavy"):
    self.combo_dataset_preset.addItem(name)
self.combo_dataset_preset.setToolTip(
    "Acquisition weight preset. Default 'tracker_default' includes tracker-side\n"
    "signals (assignment cost, track loss). Others apply to detector-only paths."
)
f_selection.addRow("Acquisition preset", self.combo_dataset_preset)
```

Also add `QComboBox` to the imports at the top of the file if not already present.

Then propagate the new params through wherever `DatasetGenerationWorker` is constructed. Search the trackerkit code for `DATASET_CONF_THRESHOLD` and `DatasetGenerationWorker(`. In each call site, add to the `params` dict passed to the worker:

```python
params["DATASET_MIN_SELECTION_SCORE"] = float(panel.spin_dataset_min_selection_score.value())
params["DATASET_AL_PRESET"] = panel.combo_dataset_preset.currentText()
```

The refactored `FrameQualityScorer.get_worst_frames` (Task 7) already reads `DATASET_MIN_SELECTION_SCORE`. Extend `FrameQualityScorer.__init__` from Task 7 to honor `DATASET_AL_PRESET`:

```python
preset_name = params.get("DATASET_AL_PRESET", "tracker_default")
base = PRESETS.get(preset_name, PRESETS["tracker_default"])
```

(Replace the `base = PRESETS["tracker_default"]` line in Task 7's implementation with this lookup. Apply this change as part of Step 3.)

- [ ] **Step 4: Run trackerkit panel smoke + regression suites**

Run: `python -m pytest tests/test_trackerkit_panels_smoke.py tests/test_dataset_generation.py tests/test_al_acquisition.py -v`
Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/trackerkit/gui/panels/dataset_panel.py src/hydra_suite/data/dataset_generation.py tests/test_trackerkit_panels_smoke.py
git commit -m "feat(trackerkit): replace Quality threshold with normalized Min selection score and preset combo"
```

---

## Final task — full-suite sanity sweep

- [ ] **Step 1: Run the entire AL test slice plus the dataset-generation suite**

Run:
```bash
python -m pytest tests/test_al_frame_source.py tests/test_al_candidate_pool.py tests/test_al_signals.py tests/test_al_acquisition.py tests/test_dataset_generation.py tests/test_detectkit_al_worker.py tests/test_detectkit_al_dialog.py -v
```
Expected: All PASS.

- [ ] **Step 2: Run broader detectkit + trackerkit slices for incidental regressions**

Run:
```bash
python -m pytest tests/test_detectkit_main_window.py tests/test_detectkit_dataset_panel.py tests/test_trackerkit_panels_smoke.py tests/test_trackerkit_workers_smoke.py -v
```
Expected: All PASS.

- [ ] **Step 3: Format + lint per project conventions**

Run:
```bash
make commit-prep
make lint-moderate
```
Expected: Both succeed cleanly. Fix any reported issues before the final commit.

- [ ] **Step 4: Final commit if formatting changes landed**

```bash
git add -A
git commit -m "chore: format AL feature additions"
```

---

## Self-review notes

**Spec coverage:**

- §2.1 frame_source -> Tasks 1, 2 ✓
- §2.2 candidate_pool -> Task 3 ✓
- §2.3 signals -> Tasks 4, 5 ✓
- §2.4 acquisition -> Task 6 (with `min_score` parameter co-located, used by Task 7's adapter and Task 10's UI) ✓
- §2.5 al_worker -> Task 8 ✓
- §2.6 active_learning dialog -> Task 9 ✓
- §2.7 tracker-side cleanup -> Tasks 7 + 10 ✓
- Error handling: empty pool (Task 8), no detector (Task 8 + Task 9 stub), cancel (Task 8 worker), preset/min-score plumbing (Tasks 7, 10) ✓
- Testing: TDD per task + final suite sweep ✓

**Type-consistency:** `ALSignals`, `AcquisitionWeights`, `FrameRef`, `ALRequest`, `ALResult`, `Detection`, `DetectorFn` are referenced consistently across tasks. `select(...)` signature includes `min_score` from Task 6 onward; Task 7 uses it; Task 10 surfaces it through the UI.

**Open follow-ups (deliberately out of scope):**

1. Extract a reusable `build_detector_fn(project)` helper from the existing dataset-prediction code in `detectkit/gui/main_window.py:1237-1395` so Task 9's `_load_active_detector_fn` can stop raising `NotImplementedError`. This is documented in the spec's design notes; doing it as part of this plan would expand scope.
2. GPU-OOM batched-inference fallback for `run_active_learning` (mirroring `_detect_batch` in `dataset_generation.py`). Punted unless encountered in practice.
