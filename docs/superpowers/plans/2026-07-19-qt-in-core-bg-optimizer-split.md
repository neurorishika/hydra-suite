# Qt-in-Core Split: bg-optimizer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the `core`→GUI dependency-direction violation in `src/hydra_suite/core/background/optimizer.py` by extracting the two `QThread` worker classes into the trackerkit app layer, leaving `core/background` Qt-free.

**Architecture:** The pure optimization math already lives as module-level functions in `optimizer.py`; only the orchestration loop (`_run_optimization`) and preview generation (`_generate_previews`) are Qt-coupled (they call `self.*_signal.emit()` and check `self._stop_requested` mid-loop). Extract those loop bodies into **pure callback-driven module functions** (taking `progress_cb`/`frame_cb`/`stop_check` callables — the idiom `_read_gray_frames` already uses), delete the `QThread` subclasses + the `PySide6` import from core, and put thin `QThread` wrappers that translate callbacks↔signals in `trackerkit/gui/workers/`.

**Tech Stack:** Python 3, PySide6 (QtCore), Optuna, numpy/opencv, pytest.

## Global Constraints

- After this change, `src/hydra_suite/core/background/` must import NO Qt (`grep -rn "PySide6\|QtCore\|QThread\|Signal" src/hydra_suite/core/background/` → empty). This is the whole point.
- **No behavior change** to the bg-parameter-helper dialog: the workers' observable interface (signal names/shapes, `stop()`, the `_cached_*` frame-cache attributes the dialog reads for preview handoff) must be preserved so the dialog works unchanged except for the import path.
- The pure helper `_suggest_trial_params` must stay importable from `hydra_suite.core.background.optimizer` (`tests/test_bg_parameter_helper.py:18` imports it). `BgOptimizationResult` and `_BgFrameCache` are pure dataclasses and STAY in core.
- Dependency direction: `core/background` must not import from any app layer; the new `trackerkit/gui/workers/bg_optimizer_worker.py` imports pure functions FROM core (allowed).
- Run `make format` before each commit; tests via `conda run -n hydra-mps python -m pytest ... -q --ignore=tests/test_identity_postprocess.py` (the ignored file has a pre-existing collection error).

## Out of scope (note, do not do here)

`core/tracking/optimization/optimizer_workers.py` has the same violation (`DetectionCacheBuildWorker(QThread)`, `TrackingPreviewWorker(QThread)` living in core). It is a parallel offender but NOT part of this task. Also NOT here: migrating the moved workers onto `widgets/workers.BaseWorker` (the plan keeps plain `QThread` to preserve the exact signal set and minimize dialog risk; BaseWorker alignment is an optional future follow-up).

---

## File Structure

- `src/hydra_suite/core/background/optimizer.py` — **modify**: add two pure callback-driven functions (`run_bg_optimization`, `generate_bg_previews`) + a small result-bundle dataclass; DELETE the two `QThread` classes and the `from PySide6.QtCore import QThread, Signal` line. Keep all pure helpers + `BgOptimizationResult` + `_BgFrameCache`.
- `src/hydra_suite/trackerkit/gui/workers/bg_optimizer_worker.py` — **create**: the two thin `QThread` wrappers (`BgSubtractionOptimizer`, `BgDetectionPreviewWorker`) translating callbacks↔signals, preserving the current signal set + `_cached_*` attrs.
- `src/hydra_suite/trackerkit/gui/dialogs/bg_parameter_helper.py` — **modify**: repoint the two-worker import to the new module (keep `BgOptimizationResult` imported from core).
- Tests: `tests/test_bg_optimizer_core.py` (new — pure functions + Qt-free guard), extend `tests/test_bg_parameter_helper.py` if needed.

---

### Task 1: Extract the optimization loop into a pure `run_bg_optimization`

Move the body of `BgSubtractionOptimizer._run_optimization` (currently lines ~623-838) into a module-level pure function, replacing Qt with injected callbacks. The pure helpers it already calls (`_read_gray_frames`, `_build_prime_frame_indices`, `_resize_roi_mask`, `_BgFrameCache`, `_suggest_trial_params`, `_init_trial_pipeline`, `_run_bg_trial_frame`, `_aggregate_trial_scores`, `BgOptimizationResult`) stay put and are reused. Also move `_build_sampler` (currently a method at ~556-603, but pure) to a module function `_build_sampler(sampler_type, n_active)`.

**Files:**
- Modify: `src/hydra_suite/core/background/optimizer.py`
- Test: `tests/test_bg_optimizer_core.py`

**Interfaces:**
- Consumes: the existing pure `_*` helpers + Optuna.
- Produces:
  ```python
  @dataclass
  class BgOptimizationRun:
      results: list[BgOptimizationResult]
      # frame cache for preview reuse (what the QThread used to stash as _cached_*):
      prime_frames: list          # np arrays
      sample_frames: list
      sample_indices: list[int]
      roi_mask: "np.ndarray | None"

  def run_bg_optimization(
      video_path: str,
      base_params: dict,
      tuning_config: dict,
      scoring_weights: dict,
      n_trials: int,
      n_sample_frames: int,
      sampler_type: str,
      *,
      progress_cb: "Callable[[int, str], None] | None" = None,
      stop_check: "Callable[[], bool] | None" = None,
  ) -> BgOptimizationRun: ...
  ```

- [ ] **Step 1: Write the failing test**

```python
# tests/test_bg_optimizer_core.py
"""Pure (Qt-free) bg-optimizer core: callback wiring + stop honoring."""
import hydra_suite.core.background.optimizer as opt


def test_run_bg_optimization_is_importable_and_qt_free():
    # The function exists and the module imports without Qt.
    assert hasattr(opt, "run_bg_optimization")
    assert hasattr(opt, "generate_bg_previews")
    assert hasattr(opt, "BgOptimizationRun")


def test_run_bg_optimization_honors_stop_check(monkeypatch, tmp_path):
    # stop_check returning True immediately must short-circuit before any Optuna work.
    progress = []
    # Force the video-open path to be a no-op / raise-free by stopping instantly.
    run = opt.run_bg_optimization(
        video_path="nonexistent.mp4",
        base_params={"MAX_TARGETS": 1, "RESIZE_FACTOR": 1.0},
        tuning_config={},
        scoring_weights={},
        n_trials=1,
        n_sample_frames=1,
        sampler_type="tpe",
        progress_cb=lambda pct, msg="": progress.append((pct, msg)),
        stop_check=lambda: True,   # stop immediately
    )
    assert isinstance(run, opt.BgOptimizationRun)
    assert run.results == []   # stopped before any trial completed
```

(Adjust the stop-path assertion to how the function actually orders its early stop-check vs `cv2.VideoCapture` open — see Step 3; the function must check `stop_check()` before/around the expensive setup so an immediate stop yields empty results without crashing on the missing file. If opening the video must happen first, monkeypatch `cv2.VideoCapture` to a fake that reports not-opened and assert the function returns an empty run gracefully.)

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n hydra-mps python -m pytest tests/test_bg_optimizer_core.py -q`
Expected: FAIL — `run_bg_optimization`/`generate_bg_previews`/`BgOptimizationRun` don't exist yet.

- [ ] **Step 3: Implement `run_bg_optimization` (extraction)**

Read the current `BgSubtractionOptimizer._run_optimization` body (lines ~623-838) and transcribe it into the new module-level `run_bg_optimization`, applying this MECHANICAL transformation:
- `self.video_path` / `self.base_params` / `self.tuning_config` / `self.scoring_weights` / `self.n_trials` / `self.n_sample_frames` / `self.sampler_type` → the corresponding function parameters.
- `self.progress_signal.emit(pct, msg)` → `if progress_cb is not None: progress_cb(pct, msg)`.
- `self._stop_requested` → `(stop_check() if stop_check is not None else False)`.
- `self._build_sampler(n_active)` → the new module `_build_sampler(sampler_type, n_active)`.
- The final `self.result_signal.emit(results)` → drop (the caller gets `results` via the return value).
- The `self._cached_prime_frames = ...` / `_cached_sample_frames` / `_cached_sample_indices` / `_cached_roi_mask` stashes (~829-834) → collect into the returned `BgOptimizationRun` fields.
- Return `BgOptimizationRun(results=results, prime_frames=..., sample_frames=..., sample_indices=..., roi_mask=...)`.
Add `from typing import Callable` and the `BgOptimizationRun` dataclass. Move `_build_sampler` to module scope (drop `self`, take `sampler_type`). Do NOT change any pure-logic call or the scoring/param math.

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n hydra-mps python -m pytest tests/test_bg_optimizer_core.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
make format
git add src/hydra_suite/core/background/optimizer.py tests/test_bg_optimizer_core.py
git commit -m "refactor(core): extract pure run_bg_optimization from the QThread optimizer"
```

---

### Task 2: Extract preview generation into a pure `generate_bg_previews`

Move the body of `BgDetectionPreviewWorker._generate_previews` (~902-1014) into a pure module function using a `frame_cb` + `stop_check`.

**Files:**
- Modify: `src/hydra_suite/core/background/optimizer.py`
- Test: `tests/test_bg_optimizer_core.py` (extend)

**Interfaces:**
- Produces:
  ```python
  def generate_bg_previews(
      video_path: str,
      base_params: dict,
      trial_params: dict,
      n_sample_frames: int,
      *,
      prime_frames: list | None = None,
      sample_frames: list | None = None,
      sample_indices: list[int] | None = None,
      roi_mask: "np.ndarray | None" = None,
      frame_cb: "Callable[[int, np.ndarray], None] | None" = None,
      stop_check: "Callable[[], bool] | None" = None,
  ) -> None: ...
  ```
  (The cached frames are passed IN as kwargs — the same cache the optimizer produced — rather than read off a worker object.)

- [ ] **Step 1: Write the failing test**

```python
def test_generate_bg_previews_emits_via_frame_cb(monkeypatch):
    import numpy as np
    import hydra_suite.core.background.optimizer as opt

    frames = []
    # Provide a tiny in-memory cache so no video is opened; one 8x8 gray sample frame.
    gray = np.zeros((8, 8), np.uint8)
    opt.generate_bg_previews(
        video_path="unused.mp4",
        base_params={"MAX_TARGETS": 1, "RESIZE_FACTOR": 1.0},
        trial_params={},
        n_sample_frames=1,
        prime_frames=[gray],
        sample_frames=[gray],
        sample_indices=[0],
        roi_mask=None,
        frame_cb=lambda idx, rgb: frames.append((idx, rgb)),
        stop_check=lambda: False,
    )
    # At least attempts a frame; the exact count depends on the pipeline, assert callback wired.
    assert isinstance(frames, list)
```

(Refine against the real pipeline — if a trial pipeline needs more setup, mock `_init_trial_pipeline`/`_run_bg_trial_frame` to return an empty detection so the drawing path runs and `frame_cb` fires once. The assertion that matters: `frame_cb` is the ONLY output path — no `emit`, no Qt.)

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n hydra-mps python -m pytest tests/test_bg_optimizer_core.py::test_generate_bg_previews_emits_via_frame_cb -q`
Expected: FAIL — function doesn't exist.

- [ ] **Step 3: Implement `generate_bg_previews`**

Transcribe `_generate_previews`'s body, applying: `self.*` → params; `self.frame_signal.emit(fi, rgb)` → `if frame_cb is not None: frame_cb(fi, rgb)`; `self._stop_requested` → `stop_check()`; the cached-frame reads (`self._cached_*`) → the `prime_frames`/`sample_frames`/`sample_indices`/`roi_mask` kwargs (falling back to `_read_gray_frames` when they're None, exactly as the current code falls back). Keep all cv2 drawing + pure pipeline calls identical.

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n hydra-mps python -m pytest tests/test_bg_optimizer_core.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
make format
git add -A
git commit -m "refactor(core): extract pure generate_bg_previews from the QThread preview worker"
```

---

### Task 3: Delete the QThread classes + PySide6 import from core

Now the pure functions exist, remove the Qt from `core/background/optimizer.py`.

**Files:**
- Modify: `src/hydra_suite/core/background/optimizer.py`
- Test: `tests/test_bg_optimizer_core.py` (add the Qt-free guard)

- [ ] **Step 1: Write the failing guard test**

```python
def test_core_background_imports_no_qt():
    import ast
    from pathlib import Path
    import hydra_suite.core.background as bg

    pkg_dir = Path(bg.__file__).parent
    offenders = []
    for py in pkg_dir.glob("*.py"):
        tree = ast.parse(py.read_text(), filename=str(py))
        for node in ast.walk(tree):
            mod = node.module if isinstance(node, ast.ImportFrom) else (
                ",".join(a.name for a in node.names) if isinstance(node, ast.Import) else None
            )
            if mod and ("PySide6" in mod or "QtCore" in mod):
                offenders.append(f"{py.name}:{node.lineno}")
    assert not offenders, "core/background must not import Qt: " + "; ".join(offenders)
```

- [ ] **Step 2: Run it — fails** (optimizer.py still imports QtCore).

Run: `conda run -n hydra-mps python -m pytest tests/test_bg_optimizer_core.py::test_core_background_imports_no_qt -q`
Expected: FAIL, offender `optimizer.py:34`.

- [ ] **Step 3: Delete the Qt**

In `optimizer.py`: delete `from PySide6.QtCore import QThread, Signal` (line ~34), delete `class BgSubtractionOptimizer(QThread):` (~523-838) in full, and `class BgDetectionPreviewWorker(QThread):` (~846-1014) in full. Keep everything else (all `_*` helpers, `_BgFrameCache`, `BgOptimizationResult`, `BgOptimizationRun`, `_build_sampler`, `run_bg_optimization`, `generate_bg_previews`).

- [ ] **Step 4: Run tests + import check**

Run:
```bash
conda run -n hydra-mps python -m pytest tests/test_bg_optimizer_core.py tests/test_bg_parameter_helper.py -q --ignore=tests/test_identity_postprocess.py
conda run -n hydra-mps python -c "import hydra_suite.core.background.optimizer; from hydra_suite.core.background.optimizer import _suggest_trial_params, BgOptimizationResult; print('core optimizer Qt-free + pure imports OK')"
grep -rn "PySide6\|QtCore\|QThread\|Signal" src/hydra_suite/core/background/
```
Expected: guard test passes; `_suggest_trial_params` + `BgOptimizationResult` still importable (existing test at `test_bg_parameter_helper.py:18` still works — though the dialog import will fail until Task 5; run `test_bg_parameter_helper.py` only after Task 5 if it imports the workers at module load — see note); grep prints nothing.

Note: `tests/test_bg_parameter_helper.py` imports the DIALOG, which still imports the workers from core (now deleted) until Task 5. So between Task 3 and Task 5 that test file will fail to import. That is expected mid-refactor; Tasks 4-5 fix it. Run only `test_bg_optimizer_core.py` here to confirm the guard.

- [ ] **Step 5: Commit**

```bash
make format
git add -A
git commit -m "refactor(core): remove QThread workers + PySide6 import from bg optimizer"
```

---

### Task 4: Create the trackerkit QThread wrappers

**Files:**
- Create: `src/hydra_suite/trackerkit/gui/workers/bg_optimizer_worker.py`
- Test: covered via the dialog test after Task 5; optionally a focused wrapper test.

**Interfaces:**
- Consumes: `run_bg_optimization`, `generate_bg_previews`, `BgOptimizationResult`, `BgOptimizationRun` from `hydra_suite.core.background.optimizer`.
- Produces: `BgSubtractionOptimizer(QThread)` and `BgDetectionPreviewWorker(QThread)` with the SAME signals/attrs the dialog uses.

- [ ] **Step 1: Write the wrapper module**

```python
# src/hydra_suite/trackerkit/gui/workers/bg_optimizer_worker.py
"""Qt wrappers for the pure bg-sub optimizer (which lives Qt-free in
core/background/optimizer). Translates the pure functions' progress/frame/stop
callbacks into Qt signals. Keeps the exact interface the bg_parameter_helper
dialog depends on."""
from __future__ import annotations

import logging

from PySide6.QtCore import QThread, Signal

from hydra_suite.core.background.optimizer import (
    run_bg_optimization,
    generate_bg_previews,
)

logger = logging.getLogger(__name__)


class BgSubtractionOptimizer(QThread):
    progress_signal = Signal(int, str)
    result_signal = Signal(list)
    finished_signal = Signal()

    def __init__(self, video_path, base_params, tuning_config, scoring_weights,
                 n_trials, n_sample_frames, sampler_type, parent=None):
        super().__init__(parent)
        self.video_path = video_path
        self.base_params = base_params
        self.tuning_config = tuning_config
        self.scoring_weights = scoring_weights
        self.n_trials = n_trials
        self.n_sample_frames = n_sample_frames
        self.sampler_type = sampler_type
        self._stop_requested = False
        # Frame cache the dialog reads (via _preview_cache_kwargs) to hand off to
        # the preview worker. Populated on completion.
        self._cached_prime_frames = None
        self._cached_sample_frames = None
        self._cached_sample_indices = None
        self._cached_roi_mask = None

    def stop(self):
        self._stop_requested = True

    def run(self):
        try:
            run = run_bg_optimization(
                self.video_path, self.base_params, self.tuning_config,
                self.scoring_weights, self.n_trials, self.n_sample_frames,
                self.sampler_type,
                progress_cb=lambda pct, msg="": self.progress_signal.emit(int(pct), msg),
                stop_check=lambda: self._stop_requested,
            )
            self._cached_prime_frames = run.prime_frames
            self._cached_sample_frames = run.sample_frames
            self._cached_sample_indices = run.sample_indices
            self._cached_roi_mask = run.roi_mask
            self.result_signal.emit(run.results)
        except Exception:  # noqa: BLE001 - surface via progress, mirror old behavior
            logger.exception("BgSubtractionOptimizer failed")
            self.progress_signal.emit(0, "Optimization failed.")
        finally:
            self.finished_signal.emit()


class BgDetectionPreviewWorker(QThread):
    frame_signal = Signal(int, object)
    finished_signal = Signal()

    def __init__(self, video_path, base_params, trial_params, n_sample_frames,
                 prime_frames=None, sample_frames=None, sample_indices=None,
                 roi_mask=None, parent=None):
        super().__init__(parent)
        self.video_path = video_path
        self.base_params = base_params
        self.trial_params = trial_params
        self.n_sample_frames = n_sample_frames
        self._prime_frames = prime_frames
        self._sample_frames = sample_frames
        self._sample_indices = sample_indices
        self._roi_mask = roi_mask
        self._stop_requested = False

    def stop(self):
        self._stop_requested = True

    def run(self):
        try:
            generate_bg_previews(
                self.video_path, self.base_params, self.trial_params,
                self.n_sample_frames,
                prime_frames=self._prime_frames,
                sample_frames=self._sample_frames,
                sample_indices=self._sample_indices,
                roi_mask=self._roi_mask,
                frame_cb=lambda idx, rgb: self.frame_signal.emit(int(idx), rgb),
                stop_check=lambda: self._stop_requested,
            )
        except Exception:  # noqa: BLE001
            logger.exception("BgDetectionPreviewWorker failed")
        finally:
            self.finished_signal.emit()
```

VERIFY against the dialog: read `bg_parameter_helper.py`'s `_preview_cache_kwargs()` (~1097-1112) and its preview-worker construction (~1134-1141) to confirm the kwarg names the dialog passes match this `__init__` signature. If the dialog passes the cache via different kwarg names, either match them here or adjust the dialog in Task 5 — keep them consistent. (The dialog currently reads `_cached_*` off the optimizer and passes them into the preview worker; preserve that exact flow.)

- [ ] **Step 2: Import check**

Run: `conda run -n hydra-mps python -c "import hydra_suite.trackerkit.gui.workers.bg_optimizer_worker as w; assert hasattr(w, 'BgSubtractionOptimizer') and hasattr(w, 'BgDetectionPreviewWorker'); print('wrappers import OK')"`
Expected: OK.

- [ ] **Step 3: Commit**

```bash
make format
git add -A
git commit -m "feat(trackerkit): Qt wrappers for the pure bg-sub optimizer"
```

---

### Task 5: Repoint the dialog import + wire the cache handoff

**Files:**
- Modify: `src/hydra_suite/trackerkit/gui/dialogs/bg_parameter_helper.py`
- Test: `tests/test_bg_parameter_helper.py`

- [ ] **Step 1: Update imports**

In `bg_parameter_helper.py` (~lines 45-48), split the import: the two workers now come from the new module; `BgOptimizationResult` stays in core:
```python
from hydra_suite.core.background.optimizer import BgOptimizationResult
from hydra_suite.trackerkit.gui.workers.bg_optimizer_worker import (
    BgDetectionPreviewWorker,
    BgSubtractionOptimizer,
)
```

- [ ] **Step 2: Confirm the cache handoff still works**

Read `_preview_cache_kwargs()` (~1097-1112): it pulls `_cached_prime_frames`/`_cached_sample_frames`/`_cached_sample_indices`/`_cached_roi_mask` off `self.optimizer` via `getattr` and passes them into the `BgDetectionPreviewWorker(...)`. The new `BgSubtractionOptimizer` wrapper still exposes those attributes (populated in `run()`), and the new `BgDetectionPreviewWorker.__init__` accepts them. If the kwarg names differ, reconcile so `_preview_cache_kwargs()`'s output maps onto the preview worker's `__init__`. Do not change the signal connections (`progress_signal`/`result_signal`/`finished_signal`/`frame_signal`) — they're unchanged.

- [ ] **Step 3: Run the dialog tests + import**

Run:
```bash
conda run -n hydra-mps python -m pytest tests/test_bg_parameter_helper.py -q --ignore=tests/test_identity_postprocess.py
conda run -n hydra-mps python -c "import hydra_suite.trackerkit.gui.dialogs.bg_parameter_helper; print('dialog imports OK')"
```
Expected: dialog imports; all `test_bg_parameter_helper.py` tests pass (they never started a real worker thread, so the wrapper move doesn't affect them — but they DO import the dialog, which now resolves).

- [ ] **Step 4: Commit**

```bash
make format
git add -A
git commit -m "refactor(trackerkit): point bg_parameter_helper at the relocated Qt workers"
```

---

## Final verification (whole plan)

- [ ] **Step 1: core/background is Qt-free**

Run:
```bash
grep -rn "PySide6\|QtCore\|QThread\|Signal" src/hydra_suite/core/background/
```
Expected: EMPTY.

- [ ] **Step 2: No stale importers of the old worker location**

Run:
```bash
grep -rn "from hydra_suite.core.background.optimizer import" src/ tests/ | grep -iE "BgSubtractionOptimizer|BgDetectionPreviewWorker"
```
Expected: EMPTY (the workers are now imported only from `trackerkit.gui.workers.bg_optimizer_worker`; `BgOptimizationResult`/`_suggest_trial_params` may still be imported from core — that's fine).

- [ ] **Step 3: Targeted suite + imports**

Run:
```bash
conda run -n hydra-mps python -m pytest tests/test_bg_optimizer_core.py tests/test_bg_parameter_helper.py -q --ignore=tests/test_identity_postprocess.py
conda run -n hydra-mps python -c "import hydra_suite.core.background.optimizer, hydra_suite.trackerkit.gui.dialogs.bg_parameter_helper, hydra_suite.trackerkit.gui.workers.bg_optimizer_worker; print('all import OK')"
```
Expected: pass; imports OK.

- [ ] **Step 4: (Optional but recommended) offscreen dialog smoke** — `QT_QPA_PLATFORM=offscreen conda run -n hydra-mps python -c "..."` constructing the `BgParameterHelper` dialog to confirm it builds with the relocated workers. Skip if dialog construction is too heavy.

- [ ] **Step 5: Format gate** — `make format-check`.

---

---

# Part B — `core/tracking/optimization/optimizer_workers.py` (parallel offender)

Same violation, same fix. `optimizer_workers.py` imports `from PySide6.QtCore import QThread, Signal` (line 15) and holds two QThread classes; its 5 pure `_preview_*` helpers (lines 51-285: `_preview_filter_cached_detections`, `_preview_compute_pose_features`, `_preview_process_matched_tracks`, `_preview_init_free_detections`, `_preview_render_tracks`) STAY in core (tested at `tests/test_tracking_optimizer_helpers.py:394-423`).

**Part B file structure:**
- `src/hydra_suite/core/tracking/optimization/optimizer_workers.py` — **modify**: extract `TrackingPreviewWorker.run`'s loop into a pure `run_tracking_preview(...)`; DELETE both QThread classes + the `PySide6` import. Keep the 5 pure helpers.
- `src/hydra_suite/trackerkit/gui/workers/param_optimizer_worker.py` — **create**: the two relocated QThread wrappers.
- `src/hydra_suite/trackerkit/gui/orchestrators/config.py` (~3408) and `src/hydra_suite/trackerkit/gui/dialogs/parameter_helper.py` (~45) — **modify**: repoint imports.

### Task B1: Extract `TrackingPreviewWorker.run` into a pure `run_tracking_preview`

`TrackingPreviewWorker.run()` (lines ~381-668) runs a tracking loop over cached detections reusing the 5 pure `_preview_*` helpers, emitting `frame_signal(np.ndarray)` per frame and checking `self._stop_requested`. `DetectionCacheBuildWorker.run()` (~315-348) is ALREADY thin (it just calls `InferenceRunner.run_batch_pass(..., progress_cb=, should_stop=)`, a pure core method) — it needs NO extraction, only relocation in Task B3.

**Files:** Modify `optimizer_workers.py`; Test `tests/test_optimizer_workers_core.py` (new).

**Interfaces:**
- Produces: `run_tracking_preview(video_path, base_params, preview_params, detection_cache_path, start_frame, end_frame, *, frame_cb: Callable[[np.ndarray], None] | None = None, stop_check: Callable[[], bool] | None = None) -> None` — the `TrackingPreviewWorker.run` body with `self.*` → params, `self.frame_signal.emit(rgb)` → `frame_cb(rgb)`, `self._stop_requested` → `stop_check()`. (Confirm the exact `__init__` fields TrackingPreviewWorker stores by reading ~361-376 and map each to a param.)

- [ ] **Step 1: Write the failing test** — assert `optimizer_workers.run_tracking_preview` exists and is Qt-free (module imports without Qt after Task B2); a minimal drive with a fake cache + `frame_cb` capturing frames + `stop_check` honoring. Mock the pure helpers as needed so no real models load.
- [ ] **Step 2: Run — fails** (function doesn't exist).
- [ ] **Step 3: Implement** the extraction per the transformation rules above (identical mechanics to Part A Task 1).
- [ ] **Step 4: Run — passes.**
- [ ] **Step 5: Commit** — `refactor(core): extract pure run_tracking_preview from the QThread preview worker`.

### Task B2: Delete both QThread classes + PySide6 import from optimizer_workers.py

- [ ] **Step 1: Guard test** — extend `tests/test_optimizer_workers_core.py` with an AST check that `core/tracking/optimization/` imports no Qt (same shape as Part A Task 3's guard, over that dir's `.py` files).
- [ ] **Step 2: Run — fails** (optimizer_workers.py:15 imports QtCore).
- [ ] **Step 3: Delete** `from PySide6.QtCore import QThread, Signal` (line 15), `class DetectionCacheBuildWorker(QThread)` (~286-348), `class TrackingPreviewWorker(QThread)` (~353-668). Keep the 5 `_preview_*` helpers + `run_tracking_preview`.
- [ ] **Step 4: Verify** — guard passes; `grep -rn "PySide6\|QtCore\|QThread\|Signal" src/hydra_suite/core/tracking/optimization/` empty; `_preview_filter_cached_detections` still importable from the module (its test at `test_tracking_optimizer_helpers.py:394` still resolves). (Consumers break until B3-B4 — expected mid-refactor.)
- [ ] **Step 5: Commit** — `refactor(core): remove QThread workers + PySide6 import from optimizer_workers`.

### Task B3: Create trackerkit wrappers + repoint consumers

**Files:** Create `trackerkit/gui/workers/param_optimizer_worker.py`; modify `trackerkit/gui/orchestrators/config.py` (~3408) + `trackerkit/gui/dialogs/parameter_helper.py` (~45).

- [ ] **Step 1: Create the wrappers.** `param_optimizer_worker.py` holds:
  - `DetectionCacheBuildWorker(QThread)` — RELOCATED as-is from the old class (it was already a thin wrapper over `InferenceRunner.run_batch_pass`; just move it, importing `InferenceRunner`/`build_inference_config_from_params` from core as it already did). Signals `progress_signal(int,str)`, `finished_signal(bool,str)`; `stop()`/`_stop_requested`; same `__init__`.
  - `TrackingPreviewWorker(QThread)` — thin wrapper: signals `frame_signal(np.ndarray)`, `finished_signal()`; `run()` calls `run_tracking_preview(..., frame_cb=lambda rgb: self.frame_signal.emit(rgb), stop_check=lambda: self._stop_requested)` in try/except with `finished_signal.emit()` in `finally`; same `__init__` fields.
- [ ] **Step 2: Repoint consumers.**
  - `orchestrators/config.py:~3408`: `from hydra_suite.core.tracking.optimization.optimizer_workers import DetectionCacheBuildWorker` → `from hydra_suite.trackerkit.gui.workers.param_optimizer_worker import DetectionCacheBuildWorker`.
  - `parameter_helper.py:~45`: `from hydra_suite.core.tracking.optimization.optimizer_workers import TrackingPreviewWorker` → same new module.
- [ ] **Step 3: Update tests that reference the workers' NEW location:**
  - `tests/test_optimizer_cache_builder.py:~37` (`ow.DetectionCacheBuildWorker`) — repoint its module import to `trackerkit.gui.workers.param_optimizer_worker`.
  - `tests/test_tracking_preview_worker_cache.py` (`ow.TrackingPreviewWorker` ~94, and it `inspect.getsource(ow.TrackingPreviewWorker.run)` ~118 to assert no legacy `DetectionCache`) — repoint to the new module. The `getsource(...run)` guard now inspects a thin wrapper; move that "no DetectionCache" assertion to inspect `run_tracking_preview`'s source in core instead (that's where the cache-read logic now lives), OR assert it against the core function. Keep the guard meaningful.
- [ ] **Step 4: Verify** — `conda run -n hydra-mps python -m pytest tests/test_optimizer_cache_builder.py tests/test_tracking_preview_worker_cache.py tests/test_tracking_optimizer_helpers.py tests/test_optimizer_workers_core.py -q --ignore=tests/test_identity_postprocess.py`; imports of `orchestrators/config.py` + `parameter_helper.py` OK; `grep -rn "from hydra_suite.core.tracking.optimization.optimizer_workers import" src/ tests/ | grep -iE "DetectionCacheBuildWorker|TrackingPreviewWorker"` empty (workers now only from trackerkit; the `_preview_*` pure imports stay).
- [ ] **Step 5: Commit** — `refactor(trackerkit): relocate param-optimizer Qt workers out of core`.

### Part B final verification

- [ ] `grep -rn "PySide6\|QtCore\|QThread\|Signal" src/hydra_suite/core/` → EMPTY (the WHOLE `core/` tree is now Qt-free — both offenders fixed).
- [ ] Full targeted suite of all affected tests passes; all three kits + the two dialogs import.
- [ ] `make format-check`.

---

## Self-Review notes

- **Goal check:** Global Constraint "core/background imports no Qt" is enforced by the Task 3 guard test + the final grep. The violation is removed by relocating only the Qt-coupled parts (QThread subclass, Signals, run/stop/emit) while the pure math stays in core.
- **Behavior preservation:** the dialog's signal set (`progress_signal(int,str)`, `result_signal(list)`, `finished_signal()`, `frame_signal(int,object)`), `stop()`, and the `_cached_*` preview-cache handoff are all preserved on the relocated wrappers, so `bg_parameter_helper` works unchanged except for the import path. The `try/except → progress emit / finished in finally` error behavior mirrors the original `run()`.
- **Test coverage:** the only existing test (`_suggest_trial_params`) keeps importing from core and is unaffected. New `test_bg_optimizer_core.py` covers the pure functions' callback/stop wiring + the Qt-free guard. The QThread `run()` bodies were untested before and remain thin translation shims (low risk).
- **Type/name consistency:** `run_bg_optimization(...) -> BgOptimizationRun`; the wrapper reads `run.prime_frames/sample_frames/sample_indices/roi_mask/results` and re-exposes them as `_cached_*`; `generate_bg_previews(..., prime_frames=, sample_frames=, sample_indices=, roi_mask=, frame_cb=, stop_check=)`. Cross-checked against the dialog's `_preview_cache_kwargs` handoff in Tasks 4-5.
```
