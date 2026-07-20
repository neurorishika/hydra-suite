# TrackingWorker Qt-in-Core Split Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the last `core`→PySide6 dependency-direction violation by splitting `src/hydra_suite/core/tracking/worker.py::TrackingWorker(QThread)` (the ~4372-line main tracking orchestrator) into a Qt-free plain `TrackingEngineCore` in core plus a thin `TrackingWorker(QThread)` wrapper in `trackerkit/gui/workers/`, leaving the **entire `core/` tree Qt-free**.

**Architecture:** `TrackingWorker` is a method-coupled QThread (dozens of methods sharing `self` state: `frame_count`, `trajectories_full`, `_density_regions`, caches, `frame_prefetcher`, `parameters`). It is too coupled for module-function extraction (the bg-optimizer approach); instead we follow the **`TrackingOptimizerCore` precedent** (Part C of the completed `done/2026-07-19-qt-in-core-bg-optimizer-split.md`): rename the class to a plain `TrackingEngineCore`, keep every method in place, replace the 6 Qt `Signal.emit(...)` calls with 6 constructor-injected callbacks, swap `QMutex`→`threading.Lock`, drop `@Slot`, and rename the QThread entry `run()`→`run_tracking()`. A thin `TrackingWorker(QThread)` wrapper owns the 6 `Signal`s + the `@Slot`, constructs a `TrackingEngineCore` wired to `self.*_signal.emit`, and delegates `set_parameters`/`update_parameters`/`stop`/`run`. The transform is **behavior-preserving by construction** — signals are UI observation, not tracking logic, so the CSV trajectory output is provably unaffected. This is gated by the same byte-identical equivalence harness used to verify the whole migration.

**Tech Stack:** Python 3, PySide6 (QtCore), numpy/opencv, pytest, the `tools/equivalence/` harness.

## Global Constraints

- After this change, `grep -rnE "PySide6|QtCore|QThread|Signal|Slot|QMutex" src/hydra_suite/core/` → **EMPTY**. The whole `core/` tree is Qt-free (this fixes the final offender; `core/background` and `core/tracking/optimization` were fixed in the prior plan). This is the whole point.
- **Behavior-preserving by construction.** No change to tracking logic, math, control flow, emit ordering, or emit payloads. The transform is mechanical: `emit`→callback, `QMutex`→`threading.Lock`, class rename. Every existing emit call site maps 1:1 to a callback call at the same point with the same arguments.
- **Preserve the exact wrapper interface** the consumers depend on: the 8-arg `__init__` signature, the 6 signals (`frame_signal(np.ndarray)`, `finished_signal(bool, list, list)`, `progress_signal(int, str)`, `stats_signal(dict)`, `warning_signal(str, str)`, `pose_exported_model_resolved_signal(str)`), `set_parameters(dict)`, `update_parameters(dict)` as a `@Slot`, `stop()`, and the inherited QThread API (`start`, `wait`, `wait(ms)`, `isRunning`, `terminate`).
- **Acceptance gate (mandatory):** the `tools/equivalence/` harness must show **byte-identical** output (pos Δ=0.000, θ Δ=0.000, identical row counts, 0 unmatched) between the pre-split branch base and the post-split branch tip, on **both** the MPS box (`hydra-mps`) and the CUDA box (mehek, `hydra-cuda`), across all 7 fixture clips. This is the attribution methodology already proven for the whole migration (memory `project-migration-verification`).
- Run `make format` before each commit. Tests via `conda run -n hydra-mps python -m pytest ... -q --ignore=tests/test_identity_postprocess.py` (the ignored file has a pre-existing collection error). `make format` may be broken in the base env (corrupted pathspec) — run `black`/`isort` directly inside `hydra-mps` if so.

## Open Questions — RESOLVED (user, 2026-07-19)

1. **Wrapper location.** ✅ RESOLVED: `src/hydra_suite/trackerkit/gui/workers/tracking_worker.py` (matches the `bg_optimizer_worker.py` / `param_optimizer_worker.py` precedent). The headless-CLI import from a `gui/` path is accepted for consistency with the prior splits.
2. **Core class name.** ✅ RESOLVED: `TrackingEngineCore` (mirrors `TrackingOptimizerCore`). The name `TrackingWorker` is retained for the Qt wrapper, so GUI/CLI consumers change only their import path.
3. **Internal API rename.** ✅ RESOLVED: repoint in-repo consumers only — **no back-compat shim**. After the split, `from hydra_suite.core.tracking import TrackingWorker` no longer resolves (core exports `TrackingEngineCore`; the Qt `TrackingWorker` lives in trackerkit). The top-level convenience import `hydra_suite.TrackingWorker` is preserved (repointed to the wrapper). The user confirmed no out-of-repo script imports `TrackingWorker` from `core`.

---

## File Structure

- `src/hydra_suite/core/tracking/worker.py` — **modify**: rename `class TrackingWorker(QThread)` → `class TrackingEngineCore:` (plain); remove the PySide6 import; add `import threading`; 6 `Signal` attrs → 6 `__init__` callback params + `_emit_*` guarded helpers; `QMutex`→`threading.Lock`; drop `@Slot`; `run()`→`run_tracking()` (the outer exception guard moves to the wrapper). Every method body stays byte-for-byte except emit/mutex/self-ref substitutions.
- `src/hydra_suite/trackerkit/gui/workers/tracking_worker.py` — **create**: `class TrackingWorker(QThread)` — the 6 Signals, `@Slot(dict) update_parameters`, 8-arg `__init__` that builds a `TrackingEngineCore` wired to emits, delegating methods, and `run()` with the QThread exception-safety fallback.
- `src/hydra_suite/core/tracking/__init__.py` — **modify**: export `TrackingEngineCore` (drop `TrackingWorker`).
- `src/hydra_suite/core/__init__.py` — **modify**: export `TrackingEngineCore` (drop `TrackingWorker`).
- `src/hydra_suite/__init__.py` — **modify**: repoint the lazy `TrackingWorker` re-export (lines ~40, 62-63) to the trackerkit wrapper.
- `src/hydra_suite/trackerkit/headless_tracking.py` — **modify**: import `TrackingWorker` from the wrapper module (line ~12).
- `src/hydra_suite/trackerkit/gui/orchestrators/tracking.py` — **modify**: repoint the two `from hydra_suite.core.tracking import TrackingWorker` imports (lines ~3800, ~4260) to the wrapper module.
- Tests — **modify**: `tests/test_tracking_worker_helpers.py`, `tests/test_tracking_worker_realtime_live_features.py`, `tests/test_worker_real_inference_integration.py`, `tests/test_trackerkit_workers_smoke.py` (repoint to `TrackingEngineCore`); `tests/test_trackerkit_tracking_orchestrator_dialogs.py` (repoint the 3 monkeypatch targets to the wrapper path). **Create**: `tests/test_tracking_engine_core_qtfree.py` (guard + wrapper smoke).

### Qt surface inventory (from the current file — use these exact sites)

- **Class decl:** `class TrackingWorker(QThread):` @106. **Import:** `from PySide6.QtCore import QMutex, QThread, Signal, Slot` @16.
- **6 Signals:** `frame_signal` @112, `finished_signal` @113, `progress_signal` @114, `stats_signal` @115, `warning_signal` @116, `pose_exported_model_resolved_signal` @117.
- **QMutex:** `self.params_mutex = QMutex()` @139; lock/unlock in `set_parameters` @162-164, `update_parameters` @169-171, `get_current_params` @176-178.
- **`@Slot(dict)`:** on `update_parameters` @166.
- **emit sites (map each to `self._emit_<name>(...)`):**
  - `frame_signal.emit`: @461
  - `finished_signal.emit`: @696, @718, @873, @984, @994, @1018, @1036, @1127, @1212, @4175
  - `progress_signal.emit`: @1284, @1287, @1341, @1368, @3868, @4372
  - `stats_signal.emit`: @3986
  - `warning_signal.emit`: @810, @1204, @4035, and the lambda `warning_cb=lambda title, msg: self.warning_signal.emit(title, msg)` @4027
  - `pose_exported_model_resolved_signal.emit`: **none** (declared + connected by orchestrator @3882/@4278, never emitted — preserve as a no-op wire).
- **`_stop_requested` (plain bool — STAYS a plain attr, no change):** init @147, set in `stop()` @390, read @412/@423/@448/@454/@702/@4018/@4176, and `should_stop=lambda: self._stop_requested` @1197.
- **`run()`/`_run_impl()` seam:** `run()` @~672 (try/except → `finished_signal.emit(False,[],[])` fallback) delegates to `_run_impl()` @~700. The core makes **zero** QThread-instance calls on `self` (verified: only `self.start_time` matches, no `msleep`/`isInterruptionRequested`/`wait`/`currentThread`), so the plain-class rename is safe.
- **Duck-typed collaborators (no Qt, keep as-is):** `csv_writer_thread.enqueue(...)` @3482/@3522/@3679; `frame_prefetcher` (`FramePrefetcher` from utils) `.start()/.read()/.stop()` @409-420, @4020-4022.

---

## Task 1: Convert the core class to a Qt-free `TrackingEngineCore`

Rename `TrackingWorker(QThread)` → plain `TrackingEngineCore`, remove all Qt, inject the 6 emits as callbacks, swap the mutex. Behavior-preserving.

**Files:**
- Modify: `src/hydra_suite/core/tracking/worker.py`
- Test: `tests/test_tracking_engine_core_qtfree.py` (new), and repoint `tests/test_tracking_worker_helpers.py` + `tests/test_trackerkit_workers_smoke.py`.

**Interfaces:**
- Consumes: nothing new (all existing core helpers stay).
- Produces:
  ```python
  class TrackingEngineCore:
      def __init__(
          self,
          video_path,
          csv_writer_thread=None,
          video_output_path=None,
          backward_mode=False,
          detection_cache_path=None,
          preview_mode=False,
          use_cached_detections=False,
          *,
          on_frame=None,                 # Callable[[np.ndarray], None]
          on_finished=None,              # Callable[[bool, list, list], None]
          on_progress=None,              # Callable[[int, str], None]
          on_stats=None,                 # Callable[[dict], None]
          on_warning=None,               # Callable[[str, str], None]
          on_pose_model_resolved=None,   # Callable[[str], None]  (currently never called)
      ): ...
      def set_parameters(self, p: dict) -> None: ...
      def update_parameters(self, new_params: dict) -> None: ...   # plain method, no @Slot
      def get_current_params(self) -> dict: ...
      def stop(self) -> None: ...
      def run_tracking(self) -> None: ...   # was _run_impl(); the pipeline
  ```
  Guarded emit helpers: `_emit_frame/_emit_finished/_emit_progress/_emit_stats/_emit_warning/_emit_pose_model_resolved`, each `if self._on_X is not None: self._on_X(...)`.

- [ ] **Step 1: Write the failing guard + helper test**

```python
# tests/test_tracking_engine_core_qtfree.py
"""core/tracking is Qt-free; TrackingEngineCore is a plain, callback-driven engine."""
import ast
from pathlib import Path


def test_core_tracking_imports_no_qt():
    import hydra_suite.core.tracking as pkg

    pkg_dir = Path(pkg.__file__).parent
    offenders = []
    for py in pkg_dir.rglob("*.py"):
        tree = ast.parse(py.read_text(), filename=str(py))
        for node in ast.walk(tree):
            mod = None
            if isinstance(node, ast.ImportFrom):
                mod = node.module
            elif isinstance(node, ast.Import):
                mod = ",".join(a.name for a in node.names)
            if mod and any(q in mod for q in ("PySide6", "QtCore")):
                offenders.append(f"{py.relative_to(pkg_dir)}:{node.lineno}")
    assert not offenders, "core/tracking must not import Qt: " + "; ".join(offenders)


def test_engine_core_is_plain_and_callback_driven():
    from hydra_suite.core.tracking.worker import TrackingEngineCore

    seen = []
    core = TrackingEngineCore("dummy.mp4", on_progress=lambda pct, msg: seen.append((pct, msg)))
    # Not a QThread — instantiable with no Qt event loop.
    assert not hasattr(type(core), "start")
    # Guarded emit helper routes to the injected callback.
    core._emit_progress(42, "hi")
    assert seen == [(42, "hi")]
    # stop() sets the plain flag and stops an active prefetcher.
    class _FakePref:
        stopped = False
        def stop(self):
            self.stopped = True
    core.frame_prefetcher = _FakePref()
    core.stop()
    assert core._stop_requested is True
    assert core.frame_prefetcher.stopped is True


def test_engine_core_param_lock_roundtrip():
    from hydra_suite.core.tracking.worker import TrackingEngineCore

    core = TrackingEngineCore("dummy.mp4")
    core.set_parameters({"A": 1})
    core.update_parameters({"A": 2, "B": 3})
    assert core.get_current_params() == {"A": 2, "B": 3}
```

- [ ] **Step 2: Run to verify it fails**

Run: `conda run -n hydra-mps python -m pytest tests/test_tracking_engine_core_qtfree.py -q`
Expected: FAIL — `TrackingEngineCore` doesn't exist yet / module still imports Qt.

- [ ] **Step 3: Apply the mechanical transform to `worker.py`**

Using the Qt surface inventory above, apply ONLY these substitutions — do not touch any tracking logic, math, loop, or helper body:

1. Line 16: delete `from PySide6.QtCore import QMutex, QThread, Signal, Slot`. Add `import threading` near the other stdlib imports.
2. Line 106: `class TrackingWorker(QThread):` → `class TrackingEngineCore:`.
3. Lines 112-117: delete the 6 `Signal(...)` class attributes.
4. `__init__` (119-158): drop the `parent=None` param and `super().__init__(parent)`; add the 6 keyword-only callback params (`on_frame=None, ... on_pose_model_resolved=None`) and store them as `self._on_frame = on_frame`, etc. Change `self.params_mutex = QMutex()` (139) → `self._params_lock = threading.Lock()`.
5. Add 6 guarded emit helpers (place right after `__init__`):
   ```python
   def _emit_frame(self, rgb):
       if self._on_frame is not None:
           self._on_frame(rgb)
   def _emit_finished(self, success, fps_list, full_traj):
       if self._on_finished is not None:
           self._on_finished(success, fps_list, full_traj)
   def _emit_progress(self, pct, msg):
       if self._on_progress is not None:
           self._on_progress(pct, msg)
   def _emit_stats(self, stats):
       if self._on_stats is not None:
           self._on_stats(stats)
   def _emit_warning(self, title, msg):
       if self._on_warning is not None:
           self._on_warning(title, msg)
   def _emit_pose_model_resolved(self, path):
       if self._on_pose_model_resolved is not None:
           self._on_pose_model_resolved(path)
   ```
6. `set_parameters` (160-164), `update_parameters` (166-172), `get_current_params` (174-179): remove the `@Slot(dict)` decorator (166); replace `self.params_mutex.lock()` / `self.params_mutex.unlock()` pairs with `with self._params_lock:` blocks (guard the same statements). Keep the `logger.info` in `update_parameters`.
7. Replace every emit call site (exact lines in the inventory) `self.<name>_signal.emit(<args>)` → `self._emit_<name>(<args>)`. Including the lambda @4027: `warning_cb=lambda title, msg: self._emit_warning(title, msg)`.
8. `run()`/`_run_impl()` seam: **delete the old `run()` method** (the try/except fallback wrapper @~672 — it moves to the Qt wrapper in Task 2) and **rename `_run_impl` → `run_tracking`** (keep `# noqa: C901`). The internal `self._stop_requested = False` reset at the top of the old `_run_impl` stays. Do NOT wrap `run_tracking` in a try/except here — an exception propagates to the wrapper, which owns the QThread exception-safety guard.
9. Leave `_stop_requested`, `stop()`, `csv_writer_thread`, `frame_prefetcher`, and every other method unchanged.

- [ ] **Step 4: Repoint the two existing tests that instantiate the class directly**

- `tests/test_tracking_worker_helpers.py`: it loads the module via `load_src_module` and uses `mod.TrackingWorker(...)` (lines 274, 304, 314, 332, 354, 386, 407, 424) and `mod.TrackingWorker._should_emit_visualization_frame` / `_realtime_yolo_micro_batch_size` (447-475). Replace `mod.TrackingWorker` → `mod.TrackingEngineCore` throughout. (These test core helpers, which now live on the core class; the plain class instantiates without Qt, so the cv2 stub is still sufficient.)
- `tests/test_trackerkit_workers_smoke.py`: `test_tracking_worker_stop_stops_active_prefetcher` imports `from hydra_suite.core.tracking.worker import TrackingWorker` (line 5) and instantiates it (14). Replace with `TrackingEngineCore` (the `stop()`+prefetcher logic lives in core). Keep the assertions.
- `tests/test_worker_real_inference_integration.py:275,283`: `from hydra_suite.core.tracking.worker import TrackingWorker` + `TrackingWorker.__new__(TrackingWorker)` → `TrackingEngineCore`.

- [ ] **Step 5: Run tests + Qt-free grep**

Run:
```bash
conda run -n hydra-mps python -m pytest tests/test_tracking_engine_core_qtfree.py tests/test_tracking_worker_helpers.py tests/test_trackerkit_workers_smoke.py tests/test_worker_real_inference_integration.py -q --ignore=tests/test_identity_postprocess.py
conda run -n hydra-mps python -c "import hydra_suite.core.tracking.worker as w; assert hasattr(w,'TrackingEngineCore') and not hasattr(w,'TrackingWorker'); print('core engine OK, no Qt name')"
grep -rnE "PySide6|QtCore|QThread|Signal|Slot|QMutex" src/hydra_suite/core/tracking/worker.py
```
Expected: tests pass; grep prints nothing. (Consumers importing `TrackingWorker` from core break until Tasks 2-3 — expected mid-refactor. `core/tracking/__init__.py` still says `from .worker import TrackingWorker`; that import now fails — fixed in Task 3. To keep this step's import check green, it targets `worker.py` directly, not the package.)

- [ ] **Step 6: Commit**

```bash
make format
git add src/hydra_suite/core/tracking/worker.py tests/test_tracking_engine_core_qtfree.py tests/test_tracking_worker_helpers.py tests/test_trackerkit_workers_smoke.py tests/test_worker_real_inference_integration.py
git commit -m "refactor(core): extract Qt-free TrackingEngineCore from TrackingWorker(QThread)"
```

---

## Task 2: Create the `TrackingWorker(QThread)` wrapper in trackerkit

**Files:**
- Create: `src/hydra_suite/trackerkit/gui/workers/tracking_worker.py`
- Test: `tests/test_tracking_engine_core_qtfree.py` (extend with a wrapper smoke test).

**Interfaces:**
- Consumes: `TrackingEngineCore` from `hydra_suite.core.tracking.worker`.
- Produces: `class TrackingWorker(QThread)` with the exact public surface the orchestrator + headless CLI use (see Global Constraints).

- [ ] **Step 1: Write the wrapper**

```python
# src/hydra_suite/trackerkit/gui/workers/tracking_worker.py
"""Qt wrapper for the Qt-free TrackingEngineCore (core/tracking/worker.py).

Owns the QThread + the 6 Signals + the update_parameters Slot, and forwards the
engine's callbacks to those signals. Keeps the exact interface the tracking
orchestrator and the headless CLI depend on."""
from __future__ import annotations

import logging

import numpy as np
from PySide6.QtCore import QThread, Signal, Slot

from hydra_suite.core.tracking.worker import TrackingEngineCore

logger = logging.getLogger(__name__)


class TrackingWorker(QThread):
    frame_signal = Signal(np.ndarray)
    finished_signal = Signal(bool, list, list)
    progress_signal = Signal(int, str)
    stats_signal = Signal(dict)
    warning_signal = Signal(str, str)
    pose_exported_model_resolved_signal = Signal(str)

    def __init__(
        self,
        video_path,
        csv_writer_thread=None,
        video_output_path=None,
        backward_mode=False,
        detection_cache_path=None,
        preview_mode=False,
        use_cached_detections=False,
        parent=None,
    ):
        super().__init__(parent)
        self._core = TrackingEngineCore(
            video_path,
            csv_writer_thread=csv_writer_thread,
            video_output_path=video_output_path,
            backward_mode=backward_mode,
            detection_cache_path=detection_cache_path,
            preview_mode=preview_mode,
            use_cached_detections=use_cached_detections,
            on_frame=lambda rgb: self.frame_signal.emit(rgb),
            on_finished=lambda ok, fps, traj: self.finished_signal.emit(ok, fps, traj),
            on_progress=lambda pct, msg: self.progress_signal.emit(int(pct), msg),
            on_stats=lambda stats: self.stats_signal.emit(stats),
            on_warning=lambda title, msg: self.warning_signal.emit(title, msg),
            on_pose_model_resolved=lambda p: self.pose_exported_model_resolved_signal.emit(p),
        )

    # --- delegation: keep the exact public surface consumers use ---
    def set_parameters(self, p: dict) -> None:
        self._core.set_parameters(p)

    @Slot(dict)
    def update_parameters(self, new_params: dict) -> None:
        self._core.update_parameters(new_params)

    def get_current_params(self) -> dict:
        return self._core.get_current_params()

    def stop(self) -> None:
        self._core.stop()

    @property
    def _stop_requested(self) -> bool:  # some call sites / tests read this
        return self._core._stop_requested

    def run(self) -> None:
        """QThread entry point. PySide6 silently swallows exceptions that escape
        a QThread.run() override, which would leave finished_signal unemitted and
        hang callers blocked on it (headless CLI's QEventLoop). Guard exactly as
        the old TrackingWorker.run() did."""
        try:
            self._core.run_tracking()
        except Exception:
            logger.exception(
                "Unhandled exception in TrackingWorker.run(); emitting "
                "finished_signal(False, ...) so callers waiting on it "
                "(e.g. the headless CLI's QEventLoop) don't hang forever."
            )
            self.finished_signal.emit(False, [], [])
```

Notes for the implementer:
- Some consumers/tests set/read attributes that were previously plain worker fields (e.g. `worker.frame_prefetcher`, `worker.individual_properties_cache_path`, `worker.detection_cache_path`). These now live on `self._core`. Grep for `tracking_worker.` attribute access in `trackerkit/` and `tests/`; for any field read/written directly on the worker OTHER than the delegated methods/signals above, forward it (add a `@property`/setter that proxies `self._core.<field>`, or repoint the caller to `worker._core.<field>`). The confirmed public surface used by orchestrator + headless is `__init__`, the 6 signals, `set_parameters`, `update_parameters`, `stop`, `start`, `wait`, `isRunning`, `terminate` — those are all covered. Verify the smoke test in Step 2 and the orchestrator import in Task 3 flush out any stragglers.

- [ ] **Step 2: Extend the test with a wrapper smoke test**

```python
def test_wrapper_delegates_and_exposes_signals(qtbot=None):
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from hydra_suite.trackerkit.gui.workers.tracking_worker import TrackingWorker

    w = TrackingWorker("dummy.mp4", preview_mode=True)
    # QThread + all 6 signals present.
    assert hasattr(w, "start") and hasattr(w, "wait") and hasattr(w, "isRunning")
    for sig in ("frame_signal", "finished_signal", "progress_signal",
                "stats_signal", "warning_signal", "pose_exported_model_resolved_signal"):
        assert hasattr(w, sig)
    # Delegation reaches the core.
    w.set_parameters({"X": 1})
    assert w.get_current_params() == {"X": 1}
    w.stop()
    assert w._stop_requested is True
```

- [ ] **Step 3: Run + import check**

Run:
```bash
QT_QPA_PLATFORM=offscreen conda run -n hydra-mps python -m pytest tests/test_tracking_engine_core_qtfree.py -q
QT_QPA_PLATFORM=offscreen conda run -n hydra-mps python -c "from hydra_suite.trackerkit.gui.workers.tracking_worker import TrackingWorker; print('wrapper OK')"
```
Expected: pass.

- [ ] **Step 4: Commit**

```bash
make format
git add src/hydra_suite/trackerkit/gui/workers/tracking_worker.py tests/test_tracking_engine_core_qtfree.py
git commit -m "feat(trackerkit): Qt wrapper for the Qt-free TrackingEngineCore"
```

---

## Task 3: Repoint all consumers + re-exports

**Files:**
- Modify: `src/hydra_suite/core/tracking/__init__.py`, `src/hydra_suite/core/__init__.py`, `src/hydra_suite/__init__.py`, `src/hydra_suite/trackerkit/headless_tracking.py`, `src/hydra_suite/trackerkit/gui/orchestrators/tracking.py`, `tests/test_trackerkit_tracking_orchestrator_dialogs.py`.

- [ ] **Step 1: Core re-exports → `TrackingEngineCore`**

- `src/hydra_suite/core/tracking/__init__.py`: `from .worker import TrackingWorker` → `from .worker import TrackingEngineCore`; `__all__ = ["TrackingWorker"]` → `["TrackingEngineCore"]`.
- `src/hydra_suite/core/__init__.py`: line 19 `from .tracking.worker import TrackingWorker` → `from .tracking.worker import TrackingEngineCore`; update the `__all__` entry (line ~22) `"TrackingWorker"` → `"TrackingEngineCore"`.

- [ ] **Step 2: Top-level lazy re-export → the wrapper**

`src/hydra_suite/__init__.py`: keep `"TrackingWorker"` in `__all__` (line ~40), but repoint the lazy loader (lines ~62-63):
```python
    if name == "TrackingWorker":
        return import_module("hydra_suite.trackerkit.gui.workers.tracking_worker").TrackingWorker
```
(So `hydra_suite.TrackingWorker` still works and returns the Qt wrapper.)

- [ ] **Step 3: Repoint the app-layer consumers**

- `src/hydra_suite/trackerkit/headless_tracking.py:12`: `from hydra_suite.core.tracking import TrackingWorker` → `from hydra_suite.trackerkit.gui.workers.tracking_worker import TrackingWorker`.
- `src/hydra_suite/trackerkit/gui/orchestrators/tracking.py`: the two local imports at ~3800 and ~4260, `from hydra_suite.core.tracking import TrackingWorker` → `from hydra_suite.trackerkit.gui.workers.tracking_worker import TrackingWorker`. (The construction sites @3867/@4262, `set_parameters`, all 6 signal `.connect(...)` incl. `pose_exported_model_resolved_signal` @3882/@4278, `parameters_changed.connect(worker.update_parameters)` @4272, and `start()/wait()/isRunning()/terminate()/stop()` are UNCHANGED — the wrapper preserves them.)

- [ ] **Step 4: Repoint the orchestrator-dialog test monkeypatches**

`tests/test_trackerkit_tracking_orchestrator_dialogs.py` patches `"hydra_suite.core.tracking.TrackingWorker"` with a `FakeTrackingWorker` (lines ~656, ~773, ~879). The orchestrator now imports `TrackingWorker` from `hydra_suite.trackerkit.gui.workers.tracking_worker`, so change all three monkeypatch targets to `"hydra_suite.trackerkit.gui.workers.tracking_worker.TrackingWorker"`. The `FakeTrackingWorker` classes (which declare the 6 signals incl. `pose_exported_model_resolved_signal`) are unchanged.

- [ ] **Step 5: Verify — whole `core/` Qt-free + no stale importers + suites green**

Run:
```bash
grep -rnE "PySide6|QtCore|QThread|Signal|Slot|QMutex" src/hydra_suite/core/ ; echo "exit=$?"
grep -rn "from hydra_suite.core.tracking import TrackingWorker" src/ tests/
grep -rn "hydra_suite.core.tracking.TrackingWorker" src/ tests/
QT_QPA_PLATFORM=offscreen conda run -n hydra-mps python -m pytest tests/test_trackerkit_tracking_orchestrator_dialogs.py tests/test_tracking_engine_core_qtfree.py tests/test_tracking_worker_helpers.py tests/test_tracking_worker_realtime_live_features.py tests/test_trackerkit_workers_smoke.py -q --ignore=tests/test_identity_postprocess.py
QT_QPA_PLATFORM=offscreen conda run -n hydra-mps python -c "import hydra_suite; hydra_suite.TrackingWorker; import hydra_suite.trackerkit.headless_tracking, hydra_suite.trackerkit.gui.orchestrators.tracking; print('all consumers import OK')"
```
Expected: first grep EMPTY (whole `core/` tree Qt-free — the milestone); the two stale-importer greps EMPTY; all suites pass; imports OK. Also repoint `tests/test_tracking_worker_realtime_live_features.py:311` (`worker_mod.TrackingWorker(...)`) if `worker_mod` is the core module — point it at the core class or the wrapper depending on what it exercises (helpers → `TrackingEngineCore`; a real QThread run → wrapper). Inspect and choose; keep the test meaningful.

- [ ] **Step 6: Commit**

```bash
make format
git add -A
git commit -m "refactor(trackerkit): repoint TrackingWorker consumers at the relocated Qt wrapper; core/ is now Qt-free"
```

---

## Task 4: Widen the Qt-free guard to the whole `core/` tree

Lock in the invariant so a future edit can't reintroduce Qt into any part of `core/`.

**Files:**
- Test: `tests/test_tracking_engine_core_qtfree.py` (add a whole-`core` guard), or extend the existing `tests/test_optimizer_workers_core.py::test_core_tracking_optimization_imports_no_qt`.

- [ ] **Step 1: Add the whole-`core` guard test**

```python
def test_entire_core_tree_imports_no_qt():
    import ast
    from pathlib import Path
    import hydra_suite.core as core_pkg

    root = Path(core_pkg.__file__).parent
    offenders = []
    for py in root.rglob("*.py"):
        tree = ast.parse(py.read_text(), filename=str(py))
        for node in ast.walk(tree):
            mod = None
            if isinstance(node, ast.ImportFrom):
                mod = node.module
            elif isinstance(node, ast.Import):
                mod = ",".join(a.name for a in node.names)
            if mod and any(q in mod for q in ("PySide6", "QtCore", "QtGui", "QtWidgets")):
                offenders.append(f"{py.relative_to(root)}:{node.lineno}")
    assert not offenders, "core/ must be Qt-free: " + "; ".join(offenders)
```

- [ ] **Step 2: Run — passes** (both prior offenders + this one fixed).

Run: `conda run -n hydra-mps python -m pytest tests/test_tracking_engine_core_qtfree.py -q`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
make format
git add tests/test_tracking_engine_core_qtfree.py
git commit -m "test(core): guard the entire core/ tree against Qt imports"
```

---

## Task 5: Equivalence parity gate (acceptance — byte-identical, both platforms)

This is the point of the whole exercise: prove the split changed nothing. Reuse the exact attribution methodology from the migration verification. **Do not skip and do not accept < byte-identical.**

**Method:** run the `tools/equivalence/` harness with the branch base (`git merge-base main HEAD`, pre-split) as `MAIN_SRC` and the branch tip (post-split) as `WT_SRC`; assert `compare.py` reports pos Δ=0.000, θ Δ=0.000, identical rows, 0 unmatched for every clip. This is an attribution run (our-change vs baseline), not a legacy-vs-migration run.

- [ ] **Step 1: Prepare two worktrees (base vs tip)**

```bash
BASE=$(git merge-base main HEAD)
git worktree add --detach .worktrees/pretty-base "$BASE"
git worktree add --detach .worktrees/pretty-tip  HEAD
```

- [ ] **Step 2: MPS run (this box)** — **conda MUST be active** (the SLEAP service spawns `conda run -n sleap`; a bare shell yields EMPTY CSVs that falsely compare "θ=0 EQUIVALENT" — see memory `project-migration-verification`).

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate hydra-mps
REPO=$PWD MAIN_SRC=$PWD/.worktrees/pretty-base/src WT_SRC=$PWD/.worktrees/pretty-tip/src \
  WT=$PWD OUT=/tmp/equiv_ttw_mps RUNTIME=mps \
  bash tools/equivalence/run_matrix.sh
```
For each of the 7 clips, run `tools/equivalence/compare.py` on `legacy` (base) vs `new_a` (tip) `_tracking_final.csv` (and `_forward.csv` for forward-only clips). **Every clip must be `EQUIVALENT ✅` with θ max=0.000, pos p99=0.000, 0 unmatched.** (Determinism `new_a` vs `new_b` is a secondary sanity check, already known to be 0.)

- [ ] **Step 3: CUDA run (mehek)** — replicate on the CUDA box, conda active:

```bash
ssh rutalab@mehek.taild08eb9.ts.net  # then, in a script sourced with conda:
#   source /home/rutalab/mambaforge/etc/profile.d/conda.sh; conda activate hydra-cuda
#   create base+tip worktrees under /home/rutalab/hydra-suite/.wt/, then:
#   REPO=... MAIN_SRC=.wt/base/src WT_SRC=.wt/tip/src WT=... OUT=... RUNTIME=cuda bash tools/equivalence/run_matrix.sh
```
Same acceptance: every clip byte-identical. Note pose/cnn clips REQUIRE the `sleap` env on the box + conda on PATH; verify the `_forward.csv` row counts are non-zero (an empty file silently "passes"). Cross-check row counts > 0 before trusting an `EQUIVALENT`.

- [ ] **Step 4: Record the result + clean up worktrees**

Append the verdict table (per clip, per platform) to the PR/commit description. Then:
```bash
git worktree remove .worktrees/pretty-base --force
git worktree remove .worktrees/pretty-tip  --force
git worktree prune
# and the equivalent on mehek; rm the /tmp/equiv_ttw_* output dirs on both boxes
```

- [ ] **Step 5: If any clip is NOT byte-identical, STOP.** A non-zero diff means the transform was not behavior-preserving — bisect the emit/mutex substitution (most likely a mis-mapped emit site, a dropped guard, or a `with self._params_lock:` block that changed which statements it covers). Do not merge until byte-identical on both platforms.

---

## Final verification (whole plan)

- [ ] `grep -rnE "PySide6|QtCore|QtGui|QtWidgets|QThread|Signal|Slot|QMutex" src/hydra_suite/core/` → **EMPTY** (whole `core/` Qt-free).
- [ ] `conda run -n hydra-mps python -m pytest tests/ -q --ignore=tests/test_identity_postprocess.py` (full suite green; at minimum every test touching the worker/orchestrator/headless path).
- [ ] `hydra_suite.TrackingWorker`, `hydra_suite.trackerkit.headless_tracking`, and `hydra_suite.trackerkit.gui.orchestrators.tracking` all import.
- [ ] Task 5 parity gate: byte-identical on MPS **and** CUDA, all 7 clips, row counts > 0.
- [ ] `make format-check`.
- [ ] Update memory `project-detector-retirement-done` (the deferred `TrackingWorker` split is now DONE) and `project-migration-verification`.

---

## Self-Review notes

- **Goal check:** the Task 4 whole-`core` guard + the Task 3 final grep enforce "core/ imports no Qt". The violation is removed by relocating ONLY the Qt (QThread base, Signals, Slot, QMutex, emits) to a thin wrapper; every tracking method stays in core.
- **Behavior preservation:** the transform is mechanical (emit→callback at identical sites/args; `QMutex`→`threading.Lock` over identical critical sections; `@Slot` removed from core but re-declared on the wrapper; `run()` exception guard preserved on the wrapper). Signals are UI observation, not tracking logic — the trajectory/CSV output cannot change. The byte-identical equivalence gate (Task 5) is the empirical proof, matching the methodology that verified the whole migration.
- **Concurrency:** `update_parameters` is connected cross-thread (`parameters_changed.connect(worker.update_parameters)` @4272) and runs on the object's owning (GUI) thread while `run_tracking` reads params on the worker thread — genuine concurrency. `threading.Lock` provides the same mutual exclusion as `QMutex` over the same tiny dict swaps. The wrapper keeps `@Slot(dict)` so queued Qt connections still marshal correctly.
- **`pose_exported_model_resolved_signal`:** declared + connected by the orchestrator but never emitted today. Preserved as a no-op: the wrapper declares it, the core exposes an unused `on_pose_model_resolved` hook. Byte-identical (never fires, same as now).
- **Attribute stragglers:** the one residual risk is a consumer/test reading a plain field directly off the worker (now on `self._core`). Task 2 Step 1's note + Task 3's import/test sweep flush these out; add proxy `@property`s as needed.
- **Type/name consistency:** core class `TrackingEngineCore` (8 ctor args + 6 keyword callbacks + `run_tracking`); wrapper `TrackingWorker(QThread)` (8 ctor args incl. `parent`, 6 Signals, `@Slot update_parameters`, delegating `set_parameters`/`get_current_params`/`stop`/`_stop_requested` property/`run`). Re-exports: `core.tracking.TrackingEngineCore`, `trackerkit.gui.workers.tracking_worker.TrackingWorker`, `hydra_suite.TrackingWorker` (→ wrapper).
- **Scope:** does NOT migrate onto `widgets/workers.BaseWorker` (keeps plain QThread to preserve the exact 6-signal set and minimize risk — optional future follow-up), and does NOT touch tracking logic.
