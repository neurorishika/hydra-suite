# GUI Post-Tracking Cutover (Slice 5) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the TrackerKit GUI drive the Qt-free `TrackingSessionCore.run_post_tracking` for all post-tracking (via a new `SessionWorker`), deleting the ~45 duplicated orchestrator methods + 11 `main_window.py` wrappers, so GUI output is byte-identical to the CLI by construction.

**Architecture:** Keep the GUI's two async `TrackingWorker` QThread passes (they already write the raw `_forward.csv`/`_backward.csv` that `run_post_tracking` re-reads). After both passes, a new `SessionWorker(BaseWorker)` runs `run_post_tracking` on those disk paths — replacing the per-pass `PostProcessWorker` + `MergeWorker` + the whole `_finish_tracking_session` chain. No `core/` change → the Slice-4 CLI equivalence gate stays valid.

**Tech Stack:** Python 3, PySide6 (QThread/Signal), pandas, pytest (no `pytest-qt`; offscreen Qt via `QT_QPA_PLATFORM=offscreen`). Conda env `hydra-mps`.

## Global Constraints

- **Design spec:** `docs/superpowers/specs/2026-08-04-gui-post-tracking-cutover-design.md`. This plan is the deferred Slice-2 Task 9, re-planned.
- **Do NOT modify `core/`.** The GUI must reproduce the CLI's `run_post_tracking` usage; the Slice-4 CLI gate must remain valid (a `core/` diff would invalidate it). Verify `git diff --stat` touches no `src/hydra_suite/core/` file.
- **Line numbers are anchors** from worktree `.worktrees/headless-qt-free` @ `a4c34b4b`; `tracking.py` is 3788 lines and shifts as methods are deleted — re-derive every cited line against live source before editing (a proven Slice-1..4 hazard: plans carry stale line numbers/keys).
- **Commit as the configured git user; NO `Co-Authored-By` trailer.**
- **Format before each commit:** `make format` is broken → `conda run -n hydra-mps black <files> && conda run -n hydra-mps isort <files>` on changed files.
- **Run tests with exactly:** `PYTHONPATH=$PWD/src KMP_DUPLICATE_LIB_OK=TRUE conda run -n hydra-mps python -m pytest <path> -q --ignore=tests/test_identity_postprocess.py` (from inside the worktree; env `hydra-mps`).
- **Full `pytest tests/` never completes** (classkit modal hang + SIGABRT) — batch per-file and judge failures as a delta vs the pre-task baseline, never in isolation.
- **When deleting a GUI method, grep `src/` AND `tests/`** for the symbol before removing it (a Slice-3 lesson: deletions break tests in other files, esp. `tests/test_trackerkit_tracking_orchestrator_dialogs.py`).

**Verified interfaces (from source @ `a4c34b4b`):**
- `BaseWorker(QThread)` (`src/hydra_suite/widgets/workers.py:6`): subclasses implement `execute()`; base signals `progress=Signal(int)`, `status=Signal(str)`, `error=Signal(str)`; `run()` (31-36) wraps `execute()` in try/except → `error`; `finished` is Qt-inherited (**never redefine it**).
- `TrackingSessionCore.__init__(self, *, video_path, config, params, paths, callbacks=None)` (`core/tracking/session.py:164`); `paths` is a **dict** read for keys `raw_csv_path`, `final_csv_path`, `detection_cache_path`, `individual_properties_cache_path`, `detected_properties_cache_path`, `detected_cnn_cache_paths`, `individual_dataset_dir`, `final_media_video_dir`, `interpolated_roi_npz_path`, `source_video_fps`.
- `SessionCallbacks(progress: Callable[[int,str],None], status: Callable[[str],None], warning: Callable[[str,str],None], stage_changed: Callable[[str],None], should_stop: Callable[[],bool])` (`session.py:141`), all no-op defaults.
- `SessionResult(success, final_csv_path, rich_export_path, media_paths, dataset_result, summary_lines, error)` (`session.py:150`).
- `run_post_tracking(self, forward_trajectories, backward_trajectories=None) -> SessionResult` (`session.py:452`) — **ignores its DataFrame args; re-reads raw `_forward.csv`/`_backward.csv` from disk** (forward-only reads `raw_csv_path`).
- CLI reference construction: `headless_tracking.py:288-308` (SessionCallbacks + `TrackingSessionCore(...)` + `run_post_tracking`).
- GUI MainWindow attrs available at finish: `self._panels.setup.csv_line.text()` (raw_csv_path, live), `self._mw.current_detection_cache_path` (set `start_tracking_on_video`), `self._mw.current_individual_properties_cache_path` / `current_detected_properties_cache_path` / `current_detected_cnn_cache_paths` (set in `_collect_worker_props_path`, `tracking.py:1705-1746`), `self._mw._session_final_csv_path`, `self._mw._session_fps_list`.

---

## Task 1: `SessionWorker(BaseWorker)`

**Files:**
- Create: `src/hydra_suite/trackerkit/gui/workers/session_worker.py`
- Test: `tests/test_session_worker.py` (create)

**Interfaces:**
- Consumes: `TrackingSessionCore`, `SessionCallbacks`, `SessionResult` (`core/tracking/session.py`); `BaseWorker` (`widgets/workers.py`).
- Produces: `class SessionWorker(BaseWorker)` with `__init__(self, *, video_path, config, params, paths)`, signals `progress_signal = Signal(int, str)`, `finished_signal = Signal(object)` (emits `SessionResult`), `error_signal = Signal(str)`, `warning_signal = Signal(str, str)`; methods `stop()`, `_should_stop() -> bool`, `execute()`. Consumed by Task 2.

- [ ] **Step 1: Write the failing test**

Create `tests/test_session_worker.py`:

```python
"""SessionWorker wraps TrackingSessionCore.run_post_tracking as a BaseWorker."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication

from hydra_suite.core.tracking import session as session_module
from hydra_suite.core.tracking.session import SessionResult
from hydra_suite.trackerkit.gui.workers.session_worker import SessionWorker


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _result() -> SessionResult:
    return SessionResult(
        success=True,
        final_csv_path="/tmp/out_final.csv",
        rich_export_path=None,
        media_paths=[],
        dataset_result=None,
        summary_lines=["final_csv=out_final.csv"],
        error=None,
    )


def test_execute_runs_service_and_emits_result(qapp, monkeypatch):
    captured = {"init": None, "ran": False, "callbacks": None}

    def _fake_init(self, *, video_path, config, params, paths, callbacks=None):
        captured["init"] = dict(video_path=video_path, paths=paths)
        captured["callbacks"] = callbacks

    def _fake_run(self, forward, backward=None):
        captured["ran"] = True
        # exercise the bridged callbacks so the test proves they are wired
        self.callbacks.progress(50, "half")
        self.callbacks.warning("W", "msg")
        return _result()

    monkeypatch.setattr(session_module.TrackingSessionCore, "__init__", _fake_init)
    monkeypatch.setattr(session_module.TrackingSessionCore, "run_post_tracking", _fake_run)
    # the fake __init__ skips storing callbacks; restore it so _fake_run can use self.callbacks
    real_init = session_module.TrackingSessionCore.__init__

    worker = SessionWorker(
        video_path="v.mp4",
        config={},
        params={"FPS": 30.0},
        paths={"raw_csv_path": "/tmp/v_tracking.csv"},
    )

    progresses: list[tuple[int, str]] = []
    warnings: list[tuple[str, str]] = []
    results: list[object] = []
    worker.progress_signal.connect(lambda v, m: progresses.append((v, m)))
    worker.warning_signal.connect(lambda t, m: warnings.append((t, m)))
    worker.finished_signal.connect(lambda r: results.append(r))

    worker.execute()  # call directly (synchronous) — do NOT start() the QThread

    assert captured["ran"] is True
    assert captured["init"]["paths"]["raw_csv_path"] == "/tmp/v_tracking.csv"
    assert results and isinstance(results[0], SessionResult) and results[0].success is True
    assert (50, "half") in progresses
    assert ("W", "msg") in warnings


def test_stop_sets_should_stop(qapp):
    worker = SessionWorker(video_path="v.mp4", config={}, params={}, paths={})
    assert worker._should_stop() is False
    worker.stop()
    assert worker._should_stop() is True
```

> Note: the `_fake_init` in the test deliberately does not store `self.callbacks`; adjust it if `execute()` builds the `SessionCallbacks` and passes them into the constructor (recommended) so that `self.callbacks` is set by the real service. If `execute()` builds callbacks and passes them to `TrackingSessionCore(...)`, then `_fake_run` can read `self.callbacks` only if `_fake_init` stores them — so have `_fake_init` also do `self.callbacks = callbacks`. Update the test accordingly when you see the final `execute()` shape.

- [ ] **Step 2: Run the test to verify it fails**

Run: `PYTHONPATH=$PWD/src KMP_DUPLICATE_LIB_OK=TRUE conda run -n hydra-mps python -m pytest tests/test_session_worker.py -q --ignore=tests/test_identity_postprocess.py`
Expected: FAIL — `ModuleNotFoundError: hydra_suite.trackerkit.gui.workers.session_worker`.

- [ ] **Step 3: Implement the worker**

Create `src/hydra_suite/trackerkit/gui/workers/session_worker.py`, mirroring `gui/workers/merge_worker.py`:

```python
"""SessionWorker — runs TrackingSessionCore.run_post_tracking on a QThread.

Mirrors the other gui/workers BaseWorker subclasses: extra Signals for the
progress/result/error/warning payloads, an `execute()` that builds the Qt-free
service and drives it, and a cooperative `stop()`/`_should_stop()` pair wired to
the service's should_stop callback.
"""

from __future__ import annotations

import logging
from typing import Any

from PySide6.QtCore import Signal

from hydra_suite.widgets.workers import BaseWorker

logger = logging.getLogger(__name__)


class SessionWorker(BaseWorker):
    progress_signal = Signal(int, str)   # (percent, message)
    finished_signal = Signal(object)     # SessionResult
    error_signal = Signal(str)
    warning_signal = Signal(str, str)    # (title, message)

    def __init__(
        self,
        *,
        video_path: str,
        config: Any,
        params: dict[str, Any],
        paths: dict[str, Any],
    ) -> None:
        super().__init__()
        self._video_path = video_path
        self._config = config
        self._params = params
        self._paths = paths
        self._stop_requested = False

    def stop(self) -> None:
        self._stop_requested = True

    def _should_stop(self) -> bool:
        return bool(self._stop_requested or self.isInterruptionRequested())

    def execute(self) -> None:
        # Import lazily so this module stays importable without pulling the
        # full core graph at GUI import time (matches merge_worker).
        from hydra_suite.core.tracking.session import (
            SessionCallbacks,
            TrackingSessionCore,
        )

        try:
            callbacks = SessionCallbacks(
                progress=lambda pct, msg: self.progress_signal.emit(int(pct), str(msg)),
                status=lambda msg: self.status.emit(str(msg)),
                warning=lambda title, msg: self.warning_signal.emit(str(title), str(msg)),
                stage_changed=lambda name: self.status.emit(str(name)),
                should_stop=self._should_stop,
            )
            service = TrackingSessionCore(
                video_path=self._video_path,
                config=self._config,
                params=self._params,
                paths=self._paths,
                callbacks=callbacks,
            )
            result = service.run_post_tracking(None, None)
            if not self._should_stop():
                self.finished_signal.emit(result)
        except Exception as exc:  # noqa: BLE001 — surface as a Qt error signal
            logger.exception("SessionWorker failed during run_post_tracking")
            self.error_signal.emit(str(exc))
```

Update the Step-1 test's `_fake_init` to also store `self.callbacks = callbacks` (so `_fake_run` can invoke them), since `execute()` passes real callbacks into the constructor.

- [ ] **Step 4: Run the test to verify it passes**

Run: `PYTHONPATH=$PWD/src KMP_DUPLICATE_LIB_OK=TRUE conda run -n hydra-mps python -m pytest tests/test_session_worker.py -q --ignore=tests/test_identity_postprocess.py`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
conda run -n hydra-mps black src/hydra_suite/trackerkit/gui/workers/session_worker.py tests/test_session_worker.py
conda run -n hydra-mps isort src/hydra_suite/trackerkit/gui/workers/session_worker.py tests/test_session_worker.py
git add src/hydra_suite/trackerkit/gui/workers/session_worker.py tests/test_session_worker.py
git commit -m "feat(trackerkit): SessionWorker wraps TrackingSessionCore.run_post_tracking"
```

---

## Task 2: Cutover wiring — GUI drives `run_post_tracking` via `SessionWorker`

Make the new path LIVE without deleting the old methods yet (so the wiring is reviewable in isolation and the smoke in Task 3 can lock it in before the big deletion).

**Files:**
- Modify: `src/hydra_suite/trackerkit/gui/orchestrators/tracking.py` (add `_run_session_worker`, `_on_session_finished`; slim `on_tracking_finished`; rewrite `_finish_tracking_session` to delegate)
- Test: `tests/test_trackerkit_session_cutover_wiring.py` (create)

**Interfaces:**
- Consumes: `SessionWorker` (Task 1); `TrackingSessionCore` inputs (the `paths`/`config`/`params` shape).
- Produces: `TrackingOrchestrator._run_session_worker(self) -> None` (builds paths/config/params, constructs+connects+starts a `SessionWorker` stored on `self._mw.session_worker`); `TrackingOrchestrator._on_session_finished(self, result) -> None` (sets `self._mw._session_final_csv_path = result.final_csv_path`, stores `summary_lines`, shows error dialog on failure, then calls `_finalize_tracking_session_ui()`). Consumed by Task 4 (which deletes what these replace).

- [ ] **Step 1: Write the failing test** (synchronous wiring; no async event loop)

Create `tests/test_trackerkit_session_cutover_wiring.py`:

```python
"""The GUI post-tracking flow builds a SessionWorker over run_post_tracking."""

from __future__ import annotations

import os
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication

from hydra_suite.core.tracking.session import SessionResult
from hydra_suite.trackerkit.gui.orchestrators import tracking as tracking_module
from hydra_suite.trackerkit.gui.orchestrators.tracking import TrackingOrchestrator


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _orchestrator(tmp_path):
    raw = str(tmp_path / "video_tracking.csv")
    panels = SimpleNamespace(
        setup=SimpleNamespace(
            csv_line=SimpleNamespace(text=lambda: raw),
            file_line=SimpleNamespace(text=lambda: "video.mp4"),
        ),
    )
    mw = SimpleNamespace(
        _stop_all_requested=False,
        _session_final_csv_path=None,
        _session_fps_list=[30.0],
        current_detection_cache_path=str(tmp_path / "cache.npz"),
        current_individual_properties_cache_path=None,
        current_detected_properties_cache_path=None,
        current_detected_cnn_cache_paths={},
        get_parameters_dict=lambda: {"FPS": 30.0},
    )
    orch = TrackingOrchestrator(main_window=mw, config=object(), panels=panels)
    orch._mw = mw
    return orch, mw, raw


def test_run_session_worker_builds_worker_over_service(qapp, tmp_path, monkeypatch):
    orch, mw, raw = _orchestrator(tmp_path)

    built = {"paths": None, "config": None, "params": None}

    class _FakeWorker:
        finished_signal = None  # set per-instance below

        def __init__(self, *, video_path, config, params, paths):
            built["paths"] = paths
            built["config"] = config
            built["params"] = params
            self._cbs = []

        # emulate just enough of the Signal API the orchestrator uses
        finished_signal = property()  # replaced in instance via SimpleNamespace below

    # Replace SessionWorker with a fake whose signals capture connections and
    # can be fired synchronously.
    class _Sig:
        def __init__(self):
            self._slots = []
        def connect(self, fn):
            self._slots.append(fn)
        def emit(self, *a):
            for fn in list(self._slots):
                fn(*a)

    class _Worker:
        def __init__(self, *, video_path, config, params, paths):
            built["paths"] = paths
            built["config"] = config
            built["params"] = params
            self.progress_signal = _Sig()
            self.finished_signal = _Sig()
            self.error_signal = _Sig()
            self.warning_signal = _Sig()
        def start(self):
            self.finished_signal.emit(
                SessionResult(True, str(0), None, [], None, ["ok"], None)
            )

    monkeypatch.setattr(tracking_module, "SessionWorker", _Worker)
    monkeypatch.setattr(
        orch, "_finalize_tracking_session_ui", lambda: mw.__setattr__("_finalized", True)
    )

    # config building: whatever the orchestrator calls to build config must be stubbed
    monkeypatch.setattr(orch, "_build_session_config", lambda: {"cfg": 1}, raising=False)

    orch._run_session_worker()

    assert built["paths"]["raw_csv_path"] == raw
    assert built["paths"]["detection_cache_path"] == mw.current_detection_cache_path
    assert built["params"] == {"FPS": 30.0}
    assert getattr(mw, "_finalized", False) is True
```

> This test pins the CONTRACT (`_run_session_worker` builds the paths dict from the MainWindow attrs, constructs a `SessionWorker`, starts it, and the finished slot reaches `_finalize_tracking_session_ui`). The exact config-building call (`_build_session_config` vs `self._mw._config_orch.build_config_dict()`) must be reconciled against source in Step 3 — update the stub name to match.

- [ ] **Step 2: Run the test to verify it fails**

Run: `PYTHONPATH=$PWD/src KMP_DUPLICATE_LIB_OK=TRUE conda run -n hydra-mps python -m pytest tests/test_trackerkit_session_cutover_wiring.py -q --ignore=tests/test_identity_postprocess.py`
Expected: FAIL — `AttributeError: 'TrackingOrchestrator' object has no attribute '_run_session_worker'`.

- [ ] **Step 3: Implement the wiring** (reconcile every cited name against live source first)

Add to `tracking.py` (import `SessionWorker` at top: `from hydra_suite.trackerkit.gui.workers.session_worker import SessionWorker`):

```python
    def _run_session_worker(self) -> None:
        """Drive all post-tracking through the Qt-free core service.

        Both tracking passes have already written the raw CSV(s) to disk; the
        service re-reads them and does postprocess/merge/rich-export/interp/
        media/dataset — replacing the old _finish_tracking_session chain.
        """
        raw_csv_path = self._panels.setup.csv_line.text()
        video_path = self._panels.setup.file_line.text()
        paths = {
            "raw_csv_path": raw_csv_path,
            "detection_cache_path": getattr(self._mw, "current_detection_cache_path", None),
            "individual_properties_cache_path": getattr(
                self._mw, "current_individual_properties_cache_path", None
            ),
            "detected_properties_cache_path": getattr(
                self._mw, "current_detected_properties_cache_path", None
            ),
            "detected_cnn_cache_paths": getattr(
                self._mw, "current_detected_cnn_cache_paths", None
            ),
        }
        worker = SessionWorker(
            video_path=video_path,
            config=self._build_session_config(),      # RECONCILE: see below
            params=self._mw.get_parameters_dict(),
            paths=paths,
        )
        self._mw.session_worker = worker
        worker.progress_signal.connect(self._on_session_progress)
        worker.warning_signal.connect(self.on_tracking_warning)
        worker.error_signal.connect(self._on_session_error)
        worker.finished_signal.connect(self._on_session_finished)
        worker.start()

    def _on_session_progress(self, value: int, message: str) -> None:
        # Mirror the progress-bar update the deleted post-workers did.
        self._mw.progress_bar.setVisible(True)
        self._mw.progress_bar.setValue(int(value))
        self._mw.progress_label.setVisible(True)
        self._mw.progress_label.setText(str(message))

    def _on_session_error(self, message: str) -> None:
        QMessageBox.critical(
            self._mw,
            "Post-Processing Error",
            f"Error during trajectory post-processing:\n{message}",
        )
        logger.error("Session post-processing error: %s", message)
        self._finalize_tracking_session_ui()

    def _on_session_finished(self, result) -> None:
        if not getattr(result, "success", False):
            QMessageBox.critical(
                self._mw,
                "Post-Processing Error",
                f"Error during trajectory post-processing:\n{result.error or 'unknown error'}",
            )
            self._finalize_tracking_session_ui()
            return
        self._mw._session_final_csv_path = result.final_csv_path
        self._mw._session_summary_lines = list(result.summary_lines or [])
        self._finalize_tracking_session_ui()
```

**RECONCILE (do these against live source before finalizing):**
1. **Config building.** Find how the GUI builds the full config dict elsewhere (Slice-3 note: `self._mw._config_orch.build_config_dict()`). Add a small private `_build_session_config()` that returns it, OR inline the call. Confirm the exact accessor. This is what `TrackingSessionCore(config=...)` consumes; it must be the same dict the deleted chain effectively used (it is the GUI's canonical config).
2. **`final_csv_path`.** The service DERIVES `{base}_final{ext}` from `raw_csv_path` and returns it as `result.final_csv_path` (session.py:481-489); `_on_session_finished` stores it. Do NOT pass `final_csv_path` in `paths` (inert — session ignores it).
3. **`_show_session_summary`.** It currently calls the (to-be-deleted) `_build_session_summary_lines`. Re-point it to read `self._mw._session_summary_lines` (set above). `_finalize_tracking_session_ui` already decides when to show the summary — leave that logic, only change the source of the lines.
4. **Slim `on_tracking_finished`** (`tracking.py:1772`): keep everything up to and including `self._collect_worker_props_path()` and `self._accumulate_session_fps(...)`; replace the `_start_postprocess_worker(...)` tail (line ~1823) with:
   ```python
       if is_backward_mode:
           self._run_session_worker()
       elif is_backward_enabled:
           self.start_backward_tracking()
       else:
           self._run_session_worker()
   ```
   (Backward pass done, or forward-only → post; forward with backward enabled → run the 2nd pass first.) Keep the `_postprocess_*` stashes only if `start_backward_tracking` still reads them; otherwise drop them.
5. **`_finish_tracking_session`** (`tracking.py:2602`): its callers are `_handle_forward_tracking_done`/`_handle_backward_tracking_done` (being removed) — after Step 4 nothing calls it. Rewrite its body to a single `self._run_session_worker()` (harmless bridge) OR leave it for deletion in Task 4; simplest: leave it, since Step 4 rewires `on_tracking_finished` to call `_run_session_worker` directly and `_handle_*` are deleted in Task 4.

- [ ] **Step 4: Run the test to verify it passes** (update the config stub name to match Step-3 reconciliation)

Run: `PYTHONPATH=$PWD/src KMP_DUPLICATE_LIB_OK=TRUE conda run -n hydra-mps python -m pytest tests/test_trackerkit_session_cutover_wiring.py -q --ignore=tests/test_identity_postprocess.py`
Expected: PASS.

- [ ] **Step 5: Regression check + commit**

Run the orchestrator smoke to confirm no import/wiring break:
`PYTHONPATH=$PWD/src KMP_DUPLICATE_LIB_OK=TRUE conda run -n hydra-mps python -m pytest tests/test_trackerkit_orchestrators_smoke.py -q --ignore=tests/test_identity_postprocess.py`
Then:
```bash
conda run -n hydra-mps black src/hydra_suite/trackerkit/gui/orchestrators/tracking.py tests/test_trackerkit_session_cutover_wiring.py
conda run -n hydra-mps isort src/hydra_suite/trackerkit/gui/orchestrators/tracking.py tests/test_trackerkit_session_cutover_wiring.py
git add src/hydra_suite/trackerkit/gui/orchestrators/tracking.py tests/test_trackerkit_session_cutover_wiring.py
git commit -m "feat(trackerkit): GUI post-tracking drives SessionWorker/run_post_tracking"
```

---

## Task 3: GUI == CLI byte-identity smoke (regression guard, before the deletion)

Prove the new GUI post-tracking path produces the **same final CSV as the CLI** for a real clip, so the Task-4/5 deletion is guarded. Runs the post-tracking synchronously (no async event loop) against pre-produced raw CSVs.

**Files:**
- Create: `tests/test_gui_session_cutover_equivalence.py`

**Interfaces:**
- Consumes: `run_tracking_cli` (CLI), `TrackingOrchestrator._run_session_worker`, the `fly_obb` fixture clip/config.

- [ ] **Step 1: Write the test**

Create `tests/test_gui_session_cutover_equivalence.py`:

```python
"""The GUI post-tracking path yields the same final CSV as the CLI."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pandas as pd
import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication

REPO = Path(__file__).resolve().parents[1]
FLY_CLIP = REPO / "tools/equivalence/fixtures/clips/fly_obb.mp4"
FLY_CONFIG = REPO / "tools/equivalence/fixtures/configs/fly_obb.json"

pytestmark = pytest.mark.skipif(
    not (FLY_CLIP.exists() and FLY_CONFIG.exists()),
    reason="fly_obb fixture missing (run tools/equivalence/fixtures/fetch_fixtures.sh)",
)


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def test_gui_run_session_worker_matches_cli(qapp, tmp_path):
    from hydra_suite.trackerkit.cli import run_tracking_cli
    from hydra_suite.trackerkit.gui.orchestrators.tracking import TrackingOrchestrator

    # 1) CLI run → canonical final CSV.
    cli_dir = tmp_path / "cli"
    cli_dir.mkdir()
    cli_clip = cli_dir / "fly_obb.mp4"
    shutil.copy(FLY_CLIP, cli_clip)
    assert run_tracking_cli([str(cli_clip)], config_path=str(FLY_CONFIG)) == 0
    cli_final = next(cli_dir.glob("*_tracking_final.csv"))

    # 2) GUI post-tracking over the SAME raw CSV the CLI produced.
    gui_dir = tmp_path / "gui"
    gui_dir.mkdir()
    gui_raw = gui_dir / "fly_obb_tracking.csv"
    shutil.copy(next(cli_dir.glob("*_tracking.csv")), gui_raw)   # raw forward CSV
    # copy the detection cache so the service reuses it identically
    cli_cache = next(cli_dir.rglob("*detection_cache*.npz"), None)

    import json
    config = json.loads(FLY_CONFIG.read_text())

    panels = SimpleNamespace(
        setup=SimpleNamespace(
            csv_line=SimpleNamespace(text=lambda: str(gui_raw)),
            file_line=SimpleNamespace(text=lambda: str(cli_clip)),
        ),
    )
    finalized = {"done": False}
    mw = SimpleNamespace(
        _stop_all_requested=False,
        _session_final_csv_path=None,
        _session_fps_list=[],
        current_detection_cache_path=str(cli_cache) if cli_cache else None,
        current_individual_properties_cache_path=None,
        current_detected_properties_cache_path=None,
        current_detected_cnn_cache_paths={},
        get_parameters_dict=lambda: dict(config),  # fly_obb: config carries the params keys
        progress_bar=SimpleNamespace(setVisible=lambda *_: None, setValue=lambda *_: None),
        progress_label=SimpleNamespace(setVisible=lambda *_: None, setText=lambda *_: None),
    )
    orch = TrackingOrchestrator(main_window=mw, config=object(), panels=panels)
    orch._mw = mw
    orch._finalize_tracking_session_ui = lambda: finalized.__setitem__("done", True)
    orch._build_session_config = lambda: dict(config)   # RECONCILE to match Task 2

    # run the SessionWorker synchronously by calling execute() rather than start()
    from hydra_suite.trackerkit.gui.workers import session_worker as sw_module
    original_start = sw_module.SessionWorker.start
    sw_module.SessionWorker.start = lambda self: self.execute()
    try:
        orch._run_session_worker()
    finally:
        sw_module.SessionWorker.start = original_start

    assert finalized["done"] is True
    gui_final = next(gui_dir.glob("*_tracking_final.csv"))
    pd.testing.assert_frame_equal(
        pd.read_csv(gui_final).reset_index(drop=True),
        pd.read_csv(cli_final).reset_index(drop=True),
    )
```

> `fly_obb` is the fast smoke clip (no SLEAP/identity), so this runs quickly and needs no `sleap` env. Reconcile the two lambdas (`get_parameters_dict`, `_build_session_config`) with what the real GUI supplies — the point is the SessionWorker receives the SAME `params`/`config` the CLI used, so the final CSVs match. If `fly_obb`'s config JSON is not directly a valid params dict, build params the way `cli_config.build_tracking_parameters` does and reuse that here.

- [ ] **Step 2: Run the test**

Run: `PYTHONPATH=$PWD/src KMP_DUPLICATE_LIB_OK=TRUE conda run -n hydra-mps python -m pytest tests/test_gui_session_cutover_equivalence.py -q --ignore=tests/test_identity_postprocess.py`
Expected: PASS (GUI final CSV byte-identical to CLI). If the frames differ, the `params`/`config`/`paths` the GUI passes differ from the CLI's — reconcile until identical. The fixture symlink for clips must exist: `ln -sfn "$REPO_MAIN/tools/equivalence/fixtures/clips" tools/equivalence/fixtures/clips` (remove after; never `git add`).

- [ ] **Step 3: Commit**

```bash
conda run -n hydra-mps black tests/test_gui_session_cutover_equivalence.py
conda run -n hydra-mps isort tests/test_gui_session_cutover_equivalence.py
git add tests/test_gui_session_cutover_equivalence.py
git commit -m "test(trackerkit): GUI post-tracking equals CLI final CSV (cutover guard)"
```

---

## Task 4: Delete the post-tracking methods from `tracking.py` + fix coupled tests

Now that the new path is live and guarded, remove the ~45 duplicated post-tracking methods. Delete leaves-first; after each removal grep `src/` AND `tests/` for the symbol.

**Files:**
- Modify: `src/hydra_suite/trackerkit/gui/orchestrators/tracking.py` (delete methods)
- Modify: `tests/test_trackerkit_tracking_orchestrator_dialogs.py` (re-point/remove tests of deleted methods)

**Interfaces:**
- Consumes: nothing new.
- Produces: `tracking.py` with the post-tracking chain gone; `on_tracking_finished` → `start_backward_tracking`/`_run_session_worker` → `_on_session_finished` → `_finalize_tracking_session_ui` the only flow.

- [ ] **Step 1: Enumerate + guard.** For EACH method in the deletion set (spec §"Deletion set"), run `grep -rn "<name>" src/ tests/`. Any hit outside `tracking.py` itself (other than the dialogs-test file handled below) is a surviving caller — STOP and resolve before deleting. Record the list of methods with zero surviving callers.

Deletion set in `tracking.py` (re-derive lines): `_rich_export_path`, `_write_rich_export_csv`, `_drop_empty_rich_export_columns`, `_remove_legacy_rich_exports`, `_build_rich_export_dataframe`, `_export_rich_csv`, `_relink_and_export_rich_csv`, `_log_rich_export_summary`, `save_trajectories_to_csv`, `_scale_trajectories_to_original_space`, `merge_and_save_trajectories`, `on_merge_progress`, `on_merge_error`, `on_merge_finished`, `_store_interpolated_pose_result`, `_store_interpolated_tag_result`, `_store_interpolated_cnn_result`, `_store_interpolated_headtail_result`, `_log_interpolated_postpass_summary`, `_count_augmented_pose_rows`, `_count_interpolated_cnn_rows`, `_on_interpolated_crops_finished`, `_generate_interpolated_individual_crops`, `_start_postprocess_worker`, `on_postprocess_progress`, `on_postprocess_finished`, `_check_pose_export_sources`, `_merge_pose_sources_into_df`, `_apply_pose_quality_postprocessing`, `_resolve_current_tag_cache_path`, `_apply_identity_postprocessing_to_df`, `_generate_final_media_export`, `_start_pending_final_media_export`, `_on_final_media_export_worker_thread_finished`, `_on_final_media_export_finished`, `_on_final_media_export_error`, `_generate_video_from_trajectories`, `_load_video_trajectories`, `_run_pending_video_generation_or_finalize`, `_generate_training_dataset`, `on_dataset_progress`, `on_dataset_finished`, `on_dataset_error`, `_build_session_summary_lines`, `_handle_forward_tracking_done`, `_handle_backward_tracking_done`, `_finish_tracking_session`. **Retain and rewire** `_show_session_summary` (reads `self._mw._session_summary_lines` per Task 2). **Retain** `_clear_session_summary_state` only if `_finalize_tracking_session_ui` still calls it; else delete. Do NOT touch `on_tracking_warning`, `_handle_tracking_failed`, `_collect_worker_props_path`, `_accumulate_session_fps`, `start_backward_tracking`, `start_tracking_on_video`, `_finalize_tracking_session_ui`, `_stop_csv_writer`, preview handlers.

- [ ] **Step 2: Capture the dialogs-suite baseline.** Run and record the pass/fail set (F0):
`PYTHONPATH=$PWD/src KMP_DUPLICATE_LIB_OK=TRUE conda run -n hydra-mps python -m pytest tests/test_trackerkit_tracking_orchestrator_dialogs.py -q --ignore=tests/test_identity_postprocess.py`

- [ ] **Step 3: Delete the methods** from `tracking.py` (leaves-first per Step 1). Also remove now-unused imports (`PostProcessWorker`, `MergeWorker`, and any rich-export/interp/media/dataset helpers imported only for these methods) — but KEEP a worker class import if still referenced. Remove any `self._mw`/panel-state reads that only these methods used.

- [ ] **Step 4: Fix the coupled tests** in `tests/test_trackerkit_tracking_orchestrator_dialogs.py`. For each test that called/patched a deleted method (functions at approx lines 892, 989, 1024, 1121, 1187, 1236, 1290, 1311, 1337 — re-derive): if the behavior now lives in a `core/post/*` stage that already has coverage (`tests/test_session_core_run.py`, `test_session_export_chain.py`), DELETE the GUI-level test (it tested deleted plumbing); if it asserted a GUI-visible behavior that survived, re-point it. Do not weaken assertions to pass; delete or re-target.

- [ ] **Step 5: Verify (delta, not absolute).** Re-run the dialogs suite (F1) and the orchestrator smoke:
```bash
PYTHONPATH=$PWD/src KMP_DUPLICATE_LIB_OK=TRUE conda run -n hydra-mps python -m pytest \
  tests/test_trackerkit_tracking_orchestrator_dialogs.py \
  tests/test_trackerkit_orchestrators_smoke.py \
  tests/test_trackerkit_session_cutover_wiring.py \
  tests/test_gui_session_cutover_equivalence.py \
  -q --ignore=tests/test_identity_postprocess.py
```
Expected: the cutover wiring + equivalence tests PASS; the dialogs suite has no NEW failures vs F0 (deleted-plumbing tests are gone, not failing). Then grep `src/ tests/` for every deleted name — zero hits outside the deletion diff.

- [ ] **Step 6: Commit**

```bash
conda run -n hydra-mps black src/hydra_suite/trackerkit/gui/orchestrators/tracking.py tests/test_trackerkit_tracking_orchestrator_dialogs.py
conda run -n hydra-mps isort src/hydra_suite/trackerkit/gui/orchestrators/tracking.py tests/test_trackerkit_tracking_orchestrator_dialogs.py
git add src/hydra_suite/trackerkit/gui/orchestrators/tracking.py tests/test_trackerkit_tracking_orchestrator_dialogs.py
git commit -m "refactor(trackerkit): delete duplicated GUI post-tracking chain (now in core service)"
```

---

## Task 5: Delete the `main_window.py` wrappers + final dangling-ref sweep

**Files:**
- Modify: `src/hydra_suite/trackerkit/gui/main_window.py` (delete 11 thin wrappers)

**Interfaces:**
- Consumes: nothing new.
- Produces: `main_window.py` with no delegators to the deleted orchestrator methods.

- [ ] **Step 1: Guard + delete.** For each wrapper (re-derive lines): `merge_and_save_trajectories`, `_store_interpolated_pose_result`, `_store_interpolated_tag_result`, `_store_interpolated_cnn_result`, `_store_interpolated_headtail_result`, `_on_interpolated_crops_finished`, `_generate_final_media_export`, `on_merge_finished`, `_generate_interpolated_individual_crops`, `_generate_training_dataset` — grep `src/ tests/` for each; any surviving caller is a Qt-signal connection to remove too (check the `__init__`/signal-wiring where these are `.connect(...)`ed). Delete the wrappers and their signal connections. Keep `_finish_tracking_session` wrapper only if Task 2 kept the orchestrator method as a bridge; otherwise delete it (and its callers).

- [ ] **Step 2: Full dangling-ref sweep.** `grep -rnE "<all deleted names, pipe-joined>" src/ tests/` → zero hits. Also `grep -rn "forward_processed_trajs\|backward_processed_trajs\|_postprocess_is_backward" src/ tests/` — remove any now-dead MainWindow state that only the deleted chain used (verify no surviving reader first).

- [ ] **Step 3: Verify.**
```bash
PYTHONPATH=$PWD/src KMP_DUPLICATE_LIB_OK=TRUE conda run -n hydra-mps python -m pytest \
  tests/test_trackerkit_orchestrators_smoke.py \
  tests/test_main_window_config_persistence.py \
  tests/test_trackerkit_session_cutover_wiring.py \
  tests/test_gui_session_cutover_equivalence.py \
  tests/test_session_worker.py \
  -q --ignore=tests/test_identity_postprocess.py
```
Expected: PASS (delta vs baseline for any pre-existing failures).

- [ ] **Step 4: Confirm no `core/` change.** `git diff --stat <slice5-base>..HEAD -- src/hydra_suite/core/` → empty. This is the invariant that keeps the Slice-4 CLI gate valid.

- [ ] **Step 5: Commit**

```bash
conda run -n hydra-mps black src/hydra_suite/trackerkit/gui/main_window.py
conda run -n hydra-mps isort src/hydra_suite/trackerkit/gui/main_window.py
git add src/hydra_suite/trackerkit/gui/main_window.py
git commit -m "refactor(trackerkit): remove main_window post-tracking wrappers"
```

---

## Task 6: Verification — GUI launch smoke + CLI-gate sanity (FINAL)

**Files:** primarily verification; optionally a fuller offscreen end-to-end smoke.

- [ ] **Step 1: End-to-end offscreen GUI smoke (best-effort, backward-enabled).** Extend `tests/test_gui_session_cutover_equivalence.py` (or add `tests/test_gui_launch_smoke.py`) with a real `MainWindow()` (offscreen, `_save/_load_advanced_config` stubbed per `tests/test_config_build_dict.py`) that runs a real forward-only session via `window._tracking_orch.start_tracking(preview_mode=False)` on `fly_obb`, driving the Qt event loop with a `QEventLoop` + `QTimer` timeout until `_finalize_tracking_session_ui` fires, and asserts a non-empty final CSV. Add a backward-enabled variant (set the backward checkbox before starting). If the async event-loop harness proves flaky in CI, keep the synchronous Task-3 equivalence test as the authoritative guard and mark this end-to-end smoke `@pytest.mark.slow`. Include one identity clip (`ant_cnn_identity`, needs `sleap` env; skip if absent) to exercise the full stage chain.

- [ ] **Step 2: CLI equivalence-gate sanity (MPS).** Because Task 5 Step 4 proves `core/` is untouched, the Slice-4 gate must still pass. Run the fast subset to confirm no accidental core/CLI change:
```bash
conda activate hydra-mps
git worktree add --detach .worktrees/equiv-slice5-base 0384265b
ln -sfn "$MAIN_REPO/tools/equivalence/fixtures/clips" tools/equivalence/fixtures/clips
REPO=$MAIN_REPO WT=$PWD MAIN_SRC=$PWD/.worktrees/equiv-slice5-base/src WT_SRC=$PWD/src \
  OUT=/tmp/equiv_slice5 RUNTIME=mps bash tools/equivalence/run_matrix.sh fly_obb ant_cnn_identity > /tmp/equiv_mps_slice5.log 2>&1
```
Expected: `fly_obb` + `ant_cnn_identity` byte-identical (same as Slice 4). Clean up the worktree + symlink after. (Full 7-clip + CUDA not required since `core/` is unchanged; this is a sanity check that nothing leaked into core/CLI.)

- [ ] **Step 3: Targeted suite green.** Run the Slice-5-touched test files together; judge failures as a delta vs the pre-slice baseline. No dangling references to any deleted symbol anywhere.

- [ ] **Step 4: Record the outcome** for the final whole-branch review and the merge.

---

## Self-Review

**Spec coverage:** Cutover contract/architecture → Tasks 1-2. SessionWorker → Task 1. `_finish_tracking_session` anchor rewrite + reconciliation (paths/params/config, props caches, final-csv-from-result) → Task 2. Deletion set (~45 methods) → Task 4; 11 wrappers → Task 5. Rewrite orchestrator-dialogs tests → Task 4 Step 4. GUI==CLI launch smoke → Task 3 (synchronous, authoritative) + Task 6 (async end-to-end, best-effort). CLI-gate sanity + no-core-change invariant → Task 5 Step 4 + Task 6 Step 2. Follow-up (shared param-builder) is out of scope by design (documented in the spec).

**Placeholder scan:** Code steps carry full source for the SessionWorker + wiring + tests. The remaining "RECONCILE" items are concrete verification steps against live source (exact config-builder accessor, exact deletion line numbers, exact props-cache attr threading), not vague placeholders — they exist because `tracking.py` line numbers/accessors shift and must be re-derived per the Global Constraints (the Slice-1..4 pattern). Each names exactly what to confirm and the expected shape.

**Type consistency:** `SessionWorker(*, video_path, config, params, paths)` + signals `progress_signal(int,str)`/`finished_signal(object)`/`error_signal(str)`/`warning_signal(str,str)` are defined in Task 1 and consumed identically in Task 2. `run_post_tracking(None, None) -> SessionResult`, `SessionResult.final_csv_path`/`.summary_lines`/`.success`/`.error`, `SessionCallbacks(progress,status,warning,stage_changed,should_stop)`, and the `paths` dict keys match the verified interfaces block and the CLI reference (`headless_tracking.py:288-308`). `_run_session_worker`/`_on_session_finished`/`_on_session_progress`/`_on_session_error` names are consistent across Tasks 2-3.
