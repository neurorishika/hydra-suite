# Qt-Free Headless — Slice 4: CLI Cutover Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `trackerkit track` fully Qt-free by rewriting `headless_tracking.py` to drive the Qt-free `TrackingEngineCore` (plain threads) then `TrackingSessionCore.run_post_tracking(...)`, and deleting the hidden-`MainWindow` bridge from `cli.py` and the `_headless_*` hooks from the GUI.

**Architecture:** The CLI stops constructing any Qt object. Forward/backward tracking each run on a plain `threading.Thread` targeting `TrackingEngineCore.run_tracking()`, writing a raw CSV through the already-Qt-free `CSVWriterThread`. The raw trajectories are read back into DataFrames and handed to `TrackingSessionCore.run_post_tracking(forward_trajectories, backward_trajectories=None)`, which owns merge/post-process/interpolation/export and returns a `SessionResult`. Ctrl-C is wired through a `signal.SIGINT`-backed stop flag surfaced as `SessionCallbacks.should_stop`.

**Tech Stack:** Python 3, PySide6 (being removed from this path), pandas, numpy, `threading`, `signal`, pytest. Conda env `hydra-mps`.

## Global Constraints

Copied verbatim from the task and spec; every task's requirements implicitly include this section.

- **This is Slice 4 (final) of the 4-slice program** in `docs/superpowers/specs/2026-07-24-headless-qt-free-session-service-design.md`. It assumes Slices 2+3 already landed: `TrackingSessionCore` (`src/hydra_suite/core/tracking/session.py`) is at full parity with the hidden-`MainWindow` bridge.
- **Treat as EXISTING** (built by Slices 2/3):
  - `TrackingSessionCore.__init__(self, *, video_path, config, params, paths, callbacks=SessionCallbacks())`
  - `TrackingSessionCore.run_post_tracking(self, forward_trajectories, backward_trajectories=None) -> SessionResult`
  - `SessionCallbacks(progress: Callable[[int, str], None], status: Callable[[str], None], warning: Callable[[str, str], None], stage_changed: Callable[[str], None], should_stop: Callable[[], bool])` — all no-op defaults.
  - `SessionResult(success: bool, final_csv_path: str | None, rich_export_path: str | None, media_paths: list[str], dataset_result: dict | None, summary_lines: list[str], error: str | None)`
- **Also EXISTING:** `TrackingEngineCore` (`src/hydra_suite/core/tracking/worker.py`) is already Qt-free and callback-driven (constructor callbacks `on_finished`, `on_progress`, `on_stats`, `on_warning`, `on_frame`, `on_pose_model_resolved`; methods `set_parameters`, `update_parameters`, `get_current_params`, `stop`, `run_tracking`; flag `_stop_requested`). It emits `on_finished(success, fps_list, full_traj)` at completion (`worker.py:4171`), where `full_traj` is `self.trajectories_full`. `CSVWriterThread` (`src/hydra_suite/data/csv_writer.py`) is already a plain `threading.Thread`.
- **Commit as the configured git user; NO `Co-Authored-By` trailer.**
- **`make format` before each commit** (or `black` + `isort` directly under `hydra-mps` if `make` is broken).
- **Run tests with:** `conda run -n hydra-mps python -m pytest <path> -q --ignore=tests/test_identity_postprocess.py`. Env is `hydra-mps`.
- **After this slice both must hold:**
  1. `grep -rnE "PySide6|QtCore|QThread|Signal|Slot|QMutex" src/hydra_suite/core/` is EMPTY (already holds — do not regress).
  2. The CLI runtime path — `trackerkit.headless_tracking` + `trackerkit.cli` minus the GUI-launcher branch — imports **no** PySide6.
- **Equivalence gate (mandatory, this slice is the most important run):** `tools/equivalence/run_matrix.sh` with the same baseline before AND after, on **both** MPS (`hydra-mps`, this box) and CUDA (mehek, `hydra-cuda`), all 7 clips, byte-identical on `_forward.csv` and `_tracking_final.csv` (positions p99 ≈ 0, θ max ≈ 0, identical row counts, 0 unmatched). Known noise floor: bistable head/tail π-flips only. **Conda MUST be active for pose/SLEAP clips or the CSVs come out empty and falsely compare EQUIVALENT — verify row counts > 1 before trusting a pass.** This slice proves the four previously-bridge-only clips (`ant_pose_headtail`, `ant_obb_sleap`, `emi_obb_identity`, `ant_cnn_identity`) are byte-identical via the direct path. Must pass on BOTH platforms before the bridge deletion is considered verified.
- **Docs follow-up (flag, do not necessarily edit):** `docs/superpowers/specs/2026-07-23-cloud-gpu-inference-design.md:123` references `QT_QPA_PLATFORM=offscreen` for the CLI container. After this slice that requirement no longer exists — note it as a docs follow-up.

---

## File Structure

| File | Change | Responsibility after this slice |
|---|---|---|
| `src/hydra_suite/trackerkit/cli_config.py` | Modify (`supports_direct_run`, line 103) | `supports_direct_run()` returns `True` unconditionally |
| `src/hydra_suite/trackerkit/headless_tracking.py` | Rewrite | Plain-threaded CLI driver: `TrackingEngineCore` → `TrackingSessionCore.run_post_tracking`; no Qt |
| `src/hydra_suite/trackerkit/cli.py` | Delete bridge | `run_tracking_cli` keeps only the direct path; no `MainWindow`, no `QMessageBox` monkeypatch |
| `src/hydra_suite/trackerkit/gui/main_window.py` | Modify (lines 390-392) | Drop the three `_headless_*` instance attributes |
| `src/hydra_suite/trackerkit/gui/orchestrators/tracking.py` | Modify (2604-2607, 2746-2751, 4589-4613) | Drop `_headless_*` reads/writes; keep GUI-only dialog paths |
| `tests/test_trackerkit_headless_tracking.py` | Rewrite | Cover the new plain-threaded driver (no `ensure_headless_qt_application`) |
| `tests/test_trackerkit_cli_cutover.py` | Create | `supports_direct_run` truth, no-bridge parity for the 4 clips |
| `tests/test_headless_tracking_qtfree.py` | Create | Qt-free guards: import-under-finder + subprocess `trackerkit track` with PySide6 blocked |

---

## Task 1: `supports_direct_run()` returns `True` unconditionally

**Files:**
- Modify: `src/hydra_suite/trackerkit/cli_config.py:103-109`
- Test: `tests/test_trackerkit_cli_cutover.py` (create)

**Interfaces:**
- Produces: `TrackerCliSession.supports_direct_run() -> bool` always `True`. Consumed by `cli.py` (Task 3) — after this task the bridge branch is dead.

The current implementation (`cli_config.py:103-109`) is:

```python
    def supports_direct_run(self) -> bool:
        """Return whether the CLI can run this session without MainWindow."""
        return not self.enable_pose_extractor and self.identity_method in {
            "",
            "none",
            "none_disabled",
        }
```

- [ ] **Step 1: Write the failing test**

Create `tests/test_trackerkit_cli_cutover.py`:

```python
"""Slice 4 CLI cutover: every session runs the direct (Qt-free) path."""

from __future__ import annotations

from hydra_suite.trackerkit.cli_config import (
    TrackerCliSession,
    TrackerCliVideoProbe,
)


def _make_session(**overrides) -> TrackerCliSession:
    base = dict(
        video_path="video.mp4",
        config_path=None,
        video_probe=TrackerCliVideoProbe(fps=30.0, total_frames=10, width=64, height=64),
        config={},
        raw_csv_path="video_tracking.csv",
        final_csv_path="video_tracking_forward_processed.csv",
        params={"FPS": 30.0},
        save_confidence_metrics=False,
        use_cached_detections=False,
        enable_backward_tracking=False,
        enable_postprocessing=True,
        interpolation_method="None",
        interpolation_max_gap_seconds=0.0,
        heading_flip_max_burst=5,
        identity_method="none_disabled",
        enable_pose_extractor=False,
    )
    base.update(overrides)
    return TrackerCliSession(**base)


def test_supports_direct_run_true_for_pose_sessions():
    session = _make_session(enable_pose_extractor=True)
    assert session.supports_direct_run() is True


def test_supports_direct_run_true_for_identity_sessions():
    session = _make_session(identity_method="apriltags")
    assert session.supports_direct_run() is True


def test_supports_direct_run_true_for_plain_session():
    assert _make_session().supports_direct_run() is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n hydra-mps python -m pytest tests/test_trackerkit_cli_cutover.py -q --ignore=tests/test_identity_postprocess.py`
Expected: FAIL — `test_supports_direct_run_true_for_pose_sessions` and `..._for_identity_sessions` assert `True` but current code returns `False`.

- [ ] **Step 3: Make the change**

Replace `cli_config.py:103-109` with:

```python
    def supports_direct_run(self) -> bool:
        """Every session runs the direct Qt-free path (Slice 4 CLI cutover).

        The hidden-MainWindow bridge was deleted once TrackingSessionCore
        reached parity, so pose/identity sessions no longer need Qt. Kept as a
        method (not deleted) so the CLI's call site stays stable.
        """
        return True
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n hydra-mps python -m pytest tests/test_trackerkit_cli_cutover.py -q --ignore=tests/test_identity_postprocess.py`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
make format
git add src/hydra_suite/trackerkit/cli_config.py tests/test_trackerkit_cli_cutover.py
git commit -m "feat(trackerkit): supports_direct_run() always True for Qt-free CLI"
```

---

## Task 2: Rewrite `headless_tracking.py` as a plain-threaded, Qt-free driver

**Files:**
- Rewrite: `src/hydra_suite/trackerkit/headless_tracking.py`
- Rewrite: `tests/test_trackerkit_headless_tracking.py`

**Interfaces:**
- Consumes: `TrackingEngineCore` (`core/tracking/worker.py`), `TrackingSessionCore` / `SessionCallbacks` / `SessionResult` (`core/tracking/session.py`), `CSVWriterThread` (`data/csv_writer.py`), `plan_tracking_cache` / `TrackingCachePlan` (`trackerkit/tracking_cache.py`), `TrackerCliSession` (`trackerkit/cli_config.py`).
- Produces:
  - `build_tracking_csv_header(save_confidence_metrics: bool, identity_method: str = "none_disabled") -> list[str]` (unchanged, still used for the raw-CSV header).
  - `_run_engine_pass(session, *, params, raw_csv_path, backward_mode, detection_cache_path, use_cached_detections, should_stop) -> tuple[bool, list[float], pandas.DataFrame | None]`
  - `run_headless_tracking_session(session: TrackerCliSession, *, should_stop: Callable[[], bool] | None = None) -> dict[str, Any]` — returns `{"success": bool, "lines": list[str], "error": str | None, "final_csv": str | None}`. Consumed by `cli.py` (Task 3).

**What is removed from the module:** the `from PySide6.QtCore import QCoreApplication, QEventLoop, QTimer` import, `ensure_headless_qt_application`, `_run_tracking_worker`, `_run_postprocess_worker`, `_run_merge_worker`, `_run_forward_only`, `_run_forward_backward`, and `save_trajectories_to_csv` (the session now owns final-CSV saving). `TrackingWorker`, `MergeWorker`, `PostProcessWorker`, and `interpolate_trajectories` imports are dropped. The empty-output guard (`_enforce_nonempty_forward` and its two helpers) is retained only if `TrackingSessionCore` does **not** already enforce it — per the spec (`design §Error handling`, "`_enforce_nonempty_forward` … moves into the service"), Slices 2/3 moved it into the service, so it is deleted here. Grep to confirm before deleting (see Step 3).

> **Note on the `paths` argument.** Slice 2 defines `paths` as a **plain dict** with keys `raw_csv_path`, `final_csv_path`, `detection_cache_path` (the service reads them via `self.paths.get(...)`). The code below passes that dict directly. Before writing Step 3, confirm against Slice 2's `session.py` — `grep -n "self.paths" src/hydra_suite/core/tracking/session.py` — and mirror the exact keys it reads. If Slice 2's implementer chose to introduce a typed `SessionPaths` dataclass instead, construct that here rather than the dict; but the default and current cross-slice contract is the plain dict.

- [ ] **Step 1: Write the failing tests**

Replace the entire contents of `tests/test_trackerkit_headless_tracking.py` with:

```python
"""Slice 4: headless_tracking drives TrackingEngineCore + TrackingSessionCore, no Qt."""

from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd

import hydra_suite.trackerkit.headless_tracking as ht
from hydra_suite.core.tracking.session import SessionResult
from hydra_suite.trackerkit.cli_config import (
    TrackerCliSession,
    TrackerCliVideoProbe,
)


def test_headless_tracking_module_imports_no_qt():
    """The CLI runtime path must not import PySide6/QtCore at any depth."""
    src = Path(ht.__file__).read_text()
    tree = ast.parse(src, filename=ht.__file__)
    offenders = []
    for node in ast.walk(tree):
        mod = None
        if isinstance(node, ast.ImportFrom):
            mod = node.module
        elif isinstance(node, ast.Import):
            mod = ",".join(a.name for a in node.names)
        if mod and any(q in mod for q in ("PySide6", "QtCore", "QtWidgets", "QtGui")):
            offenders.append(f"{mod}:{node.lineno}")
    assert not offenders, "headless_tracking must be Qt-free: " + "; ".join(offenders)


def _make_session(tmp_path, **overrides) -> TrackerCliSession:
    base = dict(
        video_path=str(tmp_path / "video.mp4"),
        config_path=None,
        video_probe=TrackerCliVideoProbe(fps=30.0, total_frames=10, width=64, height=64),
        config={},
        raw_csv_path=str(tmp_path / "video_tracking.csv"),
        final_csv_path=str(tmp_path / "video_tracking_forward_processed.csv"),
        params={"FPS": 30.0},
        save_confidence_metrics=False,
        use_cached_detections=False,
        enable_backward_tracking=False,
        enable_postprocessing=True,
        interpolation_method="None",
        interpolation_max_gap_seconds=0.0,
        heading_flip_max_burst=5,
        identity_method="none_disabled",
        enable_pose_extractor=False,
    )
    base.update(overrides)
    return TrackerCliSession(**base)


def test_forward_only_drives_engine_then_session(monkeypatch, tmp_path):
    class _CachePlan:
        inference_model_id = "bgsub_test"
        engine_model_id = None
        detection_cache_path = str(tmp_path / "cache.npz")

    monkeypatch.setattr(ht, "plan_tracking_cache", lambda *a, **k: _CachePlan())

    calls = {"engine_passes": [], "post": None}

    def _fake_engine_pass(session, *, params, raw_csv_path, backward_mode,
                          detection_cache_path, use_cached_detections, should_stop):
        calls["engine_passes"].append(backward_mode)
        assert params["INFERENCE_MODEL_ID"] == "bgsub_test"
        return True, [30.0, 30.0], pd.DataFrame({"TrajectoryID": [0], "X": [1]})

    monkeypatch.setattr(ht, "_run_engine_pass", _fake_engine_pass)

    def _fake_run_post(self, forward_trajectories, backward_trajectories=None):
        calls["post"] = (forward_trajectories, backward_trajectories)
        return SessionResult(
            success=True,
            final_csv_path=str(tmp_path / "out_final.csv"),
            rich_export_path=None,
            media_paths=[],
            dataset_result=None,
            summary_lines=["final_csv=out_final.csv"],
            error=None,
        )

    monkeypatch.setattr(
        "hydra_suite.core.tracking.session.TrackingSessionCore.run_post_tracking",
        _fake_run_post,
    )

    session = _make_session(tmp_path)
    result = ht.run_headless_tracking_session(session)

    assert result["success"] is True
    assert calls["engine_passes"] == [False]  # forward only
    assert calls["post"][1] is None  # no backward df
    assert any("avg_fps=30.0" in line for line in result["lines"])


def test_backward_enabled_runs_two_passes(monkeypatch, tmp_path):
    class _CachePlan:
        inference_model_id = "m"
        engine_model_id = None
        detection_cache_path = str(tmp_path / "cache.npz")

    monkeypatch.setattr(ht, "plan_tracking_cache", lambda *a, **k: _CachePlan())

    seen = {"passes": []}

    def _fake_engine_pass(session, *, params, raw_csv_path, backward_mode,
                          detection_cache_path, use_cached_detections, should_stop):
        seen["passes"].append((backward_mode, use_cached_detections))
        return True, [30.0], pd.DataFrame({"TrajectoryID": [0]})

    monkeypatch.setattr(ht, "_run_engine_pass", _fake_engine_pass)

    def _fake_run_post(self, forward_trajectories, backward_trajectories=None):
        assert backward_trajectories is not None  # backward df threaded through
        return SessionResult(True, str(tmp_path / "f.csv"), None, [], None, [], None)

    monkeypatch.setattr(
        "hydra_suite.core.tracking.session.TrackingSessionCore.run_post_tracking",
        _fake_run_post,
    )

    result = ht.run_headless_tracking_session(_make_session(tmp_path, enable_backward_tracking=True))
    assert result["success"] is True
    # forward pass first (uses cache flag from session), backward pass forces no-cache.
    assert seen["passes"] == [(False, False), (True, False)]


def test_forward_failure_short_circuits_before_session(monkeypatch, tmp_path):
    class _CachePlan:
        inference_model_id = "m"
        engine_model_id = None
        detection_cache_path = str(tmp_path / "cache.npz")

    monkeypatch.setattr(ht, "plan_tracking_cache", lambda *a, **k: _CachePlan())
    monkeypatch.setattr(
        ht, "_run_engine_pass",
        lambda *a, **k: (False, [], None),
    )

    def _must_not_run(*a, **k):
        raise AssertionError("session must not run after forward failure")

    monkeypatch.setattr(
        "hydra_suite.core.tracking.session.TrackingSessionCore.run_post_tracking",
        _must_not_run,
    )

    result = ht.run_headless_tracking_session(_make_session(tmp_path))
    assert result["success"] is False
    assert "forward" in result["error"].lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n hydra-mps python -m pytest tests/test_trackerkit_headless_tracking.py -q --ignore=tests/test_identity_postprocess.py`
Expected: FAIL — old module still imports `PySide6.QtCore` (fails `test_headless_tracking_module_imports_no_qt`) and has no `_run_engine_pass` (the driver tests fail with `AttributeError`).

- [ ] **Step 3: Rewrite the module**

First confirm the empty-output guard now lives in the service:

```bash
grep -n "_enforce_nonempty_forward\|_detection_cache_has_detections\|_csv_has_data_rows" src/hydra_suite/core/tracking/session.py
```

If those symbols are present in `session.py`, delete them from `headless_tracking.py` (as below). If — and only if — they are absent, keep the three helper functions verbatim from the current `headless_tracking.py:123-176` and call `_enforce_nonempty_forward(raw_csv_path, detection_cache_path)` after a successful forward pass inside `_run_engine_pass`.

Replace the entire contents of `src/hydra_suite/trackerkit/headless_tracking.py` with:

```python
"""Headless (Qt-free) tracking session runner for the TrackerKit CLI.

Drives the Qt-free ``TrackingEngineCore`` on plain threads for the forward
(and optional backward) pass, then hands the raw trajectories to the Qt-free
``TrackingSessionCore`` for post-processing/merge/export. No PySide6 import
anywhere in this module — this is the executable definition of "Qt-free CLI".
"""

from __future__ import annotations

import logging
import os
import signal
import threading
from typing import Any, Callable

import pandas as pd

from hydra_suite.core.tracking.session import (
    SessionCallbacks,
    SessionResult,
    TrackingSessionCore,
)
from hydra_suite.core.tracking.worker import TrackingEngineCore
from hydra_suite.data.csv_writer import CSVWriterThread
from hydra_suite.trackerkit.cli_config import TrackerCliSession
from hydra_suite.trackerkit.tracking_cache import plan_tracking_cache

logger = logging.getLogger(__name__)


def build_tracking_csv_header(
    save_confidence_metrics: bool, identity_method: str = "none_disabled"
) -> list[str]:
    """Build the raw tracking CSV header used by the GUI path."""
    if save_confidence_metrics:
        header = [
            "TrackID",
            "TrajectoryID",
            "Index",
            "X",
            "Y",
            "Theta",
            "FrameID",
            "State",
            "DetectionConfidence",
            "AssignmentConfidence",
            "PositionUncertainty",
            "DetectionID",
            "IdentityAssignedID",
            "IdentityAssignedLabel",
            "IdentityAssignedConfidence",
            "IdentityPosteriorMargin",
            "IdentityEntropy",
            "IdentityCommitted",
            "IdentityEvidenceSources",
            "IdentityConflictFlag",
            "IdentitySlotLockLabel",
        ]
    else:
        header = [
            "TrackID",
            "TrajectoryID",
            "Index",
            "X",
            "Y",
            "Theta",
            "FrameID",
            "State",
            "DetectionID",
            "IdentityAssignedID",
            "IdentityAssignedLabel",
            "IdentityAssignedConfidence",
            "IdentityPosteriorMargin",
            "IdentityEntropy",
            "IdentityCommitted",
            "IdentityEvidenceSources",
            "IdentityConflictFlag",
            "IdentitySlotLockLabel",
        ]
    if str(identity_method).strip().lower() == "apriltags":
        header.extend(
            [
                "DetectedTagID",
                "DetectedTagLabel",
                "DetectedTagConf",
                "DetectedTagHamming",
            ]
        )
    return header


def _read_raw_trajectories(raw_csv_path: str) -> pd.DataFrame | None:
    """Load raw tracked rows written by the engine into a DataFrame.

    Mirrors the CSV that ``PostProcessWorker`` used to read; the session's
    post-processing consumes exactly this raw shape.
    """
    if not os.path.exists(raw_csv_path):
        return None
    return pd.read_csv(raw_csv_path)


def _run_engine_pass(
    session: TrackerCliSession,
    *,
    params: dict[str, Any],
    raw_csv_path: str,
    backward_mode: bool,
    detection_cache_path: str,
    use_cached_detections: bool,
    should_stop: Callable[[], bool],
) -> tuple[bool, list[float], pd.DataFrame | None]:
    """Run one tracking pass on a plain thread; return (success, fps, raw_df).

    The engine writes its raw rows through ``CSVWriterThread`` (already a plain
    thread). We run ``run_tracking`` on a worker thread and join with a timeout
    so a SIGINT-driven ``should_stop`` can request a clean stop mid-pass.
    """
    direction = "backward" if backward_mode else "forward"
    csv_writer = CSVWriterThread(
        raw_csv_path,
        header=build_tracking_csv_header(
            session.save_confidence_metrics,
            identity_method=session.identity_method,
        ),
    )
    csv_writer.start()

    captured: dict[str, Any] = {"success": False, "fps_list": [], "finished": False}

    def _on_finished(success: bool, fps_list: list[Any], _full_traj: list[Any]) -> None:
        captured["success"] = bool(success)
        captured["fps_list"] = [f for f in (fps_list or []) if f and f > 0]
        captured["finished"] = True

    engine = TrackingEngineCore(
        session.video_path,
        csv_writer_thread=csv_writer,
        video_output_path=None,
        backward_mode=backward_mode,
        detection_cache_path=detection_cache_path,
        preview_mode=False,
        use_cached_detections=use_cached_detections,
        on_finished=_on_finished,
        on_progress=lambda pct, msg: logger.info(
            "[track %s] %d%% %s", direction, int(pct), msg
        ),
        on_warning=lambda title, msg: logger.warning("%s: %s", title, msg),
    )
    engine.set_parameters(dict(params))

    def _target() -> None:
        # TrackingEngineCore.run_tracking() does not guard every exception at the
        # top level (the old QThread wrapper did). Guard here so a crash still
        # produces finished(False) instead of a lost exception + a hung join.
        try:
            engine.run_tracking()
        except Exception:
            logger.exception("Tracking engine crashed during %s pass", direction)
            if not captured["finished"]:
                _on_finished(False, [], [])

    thread = threading.Thread(target=_target, name=f"tracking-engine-{direction}")
    thread.start()
    while thread.is_alive():
        if should_stop():
            engine.stop()
        thread.join(timeout=0.2)

    csv_writer.stop()
    csv_writer.join(timeout=10)

    if not captured["success"]:
        return False, [], None
    raw_df = _read_raw_trajectories(raw_csv_path)
    return True, list(captured["fps_list"]), raw_df


def _install_sigint_stop() -> tuple[threading.Event, Any, bool]:
    """Install a SIGINT handler that sets a stop event. Returns (event, prev, installed).

    ``signal.signal`` only works on the main thread; under a pytest worker it
    raises ``ValueError`` — in that case we skip installation and the session
    simply never self-cancels (fine for tests).
    """
    stop_event = threading.Event()
    previous = None
    installed = False
    try:
        previous = signal.getsignal(signal.SIGINT)

        def _handler(_signum, _frame):
            logger.warning("SIGINT received - requesting clean stop of tracking session.")
            stop_event.set()

        signal.signal(signal.SIGINT, _handler)
        installed = True
    except (ValueError, OSError):
        installed = False
    return stop_event, previous, installed


def _restore_sigint(previous: Any, installed: bool) -> None:
    if installed and previous is not None:
        try:
            signal.signal(signal.SIGINT, previous)
        except (ValueError, OSError):
            pass


def run_headless_tracking_session(
    session: TrackerCliSession,
    *,
    should_stop: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Run a TrackerKit session without any Qt state.

    ``should_stop`` overrides the built-in SIGINT flag (used by tests / GUI reuse).
    """
    if should_stop is None:
        stop_event, previous_handler, installed = _install_sigint_stop()
        effective_should_stop: Callable[[], bool] = stop_event.is_set
    else:
        stop_event, previous_handler, installed = None, None, False
        effective_should_stop = should_stop

    try:
        cache_plan = plan_tracking_cache(
            session.video_path,
            params=dict(session.params),
            preferred_output_dir=os.path.dirname(session.raw_csv_path),
            use_cached_detections=session.use_cached_detections,
        )
        params = dict(session.params)
        params["INFERENCE_MODEL_ID"] = cache_plan.inference_model_id
        if cache_plan.engine_model_id:
            params["ENGINE_MODEL_ID"] = cache_plan.engine_model_id
        detection_cache_path = cache_plan.detection_cache_path

        raw_base, raw_ext = os.path.splitext(session.raw_csv_path)
        if session.enable_backward_tracking:
            forward_raw_csv = f"{raw_base}_forward{raw_ext}"
            backward_raw_csv = f"{raw_base}_backward{raw_ext}"
        else:
            forward_raw_csv = session.raw_csv_path
            backward_raw_csv = None

        forward_ok, fps_fwd, forward_df = _run_engine_pass(
            session,
            params=params,
            raw_csv_path=forward_raw_csv,
            backward_mode=False,
            detection_cache_path=detection_cache_path,
            use_cached_detections=session.use_cached_detections,
            should_stop=effective_should_stop,
        )
        if not forward_ok:
            return {
                "success": False,
                "lines": [],
                "error": "An error occurred during forward tracking. Check logs for details.",
                "final_csv": None,
            }

        backward_df = None
        fps_bwd: list[float] = []
        if session.enable_backward_tracking:
            backward_ok, fps_bwd, backward_df = _run_engine_pass(
                session,
                params=params,
                raw_csv_path=backward_raw_csv,
                backward_mode=True,
                detection_cache_path=detection_cache_path,
                use_cached_detections=False,
                should_stop=effective_should_stop,
            )
            if not backward_ok:
                return {
                    "success": False,
                    "lines": [],
                    "error": "An error occurred during backward tracking. Check logs for details.",
                    "final_csv": None,
                }

        callbacks = SessionCallbacks(
            progress=lambda pct, msg: logger.info("[post] %d%% %s", int(pct), msg),
            status=lambda msg: logger.info("[post] %s", msg),
            warning=lambda title, msg: logger.warning("%s: %s", title, msg),
            stage_changed=lambda name: logger.debug("[post] stage: %s", name),
            should_stop=effective_should_stop,
        )
        service = TrackingSessionCore(
            video_path=session.video_path,
            config=session.config,
            params=params,
            paths={
                "raw_csv_path": session.raw_csv_path,
                "final_csv_path": session.final_csv_path,
                "detection_cache_path": detection_cache_path,
            },
            callbacks=callbacks,
        )
        result: SessionResult = service.run_post_tracking(
            forward_df, backward_trajectories=backward_df
        )

        if not result.success:
            return {
                "success": False,
                "lines": list(result.summary_lines or []),
                "error": result.error or "Tracker session failed.",
                "final_csv": result.final_csv_path,
            }

        lines = list(result.summary_lines or [])
        lines.insert(0, f"video={os.path.basename(session.video_path)}")
        fps_all = [f for f in (fps_fwd + fps_bwd) if f and f > 0]
        if fps_all:
            lines.append(f"avg_fps={sum(fps_all) / len(fps_all):.1f}")
        return {
            "success": True,
            "lines": lines,
            "error": None,
            "final_csv": result.final_csv_path,
        }
    finally:
        _restore_sigint(previous_handler, installed)
```

> If the grep in Step 3 showed the `paths` helper is named differently in `session.py` (e.g. a plain dict or `OutputPaths`), substitute that construction — copy the exact form used at the `TrackingSessionCore(` call site in `gui/orchestrators/tracking.py`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n hydra-mps python -m pytest tests/test_trackerkit_headless_tracking.py -q --ignore=tests/test_identity_postprocess.py`
Expected: PASS (4 passed — Qt-free guard + forward-only + backward-enabled + forward-failure).

- [ ] **Step 5: Commit**

```bash
make format
git add src/hydra_suite/trackerkit/headless_tracking.py tests/test_trackerkit_headless_tracking.py
git commit -m "feat(trackerkit): plain-threaded Qt-free headless driver via TrackingSessionCore"
```

---

## Task 3: Delete the bridge in `cli.py`

**Files:**
- Modify: `src/hydra_suite/trackerkit/cli.py` (whole file)
- Test: reuses `tests/test_trackerkit_cli_cutover.py`

**Interfaces:**
- Consumes: `run_headless_tracking_session` (Task 2 signature).
- Produces: `run_tracking_cli(video_paths, *, config_path=None, keystone_override=False) -> int` with only the direct path; no `MainWindow`, no `QApplication`, no `QMessageBox` monkeypatch.

**Deleted symbols:** `_suppress_message_boxes` (cli.py:23-56), `_ensure_qapplication` (59-67), `_prepare_video_session` (70-75), `_run_one_tracking_session` (78-97), `_run_bridge_tracking_session` (100-107), and the bridge branch + `finally` teardown inside `run_tracking_cli` (172-215).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_trackerkit_cli_cutover.py`:

```python
def test_cli_module_has_no_bridge_symbols():
    import hydra_suite.trackerkit.cli as cli

    for gone in (
        "_suppress_message_boxes",
        "_ensure_qapplication",
        "_prepare_video_session",
        "_run_one_tracking_session",
        "_run_bridge_tracking_session",
    ):
        assert not hasattr(cli, gone), f"bridge symbol still present: {gone}"


def test_cli_module_imports_no_qt():
    import ast
    from pathlib import Path

    import hydra_suite.trackerkit.cli as cli

    tree = ast.parse(Path(cli.__file__).read_text(), filename=cli.__file__)
    offenders = []
    for node in ast.walk(tree):
        mod = None
        if isinstance(node, ast.ImportFrom):
            mod = node.module
        elif isinstance(node, ast.Import):
            mod = ",".join(a.name for a in node.names)
        if mod and any(q in mod for q in ("PySide6", "QtCore", "QtWidgets", "QtGui")):
            offenders.append(f"{mod}:{node.lineno}")
    assert not offenders, "cli.py must be Qt-free: " + "; ".join(offenders)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n hydra-mps python -m pytest tests/test_trackerkit_cli_cutover.py -q --ignore=tests/test_identity_postprocess.py`
Expected: FAIL — bridge symbols still exist and `cli.py` still imports `PySide6.QtWidgets`/`QtCore` inside the deleted helpers.

- [ ] **Step 3: Rewrite `cli.py`**

Replace the entire contents of `src/hydra_suite/trackerkit/cli.py` with:

```python
"""Minimal TrackerKit CLI runner for config-driven tracking sessions (Qt-free)."""

from __future__ import annotations

import json
import logging
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Sequence

from hydra_suite.trackerkit.cli_config import (
    load_tracker_cli_config,
    load_tracker_cli_session,
)
from hydra_suite.trackerkit.headless_tracking import run_headless_tracking_session
from hydra_suite.trackerkit.session_plan import build_batch_video_plan

logger = logging.getLogger(__name__)


def run_tracking_cli(
    video_paths: Sequence[str],
    *,
    config_path: str | None = None,
    keystone_override: bool = False,
) -> int:
    """Run one or more TrackerKit sessions from the CLI (direct Qt-free path)."""

    videos = [str(path).strip() for path in video_paths if str(path).strip()]
    if not videos:
        raise ValueError("At least one video path is required.")

    for video_path in videos:
        if not Path(video_path).is_file():
            raise FileNotFoundError(f"Video not found: {video_path}")
    if config_path and not Path(config_path).is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")

    plan = build_batch_video_plan(
        videos,
        explicit_config_path=config_path,
        keystone_override=keystone_override,
    )
    if not plan:
        raise ValueError("No videos were resolved for tracking.")

    exit_code = 0
    with tempfile.TemporaryDirectory(prefix="trackerkit-cli-") as tmpdir:
        tmpdir_path = Path(tmpdir)
        baseline_config_data: dict[str, Any] | None = None

        for index, item in enumerate(plan, start=1):
            logger.info(
                "Tracker CLI: preparing video %s/%s: %s",
                index,
                len(plan),
                item.video_path,
            )
            effective_config_data = None
            if item.use_keystone_baseline and item.config_path is None:
                effective_config_data = baseline_config_data or {}
            session = load_tracker_cli_session(
                item.video_path,
                config_path=(
                    item.config_path if effective_config_data is None else None
                ),
                config_data=effective_config_data,
            )

            if index == 1:
                baseline_config_data = (
                    deepcopy(load_tracker_cli_config(item.config_path))
                    if item.config_path
                    else deepcopy(session.config)
                )

            # Persist the resolved keystone baseline for provenance/debugging; the
            # direct path consumes ``session`` directly and needs no config file.
            if item.use_keystone_baseline and item.config_path is None:
                keystone_dump = tmpdir_path / f"keystone_config_{index}.json"
                with open(keystone_dump, "w", encoding="utf-8") as handle:
                    json.dump(baseline_config_data or {}, handle, indent=2)

            result = run_headless_tracking_session(session)

            if result.get("success"):
                summary = " | ".join(result.get("lines", []))
                logger.info("Tracker CLI completed: %s", summary)
            else:
                error_message = result.get("error") or "Tracker session failed."
                logger.error(
                    "Tracker CLI failed for %s: %s",
                    item.video_path,
                    error_message,
                )
                exit_code = 1
                break

    return exit_code
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n hydra-mps python -m pytest tests/test_trackerkit_cli_cutover.py -q --ignore=tests/test_identity_postprocess.py`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
make format
git add src/hydra_suite/trackerkit/cli.py tests/test_trackerkit_cli_cutover.py
git commit -m "refactor(trackerkit): delete hidden-MainWindow bridge from CLI"
```

---

## Task 4: Delete the `_headless_*` hooks from the GUI

**Files:**
- Modify: `src/hydra_suite/trackerkit/gui/main_window.py:390-392`
- Modify: `src/hydra_suite/trackerkit/gui/orchestrators/tracking.py:2604-2607`, `2746-2751`, `4589-4613`

**Interfaces:**
- Consumes: nothing new.
- Produces: GUI orchestrator with no `_headless_*` reads/writes. `_handle_tracking_failed`, `on_postprocess_error`, and `_show_session_summary` keep only their GUI dialog behavior.

> **Re-grep first** — Slices 2/3 may already have moved parts of `_show_session_summary`. Run `grep -n "_headless_tracking_mode\|_headless_tracking_callback\|_headless_session_error" src/hydra_suite/trackerkit/gui/` and delete exactly the sites that remain, in the three methods below. The line numbers here are anchors from HEAD; re-derive them.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_trackerkit_cli_cutover.py`:

```python
def test_gui_headless_hooks_deleted():
    """No _headless_* references survive in the GUI after the bridge is gone."""
    import re
    from pathlib import Path

    import hydra_suite.trackerkit.gui as gui_pkg

    root = Path(gui_pkg.__file__).parent
    pattern = re.compile(
        r"_headless_tracking_mode|_headless_tracking_callback|_headless_session_error"
    )
    offenders = []
    for py in root.rglob("*.py"):
        for lineno, line in enumerate(py.read_text().splitlines(), start=1):
            if pattern.search(line):
                offenders.append(f"{py.relative_to(root)}:{lineno}")
    assert not offenders, "GUI still references _headless_* hooks: " + "; ".join(offenders)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n hydra-mps python -m pytest tests/test_trackerkit_cli_cutover.py::test_gui_headless_hooks_deleted -q --ignore=tests/test_identity_postprocess.py`
Expected: FAIL — lists `main_window.py:390-392` and the `orchestrators/tracking.py` sites.

- [ ] **Step 3a: Delete the init attributes in `main_window.py`**

The surrounding block (`main_window.py:388-393`) currently reads:

```python
        self._session_fps_list = []
        self._session_frames_processed = 0
        self._headless_tracking_mode = False
        self._headless_tracking_callback = None
        self._headless_session_error = None
        self._ui_settings = self._load_ui_settings()
```

becomes:

```python
        self._session_fps_list = []
        self._session_frames_processed = 0
        self._ui_settings = self._load_ui_settings()
```

- [ ] **Step 3b: Simplify `_handle_tracking_failed` in `tracking.py`**

The current `tracking.py:2601-2617` block:

```python
    def _handle_tracking_failed(self):
        """Show error dialog and finalize session when tracking did not finish normally."""
        logger.error("Tracking did not finish normally.")
        if getattr(self._mw, "_headless_tracking_mode", False):
            self._mw._headless_session_error = (
                "An error occurred during tracking. Check logs for details."
            )
        else:
            QMessageBox.warning(
                self._mw,
                "Tracking Failed",
                "An error occurred during tracking. Check logs for details.",
            )
        if self._panels.setup.g_batch.isChecked():
```

becomes:

```python
    def _handle_tracking_failed(self):
        """Show error dialog and finalize session when tracking did not finish normally."""
        logger.error("Tracking did not finish normally.")
        QMessageBox.warning(
            self._mw,
            "Tracking Failed",
            "An error occurred during tracking. Check logs for details.",
        )
        if self._panels.setup.g_batch.isChecked():
```

- [ ] **Step 3c: Simplify `on_postprocess_error` in `tracking.py`**

The current `tracking.py:2744-2757` block:

```python
        self._mw.progress_bar.setVisible(False)
        self._mw.progress_label.setVisible(False)
        if getattr(self._mw, "_headless_tracking_mode", False):
            self._mw._headless_session_error = (
                f"Error during trajectory post-processing: {error_message}"
            )
            self._finalize_tracking_session_ui()
            return
        QMessageBox.critical(
            self._mw,
            "Post-Processing Error",
            f"Error during trajectory post-processing:\n{error_message}",
        )
        logger.error(f"Trajectory post-processing error: {error_message}")
```

becomes:

```python
        self._mw.progress_bar.setVisible(False)
        self._mw.progress_label.setVisible(False)
        QMessageBox.critical(
            self._mw,
            "Post-Processing Error",
            f"Error during trajectory post-processing:\n{error_message}",
        )
        logger.error(f"Trajectory post-processing error: {error_message}")
```

- [ ] **Step 3d: Simplify `_show_session_summary` in `tracking.py`**

The current `tracking.py:4586-4615` block:

```python
    def _show_session_summary(self):
        """Show a single end-of-session summary dialog listing completed processes."""
        lines = self._build_session_summary_lines()
        error_message = str(
            getattr(self._mw, "_headless_session_error", "") or ""
        ).strip()
        if error_message:
            lines.extend(["", f"Error: {error_message}"])

        # Clean up state
        self._clear_session_summary_state()

        if getattr(self._mw, "_headless_tracking_mode", False):
            callback = getattr(self._mw, "_headless_tracking_callback", None)
            if callable(callback):
                callback(
                    {
                        "success": not bool(error_message),
                        "lines": lines,
                        "error": error_message or None,
                        "video_path": self._panels.setup.file_line.text() or None,
                        "csv_path": self._mw._session_final_csv_path
                        or self._panels.setup.csv_line.text()
                        or None,
                    }
                )
            self._mw._headless_session_error = None
            return

        QMessageBox.information(self._mw, "Tracking Complete", "\n".join(lines))
```

becomes:

```python
    def _show_session_summary(self):
        """Show a single end-of-session summary dialog listing completed processes."""
        lines = self._build_session_summary_lines()

        # Clean up state
        self._clear_session_summary_state()

        QMessageBox.information(self._mw, "Tracking Complete", "\n".join(lines))
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
conda run -n hydra-mps python -m pytest \
  tests/test_trackerkit_cli_cutover.py::test_gui_headless_hooks_deleted \
  tests/test_trackerkit_orchestrators_smoke.py \
  tests/test_trackerkit_tracking_orchestrator_dialogs.py \
  -q --ignore=tests/test_identity_postprocess.py
```
Expected: PASS — the hook-deletion guard passes and the GUI orchestrator smoke/dialog tests still pass.

- [ ] **Step 5: Commit**

```bash
make format
git add src/hydra_suite/trackerkit/gui/main_window.py src/hydra_suite/trackerkit/gui/orchestrators/tracking.py tests/test_trackerkit_cli_cutover.py
git commit -m "refactor(trackerkit): remove _headless_* bridge hooks from GUI"
```

---

## Task 5: Qt-free guard tests (executable definition of done)

**Files:**
- Create: `tests/test_headless_tracking_qtfree.py`

**Interfaces:**
- Consumes: `run_tracking_cli`, the `fly_obb` fixture clip + config.
- Produces: two guards — (a) `trackerkit.headless_tracking` + `trackerkit.cli` import under a PySide6-blocking `sys.meta_path` finder; (b) a subprocess runs `trackerkit track` on a short real clip with PySide6 blocked and writes a non-empty CSV (rows > 1).

This mirrors `tests/test_tracking_engine_core_qtfree.py` (its AST-walk `test_entire_core_tree_imports_no_qt` and module-import guards).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_headless_tracking_qtfree.py`:

```python
"""Executable definition of done: the CLI tracks with PySide6 blocked from import."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
FLY_CLIP = REPO / "tools/equivalence/fixtures/clips/fly_obb.mp4"
FLY_CONFIG = REPO / "tools/equivalence/fixtures/configs/fly_obb.json"

# A sys.meta_path finder that raises ImportError on ANY PySide6 import. Injected
# at interpreter start so even a lazy `import PySide6` deep in the CLI path fails.
_BLOCKER_PREAMBLE = textwrap.dedent(
    '''
    import sys

    class _BlockPySide6:
        def find_spec(self, name, path=None, target=None):
            if name == "PySide6" or name.startswith("PySide6."):
                raise ImportError(f"PySide6 import blocked by Qt-free guard: {name}")
            return None

    sys.meta_path.insert(0, _BlockPySide6())
    '''
)


def test_headless_tracking_imports_with_pyside6_blocked():
    """Importing the CLI runtime path must not require PySide6."""
    script = _BLOCKER_PREAMBLE + textwrap.dedent(
        '''
        import hydra_suite.trackerkit.headless_tracking  # noqa: F401
        import hydra_suite.trackerkit.cli  # noqa: F401
        import PySide6  # this line MUST raise, proving the blocker is live
        raise SystemExit("PySide6 was importable - blocker not active")
        '''
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=str(REPO),
    )
    # The final `import PySide6` raises ImportError -> nonzero exit with that text;
    # the two CLI imports above it must have SUCCEEDED (no traceback naming them).
    assert "PySide6 import blocked by Qt-free guard" in proc.stderr, proc.stderr
    assert "headless_tracking" not in proc.stderr, proc.stderr
    assert "trackerkit/cli" not in proc.stderr, proc.stderr


@pytest.mark.skipif(
    not (FLY_CLIP.exists() and FLY_CONFIG.exists()),
    reason="fly_obb fixture not present (run tools/equivalence/fixtures/fetch_fixtures.sh)",
)
def test_cli_tracks_to_completion_with_pyside6_blocked(tmp_path):
    """THE executable DoD: trackerkit track completes + writes a non-empty CSV, no PySide6."""
    clip = tmp_path / "fly_obb.mp4"
    clip.write_bytes(FLY_CLIP.read_bytes())  # copy so outputs land in tmp_path

    script = _BLOCKER_PREAMBLE + textwrap.dedent(
        f'''
        from hydra_suite.trackerkit.cli import run_tracking_cli
        code = run_tracking_cli([{str(clip)!r}], config_path={str(FLY_CONFIG)!r})
        raise SystemExit(int(code))
        '''
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=str(REPO),
    )
    assert proc.returncode == 0, f"CLI failed under blocked PySide6:\n{proc.stderr}"

    # Direct path writes <clip>_tracking_forward_processed.csv next to the clip.
    csvs = list(tmp_path.glob("*_forward_processed.csv")) + list(
        tmp_path.glob("*_final.csv")
    )
    assert csvs, f"no output CSV produced; stderr:\n{proc.stderr}"
    rows = sum(1 for _ in csvs[0].open())
    assert rows > 1, f"CSV {csvs[0]} has only {rows} line(s) (header-only or empty)"
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `conda run -n hydra-mps python -m pytest tests/test_headless_tracking_qtfree.py -q --ignore=tests/test_identity_postprocess.py`
Expected: PASS — after Tasks 2-3 the CLI path is Qt-free, so both guards pass. (If `test_cli_tracks_to_completion_with_pyside6_blocked` fails on a PySide6 import, the stderr names the offending module — fix that import to be lazy/GUI-only, then re-run.)

- [ ] **Step 3: (fix only if Step 2 fails)**

If Step 2 fails, treat the named import as the bug: move it into the GUI-launcher branch of `app.py` or behind a function-local import. Re-run Step 2 until green. (`app.py:182 check_dependencies` does NOT import PySide6, and `app.py`'s only PySide6 import is at line 253 inside the GUI-launch branch, so the `track` subcommand path is already clear — this step is a safety net.)

- [ ] **Step 4: Confirm the standing core-tree invariant**

```bash
grep -rnE "PySide6|QtCore|QThread|Signal|Slot|QMutex" src/hydra_suite/core/
```
Expected: no output. This is also covered by `tests/test_tracking_engine_core_qtfree.py::test_entire_core_tree_imports_no_qt`; run it:

Run: `conda run -n hydra-mps python -m pytest tests/test_tracking_engine_core_qtfree.py -q --ignore=tests/test_identity_postprocess.py`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
make format
git add tests/test_headless_tracking_qtfree.py
git commit -m "test(trackerkit): Qt-free guards - CLI tracks with PySide6 blocked"
```

---

## Task 6: No-bridge parity for the four previously-bridged clips

**Files:**
- Modify: `tests/test_trackerkit_cli_cutover.py`

**Interfaces:**
- Consumes: `run_tracking_cli`, the four fixture clips/configs.
- Produces: a test that runs each of `ant_pose_headtail`, `ant_obb_sleap`, `emi_obb_identity`, `ant_cnn_identity` through the CLI while asserting **no `MainWindow` is ever constructed** (patched to raise) and each writes a non-empty CSV — proving they now take the direct path.

Fixture → config mapping (from `tools/equivalence/run_matrix.sh:63-69`):

| clip | config | identity aux |
|---|---|---|
| `emi_obb_identity.mp4` | `emi_obb_identity.json` | — |
| `ant_pose_headtail.mp4` | `ant_pose_headtail.json` | `ooceraea_biroi.json` |
| `ant_obb_sleap.mp4` | `ant_obb_sleap.json` | `ooceraea_biroi.json` |
| `ant_cnn_identity.mp4` | `ant_cnn_identity.json` | `ooceraea_biroi.json` |

> Byte-identical parity is proven by the equivalence gate (Task 7). This test proves the weaker-but-executable property that matters for the cutover: these four clips complete via the **direct** path with **no bridge**. Pose/SLEAP clips require conda active + SLEAP models on the box; they SKIP when fixtures are missing but must PASS when present.

- [ ] **Step 1: Write the test**

Append to `tests/test_trackerkit_cli_cutover.py`:

```python
import pytest
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_FX = _REPO / "tools/equivalence/fixtures"

_BRIDGE_CLIPS = [
    ("emi_obb_identity.mp4", "emi_obb_identity.json"),
    ("ant_pose_headtail.mp4", "ant_pose_headtail.json"),
    ("ant_obb_sleap.mp4", "ant_obb_sleap.json"),
    ("ant_cnn_identity.mp4", "ant_cnn_identity.json"),
]


@pytest.mark.parametrize("clip_name,config_name", _BRIDGE_CLIPS)
def test_previously_bridged_clips_run_direct_no_mainwindow(
    clip_name, config_name, tmp_path, monkeypatch
):
    clip_src = _FX / "clips" / clip_name
    config = _FX / "configs" / config_name
    if not (clip_src.exists() and config.exists()):
        pytest.skip(f"fixture missing: {clip_name}")

    # Any attempt to construct MainWindow (the deleted bridge) is a hard failure.
    def _boom(*_a, **_k):
        raise AssertionError("MainWindow constructed - bridge path was taken")

    mw_mod = pytest.importorskip("hydra_suite.trackerkit.gui.main_window")
    monkeypatch.setattr(mw_mod, "MainWindow", _boom)

    clip = tmp_path / clip_name
    clip.write_bytes(clip_src.read_bytes())

    from hydra_suite.trackerkit.cli import run_tracking_cli

    code = run_tracking_cli([str(clip)], config_path=str(config))
    assert code == 0

    csvs = list(tmp_path.glob("*_forward_processed.csv")) + list(
        tmp_path.glob("*_final.csv")
    )
    assert csvs, f"{clip_name}: no output CSV"
    rows = sum(1 for _ in csvs[0].open())
    assert rows > 1, f"{clip_name}: CSV has only {rows} line(s)"
```

- [ ] **Step 2: Run the test**

Run: `conda run -n hydra-mps python -m pytest "tests/test_trackerkit_cli_cutover.py::test_previously_bridged_clips_run_direct_no_mainwindow" -q --ignore=tests/test_identity_postprocess.py`
Expected: PASS for every fixture present (conda active for SLEAP). If a clip errors, the log identifies the failing stage — fix before proceeding. The `MainWindow constructed` assertion must never fire.

- [ ] **Step 3: (no impl — behavior delivered by Tasks 1-3)**

This task adds only a test. If it fails on `MainWindow constructed`, a bridge remnant survived Task 3 — re-grep `cli.py` for `MainWindow`.

- [ ] **Step 4: Full targeted suite green**

Run:
```bash
conda run -n hydra-mps python -m pytest \
  tests/test_trackerkit_cli_cutover.py \
  tests/test_trackerkit_headless_tracking.py \
  tests/test_headless_tracking_qtfree.py \
  tests/test_tracking_engine_core_qtfree.py \
  tests/test_trackerkit_cli_config.py \
  -q --ignore=tests/test_identity_postprocess.py
```
Expected: PASS (fixture-gated tests may SKIP if fixtures/models absent).

- [ ] **Step 5: Commit**

```bash
make format
git add tests/test_trackerkit_cli_cutover.py
git commit -m "test(trackerkit): four pose/identity clips run direct (no bridge)"
```

---

## Task 7: Equivalence gate — MPS + CUDA byte-identical (FINAL)

**Files:** none (verification only). This task is the authoritative proof and must pass on BOTH platforms before the slice is done.

**Interfaces:** consumes the whole slice; produces the go/no-go signal for merge.

This is the most important equivalence run in the program: it proves the four previously-bridge-only clips (`ant_pose_headtail`, `ant_obb_sleap`, `emi_obb_identity`, `ant_cnn_identity`) are byte-identical via the direct path. Baseline = `legacy/main`; current = `HEAD` of this branch.

- [ ] **Step 1: MPS run on this box (`hydra-mps`)**

Conda MUST be active (the SLEAP service spawns `conda run -n sleap`; a bare shell yields EMPTY CSVs that FALSELY compare EQUIVALENT).

```bash
conda activate hydra-mps
bash tools/equivalence/fixtures/fetch_fixtures.sh          # once per machine
git fetch origin --tags
git worktree add --detach .worktrees/equiv-legacy legacy/main
REPO=$PWD WT=$PWD \
  MAIN_SRC=$PWD/.worktrees/equiv-legacy/src WT_SRC=$PWD/src \
  OUT=/tmp/equiv_slice4 RUNTIME=mps \
  bash tools/equivalence/run_matrix.sh
git worktree remove --force .worktrees/equiv-legacy && git worktree prune
```

- [ ] **Step 2: Verify MPS acceptance**

For every one of the 7 clips, confirm on BOTH `_forward.csv` and `_tracking_final.csv`:
- positions p99 ≈ 0, θ max ≈ 0, identical row counts, 0 unmatched;
- **row counts > 1** on the pose/SLEAP/identity clips (`ant_pose_headtail`, `ant_obb_sleap`, `emi_obb_identity`, `ant_cnn_identity`, `ant_obb_sequential`) — an empty CSV falsely passes;
- only accepted noise: bistable head/tail π-flips on head/tail clips (θ flips by π on some rows).

Expected: EQUIVALENCE at/near the DETERMINISM floor for all 7; PERFORMANCE ratio ≤ 1.25.

- [ ] **Step 3: CUDA run on mehek (`hydra-cuda`)**

```bash
ssh rutalab@mehek.taild08eb9.ts.net
cd ~/hydra-suite && git fetch origin --tags && git checkout <this-branch-sha>
source ~/mambaforge/etc/profile.d/conda.sh && conda activate hydra-cuda
bash tools/equivalence/fixtures/fetch_fixtures.sh          # once
git worktree add --detach .worktrees/equiv-legacy legacy/main
REPO=$PWD WT=$PWD MAIN_SRC=$PWD/.worktrees/equiv-legacy/src WT_SRC=$PWD/src \
  OUT=/tmp/equiv_slice4 RUNTIME=cuda nohup bash tools/equivalence/run_matrix.sh > /tmp/equiv_cuda.log 2>&1 &
```

- [ ] **Step 4: Verify CUDA acceptance**

Same acceptance criteria as Step 2, from `/tmp/equiv_cuda.log`. Pose/SLEAP clips REQUIRE the `sleap` conda env on the box + conda on PATH; verify row counts > 1. Clean up the worktree afterward (`git worktree remove --force .worktrees/equiv-legacy && git worktree prune`).

- [ ] **Step 5: Record the docs follow-up**

Log a follow-up (issue or `to_fix.md` entry) that `docs/superpowers/specs/2026-07-23-cloud-gpu-inference-design.md:123`'s `QT_QPA_PLATFORM=offscreen` requirement for the CLI container is now obsolete — the CLI path is Qt-free. Do not necessarily edit the spec in this slice; just log the follow-up.

Only when Steps 2 and 4 both pass is the bridge deletion verified and the slice complete.

---

## Self-Review

**Spec coverage (against the task's SCOPE + TESTS + GLOBAL CONSTRAINTS):**
- SCOPE 1 (rewrite `headless_tracking.py` on plain threads → engine → `run_post_tracking`; replace `ensure_headless_qt_application`/`_run_*_worker`/`_run_forward_*`/`run_headless_tracking_session`; `CSVWriterThread` already a plain thread): Task 2.
- SCOPE 2 (`supports_direct_run()` → `True`): Task 1.
- SCOPE 3 (delete `_run_bridge_tracking_session`, `_run_one_tracking_session`, `_prepare_video_session`, `_ensure_qapplication`, `_suppress_message_boxes`, MainWindow branch): Task 3.
- SCOPE 4 (delete `_headless_*` init + reads/writes in `main_window.py` and `tracking.py`, incl. `_show_session_summary`): Task 4.
- SCOPE 5 (SIGINT-backed `should_stop`; CLI callbacks progress→log, status→log, warning→WARNING, stage_changed→debug): Task 2 (`_install_sigint_stop`, `SessionCallbacks` wiring).
- TESTS — CLI parity for the 4 bridge clips via direct path: Task 6. Qt-free subprocess DoD with `sys.meta_path` finder (actual finder code + subprocess invocation provided): Task 5. `grep` core empty + import-under-finder: Task 5.
- GLOBAL — commit identity / `make format` / test command / env / equivalence gate on both platforms / docs follow-up: Global Constraints + Task 7.

**Placeholder scan:** every code step contains full source (no "similar to above", no TODO). The `paths` argument is pinned to Slice 2's plain-dict contract (`{"raw_csv_path", "final_csv_path", "detection_cache_path"}`), with a grep-to-confirm instruction in case Slice 2's implementer upgrades it to a typed dataclass.

**Type consistency:** `run_headless_tracking_session(session, *, should_stop=None) -> dict` is defined in Task 2 and consumed in Task 3. `_run_engine_pass(...) -> (bool, list[float], DataFrame|None)` is defined and monkeypatched with the same signature/return in Task 2's tests. `SessionCallbacks`/`SessionResult`/`TrackingSessionCore.run_post_tracking` names match the Global Constraints "EXISTING" contract. `supports_direct_run() -> bool` matches across Tasks 1 and 3.
</content>
