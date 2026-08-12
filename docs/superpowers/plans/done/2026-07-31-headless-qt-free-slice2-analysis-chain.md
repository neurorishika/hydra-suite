# Qt-Free Headless Session Service — Slice 2 (Analysis Chain) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Qt-free `TrackingSessionCore` (`core/tracking/session.py`) that owns the post-tracking *analysis* chain — merge → post-process → pose-source merge → identity post-pass → interpolated crops → rich export — and have the GUI orchestrator delegate to it, with the CLI path left untouched this slice.

**Architecture:** This is Slice 2 of the 4-slice program in `docs/superpowers/specs/2026-07-24-headless-qt-free-session-service-design.md`. It is a behavior-preserving-by-construction transform: every `signal.emit(...)` becomes a `callbacks.X(...)` call; every `self._mw.X` / `self._panels.X` widget read becomes a config-dict lookup, a service-state field read, or an injected callback; `QMessageBox.information/.warning` → `callbacks.warning(title, msg)`; `QMessageBox.critical` → `raise TrackingSessionError`. The interleaved GUI orchestrator methods are extracted **by name** into pure stage functions (`core/post/`, `core/individual/`) plus coupled `TrackingSessionCore` methods. The equivalence harness (byte-identical CSVs before/after) is the safety net.

**Tech Stack:** Python 3, pandas, numpy, OpenCV (`cv2`), pytest, conda env `hydra-mps`. No Qt anywhere under `src/hydra_suite/core/`.

**Slice 1 interfaces (treat as already existing — do not re-implement):**
- `hydra_suite.trackerkit.gui.orchestrators.config.build_config_dict()` — pure widget→dict body of `save_config()`, producing the lowercase config JSON dict (`enable_pose_extractor`, `identity_method`, `enable_postprocessing`, `interpolation_method`, `interpolation_max_gap_seconds`, `heading_flip_max_burst`, `resize_factor`, `fps`, `cleanup_temp_files`, `enable_backward_tracking`, `individual_interpolate_occlusions`, `pose_ignore_keypoints`, `video_output_enabled`, `video_output_path`, …), matching the vocabulary in `tools/equivalence/fixtures/configs/*.json`.
- `hydra_suite.core.tracking.session_policy` predicates, each taking a **config dict**: `is_individual_pipeline_enabled(config)`, `is_pose_inference_enabled(config)`, `is_headtail_compute_enabled(config)`, `should_export_final_canonical_images(config)`, `should_export_final_media_videos(config)`, `should_run_interpolated_postpass(config)`, `workflow_mode_key(config)`, `is_pose_export_enabled(config)`.
- `hydra_suite.core.tracking.session_summary.build_session_summary_lines(config, result)` — pure summary builder.
- A unified `TRAJECTORY_COLORS` helper on the GUI's legacy `np.random.seed(42)` + `np.random.randint(0,255,(N,3))` RNG.

## Global Constraints

- **This slice ships the service + GUI delegation only. The CLI path (`headless_tracking.py`, `cli.py`) is otherwise UNCHANGED in Slice 2** (Task 7 only re-homes the empty-output guard) — the full CLI cutover is Slice 4. `TrackerCliSession.supports_direct_run()` keeps its current value.
- Commit as the configured git user (`git config user.name` / `user.email`); **NO** `Co-Authored-By` trailer in any commit.
- Run `make format` before **each** commit (autopep8 → black → isort). If `make` is broken, run `conda run -n hydra-mps black <paths> && conda run -n hydra-mps isort <paths>` directly.
- Run tests with: `conda run -n hydra-mps python -m pytest <path> -q --ignore=tests/test_identity_postprocess.py` (that file has a pre-existing collection error — always keep the `--ignore`).
- After this slice, `grep -rnE "PySide6|QtCore|QThread|Signal|Slot|QMutex" src/hydra_suite/core/` MUST print nothing (empty; grep exit 1). Asserted by an automated test (Task 10).
- `core/` must never import from any app layer (`trackerkit`, `posekit`, …) or from `integrations`. It may import from `data/`, `training/`, `utils/`, `runtime/`.
- Non-fatal user notices go through `callbacks.warning(title, message)`; fatal errors `raise TrackingSessionError`; stages that today swallow-and-continue (e.g. the `logger.debug(..., exc_info=True)` blocks in pose merge, the rich-export trajectory count) keep doing so unchanged.
- Cancellation: check `callbacks.should_stop()` between stages and inside long loops (crop extraction), mirroring `TrackingEngineCore`'s `_stop_requested` idiom.
- Base suite has ~24 pre-existing failures — use a **delta gate** (no NEW failures), not zero-fail.
- **Equivalence gate (mandatory, Task 11):** `tools/equivalence/run_matrix.sh` with the same baseline before *and* after this slice, on **both** MPS (`hydra-mps`, this box) **and** CUDA (mehek, `hydra-cuda`), across all 7 fixture clips. Acceptance: positions p99 ≈ 0, θ max ≈ 0, identical row counts, 0 unmatched, on **both** `_forward.csv` and `_tracking_final.csv`. Known noise floor: bistable head/tail π-flips on head/tail clips only. **Conda MUST be active** for any pose/SLEAP clip or the CSVs come out empty and falsely compare EQUIVALENT — verify `wc -l` on the CSVs > 1 before trusting a pass.

---

## File Structure

**New files (all Qt-free):**
- `src/hydra_suite/core/tracking/errors.py` — `TrackingSessionError` (follows `core/individual/classification/errors.py` precedent).
- `src/hydra_suite/core/tracking/session.py` — `SessionCallbacks`, `SessionResult`, `TrackingSessionCore`, plus the moved empty-output guard functions.
- `src/hydra_suite/core/post/merge.py` — pure merge functions extracted from `MergeWorker`, plus the moved `_write_csv_artifact` / `_write_roi_npz` artifact writers.
- `src/hydra_suite/core/post/pose_merge.py` — pose-source detection + merge + quality post-pass (DataFrame→DataFrame).
- `src/hydra_suite/core/individual/postprocess_df.py` — identity post-pass (DataFrame→DataFrame).
- `src/hydra_suite/core/post/rich_export.py` — rich-export CSV builders/writers.
- `src/hydra_suite/core/post/interpolated_crops.py` — pure crop-extraction pipeline extracted from `InterpolatedCropsWorker`.
- `src/hydra_suite/trackerkit/gui/workers/session_worker.py` — thin `BaseWorker` running `TrackingSessionCore`.

**Modified files:**
- `src/hydra_suite/trackerkit/gui/workers/merge_worker.py` — `MergeWorker.execute()` becomes a thin caller of `core/post/merge.py`; the artifact writers are re-exported for compatibility.
- `src/hydra_suite/trackerkit/gui/workers/crops_worker.py` — `InterpolatedCropsWorker` becomes a thin caller of `core/post/interpolated_crops.py`.
- `src/hydra_suite/trackerkit/gui/orchestrators/tracking.py` — moved methods become thin delegations; the post-tracking chain runs via `SessionWorker`.
- `src/hydra_suite/trackerkit/headless_tracking.py` — `_enforce_nonempty_forward` and its `_csv_has_data_rows` / `_detection_cache_has_detections` helpers move to `core/`; the module re-imports them under their private names (CLI behavior byte-identical).

**Test files:**
- `tests/test_session_core_scaffold.py`, `tests/test_core_post_merge.py`, `tests/test_core_pose_merge.py`, `tests/test_core_identity_postprocess_df.py`, `tests/test_core_rich_export.py`, `tests/test_core_interpolated_crops.py`, `tests/test_nonempty_guard.py`, `tests/test_session_core_run.py`, `tests/test_session_worker.py`, `tests/test_core_qtfree_slice2.py`.

---

### Task 1: Session scaffold — `TrackingSessionError`, `SessionCallbacks`, `SessionResult`, `TrackingSessionCore.__init__`

**Files:**
- Create: `src/hydra_suite/core/tracking/errors.py`
- Create: `src/hydra_suite/core/tracking/session.py`
- Test: `tests/test_session_core_scaffold.py`

**Interfaces:**
- Produces (verbatim from spec — Slices 3 & 4 depend on these EXACT signatures):
  ```python
  @dataclass
  class SessionCallbacks:
      progress: Callable[[int, str], None] = _noop2
      status: Callable[[str], None] = _noop1
      warning: Callable[[str, str], None] = _noop2
      stage_changed: Callable[[str], None] = _noop1
      should_stop: Callable[[], bool] = _never

  @dataclass
  class SessionResult:
      success: bool
      final_csv_path: str | None
      rich_export_path: str | None
      media_paths: list[str]
      dataset_result: dict | None
      summary_lines: list[str]
      error: str | None

  class TrackingSessionCore:
      def __init__(self, *, video_path, config, params, paths, callbacks=SessionCallbacks()): ...
      def run_post_tracking(self, forward_trajectories, backward_trajectories=None) -> SessionResult: ...
  ```
- `TrackingSessionError(Exception)` in `core/tracking/errors.py`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_session_core_scaffold.py
import dataclasses

import pytest

from hydra_suite.core.tracking.errors import TrackingSessionError
from hydra_suite.core.tracking.session import (
    SessionCallbacks,
    SessionResult,
    TrackingSessionCore,
)


def test_tracking_session_error_is_exception():
    assert issubclass(TrackingSessionError, Exception)


def test_callbacks_defaults_are_silent_noops():
    cb = SessionCallbacks()
    assert cb.progress(50, "half") is None
    assert cb.status("working") is None
    assert cb.warning("Title", "Message") is None
    assert cb.stage_changed("merge") is None
    assert cb.should_stop() is False


def test_session_result_fields():
    names = {f.name for f in dataclasses.fields(SessionResult)}
    assert names == {
        "success",
        "final_csv_path",
        "rich_export_path",
        "media_paths",
        "dataset_result",
        "summary_lines",
        "error",
    }


def test_core_constructs_keyword_only_and_stores_state():
    core = TrackingSessionCore(
        video_path="/v.mp4",
        config={"enable_postprocessing": True},
        params={"FPS": 30.0},
        paths={"raw_csv_path": "/out.csv"},
    )
    assert core.video_path == "/v.mp4"
    assert core.config["enable_postprocessing"] is True
    assert core.params["FPS"] == 30.0
    assert core.paths["raw_csv_path"] == "/out.csv"
    assert isinstance(core.callbacks, SessionCallbacks)


def test_core_requires_keyword_arguments():
    with pytest.raises(TypeError):
        TrackingSessionCore("/v.mp4", {}, {}, {})  # positional not allowed
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n hydra-mps python -m pytest tests/test_session_core_scaffold.py -q --ignore=tests/test_identity_postprocess.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'hydra_suite.core.tracking.errors'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/hydra_suite/core/tracking/errors.py
"""Error types for the Qt-free tracking session service.

Follows the ``core/individual/classification/errors.py`` precedent: the service
raises a concrete type on fatal failure so the caller (GUI or CLI) decides
presentation instead of the service reaching for a widget.
"""

from __future__ import annotations


class TrackingSessionError(Exception):
    """Fatal failure inside the post-tracking session pipeline."""
```

```python
# src/hydra_suite/core/tracking/session.py
"""Qt-free post-tracking session service (Slice 2: analysis chain)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


def _noop1(_a) -> None:
    return None


def _noop2(_a, _b) -> None:
    return None


def _never() -> bool:
    return False


@dataclass
class SessionCallbacks:
    progress: Callable[[int, str], None] = _noop2
    status: Callable[[str], None] = _noop1
    warning: Callable[[str, str], None] = _noop2
    stage_changed: Callable[[str], None] = _noop1
    should_stop: Callable[[], bool] = _never


@dataclass
class SessionResult:
    success: bool
    final_csv_path: str | None
    rich_export_path: str | None
    media_paths: list[str]
    dataset_result: dict | None
    summary_lines: list[str]
    error: str | None


class TrackingSessionCore:
    """Owns the post-tracking analysis chain, Qt-free."""

    def __init__(self, *, video_path, config, params, paths, callbacks=None):
        self.video_path = video_path
        self.config = config
        self.params = params
        self.paths = paths
        self.callbacks = callbacks if callbacks is not None else SessionCallbacks()

    def run_post_tracking(self, forward_trajectories, backward_trajectories=None) -> SessionResult:
        raise NotImplementedError  # wired in Task 8
```

Note: `SessionCallbacks()` must not be used as a mutable dataclass **default argument value** in `__init__` (shared-instance footgun), so the signature uses `callbacks=None` and substitutes a fresh `SessionCallbacks()`; the observable default is still a no-op bundle, exactly as the spec's `callbacks=SessionCallbacks()` specifies.

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n hydra-mps python -m pytest tests/test_session_core_scaffold.py -q --ignore=tests/test_identity_postprocess.py`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
make format
git add src/hydra_suite/core/tracking/errors.py src/hydra_suite/core/tracking/session.py tests/test_session_core_scaffold.py
git commit -m "feat(session): scaffold Qt-free TrackingSessionCore + SessionResult/Callbacks"
```

---

### Task 2: `core/post/merge.py` — pure merge functions; rewire `MergeWorker`

Extract `MergeWorker`'s merge body (`merge_worker.py:141-215`) and its three helpers `_convert_resolved_to_dataframe` (`:63`), `_resolve_tag_identities` (`:88`), `_rescale_coordinates` (`:118`) into pure module functions, and move the two artifact writers `_write_csv_artifact` (`:218`) / `_write_roi_npz` (`:230`) into `core/post/merge.py` (Task 6 needs them Qt-free). `MergeWorker.execute()` becomes a thin caller that maps `progress_signal.emit`→a callback and `error_signal.emit`→exception handling.

**Files:**
- Create: `src/hydra_suite/core/post/merge.py`
- Modify: `src/hydra_suite/trackerkit/gui/workers/merge_worker.py`
- Test: `tests/test_core_post_merge.py`

**Interfaces:**
- Consumes: `hydra_suite.core.post.processing.resolve_trajectories`, `interpolate_trajectories` (existing).
- Produces:
  ```python
  def merge_trajectories(
      forward_trajs, backward_trajs, *, total_frames, params, resize_factor,
      interp_method, max_gap, tag_cache_path=None, heading_flip_max_burst=5,
      directed_heading_posthoc=False, enable_profiling=False,
      profile_export_path=None, progress=None, should_stop=None,
  ) -> object  # merged DataFrame (or the passthrough non-DataFrame), or None if stopped
  def convert_resolved_to_dataframe(resolved_trajectories) -> object
  def resolve_tag_identities(resolved_trajectories, *, tag_cache_path, params, progress=None) -> object
  def rescale_coordinates(resolved_trajectories, *, resize_factor) -> object
  def write_csv_artifact(path, fieldnames, rows) -> object
  def write_roi_npz(path, roi_rows, roi_corners) -> object
  ```

- [ ] **Step 1: Write the failing test**

```python
# tests/test_core_post_merge.py
import pandas as pd

from hydra_suite.core.post.merge import (
    convert_resolved_to_dataframe,
    merge_trajectories,
    rescale_coordinates,
    write_csv_artifact,
)


def _traj(tid, xs):
    return pd.DataFrame(
        {
            "TrajectoryID": tid,
            "X": xs,
            "Y": [10.0] * len(xs),
            "Theta": [0.0] * len(xs),
            "FrameID": list(range(len(xs))),
        }
    )


def test_convert_resolved_reassigns_trajectory_ids():
    out = convert_resolved_to_dataframe([_traj(99, [1.0, 2.0]), _traj(99, [3.0, 4.0])])
    assert isinstance(out, pd.DataFrame)
    assert sorted(out["TrajectoryID"].unique().tolist()) == [0, 1]


def test_rescale_coordinates_divides_by_resize_factor():
    df = _traj(0, [10.0, 20.0])
    out = rescale_coordinates(df, resize_factor=0.5)
    assert out["X"].tolist() == [20.0, 40.0]


def test_merge_reports_progress_and_returns_dataframe():
    seen = []
    merged = merge_trajectories(
        _traj(0, [1.0, 2.0, 3.0]),
        _traj(0, [1.0, 2.0, 3.0]),
        total_frames=3,
        params={},
        resize_factor=1.0,
        interp_method="none",
        max_gap=1,
        progress=lambda v, m: seen.append((v, m)),
    )
    assert isinstance(merged, pd.DataFrame)
    assert (100, "Merge complete!") in seen


def test_merge_honours_should_stop_before_completion():
    merged = merge_trajectories(
        _traj(0, [1.0, 2.0]),
        _traj(0, [1.0, 2.0]),
        total_frames=2,
        params={},
        resize_factor=1.0,
        interp_method="none",
        max_gap=1,
        should_stop=lambda: True,
    )
    assert merged is None


def test_write_csv_artifact_roundtrip(tmp_path):
    p = tmp_path / "a.csv"
    out = write_csv_artifact(str(p), ["k"], [{"k": 1}, {"k": 2}])
    assert out == str(p)
    assert p.read_text().splitlines()[0] == "k"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n hydra-mps python -m pytest tests/test_core_post_merge.py -q --ignore=tests/test_identity_postprocess.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'hydra_suite.core.post.merge'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/hydra_suite/core/post/merge.py`. Move the helper bodies **verbatim** from `merge_worker.py`, converting `self.<attr>` reads to explicit parameters and `self.progress_signal.emit(v, m)` to `progress(v, m)` (guarded by `if progress is not None`). Move `_write_csv_artifact`/`_write_roi_npz` verbatim (they already take only positional args).

```python
"""Pure trajectory-merge functions (Qt-free), extracted from MergeWorker."""

from __future__ import annotations

import csv
import logging

import numpy as np
import pandas as pd

from hydra_suite.core.post.processing import (
    interpolate_trajectories,
    resolve_trajectories,
)

logger = logging.getLogger(__name__)


def convert_resolved_to_dataframe(resolved_trajectories):
    """Convert a list of resolved trajectories to a single DataFrame."""
    if not resolved_trajectories or not isinstance(resolved_trajectories, list):
        return resolved_trajectories
    if isinstance(resolved_trajectories[0], pd.DataFrame):
        for new_id, traj_df in enumerate(resolved_trajectories):
            traj_df["TrajectoryID"] = new_id
        return pd.concat(resolved_trajectories, ignore_index=True)
    logger.warning("Received tuple format from resolve_trajectories, converting...")
    all_data = []
    for traj_id, traj in enumerate(resolved_trajectories):
        for x, y, theta, frame in traj:
            all_data.append(
                {
                    "TrajectoryID": traj_id,
                    "X": x,
                    "Y": y,
                    "Theta": theta,
                    "FrameID": frame,
                }
            )
    return pd.DataFrame(all_data) if all_data else []


def resolve_tag_identities(resolved_trajectories, *, tag_cache_path, params, progress=None):
    """Apply AprilTag identity resolution if a tag cache is available."""
    if not isinstance(resolved_trajectories, pd.DataFrame) or tag_cache_path is None:
        return resolved_trajectories
    try:
        from hydra_suite.core.post.tag_identity import (
            detect_tag_swaps,
            resolve_tag_identities as _resolve_tag_identities,
        )
        from hydra_suite.data.tag_observation_cache import TagObservationCache

        if progress is not None:
            progress(92, "Resolving tag identities...")
        tag_cache = TagObservationCache(str(tag_cache_path), mode="r")
        resolved_trajectories = _resolve_tag_identities(
            resolved_trajectories, tag_cache, params
        )
        swaps = detect_tag_swaps(resolved_trajectories, tag_cache, params)
        if swaps:
            logger.warning("Detected %d potential tag-swap events", len(swaps))
        tag_cache.close()
    except Exception:
        logger.warning("Tag identity resolution failed (non-fatal)", exc_info=True)
    return resolved_trajectories


def rescale_coordinates(resolved_trajectories, *, resize_factor):
    """Scale coordinates back to original video space."""
    if not isinstance(resolved_trajectories, pd.DataFrame):
        return resolved_trajectories
    logger.info(
        f"Pre-scaling (resize_factor={resize_factor:.3f}): "
        f"X range [{resolved_trajectories['X'].min():.1f}, {resolved_trajectories['X'].max():.1f}], "
        f"Y range [{resolved_trajectories['Y'].min():.1f}, {resolved_trajectories['Y'].max():.1f}]"
    )
    resolved_trajectories[["X", "Y"]] = resolved_trajectories[["X", "Y"]] / resize_factor
    if "Width" in resolved_trajectories.columns:
        resolved_trajectories["Width"] /= resize_factor
    if "Height" in resolved_trajectories.columns:
        resolved_trajectories["Height"] /= resize_factor
    logger.info(
        f"Post-scaling: "
        f"X range [{resolved_trajectories['X'].min():.1f}, {resolved_trajectories['X'].max():.1f}], "
        f"Y range [{resolved_trajectories['Y'].min():.1f}, {resolved_trajectories['Y'].max():.1f}]"
    )
    return resolved_trajectories


def merge_trajectories(
    forward_trajs,
    backward_trajs,
    *,
    total_frames,
    params,
    resize_factor,
    interp_method,
    max_gap,
    tag_cache_path=None,
    heading_flip_max_burst=5,
    directed_heading_posthoc=False,
    enable_profiling=False,
    profile_export_path=None,
    progress=None,
    should_stop=None,
):
    """Merge forward and backward trajectories. Returns merged DataFrame, or None if stopped."""
    from hydra_suite.core.tracking.profiler import TrackingProfiler

    def _stop() -> bool:
        return bool(should_stop()) if should_stop is not None else False

    def _emit(value, message) -> None:
        if progress is not None:
            progress(value, message)

    profiler = TrackingProfiler(enabled=enable_profiling)

    if _stop():
        return None
    profiler.phase_start("post_prepare")
    _emit(10, "Preparing trajectories...")

    def prepare_trajs_for_merge(trajs):
        if isinstance(trajs, pd.DataFrame):
            return [group for _, group in trajs.groupby("TrajectoryID")]
        return trajs

    forward_prepared = prepare_trajs_for_merge(forward_trajs)
    backward_prepared = prepare_trajs_for_merge(backward_trajs)
    profiler.phase_end("post_prepare")

    if _stop():
        return None
    profiler.phase_start("post_resolve")
    _emit(30, "Resolving trajectory conflicts...")
    resolved = resolve_trajectories(forward_prepared, backward_prepared, params=params)
    profiler.phase_end("post_resolve")

    if _stop():
        return None
    _emit(60, "Converting to DataFrame...")
    resolved = convert_resolved_to_dataframe(resolved)

    profiler.phase_start("post_interpolate")
    _emit(75, "Applying interpolation...")
    if isinstance(resolved, pd.DataFrame) and interp_method != "none":
        resolved = interpolate_trajectories(
            resolved,
            method=interp_method,
            max_gap=max_gap,
            heading_flip_max_burst=heading_flip_max_burst,
            directed_heading_posthoc=directed_heading_posthoc,
        )
    profiler.phase_end("post_interpolate")

    if _stop():
        return None
    _emit(90, "Scaling to original space...")
    profiler.phase_start("post_tag_identity")
    resolved = resolve_tag_identities(
        resolved, tag_cache_path=tag_cache_path, params=params, progress=progress
    )
    profiler.phase_end("post_tag_identity")

    profiler.phase_start("post_rescale")
    resolved = rescale_coordinates(resolved, resize_factor=resize_factor)
    profiler.phase_end("post_rescale")

    if _stop():
        return None
    profiler.log_final_summary()
    if profile_export_path:
        profiler.export_summary(profile_export_path)
    _emit(100, "Merge complete!")
    return resolved


def write_csv_artifact(path, fieldnames, rows):
    """Write a CSV artifact file. Returns the path on success, None on failure."""
    try:
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return path
    except Exception:
        return None


def write_roi_npz(path, roi_rows, roi_corners):
    """Write ROI data to a compressed NPZ file. Returns path on success, None on failure."""
    try:
        np.savez_compressed(
            str(path),
            frame_id=np.array([r["frame_id"] for r in roi_rows], dtype=np.int64),
            trajectory_id=np.array([r["trajectory_id"] for r in roi_rows], dtype=np.int64),
            filename=np.array([r["filename"] for r in roi_rows], dtype=object),
            cx=np.array([r["cx"] for r in roi_rows], dtype=np.float32),
            cy=np.array([r["cy"] for r in roi_rows], dtype=np.float32),
            w=np.array([r["w"] for r in roi_rows], dtype=np.float32),
            h=np.array([r["h"] for r in roi_rows], dtype=np.float32),
            theta=np.array([r["theta"] for r in roi_rows], dtype=np.float32),
            interp_from_start=np.array([r["interp_from_start"] for r in roi_rows], dtype=np.int64),
            interp_from_end=np.array([r["interp_from_end"] for r in roi_rows], dtype=np.int64),
            interp_index=np.array([r["interp_index"] for r in roi_rows], dtype=np.int64),
            interp_total=np.array([r["interp_total"] for r in roi_rows], dtype=np.int64),
            obb_corners=(
                np.stack(roi_corners).astype(np.float32)
                if roi_corners
                else np.zeros((0, 4, 2), dtype=np.float32)
            ),
        )
        return path
    except Exception:
        return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n hydra-mps python -m pytest tests/test_core_post_merge.py -q --ignore=tests/test_identity_postprocess.py`
Expected: PASS (5 passed).

- [ ] **Step 5: Rewire `MergeWorker` to call the core, keep writers importable**

In `merge_worker.py`, delete the three helper methods and the two module functions now living in `core/post/merge.py`, replace `execute()` with a thin caller, and re-export the writers so `crops_worker.py:26` (`from .merge_worker import _write_csv_artifact, _write_roi_npz`) keeps working.

Replace `MergeWorker.execute()` (`merge_worker.py:141-215`) with:

```python
    def execute(self):
        """Merge forward and backward trajectories (delegates to core/post/merge)."""
        from hydra_suite.core.post.merge import merge_trajectories

        try:
            merged = merge_trajectories(
                self.forward_trajs,
                self.backward_trajs,
                total_frames=self.total_frames,
                params=self.params,
                resize_factor=self.resize_factor,
                interp_method=self.interp_method,
                max_gap=self.max_gap,
                tag_cache_path=self.tag_cache_path,
                heading_flip_max_burst=self.heading_flip_max_burst,
                directed_heading_posthoc=self.directed_heading_posthoc,
                enable_profiling=self.enable_profiling,
                profile_export_path=self.profile_export_path,
                progress=lambda v, m: self.progress_signal.emit(v, m),
                should_stop=self._should_stop,
            )
            if merged is not None and not self._should_stop():
                self.finished_signal.emit(merged)
        except Exception as e:
            logger.exception("Error during trajectory merging")
            self.error_signal.emit(str(e))
```

Delete the now-dead `_convert_resolved_to_dataframe`, `_resolve_tag_identities`, `_rescale_coordinates` methods and the module-level `_write_csv_artifact`/`_write_roi_npz`. Add compatibility re-exports at the bottom of `merge_worker.py`:

```python
from hydra_suite.core.post.merge import (  # noqa: E402  (compat re-export)
    write_csv_artifact as _write_csv_artifact,
    write_roi_npz as _write_roi_npz,
)
```

Run the merge unit test plus a `crops_worker` import smoke:
`conda run -n hydra-mps python -m pytest tests/test_core_post_merge.py -q --ignore=tests/test_identity_postprocess.py && conda run -n hydra-mps python -c "import hydra_suite.trackerkit.gui.workers.crops_worker"`
Expected: PASS + no ImportError.

- [ ] **Step 6: Commit**

```bash
make format
git add src/hydra_suite/core/post/merge.py src/hydra_suite/trackerkit/gui/workers/merge_worker.py tests/test_core_post_merge.py
git commit -m "refactor(merge): extract MergeWorker logic to Qt-free core/post/merge"
```

---

### Task 3: `core/post/pose_merge.py` — pose-source merge + quality post-pass

Extract `_check_pose_export_sources` (`tracking.py:2759`), `_merge_pose_sources_into_df` (`:2827`), `_apply_pose_quality_postprocessing` (`:2985`), and `_resolve_current_tag_cache_path` (`:3125`). The `self._mw.current_*` reads become explicit fields of a `PoseSourceState` dataclass the service owns; the two remaining widget/param reads become explicit args: `spin_pose_min_kpt_conf_valid.value()` → `min_valid_conf`, `_parse_pose_ignore_keypoints()` → `ignore_keypoints` (already the config key `pose_ignore_keypoints`), `get_parameters_dict()` → `params`.

**Files:**
- Create: `src/hydra_suite/core/post/pose_merge.py`
- Test: `tests/test_core_pose_merge.py`

**Interfaces:**
- Consumes: `augment_trajectories_with_*` / `merge_interpolated_*` from `core/individual/properties/export`; `resolve_pose_group_indices` from `core/individual/pose/features`; the calibration/quality helpers from `core/individual/pose/quality`; `IndividualPropertiesCache` from `core/individual/properties/cache`.
- Produces:
  ```python
  @dataclass
  class PoseSourceState:
      individual_properties_cache_path: str | None = None
      detected_properties_cache_path: str | None = None
      detected_cnn_cache_paths: dict | None = None
      detection_cache_path: str | None = None
      interpolated_pose_csv_path: str | None = None
      interpolated_pose_df: object | None = None
      interpolated_tag_csv_path: str | None = None
      interpolated_tag_df: object | None = None
      interpolated_cnn_csv_paths: dict | None = None
      interpolated_cnn_dfs: dict | None = None
      interpolated_headtail_csv_path: str | None = None
      interpolated_headtail_df: object | None = None

  def check_pose_export_sources(state: PoseSourceState) -> tuple  # 7-tuple, same order as GUI
  def resolve_current_tag_cache_path(params, detection_cache_path) -> str
  def merge_pose_sources_into_df(trajectories_df, sources, state, *, params, min_valid_conf, ignore_keypoints) -> object
  def apply_pose_quality_postprocessing(with_pose_df, pose_labels, params, *, individual_properties_cache_path) -> object
  ```
  where `sources` is the 7-tuple returned by `check_pose_export_sources`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_core_pose_merge.py
import pandas as pd

from hydra_suite.core.post.pose_merge import (
    PoseSourceState,
    check_pose_export_sources,
    resolve_current_tag_cache_path,
)


def test_check_sources_empty_state_reports_nothing():
    (has_other, cache_path, cache_ok, interp_path, interp_ok,
     interp_mem, interp_mem_ok) = check_pose_export_sources(PoseSourceState())
    assert has_other is False
    assert cache_ok is False
    assert interp_ok is False
    assert interp_mem_ok is False


def test_check_sources_detects_in_memory_pose_df():
    state = PoseSourceState(interpolated_pose_df=pd.DataFrame({"X": [1.0]}))
    result = check_pose_export_sources(state)
    assert result[6] is True  # interp_mem_available


def test_resolve_tag_cache_returns_empty_without_apriltags():
    assert resolve_current_tag_cache_path({"USE_APRILTAGS": False}, "/nope.npz") == ""
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n hydra-mps python -m pytest tests/test_core_pose_merge.py -q --ignore=tests/test_identity_postprocess.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'hydra_suite.core.post.pose_merge'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/hydra_suite/core/post/pose_merge.py`. Re-read the current bodies in full first (`sed -n '2759,3135p' src/hydra_suite/trackerkit/gui/orchestrators/tracking.py`, re-derive line numbers with `grep -n "    def "`). Move the four method bodies verbatim, applying exactly these substitutions:
- `self._mw.current_detected_properties_cache_path` → `state.detected_properties_cache_path`
- `self._mw.current_detected_cnn_cache_paths` → `state.detected_cnn_cache_paths`
- `self._mw.current_interpolated_tag_csv_path` / `_df` → `state.interpolated_tag_csv_path` / `_df`
- `self._mw.current_interpolated_cnn_csv_paths` / `_dfs` → `state.interpolated_cnn_csv_paths` / `_dfs`
- `self._mw.current_interpolated_headtail_csv_path` / `_df` → `state.interpolated_headtail_csv_path` / `_df`
- `self._mw.current_individual_properties_cache_path` → `state.individual_properties_cache_path`
- `self._mw.current_interpolated_pose_csv_path` → `state.interpolated_pose_csv_path`
- `self._mw.current_interpolated_pose_df` → `state.interpolated_pose_df`
- `self._mw.current_detection_cache_path` → the `detection_cache_path` arg (inside `resolve_current_tag_cache_path`)
- `self._mw.get_parameters_dict()` → the `params` arg
- `self._panels.identity.spin_pose_min_kpt_conf_valid.value()` → the `min_valid_conf` arg
- `self._mw._parse_pose_ignore_keypoints()` → the `ignore_keypoints` arg
- `self._resolve_current_tag_cache_path()` → `resolve_current_tag_cache_path(params, state.detection_cache_path)`

`getattr(self._mw, "current_X", None)` reads become plain attribute access on `state` (defaults already `None`). Preserve every `try/except … logger.debug(..., exc_info=True)` swallow-and-continue block verbatim (Global Constraint: stage-level degradation unchanged). Module header: `import glob as _glob`, `import json`, `import logging`, `import os`, `import numpy as np`, `import pandas as pd`, `logger = logging.getLogger(__name__)`.

`check_pose_export_sources` body (verbatim from `tracking.py:2762-2825` with the `getattr(self._mw, …)` reads rewritten to `state.…`):

```python
@dataclass
class PoseSourceState:
    individual_properties_cache_path: str | None = None
    detected_properties_cache_path: str | None = None
    detected_cnn_cache_paths: dict | None = None
    detection_cache_path: str | None = None
    interpolated_pose_csv_path: str | None = None
    interpolated_pose_df: object | None = None
    interpolated_tag_csv_path: str | None = None
    interpolated_tag_df: object | None = None
    interpolated_cnn_csv_paths: dict | None = None
    interpolated_cnn_dfs: dict | None = None
    interpolated_headtail_csv_path: str | None = None
    interpolated_headtail_df: object | None = None


def check_pose_export_sources(state):
    _detected_props_path = str(state.detected_properties_cache_path or "").strip()
    _has_detected_props = bool(_detected_props_path and os.path.exists(_detected_props_path))
    _detected_cnn_paths = state.detected_cnn_cache_paths or {}
    _has_detected_cnn = any(
        str(path).strip() and os.path.exists(str(path).strip())
        for path in _detected_cnn_paths.values()
    )
    _has_interp_tag = bool(
        state.interpolated_tag_csv_path
        or isinstance(state.interpolated_tag_df, pd.DataFrame)
    )
    _has_interp_cnn = bool(
        state.interpolated_cnn_csv_paths or state.interpolated_cnn_dfs
    )
    _has_interp_ht = bool(
        state.interpolated_headtail_csv_path
        or isinstance(state.interpolated_headtail_df, pd.DataFrame)
    )
    _has_other_analyses = (
        _has_detected_props or _has_detected_cnn or _has_interp_tag
        or _has_interp_cnn or _has_interp_ht
    )
    cache_path = str(state.individual_properties_cache_path or "").strip()
    cache_available = bool(cache_path and os.path.exists(cache_path))
    interp_pose_path = str(state.interpolated_pose_csv_path or "").strip()
    interp_available = bool(interp_pose_path and os.path.exists(interp_pose_path))
    interp_pose_df_mem = state.interpolated_pose_df
    interp_mem_available = (
        isinstance(interp_pose_df_mem, pd.DataFrame) and not interp_pose_df_mem.empty
    )
    return (
        _has_other_analyses, cache_path, cache_available, interp_pose_path,
        interp_available, interp_pose_df_mem, interp_mem_available,
    )


def resolve_current_tag_cache_path(params, detection_cache_path):
    if not bool(params.get("USE_APRILTAGS", False)):
        return ""
    if not detection_cache_path or not os.path.exists(str(detection_cache_path)):
        return ""
    pattern = str(detection_cache_path).replace(".npz", "") + "_tags_*.npz"
    candidates = sorted(_glob.glob(pattern))
    return str(candidates[-1]) if candidates else ""
```

`merge_pose_sources_into_df(trajectories_df, sources, state, *, params, min_valid_conf, ignore_keypoints)` — move `_merge_pose_sources_into_df`'s body (`tracking.py:2845-2983`) verbatim under the substitutions above; unpack `cache_path, cache_available, interp_pose_path, interp_available, interp_pose_df_mem, interp_mem_available = sources[1:]`. Inside, the `augment_trajectories_with_pose_cache(..., ignore_keypoints=self._mw._parse_pose_ignore_keypoints(), min_valid_conf=min_valid_conf, coordinate_scale=_coord_scale)` call takes the `ignore_keypoints`/`min_valid_conf` args; `_resize_factor = float(self._mw.get_parameters_dict().get("RESIZE_FACTOR", 1.0))` becomes `_resize_factor = float(params.get("RESIZE_FACTOR", 1.0))`; `TAG_IDENTITY_LABELS` read via `params`; `self._resolve_current_tag_cache_path()` → `resolve_current_tag_cache_path(params, state.detection_cache_path)`.

`apply_pose_quality_postprocessing(with_pose_df, pose_labels, params, *, individual_properties_cache_path)` — move `_apply_pose_quality_postprocessing`'s body (`tracking.py:2986-3123`) verbatim, replacing `self._mw.current_individual_properties_cache_path` with the `individual_properties_cache_path` arg.

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n hydra-mps python -m pytest tests/test_core_pose_merge.py -q --ignore=tests/test_identity_postprocess.py`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
make format
git add src/hydra_suite/core/post/pose_merge.py tests/test_core_pose_merge.py
git commit -m "refactor(pose): extract pose-source merge + quality post-pass to core/post/pose_merge"
```

---

### Task 4: `core/individual/postprocess_df.py` — identity post-pass

Extract `_apply_identity_postprocessing_to_df` (`tracking.py:3137-3261`). Its only widget read is `params = self._mw.get_parameters_dict()` → becomes a `params` arg. The rest is pure.

**Files:**
- Create: `src/hydra_suite/core/individual/postprocess_df.py`
- Test: `tests/test_core_identity_postprocess_df.py`

**Interfaces:**
- Produces: `def apply_identity_postprocessing_to_df(with_pose_df, params) -> object`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_core_identity_postprocess_df.py
import pandas as pd

from hydra_suite.core.individual.postprocess_df import apply_identity_postprocessing_to_df


def test_empty_df_passthrough():
    empty = pd.DataFrame()
    assert apply_identity_postprocessing_to_df(empty, {}).empty


def test_annotates_summary_columns_when_solver_disabled():
    df = pd.DataFrame(
        {
            "TrajectoryID": [0, 0],
            "FrameID": [0, 1],
            "IdentityAssignedLabel": ["antA", "antA"],
        }
    )
    out = apply_identity_postprocessing_to_df(df, {"ENABLE_IDENTITY_FRAGMENT_SOLVER": False})
    assert "IdentityEvidenceSources" in out.columns
    assert "IdentityConflictFlag" in out.columns
    assert out["IdentityConflictFlag"].tolist() == [0, 0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n hydra-mps python -m pytest tests/test_core_identity_postprocess_df.py -q --ignore=tests/test_identity_postprocess.py`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

Create `src/hydra_suite/core/individual/postprocess_df.py`. Re-read the source (`sed -n '3137,3261p' src/hydra_suite/trackerkit/gui/orchestrators/tracking.py`) and move the whole `_apply_identity_postprocessing_to_df` body verbatim as a module function `apply_identity_postprocessing_to_df(with_pose_df, params)`, deleting only the `params = self._mw.get_parameters_dict()` line (now the arg). Keep the nested `_annotate_identity_summary_columns`, `_row_sources`, `_row_conflict` closures exactly as written, the `import itertools as _itertools` and the three `from hydra_suite.core...` imports inside the `try`, the entire CNN/tag label-catalog build, the `run_fragment_solver` call, `fill_identity_nans_with_consensus`, `sort_trajectories_by_identity`, and every `logger.exception` swallow block unchanged. Module header: `import logging`, `import numpy as np`, `import pandas as pd`, `logger = logging.getLogger(__name__)`.

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n hydra-mps python -m pytest tests/test_core_identity_postprocess_df.py -q --ignore=tests/test_identity_postprocess.py`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
make format
git add src/hydra_suite/core/individual/postprocess_df.py tests/test_core_identity_postprocess_df.py
git commit -m "refactor(identity): extract identity post-pass to core/individual/postprocess_df"
```

---

### Task 5: `core/post/rich_export.py` — rich-export builders/writers

Extract the rich-export helpers: `_rich_export_path` (`tracking.py:87`), `_write_rich_export_csv` (`:93`), `_drop_empty_rich_export_columns` (`:114`), `_remove_legacy_rich_exports` (`:127`), `_log_rich_export_summary` (`:1119`), `_count_augmented_pose_rows` (`:1082`), `_count_interpolated_cnn_rows` (`:1095`), `_build_rich_export_dataframe` (`:3263`), `_export_rich_csv` (`:3344`), `_relink_and_export_rich_csv` (`:3352`). These compose the Task 3/4 pure functions. The **verified** suffix constants (from `tracking.py:49-50`) are `RICH_EXPORT_SUFFIX = "_with_individual"` and `LEGACY_RICH_EXPORT_SUFFIX = "_with_pose"`.

**Files:**
- Create: `src/hydra_suite/core/post/rich_export.py`
- Test: `tests/test_core_rich_export.py`

**Interfaces:**
- Consumes: Task 3 (`PoseSourceState`, `check_pose_export_sources`, `merge_pose_sources_into_df`, `apply_pose_quality_postprocessing`), Task 4 (`apply_identity_postprocessing_to_df`), `core.post.processing.relink_trajectories_with_pose`.
- Produces:
  ```python
  RICH_EXPORT_SUFFIX = "_with_individual"
  LEGACY_RICH_EXPORT_SUFFIX = "_with_pose"

  def rich_export_path(final_csv_path, *, legacy=False) -> str
  def drop_empty_rich_export_columns(rich_df) -> object
  def write_rich_export_csv(rich_df, final_csv_path) -> str | None
  def remove_legacy_rich_exports(final_csv_path) -> None
  def log_rich_export_summary(df) -> None
  def count_augmented_pose_rows(with_pose_df) -> tuple
  def count_interpolated_cnn_rows(with_pose_df) -> str
  def build_rich_export_dataframe(final_csv_path, state, *, params, min_valid_conf, ignore_keypoints) -> object | None
  def export_rich_csv(final_csv_path, state, *, params, min_valid_conf, ignore_keypoints) -> str | None
  def relink_and_export_rich_csv(final_csv_path, state, *, params, min_valid_conf, ignore_keypoints) -> str | None
  ```

- [ ] **Step 1: Write the failing test**

```python
# tests/test_core_rich_export.py
import os

import pandas as pd

from hydra_suite.core.post.rich_export import (
    drop_empty_rich_export_columns,
    rich_export_path,
    write_rich_export_csv,
)


def test_rich_export_path_suffixes():
    assert rich_export_path("/a/b_final.csv") == "/a/b_final_with_individual.csv"
    assert rich_export_path("/a/b_final.csv", legacy=True) == "/a/b_final_with_pose.csv"


def test_drop_empty_columns_removes_all_nan():
    df = pd.DataFrame({"keep": [1, 2], "drop": [None, None]})
    out = drop_empty_rich_export_columns(df)
    assert list(out.columns) == ["keep"]


def test_write_rich_export_removes_legacy_alias(tmp_path):
    final = tmp_path / "clip_final.csv"
    legacy = tmp_path / "clip_final_with_pose.csv"
    legacy.write_text("stale\n")
    out = write_rich_export_csv(pd.DataFrame({"X": [1, 2]}), str(final))
    assert out == str(tmp_path / "clip_final_with_individual.csv")
    assert not os.path.exists(str(legacy))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n hydra-mps python -m pytest tests/test_core_rich_export.py -q --ignore=tests/test_identity_postprocess.py`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

Create `src/hydra_suite/core/post/rich_export.py`. Re-read each helper (`tracking.py:87-137`, `1082-1215`, `3263-3417`) and move each verbatim as a module function, with these substitutions:
- `self._rich_export_path` → `rich_export_path`; `self._drop_empty_rich_export_columns` → `drop_empty_rich_export_columns`; `self._write_rich_export_csv` → `write_rich_export_csv`; `self._remove_legacy_rich_exports` → `remove_legacy_rich_exports`; `self._log_rich_export_summary` → `log_rich_export_summary`; `self._count_augmented_pose_rows` → `count_augmented_pose_rows`; `self._count_interpolated_cnn_rows` → `count_interpolated_cnn_rows`.
- In `build_rich_export_dataframe`: replace `self._check_pose_export_sources()` with `sources = check_pose_export_sources(state)` (then unpack the same 7 names), `self._merge_pose_sources_into_df(trajectories_df, cache_path, cache_available, interp_pose_path, interp_available, interp_pose_df_mem, interp_mem_available)` with `merge_pose_sources_into_df(trajectories_df, sources, state, params=params, min_valid_conf=min_valid_conf, ignore_keypoints=ignore_keypoints)`, `self._apply_pose_quality_postprocessing(with_pose_df, pose_labels, params)` with `apply_pose_quality_postprocessing(with_pose_df, pose_labels, params, individual_properties_cache_path=state.individual_properties_cache_path)`, `self._apply_identity_postprocessing_to_df(with_pose_df)` with `apply_identity_postprocessing_to_df(with_pose_df, params)`, and `params = self._mw.get_parameters_dict()` with the `params` arg.
- In `relink_and_export_rich_csv`: replace `self._build_rich_export_dataframe(final_csv_path)` with `build_rich_export_dataframe(final_csv_path, state, params=params, min_valid_conf=min_valid_conf, ignore_keypoints=ignore_keypoints)`, `params = self._mw.get_parameters_dict()` with the arg, `self._export_rich_csv(final_csv_path)` with `export_rich_csv(final_csv_path, state, params=params, min_valid_conf=min_valid_conf, ignore_keypoints=ignore_keypoints)`, and `self._write_rich_export_csv` / `self._remove_legacy_rich_exports` / `self._rich_export_path` with the module functions. Keep the `from hydra_suite.core.post.processing import relink_trajectories_with_pose` import inside the function.

Preserve `_log_rich_export_summary`'s full body (the `fill`/`fill_any`/`pct` closures and per-source logging) verbatim as `log_rich_export_summary(df)`. Module header: `import logging`, `import os`, `import re`, `import pandas as pd`, the two suffix constants, `from hydra_suite.core.post.pose_merge import (apply_pose_quality_postprocessing, check_pose_export_sources, merge_pose_sources_into_df)`, `from hydra_suite.core.individual.postprocess_df import apply_identity_postprocessing_to_df`.

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n hydra-mps python -m pytest tests/test_core_rich_export.py -q --ignore=tests/test_identity_postprocess.py`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
make format
git add src/hydra_suite/core/post/rich_export.py tests/test_core_rich_export.py
git commit -m "refactor(export): extract rich-export CSV builders to core/post/rich_export"
```

---

### Task 6: `core/post/interpolated_crops.py` — pure crop-extraction pipeline

`InterpolatedCropsWorker.execute()` (`crops_worker.py:1453-1630`) and its helpers (`_validate_and_setup`, `_detect_interpolation_gaps`, `_init_interpolation_backends`, `_run_frame_tasks_loop`, `_write_interpolation_artifacts`, `_build_finished_payload`, `_cleanup_backends`, `_empty_artifact_paths`) reference only `self.csv_path`, `self.video_path`, `self.detection_cache_path`, `self.params`, `self.enable_profiling`, `self.profile_export_path`, `self._should_stop()`, and `self.progress_signal.emit`. Extract them into a pure module function `run_interpolated_crops(...)` that returns the finished-payload dict; the worker becomes a thin wrapper.

**Files:**
- Create: `src/hydra_suite/core/post/interpolated_crops.py`
- Modify: `src/hydra_suite/trackerkit/gui/workers/crops_worker.py`
- Test: `tests/test_core_interpolated_crops.py`

**Interfaces:**
- Consumes: `core.post.merge.write_csv_artifact` / `write_roi_npz` (Task 2), `IndividualDatasetGenerator`, `load_pose_backend`, `DetectionCache`, `TrackingProfiler`, `core.canonicalization.crop` helpers.
- Produces:
  ```python
  def run_interpolated_crops(
      csv_path, video_path, detection_cache_path, params, *,
      enable_profiling=False, profile_export_path=None,
      progress=None, should_stop=None,
  ) -> dict  # the payload dict the worker emitted (keys: saved, gaps, mapping_path,
             # roi_csv_path, roi_npz_path, pose_csv_path, pose_rows, tag_csv_path, tag_rows,
             # cnn_csv_paths, cnn_rows, headtail_csv_path, headtail_rows, occluded_rows,
             # interp_runs, eligible_frames, eligible_rows, roi_rows_cached, no_work_reason, ...)
  ```

- [ ] **Step 1: Write the failing test**

```python
# tests/test_core_interpolated_crops.py
from hydra_suite.core.post.interpolated_crops import run_interpolated_crops


def test_missing_csv_returns_empty_payload(tmp_path):
    result = run_interpolated_crops(
        str(tmp_path / "nope.csv"),
        str(tmp_path / "nope.mp4"),
        str(tmp_path / "nope.npz"),
        {},
    )
    # _validate_and_setup returns None on a missing CSV; the pipeline yields the
    # documented "nothing produced" payload rather than raising.
    assert isinstance(result, dict)
    assert result.get("saved", 0) == 0


def test_should_stop_before_setup_returns_empty_payload(tmp_path):
    result = run_interpolated_crops(
        str(tmp_path / "any.csv"),
        str(tmp_path / "any.mp4"),
        str(tmp_path / "any.npz"),
        {},
        should_stop=lambda: True,
    )
    assert result.get("saved", 0) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n hydra-mps python -m pytest tests/test_core_interpolated_crops.py -q --ignore=tests/test_identity_postprocess.py`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

Re-read the worker in full first (`sed -n '1,60p;1453,1630p' src/hydra_suite/trackerkit/gui/workers/crops_worker.py` and the helper methods it calls). Create `src/hydra_suite/core/post/interpolated_crops.py`. Move the entire `InterpolatedCropsWorker` **helper method set** (every method `execute()` calls) and `execute()` into the module as one public function `run_interpolated_crops(...)` plus module-level `_`-prefixed helpers, applying these substitutions throughout:
- `self.csv_path` → `csv_path`, `self.video_path` → `video_path`, `self.detection_cache_path` → `detection_cache_path`, `self.params` → `params`, `self.enable_profiling` → `enable_profiling`, `self.profile_export_path` → `profile_export_path`.
- `self._should_stop()` → an internal `_stop()` wrapper: `return bool(should_stop()) if should_stop is not None else False`.
- `self.progress_signal.emit(v, m)` → `_emit(v, m)` where `_emit` is `if progress is not None: progress(v, m)`.
- `self.finished_signal.emit(payload)` in `execute()` → `return payload` (the function returns the dict instead of emitting); the early-`return` sites inside `execute()` (`setup is None`, `gap_result is None`, `interp_saved is None`) → `return {"saved": 0, "gaps": 0}`; the `except Exception:` block → `return {"saved": 0, "gaps": 0}`.
- `from .merge_worker import _write_csv_artifact, _write_roi_npz` (used inside `_write_interpolation_artifacts`) → `from hydra_suite.core.post.merge import write_csv_artifact as _write_csv_artifact, write_roi_npz as _write_roi_npz`.

Keep the `finally: _cleanup_backends(...)` block. Helper signatures drop `self` and take whatever they read (`params`, `csv_path`, `should_stop`, etc.) as explicit args; thread the six former-instance attributes through. Preserve every log line, profiler phase, and control-flow branch verbatim. Module imports mirror `crops_worker.py:1-27` minus the `PySide6` / `BaseWorker` / `.merge_worker` lines (`gc`, `logging`, `math`, `os`, `collections.defaultdict`, `pathlib.Path`, `cv2`, `numpy`, `pandas`, and the `core.individual...` / `core.inference.api` / `data.detection_cache` / `utils.geometry` imports).

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n hydra-mps python -m pytest tests/test_core_interpolated_crops.py -q --ignore=tests/test_identity_postprocess.py`
Expected: PASS (2 passed).

- [ ] **Step 5: Rewire `InterpolatedCropsWorker` to call the core**

In `crops_worker.py`, delete the moved helper methods and replace `execute()` with:

```python
    def execute(self):
        """Generate interpolated crops (delegates to core/post/interpolated_crops)."""
        from hydra_suite.core.post.interpolated_crops import run_interpolated_crops

        payload = run_interpolated_crops(
            self.csv_path,
            self.video_path,
            self.detection_cache_path,
            self.params,
            enable_profiling=self.enable_profiling,
            profile_export_path=self.profile_export_path,
            progress=lambda v, m: self.progress_signal.emit(v, m),
            should_stop=self._should_stop,
        )
        if not self._should_stop():
            self.finished_signal.emit(payload)
```

Remove the now-unused `from .merge_worker import _write_csv_artifact, _write_roi_npz` line at `crops_worker.py:26`. Smoke-import: `conda run -n hydra-mps python -c "import hydra_suite.trackerkit.gui.workers.crops_worker"`.

- [ ] **Step 6: Commit**

```bash
make format
git add src/hydra_suite/core/post/interpolated_crops.py src/hydra_suite/trackerkit/gui/workers/crops_worker.py tests/test_core_interpolated_crops.py
git commit -m "refactor(crops): extract interpolated-crop pipeline to Qt-free core/post/interpolated_crops"
```

---

### Task 7: Move `_enforce_nonempty_forward` guard into `core/`

`_enforce_nonempty_forward` (`headless_tracking.py:157`) and its two O(1) helpers `_csv_has_data_rows` (`:123`) and `_detection_cache_has_detections` (`:136`) move into `core/tracking/session.py` so both the GUI and CLI gain the guard. The service (Task 8) calls it; `headless_tracking.py` re-imports it from `core/` under the private names (CLI behavior byte-identical).

**Files:**
- Modify: `src/hydra_suite/core/tracking/session.py`
- Modify: `src/hydra_suite/trackerkit/headless_tracking.py`
- Test: `tests/test_nonempty_guard.py`

**Interfaces:**
- Produces (in `session.py`):
  ```python
  def csv_has_data_rows(csv_path) -> bool
  def detection_cache_has_detections(detection_cache_path) -> bool
  def enforce_nonempty_forward(raw_csv_path, detection_cache_path) -> None  # raises TrackingSessionError
  ```

- [ ] **Step 1: Write the failing test**

```python
# tests/test_nonempty_guard.py
import numpy as np
import pytest

from hydra_suite.core.tracking.errors import TrackingSessionError
from hydra_suite.core.tracking.session import (
    csv_has_data_rows,
    detection_cache_has_detections,
    enforce_nonempty_forward,
)


def test_csv_has_data_rows(tmp_path):
    empty = tmp_path / "h.csv"
    empty.write_text("TrackID,X\n")
    assert csv_has_data_rows(str(empty)) is False
    full = tmp_path / "f.csv"
    full.write_text("TrackID,X\n1,2\n")
    assert csv_has_data_rows(str(full)) is True


def test_detection_cache_has_detections(tmp_path):
    p = tmp_path / "d.npz"
    np.savez(str(p), frame_0_meas=np.zeros((3, 3)))
    assert detection_cache_has_detections(str(p)) is True
    q = tmp_path / "e.npz"
    np.savez(str(q), frame_0_meas=np.zeros((0, 3)))
    assert detection_cache_has_detections(str(q)) is False


def test_enforce_raises_tracking_session_error(tmp_path):
    csv = tmp_path / "h.csv"
    csv.write_text("TrackID,X\n")  # header only
    cache = tmp_path / "d.npz"
    np.savez(str(cache), frame_0_meas=np.zeros((3, 3)))  # has detections
    with pytest.raises(TrackingSessionError):
        enforce_nonempty_forward(str(csv), str(cache))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n hydra-mps python -m pytest tests/test_nonempty_guard.py -q --ignore=tests/test_identity_postprocess.py`
Expected: FAIL with `ImportError: cannot import name 'csv_has_data_rows'`.

- [ ] **Step 3: Write minimal implementation**

Add to `session.py` (top-level, above the dataclasses). Move `_csv_has_data_rows`/`_detection_cache_has_detections` verbatim as `csv_has_data_rows`/`detection_cache_has_detections`; change `enforce_nonempty_forward` to raise `TrackingSessionError` instead of `RuntimeError` (keep the exact message). Add `import numpy as np` and `from hydra_suite.core.tracking.errors import TrackingSessionError` to `session.py`.

```python
def csv_has_data_rows(csv_path) -> bool:
    try:
        with open(csv_path, "r", encoding="utf-8") as fh:
            fh.readline()  # header
            return bool(fh.readline().strip())
    except OSError:
        return False


def detection_cache_has_detections(detection_cache_path) -> bool:
    try:
        with np.load(str(detection_cache_path), allow_pickle=True) as data:
            for key in data.files:
                if key.startswith("frame_") and key.endswith("_meas"):
                    arr = data[key]
                    if arr is not None and getattr(arr, "shape", (0,))[0] > 0:
                        return True
    except Exception:
        return False
    return False


def enforce_nonempty_forward(raw_csv_path, detection_cache_path) -> None:
    if detection_cache_has_detections(detection_cache_path) and not csv_has_data_rows(
        raw_csv_path
    ):
        raise TrackingSessionError(
            "Forward tracking produced ZERO tracked rows even though the detection "
            "cache contains detections. This indicates a silent pipeline failure "
            "(e.g. a crashed pose/identity stage or an out-of-memory abort). "
            "Refusing to emit an empty tracking CSV. "
            f"csv={raw_csv_path} detection_cache={detection_cache_path}"
        )
```

Then update `headless_tracking.py`: delete its three now-duplicated helper defs (`_csv_has_data_rows`, `_detection_cache_has_detections`, `_enforce_nonempty_forward`) and re-alias to the core, preserving the private names its callers (`_run_forward_only:320`, `_run_forward_backward:377`) use. `TrackingSessionError` is an `Exception`, so `cli.py`'s existing `except Exception` around the session still catches it (exit 1) — behavior preserved.

```python
# in headless_tracking.py, replace the three defs with:
from hydra_suite.core.tracking.session import (
    csv_has_data_rows as _csv_has_data_rows,
    detection_cache_has_detections as _detection_cache_has_detections,
    enforce_nonempty_forward as _enforce_nonempty_forward,
)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n hydra-mps python -m pytest tests/test_nonempty_guard.py -q --ignore=tests/test_identity_postprocess.py && conda run -n hydra-mps python -c "import hydra_suite.trackerkit.headless_tracking"`
Expected: PASS (3 passed) + no ImportError.

- [ ] **Step 5: Commit**

```bash
make format
git add src/hydra_suite/core/tracking/session.py src/hydra_suite/trackerkit/headless_tracking.py tests/test_nonempty_guard.py
git commit -m "refactor(guard): move nonempty-forward guard into core; raise TrackingSessionError"
```

---

### Task 8: `TrackingSessionCore.run_post_tracking` — wire the analysis chain

Implement the coupled service methods that chain the extracted stages, mirroring the GUI order in `_handle_forward_tracking_done` → `_handle_backward_tracking_done` → `merge_and_save_trajectories`/`on_merge_finished` → `_finish_tracking_session` → interp crops → `_relink_and_export_rich_csv`. Post-processing uses `process_trajectories_from_csv` on the raw CSVs in `paths` (byte-identical with both `PostProcessWorker` and the CLI). Slice-2 scope: `media_paths=[]`, `dataset_result=None`. Summary via `build_session_summary_lines(config, result)`.

**Behavioral note — postprocess reads CSV, not memory.** `PostProcessWorker` (`postprocess_worker.py:27`) runs `process_trajectories_from_csv` on the raw CSV path; the CLI does the same. To stay byte-identical, `run_post_tracking` post-processes from the raw CSV paths derived off `paths["raw_csv_path"]`, not the in-memory `forward_trajectories`/`backward_trajectories` args (which carry the raw tracked output the tracking stage already wrote to those CSVs). The trajectory args remain in the signature per spec and are accepted; the byte-identical path reads the CSVs.

**Files:**
- Modify: `src/hydra_suite/core/tracking/session.py`
- Test: `tests/test_session_core_run.py`

**Interfaces:**
- Consumes: `core.post.processing.{process_trajectories_from_csv, interpolate_trajectories}`, `core.post.merge.{merge_trajectories, rescale_coordinates}`, `core.post.pose_merge.{PoseSourceState, resolve_current_tag_cache_path}`, `core.post.interpolated_crops.run_interpolated_crops`, `core.post.rich_export.{export_rich_csv, relink_and_export_rich_csv}`, `core.tracking.session_summary.build_session_summary_lines`, `core.tracking.session_policy.should_run_interpolated_postpass`, and `enforce_nonempty_forward` (Task 7).
- `paths` dict keys: `raw_csv_path` (base for `_forward`/`_backward`/`_final`/`_forward_processed` derivations, mirroring `_handle_*`), `detection_cache_path`.
- `self.pose_state = PoseSourceState(...)` populated in `__init__` from `paths` (`detection_cache_path`) and from optional keys the GUI wrapper injects (`individual_properties_cache_path`, `detected_properties_cache_path`, `detected_cnn_cache_paths`).
- Config lookups (replacing widget reads), from `self.config`: `enable_postprocessing`, `interpolation_method`, `interpolation_max_gap_seconds`, `heading_flip_max_burst`, `enable_backward_tracking`, `individual_interpolate_occlusions`, `pose_ignore_keypoints`; from `self.params`: `FPS`, `RESIZE_FACTOR`, `POSE_MIN_KPT_CONF_VALID`, `DIRECTED_ORIENT_POSTHOC_CONSISTENCY`, `ENABLE_PROFILING`, `MIN_TRAJECTORY_LENGTH`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_session_core_run.py
import pandas as pd

from hydra_suite.core.tracking.session import (
    SessionCallbacks,
    SessionResult,
    TrackingSessionCore,
)


def _write_raw_csv(path):
    pd.DataFrame(
        {
            "TrajectoryID": [0, 0, 0],
            "X": [10.0, 11.0, 12.0],
            "Y": [5.0, 5.0, 5.0],
            "Theta": [0.0, 0.0, 0.0],
            "FrameID": [0, 1, 2],
            "State": ["tracked"] * 3,
        }
    ).to_csv(path, index=False)


def _config():
    return {
        "enable_postprocessing": False,
        "interpolation_method": "none",
        "interpolation_max_gap_seconds": 1.0,
        "heading_flip_max_burst": 5,
        "enable_backward_tracking": False,
        "individual_interpolate_occlusions": False,
    }


def test_forward_only_writes_final_csv(tmp_path):
    raw = tmp_path / "clip.csv"
    _write_raw_csv(str(raw))
    core = TrackingSessionCore(
        video_path=str(tmp_path / "clip.mp4"),
        config=_config(),
        params={"FPS": 30.0, "RESIZE_FACTOR": 1.0, "MIN_TRAJECTORY_LENGTH": 1},
        paths={"raw_csv_path": str(raw), "detection_cache_path": str(tmp_path / "d.npz")},
    )
    result = core.run_post_tracking(pd.read_csv(str(raw)))
    assert isinstance(result, SessionResult)
    assert result.success is True
    assert result.final_csv_path is not None
    assert pd.read_csv(result.final_csv_path).shape[0] == 3
    assert result.media_paths == []
    assert result.dataset_result is None
    assert isinstance(result.summary_lines, list)


def test_should_stop_between_stages_yields_unsuccessful_result(tmp_path):
    raw = tmp_path / "clip.csv"
    _write_raw_csv(str(raw))
    core = TrackingSessionCore(
        video_path=str(tmp_path / "clip.mp4"),
        config=_config(),
        params={"FPS": 30.0, "RESIZE_FACTOR": 1.0, "MIN_TRAJECTORY_LENGTH": 1},
        paths={"raw_csv_path": str(raw), "detection_cache_path": str(tmp_path / "d.npz")},
        callbacks=SessionCallbacks(should_stop=lambda: True),
    )
    result = core.run_post_tracking(pd.read_csv(str(raw)))
    assert result.success is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n hydra-mps python -m pytest tests/test_session_core_run.py -q --ignore=tests/test_identity_postprocess.py`
Expected: FAIL with `NotImplementedError`.

- [ ] **Step 3: Write minimal implementation**

Add a local `_save_trajectories_to_csv(trajectories, output_path)` to `session.py` copied verbatim from `headless_tracking.save_trajectories_to_csv` (`headless_tracking.py:94-120` — rounding of X/Y/FrameID, dropping `TrackID`/`Index`, base-column ordering), so `core/` needs no app import. Extend `__init__` to build `self.pose_state` from `paths`. Implement `run_post_tracking` and its private stage helpers:

```python
    def run_post_tracking(self, forward_trajectories, backward_trajectories=None) -> SessionResult:
        cb = self.callbacks
        try:
            if cb.should_stop():
                return self._stopped_result()

            raw_csv = self.paths.get("raw_csv_path")
            detection_cache = self.paths.get("detection_cache_path", "")
            base, ext = os.path.splitext(raw_csv) if raw_csv else ("", ".csv")
            backward_enabled = bool(self.config.get("enable_backward_tracking"))

            cb.stage_changed("postprocess")
            if raw_csv and detection_cache:
                enforce_nonempty_forward(
                    (f"{base}_forward{ext}" if backward_enabled else raw_csv),
                    detection_cache,
                )
            forward_csv = f"{base}_forward{ext}" if backward_enabled else raw_csv
            forward_processed = self._postprocess_csv(forward_csv)

            if cb.should_stop():
                return self._stopped_result()

            if backward_enabled:
                cb.stage_changed("backward_postprocess")
                backward_processed = self._postprocess_csv(f"{base}_backward{ext}")
                cb.stage_changed("merge")
                final_df = self._merge(forward_processed, backward_processed)
                final_csv = f"{base}_final{ext}"
            else:
                final_df = self._interpolate_and_scale(forward_processed)
                final_csv = f"{base}_forward_processed{ext}"

            if final_df is None or cb.should_stop():
                return self._stopped_result()
            _save_trajectories_to_csv(final_df, final_csv)

            cb.stage_changed("rich_export")
            rich_path = self._export_rich(final_csv)

            if should_run_interpolated_postpass(self.config) and not cb.should_stop():
                cb.stage_changed("interpolated_crops")
                self._run_interp_crops(final_csv)
                rich_path = self._relink_export_rich(final_csv) or rich_path

            result = SessionResult(
                success=True, final_csv_path=final_csv, rich_export_path=rich_path,
                media_paths=[], dataset_result=None, summary_lines=[], error=None,
            )
            result.summary_lines = build_session_summary_lines(self.config, result)
            cb.stage_changed("done")
            return result
        except TrackingSessionError as e:
            return SessionResult(False, None, None, [], None, [], str(e))
```

Add these helper methods:
- `_stopped_result()` → `return SessionResult(False, None, None, [], None, [], None)`.
- `_postprocess_csv(csv_path)`: reuse `PostProcessWorker`'s clean-flag param mutation verbatim (`postprocess_worker.py:36-49`): if `not self.config.get("enable_postprocessing")`, build `effective_params = dict(self.params)` with `MIN_TRAJECTORY_LENGTH=1`, `MAX_VELOCITY_BREAK=float("inf")`, `MAX_OCCLUSION_GAP=0`, `MAX_VELOCITY_ZSCORE=0.0`; else `effective_params = self.params`. Then `processed, _ = process_trajectories_from_csv(csv_path, effective_params); return processed`.
- `_interpolate_and_scale(df)`: mirror `_handle_forward_tracking_done:2454-2482`. `interp_method = str(self.config.get("interpolation_method", "none")).lower()`; if `interp_method != "none"`: `max_gap = max(1, round(float(self.config["interpolation_max_gap_seconds"]) * float(self.params["FPS"])))`, then `df = interpolate_trajectories(df, method=interp_method, max_gap=max_gap, heading_flip_max_burst=int(self.config["heading_flip_max_burst"]), directed_heading_posthoc=bool(self.params.get("DIRECTED_ORIENT_POSTHOC_CONSISTENCY", False)))`. Then scale: `return rescale_coordinates(df, resize_factor=float(self.params.get("RESIZE_FACTOR", 1.0)))` (identical to `_scale_trajectories_to_original_space`: divide X/Y and Width/Height by resize_factor).
- `_merge(forward, backward)`: read `total_frames` from `cv2.VideoCapture(self.video_path)` frame count (mirror `merge_and_save_trajectories:884-888`), then `return merge_trajectories(forward, backward, total_frames=total_frames, params=self.params, resize_factor=float(self.params.get("RESIZE_FACTOR", 1.0)), interp_method=str(self.config.get("interpolation_method", "none")).lower(), max_gap=max(1, round(float(self.config["interpolation_max_gap_seconds"]) * float(self.params["FPS"]))), tag_cache_path=resolve_current_tag_cache_path(self.params, self.paths.get("detection_cache_path")), heading_flip_max_burst=int(self.config["heading_flip_max_burst"]), directed_heading_posthoc=bool(self.params.get("DIRECTED_ORIENT_POSTHOC_CONSISTENCY", False)), enable_profiling=bool(self.params.get("ENABLE_PROFILING", False)), progress=self.callbacks.progress, should_stop=self.callbacks.should_stop)`.
- `_export_rich(final_csv)`: `return export_rich_csv(final_csv, self.pose_state, params=self.params, min_valid_conf=float(self.params.get("POSE_MIN_KPT_CONF_VALID", 0.2)), ignore_keypoints=self.config.get("pose_ignore_keypoints"))`.
- `_relink_export_rich(final_csv)`: `return relink_and_export_rich_csv(final_csv, self.pose_state, params=self.params, min_valid_conf=float(self.params.get("POSE_MIN_KPT_CONF_VALID", 0.2)), ignore_keypoints=self.config.get("pose_ignore_keypoints"))`.
- `_run_interp_crops(final_csv)`: gate on the Slice-1 policy predicate — `from hydra_suite.core.tracking import session_policy; if not session_policy.should_run_interpolated_postpass(self.config): return`. This matches the GUI's `_should_run_interpolated_postpass` gate exactly (checkbox `individual_interpolate_occlusions` AND pipeline AND one of canonical/pose-export/media), not just the raw checkbox. Then call `payload = run_interpolated_crops(final_csv, self.video_path, self.pose_state.detection_cache_path, self.params, enable_profiling=bool(self.params.get("ENABLE_PROFILING", False)), progress=self.callbacks.progress, should_stop=self.callbacks.should_stop)`, then store the payload's results back into `self.pose_state` (mirror `_store_interpolated_*` in `tracking.py:979-1037` and `_on_interpolated_crops_finished:1259-1270`): set `self.pose_state.interpolated_pose_csv_path = payload.get("pose_csv_path")` (or build a DataFrame from `payload.get("pose_rows")` into `interpolated_pose_df` when there is no CSV), and the analogous `tag`/`cnn`/`headtail` fields, so `_relink_export_rich` sees them.

Add `import os` and `import cv2` and the imports listed under **Interfaces** to `session.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n hydra-mps python -m pytest tests/test_session_core_run.py tests/test_session_core_scaffold.py -q --ignore=tests/test_identity_postprocess.py`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
make format
git add src/hydra_suite/core/tracking/session.py tests/test_session_core_run.py
git commit -m "feat(session): wire TrackingSessionCore analysis chain (merge->rich export)"
```

---

### Task 9: GUI cutover — `SessionWorker` + orchestrator delegation

Run the whole analysis chain on ONE `BaseWorker` (`SessionWorker`) instead of the current per-stage QThread chain. `on_tracking_finished` (after the preview/stop/failed guards and `_collect_worker_props_path()`) delegates to a single session run instead of starting `PostProcessWorker`. Callbacks map to the existing UI methods; `stage_changed` drives progress-label text; `warning` → `QMessageBox` on the GUI thread; a `TrackingSessionError` inside the worker surfaces via the worker's `error` signal → `QMessageBox.critical`. After the worker finishes, the existing media/dataset/finalize path stays GUI-side unchanged (Slice 3 moves media/dataset into the service).

**Files:**
- Create: `src/hydra_suite/trackerkit/gui/workers/session_worker.py`
- Modify: `src/hydra_suite/trackerkit/gui/orchestrators/tracking.py`
- Test: `tests/test_session_worker.py`

**Interfaces:**
- Consumes: `TrackingSessionCore`, `SessionCallbacks`, `SessionResult` (Task 1/8), `build_config_dict` (Slice 1), `BaseWorker`.
- `SessionWorker(BaseWorker)`: ctor `(core, forward_trajectories, backward_trajectories=None)`; signals `progress_signal = Signal(int, str)`, `status_signal = Signal(str)`, `warning_signal = Signal(str, str)`, `stage_signal = Signal(str)`, `finished_signal = Signal(object)` (emits `SessionResult`), `error_signal = Signal(str)`. Its `execute()` builds a `SessionCallbacks` whose members emit the signals and whose `should_stop` reads `self._should_stop`, assigns it to `core.callbacks`, runs `core.run_post_tracking(...)`, and emits the result (or `error_signal` on exception).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_session_worker.py  (app-layer worker; Qt imports allowed here)
import pandas as pd

from hydra_suite.core.tracking.session import TrackingSessionCore
from hydra_suite.trackerkit.gui.workers.session_worker import SessionWorker


def test_session_worker_wires_callbacks_and_returns_result(tmp_path):
    raw = tmp_path / "clip.csv"
    pd.DataFrame(
        {"TrajectoryID": [0, 0], "X": [1.0, 2.0], "Y": [3.0, 3.0],
         "Theta": [0.0, 0.0], "FrameID": [0, 1], "State": ["tracked", "tracked"]}
    ).to_csv(str(raw), index=False)
    core = TrackingSessionCore(
        video_path=str(tmp_path / "c.mp4"),
        config={"enable_postprocessing": False, "interpolation_method": "none",
                "interpolation_max_gap_seconds": 1.0, "heading_flip_max_burst": 5,
                "enable_backward_tracking": False, "individual_interpolate_occlusions": False},
        params={"FPS": 30.0, "RESIZE_FACTOR": 1.0, "MIN_TRAJECTORY_LENGTH": 1},
        paths={"raw_csv_path": str(raw), "detection_cache_path": str(tmp_path / "d.npz")},
    )
    worker = SessionWorker(core, pd.read_csv(str(raw)))
    seen = []
    worker.finished_signal.connect(lambda r: seen.append(r))
    worker.execute()  # BaseWorker.execute is plain-callable; drive synchronously
    assert seen and seen[0].success is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n hydra-mps python -m pytest tests/test_session_worker.py -q --ignore=tests/test_identity_postprocess.py`
Expected: FAIL with `ModuleNotFoundError: No module named '...session_worker'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/hydra_suite/trackerkit/gui/workers/session_worker.py
"""SessionWorker — runs the Qt-free TrackingSessionCore on one BaseWorker."""

import logging

from PySide6.QtCore import Signal

from hydra_suite.core.tracking.session import SessionCallbacks
from hydra_suite.widgets.workers import BaseWorker

logger = logging.getLogger(__name__)


class SessionWorker(BaseWorker):
    progress_signal = Signal(int, str)
    status_signal = Signal(str)
    warning_signal = Signal(str, str)
    stage_signal = Signal(str)
    finished_signal = Signal(object)
    error_signal = Signal(str)

    def __init__(self, core, forward_trajectories, backward_trajectories=None):
        super().__init__()
        self.core = core
        self.forward_trajectories = forward_trajectories
        self.backward_trajectories = backward_trajectories
        self._stop_requested = False

    def stop(self):
        self._stop_requested = True

    def _should_stop(self) -> bool:
        return bool(self._stop_requested or self.isInterruptionRequested())

    def execute(self):
        self.core.callbacks = SessionCallbacks(
            progress=lambda v, m: self.progress_signal.emit(v, m),
            status=lambda m: self.status_signal.emit(m),
            warning=lambda t, m: self.warning_signal.emit(t, m),
            stage_changed=lambda s: self.stage_signal.emit(s),
            should_stop=self._should_stop,
        )
        try:
            result = self.core.run_post_tracking(
                self.forward_trajectories, self.backward_trajectories
            )
        except Exception as e:  # TrackingSessionError or unexpected
            logger.exception("Session worker failed")
            self.error_signal.emit(str(e))
            return
        self.finished_signal.emit(result)
```

Then rewire `tracking.py` (re-derive line numbers with `grep -n "    def " …` first — methods are interleaved):
- `on_tracking_finished` (`:2619`): keep the stale-signal / stop / preview / `_collect_worker_props_path()` / `finished_normally` / fps-accounting guards verbatim, then replace the `self._start_postprocess_worker(...)` call with a new `self._start_session_worker()` delegation.
- Add `_start_session_worker()`: construct a `PoseSourceState` from the `self._mw.current_*` cache fields, build `config = build_config_dict(...)`, `params = self._mw.get_parameters_dict()`, `paths = {"raw_csv_path": self._panels.setup.csv_line.text(), "detection_cache_path": self._mw.current_detection_cache_path}`, construct `TrackingSessionCore(video_path=self._panels.setup.file_line.text(), config=config, params=params, paths=paths)`, set its `pose_state`, then start a `SessionWorker(core, forward_trajectories=<raw trajs>, backward_trajectories=None)`. Connect `progress_signal→on_merge_progress`, `warning_signal→(lambda t, m: QMessageBox.warning(self._mw, t, m))`, `stage_signal→self._on_session_stage`, `error_signal→(lambda msg: QMessageBox.critical(self._mw, "Tracking Session Error", msg))`, `finished_signal→self._on_session_finished`.
- Add `_on_session_stage(name)`: set `self._mw.progress_label` text per stage name.
- Add `_on_session_finished(result)`: store `result.final_csv_path` into `self._mw._session_final_csv_path`, then run the existing GUI-side continuation verbatim from `_finish_tracking_session:3483-3510` — set `_pending_video_csv_path`/`_pending_video_generation` from the video-output widgets, run dataset gen if enabled, then `_start_pending_final_media_export(...)` / `_run_pending_video_generation_or_finalize()`.
- Delete the methods fully superseded by the service: `merge_and_save_trajectories`, `on_merge_finished`, `on_merge_error`, `_handle_forward_tracking_done`, `_handle_backward_tracking_done`, `_start_postprocess_worker`, `on_postprocess_finished`, `on_postprocess_error`, `_check_pose_export_sources`, `_merge_pose_sources_into_df`, `_apply_pose_quality_postprocessing`, `_resolve_current_tag_cache_path`, `_apply_identity_postprocessing_to_df`, `_build_rich_export_dataframe`, `_export_rich_csv`, `_relink_and_export_rich_csv`, `_write_rich_export_csv`, `_drop_empty_rich_export_columns`, `_remove_legacy_rich_exports`, `_rich_export_path`, `_log_rich_export_summary`, `_count_augmented_pose_rows`, `_count_interpolated_cnn_rows`, `_generate_interpolated_individual_crops`, `_on_interpolated_crops_finished`, `_store_interpolated_pose_result`, `_store_interpolated_tag_result`, `_store_interpolated_cnn_result`, `_store_interpolated_headtail_result`, `_log_interpolated_postpass_summary`. Retain `on_merge_progress` (reused as the progress-callback target), `on_postprocess_progress`, and `_scale_trajectories_to_original_space` / `save_trajectories_to_csv` (still used by the media path). Remove the now-orphaned `PostProcessWorker` import if unused.

- [ ] **Step 4: Run tests + GUI import smoke**

Run: `conda run -n hydra-mps python -m pytest tests/test_session_worker.py -q --ignore=tests/test_identity_postprocess.py && conda run -n hydra-mps python -c "import hydra_suite.trackerkit.gui.orchestrators.tracking"`
Expected: PASS + no ImportError.

- [ ] **Step 5: Commit**

```bash
make format
git add src/hydra_suite/trackerkit/gui/workers/session_worker.py src/hydra_suite/trackerkit/gui/orchestrators/tracking.py tests/test_session_worker.py
git commit -m "refactor(trackerkit): delegate post-tracking analysis chain to TrackingSessionCore via SessionWorker"
```

---

### Task 10: Qt-free guard test + full delta gate

**Files:**
- Test: `tests/test_core_qtfree_slice2.py`

**Interfaces:** none (guard only).

- [ ] **Step 1: Write the guard test**

```python
# tests/test_core_qtfree_slice2.py
import importlib
import pathlib
import subprocess

CORE = pathlib.Path(__file__).resolve().parents[1] / "src" / "hydra_suite" / "core"


def test_core_has_no_qt_references():
    out = subprocess.run(
        ["grep", "-rnE", "PySide6|QtCore|QThread|Signal|Slot|QMutex", str(CORE)],
        capture_output=True, text=True,
    )
    # grep exit 1 == no matches (desired). Any stdout is a leak.
    assert out.stdout.strip() == "", f"Qt references leaked into core/:\n{out.stdout}"


def test_new_core_modules_import_without_qt():
    for mod in (
        "hydra_suite.core.tracking.session",
        "hydra_suite.core.tracking.errors",
        "hydra_suite.core.post.merge",
        "hydra_suite.core.post.pose_merge",
        "hydra_suite.core.post.rich_export",
        "hydra_suite.core.post.interpolated_crops",
        "hydra_suite.core.individual.postprocess_df",
    ):
        importlib.import_module(mod)
```

- [ ] **Step 2: Run the guard test**

Run: `conda run -n hydra-mps python -m pytest tests/test_core_qtfree_slice2.py -q --ignore=tests/test_identity_postprocess.py`
Expected: PASS. If `test_core_has_no_qt_references` FAILS, the printed path shows the leak (a stray `Signal`/`PySide6` import from an over-eager copy) — fix that module before proceeding.

- [ ] **Step 3: Full new-suite run**

Run: `conda run -n hydra-mps python -m pytest tests/test_session_core_scaffold.py tests/test_core_post_merge.py tests/test_core_pose_merge.py tests/test_core_identity_postprocess_df.py tests/test_core_rich_export.py tests/test_core_interpolated_crops.py tests/test_nonempty_guard.py tests/test_session_core_run.py tests/test_session_worker.py tests/test_core_qtfree_slice2.py -q --ignore=tests/test_identity_postprocess.py`
Expected: all PASS.

- [ ] **Step 4: Full delta gate**

Run: `conda run -n hydra-mps python -m pytest -q --ignore=tests/test_identity_postprocess.py`
Expected: no NEW failures vs the base suite (~24 pre-existing failures — delta gate, not zero-fail). Then `grep -rnE "PySide6|QtCore|QThread|Signal|Slot|QMutex" src/hydra_suite/core/` prints nothing.

- [ ] **Step 5: Commit**

```bash
make format
git add tests/test_core_qtfree_slice2.py
git commit -m "test(core): assert core/ stays Qt-free after Slice 2 analysis-chain extraction"
```

---

### Task 11: Equivalence gate (MANDATORY — MPS + CUDA, all 7 clips)

Prove the extraction is byte-identical. Run the harness with the same baseline BEFORE this slice's branch and AFTER it — per-slice runs provide attribution. The baseline worktree is the commit immediately preceding Task 1 (this slice's effect in isolation).

**Files:** none (verification only).

- [ ] **Step 1: MPS — fixtures + baseline worktree + matrix**

```bash
conda activate hydra-mps
bash tools/equivalence/fixtures/fetch_fixtures.sh          # once per machine
git fetch origin --tags
git worktree add --detach .worktrees/equiv-slice2-base <commit-before-this-slice>
REPO=$PWD WT=$PWD \
  MAIN_SRC=$PWD/.worktrees/equiv-slice2-base/src WT_SRC=$PWD/src \
  OUT=/tmp/equiv_slice2 RUNTIME=mps \
  bash tools/equivalence/run_matrix.sh
```

- [ ] **Step 2: MPS — verify acceptance for ALL 7 clips**

Clips: `emi_obb_identity`, `ant_pose_headtail`, `ant_obb_sleap`, `ant_obb_sequential`, `worm_bgsub`, `ant_cnn_identity`, `fly_obb`.
Expected per clip: EQUIVALENCE (base vs new_a) at/near its DETERMINISM floor — positions p99 ≈ 0, θ max ≈ 0, identical row counts, 0 unmatched — on BOTH `_forward.csv` and `_tracking_final.csv`. Known allowed noise: bistable head/tail π-flips on head/tail clips only.
**Before trusting any EQUIVALENT: confirm `conda` was active (pose/SLEAP clips) and `wc -l` on the emitted CSVs is > 1.** Empty CSVs falsely compare EQUIVALENT.

- [ ] **Step 3: CUDA — run the same matrix on mehek**

```bash
ssh rutalab@mehek.taild08eb9.ts.net
cd ~/hydra-suite && git fetch origin --tags && git checkout <this-slice-sha>
source ~/mambaforge/etc/profile.d/conda.sh && conda activate hydra-cuda
bash tools/equivalence/fixtures/fetch_fixtures.sh          # once
git worktree add --detach .worktrees/equiv-slice2-base <commit-before-this-slice>
REPO=$PWD WT=$PWD MAIN_SRC=$PWD/.worktrees/equiv-slice2-base/src WT_SRC=$PWD/src \
  OUT=/tmp/equiv_slice2 RUNTIME=cuda nohup bash tools/equivalence/run_matrix.sh > /tmp/equiv_cuda_slice2.log 2>&1 &
```

- [ ] **Step 4: CUDA — verify acceptance for ALL 7 clips** (same criteria as Step 2; pose/SLEAP clips REQUIRE the `sleap` conda env on the box + conda on PATH).

- [ ] **Step 5: Cleanup both boxes**

```bash
git worktree remove --force .worktrees/equiv-slice2-base && git worktree prune
```

- [ ] **Step 6: Record the gate result**

Only if BOTH platforms are byte-identical across all 7 clips (modulo the head/tail π-flip noise floor) is Slice 2 complete. If any clip diverges, treat it as a regression: attribution is exact (baseline = pre-slice commit), so bisect to the offending task's extraction, fix the widget→config or emit→callback mapping via `superpowers:systematic-debugging`, and re-run before proceeding to Slice 3.

---

## Self-Review

- **Spec coverage:** merge+post-process chaining → Tasks 2, 8, 9; `MergeWorker` logic → Task 2; pose merge + quality post-pass → Task 3; identity post-pass → Task 4; rich export → Task 5; interpolated crops → Tasks 6, 8; `_enforce_nonempty_forward` into service → Task 7; `SessionCallbacks`/`SessionResult`/`TrackingSessionCore` interfaces → Tasks 1, 8; `QMessageBox.information/.warning`→`callbacks.warning`, `QMessageBox.critical`→`TrackingSessionError` → Tasks 7–9; cancellation between stages + in crop loop → Tasks 2, 6, 8; GUI delegation via `build_config_dict()` + callbacks → Task 9; Qt-free guard → Task 10; equivalence gate → Task 11. Media export + dataset generation are explicitly Slice 3 (`media_paths=[]`, `dataset_result=None`).
- **Placeholder scan:** no `...`/TBD/"similar to above"/"add error handling" — every code step carries real code; the extraction tasks enumerate the exact `self._mw`/`self._panels`→arg substitution table and cite verbatim line ranges to copy. The rich-export suffix constants are the verified real values (`_with_individual` / `_with_pose`), not guesses.
- **Type consistency:** `PoseSourceState` field names are identical across Tasks 3, 5, 8. `check_pose_export_sources` returns the same 7-tuple order the GUI used, consumed by `build_rich_export_dataframe`. `run_interpolated_crops` returns the same payload-dict keys the worker emitted. `SessionResult`/`SessionCallbacks` signatures match the spec verbatim. `merge_trajectories`, `export_rich_csv`, `relink_and_export_rich_csv`, `run_interpolated_crops` signatures are used consistently in Task 8.
- **Out of scope (unchanged this slice):** the CLI (`headless_tracking.py`, `cli.py`) still drives its own QThread/QEventLoop workers — Task 7 only re-homes the guard; Slice 4 does the CLI cutover. Media export + dataset generation are Slice 3.

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-31-headless-qt-free-slice2-analysis-chain.md`. Two execution options:

1. **Subagent-Driven (recommended)** — dispatch a fresh subagent per task, review between tasks, fast iteration (REQUIRED SUB-SKILL: superpowers:subagent-driven-development).
2. **Inline Execution** — execute tasks in this session with checkpoints (REQUIRED SUB-SKILL: superpowers:executing-plans).
