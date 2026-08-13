# TrackerKit User/Debug Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a single global **Debug Mode** toggle to TrackerKit that cleanly separates a User-mode deliverable (one clean `<video>_tracks.csv` + optional annotated video) from Debug-mode output (byte-identical to today's full output), replacing ~25 scattered diagnostic checkboxes with one control and consolidating the duplicated terminal CSV writers.

**Architecture:** One `debug_mode` config field resolves to a boolean `DEBUG_MODE` engine param inside the shared Qt-free `build_engine_params`. That param drives (a) a mode-aware terminal writer that branches between today's `_final.csv` + `_with_individual.csv` (debug) and a single clean `<video>_tracks.csv` (user), and (b) intermediate-file cleanup. `save_confidence_metrics` is retired as a toggle — the worker now *always* emits the three confidence columns, and the User/Debug split for those columns lives only in the final writer's column selection. The GUI loses the diagnostic checkboxes in favor of one checkable toolbar button.

**Tech Stack:** Python 3.13, pandas ≥3, numpy, PyQt, pytest. Core (`core/post`, `core/tracking`) must not import app layers.

## Global Constraints

- **Debug-mode output stays byte-identical to `main` @`d6fc4f4c`.** The equivalence harness (`tools/equivalence/run_matrix.sh`) must stay at the determinism floor (positions p99 ≈ 0, θ within head/tail π-flip noise floor, identical row counts, 0 unmatched) on **MPS (this box) and CUDA (mehek)** for both `_tracking_forward.csv` and `_tracking_final.csv`.
- **All 7 gate configs already carry `save_confidence_metrics: true`** (verified), so making confidence-column emission unconditional produces byte-identical gate output. This is the key that makes retiring the toggle gate-safe.
- **Do not change the tracking math.** The worker's Kalman/assignment/detection geometry is untouched; only the `if save_confidence:` gating is removed (always take the True branch).
- **Absent `debug_mode` key ⇒ treat as Debug/legacy** (`DEBUG_MODE = config.get("debug_mode", True)`), and honor any stored granular flags. This preserves byte-identity for the gate configs (which have no `debug_mode` key) and any old user configs. The GUI always writes an explicit `debug_mode` value.
- **Identity columns are referenced via `columns.py` constants** (`C.FINAL_LABEL`, `C.FINAL_CONFIDENCE`, `C.FINAL_SOURCE`, `C.FINAL_SMOOTHED_LABEL`), never string literals, so the clean writer can't drift from the emitted schema.
- **Commit as the configured git user** (no Co-Authored-By trailer). **Do work in a git worktree branched from local HEAD.**
- Run tests with `PYTHONPATH=<worktree>/src` when in a worktree. The base suite has ~24 pre-existing failures — use a delta gate (compare against baseline), not an absolute-green gate.

## Retired / decided facts (authoritative)

- **`save_confidence_metrics` is retired as a user knob** (user decision 2026-08-11): the worker always computes and emits `DetectionConfidence`, `AssignmentConfidence`, `PositionUncertainty`. The User-mode writer surfaces only `detection_confidence` and drops the other two. This resolves the spec's "detection_confidence always" vs. "worker.py untouched" tension.
- **Debug-derived flags** (present→derive from `debug_mode`; absent→stored value):

  | Granular flag | User (debug off) | Debug on | Where consumed |
  |---|---|---|---|
  | `ENABLE_PROFILING` | false | true | `build_engine_params` → param |
  | `EXPORT_CONFIDENCE_DENSITY_VIDEO` | false | true | `build_engine_params` → param |
  | `cleanup_temp_files` (forced) | true (cleanup on) | false | core session + GUI session orchestrator |
  | `debug_logging` | false | true | GUI logger level |
  | `SHOW_KALMAN_UNCERTAINTY` | false | true | GUI display overlay only |
  | `SHOW_FG` | false | true | GUI display overlay only |
  | `SHOW_BG` | false | true | GUI display overlay only |
  | `SHOW_YOLO_OBB` | false | true | GUI display overlay only |

  `save_confidence_metrics` is **not** in this table (retired, always-on). `enable_confidence_density_map` stays driven by its own logic (unchanged).

## User-mode clean schema — `<video>_tracks.csv`

`lower_snake_case`; identity columns only if identity/tags ran; pose columns only if pose ran.

| Clean column | Source (raw column) | Notes |
|---|---|---|
| `id` | `TrajectoryID` | stable per-animal id |
| `frame` | `FrameID` | integer |
| `time_s` | `frame / fps` | new; NaN if fps unavailable |
| `x` | `X` | pixels |
| `y` | `Y` | pixels |
| `heading_deg` | `degrees(Theta) mod 360` | radian→degree, `[0,360)` |
| `state` | `State` | active/occluded/interpolated/lost |
| `detection_confidence` | `DetectionConfidence` | always |
| `identity` | `C.FINAL_LABEL` (fall back to `C.FINAL_SMOOTHED_LABEL` only when Final is empty) | only if identity ran |
| `identity_confidence` | `C.FINAL_CONFIDENCE` | only if identity ran |
| `identity_source` | `C.FINAL_SOURCE` (realtime/offline/tag) | only if identity ran |
| `<kpt>_x/_y/_conf` | `PoseKpt_<kpt>_X/_Y/_Conf` | only if pose ran; one triple per keypoint |

---

### Task 1: Retire `save_confidence_metrics` — make confidence emission unconditional

**Files:**
- Modify: `src/hydra_suite/core/tracking/worker.py` (`:2906`, `:3305`, `:3359`, `:3507` — remove the `SAVE_CONFIDENCE_METRICS` gating)
- Modify: `src/hydra_suite/trackerkit/headless_tracking.py:33-74` (`build_tracking_csv_header` — drop the param, always full header) and `:116-117` (call site)
- Modify: `src/hydra_suite/trackerkit/gui/orchestrators/tracking.py:1410-1412` (stop reading the checkbox / passing `save_confidence`)
- Test: `tests/test_headless_tracking_header.py` (new)

**Interfaces:**
- Produces: `build_tracking_csv_header(identity_method: str = "none_disabled") -> list[str]` (the `save_confidence_metrics` parameter is **removed**; header always includes `DetectionConfidence, AssignmentConfidence, PositionUncertainty`).
- Consumes: nothing from earlier tasks.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_headless_tracking_header.py
from hydra_suite.trackerkit.headless_tracking import build_tracking_csv_header


def test_header_always_has_confidence_columns():
    header = build_tracking_csv_header()
    for col in ("DetectionConfidence", "AssignmentConfidence", "PositionUncertainty"):
        assert col in header


def test_header_apriltags_appends_tag_columns():
    header = build_tracking_csv_header(identity_method="apriltags")
    assert header[-4:] == [
        "DetectedTagID",
        "DetectedTagLabel",
        "DetectedTagConf",
        "DetectedTagHamming",
    ]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_headless_tracking_header.py -v`
Expected: FAIL — `build_tracking_csv_header()` currently requires the positional `save_confidence_metrics` arg (TypeError).

- [ ] **Step 3: Make the header unconditional**

In `src/hydra_suite/trackerkit/headless_tracking.py`, replace the function signature/body (`:33-74`) so the confidence branch is the only branch:

```python
def build_tracking_csv_header(identity_method: str = "none_disabled") -> list[str]:
    """Build the raw tracking CSV header. Confidence columns are always emitted."""
    base_cols = [
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
    ]
    header = list(base_cols) + C.identity_realtime_columns()
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
```

Update the call site at `headless_tracking.py:116-117` from `build_tracking_csv_header(session.save_confidence_metrics, identity_method=...)` to:

```python
            header=build_tracking_csv_header(identity_method=session.identity_method),
```

- [ ] **Step 4: Unconditionalize the worker emission**

In `src/hydra_suite/core/tracking/worker.py`, force the confidence path on. At `:2906` replace:

```python
                save_confidence = params.get("SAVE_CONFIDENCE_METRICS", True)
                if save_confidence:
```
with:
```python
                # Confidence metrics are always computed (save_confidence_metrics retired).
                if True:
```
(Leave the `if True:` block body as-is; the `else:` at `:2918-2920` becomes dead but harmless — delete it if trivially safe.) The two matched-row enqueues gate on the same `save_confidence` name at `:3305` and `:3359` — change each `if save_confidence:` to `if True:` (their `else` branches, which append rows *without* the three columns, become dead; delete them so every enqueued row has the three columns). At `:3507` replace `_save_conf_zero = params.get("SAVE_CONFIDENCE_METRICS", True)` with `_save_conf_zero = True` and drop the now-constant `if _save_conf_zero:` guard at `:3524` (always `row_data.extend([0.0, 0.0, pos_uncertainty])`).

**Verification note:** after editing, `grep -n "save_confidence\|SAVE_CONFIDENCE_METRICS" src/hydra_suite/core/tracking/worker.py` must return nothing.

- [ ] **Step 5: Drop the GUI header-arg read**

In `src/hydra_suite/trackerkit/gui/orchestrators/tracking.py`, delete `:1410` (`save_confidence = self._panels.setup.check_save_confidence.isChecked()`) and change the `build_tracking_csv_header(save_confidence, identity_method=...)` call at `:1411-1412` to `build_tracking_csv_header(identity_method=self._mw._selected_identity_method())`. (The `check_save_confidence` widget still exists; it is removed in Task 8.)

- [ ] **Step 6: Run header + worker-helper tests**

Run: `python -m pytest tests/test_headless_tracking_header.py tests/test_tracking_worker_helpers.py -v`
Expected: PASS (header tests green; worker-helper tests unaffected).

- [ ] **Step 7: Commit**

```bash
git add src/hydra_suite/core/tracking/worker.py \
        src/hydra_suite/trackerkit/headless_tracking.py \
        src/hydra_suite/trackerkit/gui/orchestrators/tracking.py \
        tests/test_headless_tracking_header.py
git commit -m "refactor(trackerkit): retire save_confidence_metrics toggle; always emit confidence columns"
```

---

### Task 2: `debug_mode` → `DEBUG_MODE` param in `build_engine_params`

**Files:**
- Modify: `src/hydra_suite/trackerkit/engine_params.py` (add `DEBUG_MODE` + derive `ENABLE_PROFILING`/`EXPORT_CONFIDENCE_DENSITY_VIDEO`; existing lines `:888`, `:1190`, `:1296`)
- Modify: `tests/test_get_parameters_dict_characterization.py` goldens (add `DEBUG_MODE`)
- Test: `tests/test_debug_mode_params.py` (new)

**Interfaces:**
- Produces: engine params now include `DEBUG_MODE: bool`. Rule: `debug_present = "debug_mode" in cfg`; `DEBUG_MODE = bool(cfg.get("debug_mode", True))`; when `debug_present`, `ENABLE_PROFILING` and `EXPORT_CONFIDENCE_DENSITY_VIDEO` equal `DEBUG_MODE`; otherwise they keep their stored values.
- Consumes: nothing from earlier tasks.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_debug_mode_params.py
from hydra_suite.trackerkit.engine_params import build_engine_params
from hydra_suite.runtime.context import RuntimeContext  # adjust import to actual RuntimeContext location


def _rt():
    # Minimal CPU runtime context sufficient for param building.
    return RuntimeContext.for_tier("cpu")  # adjust to the real constructor used in tests


def test_debug_mode_absent_defaults_to_debug_and_keeps_stored_flags():
    cfg = {"enable_profiling": False, "export_confidence_density_video": False}
    params = build_engine_params(cfg, runtime=_rt())
    assert params["DEBUG_MODE"] is True
    assert params["ENABLE_PROFILING"] is False
    assert params["EXPORT_CONFIDENCE_DENSITY_VIDEO"] is False


def test_debug_mode_true_derives_flags_on():
    params = build_engine_params({"debug_mode": True}, runtime=_rt())
    assert params["DEBUG_MODE"] is True
    assert params["ENABLE_PROFILING"] is True
    assert params["EXPORT_CONFIDENCE_DENSITY_VIDEO"] is True


def test_debug_mode_false_derives_flags_off():
    params = build_engine_params(
        {"debug_mode": False, "enable_profiling": True}, runtime=_rt()
    )
    assert params["DEBUG_MODE"] is False
    assert params["ENABLE_PROFILING"] is False
    assert params["EXPORT_CONFIDENCE_DENSITY_VIDEO"] is False
```

> Before running, open `tests/test_get_parameters_dict_characterization.py` and copy its exact `RuntimeContext` construction into `_rt()` so the fixture matches the real API.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_debug_mode_params.py -v`
Expected: FAIL — `params["DEBUG_MODE"]` KeyError.

- [ ] **Step 3: Implement the derivation**

In `src/hydra_suite/trackerkit/engine_params.py`, inside `build_engine_params` (after `cfg` is available, near the top of the params-dict assembly), compute the debug resolution once:

```python
    _debug_present = "debug_mode" in cfg
    _debug_mode = bool(_cfg_get(cfg, "debug_mode", default=True))
```

Add `"DEBUG_MODE": _debug_mode,` to the params dict. Change the `ENABLE_PROFILING` entry (`:1296`) and `EXPORT_CONFIDENCE_DENSITY_VIDEO` entry (`:1190`) to derive when `debug_mode` is present:

```python
        "ENABLE_PROFILING": (
            _debug_mode
            if _debug_present
            else bool(_cfg_get(cfg, "enable_profiling", default=False))
        ),
        "EXPORT_CONFIDENCE_DENSITY_VIDEO": (
            _debug_mode
            if _debug_present
            else bool(_cfg_get(cfg, "export_confidence_density_video", default=False))
        ),
```

Leave `SAVE_CONFIDENCE_METRICS` (`:888`) emitting `True` (default preserved; the worker no longer reads it, but keep it for the raw header path in `session.save_confidence_metrics`). Leave `ENABLE_CONFIDENCE_DENSITY_MAP` (`:1154`) unchanged.

- [ ] **Step 4: Run the new test to verify it passes**

Run: `python -m pytest tests/test_debug_mode_params.py -v`
Expected: PASS.

- [ ] **Step 5: Update the characterization golden**

Run: `python -m pytest tests/test_get_parameters_dict_characterization.py -v` and observe the diff — it should differ only by a new `DEBUG_MODE: true` key for both `fly_obb` and `ant_cnn_identity` gate configs (they have no `debug_mode` key → `True`). Add `"DEBUG_MODE": true` to each committed golden (do NOT regenerate blindly — confirm that is the *only* delta).

Run: `python -m pytest tests/test_get_parameters_dict_characterization.py tests/test_gui_cli_param_equivalence.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/hydra_suite/trackerkit/engine_params.py \
        tests/test_debug_mode_params.py tests/test_get_parameters_dict_characterization.py
git commit -m "feat(trackerkit): derive DEBUG_MODE engine param from debug_mode config"
```

---

### Task 3: Pure clean-schema projection `project_user_tracks`

**Files:**
- Create: `src/hydra_suite/core/post/trajectory_writer.py`
- Test: `tests/test_trajectory_writer_projection.py` (new)

**Interfaces:**
- Produces: `project_user_tracks(df: pd.DataFrame, *, fps: float | None) -> pd.DataFrame` — the clean User-mode DataFrame. Later tasks import it from `hydra_suite.core.post.trajectory_writer`.
- Consumes: identity constants from `hydra_suite.core.individual.identity.columns`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_trajectory_writer_projection.py
import math

import numpy as np
import pandas as pd

from hydra_suite.core.individual.identity import columns as C
from hydra_suite.core.post.trajectory_writer import project_user_tracks


def _base_df():
    return pd.DataFrame(
        {
            "TrackID": [0, 0],
            "Index": [0, 1],
            "TrajectoryID": [3, 3],
            "FrameID": [0, 10],
            "X": [1.4, 2.6],
            "Y": [5.0, 6.0],
            "Theta": [0.0, -math.pi / 2],  # 0 rad, -90 deg
            "State": ["active", "occluded"],
            "DetectionConfidence": [0.9, 0.8],
            "AssignmentConfidence": [0.5, 0.4],
            "PositionUncertainty": [1.1, 1.2],
        }
    )


def test_core_columns_and_conversions():
    out = project_user_tracks(_base_df(), fps=10.0)
    assert list(out.columns) == [
        "id",
        "frame",
        "time_s",
        "x",
        "y",
        "heading_deg",
        "state",
        "detection_confidence",
    ]
    assert out["id"].tolist() == [3, 3]
    assert out["time_s"].tolist() == [0.0, 1.0]  # frame/fps
    # -pi/2 rad -> 270 deg (normalized to [0,360))
    assert out["heading_deg"].round(3).tolist() == [0.0, 270.0]
    # tracer-only confidences dropped
    assert "AssignmentConfidence" not in out.columns
    assert "PositionUncertainty" not in out.columns


def test_identity_columns_appear_only_when_final_present():
    df = _base_df()
    df[C.FINAL_LABEL] = ["antA", ""]
    df[C.FINAL_SMOOTHED_LABEL] = ["antA", "antB"]
    df[C.FINAL_CONFIDENCE] = [0.7, 0.6]
    df[C.FINAL_SOURCE] = ["realtime", "offline"]
    out = project_user_tracks(df, fps=10.0)
    assert out["identity"].tolist() == ["antA", "antB"]  # empty Final falls back to Smoothed
    assert out["identity_confidence"].tolist() == [0.7, 0.6]
    assert out["identity_source"].tolist() == ["realtime", "offline"]


def test_pose_triples_appear_only_when_pose_present():
    df = _base_df()
    df["PoseKpt_head_X"] = [1.0, 2.0]
    df["PoseKpt_head_Y"] = [3.0, 4.0]
    df["PoseKpt_head_Conf"] = [0.9, 0.8]
    out = project_user_tracks(df, fps=10.0)
    assert out["head_x"].tolist() == [1.0, 2.0]
    assert out["head_y"].tolist() == [3.0, 4.0]
    assert out["head_conf"].tolist() == [0.9, 0.8]


def test_no_fps_yields_nan_time():
    out = project_user_tracks(_base_df(), fps=None)
    assert out["time_s"].isna().all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_trajectory_writer_projection.py -v`
Expected: FAIL — module `trajectory_writer` does not exist.

- [ ] **Step 3: Implement the projection**

```python
# src/hydra_suite/core/post/trajectory_writer.py
"""Mode-aware terminal trajectory writers (User-mode clean CSV + Debug base-final CSV)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from hydra_suite.core.individual.identity import columns as C

_POSE_PREFIX = "PoseKpt_"
_POSE_X_SUFFIX = "_X"


def _is_empty_label(series: pd.Series) -> pd.Series:
    """True where a label is missing or an empty/whitespace string."""
    s = series.astype("string")
    return s.isna() | (s.str.strip().str.len() == 0)


def project_user_tracks(df: pd.DataFrame, *, fps: float | None) -> pd.DataFrame:
    """Project the full trajectory DataFrame to the clean User-mode schema."""
    out = pd.DataFrame(index=df.index)
    out["id"] = df["TrajectoryID"]
    out["frame"] = pd.to_numeric(df["FrameID"], errors="coerce").round().astype("Int64")
    if fps and float(fps) > 0:
        out["time_s"] = out["frame"].astype("Float64") / float(fps)
    else:
        out["time_s"] = pd.array([np.nan] * len(df), dtype="Float64")
    out["x"] = df["X"]
    out["y"] = df["Y"]
    theta = pd.to_numeric(df["Theta"], errors="coerce")
    out["heading_deg"] = np.mod(np.degrees(theta), 360.0)
    out["state"] = df["State"]
    out["detection_confidence"] = df.get("DetectionConfidence")

    # Identity — only when the resolved-final label column is present.
    if C.FINAL_LABEL in df.columns:
        label = df[C.FINAL_LABEL].astype("string")
        if C.FINAL_SMOOTHED_LABEL in df.columns:
            label = label.mask(_is_empty_label(label), df[C.FINAL_SMOOTHED_LABEL].astype("string"))
        out["identity"] = label
        if C.FINAL_CONFIDENCE in df.columns:
            out["identity_confidence"] = df[C.FINAL_CONFIDENCE]
        if C.FINAL_SOURCE in df.columns:
            out["identity_source"] = df[C.FINAL_SOURCE]

    # Pose — one <kpt>_x/_y/_conf triple per PoseKpt_<name>_X column present.
    pose_x_cols = [
        c for c in df.columns if c.startswith(_POSE_PREFIX) and c.endswith(_POSE_X_SUFFIX)
    ]
    for xcol in pose_x_cols:
        name = xcol[len(_POSE_PREFIX) : -len(_POSE_X_SUFFIX)]
        out[f"{name}_x"] = df.get(f"{_POSE_PREFIX}{name}_X")
        out[f"{name}_y"] = df.get(f"{_POSE_PREFIX}{name}_Y")
        out[f"{name}_conf"] = df.get(f"{_POSE_PREFIX}{name}_Conf")

    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_trajectory_writer_projection.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/core/post/trajectory_writer.py tests/test_trajectory_writer_projection.py
git commit -m "feat(post): add clean User-mode trajectory projection"
```

---

### Task 4: Consolidate the base-final writer; delete the orphaned twin

**Files:**
- Modify: `src/hydra_suite/core/post/trajectory_writer.py` (add `write_base_final_csv`)
- Modify: `src/hydra_suite/core/tracking/session.py:97-127` (`_save_trajectories_to_csv` delegates)
- Modify: `src/hydra_suite/core/post/rich_export.py:396` (relink bare `to_csv` routes through the shared helper)
- Delete: `src/hydra_suite/core/post/media_export.py::save_trajectories_to_csv` (`:50-101`, orphaned twin — no production caller)
- Modify: `tests/test_media_export.py:39,47` (repoint at the consolidated writer or delete the twin-specific cases)
- Test: `tests/test_trajectory_writer_base_final.py` (new)

**Interfaces:**
- Produces: `write_base_final_csv(df: pd.DataFrame, output_path: str) -> bool` — writes the debug base-final CSV (round `X`/`Y`/`FrameID` → `Int64`, drop `TrackID`/`Index`, reorder `["TrajectoryID","X","Y","Theta","FrameID"]` first). Byte-identical to today's `_save_trajectories_to_csv`.
- Consumes: nothing new.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_trajectory_writer_base_final.py
import pandas as pd

from hydra_suite.core.post.trajectory_writer import write_base_final_csv


def test_base_final_rounds_reorders_and_drops(tmp_path):
    df = pd.DataFrame(
        {
            "TrackID": [7],
            "Index": [0],
            "TrajectoryID": [2],
            "X": [1.6],
            "Y": [3.4],
            "Theta": [0.5],
            "FrameID": [10.0],
            "State": ["active"],
        }
    )
    out = tmp_path / "clip_final.csv"
    assert write_base_final_csv(df, str(out)) is True
    got = pd.read_csv(out)
    assert list(got.columns)[:5] == ["TrajectoryID", "X", "Y", "Theta", "FrameID"]
    assert "TrackID" not in got.columns and "Index" not in got.columns
    assert got["X"].tolist() == [2] and got["Y"].tolist() == [3]  # rounded


def test_base_final_empty_returns_false(tmp_path):
    assert write_base_final_csv(pd.DataFrame(), str(tmp_path / "x.csv")) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_trajectory_writer_base_final.py -v`
Expected: FAIL — `write_base_final_csv` undefined.

- [ ] **Step 3: Implement `write_base_final_csv`**

Add to `src/hydra_suite/core/post/trajectory_writer.py` (verbatim logic moved from `session._save_trajectories_to_csv`):

```python
def write_base_final_csv(df: pd.DataFrame, output_path: str) -> bool:
    """Write the debug base-final CSV: round X/Y/FrameID, drop TrackID/Index, reorder."""
    if df is None:
        return False
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected post-processed trajectories as a pandas DataFrame.")
    if df.empty:
        return False

    df_to_save = df.copy()
    for column in ["X", "Y", "FrameID"]:
        if column in df_to_save.columns:
            df_to_save[column] = pd.to_numeric(df_to_save[column], errors="coerce")
            df_to_save[column] = df_to_save[column].round().astype("Int64")

    df_to_save = df_to_save.drop(
        columns=[c for c in ["TrackID", "Index"] if c in df_to_save.columns],
        errors="ignore",
    )
    base_columns = ["TrajectoryID", "X", "Y", "Theta", "FrameID"]
    ordered_columns = base_columns + [
        c for c in df_to_save.columns if c not in base_columns
    ]
    df_to_save[ordered_columns].to_csv(output_path, index=False)
    return True
```

- [ ] **Step 4: Delegate from `session._save_trajectories_to_csv`**

Replace the body of `_save_trajectories_to_csv` (`session.py:97-127`) with a thin delegation, and add the import near the top of `session.py`:

```python
from hydra_suite.core.post.trajectory_writer import write_base_final_csv
```
```python
def _save_trajectories_to_csv(trajectories, output_path: str) -> bool:
    """Persist post-processed trajectories (delegates to the shared base-final writer)."""
    return write_base_final_csv(trajectories, output_path)
```

- [ ] **Step 5: Route the relink bare write through the helper**

In `src/hydra_suite/core/post/rich_export.py:396`, replace `relinked_base.to_csv(final_csv_path, index=False)` with:

```python
        write_base_final_csv(relinked_base, final_csv_path)
```
and add `from hydra_suite.core.post.trajectory_writer import write_base_final_csv` to the module imports. **Byte-identity check:** `relinked_base` already contains the rounded/reordered columns, so routing it through `write_base_final_csv` (which is idempotent on already-Int64 columns) must not change the output — verify in Step 7 via the equivalence smoke.

- [ ] **Step 6: Delete the orphaned twin + fix its tests**

Delete `save_trajectories_to_csv` from `src/hydra_suite/core/post/media_export.py` (`:50-101`). In `tests/test_media_export.py`, repoint the two references (`:39`, `:47`) at `hydra_suite.core.post.trajectory_writer.write_base_final_csv` (DataFrame case) or delete the list-of-tuples case if it only exercised the dead branch. Remove any now-unused imports (`csv`, `np`) from `media_export.py`.

- [ ] **Step 7: Run tests**

Run: `python -m pytest tests/test_trajectory_writer_base_final.py tests/test_media_export.py -v`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/hydra_suite/core/post/trajectory_writer.py \
        src/hydra_suite/core/tracking/session.py \
        src/hydra_suite/core/post/rich_export.py \
        src/hydra_suite/core/post/media_export.py tests/test_media_export.py
git commit -m "refactor(post): single base-final CSV writer; delete orphaned media_export twin"
```

---

### Task 5: Mode-aware terminal write (User `<video>_tracks.csv` vs Debug `_with_individual.csv`)

**Files:**
- Modify: `src/hydra_suite/core/post/rich_export.py` (`write_rich_export_csv` call sites at `:338` and `:402`; `export_rich_csv` / `relink_and_export_rich_csv` signatures)
- Modify: `src/hydra_suite/core/tracking/session.py:279-297` (`_export_rich` / `_relink_export_rich` pass `debug_mode` + `fps`)
- Test: `tests/test_rich_export_mode_aware.py` (new)

**Interfaces:**
- Produces: `write_final_trajectories(rich_df: pd.DataFrame, final_csv_path: str, *, debug_mode: bool, fps: float | None) -> str | None` in `trajectory_writer.py`. Debug: writes `_with_individual.csv` (delegates to existing `write_rich_export_csv`). User: writes `<stem>_tracks.csv` (from `project_user_tracks`) and returns its path; does not write `_with_individual.csv`.
- Consumes: `project_user_tracks` (Task 3), `write_rich_export_csv` (existing), and the `DEBUG_MODE`/`FPS` params resolved in Task 2.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_rich_export_mode_aware.py
import os

import pandas as pd

from hydra_suite.core.post.trajectory_writer import write_final_trajectories


def _rich_df():
    return pd.DataFrame(
        {
            "TrajectoryID": [1, 1],
            "FrameID": [0, 1],
            "X": [1.0, 2.0],
            "Y": [3.0, 4.0],
            "Theta": [0.0, 0.0],
            "State": ["active", "active"],
            "DetectionConfidence": [0.9, 0.8],
        }
    )


def test_user_mode_writes_tracks_csv_only(tmp_path):
    final_csv = str(tmp_path / "clip_final.csv")
    path = write_final_trajectories(_rich_df(), final_csv, debug_mode=False, fps=10.0)
    assert path.endswith("_tracks.csv")
    assert os.path.exists(path)
    assert not os.path.exists(str(tmp_path / "clip_final_with_individual.csv"))
    cols = pd.read_csv(path).columns.tolist()
    assert cols == [
        "id",
        "frame",
        "time_s",
        "x",
        "y",
        "heading_deg",
        "state",
        "detection_confidence",
    ]


def test_debug_mode_writes_with_individual(tmp_path):
    final_csv = str(tmp_path / "clip_final.csv")
    path = write_final_trajectories(_rich_df(), final_csv, debug_mode=True, fps=10.0)
    assert path.endswith("_with_individual.csv")
    assert os.path.exists(path)
    assert not os.path.exists(str(tmp_path / "clip_tracks.csv"))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_rich_export_mode_aware.py -v`
Expected: FAIL — `write_final_trajectories` undefined.

- [ ] **Step 3: Implement `write_final_trajectories` + `user_tracks_path`**

Add to `src/hydra_suite/core/post/trajectory_writer.py`:

```python
import os

from hydra_suite.core.post.rich_export import write_rich_export_csv

_FINAL_SUFFIXES = ("_final", "_forward_processed")


def user_tracks_path(final_csv_path: str) -> str:
    """Derive the clean `<stem>_tracks.csv` path from a debug final-CSV path."""
    base, ext = os.path.splitext(final_csv_path)
    for suffix in _FINAL_SUFFIXES:
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    return f"{base}_tracks{ext}"


def write_final_trajectories(
    rich_df: pd.DataFrame,
    final_csv_path: str,
    *,
    debug_mode: bool,
    fps: float | None,
) -> str | None:
    """Terminal trajectory writer. Debug → `_with_individual.csv`; User → `<stem>_tracks.csv`."""
    if debug_mode:
        return write_rich_export_csv(rich_df, final_csv_path)
    clean = project_user_tracks(rich_df, fps=fps)
    out_path = user_tracks_path(final_csv_path)
    clean.to_csv(out_path, index=False)
    return out_path
```

> Import note: `write_rich_export_csv` lives in `rich_export.py`; importing it at module top-level in `trajectory_writer.py` is safe (both are in `core/post`, no cycle since `rich_export` does not import `trajectory_writer` at module scope — it imports `write_base_final_csv` inside function bodies added in Task 4, or add that import lazily inside the relink function to avoid a cycle). If a circular import appears, move the `write_base_final_csv` import in `rich_export.py` to inside the functions that use it.

- [ ] **Step 4: Branch the rich-export path on `debug_mode`**

In `src/hydra_suite/core/post/rich_export.py`, add `debug_mode: bool = True` and `fps: float | None = None` keyword params to `export_rich_csv` and `relink_and_export_rich_csv`. At the two `write_rich_export_csv(...)` call sites (`:338`, `:402`), replace with the mode-aware writer:

```python
    from hydra_suite.core.post.trajectory_writer import write_final_trajectories
    return write_final_trajectories(with_pose_df, final_csv_path, debug_mode=debug_mode, fps=fps)
```
(and the analogous `relinked_with_pose` variant at `:402`). Default `debug_mode=True` keeps existing callers/tests byte-identical.

- [ ] **Step 5: Thread `debug_mode` + `fps` from the session**

In `src/hydra_suite/core/tracking/session.py`, update `_export_rich` and `_relink_export_rich` (`:279-297`) to forward the resolved params:

```python
    def _export_rich(self, final_csv):
        return export_rich_csv(
            final_csv,
            self.pose_state,
            params=self.params,
            min_valid_conf=float(self.params.get("POSE_MIN_KPT_CONF_VALID", 0.2)),
            ignore_keypoints=self.config.get("pose_ignore_keypoints"),
            identity_evidence_cache_path=self._identity_evidence_cache_path(),
            debug_mode=bool(self.params.get("DEBUG_MODE", True)),
            fps=self.params.get("FPS"),
        )
```
(apply the same two kwargs to `_relink_export_rich`).

- [ ] **Step 6: Run tests**

Run: `python -m pytest tests/test_rich_export_mode_aware.py tests/test_rich_export.py -v`
Expected: PASS (existing `test_rich_export.py` stays green because `debug_mode` defaults to `True`).

- [ ] **Step 7: Commit**

```bash
git add src/hydra_suite/core/post/trajectory_writer.py \
        src/hydra_suite/core/post/rich_export.py \
        src/hydra_suite/core/tracking/session.py tests/test_rich_export_mode_aware.py
git commit -m "feat(post): mode-aware terminal writer — User tracks.csv vs Debug with_individual"
```

---

### Task 6: User-mode intermediate cleanup

**Files:**
- Modify: `src/hydra_suite/core/tracking/session.py` (add end-of-run cleanup when `DEBUG_MODE` is off)
- Modify: `src/hydra_suite/trackerkit/gui/orchestrators/session.py:405` (force cleanup when debug off)
- Test: `tests/test_session_user_mode_cleanup.py` (new)

**Interfaces:**
- Consumes: `DEBUG_MODE` param (Task 2), `user_tracks_path` (Task 5).
- Produces: after a User-mode run, only `<stem>_tracks.csv` + annotated video (if enabled) + explicit export-workflow outputs remain; intermediates (`_final`/`_forward_processed`/`_forward`/`_backward` CSVs, raw `_tracking_*` CSVs, `_with_individual.csv`) are removed.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_session_user_mode_cleanup.py
from hydra_suite.core.tracking.session import _user_mode_intermediate_paths


def test_intermediate_paths_enumerated():
    paths = _user_mode_intermediate_paths(base="/out/clip", ext=".csv")
    assert "/out/clip_final.csv" in paths
    assert "/out/clip_forward.csv" in paths
    assert "/out/clip_backward.csv" in paths
    assert "/out/clip_forward_processed.csv" in paths
    assert "/out/clip_final_with_individual.csv" in paths
    # the clean deliverable must NOT be in the delete set
    assert "/out/clip_tracks.csv" not in paths
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_session_user_mode_cleanup.py -v`
Expected: FAIL — `_user_mode_intermediate_paths` undefined.

- [ ] **Step 3: Implement the enumerator + call it at end of run**

Add to `session.py`:

```python
def _user_mode_intermediate_paths(base: str, ext: str) -> list[str]:
    """Intermediate CSVs to delete after a User-mode run (never the clean tracks.csv)."""
    return [
        f"{base}_final{ext}",
        f"{base}_forward{ext}",
        f"{base}_backward{ext}",
        f"{base}_forward_processed{ext}",
        f"{base}_final_with_individual{ext}",
        f"{base}_tracking_forward{ext}",
        f"{base}_tracking_backward{ext}",
        f"{base}_tracking_final{ext}",
    ]
```

At the very end of the run method in `TrackingSessionCore` (after dataset generation, media export, and annotated-video steps at `session.py:534-538`, before returning the result), add:

```python
            if not bool(self.params.get("DEBUG_MODE", True)):
                for _p in _user_mode_intermediate_paths(base, ext):
                    try:
                        if os.path.exists(_p):
                            os.remove(_p)
                    except OSError:
                        logger.warning("Failed to remove intermediate %s", _p, exc_info=True)
```
(ensure `import os` and the module `logger` exist in `session.py`; both already do.)

- [ ] **Step 4: Force GUI cleanup when debug off**

In `src/hydra_suite/trackerkit/gui/orchestrators/session.py:405`, the current guard is `if not self._mw._postprocess_panel.chk_cleanup_temp_files.isChecked():`. After Task 8 removes that checkbox, this must read the debug state. Change it now to derive from the config's debug flag so cleanup is forced on in User mode:

```python
        _debug = bool(self._mw.config.debug_mode)
        if not _debug:  # User mode always cleans up temp files
```
(If `self._mw.config.debug_mode` does not yet exist, this task can gate on `not self._mw.config.debug_mode` after Task 8 adds the field; sequence Task 8 before this GUI edit if needed, or add the `debug_mode` field to `TrackerConfig` here as a no-op default `False`.)

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_session_user_mode_cleanup.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/hydra_suite/core/tracking/session.py \
        src/hydra_suite/trackerkit/gui/orchestrators/session.py \
        tests/test_session_user_mode_cleanup.py
git commit -m "feat(trackerkit): clean up intermediates in User mode"
```

---

### Task 7: User-mode golden across three fixtures

**Files:**
- Create: `tests/goldens/user_mode/{fly_obb,ant_cnn_identity,ant_pose_headtail}_tracks.csv` (committed goldens)
- Test: `tests/test_user_mode_golden.py` (new)

**Interfaces:**
- Consumes: the full User-mode pipeline (Tasks 1–6).
- Produces: committed golden CSVs proving the clean schema across pure-tracking / identity / pose fixtures.

- [ ] **Step 1: Generate the goldens (one-time, review before committing)**

For each fixture, run the headless pipeline with `debug_mode: false`. Copy each gate config, set `"debug_mode": false`, and run:

```bash
conda activate hydra-mps
for clip in fly_obb ant_cnn_identity ant_pose_headtail; do
  python -c "import json; c=json.load(open('tools/equivalence/fixtures/configs/${clip}.json')); c['debug_mode']=False; json.dump(c, open('/tmp/${clip}_user.json','w'))"
  # Run headless tracking on tools/equivalence/fixtures/clips/${clip}.mp4 with /tmp/${clip}_user.json
  # (use the same headless entrypoint run_matrix.sh invokes; confirm conda is active so SLEAP clips are non-empty)
done
```
Verify each produced `<clip>_tracks.csv` has `wc -l > 1` (non-empty), the expected columns per fixture (fly_obb: 8 core; ant_cnn_identity: + `identity`/`identity_confidence`/`identity_source`; ant_pose_headtail: + `<kpt>_x/_y/_conf`), then copy into `tests/goldens/user_mode/`.

- [ ] **Step 2: Write the golden test**

```python
# tests/test_user_mode_golden.py
import pandas as pd
import pytest

CASES = {
    "fly_obb": [
        "id", "frame", "time_s", "x", "y", "heading_deg", "state", "detection_confidence",
    ],
    "ant_cnn_identity": [
        "id", "frame", "time_s", "x", "y", "heading_deg", "state",
        "detection_confidence", "identity", "identity_confidence", "identity_source",
    ],
}


@pytest.mark.parametrize("clip,cols", CASES.items())
def test_user_mode_columns(clip, cols):
    golden = pd.read_csv(f"tests/goldens/user_mode/{clip}_tracks.csv")
    assert list(golden.columns) == cols


def test_pose_fixture_has_keypoint_triples():
    golden = pd.read_csv("tests/goldens/user_mode/ant_pose_headtail_tracks.csv")
    assert any(c.endswith("_conf") for c in golden.columns)
    assert any(c.endswith("_x") for c in golden.columns)
    assert "heading_deg" in golden.columns
```

- [ ] **Step 3: Run the golden test**

Run: `python -m pytest tests/test_user_mode_golden.py -v`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/goldens/user_mode/ tests/test_user_mode_golden.py
git commit -m "test(trackerkit): commit User-mode clean-CSV goldens (3 fixtures)"
```

---

### Task 8: GUI consolidation — remove diagnostic checkboxes, add Debug Mode toggle

**Files:**
- Modify: `src/hydra_suite/trackerkit/config/schemas.py` (add `debug_mode: bool = False` to `TrackerConfig`, round-trip in `to_dict`/`from_dict`)
- Modify: `src/hydra_suite/trackerkit/gui/main_window.py` (add checkable **Debug Mode** button near `:844-856`, back it with `self.config.debug_mode`)
- Modify: `src/hydra_suite/trackerkit/gui/orchestrators/config.py` (write/restore `debug_mode`; delete the 9 checkbox reads/restores listed below)
- Modify: `src/hydra_suite/trackerkit/gui/panels/{setup_panel,detection_panel,tracking_panel,postprocess_panel}.py` (delete the 9 checkbox widgets)
- Modify: `tests/test_get_parameters_dict_characterization.py` goldens (regenerate — debug-off values)
- Test: `tests/test_debug_mode_config_roundtrip.py` (new)

**Interfaces:**
- Produces: `TrackerConfig.debug_mode: bool` (default `False`); `build_config_dict` emits `"debug_mode"`; the toolbar toggle persists it.
- Consumes: `DEBUG_MODE` derivation (Task 2), cleanup wiring (Task 6).

- [ ] **Step 1: Write the failing config round-trip test**

```python
# tests/test_debug_mode_config_roundtrip.py
from hydra_suite.trackerkit.config.schemas import TrackerConfig


def test_debug_mode_defaults_false_and_roundtrips():
    cfg = TrackerConfig()
    assert cfg.debug_mode is False
    cfg.debug_mode = True
    restored = TrackerConfig.from_dict(cfg.to_dict())
    assert restored.debug_mode is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_debug_mode_config_roundtrip.py -v`
Expected: FAIL — `TrackerConfig` has no `debug_mode`.

- [ ] **Step 3: Add `debug_mode` to the schema**

In `src/hydra_suite/trackerkit/config/schemas.py`, add the field to the dataclass (`:11-30`): `debug_mode: bool = False`. Add `"debug_mode": self.debug_mode` to `to_dict` (`:32-41`) and `debug_mode=bool(data.get("debug_mode", False))` to `from_dict` (`:43-61`).

- [ ] **Step 4: Run the round-trip test to verify it passes**

Run: `python -m pytest tests/test_debug_mode_config_roundtrip.py -v`
Expected: PASS.

- [ ] **Step 5: Add the Debug Mode toggle button**

In `src/hydra_suite/trackerkit/gui/main_window.py`, mirror the `btn_preview` pattern (`:844-853`) in the same `btn_layout`:

```python
        self.btn_debug_mode = QPushButton("Debug Mode")
        self.btn_debug_mode.setCheckable(True)
        self.btn_debug_mode.setChecked(self.config.debug_mode)
        self.btn_debug_mode.setToolTip(
            "Debug Mode: retain all intermediate files, diagnostic columns, "
            "profiling, and overlays. Off = clean single trajectory CSV."
        )
        self.btn_debug_mode.setMinimumHeight(34)
        self.btn_debug_mode.clicked.connect(self._on_debug_mode_toggled)
        btn_layout.addWidget(self.btn_debug_mode)
```
Add the slot (reusing the existing `toggle_debug_logging` at `:2517`):

```python
    def _on_debug_mode_toggled(self, checked: bool) -> None:
        self.config.debug_mode = bool(checked)
        self.toggle_debug_logging(checked)  # debug_logging derives from debug_mode
```

- [ ] **Step 6: Persist/restore `debug_mode` in the orchestrator**

In `src/hydra_suite/trackerkit/gui/orchestrators/config.py`:
- In `build_config_dict` (`:1434+`), add `"debug_mode": self._mw.btn_debug_mode.isChecked(),` and **delete** the 9 diagnostic-flag writes at lines `:1525` (save_confidence_metrics), `:1723` (export_confidence_density_video), `:1741` (cleanup_temp_files), `:1783` (show_kalman_uncertainty), `:1784` (show_foreground_mask), `:1785` (show_background_model), `:1786` (show_yolo_obb), `:1788` (debug_logging), `:1789` (enable_profiling).
- In the load path, add `self._mw.btn_debug_mode.setChecked(get_cfg("debug_mode", default=False))` and **delete** the 9 restores at `:248-250`, `:812-813`, `:880-881`, `:1004-1006`, `:1007-1008`, `:1010-1011`, `:1013-1014`, `:1019-1021`, `:1022-1024`.
- In `_gui_display_overlay` (`:2053-2082`), the `SHOW_*` keys (`:2065`, `:2066`, `:2069`, `:2073-2075`) must derive from debug: replace each `...isChecked()` with `bool(self._mw.config.debug_mode)`.

- [ ] **Step 7: Delete the 9 checkbox widgets**

Delete these widget creations + their layout `addWidget`/`addRow` lines (and any now-empty group boxes / dead signal slots):
- `setup_panel.py`: `chk_debug_logging` (`:833-836`, `:847`), `chk_enable_profiling` (`:837-842`, `:848`), `check_save_confidence` (`:655-661`, `:697-698`, `:711`), `chk_show_kalman_uncertainty` (`:779-784`, `:795`).
- `detection_panel.py`: `chk_show_fg` (`:901-902`, `:906`), `chk_show_bg` (`:903-904`, `:907`), `chk_show_yolo_obb` (`:923-925`). If groups `g_overlays_bg` / `g_overlays_yolo` become empty, remove the group boxes and their visibility toggles at `:930-931`, `:1295-1296`.
- `tracking_panel.py`: `chk_export_confidence_density_video` (`:960-969`).
- `postprocess_panel.py`: `chk_cleanup_temp_files` (`:562-570`).

Grep after: `grep -rn "chk_debug_logging\|chk_enable_profiling\|check_save_confidence\|chk_show_kalman_uncertainty\|chk_show_fg\|chk_show_bg\|chk_show_yolo_obb\|chk_export_confidence_density_video\|chk_cleanup_temp_files" src/` must return nothing (all references repointed/removed).

- [ ] **Step 8: Regenerate the characterization golden**

The gate configs loaded into `MainWindow` now yield debug-off params (`DEBUG_MODE=False`, `ENABLE_PROFILING=False`, `EXPORT_CONFIDENCE_DENSITY_VIDEO=False`, `SHOW_*=False`). Run `python -m pytest tests/test_get_parameters_dict_characterization.py -v`, inspect the diff to confirm it matches exactly those expected flips, and update the committed golden.

Run: `python -m pytest tests/test_get_parameters_dict_characterization.py tests/test_gui_cli_param_equivalence.py tests/test_debug_mode_config_roundtrip.py -v`
Expected: PASS.

- [ ] **Step 9: Smoke-launch the GUI**

Run: `QT_QPA_PLATFORM=offscreen python -c "import hydra_suite.trackerkit.gui.main_window as m"` (import smoke) and, if feasible, launch `trackerkit` to confirm the Debug Mode button appears and the diagnostic checkboxes are gone.

- [ ] **Step 10: Commit**

```bash
git add src/hydra_suite/trackerkit/config/schemas.py \
        src/hydra_suite/trackerkit/gui/main_window.py \
        src/hydra_suite/trackerkit/gui/orchestrators/config.py \
        src/hydra_suite/trackerkit/gui/panels/ \
        tests/test_debug_mode_config_roundtrip.py \
        tests/test_get_parameters_dict_characterization.py
git commit -m "feat(trackerkit): single Debug Mode toggle; remove scattered diagnostic checkboxes"
```

---

### Task 9: Docstring fix + full equivalence gate (MPS + CUDA)

**Files:**
- Modify: `src/hydra_suite/data/csv_writer.py:21-28` (docstring reflects the real emitted schema)
- Verify only: `tools/equivalence/run_matrix.sh`

**Interfaces:**
- Consumes: the complete Debug-mode pipeline (Tasks 1–8).

- [ ] **Step 1: Fix the stale `CSVWriterThread` docstring**

Update `src/hydra_suite/data/csv_writer.py:21-28` so the documented columns match the real raw header from `build_tracking_csv_header` (always includes `DetectionConfidence, AssignmentConfidence, PositionUncertainty, DetectionID`, then the realtime identity block, then optional AprilTag columns). Remove any mention of a conditional confidence header.

- [ ] **Step 2: Kill stale sleap/hydra processes, then run the MPS equivalence gate**

```bash
pgrep -fl "sleap\|hydra" | grep -iv "grep\|multi-animal-tracker/\.git" || true   # inspect; kill only dead/stale sleap/hydra
conda activate hydra-mps
bash tools/equivalence/fixtures/fetch_fixtures.sh
git fetch origin --tags
git worktree add --detach .worktrees/equiv-legacy legacy/main
REPO=$PWD WT=$PWD MAIN_SRC=$PWD/.worktrees/equiv-legacy/src WT_SRC=$PWD/src \
  OUT=/tmp/equiv_debugmode RUNTIME=mps bash tools/equivalence/run_matrix.sh
```
Expected: every clip EQUIVALENT at its DETERMINISM floor for `_forward` and `_final` (positions p99 ≈ 0, θ within head/tail π-flip noise, identical row counts, 0 unmatched). Confirm CSV row counts `> 1` (conda active → non-empty). This proves Debug-mode output is byte-identical despite the always-on confidence emission (gate configs have `save_confidence_metrics: true`, no `debug_mode` key → `DEBUG_MODE=True` → debug branch).

- [ ] **Step 3: Run the CUDA gate on mehek**

```bash
ssh rutalab@mehek.taild08eb9.ts.net
cd ~/hydra-suite && git fetch origin && git checkout <this-branch-sha>
source ~/mambaforge/etc/profile.d/conda.sh && conda activate hydra-cuda
bash tools/equivalence/fixtures/fetch_fixtures.sh
git worktree add --detach .worktrees/equiv-legacy legacy/main
REPO=$PWD WT=$PWD MAIN_SRC=$PWD/.worktrees/equiv-legacy/src WT_SRC=$PWD/src \
  OUT=/tmp/equiv_debugmode RUNTIME=cuda nohup bash tools/equivalence/run_matrix.sh > /tmp/equiv_cuda.log 2>&1 &
```
Expected: same determinism-floor result on CUDA.

- [ ] **Step 4: User-mode smoke (both platforms optional; MPS required)**

Run one clip with `debug_mode: false` and confirm the output directory contains exactly `<clip>_tracks.csv` (+ annotated video if enabled) and none of the `_final`/`_forward`/`_backward`/`_with_individual` intermediates.

- [ ] **Step 5: Clean up worktrees + commit**

```bash
git worktree remove --force .worktrees/equiv-legacy && git worktree prune
git add src/hydra_suite/data/csv_writer.py
git commit -m "docs(csv_writer): refresh CSVWriterThread schema docstring"
```

---

## Self-Review notes

- **Spec coverage:** Component 1 (debug_mode field + derived flags) → Tasks 2, 8; Component 2 (UI consolidation) → Task 8; Component 3 (one mode-aware writer + three-writer consolidation) → Tasks 4, 5. User-mode schema → Task 3 + Task 7 goldens. Debug-mode byte-identity → Task 9. Config round-trip + characterization golden → Tasks 2, 8. Docstring → Task 9. `save_confidence_metrics` decision (user, 2026-08-11) → Task 1.
- **Deviation from spec, documented:** the locked flag table listed `save_confidence_metrics` as debug-gated; per the user's 2026-08-11 decision it is retired (always-on) instead, with the User/Debug split moved into the writer's column selection. Gate-safe because all 7 fixtures already have `save_confidence_metrics: true`.
- **Type consistency:** `write_base_final_csv(df, path) -> bool`, `project_user_tracks(df, *, fps) -> DataFrame`, `write_final_trajectories(rich_df, final_csv_path, *, debug_mode, fps) -> str | None`, `user_tracks_path(final_csv_path) -> str`, `build_tracking_csv_header(identity_method=...) -> list[str]`, `DEBUG_MODE` param, `TrackerConfig.debug_mode: bool` — used consistently across tasks.
- **Sequencing caveat:** Task 6 Step 4 references `self._mw.config.debug_mode`, which is added in Task 8 Step 3. If executing strictly in order, add the `debug_mode` field (Task 8 Step 3) before Task 6 Step 4, or gate Task 6's GUI edit behind a `getattr(self._mw.config, "debug_mode", False)`. The core-session cleanup (Task 6 Steps 1–3) is independent and can land first.
