# Headless Qt-Free Session Service — Slice 3 (Export Chain) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the final media export (annotated video + canonical crops/oriented videos) and active-learning dataset generation out of the GUI orchestrator into the Qt-free `TrackingSessionCore`, so the service reaches full parity with the hidden-`MainWindow` bridge and fills `SessionResult.media_paths` and `SessionResult.dataset_result`.

**Architecture:** Extract the ~1100 lines of drawing/render helpers from `trackerkit/gui/orchestrators/tracking.py` into pure functions in `core/post/media_export.py` (annotated video — Part 3b) and `core/post/dataset_export.py` (active-learning dataset — Part 3a); delegate canonical crops/oriented videos to the already-Qt-free `OrientedTrackVideoExporter` (`core/identity/dataset/oriented_video.py`). Add three inline stages to `TrackingSessionCore` (`core/tracking/session.py`, from Slice 2) that gate on the Slice-1 policy predicates, drive cooperative cancellation via `callbacks.should_stop()`, and announce `callbacks.stage_changed(name)`. The GUI's per-stage `QThread` workers are repointed to call the new core functions so behavior is byte-identical while the logic lives in `core/`. Split into **Part 3a** (canonical crops + oriented videos + dataset — mostly wiring around existing cores) and **Part 3b** (annotated-overlay-video extraction — the real code move), each independently gated so a reviewer can accept one without the other.

**Tech Stack:** Python 3, NumPy, pandas, OpenCV (`cv2`), plain `threading`/`queue` (NOT Qt), the existing `OrientedTrackVideoExporter` + `data.dataset_generation` cores, `hydra_suite.utils.video_encoder.VideoEncoder`, pytest, the `tools/equivalence/` harness. No new dependencies.

## Global Constraints

- Commit as the configured git user; **NO** `Co-Authored-By` trailer of any kind.
- Run `make format` before **each** commit (autopep8 → black → isort). If `make` is broken in the environment, run `conda run -n hydra-mps black <files>` and `conda run -n hydra-mps isort <files>` directly.
- Run tests with: `conda run -n hydra-mps python -m pytest <path> -q --ignore=tests/test_identity_postprocess.py` (the ignored file has a pre-existing collection error). The env is `hydra-mps`.
- **`core/` stays Qt-free.** After every task, `grep -rnE "PySide6|QtCore|QThread|Signal|Slot|QMutex" src/hydra_suite/core/` MUST be EMPTY. Plain `threading.Thread`/`queue.Queue` are allowed in `core/`; Qt is not.
- **`core/` must not import any app-layer package** (`trackerkit`, `posekit`, `classkit`, `refinekit`, `detectkit`, `filterkit`) or `integrations`. It may import `data`, `training`, `runtime`, `utils`, `resources`, `paths`, and other `core` modules.
- Equivalence gate (final task): `tools/equivalence/run_matrix.sh` against the same baseline **before and after**, on **both** MPS (`hydra-mps`, this box) **and** CUDA (mehek, `hydra-cuda`), across all 7 fixture clips. Acceptance: positions p99 ≈ 0, θ max ≈ 0, identical row counts, 0 unmatched, on both `_forward.csv` **and** `_tracking_final.csv`. Known noise floor: bistable head/tail π-flips on head/tail clips only. Conda **must** be active for any pose/SLEAP clip or the CSVs come out empty and falsely compare EQUIVALENT — verify `wc -l` > 1 before trusting a pass.
- Video/media output is **NOT** covered by the CSV-comparing equivalence harness. This slice adds explicit **media-parity checks** (same output file set + non-empty + same frame count, GUI vs service) in its tests — Tasks 6 and 9/Task 10.
- Cancellation: every long loop (annotated-video frame render, dataset frame scoring, crop extraction) checks `should_stop()` cooperatively. On cancel, `cv2.VideoCapture`/`VideoEncoder` are released **and the partial output file is deleted** so no half-written video survives as a valid-looking output.

---

## Interfaces from earlier slices (treat as EXISTING — do not re-implement)

**Slice 1** — `core/tracking/session_policy.py`, pure predicates over a config dict:

```python
def should_export_final_canonical_images(config: dict) -> bool: ...   # config["enable_individual_dataset"] and config["enable_individual_pipeline"]
def is_individual_image_save_enabled(config: dict) -> bool: ...        # alias of the above
def should_export_final_media_videos(config: dict) -> bool: ...        # config["enable_individual_track_videos"] and config["enable_individual_pipeline"]
```

**Slice 1** — unified `TRAJECTORY_COLORS` generator (GUI's legacy RNG), landed in `core/tracking/visualization.py`:

```python
def build_trajectory_colors(n: int) -> list[tuple[int, int, int]]:
    """np.random.seed(42) + np.random.randint(0,255,(n,3)) form — the single shared palette."""
    np.random.seed(42)
    return [tuple(int(v) for v in c) for c in np.random.randint(0, 255, (n, 3))]
```

Both `get_parameters_dict()` (GUI) and `TrackerCliSession.params` (CLI) call this so `params["TRAJECTORY_COLORS"]` is identical on both paths. `media_export` **consumes** `params["TRAJECTORY_COLORS"]` and never generates its own — Task 3 adds a test that pins this.

**Slice 2** — `core/tracking/session.py`:

```python
@dataclass
class SessionCallbacks:
    progress: Callable[[int, str], None] = _noop2
    status: Callable[[str], None] = _noop1
    warning: Callable[[str, str], None] = _noop2      # non-fatal; replaces QMessageBox.information/.warning
    stage_changed: Callable[[str], None] = _noop1     # drives GUI widget enable/disable
    should_stop: Callable[[], bool] = _never

@dataclass
class SessionResult:
    success: bool
    final_csv_path: str | None
    rich_export_path: str | None
    media_paths: list[str]          # Slice 2 leaves this []; Slice 3 fills it
    dataset_result: dict | None     # Slice 2 leaves this None; Slice 3 fills it
    summary_lines: list[str]
    error: str | None

class TrackingSessionCore:
    def __init__(self, *, video_path, config, params, paths, callbacks=SessionCallbacks()): ...
    def run_post_tracking(self, forward_trajectories, backward_trajectories=None) -> SessionResult: ...
```

`TrackingSessionCore` already runs merge → post-process → pose merge → identity post-pass → interpolated crops → rich export, and populates `SessionResult.final_csv_path`/`.rich_export_path`. This slice appends the export stages and populates `.media_paths`/`.dataset_result`.

Slice-2 fields available on `self` inside the service (set in `__init__`): `self.video_path: str`, `self.config: dict` (lowercase JSON vocabulary), `self.params: dict` (uppercase vocabulary), `self.callbacks: SessionCallbacks`, and `self.paths` — an object carrying resolved output paths. Task 8 uses `self.paths.detection_cache_path`, `self.paths.interpolated_roi_npz_path`, `self.paths.individual_dataset_dir`, `self.paths.final_media_video_dir`, `self.paths.source_video_fps`. Slice 2 introduced this object; **if any of those five fields is absent, add it there as the first step of Task 8** (they are data the GUI resolves via `_resolve_current_individual_dataset_dir` / `_resolve_current_final_media_video_dir` / `_resolve_source_video_fps`, `tracking.py:1308-1372`).

**Slice 2** — `core/post/rich_export.py`:

```python
def rich_export_path(final_csv_path: str, *, legacy: bool = False) -> str: ...
```

**NOTE on the dataset builder (verified in source):** the spec's component table says dataset generation "delegates to existing `training/` service", but the actual builder the GUI's `DatasetGenerationWorker` calls lives in `hydra_suite.data.dataset_generation` (`export_dataset` at `:764`, `FrameQualityScorer` at `:24`), imported at `dataset_worker.py:61`. There is **no** ready-made `generate_active_learning_dataset()` library function — the frame-scoring loop lives *inside* `DatasetGenerationWorker.execute` (`dataset_worker.py:57-203`). Task 7 extracts that loop into a Qt-free function. Follow the real code: delegate to `hydra_suite.data.dataset_generation`.

**Re-derive line numbers before starting** — merges shift them:

```bash
grep -nE "def (_generate_final_media_export|_get_video_draw_params|_get_pose_column_info|_preextract_traj_arrays|_draw_trail_for_track|_draw_single_track_on_frame|_render_annotated_video_frames|_open_video_cap_and_writer|_compute_video_frame_range|_generate_video_from_trajectories|_load_video_trajectories|_generate_training_dataset|_format_video_track_label|_build_video_track_label_array|_normalize_video_identity_color_key|_build_video_track_color_key_array|_build_precomputed_color_palette|_scale_trajectories_to_original_space|save_trajectories_to_csv)\b" src/hydra_suite/trackerkit/gui/orchestrators/tracking.py
```

---

## File Structure

| File | Responsibility |
|---|---|
| `core/post/media_export.py` (create) | Pure functions for the annotated-video overlay chain + `render_annotated_video()` entry point, plus `export_final_media()` (canonical crops/oriented videos via `OrientedTrackVideoExporter`), plus `scale_trajectories_to_original_space` / `save_trajectories_to_csv` / `load_video_trajectories`. **Qt-free.** |
| `core/post/dataset_export.py` (create) | `generate_active_learning_dataset()` — pure extraction of `DatasetGenerationWorker.execute` (frame scoring + `export_dataset`) with `progress`/`should_stop` callbacks. **Qt-free.** |
| `core/tracking/session.py` (modify, Slice-2 file) | Add `_run_final_media_export`, `_run_dataset_generation`, `_run_annotated_video`; call them from `run_post_tracking`; fill `SessionResult.media_paths`/`.dataset_result`. |
| `trackerkit/gui/orchestrators/tracking.py` (modify) | Repoint annotated-video methods to `media_export`; remove the moved bodies. |
| `trackerkit/gui/workers/dataset_worker.py` (modify) | `execute()` calls `dataset_export.generate_active_learning_dataset`. |
| `tests/test_media_export.py` (create) | Unit tests for all pure media functions + color-unification pin. |
| `tests/test_dataset_export.py` (create) | Unit test for `generate_active_learning_dataset`. |
| `tests/test_session_export_chain.py` (create) | Service-level tests: media_paths/dataset_result filled, cancellation deletes partial video, media-parity frame count. |
| `tests/test_core_qtfree_slice3.py` (create) | Guard: grep `core/` for Qt tokens stays empty. |

**Part mapping:** Part 3b = Tasks 1-5; Part 3a = Tasks 6-7; both wired into the service in Task 8; GUI repointed in Task 9; gated in Task 10.

---

## Task 1 (3b): Trajectory CSV + coordinate scaling — pure move

**Files:**
- Create: `src/hydra_suite/core/post/media_export.py`
- Test: `tests/test_media_export.py`
- Source: `tracking.py:743-770` (`_scale_trajectories_to_original_space`), `:772-850` (`save_trajectories_to_csv`)

**Interfaces:**
- Produces: `scale_trajectories_to_original_space(trajectories_df, resize_factor) -> pd.DataFrame`; `save_trajectories_to_csv(trajectories, output_path) -> bool`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_media_export.py
import os

import numpy as np
import pandas as pd
import pytest

from hydra_suite.core.post import media_export


def test_scale_trajectories_noop_when_factor_is_one():
    df = pd.DataFrame({"X": [10.0], "Y": [20.0], "Theta": [0.5], "FrameID": [0]})
    out = media_export.scale_trajectories_to_original_space(df, 1.0)
    assert out is df  # unchanged object when no scaling needed


def test_scale_trajectories_scales_xy_only():
    df = pd.DataFrame({"X": [10.0], "Y": [20.0], "Theta": [0.5], "FrameID": [3]})
    out = media_export.scale_trajectories_to_original_space(df, 0.5)
    assert out["X"].iloc[0] == pytest.approx(20.0)
    assert out["Y"].iloc[0] == pytest.approx(40.0)
    assert out["Theta"].iloc[0] == pytest.approx(0.5)  # angle not scaled
    assert out["FrameID"].iloc[0] == 3


def test_save_trajectories_to_csv_writes_ordered_columns(tmp_path):
    df = pd.DataFrame(
        {
            "TrajectoryID": [0, 0],
            "X": [10.4, 11.6],
            "Y": [20.0, 21.0],
            "Theta": [0.1, 0.2],
            "FrameID": [0, 1],
            "TrackID": [5, 5],
            "Extra": ["a", "b"],
        }
    )
    out = tmp_path / "traj.csv"
    assert media_export.save_trajectories_to_csv(df, str(out)) is True
    written = pd.read_csv(out)
    assert list(written.columns)[:5] == ["TrajectoryID", "X", "Y", "Theta", "FrameID"]
    assert "TrackID" not in written.columns   # unwanted dropped
    assert written["X"].iloc[0] == 10          # rounded to Int64


def test_save_trajectories_to_csv_none_returns_false(tmp_path):
    assert media_export.save_trajectories_to_csv(None, str(tmp_path / "x.csv")) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n hydra-mps python -m pytest tests/test_media_export.py -q --ignore=tests/test_identity_postprocess.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'hydra_suite.core.post.media_export'`

- [ ] **Step 3: Write minimal implementation**

Create `src/hydra_suite/core/post/media_export.py`. (`core/post/__init__.py` already exists — `processing.py` lives there — so no new `__init__` is needed.) Both functions are transcribed verbatim from `tracking.py` with `self` dropped (both are already pure — `_scale` takes no `self` state; `save_trajectories_to_csv(self: object, ...)` uses no `self`):

```python
"""Qt-free media export: trajectory persistence, coordinate scaling, and the
annotated-video overlay chain. Extracted from trackerkit/gui/orchestrators/tracking.py
(Slice 3 of the headless-session-service program)."""

from __future__ import annotations

import csv
import json
import logging
import os
import re

import cv2
import numpy as np
import pandas as pd

from hydra_suite.core.identity.dataset.oriented_video import OrientedTrackVideoExporter
from hydra_suite.core.identity.properties.export import build_pose_keypoint_labels
from hydra_suite.utils.pose_visualization import (
    is_renderable_pose_keypoint,
    normalize_pose_render_min_conf,
)
from hydra_suite.utils.video_encoder import VideoEncoder

logger = logging.getLogger(__name__)


def scale_trajectories_to_original_space(trajectories_df, resize_factor):
    """Scale trajectory coordinates from resized space back to original video space."""
    if trajectories_df is None or trajectories_df.empty:
        return trajectories_df
    if resize_factor == 1.0:
        return trajectories_df
    scale_factor = 1.0 / resize_factor
    logger.info(
        f"Scaling trajectories to original video space (resize_factor={resize_factor:.3f}, scale_factor={scale_factor:.3f})"
    )
    result_df = trajectories_df.copy()
    result_df["X"] = result_df["X"] * scale_factor
    result_df["Y"] = result_df["Y"] * scale_factor
    logger.info(
        f"Scaled {len(result_df)} trajectory points to original video coordinates"
    )
    return result_df


def save_trajectories_to_csv(trajectories, output_path):
    """Save processed trajectories to CSV. Accepts a DataFrame or list-of-tuples."""
    if trajectories is None:
        logger.warning("No post-processed trajectories to save (None).")
        return False
    if isinstance(trajectories, pd.DataFrame):
        if trajectories.empty:
            logger.warning("No post-processed trajectories to save (empty DataFrame).")
            return False
        try:
            df_to_save = trajectories.copy()
            for col in ["X", "Y", "FrameID"]:
                if col in df_to_save.columns:
                    df_to_save[col] = pd.to_numeric(df_to_save[col], errors="coerce")
                    df_to_save[col] = df_to_save[col].round().astype("Int64")
            unwanted_cols = ["TrackID", "Index"]
            df_to_save = df_to_save.drop(
                columns=[col for col in unwanted_cols if col in df_to_save.columns],
                errors="ignore",
            )
            base_cols = ["TrajectoryID", "X", "Y", "Theta", "FrameID"]
            other_cols = [col for col in df_to_save.columns if col not in base_cols]
            ordered_cols = base_cols + other_cols
            df_to_save[ordered_cols].to_csv(output_path, index=False)
            logger.info(
                f"Successfully saved {df_to_save['TrajectoryID'].nunique()} post-processed trajectories "
                f"({len(df_to_save)} rows) with {len(ordered_cols)} columns to {output_path}"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to save processed trajectories to {output_path}: {e}")
            return False

    if not trajectories:
        logger.warning("No post-processed trajectories to save.")
        return False
    header = ["TrajectoryID", "X", "Y", "Theta", "FrameID"]
    try:
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for trajectory_id, segment in enumerate(trajectories):
                for x, y, theta, frame_id in segment:
                    x_val = int(x) if not np.isnan(x) else ""
                    y_val = int(y) if not np.isnan(y) else ""
                    frame_val = int(frame_id) if not np.isnan(frame_id) else ""
                    writer.writerow([trajectory_id, x_val, y_val, theta, frame_val])
        logger.info(f"Saved trajectories to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to save trajectories to {output_path}: {e}")
        return False
```

> The `OrientedTrackVideoExporter` import is added now (used in Task 6) so the module has one import block. If Task 6 is deferred, the import is harmless.

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n hydra-mps python -m pytest tests/test_media_export.py -q --ignore=tests/test_identity_postprocess.py`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
make format
git add src/hydra_suite/core/post/media_export.py tests/test_media_export.py
git commit -m "feat(post): add media_export with trajectory scaling + CSV save (Slice 3)"
```

---

## Task 2 (3b): Identity label + color-key builders — pure move

**Files:**
- Modify: `src/hydra_suite/core/post/media_export.py`
- Test: `tests/test_media_export.py`
- Source: `tracking.py:1802-1860` (`_format_video_track_label`), `:1862-1893` (`_build_video_track_label_array`), `:1895-1929` (`_normalize_video_identity_color_key`), `:1931-1958` (`_build_video_track_color_key_array`), `:1960-2010` (`_build_precomputed_color_palette`)

**Interfaces:**
- Produces: `format_video_track_label(track_id, unique_identity_key=None) -> str`; `build_video_track_label_array(df) -> np.ndarray`; `normalize_video_identity_color_key(value) -> str`; `build_video_track_color_key_array(df) -> np.ndarray`; `build_precomputed_color_palette(colors, track_ids, color_keys) -> list`

- [ ] **Step 1: Write the failing test** — append to `tests/test_media_export.py`

```python
def test_normalize_identity_key_treats_unknown_as_empty():
    assert media_export.normalize_video_identity_color_key("unknown") == ""
    assert media_export.normalize_video_identity_color_key(np.nan) == ""
    assert media_export.normalize_video_identity_color_key(None) == ""
    assert media_export.normalize_video_identity_color_key("apriltag=3") == "apriltag=3"


def test_format_label_falls_back_to_track_id():
    assert media_export.format_video_track_label(7, None) == "ID7"
    assert media_export.format_video_track_label(7, "") == "ID7"


def test_color_key_array_prefers_identity_then_trajectory():
    df = pd.DataFrame(
        {
            "TrajectoryID": [0, 1],
            "UniqueIdentityKey": ["apriltag=3", "unknown"],
        }
    )
    keys = media_export.build_video_track_color_key_array(df)
    assert keys[0] == "identity:apriltag=3"
    assert keys[1] == "trajectory:1"


def test_precomputed_palette_uses_trajectory_colors_for_plain_tracks():
    colors = [(10, 20, 30), (40, 50, 60), (70, 80, 90)]
    track_ids = np.asarray([0, 1, 2], dtype=np.int32)
    color_keys = np.asarray(
        ["trajectory:0", "trajectory:1", "trajectory:2"], dtype=object
    )
    row_colors = media_export.build_precomputed_color_palette(
        colors, track_ids, color_keys
    )
    assert row_colors == [(10, 20, 30), (40, 50, 60), (70, 80, 90)]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n hydra-mps python -m pytest tests/test_media_export.py -k "identity or label or color_key or palette" -q --ignore=tests/test_identity_postprocess.py`
Expected: FAIL — `AttributeError: module 'hydra_suite.core.post.media_export' has no attribute 'normalize_video_identity_color_key'`

- [ ] **Step 3: Write minimal implementation** — append to `media_export.py`. Transcribe verbatim from `tracking.py`, dropping `self`, and change the internal `self._normalize_video_identity_color_key(...)` / `self._format_video_track_label(...)` calls to module-level calls:

```python
def format_video_track_label(track_id, unique_identity_key=None) -> str:
    """Return the overlay label for one rendered track row."""
    token = str(unique_identity_key).strip() if unique_identity_key is not None else ""
    if token and token.lower() != "nan":
        try:
            from hydra_suite.core.post.identity_postprocess import parse_identity_key

            parsed = parse_identity_key(token)
        except Exception:
            parsed = {}
        if parsed:
            compact_parts = []
            cnn_parts_by_label: dict[str, list[str]] = {}
            for source in sorted(parsed):
                value = str(parsed[source]).strip()
                if not value:
                    continue
                if source == "apriltag":
                    compact_parts.append(f"Tag {value}")
                    continue
                if source.startswith("cnn:"):
                    parts = source.split(":")
                    label = parts[1] if len(parts) >= 2 else source
                    compact_value = value
                    if len(parts) >= 3:
                        compact_value = value
                    elif "+" in value:
                        pieces = []
                        for item in value.split("+"):
                            item = str(item).strip()
                            if not item:
                                continue
                            if ":" in item:
                                item = str(item.split(":", 1)[1]).strip()
                            if item:
                                pieces.append(item)
                        if pieces:
                            compact_value = " / ".join(pieces)
                    if compact_value:
                        cnn_parts_by_label.setdefault(label, []).append(compact_value)
                    continue
                compact_parts.append(f"{source}={value}")
            for label in sorted(cnn_parts_by_label):
                values = [value for value in cnn_parts_by_label[label] if value]
                if not values:
                    continue
                compact_parts.append(values[0] if len(values) == 1 else " / ".join(values))
            if compact_parts:
                return " | ".join(compact_parts)
        return token
    return f"ID{track_id}"


def normalize_video_identity_color_key(value):
    """Return a stable identity color key token or an empty string."""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    token = str(value).strip()
    if not token or token.lower() == "nan":
        return ""
    if token.lower() == "unknown":
        return ""
    try:
        from hydra_suite.core.post.identity_postprocess import parse_identity_key

        parsed = parse_identity_key(token)
    except Exception:
        parsed = {}
    if parsed:
        informative_values = [
            str(v).strip()
            for v in parsed.values()
            if str(v).strip() and str(v).strip().lower() != "unknown"
        ]
        if not informative_values:
            return ""
    return token


def build_video_track_label_array(trajectories_df):
    """Precompute one overlay label per row using stable identity when available."""
    if trajectories_df is None or len(trajectories_df) == 0:
        return np.asarray([], dtype=object)
    identity_columns = [
        "UniqueIdentityKey",
        "IdentityAssignedLabel",
        "IdentityOfflineLabel",
        "IdentitySmoothedLabel",
    ]
    track_ids = trajectories_df["TrajectoryID"].tolist()
    labels = []
    for row_index, track_id in enumerate(track_ids):
        chosen_token = None
        for column in identity_columns:
            if column not in trajectories_df.columns:
                continue
            token = normalize_video_identity_color_key(
                trajectories_df.iloc[row_index][column]
            )
            if token:
                chosen_token = token
                break
        labels.append(format_video_track_label(track_id, chosen_token))
    return np.asarray(labels, dtype=object)


def build_video_track_color_key_array(trajectories_df):
    """Precompute one color key per row, preferring identity evidence over TrajectoryID."""
    if trajectories_df is None or len(trajectories_df) == 0:
        return np.asarray([], dtype=object)
    identity_columns = [
        "UniqueIdentityKey",
        "IdentityAssignedLabel",
        "IdentityOfflineLabel",
        "IdentitySmoothedLabel",
    ]
    track_ids = trajectories_df["TrajectoryID"].tolist()
    color_keys = []
    for row_index, track_id in enumerate(track_ids):
        chosen_key = ""
        for column in identity_columns:
            if column not in trajectories_df.columns:
                continue
            token = normalize_video_identity_color_key(
                trajectories_df.iloc[row_index][column]
            )
            if token:
                chosen_key = f"identity:{token}"
                break
        if not chosen_key:
            chosen_key = f"trajectory:{int(track_id)}"
        color_keys.append(chosen_key)
    return np.asarray(color_keys, dtype=object)


def build_precomputed_color_palette(colors, _track_ids, color_keys):
    """Build per-row colors, reusing one color for rows with the same identity key."""
    _category20_colors = [
        (127, 127, 31), (188, 189, 34), (140, 86, 75), (255, 127, 14),
        (214, 39, 40), (255, 152, 150), (197, 176, 213), (148, 103, 189),
        (196, 156, 148), (227, 119, 194), (199, 199, 199), (140, 140, 140),
        (23, 190, 207), (158, 218, 229), (57, 59, 121), (82, 84, 163),
        (107, 110, 207), (156, 158, 222), (99, 121, 57), (140, 162, 82),
    ]
    _n_cat = len(_category20_colors)

    def _fallback_color(_track_id):
        _tid = int(_track_id)
        return (
            tuple(colors[_tid])
            if colors and _tid < len(colors)
            else _category20_colors[_tid % _n_cat]
        )

    _identity_palette = {}
    _next_identity_color_idx = 0
    _row_colors = []
    for _tid, _key in zip(_track_ids.tolist(), color_keys.tolist()):
        _key_token = str(_key)
        if _key_token.startswith("identity:"):
            if _key_token not in _identity_palette:
                _identity_palette[_key_token] = (
                    tuple(colors[_next_identity_color_idx])
                    if colors and _next_identity_color_idx < len(colors)
                    else _category20_colors[_next_identity_color_idx % _n_cat]
                )
                _next_identity_color_idx += 1
            _row_colors.append(_identity_palette[_key_token])
            continue
        _row_colors.append(_fallback_color(_tid))
    return _row_colors
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n hydra-mps python -m pytest tests/test_media_export.py -q --ignore=tests/test_identity_postprocess.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
make format
git add src/hydra_suite/core/post/media_export.py tests/test_media_export.py
git commit -m "feat(post): move identity label + color-key builders to media_export (Slice 3)"
```

---

## Task 3 (3b): Color-unification pin — GUI/CLI drift guard

This is the concrete GUI/CLI defect the whole program fixes. The pin asserts `media_export`'s row-color palette matches the Slice-1 unified helper `build_trajectory_colors`.

**Files:**
- Test: `tests/test_media_export.py`
- Consumes: `hydra_suite.core.tracking.visualization.build_trajectory_colors` (Slice 1)

- [ ] **Step 1: Write the test** — append to `tests/test_media_export.py`

```python
def test_media_export_palette_matches_unified_trajectory_colors():
    """media_export must color plain tracks with the Slice-1 unified palette,
    not a locally-generated one — this pins the GUI/CLI color-drift fix."""
    from hydra_suite.core.tracking.visualization import build_trajectory_colors

    n = 5
    colors = build_trajectory_colors(n)

    # Reference values from the GUI's legacy RNG (np.random.seed(42)+randint).
    np.random.seed(42)
    expected = [tuple(int(v) for v in c) for c in np.random.randint(0, 255, (n, 3))]
    assert colors == expected

    df = pd.DataFrame({"TrajectoryID": [0, 1, 2, 3, 4]})
    color_keys = media_export.build_video_track_color_key_array(df)
    track_ids = df["TrajectoryID"].to_numpy(dtype=np.int32)
    row_colors = media_export.build_precomputed_color_palette(
        colors, track_ids, color_keys
    )
    assert row_colors == expected
```

- [ ] **Step 2: Run the test**

Run: `conda run -n hydra-mps python -m pytest tests/test_media_export.py::test_media_export_palette_matches_unified_trajectory_colors -q --ignore=tests/test_identity_postprocess.py`
Expected: PASS if Slice 1 is merged (it is a prerequisite). If it fails with `ImportError` on `build_trajectory_colors`, Slice 1 is not present — **stop and confirm Slice 1 is merged before proceeding.**

- [ ] **Step 3: (no implementation)** — this task pins existing behavior; no source change.

- [ ] **Step 4: Commit**

```bash
git add tests/test_media_export.py
git commit -m "test(post): pin media_export palette to unified TRAJECTORY_COLORS (Slice 3)"
```

---

## Task 4 (3b): Draw-param + pose-column + array pre-extraction — pure move

**Files:**
- Modify: `src/hydra_suite/core/post/media_export.py`
- Test: `tests/test_media_export.py`
- Source: `tracking.py:1604-1672` (`_get_video_draw_params`), `:1674-1735` (`_get_pose_column_info`), `:1737-1800` (`_preextract_traj_arrays`)

**Interfaces:**
- Produces: `build_video_draw_params(params, config, fps, trajectories_df) -> dict`; `get_pose_column_info(params, advanced_config, trajectories_df) -> tuple[list, list, bool]`; `preextract_traj_arrays(trajectories_df, show_pose, pose_column_triplets, show_trails) -> tuple`
- Config keys read by `build_video_draw_params` (verified against `config.py:1732-1738`): `video_show_labels`, `video_show_orientation`, `video_show_trails`, `video_trail_duration`, `video_marker_size`, `video_text_scale`, `video_arrow_length`.

> Confirm each of the seven keys exists in Slice-1 `build_config_dict()` output (grep the widget name — e.g. `check_show_labels` — in `save_config`, which writes them at `config.py:1732-1738`). A missing key is a Slice-1 gap to report, not a locally-invented default.

- [ ] **Step 1: Write the failing test** — append to `tests/test_media_export.py`

```python
def _draw_config():
    return {
        "video_show_labels": True,
        "video_show_orientation": True,
        "video_show_trails": False,
        "video_trail_duration": 1.0,
        "video_marker_size": 0.3,
        "video_text_scale": 0.5,
        "video_arrow_length": 0.7,
    }


def test_build_video_draw_params_reads_config_keys():
    params = {"TRAJECTORY_COLORS": [(1, 2, 3)], "REFERENCE_BODY_SIZE": 40.0,
              "ADVANCED_CONFIG": {}, "POSE_MIN_KPT_CONF_VALID": 0.2}
    df = pd.DataFrame({"TrajectoryID": [0], "X": [1.0], "Y": [2.0]})
    draw_p = media_export.build_video_draw_params(params, _draw_config(), 30.0, df)
    assert draw_p["show_labels"] is True
    assert draw_p["show_trails"] is False
    assert draw_p["marker_radius"] == int(0.3 * 40.0)
    assert draw_p["arrow_len"] == int(0.7 * 40.0)
    assert draw_p["colors"] == [(1, 2, 3)]


def test_get_pose_column_info_false_without_pose_columns():
    df = pd.DataFrame({"TrajectoryID": [0], "X": [1.0], "Y": [2.0]})
    edges, triplets, show_pose = media_export.get_pose_column_info({}, {}, df)
    assert show_pose is False
    assert triplets == []


def test_preextract_traj_arrays_indexes_by_frame():
    df = pd.DataFrame(
        {"TrajectoryID": [0, 0], "FrameID": [0, 1], "X": [1.0, 2.0],
         "Y": [3.0, 4.0], "Theta": [0.0, 0.1]}
    )
    arrays = media_export.preextract_traj_arrays(df, False, [], False)
    traj_indices_by_frame = arrays[7]
    assert traj_indices_by_frame[0] == [0]
    assert traj_indices_by_frame[1] == [1]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n hydra-mps python -m pytest tests/test_media_export.py -k "draw_params or pose_column or preextract" -q --ignore=tests/test_identity_postprocess.py`
Expected: FAIL — `AttributeError: ... has no attribute 'build_video_draw_params'`

- [ ] **Step 3: Write minimal implementation** — append to `media_export.py`. Transcribe from `tracking.py`, replacing the seven `self._panels.postprocess.<widget>.value()/.isChecked()` reads with config-dict lookups (mapping per the Interfaces block), and replacing `self._build_video_track_label_array(...)` with the module function. Keep every derived-value computation (`marker_radius`, `arrow_len`, `text_size`, `marker_thickness`, pose fields) byte-identical:

```python
def build_video_draw_params(params, config, fps, trajectories_df):
    """Return drawing parameters derived from params, config dict, and body size."""
    colors = params.get("TRAJECTORY_COLORS", [])
    reference_body_size = params.get("REFERENCE_BODY_SIZE", 30.0)
    show_labels = bool(config.get("video_show_labels", True))
    show_orientation = bool(config.get("video_show_orientation", True))
    show_trails = bool(config.get("video_show_trails", False))
    trail_duration_sec = float(config.get("video_trail_duration", 1.0))
    trail_duration_frames = int(trail_duration_sec * fps)
    marker_size = float(config.get("video_marker_size", 0.3))
    text_scale = float(config.get("video_text_scale", 0.5))
    arrow_length = float(config.get("video_arrow_length", 0.7))
    advanced_config = params.get("ADVANCED_CONFIG", {})
    marker_radius = int(marker_size * reference_body_size)
    arrow_len = int(arrow_length * reference_body_size)
    text_size = 0.5 * text_scale
    marker_thickness = max(2, int(0.15 * reference_body_size))
    pose_point_radius = int(
        max(1, advanced_config.get("video_pose_point_radius", max(2, marker_radius // 3)))
    )
    pose_point_thickness = int(advanced_config.get("video_pose_point_thickness", -1))
    pose_line_thickness = int(max(1, advanced_config.get("video_pose_line_thickness", 2)))
    pose_color_mode = str(advanced_config.get("video_pose_color_mode", "track")).strip().lower()
    pose_fixed_color_raw = advanced_config.get("video_pose_color", [255, 255, 255])
    if isinstance(pose_fixed_color_raw, (list, tuple)) and len(pose_fixed_color_raw) == 3:
        try:
            pose_fixed_color = tuple(int(max(0, min(255, float(v)))) for v in pose_fixed_color_raw)
        except Exception:
            pose_fixed_color = (255, 255, 255)
    else:
        pose_fixed_color = (255, 255, 255)
    pose_min_conf = normalize_pose_render_min_conf(params.get("POSE_MIN_KPT_CONF_VALID", 0.2))
    return dict(
        colors=colors,
        show_labels=show_labels,
        show_orientation=show_orientation,
        show_trails=show_trails,
        trail_duration_frames=trail_duration_frames,
        marker_radius=marker_radius,
        arrow_len=arrow_len,
        text_size=text_size,
        text_scale=text_scale,
        marker_thickness=marker_thickness,
        pose_point_radius=pose_point_radius,
        pose_point_thickness=pose_point_thickness,
        pose_line_thickness=pose_line_thickness,
        pose_color_mode=pose_color_mode,
        pose_fixed_color=pose_fixed_color,
        pose_min_conf=pose_min_conf,
        advanced_config=advanced_config,
    )


def get_pose_column_info(params, advanced_config, trajectories_df):
    """Return (pose_edges, pose_column_triplets, show_pose) for video rendering."""
    pose_edges = []
    pose_column_triplets = []
    show_pose = bool(advanced_config.get("video_show_pose", True))
    pose_col_pattern = re.compile(r"^PoseKpt_(.+)_(X|Y|Conf)$")
    pose_labels_available = {}
    for col in trajectories_df.columns:
        m = pose_col_pattern.match(str(col))
        if m is None:
            continue
        label = m.group(1)
        axis = m.group(2)
        pose_labels_available.setdefault(label, set()).add(axis)
    if not pose_labels_available:
        show_pose = False
    if show_pose:
        skeleton_names = []
        skeleton_file = str(params.get("POSE_SKELETON_FILE", "")).strip()
        if skeleton_file and os.path.exists(skeleton_file):
            try:
                with open(skeleton_file, "r", encoding="utf-8") as f:
                    skeleton_data = json.load(f)
                names_raw = skeleton_data.get("keypoint_names", skeleton_data.get("keypoints", []))
                skeleton_names = [str(n) for n in names_raw]
                raw_edges = skeleton_data.get("skeleton_edges", skeleton_data.get("edges", []))
                for edge in raw_edges:
                    if isinstance(edge, (list, tuple)) and len(edge) >= 2:
                        try:
                            pose_edges.append((int(edge[0]), int(edge[1])))
                        except Exception:
                            continue
            except Exception:
                pose_edges = []
        ordered_labels = build_pose_keypoint_labels(skeleton_names, len(skeleton_names))
        extras = sorted([lbl for lbl in pose_labels_available.keys() if lbl not in ordered_labels])
        ordered_labels.extend(extras)
        for label in ordered_labels:
            axes = pose_labels_available.get(label, set())
            if {"X", "Y", "Conf"}.issubset(axes):
                pose_column_triplets.append(
                    (f"PoseKpt_{label}_X", f"PoseKpt_{label}_Y", f"PoseKpt_{label}_Conf")
                )
        if not pose_column_triplets:
            show_pose = False
    return pose_edges, pose_column_triplets, show_pose


def preextract_traj_arrays(trajectories_df, show_pose, pose_column_triplets, show_trails):
    """Pre-extract trajectory arrays and index structures for O(1)/O(log N) lookups."""
    _frame_ids = trajectories_df["FrameID"].to_numpy(dtype=np.int32)
    _track_ids = trajectories_df["TrajectoryID"].to_numpy(dtype=np.int32)
    _xs = trajectories_df["X"].to_numpy(dtype=np.float64)
    _ys = trajectories_df["Y"].to_numpy(dtype=np.float64)
    _label_texts = build_video_track_label_array(trajectories_df)
    _thetas = (
        trajectories_df["Theta"].to_numpy(dtype=np.float64)
        if "Theta" in trajectories_df.columns
        else np.full(len(trajectories_df), np.nan)
    )
    _pose_kpts = None
    if show_pose and pose_column_triplets:
        _K = len(pose_column_triplets)
        _N = len(trajectories_df)
        _pose_kpts = np.full((_K, _N, 3), np.nan, dtype=np.float32)
        for _k, (_x_col, _y_col, _c_col) in enumerate(pose_column_triplets):
            if _x_col in trajectories_df.columns:
                _pose_kpts[_k, :, 0] = trajectories_df[_x_col].to_numpy(dtype=np.float32)
            if _y_col in trajectories_df.columns:
                _pose_kpts[_k, :, 1] = trajectories_df[_y_col].to_numpy(dtype=np.float32)
            if _c_col in trajectories_df.columns:
                _pose_kpts[_k, :, 2] = trajectories_df[_c_col].to_numpy(dtype=np.float32)
    traj_indices_by_frame: dict = {}
    for _i in range(len(_frame_ids)):
        _fid = int(_frame_ids[_i])
        if _fid not in traj_indices_by_frame:
            traj_indices_by_frame[_fid] = []
        traj_indices_by_frame[_fid].append(_i)
    _track_sorted_row_indices: dict = {}
    _track_sorted_frame_vals: dict = {}
    if show_trails:
        _tmp_track: dict = {}
        for _i in range(len(_track_ids)):
            _tid = int(_track_ids[_i])
            if _tid not in _tmp_track:
                _tmp_track[_tid] = []
            _tmp_track[_tid].append(_i)
        for _tid, _idxs in _tmp_track.items():
            _idx_arr = np.asarray(_idxs, dtype=np.int32)
            _order = np.argsort(_frame_ids[_idx_arr])
            _track_sorted_row_indices[_tid] = _idx_arr[_order]
            _track_sorted_frame_vals[_tid] = _frame_ids[_idx_arr[_order]]
    return (
        _frame_ids, _track_ids, _xs, _ys, _label_texts, _thetas, _pose_kpts,
        traj_indices_by_frame, _track_sorted_row_indices, _track_sorted_frame_vals,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n hydra-mps python -m pytest tests/test_media_export.py -q --ignore=tests/test_identity_postprocess.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
make format
git add src/hydra_suite/core/post/media_export.py tests/test_media_export.py
git commit -m "feat(post): move video draw-param/pose-column/array pre-extraction to media_export (Slice 3)"
```

---

## Task 5 (3b): Draw helpers + render entry point with cooperative cancellation

Highest behavioral-risk task: the render loop must (a) call a `progress` callback instead of `self._mw.progress_bar.setValue`/`QApplication.processEvents()`, (b) check `should_stop()` cooperatively, and (c) on cancel, release `cap`/`out` and **delete the partial output file**.

**Files:**
- Modify: `src/hydra_suite/core/post/media_export.py`
- Test: `tests/test_media_export.py`
- Source: `tracking.py:2012-2055` (`_draw_trail_for_track`), `:2057-2154` (`_draw_single_track_on_frame`), `:2155-2249` (`_render_annotated_video_frames`), `:2251-2270` (`_open_video_cap_and_writer`), `:2272-2286` (`_compute_video_frame_range`), `:2288-2390` (`_generate_video_from_trajectories`), `:3418-3434` (`_load_video_trajectories`)

**Interfaces:**
- Consumes: `hydra_suite.core.post.rich_export.rich_export_path` (Slice 2)
- Produces:
  - `draw_trail_for_track(frame, track_id, frame_idx, color, _xs, _ys, _track_sorted_frame_vals, _track_sorted_row_indices, trail_duration_frames, marker_thickness) -> None`
  - `draw_single_track_on_frame(frame, row_i, track_id, cx, cy, color, draw_p, _thetas, _pose_kpts, _label_texts, pose_edges) -> None`
  - `render_annotated_video_frames(cap, out, start_frame, total_frames, draw_p, pose_edges, show_pose, arrays, progress=None, should_stop=None) -> bool` — `True` if completed, `False` if cancelled.
  - `open_video_cap_and_writer(video_path, output_path) -> tuple | None`
  - `compute_video_frame_range(params, total_video_frames) -> tuple[int, int, int]`
  - `render_annotated_video(*, trajectories_df, video_path, output_path, params, config, progress=None, should_stop=None) -> str | None` — output path on success, `None` on failure or cancel (deletes any partial file on cancel).
  - `load_video_trajectories(final_csv_path) -> tuple[pd.DataFrame | None, str | None]`

- [ ] **Step 1: Write the failing test** — append to `tests/test_media_export.py`

```python
import cv2  # noqa: E402  (top-of-file imports already include os/np/pd)


def _write_black_clip(path, n_frames=10, w=64, h=48, fps=10.0):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    for _ in range(n_frames):
        vw.write(np.zeros((h, w, 3), dtype=np.uint8))
    vw.release()


def _simple_traj_df(n_frames=10):
    return pd.DataFrame(
        {
            "TrajectoryID": [0] * n_frames,
            "FrameID": list(range(n_frames)),
            "X": [30.0] * n_frames,
            "Y": [24.0] * n_frames,
            "Theta": [0.0] * n_frames,
        }
    )


def _render_params():
    return {
        "TRAJECTORY_COLORS": [(0, 255, 0)],
        "REFERENCE_BODY_SIZE": 10.0,
        "ADVANCED_CONFIG": {},
        "POSE_MIN_KPT_CONF_VALID": 0.2,
        "START_FRAME": 0,
        "END_FRAME": None,
    }


def test_render_annotated_video_writes_output(tmp_path):
    src = tmp_path / "in.mp4"
    out = tmp_path / "out.mp4"
    _write_black_clip(src, n_frames=10)
    result = media_export.render_annotated_video(
        trajectories_df=_simple_traj_df(10),
        video_path=str(src),
        output_path=str(out),
        params=_render_params(),
        config=_draw_config(),
    )
    assert result == str(out)
    assert os.path.exists(out)
    cap = cv2.VideoCapture(str(out))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 10
    cap.release()


def test_render_annotated_video_cancel_deletes_partial(tmp_path):
    src = tmp_path / "in.mp4"
    out = tmp_path / "out.mp4"
    _write_black_clip(src, n_frames=30)
    calls = {"n": 0}

    def _stop():
        calls["n"] += 1
        return calls["n"] > 3  # stop after a few frames

    result = media_export.render_annotated_video(
        trajectories_df=_simple_traj_df(30),
        video_path=str(src),
        output_path=str(out),
        params=_render_params(),
        config=_draw_config(),
        should_stop=_stop,
    )
    assert result is None
    assert not os.path.exists(out)  # partial file removed
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n hydra-mps python -m pytest tests/test_media_export.py -k "render_annotated" -q --ignore=tests/test_identity_postprocess.py`
Expected: FAIL — `AttributeError: ... has no attribute 'render_annotated_video'`

- [ ] **Step 3: Write minimal implementation** — append to `media_export.py`.

The two `_draw_*` helpers are transcribed verbatim (they use only their arguments, `self` dropped):

```python
def draw_trail_for_track(
    frame, track_id, frame_idx, color, _xs, _ys,
    _track_sorted_frame_vals, _track_sorted_row_indices,
    trail_duration_frames, marker_thickness,
):
    """Draw the fading trail for a single track on the given frame."""
    if track_id not in _track_sorted_frame_vals:
        return
    _sfv = _track_sorted_frame_vals[track_id]
    _sri = _track_sorted_row_indices[track_id]
    _lo = int(np.searchsorted(_sfv, frame_idx - trail_duration_frames, side="left"))
    _hi = int(np.searchsorted(_sfv, frame_idx, side="left"))
    if _hi - _lo < 2:
        return
    _trail_xs = _xs[_sri[_lo:_hi]]
    _trail_ys = _ys[_sri[_lo:_hi]]
    _trail_fs = _sfv[_lo:_hi]
    _trail_lw = max(1, marker_thickness // 2)
    for _seg in range(_hi - _lo - 1):
        _px1, _py1 = _trail_xs[_seg], _trail_ys[_seg]
        _px2, _py2 = _trail_xs[_seg + 1], _trail_ys[_seg + 1]
        if np.isnan(_px1) or np.isnan(_py1) or np.isnan(_px2) or np.isnan(_py2):
            continue
        _age = frame_idx - int(_trail_fs[_seg])
        _alpha = 1.0 - (_age / trail_duration_frames)
        cv2.line(
            frame, (int(_px1), int(_py1)), (int(_px2), int(_py2)),
            (int(color[0] * _alpha), int(color[1] * _alpha), int(color[2] * _alpha)),
            _trail_lw,
        )


def draw_single_track_on_frame(
    frame, row_i, track_id, cx, cy, color, draw_p,
    _thetas, _pose_kpts, _label_texts, pose_edges,
):
    """Draw circle, label, orientation arrow, and pose for a single track."""
    marker_radius = draw_p["marker_radius"]
    marker_thickness = draw_p["marker_thickness"]
    cv2.circle(frame, (cx, cy), marker_radius, color, marker_thickness)
    if draw_p["show_labels"]:
        label_offset = int(marker_radius + 5)
        cv2.putText(
            frame, str(_label_texts[row_i]),
            (cx + label_offset, cy - label_offset),
            cv2.FONT_HERSHEY_SIMPLEX, draw_p["text_size"], color,
            max(1, int(draw_p["text_scale"] * 2)),
        )
    if draw_p["show_orientation"]:
        _theta = _thetas[row_i]
        if not np.isnan(_theta):
            cv2.arrowedLine(
                frame, (cx, cy),
                (int(cx + draw_p["arrow_len"] * np.cos(_theta)),
                 int(cy + draw_p["arrow_len"] * np.sin(_theta))),
                color, marker_thickness, tipLength=0.3,
            )
    if _pose_kpts is not None:
        kpts_arr = _pose_kpts[:, row_i, :]
        if np.any(np.isfinite(kpts_arr[:, 2])):
            pose_color = (
                color if draw_p["pose_color_mode"] == "track"
                else draw_p["pose_fixed_color"]
            )
            if pose_edges:
                for e0, e1 in pose_edges:
                    if e0 < 0 or e1 < 0 or e0 >= len(kpts_arr) or e1 >= len(kpts_arr):
                        continue
                    if not is_renderable_pose_keypoint(
                        kpts_arr[e0, 0], kpts_arr[e0, 1], kpts_arr[e0, 2], draw_p["pose_min_conf"]
                    ) or not is_renderable_pose_keypoint(
                        kpts_arr[e1, 0], kpts_arr[e1, 1], kpts_arr[e1, 2], draw_p["pose_min_conf"]
                    ):
                        continue
                    cv2.line(
                        frame,
                        (int(round(float(kpts_arr[e0, 0]))), int(round(float(kpts_arr[e0, 1])))),
                        (int(round(float(kpts_arr[e1, 0]))), int(round(float(kpts_arr[e1, 1])))),
                        pose_color, draw_p["pose_line_thickness"],
                    )
            for kpt in kpts_arr:
                if not is_renderable_pose_keypoint(kpt[0], kpt[1], kpt[2], draw_p["pose_min_conf"]):
                    continue
                cv2.circle(
                    frame, (int(round(float(kpt[0]))), int(round(float(kpt[1])))),
                    draw_p["pose_point_radius"], pose_color, draw_p["pose_point_thickness"],
                )
```

`render_annotated_video_frames` — transcribed from `tracking.py:2155-2249`, with the GUI progress block (`self._mw.progress_bar.setValue(...)` + `QApplication.processEvents()`) replaced by the `progress` callback and a `should_stop` check at the top of each iteration. It returns `False` on cancel so the caller can discard the partial file. The writer thread is always torn down (put `None`, join) before returning, on both paths:

```python
def render_annotated_video_frames(
    cap, out, start_frame, total_frames, draw_p, pose_edges, show_pose, arrays,
    progress=None, should_stop=None,
):
    """Write annotated frames from cap into out. Return True if completed, False if cancelled."""
    import queue as _queue
    import threading as _threading

    (
        _frame_ids, _track_ids, _xs, _ys, _label_texts, _thetas, _pose_kpts,
        traj_indices_by_frame, _track_sorted_row_indices, _track_sorted_frame_vals,
        _row_colors,
    ) = arrays
    _write_q: _queue.Queue = _queue.Queue(maxsize=4)

    def _writer_thread():
        while True:
            _item = _write_q.get()
            if _item is None:
                break
            out.write(_item)

    _writer = _threading.Thread(target=_writer_thread, daemon=True)
    _writer.start()
    cancelled = False

    for rel_idx in range(total_frames):
        if should_stop is not None and should_stop():
            cancelled = True
            break
        frame_idx = start_frame + rel_idx
        ret, frame = cap.read()
        if not ret:
            break

        frame_row_indices = traj_indices_by_frame.get(frame_idx, [])

        if draw_p["show_trails"]:
            for row_i in frame_row_indices:
                track_id = int(_track_ids[row_i])
                color = tuple(_row_colors[row_i])
                draw_trail_for_track(
                    frame, track_id, frame_idx, color, _xs, _ys,
                    _track_sorted_frame_vals, _track_sorted_row_indices,
                    draw_p["trail_duration_frames"], draw_p["marker_thickness"],
                )

        for row_i in frame_row_indices:
            track_id = int(_track_ids[row_i])
            cx_f, cy_f = _xs[row_i], _ys[row_i]
            if np.isnan(cx_f) or np.isnan(cy_f):
                continue
            cx, cy = int(cx_f), int(cy_f)
            color = tuple(_row_colors[row_i])
            draw_single_track_on_frame(
                frame, row_i, track_id, cx, cy, color, draw_p, _thetas,
                _pose_kpts if show_pose else None, _label_texts, pose_edges,
            )

        _write_q.put(frame)

        if progress is not None and rel_idx % 30 == 0:
            pct = int(((rel_idx + 1) / total_frames) * 100)
            progress(pct, "Generating video...")

    _write_q.put(None)
    _writer.join()
    return not cancelled


def open_video_cap_and_writer(video_path, output_path):
    """Open video capture and writer; return (cap, out, fps, total_video_frames) or None on error."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    try:
        out = VideoEncoder(output_path, fps=fps, width=frame_width, height=frame_height)
    except Exception:
        logger.error(f"Failed to create output video: {output_path}")
        cap.release()
        return None
    logger.info(f"Writing video: {frame_width}x{frame_height} @ {fps} FPS")
    return cap, out, fps, total_video_frames


def compute_video_frame_range(params, total_video_frames):
    """Return (start_frame, end_frame, total_frames) clamped to video bounds."""
    start_frame = int(params.get("START_FRAME", 0) or 0)
    end_frame = params.get("END_FRAME", None)
    if end_frame is None:
        end_frame = total_video_frames - 1 if total_video_frames > 0 else 0
    end_frame = int(end_frame)
    if total_video_frames > 0:
        start_frame = max(0, min(start_frame, total_video_frames - 1))
        end_frame = max(start_frame, min(end_frame, total_video_frames - 1))
    total_frames = max(0, end_frame - start_frame + 1)
    logger.info(f"Exporting tracked frame range: {start_frame}-{end_frame} ({total_frames} frames)")
    return start_frame, end_frame, total_frames


def load_video_trajectories(final_csv_path):
    """Load best available trajectories for video generation (prefers rich export CSV)."""
    from hydra_suite.core.post.rich_export import rich_export_path

    if not final_csv_path:
        return None, None
    candidates = [
        rich_export_path(final_csv_path),
        rich_export_path(final_csv_path, legacy=True),
        final_csv_path,
    ]
    candidate = next((path for path in candidates if os.path.exists(path)), None)
    if not candidate:
        return None, None
    try:
        return pd.read_csv(candidate), candidate
    except Exception:
        logger.exception("Failed to load video trajectories from: %s", candidate)
        return None, None
```

`render_annotated_video` — the entry point rebuilt from `_generate_video_from_trajectories` (`tracking.py:2288-2390`) minus its `_complete()`/`_finalize_tracking_session_ui` GUI wiring. Both paths and `params`/`config` arrive as arguments (the original read them from widgets and via `self._mw.get_parameters_dict()`). The `_row_colors` array is assembled exactly as the original does (`build_video_track_color_key_array` → `build_precomputed_color_palette`) so the 11-element `arrays` tuple matches. On cancel it releases `cap`/`out` and deletes the partial file:

```python
def render_annotated_video(
    *, trajectories_df, video_path, output_path, params, config,
    progress=None, should_stop=None,
):
    """Generate an annotated overlay video from post-processed trajectories.

    Returns the output path on success, or None on failure/cancellation (partial
    output file is deleted on cancellation so no half-written video survives)."""
    logger.info("=" * 80)
    logger.info("Generating video from post-processed trajectories...")
    logger.info("=" * 80)

    if trajectories_df is None or trajectories_df.empty:
        return None
    if not video_path or not output_path:
        logger.error("Video input or output path not specified")
        return None

    opened = open_video_cap_and_writer(video_path, output_path)
    if opened is None:
        return None
    cap, out, fps, total_video_frames = opened

    start_frame, end_frame, total_frames = compute_video_frame_range(
        params, total_video_frames
    )
    if total_frames <= 0:
        logger.error("Invalid frame range for video generation.")
        cap.release()
        out.release()
        return None

    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    draw_p = build_video_draw_params(params, config, fps, trajectories_df)
    pose_edges, pose_column_triplets, show_pose = get_pose_column_info(
        params, draw_p["advanced_config"], trajectories_df
    )
    (
        _frame_ids, _track_ids, _xs, _ys, _label_texts, _thetas, _pose_kpts,
        traj_indices_by_frame, _track_sorted_row_indices, _track_sorted_frame_vals,
    ) = preextract_traj_arrays(
        trajectories_df, show_pose, pose_column_triplets, draw_p["show_trails"]
    )
    _color_keys = build_video_track_color_key_array(trajectories_df)
    _row_colors = build_precomputed_color_palette(draw_p["colors"], _track_ids, _color_keys)

    arrays = (
        _frame_ids, _track_ids, _xs, _ys, _label_texts, _thetas, _pose_kpts,
        traj_indices_by_frame, _track_sorted_row_indices, _track_sorted_frame_vals,
        _row_colors,
    )
    completed = render_annotated_video_frames(
        cap, out, start_frame, total_frames, draw_p, pose_edges, show_pose, arrays,
        progress=progress, should_stop=should_stop,
    )

    cap.release()
    out.release()

    if not completed:
        logger.info("Annotated video generation cancelled; removing partial output.")
        try:
            if os.path.exists(output_path):
                os.remove(output_path)
        except OSError:
            logger.warning("Could not delete partial video: %s", output_path)
        return None

    logger.info(f"✓ Video saved to: {output_path}")
    logger.info("=" * 80)
    return output_path
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n hydra-mps python -m pytest tests/test_media_export.py -q --ignore=tests/test_identity_postprocess.py`
Expected: PASS (all media_export tests)

- [ ] **Step 5: Verify `core/` stays Qt-free**

Run: `grep -rnE "PySide6|QtCore|QThread|Signal|Slot|QMutex" src/hydra_suite/core/`
Expected: EMPTY output.

- [ ] **Step 6: Commit**

```bash
make format
git add src/hydra_suite/core/post/media_export.py tests/test_media_export.py
git commit -m "feat(post): add annotated-video render entry point with cancellation (Slice 3)"
```

---

## Task 6 (3a): Canonical crops / oriented videos — `export_final_media`

Wraps the already-Qt-free `OrientedTrackVideoExporter` (`core/identity/dataset/oriented_video.py`). The GUI's `FinalMediaExportWorker` is a thin `BaseWorker` around this exporter (`video_worker.py:71-105`); this task lifts the construction + gating into a pure function so the service can call the exporter without the worker.

**Files:**
- Modify: `src/hydra_suite/core/post/media_export.py`
- Test: `tests/test_media_export.py`
- Source: `tracking.py:1284-1416` (`_generate_final_media_export` — the constructor argument mapping) → config keys per `config.py:1869-1891`, `1481-1482`
- Consumes: `OrientedTrackVideoExporter` ctor (`oriented_video.py:171-196`), `OrientedTrackVideoExportResult.to_dict` (`oriented_video.py:95-113`)

**Interfaces:**
- Produces: `export_final_media(*, final_csv_path, config, video_path, detection_cache_path, interpolated_roi_npz_path, fps, image_root, video_root, export_images, export_videos, padding_fraction, background_color, progress=None, should_stop=None) -> dict | None` — the exporter's `result.to_dict()`, or `None` when nothing was requested / preconditions unmet. Config keys read (verified against `config.py`): `suppress_foreign_obb_individual_dataset`, `suppress_foreign_obb_oriented_videos`, `individual_save_interval`, `individual_output_format`, `final_media_export_fix_direction_flips`, `final_media_export_heading_flip_burst`, `final_media_export_enable_affine_stabilization`, `final_media_export_stabilization_window`.

> Confirm each config key against Slice-1 `build_config_dict()` before writing (grep the widget in `save_config`). A missing key is a Slice-1 gap to report, not a locally-invented default.

- [ ] **Step 1: Write the failing test** — append to `tests/test_media_export.py`

```python
def test_export_final_media_returns_none_when_nothing_requested(tmp_path):
    result = media_export.export_final_media(
        final_csv_path=str(tmp_path / "final.csv"),
        config={},
        video_path=str(tmp_path / "in.mp4"),
        detection_cache_path=str(tmp_path / "cache.npz"),
        interpolated_roi_npz_path=None,
        fps=30.0,
        image_root=None,
        video_root=None,
        export_images=False,
        export_videos=False,
        padding_fraction=0.1,
        background_color=(0, 0, 0),
    )
    assert result is None


def test_export_final_media_delegates_to_exporter(tmp_path, monkeypatch):
    captured = {}

    class _FakeResult:
        def to_dict(self):
            return {"exported_videos": 2, "exported_images": 0, "output_dir": "vids"}

    class _FakeExporter:
        def __init__(self, dataset_dir, final_csv_path, **kwargs):
            captured["dataset_dir"] = str(dataset_dir)
            captured["kwargs"] = kwargs

        def export(self, progress_callback=None, should_stop=None):
            return _FakeResult()

    monkeypatch.setattr(media_export, "OrientedTrackVideoExporter", _FakeExporter)
    result = media_export.export_final_media(
        final_csv_path=str(tmp_path / "final.csv"),
        config={
            "individual_save_interval": 2,
            "individual_output_format": "png",
            "final_media_export_heading_flip_burst": 5,
            "final_media_export_stabilization_window": 5,
        },
        video_path=str(tmp_path / "in.mp4"),
        detection_cache_path=str(tmp_path / "cache.npz"),
        interpolated_roi_npz_path=None,
        fps=30.0,
        image_root=None,
        video_root=tmp_path / "vroot",
        export_images=False,
        export_videos=True,
        padding_fraction=0.1,
        background_color=(0, 0, 0),
    )
    assert result == {"exported_videos": 2, "exported_images": 0, "output_dir": "vids"}
    assert captured["kwargs"]["export_videos"] is True
    assert captured["kwargs"]["image_interval"] == 2
```

The first test's `detection_cache_path` points at a nonexistent file, but the `export_images/export_videos` are both `False`, so the early `return None` fires before the cache check — the test still passes. The second test monkeypatches the exporter, so no real cache is read.

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n hydra-mps python -m pytest tests/test_media_export.py -k "export_final_media" -q --ignore=tests/test_identity_postprocess.py`
Expected: FAIL — `AttributeError: ... has no attribute 'export_final_media'`

- [ ] **Step 3: Write minimal implementation** — append to `media_export.py`. (`OrientedTrackVideoExporter` was imported at module top in Task 1.) The argument mapping mirrors `tracking.py:1365-1399`; each `self._panels.<...>` read becomes a config-dict lookup and each `self._mw.<...>` read is an argument. `export_root` mirrors `video_root or image_root` (`tracking.py:1330`); `image_output_dir` mirrors `str((image_root / "images"))` (`tracking.py:1331-1333`). The `suppress_foreign_obb` positional (first bool at `tracking.py:1375-1379`) picks the video flag when exporting videos, else the dataset flag:

```python
def export_final_media(
    *, final_csv_path, config, video_path, detection_cache_path,
    interpolated_roi_npz_path, fps, image_root, video_root,
    export_images, export_videos, padding_fraction, background_color,
    progress=None, should_stop=None,
):
    """Export final canonical stills and/or orientation-fixed per-track videos.

    Returns the exporter result dict, or None if nothing is requested or the
    final CSV / detection cache is missing."""
    if not export_images and not export_videos:
        return None
    if not final_csv_path or not os.path.exists(final_csv_path):
        return None
    if not detection_cache_path or not os.path.exists(detection_cache_path):
        logger.warning(
            "Skipping final canonical media export: no compatible detection cache is available."
        )
        return None
    if export_images and image_root is None:
        logger.warning("Skipping final canonical image export: no image output directory found.")
        export_images = False
    if export_videos and video_root is None:
        logger.warning("Skipping final media video export: no video output directory found.")
        export_videos = False
    if not export_images and not export_videos:
        return None

    from pathlib import Path

    export_root = video_root or image_root
    image_output_dir = str((Path(image_root) / "images").expanduser()) if image_root else None

    suppress_dataset = bool(config.get("suppress_foreign_obb_individual_dataset", False))
    suppress_videos = bool(config.get("suppress_foreign_obb_oriented_videos", False))
    suppress_foreign_obb = suppress_videos if export_videos else suppress_dataset

    exporter = OrientedTrackVideoExporter(
        str(export_root),
        final_csv_path,
        video_path=video_path,
        detection_cache_path=detection_cache_path,
        interpolated_roi_npz_path=interpolated_roi_npz_path,
        fps=fps,
        padding_fraction=max(0.0, float(padding_fraction)),
        background_color=tuple(int(c) for c in background_color),
        suppress_foreign_obb=suppress_foreign_obb,
        suppress_foreign_obb_images=suppress_dataset,
        suppress_foreign_obb_videos=suppress_videos,
        export_images=export_images,
        image_output_dir=image_output_dir,
        image_interval=int(config.get("individual_save_interval", 1)),
        image_format=str(config.get("individual_output_format", "png")),
        export_videos=export_videos,
        fix_direction_flips=bool(config.get("final_media_export_fix_direction_flips", False)),
        heading_flip_max_burst=int(config.get("final_media_export_heading_flip_burst", 5)),
        enable_affine_stabilization=bool(
            config.get("final_media_export_enable_affine_stabilization", False)
        ),
        stabilization_window=int(config.get("final_media_export_stabilization_window", 5)),
        output_subdir="",
    )
    result = exporter.export(progress_callback=progress, should_stop=should_stop)
    return result.to_dict()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n hydra-mps python -m pytest tests/test_media_export.py -q --ignore=tests/test_identity_postprocess.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
make format
git add src/hydra_suite/core/post/media_export.py tests/test_media_export.py
git commit -m "feat(post): add export_final_media wrapping OrientedTrackVideoExporter (Slice 3)"
```

---

## Task 7 (3a): Active-learning dataset generation — `core/post/dataset_export.py`

Extract `DatasetGenerationWorker.execute` (`dataset_worker.py:57-203`) into a pure function with `progress`/`should_stop` callbacks. Delegates to `hydra_suite.data.dataset_generation.export_dataset` + `FrameQualityScorer`. **There is no ready-made single-function entry point in `data/` — the scoring loop lives in the worker's `execute`, so it is moved here.**

**Files:**
- Create: `src/hydra_suite/core/post/dataset_export.py`
- Test: `tests/test_dataset_export.py`
- Source: `dataset_worker.py:57-203`; `data/dataset_generation.py` (`FrameQualityScorer` ctor `:31`, `score_frame` `:81`, `get_worst_frames` `:361`, `export_dataset` `:764`), `data/detection_cache.py` (`DetectionCache`)

**Interfaces:**
- Produces: `generate_active_learning_dataset(*, video_path, csv_path, detection_cache_path, output_dir, dataset_name, class_name, params, max_frames, diversity_window, include_context, probabilistic, progress=None, should_stop=None) -> dict` — `{"success": True, "num_frames": int, "dir": str}` on success, `{"success": False, "error": str}` on failure/empty selection, `{"success": False, "cancelled": True}` on cancel.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_dataset_export.py
import pandas as pd

from hydra_suite.core.post import dataset_export


def test_generate_dataset_reports_error_on_empty_selection(tmp_path, monkeypatch):
    csv = tmp_path / "track.csv"
    pd.DataFrame({"FrameID": [0, 1], "State": ["active", "active"]}).to_csv(csv, index=False)

    class _Scorer:
        def __init__(self, params):
            pass

        def score_frame(self, frame_id, detection_data=None, tracking_data=None):
            pass

        def get_worst_frames(self, max_frames, diversity_window=30, probabilistic=True):
            return []  # nothing meets criteria

    monkeypatch.setattr(dataset_export, "FrameQualityScorer", _Scorer)
    monkeypatch.setattr(dataset_export, "export_dataset", lambda **k: "unused")

    result = dataset_export.generate_active_learning_dataset(
        video_path=str(tmp_path / "in.mp4"),
        csv_path=str(csv),
        detection_cache_path=None,
        output_dir=str(tmp_path / "out"),
        dataset_name="",
        class_name="object",
        params={},
        max_frames=5,
        diversity_window=30,
        include_context=True,
        probabilistic=False,
    )
    assert result["success"] is False
    assert "error" in result


def test_generate_dataset_success(tmp_path, monkeypatch):
    csv = tmp_path / "track.csv"
    pd.DataFrame({"FrameID": [0, 1, 2], "State": ["active"] * 3}).to_csv(csv, index=False)

    class _Scorer:
        def __init__(self, params):
            pass

        def score_frame(self, frame_id, detection_data=None, tracking_data=None):
            pass

        def get_worst_frames(self, max_frames, diversity_window=30, probabilistic=True):
            return [0, 2]

    monkeypatch.setattr(dataset_export, "FrameQualityScorer", _Scorer)
    monkeypatch.setattr(
        dataset_export, "export_dataset", lambda **k: str(tmp_path / "dataset_dir")
    )

    result = dataset_export.generate_active_learning_dataset(
        video_path=str(tmp_path / "in.mp4"),
        csv_path=str(csv),
        detection_cache_path=None,
        output_dir=str(tmp_path / "out"),
        dataset_name="",
        class_name="object",
        params={},
        max_frames=5,
        diversity_window=30,
        include_context=True,
        probabilistic=False,
    )
    assert result == {"success": True, "num_frames": 2, "dir": str(tmp_path / "dataset_dir")}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n hydra-mps python -m pytest tests/test_dataset_export.py -q --ignore=tests/test_identity_postprocess.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'hydra_suite.core.post.dataset_export'`

- [ ] **Step 3: Write minimal implementation**

Create `src/hydra_suite/core/post/dataset_export.py`. The body is `DatasetGenerationWorker.execute` (`dataset_worker.py:57-203`) with `self.progress_signal.emit(...)` → `_emit(progress, ...)`, `self.error_signal.emit(msg)` → `return {"success": False, "error": msg}`, `self.finished_signal.emit(dir, n)` → `return {"success": True, ...}`, and `self._should_stop()` → the `should_stop` callback. `FrameQualityScorer`/`export_dataset`/`DetectionCache` are imported at module top so tests can monkeypatch them:

```python
"""Qt-free active-learning dataset generation. Extracted from
trackerkit/gui/workers/dataset_worker.py (Slice 3)."""

from __future__ import annotations

import logging
import os

import pandas as pd

from hydra_suite.data.dataset_generation import FrameQualityScorer, export_dataset
from hydra_suite.data.detection_cache import DetectionCache

logger = logging.getLogger(__name__)


def _emit(progress, value, message):
    if progress is not None:
        progress(value, message)


def _stopped(should_stop) -> bool:
    return bool(should_stop is not None and should_stop())


def generate_active_learning_dataset(
    *, video_path, csv_path, detection_cache_path, output_dir, dataset_name,
    class_name, params, max_frames, diversity_window, include_context, probabilistic,
    progress=None, should_stop=None,
) -> dict:
    """Score frames and export an active-learning dataset. Pure/Qt-free."""
    detection_cache = None
    try:
        if _stopped(should_stop):
            return {"success": False, "cancelled": True}
        _emit(progress, 5, "Initializing dataset generation...")

        _emit(progress, 10, "Loading tracking data...")
        df = pd.read_csv(csv_path)

        _emit(progress, 15, "Initializing quality scorer...")
        scorer = FrameQualityScorer(params)
        if detection_cache_path and os.path.exists(detection_cache_path):
            try:
                detection_cache = DetectionCache(detection_cache_path, mode="r")
                if not detection_cache.is_compatible():
                    detection_cache.close()
                    detection_cache = None
            except Exception:
                detection_cache = None

        _emit(progress, 20, "Scoring frames...")
        unique_frames = df["FrameID"].unique()
        total_unique = len(unique_frames)

        for idx, frame_id in enumerate(unique_frames):
            if _stopped(should_stop):
                return {"success": False, "cancelled": True}
            if idx % 100 == 0:
                pct = 20 + int((idx / total_unique) * 30) if total_unique else 20
                _emit(progress, pct, f"Scoring frames ({idx}/{total_unique})...")

            frame_data = df[df["FrameID"] == frame_id]
            raw_meas, raw_shapes, raw_confidences, raw_obb_corners = [], [], [], []
            used_detection_cache = False
            if detection_cache is not None:
                try:
                    (raw_meas, _, raw_shapes, raw_confidences, raw_obb_corners, _, *_) = (
                        detection_cache.get_frame(int(frame_id))
                    )
                    used_detection_cache = True
                except Exception:
                    raw_meas, raw_shapes, raw_confidences, raw_obb_corners = [], [], [], []

            detection_count = len(raw_meas) if used_detection_cache else len(frame_data)
            detection_data = {
                "confidences": (
                    raw_confidences if raw_confidences
                    else (frame_data["DetectionConfidence"].tolist()
                          if "DetectionConfidence" in frame_data.columns else [])
                ),
                "count": detection_count,
                "measurements": raw_meas,
                "shapes": raw_shapes,
                "obb_corners": raw_obb_corners,
            }
            tracking_data = {
                "lost_tracks": int((frame_data["State"] == "lost").sum()),
                "assignment_confidences": (
                    frame_data["AssignmentConfidence"].tolist()
                    if "AssignmentConfidence" in frame_data.columns else []
                ),
                "uncertainties": (
                    frame_data["PositionUncertainty"].tolist()
                    if "PositionUncertainty" in frame_data.columns else []
                ),
            }
            scorer.score_frame(frame_id, detection_data, tracking_data)

        if _stopped(should_stop):
            return {"success": False, "cancelled": True}
        _emit(progress, 50, "Selecting challenging frames...")
        selected_frames = scorer.get_worst_frames(
            max_frames, diversity_window, probabilistic=probabilistic
        )
        if not selected_frames:
            return {"success": False, "error": "No frames met the quality criteria for export."}

        _emit(progress, 60, f"Exporting {len(selected_frames)} frames...")
        if _stopped(should_stop):
            return {"success": False, "cancelled": True}
        dataset_dir = export_dataset(
            video_path=video_path,
            csv_path=csv_path,
            frame_ids=selected_frames,
            output_dir=output_dir,
            dataset_name=dataset_name,
            class_name=class_name,
            params=params,
            include_context=include_context,
        )
        _emit(progress, 100, "Dataset generation complete!")
        return {"success": True, "num_frames": len(selected_frames), "dir": dataset_dir}
    except Exception as e:
        logger.exception("Error during dataset generation")
        return {"success": False, "error": str(e)}
    finally:
        if detection_cache is not None:
            try:
                detection_cache.close()
            except Exception:
                pass
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n hydra-mps python -m pytest tests/test_dataset_export.py -q --ignore=tests/test_identity_postprocess.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
make format
git add src/hydra_suite/core/post/dataset_export.py tests/test_dataset_export.py
git commit -m "feat(post): add Qt-free generate_active_learning_dataset (Slice 3)"
```

---

## Task 8: Wire the three export stages into `TrackingSessionCore`

Add the export chain to the Slice-2 service so `run_post_tracking` fills `SessionResult.media_paths` and `SessionResult.dataset_result`. Each stage announces `callbacks.stage_changed(name)` and is guarded by `callbacks.should_stop()`.

**Files:**
- Modify: `src/hydra_suite/core/tracking/session.py` (Slice-2 file)
- Test: `tests/test_session_export_chain.py`
- Consumes: `media_export.export_final_media`, `media_export.render_annotated_video`, `media_export.load_video_trajectories`, `dataset_export.generate_active_learning_dataset`, Slice-1 policy predicates.

**Interfaces:**
- Produces (methods on `TrackingSessionCore`): `_run_dataset_generation(self, final_csv_path) -> dict | None`; `_run_final_media_export(self, final_csv_path) -> list[str]`; `_run_annotated_video(self, final_csv_path) -> str | None`
- Slice-2 `self.paths` fields read: `detection_cache_path`, `interpolated_roi_npz_path`, `individual_dataset_dir`, `final_media_video_dir`, `source_video_fps`. **If any is absent on the Slice-2 `paths` object, add it there first** (data the GUI resolves via `_resolve_current_individual_dataset_dir` / `_resolve_current_final_media_video_dir` / `_resolve_source_video_fps`, `tracking.py:1308-1372`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_session_export_chain.py
import types

import pandas as pd

import hydra_suite.core.tracking.session as session_mod
from hydra_suite.core.tracking.session import SessionCallbacks, TrackingSessionCore


def _make_service(config, params, tmp_path):
    paths = types.SimpleNamespace(
        detection_cache_path=str(tmp_path / "cache.npz"),
        interpolated_roi_npz_path=None,
        individual_dataset_dir=tmp_path / "ds",
        final_media_video_dir=tmp_path / "vids",
        source_video_fps=30.0,
    )
    return TrackingSessionCore(
        video_path=str(tmp_path / "in.mp4"),
        config=config,
        params=params,
        paths=paths,
        callbacks=SessionCallbacks(),
    )


def test_dataset_stage_fills_dataset_result(monkeypatch, tmp_path):
    svc = _make_service(
        {"enable_dataset_generation": True, "dataset_class_name": "ant",
         "dataset_max_frames": 5, "dataset_diversity_window": 30,
         "dataset_include_context": True, "dataset_probabilistic_sampling": False},
        {}, tmp_path,
    )
    final_csv = tmp_path / "final.csv"
    final_csv.write_text("TrajectoryID,X,Y,Theta,FrameID\n0,1,2,0,0\n")
    (tmp_path / "in.mp4").write_bytes(b"x")

    monkeypatch.setattr(
        session_mod.dataset_export, "generate_active_learning_dataset",
        lambda **k: {"success": True, "num_frames": 3, "dir": "d"},
    )
    result = svc._run_dataset_generation(str(final_csv))
    assert result == {"success": True, "num_frames": 3, "dir": "d"}


def test_dataset_stage_skipped_when_disabled(tmp_path):
    svc = _make_service({"enable_dataset_generation": False}, {}, tmp_path)
    assert svc._run_dataset_generation(str(tmp_path / "final.csv")) is None


def test_annotated_video_stage_returns_path(monkeypatch, tmp_path):
    svc = _make_service(
        {"video_output_enabled": True, "video_output_path": str(tmp_path / "out.mp4")},
        {}, tmp_path,
    )
    final_csv = tmp_path / "final.csv"
    final_csv.write_text("TrajectoryID,X,Y,Theta,FrameID\n0,1,2,0,0\n")

    monkeypatch.setattr(
        session_mod.media_export, "load_video_trajectories",
        lambda p: (pd.read_csv(final_csv), str(final_csv)),
    )
    monkeypatch.setattr(
        session_mod.media_export, "render_annotated_video",
        lambda **k: k["output_path"],
    )
    out = svc._run_annotated_video(str(final_csv))
    assert out == str(tmp_path / "out.mp4")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n hydra-mps python -m pytest tests/test_session_export_chain.py -q --ignore=tests/test_identity_postprocess.py`
Expected: FAIL — `AttributeError: 'TrackingSessionCore' object has no attribute '_run_dataset_generation'`

- [ ] **Step 3: Write minimal implementation**

At the top of `core/tracking/session.py`, add these imports as module attributes so tests can monkeypatch them (ensure `import os` and a module `logger` already exist — add if missing):

```python
from hydra_suite.core.post import dataset_export, media_export
from hydra_suite.core.tracking.session_policy import (
    should_export_final_canonical_images,
    should_export_final_media_videos,
)
```

Add the three methods to `TrackingSessionCore`. Gating mirrors `tracking.py`: dataset gate `config["enable_dataset_generation"]` (`_finish_tracking_session`, `tracking.py:3491`); output dir mirrors `<video>_datasets/active_learning` (`tracking.py:4340-4344`); video gate `config["video_output_enabled"] and config["video_output_path"]` (`tracking.py:3484-3488`):

```python
    def _run_dataset_generation(self, final_csv_path):
        """Generate an active-learning dataset inline; return its result dict or None."""
        if not self.config.get("enable_dataset_generation", False):
            return None
        if self.callbacks.should_stop():
            return None
        video_path = self.video_path
        if not video_path or not os.path.exists(video_path):
            self.callbacks.warning("Dataset Generation Error", "Source video file not found.")
            return {"success": False, "error": "Source video file not found."}
        csv_path = final_csv_path
        if not csv_path or not os.path.exists(csv_path):
            self.callbacks.warning("Dataset Generation Error", "Tracking CSV file not found.")
            return {"success": False, "error": "Tracking CSV file not found."}

        output_dir = os.path.join(
            os.path.dirname(video_path),
            f"{os.path.splitext(os.path.basename(video_path))[0]}_datasets",
            "active_learning",
        )
        os.makedirs(output_dir, exist_ok=True)
        class_name = str(self.config.get("dataset_class_name", "") or "").strip() or "object"

        self.callbacks.stage_changed("dataset_generation")
        return dataset_export.generate_active_learning_dataset(
            video_path=video_path,
            csv_path=csv_path,
            detection_cache_path=self.paths.detection_cache_path,
            output_dir=output_dir,
            dataset_name="",
            class_name=class_name,
            params=self.params,
            max_frames=int(self.config.get("dataset_max_frames", 100)),
            diversity_window=int(self.config.get("dataset_diversity_window", 30)),
            include_context=bool(self.config.get("dataset_include_context", True)),
            probabilistic=bool(self.config.get("dataset_probabilistic_sampling", True)),
            progress=self.callbacks.progress,
            should_stop=self.callbacks.should_stop,
        )

    def _run_final_media_export(self, final_csv_path):
        """Export canonical stills / oriented per-track videos; return written media paths."""
        if self.callbacks.should_stop():
            return []
        export_images = should_export_final_canonical_images(self.config)
        export_videos = should_export_final_media_videos(self.config)
        if not export_images and not export_videos:
            return []
        image_root = self.paths.individual_dataset_dir if export_images else None
        video_root = self.paths.final_media_video_dir if export_videos else None

        self.callbacks.stage_changed("final_media_export")
        result = media_export.export_final_media(
            final_csv_path=final_csv_path,
            config=self.config,
            video_path=self.video_path,
            detection_cache_path=self.paths.detection_cache_path,
            interpolated_roi_npz_path=self.paths.interpolated_roi_npz_path,
            fps=self.paths.source_video_fps,
            image_root=image_root,
            video_root=video_root,
            export_images=export_images,
            export_videos=export_videos,
            padding_fraction=float(self.config.get("individual_crop_padding", 0.1)),
            background_color=tuple(self.config.get("individual_background_color", [0, 0, 0])),
            progress=self.callbacks.progress,
            should_stop=self.callbacks.should_stop,
        )
        if not result:
            return []
        media_paths = []
        for key in ("output_dir", "image_output_dir"):
            val = str(result.get(key, "")).strip()
            if val:
                media_paths.append(val)
        return media_paths

    def _run_annotated_video(self, final_csv_path):
        """Render the annotated overlay video; return its path or None."""
        if self.callbacks.should_stop():
            return None
        output_path = str(self.config.get("video_output_path", "") or "").strip()
        if not (self.config.get("video_output_enabled", False) and output_path):
            return None
        trajectories_df, loaded_path = media_export.load_video_trajectories(final_csv_path)
        if trajectories_df is None or trajectories_df.empty:
            logger.warning(
                "Skipping final video generation: no trajectories loaded from %s", final_csv_path
            )
            return None
        self.callbacks.stage_changed("annotated_video")
        return media_export.render_annotated_video(
            trajectories_df=trajectories_df,
            video_path=self.video_path,
            output_path=output_path,
            params=self.params,
            config=self.config,
            progress=self.callbacks.progress,
            should_stop=self.callbacks.should_stop,
        )
```

Finally, in `run_post_tracking` (Slice-2 body), **after the rich-export stage and immediately before constructing `SessionResult`**, insert the three stages and thread their outputs into the result. Ordering matches the GUI's `_finish_tracking_session` (`tracking.py:3469`): dataset generation before media export/cleanup:

```python
        # --- Slice 3 export chain ---
        dataset_result = None
        media_paths: list[str] = []
        if not self.callbacks.should_stop():
            dataset_result = self._run_dataset_generation(final_csv_path)
        if not self.callbacks.should_stop():
            media_paths.extend(self._run_final_media_export(final_csv_path))
        if not self.callbacks.should_stop():
            annotated = self._run_annotated_video(final_csv_path)
            if annotated:
                media_paths.append(annotated)
```

and change the result construction so `media_paths=media_paths` / `dataset_result=dataset_result` replace the Slice-2 `media_paths=[]` / `dataset_result=None`.

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n hydra-mps python -m pytest tests/test_session_export_chain.py -q --ignore=tests/test_identity_postprocess.py`
Expected: PASS

- [ ] **Step 5: Verify `core/` stays Qt-free**

Run: `grep -rnE "PySide6|QtCore|QThread|Signal|Slot|QMutex" src/hydra_suite/core/`
Expected: EMPTY.

- [ ] **Step 6: Commit**

```bash
make format
git add src/hydra_suite/core/tracking/session.py tests/test_session_export_chain.py
git commit -m "feat(session): wire media export + dataset generation into TrackingSessionCore (Slice 3)"
```

---

## Task 9: Repoint GUI orchestrator + dataset worker to core

Keep GUI behavior byte-identical while the logic lives in `core/`. The GUI's annotated-video methods and dataset worker now call the new core functions; the GUI keeps its `QThread` wrappers for responsiveness.

**Files:**
- Modify: `trackerkit/gui/orchestrators/tracking.py`
- Modify: `trackerkit/gui/workers/dataset_worker.py`
- Test: existing GUI smoke tests + the new media-parity test (Task 10)

**Interfaces:**
- Consumes: all Task 1-7 functions.

- [ ] **Step 1: Establish the GUI baseline** (find the smoke test first)

```bash
ls tests/ | grep -iE "trackerkit.*smoke|workers_smoke" || echo "no smoke test; use tests/test_trackerkit_session_plan.py as the import-safety baseline"
```

Run the identified smoke/import test:
`conda run -n hydra-mps python -m pytest tests/test_trackerkit_session_plan.py -q --ignore=tests/test_identity_postprocess.py`
Expected: PASS (baseline before change).

- [ ] **Step 2: Repoint `tracking.py`**

Add near the other `core` imports:

```python
from hydra_suite.core.post import media_export as _media_export
```

Replace the bodies of the moved orchestrator methods with delegations:

- `_scale_trajectories_to_original_space(self, df, rf)` → `return _media_export.scale_trajectories_to_original_space(df, rf)`
- `save_trajectories_to_csv(self, trajectories, output_path)` → `return _media_export.save_trajectories_to_csv(trajectories, output_path)`
- `_load_video_trajectories(self, final_csv_path)` → `return _media_export.load_video_trajectories(final_csv_path)`
- `_generate_video_from_trajectories(self, trajectories_df, csv_path=None, finalize_on_complete=True)` — keep the GUI-only progress setup and `_complete()` wiring (`tracking.py:2303-2321, 2390`), replace the open/range/draw/render block (`tracking.py:2323-2388`) with:

```python
        video_path = self._panels.setup.file_line.text()
        output_path = self._panels.postprocess.video_out_line.text()

        def _complete():
            if finalize_on_complete:
                self._finish_tracking_session(final_csv_path=csv_path)
            else:
                self._finalize_tracking_session_ui()

        params = self._mw.get_parameters_dict()
        config = self._mw.build_config_dict()  # Slice-1 pure config builder

        def _progress(pct, msg):
            self._mw.progress_bar.setValue(pct)
            QApplication.processEvents()

        _media_export.render_annotated_video(
            trajectories_df=trajectories_df,
            video_path=video_path,
            output_path=output_path,
            params=params,
            config=config,
            progress=_progress,
            should_stop=lambda: self._mw._stop_all_requested,
        )
        _complete()
```

Then delete the now-unused method bodies fully moved to `media_export`: `_format_video_track_label`, `_build_video_track_label_array`, `_normalize_video_identity_color_key`, `_build_video_track_color_key_array`, `_build_precomputed_color_palette`, `_get_video_draw_params`, `_get_pose_column_info`, `_preextract_traj_arrays`, `_draw_trail_for_track`, `_draw_single_track_on_frame`, `_render_annotated_video_frames`, `_open_video_cap_and_writer`, `_compute_video_frame_range`. First confirm no other caller in `trackerkit/` references them:

```bash
grep -rnE "_format_video_track_label|_build_video_track_label_array|_normalize_video_identity_color_key|_build_video_track_color_key_array|_build_precomputed_color_palette|_get_video_draw_params|_get_pose_column_info|_preextract_traj_arrays|_draw_trail_for_track|_draw_single_track_on_frame|_render_annotated_video_frames|_open_video_cap_and_writer|_compute_video_frame_range" src/hydra_suite/trackerkit/
```

Any remaining reference must be repointed to `_media_export.*` before deleting the method.

- [ ] **Step 3: Repoint `dataset_worker.py`**

Replace the `execute()` body with a delegation to the core function, mapping the worker fields to arguments and re-emitting through the existing signals:

```python
    def execute(self):
        from hydra_suite.core.post import dataset_export

        result = dataset_export.generate_active_learning_dataset(
            video_path=self.video_path,
            csv_path=self.csv_path,
            detection_cache_path=self.detection_cache_path,
            output_dir=self.output_dir,
            dataset_name=self.dataset_name,
            class_name=self.class_name,
            params=self.params,
            max_frames=self.max_frames,
            diversity_window=self.diversity_window,
            include_context=self.include_context,
            probabilistic=self.probabilistic,
            progress=self.progress_signal.emit,
            should_stop=self._should_stop,
        )
        if self._should_stop() or result.get("cancelled"):
            return
        if result.get("success"):
            self.finished_signal.emit(result["dir"], result["num_frames"])
        else:
            self.error_signal.emit(result.get("error", "Dataset generation failed."))
```

- [ ] **Step 4: Run the unit suite for regressions**

Run: `conda run -n hydra-mps python -m pytest tests/test_media_export.py tests/test_dataset_export.py tests/test_session_export_chain.py tests/test_dataset_generation.py tests/test_trackerkit_session_plan.py -q --ignore=tests/test_identity_postprocess.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
make format
git add src/hydra_suite/trackerkit/gui/orchestrators/tracking.py src/hydra_suite/trackerkit/gui/workers/dataset_worker.py
git commit -m "refactor(trackerkit): repoint GUI media/dataset export to core (Slice 3)"
```

---

## Task 10: Qt-free guard + media-parity + equivalence gate (FINAL)

**Files:**
- Create: `tests/test_core_qtfree_slice3.py`
- Modify: `tests/test_session_export_chain.py` (add the media-parity frame-count test)

**Interfaces:** none produced.

- [ ] **Step 1: Write the Qt-free guard test**

```python
# tests/test_core_qtfree_slice3.py
import subprocess
from pathlib import Path


def test_core_has_no_qt_imports():
    root = Path(__file__).resolve().parents[1] / "src" / "hydra_suite" / "core"
    proc = subprocess.run(
        ["grep", "-rnE", "PySide6|QtCore|QThread|Signal|Slot|QMutex", str(root)],
        capture_output=True, text=True,
    )
    # grep exit code 1 == no matches (the required state).
    assert proc.returncode == 1, f"Qt tokens found in core/:\n{proc.stdout}"


def test_media_and_dataset_export_import_clean():
    import hydra_suite.core.post.media_export  # noqa: F401
    import hydra_suite.core.post.dataset_export  # noqa: F401
```

- [ ] **Step 2: Write the media-parity frame-count test** — append to `tests/test_session_export_chain.py`

```python
import cv2
import numpy as np

from hydra_suite.core.post import media_export as _mx


def _write_black_clip(path, n_frames=12, w=64, h=48, fps=12.0):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for _ in range(n_frames):
        vw.write(np.zeros((h, w, 3), dtype=np.uint8))
    vw.release()


def test_media_parity_frame_count_matches_input(tmp_path):
    """Media output is not covered by the CSV harness — assert the rendered
    video exists, is non-empty, and has the same frame count as the input clip."""
    src = tmp_path / "in.mp4"
    out = tmp_path / "out.mp4"
    _write_black_clip(src, n_frames=12)
    df = pd.DataFrame(
        {"TrajectoryID": [0] * 12, "FrameID": list(range(12)),
         "X": [30.0] * 12, "Y": [24.0] * 12, "Theta": [0.0] * 12}
    )
    result = _mx.render_annotated_video(
        trajectories_df=df, video_path=str(src), output_path=str(out),
        params={"TRAJECTORY_COLORS": [(0, 255, 0)], "REFERENCE_BODY_SIZE": 10.0,
                "ADVANCED_CONFIG": {}, "POSE_MIN_KPT_CONF_VALID": 0.2,
                "START_FRAME": 0, "END_FRAME": None},
        config={"video_show_labels": True, "video_show_orientation": True,
                "video_show_trails": False, "video_trail_duration": 1.0,
                "video_marker_size": 0.3, "video_text_scale": 0.5, "video_arrow_length": 0.7},
    )
    assert result == str(out)
    assert out.stat().st_size > 0
    cap = cv2.VideoCapture(str(out))
    n_out = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    assert n_out == 12
```

- [ ] **Step 3: Run the guard + parity tests**

Run: `conda run -n hydra-mps python -m pytest tests/test_core_qtfree_slice3.py tests/test_session_export_chain.py::test_media_parity_frame_count_matches_input -q --ignore=tests/test_identity_postprocess.py`
Expected: PASS. If the guard fails, `grep` shows the offending file — remove the Qt import (it does not belong in `core/`).

- [ ] **Step 4: Commit the guard + parity tests**

```bash
git add tests/test_core_qtfree_slice3.py tests/test_session_export_chain.py
git commit -m "test(core): Qt-free guard + media-parity frame-count for Slice 3"
```

- [ ] **Step 5: Equivalence gate — MPS (this box), BEFORE-vs-AFTER, all 7 clips**

Conda **must** be active (pose/SLEAP clips otherwise emit empty CSVs that falsely pass).

```bash
conda activate hydra-mps
bash tools/equivalence/fixtures/fetch_fixtures.sh          # once per machine
git fetch origin --tags
git worktree add --detach .worktrees/equiv-legacy legacy/main
REPO=$PWD WT=$PWD \
  MAIN_SRC=$PWD/.worktrees/equiv-legacy/src WT_SRC=$PWD/src \
  OUT=/tmp/equiv_slice3 RUNTIME=mps \
  bash tools/equivalence/run_matrix.sh
```

Verify for every clip (`emi_obb_identity`, `ant_pose_headtail`, `ant_obb_sleap`, `ant_obb_sequential`, `worm_bgsub`, `ant_cnn_identity`, `fly_obb`): EQUIVALENCE positions p99 ≈ 0, θ max ≈ 0, identical row counts, 0 unmatched, on both `_forward.csv` and `_tracking_final.csv`. Only accepted noise: bistable head/tail π-flips on head/tail clips. Confirm CSV row counts > 1 (`wc -l`) before trusting an EQUIVALENT.

- [ ] **Step 6: Media-parity check — GUI vs service, per clip (MPS)**

The CSV harness does not compare video/media. For each clip that produces media (`ant_obb_sleap`, `ant_pose_headtail`, `ant_cnn_identity`, `emi_obb_identity` with video/dataset output enabled in their fixture configs), run the pipeline once through the GUI orchestrator and once through the service and assert:
1. the **same set** of output media files exists (annotated video path + oriented-video/image dirs + dataset dir),
2. each rendered video is **non-empty** (`st_size > 0`),
3. the annotated video's frame count equals the input clip's tracked frame range (`compute_video_frame_range`).

Record the file-set diff and frame counts in the run log. Any mismatch is a Slice-3 regression — do not proceed.

- [ ] **Step 7: Cleanup MPS worktree**

```bash
git worktree remove --force .worktrees/equiv-legacy && git worktree prune
```

- [ ] **Step 8: Equivalence gate + media-parity — CUDA (mehek)**

```bash
ssh rutalab@mehek.taild08eb9.ts.net
cd ~/hydra-suite && git fetch origin --tags && git checkout <this-slice-sha>
source ~/mambaforge/etc/profile.d/conda.sh && conda activate hydra-cuda
bash tools/equivalence/fixtures/fetch_fixtures.sh          # once
git worktree add --detach .worktrees/equiv-legacy legacy/main
REPO=$PWD WT=$PWD MAIN_SRC=$PWD/.worktrees/equiv-legacy/src WT_SRC=$PWD/src \
  OUT=/tmp/equiv_slice3 RUNTIME=cuda nohup bash tools/equivalence/run_matrix.sh > /tmp/equiv_cuda_slice3.log 2>&1 &
```

Same acceptance as Step 5 (+ Step 6 media-parity) on CUDA. Then clean up the worktree as in Step 7. Both platforms must pass before Slice 3 is complete.

- [ ] **Step 9: Final `core/` Qt-free assertion**

Run: `grep -rnE "PySide6|QtCore|QThread|Signal|Slot|QMutex" src/hydra_suite/core/`
Expected: EMPTY. Slice 3 complete — the service now does everything the bridge does for export.

---

## Self-Review

**Spec coverage** (SCOPE + spec §"Slice 3 — Export chain", component-inventory rows "Final media export", "Dataset generation"):
- Final media export methods (`_generate_final_media_export`, `_start_pending_final_media_export`, `_on_final_media_export_*`, `_get_video_draw_params`, `_get_pose_column_info`, `_preextract_traj_arrays`, label/color builders, `_draw_*`, `_render_annotated_video_frames`, `_open_video_cap_and_writer`, `_compute_video_frame_range`, `_generate_video_from_trajectories`, `_load_video_trajectories`, `_run_pending_video_generation_or_finalize`, `_scale_trajectories_to_original_space`, `save_trajectories_to_csv`) → Tasks 1, 2, 4, 5, 6, 8, 9. The pure render logic of `_render_annotated_video_frames`/`_run_pending_video_generation_or_finalize` lands in `render_annotated_video`/`load_video_trajectories`; the GUI's finalize/pending-flag bookkeeping stays in the GUI (Task 9), per spec §"What stays in the GUI". The `_start_pending_final_media_export`/`_on_final_media_export_*` worker-lifecycle callbacks are GUI-only orchestration; their payload (the exporter call) moves to `export_final_media` (Task 6), and the service runs it inline (Task 8).
- Dataset generation (`_generate_training_dataset`, `on_dataset_finished/progress/error`) → Tasks 7, 8, 9; runs **inline** in the service (no worker), delegating to `data/dataset_generation` — the frame-scoring loop is extracted from `DatasetGenerationWorker.execute` (there is no library single-function entry point).
- `SessionResult.media_paths` / `dataset_result` filled → Task 8.
- 3a (crops + dataset) / 3b (annotated video) cleanly separable → 3b = Tasks 1-5, 3a = Tasks 6-7; both wired in Task 8; the split is documented so 3b's tasks ship independently.
- Color unification consumed from Slice 1 + pinned by test → Task 3.
- Cooperative `should_stop()` in long loops + cap/writer release + partial-file delete on cancel → Task 5 (video, with cancel test), Task 7 (dataset scoring loop).
- `QMessageBox.information/.warning` → `callbacks.warning`; `QMessageBox.critical` → `TrackingSessionError` → Task 8 (`_run_dataset_generation` uses `callbacks.warning`; the exception path bubbles as the service's fatal handling from Slice 2).
- `stage_changed(name)` between stages → Task 8.
- Qt-free core guard + equivalence gate + media-parity → Task 10.

**Placeholder scan:** no "TBD"/"similar to above"/"handle edge cases". Every code step contains real, transcribed source. The one deliberately-abstracted dependency (`self.paths.*` fields from Slice 2) is named explicitly with a fallback instruction in Task 8. Unlike an earlier draft, the dataset stage calls the **verified** `export_dataset` + `FrameQualityScorer` loop, not a nonexistent `data.dataset_generation.generate_active_learning_dataset` single function; and `build_pose_keypoint_labels` is imported from its real home `core.identity.properties.export`, not `utils.pose_visualization`.

**Type consistency:** `render_annotated_video` returns `str | None` (consumed by `_run_annotated_video`); `export_final_media` returns `dict | None` (keys `output_dir`/`image_output_dir` consumed in `_run_final_media_export`); `generate_active_learning_dataset` returns `{"success", "num_frames", "dir"}` / `{"success": False, "error"}` / `{"success": False, "cancelled": True}` (consumed identically in `_run_dataset_generation` and `dataset_worker.execute`); `render_annotated_video_frames` returns `bool` (consumed by `render_annotated_video`). Function names match across producer/consumer tasks.

---

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-07-31-headless-qt-free-slice3-export-chain.md`. Two execution options:**

1. **Subagent-Driven (recommended)** — one fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
