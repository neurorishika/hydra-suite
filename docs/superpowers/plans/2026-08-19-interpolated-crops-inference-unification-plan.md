# Interpolated-Crop Inference Unification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Route `core/post/interpolated_crops.py`'s pose/CNN/AprilTag/head-tail
inference on occlusion-fill crops through the same `Pipeline` stage functions
real detections use (`extract_canonical_crops_batch`/`run_pose_batch`,
`run_cnn_batch`, `run_headtail_batch`, `extract_aabb_crops`/`run_apriltag`),
fix the geometry-sourcing/provenance/trigger bugs described in the spec, and
delete the confirmed-dead `tag_identity.py` code.

**Architecture:** A new `core/post/synthetic_detections.py` builds an
`OBBResult` from one frame's interpolated-gap tasks (mirroring real OBB
detections) and pre-filters degenerate OBBs (the stage layer doesn't). A
reworked `interpolated_crops.py` builds an `InferenceConfig` from the same
`params` dict via the *existing* `build_inference_config_from_params`, loads
only the four downstream-stage models directly (no OBB/bgsub), and replaces
its hand-rolled batching with calls into the same `*_batch` functions
`Pipeline` calls. Provenance moves from two incompatible conventions to one
(`coalesce + *Source` column) across all four signal types.

**Tech Stack:** Python, pandas, numpy, torch (existing `core/inference`
stage-function stack); no new dependencies.

**Spec:** `docs/superpowers/specs/2026-08-19-interpolated-crops-inference-unification-design.md`

## Global Constraints

- No changes to `core/inference/runner.py`, `Pipeline`, or any cache-write
  logic (spec "Out of scope").
- No changes to `run_realtime` (pre-existing, deliberately separate; out of
  scope).
- `APRILTAG_CROP_PADDING` and `cfg.apriltag.crop_padding` are already the
  SAME params key (`config.py:1074`, verified during planning: both read
  `params.get("APRILTAG_CROP_PADDING", 0.0)`) — no mismatch to reconcile,
  the spec's G8/G9 "confirm during planning" item is resolved as a
  non-issue.
- Breaking CSV schema change: `Interp*` columns are retired; `*Source`
  columns are new. No backward-compatible aliasing (spec "CSV
  schema-compatibility stance").
- All new pure functions get unit tests before being wired in (TDD); the
  wiring/replacement steps are verified via the existing test suite plus the
  characterization golden (Task 13).

## Plan-level deviation from the spec text (flag for review)

The spec's G4 decision says the adapter calls `runner.py`'s private
`_load_all_models` directly. Investigating the actual signature during
planning found a better option that still satisfies G4's real intent ("don't
reinvent bundle-loading orchestration"): `_load_all_models` *always* loads an
OBB (or bgsub) detector too, because `InferenceConfig.__post_init__` requires
`obb` xor `bgsub` to be set and `build_inference_config_from_params` always
sets `obb`. Calling `_load_all_models` would therefore load a real OBB model
the interpolated-crop pass never uses — wasted time/memory, and a reason to
reach into a private function at all.

The four downstream-stage loaders it calls are **public**, not private:
`stages.headtail.load_headtail_model`, `stages.cnn.load_cnn_model`,
`stages.pose.load_pose_model`, `stages.apriltag.load_apriltag_model`. Task 9
below calls these four directly — same "reuse existing loader code, don't
reinvent it" intent as G4, without the private import and without the
unwanted OBB load. `runner.py` is untouched either way, so this does not
violate "no changes to runner.py". Flagging this explicitly since it departs
from the spec's literal text; if the user prefers the literal `_load_all_models`
call, swap the loader calls in Task 9 for `_load_all_models` (accepting the
extra OBB-model load).

---

## Task 1: `count_by_source` helper + rich-export summary wiring

**Files:**
- Modify: `src/hydra_suite/core/post/rich_export.py`
- Test: `tests/test_rich_export.py` (create if it doesn't exist — check
  first with `ls tests/test_rich_export*.py`)

**Interfaces:**
- Produces: `count_by_source(df: pd.DataFrame, source_col: str) -> dict` with
  keys `"real"` and `"interp"`, both `int`. Used by Task 3's
  `log_rich_export_summary` rewrite and by any test asserting fill counts.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_rich_export.py
import pandas as pd

from hydra_suite.core.post.rich_export import count_by_source


def test_count_by_source_counts_real_and_interp():
    df = pd.DataFrame(
        {
            "TagSource": ["real", "real", "interp", None, "interp"],
        }
    )
    assert count_by_source(df, "TagSource") == {"real": 2, "interp": 2}


def test_count_by_source_missing_column_returns_zeros():
    df = pd.DataFrame({"Other": [1, 2, 3]})
    assert count_by_source(df, "TagSource") == {"real": 0, "interp": 0}


def test_count_by_source_empty_df():
    df = pd.DataFrame({"TagSource": []})
    assert count_by_source(df, "TagSource") == {"real": 0, "interp": 0}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_rich_export.py -v -k count_by_source`
Expected: FAIL with `ImportError: cannot import name 'count_by_source'`

- [ ] **Step 3: Implement `count_by_source`, replacing the two dead counters**

In `src/hydra_suite/core/post/rich_export.py`, delete
`count_augmented_pose_rows` (lines 84-94) and `count_interpolated_cnn_rows`
(lines 97-116) — both confirmed zero callers by the spec's dead-code audit —
and add:

```python
def count_by_source(df: pd.DataFrame, source_col: str) -> dict:
    """Real-vs-interpolated row counts for one ``*Source`` provenance column.

    Replaces the dead ``count_augmented_pose_rows``/``count_interpolated_cnn_rows``
    (zero callers) with one generic counter used uniformly for all four
    signal types (Pose/CNN/AprilTag/head-tail) now that they share the same
    coalesce-into-original-columns + explicit ``*Source`` provenance
    convention (design spec, "Provenance").
    """
    if source_col not in df.columns:
        return {"real": 0, "interp": 0}
    counts = df[source_col].value_counts()
    return {
        "real": int(counts.get("real", 0)),
        "interp": int(counts.get("interp", 0)),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_rich_export.py -v -k count_by_source`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/core/post/rich_export.py tests/test_rich_export.py
git commit -m "feat(post): add count_by_source, retire dead pose/cnn row counters"
```

---

## Task 2: Provenance — rewrite the four merge functions to coalesce + `*Source`

**Files:**
- Modify: `src/hydra_suite/core/individual/properties/export.py`
- Test: `tests/test_properties_export.py`

**Interfaces:**
- Consumes: nothing new — same four function signatures as today
  (`merge_interpolated_pose_df`, `merge_interpolated_apriltag_df`,
  `merge_interpolated_cnn_df`, `merge_interpolated_headtail_df`), same input
  DataFrame shapes interpolated_crops.py's artifact CSVs already produce
  (`frame_id`/`trajectory_id` + signal-specific columns — unchanged by this
  task; Task 10-12 don't change these interp CSV schemas).
- Produces: for AprilTag, coalesced `DetectedTagID` (unchanged real column,
  now also filled from interp) + new `TagSource` column (`"real"` | `"interp"`
  | `NaN`). For head-tail: coalesced `HeadTailAngleRad` /
  `HeadTailClassifierConf`, and `HeadingResolved`/`HeadingMethod`/
  `HeadingIsDirected` filled-if-NaN with `HeadingMethod="headtail_interp"`,
  plus new `HeadingSource` column. For CNN: `CNN_<label>_Source` (existing
  coalesce-into-original-columns behavior for `CNN_<label>_Class`/`_Conf` is
  unchanged — only the new `_Source` column is added). For pose:
  `PoseSource` (existing coalesce behavior for `PoseKpt_*`/`Pose*` summary
  columns is unchanged — only the new `PoseSource` column is added).

This task changes tests, not behavior of pose/CNN backfill (those already
coalesce) — it only *adds* `*Source` columns there. AprilTag/head-tail get
both the coalesce-target change (interp values move into the real-detection
columns) and the new `*Source` column.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_properties_export.py` (near the existing
`merge_interpolated_*` tests, e.g. after line 194's pose test and line 316's
CNN test):

```python
def test_merge_interpolated_pose_sets_pose_source():
    trajectories = pd.DataFrame(
        {
            "FrameID": [0, 1, 2],
            "TrajectoryID": [1, 1, 1],
            "PoseKpt_head_X": [10.0, np.nan, np.nan],
            "PoseKpt_head_Y": [20.0, np.nan, np.nan],
        }
    )
    interp_pose = pd.DataFrame(
        {
            "frame_id": [1],
            "trajectory_id": [1],
            "PoseKpt_head_X": [11.0],
            "PoseKpt_head_Y": [21.0],
        }
    )
    out = merge_interpolated_pose_df(trajectories, interp_pose)
    assert list(out["PoseSource"]) == ["real", "interp", np.nan] or (
        out["PoseSource"].tolist()[0] == "real"
        and out["PoseSource"].tolist()[1] == "interp"
        and pd.isna(out["PoseSource"].tolist()[2])
    )


def test_merge_interpolated_apriltag_coalesces_into_detected_tag_id():
    trajectories = pd.DataFrame(
        {
            "FrameID": [0, 1, 2],
            "TrajectoryID": [1, 1, 1],
            "DetectedTagID": [5, np.nan, np.nan],
        }
    )
    interp_tags = pd.DataFrame(
        {"frame_id": [1], "trajectory_id": [1], "tag_id": [7]}
    )
    out = merge_interpolated_apriltag_df(trajectories, interp_tags)
    assert "InterpTagID" not in out.columns
    assert out["DetectedTagID"].tolist() == [5, 7, None] or (
        out["DetectedTagID"].iloc[0] == 5
        and out["DetectedTagID"].iloc[1] == 7
        and pd.isna(out["DetectedTagID"].iloc[2])
    )
    assert out["TagSource"].iloc[0] == "real"
    assert out["TagSource"].iloc[1] == "interp"
    assert pd.isna(out["TagSource"].iloc[2])


def test_merge_interpolated_headtail_coalesces_and_sets_heading_method():
    trajectories = pd.DataFrame(
        {
            "FrameID": [0, 1, 2],
            "TrajectoryID": [1, 1, 1],
            "HeadTailAngleRad": [1.0, np.nan, np.nan],
            "HeadingResolved": [1.0, np.nan, np.nan],
            "HeadingMethod": ["headtail", np.nan, np.nan],
            "HeadingIsDirected": [True, np.nan, np.nan],
        }
    )
    interp_ht = pd.DataFrame(
        {
            "frame_id": [1],
            "trajectory_id": [1],
            "heading_rad": [2.0],
            "heading_conf": [0.9],
            "heading_directed": [1],
        }
    )
    out = merge_interpolated_headtail_df(trajectories, interp_ht)
    assert "InterpHeadingRad" not in out.columns
    assert out["HeadTailAngleRad"].iloc[1] == 2.0
    assert out["HeadingResolved"].iloc[1] == 2.0
    assert out["HeadingMethod"].iloc[1] == "headtail_interp"
    assert bool(out["HeadingIsDirected"].iloc[1]) is True
    assert out["HeadingSource"].iloc[0] == "real"
    assert out["HeadingSource"].iloc[1] == "interp"
    # row 2 (frame 2) has neither real nor interp heading -> untouched
    assert pd.isna(out["HeadingMethod"].iloc[2])


def test_merge_interpolated_cnn_sets_cnn_source():
    trajectories = pd.DataFrame(
        {
            "FrameID": [0, 1],
            "TrajectoryID": [1, 1],
            "CNN_idA_Class": ["antA", np.nan],
            "CNN_idA_Conf": [0.9, np.nan],
        }
    )
    interp_cnn = pd.DataFrame(
        {
            "frame_id": [1],
            "trajectory_id": [1],
            "class_name": ["antB"],
            "confidence": [0.7],
        }
    )
    out = merge_interpolated_cnn_df(trajectories, interp_cnn, label="idA")
    assert out["CNN_idA_Source"].iloc[0] == "real"
    assert out["CNN_idA_Source"].iloc[1] == "interp"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_properties_export.py -v -k "pose_source or apriltag_coalesces or headtail_coalesces or cnn_sets_cnn_source"`
Expected: FAIL (missing `TagSource`/`HeadingSource`/`PoseSource`/`CNN_idA_Source`
columns, and `InterpTagID`/`InterpHeadingRad` still present)

- [ ] **Step 3: Rewrite the four merge functions**

In `src/hydra_suite/core/individual/properties/export.py`:

Replace `merge_interpolated_pose_df` (keep its existing coalesce logic
exactly — it already coalesces into original `Pose*` columns) by adding a
`PoseSource` column. After the existing `merged[col] = merged[col].where(...)`
loop (right before the `merged.drop(columns=[...])` call), insert:

```python
    # PoseSource: "real" where the row already had at least one non-NaN
    # PoseKpt_* value before this merge, "interp" where this merge filled it,
    # NaN where neither a real nor an interpolated pose exists for the row.
    had_real = out[pose_cols_interp].notna().any(axis=1)
    filled_by_interp = merged[pose_cols_interp].notna().any(axis=1) & ~had_real
    merged["PoseSource"] = pd.Series(np.nan, index=merged.index, dtype=object)
    merged.loc[had_real, "PoseSource"] = "real"
    merged.loc[filled_by_interp, "PoseSource"] = "interp"
```

Replace `APRILTAG_INTERP_COLUMNS` and `merge_interpolated_apriltag_df`
(lines 1149-1201-ish) with:

```python
def merge_interpolated_apriltag_df(
    trajectories_df: pd.DataFrame,
    interp_tag_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """Merge interpolated AprilTag observations into final trajectories.

    Coalesces into the real-detection ``DetectedTagID`` column (design spec,
    "Provenance"): a row's tag id comes from a real detection if present,
    otherwise from interpolation. ``TagSource`` records which, so
    ``DetectionID``-absence is no longer the only signal for "was this
    interpolated". Retires the separate ``InterpTagID``/``InterpTagHamming``/
    ``InterpTagConf`` columns entirely -- no backward-compatible aliasing.
    """
    if trajectories_df is None or trajectories_df.empty:
        return trajectories_df
    if not _can_merge_interp(
        trajectories_df, interp_tag_df, {"frame_id", "trajectory_id", "tag_id"}
    ):
        out = trajectories_df.copy()
        if "TagSource" not in out.columns:
            out["TagSource"] = np.nan
        return out

    out, interp = _prepare_interp_join_keys(trajectories_df, interp_tag_df)
    if "DetectedTagID" not in out.columns:
        out["DetectedTagID"] = np.nan
    had_real = out["DetectedTagID"].notna()

    interp_lookup = interp[
        ["_frame_join", "_traj_join", "tag_id"]
    ].drop_duplicates(subset=["_frame_join", "_traj_join"], keep="first")

    merged = out.merge(
        interp_lookup,
        how="left",
        on=["_frame_join", "_traj_join"],
        suffixes=("", "_itag"),
        sort=False,
    )
    merged["DetectedTagID"] = merged["DetectedTagID"].where(
        merged["DetectedTagID"].notna(), merged["tag_id"]
    )
    filled_by_interp = merged["tag_id"].notna() & ~had_real
    merged["TagSource"] = pd.Series(np.nan, index=merged.index, dtype=object)
    merged.loc[had_real.values, "TagSource"] = "real"
    merged.loc[filled_by_interp.values, "TagSource"] = "interp"

    merged.drop(
        columns=["_frame_join", "_traj_join", "tag_id"],
        inplace=True,
        errors="ignore",
    )
    return merged
```

Replace `HEADTAIL_INTERP_COLUMNS` and `merge_interpolated_headtail_df` with:

```python
def merge_interpolated_headtail_df(
    trajectories_df: pd.DataFrame,
    interp_ht_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """Merge interpolated head-tail direction into final trajectories.

    Coalesces into the real-detection ``HeadTailAngleRad`` /
    ``HeadTailClassifierConf`` columns (the classifier's own raw output),
    and additionally backfills ``HeadingResolved``/``HeadingIsDirected``
    (only where those were NaN -- never overwriting an already-resolved
    heading from another source, e.g. pose or velocity) with
    ``HeadingMethod="headtail_interp"`` so the existing 4-way
    ``HeadingMethod`` vocabulary (``"headtail"``/``"pose"``/``"velocity"``/
    ``"default"``) can represent an interpolated head-tail result without a
    parallel bool. Retires the separate ``InterpHeadingRad``/
    ``InterpHeadingConf``/``InterpHeadingDirected`` columns entirely.
    """
    if trajectories_df is None or trajectories_df.empty:
        return trajectories_df
    if not _can_merge_interp(
        trajectories_df, interp_ht_df, {"frame_id", "trajectory_id", "heading_rad"}
    ):
        out = trajectories_df.copy()
        if "HeadingSource" not in out.columns:
            out["HeadingSource"] = np.nan
        return out

    out, interp = _prepare_interp_join_keys(trajectories_df, interp_ht_df)
    for col in ("HeadTailAngleRad", "HeadTailClassifierConf", "HeadingResolved",
                "HeadingMethod", "HeadingIsDirected"):
        if col not in out.columns:
            out[col] = np.nan
    had_real = out["HeadTailAngleRad"].notna()
    resolved_was_nan = out["HeadingResolved"].isna()

    ht_cols = ["heading_rad"]
    if "heading_conf" in interp.columns:
        ht_cols.append("heading_conf")
    if "heading_directed" in interp.columns:
        ht_cols.append("heading_directed")
    interp_lookup = interp[["_frame_join", "_traj_join", *ht_cols]].drop_duplicates(
        subset=["_frame_join", "_traj_join"], keep="first"
    )

    merged = out.merge(
        interp_lookup,
        how="left",
        on=["_frame_join", "_traj_join"],
        suffixes=("", "_iht"),
        sort=False,
    )
    has_interp = merged["heading_rad"].notna()

    merged["HeadTailAngleRad"] = merged["HeadTailAngleRad"].where(
        merged["HeadTailAngleRad"].notna(), merged["heading_rad"]
    )
    if "heading_conf" in merged.columns:
        merged["HeadTailClassifierConf"] = merged["HeadTailClassifierConf"].where(
            merged["HeadTailClassifierConf"].notna(), merged["heading_conf"]
        )

    backfill_resolved = has_interp.values & resolved_was_nan.values
    merged.loc[backfill_resolved, "HeadingResolved"] = merged.loc[
        backfill_resolved, "heading_rad"
    ]
    merged.loc[backfill_resolved, "HeadingMethod"] = "headtail_interp"
    if "heading_directed" in merged.columns:
        merged.loc[backfill_resolved, "HeadingIsDirected"] = merged.loc[
            backfill_resolved, "heading_directed"
        ].astype(bool)

    merged["HeadingSource"] = pd.Series(np.nan, index=merged.index, dtype=object)
    merged.loc[had_real.values, "HeadingSource"] = "real"
    merged.loc[has_interp.values & ~had_real.values, "HeadingSource"] = "interp"

    merged.drop(
        columns=["_frame_join", "_traj_join", "heading_rad", "heading_conf",
                  "heading_directed"],
        inplace=True,
        errors="ignore",
    )
    return merged
```

In `merge_interpolated_cnn_df`, keep the existing coalesce logic unchanged
and add a `CNN_<label>_Source` column. After the `merged = _backfill_interp_columns(merged, column_map)` call, insert:

```python
    source_col = f"CNN_{label}_Source"
    class_col_for_source = output_cols[0]  # CNN_<label>_Class (flat) or first wide col
    had_real = out[class_col_for_source].notna() if not uses_wide_columns else out[
        output_cols
    ].notna().any(axis=1)
    now_present = merged[class_col_for_source].notna() if not uses_wide_columns else (
        merged[output_cols].notna().any(axis=1)
    )
    filled_by_interp = now_present & ~had_real
    merged[source_col] = pd.Series(np.nan, index=merged.index, dtype=object)
    merged.loc[had_real.values, source_col] = "real"
    merged.loc[filled_by_interp.values, source_col] = "interp"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_properties_export.py -v`
Expected: PASS (all existing + 4 new tests)

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/core/individual/properties/export.py tests/test_properties_export.py
git commit -m "feat(export): unify interpolated-signal provenance into coalesce + *Source"
```

---

## Task 3: Consumer migration — `postprocess_df.py`, `rich_export.py`, existing fixtures

**Files:**
- Modify: `src/hydra_suite/core/individual/postprocess_df.py:288-290`
- Modify: `src/hydra_suite/core/post/rich_export.py` (the `InterpTagID`/
  `InterpHeadingRad` reads in `log_rich_export_summary`, ~lines 176-189)
- Modify: `tests/test_identity_evidence_vectorized.py` (drop the now-unused
  `InterpTagID` fixture column)
- Test: existing tests in both modified test files (no new tests needed —
  this task removes a now-dead column reference)

**Interfaces:**
- Consumes: `count_by_source` from Task 1.

- [ ] **Step 1: Update `postprocess_df.py`'s `has_apriltag` computation**

Read the current block first:

```bash
sed -n '280,295p' src/hydra_suite/core/individual/postprocess_df.py
```

Replace:

```python
            tag_id = _column_or_nan(frame, "DetectedTagID")
            interp_id = _column_or_nan(frame, "InterpTagID")
            has_apriltag = tag_id.notna() | interp_id.notna()
```

with:

```python
            # InterpTagID retired (design spec, "Provenance"): interpolated
            # tag ids now coalesce directly into DetectedTagID, so its
            # presence alone is authoritative. TagSource (if present) carries
            # the real-vs-interp distinction for any downstream consumer that
            # still needs it.
            tag_id = _column_or_nan(frame, "DetectedTagID")
            has_apriltag = tag_id.notna()
```

- [ ] **Step 2: Update `rich_export.py`'s summary reads**

Read the current block:

```bash
sed -n '170,195p' src/hydra_suite/core/post/rich_export.py
```

Replace the `--- interpolated AprilTag ---` and `--- interpolated head-tail ---`
blocks (which read `InterpTagID`/`InterpHeadingRad` via `fill(...)`) with:

```python
    # --- AprilTag: real vs interpolated ---
    if "TagSource" in df.columns:
        counts = count_by_source(df, "TagSource")
        if counts["interp"]:
            lines.append(
                f"  AprilTag (interpolated)  : {counts['interp']:>6,} / {total:,}  "
                f"({pct(counts['interp'])})"
            )

    # --- head-tail: real vs interpolated ---
    if "HeadingSource" in df.columns:
        counts = count_by_source(df, "HeadingSource")
        if counts["interp"]:
            lines.append(
                f"  Head-tail (interpolated) : {counts['interp']:>6,} / {total:,}  "
                f"({pct(counts['interp'])})"
            )

    # --- CNN: real vs interpolated, per label ---
    for lbl in cnn_labels:
        source_col = f"CNN_{lbl}_Source"
        if source_col in df.columns:
            counts = count_by_source(df, source_col)
            if counts["interp"]:
                lines.append(
                    f"  CNN [{lbl}] (interpolated): {counts['interp']:>6,} / {total:,}  "
                    f"({pct(counts['interp'])})"
                )
```

This also closes the spec's Motivation complaint ("the rich-export summary
never reports a CNN split at all") since CNN now gets an interpolated-count
line alongside pose/AprilTag/head-tail.

- [ ] **Step 3: Drop the now-unused `InterpTagID` fixture column**

In `tests/test_identity_evidence_vectorized.py`, remove the
`"InterpTagID": [nan] * 14,` line (~line 70) from `_build_multi_branch_df`.
It was an all-NaN column that never affected the golden output (the fixture
snapshot doesn't reference it directly), so removing it doesn't change the
test's expected values — only run the test to confirm.

- [ ] **Step 4: Run both test files**

Run: `python -m pytest tests/test_identity_evidence_vectorized.py tests/test_postprocess_df.py tests/test_rich_export.py -v`
(check `tests/test_postprocess_df.py` exists first with
`ls tests/test_postprocess_df*.py`; if the module has a different test
filename, grep for `has_apriltag` under `tests/` and use that file instead)
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/core/individual/postprocess_df.py src/hydra_suite/core/post/rich_export.py tests/test_identity_evidence_vectorized.py
git commit -m "refactor: migrate Interp* column readers to coalesce + *Source"
```

---

## Task 4: Delete dead `tag_identity.py` code + its test/whitelist/docs fallout

**Files:**
- Modify: `src/hydra_suite/core/post/tag_identity.py`
- Modify: `tests/test_tag_identity.py`
- Modify: `pyproject.toml:255`
- Modify: `docs/schematics/trackerkit_pipeline.md` (lines ~632, ~1214)

**Interfaces:** none (pure deletion).

- [ ] **Step 1: Delete `build_tag_only_trajectories` and `_interpolate_segment_rows`**

```bash
grep -n "^def build_tag_only_trajectories\|^def _interpolate_segment_rows\|^def [a-z_]*(" src/hydra_suite/core/post/tag_identity.py
```

Confirm the two functions' exact line ranges (they start at lines 95 and 332
per planning-time grep; re-verify since other edits may have shifted them),
then delete both function bodies in full, including their docstrings and any
now-unused helper functions that only `_interpolate_segment_rows` or
`build_tag_only_trajectories` called (check with
`grep -n "_interpolate_segment_rows(" src/hydra_suite/core/post/tag_identity.py`
to confirm no other caller inside the file uses it before deleting; if it's
called only from `build_tag_only_trajectories`, delete both together).

- [ ] **Step 2: Delete the two test functions**

In `tests/test_tag_identity.py`, delete `test_build_tag_only_trajectories_basic`
and `test_build_tag_only_trajectories_no_cache` (confirmed at lines 159 and
184 during planning), the `FakeTagCache` helper class *only if* nothing else
in the file uses it (grep `FakeTagCache` in the file first), and the
`build_tag_only_trajectories = mod.build_tag_only_trajectories` import line
(line 16).

- [ ] **Step 3: Remove the `pyproject.toml` whitelist entry**

In `pyproject.toml`'s `[tool.deadcode]` `ignore-names` list, delete the line
`"build_tag_only_trajectories",` (confirmed at line 255 during planning).

- [ ] **Step 4: Update the docs reference**

In `docs/schematics/trackerkit_pipeline.md`, the line at ~632
(`- **\`build_tag_only_trajectories\`** (\`tag_identity.py:332\`) is defined but ...`)
and the note at ~1214 (`> \`build_tag_only_trajectories\` (\`tag_identity.py:332\`) is dead code with ...`)
both describe now-deleted code. Replace both with a one-line note that the
function was removed in this change (or delete the lines outright if the
surrounding doc structure allows it without leaving a dangling reference —
read the surrounding context first with
`sed -n '620,640p;1205,1220p' docs/schematics/trackerkit_pipeline.md`).

- [ ] **Step 5: Verify no other callers exist and tests/lint pass**

Run:
```bash
grep -rn "build_tag_only_trajectories\|_interpolate_segment_rows" src/ tests/ docs/
python -m pytest tests/test_tag_identity.py -v
make dead-code
```
Expected: grep shows no remaining references (outside this plan/spec doc);
pytest passes; `make dead-code` doesn't flag a newly-orphaned whitelist entry.

- [ ] **Step 6: Commit**

```bash
git add src/hydra_suite/core/post/tag_identity.py tests/test_tag_identity.py pyproject.toml docs/schematics/trackerkit_pipeline.md
git commit -m "chore: delete dead build_tag_only_trajectories/_interpolate_segment_rows"
```

---

## Task 5: Postpass-trigger completeness fix

**Files:**
- Modify: `src/hydra_suite/core/tracking/session_policy.py:56-65`
- Test: `tests/test_session_policy.py` (check it exists with
  `ls tests/test_session_policy*.py`; create alongside existing tests for
  this module if not)

**Interfaces:**
- Consumes: `is_headtail_compute_enabled` (already defined in this file,
  lines 35-41) — reused, not reimplemented.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_session_policy.py (add near existing should_run_interpolated_postpass tests)
from hydra_suite.core.tracking.session_policy import should_run_interpolated_postpass


def _base_config(**overrides):
    cfg = {
        "detection_method": "yolo_obb",
        "individual_interpolate_occlusions": True,
        "enable_individual_dataset": False,
        "enable_pose_extractor": False,
        "final_media_export_videos_enabled": False,
        "cnn_classifiers": [],
        "use_apriltags": False,
        "enable_headtail_orientation": False,
        "yolo_headtail_model_path": "",
    }
    cfg.update(overrides)
    return cfg


def test_postpass_triggers_on_cnn_classifiers_alone():
    cfg = _base_config(cnn_classifiers=[{"model_path": "x.pt", "label": "id"}])
    assert should_run_interpolated_postpass(cfg) is True


def test_postpass_triggers_on_apriltags_alone():
    cfg = _base_config(use_apriltags=True)
    assert should_run_interpolated_postpass(cfg) is True


def test_postpass_triggers_on_headtail_alone():
    cfg = _base_config(
        enable_headtail_orientation=True,
        yolo_headtail_model_path="/tmp/headtail.pt",
    )
    assert should_run_interpolated_postpass(cfg) is True


def test_postpass_false_when_nothing_enabled():
    cfg = _base_config()
    assert should_run_interpolated_postpass(cfg) is False
```

(If `tests/test_session_policy.py` already exists with a `_base_config`-style
helper or fixture, reuse/extend that instead of duplicating it.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_session_policy.py -v -k postpass_triggers`
Expected: FAIL (CNN/AprilTag/head-tail-alone cases return `False` today)

- [ ] **Step 3: Fix `should_run_interpolated_postpass`**

Replace `session_policy.py:56-65`:

```python
def should_run_interpolated_postpass(config: Mapping[str, Any]) -> bool:
    if not _truthy(config, "individual_interpolate_occlusions", default=True):
        return False
    if not is_individual_pipeline_enabled(config):
        return False
    return (
        should_export_final_canonical_images(config)
        or is_pose_export_enabled(config)
        or should_export_final_media_videos(config)
        or bool(config.get("cnn_classifiers", []))
        or bool(config.get("use_apriltags", False))
        or is_headtail_compute_enabled(config)
    )
```

`cnn_classifiers`/`use_apriltags` are the same lowercase config-dict keys the
GUI already reads at `detection_panel.py:1382-1383` and
`orchestrators/config.py:1884-1888`; `is_headtail_compute_enabled` is the
existing predicate at lines 35-41 of this same file, previously defined but
never referenced from this OR-list.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_session_policy.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/core/tracking/session_policy.py tests/test_session_policy.py
git commit -m "fix(tracking): trigger interpolated post-pass for CNN/AprilTag/head-tail alone"
```

---

## Task 6: Oriented-video fallback on `interp_lookup` miss

**Files:**
- Modify: `src/hydra_suite/core/individual/dataset/oriented_video.py:642-670`
- Test: `tests/test_oriented_video.py` (check filename with
  `find tests -iname "*oriented_video*"`)

**Interfaces:**
- Consumes: `ellipse_to_obb_corners` is NOT needed here — `_build_task`
  already takes `center_x`/`center_y`/`width`/`height`/`theta` directly
  (confirmed at the call site, lines 662-669), so the fallback only needs to
  supply those five values, not build an OBB itself.

- [ ] **Step 1: Read the current block and the row object's available fields**

```bash
sed -n '595,675p' src/hydra_suite/core/individual/dataset/oriented_video.py
```

Confirm `row` (from `interp_rows_by_frame`, iterated as
`for row in interp_rows:`) exposes `.X`/`.Y`/`.Theta` in addition to
`.TrajectoryID` (it comes from `itertuples()` or similar over the tracking
CSV — verify by finding where `interp_rows_by_frame` is populated, upstream
in this same file).

- [ ] **Step 2: Write the failing test**

Add to `tests/test_oriented_video.py` a test that constructs the minimal
state needed to reach this code path with `interp_lookup` missing a
(frame_id, traj_id) key that IS present in the CSV with non-NaN X/Y/Theta,
and asserts the frame is NOT dropped (a task is still emitted for it) using
`REFERENCE_BODY_SIZE`-derived width/height. Since this function is deep
inside a class with many collaborators, base the test on the closest
existing test in this file for the same code path (search
`grep -n "def test_" tests/test_oriented_video.py` and reuse its fixture
setup), asserting on `missing_breakdown["missing_interpolated_rows"]`
staying `0` for this row and the resulting task list containing an entry for
`(frame_id, traj_id)`.

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/test_oriented_video.py -v -k fallback`
Expected: FAIL (row currently dropped, `missing_interpolated_rows` increments)

- [ ] **Step 4: Implement the fallback**

Replace lines 653-656:

```python
                        record = interp_lookup.get((frame_id, traj_id))
                        if record is None:
                            missing_breakdown["missing_interpolated_rows"] += 1
                            continue
```

with:

```python
                        record = interp_lookup.get((frame_id, traj_id))
                        if record is None:
                            # Fall back to the CSV's own X/Y/Theta instead of
                            # dropping the frame (design spec, bug fix #3):
                            # a sidecar-lookup miss no longer means "no
                            # geometry at all" once geometry sourcing
                            # respects the CSV's own interpolated value
                            # (see interpolated_crops.py's NaN-triggered
                            # priority rule).
                            row_x = getattr(row, "X", float("nan"))
                            row_y = getattr(row, "Y", float("nan"))
                            row_theta = getattr(row, "Theta", float("nan"))
                            if (
                                row_x != row_x  # NaN check without importing math/np here
                                or row_y != row_y
                                or row_theta != row_theta
                            ):
                                missing_breakdown["missing_interpolated_rows"] += 1
                                continue
                            ref_size = float(
                                self._params.get("REFERENCE_BODY_SIZE", 20.0)
                            )
                            record = {
                                "cx": float(row_x),
                                "cy": float(row_y),
                                "theta": float(row_theta),
                                "w": ref_size * 2.2,
                                "h": ref_size * 0.8,
                            }
```

Confirm `self._params` is the correct attribute name for this class's params
dict by checking how `REFERENCE_BODY_SIZE`-style keys are read elsewhere in
the same class (`grep -n "self\._params\|self\.params" src/hydra_suite/core/individual/dataset/oriented_video.py | head`);
adjust the attribute name to match if different. The `ref_size * 2.2` /
`ref_size * 0.8` width/height fallback mirrors
`interpolated_crops.py::_process_occluded_run`'s existing
`REFERENCE_BODY_SIZE`-derived default (lines 385-389 of that file) so both
fallback paths agree on the same synthetic aspect ratio.

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_oriented_video.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/hydra_suite/core/individual/dataset/oriented_video.py tests/test_oriented_video.py
git commit -m "fix(oriented-video): fall back to CSV geometry on interp_lookup miss"
```

---

## Task 7: `synthetic_detections.py` — `build_synthetic_obb_result` + degenerate-OBB pre-filter

**Files:**
- Create: `src/hydra_suite/core/post/synthetic_detections.py`
- Test: `tests/test_synthetic_detections.py`

**Interfaces:**
- Produces:
  `filter_degenerate_tasks(tasks: list[dict], geometry: CanonicalGeometry, clipping_stats: ClippingStats | None) -> list[dict]`
  and
  `build_synthetic_obb_result(frame_idx: int, tasks: list[dict]) -> OBBResult`.
  Both consumed by Tasks 10-12 and the final wiring in Task 12's caller
  (`interpolated_crops.py`). `tasks` is exactly the per-frame list
  `frame_tasks[f]` already produced by `_detect_interpolation_gaps` — each
  dict has keys `frame_id`, `cx`, `cy`, `w`, `h`, `theta`, `traj_id`,
  `interp_index`, `interp_from`, `interp_total` (unchanged by this task).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_synthetic_detections.py
import numpy as np
import pytest

from hydra_suite.core.canonicalization.geometry import (
    CanonicalGeometry,
    ClippingStats,
)
from hydra_suite.core.post.synthetic_detections import (
    build_synthetic_obb_result,
    filter_degenerate_tasks,
)


def _task(cx=50.0, cy=50.0, w=20.0, h=8.0, theta=0.0, frame_id=1, traj_id=3, interp_index=1):
    return {
        "frame_id": frame_id,
        "cx": cx,
        "cy": cy,
        "w": w,
        "h": h,
        "theta": theta,
        "traj_id": traj_id,
        "interp_index": interp_index,
        "interp_from": (0, 2),
        "interp_total": 1,
    }


def test_build_synthetic_obb_result_shapes():
    tasks = [_task(traj_id=1), _task(traj_id=2, cx=80.0)]
    obb = build_synthetic_obb_result(frame_idx=1, tasks=tasks)
    assert obb.num_detections == 2
    assert obb.corners.shape == (2, 4, 2)
    assert obb.detection_ids.shape == (2,)
    assert (obb.detection_ids < 0).all()  # negative synthetic ids
    assert obb.detection_ids[0] != obb.detection_ids[1]


def test_build_synthetic_obb_result_empty():
    obb = build_synthetic_obb_result(frame_idx=1, tasks=[])
    assert obb.num_detections == 0
    assert obb.corners.shape == (0, 4, 2)


def test_build_synthetic_obb_result_matches_ellipse_to_obb_corners():
    from hydra_suite.core.individual.geometry import ellipse_to_obb_corners

    task = _task()
    obb = build_synthetic_obb_result(frame_idx=1, tasks=[task])
    expected = ellipse_to_obb_corners(
        task["cx"], task["cy"], task["w"], task["h"], task["theta"]
    )
    np.testing.assert_allclose(obb.corners[0], expected)


def test_filter_degenerate_tasks_drops_zero_length_edge_and_tallies():
    geometry = CanonicalGeometry(canvas_wh=(64, 64), margin=1.3, aspect_ratio=2.0)
    good = _task()
    degenerate = _task(w=0.0, h=0.0, traj_id=99)
    stats = ClippingStats()
    kept = filter_degenerate_tasks([good, degenerate], geometry, stats)
    assert len(kept) == 1
    assert kept[0]["traj_id"] == 3
    assert stats.degenerate_count == 1


def test_filter_degenerate_tasks_none_clipping_stats_is_safe():
    geometry = CanonicalGeometry(canvas_wh=(64, 64), margin=1.3, aspect_ratio=2.0)
    kept = filter_degenerate_tasks([_task()], geometry, None)
    assert len(kept) == 1
```

Check `ClippingStats`'s attribute name for the degenerate tally before
writing the assertion (`grep -n "degenerate" src/hydra_suite/core/canonicalization/geometry.py`)
and adjust `stats.degenerate_count` to match the real field name.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_synthetic_detections.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `synthetic_detections.py`**

```python
"""Synthetic OBBResult construction for interpolated-crop inference.

Builds an ``OBBResult`` (the same struct real OBB detections produce) from
one frame's interpolated-gap tasks, so the SAME batched stage functions
``Pipeline`` calls for real detections
(``extract_canonical_crops_batch``/``run_pose_batch``, ``run_cnn_batch``,
``run_headtail_batch``, ``extract_aabb_crops``/``run_apriltag``) can run on
interpolated geometry unmodified. See design spec "Architecture" and
"Key architectural finding".
"""

from __future__ import annotations

import logging

import numpy as np

from hydra_suite.core.canonicalization.geometry import (
    CanonicalGeometry,
    ClippingStats,
    canonical_affine,
)
from hydra_suite.core.individual.dataset.naming import synthetic_interpolated_det_id
from hydra_suite.core.individual.geometry import ellipse_to_obb_corners
from hydra_suite.core.inference.result import OBBResult

logger = logging.getLogger(__name__)


def filter_degenerate_tasks(
    tasks: list[dict],
    geometry: CanonicalGeometry,
    clipping_stats: "ClippingStats | None",
) -> list[dict]:
    """Drop tasks whose ellipse-derived OBB is degenerate, tallying the drop.

    ``extract_canonical_crops``/``_batch`` (``stages/crops.py``) does NOT
    raise or skip on a degenerate OBB -- it silently fudges an identity
    affine (crops.py:97-98) and has no ``ClippingStats`` plumbing at all
    (design spec "Error handling", adversarial-review G2/G3). This function
    is what restores today's loud-skip-and-tally behavior: it must run
    BEFORE any task reaches ``build_synthetic_obb_result``/the batch stage
    functions, exactly mirroring what
    ``interpolated_crops.py::_compute_frame_corners_and_affines`` used to do
    inline. For kept tasks it also records the real overflow via
    ``canonical_affine``, matching what ``Pipeline`` does for real
    detections (``pipeline.py:331-338``).
    """
    kept: list[dict] = []
    for task in tasks:
        corners = ellipse_to_obb_corners(
            task["cx"], task["cy"], task["w"], task["h"], task["theta"]
        )
        try:
            canonical_affine(corners, geometry)
        except ValueError:
            if clipping_stats is not None:
                clipping_stats.record_degenerate()
            logger.warning(
                "Interp pose/CNN/tag/headtail: skipping frame_id=%s traj_id=%s "
                "-- degenerate OBB has no Layer 1 canonical transform "
                "(canonical_affine raised); the stage layer would otherwise "
                "silently fudge an identity-affine crop instead of skipping.",
                task["frame_id"],
                task["traj_id"],
            )
            continue
        if clipping_stats is not None:
            clipping_stats.record(corners, geometry)
        kept.append(task)
    return kept


def build_synthetic_obb_result(frame_idx: int, tasks: list[dict]) -> OBBResult:
    """Build an ``OBBResult`` for one frame's (already-filtered) interpolated tasks.

    ``tasks`` is ``frame_tasks[f]`` (or the ``filter_degenerate_tasks``
    output of it) -- each dict has ``cx``/``cy``/``w``/``h``/``theta``/
    ``frame_id``/``traj_id``/``interp_index``. Detection ids are negative
    and stable per (frame_id, trajectory_id, interp_index) via
    ``synthetic_interpolated_det_id`` -- the SAME scheme
    ``parse_identity_image_filename`` already uses for interpolated crop
    filenames (``naming.py``), so a synthetic id can never collide with the
    positive real-detection id space (``OBBResult.make_detection_ids``).
    """
    n = len(tasks)
    corners = np.zeros((n, 4, 2), dtype=np.float32)
    centroids = np.zeros((n, 2), dtype=np.float32)
    angles = np.zeros(n, dtype=np.float32)
    sizes = np.zeros(n, dtype=np.float32)
    shapes = np.zeros((n, 2), dtype=np.float32)
    confidences = np.ones(n, dtype=np.float32)
    det_ids = np.zeros(n, dtype=np.int64)

    for i, task in enumerate(tasks):
        corners[i] = ellipse_to_obb_corners(
            task["cx"], task["cy"], task["w"], task["h"], task["theta"]
        )
        centroids[i] = (task["cx"], task["cy"])
        angles[i] = task["theta"]
        area = float(np.pi / 4.0 * task["w"] * task["h"])
        sizes[i] = area
        aspect = float(task["w"] / task["h"]) if task["h"] else 0.0
        shapes[i] = (area, aspect)
        det_ids[i] = synthetic_interpolated_det_id(
            task["frame_id"], task["traj_id"], task["interp_index"]
        )

    return OBBResult(
        frame_idx=frame_idx,
        centroids=centroids,
        angles=angles,
        sizes=sizes,
        shapes=shapes,
        confidences=confidences,
        corners=corners,
        detection_ids=det_ids,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_synthetic_detections.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/core/post/synthetic_detections.py tests/test_synthetic_detections.py
git commit -m "feat(post): add synthetic OBBResult builder + degenerate-OBB pre-filter"
```

---

## Task 8: Geometry-sourcing priority fix (NaN-triggered CSV-first)

**Files:**
- Modify: `src/hydra_suite/core/post/interpolated_crops.py::_process_occluded_run`
  (lines 341-420)
- Test: `tests/test_core_interpolated_crops.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: no signature change to `_process_occluded_run` — same
  parameters, same return `(interp_runs, interp_gaps, j)` / `None`.

- [ ] **Step 1: Write the failing test**

Check `tests/test_core_interpolated_crops.py` for its existing fixture style
around `_process_occluded_run` or `_scan_trajectory_gaps` first
(`grep -n "_process_occluded_run\|_scan_trajectory_gaps" tests/test_core_interpolated_crops.py`),
then add a test using that same fixture-construction pattern:

```python
def test_process_occluded_run_uses_csv_value_when_present():
    """A row with non-NaN X/Y/Theta already in the CSV (mechanism-1 fill)
    must be used directly, not re-derived by linear interpolation."""
    import pandas as pd

    from hydra_suite.core.post.interpolated_crops import _process_occluded_run

    group = pd.DataFrame(
        {
            "FrameID": [0, 1, 2],
            "X": [0.0, 999.0, 20.0],  # frame 1 already filled by mechanism (1)
            "Y": [0.0, 888.0, 20.0],
            "Theta": [0.0, 1.23, 0.0],
            "State": ["tracked", "occluded", "tracked"],
            "DetectionID": [1, None, 2],
        }
    )
    frame_tasks = {}
    params = {"REFERENCE_BODY_SIZE": 20.0}
    result = _process_occluded_run(
        params,
        None,
        group,
        traj_id=5,
        last_valid_idx=0,
        i=1,
        j=2,
        detection_cache=None,
        position_scale=1.0,
        size_scale=1.0,
        frame_tasks=frame_tasks,
        interp_runs=0,
        interp_gaps=0,
    )
    assert result is not None
    task = frame_tasks[1][0]
    assert task["cx"] == pytest.approx(999.0)
    assert task["cy"] == pytest.approx(888.0)
    assert task["theta"] == pytest.approx(1.23)


def test_process_occluded_run_falls_back_when_csv_value_is_nan():
    import pandas as pd

    from hydra_suite.core.post.interpolated_crops import _process_occluded_run

    group = pd.DataFrame(
        {
            "FrameID": [0, 1, 2],
            "X": [0.0, float("nan"), 20.0],
            "Y": [0.0, float("nan"), 20.0],
            "Theta": [0.0, float("nan"), 0.0],
            "State": ["tracked", "occluded", "tracked"],
            "DetectionID": [1, None, 2],
        }
    )
    frame_tasks = {}
    params = {"REFERENCE_BODY_SIZE": 20.0}
    result = _process_occluded_run(
        params,
        None,
        group,
        traj_id=5,
        last_valid_idx=0,
        i=1,
        j=2,
        detection_cache=None,
        position_scale=1.0,
        size_scale=1.0,
        frame_tasks=frame_tasks,
        interp_runs=0,
        interp_gaps=0,
    )
    assert result is not None
    task = frame_tasks[1][0]
    # falls back to the existing linear midpoint: t=0.5 between (0,0) and (20,20)
    assert task["cx"] == pytest.approx(10.0)
    assert task["cy"] == pytest.approx(10.0)
```

(Need `import pytest` at the top of the test file if not already present.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_core_interpolated_crops.py -v -k process_occluded_run_uses_csv_value`
Expected: FAIL (current code always recomputes linearly, ignoring the
already-filled 999.0/888.0/1.23 CSV values)

- [ ] **Step 3: Implement the priority fix**

In `_process_occluded_run` (interpolated_crops.py:391-419), the `for k in
range(i, j):` loop currently always computes `cx`/`cy`/`theta` via linear
interpolation. Replace that block:

```python
    for k in range(i, j):
        if _stop():
            return None
        row = group.iloc[k]
        f = int(row["FrameID"])
        t = (f - f0) / (f1 - f0)
        cx = float(prev_row["X"]) + t * (float(next_row["X"]) - float(prev_row["X"]))
        cy = float(prev_row["Y"]) + t * (float(next_row["Y"]) - float(prev_row["Y"]))
        theta = _interp_angle(float(prev_row["Theta"]), float(next_row["Theta"]), t)
        w = w0 + t * (w1 - w0)
        h = h0 + t * (h1 - h0)
```

with:

```python
    for k in range(i, j):
        if _stop():
            return None
        row = group.iloc[k]
        f = int(row["FrameID"])
        t = (f - f0) / (f1 - f0)
        # Geometry-sourcing priority (design spec "Geometry sourcing",
        # NaN-triggered, not max_gap-triggered): if mechanism (1)'s
        # trajectory interpolation already filled this row's X/Y/Theta
        # (honoring the user's interpolation_method and heading-flip
        # correction), use it directly instead of re-deriving a bespoke
        # linear/± 180 degree estimate. Only fall back to independent
        # linear interpolation when the CSV row is genuinely NaN here
        # (interpolation_method="None" -- the GUI default -- or a gap
        # beyond max_gap).
        row_x = row["X"] if "X" in group.columns else float("nan")
        row_y = row["Y"] if "Y" in group.columns else float("nan")
        row_theta = row["Theta"] if "Theta" in group.columns else float("nan")
        if not (pd.isna(row_x) or pd.isna(row_y) or pd.isna(row_theta)):
            cx = float(row_x)
            cy = float(row_y)
            theta = float(row_theta)
        else:
            cx = float(prev_row["X"]) + t * (float(next_row["X"]) - float(prev_row["X"]))
            cy = float(prev_row["Y"]) + t * (float(next_row["Y"]) - float(prev_row["Y"]))
            theta = _interp_angle(float(prev_row["Theta"]), float(next_row["Theta"]), t)
        w = w0 + t * (w1 - w0)
        h = h0 + t * (h1 - h0)
```

Size (`w`/`h`) sourcing is unchanged per the spec ("Size ... continues to
source from the OBB detection-cache endpoints as today").

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_core_interpolated_crops.py -v`
Expected: PASS (all existing + 2 new tests)

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/core/post/interpolated_crops.py tests/test_core_interpolated_crops.py
git commit -m "fix(post): respect mechanism-1's own CSV geometry before re-deriving it"
```

---

## Task 9: Model-loading glue rewrite (config + runtime + per-stage loaders)

**Files:**
- Modify: `src/hydra_suite/core/post/interpolated_crops.py` — replace
  `_init_pose_backend` (108-170), `_resolve_backend`/`_resolved_runtime_string`
  (173-200, both become unused and are deleted), `_init_apriltag_detector`
  (203-220), `_init_cnn_backends` (223-264), `_init_headtail_analyzer`
  (267-290), `_init_interpolation_backends` (293-311)
- Test: `tests/test_core_interpolated_crops.py`

**Interfaces:**
- Consumes: `build_inference_config_from_params` (`core/inference/config.py:708`),
  `RuntimeContext.from_config` (`core/inference/runtime.py:126`),
  `load_headtail_model`/`load_cnn_model`/`load_pose_model`/
  `load_apriltag_model` (public loaders in `stages/headtail.py`,
  `stages/cnn.py`, `stages/pose.py`, `stages/apriltag.py`).
- Produces: `_init_interpolation_backends(params, output_dir, geometry) ->
  (cfg, runtime, pose_model, apriltag_model, cnn_models, cnn_labels,
  headtail_model)` — return shape changes from today's
  `(pose_backend, pose_kpt_source_names, pose_kpt_labels, apriltag_detector,
  cnn_backends, cnn_labels, headtail_analyzer, interp_cnn_rows)`. This is a
  breaking signature change to a private (`_`-prefixed) function local to
  this module — its single caller (`run_interpolated_crops`) is updated in
  Task 12, and `pose_kpt_source_names`/`pose_kpt_labels` (needed for CSV
  column naming) move to being derived from `pose_model.keypoint_names` at
  the Task 10/12 call sites instead of returned here, since `PoseModel`
  already carries them.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_core_interpolated_crops.py — add near existing backend-init tests
def test_init_interpolation_backends_returns_config_and_runtime():
    from hydra_suite.core.post.interpolated_crops import (
        _init_interpolation_backends,
    )
    from hydra_suite.core.canonicalization.geometry import canonical_geometry_from_params

    params = {
        "ENABLE_POSE_EXTRACTOR": False,
        "USE_APRILTAGS": False,
        "CNN_CLASSIFIERS": [],
        "YOLO_HEADTAIL_MODEL_PATH": "",
        "RUNTIME_TIER": "cpu",
    }
    geometry = canonical_geometry_from_params(params)
    result = _init_interpolation_backends(params, "/tmp", geometry)
    cfg, runtime, pose_model, apriltag_model, cnn_models, cnn_labels, headtail_model = result
    assert cfg.pose is None
    assert pose_model is None
    assert apriltag_model is None
    assert cnn_models == []
    assert cnn_labels == []
    assert headtail_model is None
    assert runtime.device in {"cpu", "mps", "cuda:0"}
```

This all-disabled case needs no real model files, so it's a fast, hermetic
unit test — it exercises the config/runtime plumbing without touching any
backend loader's model-file path branch.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_core_interpolated_crops.py -v -k init_interpolation_backends_returns_config_and_runtime`
Expected: FAIL (`_init_interpolation_backends` still returns the old 8-tuple
shape / doesn't return `cfg`/`runtime`)

- [ ] **Step 3: Replace the five init functions**

In `interpolated_crops.py`, delete `_init_pose_backend` (108-170),
`_resolve_backend` (173-185), `_resolved_runtime_string` (188-200),
`_init_apriltag_detector` (203-220), `_init_cnn_backends` (223-264),
`_init_headtail_analyzer` (267-290), and `_init_interpolation_backends`
(293-311) in full, and also remove the now-unused
`from hydra_suite.core.inference.api import load_pose_backend` import
(line 34). Add:

```python
def _init_interpolation_backends(params, output_dir, geometry):
    """Build the InferenceConfig/RuntimeContext and load the four
    downstream-stage models (headtail/CNN/pose/AprilTag) via their public
    stage loaders -- the SAME loaders ``Pipeline`` uses, so the interpolated
    path shares the tier->backend resolution and model-loading code instead
    of hand-rolling its own runtime-flavor ladder (design spec
    "Model-loading glue"; see the plan's "Plan-level deviation from the spec
    text" note for why the per-stage loaders are called directly here rather
    than ``runner.py``'s private ``_load_all_models`` -- that function always
    loads an OBB/bgsub detector too, which this pass never uses).

    Returns (cfg, runtime, pose_model, apriltag_model, cnn_models,
    cnn_labels, headtail_model). Any model whose config/params disable it is
    None (pose, apriltag, headtail) or empty (cnn_models/cnn_labels), mirroring
    today's opt-in behavior -- CNN no longer depends on pose being enabled
    (design spec, bug fix #1: the CNN/pose decoupling is now real, since
    ``run_cnn_batch`` builds its own classifier crops independently, unlike
    the old ``_pending_cnn_crops.append(pose_crop)``).
    """
    from hydra_suite.core.inference.config import build_inference_config_from_params
    from hydra_suite.core.inference.runtime import RuntimeContext
    from hydra_suite.core.inference.stages.apriltag import load_apriltag_model
    from hydra_suite.core.inference.stages.cnn import load_cnn_model
    from hydra_suite.core.inference.stages.headtail import load_headtail_model
    from hydra_suite.core.inference.stages.pose import load_pose_model

    cfg = build_inference_config_from_params(params)
    runtime = RuntimeContext.from_config(cfg)

    pose_model = None
    if cfg.pose is not None:
        try:
            pose_model = load_pose_model(
                cfg.pose,
                runtime,
                out_root=str(Path(output_dir).expanduser()),
            )
        except Exception as exc:
            logger.warning(
                "Interpolated pose analysis disabled (backend init failed): %s",
                exc,
            )
            pose_model = None

    apriltag_model = None
    if cfg.apriltag.enabled:
        try:
            apriltag_model = load_apriltag_model(cfg.apriltag)
        except Exception as exc:
            logger.warning("Interpolated AprilTag analysis disabled: %s", exc)
            apriltag_model = None

    cnn_models = []
    cnn_labels = []
    for cnn_cfg in cfg.cnn_phases:
        try:
            cnn_models.append(load_cnn_model(cnn_cfg, runtime))
            cnn_labels.append(cnn_cfg.label)
        except Exception as exc:
            logger.warning(
                "Interpolated CNN identity '%s' disabled: %s", cnn_cfg.label, exc
            )

    headtail_model = None
    if cfg.headtail is not None:
        try:
            headtail_model = load_headtail_model(cfg.headtail, runtime)
        except Exception as exc:
            logger.warning("Interpolated head-tail analysis disabled: %s", exc)
            headtail_model = None

    return (
        cfg,
        runtime,
        pose_model,
        apriltag_model,
        cnn_models,
        cnn_labels,
        headtail_model,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_core_interpolated_crops.py -v -k init_interpolation_backends`
Expected: PASS

(This will leave `run_interpolated_crops` and `_process_single_frame`/
`_run_frame_tasks_loop` broken against the old call signature until Task 12
rewires them — that's expected; Task 12 is the integration point. Do not run
the full `run_interpolated_crops` integration test yet.)

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/core/post/interpolated_crops.py tests/test_core_interpolated_crops.py
git commit -m "refactor(post): build model config/runtime via shared InferenceConfig plumbing"
```

---

## Task 10: Pose + CNN inference via `run_pose_batch`/`run_cnn_batch`

**Files:**
- Modify: `src/hydra_suite/core/post/interpolated_crops.py` — replace
  `_flush_pose_batch` (525-637) and `_flush_cnn_batch` (640-694)
- Test: `tests/test_core_interpolated_crops.py`

**Interfaces:**
- Consumes: `build_synthetic_obb_result`/`filter_degenerate_tasks` (Task 7),
  `extract_canonical_crops_batch`/`run_pose_batch`
  (`stages/crops.py:416`, `stages/pose.py:382`), `run_cnn_batch`
  (`stages/cnn.py:129`).
- Produces:
  `_flush_pose_cnn_window(pending_frames, pending_obbs, pose_model,
  cnn_models, cnn_labels, cfg, runtime, geometry, interp_pose_rows,
  interp_cnn_rows, gen, profiler) -> None` — replaces
  `_flush_pose_batch`/`_flush_cnn_batch`'s combined role. Appends rows to
  `interp_pose_rows`/`interp_cnn_rows` in place (same mutation contract the
  old flush functions had), now including `PoseSource="interp"` /
  `CNN_<label>_Source="interp"` per row (Task 2's new columns — these are
  the per-detection artifact-CSV rows Task 2's merge functions later read
  via `frame_id`/`trajectory_id`, so stamping `interp` here is what makes
  `TagSource`/`PoseSource`/etc. resolve to `"interp"` downstream once merged;
  see Task 2's `_can_merge_interp` note — the artifact CSVs themselves don't
  need a `*Source` column, only the trajectories DataFrame the merge
  produces does, since every row in an interp artifact CSV is by
  construction an interpolated one).

- [ ] **Step 1: Write the failing test**

Because this function needs a loaded `PoseModel`/`CNNModel` to exercise
meaningfully, and this repo's test fixtures for pose/CNN backends are
integration-level (real small model files under
`tools/equivalence/fixtures/`), write a narrower unit test against the
crop-and-source-stamping contract using a **fake** model object (matching
`PoseModel`/`CNNModel`'s duck-typed `.backend.predict_batch(...)` interface)
rather than a real network, so this stays a fast unit test:

```python
# tests/test_core_interpolated_crops.py
def test_flush_pose_cnn_window_stamps_pose_source_interp(monkeypatch):
    from hydra_suite.core.post import interpolated_crops as ic
    from hydra_suite.core.post.synthetic_detections import build_synthetic_obb_result
    from hydra_suite.core.canonicalization.geometry import canonical_geometry_from_params
    from hydra_suite.core.inference.stages.pose import PoseModel
    from hydra_suite.core.inference.result import PoseResult
    import numpy as np

    class _FakeBackend:
        def predict_batch(self, crops):
            return [
                PoseResult(
                    keypoints=np.zeros((1, 1, 3), dtype=np.float32),
                    valid_mask=np.array([True]),
                )
                for _ in crops
            ]

        # run_pose_batch checks hasattr(backend, "predict_batch_cuda") to
        # decide the CUDA branch; omit it so the CPU branch is taken.

    params = {"RUNTIME_TIER": "cpu"}
    geometry = canonical_geometry_from_params(params)
    pose_model = PoseModel(backend=_FakeBackend(), n_keypoints=1, keypoint_names=["head"])

    task = {
        "frame_id": 1, "cx": 32.0, "cy": 32.0, "w": 20.0, "h": 8.0,
        "theta": 0.0, "traj_id": 5, "interp_index": 1,
        "interp_from": (0, 2), "interp_total": 1,
    }
    obb = build_synthetic_obb_result(1, [task])
    frame = np.zeros((64, 64, 3), dtype=np.uint8)

    interp_pose_rows = []
    ic._flush_pose_cnn_window(
        pending_frames=[frame],
        pending_obbs=[obb],
        pending_tasks_by_frame=[[task]],
        pose_model=pose_model,
        cnn_models=[],
        cnn_labels=[],
        runtime=None,  # patched below via RuntimeContext.from_config if required
        geometry=geometry,
        cfg=None,
        interp_pose_rows=interp_pose_rows,
        interp_cnn_rows={},
        profiler=ic.TrackingProfiler(enabled=False) if hasattr(ic, "TrackingProfiler") else None,
    )
    assert len(interp_pose_rows) == 1
    assert interp_pose_rows[0]["PoseSource"] == "interp"
    assert interp_pose_rows[0]["trajectory_id"] == 5
```

This test's exact keyword names must match Step 3's final signature —
finalize the signature in Step 3 first if this draft and the implementation
diverge, then align the test. The core assertions to preserve either way:
one row is produced, keyed by `frame_id`/`trajectory_id`, carrying
`PoseSource == "interp"`.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_core_interpolated_crops.py -v -k flush_pose_cnn_window`
Expected: FAIL (`AttributeError: module has no attribute '_flush_pose_cnn_window'`)

- [ ] **Step 3: Implement `_flush_pose_cnn_window`**

Delete `_flush_pose_batch` (525-637) and `_flush_cnn_batch` (640-694) in
full. Add:

```python
def _flush_pose_cnn_window(
    pending_frames,
    pending_obbs,
    pending_tasks_by_frame,
    pose_model,
    cnn_models,
    cnn_labels,
    cfg,
    runtime,
    geometry,
    interp_pose_rows,
    interp_cnn_rows,
    profiler,
):
    """Run pose + CNN inference over a window of (frame, synthetic OBB) pairs.

    Calls the SAME stage functions ``Pipeline`` calls for real detections
    (``pipeline.py:367-387``): ``extract_canonical_crops_batch`` then
    ``run_pose_batch`` for pose, ``run_cnn_batch`` per CNN phase for CNN --
    instead of this module's old hand-rolled crop extraction + batching.
    ``suppress_foreign=True`` for the pose call matches today's
    intra-synthetic-batch masking of other interpolated tasks in the same
    frame (design spec, AprilTag/foreign-suppression decisions). Pose and
    CNN crops are now genuinely independent (CNN via
    ``extract_classifier_crops_batch_np`` inside ``run_cnn_batch``, not a
    reused pose crop) -- design spec bug fix #1.
    """
    from hydra_suite.core.inference.stages.cnn import run_cnn_batch
    from hydra_suite.core.inference.stages.crops import extract_canonical_crops_batch
    from hydra_suite.core.inference.stages.pose import run_pose_batch

    if not pending_frames:
        return

    if profiler is not None:
        profiler.tick("interp_pose_inference")
    if pose_model is not None:
        crop_batch = extract_canonical_crops_batch(
            pending_frames,
            pending_obbs,
            geometry,
            runtime,
            suppress_foreign=True,
            background_color=(0, 0, 0),
        )
        pose_by_frame = run_pose_batch(crop_batch, pose_model, cfg.pose, runtime, geometry)
        for frame_idx, tasks in zip(
            (obb.frame_idx for obb in pending_obbs), pending_tasks_by_frame
        ):
            pose_result = pose_by_frame.get(frame_idx)
            if pose_result is None:
                continue
            for i, task in enumerate(tasks):
                kpts = pose_result.keypoints[i] if i < len(pose_result.keypoints) else None
                pose_wide = {}
                pose_mean_conf = pose_valid_fraction = 0.0
                pose_num_valid = pose_num_keypoints = 0
                if kpts is not None:
                    conf_col = kpts[:, 2]
                    pose_num_keypoints = int(kpts.shape[0])
                    valid_mask = conf_col >= float(cfg.pose.min_keypoint_confidence)
                    pose_num_valid = int(valid_mask.sum())
                    pose_mean_conf = float(conf_col.mean()) if pose_num_keypoints else 0.0
                    pose_valid_fraction = (
                        pose_num_valid / pose_num_keypoints if pose_num_keypoints else 0.0
                    )
                    pose_wide = flatten_pose_keypoints_row(
                        kpts, build_pose_keypoint_labels(
                            pose_model.keypoint_names, pose_num_keypoints
                        )
                    )
                pose_row = {
                    "frame_id": int(task["frame_id"]),
                    "trajectory_id": int(task["traj_id"]),
                    "filename": "",
                    "PoseMeanConf": pose_mean_conf,
                    "PoseValidFraction": pose_valid_fraction,
                    "PoseNumValid": pose_num_valid,
                    "PoseNumKeypoints": pose_num_keypoints,
                    "PoseSource": "interp",
                }
                pose_row.update(pose_wide)
                interp_pose_rows.append(pose_row)
    if profiler is not None:
        profiler.tock("interp_pose_inference")

    if profiler is not None:
        profiler.tick("interp_cnn_inference")
    for cnn_model, cnn_label, cnn_cfg in zip(cnn_models, cnn_labels, cfg.cnn_phases):
        try:
            cnn_by_frame = run_cnn_batch(
                pending_frames, pending_obbs, cnn_model, cnn_cfg, runtime, geometry
            )
        except Exception as exc:
            logger.warning("Interp CNN '%s' batch failed: %s", cnn_label, exc)
            continue
        for frame_idx, tasks in zip(
            (obb.frame_idx for obb in pending_obbs), pending_tasks_by_frame
        ):
            cnn_result = cnn_by_frame.get(frame_idx)
            if cnn_result is None:
                continue
            for i, task in enumerate(tasks):
                pred = next(
                    (p for p in cnn_result.predictions if p.det_index == i), None
                )
                if pred is None:
                    continue
                row = {
                    "frame_id": int(task["frame_id"]),
                    "trajectory_id": int(task["traj_id"]),
                }
                row.update(
                    flatten_cnn_prediction_row(
                        cnn_label,
                        [f.factor_name for f in pred.factors],
                        [f.class_names for f in pred.factors],
                        [f.raw_probabilities for f in pred.factors],
                    )
                )
                row[f"CNN_{cnn_label}_Source"] = "interp"
                interp_cnn_rows.setdefault(cnn_label, []).append(row)
    if profiler is not None:
        profiler.tock("interp_cnn_inference")
```

Check `flatten_cnn_prediction_row`'s exact parameter names/order against its
real definition (`grep -n "def flatten_cnn_prediction_row" -A 15
src/hydra_suite/core/individual/properties/export.py`) before finalizing —
the old `_flush_cnn_batch` called it as
`flatten_cnn_prediction_row(label, factor_names, class_names, confidences)`
using `_pred.factor_names`/`_pred.class_names`/`_pred.confidences` from a
different (legacy classifier-backend) prediction shape; `CNNDetectionPrediction.factors`
here is a list of `CNNFactorPrediction(factor_name, class_names,
raw_probabilities)` (`result.py`, confirmed during planning) — adjust the
three list-comprehensions above if `flatten_cnn_prediction_row` expects
per-factor argmax class name + confidence rather than raw probability
vectors (check whether the function itself does the argmax, or expects it
pre-computed) and add that conversion here if needed.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_core_interpolated_crops.py -v -k flush_pose_cnn_window`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/core/post/interpolated_crops.py tests/test_core_interpolated_crops.py
git commit -m "refactor(post): route interpolated pose/CNN inference through run_pose_batch/run_cnn_batch"
```

---

## Task 11: AprilTag + head-tail inference via `run_apriltag`/`run_headtail_batch`

**Files:**
- Modify: `src/hydra_suite/core/post/interpolated_crops.py` — replace
  `_detect_apriltags_in_frame` (697-751) and `_detect_headtail_in_frame`
  (754-774)
- Test: `tests/test_core_interpolated_crops.py`

**Interfaces:**
- Consumes: `extract_aabb_crops`/`run_apriltag`
  (`stages/crops.py:110`, `stages/apriltag.py:48`), `run_headtail_batch`
  (`stages/headtail.py:245`), `build_synthetic_obb_result` (Task 7).
- Produces: `_detect_apriltags_in_frame(apriltag_model, cfg, frame, obb,
  tasks, interp_tag_rows) -> None` (per-frame, matching `Pipeline`'s
  per-frame AprilTag loop — there is no batch variant, per the spec's
  "Key architectural finding") and `_flush_headtail_window(pending_frames,
  pending_obbs, pending_tasks_by_frame, headtail_model, cfg, runtime,
  geometry, interp_headtail_rows) -> None` (windowed, matching
  `run_headtail_batch`'s signature — head-tail moves from per-frame
  `HeadTailAnalyzer.analyze_crops` to the windowed batch path, a registered
  expected difference per the spec's Testing section).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_core_interpolated_crops.py
def test_detect_apriltags_in_frame_writes_tag_source_via_run_apriltag(monkeypatch):
    from hydra_suite.core.post import interpolated_crops as ic
    from hydra_suite.core.post.synthetic_detections import build_synthetic_obb_result
    from hydra_suite.core.inference.config import AprilTagConfig
    from hydra_suite.core.inference.result import AprilTagResult
    from hydra_suite.core.inference.stages import apriltag as apriltag_stage
    import numpy as np

    task = {
        "frame_id": 1, "cx": 32.0, "cy": 32.0, "w": 20.0, "h": 8.0,
        "theta": 0.0, "traj_id": 5, "interp_index": 1,
        "interp_from": (0, 2), "interp_total": 1,
    }
    obb = build_synthetic_obb_result(1, [task])
    frame = np.zeros((64, 64, 3), dtype=np.uint8)

    def _fake_run_apriltag(cpu_crops, obb_result, model, config):
        return AprilTagResult(
            tag_ids=[7], det_indices=[0],
            centers=np.array([[32.0, 32.0]], dtype=np.float32),
            corners=np.zeros((1, 4, 2), dtype=np.float32),
        )

    monkeypatch.setattr(ic, "run_apriltag", _fake_run_apriltag, raising=False)

    interp_tag_rows = []
    ic._detect_apriltags_in_frame(
        apriltag_model=object(),
        cfg=AprilTagConfig(enabled=True),
        frame=frame,
        obb=obb,
        tasks=[task],
        interp_tag_rows=interp_tag_rows,
    )
    assert interp_tag_rows == [
        {"frame_id": 1, "trajectory_id": 5, "tag_id": 7}
    ]
```

(`monkeypatch.setattr(ic, "run_apriltag", ..., raising=False)` only works if
`run_apriltag` is imported at module scope in `interpolated_crops.py` in
Step 3 below — if it's imported inside the function instead, patch
`hydra_suite.core.inference.stages.apriltag.run_apriltag` directly instead.
Pick whichever import style Step 3 actually uses and align this test to it.)

```python
def test_flush_headtail_window_writes_heading_rows(monkeypatch):
    from hydra_suite.core.post import interpolated_crops as ic
    from hydra_suite.core.post.synthetic_detections import build_synthetic_obb_result
    from hydra_suite.core.canonicalization.geometry import canonical_geometry_from_params
    from hydra_suite.core.inference.stages.headtail import HeadTailModel
    from hydra_suite.core.inference.result import HeadTailResult
    import numpy as np

    class _FakeBackend:
        def predict_batch(self, crops):
            return [[np.array([0.9, 0.1])] for _ in crops]

    params = {"RUNTIME_TIER": "cpu"}
    geometry = canonical_geometry_from_params(params)
    headtail_model = HeadTailModel(
        backend=_FakeBackend(), input_size=(32, 32), class_names=["right", "left"]
    )
    task = {
        "frame_id": 1, "cx": 32.0, "cy": 32.0, "w": 20.0, "h": 8.0,
        "theta": 0.0, "traj_id": 5, "interp_index": 1,
        "interp_from": (0, 2), "interp_total": 1,
    }
    obb = build_synthetic_obb_result(1, [task])
    frame = np.zeros((64, 64, 3), dtype=np.uint8)

    interp_headtail_rows = []
    ic._flush_headtail_window(
        pending_frames=[frame],
        pending_obbs=[obb],
        pending_tasks_by_frame=[[task]],
        headtail_model=headtail_model,
        cfg=ic.build_inference_config_from_params(
            {**params, "YOLO_HEADTAIL_MODEL_PATH": "unused-in-test"}
        ) if hasattr(ic, "build_inference_config_from_params") else None,
        runtime=None,
        geometry=geometry,
        interp_headtail_rows=interp_headtail_rows,
    )
    assert len(interp_headtail_rows) == 1
    assert interp_headtail_rows[0]["trajectory_id"] == 5
```

Adjust the `cfg=` construction if `ic.build_inference_config_from_params` is
not re-exported from `interpolated_crops.py` (it likely isn't — import
`HeadTailConfig` directly from `hydra_suite.core.inference.config` and
construct one by hand with the fields `_assemble_headtail_result` actually
reads: `candidate_confidence_threshold`, `confidence_threshold`).

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_core_interpolated_crops.py -v -k "detect_apriltags_in_frame_writes_tag_source or flush_headtail_window"`
Expected: FAIL (`_detect_apriltags_in_frame`/`_flush_headtail_window` still
have the old signatures/bodies, or don't exist yet)

- [ ] **Step 3: Implement both replacements**

Delete `_detect_apriltags_in_frame` (697-751) in full and add, at module
scope near the top of the file, `from hydra_suite.core.inference.stages.apriltag import run_apriltag`
and `from hydra_suite.core.inference.stages.crops import extract_aabb_crops`
(alongside the existing imports at the top of the file):

```python
def _detect_apriltags_in_frame(apriltag_model, cfg, frame, obb, tasks, interp_tag_rows):
    """Detect AprilTags in one frame's interpolated crops via the SAME
    ``extract_aabb_crops``/``run_apriltag`` ``Pipeline`` uses for real
    detections (``pipeline.py:389-398``) -- no batch variant exists for
    AprilTag (design spec, "Key architectural finding"), so this stays
    per-frame like today.

    Per the design spec's AprilTag/foreign-suppression decision: unlike the
    old hand-rolled path (which foreign-masked other synthetic tasks' AABB
    regions via ``SUPPRESS_FOREIGN_OBB_REGIONS``), ``extract_aabb_crops`` has
    no suppression parameter at all -- interpolated AprilTag crops lose
    foreign-suppression of other interpolated tasks, deliberately matching
    what real detections already get.
    """
    if not tasks:
        return
    aabb_crops = extract_aabb_crops(frame, obb, padding=cfg.crop_padding)
    result = run_apriltag(aabb_crops, obb, apriltag_model, cfg)
    for tag_id, det_idx in zip(result.tag_ids, result.det_indices):
        if det_idx >= len(tasks):
            continue
        task = tasks[det_idx]
        interp_tag_rows.append(
            {
                "frame_id": int(task["frame_id"]),
                "trajectory_id": int(task["traj_id"]),
                "tag_id": int(tag_id),
            }
        )
```

Note the `interp_tag_rows` row shape drops `center_x`/`center_y`/`hamming`
(the old hand-rolled `_detect_apriltags_in_frame` wrote them, but
`AprilTagResult` doesn't carry hamming, and `merge_interpolated_apriltag_df`
after Task 2 only reads `tag_id` — the artifact CSV's own field list needs
updating in Task 12/14 to match; see Task 12's `_write_interpolation_artifacts`
update). If any other consumer of `interpolated_tags.csv` depends on
`center_x`/`center_y`/`hamming` columns, grep for them before dropping:
`grep -rn "interpolated_tags.csv\|center_x.*center_y" src/hydra_suite/ | grep -v test`.

Delete `_detect_headtail_in_frame` (754-774) in full and add:

```python
def _flush_headtail_window(
    pending_frames,
    pending_obbs,
    pending_tasks_by_frame,
    headtail_model,
    cfg,
    runtime,
    geometry,
    interp_headtail_rows,
):
    """Run head-tail classification over a window via ``run_headtail_batch``
    -- the SAME function ``Pipeline`` calls for real detections
    (``pipeline.py:342-350``). Switches from the old per-frame
    ``HeadTailAnalyzer.analyze_crops`` to the windowed batch path: a
    materially different crop-construction path, registered as an expected
    difference in the design spec's Testing section (verify equivalence
    empirically on the characterization golden, not byte-identity).
    """
    from hydra_suite.core.inference.stages.headtail import run_headtail_batch

    if not pending_frames or headtail_model is None:
        return
    headtail_by_frame = run_headtail_batch(
        pending_frames, pending_obbs, headtail_model, cfg.headtail, runtime, geometry
    )
    for frame_idx, tasks in zip(
        (obb.frame_idx for obb in pending_obbs), pending_tasks_by_frame
    ):
        result = headtail_by_frame.get(frame_idx)
        if result is None:
            continue
        for i, task in enumerate(tasks):
            if i >= len(result.heading_hints):
                continue
            interp_headtail_rows.append(
                {
                    "frame_id": int(task["frame_id"]),
                    "trajectory_id": int(task["traj_id"]),
                    "heading_rad": float(result.heading_hints[i]),
                    "heading_conf": float(result.heading_confidences[i]),
                    "heading_directed": int(result.directed_mask[i]),
                }
            )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_core_interpolated_crops.py -v -k "detect_apriltags_in_frame_writes_tag_source or flush_headtail_window"`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/core/post/interpolated_crops.py tests/test_core_interpolated_crops.py
git commit -m "refactor(post): route interpolated AprilTag/head-tail through run_apriltag/run_headtail_batch"
```

---

## Task 12: Final wiring — frame loop, artifact schema, end-to-end integration

**Files:**
- Modify: `src/hydra_suite/core/post/interpolated_crops.py` — rewrite
  `_process_single_frame` (1218-1344), `_run_frame_tasks_loop`
  (1346-1439), `_compute_frame_corners_and_affines`
  (964-995, now delegates to Task 7's `filter_degenerate_tasks`),
  `_process_single_task`/`_extract_pose_crop` (998-1148, deleted — subsumed
  by Task 10's window flush), `run_interpolated_crops` (1552-1770, update
  the `_init_interpolation_backends` call + cleanup signature), and
  `_write_interpolation_artifacts` (777-904, update the tags/pose/cnn/headtail
  fieldname lists for the new row shapes)
- Test: `tests/test_core_interpolated_crops.py`,
  `tests/test_interpolated_crops_worker.py`,
  `tests/test_interpolated_crops_worker_degenerate_obb_skip.py`,
  `tests/test_interpolated_crops_worker_layer2_fit.py`

**Interfaces:**
- Consumes: everything from Tasks 7-11.
- Produces: `run_interpolated_crops`'s public signature and return-dict
  shape are UNCHANGED (still `{"saved", "gaps", "occluded_rows", ...,
  "pose_csv_path", "tag_csv_path", "cnn_csv_paths", "headtail_csv_path",
  ...}` per `_build_finished_payload`, itself unchanged by this task) — this
  is the integration task that makes the module internally consistent again
  after Tasks 9-11 changed several private-function signatures.

- [ ] **Step 1: Rewrite `_compute_frame_corners_and_affines` to delegate to Task 7**

Replace lines 964-995:

```python
def _compute_frame_corners_and_affines(tasks, geometry, clipping_stats):
    """Degenerate-OBB pre-filter + corner geometry for one frame's tasks.

    Delegates the pre-filter itself to
    ``synthetic_detections.filter_degenerate_tasks`` (design spec, "Error
    handling") -- this wrapper now just also returns the per-task OBB
    corners for callers that still need the raw geometry (e.g. the
    interpolated-crop image-save path in ``_process_single_task``, if that
    still calls this -- see Step 3 below for whether it's still needed
    there after this task's rewrite).
    """
    from hydra_suite.core.post.synthetic_detections import filter_degenerate_tasks
    from hydra_suite.core.individual.geometry import ellipse_to_obb_corners as _e2obb

    kept_tasks = filter_degenerate_tasks(tasks, geometry, clipping_stats)
    corners = [
        _e2obb(t["cx"], t["cy"], t["w"], t["h"], t["theta"]) for t in kept_tasks
    ]
    return kept_tasks, corners
```

Note the return shape changes from `(corners, affines)` to
`(kept_tasks, corners)` — every caller must be updated (Step 2-3 below do
this). This is deliberate: since Task 10/11 build a synthetic `OBBResult`
from `kept_tasks` directly and let the stage layer own affine computation
internally (as `Pipeline` does), the adapter no longer needs to hand-carry
`_M_pose`/`cw_pose`/`ch_pose` tuples the way `_extract_pose_crop` used to.

- [ ] **Step 2: Rewrite the per-image-save path (formerly part of `_process_single_task`)**

The image-save side of `_process_single_task` (lines 1018-1065 — building
`filename`/`interp_rows`/`roi_rows`/`roi_corners` via `gen.save_interpolated_crop`)
is UNCHANGED behavior and does not go through the new inference stage
functions (it's `IndividualDatasetGenerator`'s own crop-save path, out of
scope per the spec's Architecture section — only pose/CNN/AprilTag/head-tail
*inference* moves). Keep this logic, but inline it directly into
`_process_single_frame` (Step 3) instead of as a separate
`_process_single_task` function, since the pose/CNN crop-extraction half of
the old function (lines 1066-1084, calling `_extract_pose_crop`) is deleted
outright — that inference path is now Task 10's windowed flush, not a
per-task call. Delete `_process_single_task` (998-1017, 1066-1084) and
`_extract_pose_crop` (1087-1148) in full, keeping only the
image-save-specific body (1018-1065) to be inlined next.

- [ ] **Step 3: Rewrite `_process_single_frame` and `_run_frame_tasks_loop`**

Replace `_process_single_frame` (1218-1344) with:

```python
def _process_single_frame(
    params,
    should_stop,
    progress,
    f,
    idx,
    frame,
    total_frames,
    frame_tasks,
    gen,
    save_interpolated_outputs,
    geometry,
    clipping_stats,
    apriltag_model,
    apriltag_cfg,
    interp_saved,
    interp_rows,
    roi_rows,
    roi_corners,
    interp_tag_rows,
    _pending_frames,
    _pending_obbs,
    _pending_tasks_by_frame,
):
    def _stop():
        return bool(should_stop()) if should_stop is not None else False

    def _emit(v, m):
        if progress is not None:
            progress(v, m)

    kept_tasks, corners = _compute_frame_corners_and_affines(
        frame_tasks[f], geometry, clipping_stats
    )

    for task_idx, task in enumerate(kept_tasks):
        filename = ""
        if save_interpolated_outputs:
            filename = gen.save_interpolated_crop(
                frame=frame,
                frame_id=task["frame_id"],
                cx=task["cx"],
                cy=task["cy"],
                w=task["w"],
                h=task["h"],
                theta=task["theta"],
                traj_id=task["traj_id"],
                interp_from=task["interp_from"],
                interp_index=task["interp_index"],
                interp_total=task["interp_total"],
                canonical_affine=None,
            )
        if save_interpolated_outputs and filename:
            interp_saved += 1
            interp_rows.append(
                {
                    "frame_id": int(task["frame_id"]),
                    "trajectory_id": int(task["traj_id"]),
                    "filename": filename,
                    "interp_from_start": int(task["interp_from"][0]),
                    "interp_from_end": int(task["interp_from"][1]),
                    "interp_index": int(task["interp_index"]),
                    "interp_total": int(task["interp_total"]),
                }
            )
            roi_rows.append(
                {
                    "frame_id": int(task["frame_id"]),
                    "trajectory_id": int(task["traj_id"]),
                    "filename": filename,
                    "cx": float(task["cx"]),
                    "cy": float(task["cy"]),
                    "w": float(task["w"]),
                    "h": float(task["h"]),
                    "theta": float(task["theta"]),
                    "interp_from_start": int(task["interp_from"][0]),
                    "interp_from_end": int(task["interp_from"][1]),
                    "interp_index": int(task["interp_index"]),
                    "interp_total": int(task["interp_total"]),
                }
            )
            roi_corners.append(corners[task_idx])

    if kept_tasks:
        from hydra_suite.core.post.synthetic_detections import build_synthetic_obb_result

        obb = build_synthetic_obb_result(f, kept_tasks)
        if apriltag_model is not None:
            _detect_apriltags_in_frame(
                apriltag_model, apriltag_cfg, frame, obb, kept_tasks, interp_tag_rows
            )
        _pending_frames.append(frame)
        _pending_obbs.append(obb)
        _pending_tasks_by_frame.append(kept_tasks)

    if idx % 25 == 0 or idx == total_frames:
        progress_pct = int((idx / total_frames) * 100)
        _emit(progress_pct, f"Interpolating occlusions... {idx}/{total_frames}")
    return interp_saved
```

`canonical_affine=None` is passed to `gen.save_interpolated_crop` now that
the caller no longer pre-computes a `(M, cw, ch)` tuple per task; confirm
`save_interpolated_crop` accepts `None` and recomputes its own affine
internally when not provided (`grep -n "def save_interpolated_crop" -A 30
src/hydra_suite/core/individual/dataset/generator.py`) — if it does NOT
tolerate `None`, pass `canonical_affine=canonical_affine(corners[task_idx],
geometry)[0]` instead (import `canonical_affine` from
`canonicalization.geometry` at the top of the file) so the image-save path's
behavior is unchanged from today.

Replace `_run_frame_tasks_loop` (1346-1439) with:

```python
def _run_frame_tasks_loop(
    params,
    should_stop,
    progress,
    frame_tasks,
    cap,
    gen,
    save_interpolated_outputs,
    geometry,
    clipping_stats,
    cfg,
    runtime,
    pose_model,
    cnn_models,
    cnn_labels,
    apriltag_model,
    headtail_model,
    interp_saved,
    interp_rows,
    roi_rows,
    roi_corners,
    interp_pose_rows,
    interp_tag_rows,
    interp_cnn_rows,
    interp_headtail_rows,
    profiler,
):
    def _stop():
        return bool(should_stop()) if should_stop is not None else False

    needed_frames = sorted(frame_tasks.keys())
    total_frames = len(needed_frames)
    window_batch_size = int(params.get("INTERP_POSE_INFERENCE_BATCH_SIZE", 64))
    _pending_frames: list = []
    _pending_obbs: list = []
    _pending_tasks_by_frame: list = []

    def _flush_window():
        _flush_pose_cnn_window(
            _pending_frames, _pending_obbs, _pending_tasks_by_frame,
            pose_model, cnn_models, cnn_labels, cfg, runtime, geometry,
            interp_pose_rows, interp_cnn_rows, profiler,
        )
        _flush_headtail_window(
            _pending_frames, _pending_obbs, _pending_tasks_by_frame,
            headtail_model, cfg, runtime, geometry, interp_headtail_rows,
        )
        _pending_frames.clear()
        _pending_obbs.clear()
        _pending_tasks_by_frame.clear()

    _prefetcher = _build_prefetcher(cap, needed_frames, total_frames)
    _prefetcher.start()
    for idx in range(1, total_frames + 1):
        if _stop():
            _prefetcher.stop()
            return None
        _pf_item = _prefetcher.read()
        if _pf_item is None:
            break
        f, ret, frame = _pf_item
        if not ret or frame is None:
            continue
        result = _process_single_frame(
            params, should_stop, progress, f, idx, frame, total_frames,
            frame_tasks, gen, save_interpolated_outputs, geometry,
            clipping_stats, apriltag_model, cfg.apriltag,
            interp_saved, interp_rows, roi_rows, roi_corners,
            interp_tag_rows, _pending_frames, _pending_obbs,
            _pending_tasks_by_frame,
        )
        if result is None:
            _prefetcher.stop()
            return None
        interp_saved = result
        if len(_pending_frames) >= window_batch_size or idx == total_frames:
            if _stop():
                _prefetcher.stop()
                return None
            _flush_window()
        if idx % 25 == 0:
            del frame
    _prefetcher.stop()
    return interp_saved
```

- [ ] **Step 4: Update `run_interpolated_crops`'s call sites and cleanup**

In `run_interpolated_crops` (1552-1770):

Replace the `_init_interpolation_backends(...)` call and its unpacking
(lines 1657-1666-ish) to match Task 9's new return shape:

```python
        if frame_tasks:
            (
                cfg,
                runtime,
                pose_model,
                apriltag_model,
                cnn_models,
                cnn_labels,
                headtail_model,
            ) = _init_interpolation_backends(params, output_dir, geometry)
            interp_saved = _run_frame_tasks_loop(
                params, should_stop, progress, frame_tasks, cap, gen,
                save_interpolated_outputs, geometry, clipping_stats,
                cfg, runtime, pose_model, cnn_models, cnn_labels,
                apriltag_model, headtail_model,
                interp_saved, interp_rows, roi_rows, roi_corners,
                interp_pose_rows, interp_tag_rows, interp_cnn_rows,
                interp_headtail_rows, profiler,
            )
```

Update the `_cleanup_backends` call in the `finally` block (1761-1769) to
close the new model objects instead of the old backend variables — replace:

```python
    pose_backend = None
    detection_cache = None
    cap = None
    cnn_backends = []
    cnn_labels = []
    apriltag_detector = None
    headtail_analyzer = None
    pose_kpt_source_names = []
    pose_kpt_labels = []
    interp_cnn_rows = {}
```

(the pre-try initialization block) with:

```python
    pose_model = None
    detection_cache = None
    cap = None
    cnn_models = []
    cnn_labels = []
    apriltag_model = None
    headtail_model = None
    interp_cnn_rows = {}
```

and update `_cleanup_backends`'s signature/body (907-935) to close
`pose_model`/`apriltag_model`/`cnn_models`/`headtail_model` (each has a
`.close()` method per their dataclass definitions in `stages/*.py`) instead
of the old backend objects:

```python
def _cleanup_backends(cap, detection_cache, pose_model, apriltag_model, cnn_models, headtail_model):
    """Safely close all loaded resources."""
    for resource in (cap, detection_cache, pose_model, apriltag_model, headtail_model):
        if resource is not None:
            try:
                if hasattr(resource, "release"):
                    resource.release()
                elif hasattr(resource, "close"):
                    resource.close()
            except Exception:
                pass
    for model in cnn_models or []:
        try:
            model.close()
        except Exception:
            pass
```

and its call site in the `finally` block:

```python
    finally:
        _cleanup_backends(cap, detection_cache, pose_model, apriltag_model, cnn_models, headtail_model)
```

- [ ] **Step 5: Update `_write_interpolation_artifacts`'s AprilTag/head-tail fieldnames**

In `_write_interpolation_artifacts` (777-904), the `tag_csv_path` write
(860-872) currently lists fieldnames
`["frame_id", "trajectory_id", "tag_id", "center_x", "center_y", "hamming"]`
— Task 11's new `_detect_apriltags_in_frame` only produces
`frame_id`/`trajectory_id`/`tag_id`. Replace:

```python
    if interp_tag_rows:
        result["tag_csv_path"] = _write_csv_artifact(
            parent / "interpolated_tags.csv",
            [
                "frame_id",
                "trajectory_id",
                "tag_id",
                "center_x",
                "center_y",
                "hamming",
            ],
            interp_tag_rows,
        )
```

with:

```python
    if interp_tag_rows:
        result["tag_csv_path"] = _write_csv_artifact(
            parent / "interpolated_tags.csv",
            ["frame_id", "trajectory_id", "tag_id"],
            interp_tag_rows,
        )
```

Verify with `_write_csv_artifact`'s implementation
(`grep -n "def write_csv_artifact" -A 15 src/hydra_suite/core/post/merge.py`)
whether it errors or silently drops on a row-vs-fieldname mismatch — if it
requires exact key match, this is the correct fix; if it tolerates extra
keys, no change is strictly required but keep the trimmed list for clarity.

The `pose_csv_path` fieldnames (848-858) and `headtail_csv_path` fieldnames
(891-902) are unchanged (`interp_pose_rows`/`interp_headtail_rows` still
carry the same field names as before, plus `PoseSource` added by Task 10 —
add `"PoseSource"` to the `pose_fieldnames` list at line 855, right after
`*POSE_SUMMARY_COLUMNS,`).

The `cnn_csv_paths` write (874-889) already discovers fieldnames
dynamically from the row dicts (`for _key in _cnn_row: if _key not in
fieldnames: fieldnames.append(_key)`), so it needs no change — the new
`CNN_<label>_Source` key Task 10 adds will be picked up automatically.

- [ ] **Step 6: Run the full interpolated-crops test suite**

```bash
python -m pytest tests/test_core_interpolated_crops.py tests/test_interpolated_crops_worker.py tests/test_interpolated_crops_worker_degenerate_obb_skip.py tests/test_interpolated_crops_worker_layer2_fit.py tests/test_synthetic_detections.py tests/test_properties_export.py tests/test_session_policy.py tests/test_oriented_video.py tests/test_rich_export.py -v
```

Expected: PASS. Some of the pre-existing tests
(`test_interpolated_crops_worker_degenerate_obb_skip.py`,
`test_interpolated_crops_worker_layer2_fit.py`) assert on the OLD
hand-rolled internals (`_extract_pose_crop`, `_flush_pose_batch`,
`_compute_frame_corners_and_affines`'s old `(corners, affines)` return
shape) — read each failing test, and where it's asserting on an
implementation detail this task deliberately changed (not a behavior this
task must preserve), update the test to assert the equivalent behavior
through the new functions (`filter_degenerate_tasks` for the degenerate-skip
test, `_flush_pose_cnn_window`'s `PoseSource` stamping for the layer2-fit
test) rather than deleting the test's intent.

- [ ] **Step 7: Commit**

```bash
git add src/hydra_suite/core/post/interpolated_crops.py tests/test_core_interpolated_crops.py tests/test_interpolated_crops_worker.py tests/test_interpolated_crops_worker_degenerate_obb_skip.py tests/test_interpolated_crops_worker_layer2_fit.py
git commit -m "refactor(post): wire interpolated_crops.py fully onto the Pipeline stage functions"
```

---

## Task 13: Characterization golden + pre-registered expected-difference test

**Files:**
- Create: `tests/test_interpolated_crops_characterization_golden.py`
- Create: `tests/fixtures/interpolated_crops_golden/` (golden CSVs captured
  from pre-change `main`)

**Interfaces:** none new — this is a verification harness over
`run_interpolated_crops`'s existing public contract.

- [ ] **Step 1: Capture the pre-change golden, BEFORE any of Tasks 7-12 land**

This step must run against `main` at the commit immediately before Task 7's
first commit — if Tasks 7-12 are already committed by the time this task
starts, check out that earlier commit in a throwaway worktree first:

```bash
git worktree add .worktrees/golden-capture -b golden-capture-tmp <commit-before-task-7>
cd .worktrees/golden-capture
```

Per the design spec's Testing item 3 (adversarial-review G10): confirm
whether any existing fixture in `tools/equivalence/fixtures/clips/` is
occlusion-heavy with CNN+AprilTag+head-tail signals all active
simultaneously. Check clip configs under `tools/equivalence/fixtures/` for
one that enables all four (`ENABLE_POSE_EXTRACTOR`, `CNN_CLASSIFIERS`,
`USE_APRILTAGS`, `YOLO_HEADTAIL_MODEL_PATH` all set) with a meaningful
occluded-row count. If none qualifies, build a small synthetic
occluded-CSV harness instead of a full clip: hand-write a tracking CSV with
several `State=occluded` runs (some with the row's own `X`/`Y`/`Theta`
pre-filled to exercise the NaN-triggered priority from Task 8, some left
NaN), a tiny synthetic video (a few solid-color frames via
`cv2.VideoWriter`), and a `params` dict enabling all four signal types
against small/fast model fixtures already used elsewhere in the test suite
(check `tests/fixtures/` for existing small pose/CNN/headtail/apriltag model
files other tests already load, to avoid downloading anything new).

Run `run_interpolated_crops` against this fixture/CSV pair and save its four
artifact CSVs (`interpolated_pose.csv`, `interpolated_cnn_<label>.csv`,
`interpolated_tags.csv`, `interpolated_headtail.csv`) to
`tests/fixtures/interpolated_crops_golden/` in the MAIN worktree (not the
throwaway one), then remove the throwaway worktree:

```bash
cd -
git worktree remove --force .worktrees/golden-capture
git branch -D golden-capture-tmp
```

- [ ] **Step 2: Write the diff test against the pre-registered expected-difference list**

```python
# tests/test_interpolated_crops_characterization_golden.py
"""Characterization golden for the interpolated-crop inference unification.

Diffs the overhauled ``run_interpolated_crops`` output against a golden
captured from pre-change ``main`` on an occlusion-heavy fixture. Per the
design spec's Testing section, four specific differences are EXPECTED and
must NOT fail this test -- they are registered here, not silently ignored:
CNN crop identity (now unmasked/independent of pose), AprilTag crop masking
(now unmasked, matching real-detection parity), pose crop LSB rounding
(truncate-then-mask vs round-then-mask, ~1 pixel), and head-tail
crop-construction path (HeadTailAnalyzer.analyze_crops -> run_headtail_batch,
verified by tolerance not byte-identity). Anything else diverging is a
regression.
"""
import pandas as pd
import pytest

GOLDEN_DIR = "tests/fixtures/interpolated_crops_golden"


def _run_current(tmp_path):
    """Run today's (post-unification) run_interpolated_crops on the same
    fixture/CSV/video/params used to capture the golden, writing artifacts
    into tmp_path. Returns the finished-payload dict."""
    # Mirror the fixture setup Step 1 used (same CSV, video, params) so the
    # only variable is the code under test.
    ...


def test_pose_output_matches_golden_within_registered_differences(tmp_path):
    golden = pd.read_csv(f"{GOLDEN_DIR}/interpolated_pose.csv")
    payload = _run_current(tmp_path)
    current = pd.read_csv(payload["pose_csv_path"])
    merged = golden.merge(
        current, on=["frame_id", "trajectory_id"], suffixes=("_golden", "_current")
    )
    assert len(merged) == len(golden), "row count changed unexpectedly"
    for kpt_col in [c for c in golden.columns if c.startswith("PoseKpt_") and c.endswith("_X")]:
        base = kpt_col
        # Registered difference: pose crop LSB rounding -> allow +/-1px.
        diff = (merged[f"{base}_current"] - merged[f"{base}_golden"]).abs()
        assert (diff <= 1.5).all(), f"{base} diverges beyond the registered LSB tolerance"


def test_cnn_output_diverges_only_on_frames_with_multiple_interpolated_tasks(tmp_path):
    """CNN crop identity is a REGISTERED difference only on frames with >=2
    simultaneous interpolated tasks (foreign-masking no longer shared with
    pose). On single-task frames CNN output must be unchanged."""
    ...


def test_headtail_output_agrees_within_tolerance_not_byte_identity(tmp_path):
    """run_headtail_batch is a materially different crop-construction path
    than HeadTailAnalyzer.analyze_crops -- verify agreement empirically."""
    ...


def test_tag_output_unmasked_relative_to_golden_on_multi_task_frames(tmp_path):
    """AprilTag crops lose foreign-suppression -- a registered, deliberate
    difference. Assert it's understood (tag ids may legitimately differ on
    multi-task frames) rather than asserting byte-identity."""
    ...
```

Fill in `_run_current` and the four test bodies once Step 1's concrete
fixture (real clip or synthetic harness) is chosen — the exact assertions
depend on which path Step 1 took. The four test names/docstrings above are
the required coverage; do not reduce the four registered differences to
fewer checks.

- [ ] **Step 3: Run the test**

Run: `python -m pytest tests/test_interpolated_crops_characterization_golden.py -v`
Expected: PASS. Any row/column diverging OUTSIDE the four registered
differences is a real regression — stop and fix it in the relevant Task
7-12 commit (do not paper over it here) before proceeding.

- [ ] **Step 4: Commit**

```bash
git add tests/test_interpolated_crops_characterization_golden.py tests/fixtures/interpolated_crops_golden/
git commit -m "test: add characterization golden for interpolated-crop inference unification"
```

---

## Task 14: Equivalence harness verification (MPS + CUDA)

**Files:** none (verification-only task; no code changes).

**Interfaces:** none.

- [ ] **Step 1: Kill stale sleap/hydra processes**

Per `CLAUDE.md`'s "Before any heavy run" convention:

```bash
ps aux | grep -i "sleap\|hydra" | grep -v grep
# review the list; kill only sleap/hydra processes, never anything else
```

- [ ] **Step 2: Run the full test suite delta**

```bash
python -m pytest tests/ -x -q -k "not test_" --collect-only  # sanity: verify collection isn't broken first
make pytest
```

Expected: no new failures beyond the pre-existing baseline (per memory
`project_test_suite_hardening`/`project_main_suite_blockers` — batch by file
if the whole-suite run hangs on classkit modal-dialog tests).

- [ ] **Step 3: Run the equivalence matrix on this box (MPS)**

Per `CLAUDE.md`'s "Equivalence & Benchmark Verification" section:

```bash
conda activate hydra-mps
git fetch origin --tags
git worktree add --detach .worktrees/equiv-legacy legacy/main
REPO=$PWD WT=$PWD \
  MAIN_SRC=$PWD/.worktrees/equiv-legacy/src WT_SRC=$PWD/src \
  OUT=/tmp/equiv_interp_crops RUNTIME=mps \
  bash tools/equivalence/run_matrix.sh
git worktree remove --force .worktrees/equiv-legacy && git worktree prune
```

This clip matrix exercises the surrounding tracking pipeline (not the
interpolated-crop post-pass directly, since no fixture clip triggers
`run_interpolated_crops` today per Task 13's finding) — its job here is
confirming NO COLLATERAL regression to the non-interpolated tracking path
from this change (per the design spec's Testing item 4). Verify every
clip's positions/tracking CSVs are still at/near the DETERMINISM floor.

- [ ] **Step 4: Run the equivalence matrix on mehek (CUDA)**

```bash
ssh rutalab@mehek.taild08eb9.ts.net
cd ~/hydra-suite && git fetch origin && git checkout <this-branch-or-sha>
source ~/mambaforge/etc/profile.d/conda.sh && conda activate hydra-cuda
git worktree add --detach .worktrees/equiv-legacy legacy/main
REPO=$PWD WT=$PWD MAIN_SRC=$PWD/.worktrees/equiv-legacy/src WT_SRC=$PWD/src \
  OUT=/tmp/equiv_interp_crops RUNTIME=cuda nohup bash tools/equivalence/run_matrix.sh > /tmp/equiv_cuda.log 2>&1 &
```

Wait for completion, then check `/tmp/equiv_cuda.log` for the same
EQUIVALENT verdicts as Step 3.

- [ ] **Step 5: Report results**

Summarize pass/fail for both platforms to the user before considering this
plan complete. If either platform shows a divergence beyond documented noise
(bistable head/tail π-flips, per memory `project_migration_verification`),
stop and investigate before merging — do not silently accept it.

---

## Post-merge doc lifecycle (do NOT do this until the branch is merged to `main`)

Per `CLAUDE.md`'s "Docs lifecycle" convention: once this plan's branch is
merged to `main`, `git mv` this plan and its spec
(`docs/superpowers/specs/2026-08-19-interpolated-crops-inference-unification-design.md`)
into their matching `done/` subfolders in the same commit/PR, updating only
the spec's `**Status:**` header to a `Shipped — merged to main (<sha>)` note
(no other content rewrite).
