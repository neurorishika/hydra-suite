# Combined core/post Slice — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** (Part 1) Make the trajectory head/tail orientation resolution deterministic/robust to sub-pixel input noise; (Part 2) vectorize the four row-wise pandas postproc hotspots (~28% of wall) byte-identically.

**Architecture:** Two sequenced parts in `core/post` + `core/individual/pose`. Part 1 replaces the float-tied terminal choice in `_fix_heading_globally` with a raw-directed-heading majority anchor (changes θ deterministically). Part 2 replaces per-row pandas loops with column-wise/numpy/groupby equivalents, byte-identical to the post-Part-1 output, guarded by a committed value-level golden.

**Tech Stack:** Python, NumPy, pandas, pytest; the `tools/equivalence/` harness (MPS on this box, CUDA on mehek) as the true gate.

## Global Constraints

- **Part 1 is NOT byte-identical** to current main (it changes θ on the ~12–15% of rows that previously π-flipped). Its acceptance is (a) determinism under sub-pixel perturbation and (b) correct head preserved (no systematic 180° inversion).
- **Part 2 IS byte-identical** to the post-Part-1 output: the exported `_tracking_final.csv`, `_with_individual.csv`, and `_tracks.csv` must be byte-for-byte unchanged. Do not alter dtypes/formatting at the write boundary (float32↔float conf round-trips, `Int64` NaN-aware rounding, flag/source strings).
- Preserve semantic ordering in temporal postproc: suppression→interpolation→recompute, X-before-Y; exact flag-string token order and `|`-join.
- Equivalence harness MPS + CUDA must pass at the end. Kill stale sleap/hydra procs before any heavy run. Worktree tests need `PYTHONPATH=$PWD/src`.
- One commit per task minimum; each task ends green.

---

## PART 1 — Orientation stability

### Task 1: Deterministic raw-heading anchor in `_fix_heading_globally`

**Files:**
- Modify: `src/hydra_suite/core/post/processing.py` (`_fix_heading_globally`, ~lines 4153–4162, the terminal selection + backtrack)
- Test: `tests/test_heading_global_anchor.py` (create)

**Interfaces:**
- `_fix_heading_globally(theta: np.ndarray) -> np.ndarray` — signature unchanged; only the terminal global-orientation choice changes.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_heading_global_anchor.py
import numpy as np
from hydra_suite.core.post.processing import _fix_heading_globally


def _majority_orientation_matches_raw(theta):
    out = _fix_heading_globally(theta)
    # fraction of valid frames whose corrected heading stays near the raw heading
    d = np.abs(np.angle(np.exp(1j * (out - theta))))  # circular distance to raw
    return np.mean(d[~np.isnan(theta)] < (np.pi / 2))


def test_anchor_picks_raw_majority_orientation():
    # 20 frames pointing ~0.1 rad, 3 spurious 180-flips: majority head call is 0.1
    rng = np.random.default_rng(0)
    theta = np.full(23, 0.1) + rng.normal(0, 1e-3, 23)
    theta[5] = (0.1 + np.pi) % (2 * np.pi)
    theta[11] = (0.1 + np.pi) % (2 * np.pi)
    theta[17] = (0.1 + np.pi) % (2 * np.pi)
    # >= half the frames must end up agreeing with the raw majority (0.1), NOT flipped
    assert _majority_orientation_matches_raw(theta) >= 0.5


def test_anchor_is_stable_under_subpixel_jitter():
    # The global orientation must NOT flip when inputs move by ~1e-3 rad.
    rng = np.random.default_rng(1)
    base = np.full(40, 2.0)
    base[::7] = (2.0 + np.pi) % (2 * np.pi)  # a minority of flips
    outs = []
    for k in range(8):
        jitter = rng.normal(0, 1e-3, base.size)
        out = _fix_heading_globally(base + jitter)
        # record global orientation as the mean cos/sin (sign-stable summary)
        outs.append(np.mean(np.cos(out)))  # orientation-sign summary
    outs = np.array(outs)
    # all runs agree on orientation sign (no π-flip across jitter seeds)
    assert np.all(np.sign(outs) == np.sign(outs[0]))
```

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/test_heading_global_anchor.py -q`
Expected: `test_anchor_is_stable_under_subpixel_jitter` FAILS (current terminal `dp[0]<=dp[1]` flips under jitter). `test_anchor_picks_raw_majority_orientation` may pass or fail depending on the float residual — both must pass after the fix.

- [ ] **Step 3: Implement the anchor**

Replace the terminal block (currently):
```python
    states = [None] * len(valid_idx)
    states[-1] = 0 if dp[0] <= dp[1] else 1
    for vi in range(len(valid_idx) - 1, 0, -1):
        states[vi - 1] = parent[vi][states[vi]]
```
with:
```python
    # The pairwise DP cost is invariant under a global pi-flip, so dp[0]==dp[1]
    # in exact arithmetic and the old `dp[0] <= dp[1]` terminal choice was decided
    # by ~1e-15 float rounding -- any ~1e-3 input perturbation flipped the WHOLE
    # trajectory by pi. Anchor the global orientation to the RAW directed headings
    # instead: back-track BOTH terminal states and keep the assignment that flips
    # FEWER frames away from their raw head call (summed circular distance to raw
    # == pi * flip-count), a robust integer majority. Exact tie -> terminal state 0.
    def _backtrack(term_state):
        st = [None] * len(valid_idx)
        st[-1] = term_state
        for vj in range(len(valid_idx) - 1, 0, -1):
            st[vj - 1] = parent[vj][st[vj]]
        return st

    states0 = _backtrack(0)
    states1 = _backtrack(1)
    states = states0 if sum(states0) <= sum(states1) else states1
```

- [ ] **Step 4: Run to verify it passes**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/test_heading_global_anchor.py -q`
Expected: PASS (both tests).

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/core/post/processing.py tests/test_heading_global_anchor.py
git commit -m "fix(post): anchor global head/tail orientation to raw-heading majority

_fix_heading_globally's terminal choice was a float-tied dp[0]<=dp[1] (dp[0]==dp[1]
by global-flip symmetry), so a ~1e-3 input perturbation flipped whole trajectories
by pi (the 12-15% surfaced by the crop-warp slice). Choose the global orientation
that flips fewer frames from their raw directed headings -- a robust integer
majority that preserves the correct head."
```

---

### Task 2: Re-baseline θ-dependent goldens + confirm correct head

Part 1 intentionally changes θ on the previously-bistable rows, so committed goldens that pin θ must be regenerated and their orientation sanity-checked (not blindly overwritten).

**Files:**
- Modify (regen): `tests/fixtures/postproc/resolve/fuzz_seed_*/expected.csv` (if θ columns changed), `tests/goldens/user_mode/{ant_pose_headtail,ant_cnn_identity}_tracks.csv`
- Inspect: `tests/test_postproc_equivalence.py`, `tests/test_user_mode_golden.py`, `tests/helpers/postproc_runner.py`

- [ ] **Step 1: Identify which goldens changed**

Run the affected suites and capture the diffs:
```bash
PYTHONPATH=$PWD/src python -m pytest tests/test_postproc_equivalence.py tests/test_user_mode_golden.py -q
```
Expected: failures ONLY in θ / heading_deg columns on rows that previously π-flipped (positions, IDs, frames unchanged). If ANY non-θ column changed, STOP — that is an unintended regression, not the orientation re-baseline.

- [ ] **Step 2: Verify the new orientation is correct, not inverted**

For each changed golden row set, confirm the new θ agrees with the raw directed head call (the pose/head-tail heading for that frame), i.e. the anchor picked the true head. Spot-check ≥5 flipped trajectories against their raw `Theta` in the forward CSV. Document the check in the commit message.

- [ ] **Step 3: Regenerate the goldens**

Use the repo's golden-refresh path (e.g. the runner in `tests/helpers/postproc_runner.py` / the documented `--update-golden` mechanism if present; otherwise regenerate via the same script that produced them). Do NOT hand-edit CSVs.

- [ ] **Step 4: Run to verify green**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/test_postproc_equivalence.py tests/test_user_mode_golden.py -q`
Expected: PASS against the regenerated goldens.

- [ ] **Step 5: Commit**

```bash
git add tests/fixtures/postproc tests/goldens/user_mode
git commit -m "test(post): re-baseline theta goldens for stable orientation anchor

Only heading/theta columns on previously-bistable rows change; positions/IDs/frames
unchanged. Spot-checked N flipped trajectories against raw directed head calls -- new
orientation matches the true head (no systematic 180 inversion)."
```

---

### Task 3: Part-1 acceptance — determinism under perturbation (equivalence)

Confirms the crop-perturbation no longer π-flips final θ, using the same MPS attribution that surfaced the bug (parent-vs-Slice-A-style perturbation). Not a pytest — a documented harness run. If residual θ instability remains, apply the secondary hardening (`processing.py:4144` strict `<` → epsilon-tolerant preferring "keep"; `_merge_angle_mean` `1e-12` bias) and re-run.

**Files:** none (verification), unless secondary hardening is needed (`src/hydra_suite/core/post/processing.py`).

- [ ] **Step 1: Run a perturbation-determinism equivalence on MPS**

Run the tracker on `ant_pose_headtail` twice: once normally, once with a deliberate ~1e-3 θ / sub-pixel crop perturbation (or the parent-vs-current crop pair). Compare final θ.
```bash
conda activate hydra-mps
pkill -f "sleap_service_[0-9]"; pkill -f "conda run -n sleap"
# (use tools/equivalence compare on the two _tracking_final.csv outputs)
```
Expected: final θ `theta_mean` at the determinism floor (~0), NOT ~0.4 — the whole-trajectory π-flips are gone.

- [ ] **Step 2: If residual instability remains, apply secondary hardening**

In `_fix_heading_globally`'s inner transition (`cost < best_cost`), prefer the previous state on near-ties:
```python
                if cost < best_cost - 1e-9:
                    best_cost = cost
                    best_prev_s = prev_s
```
And in `_merge_angle_mean`, replace the `1e-12` bias with a principled epsilon (e.g. `1e-6`) consistent with the anchor. Add a unit test that a near-π/2 pair resolves deterministically under 1e-3 jitter. Only include this step if Step 1 showed residual flips.

- [ ] **Step 3: Record the result**

```bash
git commit --allow-empty -m "verify(post): final theta deterministic under sub-pixel perturbation (Part 1 acceptance)"
```

---

## PART 2 — Postproc vectorization (byte-identical to post-Part-1 output)

### Task 4: Capture value-level rich-CSV golden + tighten the golden test

Closes the gap where `test_user_mode_golden.py` asserts only the column set. This golden is the byte-identical oracle for Tasks 5–8.

**Files:**
- Create: `tests/goldens/rich_export/{ant_pose_headtail,ant_cnn_identity}_with_individual.csv` (committed golden, captured on the post-Part-1 tree)
- Modify: `tests/test_user_mode_golden.py` (add value-level assertion) OR create `tests/test_rich_export_golden.py`

- [ ] **Step 1: Write the golden test (points at not-yet-committed golden)**

A test that runs the Debug rich export on the fixtures and asserts the produced `_with_individual.csv` equals the committed golden byte-for-byte (same normalization the harness uses: read as text, compare). It fails first (golden absent).

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/test_rich_export_golden.py -q`
Expected: FAIL (golden file missing).

- [ ] **Step 3: Capture the golden on the current (post-Part-1) tree**

Generate the rich CSVs from the fixtures and commit them as the golden. Verify they are non-empty and contain the full schema (pose triples, identity block, quality/temporal columns).

- [ ] **Step 4: Run to verify it passes**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/test_rich_export_golden.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/goldens/rich_export tests/test_rich_export_golden.py
git commit -m "test(post): commit value-level rich _with_individual.csv golden (vectorization oracle)"
```

---

### Task 5: Vectorize hotspot #3 — identity evidence `.apply(axis=1)`

Isolated, lowest risk. Byte-identical guarded by the Task-4 golden + `tests/test_identity_postprocess.py` / `tests/test_core_identity_postprocess_df.py`.

**Files:**
- Modify: `src/hydra_suite/core/individual/postprocess_df.py:153-156` (`_row_sources` :76-108, `_row_conflict` :110-124, `_row_top_evidence` :126-151)
- Test: `tests/test_identity_postprocess.py`, `tests/test_core_identity_postprocess_df.py`, `tests/test_rich_export_golden.py` (all must stay green)

**Semantic contract (preserve exactly):** `EVIDENCE_SOURCES` = sorted-set join (e.g. `"apriltag,cnn,offline"`) of present evidence sources; `EVIDENCE_CONFLICT_FLAG` = `int` 0/1; top-evidence = `(label, conf)` argmax over CNN conf columns (tag/apriltag precedence as in `_row_top_evidence`), with NaN/empty handling identical.

- [ ] **Step 1: Confirm the guard tests pass on current code**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/test_identity_postprocess.py tests/test_core_identity_postprocess_df.py tests/test_rich_export_golden.py -q`
Expected: PASS (baseline).

- [ ] **Step 2: Vectorize**

Replace the three `.apply(axis=1)` calls with column-wise equivalents: boolean-column combination for sources+conflict; `out[cnn_conf_cols].idxmax(axis=1)` + gather for top-evidence. Preserve sorted-set join order, `astype(int)` on conflict, and NaN→`np.nan`/empty semantics exactly.

- [ ] **Step 3: Run the guard tests**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/test_identity_postprocess.py tests/test_core_identity_postprocess_df.py tests/test_rich_export_golden.py -q`
Expected: PASS (byte-identical golden + unit behavior). If the golden diffs, the vectorization diverged — fix until identical; do not edit the golden.

- [ ] **Step 4: Commit**

```bash
git add src/hydra_suite/core/individual/postprocess_df.py
git commit -m "perf(post): vectorize identity-evidence columns (byte-identical)"
```

---

### Task 6: Vectorize hotspot #4 — pose calibration `iterrows()`

Isolated, low risk. Guarded by `tests/test_pose_quality.py` calibration tests + golden.

**Files:**
- Modify: `src/hydra_suite/core/individual/pose/quality.py` (`_collect_body_lengths` ~:381, `_accumulate_edge_samples` ~:494; both `for _, row in high_conf_df.iterrows(): _extract_keypoints_from_row(row, ...)`)
- Test: `tests/test_pose_quality.py`, `tests/test_rich_export_golden.py`

**Semantic contract:** produce the same body-length / per-edge sample lists (same rows, same order) → identical `np.median`/MAD priors.

- [ ] **Step 1: Baseline guard tests pass**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/test_pose_quality.py -q`
Expected: PASS.

- [ ] **Step 2: Vectorize**

Stack the high-conf subset's `(M,K,3)` keypoints once and compute body lengths / edge distances vectorized, yielding the same sample sequence the `iterrows` path produced (preserve row order and the NaN/valid filtering).

- [ ] **Step 3: Run guard tests**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/test_pose_quality.py tests/test_rich_export_golden.py -q`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add src/hydra_suite/core/individual/pose/quality.py
git commit -m "perf(post): vectorize pose calibration sample collection (byte-identical)"
```

---

### Task 7: Vectorize hotspot #1 — `apply_quality_to_dataframe` row loop

The largest hotspot. MED byte-identical risk (float32 round-trip + flag-string order).

**Files:**
- Modify: `src/hydra_suite/core/individual/pose/quality.py:729-774` (+ helpers `_extract_keypoints_from_row` :968-992, `assess_pose_row` :270, `_compute_quality`)
- Test: `tests/test_pose_quality.py`, `tests/test_rich_export_golden.py`, `tests/test_core_pose_merge.py`

**Semantic contract:** per-row keypoint cleaning (zero conf where `conf<min` or coord non-finite), valid-fraction rejection, quality score + state (`<0.2`/`<0.7` thresholds), body-length + edge-outlier flags; write cleaned `PoseKpt_*_Conf` back plus `PoseQualityScore/State/Flags/Source/PoseWasCleaned`. **Preserve:** float32→`float(...)` round-trip on conf; exact flag token order and `|`-join.

- [ ] **Step 1: Baseline guard tests pass**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/test_pose_quality.py -q`
Expected: PASS.

- [ ] **Step 2: Vectorize**

Stack `(N,K,3)` from the triplet columns; compute conf/coord masks, valid-fraction, score, state column-wise; assemble the flag strings from per-row boolean masks preserving token order; block-write the conf columns and the 5 metadata columns. Keep the float32 cast path identical so the CSV text matches.

- [ ] **Step 3: Run guard tests**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/test_pose_quality.py tests/test_core_pose_merge.py tests/test_rich_export_golden.py -q`
Expected: PASS (golden byte-identical). Fix divergence in code, never in the golden.

- [ ] **Step 4: Commit**

```bash
git add src/hydra_suite/core/individual/pose/quality.py
git commit -m "perf(post): vectorize apply_quality_to_dataframe row loop (byte-identical)"
```

---

### Task 8: Vectorize hotspot #2 — temporal postproc

Highest byte-identical risk (rolling ddof/window; interpolation constants; ordering).

**Files:**
- Modify: `src/hydra_suite/core/post/pose_merge.py:333-347` (groupby driver), `src/hydra_suite/core/individual/pose/quality.py` (`_suppress_temporal_outliers` :784-857, `_interpolate_gaps` :860-916, `_recompute_pose_summary` :1030-1051)
- Test: `tests/test_pose_quality.py`, `tests/test_core_pose_merge.py`, `tests/test_rich_export_golden.py`

**Semantic contract (preserve EXACTLY):** rolling `mean`/`std` with `window`, `min_periods=3`, `center=True`, `ddof=1`; z = `|series - roll_mean| / max(roll_std, 1e-6) > thr` → conf←0.0 + flag `temporal_outlier`; suppression runs X then Y, before interpolation; short-gap linear fill `t = step/(gap+1)`, conf←`0.3`, source←`"cleaned"`; summary recompute `PoseMeanConf = df[conf_cols].mean(axis=1)`, `PoseValidFraction = (df[conf_cols]>0).sum(axis=1)/K`. Flag token order preserved.

- [ ] **Step 1: Baseline guard tests pass**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/test_pose_quality.py tests/test_core_pose_merge.py -q`
Expected: PASS.

- [ ] **Step 2: Vectorize**

Replace the per-valid-index Python loops with boolean-mask assignments; use `groupby(...).rolling(...)` (or a numpy sliding-window matching pandas ddof=1/min_periods=3/center semantics) for the z-scores; vectorize the linear gap fill; compute the summary column-wise. Keep the suppression→interpolation→recompute and X-before-Y ordering, the `1e-6` epsilon, the `0.3` fill conf, and the flag token order byte-exact.

- [ ] **Step 3: Run guard tests**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/test_pose_quality.py tests/test_core_pose_merge.py tests/test_rich_export_golden.py -q`
Expected: PASS (golden byte-identical).

- [ ] **Step 4: Commit**

```bash
git add src/hydra_suite/core/post/pose_merge.py src/hydra_suite/core/individual/pose/quality.py
git commit -m "perf(post): vectorize temporal pose postprocessing (byte-identical)"
```

---

### Task 9: Final gate — equivalence MPS + CUDA + re-profile

**Files:** none (verification).

- [ ] **Step 1: Full suite green**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/ -k "post or pose or quality or rich or identity or heading" -q`
Expected: PASS.

- [ ] **Step 2: Equivalence harness MPS (this box)**

Baseline = the post-Part-1 tree (Part 2 must be byte-identical to it), so run the matrix with MAIN_SRC = a worktree at the Part-1-complete commit and WT_SRC = HEAD, on `ant_pose_headtail ant_cnn_identity fly_obb`. Kill stale sleap first. Expected: positions AND θ EQUIVALENT at the determinism floor (Part 1 already stabilized θ). Verify CSV row counts > 0.

- [ ] **Step 3: Equivalence harness CUDA (mehek)**

Per CLAUDE.md CUDA recipe with the same baseline/subset. Expected: byte-identical.

- [ ] **Step 4: Re-profile**

Re-run `profile_pipeline.py` on `ant_pose_headtail`; confirm the postproc share dropped from ~28%. Record the new breakdown.

- [ ] **Step 5: Record**

```bash
git commit --allow-empty -m "verify(post): Part 2 byte-identical equivalence MPS+CUDA; postproc share reduced"
```

---

## Self-Review

**Spec coverage:** Part 1 anchor (Task 1) + re-baseline (Task 2) + determinism acceptance/secondary hardening (Task 3); Part 2 golden (Task 4) + hotspots #3/#4/#1/#2 (Tasks 5–8) + equivalence/re-profile (Task 9). All spec sections covered. ✓

**Placeholder scan:** Part 1 code is concrete (exact replacement). Part 2 vectorization tasks are byte-identical refactors specified by (exact site + semantic contract + approach + the committed golden/unit oracle) rather than speculative literal code — the golden IS the precise spec, and writing unverified numpy here would risk float32 divergence. Task 3 Step 2 (secondary hardening) is explicitly conditional on Step 1 evidence. ✓

**Type consistency:** `_fix_heading_globally(theta)->ndarray` unchanged; `_backtrack` local helper; hotspot function names/sites consistent with Explorer map and used identically across tasks. ✓

**Ordering:** Part 1 before Part 2 (θ stabilized before the golden is captured); within Part 2, golden first, then #3/#4 (isolated) → #1 → #2 (#1 feeds #2). ✓
