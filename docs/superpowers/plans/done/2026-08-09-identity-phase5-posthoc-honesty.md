# Identity Overhaul — Phase 5: Post-hoc Self-Sufficiency + Forward-Backward Smoothing (the Honesty Fix)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the offline (post-hoc) identity decoder self-sufficient — sourced from the always-written Phase-3 `IdentityEvidenceCache` (not the decoder-populated CSV columns) — add forward-backward smoothing over calibrated evidence, run changepoint splitting on the *smoothed posterior*, resolve fragments via the shared `substrate.solve_unique_assignment`, and make post-hoc an independent toggle. **This closes the honesty bug: realtime identity influence OFF + post-hoc ON now produces real identities.**

**Architecture:** Today `run_fragment_solver` reconstructs per-trajectory identity from wide-CSV `CNN_*_Prob`/`DetectedTag*` columns (a heuristic product-of-columns → weighted-sum-of-mean-logs → single softmax) plus an online prior from `IdentityAssignedLabel`/`Confidence` that only the realtime decoder writes — so with realtime off the solver is starved. Phase 3 already writes a calibrated per-frame evidence sidecar every run regardless of the realtime flags; Phase 4 provides `fuse_log_evidence` (real posterior chaining) and `solve_unique_assignment` (the shared uniqueness solver). Phase 5 re-sources offline from that cache (join trajectory rows to evidence via `(FrameID, ln)→detection_id`), chains a real per-frame posterior forward and backward per trajectory, splits at changepoints in the smoothed posterior, and assigns fragments with the substrate solver — folding the offline collision-veto and `post/processing.resolve_simultaneous_identity_conflicts` onto that one solver. **This is a deliberate behavior change** (not byte-identical): the acceptance bar is the honesty regression + invariants + realtime-agreement, with positions byte-identical (identity is additive to geometry).

**Tech Stack:** Python 3, NumPy, pandas, SciPy, `ruptures` (PELT), pytest.

**Spec:** `docs/superpowers/specs/2026-07-22-identity-overhaul-consolidated-design.md` — Layer 4b (Post-hoc consumer), Rollout "Phase 5". Builds on Phase 3 (`IdentityEvidenceCache`) + Phase 4 (`substrate.fuse_log_evidence`/`solve_unique_assignment`/`map_tag_to_catalog`).

## Global Constraints

- **Deliberate behavior change — NOT byte-identical.** Acceptance (user-approved):
  1. **Honesty regression (primary):** realtime OFF (`ENABLE_IDENTITY_IN_TRACKING=False`) + post-hoc ON → non-empty `IdentityAssignedLabel` on the CNN-identity clip (`ant_cnn_identity`). Direct proof of the fix.
  2. **Positions byte-identical** — identity is additive; tracking geometry unchanged. Equivalence gate positions p99=0 on MPS + CUDA, baseline = Phase-5-start branch SHA.
  3. **Uniqueness invariant** — no duplicate committed identity per frame; every emitted label is a valid catalog entry.
  4. **Smoothing unit test** — a synthetic late-confident/early-ambiguous trajectory is corrected by the forward-backward pass.
  5. **Realtime-agreement (quality proxy):** with realtime ON, new post-hoc identities roughly agree with the realtime identities (both have full evidence). No ground-truth clips assumed; controller judges + reports, user confirms.
- **Source of truth = the evidence cache.** Offline must read `IdentityEvidenceCache` (calibrated per-frame `log_probs`), NOT the CSV `CNN_*_Prob`/`DetectedTag*` reconstruction (retired) nor the online-decoder columns as evidence. The online result may still be read as an optional weak *prior* only if present (do not depend on it).
- **One uniqueness solver.** Fragment assignment + `resolve_simultaneous_identity_conflicts` both go through `substrate.solve_unique_assignment`. Keep the iterative residual pass for ambiguous fragments.
- **Catalog via `resolve_catalog_spec`.** Retire the inline catalog duplicate in `postprocess_df.py:73-110`; use the Phase-1 shared resolver + `IdentityCatalog.from_spec`.
- **Isolation.** Long-lived worktree `.worktrees/identity-phases-3-7` (branch `feat/identity-phases-3-7`). No new worktree; no merge to main.
- **Dependency direction.** Offline/substrate are Core; import only Core/Runtime/Utils. `postprocess_df`/offline may not import app layers.
- **Commit as the git user;** no `Co-Authored-By`. `make format` + `make lint` before commit; revert unrelated drift. Kill stale sleap/hydra before heavy runs.
- **Verification:** unit tests + honesty test on `hydra-mps`; positions-equivalence on MPS + CUDA (mehek).

---

## Current-state anchors (verified; worktree paths; locate by content)

- **Offline (`core/individual/identity/offline.py`):** CSV reconstruction `_trajectory_cnn_log_evidence` (:188-272), `_trajectory_tag_evidence` (:390-442), `_build_prior_log_scores` (:468-485); fusion weights + blend (:741-743, :785-793), `_normalize_support_scores` (:445-465); PELT `detect_identity_changepoints` (:488-570, operates on `CNN_*_Prob` cols), `split_trajectories_at_changepoints` (:573-634); solver `_iterative_assign` (:720-1098, collision-veto greedy), `_build_traj_summaries` (:1100-1203), `solve_global_assignment` (:1206-1332), `run_fragment_solver` (:1335-1411, public entry; PELT dispatch :1392-1411). Column consts `_LABEL_COL`/`_CONF_COL` (:29-30).
- **Invocation:** `core/individual/postprocess_df.py` `apply_identity_postprocessing_to_df` — inline catalog build (:73-110), gate + call `run_fragment_solver` (:112-116), `_annotate_identity_summary_columns` (:14-56, already references `IdentityOfflineLabel`/`IdentitySmoothedLabel` at :28-31 — unwritten output slots).
- **Evidence cache read:** `IdentityEvidenceCache(path, mode="r")` — `load_frame(frame_idx)->list[IdentityEvidence]` (cache.py:238-288), `get_cached_frames()` (:290-301), `catalog_labels` (:303), `catalog_labels_for_source(src)` (:309-321). `IdentityEvidence.detection_id` = the trajectory df's `ln` column (join on `(FrameID, ln)`). `TrajectoryID` groups rows.
- **Substrate (Phase 4):** `fuse_log_evidence(log_posterior, evidence_log_probs, *, per_frame_cap=inf, prob_floor=0.0)` (substrate.py:37), `solve_unique_assignment(posterior_probs, num_known, display_threshold, *, use_scipy=True)->list[int|None]` (:137), `map_tag_to_catalog` (:364). Online sticky transition reference: `online._build_log_transition` (online.py:203-208), `_predict_belief` (:325-331).
- **Honesty-bug gate:** decoder built only if `individual_pipeline_enabled and ENABLE_IDENTITY_IN_TRACKING` (worker.py:1883-1907); blanks emitted by `_online_identity_row_values` (:1924-1980) when decoder None. Evidence cache written independent of realtime (runner.py:688-693, 1008-1011; worker `_resolve_identity_evidence_run_config` :4205-4259). Post-hoc already flag-independent (identity_schema.py:96-127; engine_params.py:1123-1128) but reads CSV. Reserved `PostHocIdentityConfig.enabled` (identity_schema.py:47) not yet emitted. Misleading tooltip `tracking_panel.py:538-544`. Trajectory-structure gating consuming `IdentityCommitted`: `core/post/processing.py:1484-1496, 2594-2615`. Third uniqueness site: `resolve_simultaneous_identity_conflicts` (`core/post/processing.py:1404`+).
- **Cache-path plumbing:** offline runs in post-processing (after tracking); the evidence sidecar path must be threaded from the worker/runner (`inference_runner.identity_evidence_sidecar_path("batch")`) into `apply_identity_postprocessing_to_df` (via params or an explicit arg). READ how post-processing is invoked from the worker to find the seam.

---

## File Structure
**Create:** `src/hydra_suite/core/individual/identity/smoothing.py` (forward-backward smoother + evidence-cache trajectory sourcing); tests under `tests/identity/`.
**Modify:** `offline.py` (source from cache, PELT-on-smoothed, substrate uniqueness); `postprocess_df.py` (catalog via resolve_catalog_spec, thread cache path, wire smoothing); `core/post/processing.py` (`resolve_simultaneous_identity_conflicts` → substrate solver); `trackerkit/config/identity_schema.py` + `engine_params.py` (emit `posthoc.enabled`); `core/tracking/worker.py`/post seam (thread evidence-cache path into post-processing); `tracking_panel.py` (tooltip).

## Interfaces (defined once)
```python
# core/individual/identity/smoothing.py
def load_trajectory_evidence(df, cache: "IdentityEvidenceCache", catalog: "IdentityCatalog"
                             ) -> dict[int, list[tuple[int, np.ndarray]]]:
    """Per TrajectoryID → ordered [(FrameID, catalog_log_probs)] pulled from the evidence cache
    by joining rows on (FrameID, ln)->detection_id (remapped to `catalog` if source basis differs).
    Rows with no cached evidence are omitted."""

def smooth_trajectory_posteriors(frame_log_probs: list[np.ndarray], transition_epsilon: float
                                 ) -> list[np.ndarray]:
    """Forward-backward (two-filter) smoothing: forward chain via fuse_log_evidence + sticky-Markov
    transition, backward chain likewise, combine per frame → smoothed per-frame log-posteriors."""

def smoothed_label_and_conf(smoothed: list[np.ndarray], catalog, display_threshold: float
                            ) -> list[tuple[str, float]]:
    """Per-frame (label, confidence) from smoothed posteriors (unknown/'' below threshold)."""
```

---

## Task 1: Offline evidence sourcing from the cache
Create `smoothing.py` + `load_trajectory_evidence`. Join trajectory rows to `IdentityEvidenceCache` via `(FrameID, ln)→detection_id`; return per-trajectory ordered per-frame catalog `log_probs`. Remap source-basis to the global catalog via the same phase→global remap the worker uses if a source's basis differs (reuse `catalog_labels_for_source`).

**Files:** Create `smoothing.py`, `tests/identity/test_offline_evidence_sourcing.py`.
- [ ] Failing test: synthetic df (`TrajectoryID`,`FrameID`,`ln`) + a written `IdentityEvidenceCache` → assert `load_trajectory_evidence` returns the right per-trajectory `(FrameID, log_probs)` sequences, joined correctly, with cache-basis remapped to the passed catalog; rows with NaN/absent `ln` omitted.
- [ ] Run RED → implement → GREEN → full `tests/identity/` no regression.
- [ ] `make format && make lint`; commit `feat(identity): offline sources per-trajectory evidence from the cache (FrameID,ln join)`.

## Task 2: Forward-backward smoothing
`smooth_trajectory_posteriors` + `smoothed_label_and_conf` using `substrate.fuse_log_evidence` + a sticky-Markov transition (reuse online's `_build_log_transition` form).

**Files:** `smoothing.py` (add); `tests/identity/test_offline_smoothing.py`.
- [ ] Failing test: a synthetic trajectory whose early frames are ambiguous (near-uniform) and late frames confidently identity B → assert the SMOOTHED early-frame posteriors favor B (a late-confident burst corrects early ambiguity), while forward-only would not. Also: single-frame trajectory returns its own posterior; empty → empty.
- [ ] Implement forward pass (chain fuse+transition), backward pass (reverse), combine (two-filter: `smoothed ∝ forward · backward / prior`, in log-space, renormalized — define precisely and test it reduces to the evidence when transition is identity).
- [ ] RED→GREEN; no regression. Commit `feat(identity): forward-backward trajectory smoothing over calibrated evidence`.

## Task 3: Changepoint on the smoothed posterior
Repoint `detect_identity_changepoints` to run PELT on the smoothed per-frame posterior signal (per trajectory) instead of the `CNN_*_Prob` columns.

**Files:** Modify `offline.py:488-570`; test.
- [ ] Failing test: a trajectory that switches identity mid-way (smoothed posterior regime change) → changepoint detected at the switch; a stable trajectory → no split. Drive `detect_identity_changepoints` with the smoothed signal.
- [ ] Implement: build the PELT signal from the smoothed posteriors (from Task 2) rather than `CNN_*_Prob`; keep `CHANGEPOINT_PENALTY`/`MIN_FRAGMENT_FRAMES`/`PELT_MODEL` params + the `ruptures`-absent graceful fallback.
- [ ] RED→GREEN; commit `refactor(identity): PELT changepoint on smoothed posterior (not CNN_*_Prob columns)`.

## Task 4: Fragment assignment + conflict resolution via the substrate solver
Replace the offline collision-veto greedy's core assignment with `substrate.solve_unique_assignment` (keep the iterative residual pass for ambiguous fragments), and fold `post/processing.resolve_simultaneous_identity_conflicts` onto the same solver.

**Files:** Modify `offline.py` (`_iterative_assign`/`solve_global_assignment`), `core/post/processing.py:1404+`; tests.
- [ ] Failing test: N fragments with overlapping same-label evidence → `solve_unique_assignment`-based global assignment yields a valid partial-injective assignment (no two overlapping fragments share a committed identity); the conflict-resolver test: two trajectories, same majority label, shared frames → one cleared, via the shared solver.
- [ ] Implement: build per-fragment catalog posteriors (from smoothed evidence) → `solve_unique_assignment`; keep the residual/veto refinement for ambiguity; rewrite `resolve_simultaneous_identity_conflicts` to call the solver. Preserve output columns (`IdentityAssignedLabel`, `IdentityFragmentScore`, `IdentityCommitted`, `IdentityConflictResolved`).
- [ ] RED→GREEN; commit `refactor(identity): offline fragment assignment + conflict-resolution via substrate.solve_unique_assignment`.

## Task 5: Wire the self-sufficient offline pipeline
`run_fragment_solver`/`postprocess_df` source from the cache → smooth → changepoint(smoothed) → assign; catalog via `resolve_catalog_spec`; thread the evidence-cache path from the worker into `apply_identity_postprocessing_to_df`; write `IdentitySmoothedLabel`/`IdentityOfflineLabel`.

**Files:** Modify `offline.py` (`run_fragment_solver`, `_build_traj_summaries` → cache-sourced), `postprocess_df.py` (catalog resolver, cache path, annotate), the worker/post seam (pass sidecar path). Test.
- [ ] **Honesty regression test (the key one):** build a tracking-output df + a written evidence cache, run `apply_identity_postprocessing_to_df` with `ENABLE_IDENTITY_IN_TRACKING`-off-style inputs (empty `IdentityAssignedLabel` columns) → assert `IdentityAssignedLabel` becomes NON-EMPTY (post-hoc self-sufficient from the cache). RED first (proves today's starvation).
- [ ] Implement the wiring; retire the CSV `CNN_*_Prob`/`DetectedTag*` reconstruction path (clean break); catalog via `resolve_catalog_spec(p["CNN_CLASSIFIERS"], p["TAG_IDENTITY_LABELS"])`.
- [ ] RRun the honesty test GREEN + full `tests/identity/`; commit `feat(identity): self-sufficient offline pipeline sourced from the evidence cache (closes honesty bug)`.

## Task 6: Independent post-hoc toggle + honest tooltip
Emit `PostHocIdentityConfig.enabled` into engine params; make offline run whenever identity classification is on (never gated on realtime); correct the misleading `tracking_panel.py` tooltip.

**Files:** `identity_schema.py` (from_engine_config emits `enabled`), `engine_params.py` (emit `IDENTITY_POSTHOC_ENABLED`), `postprocess_df.py` (honor it), `tracking_panel.py:538-544` (tooltip), golden test update.
- [ ] Update the `test_get_parameters_dict_characterization` golden for the new key (capture baseline). Tooltip: state plainly that post-hoc reads the inference-time evidence cache and works regardless of realtime.
- [ ] Commit `feat(identity): independent post-hoc toggle + honest realtime tooltip`.

## Phase-End Gate
- [ ] **Honesty regression (primary):** a real `ant_cnn_identity` run with realtime OFF + post-hoc ON → non-empty `IdentityAssignedLabel` (was empty before Phase 5). Run on hydra-mps (short range acceptable).
- [ ] **Positions byte-identical — MPS + CUDA** (baseline = Phase-5-start SHA): all clips pos p99=0 (identity is additive). Identity columns WILL differ (that's the point) — do not gate on them.
- [ ] **Uniqueness invariant** check on the identity clips: no duplicate committed identity within a frame; all labels ∈ catalog.
- [ ] **Realtime-agreement:** realtime ON, compare new post-hoc identities to the realtime identities — report agreement %; controller judges "sensible", user confirms.
- [ ] **Suite:** `tests/identity/` green.

## Self-Review (against spec Layer 4b + Phase 5)
- "Forward-backward smoothing over calibrated evidence" → Task 2. ✅
- "Changepoint on the smoothed posteriors" → Task 3. ✅
- "Fragment global assignment via the substrate uniqueness solver (+ keep iterative residual)" → Task 4. ✅ folds the 3rd uniqueness site (resolve_simultaneous_identity_conflicts).
- "Post-hoc runs from the evidence cache + final trajectories with NO dependency on the realtime decoder" → Tasks 1, 5, 6. ✅ (the honesty fix)
- "Independent toggle" → Task 6. ✅
- Retire CSV-reconstruction heuristic + inline catalog duplicate → Task 5. ✅
- Verification: honesty test + invariants + realtime-agreement + positions byte-identical (user-approved bar). ✅
- **Deferred:** provenance-explicit output columns (`Identity_Evidence_*`/`Identity_Realtime_*`/`Identity_Final_*`) → Phase 6; robustness cap/floor value-wiring → later; `map_tag_to_catalog` wiring at tag sites can land here (Task 4/5) or Phase 6.
