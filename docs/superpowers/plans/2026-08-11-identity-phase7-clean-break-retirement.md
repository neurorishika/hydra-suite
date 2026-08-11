# Identity Phase 7 — Clean-Break Retirement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Delete the retired legacy identity decision paths (no shims), leaving `IdentityEvidenceStage` + the evidence cache + the substrate as the single sourced implementation, while preserving all characterization coverage (via committed goldens where an oracle is removed).

**Architecture:** Pure removal of dead/superseded code + test migration. The live pipeline already routes identity through `IdentityEvidenceStage` (inference) → evidence sidecar → online decoder / offline fragment solver (cache-sourced). This phase removes the orphaned V3 cache, the dead detected-CNN cache plumbing, the unused `TrackCNNHistory` majority-vote, the retired `IdentityEvidenceEmitter` class (replaced by committed evidence goldens), and the offline CSV-reconstruction fallback.

**Tech Stack:** Python 3.13, pandas ≥3, pytest. Core must not import app layers.

## Global Constraints

- **Clean break, no shims** (spec + user decision 2026-08-11). Deleted symbols are removed, not aliased.
- **Positions byte-identical:** none of these removals touch the Kalman/assignment/detection geometry. The equivalence harness (positions p99 = 0, θ within head/tail π-flip noise floor, unmatched = 0) must stay green on MPS (+ CUDA at the pre-merge gate). Baseline for attribution = branch commit at phase start `1d6b5b93`.
- **Preserve `build_evidence_cache_path`** — it is a path helper imported by `core/inference/runner.py:1068` from `evidence_emitter.py`; it must survive the emitter-class deletion (move it to a neutral module or keep the module with only that helper).
- **Target 4 handled as delete-emitter + committed-golden** (user + author decision): snapshot the emitter's evidence output to a committed fixture BEFORE deleting the emitter, then repoint the parity tests at the frozen snapshot.
- **Only Target 7 can change output**, and only for the no-evidence-sidecar path (which Phase 3 made not happen in production); document it.
- After each deletion task, run `make dead-code` scope check is optional; the gate is the identity + headless test suites staying green (modulo the known pre-existing failures: 9 `test_postproc_invariants` KeyErrors + the `get_parameters_dict` `canonical_margin` golden — both fail at baseline too).

## Retirement inventory (authoritative — from the Phase-7 inventory)

Already done (NO action): factor-encoding landmine (structured catalog, Phase 1); triplicated uniqueness (single `substrate.solve_unique_assignment`, Phase 4); tracking-time emitter *wiring* (Phase 3 — worker loop no longer feeds an emitter).

Real work, dependency order: **Task 1** V3 `CNNIdentityCache` → **Task 2** `detected_cnn_cache_paths` + augment fn (reads the V3 cache) → **Task 3** `TrackCNNHistory` → **Task 4** `IdentityEvidenceEmitter` class → goldens → **Task 5** offline CSV-reconstruction fallback.

---

### Task 1: Delete the orphaned V3 `CNNIdentityCache`

**Files:**
- Modify: `src/hydra_suite/core/individual/classification/cnn.py` (delete `class CNNIdentityCache` ~:86 + its `.save()`/load methods ~:157-228)
- Modify: `src/hydra_suite/core/individual/__init__.py`, `src/hydra_suite/core/individual/classification/__init__.py` (remove the two re-exports)
- Modify: `src/hydra_suite/core/tracking/worker.py` (delete the guarded dead read ~:1652-1671 + `_build_cnn_identity_cache_path` ~:680 if unused elsewhere)
- Modify: `src/hydra_suite/core/individual/properties/export.py` (remove the `CNNIdentityCache` import + its use in `augment_trajectories_with_detected_cnn_cache` — but that fn is deleted in Task 2; in THIS task just remove the import if it dangles, else defer to Task 2)
- Modify: `src/hydra_suite/core/inference/cache/base.py` (scrub the comment referencing it, ~:9)
- Delete: `tests/test_hydra_cnn_identity.py` (entirely — it only round-trips the V3 cache)
- Modify: `tests/test_properties_export.py` (remove the CNN-cache round-trip tests ~:298,340), `tests/test_tracking_worker_helpers.py` (remove the `classification_cnn.CNNIdentityCache = object` monkeypatch ~:177)

**Interfaces:** Produces: nothing (removal). Consumes: nothing.

- [ ] **Step 1: Confirm no writer.** `grep -rn "CNNIdentityCache(" src/ tests/` — verify no `.save()` call exists in `src/` (only tests). Record the finding.
- [ ] **Step 2: Delete the class + methods** in `cnn.py`; remove the two `__init__` re-exports; run `grep -rn "CNNIdentityCache" src/` and confirm only removable references remain.
- [ ] **Step 3: Remove the dead worker read** (`worker.py:1652-1671`) and `_build_cnn_identity_cache_path` if now unused (`grep` it). Remove the `base.py:9` comment.
- [ ] **Step 4: Delete/trim the tests** listed above.
- [ ] **Step 5: Run** `python -m pytest tests/test_properties_export.py tests/test_tracking_worker_helpers.py tests/identity/ tests/test_fragment_solver.py -q` — green (Task 2 handles the augment fn; if `test_properties_export` still imports the augment fn, leave that test for Task 2).
- [ ] **Step 6: Commit** — `refactor(identity): delete orphaned V3 CNNIdentityCache (Phase 7 Task 1)`.

---

### Task 2: Delete `detected_cnn_cache_paths` + `augment_trajectories_with_detected_cnn_cache`

**Files:**
- Modify: `src/hydra_suite/core/post/pose_merge.py` (remove the `detected_cnn_cache_paths` dataclass field ~:24 + the merge loop ~:130-139 + its use in `check_pose_export_sources` ~:43)
- Modify: `src/hydra_suite/core/tracking/session.py` (remove `detected_cnn_cache_paths=self.paths.get(...)` ~:179)
- Modify: `src/hydra_suite/core/individual/properties/export.py` (delete `augment_trajectories_with_detected_cnn_cache` ~:761 + `build_detected_cnn_lookup_dataframe` ~:645 if only used by it)
- Modify: `tests/test_properties_export.py` (remove the augment-fn unit tests ~:296,325,336,367)
- Modify: GUI cutover-wiring tests that pass `current_detected_cnn_cache_paths` — `tests/test_trackerkit_session_cutover_wiring.py:41,144`, `tests/test_trackerkit_tracking_orchestrator_dialogs.py:151`, `tests/test_gui_session_cutover_equivalence.py:159` (drop the dead key from those fixtures)

**Interfaces:** Consumes: Task 1 (the augment fn read the now-deleted `CNNIdentityCache`).

- [ ] **Step 1: Confirm dead.** `grep -rn "detected_cnn_cache_paths" src/` — verify no writer sets `paths["detected_cnn_cache_paths"]`. Record.
- [ ] **Step 2: Delete** the dataclass field, the `pose_merge` loop, the `check_pose_export_sources` branch, the `session.py:179` line, and the two export.py functions.
- [ ] **Step 3: Trim tests** (unit + GUI cutover fixtures).
- [ ] **Step 4: Run** `python -m pytest tests/test_properties_export.py tests/test_trackerkit_session_cutover_wiring.py tests/test_gui_session_cutover_equivalence.py tests/test_core_qtfree_slice2.py tests/test_trackerkit_headless_tracking.py -q` — green.
- [ ] **Step 5: Commit** — `refactor(identity): delete dead detected_cnn_cache_paths + augment fn (Phase 7 Task 2)`.

---

### Task 3: Delete `TrackCNNHistory` majority-vote

**Files:**
- Modify: `src/hydra_suite/core/individual/classification/cnn.py` (delete `class TrackCNNHistory` ~:584 + `majority_class` ~:633,657)
- Modify: `src/hydra_suite/core/individual/__init__.py` (~:17,56), `src/hydra_suite/core/individual/classification/__init__.py` (~:16,43) — remove the re-exports
- Modify: `src/hydra_suite/core/tracking/frame_result_bridge.py` (scrub the docstring mention ~:73; KEEP the module — it is live)

**Interfaces:** none.

- [ ] **Step 1: Confirm never instantiated.** `grep -rn "TrackCNNHistory" src/ tests/` — zero constructor calls. Record.
- [ ] **Step 2: Delete** the class + `majority_class`; remove the re-exports; scrub the docstring.
- [ ] **Step 3: Run** `python -m pytest tests/test_worker_live_store_population.py tests/test_worker_real_inference_integration.py tests/identity/ -q` — green.
- [ ] **Step 4: Commit** — `refactor(identity): delete unused TrackCNNHistory majority-vote (Phase 7 Task 3)`.

---

### Task 4: Retire `IdentityEvidenceEmitter` class → committed evidence goldens

**Files:**
- Create: `tests/data/identity_evidence_goldens/<case>.npz` (or `.json`) — frozen snapshots of the emitter's evidence output for the parity cases.
- Modify: `tests/identity/test_evidence_builder_parity.py`, `tests/identity/test_evidence_phase_basis_parity.py` — repoint from "run the emitter as oracle" to "compare against the committed golden".
- Modify: `src/hydra_suite/core/tracking/identity/evidence_emitter.py` — delete `class IdentityEvidenceEmitter` (~:40); PRESERVE `build_evidence_cache_path` (imported by `runner.py:1068`). If the file would be left with only `build_evidence_cache_path`, either keep the file with just that helper (+ update its module docstring) or move the helper to a neutral module (e.g. `core/inference/cache/paths.py`) and update the `runner.py:1068` import. Pick the lower-churn option and state it.
- Modify: `tests/test_worker_real_inference_integration.py:308` — delete the vestigial `worker_obj._evidence_emitters = []` line.
- Modify: docstrings that name the emitter (`evidence_builder.py`, `runner.py:401`, `identity_evidence.py:34,81`, `identity_evidence_config.py:79`, `cache.py:99`) — update to reference `IdentityEvidenceStage` (optional polish; do the ones that would otherwise read as stale/incorrect).

**Interfaces:** Consumes: the emitter (one last time) to generate the goldens.

- [ ] **Step 1: Generate the goldens.** In a scratch step, run each parity test's setup through the EXISTING emitter and dump its evidence output (the exact arrays the tests currently assert on) to the committed fixture files. Verify the current parity tests still pass against the emitter (baseline).
- [ ] **Step 2: Repoint the parity tests** to load the golden fixtures instead of instantiating the emitter; assert `IdentityEvidenceStage`/`EvidenceBuilder` output equals the golden (same tolerance/byte-equality as before). Run them GREEN (still with the emitter present, now unused by the tests).
- [ ] **Step 3: Delete the emitter class**; preserve/move `build_evidence_cache_path`; update the `runner.py` import if moved. Delete the vestigial test line.
- [ ] **Step 4: Run** `python -m pytest tests/identity/test_evidence_builder_parity.py tests/identity/test_evidence_phase_basis_parity.py tests/test_identity_evidence_pipeline.py tests/test_worker_real_inference_integration.py -q` — green; `grep -rn "IdentityEvidenceEmitter" src/` returns nothing (docstrings optional).
- [ ] **Step 5: Commit** — `refactor(identity): retire IdentityEvidenceEmitter; parity tests use committed goldens (Phase 7 Task 4)`.

---

### Task 5: Delete the offline CSV-reconstruction fallback

**Files:**
- Modify: `src/hydra_suite/core/individual/identity/offline.py` — delete the CSV-decode helpers (`_build_cnn_label_specs`, `_build_cnn_probability_prefix_map`, `_trajectory_mean_cnn_probs`, `_trajectory_cnn_log_evidence`, `_trajectory_per_row_probs`, `_sanitize_probability_token`, ~:43-361 — audit exact set) and the `evidence_by_traj is None` reconstruction branch in `_build_traj_summaries` (~:1264-1327). `solve_global_assignment` should REQUIRE `evidence_by_traj` (cache-sourced); when it is `None`/empty, produce empty/unknown Final identity (the documented no-sidecar degrade) rather than reconstruct from CSV columns. Removes the last `split("_")` decode.
- Modify: `tests/test_fragment_solver.py` — rewrite the ~13 `solve_global_assignment(df, catalog, params)` / `_build_traj_summaries(df, catalog)` call sites (lines ~168,186,213,245,278,550,595,669,726,757,836,904,922,985,1007,1034) to build a real `IdentityEvidenceCache` (mirror `tests/identity/test_honesty_fix.py::_write_cache`) and pass `evidence_by_traj`/`cache`, instead of hand-built `CNN_*_Prob` columns.
- Modify: `tests/identity/test_honesty_fix.py` — the negative-control test that exercises the no-cache reconstruction: update to assert the new no-sidecar behavior (empty/unknown), preserving the discriminating power (WITH cache → correct identities; WITHOUT → no reconstruction).

**Interfaces:** Consumes: the evidence-cache test helper pattern from `test_honesty_fix.py`.

- [ ] **Step 1: Write/adjust the failing tests first** — convert the fragment-solver tests to feed a real evidence cache; run them RED against the current fallback-present code (they should still pass with the fallback, so instead: add ONE new test asserting `solve_global_assignment(df, catalog, params, evidence_by_traj=None)` on a df with `CNN_*_Prob` columns now yields empty/unknown Final (no reconstruction) — RED before the deletion, GREEN after).
- [ ] **Step 2: Delete** the decode helpers + the `None` reconstruction branch; make `_build_traj_summaries` cache-only.
- [ ] **Step 3: Rewrite** the ~16 dependent assertions to feed a cache.
- [ ] **Step 4: Run** `python -m pytest tests/test_fragment_solver.py tests/identity/ tests/test_core_identity_postprocess_df.py -q` — green. `grep -n 'split("_")' src/hydra_suite/core/individual/identity/offline.py` returns nothing.
- [ ] **Step 5: Commit** — `refactor(identity): delete offline CSV-reconstruction fallback; solver is cache-only (Phase 7 Task 5)`.

---

## Phase-end gate (controller, after Task 5)

1. **No-legacy grep:** `CNNIdentityCache`, `TrackCNNHistory`, `IdentityEvidenceEmitter`, `detected_cnn_cache_paths`, `augment_trajectories_with_detected_cnn_cache` return zero live `src/` references.
2. **Positions equivalence (MPS + CUDA):** baseline `1d6b5b93` vs Phase-7 HEAD — positions byte-identical all 7 clips (these are dead-code/test-only removals + a no-sidecar-only behavior change; production positions unchanged).
3. **Real-clip identity smoke:** `ant_cnn_identity` (realtime on + off) still produces correct provenance-explicit identity (the cache path is unaffected).
4. **Full identity + headless suite** green (modulo the known pre-existing failures).
5. **Whole-branch review** (opus) over the Phase-7 range.
6. Bring the result to the user (checkpoint) — then the SINGLE `--no-ff` merge of `feat/identity-phases-3-7` to main (all of Phases 3-7), with the deferred-minor follow-ups listed.

## Self-Review notes (author)

- Dependency order respected: V3 cache (1) before its reader the augment fn (2); `TrackCNNHistory` (3) independent; emitter→goldens (4) preserves `build_evidence_cache_path`; CSV fallback (5) is the only behavior-affecting one (no-sidecar path) and the highest test-churn.
- Spec coverage: Targets 1/2/3/4/7 addressed; Targets 5/6 + emitter-wiring already done (no task).
- Risk: Task 5's no-sidecar behavior change — mitigated by Phase 3's unconditional sidecar write; documented in the commit + gate step 3.
- Golden generation (Task 4 Step 1) must run the emitter BEFORE deletion — ordering is load-bearing.
