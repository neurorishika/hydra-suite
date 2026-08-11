# Identity Phase 6 — Provenance Columns + Honest UI Reorg Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the single, clobbered `IdentityAssigned*` output column with three provenance-explicit column families (`IdentityEvidence*`, `IdentityRealtime*`, `IdentityFinal*`) that no stage overwrites, and re-scope the three identity UI panels so each control is honest about what it does and when it is available.

**Architecture:** One shared identity-column vocabulary (a new `core/individual/identity/columns.py` constants + ordered-header module) is the single source of truth for the CSV contract, consumed by both header builders and the worker's positional row writer (killing today's triple-coupling). Each pipeline stage writes only its own family: the online decoder writes `IdentityRealtime*`; the rich-export evidence summary writes `IdentityEvidence*`; the offline solver + post-hoc resolution write `IdentityFinal*` (with `IdentityFinalSource ∈ {realtime, offline, tag}`). When post-hoc did not run, `IdentityFinal*` mirrors the realtime decision once, non-destructively. `UniqueIdentityKey` remains a derived compatibility column.

**Tech Stack:** Python 3.13, pandas ≥3, PySide6 (Qt), pytest. Core/Training must not import app-layer packages.

## Global Constraints

- **Clean rename (user decision, 2026-08-10):** the legacy names `IdentityAssignedID/Label/Confidence`, `IdentityPosteriorMargin`, `IdentityEntropy`, `IdentityCommitted`, `IdentitySlotLockLabel`, `IdentityEvidenceSources`, `IdentityConflictFlag`, `IdentityOfflineLabel/Confidence`, `IdentitySmoothedLabel/Confidence`, `IdentityFragmentScore`, `IdentityConflictResolved` are REMOVED from output. No back-compat aliases for these (only `UniqueIdentityKey` is retained). Every in-repo reader migrates in lockstep.
- **Provenance over mutation:** no stage may overwrite a column another stage owns. The offline solver must NOT write into any `IdentityRealtime*` column. The rich-export evidence summary must NOT overwrite a realtime column.
- **Positions byte-identical:** identity columns are additive/renamed post-processing only. Nothing in this phase may touch Kalman / assignment / detection geometry. The equivalence harness (positions p99 = 0, θ within the documented head/tail π-flip noise floor, unmatched = 0) must stay green on MPS + CUDA. Baseline for attribution = branch commit at phase start `0a0a0091`.
- **Two duplicated CSV headers must converge:** `trackerkit/headless_tracking.py:build_tracking_csv_header` and the inline copy in `trackerkit/gui/orchestrators/tracking.py` (~:1410-1462) must both route through the new shared header helper; the positional writer in `core/tracking/worker.py` (`_online_identity_row_values`, ~:1924-1979) must derive its order from the same source.
- **One typed source of truth:** UI panels round-trip through the flat cfg dict → `IdentityConfig` (`trackerkit/config/identity_schema.py`) → engine `IDENTITY_*` params. No new widget-attribute state.
- **`IdentityFinalSource` vocabulary:** exactly `realtime` | `offline` | `tag` (lowercase), or empty string when no identity was resolved for that row.

---

## Canonical column mapping (old → new)

Implementers: this table is the authoritative rename map. Apply it everywhere.

**Naming convention (user decision, 2026-08-10): PascalCase, no underscores** — matches the rest of the CSV (`TrajectoryID`, `DetectionConfidence`, …). Family prefixes are `IdentityEvidence`, `IdentityRealtime`, `IdentityFinal`. Prefix-filter with `df.filter(regex="^IdentityFinal")`.

| Old column | New column | Family / owner |
|---|---|---|
| `IdentityAssignedID` | `IdentityRealtimeID` | Realtime (online decoder) |
| `IdentityAssignedLabel` | `IdentityRealtimeLabel` | Realtime |
| `IdentityAssignedConfidence` | `IdentityRealtimeConfidence` | Realtime |
| `IdentityPosteriorMargin` | `IdentityRealtimeMargin` | Realtime |
| `IdentityEntropy` | `IdentityRealtimeEntropy` | Realtime |
| `IdentityCommitted` | `IdentityRealtimeCommitted` | Realtime |
| `IdentitySlotLockLabel` | `IdentityRealtimeSlotLock` | Realtime |
| `IdentityEvidenceSources` | `IdentityEvidenceSources` (unchanged name, now Evidence-owned) | Evidence (rich-export summary) |
| *(new)* | `IdentityEvidenceTopLabel` | Evidence |
| *(new)* | `IdentityEvidenceConfidence` | Evidence |
| `IdentityConflictFlag` | `IdentityEvidenceConflictFlag` | Evidence (it flags conflicting *evidence*) |
| `IdentityOfflineLabel` | `IdentityFinalLabel` | Final (offline solver / resolved) |
| `IdentityOfflineConfidence` | `IdentityFinalConfidence` | Final |
| `IdentityFragmentScore` | `IdentityFinalFragmentScore` | Final |
| *(new)* | `IdentityFinalID` | Final (catalog index of the resolved label) |
| *(new)* | `IdentityFinalSource` | Final (`realtime`\|`offline`\|`tag`\|"") |
| `IdentitySmoothedLabel` | `IdentityFinalSmoothedLabel` | Final (per-frame smoothed diagnostic) |
| `IdentitySmoothedConfidence` | `IdentityFinalSmoothedConfidence` | Final |
| `IdentityConflictResolved` | `IdentityFinalConflictResolved` | Final (post-hoc resolution marker) |
| `UniqueIdentityKey` | `UniqueIdentityKey` (retained) | Derived compat, from `IdentityFinal*` |

**Resolution semantics for the Final family (single writer):** after the offline stage,
for every trajectory/row:
- If the offline fragment solver produced a label → `IdentityFinal{Label,ID,Confidence,FragmentScore}` = offline result, `IdentityFinalSource = "offline"`.
- Else if a tag observation resolved it → tag label, `IdentityFinalSource = "tag"`.
- Else if the realtime decoder ran and committed a label → mirror `IdentityRealtime{Label,ID,Confidence}` into `IdentityFinal*`, `IdentityFinalSource = "realtime"` (a one-time non-destructive copy — the realtime columns are left intact).
- Else → `IdentityFinal*` empty, `IdentityFinalSource = ""`.

---

## File Structure

**Create:**
- `src/hydra_suite/core/individual/identity/columns.py` — canonical identity-column name constants (the three families + `UniqueIdentityKey`), the ordered `identity_tracking_columns(identity_method, save_confidence_metrics)` helper (returns the realtime + evidence-source columns appended to the raw tracking header, in writer order), and `IdentityFinalSource` value constants. Pure, no imports from app layers.
- `tests/identity/test_identity_columns.py` — locks the constant names + ordered-header contract.
- `tests/identity/test_provenance_no_clobber.py` — the phase's headline invariant test (offline never mutates a `IdentityRealtime*` column; Final mirrors realtime when post-hoc is off).

**Modify (core writers):**
- `src/hydra_suite/core/tracking/worker.py` — `_online_identity_row_values` (~:1924-1979) writes realtime-family values in the shared order.
- `src/hydra_suite/trackerkit/headless_tracking.py` — `build_tracking_csv_header` (:32-88) routes through `columns.py`.
- `src/hydra_suite/trackerkit/gui/orchestrators/tracking.py` — inline header (~:1410-1462) routes through `columns.py`.
- `src/hydra_suite/core/individual/identity/offline.py` — `_LABEL_COL`/`_CONF_COL` constants (:35-36), `solve_global_assignment` write block (:1537-1586), `_annotate_smoothed_labels` (:1591-1638): write the Final family + `IdentityFinalSource`; never touch realtime columns.
- `src/hydra_suite/core/individual/postprocess_df.py` — `_annotate_identity_summary_columns` (:57-127): write `IdentityEvidenceSources`/`IdentityEvidenceConflictFlag`/`IdentityEvidenceTopLabel`/`IdentityEvidenceConfidence`; read Final for the summary.
- `src/hydra_suite/core/post/identity_postprocess.py` — consensus fill (:131-230) operates on `IdentityFinal*`; derive `UniqueIdentityKey` from Final.
- `src/hydra_suite/core/post/processing.py` — relink/conflict (`_IDENTITY_LABEL_COL` etc. :1311-1400, :402-416, :1547-1559, :2657-2678, :3660-3664): operate on `IdentityFinal*`; set `IdentityFinalConflictResolved`.
- `src/hydra_suite/core/post/media_export.py` — overlay priority list (:193-224): `["UniqueIdentityKey", "IdentityFinalLabel", "IdentityFinalSmoothedLabel"]`.

**Modify (UI):**
- `src/hydra_suite/trackerkit/gui/panels/identity_panel.py` — "Identity Models" scope: status line, calibration affordance, corrected tooltips.
- `src/hydra_suite/trackerkit/gui/panels/tracking_panel.py` — "Realtime Identity" scope: labels/tooltips scoped strictly to realtime association.
- `src/hydra_suite/trackerkit/gui/panels/postprocess_panel.py` — "Post-hoc Identity" scope: first-class independent toggle + smoothing control; corrected tooltip.
- `src/hydra_suite/trackerkit/config/identity_schema.py` — emit the reserved `PostHocIdentityConfig.smoothing_enabled`/`changepoint_enabled` to engine params.
- `src/hydra_suite/trackerkit/engine_params.py` — thread the new post-hoc engine keys.

**Migrate tests (in the task that renames the columns they cover):**
- `tests/test_fragment_solver.py`, `tests/identity/test_honesty_fix.py`, `tests/test_identity_postprocess.py`, `tests/test_identity_conflict_resolution.py`, `tests/test_postproc_invariants.py`, `tests/test_core_identity_postprocess_df.py`, `tests/test_post_tracklet_relinking.py`, `tests/test_media_export.py`, `tests/test_postproc_identity_gating.py`, `tests/test_core_qtfree_slice2.py`, `tests/identity/test_evidence_sidecar_consumption.py`, `tests/test_trackerkit_tracking_orchestrator_dialogs.py`.

---

### Task 1: Canonical identity-column vocabulary (single source of truth)

**Files:**
- Create: `src/hydra_suite/core/individual/identity/columns.py`
- Create: `tests/identity/test_identity_columns.py`

**Interfaces:**
- Produces:
  - Name constants (module-level `str`): `REALTIME_ID, REALTIME_LABEL, REALTIME_CONFIDENCE, REALTIME_MARGIN, REALTIME_ENTROPY, REALTIME_COMMITTED, REALTIME_SLOTLOCK`; `EVIDENCE_TOPLABEL, EVIDENCE_CONF, EVIDENCE_SOURCES, EVIDENCE_CONFLICT_FLAG`; `FINAL_LABEL, FINAL_ID, FINAL_CONFIDENCE, FINAL_SOURCE, FINAL_FRAGMENT_SCORE, FINAL_SMOOTHED_LABEL, FINAL_SMOOTHED_CONFIDENCE, FINAL_CONFLICT_RESOLVED`; `UNIQUE_IDENTITY_KEY = "UniqueIdentityKey"`.
  - `class IdentityFinalSource: REALTIME="realtime"; OFFLINE="offline"; TAG="tag"; NONE=""`.
  - `def identity_realtime_columns() -> list[str]` — the 7 realtime columns + `EVIDENCE_SOURCES` + `EVIDENCE_CONFLICT_FLAG` + `REALTIME_SLOTLOCK`, in the exact order the worker positional writer emits (`[REALTIME_ID, REALTIME_LABEL, REALTIME_CONFIDENCE, REALTIME_MARGIN, REALTIME_ENTROPY, REALTIME_COMMITTED, EVIDENCE_SOURCES, EVIDENCE_CONFLICT_FLAG, REALTIME_SLOTLOCK]`). This is the block appended to each raw tracking row.

- [ ] **Step 1: Write the failing test** (`tests/identity/test_identity_columns.py`)

```python
from hydra_suite.core.individual.identity import columns as C


def test_family_prefixes_are_provenance_explicit():
    assert C.REALTIME_LABEL == "IdentityRealtimeLabel"
    assert C.FINAL_LABEL == "IdentityFinalLabel"
    assert C.FINAL_SOURCE == "IdentityFinalSource"
    assert C.EVIDENCE_SOURCES == "IdentityEvidenceSources"
    assert C.UNIQUE_IDENTITY_KEY == "UniqueIdentityKey"
    # PascalCase house style: no underscores inside a column name.
    for k, v in vars(C).items():
        if k.isupper() and isinstance(v, str) and v.startswith("Identity"):
            assert "_" not in v, v


def test_no_legacy_names_leak():
    legacy = {"IdentityAssignedLabel", "IdentityAssignedConfidence",
              "IdentityOfflineLabel", "IdentityPosteriorMargin"}
    allnames = {v for k, v in vars(C).items() if isinstance(v, str) and k.isupper()}
    assert allnames.isdisjoint(legacy)


def test_realtime_row_order_is_the_worker_contract():
    order = C.identity_realtime_columns()
    assert order == [
        C.REALTIME_ID, C.REALTIME_LABEL, C.REALTIME_CONFIDENCE,
        C.REALTIME_MARGIN, C.REALTIME_ENTROPY, C.REALTIME_COMMITTED,
        C.EVIDENCE_SOURCES, C.EVIDENCE_CONFLICT_FLAG, C.REALTIME_SLOTLOCK,
    ]


def test_final_source_vocabulary():
    assert C.IdentityFinalSource.OFFLINE == "offline"
    assert C.IdentityFinalSource.REALTIME == "realtime"
    assert C.IdentityFinalSource.TAG == "tag"
    assert C.IdentityFinalSource.NONE == ""
```

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/identity/test_identity_columns.py -q`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement `columns.py`** with the constants, `IdentityFinalSource`, and `identity_realtime_columns()` exactly as the test asserts. Pure module; no imports from `trackerkit`/`classkit`/other app layers.

- [ ] **Step 4: Run to verify it passes.** Run the same command; expected PASS.

- [ ] **Step 5: Commit** — `git add` the two files; `git commit -m "feat(identity): canonical provenance-explicit column vocabulary (Phase 6 Task 1)"`.

---

### Task 2: Route both CSV headers + the worker row writer through `columns.py` (realtime family rename)

**Files:**
- Modify: `src/hydra_suite/trackerkit/headless_tracking.py` (:32-88)
- Modify: `src/hydra_suite/trackerkit/gui/orchestrators/tracking.py` (~:1410-1462)
- Modify: `src/hydra_suite/core/tracking/worker.py` (`_online_identity_row_values` ~:1924-1979; and the header/positional coupling)
- Test: `tests/test_core_qtfree_slice2.py`, `tests/test_trackerkit_tracking_orchestrator_dialogs.py` (update header expectations)

**Interfaces:**
- Consumes: `identity_realtime_columns()` from Task 1.
- Produces: both `build_tracking_csv_header(...)` and the GUI inline header emit the raw columns followed by `identity_realtime_columns()` (then the apriltag `DetectedTag*` block iff `identity_method == "apriltags"`). The worker's `_online_identity_row_values` returns values positionally matching `identity_realtime_columns()`.

- [ ] **Step 1: Write/adjust the failing test.** In `tests/test_core_qtfree_slice2.py` (or a focused new assertion), assert `build_tracking_csv_header(save_confidence_metrics=True, identity_method="cnn_classifier")` contains `IdentityRealtimeLabel` and `IdentityEvidenceSources` and does NOT contain `IdentityAssignedLabel`. Also assert the GUI orchestrator header equals `build_tracking_csv_header(...)` for the same args (kills the duplication drift).

```python
from hydra_suite.trackerkit.headless_tracking import build_tracking_csv_header
from hydra_suite.core.individual.identity import columns as C

def test_header_uses_realtime_family_and_no_legacy():
    hdr = build_tracking_csv_header(True, identity_method="cnn_classifier")
    assert C.REALTIME_LABEL in hdr and C.EVIDENCE_SOURCES in hdr  # IdentityRealtimeLabel, IdentityEvidenceSources
    assert "IdentityAssignedLabel" not in hdr
    # positional block appears contiguously in writer order
    i = hdr.index(C.REALTIME_ID)
    assert hdr[i:i + len(C.identity_realtime_columns())] == C.identity_realtime_columns()
```

- [ ] **Step 2: Run to verify it fails.** Expected FAIL (legacy names still present).

- [ ] **Step 3: Implement.** Replace the hard-coded identity block in `build_tracking_csv_header` with `list(base_cols) + C.identity_realtime_columns()` (+ apriltag block). Make the GUI orchestrator inline header call `build_tracking_csv_header(...)` instead of re-listing (import it; it is Qt-free). Leave the `DetectedTag*` apriltag append exactly as today. In `worker.py`, keep `_online_identity_row_values` returning the same 9 values in the same order (they now correspond to the renamed columns — no value-logic change).

- [ ] **Step 4: Run the header tests + a fast equivalence smoke** (`fly_obb`) to confirm the raw CSV still parses and positions are byte-identical. Expected: header tests PASS; `fly_obb` EQUIVALENT.

- [ ] **Step 5: Commit** — `feat(identity): route CSV headers + row writer through shared realtime-family columns (Phase 6 Task 2)`.

---

### Task 3: Offline solver writes the Final family (stop clobbering realtime)

**Files:**
- Modify: `src/hydra_suite/core/individual/identity/offline.py` (:35-36 constants; :1537-1586 write block; :1591-1638 `_annotate_smoothed_labels`; `run_fragment_solver` :1643+)
- Test: `tests/test_fragment_solver.py`, `tests/identity/test_honesty_fix.py`, `tests/identity/test_provenance_no_clobber.py` (new)

**Interfaces:**
- Consumes: `columns.py` constants; `IdentityFinalSource`.
- Produces: `solve_global_assignment` writes `IdentityFinalLabel/ID/Confidence/FragmentScore` + `IdentityFinalSource` (= `offline` for solver-assigned rows, `""` for solver-`unknown` rows), and NEVER writes any `IdentityRealtime*`. `_annotate_smoothed_labels` writes `IdentityFinalSmoothedLabel/Confidence`. All Final label columns initialized as **object dtype** (carry forward the Phase-5 `LossySetitemError` fix).

- [ ] **Step 1: Write the failing invariant test** (`tests/identity/test_provenance_no_clobber.py`)

```python
import numpy as np, pandas as pd
from hydra_suite.core.individual.identity import columns as C
# (build a small trajectories df with populated IdentityRealtime* + a real
#  evidence cache, run run_fragment_solver, then:)

def test_offline_never_mutates_realtime_columns(fragment_df_with_realtime, catalog, cache, params):
    before = fragment_df_with_realtime[C.REALTIME_LABEL].copy()
    out = run_fragment_solver(fragment_df_with_realtime, catalog, params, cache=cache)
    pd.testing.assert_series_equal(out[C.REALTIME_LABEL], before, check_names=False)
    assert (out[C.FINAL_SOURCE].isin(["offline", ""])).all()
    assert out.loc[out[C.FINAL_LABEL].notna() & (out[C.FINAL_LABEL] != "unknown"),
                   C.FINAL_SOURCE].eq("offline").all()
```

(Provide the fixtures inline in the test module, mirroring `tests/identity/test_honesty_fix.py`'s `_build_realtime_off_df`/`_write_cache`, but with the realtime columns populated to prove non-clobber.)

- [ ] **Step 2: Run to verify it fails.** Expected FAIL (offline still writes `IdentityAssignedLabel`).

- [ ] **Step 3: Implement.** Set `_LABEL_COL = C.FINAL_LABEL`, `_CONF_COL = C.FINAL_CONFIDENCE`; rename the offline-mirror writes to `C.FINAL_*`; add `out[C.FINAL_SOURCE]` assignment (`offline` when a label is committed, `""` on `unknown`); rename smoothed columns to `C.FINAL_SMOOTHED_*`. Keep the object-dtype init (Phase-5 fix) on every Final label column. Remove all references to the legacy names in this file. Migrate `test_fragment_solver.py` + `test_honesty_fix.py` assertions to the new names.

- [ ] **Step 4: Run** `tests/test_fragment_solver.py tests/identity/ -q`. Expected PASS.

- [ ] **Step 5: Commit** — `feat(identity): offline solver writes IdentityFinal* (provenance, no realtime clobber) (Phase 6 Task 3)`.

---

### Task 4: Rich-export evidence summary writes the Evidence family (single owner)

**Files:**
- Modify: `src/hydra_suite/core/individual/postprocess_df.py` (`_annotate_identity_summary_columns` :57-127)
- Test: `tests/test_core_identity_postprocess_df.py`

**Interfaces:**
- Consumes: `columns.py`; the Final family (Task 3); `CNN_*_Class`, `DetectedTag*` inputs; the `IdentityEvidenceCache` top-label/conf if available.
- Produces: `IdentityEvidenceSources` (`,`-joined among `apriltag`/`cnn`/`offline`/`realtime`), `IdentityEvidenceConflictFlag` (int), and the two new `IdentityEvidenceTopLabel`/`IdentityEvidenceConfidence` (per-row top calibrated evidence summary; source from the evidence cache when threaded, else the max `CNN_*_Prob`). Reads `IdentityFinalLabel` (not the legacy assigned) for the `offline` source signal. Does NOT write any realtime column.

- [ ] **Step 1: Write the failing test.** Assert the summary produces `IdentityEvidenceSources`/`IdentityEvidenceConflictFlag` (+ `TopLabel`/`Conf` when CNN cols present) and no legacy `IdentityEvidenceSources`/`IdentityConflictFlag`.
- [ ] **Step 2: Run — fails.**
- [ ] **Step 3: Implement** the rename + the two new evidence columns; drop the double-ownership (the online decoder no longer needs to pre-seed evidence-source columns — they are computed once here).
- [ ] **Step 4: Run** `tests/test_core_identity_postprocess_df.py -q`. PASS.
- [ ] **Step 5: Commit** — `feat(identity): rich-export writes IdentityEvidence* summary (single owner) (Phase 6 Task 4)`.

---

### Task 5: Post-hoc consumers + overlay migrate to Final; derive UniqueIdentityKey; realtime mirror

**Files:**
- Modify: `src/hydra_suite/core/post/identity_postprocess.py` (:131-230)
- Modify: `src/hydra_suite/core/post/processing.py` (:402-416, :1311-1400, :1547-1559, :2657-2678, :3660-3664)
- Modify: `src/hydra_suite/core/post/media_export.py` (:193-224)
- Test: `tests/test_identity_postprocess.py`, `tests/test_identity_conflict_resolution.py`, `tests/test_postproc_invariants.py`, `tests/test_post_tracklet_relinking.py`, `tests/test_media_export.py`, `tests/test_postproc_identity_gating.py`

**Interfaces:**
- Consumes: Final family (Task 3), Evidence family (Task 4), Realtime family (Task 2).
- Produces:
  - Consensus fill + conflict resolution + tracklet relinking operate on `IdentityFinal*` and set `IdentityFinalConflictResolved`. The conflict resolver's per-label pairwise-overlap dominance (Phase 5) is preserved — only the column names change.
  - `UniqueIdentityKey` derived from `IdentityFinalLabel` (+ source), computed once at the end of post-processing.
  - **Realtime→Final mirror:** a single non-destructive step (best placed at the entry of `apply_identity_postprocessing_to_df` when the fragment solver did NOT run, or after it for rows the solver left empty) copies `IdentityRealtime{Label,ID,Confidence}` into `IdentityFinal*` with `IdentityFinalSource = "realtime"`, leaving realtime columns intact. Tag-resolved rows get `IdentityFinalSource = "tag"`.
  - Video overlay priority list → `[UNIQUE_IDENTITY_KEY, FINAL_LABEL, FINAL_SMOOTHED_LABEL]` (i.e. `["UniqueIdentityKey", "IdentityFinalLabel", "IdentityFinalSmoothedLabel"]`).

- [ ] **Step 1: Write/adjust the failing tests** across the listed files to the new names + add an assertion that with post-hoc OFF, `IdentityFinalLabel` equals `IdentityRealtimeLabel` and `IdentityFinalSource == "realtime"`.
- [ ] **Step 2: Run — fails.**
- [ ] **Step 3: Implement** the renames + the realtime→Final mirror + `UniqueIdentityKey` derivation. Update `_IDENTITY_LABEL_COL`/`_IDENTITY_CONFLICT_COL` etc. constants in `processing.py` to the Final names. Ensure no path writes a realtime column.
- [ ] **Step 4: Run** the full listed test set. PASS.
- [ ] **Step 5: Commit** — `feat(identity): post-hoc + overlay on IdentityFinal*; realtime→Final mirror; derive UniqueIdentityKey (Phase 6 Task 5)`.

---

### Task 6: Honest UI — Identity Models panel

**Files:**
- Modify: `src/hydra_suite/trackerkit/gui/panels/identity_panel.py` (:46, :64, :87-394, CNN row :759-984)
- Test: `tests/test_trackerkit_tracking_orchestrator_dialogs.py` (extend the existing panel round-trip test) or a focused new panel test.

**Interfaces:**
- Consumes: `IdentityConfig` via the cfg-dict round-trip (`gui/orchestrators/config.py`).
- Produces: a status line under the master toggle stating verbatim: *"Identity evidence is computed during inference and cached — available to both realtime and post-hoc."*; a per-`unique_identifier` CNN-row calibration status/affordance (fitted / not fitted + a fit action hook); corrected tooltips (no claims about realtime/post-hoc coupling). No config-key changes here — labels/tooltips/status only.

- [ ] **Step 1: Write the failing test** asserting the status-line text exists and the calibration status widget is present for a `unique_identifier` CNN row.
- [ ] **Step 2: Run — fails.**
- [ ] **Step 3: Implement** the status line + calibration affordance + tooltip corrections.
- [ ] **Step 4: Run** the panel test (headless Qt via `QT_QPA_PLATFORM=offscreen`). PASS.
- [ ] **Step 5: Commit** — `feat(trackerkit): Identity Models panel — honest status + calibration affordance (Phase 6 Task 6)`.

---

### Task 7: Honest UI — Realtime Identity (tracking panel) + Post-hoc Identity (post-process panel)

**Files:**
- Modify: `src/hydra_suite/trackerkit/gui/panels/tracking_panel.py` (:516-707)
- Modify: `src/hydra_suite/trackerkit/gui/panels/postprocess_panel.py` (:573-830)
- Modify: `src/hydra_suite/trackerkit/config/identity_schema.py` (:52-55 emit reserved fields)
- Modify: `src/hydra_suite/trackerkit/engine_params.py` (:604-635, :710 thread post-hoc keys)
- Test: `tests/test_postproc_identity_gating.py`, panel round-trip test.

**Interfaces:**
- Consumes: `IdentityConfig` (realtime + posthoc).
- Produces:
  - Tracking panel: the realtime group title/labels/tooltips scoped strictly to *association influence*; the ~:538 tooltip states plainly it affects realtime only and says nothing about post-hoc.
  - Post-process panel: "Assign identities from final trajectories" as a first-class independent toggle (already decoupled in Phase 5 — surface it as the group's own enable, never gated on the realtime flag); add a forward-backward **smoothing** enable that maps to `PostHocIdentityConfig.smoothing_enabled` → a new `IDENTITY_ENABLE_SMOOTHING` engine key consumed by `run_fragment_solver` (the smoother currently always runs when cache present; make it honor the flag, default True). Corrected tooltip (no false "still works with realtime off" claim — it is now literally true, so state it correctly).
  - `identity_schema.py` emits `smoothing_enabled`/`changepoint_enabled`; `engine_params.py` threads them.

- [ ] **Step 1: Write the failing test** asserting: post-hoc toggle enable-state is independent of `enable_identity_in_tracking`; the smoothing checkbox round-trips to `IDENTITY_ENABLE_SMOOTHING`; the tracking tooltip no longer contains any post-hoc claim.
- [ ] **Step 2: Run — fails.**
- [ ] **Step 3: Implement** the label/tooltip re-scope + the smoothing control + the engine-key threading + honor the flag in `run_fragment_solver`.
- [ ] **Step 4: Run** the gating + panel tests. PASS.
- [ ] **Step 5: Commit** — `feat(trackerkit): realtime vs post-hoc panels honest + independent; wire smoothing toggle (Phase 6 Task 7)`.

---

### Task 8: Sweep for stragglers + full-suite green

**Files:**
- Modify: any remaining legacy-name references surfaced by grep (`tests/identity/test_evidence_sidecar_consumption.py`, tooltips in `postprocess_panel.py:780`/`identity_panel.py:806-809`, docstrings).

- [ ] **Step 1:** `grep -rnE "IdentityAssigned|IdentityOffline|IdentitySmoothed|IdentityPosteriorMargin|IdentityEntropy(?!)|IdentityCommitted|IdentitySlotLockLabel|IdentityEvidenceSources|IdentityConflictFlag|IdentityConflictResolved|IdentityFragmentScore" src/ tests/` — expect only intentional retirements (none in output columns). Fix each straggler.
- [ ] **Step 2:** Run the identity-focused suites + the headless suite: `tests/identity/ tests/test_fragment_solver.py tests/test_identity_postprocess.py tests/test_identity_conflict_resolution.py tests/test_postproc_invariants.py tests/test_post_tracklet_relinking.py tests/test_media_export.py tests/test_core_identity_postprocess_df.py tests/test_postproc_identity_gating.py tests/test_core_qtfree_slice2.py tests/test_trackerkit_headless_tracking.py tests/test_trackerkit_tracking_orchestrator_dialogs.py`. All PASS.
- [ ] **Step 3:** `make format-check` + `make lint-moderate` on changed files.
- [ ] **Step 4: Commit** — `chore(identity): retire last legacy identity-column names; Phase 6 sweep green (Phase 6 Task 8)`.

---

## Phase-end gate (controller, after Task 8)

1. **Provenance invariant:** `tests/identity/test_provenance_no_clobber.py` green; grep confirms no legacy output-column names remain.
2. **Positions equivalence (MPS + CUDA):** baseline `0a0a0091` vs Phase-6 HEAD — positions p99 = 0, θ within head/tail π-flip noise floor, unmatched = 0, on all 7 clips. Identity columns are renamed/additive and NOT gated.
3. **Real-clip provenance check:** on `ant_cnn_identity` realtime-ON and realtime-OFF, confirm `IdentityRealtime*` present iff realtime ran, `IdentityFinalSource` ∈ {offline (post-hoc on), realtime (post-hoc off + realtime on)}, realtime columns never clobbered, `UniqueIdentityKey` derived.
4. **Whole-branch review** (opus) over the Phase-6 commit range.
5. Bring the result to the user (checkpoint) before checkpointing Phase 6 on the branch. No merge (Phase 7 pending).

## Self-Review notes (author)

- Spec coverage: Layer 5 three families ✔ (Tasks 1-5), UI reorg three panels ✔ (Tasks 6-7), tooltip correction ✔, `UniqueIdentityKey` retained ✔, "no stage overwrites" invariant ✔ (Task 3 + `test_provenance_no_clobber`).
- The `IdentityEvidenceTopLabel`/`Conf` are genuinely new (no legacy source) — Task 4 sources them from the evidence cache when threaded, else max `CNN_*_Prob`; acceptable summary.
- `IdentitySmoothed*` is not in the spec's Layer-5 list; retained as `IdentityFinalSmoothed*` diagnostics rather than dropped, to avoid losing the Phase-5 per-frame smoothed decode. Flag for user if they'd rather drop it.
- Risk: the two-header duplication + positional writer (Task 2) is the byte-alignment landmine; the `fly_obb` smoke in Task 2 Step 4 catches misalignment early.
