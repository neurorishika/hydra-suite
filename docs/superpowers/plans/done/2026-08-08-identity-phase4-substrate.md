# Identity Overhaul — Phase 4: Shared Substrate + Realtime Read-Through Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract the identity decoders' shared primitives into one `core/individual/identity/substrate.py` — `fuse_log_evidence` (log-space Bayesian fusion, with a no-op robustness cap/floor capability), `solve_unique_assignment` (the partial-injective Hungarian-with-dummy-columns solver), and the single `map_cnn_to_catalog` / `map_tag_to_catalog` mapping — and refactor `online.py` and `evidence_builder.py` to delegate to it. **Behavior-preserving; byte-identical target.**

**Architecture:** Phase 3 already made the online decoder consume pre-built `IdentityEvidence` from the inference sidecar (the "realtime read-through" is done — `online.py` does no catalog mapping, only fuses `log_probs`). Phase 4 removes the *duplication* the spec flags (Layer 3): the online decoder's `_fuse_evidence` and `_hungarian_assignment` become the canonical `substrate.fuse_log_evidence` / `substrate.solve_unique_assignment`; the factor→catalog mapping currently in `evidence_builder.py` (lifted from the old emitter in Phase 3) becomes `substrate.map_cnn_to_catalog` and `evidence_builder` delegates. This is a lift-and-delegate refactor — every extracted function keeps identical math — so `tests/test_identity_online.py` and the Phase-3 evidence tests stay green and the equivalence matrix stays byte-identical. The offline decoder's *different* fusion/uniqueness algorithms (mean-log-evidence, weighted-sum, collision-veto) are intentionally NOT folded here — Phase 5 repoints offline and can adopt `solve_unique_assignment` with its own gate.

**Tech Stack:** Python 3, NumPy, SciPy (`linear_sum_assignment`), pytest.

**Spec:** `docs/superpowers/specs/2026-07-22-identity-overhaul-consolidated-design.md` — Layer 3 (Shared substrate), Layer 4a (Realtime consumer), Rollout "Phase 4". Builds on Phase 3 (`EvidenceBuilder`, evidence sidecar).

## Global Constraints

- **Byte-identical is the gate.** Pure extraction: every substrate function reproduces the exact math it was lifted from. Gate: full equivalence matrix, **MPS + CUDA**, baselined against the **branch commit at the start of Phase 4** (record it). All clips at their determinism floor — including `ant_cnn_identity` identity columns, which must be **unchanged from the Phase-3 branch state** (Phase 4 changes no evidence/decision, only where the fuse/solve code lives). Verify CSV row counts > 1.
- **No algorithm change.** `fuse_log_evidence` defaults to cap=+∞ / floor=0 (exact no-op vs today's `_fuse_evidence`). `solve_unique_assignment` reproduces `_hungarian_assignment` + the greedy fallback bit-for-bit. `map_cnn_to_catalog` reproduces `EvidenceBuilder._factor_log_prob` + `_build_log_probs_from_posteriors` bit-for-bit (Phase-3 parity tests guard it).
- **Offline is out of scope.** Do NOT touch `offline.py`'s `_iterative_assign`/fusion or `post/processing.resolve_simultaneous_identity_conflicts` — those are different algorithms folded in Phase 5.
- **Isolation.** Work in the existing long-lived worktree `.worktrees/identity-phases-3-7` (branch `feat/identity-phases-3-7`). Do NOT create a new worktree; do NOT merge to main.
- **Dependency direction.** `substrate.py` is Core — imports only Core/stdlib/numpy/scipy. `online.py` and `evidence_builder.py` (both Core) import it.
- **Commit as the configured git user.** No `Co-Authored-By: Claude` trailer.
- **Before commit:** `make format` then `make lint`. Revert unrelated drift. Kill stale `sleap`/`hydra` before heavy runs.
- **Verification:** unit tests on `hydra-mps`; equivalence gate on MPS + CUDA (mehek).

---

## Current-state anchors (verified; worktree paths; line numbers may drift — locate by content)

Base: `.worktrees/identity-phases-3-7/src/hydra_suite/core/individual/identity/`

- **`online.py` `OnlineIdentityDecoder`:** `_fuse_evidence(belief, evidences)` (L333-351): per-evidence `belief.log_posterior += ev.log_probs` (skips size mismatch with a warning), then `-= logaddexp.reduce(...)`, `hit_count += 1` — **no cap/floor today**. `_hungarian_assignment(visible_slots, linear_sum_assignment)` (L515-551): N×(K+N) cost (K identity cols `-log(prob[j+1])` + N dummy cols `-log(prob[0])`), `linear_sum_assignment`, display-threshold gate → `label_of` or None. `_solve_visible_assignment` (L497-513): scipy-or-greedy dispatcher. `_greedy_assignment` (L553-569): argmax over known, display-threshold + `used` set. `_posterior_probs` (L366-370), `_renormalize_log_probs` (L240-243). `update_frame` (L381-495) orchestrates predict→fuse→swap→bias→assign→commit.
- **`evidence_builder.py` `EvidenceBuilder`:** `_factor_class_to_catalog` map built in `__init__` (L154-167); `_factor_log_prob(factor_index, factor_probs)` (L229-279); `_calibrate_posterior` (L281-296); `_build_log_probs_from_posteriors` (L298-317, product over factors in log-space); `build_frame_evidences` (L186-223).
- **Tag mapping:** `catalog.py` `IdentityCatalog.apriltag_log_prior(tag_id, tag_to_label, floor=1e-4)` — the one tag→catalog log-prior (used by `IdentityEvidenceStage`/`from_apriltag`).
- **Guardrail tests:** `tests/test_identity_online.py` (uniqueness/conflict, commit override, Hungarian display-threshold, blocked-label prior, respawn, full swap suite — all drive `update_frame`). Phase-3 evidence tests: `tests/identity/test_evidence_builder_parity.py`, `test_evidence_phase_basis_parity.py`, `test_evidence_sidecar_consumption.py`, `test_evidence_stage_runner.py`.
- **`substrate.py` does NOT exist yet.**

---

## File Structure
**Create:** `src/hydra_suite/core/individual/identity/substrate.py`; `tests/identity/test_substrate.py`.
**Modify:** `online.py` (delegate `_fuse_evidence` + `_hungarian_assignment`/`_greedy_assignment` to substrate); `evidence_builder.py` (delegate the CNN factor→catalog mapping to substrate).

## Interfaces (defined once)
```python
# core/individual/identity/substrate.py
def fuse_log_evidence(log_posterior: np.ndarray, evidence_log_probs: np.ndarray,
                      *, per_frame_cap: float = float("inf"), prob_floor: float = 0.0) -> np.ndarray:
    """One log-space Bayesian update: add evidence (optionally cap the per-entry log shift and
    floor probabilities), then renormalize (logaddexp). cap=inf/floor=0 == today's exact behavior."""

def solve_unique_assignment(posterior_probs: list[np.ndarray], num_known: int,
                            display_threshold: float, *, use_scipy: bool = True
                            ) -> list[int | None]:
    """Partial-injective assignment: N×(K+N) Hungarian (K identity + N dummy-unassigned cols),
    display-threshold gate; scipy or greedy fallback. Returns per-slot known-index (1-based into
    catalog) or None. Bit-for-bit reproduction of online._hungarian_assignment/_greedy_assignment."""

def map_cnn_to_catalog(per_factor_probs: list[np.ndarray],
                       factor_class_to_catalog: dict[tuple[int, str], list[int]],
                       factor_class_names: list[list[str]], catalog_size: int) -> np.ndarray:
    """CNN per-factor posteriors → calibrated catalog log-probs (product over factors). Bit-for-bit
    reproduction of EvidenceBuilder._factor_log_prob + _build_log_probs_from_posteriors."""

def map_tag_to_catalog(catalog, tag_id, tag_to_label, floor: float = 1e-4) -> np.ndarray:
    """Tag → catalog log-prior. Delegates to catalog.apriltag_log_prior (the one tag mapping)."""
```

---

## Task 1: `substrate.solve_unique_assignment` (lift the Hungarian + greedy)

**Files:** Create `substrate.py` (this function); Test `tests/identity/test_substrate.py`.

- [ ] **Step 1: Failing test.** READ `online.py:497-569` verbatim first. Write tests asserting `solve_unique_assignment` reproduces the exact assignment for hand-built posteriors: (a) two slots, disjoint high-confidence identities → each gets its own; (b) two slots both favoring the same identity → uniqueness forces one to a dummy/None; (c) below-`display_threshold` → None; (d) `use_scipy=False` greedy path matches scipy path on a non-degenerate case. Assert exact index/None equality.

```python
# tests/identity/test_substrate.py (sketch — fill from the real online.py logic)
import numpy as np
from hydra_suite.core.individual.identity.substrate import solve_unique_assignment

def _p(*ps):  # build a catalog posterior [unknown, k1, k2, ...]
    a = np.array(ps, dtype=np.float64); return a / a.sum()

def test_disjoint_identities_assigned():
    # 2 slots, 2 known; slot0->k1, slot1->k2
    post = [_p(0.01, 0.98, 0.01), _p(0.01, 0.01, 0.98)]
    out = solve_unique_assignment(post, num_known=2, display_threshold=0.6)
    assert out == [1, 2]

def test_uniqueness_forces_one_off():
    post = [_p(0.01, 0.98, 0.01), _p(0.02, 0.97, 0.01)]  # both want k1
    out = solve_unique_assignment(post, num_known=2, display_threshold=0.6)
    assert 1 in out and out.count(1) == 1  # only one gets k1; other -> None or k2-below-thresh->None

def test_below_display_threshold_is_none():
    post = [_p(0.5, 0.5, 0.0)]  # k1 prob 0.5 < 0.6
    assert solve_unique_assignment(post, num_known=2, display_threshold=0.6) == [None]

def test_greedy_matches_scipy():
    post = [_p(0.01,0.9,0.09), _p(0.01,0.2,0.79)]
    assert solve_unique_assignment(post, 2, 0.6, use_scipy=True) == solve_unique_assignment(post, 2, 0.6, use_scipy=False)
```

- [ ] **Step 2: Run — FAIL** (module missing).
- [ ] **Step 3: Implement** by lifting `_hungarian_assignment` (L515-551) + `_greedy_assignment` (L553-569) into `solve_unique_assignment`, parameterized by `posterior_probs` (list of catalog prob vectors), `num_known`, `display_threshold`, `use_scipy`. Keep the cost-matrix construction, `-log(max(p,1e-300))`, dummy columns, and threshold gate identical. Return per-slot 1-based known index or None (the caller maps index→label).
- [ ] **Step 4: Run — PASS.**
- [ ] **Step 5: `make format && make lint`; commit** `feat(identity): substrate.solve_unique_assignment (partial-injective Hungarian, lifted from online)`.

---

## Task 2: `substrate.fuse_log_evidence` (lift fusion + no-op cap/floor)

**Files:** `substrate.py` (add); `tests/identity/test_substrate.py` (append).

- [ ] **Step 1: Failing test.** READ `online.py:333-351`. Test: (a) `fuse_log_evidence(lp, ev)` with default cap/floor equals `lp + ev` then `-= logaddexp.reduce` (exact, `np.array_equal`); (b) a finite `per_frame_cap` clamps the per-entry log shift to ±cap; (c) `prob_floor>0` keeps every entry ≥ floor after renorm; (d) size-mismatch handling matches online (the caller currently skips mismatched evidence — decide whether substrate raises or the caller guards; mirror online's skip-with-warning at the CALLER, keep substrate pure/raising on mismatch).

- [ ] **Step 2: Run — FAIL.**
- [ ] **Step 3: Implement** — default path is exactly `renorm(log_posterior + evidence_log_probs)`. Cap: clip `(evidence_log_probs)` contribution to `[-per_frame_cap, +per_frame_cap]` before adding (only when finite). Floor: after renorm, `log_posterior = logaddexp(log_posterior, log(prob_floor))`-style flooring then renorm (define precisely so floor=0 is a no-op). Keep the default byte-identical to `_fuse_evidence`'s core.
- [ ] **Step 4: Run — PASS.**
- [ ] **Step 5: commit** `feat(identity): substrate.fuse_log_evidence (log-space fusion + no-op cap/floor)`.

---

## Task 3: `online.py` delegates to the substrate

**Files:** Modify `online.py`; guardrail = `tests/test_identity_online.py` + equivalence gate.

- [ ] **Step 1:** Refactor `_fuse_evidence` (L333-351) to call `substrate.fuse_log_evidence(belief.log_posterior, ev.log_probs)` per evidence (keeping the size-mismatch skip+warning + `hit_count += 1` in `online`). Refactor `_solve_visible_assignment`/`_hungarian_assignment`/`_greedy_assignment` (L497-569) to build the per-slot `posterior_probs` list and call `substrate.solve_unique_assignment(...)`, then map returned indices → labels via `catalog.label_of`. Keep all surrounding logic (predict, slot-lock bias, swap, commitment) untouched.
- [ ] **Step 2:** Run `python -m pytest tests/test_identity_online.py -v` — ALL pass (uniqueness/conflict, Hungarian display-threshold, swap suite, respawn, blocked-label). This is the black-box proof the refactor is behavior-preserving.
- [ ] **Step 3:** Run `tests/identity/` — no regression.
- [ ] **Step 4: Equivalence smoke** (fastest clips + `ant_cnn_identity`) on MPS vs the Phase-4-start baseline: positions byte-identical AND `ant_cnn_identity` identity columns unchanged from the Phase-3 branch state. (Phase 4 must not perturb identity at all.)
- [ ] **Step 5: commit** `refactor(identity): online decoder uses shared substrate (fuse + unique-assignment)`.

---

## Task 4: `substrate.map_cnn_to_catalog` / `map_tag_to_catalog` + `evidence_builder` delegates

Unify the factor→catalog mapping so Layer 2 (evidence stage, via `evidence_builder`) and future Layer 4b (offline, Phase 5) share one implementation.

**Files:** `substrate.py` (add map fns); `evidence_builder.py` (delegate); `tests/identity/test_substrate.py` (append).

- [ ] **Step 1: Failing test.** Assert `substrate.map_cnn_to_catalog(...)` reproduces `EvidenceBuilder._build_log_probs_from_posteriors` output bit-for-bit for a multi-factor case with a `_`-containing class (reuse the Phase-3 parity fixtures). Assert `map_tag_to_catalog(catalog, tag_id, tag_to_label)` == `catalog.apriltag_log_prior(tag_id, tag_to_label)`.
- [ ] **Step 2: Run — FAIL.**
- [ ] **Step 3: Implement** `map_cnn_to_catalog` by lifting `EvidenceBuilder._factor_log_prob` + `_build_log_probs_from_posteriors` (keep floor `1e-6`, product-over-factors, renorm identical); `map_tag_to_catalog` delegates to `catalog.apriltag_log_prior`. Then refactor `EvidenceBuilder` to call `substrate.map_cnn_to_catalog` (passing its `_factor_class_to_catalog`) — keeping `EvidenceBuilder`'s construction of that map + calibration.
- [ ] **Step 4: Run** `tests/identity/test_evidence_builder_parity.py test_evidence_phase_basis_parity.py test_substrate.py` — all pass (the Phase-3 parity tests prove `evidence_builder` still produces identical evidence).
- [ ] **Step 5: commit** `refactor(identity): substrate owns the one factor→catalog + tag→catalog mapping; evidence_builder delegates`.

---

## Phase-End Gate
- [ ] **Full equivalence — MPS**, baseline = Phase-4-start branch SHA vs branch HEAD. Every clip byte-identical INCLUDING identity columns (Phase 4 changes no decisions). `ant_cnn_identity` must match the Phase-3 branch state exactly (same accepted improvement, no new delta). Verify row counts > 1.
- [ ] **Full equivalence — CUDA (mehek)** same baseline.
- [ ] **Guardrail suites green:** `tests/test_identity_online.py` + `tests/identity/`.

## Self-Review (against spec Layer 3 + Phase 4)
- "substrate centralizes map/fuse/solve" → Tasks 1,2,4. ✅
- "used by online per-frame" → Task 3. ✅ "also used by Layer 2 (evidence stage)" → Task 4 (evidence_builder delegates). ✅
- "fuse with robustness cap/floor" → Task 2 (capability added, no-op default; RobustnessConfig population/application deferred — no values wired yet). ✅ documented.
- "one uniqueness solver used by online, offline global assignment, post conflict-resolution" → online in Task 3; **offline + post folded in Phase 5** (different algorithms; out of scope here). ✅ documented.
- Byte-identical: pure lift-and-delegate; guarded by `test_identity_online.py`, Phase-3 parity tests, and the equivalence gate. ✅
- **Deferred:** offline `_iterative_assign`/fusion + `resolve_simultaneous_identity_conflicts` unification (Phase 5); RobustnessConfig cap/floor value wiring + application (later).
