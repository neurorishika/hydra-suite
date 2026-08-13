# Identity Subsystem — Scientific-Logic Audit

**Date:** 2026-08-13
**Scope:** `src/hydra_suite/core/individual/identity/` (online + offline), its consumers
(`core/assigners/hungarian.py`, `core/tracking/worker.py`, `core/post/processing.py`,
`core/individual/postprocess_df.py`, `core/post/identity_postprocess.py`), and the config
surface (`trackerkit/config/identity_schema.py`, `trackerkit/engine_params.py`).
**Method:** static probabilistic reasoning over the code as written, plus one targeted
empirical probe run against the real `OnlineIdentityDecoder` (no video required — see
Appendix A). No production code was changed.

---

## 1. Executive summary — the five most important scientific flaws, ranked

1. **Uncapped naive-Bayes fusion of correlated evidence (F1).** The decoder sums per-frame
   CNN log-posteriors as if consecutive crops were independent observations. They are not
   (same pose, same lighting, same occluder), so confidence inflates roughly linearly in
   nats per frame with no ceiling. **Measured on the real decoder: a cold slot commits at
   0.85 after 4 frames of 0.6-confidence evidence; a fully committed *and slot-locked*
   identity on an isolated animal flips to a different identity after 5 frames of
   0.6-confidence wrong evidence, and flips back 5 frames later.** The robustness
   mechanisms designed to prevent exactly this (`per_frame_evidence_cap`, `prob_floor`)
   exist in `substrate.fuse_log_evidence` but are dead code: no call site passes them and
   no engine param emits them. This is the root of failure modes 1 and 2.

2. **The system cannot say "I don't know" (F2).** Every CNN evidence vector assigns the
   `unknown` state a hard floor of 1e-6 *before normalization* (`catalog.cnn_log_prior`,
   `substrate._factor_log_prob`). One frame of any CNN output drives
   p(unknown) to ~1e-8 (measured). The unknown state — the only fail-safe hypothesis in
   the model — is annihilated by construction, so every ambiguous or out-of-catalog animal
   is forced into a catalog identity. All downstream "unassigned" logic (Hungarian dummy
   columns priced at −log p(unknown) ≈ 18) is thereby neutered.

3. **The two anti-flip guards are, respectively, a sign bug and a tautology (F3, F4).**
   The slot-lock "boost" adds `log(0.9) = −0.105` to the locked label — it *penalizes*
   the locked identity every frame (measured: p(locked) decreases after the bias). And the
   0.5 commit-override margin is vacuous: a challenger needs confidence ≥ 0.85 to reach the
   check at all, which forces the incumbent ≤ 0.15, so the margin ≥ 0.70 > 0.5 holds
   automatically. Net effect: the *only* real protection on a committed identity is the
   commit threshold itself, which flaw #1 makes trivially reachable in ~5 frames.

4. **Motion and appearance never share a probability model (F6, F7, F11).** Motion enters
   identity nowhere (the belief is appearance + stickiness only); identity enters motion
   through three separately-thresholded side channels (capped association addon,
   identity-first rejoin at a *single-frame* 0.5 evidence threshold, offline
   bridge-velocity veto with a 30-frame gap cap). Each channel can override the others,
   there is no joint likelihood anywhere, and the rejoin channel in particular converts one
   misclassified frame into a within-trajectory teleport (failure mode 3).

5. **The offline "correction" layer cannot correct the dominant realtime failure (F14),
   and its objective is not a probability (F11).** With PELT splitting off by default, the
   fragment solver can only relabel *whole trajectories*; a mid-track flip inside an
   unbroken track (failure mode 1) is invisible to it — majority evidence wins the whole
   fragment. Its objective is a weighted blend of geometric-mean log-evidence,
   a spatial Gaussian, and a length heuristic, compared against an absolute margin of 0.10
   — a score with no likelihood semantics, later written to the output as
   `IdentityFinalConfidence`.

A cross-cutting conclusion for the config audit (§3): the shipped defaults are **not** the
main problem. The disabled features (robustness caps, PELT, z-score breaks, the
identity→geometry coupling) would mitigate some failures, but the flaws above live in code
that *is* active by default. "A correct algorithm switched off" describes the robustness
caps only; the rest is "an unsound algorithm switched on."

---

## 2. The system model as actually built

### 2.1 Realtime path (active by default whenever detection is YOLO-OBB and identity models are configured)

Per frame, after geometric assignment (`worker.py:2935-3075`, `online.py:382-496`):

1. **Predict.** Each visible slot's log-posterior over the catalog (unknown at index 0) is
   pushed through a sticky Markov transition, ε = 0.02 off-diagonal
   (`online.py:326-332`). Note: this transition is a *leak toward uniform* — a forgetting
   factor. It reduces confidence each frame; it does not inflate it.
2. **Fuse.** Every evidence item (calibrated CNN sidecar posterior remapped to the global
   catalog, and/or an AprilTag prior) is added in log space and renormalized
   (`online.py:349-351` → `substrate.fuse_log_evidence:96-97`). This is naive-Bayes
   fusion: it is a valid Bayesian update **iff** each item is a conditionally independent
   likelihood. The cap/floor arguments of `fuse_log_evidence` default to no-ops and no
   caller overrides them.
3. **Swap detection** on pre-lock posteriors: pairs of *committed, visible* slots whose
   posteriors mutually prefer each other's committed labels by ≥ 0.2 above 0.6 for 8
   consecutive qualifying frames exchange committed labels (`online.py:620-753`).
4. **Slot-lock bias:** the committed label's log-posterior gets `+log(strength=0.9)`, i.e.
   −0.105 nats (`online.py:354-365`).
5. **Uniqueness assignment** among *currently visible* slots only: Hungarian on
   −log p with N dummy columns priced at −log p(unknown), then a 0.6 display gate
   (`online.py:498-522`, `substrate.py:177-211`).
6. **Commit state machine** (`online.py:524-618`): commit at confidence ≥ 0.85 with ≥ 5
   evidence hits, blocked if any other slot (visible or lost) holds the label; commitment
   revision and lock release both gated by the (vacuous) 0.5 margin; slot-lock engages
   after 30 stable frames. The *reported* label is the committed label regardless of the
   current posterior argmax (`online.py:481-482`).

Slot lifecycle: lost slots' committed beliefs are kept and decayed
(`decay_absent_slot_beliefs`); respawned slots get a fresh prior mixed with a
0.75-strength, 0.97-per-frame-decayed copy of the *previous occupant's* posterior for up
to 120 frames (`online.py:270-300`) — an identity prior keyed to the **slot index**, not
to space or appearance.

**Identity→geometry coupling (OFF by default,** `ENABLE_IDENTITY_ONLINE_DECODER=False`):
(a) a capped additive term `α·(−log ⟨belief, evidence⟩)` on the assignment cost
(`hungarian.py:223-271`); (b) identity-first rejoin of committed-lost slots: any
unassigned detection whose single-frame evidence inner product with the slot's belief
exceeds 0.5, within a motion budget that grows linearly with lost duration, is claimed —
KF hard-reset to the detection, trajectory ID preserved (`hungarian.py:801-868`,
`worker.py:3132-3145`); (c) committed-lost slots are reserved from slot reuse
(`worker.py:3384-3389`).

**Deviations from a coherent Bayesian treatment:** correlated evidence fused as
independent (F1); posteriors fused as likelihoods with an emission model that structurally
excludes `unknown` (F2); mutual exclusion enforced only among visible slots per frame,
softly, post-hoc (F9); an irreversible discrete commit layer whose guards don't function
(F3, F4); motion absent from the model (F6/F7); and when the coupling flag is on, the same
evidence is used both to *decide the assignment* and, after assignment, to *update the
belief* that produced the assignment cost — a feedback loop (F6).

### 2.2 Offline path (fragment solver — ON by default)

`postprocess_df.apply_identity_postprocessing_to_df:273-289` → `run_fragment_solver`
(`offline.py:1270-1441`):

1. Per-trajectory evidence pulled from the always-written evidence sidecar, multi-source
   fused per detection (`smoothing.load_trajectory_evidence`).
2. Forward-backward smoothing with the same ε = 0.02 sticky chain (default on). The
   two-filter combination `α + β − evidence` is **mathematically correct** for a uniform
   initial prior — this part is sound.
3. Optional PELT changepoint split on the smoothed posterior — **off by default**, so
   fragments = trajectories.
4. `solve_global_assignment` → `_iterative_assign` (`offline.py:470-847`): per-fragment
   label support = normalized `exp(0.40·mean-log-CNN + 0.15·log-tag + 0.25·log-online-prior)`;
   candidate score = support × spatial-Gaussian(implied bridge velocity, σ = 50 px/f,
   gap capped at 30 frames) × length factor; hard vetoes for time-overlap collision and
   velocity > 50 px/f; greedy doubt-ordered relabeling with a 0.10 absolute margin gate,
   one-step swaps, and an Unknown-rescue pass. Base assignment via the shared Hungarian
   solver per temporal-overlap component (a sound construction).
5. `resolve_simultaneous_identity_conflicts` (`processing.py:1485-1552`): pairwise
   dominance per label; the loser's identity columns are *stripped to unknown*, not
   re-solved.

**Inactive/dead pieces under shipped defaults:** PELT (off), velocity z-score breaks
(`MAX_VELOCITY_ZSCORE` default 0.0, `engine_params.py:913`), robustness cap/floor (never
emitted, never passed), `source_weights` (no consumer anywhere in `src/`), and the
solver's 0.25-weight "online prior" (reads `IdentityFinalLabel`, which does not exist yet
at solver time — see F12).

---

## 3. Config-defaults audit

| Knob (schema → engine key) | Default | Actually active on shipped path? |
|---|---|---|
| `IdentityConfig.enabled` | True | Yes (gates CNN classifier config in GUI) |
| `realtime.enabled` → `ENABLE_IDENTITY_IN_TRACKING` | True | **Yes** — decoder instantiated & labels tracks whenever pipeline is YOLO-OBB + catalog non-empty (`worker.py:1822-1858`) |
| `realtime.bayesian_cost_enabled` → `ENABLE_IDENTITY_ONLINE_DECODER` | **False** | Gate for *all three* identity→geometry couplings: Bayesian cost (`hungarian.py:239`), committed-slot map → identity rejoin + slot reservation (`worker.py:2837-2847`). All OFF by default. |
| `realtime.association_weight` → `ASSOCIATION_IDENTITY_HINT_SCALE` | 1.0 | Emitted, but moot while the flag above is off. (Note: the code-level fallback is 0.3, the emitted default is 1.0.) |
| `posthoc.enabled` → `IDENTITY_POSTHOC_ENABLED` | True | Yes |
| `posthoc.fragment_solver_enabled` → `ENABLE_IDENTITY_FRAGMENT_SOLVER` | **True** on the shipped path (`from_engine_config`: `enable_postprocessing=True` ∧ mode default `"Fragment Solver"`, `identity_schema.py:110-115`) | Yes, whenever the catalog has entries. The dataclass default `False` is misleading; the derivation always runs. **Seed hypothesis "solver disabled by default" is REFUTED.** |
| `posthoc.smoothing_enabled` → `IDENTITY_ENABLE_SMOOTHING` | True | Yes — and it feeds the solver's evidence stats (see F13) |
| `posthoc.changepoint_enabled` → `ENABLE_PELT_SPLITTING` | False | **Off** — solver granularity is whole trajectories (F14) |
| `RobustnessConfig.per_frame_evidence_cap` / `prob_floor` | 0.0 / 0.0 | **Dead**: no engine key exists; every call to `fuse_log_evidence` (online `online.py:349`, offline `smoothing.py:133,265,271`) uses the no-op defaults |
| `RobustnessConfig.source_weights` | `{}` | **Dead**: zero consumers in `src/` |
| `IDENTITY_SLOT_LOCK_MIN_FRAMES/STRENGTH/OVERRIDE_MARGIN`, `IDENTITY_COMMIT_MIN_HITS`, `IDENTITY_RESPAWN_PRIOR_*` | 30 / 0.9 / 0.5 / 5 / 0.75-0.97-120 | Hardcoded decoder defaults — **not emitted by `engine_params.py` at all**, so not user-tunable |
| `MAX_VELOCITY_ZSCORE` | 0.0 | **Off** — the kinematic swap detector never runs; only the hard 50 px/f break and cross-gap spatial split act |
| `IDENTITY_GATES_TRAJECTORY_STRUCTURE` | True | Yes (identity-disagree splits in merge) |
| `calibration_required` | False | Calibration optional; evidence may be raw softmax |

**Answer to the key question:** the shipped default system is (decoder labels tracks) +
(fragment solver relabels whole trajectories). The degraded pieces — no robustness caps,
no changepoints, no kinematic breaks, no geometry coupling — *worsen* the failures but do
not explain them: F1–F5 (the flip machinery) are fully active by default, and F11–F14 (the
offline blind spots) are properties of the active solver. One symptom *is* purely
config-dependent: the within-trajectory geometric teleport (mode 3, path A) requires
`ENABLE_IDENTITY_ONLINE_DECODER=True`, so if it was observed, that run had the flag on —
worth confirming against the run's config.

---

## 4. Findings

Format: **name** · type · evidence · flawed assumption · failure modes · severity ·
confirmation.

### F1 — Correlated-evidence naive-Bayes fusion, uncapped
- **Type:** independence fallacy + miscalibration.
- **Evidence:** `online.py:349-351` → `substrate.fuse_log_evidence` (`substrate.py:96-97`);
  cap/floor parameters exist (`substrate.py:41-43`) but every caller uses defaults;
  `RobustnessConfig` (`identity_schema.py:63-67`) is never emitted
  (no `PER_FRAME_EVIDENCE_CAP`/`PROB_FLOOR` key in `engine_params.py`).
- **Flawed assumption:** per-frame CNN posteriors are conditionally independent
  likelihoods given identity. Consecutive crops of the same animal are highly correlated;
  a systematic misread (pose, lighting, partial occlusion) produces a *streak*, and the sum
  of a streak's log-odds is treated as k independent confirmations.
- **Consequence (measured, Appendix A):** cold start → commit in 4 frames at 0.6
  confidence; committed+locked → flipped in 5 frames of 0.6-confidence wrong evidence;
  flips back in 5. The posterior saturates within ~3 frames of any consistent signal.
- **Failure modes:** 1 (isolated flip), 2 (classification overrides everything).
- **Severity:** critical. **Confirmed:** by probe run.

### F2 — Unknown-state annihilation (no open-set model)
- **Type:** miscalibration / fail-confident by construction.
- **Evidence:** `catalog.py:220-221` (`cnn_log_prior`: unknown = floor 1e-6),
  `substrate.py:257-258` (`_factor_log_prob`: same), `catalog.py:188-189` (AprilTag:
  unknown = 1e-4). The CNN has no "none of the above" output, so unknown always gets the
  floor.
- **Flawed assumption:** that a K-way closed-set softmax says anything about
  p(unknown). Fusing one such vector multiplies unknown's mass by ~1e-6 relative to knowns.
  Measured: p(unknown) = 2.3e-7 after **one** frame.
- **Consequences:** (a) the belief can never retreat to "uncertain" — the fail-safe state
  is unreachable; (b) the Hungarian dummy columns (cost −log p(unknown) ≈ 18 vs ≈ 0 for
  labels) make "leave unassigned" essentially never optimal, so uniqueness pressure always
  resolves by assigning *some* label (the 0.6 display gate is the only remaining brake);
  (c) out-of-catalog animals are forced into catalog identities.
- **Failure modes:** 1, 2. **Severity:** critical. **Confirmed:** code + probe.

### F3 — Slot-lock bias has the wrong sign (a "lock" that penalizes)
- **Type:** implementation bug in a core stabilizer.
- **Evidence:** `online.py:363-365`: `log_bias = log(max(strength, 1e-6))`; strength
  default 0.9 → bias = −0.105 added to the locked label, then renormalize. The docstring
  ("fraction of probability mass the lock diverts toward the locked label",
  `online.py:72-73`) describes a mechanism that is not implemented.
- **Measured:** p(locked label) *decreases* after `_apply_slot_lock_bias` (0.999343 →
  0.999270). Because the posterior persists across frames, this is a small recurring
  erosion of the committed identity, on top of the ε-transition leak.
- **Failure modes:** 1, 2 (the intended defense is absent; actual effect mildly
  destabilizing). **Severity:** high (mechanism inert), low (magnitude).
  **Confirmed:** probe.

### F4 — The 0.5 override margin is a tautology; commit revision is single-gate
- **Type:** tautology / threshold interaction.
- **Evidence:** `online.py:543-548` (must have confidence ≥ 0.85 to reach the check);
  `online.py:555-561` and `573-580` (require `confidence − incumbent ≥ 0.5`). Since the
  posterior sums to 1, confidence ≥ 0.85 ⟹ incumbent ≤ 0.15 ⟹ margin ≥ 0.70. The gate can
  never fail. The same is true of the lock-release check.
- **Flawed assumption:** that a probability-space margin adds protection beyond the commit
  threshold. It cannot, because both quantities live on the same simplex.
- **Failure modes:** 1, 2 — combined with F1, commitment is revised the moment the
  posterior crosses 0.85, i.e. after a handful of wrong frames.
- **Severity:** high. **Confirmed:** algebraic + probe (flip occurred the frame p(B) hit
  0.947 with zero extra resistance).

### F5 — Sticky-prior circularity: seed hypothesis **refuted**, replaced by a real reporting inconsistency
- **Type:** (refutation + minor finding).
- The seed hypothesis said the sticky transition prior inflates confidence because the
  slot was previously committed — circular. **This is wrong:** the ε-transition
  (`online.py:326-332`) mixes the posterior *toward uniform* each frame; it is a
  forgetting factor, the correct hedge. Confidence inflation comes from F1, not the prior.
- The real (smaller) issue: once committed, the *reported* label is `committed_label`
  regardless of the current posterior (`online.py:481-482`), while the reported
  `confidence` is the posterior of the *assigned* label — so during a contested period the
  output row can pair a stale committed label with a confidence belonging to a different
  label. Provenance columns are therefore not self-consistent row-wise.
- **Severity:** low-medium (analysis hazard). **Confirmed:** reasoning.

### F6 — Assignment→evidence feedback loop and inconsistent twin likelihoods (flag-gated)
- **Type:** double-counting / feedback.
- **Evidence:** with `ENABLE_IDENTITY_ONLINE_DECODER=True`, the same frame's identity
  evidence is used (a) in the association cost that decides which track gets which
  detection (`worker.py:2704-2755` → `hungarian.py:223-271`), then (b) fused into the
  *winning* track's belief (`worker.py:2966-3072`). Belief → cost → assignment → belief.
- Additionally the two uses build **different likelihoods from the same data**: the cost
  path reconstructs an *uncalibrated* top-1 spread (`worker.py:2721-2746`,
  `catalog.cnn_log_prior`), the fusion path uses the calibrated full sidecar posterior
  (`worker.py:3044-3068`). Same observation, two incompatible probability models.
- **Flawed assumption:** that routing evidence by a belief and then updating that belief
  with the routed evidence is a valid inference step. It is a self-confirming loop: a
  wrong belief biases assignment toward identity-consistent detections, which then
  reinforce the wrong belief.
- **Failure modes:** 2, 3 (when flag on). **Severity:** high (flag-gated).
  **Confirmed:** reasoning over code.

### F7 — Identity-first rejoin: one uncalibrated frame ⇒ within-trajectory teleport
- **Type:** unenforced constraint / fail-confident, threshold interaction.
- **Evidence:** `hungarian.py:801-868`. Score = `logsumexp(log_post + log_like)` =
  log⟨posterior, evidence⟩ — an inner product of two *distributions*, not p(same
  identity). For a committed slot (posterior ≈ δ_L) the criterion reduces to: **one
  frame's** detection evidence at L > 0.5 — evidence built from the *uncalibrated top-1*
  path (F6), for which unknown is annihilated (F2). Note 0.5 < display threshold 0.6: a
  label too weak to display can still teleport a track.
- Motion budget (`hungarian.py:825-832`): `max(2·body, lost_frames · 2·body · 1.5)` —
  grows linearly without cap; after 100 lost frames at body = 20 px the budget is 6000 px.
  The `missed_frames is None` bypass (`:826-827`) is real but **unreachable on the live
  path** — the worker always passes `missed_frames` (`worker.py:2860`). Seed hypothesis
  "gate disabled in production" is **refuted**; the exposure is the growth law and the
  single-frame trigger, not a disabled gate.
- On success the KF is hard-reset to the detection with zeroed velocity and the
  **trajectory ID is preserved** (`worker.py:3132-3145`) — this is the only code path that
  moves an existing trajectory a long distance in one frame. No confirmation window, no
  hysteresis: if the claim was wrong, the slot is typically outcompeted within frames,
  goes lost again, and can rejoin back near the original location — the full
  jump-and-return signature.
- **Failure modes:** 3 (primary), 2. **Severity:** critical when flag on.
  **Confirmed:** reasoning; empirical attribution experiment proposed (§7).

### F8 — Swap correction structurally cannot fix an isolated flip
- **Type:** design gap (confirmed seed hypothesis).
- **Evidence:** `online.py:620-716` requires **two committed, visible** slots with
  *mutual* mismatch (each ≥ 0.6 and ≥ 0.2 above own label) sustained 8 consecutive
  qualifying frames; the pair counter is dropped on any frame the pair isn't
  both-committed-visible-mismatched (`online.py:677,679-681`).
- For failure mode 1 (no other animal nearby) there is no partner, so the only corrective
  path is commitment revision — which, per F1+F4, is exactly the mechanism that caused the
  flip and will flap back and forth with the evidence streaks. The consecutive-frames
  requirement also makes the swap fragile precisely during the occlusion-heavy
  interactions where true swaps happen.
- **Failure modes:** 1. **Severity:** high. **Confirmed:** code + probe (H-C/H-E flapping).

### F9 — Mutual exclusion is per-frame, visible-only, and soft; lost committed slots create deadlocks and slot-keyed priors
- **Type:** unenforced physical constraint.
- **Evidence:** uniqueness solved only over *visible* slots (`online.py:443,498-522`); a
  lost committed slot retains its label indefinitely and blocks any other slot from
  *committing* it (`online.py:533-541`) while the visible animal can still be *assigned*
  and displayed with that label — two slots claiming one identity across the
  visible/lost boundary. With the decoder flag on, committed-lost slots are additionally
  reserved from reuse (`worker.py:3384-3389`) and recoverable only via rejoin (F7): the
  real animal's new slot can never commit, while the stale slot waits to teleport onto any
  >0.5 detection. With the flag off (default), the slot is eventually reused and its
  belief is carried into the *new occupant* via `RespawnPrior` at strength 0.75
  (`online.py:270-300`) — an identity prior attached to a **slot index**, which is an
  arbitrary bookkeeping quantity, not a physical one.
- Also: `_initial_log_posterior` blocks other slots' committed labels at 1e-300
  (`online.py:222-239`), but one ε-transition step restores ~ε/(C−1) mass — the block is
  one-frame cosmetics; semantics are neither hard nor soft (F10 folded in here).
- **Failure modes:** 1, 3. **Severity:** high. **Confirmed:** reasoning.

### F11 — The offline objective is a category error, and its output is mislabeled "confidence"
- **Type:** category error / pseudo-probability (confirmed seed hypothesis).
- **Evidence:** `offline.py:529-543`: per-label support =
  `normalize(exp(0.40·mean-log-CNN + 0.15·log-tag + 0.25·log-prior))`. The mean-log
  (geometric mean) discards **evidence count** — a 5-frame and a 500-frame fragment with
  the same per-frame mean have identical evidence support; length re-enters only through a
  separate multiplicative heuristic (`offline.py:511-525`). Candidate score = support ×
  spatial Gaussian × length factor (`offline.py:594-620`), accepted if the improvement
  beats an absolute 0.10 margin (`offline.py:795,804`) — a threshold on a quantity whose
  scale depends on catalog size, fragment pool, and weights. The converged score is then
  written as `IdentityFinalConfidence` (`offline.py:1204`) — it is not a probability of
  anything.
- The weights (0.40/0.15/0.25/0.60) mix log-likelihood-ish terms with heuristics in one
  additive/multiplicative blend; the "global objective" is greedy coordinate ascent with
  vetoes, not MAP under any generative model. (The per-component Hungarian *base*
  assignment and the collision veto are sound; the scoring is not.)
- **Failure modes:** 2 (offline flavor), 3 (path C below). **Severity:** high.
  **Confirmed:** reasoning.

### F12 — The solver's "online prior" (weight 0.25) is dead on the standard pipeline
- **Type:** disabled-in-practice.
- **Evidence:** `_build_traj_summaries` reads `OnlineLabel` from **`IdentityFinalLabel`**
  (`offline.py:973-993`), but the solver runs *before* the realtime→final mirror
  (`postprocess_df.py:281` vs `:289`), so the column is empty → label "unknown" →
  `_build_prior_log_scores` returns all-zero (uniform) log-scores (`offline.py:107-108`).
  The realtime decoder's actual output (`IdentityRealtime*`) is never consulted by the
  solver. The prior term only activates when re-running the solver over an already-solved
  CSV — where it then double-counts the previous offline pass's own decision as "online"
  evidence.
- **Severity:** medium (a designed input silently absent; and a latent self-feedback on
  re-runs). **Confirmed:** reasoning over call order.

### F13 — Offline evidence statistics computed on *smoothed* posteriors: temporal double-counting
- **Type:** double-counting / miscalibration.
- **Evidence:** `run_fragment_solver` smooths first (default on, `offline.py:1379-1390`),
  then `_evidence_dicts_for_fragment` takes per-frame means of the **smoothed** log
  posteriors (`offline.py:874-887`), and `_fragment_stability` computes agreement/margin
  on the same smoothed vectors (`offline.py:40-75`). After forward-backward smoothing,
  every frame's posterior contains (transition-discounted) copies of *all* frames'
  evidence: the fragment mean re-counts each observation many times, and stability's
  "per-frame agreement" is agreement of a signal that has been forced to agree.
  Anchor confidence and the doubt ordering are therefore systematically inflated.
  PELT (when enabled) also runs on the smoothed signal, where the ε-sticky chain blurs
  exactly the changepoints it is looking for.
- **Severity:** medium-high. **Confirmed:** reasoning.

### F14 — Fragment granularity = trajectory granularity by default: mode-1 errors are offline-invisible
- **Type:** disabled-by-default (PELT) + design consequence.
- **Evidence:** `ENABLE_PELT_SPLITTING` default False (`engine_params.py:942`,
  `identity_schema.py:57`); `solve_global_assignment` assigns **one label per
  trajectory** (`offline.py:1183-1206`). A mid-track identity flip that did not break the
  track geometrically (the isolated-animal case: same detection chain, only the label
  flipped) produces no fragment boundary, so the solver can only pick the majority label
  for the whole track; the minority segment is silently mislabeled. The realtime flip is
  visible in `IdentityRealtime*` but nothing downstream splits on committed-label
  switches.
- **Failure modes:** 1 (uncorrectable offline). **Severity:** high. **Confirmed:** reasoning.

### F15 — Spatial veto's gap cap licenses large teleports; kinematic break disabled; z-test one-sided
- **Type:** threshold interaction + disabled-by-default.
- **Evidence:** `_spatial_score_for_fragment` clamps the gap at `MAX_BRIDGE_GAP_FRAMES=30`
  (`offline.py:283-352`) — intended to stop long gaps from excusing jumps, but combined
  with `MAX_VELOCITY_BREAK=50` px/f it *fixes* the allowed same-identity displacement at
  up to 1500 px for **any** gap ≥ 30 frames, and the score at the veto boundary
  (exp(−2) ≈ 0.14) still clears the 0.05 veto. Meanwhile `MAX_VELOCITY_ZSCORE` defaults to
  0.0 = disabled (`engine_params.py:913`, `processing.py:890`), and when enabled the test
  is positive-z only (`processing.py:528-531`) — a teleport that manifests as an abrupt
  *stop* (or the return leg landing inside a low-motion window) is not tested. (Both legs
  of a jump-and-return are accelerations, so the one-sidedness is a secondary gap; the
  primary gap is that the detector is off.)
- **Failure modes:** 3 (offline flavor). **Severity:** medium. **Confirmed:** reasoning.

### F16 — Conflict resolution discards rather than re-solves
- **Type:** heuristic uniqueness, information-destroying.
- **Evidence:** `resolve_simultaneous_identity_conflicts` (`processing.py:1485-1552`)
  strips the losing trajectory's identity columns to unknown. The loser's rows are not
  offered their second-best identity; the pairwise dominance score is yet another bespoke
  blend (agreement × conf × length + tag bonus + forward tiebreak) — the third distinct
  scoring system for the same question. At least it fails *safe* (to unknown) — the only
  mechanism in the stack that does.
- **Severity:** low-medium. **Confirmed:** reasoning.

### F17 — Three uncoordinated scoring vocabularies
- **Type:** architectural incoherence (summary finding).
- The same question — "which identity does this evidence support, and how strongly?" — is
  answered by (1) the decoder's fused posterior (calibrated sidecar evidence), (2) the
  association/rejoin inner-product on uncalibrated top-1 spreads, and (3) the offline
  weighted-blend support score; with four distinct uniqueness mechanisms (decoder
  Hungarian, commit-blocking, offline collision veto, post-hoc pairwise dominance) none of
  which share state. Inconsistency between paths is guaranteed by construction, which is
  the general form of the seed hypothesis "motion and appearance can override each other"
  — **confirmed**, and broader than motion-vs-appearance.

---

## 5. Failure-mode traces

### Mode 1 — identity flips mid-track with no other animal nearby
Primary path (active by default): CNN produces a correlated wrong-label streak (pose
change, shadow, partial occlusion). Each frame's calibrated posterior is fused as
independent evidence (**F1**, `online.py:349`); unknown cannot absorb the ambiguity
(**F2**); the slot lock provides no resistance — it slightly helps the flip (**F3**);
after ~4–6 frames p(wrong) ≥ 0.85 and `_update_commitment` revises the committed label,
the margin gate passing automatically (**F4**, `online.py:573-580`). Swap correction is
inapplicable — no partner (**F8**). When the streak ends, the same machinery flips back
(**probe H-E**), producing label flapping. Offline, the track has no geometric break, PELT
is off, so the fragment solver assigns the majority label to the whole trajectory
(**F14**) — the flip either disappears (majority correct) or contaminates the entire track
(majority wrong); the frame-accurate flip point is never recovered. Secondary paths:
respawn-prior contamination when the slot was recently reused (**F9**), and
`fill_identity_nans_with_consensus` back-filling a flipped consensus across NaN rows.

### Mode 2 — overconfident classification overrides motion continuity
Motion never enters the identity belief — there is no motion likelihood anywhere in
`online.py`; the belief is appearance + stickiness, so *any* sufficiently long appearance
streak wins over arbitrarily strong motion evidence by construction (**F1/F6/F17**). With
the decoder flag off (default), the override is label-only (mode 1's mechanism). With the
flag on: the Bayesian cost addon reorders geometrically-valid matches (capped —
`hungarian.py:266-269` — a reasonable design), but the rejoin path is uncapped in the
evidence dimension: a single uncalibrated frame at 0.5 claims a detection against motion
(**F7**), and the feedback loop (**F6**) then routes further evidence to confirm the
claim. Offline: evidence weight 0.40 against a spatial score whose no-neighbor default is
0.3 and whose veto tolerates 50 px/frame over a capped gap (**F11/F15**) — a
high-mean-CNN short fragment can take a label from a spatially coherent schedule.

### Mode 3 — sudden jump to a far point, then return
Path A (realtime, requires `ENABLE_IDENTITY_ONLINE_DECODER=True`): committed slot goes
lost → belief reserved (**F9**) → some frames later a far detection (another animal, or a
false positive) gets one CNN frame ≥ 0.5 at the lost slot's label (**F2** guarantees
p(unknown)≈0, **F7**'s threshold is below the display threshold) and sits within the
linearly-grown motion budget → identity rejoin fires: KF hard-reset to the far detection,
**same trajectory ID** (`worker.py:3132-3145`) → jump. The claimed detection's true owner
usually wins it back within frames (it is the geometrically consistent match), the rejoined
slot goes unmatched → lost → rejoins or is respawned back near its original, internally
consistent location → return. The velocity z-score detector that should flag both legs is
disabled by default (**F15**); the hard 50 px/f break fires only if the jump is written as
consecutive-frame motion, and the cross-gap splitter (`processing.py:609+`) uses
`dist/frame_gap`, which a long gap dilutes.
Path B (offline, always on): the fragment solver assigns a far-away fragment to identity L
between two of L's segments — allowed because the bridge-gap cap fixes tolerable
displacement at ≤1500 px regardless of true gap (**F15**) and evidence can outvote a 0.14
spatial score (**F11**). Any per-identity time series (`sort_trajectories_by_identity`
orders fragments by label) then shows teleport-out/teleport-back at the fragment
boundaries.
Path C (merge): forward/backward disagreement runs shorter than
`IDENTITY_DISAGREE_MIN_RUN=5` are averaged through (`processing.py:1595-1639`), which can
splice a brief wrong-association excursion into an otherwise consistent merged track.

---

## 6. Incremental fixes (within the current architecture)

Ordered by leverage; each tied to findings.

1. **Fix the slot-lock sign or delete the mechanism (F3).** If kept, implement the
   documented semantics: `p ← (1−s)·p + s·δ_lock` or an explicit log-odds bonus
   `+log(1/(1−s))`. One-line change, currently the mechanism is inverted. *Risk: low.*
2. **Wire the existing robustness cap/floor and set defaults (F1, F2).** Emit
   `PER_FRAME_EVIDENCE_CAP` (≈ 0.5–1.0 nats) and `PROB_FLOOR` (≈ 1e-3) from
   `RobustnessConfig`, and pass them at all three `fuse_log_evidence` call sites. This is
   the intended Phase-3 design, already implemented and tested in `substrate.py` — it is
   only unplugged. Expected effect: commit time stretches from ~4 frames to a tunable
   horizon; single-streak flips of a committed identity become much harder; the unknown
   state stays reachable. *Risk: low-medium (slower commits; re-tune commit_min_hits).*
3. **Better: temper evidence by an effective-sample-size factor (F1).** Multiply each
   frame's log-evidence by `1/τ` where τ ≈ the autocorrelation window of crops (measurable
   from the sidecar: lag-1 correlation of per-frame posteriors), or fuse only every τ-th
   frame. Converts "5 correlated frames = 5 confirmations" into "≈1 confirmation."
   *Risk: medium (needs τ estimation; a fixed τ=5–10 is already a large improvement).*
4. **Give `unknown` a real emission (F2).** Replace the 1e-6 floor with
   p(evidence | unknown) = uniform over the model's classes (1/K), or a learned
   open-set/outlier score. With calibrated evidence this makes p(unknown) grow under
   ambiguity instead of vanishing. *Risk: medium — changes every posterior; re-tune
   display/commit thresholds.*
5. **Make commitment revision require sustained counter-evidence (F4, F8).** Replace the
   vacuous probability-margin gate with: revise only after M consecutive frames where the
   posterior argmax ≠ committed label (M ≈ swap_min_frames), mirroring the swap counter so
   isolated flips get the same debounce that paired swaps already have. *Risk: low.*
6. **Harden the rejoin trigger (F7).** Require (a) calibrated sidecar evidence, not the
   top-1 reconstruction; (b) a confirmation window: the same detection/tracklet must
   support the label for M ≥ 3 consecutive frames before the claim executes; (c) a
   threshold ≥ display threshold; (d) a sublinear or capped motion budget
   (e.g. `min(budget, arena_scale)` or √t diffusion growth, which is the physically
   correct law for an unobserved random walk). *Risk: medium (slower re-ID after
   occlusion — acceptable trade).*
7. **Remove the evidence double-use, or make the coupling one-way (F6).** Either the
   association cost consumes only the *prior* belief (fine) and fusion uses evidence not
   used in the cost (e.g. fuse with weight reduced by the assignment's identity-cost
   contribution), or more simply: keep the cost term but build it from the same calibrated
   sidecar vectors as fusion so at least the two paths share one likelihood. *Risk: low
   for the consistency half; the full de-double-counting needs care.*
8. **Point the solver's online prior at `IdentityRealtime*` (F12)** — or delete the
   0.25-weight term. Also guard re-runs: never read `IdentityFinalLabel` produced by a
   previous offline pass as "online" evidence. *Risk: low.*
9. **Feed the solver raw (unsmoothed) per-frame evidence for support/stability (F13).**
   Keep smoothing for the per-row `IdentityFinalSmoothedLabel` display columns; compute
   `CNNLogEvidence` (as a *sum*, not mean — restoring evidence count, optionally tempered
   per fix 3) and `Stability` from the raw sequences. *Risk: medium (changes solver
   behavior; gate with the equivalence harness on `ant_cnn_identity`/`emi_obb_identity`).*
10. **Split fragments at committed-label switches (F14).** Even without PELT: the realtime
    decoder already logs commitment revisions; splitting trajectories at
    `IdentityRealtimeLabel` change points (with the min_run debounce) gives the solver
    exactly the boundaries mode-1 errors create. Alternatively enable PELT on raw
    evidence. *Risk: medium.*
11. **Scale the offline bridge veto with the true gap (F15):** replace the 30-frame clamp
    with a diffusion-scaled allowance `dist ≤ v_max·min(gap, cap) + c·√gap·body`, and
    enable `MAX_VELOCITY_ZSCORE` (two-sided |z|) by default. *Risk: low-medium.*
12. **Emit the hardcoded decoder knobs** (`IDENTITY_SLOT_LOCK_*`, `IDENTITY_COMMIT_MIN_HITS`,
    respawn-prior trio) from the schema so they are tunable and auditable. *Risk: trivial.*
13. **Respawn prior: key on space, not slot (F9).** Carry a lost identity's prior into a
    respawned slot only if the respawn position is within the lost slot's motion budget of
    its last position; otherwise start uninformative. Add a TTL to committed-lost beliefs
    (e.g. respawn_prior_max_gap) after which the slot un-commits, releasing the label.
    *Risk: low.*

---

## 7. Ground-up redesign — the theoretically sound version

**Generative model.** K known animals (+ an open-set "clutter/unknown" class). Latent
state per animal: kinematic state x_t (the existing Kalman model — unchanged) and, per
detection, an identity indicator. One joint model:

p(detections | assignment, identities) = ∏ p(position | x_t) · p(appearance | identity)

with the **hard constraint** that the map detections→identities is injective per frame
(and, over time, that each identity occupies one path).

**Layer 1 — per-frame association (realtime).** Keep the Hungarian assigner, but make the
cost the negative log of one joint likelihood:
`cost(i,j) = −log N(z_j; Hx_i, S_i) − λ·log p(z_j^app | id belief_i)` with the appearance
term *tempered* (λ = 1/τ from the evidence-autocorrelation estimate) and computed from
calibrated evidence with a real unknown emission. This subsumes today's capped hint, the
rejoin path (a lost track is just a row whose motion likelihood is a gap-widened
diffusion N(·; x, S + tQ) — long-range re-ID falls out naturally, with √t growth, no
special-case budget), and the density gate. One likelihood, one solver, no side channels.

**Layer 2 — identity belief (realtime).** Per-track HMM exactly as now (sticky transition
is fine), but: evidence tempered (ESS), unknown reachable, **no internal commit state
machine**. "Committed" becomes a *reporting* concept: display argmax with its calibrated
posterior; downstream consumers threshold as they wish. Uniqueness is enforced where it is
physical — in the Layer-1 assignment and the Layer-3 global solve — not by blocking
commits on stale slots. Swaps need no special detector: with a joint cost, a sustained
mutual mismatch simply makes the swapped assignment cheaper and Layer 1 executes it as an
ordinary re-association (the current swap machinery, lock machinery, respawn-prior
machinery, and commit-blocking all disappear).

**Layer 3 — offline global MAP (replaces the fragment solver).** Standard tracklet-graph
formulation: nodes = tracklets split at genuine ambiguity events (occlusion contacts,
detection gaps, evidence changepoints on *raw* evidence); identities = K source-sink paths
through the graph; edge weights = −log[motion likelihood of the bridge (gap-dependent
diffusion) · appearance likelihood of the downstream tracklet under the identity (tempered
sum of calibrated log-evidence — this restores evidence *count*, making the ad-hoc length
factor unnecessary)]. Solve as min-cost flow / k-disjoint-shortest-paths (or a small ILP —
tracklet counts here are tiny). This is exact MAP under the stated model: mutual exclusion
is a flow constraint (hard, global, no pairwise post-hoc stripping), "length weight" is
likelihood mass, "spatial score" is the motion likelihood, "doubt ordering / margin /
unknown rescue / swap moves" are all replaced by optimality. `IdentityFinalConfidence`
becomes a real quantity: the min-marginal (cost gap to the best solution with that
tracklet-identity edge removed).

**Why this removes the root fallacies:** F1/F13 → tempering + sum-with-ESS makes fused
confidence match information content; F2 → explicit unknown emission and an unassigned
option priced by the model; F3/F4/F5/F8/F9 → the commit/lock/swap/respawn state machine is
deleted, so its bugs and tautologies go with it; F6 → evidence enters the joint likelihood
exactly once; F7 → re-ID is the same likelihood as tracking, with diffusion-correct
distance scaling; F11/F14/F15/F16/F17 → one objective, exact hard uniqueness, fragment
boundaries from the model, one scoring vocabulary end-to-end.

**Mapping onto existing modules:** `evidence.py`/`evidence_builder.py`/`calibration.py`/
`cache.py` survive unchanged (the sidecar is exactly the right substrate — add an ESS/τ
estimator over it); `catalog.py` gains the unknown emission; `online.py` shrinks to the
tempered HMM filter (≈150 lines); `substrate.py`'s solver remains for Layer 1;
`hungarian.py` gets the joint cost (replacing `_apply_bayesian_identity_cost` +
`_assign_respawn`'s identity branch); `offline.py` is replaced by the tracklet-graph
solver (smoothing.py's forward-backward survives for per-row smoothed decode);
`processing.py`'s identity conflict resolver is deleted (subsumed by flow constraints).
Calibration becomes load-bearing: `calibration_required` should default True, with
`fit_temperature`/ECE (already implemented in `calibration.py`) run at model-publish time.

---

## 8. Open questions

1. **Was the observed mode-3 teleport produced with `ENABLE_IDENTITY_ONLINE_DECODER=True`?**
   The within-trajectory jump path (F7) requires it. Check the run's saved config /
   `IdentityRealtime*` columns. If the flag was off, mode 3 came from paths B/C (offline
   assignment or merge) and the priority of fix 6 vs fixes 9-11 changes.
2. **Are calibration temperatures actually fitted in practice?** `CalibrationModel`
   defaults to identity (T=1); `calibration_required` defaults False. If production models
   run uncalibrated, F1's inflation is compounded by softmax overconfidence — worth
   measuring ECE on a held-out identity set for the shipped classifiers.
3. **What is the empirical crop-evidence autocorrelation τ?** Determines the right
   tempering constant (fix 3). Measurable directly from an existing evidence sidecar
   (lag-k agreement of per-frame argmax / correlation of log-posteriors) on
   `ant_cnn_identity` / `emi_obb_identity` — no new tracking runs needed.
4. **Attribution runs (proposed, not executed):** on `emi_obb_identity` and
   `ant_cnn_identity` fixtures, (a) count `IdentityRealtimeLabel` flip events per 1k
   frames with defaults vs with `fuse_log_evidence` cap = 1.0 nat (worktree toggle) —
   directly tests F1's share of mode 1; (b) with the decoder flag ON, count
   trajectory-internal displacement spikes > 5×body with and without a 3-frame rejoin
   confirmation window — tests F7's share of mode 3; (c) enable PELT on raw vs smoothed
   evidence and count recovered mid-track label switches — tests F13/F14. Each is a single
   named-hypothesis run under `tools/equivalence/` fixtures per the repo's harness
   conventions (conda env active; verify CSV row counts > 1).
5. **Is the swap detector ever firing in practice?** Its consecutive-frame requirement
   (F8) plus the lock erosion (F3) make its firing conditions narrow; the
   `Identity swap fired` log line frequency in real runs would show whether it is a live
   mechanism or dead weight.
6. **`IDENTITY_WEIGHT` semantics:** the emitted `ASSOCIATION_IDENTITY_HINT_SCALE` default
   is 1.0 (schema) while the code fallback is 0.3 (`hungarian.py:241`); if users enable
   the decoder flag without touching the weight they get a 3.3× stronger coupling than the
   code's own default suggests. Which value was intended?

---

## Appendix A — Empirical probe (run 2026-08-13)

Script: `scratchpad/identity_decoder_probe.py` (session scratchpad), run as
`PYTHONPATH=$PWD/src conda run -n hydra-mps python identity_decoder_probe.py` against the
real `OnlineIdentityDecoder` with all-default params and a 4-identity catalog, evidence
built via the code's own `catalog.cnn_log_prior` (i.e., exactly the worker's top-1
construction).

| Hypothesis | Result |
|---|---|
| H-A: commit speed, cold start, evidence 0.6-on-A per frame | conf 0.868 after 2 frames, 0.988 after 4; committed at frame 5 (hit gate, not confidence gate, was binding) |
| H-D: unknown annihilation | p(unknown) = 2.3e-7 after 1 frame, 8.5e-9 steady state |
| H-B: slot-lock bias sign | p(A) 0.999343 → 0.999270 after `_apply_slot_lock_bias` (strength 0.9): the lock **decreases** the locked label's probability |
| H-C: committed + locked slot, isolated, wrong evidence B@0.6 | committed label flipped A→B after **5 frames**; lock destroyed; no gate resisted |
| H-E: recovery | flipped back B→A after 5 frames of A@0.6 — symmetric flapping |

These are unit-level dynamics of the decoder, independent of detector/video; they confirm
F1–F4 quantitatively.
