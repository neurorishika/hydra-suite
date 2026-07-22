# Identity Overhaul — Consolidated Design

Date: 2026-07-22
Status: Proposed
Supersedes: `trackerkit-identity-overhaul-spec.md` (2026-04-26)

Related docs:

- `docs/superpowers/specs/trackerkit-identity-overhaul-spec.md` (superseded; ideas folded in)
- `docs/superpowers/specs/2026-07-19-bgsub-inference-unification-design.md` (the caching pattern this reuses)
- `docs/developer-guide/runtime-integration.md`
- `docs/developer-guide/extending-identity.md`

## Objective

Consolidate the identity subsystem onto a single, honest architecture:

- **Identity evidence is an inference-time artifact** produced by `InferenceRunner`, alongside the raw CNN/AprilTag caches — not synthesized by the tracker.
- **Realtime identity-aware tracking** and **post-hoc identity assignment** are two *independent* consumers of that one evidence layer. Either, both, or neither can run. Honesty is structural.
- **Calibrated (honest) uncertainty is mandatory** before any Bayesian decoder runs.
- The two decoders (causal online filter, full-context offline smoother) stay distinct algorithms but **share one substrate** (catalog, factor→catalog mapping, uniqueness constraint, log-space fusion) so they cannot disagree about what the evidence means.
- Output columns are **provenance-explicit and never clobbered**.
- All identity state lives in one **typed `IdentityConfig`** driving both the UI and core.

This overhaul is a **consolidation + gap-completion**, not a rewrite: most of the machinery (`catalog.py`, `evidence.py`, `cache.py`, `online.py`, `fragment_solver.py`, `calibration.py`, calibrated-posterior CNN path) already exists. The work is to relocate, unify, and de-dupe it, and to complete the missing pieces.

## Scope

In scope:

1. Move calibrated evidence generation into the inference pass.
2. Persist the identity catalog and calibration as config artifacts known before inference.
3. A real calibration-fitting workflow (temperature scaling on the CNN held-out validation split), with a mandatory-calibration gate and runtime robustness knobs.
4. Extract a shared substrate consumed by both decoders; make the offline path read the evidence cache instead of reconstructing from CSV.
5. Add true offline forward-backward smoothing feeding the changepoint/fragment/global-assignment solver.
6. Provenance-explicit output columns.
7. A typed `IdentityConfig` and a reorganized, honest TrackerKit UI.
8. **Clean-break retirement** of legacy paths (see Retirement).

Out of scope:

- Redesigning the Kalman tracker, OBB detection, or geometric association.
- Redesigning AprilTag detection itself.
- Open-world re-identification beyond the configured catalog.
- Pose-model or ViTPose changes (pose identity features remain as-is).

## Current State (verified at HEAD)

### What already exists and is sound

- `core/identity/catalog.py` — `IdentityCatalog` (index 0 = unknown), prior builders.
- `core/identity/evidence.py` — `IdentityEvidence` with a full calibrated catalog `log_probs`; `from_apriltag`, `from_cnn`, `missing`.
- `core/identity/cache.py` — `IdentityEvidenceCache` NPZ sidecar reader/writer.
- `core/identity/calibration.py` — `CalibrationModel` (temperature scaling, content-hash signature).
- `core/identity/online.py` — `OnlineIdentityDecoder`: log-space predict→fuse→Hungarian→commit, swap detection, slot-lock, respawn priors.
- `core/identity/fragment_solver.py` — offline changepoint + iterative global assignment.
- CNN calibrated-posterior path: `predict_batch_posteriors`, V3 cache with full per-factor probability vectors.
- Inference-time raw caches: `CNNCacheHandle` (raw posteriors) and `AprilTagCacheHandle` are written by `InferenceRunner` — before tracking in non-realtime (`run_batch_pass`), inline in realtime (`run_realtime`).

### Verified flaws this overhaul fixes

1. **Evidence is a tracking-time artifact.** `IdentityEvidenceEmitter` is fed per-frame inside the tracking loop (`worker.py:2510-2522`) and flushed at loop end (`worker.py:4047-4052`), in *both* realtime and non-realtime modes. This violates "all inference caches written before tracking."
2. **The honesty bug.** With `ENABLE_IDENTITY_IN_TRACKING` off, `OnlineIdentityDecoder` is never built (`worker.py:1820-1821`), so `IdentityAssignedLabel`/`IdentityCommitted` come out empty, and the offline fragment solver — which reconstructs evidence from those CSV columns — is starved. The tracking-panel tooltip claiming "offline post-processing still works" is therefore literally false.
3. **Offline reads CSV, not evidence.** `fragment_solver.py:263-270` reconstructs probabilities from exported columns via a heuristic weighted-support blend that is *not* a posterior — the bird's-eye pass is fed worse data than the online path, ignoring the calibrated evidence cache.
4. **Duplicated, divergable substrate.** Catalog assembly (`worker.py:1817-1915`) and multi-factor→catalog mapping are implemented twice (worker inline vs `fragment_solver` CSV reconstruction) and can disagree. Uniqueness is implemented three times (online Hungarian, offline collision-veto, post conflict-resolution).
5. **Factor-encoding landmine.** Composite identities are keyed on `"_".join(...)` / `label.split("_")`; any factor class name containing `_` silently breaks decomposition, dropping that identity's evidence to a `1e-9` floor (`worker.py:3164`).
6. **Information loss.** The online path reconstructs per-factor distributions from **top-1 confidence only** (`worker.py:3142-3155`), discarding the true per-head softmax that the V3 cache already stores.
7. **Destructive column clobber.** `IdentityAssignedLabel` is written by the online decoder, then overwritten in place by the fragment solver (`fragment_solver.py:1324`); `Confidence`/`Margin`/`Entropy` are left describing the stale online posterior, now inconsistent with the overwritten label.
8. **Orphaned/dead artifacts.** The legacy V3 `CNNIdentityCache` is written only by the export helper and read-only in the worker; `detected_cnn_cache_paths` (`worker.py:153`) is initialized to `{}` and never populated.
9. **No typed config.** Identity is ~20 flat UPPERCASE keys assembled in an orchestrator, split across three UI panels, absent from `TrackerConfig` — contrary to the project's typed-schema design principle.
10. **Overconfident models.** Raw softmax confidences are not honest uncertainty estimates; feeding them into a log-space Bayesian accumulator compounds into false certainty. Calibration is currently never actually *fitted*.

## Design Principles

- **Honest uncertainty is a first-class requirement.** Only calibrated posteriors enter the Bayesian decoders. The pipeline refuses to run the decoders on uncalibrated models unless the user explicitly overrides.
- **Evidence is an inference-time artifact.** Whatever the inference pass caches, the tracker only reads (non-realtime); in realtime the same contract is emitted inline. Nothing that can be precomputed is synthesized at tracking time.
- **One substrate, two decoders.** Filtering (causal) and smoothing (full-context) are different problems and stay different algorithms; they share catalog, factor-mapping, uniqueness, and fusion primitives.
- **Independent consumers.** Realtime influence and post-hoc assignment are orthogonal toggles over the same evidence. Neither depends on the other.
- **Provenance over mutation.** Raw / realtime / final identity live in separate columns; no stage overwrites another stage's decision.
- **One typed source of truth.** `IdentityConfig` drives both UI and core; no scattered flat keys, no widget-attribute state.
- **Clean break.** Legacy decision paths are deleted, not shimmed (the equivalence + identity gates are the safety net).

## Target Architecture

### Layer 0 — Config (persisted, typed, known before inference)

New typed schema `trackerkit/config/identity_schema.py` (dataclass `IdentityConfig`), referenced from `TrackerConfig`:

```python
@dataclass
class IdentityModelConfig:
    kind: str                 # "cnn" | "apriltag" | "color_tag"
    name: str
    path: str | None
    unique_identifier: bool   # participates in the catalog
    factors: tuple[str, ...]  # structured factor names (NOT "_"-joined)
    calibration: CalibrationRef | None   # fitted temperature + signature

@dataclass
class RealtimeIdentityConfig:
    enabled: bool                 # influence association at all
    bayesian_cost_enabled: bool
    association_weight: float
    rejoin_threshold: float
    commit_threshold: float
    display_threshold: float
    transition_epsilon: float
    unknown_prior: float
    swap_enabled: bool
    slot_lock: SlotLockConfig

@dataclass
class PostHocIdentityConfig:
    enabled: bool                 # independent of realtime
    smoothing_enabled: bool       # forward-backward
    changepoint_enabled: bool     # PELT
    fragment_min_frames: int
    ambiguity_margin: float
    gates_trajectory_structure: bool
    disagree_min_run: int

@dataclass
class RobustnessConfig:
    per_frame_evidence_cap: float # bound single-frame log shift
    prob_floor: float             # no frame ever fully certain
    source_weights: dict[str, float]  # {"cnn": .., "apriltag": ..}

@dataclass
class IdentityConfig:
    enabled: bool                 # master: identity classification on
    catalog: IdentityCatalogSpec  # persisted domain (see Layer 1)
    models: list[IdentityModelConfig]
    calibration_required: bool
    realtime: RealtimeIdentityConfig
    posthoc: PostHocIdentityConfig
    robustness: RobustnessConfig
```

`get_parameters_dict()` derives the legacy flat keys from `IdentityConfig` during migration, then those call sites are converted to read `IdentityConfig` directly.

### Layer 1 — Identity catalog & calibration as artifacts

**Catalog.** The identity domain is resolved **once**, up front (not per-run inside the worker, not again inside the orchestrator), from the `unique_identifier` models + tag labels, and persisted as `IdentityCatalogSpec` in the config. `IdentityCatalog.from_spec(...)` rebuilds the frozen runtime object identically in inference, tracking, and post-hoc.

Fixes: single ownership of the domain; **structured factor keys** replace `"_"`-joins — a composite label is a tuple of `(factor, class)` pairs, so class names may contain any character; per-model namespacing prevents two classifiers' identical class strings from collapsing.

**Calibration.** A new workflow fits a temperature per CNN model (optionally per factor) on the model's **held-out validation split**, minimizing NLL (report ECE before/after). The fitted temperature + content-hash signature are stored with the model and referenced from `IdentityModelConfig.calibration`.

- New: `core/identity/calibration_fit.py` — `fit_temperature(logits, labels) -> CalibrationModel`; CLI/GUI entry to fit and store.
- Gate: if `calibration_required` and any `unique_identifier` model lacks a matching-signature calibration, the Bayesian decoders refuse to run (loud error naming the fit step). A user override downgrades to a warning.

### Layer 2 — Evidence layer (inference-time)

`InferenceRunner` gains an **`IdentityEvidenceStage`** that, given the raw CNN/AprilTag caches + catalog + calibration, produces `IdentityEvidence` (full calibrated catalog `log_probs`, including the unknown slot) and writes the `IdentityEvidenceCache` sidecar.

- **Non-realtime:** written in `run_batch_pass` alongside the raw caches (`core/inference/cache/writer.py`), **before tracking**. Tracking and post-hoc both only *read* it.
- **Realtime:** emitted inline in `run_realtime`, identical contract.

The stage owns the **single** factor→catalog mapping: it consumes the true per-factor softmax from the raw CNN cache (no top-1 reconstruction), forms the joint as a product over factors in log-space, applies calibration and the robustness floor/cap/source-weight, and maps into the catalog by structured factor keys.

Deleted as a consequence: tracking-time `IdentityEvidenceEmitter` construction/feeding/flush in the worker; the worker's inline `from_apriltag`/`from_cnn` construction; the top-1 pseudo-distribution reconstruction; the orphaned V3 `CNNIdentityCache`; `detected_cnn_cache_paths`.

### Layer 3 — Shared substrate

`core/identity/substrate.py` (new) centralizes what both decoders currently reimplement:

- `map_cnn_to_catalog(...)` / `map_tag_to_catalog(...)` — the one factor→catalog mapping (also used by Layer 2).
- `fuse_log_evidence(...)` — log-space Bayesian fusion with robustness cap/floor.
- `solve_unique_assignment(...)` — the one partial-injective uniqueness solver (Hungarian with dummy-unassigned columns), used by online per-frame, by offline global assignment, and by post conflict-resolution.

Both `online.py` and `fragment_solver.py` are refactored to call these; the triplicated logic is removed.

### Layer 4a — Realtime consumer (causal filter)

`OnlineIdentityDecoder` keeps its role: sticky-Markov predict → fuse (via substrate) → unique assignment (via substrate) → commitment/slot-lock. It now **reads evidence** from the cache/stream rather than building it. Its influence on association stays an explicit, independent toggle (`RealtimeIdentityConfig.enabled` / `bayesian_cost_enabled` / `association_weight`). With realtime off, the decoder simply does not run — and post-hoc is unaffected because it reads the same evidence cache.

### Layer 4b — Post-hoc consumer (full-context smoother)

The offline path is completed and re-pointed at the evidence cache:

1. **Forward-backward smoothing** (new `smooth_trajectory_posteriors`) per final trajectory over the calibrated evidence — a confident late burst corrects ambiguous early frames.
2. **Changepoint detection** (PELT) on the *smoothed* posteriors → split mixed-identity trajectories at evidence regime changes.
3. **Fragment global assignment** via the substrate uniqueness solver (replacing the standalone greedy collision-veto with the shared partial-injective solver; keep the iterative residual pass for ambiguous fragments).

Post-hoc runs from the evidence cache + final trajectories **with no dependency on the realtime decoder**. This is the core honesty fix.

### Layer 5 — Output (provenance-explicit)

Three column families, none overwriting another:

- `Identity_Evidence_TopLabel`, `Identity_Evidence_Conf`, `Identity_Evidence_Sources` — per-detection raw calibrated evidence summary.
- `Identity_Realtime_Label`, `_ID`, `_Confidence`, `_Margin`, `_Entropy`, `_Committed`, `_SlotLock` — the online filter decision (present iff realtime ran).
- `Identity_Final_Label`, `_ID`, `_Confidence`, `_Source` (`realtime` | `offline` | `tag`), `_FragmentScore` — the resolved identity. If post-hoc ran, it is the offline result; otherwise it mirrors realtime (with `_Source=realtime`); if neither ran, empty.

`UniqueIdentityKey` is retained as a derived compatibility/presentation column. The raw NPZ evidence cache remains the authoritative posterior store; CSV columns are summaries.

## UI Reorganization (honest, layered)

The three scattered panels are re-scoped to mirror the architecture. Each section's enable state is honest — a control is shown as available only when its inputs actually exist.

1. **Identity Models** (identity panel): master "Enable Identity Classification"; configure CNN classifiers (with `unique_identifier`), AprilTags, color tags, head-tail, pose. A status line states plainly: *"Identity evidence is computed during inference and cached — available to both realtime and post-hoc."* A calibration status/affordance per `unique_identifier` model (fitted / not fitted, fit button).
2. **Realtime Identity** (tracking panel): "Use identity to influence tracking" + the Bayesian cost term and its weights/thresholds. Scoped *only* to realtime association influence. Tooltip corrected — it no longer claims anything about post-hoc.
3. **Post-hoc Identity** (post-process panel): "Assign identities from final trajectories" as a **first-class independent toggle**, enabled whenever identity classification is on (never gated on the realtime flag). Sub-controls: forward-backward smoothing, PELT changepoint splitting, fragment-solver knobs, `gates_trajectory_structure`.

All three read/write the single `IdentityConfig`.

## Retirement (clean break)

Deleted in this overhaul (no shims):

- `TrackCNNHistory` majority-vote as a decision path (and its use in `frame_result_bridge.py`).
- The orphaned V3 `CNNIdentityCache` write/read path (`export.py:767`, `worker.py:1573`).
- Dead `detected_cnn_cache_paths` + `augment_trajectories_with_detected_cnn_cache`.
- Tracking-time `IdentityEvidenceEmitter` wiring in the worker.
- `"_".join` / `split("_")` factor encoding (replaced by structured factor keys).
- Triplicated uniqueness logic (folded into the substrate solver).
- The heuristic CSV-reconstruction evidence path in `fragment_solver.py`.

Safety net: the equivalence harness (positions byte-identical when identity influence is off) plus new identity-specific tests/metrics, on both MPS and CUDA.

## Rollout Plan

Each phase is independently shippable and gated.

- **Phase 0 — Typed config + persisted catalog.** Introduce `IdentityConfig`, `IdentityCatalogSpec`, structured factor keys; migrate `get_parameters_dict()` to derive from it. No behavior change. Catalog resolved once, persisted.
- **Phase 1 — Calibration workflow.** `calibration_fit.py`, validation-split fitting, storage, signature, mandatory-calibration gate, robustness knobs. Report ECE.
- **Phase 2 — Evidence as inference artifact.** `IdentityEvidenceStage` in `InferenceRunner` (batch + realtime); single factor→catalog mapping using true per-factor softmax; remove tracking-time emitter. Evidence cache written before tracking (non-realtime).
- **Phase 3 — Shared substrate + realtime read-through.** Extract `substrate.py`; refactor `online.py` to read evidence and use substrate fusion/uniqueness.
- **Phase 4 — Post-hoc self-sufficiency + smoothing.** Offline reads the evidence cache; add forward-backward smoothing; fragment solver uses substrate uniqueness. Post-hoc runs with realtime off. **This closes the honesty bug end-to-end.**
- **Phase 5 — Provenance columns + UI reorg.** Split output columns; reorganize the three panels; correct tooltips.
- **Phase 6 — Clean-break retirement.** Delete legacy paths once the identity + equivalence gates pass on MPS and CUDA.

## Testing & Verification

Unit:

- Catalog round-trip from spec; structured-factor labels with `_` in class names.
- Calibration fit reduces ECE on a synthetic overconfident set; signature match/mismatch gating.
- Factor→catalog joint = product of true per-factor softmax (not top-1).
- Substrate: log-space fusion with cap/floor; partial-injective uniqueness solver.
- Evidence cache round-trip; identical bytes whether emitted batch vs realtime.
- Forward-backward smoothing corrects a late-confident/early-ambiguous trajectory.

Integration:

- Realtime OFF + post-hoc ON produces non-empty `Identity_Final_*` (the regression that proves the honesty fix).
- AprilTag-only, CNN-only, and conflicting-evidence runs.
- Occlusion/reappearance with slot reservation.
- Offline global solver resolves overlapping same-label fragments.

Gates:

- **Equivalence harness:** tracking positions byte-identical vs baseline when realtime identity influence is off (identity is additive columns). Both MPS + CUDA.
- **Identity metrics:** IDF1, identity switches, duplicate-ID-per-frame violations, unresolved fraction, occlusion recovery delay, online-vs-offline agreement.

## Risks & Open Questions

- Moving evidence into inference requires the catalog + calibration to be resolved before the inference pass; a run configured with identity but no fitted calibration must fail clearly (mandatory gate) rather than silently use raw softmax.
- Validation-split calibration reflects training distribution, not necessarily deployment conditions; the runtime robustness knobs (cap/floor/source-weight) are the mitigation. Revisit tag-as-free-label calibration later if drift is observed.
- Clean-break retirement means no fallback if a subtle regression slips the gates; the equivalence gate must be run before/after Phase 6 on both platforms.
- Forward-backward smoothing cost on long trajectories with a large catalog must stay bounded.

## Acceptance Criteria

- Identity evidence is written during the inference pass (before tracking in non-realtime), and both realtime and post-hoc consume the same cache.
- Post-hoc identity assignment runs correctly with realtime identity influence off (non-empty `Identity_Final_*`).
- Only calibrated posteriors enter the decoders; uncalibrated `unique_identifier` models are gated with a clear error.
- One catalog and one factor→catalog mapping are used everywhere; class names may contain any character.
- Realtime and post-hoc are independent toggles over one `IdentityConfig`; the UI states this honestly and no tooltip is misleading.
- Output columns are provenance-explicit; no stage overwrites another's decision.
- Legacy majority-vote, V3 cache, dead detected-CNN path, and string-join encoding are deleted.
- Equivalence gate (positions byte-identical, identity off) passes on MPS and CUDA.
