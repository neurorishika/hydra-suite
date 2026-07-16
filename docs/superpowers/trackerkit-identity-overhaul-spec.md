# TrackerKit Identity Overhaul Spec

Date: 2026-04-26
Status: Proposed

Related docs:

- `architecture.md`
- `runtime-integration.md`
- `trackerkit-streaming-individual-analysis-plan.md`
- `tracking-algorithm-deep-dive.md`

## Objective

Overhaul TrackerKit identity handling so that both realtime tracking and offline post-processing preserve uncertainty, accumulate evidence over time, and enforce the hard uniqueness constraint that one visible animal can own a given identity at a time.

This spec is intentionally narrower than the general tracking architecture docs. It focuses only on identity inference, identity persistence, identity-aware slot reuse, and identity-aware post-processing.

This identity plan assumes the streaming-first execution substrate from `trackerkit-streaming-individual-analysis-plan.md`. The streaming plan owns the filtered-detection payload, canonical crop transport, runtime execution, and live versus replay parity. This plan owns the probabilistic identity model that must run on top of that substrate.

## Problem Statement

The operating conditions are simpler than general multi-object tracking, but stricter than the current implementation assumes:

- Geometry and motion association already produce trajectories or track slots.
- For each visible trajectory or detection, TrackerKit has identity evidence with confidence.
- Identities come from a known set of `N` unique types.
- At any frame, the visible assignment from trajectory to identity must be a partial injection.
- Not all animals are always visible.
- The detector and classifier are noisy, so the system must clean spurious errors without discarding useful uncertainty.

The current system loses too much information by collapsing identity evidence into hard labels too early. The correct abstraction is probabilistic identity accumulation under a global uniqueness constraint.

## Current Codebase Findings

The current implementation already has useful hooks, but the uncertainty-preserving model does not exist yet.

### Realtime path

- `src/hydra_suite/core/tracking/worker.py` reads per-frame tag and CNN outputs during tracking.
- `src/hydra_suite/core/tracking/tag_features.py` keeps a rolling majority-vote tag history per track slot.
- `src/hydra_suite/core/identity/classification/cnn.py` exposes `TrackCNNHistory`, which stores tuples of hard classes and confidences, then reduces them to majority identities.
- `src/hydra_suite/core/tracking/cnn_features.py` converts cached predictions plus track history into `detection_classes` and `track_identities` for the assigner.
- `src/hydra_suite/core/assigners/hungarian.py` only sees hard identity overlays as additive match and mismatch bonuses in the geometric cost matrix.

### Post-processing path

- `src/hydra_suite/trackerkit/gui/orchestrators/tracking.py` builds the rich export dataframe after tracking by merging pose, CNN, AprilTag, and interpolation outputs.
- `src/hydra_suite/core/post/identity_postprocess.py` performs a heuristic split and re-chain pass using exported `TagID`, `InterpTagID`, `DetectedTagID`, and flattened CNN class/conf columns.
- `src/hydra_suite/core/post/processing.py` uses `UniqueIdentityKey` only as a relink veto or compatibility check after trajectories already exist.

### Evidence persistence

- `src/hydra_suite/core/identity/classification/cnn.py` persists only top class names and top confidences per factor in `CNNIdentityCache`.
- `src/hydra_suite/core/identity/properties/export.py` exports only flattened `CNN_<label>_Class` and `CNN_<label>_Conf` columns.
- The current pipeline does not persist calibrated posterior vectors over the full identity catalog, so offline post-processing cannot recover information that was already discarded.

### Slot lifecycle

- `src/hydra_suite/core/tracking/worker.py` clears tag and CNN histories when a lost slot respawns.
- Lost slots are freely reused with a new `TrajectoryID` and no identity reservation semantics.
- The current code therefore has no notion of long-lived slot commitment or identity-aware slot reuse.

## Design Principles

- Preserve the full posterior over identities as long as possible.
- Separate geometric association from identity decoding.
- Enforce uniqueness globally across visible trajectories, not greedily per trajectory.
- Use the same identity model online and offline.
- Keep raw evidence and cleaned outputs side by side for auditability.
- Treat low-confidence evidence as weak evidence, not as missing evidence.
- Make identity commitment and slot reuse policy explicit, testable, and configurable.

## Dependency On The Streaming Plan

The identity overhaul should not be executed as an isolated refactor.

It depends on the streaming plan for:

- a stable filtered-detection payload after head-tail and canonical reorientation
- one live and replay-compatible place to emit identity evidence caches
- runtime-consistent CNN execution paths that can expose posterior information without duplicating inference code

Because of that dependency, the streaming plan should establish the transport layer first. The identity overhaul should begin only after the shared payload contract and CNN posterior-emission hook exist.

## Functional Requirements

- Support a known catalog of unique identities plus an explicit `unknown` or `unobserved` state.
- Fuse multiple evidence sources, including AprilTag and CNN identity models.
- Preserve multihead uncertainty instead of collapsing it into a tuple string.
- Produce a stable online identity assignment with bounded latency.
- Produce a stronger offline identity result using future evidence and global fragment constraints.
- Keep compatibility with the current CSV and rich-export workflows during migration.
- Avoid violating the rule that two visible trajectories cannot own the same identity at the same time.
- Allow animals to be absent without forcing fabricated identities.

## Non-Goals

- Replacing the Kalman tracker or geometric detection-to-track association in this project.
- Redesigning AprilTag detection itself.
- Solving generic open-world re-identification beyond the configured identity catalog.
- Removing existing CSV exports or GUI affordances during the first rollout phase.

## Target Architecture

The system should use one identity model and one evidence representation in both live and offline modes.

### Identity Catalog

Introduce a canonical identity catalog that all sources map into.

Proposed module:

- `src/hydra_suite/core/identity/catalog.py`

Proposed responsibilities:

- Define the ordered identity domain for one run.
- Represent the `unknown` or `unobserved` state explicitly.
- Map AprilTag IDs, classifier labels, and future appearance embeddings into the same index space.
- Validate that configured unique-ID classifiers are compatible with the catalog.

Proposed data model:

```python
@dataclass(frozen=True)
class IdentityCatalog:
    labels: tuple[str, ...]
    unknown_index: int
    source_mappings: dict[str, dict[str, int]]

    @property
    def size(self) -> int: ...

    def index_of(self, label: str) -> int: ...
    def label_of(self, index: int) -> str: ...
```

### Identity Evidence

Introduce one common evidence contract that every source emits.

Proposed modules:

- `src/hydra_suite/core/identity/evidence.py`
- `src/hydra_suite/core/identity/cache.py`

Proposed data model:

```python
@dataclass(frozen=True)
class IdentityEvidence:
    frame_idx: int
    detection_index: int
    detection_id: int | None
    source_name: str
    log_probs: np.ndarray  # shape: [catalog_size]
    observed_mask: np.ndarray | None
    confidence_scale: float
    metadata: dict[str, Any]


class IdentityEvidenceCache:
    def save_frame(self, frame_idx: int, evidences: list[IdentityEvidence]) -> None: ...
    def load_frame(self, frame_idx: int) -> list[IdentityEvidence]: ...
    def flush(self) -> None: ...
```

Rules:

- AprilTag evidence should be represented as a very sharp categorical distribution over one identity plus a small floor elsewhere.
- CNN evidence should be stored as a calibrated posterior over the catalog, not only the top class.
- Multihead models must preserve per-factor uncertainty until the factor-to-catalog mapping step is complete.
- Missing evidence should be represented explicitly, not by dropping the detection from the cache.
- The evidence cache must be emitted equivalently from both the streaming live path and the replay fallback path defined in `trackerkit-streaming-individual-analysis-plan.md`.

### Online Identity Decoder

Add a new online decoder layer that runs after geometry assignment and before final identity publication.

Proposed module:

- `src/hydra_suite/core/identity/online.py`

Proposed data model:

```python
@dataclass
class TrackIdentityBelief:
    slot_index: int
    trajectory_id: int
    log_posterior: np.ndarray  # shape: [catalog_size]
    committed_identity: int | None
    commitment_age: int
    stable_frames: int
    last_visible_frame: int | None
    slot_lock_identity: int | None
    slot_lock_strength: float


@dataclass(frozen=True)
class IdentityAssignment:
    slot_index: int
    identity_index: int | None
    confidence: float
    margin: float
    committed: bool
```

Proposed API:

```python
class OnlineIdentityDecoder:
    def __init__(self, catalog: IdentityCatalog, params: dict[str, Any]) -> None: ...

    def ensure_slots(self, n_slots: int) -> None: ...

    def update_matched_slots(
        self,
        frame_idx: int,
        matched_slot_to_detection: dict[int, int],
        evidences: list[IdentityEvidence],
        slot_states: list[str],
    ) -> None: ...

    def decay_unmatched_slots(self, frame_idx: int, visible_slots: list[int]) -> None: ...

    def decode_visible_assignments(
        self,
        visible_slots: list[int],
    ) -> list[IdentityAssignment]: ...

    def reserve_slot_on_commit(self, slot_index: int) -> None: ...
    def clear_slot(self, slot_index: int, reason: str) -> None: ...
```

Online algorithm:

1. Predict each slot belief forward with a sticky transition matrix.
2. Fuse all matched evidence for that slot in log-space.
3. Keep the full posterior over the catalog.
4. Solve a visible-slot to identity assignment with dummy unassigned states.
5. Publish identity only when posterior and stability cross configurable thresholds.
6. Retain the full posterior even when nothing is committed yet.

### Offline Identity Decoder

Add a probabilistic offline decoder that consumes the same evidence cache and final trajectories.

Proposed modules:

- `src/hydra_suite/core/identity/offline.py`
- `src/hydra_suite/core/identity/fragments.py`

Proposed API:

```python
def smooth_trajectory_identity_posteriors(
    trajectories_df: pd.DataFrame,
    evidence_cache: IdentityEvidenceCache,
    catalog: IdentityCatalog,
    params: dict[str, Any],
) -> pd.DataFrame: ...


def build_identity_fragments(
    smoothed_df: pd.DataFrame,
    params: dict[str, Any],
) -> pd.DataFrame: ...


def solve_fragment_identity_assignment(
    fragments_df: pd.DataFrame,
    catalog: IdentityCatalog,
    params: dict[str, Any],
) -> pd.DataFrame: ...


def run_identity_residual_assignment(
    fragments_df: pd.DataFrame,
    params: dict[str, Any],
) -> pd.DataFrame: ...
```

Offline algorithm:

1. Run forward-backward or Viterbi per trajectory on the stored posterior evidence.
2. Segment trajectories into fragments where the smoothed identity is stable.
3. Accumulate fragment log-likelihoods over the catalog.
4. Solve a global uniqueness-constrained assignment across overlapping fragments.
5. Lock in confident fragments and iterate on ambiguous fragments.
6. Detect likely mixed-identity trajectories and split at evidence regime changes.
7. Write both summary outputs and audit-friendly posterior diagnostics.

### Slot Reservation Policy

Support the bounded-slot rule explicitly.

Policy statement:

- Once a slot has held one identity with high posterior for long enough, reuse of that slot should be strongly biased toward the same identity.
- After a stronger threshold, the slot may become hard-locked to that identity until a timeout or contradiction condition is reached.

This policy belongs in the online decoder, not in the geometric assigner.

Initial rollout rule:

- soft lock only
- no hard rejection in the first phase
- log every time a lost slot is reused by a different decoded identity than its previous committed identity

Later rollout rule:

- optional hard lock after `IDENTITY_SLOT_LOCK_MIN_FRAMES`
- allow override if the posterior for a new identity exceeds the current lock by a configured margin for a configured dwell time

## Proposed File-Level Changes

### New modules

- `src/hydra_suite/core/identity/catalog.py`
- `src/hydra_suite/core/identity/evidence.py`
- `src/hydra_suite/core/identity/cache.py`
- `src/hydra_suite/core/identity/online.py`
- `src/hydra_suite/core/identity/offline.py`
- `src/hydra_suite/core/identity/fragments.py`
- `src/hydra_suite/core/identity/calibration.py`

### Existing files to change

#### `src/hydra_suite/core/identity/classification/cnn.py`

- Add a posterior-producing path that returns calibrated probabilities over classes, not only top-1 names and confidences.
- Preserve multihead posterior information until the mapping into the identity catalog is complete.
- Keep the current `ClassPrediction` path for compatibility during migration.

#### `src/hydra_suite/core/tracking/precompute.py`

- Add an `IdentityEvidencePhase` that writes the new sidecar evidence cache.
- Keep existing `CNNIdentityCache` writing during migration.
- Ensure live streaming and replay fallback both emit the same evidence contract using the shared filtered-detection payload.

#### `src/hydra_suite/core/tracking/worker.py`

- Replace hard-label history as the primary live identity state with `OnlineIdentityDecoder`.
- Use identity posteriors for slot commitment, uniqueness-aware decoding, and slot reservation.
- Keep track-level CSV outputs compatible while adding new summary columns.

#### `src/hydra_suite/core/tracking/cnn_features.py`

- Deprecate direct hard-label track-history usage.
- Convert this module into a thin evidence adapter during migration.

#### `src/hydra_suite/core/tracking/tag_features.py`

- Convert from majority-vote tag history into an evidence builder for AprilTag observations.
- Keep deterministic tag helper functions where useful for UI and diagnostics.

#### `src/hydra_suite/core/assigners/hungarian.py`

- Reduce identity overlay responsibility.
- Continue using identity only as a compatibility term for geometry association.
- Do not make this module the global identity uniqueness solver.

#### `src/hydra_suite/core/post/identity_postprocess.py`

- Replace the current heuristic split-and-rechain implementation with a compatibility shim over the new offline decoder.
- Preserve audit columns such as `UniqueIdentityKey` while changing how they are computed.

#### `src/hydra_suite/trackerkit/gui/orchestrators/tracking.py`

- Load the new identity evidence sidecar when building the rich export dataframe.
- Run the offline decoder after rich evidence assembly.
- Continue emitting label-friendly summary columns for video export and CSV export.

#### `src/hydra_suite/core/identity/properties/export.py`

- Continue exporting current summary columns.
- Add new summary fields for decoded identity, commitment state, entropy, and margins.
- Do not use this CSV as the primary posterior store.

## Proposed Data Artifacts

### Raw evidence artifact

Introduce a new sidecar file next to the detection and CNN caches.

Suggested naming:

- `<base>_identity_evidence_<signature>.npz`

Required contents per frame and detection slot:

- `frame_idx`
- `detection_index`
- `detection_id`
- `source_name`
- `log_probs`
- `catalog_labels`
- `calibration_signature`
- `runtime_signature`
- `observed_mask`

### Rich export summary columns

Add new summary columns to the rich dataframe and exported CSV.

Suggested columns:

- `IdentityAssignedID`
- `IdentityAssignedLabel`
- `IdentityAssignedConfidence`
- `IdentityPosteriorMargin`
- `IdentityEntropy`
- `IdentityCommitted`
- `IdentityEvidenceSources`
- `IdentityConflictFlag`
- `IdentitySlotLockLabel`
- `IdentitySlotLockStrength`

Existing columns such as `UniqueIdentityKey` can remain as compatibility and UI presentation fields.

## Configuration and Rollout Flags

Add the new behavior behind explicit configuration so migration is reversible.

Suggested runtime parameters:

- `ENABLE_IDENTITY_POSTERIOR_CACHE`
- `ENABLE_IDENTITY_ONLINE_DECODER`
- `ENABLE_IDENTITY_OFFLINE_DECODER`
- `IDENTITY_TRANSITION_EPSILON`
- `IDENTITY_UNKNOWN_PRIOR`
- `IDENTITY_COMMIT_THRESHOLD`
- `IDENTITY_COMMIT_MIN_HITS`
- `IDENTITY_DISPLAY_THRESHOLD`
- `IDENTITY_SLOT_LOCK_MIN_FRAMES`
- `IDENTITY_SLOT_LOCK_STRENGTH`
- `IDENTITY_SLOT_LOCK_OVERRIDE_MARGIN`
- `IDENTITY_OFFLINE_FRAGMENT_MIN_FRAMES`
- `IDENTITY_OFFLINE_AMBIGUITY_MARGIN`
- `IDENTITY_OFFLINE_MIN_COST_FLOW`
- `IDENTITY_CALIBRATION_REQUIRED`

Migration rule:

- Old and new identity pipelines must be runnable side by side until parity and auditability are satisfactory.

## Detailed Rollout Plan

### Phase 0: Preserve Posterior Information

Goal:

- Stop losing information at cache write time.

Tasks:

- Add posterior vector output to CNN identity inference.
- Add calibration metadata plumbing.
- Introduce `IdentityEvidenceCache`.
- Implement the evidence cache on top of the shared payload and cache-emission points from the streaming plan rather than creating a parallel path.
- Keep current top-1 cache and CSV export unchanged.

Exit criteria:

- Every relevant frame can be reconstructed from a posterior artifact without rerunning the model.

### Phase 1: Realtime Decoder Behind Flag

Goal:

- Add a new online identity decoder without removing current behavior.

Tasks:

- Instantiate `IdentityCatalog` at run start.
- Build evidence vectors from AprilTag and CNN outputs.
- Update slot beliefs online.
- Add uniqueness-aware visible assignment with dummy unassigned identities.
- Emit summary diagnostics and logs.
- Consume live evidence from the streaming forward path, not from a replay-only precompute phase.

Exit criteria:

- The live path produces posterior-aware identity outputs under a feature flag.

### Phase 2: Commitment and Slot Reservation

Goal:

- Stabilize live identity output and implement the bounded-slot rule.

Tasks:

- Add commitment state and stability counters.
- Add soft slot lock behavior.
- Add later optional hard slot lock behavior.
- Record lock overrides and contradictions.

Exit criteria:

- Realtime identity outputs are stable and slot reuse is identity-aware.

### Phase 3: Offline Smoothing

Goal:

- Use future evidence to clean the final result.

Tasks:

- Add forward-backward and Viterbi smoothing over final trajectories.
- Emit smoothed posterior summaries.
- Preserve raw and smoothed outputs together.
- Consume the same identity evidence sidecar emitted by both streaming and replay paths.

Exit criteria:

- A confident late identity burst can improve ambiguous early frames in the same trajectory.

### Phase 4: Global Fragment Assignment

Goal:

- Enforce uniqueness globally across overlapping fragments.

Tasks:

- Build fragment-level evidence summaries.
- Solve overlap-constrained assignment using min-cost flow or equivalent integer optimization.
- Add residual reassignment for ambiguous fragments.

Exit criteria:

- No overlapping decoded fragments share the same identity unless explicitly marked unresolved.

### Phase 5: Retire Legacy Identity Heuristics

Goal:

- Remove old majority-vote identity behavior once the new path is validated.

Tasks:

- Decommission `TrackCNNHistory` and `TrackTagHistory` as the primary live decision machinery.
- Remove heuristic-only split/join as the main offline decoder.
- Keep compatibility adapters only where UI or exports still depend on them.

Exit criteria:

- The new posterior-based pipeline is the default identity system.

## Testing Strategy

### Unit tests

- `IdentityCatalog` mapping correctness.
- Calibration application and numerical stability.
- Posterior cache roundtrip.
- Online transition and evidence fusion logic.
- Unique assignment decoder with dummy unassigned states.
- Slot lock and override behavior.
- Offline fragment likelihood accumulation.

### Integration tests

- Realtime run with AprilTag only.
- Realtime run with CNN unique-ID only.
- Realtime run with AprilTag plus CNN evidence conflict.
- Occlusion and reappearance with slot reservation.
- Offline smoothing correcting early ambiguity.
- Offline global fragment solver resolving overlapping conflicts.

### Regression scenarios

- Two live tracks both moderately favor the same identity.
- One track has low-confidence diffuse evidence across many frames.
- One multihead factor is missing while others remain informative.
- A long-lived slot respawns near a different animal.
- A trajectory contains a clean internal identity switch point.

### Metrics

- IDF1
- identity switch count
- duplicate-ID violations per frame
- unresolved or tentative fraction
- recovery delay after occlusion
- agreement between online committed identity and offline final identity

## Risks and Open Questions

- A true identity catalog is straightforward for AprilTag, but may require explicit configuration for CNN classifiers whose labels are not directly the unique animal IDs.
- If existing models cannot emit full calibrated posterior vectors, Phase 0 will require backend work before the larger decoder rollout can start.
- The online decoder should remain low-latency; if the catalog becomes large, visible-slot decoding must remain bounded and efficient.
- The slot lock policy must start soft to avoid preserving wrong commitments too aggressively.
- The offline solver should degrade gracefully when the evidence is genuinely ambiguous.

## Acceptance Criteria

- The system preserves raw identity uncertainty in a persisted artifact.
- The online path can output tentative and committed identities without discarding low-confidence evidence.
- The offline path uses future evidence to improve final identity results.
- The uniqueness constraint is enforced globally for visible trajectories in both live decoding and offline fragment assignment.
- Slot reuse can be configured to prefer or require the same identity after sufficient commitment time.
- Existing CSV, rich export, and video-label flows remain functional during migration.
- The identity pipeline behaves the same regardless of whether evidence was produced by the live streaming path or replay fallback.

## Recommended Joint Execution Order

If the streaming plan and identity overhaul are being executed together, the preferred combined order is:

1. Streaming Phase 0: instrumentation and artifact-contract audit.
2. Streaming Phase 1: shared oriented-analysis payload.
3. Streaming Phase 2: GPU-native CNN runtime path plus posterior-output hook.
4. Identity Overhaul Phase 0: identity catalog, calibrated posterior contract, and evidence sidecar cache.
5. Streaming Phase 3 and Phase 4: live pose and CNN dispatch on the shared payload, including evidence emission.
6. Identity Overhaul Phase 1 and Phase 2: online decoder, commitment, and slot reservation on the live path.
7. Streaming Phase 5: make streaming the default forward path.
8. Identity Overhaul Phase 3 and Phase 4: offline smoothing and global fragment assignment on top of the preserved evidence.
9. Streaming Phase 6 and Identity Overhaul Phase 5: retire replay-first and hard-label legacy behavior.

Rationale:

- the streaming plan owns the transport and runtime substrate
- the identity overhaul owns the evidence model and decoders
- building the identity overhaul first would force rework when the transport path changes
- making streaming default too early would leave the live path without the probabilistic decoder that justifies the transition

## Recommended First Implementation Slice

The first engineering slice should be intentionally narrow and unblock the rest of the roadmap.

1. Complete Streaming Phase 1 so the filtered-detection payload and stable detection-slot indexing exist.
2. Complete Streaming Phase 2 so CNN execution has a posterior-producing hook.
3. Add posterior-producing CNN output plus calibration metadata in `cnn.py`.
4. Add the new identity evidence sidecar cache in `precompute.py` using the shared streaming payload path.
5. Build `IdentityCatalog` and `IdentityEvidence` contracts.
6. Fix live multihead history initialization so it no longer collapses to `("flat",)`.
7. Add a feature-flagged online decoder that computes posteriors without yet changing the final assignment behavior.

This slice provides the data preservation and observability needed before replacing the current live and offline heuristics, and it does so on top of the same execution substrate that the final product will use.
