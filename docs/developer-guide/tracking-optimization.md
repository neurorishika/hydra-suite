# Tracking Auto-Tuner

TrackerKit's tracking auto-tuner is an **unlabeled candidate recommender**. It
does not estimate ground-truth tracking accuracy and must not describe a result
as mathematically optimal or “the best” configuration. Without labeled identity
trajectories, a tracker can be consistently wrong.

## Decision model

The tuner separates cheap proposal generation from the decision to recommend a
change:

1. Raw detections are cached once. A production cache may be passed as either
   its `detection.npz` member or its containing directory; the tuner canonicalizes
   both forms to the cache directory. Confidence, IoU, size, aspect, count,
   source, and ROI filtering are re-applied with the same `filter_for_source`
   path used by production inference. Background-subtraction detections therefore
   bypass YOLO-only confidence filtering.
   Candidate confidence-density regions are built from those same filtered
   detections after any ROI mask is resampled to native cache-frame coordinates;
   raw cached detections are never substituted for density evidence. Sparse
   cache keys retain their absolute video-frame numbers and break temporal
   smoothing/regions at each missing frame rather than creating a false bridge.
   A cache prepared from the dialog runs raw detection and every configured
   downstream head-tail, CNN, pose, and AprilTag stage needed for
   production-faithful replay; it is not a detection-only shortcut.
   Read-only density replay is admitted only when a conservative estimate of
   its raw, smoothing, binary, labeling, and arena-mask buffers fits
   `AUTOTUNE_DENSITY_MAX_BYTES` (512 MiB by default). Legacy monolithic caches
   use the requested span as a conservative count without loading their payload.
   An over-budget or cancelled build is explicit missing validation evidence,
   never a silent no-density fallback.
2. Optuna explores the user-selected parameter dimensions on a chronological
   training slice. Its scalar loss is only a search heuristic. The current
   production settings are also evaluated exactly and are never clamped into
   the search space.
3. The current settings and a bounded shortlist of proposals are replayed on a
   held-out tail through `TrackingEngineCore` itself, with the inference cache
   opened read-only. Replay runs an unscored pre-roll from the original tracking
   start so tracker maturity, lifecycle, and identity state enter the tail as
   they would in production. Cache provenance remains tied to the original
   production configuration rather than the tail's loop bounds. This is the
   apply/no-apply evidence; the lightweight search loop is not treated as
   production-equivalent.
4. Held-out outputs are compared on explicit lower-is-better losses:

   - forward/backward cycle disagreement, normalized by body size and globally
     aligned for arbitrary startup slot permutations;
   - missing observations (coverage loss);
   - present/missing transitions (fragmentation loss);
   - second-difference motion roughness computed from exported detections, not
     hidden Kalman posterior positions.

   One global slot alignment is reused across four paired temporal regions, so
   a region-local identity swap cannot masquerade as a consistent result.
   Temporal transitions and triplets at region boundaries remain in the
   evidence. Output safeguards also reject clearly pathological proposals:
   duplicate/colliding slots, source detections measured before the final
   target-count cap, and a starved worst track. These safeguards are still not
   labels or an accuracy claim.
5. Candidates receive Pareto fronts rather than being collapsed into a
   user-weighted accuracy claim. A candidate is automatically recommended only
   when it dominates the exact baseline and other baseline-safe candidates, and
   its held-out improvement clears an uncertainty-aware margin without a
   conservative regression on any metric. Otherwise the UI says **Keep current**.
   Proposals without complete finite held-out metrics remain inspection-only and
   cannot be applied or persisted as a recommendation.

The user can still preview any proposal for inspection. Applying settings is
limited to current settings or a production-validated candidate; neither action
is an automatic accuracy claim.

## Components

- `core/tracking/optimization/detection_config.py` builds the source-aware
  inference configuration shared by cache building, search replay, and preview.
- `core/tracking/optimization/optimizer.py` owns proposal search, exact baseline
  evaluation, shortlist orchestration, and result metadata.
- `core/tracking/optimization/parameter_contract.py` is the Qt-free typed
  search/display contract shared by core proposal generation and TrackerKit.
- `core/tracking/optimization/production_replay.py` adapts the production
  `TrackingEngineCore` into a read-only observed-position evaluator.
- `core/tracking/optimization/unlabeled_scoring.py` contains the pure NumPy
  cycle, output-quality, aggregation, Pareto, and baseline-protection logic.
- `trackerkit/gui/autotune_contract.py` is the single apply contract. It rejects
  unsupported values before changing any widget and explicitly converts core
  frame counts to UI seconds using the active FPS.

## Parameter contract

Every core search key must have a corresponding TrackerKit control. The contract
test fails if an unapplyable search dimension is introduced. The shared typed
contract owns the search bounds, storage units, and each real Qt control's
decimal precision. Every seed, random/restart point, and Optuna proposal is
quantized to that precision before it is evaluated; applying it then writes the
same canonical value. This prevents the GUI from silently rounding a validated
candidate into a different configuration. Values are range-validated atomically
before any widget changes; they are not clamped on write-back.

Lifecycle values remain integer frame counts in the engine and appear as seconds
in the UI. The apply contract converts them using the active FPS and the
seconds-widget precision, preserving the frame value through the supported
frame/FPS ranges. Their search neighborhood is expressed as dimensionless
factors around the current duration; this preserves comparable real-time
exploration across different video FPS. Dynamic ranges are capped at the
corresponding seconds-control limits so every generated candidate remains
representable on write-back.

Candidate evaluation uses the same public-control merge that TrackerKit applies.
Engine-only dependent values are always recalculated after a candidate override:
changing the public longitudinal Kalman multiplier leaves the retained hidden
lateral multiplier unchanged and recomputes anisotropy as
`max(1, longitudinal / max(lateral, 1e-6))`, exactly as the engine builder
does. Thus tuning longitudinal noise is a long-only production change, not a
proportional change to both axes. Likewise, a distance-multiplier candidate
rebuilds its pixel distance threshold from the effective body size. Derived
fields are not applied directly to hidden widgets; they are only replay-time
representations of the public candidate.

The selectable dimensions are source- and evidence-aware. Background
subtraction does not show inert YOLO confidence or IoU controls. For YOLO,
confidence and IoU are disabled with an explicit reason when the raw cache also
contains downstream head-tail, CNN, pose, or AprilTag evidence: changing the
post-cache filter can change membership and that evidence cannot safely be
re-indexed in read-only replay. Core exposes these exclusions as
`disabled_tuning_dimensions`, and the dialog applies them even for restored
settings.

Sequential YOLO stage 2 runs at a permissive raw confidence floor. The final
confidence threshold is applied after cache loading, so a trial can recover
detections below the threshold active when the cache was first built.
Sequential cache keys include stage-1 confidence, crop geometry, stage-2
extraction settings, target/count caps, slicing configuration, and any ROI mask
used for sliced stage 1; changing a raw-generation setting cannot silently reuse
incompatible detections. Direct, sequential, and background-subtraction raw
keys also include `emit_native_geometry`: compact tracking caches do not
serialize native polygons, so a configuration that requires them bypasses raw
cache reads and writes and recomputes live instead of mistaking a polygon-free
cache hit for export-ready evidence.

## State and cancellation

Saved result rankings are keyed by raw cache/data-manifest and video signatures,
frame range, the complete canonical production-replay parameter mapping, selected
tuning dimensions, proposal weights, trial/seeding settings, plateau behavior,
sampler, and objective version. The parameter snapshot fingerprints ndarray
contents (for example ROI masks) as well as scalar, nested, arena, association,
identity, density, and other fixed replay settings. Mutable autotune sidecars,
density artifacts, and diagnostics are deliberately excluded from the cache
signature, so saving a ranking cannot invalidate itself. Rebuilding an input or
changing any evaluation-semantic configuration invalidates old rankings.
Detection caches and production validation replays are read-only.

Automatic promotion requires enough held-out support for four paired regions at
the largest active temporal horizon. A short range can still show proposals,
but it retains the current settings and explains that held-out validation cannot
support a recommendation. Frame count alone is not evidence: each region must
also contain observed forward/backward motion triplets and robust shared cycle
observations before it contributes to promotion. Cycle evidence has both an
absolute shared-observation floor and a horizon/slot-scaled coverage floor.
When a candidate changes a lifecycle threshold, held-out output must also
exercise a threshold that distinguishes it from baseline: maturity needs a
consecutive observed run one frame longer than the lower baseline/candidate
age (a newly bootstrapped track starts at continuity zero), and loss needs a
bracketed missing run reaching the lower value. A run or gap between those
values exercises one lifecycle policy while the other remains unchanged.
Otherwise the current settings are retained.

Cancel, window close, `reject()`, `accept()`, and direct `done()` all request
optimizer and preview cancellation before a terminal dialog transition. The
dialog waits only a bounded time. If a worker has not stopped, the dialog stays
open with a stopping status and may retry once the worker cooperates; it never
destroys a live Qt thread. Plateau stopping is separate from explicit user
cancellation: only the former sets search convergence, and a plateau-converged
search still performs held-out validation. An explicit cancellation discards
partial proposals and held-out state without ranking, emitting, saving,
previewing, or enabling an apply action from that stale evidence.

## Remaining limitation

Forward/backward agreement and output stability are useful rejection signals,
not identity or localization oracles. A future labeled evaluation path should be
added alongside this workflow and should take precedence whenever trustworthy
annotations become available.
