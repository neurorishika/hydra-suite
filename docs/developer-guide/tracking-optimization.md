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
   duplicate/colliding slots, excess source detections, and a starved worst
   track. These safeguards are still not labels or an accuracy claim.
5. Candidates receive Pareto fronts rather than being collapsed into a
   user-weighted accuracy claim. A candidate is automatically recommended only
   when it dominates the exact baseline and other baseline-safe candidates, and
   its held-out improvement clears an uncertainty-aware margin without a
   conservative regression on any metric. Otherwise the UI says **Keep current**.

The user can still preview and manually choose any proposal. That is an explicit
human decision, not an automatic accuracy claim.

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
incompatible detections.

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
support a recommendation.

Cancel, window close, `reject()`, `accept()`, and direct `done()` all request
optimizer and preview cancellation before a terminal dialog transition. The
dialog waits only a bounded time. If a worker has not stopped, the dialog stays
open with a stopping status and may retry once the worker cooperates; it never
destroys a live Qt thread. Plateau stopping is separate from explicit user
cancellation: only the former sets search convergence, and a plateau-converged
search still performs held-out validation.

## Remaining limitation

Forward/backward agreement and output stability are useful rejection signals,
not identity or localization oracles. A future labeled evaluation path should be
added alongside this workflow and should take precedence whenever trustworthy
annotations become available.
