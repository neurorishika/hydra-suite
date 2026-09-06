# Tracking Auto-Tuner

TrackerKit's tracking auto-tuner is an **unlabeled candidate recommender**. It
does not estimate ground-truth tracking accuracy and must not describe a result
as mathematically optimal or “the best” configuration. Without labeled identity
trajectories, a tracker can be consistently wrong.

## Decision model

The tuner separates cheap proposal generation from the decision to recommend a
change:

1. Raw detections are cached once. Confidence, IoU, size, aspect, count, source,
   and ROI filtering are re-applied with the same `filter_for_source` path used
   by production inference. Background-subtraction detections therefore bypass
   YOLO-only confidence filtering.
2. Optuna explores the user-selected parameter dimensions on a chronological
   training slice. Its scalar loss is only a search heuristic. The current
   production settings are also evaluated exactly and are never clamped into
   the search space.
3. The current settings and a bounded shortlist of proposals are replayed on a
   held-out tail through `TrackingEngineCore` itself, with the inference cache
   opened read-only. This is the apply/no-apply evidence; the lightweight search
   loop is not treated as production-equivalent.
4. Held-out outputs are compared on explicit lower-is-better losses:

   - forward/backward cycle disagreement, normalized by body size and globally
     aligned for arbitrary startup slot permutations;
   - missing observations (coverage loss);
   - present/missing transitions (fragmentation loss);
   - second-difference motion roughness computed from exported detections, not
     hidden Kalman posterior positions.

   Metrics are repeated over several temporal regions to expose instability.
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
- `core/tracking/optimization/production_replay.py` adapts the production
  `TrackingEngineCore` into a read-only observed-position evaluator.
- `core/tracking/optimization/unlabeled_scoring.py` contains the pure NumPy
  cycle, output-quality, aggregation, Pareto, and baseline-protection logic.
- `trackerkit/gui/autotune_contract.py` is the single apply contract. It rejects
  unsupported values before changing any widget and explicitly converts core
  frame counts to UI seconds using the active FPS.

## Parameter contract

Every core search key must have a corresponding TrackerKit control. The contract
test fails if an unapplyable search dimension is introduced. Detection
thresholds and assignment controls use the same supported bounds as the UI.
Lifecycle values remain integer frame counts in the engine, but their search
neighborhood is expressed as dimensionless factors around the current duration;
this preserves comparable real-time exploration across different video FPS.
Dynamic ranges are capped at the corresponding seconds-control limits so every
generated candidate remains representable on write-back.

Sequential YOLO stage 2 runs at a permissive raw confidence floor. The final
confidence threshold is applied after cache loading, so a trial can recover
detections below the threshold active when the cache was first built.
Sequential cache keys include stage-1 confidence, crop geometry, stage-2
extraction settings, target/count caps, slicing configuration, and any ROI mask
used for sliced stage 1; changing a raw-generation setting cannot silently reuse
incompatible detections.

## State and cancellation

Saved result rankings are keyed by the cache and video file signatures, frame range, base/domain
parameters, selected tuning dimensions, proposal weights, trial/seeding settings,
plateau behavior, sampler, and objective version. Rebuilding an input or changing
the objective configuration invalidates old rankings. Detection caches and
production validation replays are read-only.

For ranges shorter than eight frames, there is not enough data for a held-out
split. Proposals may still be shown, but automatic recommendation retains the
current settings and reports that held-out validation was unavailable.

## Remaining limitation

Forward/backward agreement and output stability are useful rejection signals,
not identity or localization oracles. A future labeled evaluation path should be
added alongside this workflow and should take precedence whenever trustworthy
annotations become available.
