# Post-processing

Post-processing is where HYDRA Suite turns raw online trajectories into something you can trust for analysis.

The current implementation is deliberately conservative: it would rather keep two defensible fragments than one polished but incorrect identity trace.

## Why Post-processing Exists

Online tracking has to make decisions immediately. That is useful for visualization and throughput, but it also means difficult events can slip through:

- crossings,
- temporary occlusions,
- missing detections,
- abrupt detector failures,
- and short identity swaps.

Post-processing gets a second chance to inspect those events with more context.

## Current Processing Stages

### 1. Remove weak fragments

Very short trajectories are dropped using `MIN_TRAJECTORY_LENGTH`.

This removes:

- detector noise,
- unstable startup segments,
- and partial fragments that are too small to support reliable identity claims.

### 2. Split implausible motion

The pipeline can break a trajectory when motion becomes physically implausible.

Controls:

- `MAX_VELOCITY_BREAK`
- `MAX_VELOCITY_ZSCORE`
- `VELOCITY_ZSCORE_WINDOW`
- `VELOCITY_ZSCORE_MIN_VELOCITY`

Use the fixed threshold when you know the approximate speed envelope. Use the z-score breaker when failures look like sudden outliers relative to each trajectory's own history.

### 3. Split long occlusion runs

If a trajectory stays in `occluded` state for too long, the current implementation can split at that gap rather than pretending continuity survived a long missing period.

Control:

- `MAX_OCCLUSION_GAP`

### 4. Merge forward and backward hypotheses

If backward tracking was enabled, the post-processor compares forward and backward outputs and merges only the segments that genuinely agree.

Controls:

- `AGREEMENT_DISTANCE`
- `MIN_OVERLAP_FRAMES`

### 5. Stitch broken fragments

After merge resolution, nearby fragments can be stitched across short gaps when the geometry still makes sense.

This is a pragmatic recovery step for tracks broken by:

- quick turns,
- short detector dropouts,
- or ambiguity during close interactions.

### 6. Relink with motion and optional pose continuity

The current relinking step is stronger than simple gap stitching. It summarizes fragment endpoints and asks whether the later fragment could plausibly be the continuation of the earlier one.

It uses:

- motion extrapolation,
- heading compatibility,
- and pose-shape compatibility when pose data exists.

Controls:

- `ENABLE_TRACKLET_RELINKING`
- `RELINK_POSE_MAX_DISTANCE`
- `MAX_OCCLUSION_GAP`
- `MAX_VELOCITY_BREAK`

### 7. Interpolate short gaps

Interpolation is the last stage. It is not intended to rescue fundamentally bad tracking.

Controls:

- `INTERPOLATION_METHOD`
- `INTERPOLATION_MAX_GAP`

Available modes:

- `none`
- `linear`
- `cubic`
- `spline`

## Recommended Defaults

For most experiments, a safe default posture is:

- enable post-processing,
- keep merge thresholds conservative,
- enable relinking only when fragmenting is a real problem,
- use linear interpolation with a small max gap,
- and visually spot-check crossings before trusting derived statistics.

## When to Be More Conservative

Tighten the post-processing stack if:

- animals are visually similar and identity errors are costly,
- social interactions create frequent overlaps,
- or downstream metrics depend heavily on exact individual continuity.

## When to Be More Permissive

Relax it only when:

- detections are clean,
- crossings are rare,
- and fragmentation is the main failure mode.

## Identity Evidence and the Quality Breaker

When identity is enabled, the fragment solver can split a trajectory where
the identity evidence changes (PELT changepoint detection) before assigning
labels. Two behaviours are worth knowing:

- **The PELT penalty is in probability units.** The signal is the smoothed
  per-label posterior on the simplex, `[0, 1]`, used raw. Earlier versions
  z-scored it per trajectory, which made the penalty's units depend on the
  trajectory and inflated float noise on a flat posterior into unit-variance
  "signal" -- 660 spurious splits over 128 tracks in one measured run. Because
  the units are now absolute, a penalty tuned on one video means the same
  thing on the next.

- **Uninformative evidence is refused, not acted on.** Before splitting, the
  solver assesses evidence quality: the fraction of detections confident
  above `EVIDENCE_CONF_LEVEL` (0.5) must reach `EVIDENCE_MIN_CONF_FRAC`
  (10%), and the distinct-labels-per-frame diversity must reach
  `EVIDENCE_MIN_DIVERSITY` (0.30) of what is achievable. If either fails, the
  solver logs an ERROR and refuses to split or assign:

  ```
  fragment_solver: identity evidence is uninformative (confident=..% of
  detections, diversity=.. over .. frames) -- refusing to split or assign
  identities. Check the classifier's fit_policy / preprocessing.
  ```

  Trajectories are then left intact rather than shredded by a source that
  does not know anything. The smoothed per-row posterior is still written to
  `IdentityFinalSmoothedLabel` / `...Confidence` so you can see what the
  (uninformative) source thought. Seeing this ERROR almost always means the
  classifier is being fed the wrong crop shape -- see
  [Older Classifiers and Crop Preprocessing](individual-analysis.md#older-classifiers-and-crop-preprocessing).

## Related Reading

- [Tracking, Identity Continuity, and Merging](tracking-and-merging.md)
- [Tracking Algorithm Deep Dive](../developer-guide/tracking-algorithm-deep-dive.md)
