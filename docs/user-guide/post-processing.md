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

## Honest Identity Evidence

Three behaviours keep the fragment solver from treating weak identity
evidence as strong. The first is on by default; the other two are opt-in
(see *How the defaults were chosen* below):

- **Non-identifying mass stays "unknown".** When a classifier puts
  probability on a class that is not in the identity catalog -- a declared
  non-identifying class such as `notag`, or anything out of vocabulary --
  that probability now goes to the catalog's `unknown` slot. Before, it was
  dropped and the remainder renormalized, so a crop that was 90 % `notag`
  and 10 % one tag read as a confident tag. Config key
  `identity_noncatalog_to_unknown` (engine `IDENTITY_NONCATALOG_TO_UNKNOWN`,
  default `true`). It applies to the realtime decoder and the offline solver
  alike.
- **Absolute support gate.** `fragment_min_support` (default 0.5) is a floor
  on *relative* support -- support renormalized over the known labels, which
  saturates towards 1 on any long trajectory whatever the absolute evidence.
  `fragment_min_absolute_posterior` (engine `FRAGMENT_MIN_ABSOLUTE_POSTERIOR`,
  default `0` = off, `0.3` recommended when enabled) additionally requires the fragment's mean
  *absolute* posterior on a label (the `unknown` mass kept in the
  denominator) to reach this value before the label is a candidate at all.
  It only applies to fragments that have cache evidence; a fragment with
  none keeps whatever the online prior supports.
- **Absolute seeding.** Fragments claim labels in descending
  `duration x evidence` order; with `fragment_absolute_seeding` (engine
  `FRAGMENT_ABSOLUTE_SEEDING`, default `false`) the evidence term is the
  absolute posterior, so a long weakly-identified trajectory no longer
  claims a label ahead of a shorter, clearly-identified one.

`fragment_length_weight` has always been clamped to `[0, 1]`; a value outside
that range now logs a warning instead of being clamped silently.

## Evidence-Trust Layer (optional)

`identity_evidence_trust` (engine `IDENTITY_EVIDENCE_TRUST`, default `false`)
decides, from detection geometry alone -- never from the classifier -- which
evidence frames show one animal clearly, and lets those drive identity:

1. **Trusted frames.** A detection-backed frame is trusted when all active
   checks pass:
   - *isolation*: the nearest other detection centre in the same frame is
     more than `identity_trust_isolation_bl` (1.0) body lengths away;
   - *single-animal shape*: OBB area within `+-identity_trust_area_tolerance`
     (0.35) of the single-animal prior, and elongation at least
     `identity_trust_min_aspect_fraction` (0.65) of the prior's;
   - *keypoint completeness* (only when the run has pose): at least
     `identity_trust_min_keypoint_fraction` (0.75) of keypoints with
     confidence `>= identity_trust_keypoint_conf` (0.5).
2. **Per-video calibration.** The body length and the shape prior are the
   medians over this video's *isolated* detections (nearest neighbour more
   than `identity_trust_prior_isolation_bl`, 2.0, body lengths away) -- a
   whole-video median is dominated by touching animals in a crowded video.
   The shape check switches itself off when fewer than
   `identity_trust_min_prior_detections` (200) isolated detections exist, the
   keypoint check when there is no pose, and the whole layer when the
   detection cache (the geometry source) is unavailable. There is no
   stationarity filter.
3. **Distinct views.** Consecutive trusted frames collapse into one *view*
   until the animal moves `identity_trust_view_move_bl` (0.5) body lengths
   or turns `identity_trust_view_turn_deg` (30) degrees, so an object seen
   10 000 times standing still counts once.
4. **Tiered solving.** Fragments with at least `identity_trust_min_views`
   (1) trusted views form the trusted tier: their support is the mean
   absolute posterior over views and they all claim labels first (ordered
   by `views x support`). Every other fragment uses the honest evidence
   above, seeds after the trusted tier, and may never displace a trusted
   fragment -- it can only take labels the trusted tier left free.
5. **Splitting on trusted switches.** With `enable_pelt_splitting` on and the
   trust layer active, PELT on every frame's smoothed posterior is replaced
   by a split at confident label switches in the *trusted-view* sequence:
   label A on one side, B on the other, each side at least
   `identity_trust_split_min_views` (3) views with mean posterior
   `>= identity_trust_split_min_confidence` (0.5), and the two-label
   explanation better than the best single label by
   `identity_trust_split_min_log_odds` (8.0) nats of evidence. The cut is
   placed in the untrusted gap between the two views. With the trust layer
   off, `enable_pelt_splitting` keeps the original PELT behaviour.

Each run logs one summary line: trusted fraction, body length, prior,
and which checks were active. The trust keys are config-only for now (no
TrackerKit controls). The layer was validated together with
`fragment_min_absolute_posterior: 0.3` and `fragment_absolute_seeding: true`
(the untrusted tier's evidence); enable all three together.

### How the defaults were chosen

Each piece was replayed through the real fragment solver on six crowded
full-arena videos (central dish labels compared against a dish-only
reference) and switched on by default only if, on every video, agreement did
not drop, the wrong-label rate did not rise by more than 0.005, and the
labels it removed had a median per-frame posterior below 0.3 (i.e. it removed
unsupported labels, not real ones).

- *Non-identifying mass to unknown*: agreement never lower by more than
  0.0001 and up to +0.14 higher, wrong rate never up by more than 0.003. It is a correctness
  fix and ships on; on one video the few labels it removed (388 rows) had a
  median posterior of 0.35.
- *Absolute gate + seeding*: agreement up on five videos (up to +0.23) but
  the wrong-label rate rose by 0.011 and 0.007 on two -- off by default.
- *Trust layer*: agreement up on all six (up to +0.37) and the wrong-label
  rate down or within 0.001 on all six, but it also removes labels whose per-frame posterior is
  moderate (median 0.36-0.61 on four videos) and lowers coverage of crowded
  regions by up to 6 points -- off by default. Adding trusted-view splitting
  raises agreement further (0.87-0.97 on five videos, 0.75 on the most
  crowded) and fixes most relabel-to-the-wrong-
  animal cases, at the same coverage trade-off.

## Related Reading

- [Tracking, Identity Continuity, and Merging](tracking-and-merging.md)
- [Tracking Algorithm Deep Dive](../developer-guide/tracking-algorithm-deep-dive.md)
