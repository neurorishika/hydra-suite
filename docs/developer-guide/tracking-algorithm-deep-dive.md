# Tracking Algorithm Deep Dive

This page documents the current tracking implementation as it exists in the codebase today.

Primary modules:

- `hydra_suite.core.tracking.worker`
- `hydra_suite.core.filters.kalman`
- `hydra_suite.core.assigners.hungarian`
- `hydra_suite.core.post.processing`
- `hydra_suite.core.individual.runtime_api`

## Design Goals

The current implementation optimizes for four things at the same time:

- stable online tracking during long videos,
- reproducible reruns through detection caching,
- conservative identity handling in difficult scenes,
- and practical support for richer cues such as pose-derived direction.

The design is therefore not a single elegant algorithm block. It is a staged pipeline with explicit checkpoints.

## Runtime Topology

`TrackingWorker` is the top-level orchestrator. It owns the run lifecycle, frame iteration, detector construction, Kalman manager, assignment logic, visualization, caching, pose precompute, and final signal emission.

The main control branches are:

- forward live detection,
- forward cached replay,
- backward cached replay,
- preview mode,
- and YOLO two-phase mode with batched prepass.

## Data Contracts

### Track state

Each Kalman slot stores a state vector:

```text
state = [x, y, theta, vx, vy]
```

with:

- `x`, `y`: center position in resized-frame coordinates,
- `theta`: orientation in radians,
- `vx`, `vy`: velocity components in pixels per frame.

### Measurement state

Online measurement updates operate in:

```text
measurement = [x, y, theta]
```

### Auxiliary detection attributes

Association also consumes parallel per-detection arrays:

- area,
- aspect ratio,
- confidence,
- OBB corners,
- `DetectionID`,
- pose-derived heading,
- normalized pose keypoints,
- pose visibility,
- and crop quality heuristics.

## Detection Layer

### Background subtraction path

The background-subtraction branch:

1. converts the frame to grayscale,
2. applies brightness, contrast, gamma, and optional lighting stabilization,
3. updates the background model,
4. generates a foreground mask,
5. applies ROI masking,
6. optionally applies conservative split morphology,
7. and extracts measurement candidates.

This branch is lightweight and self-contained, but it depends strongly on scene stability.

### YOLO OBB path

The YOLO path:

1. runs object detection on the BGR frame,
2. preserves raw detections,
3. generates deterministic `DetectionID` values from absolute frame index,
4. applies filtering after inference,
5. and optionally writes raw detections to cache for later replay.

The filtering stage is important because it decouples:

- expensive inference,
- ROI and size filtering,
- downstream pose extraction,
- and tracking experiments.

### Detection cache semantics

The detection cache is central to the current architecture.

It is used for:

- backward tracking,
- replay without re-running inference,
- pose-property precompute,
- and reproducible tuning.

The worker validates that the cache:

- exists,
- matches the requested frame range,
- and has a compatible format.

Backward mode refuses to run without a compatible cache.

## Kalman Filter Implementation

The filter manager is vectorized across all target slots.

### State transition

The transition matrix is:

```text
[1 0 0 damp 0   ]
[0 1 0 0    damp]
[0 0 1 0    0   ]
[0 0 0 damp 0   ]
[0 0 0 0    damp]
```

with `damp = KALMAN_DAMPING`.

That means:

- position is advanced using damped velocity,
- velocity persists but decays,
- and orientation is modeled as a carried state rather than a velocity-driven state.

### Process noise

The process-noise model is anisotropic. Longitudinal and lateral velocity noise are rotated into the current heading frame before they are injected into the covariance.

Operationally, this means the tracker can assume:

- more uncertainty along the direction of travel,
- less uncertainty sideways,
- and a more biologically plausible motion envelope than isotropic noise would provide.

### Correction

Measurement correction uses:

- circular innovation logic for `theta`,
- Joseph-form covariance update for numerical stability,
- and a post-correction speed clamp based on `REFERENCE_BODY_SIZE`.

### Age-dependent motion restraint

Young tracks are intentionally conservative. Before `KALMAN_MATURITY_AGE`, velocity is attenuated toward zero according to `KALMAN_INITIAL_VELOCITY_RETENTION`.

This reduces the chance that a newly initialized slot immediately predicts itself into a nearby wrong animal.

## Orientation Handling

The orientation logic deserves separate attention because it affects both assignment and visualization.

### Axis ambiguity collapse

OBB orientation is treated as a body axis unless pose provides a directed heading. The worker compares `theta` and `theta + pi` against the last reliable orientation and chooses whichever is closer.

This avoids unhelpful 180-degree oscillations.

### Pose-derived heading

If pose extraction is enabled and both configured keypoint groups are visible enough:

- the worker computes a directed posterior-to-anterior heading,
- normalizes it into `[0, 2*pi)`,
- and marks the detection as directed.

If pose does not provide a valid directed heading, the fallback path returns to axis-based orientation collapse.

### Motion-conditioned smoothing

After correction, orientation is further smoothed:

- low-speed tracks retain historical orientation unless the change is small enough,
- high-speed tracks can flip by 180 degrees if motion direction indicates the heading is reversed.

This is a pragmatic mix of body-axis and motion-direction reasoning.

## Association Stack

The cost matrix is implemented in `TrackAssigner`.

### Baseline cost

The base cost per track-detection pair is:

```text
cost =
  W_POSITION * position_distance
  + W_ORIENTATION * orientation_difference
  + W_AREA * area_difference
  + W_ASPECT * aspect_difference
```

Position distance can be either:

- Euclidean distance, or
- Mahalanobis distance using the predicted innovation covariance.

Orientation difference is circular and respects the directed-vs-axis distinction.

### Stage-1 candidate gate

When advanced association data is available, a coarse candidate gate runs before full cost scoring. It rejects pairs that exceed a local motion envelope derived from:

- global culling threshold,
- per-track uncertainty,
- per-track average step size,
- maximum allowed area ratio,
- and maximum allowed aspect-ratio change.

This keeps the expensive stage focused on plausible candidates.

### Pose rejection

If normalized pose keypoints are available, the assigner can compute a paired pose distance between:

- the current detection,
- and the track's stored pose prototype.

If visibility is high enough and the pose distance exceeds `POSE_REJECTION_THRESHOLD`, the candidate is vetoed even if motion looks acceptable.

This is a strong identity-protection mechanism when pose is reliable.

### Assignment phases

The assignment output is not a single one-shot matching step.

#### Established tracks

Tracks with `tracking_continuity >= CONTINUITY_THRESHOLD` are handled first.

Options:

- Hungarian global assignment,
- or greedy assignment when throughput matters more.

#### Unstable tracks

Lower-continuity tracks are filled greedily from the remaining detections.

#### Lost-track respawn

Free detections can be assigned to lost slots if they are far enough from non-lost predictions, using `MIN_RESPAWN_DISTANCE`.

This is a controlled reuse policy, not a free-for-all.

## Track Memory Beyond Kalman State

The worker keeps more than just the Kalman filter state.

Per-track memory includes:

- `orientation_last`,
- `last_shape_info`,
- `track_pose_prototypes`,
- `track_avg_step`,
- continuity count,
- missed-frame count,
- local CSV row count,
- and recent positions for speed estimation.

This extra memory is what lets the tracker remain practical in crowded scenes without inflating the Kalman state unnecessarily.

## Pose-Enhanced Tracking Path

Pose extraction is not only a downstream analysis feature. In the current implementation it also feeds back into tracking.

### Precompute stage

If pose extraction is enabled in YOLO OBB mode, the worker can precompute pose properties from cached detections before online tracking begins.

That precompute produces a deterministic individual-properties cache keyed by:

- video,
- frame range,
- detection hash,
- filter settings hash,
- and extractor hash.

### Runtime use

During frame processing the worker can recover, per detection:

- pose keypoints,
- pose visibility,
- normalized pose prototype,
- and pose-derived heading.

Those values then influence:

- orientation override,
- association vetoing,
- track prototype updates,
- relinking,
- and optional export.

## Forward and Backward Passes

Backward mode is not a special detector. It is a special playback mode over the same cached detections.

Important properties:

- it reuses the requested frame range,
- it can skip frame reads entirely for speed,
- it writes a second trajectory hypothesis,
- and orientation handling includes a backward fallback correction for non-pose-directed cases.

The purpose is not to generate a prettier trajectory. The purpose is to offer a second causal interpretation for later consensus.

## Post-Processing Pipeline

The main post-processing functions are in `hydra_suite.core.post.processing`.

### Cleaning and break detection

The cleaning stage can:

- remove short fragments,
- split on excessive absolute velocity,
- split on abnormal velocity z-scores,
- and split across long occlusion runs.

These breakers exist because online assignment is allowed to be imperfect if later consistency checks can safely cut bad sections.

### Conservative forward/backward resolution

`resolve_trajectories` does not blindly average two passes.

It:

1. converts trajectory inputs into DataFrames,
2. removes trivially bad fragments,
3. finds forward/backward overlap candidates,
4. keeps only pairs with enough agreeing frames,
5. performs conservative segment-level merging,
6. removes redundant fragments,
7. merges overlapping agreeing fragments,
8. and stitches nearby fragments across short gaps.

The result is intentionally fragmentation-tolerant and identity-protective.

### Motion-and-pose relinking

`relink_trajectories_with_pose` summarizes each fragment by:

- start and end frames,
- start and end position,
- start and end heading,
- short-window velocity,
- and optional pose prototypes at both ends.

Candidate relinks are accepted only if:

- gap length is within `MAX_OCCLUSION_GAP`,
- predicted motion reaches the next fragment within an allowed distance,
- heading is compatible when motion is informative,
- and pose distance is below `RELINK_POSE_MAX_DISTANCE` when both sides have usable pose.

### Interpolation

`interpolate_trajectories` reindexes each trajectory onto a complete frame range, fills missing state labels as `occluded`, and interpolates:

- `X`,
- `Y`,
- and `Theta`.

Theta interpolation uses circular logic, which prevents wrap-around artifacts near `0` and `2*pi`.

## Performance Model

The implementation includes several explicit performance levers:

- batched YOLO prepass,
- detection cache reuse,
- optional frame prefetching,
- optional KD-tree candidate pruning,
- optional greedy assignment,
- Numba kernels for cost computation and post-processing inner loops,
- and visualization-free cached replay.

For larger target counts, enabling spatial optimization is usually worth it.

## Parameter Surfaces That Matter Most

| Parameter | Role in current implementation | Failure if too small | Failure if too large |
|---|---|---|---|
| `REFERENCE_BODY_SIZE` | Scales motion, velocity, and geometric heuristics | tracker becomes too tight | tracker becomes overly permissive |
| `MAX_DISTANCE_THRESHOLD` | hard assignment acceptance ceiling | fragmentation | swaps and duplicates |
| `LOST_THRESHOLD_FRAMES` | occlusion tolerance | premature track death | stale tracks survive too long |
| `KALMAN_DAMPING` | motion persistence | jerky short-term prediction | overshoot after stops |
| `POSE_REJECTION_THRESHOLD` | pose veto strictness | true matches rejected | pose provides little protection |
| `AGREEMENT_DISTANCE` | forward/backward merge tolerance | under-merged outputs | over-merged outputs |
| `MAX_OCCLUSION_GAP` | relinking and splitting window | missed recoveries | speculative reconnects |

## Source-of-Truth Files

If you need to audit behavior, these files are the primary reference points:

- `src/hydra_suite/core/tracking/worker.py`
- `src/hydra_suite/core/filters/kalman.py`
- `src/hydra_suite/core/assigners/hungarian.py`
- `src/hydra_suite/core/post/processing.py`
- `src/hydra_suite/core/individual/runtime_api.py`
- `tests/test_tracking_pipeline_synthetic.py`
- `tests/test_post_tracklet_relinking.py`

## Related Documentation

- [Tracking, Identity Continuity, and Merging](../user-guide/tracking-and-merging.md)
- [Post-processing](../user-guide/post-processing.md)
- [Technical Reference (LaTeX)](../reference/technical-reference.md)
