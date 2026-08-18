# Confidence Metrics

Confidence-related outputs combine detection and tracking signals.

## Typical Metrics

- Detection confidence (detector quality signal)
- Assignment confidence (match quality signal)
- Position uncertainty (state covariance-derived signal)

## Why They Matter

- Identify hard frames for active learning.
- Detect parameter regimes that overfit or underfit scene dynamics.
- Prioritize manual review where trajectory quality is weakest.

## Integration Points

- Collected during tracking worker pipeline.
- Optionally persisted to CSV when enabled.
- Consumed by dataset generation/scoring workflows.

## Frame-Selection Metrics (`data/al/`)

Active-learning frame selection (`hydra_suite.data.al.signals`,
`hydra_suite.data.al.acquisition`) scores every tracked frame on a fixed set
of named channels, then ranks frames by a weighted composite. Every channel
is an **absolute severity** in `[0, 1]`, exactly `0.0` on a frame with no
problem on that axis — not a within-run rank. A previous implementation
min-max normalized each channel across the run, which meant the top frame of
even a cleanly tracked video always scored near `1.0` and `min_score` could
never gate anything; that normalization has been removed and must not be
reintroduced (`_composite_score` in `acquisition.py` documents this in a
code comment for exactly that reason).

The channels:

| Channel | Meaning | Absolute definition |
|---|---|---|
| `uncertainty` | Low mean detection confidence | `0` if `mean_conf >= conf_floor`, else `(conf_floor - mean_conf) / conf_floor` |
| `count` | Detection count vs. expected target count | asymmetric: under-count `(expected - n) / expected`; over-count `min((n - expected) / expected, 1) * 0.5` |
| `crowd` | Animals genuinely overlapping/touching | max pairwise polygon-overlap ratio across detection pairs |
| `fragmentation` | One animal apparently split into two detections | proximity + overlap + both-smaller-than-typical heuristic, with a 0.45 suspicion gate |
| `edge` | A detection near the frame border | border-proximity, computed against the real frame shape |
| `assignment` | Poor track-to-detection matches | `1 - mean(assignment_confidence)`, or a cost-based fallback |
| `track_loss` | Coasted/lost tracks | `min(lost / max_targets, 1)` |
| `position_uncertainty` | High state-covariance uncertainty | `min(mean(uncertainty) / 50, 1)` |
| `nms_instability` | Detections unstable under threshold perturbation (DetectKit path) | `1 - mean(set IoU)` |

**`fragmentation` and `crowd` are separate channels with separate weights and
separate UI controls.** They measure different phenomena: `fragmentation` is
evidence that a single animal was split into two nearby, undersized
detections; `crowd` is evidence that two full-size animals are genuinely
touching or overlapping. An earlier version of the tracker path had the
`METRIC_FRAGMENTED_DETECTIONS` checkbox gating the `crowd` weight instead of
a fragmentation channel — the fragmentation signal was computed but never
consulted — which is not what the control's label said. That has been
corrected: the checkbox now gates `fragmentation`, and `crowd` has its own
independent control.

When a selection run comes back empty (no frame clears `min_score`),
`hydra_suite.data.al.acquisition.explain()` reports the observed maximum of
each weighted channel, so the caller can report *why* nothing was selected
instead of a bare empty result.
