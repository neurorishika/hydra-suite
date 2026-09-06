# Direct-path scoring unification — measured attribution (D7 / D9 / D8)

Task 5 of `docs/superpowers/plans/2026-09-06-direct-path-scoring-unification.md`.

**Why this file lives here and not in `docs/superpowers/`.** It is evidence tightly
coupled to the four JSON payloads and the scorer sitting beside it, and it must not
travel when the plan is `git mv`'d into `plans/done/` at merge time. A reader who
opens `before.json` should find the explanation in the same directory.

## What is in this directory

| File | What it is |
| --- | --- |
| `corpus.json` | The frozen, hand-built geometry corpus. 24 cases (8 shapes x 3 tasks). Never regenerated. |
| `generate_corpus.py` | Corpus generator (one-time). |
| `generate_before.py` | One-time legacy snapshot generator. **Do not re-run** — it would capture the NEW rules under the OLD label. |
| `before.json` | Frozen legacy scores, generated at `eb913f16`. Historical evidence, **no standing assertion**. |
| `generate_after.py` | Stage-agnostic scorer. Run against any `src/` tree; records the resolved package path and that tree's git sha into the payload. |
| `stage_a_d7_matcher.json` | Corpus scored at `5700e7ec` — D7 matcher only, no area band. |
| `stage_b_d7_d9.json` | Corpus scored at `8c7d2230` — D7 + D9 band. |
| `stage_c_d7_d9_d8.json` | Corpus scored at `546b9989` — D7 + D9 + D8 recommender. |
| `after.json` | Corpus scored at HEAD (`1efaf026`). Reproduced by a standing test. |

Each payload carries `generated_at_commit` and `hydra_suite_package_root`. All four
shas are distinct and match the intended stage — that is the check that makes this
attribution *measured* rather than four re-runs of HEAD (`PYTHONPATH` is routinely
shadowed by an editable install; `--expect-src` refuses to run when it is).

## The three rulings, and the commits they landed in

| Stage | Commit | Ruling |
| --- | --- | --- |
| before | `eb913f16` | legacy: hard `IoU >= 0.5` sole criterion; no shape prior; F1-tolerance + Pareto + fastest recommender |
| (a) | `5700e7ec` | **D7** — shared containment matcher (representative point + containment, quality ranking), hard IoU gate removed |
| (b) | `8c7d2230` | **D9** — one shared `AreaBand` fitted over the label set, pooled per evidence set |
| (c) | `546b9989` | **D8** — recall-first recommendation with quality floors; F1 retired as an optimisation target |
| after | `1efaf026` | persistence-only follow-up (`mean_quality` survives a profile reload) |

Verified: `(c)` and `after` produce **byte-identical case scores and an identical
recommender demo** — `1efaf026` changes no scoring behaviour, as its message claims.
`(b)` and `(c)` produce **identical case scores**; D8 touches only the recommender.

## Headline numbers

Corpus-wide, summing all 24 cases:

| | before | (a) +D7 | (b) +D9 | after |
| --- | --- | --- | --- | --- |
| matched | 250 | 260 | 257 | 257 |
| missed | 17 | 7 | 10 | 10 |
| extra | 20 | 10 | 13 | 13 |
| recall | 0.9363 | 0.9738 | 0.9625 | **0.9625** |
| precision | 0.9259 | 0.9630 | 0.9519 | **0.9519** |
| pooled `mean_iou` | 0.7393 | 0.7235 | 0.7305 | **0.7305** |

The three `bulk_easy_matches_*` cases contribute 240 trivially-easy matches and swamp
the signal. Excluding them, over the 21 hard cases:

| | before | (a) +D7 | (b) +D9 | after |
| --- | --- | --- | --- | --- |
| matched / missed / extra | 10 / 17 / 20 | 20 / 7 / 10 | 17 / 10 / 13 | 17 / 10 / 13 |
| recall | 0.3704 | 0.7407 | 0.6296 | **0.6296** |
| pooled `mean_iou` | 0.7454 | 0.5369 | 0.6109 | **0.6109** |

### `mean_iou` means something different after this change — read this before comparing it

`mean_iou` went **down** (0.7393 -> 0.7305 overall; 0.7454 -> 0.6109 on the hard
cases) while recall went **up**. That is not a localization regression. It is a
population change:

- **Before**, `mean_iou` was averaged over pairs that `IoU >= 0.5` had *already
  selected*. It was arithmetically incapable of falling below 0.5 on any matched
  pair. It also doubled as a gate: `MIN_LOCALIZATION = 0.5` in the recommender.
- **After**, `MIN_LOCALIZATION` is **deleted**, `mean_iou` is not a target, and it is
  averaged over pairs an *unrelated* criterion (containment + match quality) accepted.
  Pairs with IoU 0.43 are now in the population because they are correct silhouettes
  with a different extent convention — exactly the pairs D7 exists to stop discarding.

Same column name, different population. A before/after comparison of `mean_iou`
without this paragraph will read as "the change made localization worse". It did not;
it stopped throwing away the low-IoU-but-correct matches that used to be invisible.
The quality-of-match number to watch after the change is `mean_quality`
(`MIN_MEAN_QUALITY = 0.35`), which did not exist before.

## Per-case attribution

`m/mi/ex` = matched / missed / extra. `<-` marks the stage that moved the case.

| case | before | (a) D7 | (b) +D9 | after | attributed to |
| --- | --- | --- | --- | --- | --- |
| `d7_extent_convention_inflation_detect` | 0/1/1 | 1/0/0 `<-` | 1/0/0 | 1/0/0 | **D7** |
| `d7_extent_convention_inflation_obb` | 0/1/1 | 1/0/0 `<-` | 1/0/0 | 1/0/0 | **D7** |
| `d7_extent_convention_inflation_segment` | 0/1/1 | 1/0/0 `<-` | 1/0/0 | 1/0/0 | **D7** |
| `d7_nonconvex_rotated_obb` | 0/1/1 | 1/0/0 `<-` | 1/0/0 | 1/0/0 | **D7** |
| `d7_nonconvex_rotated_segment` | 0/1/1 | 1/0/0 `<-` | 1/0/0 | 1/0/0 | **D7** |
| `d7_nonconvex_rotated_detect` | 0/1/1 | 1/0/0 `<-` | 0/1/1 `<-` | 0/1/1 | **D7 then D9** (credited, then rejected) |
| `rotated_task_divergence_obb` | 0/1/1 | 1/0/0 `<-` | 1/0/0 | 1/0/0 | **D7** |
| `rotated_task_divergence_segment` | 0/1/1 | 1/0/0 `<-` | 1/0/0 | 1/0/0 | **D7** |
| `rotated_task_divergence_detect` | 1/0/0 | 1/0/0 | 1/0/0 | 1/0/0 | unmoved (counts) — see below |
| `nonconvex_centroid_outside_obb` | 0/1/1 | 1/0/0 `<-` | 0/1/1 `<-` | 0/1/1 | **D7 then D9** |
| `nonconvex_centroid_outside_segment` | 0/1/1 | 1/0/0 `<-` | 0/1/1 `<-` | 0/1/1 | **D7 then D9** |
| `nonconvex_centroid_outside_detect` | 0/1/1 | 0/1/1 | 0/1/1 | 0/1/1 | unmoved |
| `d9_blob_spans_two_animals_{detect,obb,segment}` | 1/1/0 | 1/1/0 | 1/1/0 | 1/1/0 | **unmoved — see the naming warning below** |
| `dense_cluster_label_steal_*` | 1/1/1 (+1 dup) | unchanged | unchanged | unchanged | unmoved |
| `degenerate_geometry_*` | 1/0/2 | unchanged | unchanged | unchanged | unmoved |
| `bulk_easy_matches_*` | 80/0/0 | unchanged | unchanged | unchanged | unmoved |

### D7's contribution

D7 is responsible for every count change between `before` and `(a)`: **10 cases** flip
from a simultaneous miss+extra to a match. That double-penalty was the ruling's stated
motivation — a correct silhouette with an inflated extent convention scored as *both*
a missed animal and a spurious detection in the same frame.

Concretely, `d7_extent_convention_inflation_obb`: label 400 px², prediction 680 px²
(1.7x, the appendage-tracing overshoot), IoU 0.432. Under the legacy hard gate,
0.432 < 0.5 -> 0 matched, 1 missed, 1 extra. Under the containment matcher -> 1
matched, `mean_iou` 0.432, `mean_quality` 0.566.

### D9's contribution

D9 is responsible for every count change between `(a)` and `(b)`: **3 cases** that D7
had credited are taken back, all of them the `nonconvex_centroid_outside` family
(and its `detect` sibling under `d7_nonconvex_rotated`).

`nonconvex_centroid_outside_obb` is the case that actually demonstrates D9. The label
is a 144 px²-median body; the fitted band is `min 43.2 / max 360.0 / median 144.0 /
n 1`. The horseshoe silhouette is roughly **25x** the label area, far outside the
ceiling, so the band rejects it. Stage-by-stage that case reads `0 matched` (legacy
IoU gate) -> `1 matched` (D7 containment credits it — the label's representative point
*is* inside the horseshoe) -> `0 matched` (D9 rejects it on size). This is exactly why
the attribution needs four columns and not a before/after delta: a two-column view
shows `0 -> 0` and erases both rulings' evidence simultaneously.

**Naming warning — `d9_blob_spans_two_animals_*` does NOT exercise D9.** Despite its
name it is unmoved at every stage. Its two labels are 400.0 px² each, the prediction
is 624.0 px², and the fitted band is `min 120.0 / max 1000.0 / median 400.0 / n 2`.
624 < 1000, so the band *admits* the blob; nothing about D9 fires. Do not cite this
case as D9 evidence. It could not be renamed: the corpus is frozen, `before.json` is
frozen, and `test_before_json_case_names_match_corpus_case_names` asserts the two name
sets are equal — renaming the case would require regenerating `before.json`, which the
plan forbids. The name is wrong; the number is right; this paragraph is the fix.

### `rotated_task_divergence_*` — the divergence migrated from counts to values

Before the swap, this case made `detect` differ from `obb`/`segment` in *outcome*:
detect scored 1/0/0, obb and segment scored 0/1/1. After D7 all three match, so an
assertion on match **counts** would no longer detect a regression here. The divergence
did not disappear — it moved into the values:

| task | after `mean_iou` | after `mean_quality` |
| --- | --- | --- |
| `detect` | 0.5111 | 0.7995 |
| `obb` | 0.4643 | 0.7743 |

`detect` scores higher because both polygons are reduced to their axis-aligned
bounding boxes before scoring, which is a lossy, IoU-inflating reduction on rotated
geometry. Any standing assertion about this case must therefore target **values, not
counts**. The AABB reduction itself is separately and provably live: an independent
unit test, `tests/test_direct_calibration.py:244`
(`test_rotated_prediction_is_scored_as_its_aabb_under_detect`), asserts
`detect_score.mean_iou > obb_score.mean_iou` on identical hand-built geometry.

### D8's contribution

D8 changes **no case score at all** — `(b)` and `(c)` are identical on every case.
Its entire effect is on the recommendation:

| | before / (a) / (b) | (c) / after |
| --- | --- | --- |
| `rule_id` | `balanced-pareto-fastest-v1` | `recall-first-quality-floors-v1` |
| objective | Pareto over (missed, extra, seconds), then within `F1_TOLERANCE=0.01` of best F1, then fastest | recall floor 0.90, then mean-quality floor 0.35, then `matched >= 60`, then fastest |
| eligibility gate | `MIN_MATCHED_INSTANCES=60`, `MIN_LOCALIZATION=0.5` | `MIN_RECALL=0.90`, `MIN_MEAN_QUALITY=0.35`, `MIN_MATCHED_INSTANCES=60` (`MIN_LOCALIZATION` deleted) |
| chosen | `synthetic_near_best_faster` | `synthetic_near_best_faster` |
| justification reported | F1 0.9937, 0.50 s/frame | recall 0.9875, mean quality 0.8231, F1 0.9937 *(reported, not optimised)*, 0.50 s/frame |

**Both rules pick the same candidate on this synthetic set.** That is stated plainly
because it is a limitation, not a result: the six synthetic operating points were
constructed (in Task 3) to exercise the *legacy* rule's Pareto/F1-tolerance/fastest
tie-break, and they happen not to separate the two rules. What the gate does show is
that the rule id, the eligibility criteria, and the reported justification all changed
together, and that the new rule's floors admit the candidate rather than refusing it.
It does **not** show a case where recall-first and F1-balanced disagree. Constructing
such a case would require touching the frozen corpus.

One deliberate construction change on the after side: `generate_after.py` seeds
`mean_quality` onto the synthetic scores (perturbed in lockstep with `mean_iou`, as
the before-side already did for `mean_iou`). `generate_before.py` predates the field,
so it left it at 0.0; under D8's 0.35 quality floor every candidate would have been
refused and the demo would have reported a construction artifact instead of the rule.
The legacy recommender ignores `mean_quality` entirely, so this does not change what
the legacy rule would have chosen — comparability holds.

## What this gate does NOT establish

Read this before quoting any number above.

1. **The corpus is synthetic hand-built geometry.** Rectangles, a horseshoe, a rotated
   non-convex outline, and 80 trivially-easy squares per task, written by hand to make
   specific mechanisms fire. It is **not** organic sweep output from a real model on
   real frames. It demonstrates *mechanism*; it does **not** estimate effect size on
   real data. "Recall 0.9363 -> 0.9625" is a property of this file, not of any dataset.
2. **The six `DirectCalibrationPoint` rows driving the recommender are synthetic.**
   There is no model, no confidence axis, no measured wall-clock. The timings
   (0.05–0.90 s/frame) are invented to create a Pareto frontier.
3. **No model was ever run.** No `ultralytics`, no inference, at any stage.
4. **The recommender comparison does not separate the two rules** (see above).
5. **The `bulk_easy_matches_*` cases dominate the corpus-wide totals** (240 of 267
   labelled instances). Quote the hard-case row, or quote both.
6. **Precision is not independently meaningful here.** Every corpus case has a
   hand-chosen number of predictions; "precision 0.9259 -> 0.9519" reflects the same
   10 miss+extra flips already attributed to D7, not a separate finding.
7. **`mean_iou` is not comparable across the boundary** — see the population note above.
8. **Nothing here speaks to the semantic path.** The semantic path's behaviour is
   equivalence-bound across Task 1 and is covered by
   `tests/test_semantic_calibration.py`, unmodified.

## Reproducing this

```bash
WT=<worktree>
# after.json (HEAD)
PYTHONPATH=$WT/src python $WT/tests/data/direct_calibration_golden/generate_after.py \
    --out $WT/tests/data/direct_calibration_golden/after.json \
    --stage after --expect-src $WT/src

# an intermediate stage, without ever checking out in the shared worktree
git worktree add --detach /tmp/stage_a 5700e7ec
PYTHONPATH=/tmp/stage_a/src python $WT/tests/data/direct_calibration_golden/generate_after.py \
    --corpus $WT/tests/data/direct_calibration_golden/corpus.json \
    --out /tmp/stage_a.json --stage a_d7_matcher \
    --no-area-band --expect-src /tmp/stage_a/src
git worktree remove --force /tmp/stage_a && git worktree prune
```

`--no-area-band` is what makes stage (a) faithful to `5700e7ec`: that commit adds the
`area_band` parameter with default `None` and no caller passes one, so there is zero
band behaviour there. The fitter and its call site arrive at `8c7d2230`.

The standing check that `after.json` still describes live code is
`tests/test_direct_calibration_after_gate.py`. `before.json` deliberately has **no**
standing assertion against live code — it is frozen historical evidence.
