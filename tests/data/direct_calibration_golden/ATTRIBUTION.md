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
| `stage_d_pooled_quality.json` | Corpus scored at `b796931d` — quality pooled per matched pair (this stage). |
| `after.json` | Corpus scored at HEAD. Reproduced by a standing test. |

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
| (d) | `b796931d` | **D8 aggregation fix** — `mean_quality` pooled per matched pair; a zero-match frame contributes no sample |
| after | `b796931d` | same tree as (d) |

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
because it is a limitation, not a result: the five synthetic operating points were
constructed (in Task 3) to exercise the *legacy* rule's Pareto/F1-tolerance/fastest
tie-break, and they happen not to separate the two rules. What the gate does show is
that the rule id, the eligibility criteria, and the reported justification all changed
together, and that the new rule's floors admit the candidate rather than refusing it.
It does **not** show a case where recall-first and F1-balanced disagree on THIS
corpus. Constructing such a case here would require touching the frozen corpus, but
the disagreement itself is separately and provably live: an independent standing
test, `tests/test_direct_calibration.py:126`
(`test_f1_no_longer_influences_selection`), asserts the two rules pick different
points at equal recall -- F1 0.741 vs 0.952, with recall-first choosing the cheaper,
lower-F1 point. So the limitation is that this particular synthetic gate does not
exercise that disagreement, not that the disagreement is untested.

One deliberate construction change on the after side: `generate_after.py` seeds
`mean_quality` onto the synthetic scores (perturbed in lockstep with `mean_iou`, as
the before-side already did for `mean_iou`). `generate_before.py` predates the field,
so it left it at 0.0; under D8's 0.35 quality floor every candidate would have been
refused and the demo would have reported a construction artifact instead of the rule.
The legacy recommender ignores `mean_quality` entirely, so this does not change what
the legacy rule would have chosen — comparability holds.

### Stage (d) — the D8 quality floor now gates the quantity it was calibrated on

`MIN_MEAN_QUALITY = 0.35` was documented as "taken from the semantic path", and it was
— but until this stage it gated a **differently aggregated statistic**. The semantic
path (`core/inference/semantic/calibration.py`) accumulates ONE `qualities` list over
the whole evidence set, i.e. it pools **per matched pair**. `score_frames` averaged
**per-frame means**, and `match_frame` reports `mean_quality = 0.0` for a frame that
matched nothing, so every zero-match frame injected a hard zero.

Two consequences, both arithmetic rather than geometric:

- a frame with 1 match weighed as much as a frame with 16;
- a configuration that missed one whole frame out of twenty was charged a 0.0 quality
  sample for it *on top of* the recall it already lost. Worked example (not corpus
  data): 20 frames x 2 labels with one all-missed frame gives recall 0.95 — clearing
  the 0.90 floor — while `mean_quality` is dragged 0.694 -> 0.660. At a floor of 0.35
  the same mechanism flips a good configuration into the **"Mistargeted"** refusal,
  which is a positive claim *about detection geometry* produced by aggregation.

Absence of evidence is not evidence of bad geometry, so a zero-match frame now
contributes **no sample**. It still costs recall through `missed`, which is where that
failure belongs. An entirely empty sample reads a measured `0.0` — never `None`, which
is reserved for "this profile predates the metric" — so a configuration that matches
nothing anywhere is refused at the **recall** floor, and an evidence-poor one at the
**matched-instances** floor (`MIN_MATCHED_INSTANCES = 60`), never vacuously admitted.

`MIN_RECALL`, `MIN_MATCHED_INSTANCES` and `MIN_MEAN_QUALITY` are all **unchanged**.
`f1` is still computed and still reported, and still appears nowhere in the objective.

The per-frame averaging is **pre-existing** — it predates the D7/D9/D8 branch. That
branch's contribution was making it *load-bearing*, by putting a semantically
calibrated floor on top of it.

#### What moved on this corpus: essentially nothing, and that is the honest result

| | (c)/(after, pre-fix) | (d) pooled |
| --- | --- | --- |
| every case's matched / missed / extra / duplicate | unchanged | unchanged |
| every case's `mean_iou` | unchanged | unchanged |
| `mean_quality`, 23 of 24 cases | unchanged | unchanged |
| `bulk_easy_matches_detect` `mean_quality` | 0.834771562840346 | **0.8347715628403456** |
| `recommend_balanced_demo` (chosen, recall, quality, F1, explanation) | unchanged | unchanged |

The single moved number is a **4.4e-16 floating-point reassociation** — one ULP — from
summing 80 samples once instead of averaging five 16-sample frame means. Nothing else
in the payload differs.

That is a property of the frozen corpus, not evidence that the change is inert. Every
non-`bulk` case has exactly **one frame**, where the two aggregations are identically
equal, and the three `bulk_easy_matches_*` cases have five frames with an identical
16 matches each, where the weighted and unweighted means coincide. **The corpus cannot
exercise this defect**: it contains no case with heterogeneous per-frame match counts
and no case with a zero-match frame alongside a matched one, and a case cannot be added
(`test_before_json_case_names_match_corpus_case_names` freezes the name set against the
frozen `before.json`).

The behavioural evidence therefore lives in fail-first unit tests rather than in this
corpus:

- `tests/test_direct_calibration.py::test_mean_quality_pools_per_matched_pair_not_per_frame`
  — two frames with 2 and 1 matches; pooled 0.7165 vs the old per-frame 0.3583. FAILED
  before the fix.
- `tests/test_direct_calibration.py::test_a_zero_match_frame_contributes_no_quality_sample`
  — a matched frame plus an all-missed one; the pooled mean equals the matched frame's
  quality instead of half of it, while `missed` still records the miss. FAILED before
  the fix.
- `tests/test_direct_calibration.py::test_an_empty_quality_sample_cannot_pass_the_recommender`
  — an empty sample is refused at the recall floor (not reported as "Mistargeted"), and
  a well-targeted but evidence-poor point is refused at the matched-instances floor.
  This one **passed before the fix too**; it is a standing guard against the empty-sample
  hole the fix could have opened, not evidence of the fix.

`after.json` was regenerated for this stage. That is a deliberate, reported act: the
only difference is the one ULP above. **`before.json` is untouched.**

## What this gate does NOT establish

Read this before quoting any number above.

1. **The corpus is synthetic hand-built geometry.** Rectangles, a horseshoe, a rotated
   non-convex outline, and 80 trivially-easy squares per task, written by hand to make
   specific mechanisms fire. It is **not** organic sweep output from a real model on
   real frames. It demonstrates *mechanism*; it does **not** estimate effect size on
   real data. "Recall 0.9363 -> 0.9625" is a property of this file, not of any dataset.
2. **The five `DirectCalibrationPoint` rows driving the recommender are synthetic.**
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
   It is also still aggregated as a mean of per-frame means after stage (d): pooling was
   applied to `mean_quality` only, because that is the number a floor gates, and the
   semantic path reports a *median* IoU rather than a pooled mean.
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
