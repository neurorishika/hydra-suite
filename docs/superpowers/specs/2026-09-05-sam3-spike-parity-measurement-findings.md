# SAM3 spike-parity — measurement findings (the premise is refuted)

**Status:** final. This is the durable, committed record of the measurement
phase of the `codex/sam3-spike-parity` branch. It supersedes the motivating
evidence of
`docs/superpowers/plans/2026-09-05-sam3-spike-parity-finetuning.md`.

**Date:** 2026-09-05 · **Box:** `courtship` (RTX 4090) · **Env:** `hydra-cuda`
**Branch:** `codex/sam3-spike-parity` (`c1dcd74a`, `80685249`, `82327fd6`)
**Artifact:** `tools/sam3_parity/baseline.json` (pre-registered; never re-run)

---

## 1. The one-paragraph verdict

The programme was built on the belief that the research spike's 314-module LoRA
adapter surface produced a materially better detector than production's
206-module surface — specifically **+1.2–2.0 false positives per frame** in our
favour-adjusted comparison. On a genuine 16-frame held-out split the
pre-registered paired comparison is **null**, and on average precision **our
checkpoint is better**. Separately, the shared recall plateau the plan set out
to explain turned out to be a **defect in the scoring harness**, not in either
model. No adapter-surface change can move a ceiling imposed by the scorer.
**Do not re-run this programme on the original premise.**

## 2. The pre-registered result (null)

Criterion, fixed before the run: *a difference is real only if the paired 95 %
CI excludes zero.* `a` = ours, `b` = spike.

| Granularity | n | mean diff (ours − spike, extras) | 95 % CI | sign p | Verdict |
|---|---|---|---|---|---|
| **source frame (primary)** | 16 | **+0.4375** | **[−0.4375, +1.4375]** | 0.34 | **null — CI includes zero** |
| tile (secondary) | 576 | +0.0122 | [−0.0035, +0.0295] | 0.21 | null |

**Power caveat — read this before citing the null.** The primary CI's upper
bound (+1.4375) OVERLAPS the original +1.2–2.0 claim. A null here means the
comparison FAILS TO DEMONSTRATE the effect at n=16 frames; it does NOT exclude
the lower half of the claimed band. Absence of evidence, not evidence of
absence. The programme's premise is called refuted on four INDEPENDENT legs —
this null, the AP inversion (ours 0.530 > spike 0.472), the scoring-harness
defect (§ below), and the ~19× training-data confound — and would not be on
the null alone.

**AP (no threshold matching): ours 0.530, spike 0.472 — ours is better.**

`extras_per_frame_at_target_recall` is **null for both**: 0.9 recall is not
reached anywhere on the confidence grid under the shipped matcher. That
non-result is itself a symptom of §4.

## 3. Two confounds that make this not an adapter-surface test at all

**(a) Training contamination.** One of the 16 held-out frames, `f008975`, is in
the **spike's own training set** (`fold_all` train split =
`{f008078, f008161, f008975}`). It was **not** dropped — a post-hoc frame-set
change is itself a protocol deviation — so the pre-registered 16-frame number
stands and the sensitivity is reported instead:

| Frame set | n | mean diff | 95 % CI | Verdict |
|---|---|---|---|---|
| all 16 (pre-registered) | 16 | +0.4375 | [−0.4375, +1.4375] | null |
| excluding `f008975` | 15 | +0.0667 | [−0.600, +0.667] | null |

`f008975` is the single largest per-frame divergence (**+6**; next largest −3),
i.e. the contaminated frame supplies most of the point estimate. **The verdict
does not flip.**

**(b) ~19× data confound.** The spike's `fold_all` dataset is **3 source frames
/ 108 tiles / 177 instances**, and its train and valid splits are *the same 108
files* — **the spike had no held-out set at all**. Ours is **62 frames / 2 232
tiles / 3 286 instances**. Any difference between these checkpoints is
dominated by training data, not by adapter surface. `baseline.json` therefore
answers "which checkpoint deploys better on our data" (answer: no significant
difference in extras; ours is ahead on AP), **not** "did our adapter surface
reproduce theirs".

**(c) The measured "ours" is not the surface this branch built.** The measured
checkpoint is `20260904-143950_semantic_sam3_ca1bd031.pt`, trained on the **old
206-module surface**. The **312-module** surface this branch created (206 + the
clone-MHA pass + the six geometry Linears; scoring head defaults OFF) has
**zero training evidence**. Plan RUN A and RUN B were never executed.

## 4. The recall plateau is a scoring-harness defect in production code

`src/hydra_suite/core/inference/semantic/calibration.py`

`match_one_to_one` admits a pair only if the prediction's centroid lies inside
the label *or* the label's centroid inside the prediction. Its `_centroid` is
the **mean of polygon vertices**, not the area centroid. For a curved,
elongated, densely-sampled ant outline the vertex mean routinely falls
**outside the outline**. Measured directly on the held-out labels:

> **128 of 805 (15.9 %) ground-truth ant polygons do not contain their own
> vertex-mean.**

That 15.9 % is the ceiling. The containment gate vetoes near-perfect masks —
one concrete verified case: **IoU 0.904**, area 1 226 px² vs the label's
1 217 px², claimed by no other label, scored a **miss**.

Same predictions, three matchers:

| conf | | recall shipped | recall IoU ≥ 0.5 | recall area-centroid | extras/tile shipped | extras/tile IoU |
|---|---|---|---|---|---|---|
| 0.20 | ours | 0.867 | **0.962** | 0.929 | 0.20 | 0.07 |
| 0.20 | spike | 0.846 | **0.935** | 0.911 | 0.20 | 0.08 |
| 0.50 | ours | 0.858 | **0.957** | 0.921 | 0.19 | 0.05 |
| 0.50 | spike | 0.836 | **0.935** | 0.901 | 0.18 | 0.04 |

**Both checkpoints move together** (+9.4 and +8.9 recall points) — what a
shared harness defect predicts and a model difference would not.

Consequences:

* Misses were checked against the obvious alternatives and cleared: not small
  (missed median area 1 194 px² vs all-GT median 1 153 px²), not area-gate
  rejected (0 outside the fitted band), not seam-adjacent (median centroid–edge
  distance 201 px), only mildly cluster-enriched (62 % vs a 53 % base rate).
* **~65 % of "extras" are already-labelled ants whose match was vetoed** —
  neither clutter nor unlabelled ants, but bookkeeping. Restated per frame at
  confidence 0.2 the real operating point is roughly **0.96 recall and
  ~2.5 extras/frame**, not "0.72 recall and 5–8 extras/frame".
* The same `match_one_to_one` gates production `calibrate()`, so **every
  calibration number the product has shown for a curved animal is biased
  downward**, and `MIN_RECALL = 0.90` (calibration.py:61) may be unreachable
  for **scoring** rather than detection reasons.

### Deliberately NOT fixed on this branch

Re-running a pre-registered comparison under a changed metric after seeing the
result is a post-hoc protocol change. The matcher fix belongs on its own branch
with its own before/after. `baseline.json` was likewise deliberately **not**
re-run under a corrected matcher.

**Recommended fix (for that branch):** replace the vertex mean in
`calibration._centroid` with an inside-guaranteed representative point (area
centroid via `cv2.moments`, falling back to a `pointPolygonTest`-verified
interior point), or drop containment in favour of an IoU precondition. It is a
behaviour change to shipped GUI calibration and needs its own tests and gates.

## 5. Limitations — what was not measured

* The "absent from the candidate set entirely" axis is **not** cleanly
  answered: the statistic that would have answered it used the same vertex-mean
  containment test under investigation, so it is discarded. The residual ~3.8 %
  of GT still missed under IoU ≥ 0.5 (~31 instances) was not characterised.
* The residual extras (~0.07/tile) were **not adjudicated by eye** — plan
  Task 0 Step 5's literal ask remains open.
* Tiles overlap 25 %, so the same ant is scored in several tiles; this
  correlates tiles and is why frame-level pairing is the primary statistic.
  Only 169 of 576 tiles carry labels; the other 407 are label-empty, where
  every detection scores as an extra by construction.
* No CUDA-vs-MPS cross-check: SAM3 inference is CUDA-only here.

## 6. Defects fixed en route

1. `np.trapezoid` is numpy ≥ 2.0 only — `average_precision` died on every
   numpy 1.x host (the dev box and the `hydra-sam3` sidecar are both 1.26.4).
   Fixed in `c1dcd74a`.
2. `tools/sam3_parity/README.md` named the wrong env: inference goes through
   `ultralytics.models.sam.SAM3SemanticPredictor`, which the `hydra-sam3`
   training sidecar does not have. `hydra-cuda` is correct.
3. The tool paired per **tile**, not per frame, while the criterion is per
   frame. `80685249` adds the source-frame rollup as the primary statistic.

## 7. What should happen next

1. **Fix the matcher on its own branch** (§4). It outranks every remaining task
   here: ~9 recall points and two thirds of the "extras" are recovered by
   changing nothing but the scorer.
2. **Do not resume Tasks premised on closing the spike gap.** There is no
   measured gap.
3. If the 312-module surface is ever to be judged, it needs its own training
   run and its own comparison against a 206-module run on **identical data** —
   the manifest now records `adapted_modules` and
   `adapter_trainable_params` (publish_worker.py) precisely so such a run is
   attributable to a surface.
