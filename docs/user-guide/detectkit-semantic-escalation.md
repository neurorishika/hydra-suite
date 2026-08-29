# DetectKit: Semantic Escalation (SAM3)

DetectKit's **Escalation** group in the Tools panel offers two distinct
operations. Both live under `detectkit`, in the same "Escalation" section.

## Two escalations, two different jobs

- **Geometry escalation (SAM2)** converts boxes you already have into masks.
  It refines the geometry of an existing label — it cannot add an animal
  that was never labelled.
- **Semantic escalation (SAM3)** is the subject of this page. It takes a
  text prompt (e.g. `ant`) and finds instances of that concept across an
  image source, including animals that are missing from the current
  labels entirely. It needs no existing labels to start.

Use geometry escalation to upgrade what you have; use semantic escalation
to find what you don't.

## The prompt

The prompt is a short noun phrase (the default is `ant`). It is yours to
vary, but **wording matters far less than tile size**. If results look
wrong, try the "Preview one tile" button with a different prompt before
assuming the prompt is the problem — tiling (below) is usually the bigger
lever.

## Calibration: fitting the run to your data, not the other way around

Before committing to a full run, calibrate against your own labelled
frames. Calibration needs a labelled frame at *any* geometry level (AABB,
OBB, or polygon) — it only needs instance counts to compare against SAM3's
output, not masks — so you don't need polygon labels to calibrate.

Calibration fits **two** parameters against your labelled frames:

1. **Tile fraction** — whether (and how finely) the frame is tiled before
   inference, and
2. **Confidence** — the score threshold used to keep or drop a detected
   instance.

**The tile-fraction default shown in the dialog (`0.05`) is an unvalidated
starting guess taken from a single measured configuration on one dataset.
It is not a tuned value, and it should not be treated as one.** It exists
only to prefill the dialog if you skip calibration. Calibration is how you
actually fit the tile fraction (and the confidence) to your own images.

This asymmetry matters operationally: **changing the tile fraction requires
a full re-run** (tile geometry is baked into what gets inferred), while
**changing the confidence does not** — a staged run keeps every candidate
detection in a cache, and re-thresholding it to a different confidence is
free (no inference, just re-filtering the cache). Calibrate confidence
liberally; calibrate tile fraction only when you actually plan to re-run.

### The exhaustive-labelling checkbox

Calibration measures how many labelled animals SAM3 catches (recall) and
how many extra polygons it produces that don't match a label. That second
number is only meaningful if your labelled frames mark *every* animal in
the frame. If some real animals are unlabelled, SAM3 correctly finding them
looks like a false positive and biases the recommended threshold upward
(toward missing more real animals). The checkbox — "My labelled frames are
exhaustively labelled" — is a required confirmation before calibration
runs, precisely because this bias is easy to introduce by accident.

### What the recommendation optimises

The recommendation is **not** the F1-optimal point. It takes the cheapest
tiling (fewest tiles per frame) that clears a recall floor, then breaks
ties toward higher confidence (fewer polygons to delete). It deliberately
does not maximise F1, because the two kinds of mistake are not
symmetric in cost: a spurious polygon is deleted with one click during
review, but a missed animal must be found by eye, and may not be. When
there's a choice, recall is worth much more than precision here.

If calibration can't reach the recall floor on your frames, or has too few
matched instances to trust, it will say so and refuse to recommend a
point — that's not a bug, it's calibration declining to guess.

## Running the escalation

Semantic escalation is a batch job, not an interactive one — budget it in
hours, not minutes, for anything beyond a handful of frames. This is exactly
why it is cancellable and resumable (below): a run you can't finish in one
sitting is still a run you can make progress on. No fixed per-frame time is
quoted here, because the real rate depends on your GPU, tile count, and
image size — **calibration measures the actual per-frame time on your own
hardware and your own data**, and that measured number, not any figure in
this page, is the one to plan a run from. For anything beyond a handful of
frames, **point the run at the CUDA box** rather than running it on a
laptop.

The run is:

- **Cancellable** — it stops at the next tile or frame boundary, not
  instantly, but it does stop cleanly.
- **Resumable** — a cancelled or interrupted run picks back up where it
  left off rather than restarting from scratch, as long as the run's
  parameters (prompt, model variant, tile size, and related settings)
  haven't changed. Changing a parameter that affects what gets inferred
  starts a fresh run.
- **Re-thresholdable for free** — as noted above, once a run has produced
  a candidate cache, you can change the confidence threshold and get new
  results instantly, with no new inference.

## Accepting results: a new sibling source, never in place

When you review and accept a staged semantic escalation, DetectKit creates
a **brand-new sibling source** alongside the one you escalated. **It never
touches the original source's labels.** This is deliberate and is the
whole reason the result lands in a new source rather than being merged in:
SAM3's masks are a different instance set, at a different geometry
convention, from your existing labels (see below) — merging them in place
would risk silently degrading a dataset you've already curated. The new
source starts unreviewed; you keep, merge, or delete it with the same
tools you'd use for any other source, on your own schedule.

## A convention gap that review has to settle

SAM3's masks trace an animal's full visible extent — legs, antennae, and
all — while tracking-derived labels typically bound just the body core.
These are two different, both-legitimate conventions for "the boundary of
an animal," and semantic escalation does not attempt to reconcile them
automatically. This is precisely why acceptance produces a reviewable
sibling source rather than a silent merge: review is where you decide,
for your dataset, which convention should stand — or whether the two need
to be visually reconciled before you train on the combined set.
