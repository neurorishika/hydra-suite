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

## Installing SAM3

Semantic escalation is an optional extra. Two steps, because one of SAM3's
dependencies is not on PyPI:

```bash
pip install 'hydra-suite[sam3]'
pip install git+https://github.com/ultralytics/CLIP.git
```

The second line installs OpenAI's `clip` package. It cannot be listed as a
dependency of the `sam3` extra — PyPI rejects direct URL references in
uploaded package metadata — so it has to be installed by hand. Without it,
DetectKit disables the semantic escalation button and the tooltip names the
missing package.

### The model checkpoint (3.45 GB, downloaded once)

The `sam3` checkpoint is about **3.45 GB** and is fetched from the public
`facebook/sam3` Hugging Face repository the first time you run. DetectKit
never downloads it behind your back: if it is not already on the machine,
the escalation dialog shows a warning up front and asks for confirmation
before the run (or a random-image check, or calibration) starts. The button
stays enabled in that state — the download offer is inside the dialog, so
disabling the button would put it out of reach.

## The prompt

The prompt is a short noun phrase (the default is `ant`). It is yours to
vary, but **wording matters far less than tile size**. If results look
wrong, try the "Test random image" button with a different prompt before
assuming the prompt is the problem — tiling (below) is usually the bigger
lever.

### "Test random image"

The check chooses one random image from the selected sources and processes
the **complete image with the current run settings**. If tiling is enabled,
SAM3 runs every tile and merges the results exactly as it would during
escalation; "complete image" does not mean tiling is bypassed.

The result opens as a zoomable, pannable overlay. Predictions are blue and
dashed; existing ground-truth polygons are green when the sampled image is
labelled. Nothing is written back to the source. DetectKit also reports the
time the complete image actually took on this machine and extrapolates that
measurement across the selected images (and the whole project when those
counts differ). It is an estimate from one random image, so content and
hardware-load differences can move the final run time. The per-image timing
excludes one-time model loading; no timing figure in DetectKit or in this
page is hardcoded.

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

### Reference body size, and where it comes from

Tiling needs to know roughly how large one animal is, in pixels: the tile
edge is `reference body size / tile fraction`. DetectKit resolves it from
the first of these that yields a value:

1. the project's sliced-training reference body size,
2. the **median longest side of the labels you already have** in the
   selected sources, then
3. **you** — the dialog's "Reference body size (px)" field is editable, and
   shows which of the three it was prefilled from.

If none of them resolves, tiling switches **off**, which is the worst
configuration measured for small animals. The dialog says so explicitly
rather than proceeding quietly.

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
  results instantly, with no new inference. Candidates are cached down to
  the bottom of the calibration grid, not down to the confidence you ran at,
  so re-thresholding *downward* is complete rather than truncated. (A cache
  staged by an older version records the higher floor it was collected at,
  and DetectKit refuses to re-threshold below it rather than quietly
  returning a short list.)
- **Reported honestly when cancelled** — a cancelled run says so, and says
  how many frames it got through, instead of reporting a partial result as
  a clean success.

## Reviewing results: frame by frame, into the source you ran on

A staged semantic escalation is reviewed **frame by frame**, and accepting
a frame writes **into the source you escalated** — SAM3 no longer creates a
sibling source. This is the same `StagedReview` flow used for geometry
escalation (SAM2) and for staged dataset predictions (below); all three
producers share one review path.

While a source has a staged review, a **review bar** appears above the
canvas with four operations, applied to the frame on screen:

- **Replace** — the staged labels replace this frame's.
- **Add New** — this frame's existing labels are kept; only the staged
  instances that don't overlap one already there are appended.
- **Reject** — the staged labels for this frame are discarded.
- **Accept All / Reject All** — the same, over every frame not yet
  decided.

Plus **Next Undecided**, to jump to the next frame with an outstanding
decision, and a `23/140 decided` counter showing review progress across
the source.

Accepts apply **immediately** to the source's real labels — the result
appears on the ground-truth layer as you work, rather than accumulating
into a pending set you review later. The staged (magenta) proposal
disappears from a frame once it is decided.

**Revert Review** restores the source's labels, geometry level, and class
list to their state before the review started — but only while the review
is open. Finishing the review (every frame decided) deletes the staging
directory and the snapshot it depends on, so revert is no longer available
after that point.

### A convention gap review has to settle

SAM3's masks trace an animal's full visible extent — legs, antennae, and
all — while tracking-derived labels typically bound just the body core.
These are two different, both-legitimate conventions for "the boundary of
an animal," and semantic escalation does not attempt to reconcile them
automatically. Reviewing frame by frame, with **Replace** vs. **Add New**
as an explicit per-frame choice, is where you decide which convention
should stand for a given frame, or whether the two conventions need to be
visually reconciled before you train on the result.

### Accepting polygons into a box source promotes it

SAM3 stages polygon-level masks. If the source you escalated is still at
OBB (or AABB), accepting a staged polygon **promotes the source to
polygon**: its existing box labels are lifted to 4-point polygons (no
points move — an OBB quad is already a valid polygon), and the rest of the
review proceeds at the new level. Promotion is a one-way, source-level
change that happens on the first promoting accept, not per frame.

## Staging dataset predictions for review

Model inference (Batch Predict / dataset predictions) can also be staged
for the same review flow, via **Stage Predictions for Review** in the
Tools panel. Staging is explicit: merely running inference to preview
predictions on the canvas does not create anything reviewable, and only
the predictions currently visible at the confidence slider are staged —
raise or lower the slider to the set you want reviewed before staging.
