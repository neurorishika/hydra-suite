# DetectKit: Semantic Escalation (SAM3)

> This page also documents SAM3 LoRA **finetuning** (below), which produces
> the checkpoints this page's escalation dialog can select from. Escalation
> itself needs none of the packages the finetuning section names — only the
> `sam3` extra.

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

## Finetuning your own SAM3 checkpoint

The stock `sam3` checkpoint is a general-purpose model. If your animals or
imaging conditions are far from what it saw in pretraining, DetectKit can
finetune a SAM3 LoRA adapter on your own polygon labels and publish a
merged checkpoint that then shows up as another option in this page's
escalation dialog.

This is a **DetectKit training role** ("Semantic" mode, in the training
dialog's mode selector), not part of the escalation workflow above. It
lives in its own "SAM3" tab, which only appears once Semantic mode is
selected — the mode also force-selects the `segment` task, since a SAM3
checkpoint is only ever consumed as a segmentation model.

### Installing the training dependencies

Finetuning needs a larger, CUDA-only dependency stack. **Escalation
(inference) needs none of it** — that is the entire point of publishing a
merged checkpoint rather than shipping the training code path to every
machine that just wants to run segmentation.

```bash
pip install 'hydra-suite[sam3,sam3-train]'
pip install git+https://github.com/ultralytics/CLIP.git
pip install git+https://github.com/facebookresearch/sam3.git
```

The `sam3` package itself, like `clip` above, is a git dependency and
cannot be listed in a published PyPI extra (the same PEP 508 metadata
constraint documented for `clip` earlier on this page), so it is always a
manual install. The `sam3-train` extra covers everything else training
needs beyond the `sam3` extra: `torchmetrics`, `scipy`, `einops`,
`decord`, `iopath`, and the `numpy<2` pin that `sam3` itself requires.

`iopath` is listed even though `sam3` declares it, because a *source
checkout* of `sam3` placed on `PYTHONPATH` -- rather than a `pip install`
of it -- never resolves that dependency, and the failure surfaces only
once training reaches `sam3.train.loss`, as a bare `ModuleNotFoundError`.

### CUDA only, and not a small GPU

SAM3 LoRA training runs on CUDA exclusively — there is no CPU or MPS path.
Preflight checks free VRAM before touching a weight and refuses the run
below **32 GB free**, with a further warning below 40 GB. That floor sits
above the ~29 GB the training spike actually measured at batch size 1: the
margin exists so the same GPU load that OOMs a real run also fails
preflight, in milliseconds, rather than after an hour of GPU time. Point
this role at the CUDA box, not a laptop.

### The label-quality acknowledgement

Training uses **every label** on the source you point it at — there is no
provenance filter that separates labels you drew by hand from labels a
prior semantic-escalation run produced and you accepted. That is a
deliberate simplification, not an oversight: provenance does not survive
review in a form training could reliably filter on. Because of that, the
SAM3 tab has a checkbox — unchecked by default — asking you to confirm the
source's labels are good enough to train on, including any SAM3 output
already folded in. Preflight refuses to start the run without it.

### Dataset and tiling

Training builds a COCO instance-segmentation dataset from the source's
polygon labels, tiled with the same shared tile-geometry helpers
(`utils/slice_geometry`) that sliced training and sliced inference already
use elsewhere in DetectKit — so a training tile and an inference tile are
built the same way. Polygon labels are the only geometry level training can
use; AABB/OBB-only sources need geometry escalation first.

### Defaults

The panel's defaults come from a training spike, not from taste — see the
comments in `Sam3LoraParams` if you want the exact rationale per value:

| Setting | Default |
|---|---|
| LoRA rank / alpha | 16 / 32 |
| Learning rate | 5e-5 |
| Epochs | 10 |
| Batch size / grad-accum | 1 / 8 (effective batch 8) |
| Mixed precision | bf16 |
| Input size | 1008 (SAM3's architecture size; not configurable) |

On a small internal check — 3 leave-one-frame-out folds over 3 labelled
frames on one dataset — AP75 rose from roughly 0.000 (stock checkpoint) to
roughly 0.624 (finetuned). That is a small-sample result on a single
dataset, not a general performance claim; treat it as evidence the
approach works at all, not as a number to expect on your own data.

### Checkpoint selection is always "last", never "best"

On that same spike, validation loss was **anti-correlated** with held-out
AP: the fold with the worst validation loss had the best held-out AP75.
Training therefore always keeps the **last** epoch's checkpoint. It still
computes and reports validation-loss statistics during the run, because
they are informative to watch, but nothing in the pipeline uses them to
pick which weights to keep. Do not "fix" this by wiring in best-checkpoint
selection later without re-running the measurement that ruled it out.

### Publishing

A finished run publishes a **merged, full checkpoint** — the LoRA adapter
folded into the base weights, not the adapter alone — to
`get_models_dir() / "sam3_finetuned"`, alongside a JSON sidecar recording
the run's parameters and dataset fingerprint, and registers it in the
model registry. From then on, the escalation dialog's model selector
offers it next to the stock `sam3` checkpoint.

A load guard (`assert_checkpoint_loaded`) checks that a selected checkpoint
actually loaded before it is used for inference. This exists because
ultralytics loads state dicts with `load_state_dict(strict=False)` and
discards the result it returns — without an explicit check, a checkpoint
that fails to match the model's shape would silently fall back to serving
stock weights instead of raising.
