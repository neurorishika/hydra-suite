# DetectKit: Semantic Escalation (SAM3)

> This page also documents SAM3 LoRA **finetuning** (below), which produces
> the checkpoints this page's escalation dialog can select from. Escalation
> runs on every device and needs none of the finetuning packages. Training
> is CUDA-only and lives in an optional sidecar env. See
> [SAM2 and SAM3: install and run](sam-install-and-run.md).

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

Nothing extra to install: `python install.py` installs SAM3 inference on
every tier (CPU, Apple MPS, CUDA). That includes the `sam` extra and the
pinned ultralytics CLIP fork, which is git-only and so cannot be a package
dependency. If either is missing, DetectKit disables the semantic escalation
button and the tooltip names the missing piece. See
[SAM2 and SAM3: install and run](sam-install-and-run.md) for the device
matrix, the `detectkit escalate sam3` command line, and troubleshooting.

### The model checkpoint (3.45 GB, licence-gated, downloaded once)

The stock `sam3` checkpoint is about **3.45 GB** and is fetched from the
**licence-gated** `facebook/sam3` Hugging Face repository the first time you
run. Before that first run, accept the licence at
<https://huggingface.co/facebook/sam3> and run `hf auth login` (or set
`HF_TOKEN`) once on the machine. DetectKit never downloads it behind your
back: if it is not already on the machine, the escalation dialog shows a
warning up front and asks for confirmation before the run (or a random-image
check, or calibration) starts. The button stays enabled in that state,
because the download offer is inside the dialog and disabling the button
would put it out of reach.

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

1. **Object scale** (the tile fraction, `--tile-fraction` on the command
   line) — whether (and how finely) the frame is tiled before inference.
   Tile size = body size ÷ object scale; `0` shows as "full frame (no
   tiling)", and
2. **Confidence** — the score threshold used to keep or drop a detected
   instance.

**The object-scale default shown in the dialog (`0.05`) is an unvalidated
starting guess taken from a single measured configuration on one dataset.
It is not a tuned value, and it should not be treated as one.** It exists
only to prefill the dialog if you skip calibration. Calibration is how you
actually fit the object scale (and the confidence) to your own images.

This asymmetry matters operationally: **changing the object scale requires
a full re-run** (tile geometry is baked into what gets inferred), while
**changing the confidence does not** — a staged run keeps every candidate
detection in a cache, and re-thresholding it to a different confidence is
free (no inference, just re-filtering the cache). Calibrate confidence
liberally; calibrate the object scale only when you actually plan to re-run.

### Body size, and where it comes from

Tiling needs to know roughly how large one animal is, in pixels: the tile
edge is `body size / object scale`, shown read-only in the **Resolved tile**
row. DetectKit resolves the body size from the first of these that yields a
value:

1. the project's sliced-training reference body size (badge `project`),
2. the **median longest side of the labels you already have** in the
   selected sources (badge `dataset`), then
3. **you**: an unknown body size ("unknown (tiling off)") stays editable.
   A derived one is read-only until you check **Override**.

A finetuned model's stamped body size (badge `stamped`) is used when the
model was trained at one and the project chain is empty. If no body size
resolves, tiling switches **off**, which is the worst configuration measured
for small animals. The dialog says so explicitly rather than proceeding
quietly.

The dialog opens with the settings you last accepted for that model
variant, else its saved calibration, else the model's stamped scale, else
the `0.05` starting guess at the body size above. The headless
`detectkit escalate sam3` command resolves its tiling the same way, so a
run with no tiling flags stages what the dialog would have opened with.
`--tile-fraction` and `--reference-body-px` override it (see [SAM2 and SAM3:
install and run](sam-install-and-run.md#running-from-the-command-line)).
The tiling rows are the shared SAHI widget. [SAHI Settings](sahi-settings.md)
describes every row, badge, and the whole-animal overlap hint, including
why a deliberate overlap of 0.5 is never nudged down.

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

### Three environments, not one

Meta's `sam3` pins `numpy<2`, and DetectKit's own runtime (`hydra-mps` /
`hydra-cuda`) needs numpy 2.x — those two dependency sets cannot coexist in
one Python environment. Rather than fight that, SAM3 training runs as a
**subprocess in a dedicated sidecar conda environment**, the same pattern
already used for the SLEAP integration:

- **`hydra-mps` / `hydra-cuda`** — the environment DetectKit itself runs
  in. It never installs `sam3` or any of its training dependencies, and its
  numpy version is untouched by anything below.
- **`hydra-sam3`** (the sidecar) — a separate conda env that owns `sam3`,
  its `numpy<2` pin, and the training loop. The GUI launches training in
  this env as a child process (`conda run -n hydra-sam3 ...`) and streams
  its progress back; it never imports `sam3` itself.
- **Neither** — escalation (inference with a published checkpoint) needs
  none of this. That is the whole point of publishing a merged checkpoint
  rather than shipping the training code path to every machine that just
  wants to run segmentation.

The SAM3 training tab has an env row (default `hydra-sam3`) where you name
which conda env to launch training in, and a "Check" button that probes
whether that env can actually import what training needs — it reports the
child's real failure text (e.g. a missing package name) rather than a
generic "unavailable", so you know exactly what to fix. The probe spawns a
subprocess and is not run automatically on every keystroke; it runs once
when the tab is first shown and whenever you click "Check".

#### Building the `hydra-sam3` env

On a CUDA machine:

```bash
python install.py --with-sam3-train      # or: make setup-sam3-train
```

This is the only supported recipe. It builds `hydra-sam3` with the matching
CUDA torch, `sam3` pinned to a tested commit, and the **same hydra-suite
source** as the main env. It never installs from PyPI. The env is reused if
it already exists. See
[SAM2 and SAM3: install and run](sam-install-and-run.md#sam3-training-optional-cuda-only)
for details and troubleshooting.

#### Hugging Face access is required on every training machine

Building the model calls `build_sam3_image_model`, which fetches the SAM3
config from the **gated** `facebook/sam3` repo on Hugging Face. Having the
3.45 GB checkpoint already on disk does **not** remove this requirement:
the local file supplies weights, not the architecture config.

Escalation with the **stock** `sam3` checkpoint needs the same licence and
login to download the weights. Escalation with a published finetuned
checkpoint needs neither. So on each machine that will train, or download
the stock weights:

1. Accept the licence at <https://huggingface.co/facebook/sam3> with the
   account you will authenticate as.
2. Authenticate on that machine:

```bash
hf auth login       # or: export HF_TOKEN=hf_...
```

Without it, model construction fails with
`huggingface_hub.errors.GatedRepoError: 401`. Preflight checks for a
credential up front and refuses in milliseconds, rather than letting the run
die minutes later inside the sidecar subprocess. However, preflight can only
see whether a token EXISTS. If the token's account has not accepted the
licence, the 401 still arrives at build time.

#### Hardware requirements

| Resource | Requirement |
|---|---|
| GPU | NVIDIA, compute capability ≥ 8.0 with bf16 |
| VRAM | **12 GiB** free (the measured bf16 device peak preflight admits against, batch 1, rank 16, 1008 px tiles) |
| Host RAM | ~16 GB |
| Free disk | 8 GB (3.45 GB base checkpoint + ~3.2 GB merged artifact) |

A 24 GB RTX 4090 (capability 8.9) trains this role comfortably. Short probes
have under-reported the full-run peak, so use the 12 GiB figure rather than
any shorter measurement. Every run logs its own `vram_peak` on each progress
line, so you can re-derive the figure on your own hardware.

#### Why the sidecar pins what it pins

The pins live in `install.py` (`sam3_train_steps`). Several are not obvious,
and each was found the hard way:

- **`--no-deps` on the hydra-suite install.** `pyproject.toml`'s core
  dependency is `numpy>=1.24`, which pip would resolve to numpy 2.x and so
  break `sam3`'s `numpy<2` pin. Every runtime dependency the training CLI
  imports is installed explicitly, so `--no-deps` costs nothing.
- **`setuptools<81`.** setuptools 81 removed `pkg_resources`, which
  `sam3/model_builder.py` imports at module scope.
- **`einops`, `pycocotools`, `psutil`.** `sam3` imports these at module
  scope but does not declare them.
- **`scipy<1.14` and `opencv-python-headless<4.12`.** Newer releases
  require `numpy>=2`. They are installed in the same resolver call as
  `numpy<2` so nothing drags numpy 2 back in.
- **`pandas`/`numba`.** The in-env CLI runs as
  `python -m hydra_suite.training.sam3_lora.cli`, and importing
  `hydra_suite` that way pulls them in.

If you run `conda run -n hydra-sam3 ...` commands by hand (the GUI sets this
for you), set **`KMP_DUPLICATE_LIB_OK=TRUE`** first. Without it, a bare
`import torch` aborts with `OMP Error #15` (double-linked libomp).

### Platform: CUDA only

SAM3 training requires an **NVIDIA GPU with compute capability ≥ 8.0 and
bf16**. This is a hard gate, not a best-effort probe:

- `python install.py --with-sam3-train` refuses to build the sidecar on a
  non-CUDA host.
- On a machine without CUDA, the SAM3 training tab says that training needs
  an NVIDIA GPU and that SAM3 *inference* and SAM2 escalation still work.
- Preflight refuses a run with no CUDA device, a GPU below compute
  capability 8.0, or any precision other than bf16. There is no fp32 fallback.

**macOS (MPS)** has a second blocker as well. `sam3/__init__.py` reaches
`import triton` at module scope through the video-tracker import path
(`model_builder.py` → `sam1_task_predictor.py` → `sam3_tracker_base.py` →
`sam3_tracker_utils.py` → `edt.py`), and `triton` ships no macOS wheel. So
`import sam3` fails on a Mac today. This is a packaging problem, not a memory
problem. Even if that import became optional upstream, MPS training would
still need the CUDA/bf16 gates above lifted and the path verified. It would
not "just work".

**VRAM admission.** Preflight admits a run against the measured bf16 device
peak, 12 GiB at batch 1, before any weight is loaded. The check runs in the
launching process and **again inside the sidecar**: the launcher re-observes
the selected GPU while it holds the GPU lease, and the training child repeats
the CUDA/bf16/compute-capability gate before importing `sam3`. A busy GPU
therefore fails in seconds, not after an hour of compute.

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

The panel's **Tiling (SAHI geometry)** group is the shared SAHI widget (see
[SAHI Settings](sahi-settings.md)). It defaults to "Fit to animal size", an
object scale of `0.055` of the 1008 px input, and a tile overlap of `0.25`.
Fragments below the minimum retained area are kept but marked `is_crowd`
rather than dropped. The `0.055` training default is deliberately not the
escalation dialog's `0.05` starting guess.

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

### Checkpoint selection

Which epoch's weights become the run's adapter is a per-run setting,
`checkpoint_selection`. The default, `best_val_loss`, keeps the epoch with the
lowest validation loss (ties go to the earlier epoch). `last` keeps the final
epoch. The choice is recorded in the run's `checkpoint_selection.json`.

This default was a deliberate user decision (2026-09-15), made knowing the
earlier spike evidence pointed the other way. On that spike, validation loss
was **anti-correlated** with held-out AP: the fold with the worst validation
loss had the best held-out AP75. That spike's validation split later turned
out to be byte-identical to its training split, so the comparison is not
usable evidence either way. See the module docstring of
`hydra_suite.training.sam3_lora.cli` for the full history.

### Publishing

A finished run publishes a **merged, full checkpoint** — the LoRA adapter
folded into the base weights, not the adapter alone — to
`get_models_dir() / "sam3_finetuned"`, alongside a JSON sidecar recording
the run's parameters and dataset fingerprint
(`<checkpoint>.sam3_meta.json`), and registers it in the model registry.
Publishing also writes the tiling geometry to
`<checkpoint>.slice_meta.json` (schema 3, `model_family: "sam3"`), the
same sidecar format YOLO models use. The geometry stays in
`.sam3_meta.json` too, so older readers keep working. From then on, the escalation dialog's model selector
offers it next to the stock `sam3` checkpoint.

A load guard (`assert_checkpoint_loaded`) checks that a selected checkpoint
actually loaded before it is used for inference. This exists because
ultralytics loads state dicts with `load_state_dict(strict=False)` and
discards the result it returns — without an explicit check, a checkpoint
that fails to match the model's shape would silently fall back to serving
stock weights instead of raising.
