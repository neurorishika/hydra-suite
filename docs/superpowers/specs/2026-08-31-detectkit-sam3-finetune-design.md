# DetectKit — SAM3 LoRA finetuning as a training role

**Status:** pending implementation plan

## Why

DetectKit's semantic escalation runs SAM3 zero-shot from a text prompt. On the
`improved_ant_detection` project that is not good enough, and a YOLO-seg model
trained on the same three frames is worse. A leave-one-frame-out spike (`spike/sam3-lora`,
`scratch/sam3_lora_spike/FINDINGS.md`, result JSONs committed under
`scratch/sam3_lora_spike/results/`) measured, **on one held-out frame after one
epoch of one fold**:

| arm (held-out frame f008078)  | AP50  | AP75  | mean matched IoU |
|-------------------------------|-------|-------|------------------|
| YOLO-seg, best tile size      | 0.483 | 0.220 | 0.761            |
| SAM3 zero-shot                | 0.447 | 0.001 | 0.584            |
| SAM3 + LoRA, **one epoch**    | 0.737 | 0.563 | 0.769            |

**How much weight this evidence carries.** One fold, one epoch, 24 GT instances
on a single frame from a single video. Both known biases favour YOLO (its
training set included this frame; its tile size was swept on the test frames),
so the direction is conservative -- but the magnitude is noisy and should not be
quoted as a expected gain. Two further caveats the numbers do not carry on their
face: `mean matched IoU` is averaged only over the GT each arm actually found,
so zero-shot's 0.584 is computed over the 62% it hit and is a selection effect;
and the scorer's greedy argmax matching is not COCO matching (a prediction whose
best GT is already taken becomes an FP even if a second GT would clear
threshold), which deflates all arms but makes these numbers non-comparable to
published COCO AP. Folds 2-3 and the AP-vs-epoch curve are in flight; this design
is justified by the *direction* and by the mechanism below, not by 0.737.

The diagnosis is specific and it is what makes finetuning the right lever:
**zero-shot SAM3 already localises the animals as well as a trained YOLO, but
its mask geometry is wrong** — it overshoots, tracing legs, the same ~1.7x area
bias the shape-aware calibration work documented. AP75 of 0.001 is that bias.
One epoch on ~48 example polygons fixed it. Localisation is expensive to teach;
shape is cheap, and the user already has the labels.

Confidence threshold is not the lever: zero-shot AP50 is 0.465/0.465/0.462 at
conf 0.1/0.2/0.4.

**The numbers above were measured through Meta's native `sam3` stack, not
through the stack that will ship.** The spike scored
`pred_logits.sigmoid().max(-1)` with no presence term and no per-tile NMS;
ultralytics' `SAM3SemanticPredictor.postprocess` multiplies
`sigmoid(logits) * presence` and applies NMS at `iou=0.7`
(`ultralytics/models/sam/predict.py:2299-2315`, pinned as `PREDICTOR_NMS_IOU`
in `semantic/sam3.py:31`). Confidence thresholds are therefore **not comparable
between the two stacks**, and no run has yet put a merged checkpoint through the
ultralytics graph. Closing that gap is an acceptance criterion, not a follow-up:
see "Acceptance".

## Scope

A new first-class training role that finetunes SAM3 on a DetectKit source's
polygon labels and publishes a checkpoint the existing escalation path can load.

Out of scope: video/tracking SAM3, prompt-free operation, multi-class SAM3,
adapter stacking, and any change to how escalations are reviewed.

## Decisions taken

1. **Merged full checkpoint, not a shipped adapter.** LoRA is merged into the
   base weights at publish time and written as a ~3.4 GB `.pt` in Meta key
   layout. Justification below.
2. **Training uses every label in the chosen source.** Per-frame provenance does
   not survive a review (see "Provenance is unavailable"), so a human-only mode
   would be a lie. The user is warned instead.
3. **A full training role**, in the existing contracts/runner/service/publish
   machinery — not a side door in the escalation dialog.
4. **We own the trainer.** ~350 lines against Meta's `sam3` package, importing
   Meta's own loss and matcher.

### Why a merged checkpoint

`ultralytics/models/sam/build_sam3.py:357` builds its state dict as
`{k.replace("detector.", ""): v for k, v in ckpt.items() if "detector" in k}`
and then calls `load_state_dict(..., strict=False)`. The published
`facebook/sam3` `sam3.pt` is a raw Meta-layout state dict with `detector.`
prefixes, so **a merged checkpoint written in that same layout loads into the
existing ultralytics escalation path with no inference-side code change and no
new runtime dependency.** The alternative — shipping the 46 MB adapter — needs a
second SAM3 inference backend built on Meta's package, which drags `decord`,
`torchmetrics`, `scipy` and `einops` into the runtime install for every user.
Disk is the cheaper currency here.

That `strict=False` is a footgun and this design must guard it: see
"The silent-load guard".

### Why we own the trainer

`Sompote/sam3_lora` proved the approach and is what the spike used, but its
value was concentrated in one fact: **it does not reimplement the objective.**
It imports Meta's `Sam3LossWrapper` and `BinaryHungarianMatcherV2` from the
official `sam3` package and wraps them in a LoRA injection, a COCO dataloader
and a training loop. Those three are standard code we should own and test.
The repo itself is a poor dependency — 70 MB, thirteen training scripts, a
60 KB trainer, a vendored copy of `sam3`, and a bundled `Machinery.zip`.
Vendoring it is a maintenance liability; treating it as an external tool under
`integrations/` (the SLEAP precedent) is unjustified for a two-week-old
single-contributor repo.

### Provenance: a hard ordering dependency on the review redesign

**Today, source choice IS the provenance filter.**
`accept_pending_semantic_escalation` (`detectkit/jobs/semantic_escalation.py:1174`)
promotes staged SAM3 output to a **new sibling source** and deliberately never
touches the origin's labels -- its docstring says overwriting them "would be
worse". So a user can train on hand-labelled polygons today simply by selecting
the hand-labelled source.

That changes only when
`2026-08-31-detectkit-frame-granular-review-design.md` lands. After it, accepted
labels are written **in place** into the source's `labels/`, byte-indistinguishable
from hand-drawn polygons; `decisions.json` holds per-frame outcomes only for the
review's lifetime ("completing it removes the staging dir"); and `producer` is
source-level and explicitly "load-bearing for *nothing*". At that point, and only
then, "train on human labels only" stops being expressible.

**This is therefore a stated ordering dependency, not a background assumption:**

- If this role ships **before** the review redesign, the training panel must say
  that selecting a non-escalation source is the provenance filter, and the
  warning is advisory.
- Once the review redesign lands, the distinction is gone and the warning becomes
  the only control.

Either way the role trains on whatever is in the selected source's `labels/`, and
**requires the user to acknowledge a warning before the run starts**: SAM3 will
learn any systematic error in those labels, including its own accepted output.
The acknowledgement and the source fingerprint are recorded in the run record so
a suspect model can be traced back to its labels.

## Architecture

Dependency direction is unchanged: `training/` and `core/inference/semantic/`
are below the app layer; DetectKit imports them and neither imports DetectKit.

### 1. Contracts (`training/contracts.py`)

```python
class TrainingRole(str, Enum):
    ...
    SEMANTIC_SAM3 = "semantic_sam3"


@dataclass(slots=True)
class Sam3LoraParams:
    prompt: str = ""              # the concept text; required, no default concept
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.1
    lr: float = 5e-5
    epochs: int = 20
    batch: int = 1                # 1008 px + ViT backprop; see VRAM preflight
    grad_accum: int = 8
    mixed_precision: str = "bf16"
    num_negatives: int = 3        # hard-negative prompts that must return nothing
    # Which submodules receive adapters. Text encoder is False by default as a
    # precaution against eroding prompt discrimination -- untested here; the
    # spike froze it in every configuration.
    adapt_vision_encoder: bool = True
    adapt_text_encoder: bool = False
    adapt_geometry_encoder: bool = True
    adapt_detr_encoder: bool = True
    adapt_detr_decoder: bool = True
    adapt_mask_decoder: bool = True
    # Tiling, mirroring the SAHI sliced-training knobs.
    geometry_mode: str = "auto_object"   # auto_object | auto_model | custom
    object_tile_fraction: float = 0.055
    slice_width: int = 0                 # custom mode only; 0 => fall back to imgsz
    slice_height: int = 0                # custom mode only
    tile_overlap: float = 0.25
    keep_empty_tiles: bool = True
```

`TrainingRunSpec` gains `sam3_params: Sam3LoraParams | None = None`, exactly as
it already carries `tiny_params` and `custom_params`.

`object_tile_fraction` defaults to 0.055 because that is what the spike
measured, not because it is principled: 55 px animals in a 1008 px tile. It is
a knob with a measured default, and the design says so rather than dressing it
up. It deliberately diverges from the `0.15` used by the SAHI consumers
(`core/inference/config.py:74`, `training/sliced_dataset.py:92`) and the `0.05`
escalation seed (`semantic/tiling.py:32`) because SAM3's fixed 1008 input makes
the trade-off a different one.

`epochs = 20` is a **placeholder, and weaker than it looks**. The spike hit
AP75 0.563 after one epoch, but with `warmup_steps: 50` and ~9 optimiser steps
per epoch that checkpoint trained entirely inside LR warmup — it may not reflect
the configured learning rate at all. Fold-1 val loss then rose from 0.884 to
0.997 and never cleanly improved. The AP-vs-epoch curve (fold 1 snapshots, eval
pending) is a **blocking input** to this default: do not implement against 20
without it.

### 2. Dataset builder (`training/dataset_builders.py`)

`prepare_role_dataset` gains a `SEMANTIC_SAM3` branch producing a COCO
instance-segmentation dataset of tiles:

```
<derived>/semantic_sam3/<run>/
    train/_annotations.coco.json  + tile jpgs
    valid/_annotations.coco.json  + tile jpgs
    build_manifest.json
```

- Tile size from `slice_geometry.tile_size_for_mode(...)` using the source's
  measured `reference_body_px` — the same measurement the SAHI builder already
  makes and stamps. `custom` mode needs `slice_width`/`slice_height`, which is
  why `Sam3LoraParams` carries them.
- **SAM3's input is a fixed 1008 px square.** A `tile_px` tile is cut from the
  frame and resized to 1008 before it reaches the model, so the scale the model
  sees is `1008 / tile_px`. The spike's "native resolution" case is only the one
  where `tile_px` happened to be 1008 and the resize was the identity; that is
  not an invariant and the builder must not assume it. `object_tile_fraction` is
  precisely the knob that sets this ratio — it fixes how many of the 1008 px an
  animal occupies (0.055 => ~55 px) regardless of its size in the source frame.
  Training and inference must resolve `tile_px` from the **same** stamped
  `reference_body_px` and fraction, or the model is finetuned at a scale
  escalation never reproduces.
- Tile positions from `slice_geometry.plan_tiles`, polygons clipped with
  `slice_geometry.clip_polygon_to_tile`. The spike's throwaway tiler
  reimplemented all of this; the role must not.
- An instance clipped below `MIN_RETAINED_AREA_FRAC = 0.5` by a tile seam is
  written with `iscrowd = 1` so it neither rewards nor punishes. Dropping it
  instead would teach SAM3 that a visible half-animal is background.
- Empty tiles are **kept** when `keep_empty_tiles`. False positives on
  background are half of what "is it good" means, and the trainer's
  hard-negative prompts cover prompt discrimination, not empty scenes.
- On a multi-class source the builder emits **only the selected class**.
- `category.name` is `Sam3LoraParams.prompt`, so the concept the model is
  trained on and the concept escalation prompts with are the same string by
  construction.

The split is by **frame**, never by tile: tiles from one frame overlap, so a
tile-level split leaks pixels across train/valid.

### 2b. What this role does NOT get for free

Adding a role is not a drop-in. Every item below is a concrete breakage on the
existing path, verified in code, and each needs work the implementation plan
must budget for:

| # | Site | What happens today | Needed |
|---|------|--------------------|--------|
| 1 | `validation.py:439` | `validate_role_dataset` calls `inspect_obb_or_detect_dataset` **unconditionally, before any role branch**, and that inspector **raises** `RuntimeError("No valid OBB/detect dataset layout found…")` on a COCO layout (`dataset_inspector.py:233-247`). `service.build_role_dataset:400` calls it on every derived dataset. | Branch on the role **before** the inspector runs; add a COCO-layout validator. **This is a crash, not a gap** — the freshly built dataset fails validation immediately. |
| 2 | `dataset_builders.py:1001-1016` | `role_min_level` raises `RuntimeError("Role has no geometry-level requirement")` for unknown roles, and `prepare_role_dataset:1044` calls it first thing. | `SEMANTIC_SAM3: GeometryLevel.POLYGON` in `_ROLE_MIN_LEVEL`. |
| 3 | `model_publish.py:42`, `:72` | `_repo_dir_for_role` and `_task_usage_for_role` both raise `"Unsupported publish role"`. | Either extend both, or fork publish for this role — and say which. |
| 4 | `service.py:465-472` | Auto-publish routes through `_publish_training_artifacts` → `publish_trained_model`, which hits #3. | A forked publish path must still register in `model_registry.json`. A directory scan of `<models>/sam3/` would be a **second registry convention**, in direct tension with `2026-07-29-model-registry-unification-design.md`. Registry, not scan. |
| 5 | `dataset_builders.py:1031` | `prepare_role_dataset`'s first positional is `merged_obb_dataset_dir`; the service merges sources first (`service.py:328-347`). | Decide and state whether the SAM3 branch consumes the merged dir or a single raw source. "The chosen source" (singular) elsewhere in this spec implies the latter; the merge step then has to be skipped explicitly. |
| 6 | `runner.py:2300` | Any role that falls past the custom-classify branches reaches `build_ultralytics_command`. | Dispatch `SEMANTIC_SAM3` **before** that fall-through, or it silently builds a nonsense `yolo` command. |

Only `TrainingRunSpec.sam3_params` genuinely slots in as-is: it mirrors
`custom_params`, and `to_dict`'s `asdict` handles slots dataclasses
(`contracts.py:196-217`).

### 3. Trainer (`training/sam3_lora/`, Qt-free)

A **package**, not a module — the seams below are separate files, per the
no-god-object rule. Dispatched from `runner.run_training` before the ultralytics
fall-through, in the same shape as `_CUSTOM_CLASSIFY_ROLES`.

Four units, each independently testable:

- `lora.py` seam — `LoraConfig`, `inject_adapters(model, cfg) -> int`,
  `adapter_state_dict(model)`, `merge_adapters(base_state_dict, adapters)`.
  Pure tensor code, no SAM3 import, so it is unit-testable on a toy module.
- `dataset.py` seam — COCO tiles → the `Datapoint` / `FindQueryLoaded` /
  `collate_fn_api` structures Meta's model consumes, plus negative-prompt
  sampling.
- The loop — AdamW, cosine schedule with warmup, grad accumulation, bf16,
  gradient clipping at 1.0. Emits `log_cb`/`progress_cb` and honours
  `should_cancel` between steps, like every other trainer here.
- `probe_sam3_training_availability()` — mirrors
  `semantic/checkpoints.py:probe_availability`: checks `sam3`, `torchmetrics`,
  `scipy`, `einops`, `decord` and the base checkpoint **without importing
  ultralytics or downloading anything**, returning a structured reason the GUI
  shows on a disabled action instead of failing at click time.

Objective and matcher are imported from Meta's package
(`sam3.train.loss.sam3_loss.Sam3LossWrapper`,
`sam3.train.matcher.BinaryHungarianMatcherV2`). We do not reimplement them.

Optimiser state is not checkpointed; `resume_from` is unsupported for this role
and rejected in preflight rather than silently ignored.

### 4. Publish (`training/model_publish.py`)

On success the trainer writes `adapters.pt` (~46 MB) into the run dir, and
publish merges:

1. Load the base `sam3.pt` state dict (Meta layout, `detector.`-prefixed).
2. Apply `merge_adapters` in that layout.
3. Write `<models>/sam3/<run_id>_semantic_sam3.pt` plus a
   `<artifact>.sam3_meta.json` sidecar:

```json
{
  "base_variant": "sam3",
  "prompt": "ant with color patch",
  "tile_px": 1008,
  "reference_body_px": 55.4,
  "object_tile_fraction": 0.055,
  "stripped_keys": ["backbone.vision_backbone...", "..."],
  "source_fingerprint": "<dataset fingerprint>",
  "label_quality_acknowledged": true
}
```

The adapters are retained in the run dir. They are the cheap artifact to keep
for re-merging against a future base checkpoint; the 3.4 GB file is the
disposable one.

### 5. Escalation consumption

- `checkpoints.py` grows `resolve_checkpoint(variant_or_model_key)` which
  returns a stock variant's path **or** a published SAM3 artifact's path, and
  `available_models()` returning stock variants plus published ones.
- `Sam3SemanticLabeler.from_variant` gains a `checkpoint:` override so the
  labeler can be built from a published artifact. Nothing else in the SAM3
  inference path changes.
- `SemanticEscalationDialog`'s Model combo lists published models alongside
  stock variants; choosing one **prefills prompt and tile fraction from the
  sidecar**, the same pattern TrackerKit already uses to prefill its SAHI panel
  from `.slice_meta.json`. Prefill is a default, not a lock — `REFERENCE_BODY_SIZE`
  precedent applies: the measured value is sacrosanct, the derived one is a
  starting point.

### The silent-load guard

`_load_checkpoint` **discards** the return value of
`load_state_dict(..., strict=False)` (`ultralytics/models/sam/build_sam3.py:381`),
and the model is built lazily inside the predictor's first call. So the labeler
cannot simply read `missing_keys`/`unexpected_keys` off the load. If a future
ultralytics renames keys, a finetuned checkpoint loads **zero** tuned weights and
escalation silently reverts to zero-shot quality — no error, no symptom except
worse masks, which is the hardest failure to notice.

The guard therefore needs a stated mechanism, not an intention:

1. **Publish** records in the sidecar the full **stripped key list** the merged
   checkpoint contributes — i.e. the result of ultralytics' own transform,
   `{k.replace("detector.", ""): v for k, v in ckpt.items() if "detector" in k}`
   (note: a substring test, not a prefix match), reduced to sorted key names.
   A count alone cannot name what drifted; the list can.
2. **Load** forces eager `setup_model()` at labeler construction rather than on
   first inference, so the built model's `state_dict().keys()` is observable.
3. **Assert** the sidecar's key set is a subset of the model's, and raise naming
   the missing keys on mismatch.

Reading the key index does **not** require materialising 3.4 GB: `torch.load`
with `mmap=True` (or reading the zip directory) yields key names cheaply.

Two honest limits. Shape-mismatched keys raise even under `strict=False`, so key
identity is the only thing needing a guard. And a **stock** variant is
deliberately unguarded — it ships no sidecar and makes no claim — so an upstream
rename degrades stock escalation with the same silence. That is pre-existing, not
introduced here, and worth a separate issue rather than scope creep.

### 6. Preflight

Before any weights load:

- **Device and VRAM.** The spike measured ~29 GB at batch 1 (batch 2 OOMed on a
  47 GB card). The threshold is therefore **refuse below 32 GB free**, warn
  below 40 GB — not "warn below 24", which would wave through the 24-29 GB band
  straight into the OOM the preflight exists to prevent.
- **Where it can run.** CUDA only. That means the role **cannot run in-process on
  the macOS box where DetectKit is normally driven** — `run_training` executes in
  the caller's process, like the custom-classify roles. This spec does **not**
  solve remote training; it requires the GUI to state plainly that the role needs
  a CUDA host, and it scopes the first slice to running there. A remote-execution
  story is a separate design.
- **bf16.** Default `mixed_precision: "bf16"` requires compute capability >= 8.0.
  Probe and fall back to fp16/fp32 with a logged notice rather than failing deep
  in the loop.
- **Disk.** The merged artifact is ~3.4 GB and the merge needs the base
  checkpoint resident. Check free space before training starts, not at publish
  time after an hour of GPU.
- **Prompt.** Empty prompt rejected; there is no defensible default concept.
- **Instances.** Below `MIN_TRAIN_INSTANCES = 20` (matching `calibration.py:58`'s
  existing floor) the run is refused as unmeasurable.
- **Label quality.** The acknowledgement described above.
- **`resume_from`** set => rejected (optimiser state is not checkpointed).

### 7. Things this design fixes the meaning of

- **Determinism.** The trainer consumes `TrainingRunSpec.seed` (`contracts.py:203`)
  for torch/numpy/python RNG and the tile-order shuffle. Byte-identical reruns are
  **not** promised — cuDNN autotuning and bf16 reduction order are not pinned —
  but seed-controlled reruns are.
- **Multi-class sources.** `category.name = prompt` is single-concept by
  construction. On a source with multiple classes the builder trains the
  **selected class only** and records which; it does not silently merge classes
  into one concept.
- **Cancellation and progress.** `should_cancel` is honoured between optimiser
  steps, and the multi-minute base-checkpoint load + merge at publish reports
  progress rather than appearing hung.
- **Equivalence gates.** Nothing here is on the tracking path, and semantic
  escalation has no equivalence-fixture coverage at all. So there is no gate to
  run — and equally, no safety net. The Testing section is the only net.

## Testing

1. **Tiling/COCO builder** — clip fractions at the `iscrowd` boundary in both
   directions; empty-tile retention; frame-level (not tile-level) split;
   `category.name` equals the prompt.
2. **LoRA seam** — inject/merge round-trip on a toy `nn.Module`: merged weights
   equal base + BA*scale; `inject_adapters` returns the expected count; merging
   a zero-initialised adapter is a no-op on the base state dict.
3. **Merged-checkpoint key layout** — the published artifact's keys are a
   superset of what `build_sam3.py`'s `detector.`-strip produces, and
   the sidecar's `stripped_keys` matches. This is the test that actually protects the
   integration; it needs no GPU and no real weights, only key names.
4. **Silent-load guard** — a deliberately renamed key raises instead of loading
   partially.
5. **Preflight** — each refusal path returns its structured reason and does not
   import ultralytics or touch the network.
6. **Availability probe** — a missing training dep yields a disabled action with
   a reason, never an AutoUpdate pip install.

An end-to-end training test is not in the suite: it needs a licence-gated 3.4 GB
checkpoint and a 32 GB CUDA card. The manual gate is below.

## Acceptance

Automated tests cannot cover the integration that actually matters, so these are
the conditions for calling the role done. Both run on mehek.

1. **Stack-parity gate (blocking).** The merged checkpoint, loaded through the
   **ultralytics** path, must reproduce the native-`sam3`-path AP on at least one
   held-out frame within tolerance. This is the criterion the original spec was
   missing: every number motivating this work was measured on the native stack,
   and the two stacks score differently (presence multiplication, NMS at 0.7).
   Until this passes, nothing is known about what ships.

   The spike harness cannot run this today — `Sam3SemanticLabeler.from_variant`
   has no checkpoint override (`semantic/sam3.py:75-84`). The `checkpoint:`
   override this design adds (§5) makes it a ~10-line change to
   `scratch/sam3_lora_spike/arm_sam3.py`, which already drives the ultralytics
   path. Add the override first, then this gate is cheap.

2. **Beats the tuned baseline (blocking).** Through the ultralytics path, on
   held-out frames, the finetuned model beats the best-configured YOLO-seg arm on
   AP75. AP50 parity is not sufficient — zero-shot already achieves that, and mask
   geometry is the entire thesis.

3. **Scale round-trip (blocking).** A model trained at `tile_px` X and escalated
   at `tile_px` X reproduces its training-time quality; the sidecar prefill is
   what guarantees it. A deliberate mismatch should visibly degrade — if it does
   not, the scale coupling is not doing what this design claims.

## Risks

- **3.4 GB per published model.** Accepted deliberately (see above). Mitigated
  by keeping adapters as the re-mergeable artifact.
- **Self-training loop.** Unavoidable today; warned about, and the source
  fingerprint is recorded so a suspect model can be traced to its labels.
- **Single-video evidence.** The spike's three frames share one arena and
  lighting. Leave-one-frame-out controls overfitting to a *frame*, not to a
  *setup*. The AP numbers above justify building this; they do not promise a
  number on a new rig.
- **Meta `sam3` is an unversioned dependency.** Pin it, and treat the
  stripped-key-list guard as the tripwire for upstream drift.
- **Two scoring stacks.** Training uses Meta's `sam3`; inference uses
  ultralytics' wrapper over the same weights. They agree on weights, not on
  post-processing. The stack-parity gate is the only thing standing between that
  and a silent quality gap; it is blocking for exactly that reason.
- **The role cannot run where the GUI runs.** CUDA-only, ~32 GB. Until a remote
  execution story exists, this is a lab-box feature exposed in a desktop app —
  the GUI must say so rather than offering a button that always fails.
