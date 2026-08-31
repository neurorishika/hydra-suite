# DetectKit — SAM3 LoRA finetuning as a training role

**Status:** pending implementation plan

## Why

DetectKit's semantic escalation runs SAM3 zero-shot from a text prompt. On the
`improved_ant_detection` project that is not good enough, and a YOLO-seg model
trained on the same three frames is worse. A leave-one-frame-out spike
(`spike/sam3-lora`, `scratch/sam3_lora_spike/FINDINGS.md`) measured all three
arms on held-out frames through their real tiling paths:

| arm (held-out frame f008078)  | AP50  | AP75  | mean matched IoU |
|-------------------------------|-------|-------|------------------|
| YOLO-seg, best tile size      | 0.483 | 0.220 | 0.761            |
| SAM3 zero-shot                | 0.447 | 0.001 | 0.584            |
| SAM3 + LoRA, **one epoch**    | 0.737 | 0.563 | 0.769            |

The diagnosis is specific and it is what makes finetuning the right lever:
**zero-shot SAM3 already localises the animals as well as a trained YOLO, but
its mask geometry is wrong** — it overshoots, tracing legs, the same ~1.7x area
bias the shape-aware calibration work documented. AP75 of 0.001 is that bias.
One epoch on ~48 example polygons fixed it. Localisation is expensive to teach;
shape is cheap, and the user already has the labels.

Confidence threshold is not the lever: zero-shot AP50 is 0.465/0.465/0.462 at
conf 0.1/0.2/0.4.

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

### Provenance is unavailable, so we warn

After `2026-08-31-detectkit-frame-granular-review-design.md` lands, accepted
escalation labels are written **in place** into the source's `labels/`,
byte-indistinguishable from hand-drawn polygons. `decisions.json` holds the
per-frame outcome only for the review's lifetime — "completing it removes the
staging dir" — and `producer` is source-level and explicitly "load-bearing for
*nothing*". So "train on human labels only" cannot be expressed.

The role therefore trains on whatever is in `labels/` and **requires the user to
acknowledge a warning before the run starts**: SAM3 will learn any systematic
error present in those labels, including its own accepted output. The
acknowledgement and the source fingerprint are recorded in the run record.

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
    # Which submodules receive adapters. Text encoder is False by default and
    # adapting it erodes prompt discrimination.
    adapt_vision_encoder: bool = True
    adapt_text_encoder: bool = False
    adapt_geometry_encoder: bool = True
    adapt_detr_encoder: bool = True
    adapt_detr_decoder: bool = True
    adapt_mask_decoder: bool = True
    # Tiling, mirroring the SAHI sliced-training knobs.
    geometry_mode: str = "auto_object"   # auto_object | auto_model | custom
    object_tile_fraction: float = 0.055
    tile_overlap: float = 0.25
    keep_empty_tiles: bool = True
```

`TrainingRunSpec` gains `sam3_params: Sam3LoraParams | None = None`, exactly as
it already carries `tiny_params` and `custom_params`.

`object_tile_fraction` defaults to 0.055 because that is what the spike
measured, not because it is principled: 55 px animals in a 1008 px tile. It is
a knob with a measured default, and the design says so rather than dressing it
up. `epochs` defaults to 20 on the same footing: the spike reached AP75 0.563
after **one** epoch and its train loss stopped improving around epoch 20, but
the AP-vs-epoch curve that would settle this is still running. Revisit the
default when that curve exists; do not treat 20 as load-bearing. `geometry_mode`/`tile_size_for_mode` semantics are inherited unchanged from
`utils/slice_geometry.py`.

### 2. Dataset builder (`training/dataset_builders.py`)

`prepare_role_dataset` gains a `SEMANTIC_SAM3` branch producing a COCO
instance-segmentation dataset of native-resolution tiles:

```
<derived>/semantic_sam3/<run>/
    train/_annotations.coco.json  + tile jpgs
    valid/_annotations.coco.json  + tile jpgs
    build_manifest.json
```

- Tile size from `slice_geometry.tile_size_for_mode(...)` using the source's
  measured `reference_body_px` — the same measurement the SAHI builder already
  makes and stamps. Tiles are cut at **native resolution** and handed to SAM3
  at its native 1008 px input; no downscale.
- Tile positions from `slice_geometry.plan_tiles`, polygons clipped with
  `slice_geometry.clip_polygon_to_tile`. The spike's throwaway tiler
  reimplemented all of this; the role must not.
- An instance clipped below `MIN_RETAINED_AREA_FRAC = 0.5` by a tile seam is
  written with `iscrowd = 1` so it neither rewards nor punishes. Dropping it
  instead would teach SAM3 that a visible half-animal is background.
- Empty tiles are **kept** when `keep_empty_tiles`. False positives on
  background are half of what "is it good" means, and the trainer's
  hard-negative prompts cover prompt discrimination, not empty scenes.
- `category.name` is `Sam3LoraParams.prompt`, so the concept the model is
  trained on and the concept escalation prompts with are the same string by
  construction.

The split is by **frame**, never by tile: tiles from one frame overlap, so a
tile-level split leaks pixels across train/valid.

### 3. Trainer (`training/sam3_lora.py`, Qt-free)

Dispatched from `runner.run_training` before the ultralytics branch, in the
same shape as `_CUSTOM_CLASSIFY_ROLES`.

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
  "expected_loaded_keys": 1465,
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

`build_sam3.py` loads non-strictly. If a future ultralytics renames keys, a
finetuned checkpoint loads **zero** tuned weights and escalation silently
returns to zero-shot quality — a failure with no error and no visible symptom
except worse masks, which is the hardest kind to notice.

So: publish records `expected_loaded_keys`, and the labeler asserts the count of
matched keys when loading a finetuned artifact, raising with an actionable
message on mismatch. A stock variant skips the check (no sidecar, no claim).

### 6. Preflight

Before any weights load:

- **Device.** Training needed ~29 GB at batch 1 on the spike. Refuse anything
  but CUDA with a message naming the measured requirement, rather than OOMing
  after a 3.4 GB model load. A CUDA card below ~24 GB free warns.
- **Prompt.** Empty prompt is rejected; there is no defensible default concept.
- **Instances.** Below a floor (`MIN_TRAIN_INSTANCES = 20`, matching
  `calibration.py`'s existing floor) the run is refused as unmeasurable.
- **Label quality.** The acknowledgement described above.
- **`resume_from`** set → rejected.

## Testing

1. **Tiling/COCO builder** — clip fractions at the `iscrowd` boundary in both
   directions; empty-tile retention; frame-level (not tile-level) split;
   `category.name` equals the prompt.
2. **LoRA seam** — inject/merge round-trip on a toy `nn.Module`: merged weights
   equal base + BA*scale; `inject_adapters` returns the expected count; merging
   a zero-initialised adapter is a no-op on the base state dict.
3. **Merged-checkpoint key layout** — the published artifact's keys are a
   superset of what `build_sam3.py`'s `detector.`-strip produces, and
   `expected_loaded_keys` matches. This is the test that actually protects the
   integration; it needs no GPU and no real weights, only key names.
4. **Silent-load guard** — a deliberately renamed key raises instead of loading
   partially.
5. **Preflight** — each refusal path returns its structured reason and does not
   import ultralytics or touch the network.
6. **Availability probe** — a missing training dep yields a disabled action with
   a reason, never an AutoUpdate pip install.

An end-to-end training test is not in the suite: it needs a licence-gated 3.4 GB
checkpoint and a 29 GB CUDA card. The spike harness on `spike/sam3-lora` is the
manual gate, run on mehek.

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
  `expected_loaded_keys` guard as the tripwire for upstream drift.
