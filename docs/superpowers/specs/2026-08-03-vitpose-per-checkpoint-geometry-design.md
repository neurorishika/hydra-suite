# ViTPose per-checkpoint geometry — design

Date: 2026-08-03. Branch: `feat/vitpose-external-probe` (continues on top of the
external-checkpoint probe).

## Why

`vitpose/config.py` fixes the model input at `IMAGE_SIZE_WH = (192, 256)` and the
heatmap at `HEATMAP_SIZE_WH = (48, 64)` for every checkpoint, process-wide. That is
the COCO **human portrait** aspect ratio, 0.75.

Our animals are not shaped like people. Crops arrive from OBB tracking roughly
square, and `transforms.box2cs` grows every one of them to 3:4 before warping — so a
square ant crop spends about a quarter of its pixel budget on padding, and the
effective resolution on small structures is correspondingly lower. The
`2026-08-03-external-vitpose-checkpoint-probe` run found the ant model's antenna tips
to be its weakest keypoints, which is exactly where that lost resolution would show.

The collaborator trained both their ant and fly models at square `256x256`. Treating
that as a coincidence is not credible.

So the goal is **models that fit our animals**, not merely compatibility. Loading
external checkpoints first-class falls out of the same change for free.

## Non-goals

- No change to the default. `DEFAULT_GEOMETRY` stays `(192, 256)`, so every existing
  code path and the ~25 test files that hardcode 256x192 behave identically.
- No GUI work in Slice 1. The size is reachable from `run.json`; the PoseKit control
  and dataset auto-measurement are Slice 2.
- No model-registry unification. Per-checkpoint geometry is metadata that the
  deferred registry work will eventually own; see Sequencing.
- No quantitative re-evaluation of the collaborator's checkpoints. That is separate
  follow-up work described in `tools/vitpose/external_ckpt/FINDINGS.md`.

## Parity position

Verified: **no tracking-equivalence fixture exercises ViTPose.** All seven configs
under `tools/equivalence/fixtures/configs/` set `pose_model_type` to `sleap` or
`yolo`. The byte-identical tracking gates therefore do not cover this code, and this
change cannot regress them.

The gate that *does* apply is `tools/equivalence/verify_vitpose_runtimes.py`, which
compares torch against CoreML and TensorRT for ViTPose. Geometry flows into the
exported artifacts, so that check must still pass.

---

# Slice 1 — geometry becomes per-checkpoint

## 1. The value object

New file `src/hydra_suite/core/identity/pose/vitpose/geometry.py`:

```python
@dataclass(frozen=True)
class PoseGeometry:
    image_size_wh: tuple[int, int]        # (W, H), both % 32 == 0, both > 0

    @property
    def heatmap_size_wh(self) -> tuple[int, int]:   # (W // 4, H // 4)

    @property
    def patch_grid_hw(self) -> tuple[int, int]:     # (H // 16, W // 16)

    @property
    def num_tokens(self) -> int:                    # patch_grid product + 1 (cls slot)

    def to_hw(self) -> list[int]:                   # [H, W] for serialization

    @classmethod
    def from_hw(cls, hw: Sequence[int]) -> "PoseGeometry":

DEFAULT_GEOMETRY = PoseGeometry((192, 256))
```

**The heatmap stops being an independent constant.** `ClassicHead` is two stride-2
`ConvTranspose2d` layers applied to the patch grid, so its output is always
`image / 16 * 4 = image / 4`. The current pair of constants agree only by hand
(256/4 = 64, 192/4 = 48); deriving one from the other removes the possibility of them
disagreeing.

**Why multiples of 32, not 16.** Patch-16 embedding requires 16. ClassKit already
snaps classifier input sizes to multiples of 32
(`classkit/gui/dialogs/training.py:105-113`). 32 satisfies both and keeps one
convention across kits. It also keeps the heatmap dimension divisible by 8.

Validation raises `ValueError` naming the offending dimension.

## 2. Threading

`geom: PoseGeometry` becomes an explicit parameter — no module-level mutable state,
no implicit context — at these call sites:

| File | Function | Current constant use |
|---|---|---|
| `transforms.py:29` | `box2cs` | aspect ratio |
| `transforms.py:87` | `affine_matrix` | UDP warp destination |
| `transforms.py:102` | `top_down_affine` | `warpAffine` output size |
| `infer.py:45` | `decode_and_project` | heatmap decode scale |
| `heads.py:49` | `SimpleHead.forward` | interpolate target |
| `training/dataset.py:15,87,96` | `FEAT_STRIDE`, bounds mask, target gen | both |
| `training/validate.py:44` | `run_validation` | heatmap |
| `training/train.py:162` | `_write_val_overlays` | heatmap |
| `export.py:77,240` | ONNX / CoreML dummy input | image |
| `vitpose.py:31` | `build_vitpose` | must pass `img_size_hw` to `ViT` |
| `training/model_setup.py:25-30` | `build_finetune_model` | same |

`config.py` keeps `IMAGE_SIZE_WH` and `HEATMAP_SIZE_WH` as aliases derived from
`DEFAULT_GEOMETRY`, so existing tests and any external reader keep working.

`ViT.__init__` already accepts `img_size_hw`; today neither production constructor
passes it, so the `(256, 192)` default is the only live path. Both constructors must
now pass geometry through.

### SimpleHead

`SimpleHead` hardcodes `F.interpolate(size=(64, 48))`, so it currently cannot follow a
new geometry; `ClassicHead` scales naturally. `SimpleHead` takes the geometry at
construction and interpolates to `geom.heatmap_size_wh`. Both head types then honour
geometry uniformly.

## 3. `pos_embed` interpolation — the enabling piece

`training/model_setup.py:44` `load_finetune_init` passes `backbone.pos_embed` straight
into `load_state_dict`. PyTorch raises on any token-count difference even under
`strict=False`, and that is re-raised as `CheckpointKeyError`. **So today it is
impossible to fine-tune at any size other than the pretrained checkpoint's.** A COCO
ViTPose-B carries 193 tokens (12x16 grid); square 256x256 needs 257.

New helper, `vitpose/pos_embed.py`:

```python
def resize_pos_embed(pos_embed: Tensor, src_grid_hw, dst_grid_hw) -> Tensor
```

Splits the leading cls slot from the patch grid, reshapes the grid to
`(1, D, gh, gw)`, resizes bicubic with `align_corners=False`, flattens back and
re-concatenates the cls slot. Returns the input unchanged when source and target
grids are equal, so the default path is bit-for-bit untouched.

`load_finetune_init` calls it when the incoming token count differs from the model's.

### Recovering the source grid

A bare token count does not determine `(gh, gw)`: 256 patches could be 16x16 or 8x32.
Resolution order, and it must be exactly this:

1. If the checkpoint stores `input_size`, use it. Authoritative.
2. Else if `N` is a perfect square, assume square. (257 tokens -> 16x16 — the
   collaborator's checkpoints.)
3. Else if `N` factors to the default 0.75 aspect, use that. (193 -> 12x16 — every
   upstream ViTPose release.)
4. Else raise `ValueError` naming the token count and asking for an explicit
   `input_size`. **Never guess past this point** — a wrong grid silently produces a
   plausible model that is subtly wrong everywhere.

## 4. Checkpoint metadata

Serialized key is `input_size`, value `[H, W]` — the same name and the same H,W order
as the classifier stack (`core/identity/classification/backend.py:82-97`, which
explicitly rejects `[W, H]`). Internally geometry stays `(W, H)`; conversion happens
only in `PoseGeometry.to_hw` / `from_hw`.

- `training/train.py:131-139` payload gains `"input_size": geom.to_hw()`, where `geom`
  is resolved once at the top of `train()` from `cfg.input_size` (section 6),
  falling back to `DEFAULT_GEOMETRY` when it is `None`, and then threaded to every
  consumer rather than re-derived.
- `adapter.FinetuneMeta` gains `geometry: PoseGeometry`.
- `adapter.load_finetuned_checkpoint` resolves geometry by the order in section 3,
  then builds the model at that geometry. The collaborator's 256x256 checkpoints
  therefore load with no flags and no caller changes.

`adapter._infer_variant` already reads `pos_embed` and uses only `shape[-1]`
(embed dim); `shape[1]` — the token count that carries the size signal — is sitting
right there unused. That is where inference hooks in.

## 5. Export signature

`backends/vitpose.py:151-153` previously produced
`vitpose-v1|{flavor}|opset17|fp32|{fingerprint}`. Geometry changes the exported graph, so
a naive read of "the signature must discriminate on everything that changes the graph"
suggests putting geometry in the signature directly:

```
vitpose-v2|{flavor}|{H}x{W}|opset17|fp32|{fingerprint}
```

**As implemented, the signature does NOT carry geometry.** It is:

```
vitpose-v2|{flavor}|opset17|fp32|{fingerprint}
```

This was a deliberate decision, reviewed and upheld during implementation, not an
oversight relative to the original plan above. The signature is computed on every cache
probe, before the checkpoint is loaded — adding geometry to it would force a full
`torch.load` of the (up to ~1 GB) checkpoint on every probe just to read `input_size`,
for no discriminating power: geometry is a deterministic function of the checkpoint's
bytes, and `path_fingerprint_token` (resolved path + `mtime_ns` + size) already
identifies that exact file. Two different geometries can only come from two different
checkpoint files, which the fingerprint already distinguishes.

The `v1 -> v2` recipe-tag bump is retained and still does the invalidation work: it
forces every pre-existing (v1-signed) artifact to rebuild exactly once, since v1
artifacts were built from a process-wide-constant geometry assumption that no longer
holds. Without that bump, a CoreML package compiled at 192x256 under the old recipe
could be silently reused post-migration — wrong numbers, no error. After that one-time
invalidation, the fingerprint alone keeps artifacts correctly scoped per checkpoint file
without re-reading checkpoint bytes on every probe.

`backends/vitpose.py:82` `preferred_input_size` returns
`max(geometry.image_size_wh)` instead of the constant. It has no production call site
(only the `types.py:69` Protocol, also implemented by the yolo and sleap backends, and
one test), so the Protocol and the other two backends are deliberately left alone —
removing it is a separate cleanup, not this slice's business.

## 6. Training config

`training/config.py` `RunConfig` gains `input_size: list[int] | None = None`, where
`None` means `DEFAULT_GEOMETRY`. The duplicated `_FIELDS` whitelist at `config.py:8-26`
must gain the same key or `validate_run_config` rejects the `run.json`.

`validate_run_config` also validates it: a 2-element list of positive multiples of 32.

## 7. Error handling

| Condition | Behaviour |
|---|---|
| Geometry dim not a positive multiple of 32 | `ValueError` naming the dimension |
| `run.json` `input_size` malformed | `ValueError` from `validate_run_config` |
| Token count not resolvable to a grid | `ValueError` asking for explicit `input_size` |
| Fine-tune init across differing grids | Resize silently; log source and target grid |
| Checkpoint state/geometry shape mismatch after resize | Existing `CheckpointKeyError` |

Nothing falls back to a default on ambiguity. A wrong geometry is not a degraded
result, it is a silently wrong one.

## 8. Testing

New:
- `PoseGeometry` validation, heatmap/grid/token derivation, `to_hw`/`from_hw` round-trip.
- `resize_pos_embed`: identity when grids match; correct output shape on 193 -> 257;
  a model built at the target geometry forwards successfully after the resize.
- Source-grid resolution: 257 -> 16x16, 193 -> 12x16, an ambiguous count raises.
- `adapter` returns the right geometry for a stored value and for an inferred one.
- Export signature carries the `v1 -> v2` recipe-tag bump (so every pre-migration
  artifact rebuilds once) but deliberately does NOT encode the geometry itself — see
  Section 5's rationale. Tests assert the tag appears in an actually generated
  signature, and that `export_onnx`/`export_coreml` honour a non-default `geom` kwarg
  by asserting the exported graph's spatial input dims, independent of the signature.
- `SimpleHead` emits `geom.heatmap_size_wh` at a non-default geometry.

Unchanged: the ~25 existing test files that assert 256x192 / 64x48 continue to pass
untouched, because `DEFAULT_GEOMETRY` is unchanged. That is the regression net.

**End-to-end proof.** The probe tool in `tools/vitpose/external_ckpt/` already produces
known-good coordinates from the collaborator's 256x256 checkpoint via its own local
geometry. Slice 1 is correct when `load_finetuned_checkpoint` loads that same file with
no special flags and reproduces those coordinates. Marked skip-if-absent, since the
checkpoint is 1 GB and not in the repo. This is why Slice 1 builds on the probe branch
rather than a fresh one.

**Gate before merge:** `tools/equivalence/verify_vitpose_runtimes.py` must pass.

---

# Slice 2 — auto-sizing from the dataset (notes)

To be implemented on this same branch before merge. Recorded here so Slice 1's
interfaces anticipate it.

**Measurement lives outside the GUI.** Follow the DetectKit precedent
(`training/sliced_dataset.py:51-56` measures, the GUI only consumes), not the ClassKit
one, which duplicates the same logic in two GUI files
(`classkit/gui/dialogs/training.py:117-160` and
`classkit/gui/main_window.py:3348-3395`).

New `src/hydra_suite/training/pose_geometry_measure.py`:
`measure_pose_geometry(dataset_dir) -> MeasuredPoseGeometry` reading the COCO
`annotations.json` the ViTPose trainer already consumes
(`training/dataset.py:27`). Per instance, take the keypoint bounding box; report
median aspect ratio and median longest side, plus the sample count.

Recommendation: long side = median longest side x detail factor, short side from the
median aspect, both snapped to a multiple of 32 and clamped `[64, 512]`.

**Stamping and flow**, mirroring the established convention:
`measured_input_size` into the dataset manifest -> auto-fill the training setting only
when it is unset (DetectKit's `populate_measured_reference`,
`detectkit/gui/models.py:127-137`, never overwrites a user value) -> `RunConfig.input_size`
-> checkpoint `input_size` -> `adapter` at inference.

**GUI**: `posekit/gui/dialogs/training.py` — the ViTPose group at `:688-702` gains a
size control plus an "Auto from dataset" action and a Rescale spin box
(x0.25-4.0, step 0.25 — ClassKit's `_auto_size_scale_spin`,
`classkit/gui/dialogs/training.py:848-856`), then threading through
`_start_vitpose_training` `:1560-1626`, `ViTPoseTrainingWorker.__init__` `:311-341`,
and its `params` dict `:365-373`. PoseKit currently has no auto-sizing at all — its
`imgsz` spin box at `:737-740` is a bare default of 640.

---

# Sequencing

The deferred model-registry unification
(`docs/superpowers/specs/2026-07-29-model-registry-unification-design.md`) is now
unblocked and will eventually own checkpoint metadata. `input_size` is exactly such a
field. This spec deliberately ships first and keeps the field small and conventional —
one key, `[H, W]`, matching the classifier stack — so the registry work absorbs it
later without redesign.

# Risks

| Risk | Mitigation |
|---|---|
| Wide mechanical diff across ~11 files | Default unchanged; existing tests are the regression net |
| Ambiguous grid recovery from token count | Explicit resolution order; raise rather than guess |
| Stale exported artifacts reused at the wrong shape | Geometry in the signature + `v1 -> v2` tag bump |
| Bicubic `pos_embed` resize degrades fine-tune quality | Standard ViT practice; identity when grids match, so the default path is unaffected |
| `simple`-head models silently ignore geometry | `SimpleHead` takes geometry; covered by a test at non-default size |
