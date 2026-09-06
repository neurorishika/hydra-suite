# SAM3 LoRA training run — improved_ant_detection / obb-6609f9b105

**Status: IN PROGRESS** (this file is updated as the run proceeds; final status at the bottom).

Date: 2026-09-06. Training box: `mehek` (RTX 6000 Ada, 49 GB). Eval box: `courtship` (RTX 4090).
Code: `~/sam3_train_wt` on mehek, pinned `b9e92bc7`; `~/sam3_eval_wt` on courtship, `git archive` of the
same commit (`PINNED_COMMIT` file records it). `~/hydra-suite` on mehek was NOT touched.

## 1. Dataset rebuild on mehek — 1871 SURVIVED

Source rsynced to `mehek:~/sam3_lora_runs/ant_2026-09-06/source/obb-6609f9b105`.
Pre-build check on mehek: 79 `.txt` files (78 labels + `classes.txt`), **1871 non-empty label lines**.

`--prepare-only` build manifest diffed against the local one. The ONLY differences:

| key | local | mehek |
|---|---|---|
| `created_at` | 2026-09-06T01:01:59 | 2026-09-06T02:02:53 |
| `source` | mac path | mehek path |
| `reference_body_px` | 97.13015937805176 | 97.13015747070312 |

Everything else byte-identical, including:

- `fragment_counts.train` = tiles 486, annotations 3363, fragment_annotations 21, downgraded_tiles 20, fragment_only_tiles 1
- `fragment_counts.valid` = tiles 147, annotations 1183, fragment_annotations 14, downgraded_tiles 14, fragment_only_tiles 0
- `tile_px` = [1766, 1766], `tile_overlap` 0.25, `seed` 42, `selected_class` "ant"
- `frame_split.valid` = exactly the 16 held-out stems (same set)

The `reference_body_px` delta is 1.9e-6 px — a float32 rounding difference between the macOS and Linux
measurement path. It does not change `tile_px`, so the tiling is identical, which the identical tile and
annotation counts confirm. **The build is deterministic and reproduced exactly, so the local 1871-survival
proof transfers.** Total annotations 3363+1183 = 4546, the same overlap-duplicated total as locally.

Three separate prepared datasets were built during this session (prepare-only, the failed first launch, and
the real launch); all three manifests carry identical counts.

## 2. Preflight on mehek — ADMITTED

`assess_preflight` run under `hydra-cuda` with `PYTHONPATH=~/sam3_train_wt/src`:

```
admitted: true
refusals: []
warnings: []
estimator_version: sam3-lora-streaming-v2
```

| | bytes | GiB |
|---|---|---|
| accelerator peak (phase `training`) | 13,275,856,896 | 12.364 |
| accelerator steady (training) | 8,863,616,000 | 8.255 |
| accelerator peak (phase `model_load`) | 8,589,934,592 | 8.000 |
| host peak (phase `model_load`) | 9,692,834,633 | 9.027 |
| host steady (training) | 6,654,063,433 | 6.197 |
| disk transient | 91,227,136 | 0.085 |
| containment soft host | 10,862,817,796 | 10.116 |
| containment hard host | 12,344,111,132 | 11.496 |

CUDA device observed: RTX 6000 Ada, free 50,387,222,528 B (46.9 GiB) of 51,527,024,640 B. Free disk 1.99 TB.
Dataset profile accepted: train_tiles 486, validation_tiles 147, validation_present true, train_instances
3342, max_active_instances_per_tile 24, polygon_count 4546, polygon_vertices 54150. Limits: batch_size 1,
workers 0, prefetch_batches 0, tiles 1, candidates 24.

No refusal was bypassed — none fired.

## 3. Launch

(pending — filled in below)

## 3. Launch

First launch **FAILED** immediately after dataset prep:

```
[semantic_sam3] EnvironmentLocationNotFound: Not a conda environment: /home/rutalab/mambaforge/envs/hydra-sam3
```

`resolve_sam3_env` falls back to `DEFAULT_SAM3_ENV = "hydra-sam3"`; mehek's sidecar env is named `sam3-lora`.
Fixed by exporting `HYDRA_SAM3_ENV=sam3-lora` in the unit. This is a box-naming difference, not a code
defect, and the error was loud and immediate — no silent fallback. No code change was needed.

Working launch:

```bash
systemd-run --user --unit=sam3-train-ant -p MemoryMax=60G -p MemorySwapMax=0 -p TasksMax=4096 \
  /usr/bin/bash -c "source ~/mambaforge/etc/profile.d/conda.sh && conda activate hydra-cuda && \
  export PYTHONPATH=/home/rutalab/sam3_train_wt/src KMP_DUPLICATE_LIB_OK=TRUE HYDRA_SAM3_ENV=sam3-lora && \
  cd ~/sam3_lora_runs/ant_2026-09-06 && exec python -u -m hydra_suite.detectkit.cli \
  --config ~/sam3_lora_runs/ant_2026-09-06/training_plan.json > ~/sam3_lora_runs/ant_2026-09-06/train.log 2>&1"
```

`publish.auto_import` was flipped to `true` in the mehek copy of the plan (the prep report left this
decision to the run owner; publishing is part of this brief).

### The 60 GB cap was redundant — the pipeline caps itself harder

The trainer does NOT run inside the `sam3-train-ant` cgroup. `process_supervisor` moves the sidecar into its
own transient scope, and that scope carries limits derived from preflight's containment numbers:

```
hydra-job-c6ce34e9548640748cee453e176add37.scope
  MemoryHigh=10,862,817,796   MemoryMax=12,344,111,132   MemorySwapMax=0   TasksMax=512
```

i.e. a hard 11.5 GiB host cap with swap disabled, matching `containment_hard_host_bytes` exactly. The
"~128 GB on a 125 GB box" failure mode is structurally prevented by the product itself, not just by the
outer `systemd-run` flag. Worth knowing: the outer `MemoryMax=60G` would never have fired.

## 4. Training

Startup lines (the adapter-surface confirmation):

```
Patched vitdet.addmm_act with a grad-safe eager equivalent.
Injected LoRA adapters into 312 Linear modules.
Verified LoRA-only optimizer scope: 11,403,392 parameters in 624 tensors across 312 adapters.
training shape: 486 tiles, 1944 datapoints, batch=1 grad_accum=8, 1944 micro-batches/epoch,
                ~243 steps/epoch x 10 epochs
```

**312 modules confirmed at runtime** (the figure the prep report could only quote). The dataloader is
non-empty — 486 tiles / 1944 datapoints — so the zero-init-LoRA no-op trap does not apply.

### VRAM — the probe under-measured; a finding in its own right

The `vram_peak=` printed each logged step is `torch.cuda.max_memory_reserved()` (`cli.py::_peak_vram_gib`).
That is the **same metric** `tools/sam3/measure_bf16_peak.py` records as `reserved_peak_bytes` and the same
one `preflight.py::_MEASURED_BF16_DEVICE_PEAK_BYTES = 12 GiB` gates against — so the comparison below is
exactly apples-to-apples. It is a never-reset running maximum, so every new value is a real high-water mark.

Observed new maxima (step at which each first appeared):

| step | new max vram_peak |
|---|---|
| 1 | 7.60 GiB |
| ~30 | 8.89 GiB |
| ~70 | 10.22 GiB |

(table completed at end of run)

### *** ADMISSION-GATE BREACH — observed VRAM exceeded the ESTIMATE, not just the constant ***

At **epoch 1 step 320** `vram_peak` reached **12.99 GiB**. The arithmetic, exactly:

| quantity | bytes | GiB |
|---|---|---|
| `_MEASURED_BF16_DEVICE_PEAK_BYTES` (the envelope constant) | 12,884,901,888 | 12.000 |
| + LoRA/optimizer delta + dense device masks | 390,955,008 | 0.364 |
| **= preflight `training.accelerator_peak_bytes` (the whole estimate)** | **13,275,856,896** | **12.364** |
| **observed `max_memory_reserved` at epoch 1 step 320** | **~13,947,906,294** | **12.99** |

So the breach is not merely of the 12 GiB base term — **the composed admission estimate itself is exceeded,
by ~0.63 GiB (5.1%), and the run is only at step 320 of ~2430.** The 7.72 GiB the 30-step probe recorded is
**41% below** the observed peak.

What this does and does not mean, precisely:

- Admission applies `accelerator_safety_fraction = 0.85`, so preflight demands `12.364 / 0.85 = 14.55 GiB`
  free before admitting. The observed 12.99 GiB still fits under 14.55 GiB, so the safety fraction is
  currently absorbing the error. **No OOM has occurred and none is predicted on this card.**
- But the safety fraction exists to cover allocator noise, not to cover a wrong central estimate. It has
  been silently consumed: intended slack 2.18 GiB, remaining slack 1.56 GiB, and still climbing.
- If the peak reaches 14.55 GiB, a card that preflight ADMITS will OOM. That is the user's stated
  "no OOM errors" goal, and it is a shipped-product defect, not a lab curiosity.

### What the logged `loss` number actually is (mechanism, stated precisely)

`cli.py:743-748` prints `loss_window.summary()` and then calls `loss_window.reset()` **on every optimizer
step**. `_LossWindow` (cli.py:474) accumulates and then divides by `n`. So the printed value is:

> **the MEAN core loss across the `grad_accum = 8` micro-batches accumulated into that one optimizer step**

It is *not* a running window spanning multiple steps, and it is *not* a single micro-batch either. This
matters for reading the spikes: a logged value of 49.5 is an average of 8 micro-batches, so either several
micro-batches in that window scored badly or one scored ~8x worse than the window mean — a lone mild
outlier cannot produce it.

The docstring explains why the mean is used at all: logging the last micro-batch of the window always lands
on the same query TYPE (negatives are interleaved one per tile), which previously made a healthy run read
as a total collapse with all terms at 0.0000.

### Loss spikes are classification-driven, and they are diminishing

The bimodal high/low alternation is entirely `loss_ce`; the geometry and mask terms barely move:

```
step 640 (spike)  loss 49.53  loss_ce=0.0740  giou=0.0295  dice=0.0206
step 650 (normal) loss  6.59  loss_ce=0.0060  giou=0.0150  dice=0.0165
step 700 (spike)  loss 38.76  loss_ce=0.0648  giou=0.0364  dice=0.0331
step 710 (normal) loss  6.26  loss_ce=0.0054  giou=0.0214  dice=0.0172
```

`loss_ce` moves ~10x on spike steps. This is the presence/scoring head mis-scoring particular batches, not
geometry or mask quality degrading. At `batch=1 x grad_accum=8` that level of variance is unremarkable.

Per-epoch statistics (over the logged steps — the trainer logs step 1,2,3 then every 10th, so these are a
systematic 1-in-10 sample of ~243 steps/epoch, n per epoch shown):

| epoch | n | mean loss | median | min | max | steps >= 20 | mean loss_ce | mean dice | mean giou | vram max | skipped |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 27 | 34.05 | 26.40 | 5.38 | 95.80 | 14 (51.9%) | 0.0422 | 0.0398 | 0.0550 | 10.22 | 0 |
| 1 | 24 | 29.20 | 22.11 | 4.63 | 79.84 | 15 (62.5%) | 0.0418 | 0.0238 | 0.0281 | 12.99 | 0 |
| 2 | 24 | 19.08 | 8.51 | 4.81 | 49.53 | 10 (41.7%) | 0.0266 | 0.0190 | 0.0238 | 12.99 | 0 |

**Epoch means are falling monotonically (34.05 -> 29.20 -> 19.08), medians faster still
(26.40 -> 22.11 -> 8.51), and the proportion of high-loss batches is falling after epoch 1
(51.9% -> 62.5% -> 41.7%) alongside `mean loss_ce` (0.0422 -> 0.0418 -> 0.0266).** The scoring head is
learning the hard cases rather than plateauing on them.

Relevance, without overclaiming: the scoring/classification head is exactly the component the earlier parity
programme ranked as the top suspected cause of our precision behaviour (`dot_prod_scoring`), and
`adapt_scoring_head` is the scope deliberately left OFF for this run for lack of a measured sizing
coefficient. `loss_ce` being the noisy, slowest-converging term is therefore a reason to consider testing
that scope later — but this is a training-curve observation only and says nothing about held-out accuracy.
The held-out evaluation is the verdict.

## 5. Evaluation staging (prepared during training; run after it finished)

**Evaluation was moved to mehek, not courtship.** A matched run is in flight on courtship under
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (same corpus, same 2430-step horizon, verified identical
dataset manifest) — courtship's GPU was measured at 96% util / 7980 MiB, so it was left alone. This run is
therefore the **default-allocator arm** of that controlled pair; the VRAM ladder below is one half of it.

Staged onto mehek (read-only pulls from courtship, routed via the Mac since mehek->courtship ssh has no
host key; no GPU work on courtship):

| artifact | mehek path | check |
|---|---|---|
| held-out tiles + COCO | `~/sam3_eval/data/valid` | 576 images, 805 annotations, exactly the 16 held-out stems — identical corpus to the pre-registered baseline |
| 2026-09-04 reference ckpt + sidecar | `~/sam3_eval/refs/20260904-143950_semantic_sam3_ca1bd031.pt` | md5 `97288cddb7a9ef203f062c64d16008e1` matched against courtship's copy |
| spike ckpt | `~/sam3_spike/out/sam3_ant_finetuned.pt` | md5 `5d41842564f638b3e0b1b5579afa46f3`, **identical to courtship's copy** — so the artifact named in the brief is the one evaluated |

Corrected matcher confirmed present in the pinned tree on mehek
(`~/sam3_train_wt/src/hydra_suite/core/inference/semantic/calibration.py`): ranks by match QUALITY, keeps the
containment gate, tests `representative_point` rather than the vertex mean, and the IoU second admissibility
route is REMOVED (it survives only in explanatory comments). Its docstring records the measured effect with
predictions held fixed: recall 0.867 -> 0.988, extras/tile 0.203 -> 0.035.

Environment dry-run on mehek `hydra-cuda` (imports only, no GPU allocation while training held the card):
`compare_models.py --help` exits 0; ultralytics 8.4.41, numpy 2.4.3 (`np.trapezoid` present, so the 1.x
fallback is not exercised), `SAM3SemanticPredictor` imports, `scikit-learn`/`shapely` present,
`hydra_suite.__file__` = `~/sam3_train_wt/src/hydra_suite/__init__.py`.

`tools/sam3_parity/baseline.json` is NOT written: `--out` is overridden to new files under
`~/sam3_eval/results/`. Every other argument is copied verbatim from `baseline.json`'s recorded values
(`prompt=ant`, `reference_body_px=97.13015747070312`, `tile_fraction` omitted/None, `seam_margin_px=8`,
`merge_iou=0.5`, `compare_confidence=0.5`, `target_recall=0.9`) so that the matcher is the only difference
from the pre-registered run.

## 6. Checkpoint ladder (added mid-run, to test the last-epoch selection default)

`cli.py:20` selects the LAST epoch's weights, always. `cli.py:267` sets `KEEP_EPOCH_CHECKPOINTS = 3`.
Because the epoch means stalled after epoch 2, a ladder of per-epoch checkpoints was preserved so that
last-epoch selection can be tested against held-out metrics rather than assumed.

**`epoch_001.pt` and `epoch_002.pt` were ALREADY PRUNED by the time this was set up** — only
`epoch_003.pt`, `epoch_004.pt`, `epoch_005.pt` remained. Stated rather than silently substituted. Those
three were copied to `~/sam3_eval/ladder/` (outside the pruning path) and a 30-second copy loop
(`~/sam3_eval/preserve.sh`) now preserves every subsequent epoch checkpoint.

Epoch checkpoints are **adapters only (45,820,383 bytes)**. `Sam3SemanticLabeler.from_variant` needs a
merged artifact plus its `.sam3_meta.json` sidecar (the load guard reads the sidecar), so each rung must be
published before it can be evaluated. That is done with `publish_sam3_model`, mirroring
`training/service.py:184`'s SEMANTIC_SAM3 branch, but with `models_root` pointed at an isolated
`~/sam3_eval/ladder_models` so the real registry is not polluted with ladder rungs.

### Finding: publish is serialized behind the host-memory lease

Attempting a ladder publish while training was still running was REFUSED:

```
ResourceBusyError: Resource 'mehek:host-memory' is already leased by PID 2791995
Sam3PublishError: SAM3 publish sidecar launch refused
```

This is correct, intentional behaviour — one heavy host job at a time — and it is a loud refusal, not a
silent degradation. Consequence for scheduling: ladder publishes cannot be overlapped with training to save
wall-clock; they must run after the run releases its lease. Publishing itself is CPU-only
(`CUDA_VISIBLE_DEVICES=""`), so it does not contend for the GPU.

## 7. Training outcome — COMPLETED, all 10 epochs

```
progress 10/10
adapters.pt          45,820,383 bytes
adapters.pt.complete.json  sha256 738749e2dfe25d7871089ddc5b0361c6f3df4105ece133fc16b360ae41b8a60d
val_stats.json       {"val_loss_mean": 0.9056838208355572, "val_batches": 588,
                      "note": "informational only; checkpoint selection is always 'last'"}
```

Wall clock 02:04:37 -> 04:17 EDT, ~2h12m for 2430 optimizer steps. Training worker exit code **0**.

- **Skipped steps: 0** across all 2430 optimizer steps. No non-finite loss, no gradient skip, so the
  `MAX_CONSECUTIVE_SKIPPED_STEPS` guard never engaged.
- **Dataloader non-empty**: 486 tiles / 1944 datapoints / 588 validation batches — the zero-init-LoRA no-op
  trap (a "successful" run publishing a checkpoint byte-identical to stock SAM3) does not apply.
- No OOM, no exit 137, no hang.

Per-epoch loss (1-in-10 systematic sample of ~243 steps/epoch; the trainer logs step 1,2,3 then every 10th,
so n ~ 24-27 per epoch; the distribution is strongly bimodal so these means carry a wide error bar):

| epoch | n | mean | median | min | max | steps >= 20 | mean loss_ce | vram max | skipped |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 27 | 34.05 | 26.40 | 5.38 | 95.80 | 51.9% | 0.0422 | 10.22 | 0 |
| 1 | 24 | 29.20 | 22.11 | 4.63 | 79.84 | 62.5% | 0.0418 | 12.99 | 0 |
| 2 | 24 | 19.08 | 8.51 | 4.81 | 49.53 | 41.7% | 0.0266 | 12.99 | 0 |
| 3 | 25 | 19.68 | 12.30 | 3.63 | 49.99 | 32.0% | 0.0267 | 12.99 | 0 |
| 4 | 24 | 19.85 | 13.81 | 5.49 | 60.14 | 41.7% | 0.0273 | 12.99 | 0 |
| 5 | 24 | 20.85 | 15.30 | 3.73 | 56.87 | 41.7% | 0.0281 | 12.99 | 0 |
| 6 | 25 | 19.31 | 9.84 | 4.21 | 70.64 | 40.0% | 0.0264 | 12.99 | 0 |
| 7 | 24 | 18.13 | 9.80 | 3.38 | 48.62 | 41.7% | 0.0249 | 12.99 | 0 |
| 8 | 24 | 19.96 | 11.61 | 2.86 | 63.11 | 41.7% | 0.0279 | 12.99 | 0 |

**Learning is visible for epochs 0-2 (34.05 -> 29.20 -> 19.08) and then stops.** Epochs 2-8 oscillate in a
flat band of 18.1-20.9 with no trend in either direction; `mean loss_ce` is flat at 0.025-0.028 and the
high-loss-batch fraction is pinned at 40-42% for six consecutive epochs. Nothing after epoch 2 is
distinguishable from sampling noise. That is what motivated the checkpoint ladder in section 6.

### Final VRAM ladder (the controlled pair's default-allocator arm)

| new maximum | first appeared at |
|---|---|
| 7.60 GiB | epoch 0, step 1 |
| 8.89 GiB | epoch 0, step 30 |
| 10.22 GiB | epoch 0, step 70 |
| 11.59 GiB | epoch 1, step 310 |
| **12.99 GiB (final)** | **epoch 1, step 320** |

**The peak last moved at step 320 and then held for 2110 consecutive steps (8.7 epochs) to the end of the
run.** So it does plateau — but only after crossing the admission estimate, and only after a 230-step flat
stretch at 10.22 GiB that spanned a whole epoch boundary and would have fooled any flatness-based rule.

Final honest numbers: **full-run peak 12.99 GiB**, versus a 30-step probe reading of **7.72 GiB** (41% low)
and a preflight estimate of **12.364 GiB** (5.1% low). No OOM occurred, because the 0.85 safety fraction
demands 14.55 GiB free and 12.99 fits under it — the error was absorbed by margin that exists for other
reasons.

### Recommended stopping rule for `tools/sam3/measure_bf16_peak.py`

The probe currently stops after `--optimizer-steps` (default 5; the recalibration used 30). Both are far too
early, and the failure is not "N was too small" but "step count is the wrong variable". Evidence: the peak
was still flat at 10.22 GiB at step 300 and jumped twice within the next 20 steps.

Root cause of the growth is almost certainly the tile with the most simultaneous instances —
`max_active_instances_per_tile` is 24 and preflight already models device peak as a function of it — so the
peak is reached when the loader first draws a worst-case tile, and with `batch=1` over 486 shuffled tiles
that can land anywhere in the first few hundred steps.

Recommended, in preference order:

1. **Stop on dataset coverage, not step count: run at least one full epoch** (486 optimizer-step-equivalents
   here). This is the only rule that guarantees every tile — including the worst-case one — has been seen.
   Cost here would have been ~13 minutes, against a 4.3-hour run.
2. **If a full epoch is too expensive, order the probe's tiles by descending instance count** and run the top
   K (say K=16 covering the largest tiles), rather than sampling the shuffled order. This targets the
   variable that actually drives the peak, and would likely find the true maximum in far fewer steps than
   random order.
3. **Never accept a plateau as evidence.** Require that the peak has not moved for a stated number of steps
   AND that dataset coverage is complete. A pure flatness test would have stopped at step 300 with 10.22.

Also worth noting: the probe records `reserved_peak_bytes` from `torch.cuda.max_memory_reserved()`, which is
exactly the metric the training loop logs and the metric admission is written against — that part of the
methodology is sound. Only the stopping rule is wrong.

**No source constant was edited.** `_MEASURED_BF16_DEVICE_PEAK_BYTES` is left at 12 GiB; the recommendation
is that a corrected measurement be persisted through `runtime/memory_profiles.py` (as
`measure_bf16_peak.py` already does via `MemoryProfileStore`), re-measured under rule 1 above.

## 8. Publish — FAILED, root-caused, FIXED (a real pipeline defect)

After the completed 10-epoch run, publish died. `training_result.json` recorded
`"success": false`, `published_registry_key: ""`, `published_model_path: ""` — the training worker itself
exited 0 and `adapters.pt` was safely on disk, but **no artifact, no sidecar, no registry entry**.

```
Worker exited with code 130. Child output tail:
OpenBLAS blas_thread_init: pthread_create failed for thread 31 of 32: Resource temporarily unavailable
OpenBLAS blas_thread_init: RLIMIT_NPROC 513222 current, 513222 max
...
  File ".../training/sam3_lora/publish_worker.py", line 31, in <module>
    from hydra_suite.utils.sam3_constants import PREDICTOR_IMGSZ
  File ".../hydra_suite/utils/__init__.py", line 8, in <module>
    from .frame_prefetcher import FramePrefetcher
  File ".../hydra_suite/utils/frame_prefetcher.py", line 14, in <module>
    import cv2
KeyboardInterrupt
```

Reproduced standalone (so: deterministic, not a transient), then isolated with four controlled tests:

| test | result |
|---|---|
| `conda run`, no containment | **OK** (cv2 imports in 0.1 s) |
| direct python, `TasksMax=64` | **OK** |
| **`conda run` + `TasksMax=64`** | **FAILS — `pthread_create failed for thread 31 of 32`, `ImportError: numpy._core.multiarray failed to import`, segfault** |
| `conda run`, `TasksMax=512` | **OK** |

**Root cause:** `publish.py:54` set `MAX_PROCESSES = 64`. The publish sidecar is launched through `conda run`,
and conda's process tree plus numpy/OpenBLAS's core-count-sized thread pool (32 on this box) exhausts a
`pids.max` of 64 before the worker finishes importing. Training's own scope runs at `TasksMax=512` and is
unaffected — which is why 4.3 hours of training succeeded and the 3-minute publish could never have.

**Fix:** `MAX_PROCESSES = 64 -> 512`, with the measurement recorded in-comment. Committed locally by explicit
path with `-n` as **`9c50a40b`** (NOT pushed). With the fix applied, publish succeeded on the first attempt.

### Contributing defect (not fixed — reporting only)

`utils/sam3_constants.py` opens with: *"Deliberately a dependency-free leaf module: it imports nothing. The
SAM3 training sidecar runs in a minimal conda env with no numba, sklearn, cv2 or ultralytics..."* — but
`hydra_suite/utils/__init__.py:8` eagerly does `from .frame_prefetcher import FramePrefetcher`, which imports
`cv2`. Importing that one integer therefore executes the whole heavy graph, defeating the module's stated
design intent and putting OpenBLAS in the crash path in the first place. It only works at all on this box
because its `sam3-lora` env happens to have cv2, contrary to the docstring's assumption. Fixing this would
independently have prevented the crash.

### Context: publish is the least-exercised path in the pipeline

`publish.py:378` carries a comment from a previous incident: *"Publishing therefore failed for EVERY SAM3 run
with 'must be numeric'; the pipeline had never reached publish before, so nothing caught it."* This run found
the next defect in the same stretch of code. Publish is reached only after a multi-hour training run
succeeds, so it gets almost no coverage — worth a fast synthetic publish test with a tiny adapter.

## 9. Published artifacts — all three present and verified

| artifact | value |
|---|---|
| `.pt` | `~/.local/share/hydra-suite/models/sam3_finetuned/20260906-020455_semantic_sam3_ee10d3d0.pt` (3,450,158,878 B) |
| sidecar | `...ee10d3d0.pt.sam3_meta.json` (74,836 B) |
| registry | `model_registry.json` schema_version 2, key `sam3_finetuned/20260906-020455_semantic_sam3_ee10d3d0.pt` |

Sidecar stamps, both confirmed against the runtime log:

```
adapted_modules          = 312          <- matches the 312-module surface
adapter_trainable_params = 11403392     <- matches "11,403,392 parameters in 624 tensors across 312 adapters"
base_variant = sam3   imgsz = 1008   prompt = ant
object_tile_fraction = 0.055   train_tile_px = 1766   reference_body_px = 97.13015747070312
label_quality_acknowledged = True
source_fingerprint = 87e03e24554704ea3fe5bb5d7cb830fdfd71d1700745cb0f08846b72110d3ae0
```

Registry entry carries `stored_path`, `sidecar_path`, `trained_from_run_id`, `dataset_fingerprint`,
`task_family: semantic`, `usage_role: semantic_sam3`, `added_at: 2026-09-06T04:20:19`.

**Load guard passes**: `Sam3SemanticLabeler.from_variant(checkpoint=...)` returns a live labeler; the
`assert_checkpoint_loaded` guard (which compares the live state dict against the sidecar's
`tuned_fingerprints`/`stripped_keys` at `imgsz=1008`) raised nothing.

## 10. THE GEOMETRY CONFOUND — read this before any number in section 11

The 2026-09-06 run and the 2026-09-04 reference were **not trained at the same tile geometry**:

| | new (2026-09-06) | reference (2026-09-04) |
|---|---|---|
| `object_tile_fraction` | **0.055** | **0.10** |
| `train_tile_px` | **1766** | **971** |

SAM3 resizes every tile to its native 1008 px. So a 97.13 px ant is presented to each model at a different
apparent scale:

- reference: tile 971 -> 1008, ant appears at 97.13 x 1008/971 ~= **101 px**
- new: tile 1766 -> 1008, ant appears at 97.13 x 1008/1766 ~= **55 px**

The pre-registered evaluation corpus (`~/sam3_eval/data/valid`) is **971 x 971** tiles — built at the
REFERENCE model's geometry. Evaluating both models on it tests the reference at the scale it trained on and
the new model at a scale ~1.82x away from the one it trained on. **Any single-geometry comparison is
therefore confounded with a train/serve scale mismatch that handicaps exactly one arm.**

### Root cause — a process defect, not a code bug

`training/contracts.py:250` declares `object_tile_fraction: float = 0.055`. The 2026-09-06 plan did not set
the field, so dataset prep silently took the default; the 2026-09-04 reference had used 0.10. **This was
never a like-for-like retrain**, and the divergence entered through prep quietly accepting a default rather
than matching the checkpoint the run existed to be compared against.

**Could anything have warned?** No. Searching `src/` for any warning or mismatch check on
`object_tile_fraction` returns nothing, and `train_tile_px` has exactly two references in the whole tree —
`publish_worker.py:267` (which writes it) and a comment in `publish.py:378`. Nothing reads a prior
checkpoint's sidecar geometry at plan time, and nothing compares a new run's geometry against it. The
geometry IS faithfully stamped on the artifact, so the information exists; it is simply never checked.

**Recommendation:** when a run exists to be compared against a prior checkpoint, its geometry parameters
should be read from that checkpoint's `.sam3_meta.json` rather than from `contracts.py` defaults — or, at
minimum, a plan that names a comparison baseline should warn when `object_tile_fraction` / `tile_px` differ
from it.

### Provenance caveat on the reference

The 2026-09-04 sidecar's `source_fingerprint` is the placeholder string
`courtship:20260904-143950_semantic_sam3_ca1bd031`, not a dataset hash, so **same-corpus could not be proved
from metadata.** Its `reference_body_px` is identical to this run's (97.13015747070312), which is suggestive
of the same source labels, but that is not proof and is not upgraded to a claim here.

### The clean good news, independent of every confound above

**The corrected matcher is demonstrably working.** On these same 16 held-out frames, recall is **0.98**
under the corrected matcher versus **0.867** recorded in `baseline.json` under the old vertex-mean matcher.
That reproduces the fix's own predicted effect (0.867 -> 0.988) on an independent run, and it is unaffected
by the tile-geometry question, which touches the models rather than the matcher.

`tools/sam3_parity/baseline.json` was NOT modified. All results are written to new files under
`~/sam3_eval/results/`.

## 11. Held-out evaluation — 16 frames / 576 tiles / 805 instances, corrected matcher

All runs: `--prompt ant --reference-body-px 97.13015747070312 --seam-margin-px 8 --merge-iou 0.5
--compare-confidence 0.5 --target-recall 0.9`, no `--tile-fraction` — every value copied verbatim from
`baseline.json` so the matcher and the corpus geometry are the only things that differ from the
pre-registered run. Results written to `~/sam3_eval/results/*.json`; **`baseline.json` was not touched.**

Note on a naming quirk: `operating_points[*].extra_per_frame` is per-TILE despite its name (it matches the
`extras/tile` figures in `baseline.json`'s caveats). The explicitly-named `paired_extras_per_frame` and
`paired_extras_per_tile` blocks are per-frame and per-tile respectively.

**`median IoU` is NOT reported: `compare_models.py` does not emit it.** There is no such key in its output
and no code path computes one (the only IoU uses are `--merge-iou` and the matcher's internal
`match_quality`). Recording this as unavailable rather than inventing a number.

### 11a. Cross-model comparisons on the 971-px corpus — CONFOUNDED, not model claims

Read section 10 first. On this corpus the reference model is at its trained scale and the new model is
1.82x away from its own.

| | new (2026-09-06) | ref (2026-09-04) | spike |
|---|---|---|---|
| recall @ conf 0.5 | 0.9801 | 0.9801 | 0.9565 |
| extras/tile @ conf 0.5 | 0.1094 | 0.0191 | 0.0087 |
| max recall | 0.9975 @0.20 | 0.9876 @0.20 | 0.9689 @0.10 |
| AP | 0.6135 | 0.6930 | 0.6081 |
| extras/frame @ target recall 0.9 | 0.0516 | 0.0056 | 0.0035 |

- **new vs ref:** paired extras/frame **+3.25, 95% CI [2.8125, 3.75]**, sign p = 3.05e-05; per-tile
  **+0.0903, CI [0.0660, 0.1163]**, p = 9.06e-14. CI excludes zero.
- **new vs spike:** paired extras/frame **+3.625, 95% CI [3.00, 4.375]**, sign p = 3.05e-05. CI excludes zero.

Both differences are "real" by the pre-registered criterion, and both say the new model emits more extras at
equal-or-better recall. **Neither is presented as a model claim**: the tile-geometry confound handicaps the
new model in both, and for the spike the brief's own caveats apply independently (3-frame training set with
train == valid, and held-out frame `f008975` sits inside it).

### 11b. Checkpoint ladder — internally clean, and the most informative result

All four arms come from THIS run, share `object_tile_fraction = 0.055` / 1766-px training geometry, and are
scored on the same 971-px corpus. The scale handicap is therefore identical across arms and cancels. (State
that shared handicap whenever these numbers are quoted — the absolute values are depressed for all four.)

Each rung is `a`, the final checkpoint is `b`:

| rung | AP | recall @0.5 | extras/tile @0.5 | extras/frame @ recall 0.9 | paired extras/frame vs final | 95% CI | sign p |
|---|---|---|---|---|---|---|---|
| epoch_003 (end ep 2) | **0.6379** | 0.9789 | **0.0764** | **0.0232** | **-1.1875** | [-1.6875, -0.75] | 4.88e-04 |
| epoch_005 (end ep 4) | 0.5985 | 0.9839 | 0.1354 | 0.0385 | +0.9375 | [0.125, 1.75] | 0.146 |
| epoch_008 (end ep 7) | 0.6202 | 0.9814 | 0.0972 | 0.0501 | -0.4375 | [-0.6875, -0.1875] | 0.0156 |
| **final (end ep 9)** | 0.6135 | 0.9801 | 0.1094 | 0.0516 | — | — | — |

**Interpretation, stated conservatively.**

- By AP the ordering is epoch_003 (0.6379) > epoch_008 (0.6202) > **final (0.6135)** > epoch_005 (0.5985).
  The always-last-epoch default picked a checkpoint that is **third of the four sampled**.
- **The ladder is NOT monotonic in training duration.** If longer training were systematically harmful,
  epoch_005 would sit between epoch_003 and the final; instead it is the worst arm. So this is NOT evidence
  of a duration effect — it is at least as consistent with substantial epoch-to-epoch variance in the
  adapter, with epoch_003 a good draw and epoch_005 a bad one.
- Recall is essentially constant across all four arms (0.9789-0.9839, a spread of 0.005). Everything that
  moves, moves in precision.
- These are **three comparisons against a common reference, uncorrected for multiplicity**; the
  pre-registered criterion was written for a single comparison.
- **Methodological flag:** for epoch_005 the tool reports `"significant": true` (CI [0.125, 1.75] excludes
  zero) while `sign_test_p = 0.146` would not reject. The two disagree; the pre-registered rule is the CI.
  epoch_003 is the sturdier result because both agree there (CI excludes zero AND p = 4.88e-04).

**What survives:** ten epochs bought nothing measurable over three — no rung beats epoch_003, and the loss
curve independently stopped improving after epoch 2. Last-epoch selection is not obviously right, but this
data does not establish a better rule either. It is real evidence to put against `cli.py:20`'s default,
where previously there was none; it is not yet a mandate to change it.

### 11c. The 2x2 — the model effect FLIPS SIGN between geometries

Running the same pair on both corpora separates the model effect (down a column, geometry fixed) from the
geometry effect (across a row, model fixed). `a` = new (trained 1766), `b` = ref 2026-09-04 (trained 971).

|  | corpus 971 px (576 tiles) | corpus 1766 px (147 tiles) |
|---|---|---|
| **paired extras/frame (a - b)** | **+3.25**, CI [+2.81, +3.75], p = 3.05e-05 | **-2.8125**, CI [-3.9375, -1.8125], p = 1.22e-04 |
| paired extras/tile (a - b) | +0.0903, CI [+0.066, +0.116] | -0.3061, CI [-0.415, -0.204] |
| recall @0.5 — a / b | 0.9801 / 0.9801 | 0.9713 / 0.9383 |
| extras/tile @0.5 — a / b | 0.1094 / 0.0191 | 0.1361 / 0.4422 |
| extras/frame @ recall 0.9 — a / b | 0.0516 / 0.0056 | 0.0272 / 0.1298 |
| AP — a / b | 0.6135 / 0.6930 | 0.9622 / 0.9676 |

**On the pre-registered metric (paired extras/frame) the sign FLIPS: +3.25 on the 971-px corpus,
-2.8125 on the 1766-px corpus, both with CIs excluding zero. Each model wins decisively on its own training
geometry. Neither single-geometry comparison supports a model claim.**

One precision, because it matters and the flip is not uniform across metrics: **AP does not flip.** It
mildly favours the reference on both corpora (0.6135 vs 0.6930 at 971; 0.9622 vs 0.9676 at 1766) — but the
gap collapses from 0.080 to 0.005 when the new model is scored at its own geometry. So the flip is a
property of the extras metric the criterion was registered on; AP shows the same directional pull, greatly
attenuated.

**AP is not comparable across corpora.** Both models score ~0.96 at 1766 px versus 0.61-0.69 at 971 px. The
larger-tile corpus is simply an easier scoring regime (fewer, larger tiles; 147 vs 576), so the 0.96 figures
must never be compared against the 0.61-0.69 ones.

**Practical conclusion: `object_tile_fraction` dominated every model difference measured tonight.** It
entered silently as the `contracts.py:250` default (0.055) instead of being read from the reference
checkpoint's sidecar (0.10), and its effect on the measured comparison is larger than, and opposite in sign
to, whatever real difference exists between the two adapters.

### 11d. The deployment-realistic diagonal (descriptive only)

Each model scored at the geometry stamped in its own sidecar — how each would actually be served, since
TrackerKit reads `train_tile_px` back to prefill SAHI. **This is NOT a controlled comparison**: the two
arms use different corpora, so there is no paired statistic, no CI, and AP is not comparable (see above).

| | new @ 1766 px | ref 09-04 @ 971 px |
|---|---|---|
| recall @ conf 0.5 | 0.9713 | 0.9801 |
| max recall | 0.9924 @0.10 | 0.9876 @0.20 |
| extras/frame @ recall 0.9 | 0.0272 | 0.0056 |
| AP (NOT comparable across corpora) | 0.9622 | 0.6930 |

On the only two figures that are even loosely comparable — per-frame recall and per-frame extras at matched
recall — the 2026-09-04 reference at its own geometry is modestly ahead: slightly higher recall at conf 0.5
(0.9801 vs 0.9713) and ~4.8x fewer extras per frame at target recall. Treat as descriptive, not a verdict.

## 12. Artifact locations

**On the Mac** (`.../Behavior/sam3_lora_runs/ant_2026-09-06/published/`) — all md5-verified against mehek,
15/15 files matching:

```
20260906-020455_semantic_sam3_ee10d3d0.pt               62e909168c2cbcc68aa2ee3e8720070a  (3,450,158,878 B)
20260906-020455_semantic_sam3_ee10d3d0.pt.sam3_meta.json 7c09e88da83832272e109f6d39b07d6f
adapters.pt                                              37bc1f7a8d492c348b8dea1604e0d840
adapters.pt.complete.json                                0b0336784798fbe489d132690d75d1b1
val_stats.json                                           3d133535b5c5f628163130d5d7c29259
spec.json                                                87858d8f3d1ab19dd58b0642bb984729
resource_preflight.json                                  e73a65f10b670c281449db006829f545
train.log                                                346b7c2c92eb9b650c7c44c163604465
ladder_adapters/epoch_003..009.pt   (7 files, all md5-matched)
eval_results/*.json                 (6 result files + 2 run logs)
```

**Left in place on mehek** (not deleted):

| what | path |
|---|---|
| published model + sidecar + registry | `~/.local/share/hydra-suite/models/sam3_finetuned/` |
| run dir (adapters, spec, preflight, val_stats) | `~/sam3_lora_runs/ant_2026-09-06/workspace/runs/20260906-020455_semantic_sam3_ee10d3d0/` |
| preserved ladder adapters (7) | `~/sam3_eval/ladder/` |
| ladder rungs published (isolated root) | `~/sam3_eval/ladder_models/sam3_finetuned/` |
| evaluation results (6 JSON) | `~/sam3_eval/results/` |
| held-out eval corpus (971 px) + reference checkpoints | `~/sam3_eval/data/valid/`, `~/sam3_eval/refs/` |
| pinned code tree (b9e92bc7 + the publish fix) | `~/sam3_train_wt/` |

`~/hydra-suite` on mehek was never touched. Courtship was used read-only (file pulls); no GPU work ran there.

## 13. Pipeline defects found — the answer to "does SAM3 training work end to end?"

**Training: YES. Publish: NO, until fixed tonight.**

1. **`publish.py` `MAX_PROCESSES = 64` — FIXED (`9c50a40b`, local, not pushed).** 100% reproducible publish
   failure after a completed multi-hour run: `conda run` + numpy/OpenBLAS under `pids.max=64` exhausts the
   cgroup mid-import (`pthread_create failed for thread 31 of 32` -> `ImportError: numpy._core.multiarray
   failed to import` -> segfault, exit 130). Raised to 512, matching the training scope. **Nobody running
   this pipeline could have published a SAM3 model before this fix.**
2. **`utils/sam3_constants.py` is a "dependency-free leaf module" that isn't — NOT fixed, reported.**
   `hydra_suite/utils/__init__.py:8` eagerly imports `frame_prefetcher` -> `cv2`, so importing one integer
   executes the whole heavy graph, defeating the module's explicit documented purpose and putting OpenBLAS
   in the crash path of defect 1. Fixing it would independently have prevented that crash.
3. **VRAM admission is under-evidenced.** Full-run peak 12.99 GiB vs a 12.364 GiB preflight estimate (5.1%
   low) and a 7.72 GiB probe reading (41% low). No OOM, because the 0.85 safety fraction absorbed it. The
   probe's step-count stopping rule is the root cause — see section 7 for the recommended coverage-based
   rule. No constant edited.
4. **No geometry guard against a comparison baseline — process defect.** `object_tile_fraction` silently
   took its `contracts.py` default and diverged from the checkpoint this run existed to be compared with.
   Nothing in `src/` warns; `train_tile_px` is written but never read back for validation.
5. **Publish is the least-covered path in the pipeline.** `publish.py:378` documents a prior incident with
   the same signature ("the pipeline had never reached publish before, so nothing caught it"). Two publish
   defects in two attempts. A fast synthetic publish test with a tiny adapter would pay for itself.
6. **Minor / environmental:** `DEFAULT_SAM3_ENV = "hydra-sam3"` does not match mehek's `sam3-lora`
   (loud, immediate failure; fixed with `HYDRA_SAM3_ENV`). And `compare_models.py` reports
   `"significant": true` from the CI even when the sign test disagrees (epoch_005: CI [0.125, 1.75] excludes
   zero, sign p = 0.146) — the two criteria can conflict and only the CI is pre-registered.

**Status: COMPLETE.**

## 14. Postscript — the paired allocator arm

While this run was in flight, the matched arm on courtship
(`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, same corpus, same 2430-step horizon) completed, and its
result landed on local `main` as **`e65bdb2a` — "perf(sam3): set expandable_segments for the sidecar --
45% of peak VRAM was fragmentation"**.

That directly explains the finding in section 7: **this run is the default-allocator arm, and roughly 45% of
its 12.99 GiB reserved peak was allocator fragmentation rather than live tensors.** `max_memory_reserved`
counts blocks the caching allocator holds but has not returned to the driver, so a fragmenting allocator
inflates exactly the metric admission is written against.

Two consequences worth carrying forward:

1. The VRAM ladder in section 7 (7.60 -> 8.89 -> 10.22 -> 11.59 -> 12.99 GiB) should be read as the
   **default-allocator upper bound**, not as the intrinsic memory requirement of this adapter surface.
2. The recommendation in section 7 stands and gains a rider: any re-measurement persisted through
   `runtime/memory_profiles.py` must record **which allocator configuration it was taken under**, because
   the two differ by nearly a factor of two on the same workload. A single unlabelled number would be
   actively misleading.
