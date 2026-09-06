# DetectKit Headless Training

DetectKit can prepare datasets and train detector roles on a server without a
desktop session or Qt display:

```bash
detectkit train --config training.json
```

The CLI uses the same dataset preparation, validation, training runner, run
registry, and publishing services as the DetectKit training dialog. Relative
workspace, source, checkpoint, and explicit model paths are resolved relative
to the configuration file, making a plan portable with its datasets.

## Configuration

Save a JSON file like the following. Plain Ultralytics model names such as
`yolo26s-obb.pt` remain model identifiers; paths beginning with `./`, `../`, or
`/` are resolved as files.

```json
{
  "version": 1,
  "workspace": "./training-workspace",
  "sources": [
    {
      "path": "./datasets/day-1",
      "name": "day-1",
      "level": "obb"
    }
  ],
  "class_names": ["ant"],
  "species": "ant",
  "model_tag": "server",
  "dataset": {
    "split": {"train": 0.8, "val": 0.2, "test": 0.0},
    "deduplicate": true,
    "crop_pad_ratio": 0.15,
    "min_crop_size_px": 64,
    "enforce_square": true,
    "slicing": {
      "enabled": false,
      "geometry_mode": "auto_object",
      "overlap": 0.2,
      "min_area_ratio": 0.1,
      "negative_tile_fraction": 0.15,
      "target_size_fractions": [0.3125, 0.46875, 0.625],
      "full_frame_mix": true
    }
  },
  "training": {
    "device": "0",
    "seed": 42,
    "epochs": 100,
    "batch": 16,
    "lr0": 0.01,
    "patience": 30,
    "workers": 8,
    "cache": false,
    "augmentation": {
      "enabled": true,
      "args": {
        "fliplr": 0.5,
        "flipud": 0.0,
        "degrees": 0.0,
        "mosaic": 1.0,
        "mixup": 0.0,
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4
      }
    }
  },
  "roles": [
    {
      "role": "obb_direct",
      "model": "yolo26s-obb.pt",
      "imgsz": 640
    }
  ],
  "publish": {
    "auto_import": false,
    "auto_select": false
  }
}
```

Supported DetectKit roles are `obb_direct`, `detect_direct`,
`segment_direct`, `seq_detect`, `seq_crop_obb`, `seq_crop_segment`, and
`semantic_sam3`. Sequential plans list their stages in execution order. For
example, a sequential OBB plan lists `seq_detect` before `seq_crop_obb`; the
second run is linked to the first through its parent run ID.

Set each source's `level` to its native annotation fidelity: `aabb`, `obb`, or
`polygon`. Preflight validates labels according to that geometry and dataset
derivation refuses roles that require unavailable fidelity.

Publishing is disabled by default for headless plans. Successful checkpoints
remain in `<workspace>/runs/<run-id>/`. Enable `publish.auto_import` only when
the server's HYDRA model registry is the intended destination.

### Batch size

`training.batch` defaults to `16`. Setting it to `-1` means "let Ultralytics
size the batch before launch": a short, contained resolution child runs
Ultralytics' own autobatch, and the number it reports is clamped to
`[1, 64]` — `64` is the largest batch Ultralytics actually profiles rather than
extrapolates to from a linear fit.

The resolved value and where it came from land in
`<run_dir>/batch_resolution.json`:

```json
{
  "requested": -1,
  "resolved": 24,
  "provenance": "ultralytics_autobatch",
  "fingerprint": "",
  "degraded_reasons": [],
  "effective_batch": 24,
  "measured_envelope_bytes": 0,
  "free_bytes": 0,
  "resolved_at_unix_ns": 1757030400000000000
}
```

The file shares its **core keys** with the SAM3 block below, which carries
several more. `measured_envelope_bytes` and `free_bytes` stay `0` on this path
on purpose: nothing here was measured by HYDRA, so nothing claims to have been.
`degraded_reasons` lists anything that made the estimate weaker, such as a
missing train-label root.

`resolved` and `effective_batch` mean different things and can differ:

- `resolved` is what **resolution chose**, written before the run starts so a
  crash still leaves the provenance behind.
- `effective_batch` is the batch of the **final attempt** — what trained, or,
  for a run that failed every attempt, the last batch tried. It is `null`
  until the bounded OOM-retry ladder settles, at which point the file is
  rewritten with the settled value. Because that ladder halves the batch in
  a fresh child on a classified out-of-memory exit, `effective_batch` can be
  half or a quarter of `resolved`. Read this key, not `resolved`, when you want
  to know what ran.

`provenance` is one of `explicit` (you set a positive batch), `ultralytics_autobatch`,
`ultralytics_autobatch_clamped`, `default_non_cuda` (Ultralytics only measures on
CUDA, so off CUDA the default is used unchanged), or `fallback` (the resolution
child failed or left no readable report — the default is used and `-1` is never
passed on).

The resolved number is **Ultralytics' estimate**, not a measured peak and not a
guarantee that the run will fit: it profiles a single step and extrapolates.
The bounded OOM-retry ladder — which halves the batch in a fresh child on a
classified out-of-memory exit — is what actually protects the run.

`training.batch` is **opt-in**: `16` is a default shared with PoseKit, ClassKit
and TrackerKit, so omitting `training.batch` still gives you `16`. You have to
write `-1` to get auto batch.

Ultralytics device conventions work here: `"0"`, `"0,1"` and `"cuda:0"` all
reach resolution, as does `"auto"` on a CUDA host. A multi-GPU value resolves
against the **first** device only — Ultralytics profiles one device, and sizing
against the whole set would overstate capacity. The launch command still
receives the device string exactly as you wrote it.

Auto batch is not reproducible across machines, because it depends on the GPU
it measures on. **Set a fixed positive `training.batch` for byte-reproducible
runs.**

### SAM3 batch

`sam3.batch` accepts `-1` too, but it means something quite different from
`training.batch: -1` and is **not** Ultralytics' autobatch.

`-1` makes HYDRA measure this workload on this card before launching. It walks
a ladder of contained probe children (batch 1, 2, 4, 8), each a real SAM3 LoRA
run taken through two full optimizer steps on the densest tiles, and records
the reserved device peak each one reaches. An out-of-memory inside a probe
child is a **measurement**, not a failure.

The chosen batch then has to be safe under **both** the measurement and a
conservative analytic estimate — the requirement is `max(analytic, measured)`.
A measurement may only **raise** the estimate, never lower it, because a short
probe systematically under-reports (the same run measured 7.34 GiB over two
steps, 9.93 GiB over sixty, and 12.99 GiB over a full run; the cause is CUDA
allocator fragmentation).

Be clear-eyed about what that buys you: this is **not** an optimal-batch
search, and it is not tuned for throughput. The analytic term currently
dominates on the cards this role targets, so **`-1` will often resolve to
`1`.** Nothing here promises the largest batch that would fit.

**It can also refuse the run.** If batch 1 does not fit in the free VRAM at the
safety margin, training fails with an explicit refusal naming the requirement
and the free bytes. The measurement is still cached, so retrying on a quieter
GPU reuses it rather than re-probing.

**Where the measurements live.** Records are keyed by a workload fingerprint —
checkpoint, sidecar-env package set, GPU model, precision, LoRA rank and
adapter scope, tile geometry, dataset density, and the probe protocol itself —
and stored **globally, across runs**, at
`<HYDRA_DATA_DIR>/memory_profiles/sam3_lora.json`. A second run with the same
fingerprint reuses the cached records and launches no probe. Anything that
changes the memory changes the key, including bumping the probe step count.

**Escape hatch.** Set `HYDRA_SAM3_FORCE_PROBE=1` to re-measure even when the
store already has records for this exact fingerprint — for example after a
driver upgrade, which the key does not capture.

**Keys.** Every key `<run_dir>/batch_resolution.json` carries on the SAM3
path, including the ones shared with the YOLO block:

| Key | Meaning |
| --- | --- |
| `fingerprint` | The workload key the records were stored under. |
| `provenance` | `measured` (we probed this run), `cached`, or `explicit`. |
| `requirement_provenance` | Which side of `max(analytic, measured)` decided the requirement that was cleared. |
| `requirement_basis` | `measured` if the chosen batch is a rung we actually observed, `extrapolated` if it fell between rungs. |
| `requirement_bytes` | The requirement the chosen batch cleared. |
| `requirement_measured_extrapolated` | Whether the measured side of that requirement was itself extrapolated. |
| `measured_envelope_bytes` | `max(fitted curve, every observed peak at or below this batch)`. An envelope, not a raw observation — at an observed rung it can still exceed that rung's peak. |
| `ladder_terminated_by` | Why the probe ladder stopped. |
| `degraded_reasons` | Anything that weakened the fingerprint, such as a checkpoint that could not be stat'ed. |
| `free_bytes` | Free VRAM on the selected GPU at selection time. |

Like YOLO's, SAM3 auto batch is not reproducible across machines. **Set a fixed
positive `sam3.batch` for reproducible runs.**

## Validate and prepare

Validate the configuration and inspect all resolved paths without creating the
workspace or reading the dataset:

```bash
detectkit train --config training.json --dry-run
```

Run source preflight and build the merged, sliced, and role-specific datasets
without training:

```bash
detectkit train --config training.json --prepare-only
```

Each invocation records `resolved_training_plan.json`, `preflight.json`, and
`prepared_datasets.json` under a unique
`<workspace>/sessions/<session-id>/` directory. A completed training session
additionally records `training_result.json` with run IDs, status, metrics, and
artifact paths. Failed preflight, cancellation, and unexpected failures also
write a structured `training_result.json`, so an earlier session is never
overwritten by a later run.

Only one DetectKit training invocation may use a workspace at a time. A second
invocation exits with a configuration error instead of racing dataset builders
or corrupting session state. Use a distinct workspace when the server should
run independent jobs concurrently. The global training-run registry serializes
updates from jobs that use different workspaces.

## Resume a run

For a configuration containing exactly one Ultralytics role, resume from its
last checkpoint with:

```bash
detectkit train --config training.json \
  --resume ./training-workspace/runs/<run-id>/weights/last.pt
```

Resume is deliberately rejected for multi-role and SAM3 plans so a checkpoint
cannot silently attach to the wrong stage.

## Scheduler example

For Slurm, activate the CUDA environment in the batch script and let the
scheduler capture normal stdout and stderr:

```bash
#!/usr/bin/env bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --output=detectkit-%j.log

set -euo pipefail
source ~/mambaforge/etc/profile.d/conda.sh
conda activate hydra-cuda
detectkit train --config /shared/experiment/training.json
```

`SIGINT` and `SIGTERM` request cancellation through the same cancellation path
used by the GUI. For Ultralytics jobs, DetectKit continues checking for a
request even while the trainer is silent and terminates the trainer process
group so data-loader or distributed-training children are not left behind.
Repeated signals remain cooperative while cleanup completes. Exit status is
`0` on success, `1` for a failed training run, `2` for a configuration or
preflight error, and `130` when canceled.
