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

The workspace records `resolved_training_plan.json`, `preflight.json`, and
`prepared_datasets.json`. A completed training session additionally records
`training_result.json` with run IDs, status, metrics, and artifact paths.

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
used by the GUI. Exit status is `0` on success, `1` for a failed training run,
`2` for a configuration or preflight error, and `130` when canceled.
