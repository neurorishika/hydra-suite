# Inference-pipeline equivalence harness

Verifies that the **new** inference pipeline (this `feature/inference-pipeline-redesign`
worktree) produces output equivalent to the **legacy** pipeline (`main`), across
devices (CPU / MPS / CUDA), so we can decide whether the redesign is ready to merge.

## How it works

The pipeline that runs is decided purely by which `hydra_suite` is importable —
set `PYTHONPATH` to a source tree. So "legacy vs new" = "main `src/` vs worktree
`src/`", same conda env, same models, same config.

Every run is isolated: `runner.py` symlinks the source video into its output dir,
so detection caches (`<stem>_caches/`) and trajectory CSVs land there and your real
data directory under `MultiTrackerData/` is never touched. Pure side-outputs
(rendered video, datasets, crop images, density map) are disabled; the full
detect → headtail → pose → identity → track path is preserved. Detections are
always recomputed fresh (`use_cached_detections=False`).

## Files

- `runner.py` — run one session into `--outdir`; records `meta.json` (device, torch,
  git branch/commit of the src used, runtime). `--runtime` overrides every stage
  runtime (`cpu|mps|cuda|onnx_cpu|onnx_cuda|tensorrt`), or `config` to keep the
  config's own.
- `compare.py` — compare two CSVs. **Positional** view matches detections per frame
  by nearest (X,Y) — robust to track-ID renumbering. **Keyed** view aligns rows by
  (FrameID, track id) when schemas match.
- `run_matrix.sh` — one-shot: legacy ×1, new ×2, per video, then prints the
  determinism baseline and the equivalence comparison.

## Portable fixtures (default)

For cross-machine runs without copying the full datasets, the harness ships short
clips + the exact models they need, hosted as a GitHub Release (only small
portable files live in git: configs, skeleton, manifest, scripts).

`fixtures/`:
- `clips/` (gitignored) — short clips, fetched from the release.
- `configs/` — portable per-clip tracking configs (paths blanked; set at run time).
- `ooceraea_biroi.json` — pose skeleton (passed via `runner --skeleton`).
- `manifest.json` — release tag + sha256/size of every asset.
- `fetch_fixtures.sh` — download + verify clips, extract models into the models dir.
- `generate_clips.py` / `make_manifest.py` — regenerate fixtures from source data
  (dev-only; run where the full videos + models exist).

Clips currently cover (one representative per demo, ~500 frames each):

| clip | path / features exercised |
|---|---|
| `emi_obb_identity` | OBB-direct + identity online decoder |
| `ant_pose_headtail` | OBB + head-tail + SLEAP pose + identity (realtime) |
| `ant_obb_sleap` | OBB-direct + SLEAP pose + identity-analysis (no head-tail) |
| `ant_obb_sequential` | **OBB-sequential** (detect stage -> crop -> OBB stage) + SLEAP pose + identity-analysis; reuses the `ant_obb_sleap` clip with `yolo_obb_mode=sequential` and the `detection/20260305-175022_26x_obiroi_v1.pt` + `obb/cropped/20260305-175049_26s_obiroi_obbcrop.pt` model pair |
| `worm_bgsub` | **background-subtraction** detection path |
| `ant_cnn_identity` | OBB + **CNN multihead identity** + head-tail + pose-direction |
| `fly_obb` | OBB-direct, fly, fresh detection |

No AprilTag clip yet (none of the source demos enable AprilTags).

`ant_obb_sequential`'s two-stage models are now listed in `manifest.json`'s
`models_contained` (added via `make_manifest.py`'s `EXTRA_MODEL_CONFIGS`, since
the clip is reused rather than a separate asset), and the regenerated
`models.tar.gz` (384 MB, includes both models) has been uploaded to the
`equiv-fixtures-v2` release, replacing the previous asset — a fresh machine's
`fetch_fixtures.sh` can now pull `ant_obb_sequential`'s models.

On a fresh machine:
```bash
conda activate hydra-mps                        # or hydra-suite-cuda
bash tools/equivalence/fixtures/fetch_fixtures.sh   # downloads clips + models
bash tools/equivalence/run_matrix.sh                # FIXTURES=1 is the default
```

Run only specific clips (so you don't rerun the whole matrix) — pass names as
arguments or via `ONLY=` (space- or comma-separated):
```bash
bash tools/equivalence/run_matrix.sh ant_pose_headtail worm_bgsub
ONLY=ant_pose_headtail bash tools/equivalence/run_matrix.sh
```
Clip names: `emi_obb_identity`, `ant_pose_headtail`, `ant_obb_sleap`, `ant_obb_sequential`,
`worm_bgsub`, `ant_cnn_identity`, `fly_obb`.

To regenerate/refresh the fixtures (on a machine with the full data):
```bash
python tools/equivalence/fixtures/generate_clips.py
python tools/equivalence/fixtures/make_manifest.py
# then create the release and upload the assets it lists (clips + models.tar.gz)
```

## Quick start (full local videos)

```bash
conda activate hydra-mps          # or hydra-suite-cuda on an NVIDIA box
FIXTURES=0 bash tools/equivalence/run_matrix.sh    # uses $DATA full videos
```

Override anything via env vars (see top of `run_matrix.sh`), e.g. force a device:

```bash
RUNTIME=cpu bash tools/equivalence/run_matrix.sh
```

## Interpreting results — read these two together

For each video/CSV the harness prints two comparisons:

1. **DETERMINISM (new_a vs new_b)** — the same pipeline run twice. This is the
   noise floor. If GPU inference is non-deterministic, this will be non-zero, and
   small differences cascade through tracking. **Equivalence cannot be tighter than
   this.**
2. **EQUIVALENCE (legacy vs new_a)** — legacy vs new.

Decision rule: the redesign is equivalent on a given device if the EQUIVALENCE
numbers are at or near the DETERMINISM numbers. If equivalence is far worse than
the determinism floor, that gap is a real regression to investigate.

`compare.py` exit code: 0 within tolerance, 1 otherwise (tolerances via
`--gate`, `--pos-atol`, `--theta-atol`).

## Performance check

Each run records its wall-clock and FPS in `meta.json` (`tracking_seconds`,
`fps`, `n_frames`), and the matrix prints a **PERFORMANCE** line per video:

```
>>> ant_pose_headtail : performance
--- PERFORMANCE  legacy vs new_a (tolerance 1.25x) ---
  legacy: 4.7s (106.8 fps)   new: 12.3s (40.7 fps)
  new/legacy time ratio = 2.62x  ->  PERFORMANCE: SLOWER ❌
```

The new pipeline must not be meaningfully slower than legacy: `PERFORMANCE: SLOWER ❌`
when `new/legacy` wall-clock ratio exceeds `PERF_TOLERANCE` (default 1.25). This
catches throughput regressions that the CSV comparison can't — e.g. a cold
per-frame pose service running seconds/frame even when the trajectories match.

## Cross-device merge-readiness

Run `run_matrix.sh` on each target machine (Apple Silicon → `mps`, NVIDIA →
`cuda`, plus a `cpu` baseline anywhere). Collect the `meta.json` + printed reports.
The system is merge-ready when every device shows equivalence within its own
determinism floor for both `forward` and `final` CSVs on every video.
