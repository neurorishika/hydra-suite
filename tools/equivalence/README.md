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

Clips currently cover: `emi_obb_identity` (OBB + identity online decoder) and
`ant_pose_headtail` (OBB + head-tail + SLEAP pose + identity). No AprilTag clip yet.

On a fresh machine:
```bash
conda activate hydra-mps                        # or hydra-suite-cuda
bash tools/equivalence/fixtures/fetch_fixtures.sh   # downloads clips + models
bash tools/equivalence/run_matrix.sh                # FIXTURES=1 is the default
```

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

## Cross-device merge-readiness

Run `run_matrix.sh` on each target machine (Apple Silicon → `mps`, NVIDIA →
`cuda`, plus a `cpu` baseline anywhere). Collect the `meta.json` + printed reports.
The system is merge-ready when every device shows equivalence within its own
determinism floor for both `forward` and `final` CSVs on every video.
