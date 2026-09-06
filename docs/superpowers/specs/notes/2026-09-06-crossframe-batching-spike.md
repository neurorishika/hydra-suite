# Cross-frame crop batching (Option B) — measurement spike

**Status:** measurement only. No pipeline code changed. Run on `courtship`
(RTX 4090, torch 2.11.0+cu128) against repo `17eb4c48`.

## Question

`Pipeline._process_downstream_frame` hands the batch stage functions exactly
one frame at a time (`frames = [frame]`), a deliberate crop-memory bound
introduced by `893cde19`. **Option B** would reintroduce cross-frame crop
accumulation via an independent, crop-count-bounded accumulator, decoupled
from `detection_batch_size`. Is the speedup worth the cache-ordering and
numerics cost?

## Method

`tools/spikes/crossframe_batching/spike_crossframe.py` builds the REAL stages
from a real fixture config + clip (`build_tracking_parameters` →
`build_inference_config_from_params` → `InferenceRunner`), runs detection
ONCE to fix the inputs, then calls the REAL batch stage functions two ways
over the SAME detections:

- **per_frame**   — `run_*_batch([f], [obb])` once per frame  (today)
- **accumulated** — `run_*_batch(frames, obbs)` once per group (Option B)

`torch.cuda.synchronize()` around every timing; 2 warmup, median of 5.
Numeric agreement is reported harness-style: an `accumulated` vs `accumulated`
DETERMINISM floor first, then `per_frame` vs `accumulated` against it.

Clips: `ant_cnn_identity` (head-tail + multi-head identity CNN, ~17 det/frame),
`ant_pose_headtail` (head-tail only, ~18 det/frame). 32 frames each.

## Results — speedup (per_frame / accumulated, >1 = B is faster)

`ant_cnn_identity`, gpu tier (native torch CUDA):

| classifier batch knob | group | warp | head-tail | CNN |
|---|---|---|---|---|
| 25 (fixture value) | 1 (null) | 1.00× | 0.98× | 1.00× |
| 25 | 2 | 0.92× | 1.02× | 1.02× |
| 25 | 4 | 0.81× | 1.09× | 1.04× |
| 25 | 8 | 0.89× | 1.11× | 1.09× |
| **64 (code default)** | 1 (null) | 1.00× | 1.07× | 0.99× |
| **64** | 2 | 0.92× | **1.18×** | **1.18×** |
| **64** | 4 | 1.09× | **1.16×** | 1.07× |
| **64** | 8 | 1.02× | **1.23×** | **1.17×** |
| 256 | 2 | 0.93× | 1.16× | 1.19× |
| 256 | 4 | 1.11× | 1.31× | 1.14× |
| 256 | 8 | 1.01× | 1.20× | 1.11× |

`ant_cnn_identity`, gpu_fast tier (ONNX Runtime; classifier `.onnx` artifacts
confirmed present on the box):

| batch knob | group | warp | head-tail | CNN |
|---|---|---|---|---|
| 25 | 4 | 1.09× | 1.09× | 0.99× |
| 25 | 8 | 0.98× | 1.11× | 1.02× |
| 256 | 4 | 1.07× | 1.07× | 0.96× |
| 256 | 8 | 0.94× | 0.99× | 0.94× |

`ant_pose_headtail`, gpu tier, knob 256 (second clip, confirmation):

| group | warp | head-tail |
|---|---|---|
| 1 (null) | 1.00× | 0.97× |
| 4 | 1.19× | 1.26× |
| 8 | 0.95× | 1.22× |
| 16 | 1.07× | 1.18× |

### Noise floor — read the tables with this

Group 1 is the null arm: both arms do identical work by construction, so its
deviation from 1.00× IS the noise. It reads **0.94–1.07×**, and absolute warp
time for the same 536 crops varied 55–98 ms across runs (clock/thermal ramp;
the two arms run sequentially, which biases toward whichever runs second).
**Treat anything under ~1.10× as noise.** On that basis the b25 rows carry no
signal; the b64 and b256 head-tail/CNN gains do.

## Results — numerics

| stage | tier | determinism floor | per_frame vs accumulated |
|---|---|---|---|
| head-tail | gpu | 0 flips, Δ=0.0 | **0 flips, Δ=0.0** (byte-identical, all knobs/groups, both clips) |
| CNN | gpu | 0 flips, Δ=0.0 | 0 argmax flips / 1072 factor preds, **max \|Δprob\| ≈ 0.0022** |
| head-tail, CNN | gpu_fast | 0 flips, Δ=0.0 | **0 flips, Δ=0.0** (byte-identical) |

The head-tail π-flip worry is **dismissed on this evidence** — head-tail is
byte-identical under regrouping on both tiers.

CNN on the gpu tier is not: chunk composition (one chunk of 136 crops from 8
frames vs 8 chunks of ~17) shifts probabilities by up to 2.2e-3 against a
perfectly reproducible 0.0 floor. No decision changed on these clips, but
Debug CSVs carry `IdentityEvidence*` probability columns, so **B breaks
byte-identity of the equivalence gate on identity clips**, and Bayes
log-compat accumulation over thousands of frames could turn 2e-3 into a flip
somewhere. gpu_fast is immune (fixed-shape ONNX path).

## Verdict for CUDA classifiers

**Marginal.** At the code-default batch knob (64), B buys ~1.2× on head-tail
and ~1.2× on CNN, gpu tier only; gpu_fast shows nothing above noise. Crop warp
does not shrink (~1.0×), confirming the prior — B accelerates only the backend
forward, not the ~32%-of-wall warp. The cost is the cache-ordering rework
(`write_downstream` deferral vs `runner.py`'s strict window order + the
explicit-empty cancel invariant) plus loss of identity byte-identity.

## What is NOT measured (and why the overall go/no-go is still open)

Both of the arguments that actually motivated B are unmeasured here:

1. **SLEAP pose round-trip** — hypothesized to be the largest lever
   (`project_sleap_gpu_mehek`: "lever = batch frames per request").
   **Blocked on courtship:** its `sleap` conda env is TensorFlow SLEAP 1.4.1
   on Python 3.7, not the sleap-nn/PyTorch runtime the pipeline requires
   (`No module named 'torch'` inside the env). No exported SLEAP `.onnx` on
   the box either, so the gpu_fast in-process pose arm is unavailable too.
   Needs mehek or a sleap-nn env.
2. **CoreML classifier batching** — the 2026-07-03 spec measured **7.8×** per
   frame for batched CoreML classification. That is Apple-only and is the one
   place B could be a large win rather than a marginal one. Not reproducible
   on CUDA hardware; needs the MPS box.

The spike script is portable — `--tier gpu` gives the SLEAP service arm and
`--tier gpu_fast` gives CoreML classifiers on the MPS box.

## Reproducing

```bash
conda activate hydra-cuda          # or hydra-mps
export PYTHONPATH=$PWD/src KMP_DUPLICATE_LIB_OK=TRUE SPIKE_NO_POSE=1
python tools/spikes/crossframe_batching/spike_crossframe.py \
  --config tools/equivalence/fixtures/configs/ant_cnn_identity.json \
  --clip   tools/equivalence/fixtures/clips/ant_cnn_identity.mp4 \
  --tier gpu --frames 32 --groups 1 2 4 8 --repeats 5 --warmup 2 \
  --skeleton <skeletons>/ooceraea_biroi.json --out results.json
```

Env overrides: `SPIKE_NO_POSE=1`, `SPIKE_CLS_BATCH=<n>` (head-tail + CNN
batch knob), `SPIKE_POSE_BATCH=<n>`.

`--skeleton` is required for SLEAP configs because the fixture configs carry
`pose_skeleton_file: ""` and `runner.py`'s `load_pose_model` call passes no
`keypoint_names` override — the pre-existing main breakage recorded in
`project_main_pose_sleap_breakage`.

Raw JSON reports: `tools/spikes/crossframe_batching/results/`.
