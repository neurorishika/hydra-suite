# Cross-frame crop batching (Option B) — measurement spike

**Status:** measurement complete, both platforms, all stages. No pipeline code
changed. **Verdict: do not implement.**

## Question

`Pipeline._process_downstream_frame` hands the batch stage functions exactly
one frame at a time (`frames = [frame]`), a deliberate crop-memory bound
introduced by `893cde19`. **Option B** would reintroduce cross-frame crop
accumulation via an independent, crop-count-bounded accumulator, decoupled
from `detection_batch_size`. Is the speedup worth the cache-ordering and
numerics cost?

Two specific hypotheses motivated it:

1. **SLEAP pose round-trip** — `project_sleap_gpu_mehek` concluded "pose
   slowness = per-frame service round-trip overhead; lever = batch frames per
   request."
2. **CoreML classifier 7.8×** — `2026-07-03-tensorrt-coreml-cross-frame-batching-design.md`
   measured a 7.8× per-frame speedup from batched CoreML classification.

**Both are refuted below.**

## Method

`tools/spikes/crossframe_batching/spike_crossframe.py` builds the REAL stages
from a real fixture config + clip (`build_tracking_parameters` →
`build_inference_config_from_params` → `InferenceRunner`), runs detection ONCE
to fix the inputs, then calls the REAL batch stage functions two ways over the
SAME detections:

- **per_frame**   — `run_*_batch([f], [obb])` once per frame  (today)
- **accumulated** — `run_*_batch(frames, obbs)` once per group (Option B)

`torch.cuda.synchronize()` around every timing; warmup then median of N.
Numeric agreement is reported harness-style: an accumulated-vs-accumulated
DETERMINISM floor first, then per_frame vs accumulated against it.

Hardware: `courtship` RTX 4090 (torch 2.11.0+cu128, sleap-nn 0.1.3 in a
purpose-built `sleap-nn` env) and this Apple Silicon box (`hydra-mps`,
sleap-nn 0.1.3). Clips `ant_cnn_identity` (HT + multi-head identity CNN,
~17 det/frame) and `ant_pose_headtail` (HT + SLEAP pose, ~18 det/frame).

## Results — best speedup per stage (per_frame / accumulated)

| platform | tier | backend | stage | best | numerics vs 0.0 floor |
|---|---|---|---|---|---|
| CUDA | gpu | torch | head-tail | **1.23×** | 0.0 — byte-identical |
| CUDA | gpu | torch | CNN | **1.18×** | Δprob 2.2e-3, 0 argmax flips |
| CUDA | gpu_fast | ONNX/TRT | HT + CNN | 0.94–1.11× (noise) | 0.0 — byte-identical |
| CUDA | gpu | sleap-nn service | **pose** | **1.04×** | Δ up to **1.045 px** |
| MPS | gpu | torch | head-tail | **1.31×** | 0.0 — byte-identical |
| MPS | gpu | torch | CNN | **1.14×** | Δprob 1.4e-6 |
| MPS | gpu_fast | **CoreML** | head-tail | **1.15×** | 0.0 — byte-identical |
| MPS | gpu_fast | **CoreML** | CNN | **1.15×** | Δprob **0.048**, 0 argmax flips |
| MPS | gpu | sleap-nn service | **pose** | 1.09× @ batch 4, **1.35×** @ batch 64 | 0.0 |
| both | all | — | **crop warp** | ~1.0× | n/a |

### Noise floor — read every row with this

Group 1 is a null arm: both sides do identical work by construction, so its
deviation from 1.00× IS the noise. Across all runs it reads **0.93–1.29×**.
Absolute times for identical work swung up to 40% between runs (clock/thermal
ramp; the arms run sequentially, biasing whichever runs second). One MPS row
is a visible outlier (head-tail g4 @ pose-batch 4 read 3.46× — a contaminated
sample, not a real effect). **Treat anything under ~1.15× as unproven.**

## The two motivating hypotheses, tested

**1. SLEAP round-trip — REFUTED.** Cross-frame pose accumulation buys
**1.01–1.04×** on the RTX 4090 at either SLEAP batch setting. Batching frames
per request is not the lever the earlier audit assumed. On MPS pose reaches
1.35×, but only with `PoseSLEAPConfig.batch_size` raised 4 → 64 *and* group-8
accumulation; at the default batch it is 1.04–1.09×.

**The knob alone is inert — it is not a separate cheap win.** Compare the
per_frame arm (what the pipeline does today) at batch 4 vs 64:

| | g1 | g4 | g8 |
|---|---|---|---|
| MPS per_frame, batch 4 | 975 ms | 919 ms | 817 ms |
| MPS per_frame, batch 64 | 1184 ms | 1064 ms | 886 ms |
| CUDA per_frame, batch 4 | 288 ms | 285 ms | 281 ms |
| CUDA per_frame, batch 64 | 292 ms | 288 ms | 299 ms |

Flat or slower. With ~18 crops per frame, 5 chunks of 4 costs the same as 1
chunk of 18 — which is itself further evidence that chunk/round-trip count is
cheap. The knob only does anything once accumulation has already produced a
144-crop call, i.e. it is a multiplier on B, not an alternative to it.

**2. CoreML 7.8× — REFUTED as an argument for B.** (CoreML genuinely ran:
`.mlpackage` artifacts are present for both the orientation and identity
models, and CNN's Δprob of 0.048 vs 1.4e-6 on the torch-MPS tier confirms a
different backend was exercised.) Measured CoreML classifier
gain from cross-frame accumulation is **1.15×**, not 7.8×. The original 7.8×
came from fixing a call-count bug (`_forward_coreml` looping per crop instead
of one batched `predict()`), and that fix already shipped — today's per-frame
call already batches all ~17 of a frame's crops. Accumulating *more* on top of
an already-batched call adds ~15%, which is the same ~15% every other backend
shows. There is no Apple-specific windfall left to collect.

## Numerics — three findings

- **Head-tail is byte-identical** under regrouping on every platform/tier
  (0 flips, Δ=0.0, 1000+ predictions). The θ π-flip worry is **dead**.
- **Pose is byte-identical on MPS**; on CUDA five of six rows are ≤0.001 px
  but **one row (batch 64, group 4) drifts 1.045 px** on a keypoint against a
  0.0 determinism floor — outlier-class, not systematic, but real.
- **CNN drift is backend-dependent and worst on CoreML**: Δprob 1.4e-6
  (torch-MPS) → 2.2e-3 (torch-CUDA) → **0.048 (CoreML)**, always with 0 argmax
  flips. 0.048 is ~5 percentage points on a probability.

Because Debug CSVs carry `IdentityEvidence*` probability columns and pose
keypoint coordinates, **B breaks equivalence byte-identity on identity and
pose clips on three of the four backends measured.**

## Verdict

**Do not implement Option B.** Every stage, both platforms, every tier lands
in a 1.0–1.35× band, mostly within ~2× of the measurement noise floor. Crop
warp — ~32% of wall per `project_sleap_roundtrip_audit` — does not shrink at
all, because B only affects the backend forward. Against that, B costs:
deferring `write_downstream` against `runner.py`'s strict window order, a new
cancel-path flush invariant, a crop-count-bounded accumulator, a
pre-extracted-batch parameter for `run_cnn_batch` (it builds its own crops),
and loss of gate byte-identity on identity + pose clips.

**End-to-end, the ceiling is negligible.** `project_sleap_roundtrip_audit`
measured pose at 4.6% of wall, so even the best-case 1.35× on pose is ~1.2%
end-to-end. The classifier gains sit on a similarly small slice.

**Where to look instead:** crop warp, not the backend forward. It is ~32% of
wall and B provably does not touch it. Note also that no batch-knob tuning is
available as a consolation prize — the classifier knobs (64) already exceed
typical per-frame detection counts and so are inert, and the SLEAP knob is
inert without accumulation (above).

## Incidental findings

- **`POSE_SLEAP_ENV` and `pose_sleap_batch` are dead keys.**
  `build_inference_config_from_params` constructs `PoseSLEAPConfig` with only
  `model_path` and `POSE_BATCH_SIZE` (`config.py:1097-1100`), so `conda_env`
  is always the `"sleap"` default no matter what the config says. Same class
  as the `DETECTION_BATCH_SIZE` dead key. The spike works around it by
  mutating the built config.
- **`pip install "sleap[nn,nn-export-gpu]"` unpinned installs sleap-nn 0.3.3,
  which breaks the pipeline's shared-memory transport** (`findDecoder
  imread_('inmem_crop_000000')`). The working version is sleap-nn 0.1.3 via
  `sleap==1.6.2`. `docs/getting-started/integrations.md` gives the unpinned
  command.
- On courtship, `sleap[nn,nn-export-gpu]` pulls torch cu130, which the box's
  535 driver cannot run — reinstall torch/torchvision from the cu128 index.
- The equivalence clips are gitignored, so **a git worktree contains no
  fixtures**; a first run silently measured 0 detections and would have
  reported meaningless ratios. The harness now aborts loudly on 0 frames or 0
  detections.

## Reproducing

```bash
conda activate hydra-cuda          # or hydra-mps
export PYTHONPATH=$PWD/src KMP_DUPLICATE_LIB_OK=TRUE
python tools/spikes/crossframe_batching/spike_crossframe.py \
  --config <fixtures>/configs/ant_cnn_identity.json \
  --clip   <fixtures>/clips/ant_cnn_identity.mp4 \
  --tier gpu --frames 32 --groups 1 2 4 8 --repeats 5 --warmup 2 \
  --skeleton <skeletons>/ooceraea_biroi.json --out results.json
```

Point `--clip`/`--config` at the PRIMARY checkout's fixtures (absolute paths).
Env overrides: `SPIKE_NO_POSE=1`, `SPIKE_CLS_BATCH=<n>`,
`SPIKE_POSE_BATCH=<n>`, `SPIKE_SLEAP_ENV=<env>`.

`--skeleton` is required for SLEAP configs: the fixture configs carry
`pose_skeleton_file: ""` and `runner.py`'s `load_pose_model` call passes no
`keypoint_names` override — the pre-existing main breakage recorded in
`project_main_pose_sleap_breakage`.

Raw JSON reports: `tools/spikes/crossframe_batching/results/`.
