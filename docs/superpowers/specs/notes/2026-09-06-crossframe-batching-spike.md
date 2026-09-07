# Cross-frame crop batching (Option B) — measurement spike

**Status:** COMPLETE — stage-level microbenchmark, end-to-end profile, and an
in-pipeline A/B on both platforms. **Verdict: do not implement Option B.** The
in-pipeline A/B is the decisive evidence and is at the top of this note; the
earlier sections are kept because the route to the answer matters. No
production code changed (the A/B patch is env-gated on a throwaway branch).

# THE ANSWER: in-pipeline A/B

Same binary, same clip, same `detection_batch_size`, one env flag
(`HYDRA_AB_CROSSFRAME=1`) switching the downstream call between per-frame and
whole-window. 489 frames, `ant_cnn_identity`, `HYDRA_PROFILE=1`.

**B/A ratios (>1 means Option B is SLOWER):**

| | span | CUDA d2/b2 | CUDA d1/b8 | MPS d2/b2 | MPS d1/b8 |
|---|---|---|---|---|---|
| | `headtail` | 1.005 | **1.000** | **1.580** | 1.177 |
| | `cnn` | 0.991 | 0.995 | 0.890 | 1.133 |
| | `pose` | — | — | 1.092 | 0.813 |
| | `backend_forward` | 1.000 | 0.995 | 1.165 | 1.060 |
| | **wall clock** | 0.958 | 0.998 | **1.074** | **1.027** |

**On CUDA, Option B does exactly nothing** — every stage within ±0.5%. The
`d1/b8` column is the microbenchmark's own g8 condition (8 frames × ~17
detections = 136 crops in one call vs 8 calls of 17), where the isolated spike
predicted **1.23×**. In the pipeline it is **1.000**.

**On MPS it is neutral-to-worse**, up to 1.58× slower on head-tail.

The idle-GPU microbenchmark simply does not transfer. What it measured —
per-call launch overhead — is not what the pipeline is spending time on.

**Numerics, with a real floor:** an A-vs-A rerun is **byte-identical**, so
everything below is attributable to B alone. B changes **9,413 of 12,225 rows
(77%)** on MPS — but exclusively in three columns:

| column | rows differing | max abs delta |
|---|---|---|
| `IdentityRealtimeEntropy` | 9,253 | 4.0e-06 |
| `IdentityRealtimeMargin` | 8,997 | 5.2e-06 |
| `IdentityRealtimeConfidence` | 8,714 | 2.6e-06 |

Positions, angles, track IDs and identity assignments are unchanged. CUDA
(pose off) is fully byte-identical. So the drift is smaller than the isolated
microbenchmark suggested (5e-6, not 2.2e-3) — but it is real, it is 100%
attributable, and it breaks the byte-identity gate for zero measured speed.

## Two follow-on questions raised by the A/B

**1. `pipeline_depth` — measured properly, the current default is already
right.** The first read of this ("depth 2 is a net loss on MPS") came from
comparing depth 1 / batch 8 against depth 2 / batch 2 — the depth and the
batch varied together. **Re-run at constant `detection_batch_size=2`, 2–3
reps per cell:**

| | depth 1 | depth 2 | d2/d1 |
|---|---|---|---|
| **CUDA wall** (median of 3) | 72.0 s | **63.5 s** | **0.881** |
| CUDA reps | 71.9 / 72.0 / 72.3 | 63.5 / 63.9 / 63.2 | — |
| **MPS wall** (median of 2) | 152.6 s | 148.8 s | 0.975 |
| MPS reps | 150.5 / 154.8 | 152.9 / 144.7 | — |

**CUDA: depth 2 is a real 12% win**, with reps tight to ±0.4 s. **MPS: no
significant difference** — the reps overlap, so 0.975 is noise. Output is
**byte-identical across depths on both platforms** (and depth-1 rep-to-rep is
byte-identical, giving a clean floor).

So `pipeline_depth=2` is correct on CUDA and harmless on MPS. **No change
warranted — no per-platform default needed.**

Methodological note: the *span totals* look alarming at depth 2 (MPS head-tail
44.3 s vs 18.6 s, 2.37×) because two threads are running concurrently and each
stage's wall-time inflates while it waits on the shared device. That is what
overlap looks like; it is not a cost. **For concurrent pipelines, wall clock is
the only honest metric — per-stage span totals are not additive and must not be
compared across depths.**

**2. `detection_batch_size` > 2 is unusable on 4K clips.** At depth 2 the
frame-buffer admissibility check (`pipeline.py`, `retained_windows =
queue_bound + 3`) rejects batch 8 on this clip at runtime:

```
one frame=61074432 bytes, batch=8, live_windows=4,
estimated=1954381824 bytes exceeds the 536870912-byte pipeline budget
```

The GUI spinbox offers 1–64 with no indication that anything above 2 will
abort the run on 4K video. That is a real usability bug: the control's range
is not the admissible range.

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

## Correction — the first verdict was wrong, and is retracted

The first pass concluded "do not implement, ~1.2% end-to-end." **That ceiling
was wrong.** It anchored on `project_sleap_roundtrip_audit`'s "pose = 4.6% of
wall", which was measured on a different clip/config. An actual
`HYDRA_PROFILE=1` run of `ant_cnn_identity` (MPS, gpu tier, 489 frames,
246.5 s) shows the opposite:

| span | s | % wall | thread |
|---|---|---|---|
| `window` (consumer critical path) | 238.5 | 96.8% | tracking-engine-forward |
| `headtail` | 90.5 | **36.7%** | consumer |
| `cnn` | 74.1 | **30.1%** | consumer |
| `pose` | 69.3 | **28.1%** | consumer |
| `detect` / `run_obb` | 93.4 | 37.9% | **producer — fully hidden** |
| `backend_forward` (all stages) | 174.6 | 70.8% | consumer |
| `crop_extract` | 32.0 | 13.0% | consumer |

**Downstream crop consumers are 94.9% of the critical path.** Detection runs
on the producer thread and is entirely overlapped. So B targets the dominant
cost, not a 4.6% sliver — the stage the user cares about is exactly the one
that is slow.

### But the stage-level ratios still cannot be multiplied through

Per-frame head-tail cost, same clip, same tier, same model:

- spike, `SPIKE_NO_POSE=1`, GPU otherwise idle: 831 ms / 32 = **26 ms/frame**
- pipeline, `HYDRA_PROFILE=1`: 90.5 s / 489 = **185 ms/frame**

**7× slower inside the pipeline.** The pipeline runs the OBB producer thread,
the consumer thread, and the SLEAP service process concurrently on one device.
The profile is therefore contention-bound, while the 1.2× was measured with
zero contention. Under contention, fewer/larger calls could help (less
interleaving), do nothing (device already saturated), or hurt. Multiplying
233.9 s × (1 − 1/1.2) assumes an answer that has not been measured.

**No end-to-end percentage should be quoted until an in-pipeline A/B is run.**

### Contention isolation — SLEAP is NOT the co-tenant; the detector is

Rerunning the same clip with `enable_pose_extractor: false`:

| | pose ON | pose OFF |
|---|---|---|
| head-tail | 185.1 ms/frame | **187.3 ms/frame** |
| cnn | 151.5 ms/frame | 140.7 ms/frame |
| detect (producer thread) | 191.0 ms/frame | 198.2 ms/frame |
| `window` (consumer) | 487.8 ms/frame | 331.9 ms/frame |

Head-tail is **unchanged** with the SLEAP service process gone, so SLEAP is not
what inflates it. What remains concurrent is the **OBB producer thread**, doing
~198 ms/frame of detection on the same device while the consumer runs
head-tail + CNN. Per-frame GPU work (198 + 187 + 141 = 526 ms) exceeds the
332 ms consumer window, confirming real overlap — the device is saturated by
two threads, and the consumer stages pay for it.

This is the crux for Option B: on a device that is **already saturated by a
concurrent producer**, issuing fewer/larger classifier calls may recover little
or nothing, because the win B was measured on (per-call launch overhead) is not
what is costing 185 ms/frame here. It may also be the more valuable finding in
its own right — a 7× gap between a stage's uncontended and in-pipeline cost is
a bigger lever than anything cross-frame batching offers.

## What the stage-level data still supports

Every stage, both platforms, every tier lands in a 1.0–1.35× band on an idle
device, mostly within ~2× of the noise floor. Crop warp does not shrink at
all, because B only affects the backend forward. Against that, B costs:
deferring `write_downstream` against `runner.py`'s strict window order, a new
cancel-path flush invariant, a crop-count-bounded accumulator, a
pre-extracted-batch parameter for `run_cnn_batch` (it builds its own crops),
and loss of gate byte-identity on identity + pose clips.

## Superseded judgement (kept for the record)

Before the in-pipeline A/B, the conditional judgement here was "head-tail-only
accumulation is the candidate worth pursuing" — largest consumer stage,
byte-identical under regrouping. **The A/B killed it:** head-tail is exactly
1.000 on CUDA and 1.58× SLOWER on MPS. The stage-level 1.23× was an artifact
of measuring on an idle device.

**CNN and pose accumulation should stay off the table**, and the "small
differences add up over long runs" argument is precisely why: 10 h × 30 fps ≈
1M frames × ~17 detections, each carrying 2.2e-3 (CUDA) or 0.048 (CoreML)
probability drift into Bayes log-compat identity accumulation, plus up to
1 px pose drift — while simultaneously breaking the byte-identity gate that
is the only thing protecting runs of that length.

All three planned measurements are now DONE (contention isolation, MPS A/B,
CUDA A/B) and all three point the same way. See "THE ANSWER" at the top.

The A/B patch lives on branch `spike/crossframe-ab` (env-gated behind
`HYDRA_AB_CROSSFRAME`), worktrees `.worktrees/ab-crossframe` on both boxes.
It is a measurement artifact and must not be merged.

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
