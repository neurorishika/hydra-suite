# TrackerKit inference autotuner: Mehek measurement study

**Date:** 2026-09-06
**Status:** Completed measurement study; no production autotuner implemented
**Study branch:** `codex/inference-autotuner-study`
**Remote checkout:** `~/inference-autotuner-study` on Mehek
**Purpose:** Determine whether detector-frame, SAHI-tile, pose, head-tail,
identity, and pipeline-depth batching can be selected automatically and define
the evidence-based constraints for a persistent system-specific tuner.

## Bottom line

Full automation is feasible, but an independent "largest batch that fits"
policy is not. The measured optima are non-monotonic, depend on frame geometry
and the complete active model combination, interact with pipeline depth, and
can change model outputs. The required architecture is a bounded,
system-specific, correctness-gated coordinate tuner with analytical admission,
fresh-process containment, an end-to-end acceptance test, and a persistent
versioned profile cache.

No production behavior was changed in this study.

## Test system

| Component | Measured value |
|---|---|
| Host | Mehek, Linux 6.8.0-138-generic x86_64 |
| CPU | Intel Core i9-14900K, 32 logical CPUs |
| Host memory | 134,759,956,480 bytes |
| GPU | NVIDIA RTX 6000 Ada Generation |
| GPU UUID | `GPU-088a4fff-9dff-dcce-5c6e-b7b29fc28177` |
| VRAM | 49,140 MiB |
| Driver | 595.84 |
| Python | 3.13.13 |
| PyTorch | 2.11.0+cu130 |
| CUDA runtime | 13.0 |
| cuDNN | 9.19.0 (`91900`) |
| TensorRT | 10.16.1.11 |
| Ultralytics | 8.4.41 |
| OpenCV / NumPy | 4.13.0 / 2.4.3 |
| Final study commit on Mehek | `6f73de1e7072ef9b2184afc705f3a7f5756c5adf` |

The `hydra-cuda` environment required cuDNN to precede the conda library
directory in `LD_LIBRARY_PATH`; the default inherited loader order found cuDNN
9.10.2 while PyTorch required 9.19. A production tuner must record the resolved
runtime libraries and fail preflight rather than benchmark a partially broken
environment.

## Method

The study harness is
`tools/experiments/inference_autotuner_study.py`. Its case matrices are
`inference_autotuner_mehek_screen.json` and
`inference_autotuner_mehek_joint.json` in the same directory.

Each trial:

1. ran in a fresh subprocess and private output/config directory;
2. used the real `tools/equivalence/runner.py` and production inference path;
3. disabled backward tracking, postprocessing, rendered video, dataset export,
   and density output so the measured surface was forward inference;
4. enabled `HYDRA_PROFILE=1` and retained stage-span timing;
5. sampled total GPU memory, utilization, power, and temperature with
   `nvidia-smi` every 100 ms and sampled process-tree RSS with `psutil`;
6. atomically wrote a result after every case; and
7. used isolated detection/result caches, never a previous trial's outputs.

The full set comprised 196 fresh runner launches: 191 successful runs and five
intentional or diagnostic admission failures. Before each heavy run, Mehek was
checked for unrelated SLEAP/Hydra jobs and for idle GPU state. No unrelated
process was stopped.

Screening used short clips and two or three repeats. Apparent knees were then
confirmed with 3-5 longer repeats. Candidates were grouped rather than
randomized; the production specification corrects this limitation by requiring
interleaved randomized blocks. Thermal peaks reached 84 C during some builds,
but the confirmed per-candidate ranges were narrow.

## Results

Times below are profiler forward-wall medians. Memory is the maximum sampled
total GPU allocation for the process during those repeats. FPS includes the
same forward region for every row in a table.

### Native detector frame batching is geometry-specific

Five complete 500-frame repetitions on the 1200 x 1200 fly clip:

| Frame batch | Median wall | Range | Median FPS | Peak VRAM |
|---:|---:|---:|---:|---:|
| 1 | 5.960 s | 5.912-6.018 | 83.89 | 1,293 MiB |
| **4** | **5.129 s** | **5.087-5.146** | **97.48** | **1,977 MiB** |
| 8 | 5.194 s | 5.183-5.233 | 96.26 | 2,651 MiB |
| 16 | 5.290 s | 5.255-5.305 | 94.52 | 4,677 MiB |

Batch 4 improved throughput 16.2% over batch 1 and was also faster and leaner
than batches 8 and 16. The optimum was a knee, not the largest feasible batch.

Three 96-frame repetitions on the 2632 x 2632 ant pose clip, detector-only:

| Frame batch | Median wall | Median FPS | Peak VRAM |
|---:|---:|---:|---:|
| **1** | **2.223 s** | **43.19** | **1,085 MiB** |
| 2 | 2.281 s | 42.08 | 1,431 MiB |
| 4 | 2.252 s | 42.63 | 1,947 MiB |
| 8 | 2.366 s | 40.57 | 2,845 MiB |

At 4512 x 4512, 64-frame screening selected batch 1 by the conservative
near-optimal rule: batch 1 and 4 had approximately equal wall time (3.07 vs
3.08 s), batch 8 regressed to 3.24 s, and peak VRAM rose from 1,669 to 3,405
to 5,903 MiB. Batches 16 and 32 were rejected by the existing frame-buffer
guard before inference.

An earlier full-combination batch-8/depth-2 case quantified that guard:
one decoded frame was 61,074,432 bytes; eight frames across four retained live
windows estimated 1,954,381,824 bytes, above the 536,870,912-byte budget. This
must be an analytical candidate-pruning step, not an OOM experiment.

### SAHI tile batching is non-monotonic

Three 64-frame repetitions on the 4512 x 4512 ant clip:

| Tile batch | Median wall | Range | Median FPS | Peak VRAM |
|---:|---:|---:|---:|---:|
| **1** | **38.734 s** | **38.270-39.124** | **1.65** | **1,669 MiB** |
| 2 | 42.555 s | 42.491-42.646 | 1.50 | 2,253 MiB |

The 16-frame screen showed monotonically worse throughput from requested
batches 1, 2, 4, 8, and 16. Requested batches 32 and 64 converged on the same
~10,859 MiB peak, demonstrating admission clamping. Batch 1 was about 10%
faster than batch 2; doubling until OOM would move in exactly the wrong
direction.

The current opt-in, process-local SAHI autotuner was then tested twice in fresh
processes over 64 frames:

| Mode | Median wall | Median FPS | Peak VRAM |
|---|---:|---:|---:|
| Manual tile batch 1 | 38.734 s | 1.65 | 1,669 MiB |
| Current process-local auto | 43.734 s | 1.46 | 15,249 MiB |

Both auto runs repeated the tuning work. They were 11.5% slower than manual
batch 1 and used about 9.1 times the peak VRAM. The isolated tile-predict rate
objective did not predict full-run throughput.

### Crop-stage batches need separate optima and real warmup

Three 96-frame repetitions around each apparent knee:

| Tunable | Candidate | Median wall | Median FPS | Peak VRAM |
|---|---:|---:|---:|---:|
| Pose | 1 | 16.393 s | 5.86 | 2,870 MiB |
| Pose | **4** | **15.746 s** | **6.10** | 3,696 MiB |
| Pose | 8 | 15.768 s | 6.09 | 3,696 MiB |
| Head-tail | 1 | 15.248 s | 6.30 | 3,694 MiB |
| Head-tail | **4** | **14.948 s** | **6.42** | 3,696 MiB |
| Head-tail | 8 | 16.157 s | 5.94 | 3,700 MiB |
| Head-tail | 16 | 15.969 s | 6.01 | 3,696 MiB |

The 32-frame screen incorrectly made head-tail batch 1 look much faster than
batch 4. At 96 frames, after startup was amortized, batch 4 won. A tuner needs
explicit warmup and sequential stopping; a fixed small number of frames is not
reliable.

Identity batches over 96 frames:

| Identity batch | Median wall | Median FPS | Peak VRAM |
|---:|---:|---:|---:|
| 1 | 23.254 s | 4.13 | 3,594 MiB |
| **8** | **20.311 s** | **4.73** | **3,592 MiB** |
| 16 | 19.934 s | 4.82 | 3,594 MiB |
| 25 | 19.979 s | 4.81 | 3,592 MiB |
| 64 | 19.872 s | 4.83 | 3,592 MiB |

Batch 8 was the smallest candidate within about 2% of the faster plateau, but
the strict correctness test below disqualified it for this configuration.
Values above the number of crops in a frame did not buy meaningful throughput.

### Pipeline depth and batch choices interact

At detector batch 8 on the 1200 x 1200 fly clip, pipeline depths 1-4 had median
64-frame walls of 1.239, 1.230, 1.217, and 1.226 s. The maximum apparent gain
was under 2%; a conservative tie policy would keep the shallower option for
this detector-only workload.

That conclusion did not transfer to the full ant pipelines. Independently
selected stage settings combined at depth 1 regressed:

| Full pipeline | Configuration | Median wall | Median FPS | Peak VRAM |
|---|---|---:|---:|---:|
| Pose/head-tail | manual: depth 2, pose 25, head-tail 25 | 16.987 s | 7.54 | 2,876 MiB |
| Pose/head-tail | naive: depth 1, pose 4, head-tail 4 | 17.776 s | 7.20 | 2,766 MiB |
| Identity | manual: depth 2, pose 25, head-tail 25, CNN 25 | 23.340 s | 5.48 | 3,010 MiB |
| Identity | naive: depth 1, pose 4, head-tail 4, CNN 8 | 26.428 s | 4.84 | 2,944 MiB |

Targeted coordinate trials at depth 2 found:

| Full pipeline | Candidate | Median wall | Median FPS | Decision |
|---|---|---:|---:|---|
| Pose/head-tail | pose 25, head-tail 4 | 16.725 s | 7.65 | +1.5%; marginal |
| Pose/head-tail | **pose 4, head-tail 4** | **16.595 s** | **7.71** | **+2.3%; candidate** |
| Identity | pose 25, head-tail 4, CNN 25 | 23.569 s | 5.43 | slower; reject |
| Identity | pose 25, head-tail 4, CNN 8 | 24.336 s | 5.26 | slower; reject |

The correct identity outcome is to retain the manual baseline. "No change" is
a first-class successful tuning result.

### TensorRT preparation and reuse are separate concerns

The `gpu_fast` fly screen encountered missing batch-profile artifacts:

| Profile | Cold wall | Hot second wall | Cold peak | Hot peak |
|---:|---:|---:|---:|---:|
| batch 8 | 254.543 s | 3.928 s | 8,041 MiB | 2,909 MiB |
| batch 16 | 310.359 s | 3.903 s | 14,537 MiB | 4,743 MiB |

Cold engine compilation dwarfs inference and cannot be included in the
steady-state objective. The current export also stages generic `.onnx` and
`.engine` files in the shared model directory before producing `_b8.engine`
or `_b16.engine`. A production tuner needs a per-artifact build lock, private
temporary staging, atomic promotion, and separate `prepare_seconds` telemetry.

The study caused the normal runtime exporter to create/update the following
Mehek artifacts; they were not deleted after measurement:

- `20260503-171130_26x_fly_train7.onnx`
- `20260503-171130_26x_fly_train7.engine`
- `20260503-171130_26x_fly_train7_b1.engine` and metadata
- `20260503-171130_26x_fly_train7_b8.engine` and metadata
- `20260503-171130_26x_fly_train7_b16.engine` and metadata

### Correctness and determinism

| Comparison | Result |
|---|---|
| Selected pose profile vs manual, 2,472-row forward CSV | Exact geometry and row equality |
| Selected pose rich CSV, 2,472 x 55 | Every keyed column identical |
| Selected pose final tracking CSV, 3,200 x 21 | Equivalent; zero positional/angular delta |
| Fly detector batch 1 vs 4, 1,491 rows | Equivalent; p99 position delta 0, rare max 1 px, theta mean `3.382e-05` rad |
| SAHI tile batch 1 vs 2 | Rejected: 1,074 vs 1,080 rows; 1/7 unmatched |
| Identity batch 1 vs 8, 1,541 x 68 | Rejected strictly: one categorical identity row differed in two derived columns |
| SAHI batch 1 repeat 1 vs repeat 2 | Exact |
| Identity batch 1 repeat 1 vs repeat 2 | Exact across all 68 columns |

The SAHI and identity differences exceeded their exact same-setting
determinism floors. Batching is not semantically free; categorical identity and
detection-count gates must be stricter than geometry tolerance alone.

## Conclusions that constrain the design

1. Candidate capacity must be analytically admitted before models are loaded.
2. Throughput is non-monotonic; largest-fit and doubling-to-OOM are invalid.
3. Frame geometry, slice geometry, target density, full active model set, and
   software/runtime versions all belong in the persistent key.
4. Stage-local timing is useful for screening but cannot authorize a profile.
   A full-pipeline coordinate trial is required.
5. Explicit warmup and sufficiently long, interleaved measurements are
   required; 32 frames misranked head-tail candidates.
6. The selected result must clear a meaningful-gain/confidence threshold. The
   smallest-memory candidate inside the near-optimal band wins.
7. Output comparison against the same-setting determinism floor is a hard
   gate. Categorical identity must remain exact by default.
8. TensorRT artifact preparation, steady-state tuning, and tuning-profile
   persistence are three different caches with different locking/invalidation.
9. A cache hit must still be down-admitted against current free memory and
   system load.
10. Retaining the current settings is a valid and frequently correct outcome.

## Limits

This is one machine, one RTX 6000 Ada, one CUDA stack, and a small fixture set.
It establishes the architecture and invalidates several tempting heuristics;
it does not establish portable batch constants for other GPUs, MPS, CPU,
different model versions, different slice geometry, or materially different
animal density. The proposed cache is deliberately exact and system-specific
for that reason.

Raw manifests remain on Mehek under
`~/inference-autotuner-results/{detector-screen-v1,sahi-screen-v1,pose-headtail-screen-v1,identity-screen-v1,depth-screen-v1,detector-confirm-v1,sahi-confirm-v1,pose-headtail-confirm-v1,crop-neighbors-confirm-v1,identity-low-confirm-v1,ant2632-detector-v1,joint-confirm-v1,joint-interactions-v1,current-sahi-auto-v1}`.
