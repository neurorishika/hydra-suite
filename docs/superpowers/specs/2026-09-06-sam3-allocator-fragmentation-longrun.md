# SAM3 LoRA — does `expandable_segments:True` hold flat at FULL run length?

Date: 2026-09-06. Box: **courtship** (RTX 4090, 24 GB, driver 535.309.01, CUDA 12.2), idle at 215 MiB
before and between runs. mehek was not touched.

**ANSWER: yes. The staircase is eliminated, not deferred.** Peak `max_memory_reserved` over a complete
**2430-optimizer-step** run (10 epochs x 243 steps — the live run's exact horizon) is **7.1250 GiB**, and the
**last new maximum appeared at step 302**. Steps 303 through 2430 — 2128 consecutive steps spanning eight
epoch boundaries — produced **no new maximum in either reserved or allocated**.

## Provenance

| | |
|---|---|
| Commit | **`9598c9e0938c699116ecce2edf60a26b4044e835`** (local `main`; newer than `train-pin` b9e92bc7) |
| Transport | incremental `git bundle 3c57045f8..main`, fetched into a **fresh** scratch clone `~/sam3_expandable_probe/hydra-suite` (cloned read-only from `~/hydra-suite`). Nothing pushed to origin. |
| Parent env | `hydra-cuda`, `PYTHONPATH=~/sam3_expandable_probe/hydra-suite/src` |
| Sidecar env | **`hydra-sam3`** (torch 2.11.0+cu128, numpy 1.26.4, `sam3` importable, CUDA true) |
| Verified in-process | `PROBE env: PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'` and `PROBE hydra_suite=/home/rutalab/sam3_expandable_probe/hydra-suite/src/hydra_suite/__init__.py`, printed by the training process itself, with a hard abort if the flag is absent |
| Containment | `systemd-run --user --unit=sam3-expandable-probe -p MemoryMax=60G -p MemorySwapMax=0 -p TasksMax=4096`; peak tree RSS 6.78 GiB, min system available 113 GiB — nowhere near the cap |
| Config | batch 1, grad_accum 8, rank 16 / alpha 32, bf16, lr 5e-5, 10 epochs, `keep_empty_tiles=false`, `label_quality_acknowledged=true`, `adapt_scoring_head` **not** enabled, tiles 1766x1766 |
| Result | `success`, exit 0, adapters written; `peak_observed_device_used_bytes` (nvidia-smi level) 8.16 GiB |

### Dataset — 1871 survived

Rebuilt on-box from the same source (`obb-6609f9b105`, rsynced): 78 label files, **1871 non-empty lines, all
class 0**. The `build_manifest.json` matches the local build **exactly**: train 486 tiles / 3363 ann /
21 fragment / 20 downgraded / 1 fragment-only; valid 147 / 1183 / 14 / 14 / 0; `tile_px` [1766,1766]; the
same 16 held-out stems. Only `reference_body_px` differs in the last float32 digits
(97.13015747070312 on courtship vs 97.13015937805176 on the Mac) — does not change tiling.
Loader shape confirmed non-empty: `486 tiles, 1944 datapoints, batch=1 grad_accum=8, 1944 micro-batches/epoch,
243 steps/epoch x 10 epochs`.

### Instrumentation (probe-only, not committed)

`logging_steps` 10 -> **1**, and the step line was extended to emit peak **and** current, reserved **and**
allocated. 2430 step records, one per optimizer step, no gaps.

## Maxima table — every distinct new maximum, to 2430 steps

Reserved peak (`torch.cuda.max_memory_reserved`):

| step | epoch | reserved peak (GiB) | allocated peak then | gap |
|---:|---:|---:|---:|---:|
| 1 | 0 | 6.6953 | 6.2488 | 0.4465 |
| 24 | 0 | 6.9297 | 6.4957 | 0.4340 |
| 67 | 0 | 6.9492 | 6.5023 | 0.4469 |
| 170 | 0 | 7.0664 | 6.5448 | 0.5216 |
| 214 | 0 | 7.1055 | 6.5742 | 0.5313 |
| **302** | 1 | **7.1250** | 6.6337 | 0.4913 |
| — | — | *no further maximum through step 2430* | | |

Allocated peak (`torch.cuda.max_memory_allocated`):

| step | epoch | allocated peak (GiB) | reserved then | gap |
|---:|---:|---:|---:|---:|
| 1 | 0 | 6.2488 | 6.6953 | 0.4465 |
| 17 | 0 | 6.3386 | 6.6953 | 0.3567 |
| 24 | 0 | 6.4957 | 6.9297 | 0.4340 |
| 67 | 0 | 6.5023 | 6.9492 | 0.4469 |
| 76 | 0 | 6.5448 | 6.9492 | 0.4044 |
| 214 | 0 | 6.5742 | 7.1055 | 0.5313 |
| **302** | 1 | **6.6337** | 7.1250 | 0.4913 |
| — | — | *no further maximum through step 2430* | | |

Checkpoints along the flat tail (all identical): steps 486, 729, 1000, 1215, 1500, 1701, 2000, 2187, 2400,
2430 — reserved 7.1250, allocated 6.6337, gap 0.4913, `res_now` 7.1250, `alloc_now` 3.6949.

**Scope of the 7.1250 GiB figure:** it is the peak over the *training loop* (optimizer steps 1-2430).
Validation (`_evaluate_and_write`) runs after step 2430 in the same process and is **not** instrumented, so
if eval allocated more than training this would under-state the process peak. The outer bound that does
cover eval is the containment telemetry: `peak_observed_device_used_bytes` = **8.16 GiB** device-level
(includes the CUDA context and the 215 MiB desktop) — consistent with no dramatic eval spike.

The gap never exceeds **0.53 GiB** and is essentially flat from step 1. Total reserved growth over the whole
run is **+6.4% (6.6953 -> 7.1250)**, and allocated grows **+6.2%** over the same span — the two series move
together. That is the signature of *real* transient demand (largest-tile / most-instances batches being met
for the first time), not of fragmentation. Contrast the default allocator's measured behaviour, where
reserved grew +35% while allocated grew +2%.

## Verdicts

**(a) Does expandable_segments hold flat to full run length? YES.** Final peak **7.1250 GiB reserved /
6.6337 GiB allocated**, reached at **step 302 of 2430** and never exceeded again across 2128 further steps
and eight epoch boundaries. The 120-step observation from the earlier session was not a deferral artefact.

**(b) True fragmentation overhead: ~5.87 GiB, i.e. the default allocator reserves ~1.82x what the workload
needs.** Default-allocator peak 12.99 GiB (mehek) minus the expandable steady state 7.125 GiB = **5.865 GiB**
of pure allocator fragmentation — 45% of the default-allocator peak is caching-allocator waste, not model,
activations or optimizer state. Caveat: the 12.99 figure is from mehek (RTX 6000 Ada) and the 7.125 from
courtship (RTX 4090). Same code, same dataset, same config, so the comparison is meaningful, but it is not
a same-box A/B. I attempted a same-box 243-step default-allocator reference run; **launching it was blocked
by the permission classifier**, so the cross-box caveat stands and is not resolved here.

**(c) Would a short probe have bounded the full-run peak? Not exactly — but the shortfall is small, bounded
and known.**

| probe length | reserved peak observed | full-run peak | shortfall |
|---:|---:|---:|---:|
| 2 steps | 6.6953 | 7.1250 | +0.4297 GiB (**+6.42%**) |
| 30 steps | 6.9297 | 7.1250 | +0.1953 GiB (**+2.82%**) |
| 60 steps | 6.9297 | 7.1250 | +0.1953 GiB (**+2.82%**) |
| 120 steps | 6.9492 | 7.1250 | +0.1758 GiB (**+2.53%**) |

No probe length short of ~302 steps sees the true peak. But every one of them **under**-reads by at most
6.4%, and 30+ steps by under 3% — a bounded, monotone, small error, unlike the default allocator's +71%
climb from step 1 to step 320. **A 30-60 step probe under expandable_segments plus a ~10% safety margin
genuinely bounds the full-run peak; a 2-step probe needs ~10-15%.** Auto batch sizing is tractable under
this flag and is not tractable without it.

**(d) Recommendation: set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` by default for SAM3 LoRA
training.** It converts an unboundable 12.99 GiB into a 7.13 GiB peak that a short probe can bound, frees
~5.9 GiB of a 24 GB card, and is the difference between "this role needs a 16 GiB+ card and we cannot say
how much more" and a defensible hardware requirement. Downsides observed: **none.** No allocator warning of
any kind in the 2466-line log. Throughput 2430 steps in ~125 min = **~3.09 s/step** (spot-measured
3.16 s/step early on), loss descended normally (92 -> ~5-47 band), zero skipped steps, run exited 0 with a
validated adapter. Honest limit: with the same-box default-allocator run blocked, I have **no matched
throughput baseline on courtship**, so I can state that the flag caused no *observable* slowdown but cannot
put a number on a possible few-percent step-time cost. Set the variable in `sam3_env_environ()` alongside
`KMP_DUPLICATE_LIB_OK` so the sidecar gets it regardless of how the parent was launched — the parent's
export reaching the child was verified in this run (child inherits `os.environ`), but making it explicit
removes the dependency on the caller's shell.

## Blockers

One, non-fatal: the optional same-box default-allocator reference run could not be launched (permission
classifier denied the second `systemd-run`). Everything the task asked for was measured. Scratch dirs on
courtship (`~/sam3_expandable_probe`) removed after the log was copied back; `~/hydra-suite` on courtship
was cloned from but never written to.
