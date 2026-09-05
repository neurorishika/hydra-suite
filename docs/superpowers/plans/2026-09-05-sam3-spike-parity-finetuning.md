# SAM3 spike-parity finetuning implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reproduce the research spike's *learning* configuration inside the
production pipeline, so our finetuned checkpoints match or beat the spike's
measured accuracy — then keep every production guard that provably cannot
affect the trained model.

**Architecture:** Parity is the default; each surviving deviation must be
named, justified, and individually testable. Three groups:

1. **Keep our deviations** (infrastructure, provably model-neutral): preflight,
   sidecar env, atomic artifacts, streaming logs, NaN abort, non-finite
   gradient skip, per-epoch checkpoints.
2. **Revert to spike parity** (learning design, no justification found):
   adapter surface, training precision, optimizer cadence.
3. **Deviate deliberately** (the spike is wrong): tile-seam fragment handling.

**Tech Stack:** SAM3 LoRA sidecar (`hydra_suite.training.sam3_lora`), Meta
`sam3`, calibration harness `core/inference/semantic/calibration.py`.

**Spec:** `docs/superpowers/specs/2026-09-05-sam3-finetune-quality-audit.md`
(the measured accuracy gap and its root-cause ranking) and
`docs/superpowers/specs/2026-09-04-sam3-spike-vs-build-consolidation.md`
(the earlier design comparison — **two of its conclusions are overturned by
the newer audit; see Global Constraints**).

## The evidence this plan answers

Both checkpoints, same 12 labelled frames, 275 instances, tile fraction 0.1:

| conf | spike recall | spike extra/f | ours recall | ours extra/f |
|---|---|---|---|---|
| 0.30 | 0.727 | 6.83 | 0.724 | 8.33 |
| 0.50 | 0.720 | 6.33 | 0.720 | 6.92 |
| 0.65 | 0.720 | 5.83 | 0.702 | 6.42 |
| 0.85 | 0.611 | 5.00 | 0.534 | 5.50 |

Median IoU 0.868 vs 0.863. Recall parity, **+1.2–2.0 false positives per
frame**, faster high-confidence collapse — and the frames are from *our*
training source, so the comparison already favours us.

---

## Global Constraints

- **Parity is the default. Every deviation must be justified in a comment at
  the point of deviation**, naming what it costs and why it is worth it.
  "Ours is more modern/safer/cleaner" is not a justification for a change that
  alters what the model learns.
- **Two conclusions of the 2026-09-04 audit are overturned** by the 2026-09-05
  one and must NOT be re-applied:
  (a) "the spike never adapted SAM3's own clone MHA" — **disproved by its
  checkpoint**, those sites carry non-zero trained adapters;
  (b) "keep the last-epoch checkpoint because val loss anti-correlates" —
  that rationale rests on a fold whose val split is byte-identical to train.
- **Do not fix `is_crowd` by copying the spike.** Both implementations train
  truncated instances as full positives and nothing in either SAM3 tree reads
  `is_crowd`. Parity here would reproduce a bug.
- **Change one axis per run.** The current mess exists because surface,
  precision, batch schedule, data geometry and SAM3 version all changed at
  once, leaving nothing attributable. Each task below is a separately
  measurable run.
- **Keep SAHI tiling.** 4512² frames with ~97 px animals need it; the audit's
  doubt is noted but whole-image training is out of scope here.
- **Never bake a measurement into a source constant** (see
  `_MEASURED_BF16_DEVICE_PEAK_BYTES`, which was 3.7× the real figure and
  excluded every card below ~32 GiB).
- **GPU etiquette:** `courtship.taild08eb9.ts.net` (RTX 4090, 24 GB) is the
  training box; `mehek.taild08eb9.ts.net` is shared. Check
  `nvidia-smi` and `systemctl --user is-active sam3-*` first; never kill
  another user's process. Launch under
  `systemd-run --user --unit=... -p MemoryMax=60G -p MemorySwapMax=0 -p TasksMax=4096`.
- **Isolation:** `git worktree add .worktrees/sam3-parity -b codex/sam3-spike-parity HEAD`.
  Never `git stash`, never `git clean -fdx`, never `rm -rf`.
- Tests: `conda activate hydra-mps`, `KMP_DUPLICATE_LIB_OK=TRUE`,
  `python -m pytest tests/ -k "sam3" -q`.

---

## File Structure

- **Modify** `src/hydra_suite/training/sam3_lora/lora.py` — new adapter scope
  for `dot_prod_scoring`; clone-MHA replacement; `SUBMODULE_PREFIXES`.
- **Modify** `src/hydra_suite/training/contracts.py` — new `adapt_*` flags,
  precision default.
- **Modify** `src/hydra_suite/training/sam3_lora/sizing.py` — per-flag
  trainable-parameter counts (measured, not derived).
- **Modify** `src/hydra_suite/training/sam3_lora/publish.py` — merge path for
  any newly reachable fused/clone attention.
- **Modify** `src/hydra_suite/training/sam3_lora/cli.py` — optimizer cadence.
- **Modify** `src/hydra_suite/training/sam3_lora/dataset_build.py` — fragment
  handling.
- **Tests:** `tests/test_sam3_split_mha.py`, `tests/test_sam3_lora_scopes.py`
  (new), `tests/test_sam3_dataset_build.py`, `tests/test_sam3_publish.py`.

---

## Task 0: Establish the parity baseline (measurement only, no code)

Before changing anything, make the comparison reproducible and record the
starting point. Without this, later tasks cannot be attributed.

- [ ] **Step 1:** Re-run the two-model comparison on the **16 held-out
  validation frames** from the existing build, not the 12 training frames.
  The current numbers are biased toward our model; the parity target must be
  measured on frames neither model trained on. (The spike trained on different
  data entirely, so only our model has any exposure.)
- [ ] **Step 2:** Record both models' full confidence sweeps to
  `docs/superpowers/specs/2026-09-05-parity-baseline.json`, committed.
- [ ] **Step 3:** Note in the ledger which frames were used and the exact
  calibration arguments, so every later run uses identical settings.

**Acceptance:** a committed baseline both later tasks are measured against.

---

## Task 1: Adapt `dot_prod_scoring` (the top-ranked cause)

**Files:** `lora.py:50-62`, `contracts.py:233-238`, `sizing.py`,
`tests/test_sam3_lora_scopes.py`

`lora.py:50-53` states `dot_prod_scoring` is "covered by NO flag,
deliberately". The audit found it is the spike's **single biggest mover**
(`prompt_proj` lora_B norm 0.174) and is the head that emits detection
confidence — a frozen scoring pathway over adapted features is the leading
explanation for excess false positives at unchanged IoU.

**Interfaces:**
- Produces: `Sam3LoraParams.adapt_scoring_head: bool = True` and
  `SUBMODULE_PREFIXES["adapt_scoring_head"] = ("dot_prod_scoring",)`.

- [ ] **Step 1: Write the failing test** — with the flag on, injection reaches
  `dot_prod_scoring.prompt_proj` and `dot_prod_scoring.hs_proj`; with it off,
  neither is wrapped; the flag defaults to on.
- [ ] **Step 2:** Run it; expect failure (no such flag).
- [ ] **Step 3:** Implement. Replace the "deliberately excluded" comment with
  the evidence that overturned it — do not delete the reasoning, correct it.
- [ ] **Step 4:** Measure the new per-flag parameter count on a live
  `build_sam3_image_model` and update `sizing.py` from the measurement.
- [ ] **Step 5:** Verify the merge round-trip: these are plain `nn.Linear`s,
  so `merge_adapters` should already resolve them — assert it, don't assume.
- [ ] **Step 6:** Full `-k "sam3"` suite green. Commit.
- [ ] **Step 7: RUN A.** Retrain on the identical prepared dataset, same
  epochs/precision/cadence as the 2026-09-04 run, changing only this flag.
  Re-run the Task 0 sweep. **This is the decisive test of the top cause.**

**Acceptance:** extras/frame move toward the spike's at matched recall. Record
the result whichever way it goes — a null result convicts a different cause
and is equally valuable.

---

## Task 2: Reach the clone-MHA sites

**Files:** `lora.py`, `publish.py`, `tests/test_sam3_split_mha.py`

The 2026-09-04 audit said the spike never adapted SAM3's own
`model_misc.MultiheadAttention` clone; the 2026-09-05 audit **disproved that
from the checkpoint**. Those sites are part of the 108 modules we cannot
reach. Our existing skip (Linears whose parent has `in_proj_weight`) covers
both torch's MHA — now handled by `SplitMultiheadAttention` — and the clone,
which is still skipped.

- [ ] **Step 1: Write the failing tests** — the clone is replaced and its four
  projections wrapped; forward parity against the unmodified clone across the
  argument shapes SAM3 uses at those call sites (attn masks, key padding
  masks, `batch_first` both ways); merged `in_proj_weight` reassembles exactly;
  zero-init merge is bit-identical to base.
- [ ] **Step 2-5:** Fail, implement, measure `sizing.py`, verify the publish
  round-trip produces a key-identical stock state dict.
- [ ] **Step 6:** Confirm the adapted-module count reaches the spike's **314**
  under matching flags (text encoder OFF, matching its config). A different
  number means the surfaces still differ — investigate before proceeding.
- [ ] **Step 7: RUN B.** Retrain, re-sweep, compare against Task 1.

**Ruling if forward parity cannot be achieved** for some call site: skip that
site, record it in the ledger with the shapes that failed, and proceed. Do not
ship an attention module that is "close enough" — a silently wrong forward
would poison training in a way no test here would catch.

---

## Task 3: fp32 training parity

**Files:** `contracts.py`, `preflight.py`, `cli.py`

The spike trained in **pure fp32** (its `mixed_precision: bf16` is dead
config). We mandate bf16. The earlier rejection of fp32 ("~58 GiB, does not
fit") was arithmetic on the since-corrected 29 GiB constant; with the measured
~8–10.4 GiB bf16 peak, fp32 is expected around 16–19 GiB and **fits the
24 GB 4090**.

- [ ] **Step 1:** Measure, do not assume. Run a short fp32 probe on courtship
  and read `vram_peak` from the existing telemetry.
- [ ] **Step 2:** Set `_FP32_DEVICE_PEAK_MULTIPLIER` from that measurement.
- [ ] **Step 3: RUN C.** Retrain in fp32, changing only precision. Re-sweep.
- [ ] **Step 4:** If fp32 does not fit or does not help, keep bf16 and record
  the measurement — the point is to settle it with a number.

---

## Task 4: Optimizer cadence parity

**Files:** `cli.py`, `contracts.py`

Spike: batch 4, `optimizer.step()` every batch, no accumulation, no clipping,
no scheduler. Ours: batch 1, accum 8, clip 1.0, warmup+cosine.

- [ ] **Step 1:** Determine whether batch 4 fits at 1008 px (the probe from
  the auto-batch plan, or a direct measurement). `contracts.py:217` claims
  "batch 2 OOMs at 1008 px on a 47 GB card" — that predates `perflib_compat`
  and the streaming dataloader and is **unverified**.
- [ ] **Step 2: RUN D.** Match the spike's effective batch and cadence,
  changing only this. Re-sweep.
- [ ] **Step 3: KEEP the non-finite gradient skip regardless.** It is not part
  of the spike's design, it protects against a failure that cost us three
  runs, and it is inert when gradients are finite.

---

## Task 5: Fragment handling (deliberate deviation)

**Files:** `dataset_build.py:56-57,239,462`, `tests/test_sam3_dataset_build.py`

`MIN_RETAINED_AREA_FRAC = 0.5` marks tile-clipped instances `iscrowd=1`, and
**nothing in either SAM3 tree reads `is_crowd`** — so 380/3286 truncated
animals (median 30% of full area, no lower bound) train as exhaustive full
positives. The spike shares this defect at a similar rate, so it does not
explain the gap, but it plausibly sets the ~5–8 extras/frame floor *both*
models share.

- [ ] **Step 1: Write the failing tests** — a fragment below the retained-area
  floor is excluded from the query's `object_ids_output` and its tile is
  marked `is_exhaustive=False`; a fragment above the floor is retained as a
  normal positive; totals are reported in the build manifest.
- [ ] **Step 2-4:** Fail, implement, pass.
- [ ] **Step 5:** Do NOT silently drop fragments while leaving
  `is_exhaustive=True` — that teaches the model the animal is absent, which is
  worse than the current bug. The tile's exhaustiveness claim and its
  instance list must stay consistent.
- [ ] **Step 6: RUN E.** Rebuild the dataset, retrain, re-sweep.

---

## Verification

- [ ] `python -m pytest tests/ -k "sam3" -q` green after every task.
- [ ] `make lint-moderate`.
- [ ] Each RUN's sweep appended to the baseline JSON with the single axis that
  changed, so the contribution of each change is attributable.
- [ ] A published checkpoint from the final configuration loads through
  `Sam3SemanticLabeler.from_variant` with the load guard passing.
- [ ] **Success criterion:** on the held-out frames, extras/frame at matched
  recall ≤ the spike's, with median IoU no worse.

## Known reference points

- Spike checkpoint: 314 adapted modules, rank 16, text encoder OFF, 10 epochs
  / ~1080 optimizer steps, pure fp32, batch 4.
- Ours (2026-09-04): 206 modules, rank 16, 3 epochs / ~3348 steps, bf16,
  batch 1 × accum 8, `val_loss_mean` 0.166, 1 skipped step.
- VRAM: 7.83 GiB steady, 10.43 GiB peak at bf16 batch 1 — identical on
  RTX 6000 Ada and RTX 4090.
