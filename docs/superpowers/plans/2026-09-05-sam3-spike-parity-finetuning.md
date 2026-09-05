# SAM3 spike-parity finetuning implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Status:** revised 2026-09-05 after an adversarial review found four blocking
defects (`docs/superpowers/specs/2026-09-05-parity-plan-review.md`). The
revision corrects the spike's actual hyperparameters, replaces an unachievable
acceptance criterion, fixes an interface that would have wrapped zero modules,
and rebuilds the measurement methodology. Task 4 is demoted; two tasks are
added.

**Goal:** Reproduce the research spike's *learning* configuration inside the
production pipeline, so our finetuned checkpoints match or beat the spike's
measured accuracy — keeping every production guard that provably cannot affect
the trained model.

**Architecture:** Parity is the default; each surviving deviation must be
named, justified, and individually testable.

**Tech Stack:** SAM3 LoRA sidecar (`hydra_suite.training.sam3_lora`), Meta
`sam3`, calibration harness `core/inference/semantic/calibration.py`.

**Spec:** `docs/superpowers/specs/2026-09-05-sam3-finetune-quality-audit.md`
(root-cause ranking), `docs/superpowers/specs/2026-09-05-parity-plan-review.md`
(this plan's review — **read it; every correction below traces to a finding**),
`docs/superpowers/specs/2026-09-04-sam3-spike-vs-build-consolidation.md`
(earlier design comparison, partially superseded).

## The evidence this plan answers

Both checkpoints, same 12 labelled frames, 275 instances, tile fraction 0.1:

| conf | spike recall | spike extra/f | ours recall | ours extra/f |
|---|---|---|---|---|
| 0.30 | 0.727 | 6.83 | 0.724 | 8.33 |
| 0.50 | 0.720 | 6.33 | 0.720 | 6.92 |
| 0.65 | 0.720 | 5.83 | 0.702 | 6.42 |
| 0.85 | 0.611 | 5.00 | 0.534 | 5.50 |

Median IoU 0.868 vs 0.863. Recall parity, **+1.2–2.0 extras/frame**, faster
high-confidence collapse — measured on frames from *our* training source, so
the comparison already favours us. **This delta is only a 2–3σ effect at the
available sample size; see Task 0, which must be fixed before it can judge
anything.**

## What the spike actually ran (corrected)

The earlier draft cited `full_lora_config.yaml`, a **template**. The run that
produced the shipped checkpoint used
`~/sam3_spike/work/configs/fold_all_r16.yaml` on mehek:

```
rank: 16   alpha: 32   batch_size: 1   learning_rate: 5.0e-5   num_epochs: 10
```

**`batch_size: 1`, not 4.** Rank, alpha and learning rate are therefore
**already identical to ours**. One DataLoader element is one image carrying
all its queries, which is the only reading consistent with its ~1080 optimizer
steps (108 images × 10 epochs). The cadence gap the earlier draft described
was largely fictional — see Task 4.

---

## Global Constraints

- **Parity is the default. Every deviation must be justified in a comment at
  the point of deviation**, naming what it costs and why. "Ours is more
  modern/safer/cleaner" does not justify changing what the model learns.
- **The authoritative spike reference is on mehek** at
  `~/sam3_spike/sam3_lora/` (with its own older SAM3 at
  `~/sam3_spike/sam3_lora/sam3/`) and `~/sam3_spike/work/configs/`. The local
  `/tmp/sam3_lora_vet/` copy is **hollow (0-byte files)** — do not read it.
- **Two conclusions of the 2026-09-04 audit are overturned** (both verified by
  the review) and must NOT be re-applied: (a) "the spike never adapted SAM3's
  own clone MHA" — disproved by its checkpoint; (b) "keep the last-epoch
  checkpoint because val loss anti-correlates" — that rationale rests on a
  fold whose val split is byte-identical to train.
- **Its D4 verdict still stands.** Gradient accumulation, clipping and the
  scheduler were a KEEP, and the new evidence did not overturn them. Do not
  revert them wholesale (see Task 4).
- **Do not fix `is_crowd` by copying the spike.** Both implementations train
  truncated instances as full positives and nothing in either SAM3 tree reads
  `is_crowd`. Parity here would reproduce a bug.
- **One knob per run.** Not "one task per run" — Task 4's original form bundled
  four knobs and called it one axis.
- **Keep SAHI tiling.** 4512² frames with ~97 px animals need it.
- **Never bake a measurement into a source constant.** Store it via the
  profile store (`runtime/memory_profiles.py`); `_MEASURED_BF16_DEVICE_PEAK_BYTES`
  was 3.7× the real figure and excluded every card below ~32 GiB.
- **GPU etiquette:** courtship (RTX 4090, 24 GB) is the training box; mehek is
  shared. Check `nvidia-smi` and `systemctl --user is-active sam3-*` first;
  never kill another user's process. Launch under
  `systemd-run --user --unit=... -p MemoryMax=60G -p MemorySwapMax=0 -p TasksMax=4096`.
- **Isolation:** `git worktree add .worktrees/sam3-parity -b codex/sam3-spike-parity HEAD`.
  Never `git stash`, never `git clean -fdx`, never `rm -rf`.
- Tests: `conda activate hydra-mps`, `KMP_DUPLICATE_LIB_OK=TRUE`,
  `python -m pytest tests/ -k "sam3" -q`.

---

## Task 0: A measurement that can actually resolve the effect

**Files:** Create `tools/sam3_parity/compare_models.py`,
`tools/sam3_parity/baseline.json`. (Not `specs/` — this is generated data.)

The original Task 0 could not have distinguished the effect it was built to
judge: at 5–8 extras/frame, Poisson noise on a 16-frame mean is
sd ≈ √(7/16) ≈ **0.6–0.7 extras/frame** against a 1.2–2.0 target. Inference is
deterministic, so repeats do not help — the variance is frame sampling.

- [ ] **Step 1: Paired per-frame comparison.** Report
  `extras_ours − extras_spike` **per frame**, not two independent means. Frame
  difficulty cancels. Report a sign test or bootstrap CI over the paired
  differences.
- [ ] **Step 2: State the significance criterion in the plan before running.**
  A change counts as real only if the paired CI excludes zero. Any per-task
  "movement" smaller than that is noise and must be reported as null.
- [ ] **Step 3: Define "matched recall" procedurally** — which model's
  operating point is held fixed, and how the other is interpolated onto it.
  Ambiguity here silently changes the answer.
- [ ] **Step 4: Report AP / the full PR curve alongside**, which needs no
  threshold matching and no matched-recall procedure at all.
- [ ] **Step 5: Adjudicate the delta extras once, by eye.** Take the
  detections one model emits and the other does not, and check whether they
  are clutter or **unlabelled ants**. Extras are scored against labels that
  may themselves be incomplete, so a genuinely better detector can measure
  worse. This bias only cancels between models if both emit the same extras —
  which is exactly what is under test.
- [ ] **Step 6:** Run on the **16 held-out validation frames** (576 tiles, 805
  instances, verified present in `_annotations.coco.json`), commit
  `baseline.json` with the exact calibration arguments and frame list.

**Acceptance:** a committed baseline, a stated significance criterion, and an
answer to whether the delta extras are clutter or missing labels.

---

## Task 1: Adapt `dot_prod_scoring` (top-ranked cause)

**Files:** `lora.py:50-62,66-80`, `contracts.py:233-238`, `sizing.py`,
`tests/test_sam3_lora_scopes.py`

`lora.py:50-53` excludes `dot_prod_scoring` "deliberately". The audit found it
is the spike's single biggest mover (`prompt_proj` lora_B norm 0.174) and is
the head that emits detection confidence.

**Honesty about the premise (M5):** a large lora_B norm shows the head *moved*,
not that it moved *toward precision*. This is the best available hypothesis,
not directional evidence. RUN A is a genuine test, not a confirmation.

**Interfaces (corrected — C3):** a prefix alone wraps **zero** modules, because
`inject_adapters` requires the leaf name in `TARGET_SUFFIXES` (`lora.py:412-419`)
and neither `prompt_proj` nor `hs_proj` is there. The interface is:
- `Sam3LoraParams.adapt_scoring_head: bool = False` — **defaults OFF until
  RUN A validates it** (M5).
- `SUBMODULE_PREFIXES["adapt_scoring_head"] = ("dot_prod_scoring",)`, **plus**
  an explicit per-module path allowlist for `prompt_proj` and `hs_proj`.
  Prefer a path allowlist over broadening `TARGET_SUFFIXES`: suffix additions
  have blast radius across every other scope.
- The head contains **4** Linears (`lora.py:52`) but the spike trained exactly
  **2**. Wrapping 4 silently exceeds the spike surface and breaks Task 2's
  arithmetic (316, not 314).
- `sam3_image.py:83-85` holds `instance_dot_prod_scoring = deepcopy(...)`.
  Prefix matching correctly excludes it, and that exclusion **is** intended
  parity — the spike checkpoint has no adapters for it. Assert it.

- [ ] **Step 1: Write the failing test** — exactly `prompt_proj` and `hs_proj`
  are wrapped (assert the count is 2, not "≥1"); `instance_dot_prod_scoring`
  is not; flag off wraps zero.
- [ ] **Step 2:** Run; expect failure.
- [ ] **Step 3:** Implement. Replace the "deliberately excluded" comment with
  the evidence that overturned it — correct the reasoning, do not delete it.
- [ ] **Step 4:** Measure the per-flag parameter count on a live
  `build_sam3_image_model`; update `sizing.py` from the measurement.
- [ ] **Step 5:** Assert the merge round-trip resolves these paths.
- [ ] **Step 6:** `-k "sam3"` green. Commit.
- [ ] **Step 7: RUN A.** Retrain with only this flag changed; re-run Task 0's
  paired comparison. Record the result either way — a null convicts a
  different cause and is equally valuable.

---

## Task 2: Reach the clone-MHA sites and the geometry Linears

**Files:** `lora.py`, `publish.py`, `tests/test_sam3_split_mha.py`

**Corrected arithmetic (C2).** The spike's 314 decomposes as: vision 128,
encoder 60, decoder 84, geometry **36**, segmentation_head 4, dot_prod 2.
Ours is 206 (vision 128 + enc 12 + dec 60 + geom 6). The 108-module delta is:

| component | count | covered by |
|---|---|---|
| enc/dec/seg-head clone-MHA projections | 76 | this task |
| geometry-encoder clone-MHA projections | 24 | this task |
| geometry plain Linears with unmatched leaf names | 6 | this task, **new** |
| `dot_prod_scoring` | 2 | Task 1 |

Expected intermediate counts: **206 → 308** after the clone-MHA pass, **→ 314**
after the 6 geometry Linears. The earlier draft's single "reach 314" check
would have dead-ended at 308.

The 6 are `boxes_direct_project`, `boxes_pos_enc_project`,
`points_direct_project`, `points_pool_project`, `points_pos_enc_project`,
`final_proj` (`geometry_encoders.py:543-566`). Matching is exact on the last
dotted component (`lora.py:415`), so `proj` does not match `final_proj`. Use a
**path allowlist**, not suffix broadening.

Note (N1): 11 geometry modules had `lora_B ≡ 0` in the spike — they never
forward-ran under text prompting. Their contribution is parity-cosmetic, not
functional. Do not spend effort making them "work".

**Feasibility corrections (M1):**
- The geometry encoder's attention **is** the `model_misc` clone
  (`model_builder.py:259-280`).
- The clone takes an **`attn_bias` kwarg**, passed at `decoder.py:892,925`.
  A split reimplementation must accept and apply it.
- Its Vanilla path uses **SDPA, not eager math**, so **bitwise parity is
  impossible on CUDA**. Tests must assert a tolerance, not equality — pick and
  justify the tolerance.
- In the **old** tree the clone was a torch-MHA **subclass**, which is how the
  spike reached those sites. This corrects the framing in both prior audits.
- `lora.py` is deliberately **sam3-import-free**. Identifying a sam3 class
  requires a seam (duck-typing on `in_proj_weight` + `attn_bias`, or a lazy
  import behind a function). State which, and keep module-scope clean.

- [ ] **Step 1: Write the failing tests** — replacement + wrapping; forward
  parity within tolerance across the real call-site shapes **including
  `attn_bias`**, attn masks, key-padding masks, `batch_first` both ways;
  merged `in_proj_weight` reassembles exactly; zero-init merge bit-identical.
- [ ] **Step 2-5:** Fail, implement, measure `sizing.py`, verify publish
  produces a key-identical stock state dict.
- [ ] **Step 6:** Assert **308** after the MHA pass and **314** after the
  geometry Linears, under matching flags (text encoder **OFF**, as the spike).
- [ ] **Step 7: RUN B.** Retrain, re-compare against RUN A.

**Ruling if forward parity cannot be achieved at some call site:** skip that
site, record it with the failing shapes, proceed. Never ship an attention
module that is "close enough" — a silently wrong forward poisons training in a
way no test here would catch.

---

## Task 3: fp32 training parity

**Files:** `contracts.py`, `preflight.py`, `cli.py`

**Corrected mental model (m1).** Under autocast, weights, gradients and
optimizer state are **already fp32**; only activations and some matmul
intermediates are bf16. So fp32 does not "double" the peak — the earlier
16–19 GiB extrapolation from an 8–10.4 GiB bf16 peak was the same class of
guess as the 29 GiB constant this plan criticises. It errs high, which is safe,
but it is not a basis for a constant.

- [ ] **Step 1:** Measure. Short fp32 probe on courtship; read `vram_peak`
  from the existing telemetry.
- [ ] **Step 2:** Persist the measurement through the profile store. Do **not**
  hardcode `_FP32_DEVICE_PEAK_MULTIPLIER` from it — that violates this plan's
  own constraint.
- [ ] **Step 3: RUN C.** Retrain in fp32, precision the only change.
- [ ] **Step 4:** If fp32 does not fit or does not help, keep bf16 and record
  the number. The point is to settle it with a measurement.

---

## Task 4 (DEMOTED): optimizer cadence

**Corrected (C1, M3).** The spike ran **batch_size 1**, same as ours. The real
remaining differences are accumulation (ours 8, spike 1), clipping (ours 1.0,
spike none) and the scheduler (ours warmup+cosine, spike none) — and the
2026-09-04 audit's **D4 KEEP verdict on all three still stands**; the new
evidence did not touch it.

This task is therefore **optional and last**, run only if Tasks 1–3 leave an
unexplained gap.

- [ ] **Step 1:** If run at all, change **one knob at a time** — accumulation,
  then clipping, then scheduler — each its own run and its own comparison.
- [ ] **Step 2: KEEP the non-finite gradient skip in every configuration.** It
  is not part of the spike's design, it protects against a failure that cost
  three runs, and it is inert when gradients are finite.
- [ ] **Step 3:** Do not act on `contracts.py:217`'s "batch 2 OOMs at 1008 px
  on a 47 GB card" — it predates `perflib_compat` and the streaming
  dataloader and is unverified. Measure if it matters.

---

## Task 5: Fragment handling (deliberate deviation)

**Files:** `dataset_build.py:56-57,239,462`, `tests/test_sam3_dataset_build.py`

`MIN_RETAINED_AREA_FRAC = 0.5` marks tile-clipped instances `iscrowd=1`, and
nothing in either SAM3 tree reads `is_crowd` — so 380/3286 truncated animals
(median 30% of full area, **no lower bound**) train as exhaustive full
positives.

**Quantified side effect (M2).** A non-exhaustive query has its no-object BCE
nullified and is excluded from FP penalties
(`train/loss/loss_fns.py:452-456,1167-1221`). **249 of 545 annotated train
tiles (45.7%) contain ≥1 sub-floor fragment**, and 32 contain *only*
fragments. Marking those non-exhaustive removes precision pressure from nearly
half the annotated stream — on exactly the seam-adjacent tiles where FP
discipline matters most. This fix could plausibly **increase** extras/frame,
the metric the whole programme optimises.

- [ ] **Step 1: Pick the floor and say why.** The plan previously named none.
  Current `MIN_RETAINED_AREA_FRAC = 0.5` excludes every current fragment; the
  audit floated 0.30. Prefer **0.2–0.3** so mostly-visible animals stay full
  positives and far fewer tiles are downgraded.
- [ ] **Step 2: Write the failing tests** — sub-floor fragments excluded from
  `object_ids_output` with the tile marked `is_exhaustive=False`; above-floor
  fragments retained as normal positives; counts reported in the manifest,
  including how many tiles were downgraded.
- [ ] **Step 3-4:** Fail, implement, pass.
- [ ] **Step 5:** Never drop fragments while leaving `is_exhaustive=True` —
  that teaches "the animal is absent", which is worse than the current bug.
  The exhaustiveness claim and the instance list must stay consistent.
- [ ] **Step 6:** Evaluate the audit's alternative (keep fragments as
  instances, route through `FilterCrowds` + exhaustive downgrade) before
  committing to the chosen design.
- [ ] **Step 7: RUN E, pre-registered as two-sided.** A worsening reverts the
  change. State that before running.

---

## Task 6 (NEW): Validate the combined configuration

**M4.** RUNs A–E each isolate one change; nothing validates them **together**,
and interactions (surface × precision especially) are plausible.

- [ ] **Step 1:** Train the final configuration — every change that survived
  its own run, enabled together.
- [ ] **Step 2:** Run Task 0's paired comparison against both the spike and
  the 2026-09-04 checkpoint.
- [ ] **Step 3:** Publish it and confirm it loads through
  `Sam3SemanticLabeler.from_variant` with the load guard passing.
- [ ] **Step 4:** If the combination is worse than the best single-change run,
  say so and ship the better one. Additivity is an assumption, not a result.

---

## Task 7 (NEW): Interrogate the shared recall plateau

**m3.** Both models sit at ~0.72 recall and ~5–8 extras/frame. That is a
**shared ceiling no adapter-surface change will move**, and it may dominate
everything this plan does.

- [ ] **Step 1:** Take the ~6.3 missed instances/frame at the best operating
  point and characterise them: are they small, occluded, clustered,
  seam-adjacent, or absent from the model's candidate set entirely (i.e. the
  confidence floor never saw them)?
- [ ] **Step 2:** Decide whether the ceiling is a labelling artefact, a tiling
  artefact, or a genuine model limit — and record which.
- [ ] **Step 3:** If it is a tiling or labelling artefact, that finding is
  worth more than any remaining task here. Re-prioritise accordingly.

---

## Task 8 (NEW): Correct the stale rationale in `cli.py`

**m2.** The audit's third fix recommendation was dropped from the plan.
`cli.py:20-24` justifies last-epoch checkpoint selection with a val-loss
anti-correlation argument that rests on a fold whose val split is
byte-identical to train.

- [ ] **Step 1:** Correct the comment to state what is actually known. Keep
  last-epoch selection (no evidence against it); remove the false rationale.

---

## Verification

- [ ] `python -m pytest tests/ -k "sam3" -q` green after every task.
- [ ] `make lint-moderate`.
- [ ] Every RUN appended to `tools/sam3_parity/baseline.json` with the single
  knob that changed and its paired CI.
- [ ] **Success criterion:** on held-out frames, paired extras/frame at matched
  recall ≤ the spike's with a CI excluding zero, median IoU no worse.

## Known reference points

- Spike: 314 adapted modules, rank 16, alpha 32, lr 5e-5, **batch_size 1**,
  text encoder OFF, 10 epochs / ~1080 steps, pure fp32.
- Ours (2026-09-04): 206 modules, rank 16, 3 epochs / ~3348 steps, bf16,
  batch 1 × accum 8, `val_loss_mean` 0.166, 1 skipped step.
- VRAM: 7.83 GiB steady, 10.43 GiB peak at bf16 batch 1 — identical on
  RTX 6000 Ada and RTX 4090.
- Held-out validation split: 16 frames, 576 tiles, 805 instances.
