# SAM3 LoRA: spike vs current build — consolidation decisions

**Status:** decision document (audit only, no code changed)
**Date:** 2026-09-04
**Scope:** every substantive difference between (A) the empirically successful
spike (`~/sam3_spike/sam3_lora/` on mehek; read-only copy `/tmp/sam3_lora_vet/`)
and (B) the productionised build (`src/hydra_suite/training/sam3_lora/`), judged
by effect on fine-tuning QUALITY and STABILITY of the trained model.

Given findings 1–9 from the audit brief are treated as established. Nothing
found here contradicts them; two are EXTENDED (see "Ledger" at the end).

---

## New evidence established during this audit

These were verified directly, not assumed:

- **E1 — the successful run was rank 16, not 32.** The shipped ant checkpoint
  (`~/sam3_spike/out/sam3_ant_finetuned.pt`) comes from `out/fold_all_r16/`;
  `best_lora_weights.pt` holds **628 tensors = 314 adapted modules**, and
  `backbone.vision_backbone.trunk.blocks.0.attn.qkv.lora.lora_A` has shape
  `(1024, 16)`. `configs/full_lora_config.yaml:12-13` (rank 32 / alpha 64) is a
  generic template, NOT the ant run's config. B's defaults
  (`contracts.py:211-212`, rank 16 / alpha 32) match the real run.
- **E2 — the spike trained in pure FP32.** `train_sam3_lora_native.py` contains
  no `autocast`, `bfloat16`, `half`, or `GradScaler` anywhere; the vendored old
  `model_builder.py` has no dtype casting; the old vitdet uses timm's `Mlp`
  (plain fp32 module calls). The config's `mixed_precision: "bf16"` is dead
  config, exactly like its `warmup_steps`/`max_grad_norm`/`grad_accum`. B
  trains everything under `torch.autocast(bfloat16)` (`cli.py:556,573`) and its
  admission gate *mandates* bf16 (`cli.py`, `_runtime_admission_refusal`).
- **E3 — internal training-mode matching is equivalent to the spike's explicit
  attachment.** New SAM3 `sam3_image.py:495-496` calls `_compute_matching`
  whenever `self.training` (one grounding step per stage in training, since
  `num_interactive_steps = 0` at `:578`), and `_compute_matching`
  (`:607-610`) attaches `indices` to the main output **and every
  `aux_outputs` entry** using `self.matcher`. Old tree is line-for-line the
  same (`:579-581`). The spike's `ALL_STEPS_PER_STAGE` loop
  (`train_sam3_lora_native.py:1250-1263`) does the identical thing externally.
- **E4 — B's perflib patch is faithful to what the spike actually trained
  through.** Old vitdet has no local `Mlp`; it imports **timm's `Mlp`**
  (`old vitdet.py:23,26`), whose forward is `fc2(drop(norm(drop(act(fc1(x))))))`
  via plain module calls. New vitdet's `Mlp.forward` (`new vitdet.py:70-76`)
  replaces only the `fc1 → act` step with `addmm_act(type(self.act), self.fc1, x)`;
  the fused kernel (`perflib/fused.py`) raises under grad, detaches weights,
  and hard-casts to bf16. B's `eager_addmm_act` (`perflib_compat.py`) computes
  `act(self.fc1(x))` by **calling the module** — i.e. exactly timm's semantics,
  with LoRA live and gradients attached, deliberately without the hard bf16
  casts (autocast owns precision).
- **E5 — torch `nn.MultiheadAttention` still exists in the NEW SAM3 tree** at
  the same sites the spike adapted: `decoder.py:54,59` (`ca_text`,
  `self_attn`) in both trees, plus the language backbone. SAM3's own clone
  (`model_misc.py:470+`) is an `nn.Module` (not a subclass of torch's MHA)
  whose forward routes through a functional
  `multi_head_attention_forward(... in_proj_weight, out_proj.weight ...)`
  (`model_misc.py:90-149,449`) — the reason B's skip rule exists.
- **E6 — the two SAM3 trees differ broadly** (`diff -rq`: nearly every model
  file differs), but the two files that decide equivalence for this programme —
  `sam3_image.py` matching/`back_convert` and the matcher — are identical
  modulo line offsets; the load-bearing behavioural change is the perflib
  fused MLP path, which B patches (E4).

---

## Difference-by-difference decisions

Format: what A does / what B does → direction → confidence → risk.

### D1. Adaptation surface: fused-attention (`in_proj`) modules

- **A:** replaces every `nn.MultiheadAttention` in scope with a split
  q/k/v/out eager reimplementation (`lora_layers.py:14-176`,
  `apply_lora_to_model` STEP 1 at `:420-455`), then LoRA-wraps the resulting
  `q_proj/k_proj/v_proj/out_proj`. Merge reassembles stock `in_proj_weight`
  (`merge_lora_weights.py:82-131`, `restore_multihead_attention`) and asserts
  the merged state dict is key-identical to stock SAM3 (`:203-214`). Net: 314
  adapted modules (E1); the merged ant checkpoint changed 36 `in_proj_weight`
  and 37 `out_proj.weight` tensors at 5–15% relative delta (given finding 2).
- **B:** wraps only free-standing `nn.Linear`s (158 modules) and explicitly
  skips any Linear whose parent has `in_proj_weight`
  (`lora.py:130-146,154`), because both torch's MHA and SAM3's clone read
  `.out_proj.weight` functionally, making a wrapper either crash (loud) or be
  dead weight.
- **Verdict: ADOPT A's direction (module replacement), as a deliberate
  slice.** The empirically successful checkpoint did a large share of its
  work inside exactly the modules B cannot reach — the DETR decoder's
  self/text-cross attention is the seam where a new text concept is grounded,
  and finding 2 shows those weights moved substantially. B's current surface
  has never produced a validated checkpoint at parity with the spike's; the
  spike's has. Port the *mechanism* into B's idiom: replace-in-scope torch
  `nn.MultiheadAttention` (E5 confirms they exist unchanged in the new tree)
  with a split-projection module, inject `LoraLinear` on the four
  projections, and keep B's guards.
- **Merge/publish round-trip:** B's `merge_adapters` resolves
  `{path}.lora_A/B → detector.{path}.weight` (`lora.py:174-260`); split
  `q_proj/k_proj/v_proj` paths resolve to keys that do not exist in the stock
  checkpoint, so a naive adoption **fails loud** (`KeyError`, `lora.py` hard
  error) — in A-into-B's favour, no silent corruption is possible. Publish
  needs one special case: fold q/k/v deltas into `detector.{mha}.in_proj_weight`
  row slices `[0:E]`, `[E:2E]`, `[2E:3E]` and out_proj through the existing
  formula; biases untouched (LoRA touches weights only). The spike's
  `restore_multihead_attention` proves the reassembly is exact. Also update
  `sizing.expected_lora_trainable_params` and `_validated_lora_trainables`
  (`cli.py:476-486`) counts, or the invariant refuses the run.
- **Numerical-fidelity risk:** the replacement forward must match
  `F.multi_head_attention_forward` for the argument shapes SAM3 actually uses.
  The spike's eager module ran an entire successful training and its merged
  checkpoint validated back through stock SAM3 inference, which bounds this
  risk empirically, but the port must be tested per call-site (decoder
  `self_attn` uses attn masks; `ca_text` uses key_padding_mask). Do NOT touch
  SAM3's own clone (`model_misc.MultiheadAttention`) in a first slice — the
  spike never adapted it either (it is not an `nn.MultiheadAttention`
  instance), so it is outside the empirically validated surface.
- **Confidence:** medium-high that this materially helps (spike evidence +
  location of the deltas); would rise to high with the named experiment
  (D-EXP-1): train B twice on the same dataset, with and without MHA
  replacement, compare held-out AP75.

### D2. `.weight`/`.bias` proxy properties on the LoRA wrapper

- **A:** `LoRALinear` exposes `weight`/`bias` properties proxying to the
  frozen base (`lora_layers.py:256-265`) so direct-weight readers keep
  working.
- **B:** no proxy; direct-weight readers would raise `AttributeError`, hence
  the skip rule (`lora.py:130-146`).
- **Verdict: KEEP B — do not adopt the proxy.** The proxy is doubly wrong for
  quality: a parent that reads `.weight` never *calls* the module, so the
  adapter's forward never runs — its gradient is identically zero and it is a
  mathematically dead parameter that still consumes optimizer state and
  inflates the "trainable params" count, while *appearing* to adapt the layer.
  A loud `AttributeError` on first forward is strictly better than a silent
  no-op adapter: the failure mode B chose is the recoverable one. In the
  spike the proxy was harmless only because STEP 1 had already replaced every
  reachable MHA, leaving (almost) no direct-weight readers; adopt D1's
  replacement and the proxy has no remaining purpose. Note one residual
  reader even under D1: new vitdet's fused `addmm_act` reads
  `linear.weight/.bias` — B already solved that correctly by calling the
  module in `eager_addmm_act` (E4).
- **Confidence:** high. **Risk of adopting A:** silent dead adapters and a
  publish path that merges zero-deltas indistinguishable from trained ones
  (until `_validate_adapter_state`'s no-op check fires only if *all* are
  zero).

### D3. `perflib_compat` (vision-trunk MLP differentiability)

- **A:** never needed it — old vitdet uses timm's `Mlp` (E4).
- **B:** rebinds `vitdet.addmm_act` to `eager_addmm_act`
  (`perflib_compat.py`), installed before model build (`cli.py:465`).
- **Verdict: KEEP B; the patch is faithful.** `act(fc1(x))` via module call is
  exactly what the spike trained through (timm), keeps LoRA on `fc1` live, and
  correctly declines to reproduce the fused kernel's hard bf16 casts (autocast
  handles precision at the right level; a hard cast would silently downcast an
  fp32 run — relevant given D5). One asymmetry worth knowing, not fixing:
  *inference* through the published checkpoint on stock SAM3 uses the fused
  bf16 kernel, so train/serve MLP numerics differ at bf16 rounding level —
  the same asymmetry the spike had (fp32 train, fused-bf16 serve on the new
  tree) and empirically tolerable.
- **Confidence:** high (primary-source comparison done). **Risk:** none beyond
  status quo.

### D4. Optimizer dynamics (effective batch, clipping, scheduler)

- **A:** batch_size 4 (datapoints), `zero_grad → backward → step` every batch
  (`train_sam3_lora_native.py:1272-1275`), constant lr 5e-5
  (`:1030-1034`), **no** accumulation, **no** clipping, **no** scheduler —
  the config's `gradient_accumulation_steps: 8`, `max_grad_norm: 1.0`,
  `warmup_steps`, `lr_scheduler: cosine` are all dead (given finding 5).
- **B:** batch 1 query × grad_accum 8, `clip_grad_norm_(1.0)`, cosine with
  `warmup = min(50, total//4)` (`cli.py:544-556,586-589`).
- **Verdict: KEEP B's mechanics.** Effective tokens/step are comparable
  (4 datapoints ≈ 8 query-rows), warmup+cosine+clipping are strictly
  stabilising, and the spike's success without them shows they are not
  *necessary*, not that they are harmful. They are not credible causes of the
  epoch-5 NaN: clipping rescales finite gradients (it does NOT sanitise
  non-finite ones — `clip_grad_norm_` happily propagates inf/NaN norms), and
  the scheduler only lowers lr over time.
- **Epoch-5 NaN attribution (finding 8):** the Hungarian
  `ValueError` is the *symptom site*, not the cause — B's own
  `_assert_finite_loss` docstring records the matcher error firing ~700 steps
  after the loss first went bad. Ranked candidates, given this audit:
  1. **bf16 autocast (E2)** — the single largest regime change from the
     proven run; focal CE / presence terms on large logits and the matcher's
     cost matrix are computed under autocast (`cli.py:573`; internal matching
     runs inside `model(input)`, hence inside autocast — the spike matched in
     fp32 outside the model).
  2. **Negative-heavy stream** — the NaN'd run deviated from defaults
     (1 negative, empty tiles dropped → 50% negatives; finding 7), starving
     accumulation windows of matched positives.
  3. Longer run simply reaching the divergence that a 2-epoch run never met.
  The now-present `_assert_finite_loss` per micro-batch will localise the
  first bad step on the next run.
- **Confidence:** high on keeping B's mechanics; medium on NaN attribution
  pending D-EXP-2 (below).

### D5. Training precision: FP32 vs mandated BF16  *(new difference, extends finding 5)*

- **A:** pure FP32 (E2). **B:** bf16 autocast, with an admission gate that
  *refuses* anything else (`cli.py`, `_runtime_admission_refusal`: "SAM3
  training supports only CUDA BF16 ... fp16/fp32 modes ... disabled").
- **Verdict: NEEDS EXPERIMENT (D-EXP-2), and the gate's rationale is
  partially stale.** The gate's justification ("fails against SAM3's BF16
  activation path") was true of the unpatched fused kernel; with
  `perflib_compat` installed the only hard bf16 dependency is gone, so an
  fp32 (or fp32-with-bf16-matmul) run is now *feasible* — and it is the
  regime the only successful checkpoint used. bf16 halves memory and roughly
  doubles throughput, so it should not be abandoned on suspicion alone.
  **D-EXP-2:** rerun the 10-epoch config with defaults restored
  (num_negatives=3, keep_empty_tiles=True) under (a) bf16 and (b) fp32; the
  finiteness assert localises any divergence. If (a) NaNs where (b) does not,
  precision is convicted; then prefer selective fp32 (matcher + loss outside
  autocast) before abandoning bf16 wholesale.
- **Confidence:** high that the difference is real; low-medium on which
  precision B should end up with. **Risk of adopting A (fp32):** ~2× memory
  and time; the preflight/sizing budgets and OOM-hardening assumptions are
  calibrated for bf16.

### D6. Matcher index attachment (train path)

- **A:** explicit `outputs["indices"] = matcher(...)` for main + aux under
  `SAM3Output.iteration_mode(ALL_STEPS_PER_STAGE)`
  (`train_sam3_lora_native.py:1250-1263`), model left in whatever mode.
- **B:** relies on the model's internal training-mode matching, sharing ONE
  matcher instance by assigning `model.matcher = matcher` (`cli.py:506-517`);
  explicit attachment only in eval (`cli.py:152-160,656`).
- **Verdict: KEEP B — verified equivalent (E3).** Internal
  `_compute_matching` covers main output and all `aux_outputs` per stage with
  the same matcher and the same `back_convert` targets; with
  `num_interactive_steps=0` in training there is exactly one step per stage,
  so `ALL_STEPS_PER_STAGE` adds nothing. The `model.matcher` mutation is not a
  hazard: `build_sam3_image_model(eval_mode=False)`'s model *requires* a
  matcher for its training forward, and sharing the instance guarantees loss
  and model can never diverge in matcher hyperparameters (the spike achieved
  the same by construction). One real remaining asymmetry: B's internal
  matching runs under autocast (see D4/D5); if D-EXP-2 convicts precision, an
  fp32-matcher wrapper is the surgical fix.
- **Confidence:** high (source verified on both trees).

### D7. Negative sampling and empty-tile retention

- **A:** per image, ALL absent dataset categories (tier 1) + 3 sampled
  generic out-of-domain negatives (tier 2) (`dataset_formats.py:213-249`,
  `full_lora_config.yaml:74`), and background images retained.
- **B:** defaults `num_negatives=3`, `keep_empty_tiles=True`
  (`contracts.py:221,245`; `dataset_build.py:241`), negatives sampled per
  tile from the manifest-resolved pool (`dataloader.py`,
  `_negative_prompts_for`, `build_descriptors`). **The NaN'd run deviated
  from these defaults** (1 negative, empty tiles dropped → 50% negatives).
- **Verdict: KEEP B's defaults — they already ARE the spike's direction —
  and stop deviating.** For a single-concept prompt there is no in-domain
  tier, so 3 generic negatives per tile matches the spike's tier-2 exactly;
  retained empty tiles are the analogue of the spike's background images and
  are the *positive-prompt* hard negatives that buy precision (a prompt that
  must return nothing on background). Dropping them removes precision
  supervision; cutting negatives to 1 while also dropping empties skews the
  stream to 50% pure-negative rows, which both destabilises windows (D4) and
  weakens recall-side gradient signal per step.
- **Confidence:** high that defaults > the deviated config; medium on the
  exact ratio being optimal (D-EXP-3: sweep num_negatives ∈ {1,3,5} with
  empties kept, score precision/recall on held-out frames).

### D8. LoRA scope, rank/alpha, dropout, lr

- **A (as actually run, E1):** rank 16, alpha 32 (628 tensors, `_r16`
  folds), dropout 0.1, lr 5e-5, text encoder frozen, mask decoder +
  geometry encoder adapted (`full_lora_config.yaml:57-63` scope flags; the
  yaml's 32/64 rank numbers are a template, not the run).
- **B:** identical — rank 16 / alpha 32 / dropout 0.1 / lr 5e-5, same six
  scope flags with the same values (`contracts.py:211-238`).
- **Verdict: KEEP B (no difference in substance).** The apparent rank
  discrepancy dissolves under E1: B's "defaults measured on the spike" claim
  is *correct*; the template yaml is a red herring. No experiment needed.
- **Confidence:** high (adapter tensors inspected).

### D9. Dataset geometry: whole images vs SAHI tiles

- **A:** whole COCO images resized to model RES; queries per image
  (`dataset_formats.py`).
- **B:** SAHI-style tiles sized from measured object scale
  (`dataset_build.py`, `geometry_mode="auto_object"`), resized to
  `PREDICTOR_IMGSZ` with polygons rescaled (`datapoints.py:31-50`).
- **Verdict: KEEP B.** Training geometry matching serving-time tiling is the
  entire point of the sliced-training programme (objects appear at the scale
  inference will see); the spike's whole-image resize is the less faithful
  regime for this deployment. Not implicated in stability.
- **Confidence:** medium-high; D-EXP-1's runs double as evidence here since
  they train B's geometry end-to-end to a validated checkpoint.

### D10. Which SAM3 tree to target

- **A:** succeeded on the older vendored tree (no `perflib/fused.py`).
- **B:** targets the newer tree at `~/sam3_spike/sam3/sam3/`.
- **Verdict: KEEP B on the newer tree.** The two files that govern training
  semantics for this loop — matching/`back_convert` in `sam3_image.py` and the
  matcher — are identical across trees (E3); the one behavioural landmine
  (fused inference-only MLP) is patched faithfully (E4/D3); torch
  `nn.MultiheadAttention` sites needed for D1 are unchanged (E5). Pinning to a
  frozen old vendored snapshot would orphan B from upstream fixes for no
  measured benefit. Evidence raiser: a reviewed `diff -r` of
  `sam3/model/` + `sam3/train/` between trees (E6 confirmed "broadly differs"
  but only spot-checked the load-bearing files).
- **Confidence:** medium-high.

### D11. Checkpoint selection, validation, artifact discipline

- **A:** saved best-by-val-loss and last (`out/fold_*/best_lora_weights.pt`);
  fold data shows val loss still falling at epoch 10.
- **B:** always last-epoch, val loss informational only, atomic validated
  artifact write with no-op detection (`cli.py` module docstring,
  `_write_validated_adapter_artifact`, `_validate_adapter_state`).
- **Verdict: KEEP B.** The last-epoch rule encodes the spike's own measured
  anti-correlation between val loss and held-out AP; the artifact validation
  (finite, complete pairs, non-zero delta) has no spike counterpart and
  protects the published-checkpoint contract.
- **Confidence:** high.

---

## Ranked recommendations

### 1) Adopt from spike into current build
1. **D1 — MultiheadAttention replacement (adapt `in_proj` q/k/v/out).** The
   validated checkpoint did much of its work there; B cannot reach it.
   Port as replacement modules in B's idiom + one publish special case
   (in_proj row-slice reassembly) + sizing/invariant updates. B's hard-error
   merge means a botched adoption fails loud.
2. **D7 (operationally) — run with the spike-equivalent negative regime**,
   which is B's own defaults: `num_negatives=3`, `keep_empty_tiles=True`.
   The deviation, not the design, produced the 50%-negative stream.

### 2) Keep current build's behaviour
3. **D2 — loud skip over `.weight` proxy** (silent dead adapters are the worse
   failure; proxy becomes moot under D1).
4. **D3 — `perflib_compat` eager patch** (verified faithful to the timm `Mlp`
   the spike trained through).
5. **D6 — internal training-mode matching + shared `model.matcher`**
   (verified equivalent to the spike's explicit attachment, aux included).
6. **D4 — grad accumulation, clipping, warmup+cosine** (strictly stabilising;
   not the NaN's cause).
7. **D8 — rank 16 / alpha 32 / dropout 0.1 / lr 5e-5 / scope flags** (they ARE
   the successful run's values; the yaml's 32/64 was a template).
8. **D9 — SAHI tile geometry** (matches serving; the spike's whole-image
   resize is less faithful for this deployment).
9. **D10 — newer SAM3 tree** (load-bearing training semantics verified
   identical; fused-MLP landmine already patched).
10. **D11 — last-epoch checkpoint + validated atomic artifact.**

### 3) Needs an experiment
11. **D5 — precision (fp32 vs bf16), the prime epoch-5-NaN suspect.**
    **D-EXP-2:** 10 epochs, defaults restored, (a) bf16 vs (b) fp32 (feasible
    now that `perflib_compat` removed the hard bf16 dependency — the
    admission gate's "no safe FP32" rationale is stale and should be revisited
    after the experiment). If bf16 alone NaNs, prefer surgical fp32 for
    matcher+loss before abandoning autocast.
12. **D1 confirmation — D-EXP-1:** same dataset, B with vs without MHA
    replacement, held-out AP75; also serves as the end-to-end validation of
    B's tile geometry (D9).
13. **D7 ratio — D-EXP-3:** `num_negatives ∈ {1,3,5}` with empties kept;
    score precision/recall of the tuned concept on held-out labelled frames.

---

## Ledger vs the nine given findings

- **No contradictions found.**
- **Finding 5 extended:** `mixed_precision: bf16` is *also* dead config — the
  spike trained in pure FP32 (E2). This turns precision into a first-class
  difference (D5) and the leading NaN suspect.
- **Finding 7 clarified:** the 1-negative / empties-dropped regime was a
  per-run deviation; B's *defaults* already match the spike's direction (D7).
- **Supplementary to finding 2/3:** the successful adapter file has 628
  tensors (314 modules) at rank 16 — so the surface gap vs B's 158 modules is
  even larger than the merged-tensor count suggested, and the template yaml's
  rank 32/alpha 64 was not what trained the ant checkpoint (E1, D8).
  The counts also reconcile rather than conflict: each replaced fused MHA
  collapses its three q/k/v adapter modules into ONE merged `in_proj_weight`
  tensor, so touched base tensors ≈ 314 − 2×(replaced MHAs) ≈ 235–242 for
  ~36–40 replacements — matching finding 2's 235 exactly, which corroborates
  that `fold_all_r16` and the merged ant checkpoint describe the same run.
