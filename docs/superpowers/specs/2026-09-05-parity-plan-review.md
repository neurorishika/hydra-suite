# Adversarial review: SAM3 spike-parity finetuning plan

**Status:** review findings (no code or plan changes made)
**Date:** 2026-09-05
**Target:** `docs/superpowers/plans/2026-09-05-sam3-spike-parity-finetuning.md`
**Method:** every challenged claim was checked against primary sources — our
`src/hydra_suite/training/sam3_lora/`, the spike tree and run artifacts on
mehek (`~/sam3_spike/`, read-only), the new SAM3 tree
(`~/sam3_spike/sam3/sam3/`), the spike checkpoint
(`~/sam3_spike/out/fold_all_r16/best_lora_weights.pt`, key-listed on CPU),
and the built datasets on courtship. The local `/tmp/sam3_lora_vet/` copy is
**hollow** (directory skeleton, 0 bytes of source) — anything "verified"
against it was verified against nothing; all spike reads below are from mehek.

Overall verdict is at the end.

---

## Critical

### C1. Task 4's parity target ("spike: batch 4") is contradicted by the spike's own run config

- **Claim challenged:** "Spike: batch 4, `optimizer.step()` every batch" and
  the Known-reference "pure fp32, batch 4".
- **Evidence:** mehek `~/sam3_spike/work/configs/fold_all_r16.yaml:15` —
  `batch_size: 1`. `train_sam3_lora_native.py:1150-1155` builds its
  `DataLoader` directly from that value, and one dataset element is **one
  image carrying all of its queries** (`__getitem__` at `:315` loads one
  image; queries — positive, paraphrase, and hard negatives — are attached to
  that image's datapoint at `:435-463`, collated by `collate_fn_api` at
  `:1128-1129`). The plan's own "~1080 optimizer steps" is only consistent
  with batch 1: 108 train images × 10 epochs = 1080 steps; batch 4 would give
  270. The plan is internally inconsistent *and* wrong against the yaml.
- **Why it matters:** Task 4's entire purpose is "match the spike's effective
  batch and cadence". As written it would set batch 4 — a *third* regime that
  is neither ours nor the spike's — and Step 1 (probe whether batch 4 fits at
  1008 px) answers a question nothing depends on. The "batch 4 (datapoints)"
  figure comes from the 2026-09-04 audit's D4, which read a config whose
  accumulation/scheduler keys were *known to be dead*; the 2026-09-05 audit's
  N4 already corrected it to batch 1 — the plan sides with the overturned
  number while claiming to be built on the newer audit.
- **Correction:** Task 4's target is: 1 image-datapoint per step (which on
  our tiled data means 1 tile-datapoint = 1 positive + 3 negative query rows),
  step every batch, constant lr 5e-5, no accumulation/clipping/scheduler.
  Define "effective batch" in query-rows and state the mapping between the
  spike's image-queries and our tile-queries explicitly. Delete or repurpose
  the batch-4 probe. Also fix the Known-reference block.

### C2. Task 2's acceptance ("reach 314") is unachievable under the task's stated scope

- **Claim challenged:** "Confirm the adapted-module count reaches the spike's
  **314** under matching flags" via clone-MHA replacement.
- **Evidence:** the spike checkpoint's 314 modules decompose (key-listed on
  mehek) as: `backbone.vision_backbone` 128, `transformer.encoder` 60,
  `transformer.decoder` 84, `geometry_encoder` **36**, `segmentation_head` 4,
  `dot_prod_scoring` 2. Ours is 206 (= vision 128 + enc 12 + dec 60 + geom 6).
  The delta of 108 is: enc/dec/seg-head clone-MHA projections **76**,
  dot_prod **2** (Task 1), geometry-encoder **30**. Of geometry's 30: 24 are
  clone-MHA projections (`geometry_encoder.encode.{0-2}.{self_attn,
  cross_attn_image}.{q,k,v,out}_proj` — the geometry encoder's attention IS
  the `model_misc` clone, built at new-tree `model_builder.py:259-280`), and
  **6 are plain Linears whose leaf names match NO current target suffix**:
  `boxes_direct_project`, `boxes_pos_enc_project`, `points_direct_project`,
  `points_pool_project`, `points_pos_enc_project`, `final_proj`
  (`geometry_encoders.py:543-566`; `TARGET_SUFFIXES` at `lora.py:66-80`
  contains `proj` but not `final_proj` etc., and matching is exact on the
  last dotted component, `lora.py:415`).
- **Why it matters:** clone-MHA replacement alone lands at 206+2+76+24 = 308.
  The remaining 6 require **adding new target suffixes**, which the plan
  nowhere mentions — and suffix additions have blast radius across every
  other scope (any module elsewhere whose leaf is `final_proj` would also get
  wrapped). Without this, Step 6 fails and the task dead-ends at "investigate
  before proceeding" on a gap that was knowable in advance.
- **Correction:** add the 6 geometry Linears to the task explicitly (either
  via per-module path allowlist — safer than suffix broadening — or suffixes
  with a measured blast-radius check), state the expected intermediate counts
  (308 after MHA pass, 314 after), and note that per N1 11 of the geometry
  modules had lora_B ≡ 0 in the spike (never forward-ran under text
  prompting), so their contribution is parity-cosmetic, not functional.

### C3. Task 1's stated interface is a silent no-op under the current injection rules

- **Claim challenged:** "Produces: `SUBMODULE_PREFIXES["adapt_scoring_head"]
  = ("dot_prod_scoring",)`" (plus the flag) as the interface for reaching
  `prompt_proj`/`hs_proj`.
- **Evidence:** `inject_adapters` wraps a Linear only if
  `name.split(".")[-1] in cfg.target_suffixes` (`lora.py:412-419`).
  `TARGET_SUFFIXES` (`lora.py:66-80`) contains `proj`, `q_proj`, … but **not**
  `prompt_proj` or `hs_proj`. A prefix alone therefore wraps **zero** modules
  — exactly the "flag silently turning into a no-op" failure mode
  `lora.py:31-36`'s own comment warns about.
- **Why it matters:** the plan's Step 1 failing test would catch it, but the
  interface spec is what a task-executing agent implements first; as written
  it guarantees a detour. Worse, the *fix* is underdetermined: adding `hs_proj`
  / `prompt_proj` as suffixes vs. an explicit path list decides whether the
  flag wraps exactly the spike's 2 modules or more. `lora.py:52` says
  `dot_prod_scoring` contains **4** Linears (the checkpoint shows the spike
  trained exactly **2**: `prompt_proj`, `hs_proj`); a careless suffix choice
  adapts 4 and silently exceeds the spike surface, breaking both parity and
  C2's 314 arithmetic (316). Note also `sam3_image.py:83-85` holds an
  `instance_dot_prod_scoring = deepcopy(dot_prod_scoring)`; prefix matching
  on `"dot_prod_scoring"` correctly excludes it (startswith fails), but the
  plan should say whether excluding it is intended parity (it is: the
  checkpoint has no `instance_dot_prod_scoring` adapters).
- **Correction:** specify the full interface: prefix + the suffix (or path)
  additions, an assertion that exactly `prompt_proj` and `hs_proj` are
  wrapped (2 modules, not 4), and a statement about
  `instance_dot_prod_scoring`.

### C4. Task 0 cannot statistically resolve the effect every later task is judged by

- **Claim challenged:** the implicit assumption that "extras/frame at matched
  recall" on 16 frames distinguishes the 1.2–2.0/frame delta, and Task 1's
  acceptance "extras/frame move toward the spike's".
- **Evidence:** courtship valid split = **16 source frames** / 576 tiles /
  805 instances (verified in `_annotations.coco.json`) — the frames exist, so
  Step 1 is feasible. But at a base rate of 5–8 extras/frame, Poisson-scale
  noise on a 16-frame mean is sd ≈ √(7/16) ≈ **0.6–0.7 extras/frame**; the
  target delta (1.2–2.0) is a 2–3σ effect and any per-task "movement" of
  <1/frame is inside noise. Inference is deterministic, so repeats do not
  help — the variance is frame sampling.
- **Why it matters:** five sequential runs each judged by a sub-σ criterion
  is a recipe for misattribution — the exact disease the plan says it exists
  to cure. Additionally, "extras" are counted against labels that may
  themselves miss ants; an unlabeled true ant scores as a false positive, so
  a genuinely *better* detector can measure *worse*. That bias only cancels
  between models if both emit the same extras — which is precisely what is
  under test.
- **Correction:** (a) make the comparison **paired per-frame** (extras_ours −
  extras_spike on each frame; frame variance cancels; report a sign test or
  bootstrap CI) and state a significance criterion in the plan; (b) define
  the "matched recall" procedure (which model's operating point is matched,
  interpolated how); (c) add a one-time human adjudication of the *delta*
  extras (the detections one model emits and the other doesn't) to establish
  whether they are clutter or unlabeled ants; (d) consider reporting AP /
  the full PR curve alongside, which needs no threshold matching.

---

## Major

### M1. Task 2's parity-test list omits the arguments and semantics that actually differ

- **Claim challenged:** "forward parity against the unmodified clone across
  the argument shapes SAM3 uses (attn masks, key padding masks, batch_first
  both ways)".
- **Evidence:** the new tree's clone (`model_misc.py:470-733`) differs from
  torch-MHA semantics in ways the plan's list misses:
  - it accepts an **`attn_bias` kwarg** (`:589`), and real call sites pass it
    (`decoder.py:892,925`; encoder sites have it commented out). Our
    `SplitMultiheadAttention` has no such parameter — a drop-in replacement
    raises `TypeError` at the first decoder cross-attn call (loud, but the
    plan should say the replacement must accept it and define behavior for
    non-None).
  - its Vanilla path runs **`F.scaled_dot_product_attention`** with all three
    SDP backends enabled (`model_misc.py:396-415`), not torch's eager math —
    our SplitMHA docstring promises "torch's *eager*
    `F.multi_head_attention_forward` math" (`lora.py:138-147`). Bitwise
    forward parity on CUDA is therefore not achievable; "parity" needs a
    stated dtype/device/tolerance and must be tested in eval mode (train-mode
    dropout RNG streams differ regardless).
  - it returns `(out, None)` when `need_weights=False`; call sites index
    `[0]` — arity must be preserved.
  - the builder constructs it with `use_fa3` threaded through
    (`model_builder.py:136-231`); the replacement must refuse or reproduce
    `use_fa3=True` / non-Vanilla `attn_type` / `use_act_checkpoint=True`
    rather than silently ignore them.
- **Also worth stating (corrects both audits' framing):** the reason the
  spike reached these sites at all is that the OLD vendored tree's
  "clone" is a thin **subclass of torch's `nn.MultiheadAttention`**
  (`~/sam3_spike/sam3_lora/sam3/model/model_misc.py:31-34`,
  `MultiheadAttentionWrapper(nn.MultiheadAttention)` forcing
  `need_weights=False`), so the spike's isinstance replacement caught them.
  The new tree renamed a genuine nn.Module clone to the same alias
  (`model_misc.py:731`). Our injection uses `type(mod) is
  nn.MultiheadAttention` (`lora.py:373`) — exact type — so even a torch-MHA
  subclass would be skipped today; Task 2's implementation must key on the
  sam3 clone class specifically.
- **One structural conflict the plan ignores:** `lora.py`'s module docstring
  declares it "deliberately free of any SAM3 import" (line 3). Identifying
  `model_misc.MultiheadAttention` instances requires knowing that class.
  Solvable (injected predicate, qualified-name string match), but the plan's
  file list and interface say nothing about how the seam stays sam3-free.

### M2. Task 5's `is_exhaustive=False` fix would strip false-positive suppression from ~46% of annotated train tiles

- **Claim challenged:** "a fragment below the retained-area floor is excluded
  … and its tile is marked `is_exhaustive=False`" as an improvement.
- **Evidence:** per N5 (verified semantics), a non-exhaustive query has its
  negative/no-object BCE nullified and is excluded from FP penalties
  (`train/loss/loss_fns.py:452-456`, `:1167-1221`). Measured on courtship's
  train split (`dataset-preparation-3365d6c8…`): **249 of 545 annotated
  tiles (45.7%) contain ≥1 sub-floor fragment** (and 32 tiles contain *only*
  fragments). Marking those tiles' positive query non-exhaustive removes all
  precision pressure on the retained full ants' clutter in nearly half of the
  annotated stream — the model is never told "and nothing else here is an
  ant" exactly on the seam-adjacent tiles where FP discipline matters most.
- **Why it matters:** the fix is directionally defensible (it stops teaching
  "fragment = full ant" without teaching "fragment = background") but its
  side effect is large and unquantified in the plan, and it plausibly
  *increases* extras/frame — the metric the whole programme optimizes. The
  negative-query rows (3 per tile) stay exhaustive-empty and keep some
  pressure, but they supervise different prompts, not the positive concept.
- **Correction:** state the trade-off and the 249/545 number in the task;
  specify the floor value (the plan never picks one — current
  `MIN_RETAINED_AREA_FRAC = 0.5` at `dataset_build.py:57`, the audit floated
  0.30; at 0.5 every current `iscrowd` fragment is excluded); consider the
  audit's alternative (keep fragments as instances but route them through
  `FilterCrowds` + exhaustive downgrade) and a floor low enough (e.g. 0.2–0.3)
  that mostly-visible animals remain full positives and far fewer tiles are
  downgraded. Treat RUN E as genuinely two-sided: pre-register that a
  worsening reverts the change.

### M3. The plan discards the earlier audit's still-standing D4 verdict without new evidence, and misses that its Task 4 bundles four knobs into "one axis"

- **Claim challenged:** the Global Constraint that exactly two 2026-09-04
  conclusions are overturned, combined with Task 4 reverting accumulation,
  clipping, warmup and cosine wholesale.
- **Evidence:** the 2026-09-04 audit's D4 verdict was "KEEP B's mechanics"
  (accumulation/clipping/scheduler "strictly stabilising"). The 2026-09-05
  audit did **not** overturn it — its cause 3 is "LOW-MEDIUM confidence …
  ranked here because it is the largest remaining regime difference", i.e.
  needs-experiment, same as before. Running the experiment is fine; the plan
  instead frames the revert as parity-by-default ("no justification found"),
  which contradicts its own source document. Meanwhile Task 4 changes batch
  semantics, accumulation, clipping and the scheduler in a single RUN D —
  violating the plan's own "change one axis per run" constraint (an axis is a
  knob, and D contains four). If RUN D moves the metric, nothing says which
  knob did it; if it NaNs (clipping removed under bf16 — the earlier audit's
  prime NaN suspect regime, mitigated only by the kept non-finite skip),
  nothing says why.
- **Correction:** either declare "optimizer cadence" one composite axis
  explicitly and accept coarse attribution (defensible, cheaper), or split D
  into cadence (step/lr schedule) and clipping. And correct the Global
  Constraints text: D4's keep-verdict stands unrefuted; Task 4 is an
  experiment against it, not a correction of it.

### M4. The plan is ambiguous about whether RUNs A–E stack, and the final configuration is never validated as a whole

- **Claim challenged:** "Each task below is a separately measurable run" vs
  Task 2's "RUN B. Retrain, re-sweep, **compare against Task 1**" and Task 3's
  "changing only precision".
- **Evidence:** if runs are isolated (each = 2026-09-04 config + one change),
  the plan's end state — the union of every accepted change — is a
  configuration that never trained once; interactions (surface × precision:
  fp32 convergence of a *larger* adapter set; cadence × surface: constant-lr
  stability with the scoring head trainable and its ±12 logit clamp,
  `model_misc.py:748-751`; data × everything for RUN E's rebuilt dataset)
  go unmeasured. If runs stack, "changing only X" is relative to a moving
  base and the one-at-a-time ordering can walk into a local optimum, with
  early axes credited for effects later axes would have produced.
- **Correction:** state the design (stacked is more coherent given "compare
  against Task 1"), and add a mandatory final run: the full accepted
  configuration, swept on the Task 0 frames, as the checkpoint that actually
  ships. Cheap: it is RUN E if stacking is declared.

### M5. Task 1's premise is the best available hypothesis, but the plan overstates it as directional and defaults the flag on before RUN A validates it

- **Claim challenged:** "a frozen scoring pathway over adapted features is
  the leading explanation for excess false positives" and "the flag defaults
  to on".
- **Evidence:** the audit's N1 establishes the spike adapted
  `dot_prod_scoring` and that it moved most (lora_B 0.174) — that is
  evidence the spike *changed* its scoring head, not evidence the change
  reduced FPs. Direction is not derivable from a weight norm: under the
  shared loss (BCE-focal `pos_weight=10`, N6) with 11–15% fragment positives,
  a trainable scoring head can as easily learn to up-score fragment-like
  clutter. The mechanism story (frozen calibration → less separation → more
  clutter above threshold AND compressed TP confidence) is coherent and
  uniquely fits *both* halves of the signature, which is why Task 1's
  ordering is defensible — but the comparison it rests on is confounded by
  everything else that differs (data, scale, tree, steps), and the audit's
  own cause 2 concedes an unquantified residual delta from tile-seam-shaped
  fragments transferring to inference seams more directly than the spike's
  whole-image fragments. RUN A is exactly the right disambiguator; the plan
  correctly treats a null as informative.
- **Why it matters:** two lesser errors follow from the overstatement:
  (a) `adapt_scoring_head` defaulting to **True** before RUN A reports means
  any user run in the interim trains an unvalidated surface; (b) acceptance
  "extras/frame move toward the spike's" is unfalsifiable at C4's noise
  floor — a 0.4/frame "movement" will be read as confirmation.
- **Correction:** default the flag off until RUN A validates (flip the
  default in the same commit that records the result); tie acceptance to the
  C4 significance criterion; phrase the premise as "leading hypothesis,
  decided by RUN A", which is also more honest to the audit's own language.

---

## Minor

### m1. Task 3's 16–19 GiB extrapolation is the wrong mental model, though conservative — and Step 2 violates the plan's own constraint

Under `torch.autocast(bfloat16)`, parameters, gradients, and optimizer state
are already fp32 (autocast casts op inputs, not weights; `perflib_compat`
deliberately dropped the hard bf16 casts). Only activation memory ≈doubles,
so true fp32 peak is < 2× the 10.43 GiB bf16 peak — likely ~13–17 GiB, not a
clean ×2. The plan's estimate errs high, which is safe on a 24 GB card and
Step 1 measures anyway, so this is minor — but it is the same *class* of
reasoning (unmeasured multiplier) the plan's Global Constraints condemn, and
`preflight.py:80` already hardcodes `_FP32_DEVICE_PEAK_MULTIPLIER = 2.0`.
Sharper: **Step 2 ("Set `_FP32_DEVICE_PEAK_MULTIPLIER` from that
measurement") literally instructs baking a measurement into a source
constant, which the Global Constraints forbid in bold.** Reword one of them
(e.g. the constraint means "never bake an *unverified* figure; measured
constants must cite their measurement and date").

### m2. The audit's fix recommendation 3 (correct `cli.py:20-24`) is dropped

The audit's confident recommendations were three; the plan implements two.
The stale rationale ("val loss anti-correlated with held-out AP" — resting
partly on a fold whose val split was byte-identical to train) remains in
`cli.py:21-24` and actively forbids best-checkpoint selection for future
readers. One doc-only commit; add it (Task 4 or a standalone step).

### m3. The shared ~0.72 recall plateau is never interrogated

Both models plateau at ~0.72 recall and ~5–8 extras/frame. If the missed 28%
are unlabeled/ambiguous ground truth or a genuine model ceiling, "reproduce
the spike" optimizes a 1.5/frame delta while ignoring a 10× larger shared
deficit. Task 5 attacks the FP floor; nothing attacks (or explains) the
recall floor. Cheap addition to Task 0: adjudicate the misses on the 16
frames once (occlusion? tiny? label absent?) so the programme knows whether
parity is the right goal or merely the measurable one.

### m4. Task 0's baseline JSON lives in `specs/`

`docs/superpowers/specs/2026-09-05-parity-baseline.json` puts measurement
artifacts in the design-spec tree, which the docs lifecycle rules move to
`done/` on merge. Harmless, but a `tools/` or fixtures location (or the
ledger) is a better home. Also: the calibration sweep needs the 16 *full
frames plus labels*, not the derived tiles — the plan should name the exact
source path so "identical settings" is reproducible.

### m5. Known-reference block errors

Besides C1's "batch 4": "Ours … 3 epochs / ~3348 steps" matches N7, fine,
but "spike … 10 epochs / ~1080 optimizer steps" should be labeled batch 1 so
future readers do not re-derive batch 4 from it. And the plan's Evidence
table is the *training-frame* comparison it immediately declares biased —
fine as motivation, but the parity target row should be re-stated from Task
0's held-out numbers once they exist, or later tasks will keep being judged
against the biased table.

---

## What the plan gets right (so the fixes do not throw it away)

- Parity-as-default with named deviations, one-axis discipline, and
  measure-before-assume are the correct posture, and Task 0-before-anything
  is the single best decision in the plan.
- The two overturned-conclusion claims in Global Constraints are **both
  correct** as stated: (a) the spike's checkpoint does carry trained adapters
  at the clone sites (verified by key listing: encoder 60/decoder 84 include
  the `cross_attn_image`/`cross_attn` q/k/v/out projections; the old tree's
  wrapper being a torch-MHA subclass explains *how*); (b) the last-epoch
  rationale does rest on a fold whose val==train (N4) — though note the
  earlier audit's D11 verdict (keep last-epoch) was *retained*, not reversed,
  by the new audit; the plan's wording "must NOT be re-applied" slightly
  overshoots for (b), since best==last made selection moot.
- Refusing to "fix" `is_crowd` by copying the spike, keeping the non-finite
  gradient skip, and the "record a null result" stance in Task 1 are all
  sound.
- Task 3's measure-first structure is exactly right; only its arithmetic
  framing and Step 2's constant-baking need the m1 rewording.

## Verdict

**Not fit to execute as written; fixable by amendment, not redesign.** The
skeleton (Task 0 baseline → surface → precision → cadence → data, one axis,
recorded either way) is sound. But: Task 4 aims at a batch size the spike's
run config disproves (C1); Task 2's acceptance number is unreachable under
its stated scope and needs the geometry-encoder work spelled out (C2); Task
1's interface is a silent no-op under the current suffix rules and
under-pins how many modules get wrapped (C3); and Task 0 lacks the
statistical machinery (paired frames, significance criterion, extras
adjudication) to detect the effects every downstream accept/reject decision
depends on (C4). Fix those four plus M1–M5 and the plan is executable.
