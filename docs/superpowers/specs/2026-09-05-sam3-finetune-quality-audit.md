# SAM3 finetune quality audit: why our checkpoint emits more false positives than the spike's

**Status:** root-cause analysis (read-only audit, no code changed)
**Date:** 2026-09-05
**Prior work:** `docs/superpowers/specs/2026-09-04-sam3-spike-vs-build-consolidation.md`
(the design-grounds audit). This document explains a *measured* accuracy gap and
explicitly overturns or confirms that audit's judgement calls where the new
evidence bears on them.

## The thing to explain

Same 12 labelled frames (275 instances), same tiling/merge/prompt. Recall and
median IoU are a wash (0.868 vs 0.863). The gap has a specific signature:

1. **Ours emits +1.2–2.0 extra detections/frame at every confidence** in the
   usable range (e.g. 8.33 vs 6.83 at conf 0.30).
2. **Ours' true-positive confidence is compressed**: between conf 0.65 and
   0.85 spike recall falls 0.720→0.611 while ours falls 0.702→0.534.

The eval frames come from OUR training source (bias in our favour), and we
still lost. A credible root cause must predict *excess false positives and
compressed TP confidence*, not generic quality loss.

## New evidence established in this audit

All verified directly on primary sources (mehek `~/sam3_spike/`, courtship
`~/sam3_train/`, local `src/hydra_suite/training/sam3_lora/`), read-only.

- **N1 — our adapter surface is a strict subset of the spike's; the 108
  missing modules include the confidence-producing scoring head, and 97 of the
  108 genuinely trained.** `torch.load` of ours
  (courtship `~/sam3_train/ws_full/runs/20260904-143950_semantic_sam3_ca1bd031/adapters.pt`,
  412 tensors / 206 modules) vs spike
  (mehek `~/sam3_spike/out/fold_all_r16/best_lora_weights.pt`, 628 tensors /
  314 modules): after path normalization, **ours ⊂ spike exactly; 0 modules
  are ours-only**. The 108 spike-only modules, grouped:
  - `transformer.encoder.layers.*.self_attn.{q,k,v,out}_proj` (24) and
    `transformer.encoder.layers.*.cross_attn_image.{q,k,v,out}_proj` (24) —
    the DETR **fusion encoder's** attention, where the text prompt is fused
    into image features;
  - `transformer.decoder.layers.*.cross_attn.{q,k,v,out}_proj` (24) — decoder
    **image cross-attention**;
  - `geometry_encoder.*` (30);
  - `segmentation_head.cross_attend_prompt.{q,k,v,out}_proj` (4);
  - **`dot_prod_scoring.prompt_proj` and `dot_prod_scoring.hs_proj`** (2, plus
    2 geometry projections) — the text/vision similarity head that scores
    every query, i.e. the module that literally emits detection confidence.
  Trained-ness check (lora_B is zero-init, so a module whose forward never ran
  has lora_B ≡ 0): only **11 of 108 have lora_B exactly zero** (all in
  `geometry_encoder`, unused for text-prompt training). The **largest lora_B
  norms among spike-only modules are `dot_prod_scoring.prompt_proj` (0.174 —
  the single biggest mover in the set), `segmentation_head.cross_attend_prompt.*`
  (~0.10), `dot_prod_scoring.hs_proj` (0.097) and decoder `cross_attn` layers
  0–1 (~0.10)**, against a spike-only median of 0.065 (shared-module median
  0.104). The spike's advantage lives disproportionately in scoring and
  prompt-grounding attention that our build cannot reach.
- **N2 — nothing in either SAM3 tree's loss/matcher/collator consumes
  `is_crowd`.** `grep -rn "iscrowd\|is_crowd"` over the new tree's
  `train/loss/` and `train/matcher.py`: zero hits. `BatchedFindTarget`
  (`train/data/collator.py:167-181,296`) carries no crowd field. The only
  consumer is the opt-in `FilterCrowds` transform
  (`train/transforms/filter_query_transforms.py:523-536`), which neither
  implementation uses. An `iscrowd=1` instance is matched and supervised
  exactly like a normal positive (box L1+gIoU, CE, mask+dice). This confirms
  hypothesis 1's premise — but see N3.
- **N3 — the spike ALSO trained crowd fragments as full positives, at a
  similar rate.** Spike fold_all train set: 108 images / 177 instances, **26
  iscrowd=1 (14.7%)**, ingested without ever reading `iscrowd`
  (`train_sam3_lora_native.py:725,934` only *write* `iscrowd: 0` into
  eval-time dicts). Ours (courtship, all three
  `~/sam3_train/ws_full/prepared/*/derived/semantic_sam3/` builds are
  statistically identical): train 2232 tiles / 3286 anns, **380 iscrowd=1
  (11.6%)**, median crowd area 348 px² vs 1175 px² for clean (≈30% of a full
  instance); valid 576/805 with 88 crowd (10.9%). So fragment-positives are a
  **shared** defect, not the differentiator — consistent with BOTH models
  emitting a high absolute FP floor (5–8 extras/frame).
- **N4 — spike run facts.** `~/sam3_spike/out/fold_all_r16/` +
  `~/sam3_spike/work/configs/fold_all_r16.yaml`: **10 epochs actually ran**;
  per-epoch val loss 0.800…0.403, monotone-ish with the minimum at epoch 10,
  so **best_lora_weights.pt == last epoch** (selection at
  `train_sam3_lora_native.py:1327-1346`). Batch size 1, **no** accumulation /
  scheduler / warmup / clipping (optimizer built at `:1030-1034`; step loop
  `:1269-1272`; the yaml's `gradient_accumulation_steps: 8`,
  `warmup_steps`, `lr_scheduler: cosine`, `max_grad_norm` are all dead) →
  ~**1080 optimizer steps** at constant lr 5e-5. Caveat: fold_all's valid
  split is **byte-identical to train** (md5-equal), so its "val loss" is
  train loss.
- **N5 — `is_exhaustive` semantics are identical on both sides.** Spike's
  `dataset_formats.py:182-198` sets `is_exhaustive=True` unconditionally (as
  do ours: `datapoints.py:155`). In the loss, non-exhaustive queries have
  their negative/no-object BCE nullified (`train/loss/loss_fns.py:452-456`)
  and FP penalties gated to exhaustive batches (`:1167-1221`). Both regimes
  therefore apply full negative pressure to unmatched slots and full positive
  supervision to every annotation, crowd fragments included. Not a
  differentiator.
- **N6 — negative weighting (shared loss config):** classification is
  BCE-focal with `pos_weight=10.0`, `alpha=0.25`, plus `loss_mask=200.0`,
  `loss_dice=10.0` applied to matched positives only
  (`train_sam3_lora_native.py:1042-1085`; loss defaults mirrored by our
  build). Negative supervision exists but is comparatively weak; positive
  supervision on fragments is strong. Shared, not a differentiator, but it
  amplifies whatever positives the dataset asserts.
- **N7 — our dataset composition:** 2232 train tiles (971×971 upscaled to
  1008), **75.6% with zero annotations** (1687 empty tiles kept), overlap
  0.25, `reference_body_px` 97.13, `object_tile_fraction` 0.1, negatives
  `["background","shadow","debris"]`. 8928 datapoints = 2232 tiles × (1
  positive + 3 negative queries). ~1116 micro-batches/epoch at grad_accum 8 →
  ~**3348 optimizer steps over 3 epochs** under warmup+cosine
  (`cli.py:610-611,645`). Per-instance positive exposure: ours 3286×3 ≈ 9.9k
  instance-exposures vs spike 177×10 ≈ 1.8k — ours is not starved in raw
  counts.

---

## Ranked root causes

### 1. Adapter surface: ours cannot reach the scoring head or the prompt-fusion attention the spike's gains live in — HIGH confidence, explains the FP signature directly

**Mechanism.** SAM3 scores each candidate by a dot product between projected
decoder outputs and the projected prompt embedding (`dot_prod_scoring.hs_proj`
/ `prompt_proj`), after the prompt is fused into image features by the DETR
encoder's `self_attn`/`cross_attn_image` and read back by the decoder's
`cross_attn`. Fine-tuning these is what *sharpens the meaning of "ant"* —
raising scores on true ants and lowering them on ant-like clutter. The spike
adapted all of them and its checkpoint moved them substantially (N1: the
scoring head is its single biggest mover). Our build structurally excludes
them: `dot_prod_scoring` is covered by no `adapt_*` flag, deliberately
(`lora.py:48-52`: "covered by NO flag, deliberately"), and the encoder
`cross_attn_image` / decoder `cross_attn` / `segmentation_head.
cross_attend_prompt` sites are SAM3's clone MHA (`model_misc.py`), which
`inject_adapters` skips because the clone reads `out_proj.weight` functionally
(`lora.py:333-364` replaces only true torch `nn.MultiheadAttention`). Net: our
206 modules tune features and box/mask geometry (hence IoU parity, 0.863 vs
0.868) but leave the *score calibration pathway* frozen at the base model.

**Predicts the signature?** Yes, both halves. A frozen scoring pathway on top
of shifted features means the score distribution separating ants from clutter
is less sharpened: more clutter above any threshold (+1.2–2.0 extras/frame)
and true positives not pushed toward high confidence (recall collapse at 0.85:
0.534 vs 0.611). IoU parity is exactly what this predicts — localization
heads WERE adapted.

**Evidence for:** N1 (strict subset; 97/108 spike-only modules trained;
scoring head is the top mover). **Against:** none found; the alternative
"those spike adapters were dead proxies" is refuted by the lora_B norms.

**vs the prior audit:** *Confirms and upgrades D1* from "medium-high" to the
top-ranked measured cause — and the audit's D-EXP-1 has effectively now run in
reverse (same-data comparison, restricted surface lost). It also *corrects
E5/D1's scoping*: the audit asserted "the spike never adapted SAM3's own clone
(it is not an `nn.MultiheadAttention` instance)". The checkpoint disproves
that: spike adapters at `cross_attn_image`, decoder `cross_attn` and
`segmentation_head.cross_attend_prompt` carry nonzero trained lora_B — in the
OLD vendored tree those projections were reachable, callable Linears. The
"empirically validated surface" therefore INCLUDES the clone sites, so a D1
port limited to torch-MHA replacement (what our build now has via
`SplitMultiheadAttention`, 206 modules) recovers only part of the spike's
surface.

### 2. Shared-defect FP floor: tile-seam fragments trained as full, exhaustive positives — MEDIUM confidence as a large contributor to the absolute FP rate; LOW as the delta's cause

**Mechanism.** `dataset_build.py:54-58,239` marks any instance retaining
<50% of its area after tile clipping as `iscrowd=1` (no lower bound — a 2%
sliver still qualifies) and keeps it; `datapoints.py:96-100` feeds it to the
loss as a positive under `is_exhaustive=True`; N2 proves nothing downstream
ignores it. 380/3286 train annotations (11.6%) are such fragments at a median
30% of full-instance area (N3). The model is explicitly taught that partial
ant fragments are complete, mandatory detections — at exactly the scale and
seam geometry inference tiles reproduce.

**Predicts the signature?** It predicts excess FPs in general (seam-fragment
detections that fail `merge_iou 0.5` across overlapping tiles surface as
extras), but **it cannot be the primary explanation of the spike-vs-ours
delta**: the spike trained on a *similar fraction* of crowd-fragment positives
(26/177 = 14.7%, N3) with identical loss semantics (N2, N5). It IS the best
available explanation for why **both** models emit 5–8 extras/frame — an
absolute precision ceiling worth fixing regardless of the comparison.
Residual delta contribution is plausible but second-order: our fragments are
tile-seam-shaped at inference scale, the spike's were whole-image-resize
artifacts at a different scale, so ours transfer to inference seams more
directly. Unquantified.

**vs the prior audit:** consistent with its findings; extends them with the
loss-side proof (N2, N5) and dataset-side quantification (N3).

### 3. Training regime: bf16 autocast + warmup/cosine vs the spike's pure-fp32 constant-lr — LOW-MEDIUM confidence, plausible secondary contributor

**Mechanism.** Ours trains everything (including internal matching and the
loss) under `torch.autocast(bfloat16)` with warmup(≤50 steps)+cosine decay
over ~3348 steps (`cli.py:610-611`, N7); the spike ran pure fp32 at constant
5e-5 for ~1080 steps (N4, confirming the prior audit's E2 and the "dead
config" claim in full). Cosine halves the average lr relative to peak; bf16
adds rounding noise to focal-CE logits and matcher costs. Either could leave
the concept slightly less converged — consistent with compressed TP
confidence.

**Predicts the signature?** The confidence-compression half, weakly; not
specifically the FP excess. **Against:** ours had 3× the optimizer steps and
~5× the per-instance exposures (N7), and its final val_loss_mean (0.166) is
far below the spike's train-set loss (0.403) — though those numbers are not
comparable across datasets. No direct evidence of harm; ranked here because it
is the largest remaining regime difference after 1 and 2.

**vs the prior audit:** D4/D5 unchanged ("needs experiment"); nothing here
convicts bf16, and this run did not NaN.

### 4. Positive-stream dilution: 75.6% empty tiles — LOW confidence, direction actually favours precision

Ours' stream is dominated by background-only tiles each carrying an exhaustive
empty "ant" query plus 3 negatives (N7). That is *precision supervision* — it
pressures scores on background DOWN. It cannot explain excess FPs; if
anything it makes the observed gap more damning for cause 1. Its real cost is
compute efficiency (¾ of every epoch spent on background). Prior audit D7's
"keep empties" stands, though the ratio (75%) is far beyond anything the spike
saw and D-EXP-3 (ratio sweep) remains worth running for throughput reasons.

### 5. Checkpoint selection (last vs best) — REFUTED as a cause

The spike's `best_lora_weights.pt` **is** its last epoch: val loss hit its
minimum at epoch 10 (N4). Selection policy made zero difference to the
artifact we lost to. Additionally, `cli.py:20-22`'s stated rationale ("the
spike found val loss anti-correlated with held-out AP") should be treated
with suspicion going forward: for fold_all the "validation" split was
byte-identical to train (N4), so that anti-correlation claim rests on the
other folds only. Our 3-epoch run also has no surviving per-epoch salvage
checkpoints to compare (only `adapters.pt` remains in the run dir). Prior
audit D11's verdict survives, but on weaker evidence than it claimed.

### 6. Negative-prompt semantics — REFUTED as a differentiator

Spike: 3 generic negatives (word-overlap filtered) + in-domain absent
categories (`dataset_formats.py:213-247`); with a single-category corpus the
in-domain tier contributes nothing beyond `num_cross_negatives: 2` samples.
Ours: 3 fixed curated negatives (`background`, `shadow`, `debris`) per tile
(`dataset_build.py:52`, `dataloader.py:76-104`). Both exhaustive-empty query
rows with identical loss treatment (N5). The semantic difference is
cosmetic at this corpus size.

### 7. Architecture/numerics of the newer tree (`perflib_compat`) — REFUTED as a cause on current evidence

The prior audit's E4 verified the eager patch reproduces timm-`Mlp` semantics
(what the spike trained through); the load-bearing training files
(matching, matcher) are identical across trees (E3/E6). Nothing in this
audit's measurements implicates it — the deficit localizes to *unadapted*
modules (N1), not to differently-computed adapted ones. The one place tree
differences DO matter is cause 1's mechanism: the newer tree's clone MHA is
what makes the fusion/cross-attention sites unreachable by plain Linear
wrapping, whereas the old tree exposed them (see cause 1, "vs the prior
audit").

---

## Prior-audit judgement calls this evidence bears on

- **D1 (adapt MHA surface): confirmed, and its top-priority ranking is now
  empirically vindicated** — but its scope was too narrow. The spike's
  validated surface includes SAM3-clone attention sites
  (`cross_attn_image`, decoder `cross_attn`, `segmentation_head.
  cross_attend_prompt`) and `dot_prod_scoring`, which E5/D1 explicitly walled
  off ("do NOT touch SAM3's own clone … the spike never adapted it either" —
  false, per N1's lora_B norms). The port must extend to a split-projection
  replacement of `model_misc.MultiheadAttention` call-sites *and* add a
  `dot_prod_scoring` scope flag.
- **D9 (SAHI tiles > whole images, "medium-high"): weakened.** A tile-trained
  model evaluated by tiled inference *over its own training source* lost on
  precision to a whole-image-trained model on foreign data. Tiling also
  *created* the 380 fragment-positives (cause 2), a defect the whole-image
  regime produces less of per-inference-seam. D9's premise (train/serve scale
  match) still stands for recall/IoU — which are at parity — but it is no
  longer safe to call the tile regime unambiguously better without fixing
  fragment labeling.
- **D11 (last-epoch selection): survives, but its evidentiary basis is weaker
  than stated** — fold_all's val==train, and best==last anyway (cause 5).
- **D4/E2 (dead config, fp32): confirmed in detail** (N4).

---

## Cheapest decisive experiment (top cause)

**One-variable retrain: extend the adapter surface, same everything else.**
On courtship (RTX 4090, free), retrain from the *same* prepared dataset
(`~/sam3_train/ws_full/prepared/dataset-preparation-3365d6c8…`) and same
spec, with the surface extended in two steps of increasing cost:

1. *Two-line scope experiment (hours):* add a flag exposing
   `dot_prod_scoring.{hs_proj,prompt_proj}` (plain, callable `nn.Linear`s —
   wrappable today with zero architectural work; they are excluded only by
   `lora.py`'s prefix list). Re-run the identical 12-frame calibration sweep.
   If extras/frame drop ~1/frame toward spike levels and the conf-0.85 recall
   gap closes materially, the scoring-pathway mechanism is convicted on the
   cheapest possible surface.
2. *Full D1-extended slice (a day):* additionally replace clone-MHA sites
   (`cross_attn_image`, decoder `cross_attn`,
   `segmentation_head.cross_attend_prompt`) with split-projection modules and
   re-sweep. Publish-side merging follows the audit's D1 recipe (in_proj
   row-slice folding); the spike's `restore_multihead_attention` proves
   exactness.

Success criterion: ours ≤ spike extras/frame at conf 0.30–0.65 and recall at
0.85 within 0.03 of spike, on the same 12 frames.

## Fix recommendations (confident)

1. **Extend the LoRA surface** per the experiment above; make the extended
   surface the default once step 2 validates. Update
   `sizing.expected_lora_trainable_params` and the cli invariant counts.
2. **Stop asserting sub-50% tile fragments as exhaustive full positives**
   (shared FP-floor fix, benefits any surface): either drop instances below a
   retained-area floor (e.g. <30%) AND mark the affected query
   `is_exhaustive=False` (the loss then nullifies negative pressure on that
   tile, `loss_fns.py:452-456`, so the un-annotated fragment is neither a
   positive nor punished as an FP), or route fragments through the loss's
   actual ignore mechanism by adding `FilterCrowds`
   (`filter_query_transforms.py:523-536`) plus the same exhaustive downgrade.
   Do NOT simply drop them while keeping `is_exhaustive=True` — that re-creates
   the earlier "teach fragments = background" bug the current code's comment
   warns about (`dataset_build.py:54-58`).
3. **Correct the record in `cli.py:20-22`** (doc-only): the spike's shipped
   fold_all checkpoint was best==last, and fold_all's val split was its train
   split; the anti-correlation rationale should cite the other folds or be
   softened.
