# SAM3 LoRA: measured device-memory cost per extra batch item

**Date:** 2026-09-06 · **Box:** courtship (RTX 4090, 24564 MiB) · **Commit:** `e65bdb2aa40b6bd1aa2a8e937cff1d2bb7c7068d` (`main`, bundle-transported to a fresh scratch checkout `~/sam3_batch_sweep/repo`; nothing pushed)
**Env:** `hydra-sam3` (`/home/rutalab/anaconda3/envs/hydra-sam3`), `PYTHONPATH=~/sam3_batch_sweep/repo/src` (verified `hydra_suite.__file__` inside the process), `KMP_DUPLICATE_LIB_OK=TRUE`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (asserted inside the training process before CUDA init; the probe aborts with exit 2 if absent).
**Containment:** `systemd-run --user --unit=sam3-batch-sweep -p MemoryMax=60G -p MemorySwapMax=0 -p TasksMax=512`, one child process per batch size.

## Configuration (held fixed across all four points)

rank 16 · alpha 32 · bf16 · grad_accum 8 · seed 42 · **312 adapted Linear modules**
(`adapt_scoring_head` OFF — production surface) · tiles 1766x1766 (`auto_object`,
`object_tile_fraction` 0.055, `reference_body_px` 97.13) · `tile_overlap` 0.25 ·
`keep_empty_tiles=False` · `label_quality_acknowledged=True`.

**Corpus:** `improved_ant_detection` / `artifacts/imported_sources/obb-6609f9b105`.
Verified 78 frames / **1871 instances** before build. Build produced exactly the expected
figures: **486 train tiles, 3363 train annotations, 16 held-out valid stems**, tile 1766x1766,
1944 train datapoints (243 optimizer steps/epoch at batch 1).

Each batch size ran through the production entry point `cli.run_training` against the same
prepared dataset, stopped by counting `torch.optim.AdamW.step`. Step budgets were chosen so
every point consumes a comparable slice of the corpus (~1.6-2.0 epochs of datapoints), which
matters because the peak tracks per-tile CONTENT: b1 = 400 steps, b2 = 220, b4 = 120, b8 = 60.
All four are `stopped_early` and are therefore **LOWER BOUNDS**, not completed-run peaks.

## (a) Results

| batch | reserved peak | allocated peak | gap | reserved/alloc | step of last new reserved max | step of last new alloc max | steps run | outcome |
|---|---|---|---|---|---|---|---|---|
| 1 | **7.125 GiB** | 6.634 GiB | 0.491 GiB | 1.074 | 214 | 315 | 400 | stopped_early (lower bound) |
| 2 | **8.434 GiB** | 7.913 GiB | 0.521 GiB | 1.066 | 158 | 158 | 220 | stopped_early (lower bound) |
| 4 | **11.229 GiB** | 10.699 GiB | 0.530 GiB | 1.050 | 1 | 28 | 120 | stopped_early (lower bound) |
| 8 | **18.748 GiB** | 17.853 GiB | 0.895 GiB | 1.050 | 4 | 60* | 60 | stopped_early (lower bound) |

\* b8's last new allocated max coincides with the last recorded step; its allocated peak may
still be a slightly looser lower bound than the others.

Host RSS peak 7.44 GiB at every batch size. Zero allocator warnings, zero skipped/non-finite
optimizer steps, normal loss descent at every batch size (b8: 123.4 -> 51.7 over 50 steps).

**Fragmentation signal: clean.** Reserved and allocated move together; the reserved/allocated
ratio does not grow with batch (1.074 -> 1.050). The absolute gap widens at b8 (0.90 GiB) but
proportionally it is the *smallest* of the four. The measurement is not contaminated.

**Corroboration:** the batch-1 figure, 7.125 GiB, is **exactly** the full 2430-step
expandable-segments run recorded in `sam3_lora/env.py`. A 400-step probe at b1 reproduced a
completed run's peak to three decimals — evidence that under this allocator these probes are
close to converged, though still formally lower bounds.

## (b) Marginal device cost per extra item

| interval | Δ reserved | items | **GiB / extra item** | Δ allocated / item |
|---|---|---|---|---|
| 1 -> 2 | 1.309 GiB | 1 | **1.309** | 1.279 |
| 2 -> 4 | 2.795 GiB | 2 | **1.397** | 1.393 |
| 4 -> 8 | 7.520 GiB | 4 | **1.880** | 1.789 |

**It is not linear. It is mildly SUPERLINEAR (convex).** The marginal cost per item rises
monotonically, 1.31 -> 1.40 -> 1.88 GiB. A least-squares line `a + b*n` gives a = 5.11 GiB,
b = 1.674 GiB/item with residuals up to 0.58 GiB (5% at n=4) whose signs form the U-shape
(+0.34, -0.03, -0.58, +0.25) characteristic of convex data. Four points cannot distinguish
among convex candidates (quadratic, `n*log n`, saturating-max models); what they *do* support
is a direction, and the direction is convex, not concave.

**Mechanism (hypothesis, consistent with the data):** SAM3's collator pads a batch to its
maximum instance/mask count, so a batch's cost is roughly `n x f(densest tile IN the window)`.
As `n` grows, the per-window maximum density rises toward the corpus maximum, so the per-item
factor grows too. This is directly visible in the "step of last new maximum" column: at b1 the
peak needed **214 steps** to encounter the densest tile; at b4/b8 it was reached at **step 1-4**,
because a window of 4-8 tiles almost always contains a dense one. Under this model the convexity
must *saturate* once `n` is large enough that the window max equals the corpus max — so
extrapolating the 1.88 GiB/item slope past 8 would over-estimate.

**Disagreement with the mehek 2-step figures.** Those (7.34 / 10.16 / 13.98 GiB -> 2.82 then
1.91 GiB/item) suggest the OPPOSITE shape, sublinear. My data contradicts it, and the
truncation argument runs the *other* way from what would rescue them: a 2-step probe under-reads
b1 most (b1 needs ~214 steps to find its peak) and b4/b8 barely at all (peak by step 4), which
would manufacture *false superlinearity*, i.e. it can only exaggerate my effect, not create
mehek's sublinearity. My own step-2 slice reproduces their protocol on my corpus and still comes
out convex (6.695 / 8.023 / 11.229 / 18.746 -> 1.33, 1.60, 1.88 GiB/item). So the mehek shape is
most plausibly a corpus difference (their densest tiles were pre-selected, flattening the
content-sampling effect that produces convexity here) or a card/kernel difference, not a
truncation artefact. **Either way, both datasets agree the per-item cost is NOT a constant** —
they merely disagree on the sign of its drift.

## (c) Recommendation for `_EXTRA_BATCH_DEVICE_BYTES`

**The constant's model is wrong, but a constant can still be made SAFE over the measured range.**

- Current value **18 GiB is ~10x the largest measured marginal (1.88 GiB)** and ~13x the
  b1->b2 marginal. It makes the analytic term dominate `max(analytic, measured)` at every batch
  above 1, so the measured-auto-batch feature can never raise a batch size on any card it targets.
  That is a real, measured defect.
- If the constant is kept as-is structurally, the defensible value is
  **`_EXTRA_BATCH_DEVICE_BYTES = 2 * GiB`.**
  Derivation: a secant bound, not a fit. `max over n of (peak(n) - peak(1)) / (n - 1)` =
  max(1.309, 1.368, 1.660) = **1.66 GiB/item**; 2 GiB adds ~20% margin and dominates every
  measured point (`7.125 + 7*2 = 21.1 >= 18.75` at n=8). A *fitted* slope (1.674) or the b1->b2
  marginal (1.31) would UNDER-admit-safely at n=8 and must not be used: the correct quantity for
  an admission floor is the secant from n=1, not any local marginal.
- **Preferred: replace the term rather than recalibrate it.** The honest model over 1..8 is a
  base plus a superlinear term. The minimal defensible replacement that these four points support
  is a piecewise-constant secant table (`<=2: 1.4, <=4: 1.5, <=8: 2.0 GiB/item`), or simply
  refusing to extrapolate: admission for batch `n` should consult the measured envelope for `n`
  and only fall back to the analytic term when no measurement exists.
- **Do NOT scale it by precision naively at large batch.** The fp32 x2 multiplier is applied to
  this term today; at batch 8 that projects 7 x 36 GiB with the current constant, and even at
  2 GiB it projects 28 GiB of "extra" — plausible for activations, but unmeasured. Untested.

**Limits of this number, stated plainly:** one corpus (1871 ant instances, 486 tiles at 1766 px),
one card (RTX 4090, sm_89), one allocator config (`expandable_segments:True`), one adapter
surface (312 modules, rank 16), bf16 only, grad_accum 8, and four points, all of them formally
lower bounds from truncated probes. Peak is content-dependent, so a denser corpus can raise every
number here. This is a calibration for an admission FLOOR, not a prediction of any particular run.
It does not transfer to the default allocator: under it, no probe of any length bounds the peak.

I did not edit the constant.

## (d) Largest batch that fits, and what admission would say

**Batch 8 fits and trains stably on a 24 GB RTX 4090.** 18.748 GiB reserved against 24564 MiB
total, 60 optimizer steps across two epoch boundaries, exit 0, no OOM, no allocator warnings, no
skipped steps, loss descending. Batch 8 has zero measured headroom to spare against a 0.8 usable
fraction (18.75 vs 19.20 GiB usable) and would be a genuinely marginal admission; batch 4
(11.23 GiB) has ~8 GiB of margin. **No batch size OOMed at any point in the sweep.**

Admission arithmetic on this card (free 24 GiB, usable 19.20 at the 0.8 fraction; the composed
analytic requirement at batch 1 is 12.36 GiB, of which 12 GiB is the inflated base envelope):

| batch | analytic @ 18 GiB (today) | analytic @ 2 GiB (corrected) | measured envelope | actually needs | admitted today? | admitted corrected? |
|---|---|---|---|---|---|---|
| 1 | 12.36 | 12.36 | 7.27 | 7.13 | YES | YES |
| 2 | 30.73 | 14.36 | 9.21 | 8.43 | no | **YES** |
| 4 | 67.45 | 18.36 | 13.67 | 11.23 | no | **YES** |
| 8 | 138.9 | 26.36 | — | 18.75 | no | no (refused; it would in fact have worked) |

So the corrected constant turns the feature on: it would admit **batch 4** on a 24 GB card, which
the measurement shows runs with a wide margin. Batch 8 stays refused — conservative but not
absurd, since the 12 GiB base is itself ~1.7x the measured b1 envelope and b8's true requirement
is within 3% of the 0.8-fraction budget.

## (e) `contracts.py:217` — "batch 2 OOMs at 1008 px on a 47 GB card"

**FALSE / stale.** Batch 2 peaked at 8.43 GiB reserved on a 24 GB card — under a fifth of a 47 GB
card's capacity — and batch 8 completed at 18.75 GiB. Nothing in the 1..8 range comes near an OOM
on hardware half that size. The comment predates `perflib_compat` and the streaming dataloader.
Another session has already corrected it on their branch from mehek measurements; my figures agree
with that correction (batch 2 does not OOM) while disagreeing with their *shape* — see (b).

## Artifacts

Per-step records (reserved/allocated at every optimizer step, new-max flags, timings):
`.superpowers/sdd/sam3-batch-scaling-data/result_b{1,2,4,8}.json` (kept next to this report); produced by
`~/sam3_batch_sweep/probe_batch.py` on courtship. Scratch on courtship removed after collection.

## Note on `sam3_env_environ()` and the flag

Verified statically, since this measurement did not go through the launcher:
`train.py:185` merges `sam3_env_environ()` into the child environment and passes it to
`build_limited_launch(..., environment=child_environment)`, which at
`runtime/resource_limits.py:165` does `child_env = dict(environment)` — so a real sidecar launch
does export `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to the training child.
**It is NOT set on a direct `conda run` / `systemd-run` path** (nothing in that path calls
`sam3_env_environ()`), which is why this probe sets it explicitly and asserts it in-process.
