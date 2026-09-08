# Multi-scale SAHI training for SAM3 — porting the YOLO scale set

**Spec:** `docs/superpowers/specs/2026-09-06-unified-sahi-training-geometry-design.md`
(§3.3 multi-scale for SAM3, §3.5 stamping, §4.2 R2/R3/R3b, §4.3 full frames)
**Status:** Shipped — merged to main (`50cb5b94`). Tasks 1-7 implemented; Task 8's measured
GPU gate ran on `firebrat` (RTX 4090) with no OOM. See the Task 8 Results subsection below.
**Base:** local `main` @ `e29838cb`
**User direction (verbatim):** *"the sahi training needs to become like the sahi training we
do for yolo models — multi scale multi level (a variety of tile sizes including mixed full
frames). It would increase the dataset size but it would make it more robust."*

> **WARNING — stale spec text.** The spec still carries pre-ruling bullets for **D7-D11**
> *below* their RESOLVED block. Those are SUPERSEDED. Nothing in this plan depends on them.
> Separately, spec §3.3 says full frames arrive via `plan_tiles(..., full_frame=True)`. That
> is **inaccurate against the code**: the YOLO builder emits full frames in a separate branch
> (`training/sliced_dataset.py:329-342`) and `SlicePlan.full_frame` is never set by it. Follow
> the code, not §3.3's sentence.

## Global Constraints

- Work in a git worktree branched from local HEAD (`git worktree add .worktrees/<n> -b <b> HEAD`).
  Never push. Commit as the configured git user, no `Co-Authored-By: Claude` trailer.
- Never `git stash` bare, never `git clean -fdx`, never `rm -rf`.
- Tests: `PYTHONPATH=<worktree>/src python -m pytest tests/test_sam3_*.py tests/test_slice*.py`.
  The whole suite hangs on `main` (classkit modal) — batch per file.
- SAM3 training is **CUDA-only** and runs in a sidecar conda env; any task that needs a real
  train step runs on `mehek` (`rutalab@mehek.taild08eb9.ts.net`, env `hydra-cuda` + the SAM3
  sidecar env). Dataset-build tasks are testable on this box with no GPU.
- **`tile_size_for_mode` semantics are frozen** (`utils/slice_geometry.py:122-124`). This work
  adds a loop *around* it; it must not touch the formula, its clamps, or its rounding (R1).
- **Default = today's behaviour.** A one-element scale set must produce a **byte-identical**
  dataset tree to the current builder. Multi-scale is opt-in for this change (D6/D17 stay
  open, see Out of scope).
- **D16 must remain open.** Three fractions are live in the tree — SAM3 training `0.055`
  (`training/contracts.py:264`), YOLO training `0.10` (`detectkit/config/training.py:161`,
  `training/sliced_dataset.py:109`), inference `0.15` (`core/inference/config.py:85`). This
  plan ships a *configurable* scale set and **must not** change SAM3's default fraction or
  bake a recommended set into code. The pre-registered 2x2 ablation decides that later.

---

## Current state (mapped 2026-09-06, file:line verified against `e29838cb`)

### YOLO multi-scale, as it actually works

- User-facing knob: `SliceTrainingConfig.target_size_fractions = (0.05, 0.10, 0.15, 0.20)`
  (`detectkit/config/training.py:168`), validated to `(0, 1]` (`:243-246`). Legacy absolute
  `target_sizes = (32, 64, 96, 128)` (`:170-172`) is demoted: `target_fractions()` (`:264-267`)
  prefers the fractions and only falls back to `target_sizes / 640.0`; supplying `target_sizes`
  without fractions explicitly blanks the fractions (`:220-222`). The GUI mirror is
  `detectkit/gui/models.py:180,254` (same `/640.0` fallback at `:258-260`).
- Resolution to pixels: `target_sizes_for(imgsz)` = `fraction * imgsz` (`training.py:269-271`),
  called once per role at `detectkit/jobs/training.py:220` with `request.imgsz_for(role)`.
- Tiling fan-out: `SliceBuildParams.target_sizes` (pixels, `training/sliced_dataset.py:115`)
  → `_tile_sizes_for_params` (`:171-206`) converts each back with `frac = target / imgsz`
  (`:185`) and calls `tile_size_for_mode(object_tile_fraction=frac)` (`:186-193`), deduping
  sizes. The scalar `params.object_tile_fraction` is used **only** when the fan-out is empty
  (`:198-206`).
- Dataset multiplication: the per-image loop tiles once per resolved size
  (`:302-327`); tiles are named `{stem}_t{W}x{H}_{i:04d}` (`:318`) — a **parsed interface**,
  not just provenance (below). Negative tiles are sampled at `negative_tile_fraction=0.15`
  (`:114`, `:316-317`).
- Mixed full frames: a separate branch, one un-tiled copy per image named `{stem}_full`
  (`:329-342`), gated on `full_frame_mix=True` (`:116`). It is **not** `SlicePlan.full_frame`.
- Scale balancing (shipped `306738ec`, hardened `4baab88c`):
  `training/ultralytics_scale_balance.py` — `scale_group_for_path` (`:109-118`) parses
  `_t{W}x{H}_` / `_full`; `scale_group_weights` (`:121-143`) inverse-frequency weights with
  `full`/`other` pinned to 1.0; `ScaleGroupedBatchSampler` (`:146-192`) makes batches
  scale-homogeneous. Installed from the manifest by
  `install_sahi_scale_balance_and_grouped_sampling` (`:265`), which **raises**
  `ScaleGroupedSamplingUnsupportedError` when `world_size > 1` unless
  `HYDRA_SAHI_DDP_ALLOW_UNGROUPED=1` (`:284-300`), and always writes a realised stamp
  (`:302-325`, `REALISED_STAMP_FILENAME`), carried onto the published sidecar by
  `training/service.py:~186-196` + `apply_realised_balance_stamp`.
- Manifest: `_slice_geometry_manifest` (`sliced_dataset.py:410-432`) stamps `target_sizes`
  (pixels) + `imgsz` + `multiscale_loss_balance{enabled,power}`. It does **not** stamp
  `target_size_fractions` (spec §3.5 amendment (a) — still unimplemented) and does not stamp
  realised per-scale counts (§3.3 surviving requirement — still unimplemented).

### SAM3 tiling, as it works today — exactly ONE scale

- `Sam3LoraParams.object_tile_fraction = 0.055`, `geometry_mode="auto_object"`,
  `tile_overlap=0.25`, `keep_empty_tiles=True` (`training/contracts.py:262-267`).
- One tile size for the whole build: `tile_size_for_mode(...)` called **once**, outside every
  loop, at `training/sam3_lora/dataset_build.py:390-397`, with `imgsz=_SAM3_IMGSZ` (`:105`,
  1008) which is inert because the `auto_object` branch fires whenever `reference_body_px > 0`.
- `reference_body_px` is the **mean of the middle 1-2 per-frame medians** (`:369-388`) — a
  different estimator from YOLO's global median over all majors
  (`sliced_dataset.py:344-348`); spec D1, out of scope here but must not be perturbed.
- `_tile_frame` (`:247-285`) plans tiles for one `(tile_w, tile_h)` and clips polygons;
  `retained_frac < MIN_RETAINED_AREA_FRAC` downgrades an instance to `iscrowd`
  (`:280-282`) rather than dropping it (YOLO drops below `min_area_ratio`).
- `_build_split` (`:493-556`) writes `{stem}_tile{idx:03d}.jpg` — **no scale token** — plus
  COCO `images`/`annotations` records keyed on integer image ids.
- No full-frame arm anywhere in the SAM3 builder.
- Manifest: `build_manifest.json` fields at `:608-628` — `tile_px: [w, h]`,
  `reference_body_px`, `object_tile_fraction`, `tile_overlap`, `fragment_counts` per split.
- Every tile is stretched to `RES`x`RES` = 1008 with **no aspect preservation**
  (`sam3_lora/datapoints.py:165-171`) — the R3b mechanism.
- Training loop: hand-rolled. `dataloader.build_descriptors` (`:166`) →
  `shuffled_batches` / `collate_batches` (`:284-332`). **No `torch.utils.data.DataLoader`, no
  sampler object, no `DistributedSampler`.** `train.py:395-411` pins exactly one CUDA device
  by UUID and *logs* that a multi-GPU device string is narrowed to one. Epoch count is fixed
  (`cli.py:944 for epoch in range(params.epochs)`), so steps/epoch ∝ dataset size
  (`cli.py:910,940`).

### Where a scale set would enter

Exactly one place in the builder: `dataset_build.py:390-397` becomes a resolved **list** of
`(w, h)`, and `_build_split`'s inner loop (`:527-534`) iterates scales × `_tile_frame`.
Everything downstream (`tile_px`, publish, sidecar, drift guard) is scalar-shaped and is the
real work — see Tasks 4-6.

---

## THE DENOMINATOR — the crux, stated once

`object_tile_fraction` is relative to **measured body size** (`tile_px = ref_px / frac`,
`slice_geometry.py:130-134`). `target_size_fraction` is relative to **model input size**
(`target_px = frac * imgsz`, `training.py:269-271`).

They are **numerically the same number** for a square tile that is resized to the model input:
apparent = ref·imgsz/tile, so apparent/imgsz = ref/tile. The YOLO builder relies on precisely
this identity — it divides the pixel target by `imgsz` and feeds the result straight in as
`object_tile_fraction` (`sliced_dataset.py:185-190`). **So porting a YOLO scale set to SAM3 is
the identity map on fractions**: `{0.05, 0.10, 0.15, 0.20}` means the same geometry on both
sides. Do not "convert".

What breaks if the two are conflated:

1. **Absolute `target_sizes` are anchored to 640.** `target_fractions()` divides by a literal
   `640.0` (`training.py:267`, `gui/models.py:260`). SAM3's input is **1008**
   (`dataset_build.py:105`, `datapoints.py` `RES`). Feeding SAM3 a legacy pixel list through
   the YOLO fallback shifts every scale by 1008/640 = **1.575x** — the same class of silent
   geometry drift as the 0.055 incident, and it would look like a plausible number in every
   artifact. **SAM3 must consume FRACTIONS only, never a pixel list, and must never divide by
   an imgsz it did not itself use.**
2. **The identity assumes resize-to-input.** Edge tiles are not square (`plan_tiles` flushes
   the last tile to the edge, `slice_geometry.py:41-45`, and the builder further clips to the
   frame, `dataset_build.py:264-267`), and `datapoints.py:167-168` stretches anisotropically.
   For those tiles the apparent size differs per axis and `object_tile_fraction` no longer
   equals the realised `target_size_fraction`. This is R3b and it is why Task 7 counts them.
3. **The two knobs are on different UI surfaces.** SAM3's is a single spin box
   (`gui/panels/sam3_training_panel.py:472-476,560,604`); the YOLO set lives in
   `gui/panels/slice_settings_widget.py:526`, which already collapses the set to
   `median(fractions)` for its own scalar (`:517`). A shared widget is *not* in scope; a
   shared *vocabulary* is (Task 2).

---

## Task 1 — Freeze the before-gate: a byte-identical SAM3 dataset golden

Build a committed characterization golden for the **current single-scale** builder, before any
production code changes.

- Deterministic tiny corpus (a handful of synthetic frames with polygon labels, generated from
  a committed seed/script), built through `build_sam3_coco_dataset` at today's defaults.
- Golden = the tree hash of the output: sorted relative paths + SHA-256 of every tile image and
  of `train/valid` COCO json (with `created_at` and absolute `source` normalised out) +
  `build_manifest.json` minus the same volatile keys.
- Commit the golden under `tests/data/sam3_multiscale_golden/` with a short `ATTRIBUTION.md`
  recording the commit it was produced at.

**Verification:** a new `tests/test_sam3_multiscale_gate.py::test_single_scale_golden` passes at
this commit and is re-run unchanged after every later task. This is R1's own mitigation and it
is what makes "multi-scale did X" a **measured** claim rather than an asserted one.

**What silently goes wrong without it:** a scale-set refactor that changes tile ordering,
naming, or the dedup rule produces a *plausible* dataset that trains differently, and nothing
in the tree would notice. The hash notices.

## Task 2 — `resolve_scales` in the shared layer (pure, no behaviour change)

Add to `utils/slice_geometry.py` (or a new Qt-free `utils/tiling_plan.py` beside it — pick one
and state why in the PR; the module must import nothing from `training/` or `core/`):

```
resolve_scales(*, geometry_mode, imgsz, reference_body_px, fractions, slice_width,
               slice_height) -> list[tuple[int, int]]
```

carrying `_tile_sizes_for_params`'s exact semantics: clamp each fraction into `[0.01, 0.9]` via
`tile_size_for_mode`, dedupe **preserving first-seen order**, and fall back to the single
`geometry_mode` size when `fractions` is empty or `reference_body_px <= 0`.

Then make `training/sliced_dataset.py:_tile_sizes_for_params` a thin caller of it (converting
its pixel `target_sizes` to fractions at the call site, as it already does at `:185`). **Do not
change the SAM3 builder in this task.**

**Verification:** existing `tests/test_sliced_dataset.py` + `tests/test_slice_geometry*.py` pass
**unmodified**; a new property test asserts `resolve_scales` output equals the old
`_tile_sizes_for_params` output over a grid of (fractions, ref_px, imgsz) including the
degenerate cases (empty list, ref 0, duplicate fractions that collapse after rounding,
fractions outside the clamp).

**Silent failure it prevents:** a second copy of the tile-size formula. The spec records five
copies already; a sixth is how the two sides drift apart again.

## Task 3 — SAM3 builds a scale SET (behaviour-preserving at one scale)

In `training/contracts.py`, add to `Sam3LoraParams`:

- `object_tile_fractions: tuple[float, ...] = ()` — **empty means "use the scalar
  `object_tile_fraction`"**, so today's default is bit-for-bit unchanged (D6/D16 stay open).
- `full_frame_mix: bool = False` — SAM3 has no full-frame arm today; default-off keeps that
  true. (Spec §4.3 refuses to assert full frames help; the switch must be explicit and
  recorded, never an unstated default.)
- Validation in `detectkit/config/training.py` beside the existing
  `object_tile_fraction <= 0.0` check (`:738`): every entry in `(0, 1]`, list length capped
  (mirror `dataset_preparation_sidecar.py:146-151`'s `MAX_CLASSES` cap).

In `dataset_build.py`: replace the single `tile_size_for_mode` call (`:390-397`) with
`resolve_scales(...)`; loop `_tile_frame` over the resolved sizes inside `_build_split`
(`:527-534`); emit the **full-frame arm** as its own pass when `full_frame_mix` (mirroring
`sliced_dataset.py:329-342`, i.e. the whole image with frame-space polygons, not
`plan_tiles(full_frame=True)`).

**Fork on set-emptiness — this is mandatory, not stylistic.** With `object_tile_fractions=()`
the builder takes literally today's path: legacy `{stem}_tile{idx:03d}.jpg` names, no
`scale_group` field, identical COCO records. Only when a set is supplied do tile names become
`{stem}_t{W}x{H}_{idx:04d}.jpg` / `{stem}_full.jpg` (matching `sliced_dataset.py:318`) and does
each COCO `images` record gain — per D19 — a `scale_group` and `tile_px: [w,h]` field, with the
filename token demoted to a legacy fallback. Without the fork, Task 1's gate is unpassable: the
name appears in the COCO `file_name` too, so a rename changes the tree hash twice over.

**Verification:**
- Task 1's golden **still passes byte-identically** with `object_tile_fractions=()`. If it does
  not, the loop changed something and the task is not done.
- New test: a 2-fraction build produces tiles from both sizes, with per-scale tile counts equal
  to `len(plan_tiles(...).tiles)` per scale, no stem collisions, and every COCO image record
  carrying a `scale_group` that matches its filename token.
- New test: duplicate fractions that resolve to the same tile size produce **one** copy of
  those tiles (the dedup in Task 2), not two — a duplicated-supervision bug that no downstream
  counter would ever reveal.
- New test: an existing image-stem collision guard (`dataset_build.py:355-363`) still fires.

## Task 4 — Stamp the SCALE SET, not a collapsed scalar (§3.5)

**Producers of the collapsed value today:**
- `dataset_build.py:423,614,680` — `tile_px: [w, h]` in the log, the manifest, and the return.
- `publish_worker.py:267-269` — sidecar `train_tile_px = build_manifest["tile_px"]`,
  `object_tile_fraction = build_manifest["object_tile_fraction"]`. Note its `build_manifest`
  arg is **not** a re-read of `build_manifest.json`: it is the parent's request payload
  (`publish.py:407-412` → `publish_cli.py:55`), which `publish.py:374-401` has already
  **collapsed to a scalar**. Verified on a real sidecar: `train_tile_px: 971` (scalar),
  `object_tile_fraction: 0.1`. The comments at `geometry_drift.py:126-129` and
  `dataset_build.py:446-447` claiming the sidecar carries a `[w,h]` pair are therefore
  **stale**; fix them in this task.
- `sliced_dataset.py:410-432` — the YOLO manifest (`target_sizes` + `imgsz`).

**Consumers, and exactly what each expects:**
- `publish.py:374-400` — **HARD RAISES** on a `tile_px` that is not a number or a 2-element
  numeric pair, and raises again if the pair is non-square. A list-of-sizes stamp makes
  **every SAM3 publish fail**. This is the single hardest break and must be handled here.
- `core/inference/geometry_drift.py:_as_geometry_value` (`:130-155`) — parses a scalar or a 1/2
  element sequence; **any longer sequence returns `None`, i.e. `NO_STAMPED`**. So a scale-set
  stamp does not error, it **silently disables the drift guard** — the exact failure mode the
  module was written to end. It handles `float | tuple[float,float]` element-wise; it **cannot
  carry a set** without a new comparison rule.
  Callers: `sliced_dataset.py:274-287`, `dataset_build.py:436-455`.
- `core/inference/slice_meta.py:_training_values` (`:286-317`) — already multi-scale-aware for
  YOLO: stamps the whole `target_sizes` list and derives the prefill as
  `median(target_sizes)/imgsz`. This is the shape to reuse.
- `direct_calibration_grid.py:67` — `base_fraction = training_geometry["object_tile_fraction"]
  or 0.15`, then multiplicative `FRACTION_STEPS`. A collapsed median here is acceptable **only
  if it is named as a prefill**.
- GUI prefill: `semantic_escalation_dialog.py:635-643`, `trackerkit .../detection_panel.py:2876,2921`.

**The change:**
1. Manifest gains `tile_px_set: [[w,h], ...]`, `object_tile_fractions: [...]`,
   `full_frame_mix`, `scale_range_px: [min, max]`, and **per-scale realised counts**
   (tiles, annotations, `downgraded_tiles`, `fragment_only_tiles`, non-square tiles — Task 7).
   `tile_px` / `object_tile_fraction` are **retained** as the single-scale values when the set
   has one entry, and for a multi-scale build are written as the **median**, under the new
   explicit key `prefill_object_tile_fraction` / `prefill_tile_px` — the legacy scalar keys are
   then **omitted**, not filled with the median.
2. **Items 1 and 2 must land in the SAME commit.** Omitting the legacy scalar keys before the
   publish guard understands the new ones leaves an intermediate state in which **every**
   multi-scale publish raises.
   `publish.py`'s guard is extended to accept the new keys and to raise a *specific* message
   for a multi-scale artifact whose consumer surface has not been updated — never to silently
   halve or median-collapse.
3. `geometry_drift.py` gains an explicit set-valued comparison (`MISMATCH` when the sets
   differ, `MATCH` when equal, and a distinct verdict for "stamped a set, serving a scalar
   inside the set" — that is *not* a mismatch, and it is *not* a match either).
4. **Which sidecar surface carries it is D2 and is a USER DECISION.** Either SAM3 artifacts
   adopt the v2 `.slice_meta.json` surface, or `.sam3_meta.json` gains an embedded v2-shaped
   `training_geometry` block. **Do not pick one; implement behind whichever the user rules for,
   and stop and ask if no ruling exists.** Note the hazard either way: the SAM3 serving path
   **refuses to serve on a malformed sidecar**, so a shape change without a back-compat reader
   is a hard outage for already-published models, not a degradation.

**Verification:** a single-scale build's sidecar is **unchanged** (round-trip test against a
committed real-sidecar fixture); a multi-scale build's sidecar carries the full set and
**no** bare `object_tile_fraction`; `publish_sam3_model` succeeds for both; a drift-guard test
asserts a set-vs-set match, a set-vs-different-set mismatch, and — the regression that matters
— that a 4-element stamp **never** reads as `NO_STAMPED`.

## Task 5 — Scale-grouped batching for SAM3, or an explicit refusal

Promote the three framework-agnostic functions out of `ultralytics_scale_balance.py`
(`scale_group_for_path`, `scale_group_weights`, `ScaleGroupedBatchSampler`) into the shared
layer, leaving the Ultralytics installer as one adapter (spec §3.3's table). Then either:

(a) add a small SAM3 adapter over `dataloader.collate_epoch_batches` (`dataloader.py:322-332`,
which shuffles descriptor indices and flushes every `batch_size` **queries**, not tiles —
`:297-320`) that groups by the COCO record's `scale_group`, **preferred**, because grouping then
reads a field rather than parsing a filename; or
(b) explicitly decline grouping for SAM3 in this change and record that decision in the
manifest as `scale_grouped_batching: false`.

Whichever ships, the run must write a **realised** stamp in the shape `4baab88c` established
(`REALISED_STAMP_FILENAME`, requested-vs-applied blocks) so a grouped and an ungrouped SAM3 run
are never confusable afterwards.

**The DDP answer, so nobody ports the hazard by reflex:** SAM3 training is **single-process,
single-GPU by construction** — a grep for
`DataLoader|DistributedSampler|torch.distributed|init_process_group|spawn|world_size|local_rank|DDP`
across `training/sam3_lora/` returns **zero matches**; `train.py` launches exactly one sidecar
child (probe `:303`, train `:968`) via `python -m ...sam3_lora.cli`, with no `torchrun` and no
rank plumbing; and `train.py:402-411` pins one device by UUID and loudly narrows a multi-GPU
device string. `install_sahi_scale_balance_and_grouped_sampling` has exactly one caller in the
tree, `training/ultralytics_entrypoint.py:42-51`, and is **not** imported by any SAM3 module. So SAM3 **does not
inherit R7's DDP hazard today**. It inherits the *rule*: if SAM3 ever grows multi-device
training, an ungrouped fallback must raise, not warn. Encode that as an assertion now
(a test that fails the moment `sam3_lora/` gains a distributed launch), not as a comment.

**A SAM3-specific reason grouping matters more here than on the YOLO side:** the SAM3 collator
pads to the batch's max instance count, and marginal VRAM per item is superlinear
(1.31 → 1.40 → 1.88 GiB). Coarse tiles hold many more instances than fine ones, so a
scale-*heterogeneous* batch is padded to the coarse tile's instance count — multi-scale
without grouping raises peak VRAM for reasons that have nothing to do with tile pixels.
**This is a hypothesis from the batch-scaling measurement, not a measurement of mixed batches;
Task 8 measures it.**

**Verification:** grouping unit tests on synthetic path/record lists (including the R8 case —
a stem that fails the regex must be *counted and warned*, never silently weighted 1.0);
a test that the realised stamp is written on both arms.

## Task 6 — Autobatch fingerprint must include the whole scale set

`sam3_workload_fingerprint` (`autobatch.py:428-512`) hashes `object_tile_fraction=` into the
task payload (`:492`) but **never the resolved tile pixels** — `slice_width`/`slice_height` are
hashed only in `geometry_mode == "custom"`, otherwise the literal `"mode_derived"` is used
(`:483-489`). The only other scale-sensitive term is the density hash
(`_dataset_density_hash`, `:377-397`: max/p95 instances per tile, negatives). With a scale set
the key becomes **wrong**: a list f-string-interpolated at `:492` keys distinctly by accident of
its `repr` (order-sensitive, type-unvalidated), and two builds with different sets but the same
scalar collide — a stale VRAM probe reused across geometries. Given the measured history here — every
short probe *under*-reported the full-run peak (7.72 → 12.99 GiB) and the flatness-then-jump
shape defeats any early-stopping rule — a wrongly-reused probe is an OOM, not an inefficiency.

Include the resolved `tile_px_set` (not the fractions — the tiles are what the GPU sees) and
`full_frame_mix` in the fingerprint. Full frames are the largest images in the set and will
dominate the peak; if the probe samples the dataset, it must sample the **coarsest** scale and
the full-frame arm, not a uniform sample.

**Verification:** two specs differing only in `object_tile_fractions` produce different
fingerprints; a spec with an unchanged single scale produces the **same** fingerprint as before
this change (no gratuitous cache invalidation for existing runs).

## Task 7 — Per-scale counters, including the non-square count (R2 + R3b)

`SplitCounts` (`dataset_build.py:~51-63`) becomes per-scale: for each scale, `tiles`,
`annotations`, `fragment_annotations`, `downgraded_tiles`, `fragment_only_tiles`, **and a new
`non_square_tiles`** (tiles whose clipped crop is not `W == H`, i.e. those anisotropically
stretched to 1008x1008 by `datapoints.py:167-168`). Aggregate totals are retained for
back-compat. Log the table **before** any GPU time is spent, in the shape
`log_effective_geometry` already uses (`dataset_build.py:415-434`).

Why this is not optional: `retained_frac` is measured against the instance's **full frame-space
area** (`:277-280`), so the *test* composes across scales — but the *rate* does not. Fine tiles
cross far more seams, so a multi-scale build downgrades a much larger share of instances at the
fine end, removing precision pressure exactly where the eval spec decides extras/frame. Adding
scales likewise multiplies the edge-tile population, so a 4-scale dataset carries
proportionally more distorted supervision than a 1-scale one — **with no counter reporting it
today**. Neither effect appears in the loss.

**Verification:** a synthetic frame whose dimensions are not a multiple of the tile step
produces a known, asserted `non_square_tiles` count per scale; a frame with an instance
straddling a seam produces the expected `downgraded_tiles` at the fine scale and zero at the
coarse one.

## Task 8 — The measured after-gate (mehek, CUDA)

Only after Tasks 1-7. On `mehek`, with the SAM3 sidecar env:

1. **Dataset arm (no GPU needed, run anywhere):** build the same real corpus at (i) today's
   single scale, (ii) a 2-scale set, (iii) a 4-scale set. Record actual tile counts, per-scale
   fragment/non-square counts, and on-disk size. This is the *measured* dataset multiplier —
   the estimate below is arithmetic and must not be reported as a result.
2. **VRAM arm:** run the autobatch probe on each, with `expandable_segments` on, recording
   **reserved and allocated** (a default-allocator probe gets the *curvature* wrong, not just
   the level). Compare grouped vs ungrouped batches at the same batch size to test Task 5's
   padding hypothesis.
3. **Wall-clock arm:** one short run (fixed small epoch count) per arm to get s/step and
   confirm steps/epoch scales as `tiles / batch` (`cli.py:910,940`).

Commit the numbers as `docs/superpowers/specs/`-side evidence with the arms, seeds, and the
commit sha. **Attribution must be measured, not asserted:** report the dataset multiplier, the
VRAM delta, and the wall-clock delta as three separate numbers, because they do not move
together.

### Task 8 — Results (measured on `firebrat`, RTX 4090 24 GB, one corpus)

**Status: COMPLETE.** Ran on GPU (`firebrat`, CUDA, sidecar `sam3-lora`), three arms — A
single-scale, B multi-scale `{0.055, 0.11}` grouped, C multi-scale `{0.055, 0.11}`
ungrouped — 1 epoch each, corpus `20260827_222150-sam3-ant` (single YOLO-seg source, class
`ant`), seed 42, `imgsz=1008`. **No OOM on any arm.** Full detail, per-scale tables, and raw
artifact paths: `task-8-gate-report.md` / `task-8-gate-results.json` in the
`2026-09-06-multiscale-sam3-training` SDD ledger (not tracked in this repo — ledger is
gitignored `.superpowers/`).

**Every number below is corpus-relative** — one 78-image, one-species corpus, one card. Do
not treat any ratio here as a general constant.

- **Dataset multiplier: measured 6.44x tiles, not the 4.1x predicted below.** The 4.1x
  arithmetic assumed a 4512x4512 frame at reference body 97 px (tiles 1764/882 px); this
  corpus actually resolves to 1862/931 px tiles, so the fine scale fans out harder than the
  estimate. The prediction's *direction* held (fine scale dominates the multiplier); its
  *magnitude* was 57% low.
- **VRAM: roughly flat, as predicted.** Resolved batch 4 on all arms (13.2-13.3 GiB of 22.0
  GiB free). Training-loop peak reserved: A 11.99 GiB, B (grouped) 12.20 GiB, C (ungrouped)
  12.63 GiB. Task 5's padding hypothesis is supported but the effect is small: **ungrouped
  costs +0.43 GiB (+3.5%) over grouped** at the same batch size and dataset.
- **Wall-clock: linear in tiles, as predicted.** Epoch ratio multi/single = 6.51x against a
  measured tile ratio of 6.44x. **Grouping is wall-clock-neutral**: grouped vs ungrouped
  epoch times differ by 0.06%.
- **Verified on hardware, not assumed:** the autobatch fingerprint genuinely differs between
  single- and multi-scale builds (both the density hash and the new `tile_px_set` suffix);
  the realised requested-vs-applied stamp landed correctly on all three arms; and
  `expandable_segments:True` was confirmed live via `/proc/<pid>/environ` of the sidecar
  child, not just assumed from the parent's env.
- **`non_square_tiles = 0` on this corpus** (both single- and multi-scale builds) — this
  corpus's frames tile evenly at both resolved sizes. The R2/R3b counter is therefore
  **unexercised here, not disproven**; its non-zero path is covered by the committed
  synthetic unit test, not by this gate.
- **The 4-scale (18.4x) arm was deliberately not run** — the pre-flight ruling called it
  out of the GPU window available for this gate. It remains a gap in this evidence, not an
  accidental omission.

### Cost estimate — arithmetic, superseded by Task 8's measurement above for the 2-scale row

(Kept for the reasoning and the un-run 4-scale/18.4x extrapolation, which Task 8 did not
measure. For the 2-scale row, use the measured 6.44x above, not the 4.1x below.)

Tiles/frame ∝ (frame/tile)² and tile = ref/frac, so tiles ∝ frac² asymptotically; edge-flushing
(`slice_geometry.py:41-45`) inflates that at coarse scales. Computed with the repo's own
planner for the measured run's geometry (4512x4512 frame, ref 97 px, overlap 0.25):

| scale set | tile px | tiles/frame | vs today |
|---|---|---|---|
| `{0.055}` (today) | 1764 | 16 | 1.0x |
| `{0.055, 0.11}` | 1764, 882 | 65 | **4.1x** |
| `{0.05,0.10,0.15,0.20}` (the YOLO set) | 1940, 970, 647, 485 | 295 | **18.4x** |

Two amplifiers: SAM3 keeps **every** empty tile (`keep_empty_tiles=True`,
`contracts.py:266`) where YOLO samples negatives at 0.15, so SAM3's multiplier is the raw tile
multiplier; and `MAX_TILES_PER_FRAME = 4096` (`slice_geometry.py:20`) is checked **per scale** —
`plan_tiles` raises, and the YOLO builder swallows that with `except ValueError: continue`
(`sliced_dataset.py:307-308`), which for SAM3 would silently drop an entire scale. **Make SAM3
refuse loudly instead.**

Consequences, against the measured single-scale baseline (auto-batch 8, 18.9 of 47.1 GiB,
~10 epochs):
- **VRAM: roughly flat.** Every tile is stretched to 1008x1008 regardless of source tile size,
  so per-item cost does not depend on the scale. The two things that *can* move it are the
  full-frame arm (also 1008 after stretch — so also flat) and batch instance-count padding
  (Task 5's hypothesis). Expect batch 8 to still fit; **verify, do not assume**.
- **Wall-clock: linear in tiles.** `query_count = tiles x (1 + num_negatives)`
  (`dataloader.py:293-295`) → `n_batches` (`cli.py:908`) → `steps_per_epoch` (`:909`) →
  `total_steps` (`:910`), with `params.epochs` a fixed hyperparameter (`:944`). So an 18.4x
  dataset is an ~18.4x longer run at the same epoch count — days, not hours. Two riders:
  `warmup_steps = min(50, total_steps // 4)` (`:911`) pins at 50, so the warmup *fraction*
  shrinks by the same factor and the LR schedule is not the same schedule; and per-epoch
  validation + checkpoint writes (`:1022-1024`, `:1112-1113`) scale too. **This is the single biggest practical finding in this plan.**
  Mitigations to choose between explicitly (not silently): a smaller/coarser scale set
  (`{0.055, 0.11}` at 4.1x is the pragmatic first port), a reduced epoch count at fixed
  *step* budget, or per-scale negative-tile sampling ported from YOLO's
  `negative_tile_fraction`. Do not pick one without measuring.

## Out of scope — named so no implementer wanders in

- **D16** (should SAM3 adopt 0.10?). This plan ships a configurable set and changes **no**
  default fraction. The 2x2 ablation decides it later.
- **D6 / D17** (multi-scale default-ON for SAM3). Ships opt-in; the YOLO precedent for
  default-on is noted, not followed.
- **D1** (reference-body estimator: SAM3's mean-of-middle-medians vs YOLO's global median).
  Do not touch `dataset_build.py:369-388` — it changes the tile size of every existing build.
- **D3/D4/D5** (seam-fragment policy, empty-tile policy, overlap defaults), the corpus-derived
  scale-set *resolver* of §3.6 (this plan takes a user-supplied set), the unified
  `TilingContract`, the calibration harnesses, and the full-frame *efficacy* question (§4.3).
- Any change to `tile_size_for_mode`, and any GUI redesign beyond the minimum needed to enter a
  scale set in `sam3_training_panel.py`.

## What could NOT be determined from the code, and must be measured or decided

- **Whether multi-scale actually improves SAM3.** No evidence in the tree either way. The user's
  "more robust" is the hypothesis this plan makes testable; it is not established.
- **Whether full frames help SAM3.** Spec §4.3 refuses to assert it: at 1008 input, a 97-px
  animal in a 4512-px frame is ~21 px. Ships off by default, per-project switch.
- **Whether the collator's padding actually rises for scale-heterogeneous batches.** Inferred
  from the batch-scaling measurement, not measured on mixed batches. Task 8 arm 2.
- **D2 — which sidecar surface carries the scale set.** A user decision; the plan is written to
  accept either.
- **The right scale set for this corpus.** Deliberately not proposed. Three fractions exist in
  the tree for three different reasons and none of them was calibrated for SAM3 training.
