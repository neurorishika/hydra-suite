# Unified SAHI training geometry: multi-scale SAM3, calibrated scale sets, one interface

**Status:** design proposal, pending review. No implementation.
**Repo:** `/Users/neurorishika/Projects/Rockefeller/Kronauer/multi-animal-tracker` @ `main` (`097408af`), read-only audit.
**Motivating evidence:** `docs/superpowers/specs/2026-09-06-sam3-training-run-and-evaluation.md` §10, §11c.
**Related, already merged:** `docs/superpowers/specs/done/2026-09-01-detectkit-sahi-calibration-profiles-design.md`,
`docs/superpowers/specs/done/2026-09-05-calibration-profile-headless-parity-{audit,design}.md`.

Every claim below is tagged **[V]** verified by reading the cited line, or **[I]** inferred.
All paths are under `src/hydra_suite/` unless noted.

---

## 0. Executive summary — three findings that reshape the brief

**F-A. The tile planner is ALREADY shared. The divergence is one layer up.** Both trainers
call `utils/slice_geometry.py`: YOLO at `training/sliced_dataset.py:21-26`, SAM3 at
`training/sam3_lora/dataset_build.py:26-30`, and tiled *inference* at
`core/inference/semantic/tiling.py:20` and `core/inference/stages/slicing.py:10`. **[V]**
So "unify the tiling geometry interface" is not a rewrite of tile planning. It is a
unification of the **parameter contract**, the **scale-set policy**, and the **stamping /
read-back contract** that sit on top of one already-shared planner.

**F-B. YOLO sliced training is ALREADY multi-scale, with full-frame mixing. SAM3 is the
single-scale outlier.** `training/sliced_dataset.py:159-193` (`_tile_sizes_for_params`)
fans out one square tile size per entry of `params.target_sizes`, converting each target
*apparent* object size to a fraction via `target / imgsz`; `:248-260` additionally emits an
untiled full frame per image when `full_frame_mix` is true; defaults
`target_sizes = [200.0, 300.0, 400.0]`, `full_frame_mix = True` at
`training/sliced_dataset.py:105-106` and `detectkit/config/training.py:169-170`. **[V]**
SAM3 resolves exactly ONE `(tile_w, tile_h)` for the whole dataset at
`training/sam3_lora/dataset_build.py:373-380`. **[V]**
**The user's multi-scale request is therefore mostly a port, not an invention.**

**F-C. The task brief's premise that "the CLI ignores calibration profiles" is STALE — it
was fixed.** `_slice_profile_overlay` (`trackerkit/engine_params.py:363-436`) is applied in
the shared builder at `:968-986`; `apply_sahi_profile_override`
(`trackerkit/cli_config.py:143-170`) and the `--sahi-profile` flag (`trackerkit/app.py:112`,
`:253`; `trackerkit/cli.py:13, 28, 66-73`) close it, and both parity docs are in
`specs/done/`. **[V]** This matters: the profile-overlay + sidecar-v2 + `--sahi-profile`
path is **the plumbing this design should ride**, not a gap to design around.

Two gaps from that audit do appear to remain and threaten any stamping design:
`training/model_publish.py:874-879` still gates the whole `.slice_meta.json` sidecar
copy/merge on `slice_geometry` being truthy **[V]**, so a publish that passes no training
geometry silently drops all calibration profiles.

---

## Part 1 — What exists on the YOLO / SAHI side

### 1.1 The shared planner — `utils/slice_geometry.py`

| Symbol | Line | Contract |
|---|---|---|
| `MAX_TILES_PER_FRAME = 4096` | :20 | Hard ceiling; `plan_tiles` raises `ValueError` above it **[V]** |
| `SlicePlan` | :24 | `tiles`, `full_frame`, `slice_wh`, `frame_wh` **[V]** |
| `get_slice_bboxes` / `_axis_starts` | :37-81 | Axis start positions, step `= size*(1-overlap)` **[V]** |
| `tiles_overlap` | :83-112 | Overlap detection for merge decisions **[V]** |
| `tile_size_for_mode(...)` | :113-137 | The one geometry-mode resolver. `custom` -> `slice_w/h` or `imgsz`; `auto_object` with `reference_body_px>0` -> `round(reference_body_px / clamp(fraction, 0.01, 0.9))`, clamped to `[64, 4096]`, always **square**; else `auto_model` -> `(imgsz, imgsz)` **[V]** |
| `plan_tiles(frame_hw, w, h, ov_w, ov_h, *, full_frame, roi_mask)` | :139-198 | Grid + ROI gating (mask-shape mismatch downgrades to no gating with a warning; an all-empty gate falls back to keeping all tiles) **[V]** |
| `polygon_area`, `clip_polygon_to_tile` | :200-... | Sutherland–Hodgman clip used by both builders **[V]** |

**Note:** `tile_size_for_mode` is documented as reproducing `stages/slicing.py:_tile_size`
"exactly (semantics are frozen)" (:122-124) **[V]**. Any change here changes inference.

### 1.2 The YOLO sliced-training builder — `training/sliced_dataset.py`

- `SliceBuildParams` (:95-107) **[V]**, verbatim defaults: `geometry_mode="auto_object"`,
  `imgsz=640`, `object_tile_fraction=0.15`, `slice_width=0`, `slice_height=0`, `overlap=0.2`,
  `min_area_ratio=0.1`, `negative_tile_fraction=0.15`,
  `target_sizes=[200.0, 300.0, 400.0]`, `full_frame_mix=True`, `reference_body_px=0.0`.
- `measure_reference_body_px` = **median** `minAreaRect` major axis over a frame's labels
  (:57-...); `object_major_axes_px` (:42-55) **[V]**. Dataset-level reference is the mean of
  per-frame values accumulated in `all_majors` -> `np.median` at :263-267 **[V]**.
- Multi-scale fan-out: `_tile_sizes_for_params` (:159-193). Only active when
  `geometry_mode == "auto_object"` AND `reference_body_px > 0` AND `target_sizes` non-empty;
  otherwise single size. Dedupes identical `(w,h)` **[V]**.
- Per-tile emission (:230-247): drops a clipped instance below `min_area_ratio` of its
  original area (`_tile_one_image` :114-137) — **[V]** note this is a *drop*, not a downgrade.
  Negative (instance-free) tiles are sampled at `negative_tile_fraction` (:235-236) **[V]**.
  Tile stems encode the scale: `f"{stem}_t{tile_w}x{tile_h}_{ti:04d}"` (:237) **[V]**.
- Full-frame mixing (:248-260): whole image written with stem `f"{stem}_full"` **[V]**.
- Manifest (:270-278) records `measured_reference_body_px` and a `slice_geometry` block from
  `_slice_geometry_manifest` (:329-347) carrying `geometry_mode, imgsz,
  object_tile_fraction, slice_width, slice_height, overlap, min_area_ratio,
  negative_tile_fraction, target_sizes, full_frame_mix, reference_body_px` **[V]**.
- That manifest block reaches publish via `training/service.py:136-151, 222`
  (`_slice_geometry_for_publish`) -> `model_publish.py:805, 874-890, 941-942` **[V]**.

**There is no per-scale instance balancing.** Every scale contributes every tile it produces,
so a small tile size contributes quadratically more tiles (and more instance copies) than a
large one. **[V by absence — no weighting code in `build_sliced_obb_dataset`]**

### 1.3 The sidecar — `core/inference/slice_meta.py` (v2)

- Document shape: `training_geometry()` promoting flat v1 payloads (:50-53); `profiles[]`
  with `id/name/settings`, plus `primary_profile_id`; `normalized_slice_meta` never invents a
  primary (:66-73) **[V, via the merged audit's field table, lines re-checked at :226-248,
  :287-317, :321-382]**.
- **`_training_values` already collapses a multi-scale set to one number** (:287-317): if
  `target_sizes` is non-empty and `imgsz > 0`, the derived prefill fraction is
  `median(target_sizes) / imgsz`, clamped to `[0.01, 0.9]` **[V]**. This is the existing,
  shipped answer to "what does a multi-scale model prefill?" — see Part 3.5.
- `profile_evidence_state` (:226-248) SHA-256s the checkpoint for a staleness label **[V]**.
- `resolve_slice_profile_values` / `slice_meta_to_panel_values` (:250-268, :321-382) give the
  3-rung ladder requested-id -> primary -> `__training__`, with a `resolution` field
  explaining which rung fired **[V]**.

### 1.4 Calibration — two independent flows already exist

**(a) Direct-detector (YOLO/OBB) calibration.** `core/inference/direct_calibration_grid.py`:
`FRACTION_STEPS = (0.75, 1.0, 1.5)`, `OVERLAP_STEPS = (0.1, 0.2, 0.3)`,
`DEFAULT_MAX_TOTAL_TILES = 20000` (:18-20) **[V]**. `build_candidate_grid` (:59-...) emits a
**full-frame `enabled=False` baseline first** (:72-83) then fraction x overlap candidates
**relative to the model's stamped training geometry** (`base_fraction`, `base_overlap`,
`body_px` read from `training_geometry`, :67-72) **[V]**. Candidates carry `SLICE_*` PARAMS,
never a hand-built `SliceConfig` (:34-48, and the module docstring's explicit rationale
:1-7) **[V]** — this is the property that makes a measured point expressible as TrackerKit
settings, and it is exactly the property a training-side reuse must preserve.
Sweep/objective in `direct_calibration_sweep.py`; job wrapper `detectkit/jobs/direct_calibration.py`;
UI `detectkit/gui/dialogs/direct_calibration_{wizard,results}.py`; the results dialog writes
profile settings at `direct_calibration_results.py:481-507` **[V, via merged audit]**.

**(b) Semantic (SAM3-inference) calibration.** `core/inference/semantic/tiling.py:26-38`:
`SEMANTIC_TILE_FRACTION_SEED = 0.05` with an explicit in-code disclaimer that it "was
back-derived from a single measured-good configuration (1504 px tile at body 80 px) on a
single dataset, and has no independent grounding ... It exists to prefill the dialog when
the user skips calibration"; `TILE_FRACTION_GRID = (0.03, 0.05, 0.10, None)` where **`None`
means one full-frame pass** — "the right answer on a rig where animals are already large at
native resolution, where tiling HURTS" **[V]**. `DEFAULT_OVERLAP = 0.5`,
`DEFAULT_SEAM_MARGIN_PX = 4`, `DEFAULT_MERGE_IOU = 0.5` (:39-41) **[V]**.
`semantic/calibration.py:1-24` fits **tile fraction AND confidence** against the user's own
labelled frames, objective = the missed-vs-to-delete frontier (recall-first), matching gated
by a per-project fitted `shape_prior` area band **[V]**.

**This (b) grid is already a per-project, calibration-chosen scale sweep including a
full-frame arm — for SAM3 inference. It is the single closest existing fit to what the user
wants for SAM3 *training*.**

### 1.5 Profile -> run path (post-parity-fix)

GUI: `detection_panel.py:2643-2765` applies `slice_meta_to_panel_values`. Headless: config
carries `slice_profile_id` + `slice_profile_settings`; `engine_params.py:968-986` overlays
them onto the `SLICE_*` advanced keys, with a hard error for an unresolvable explicit
`--sahi-profile` (`cli_config.py:143-170`) and a log warning when a named profile is missing
from the sidecar (`engine_params.py:402-410`) or when confidence differs from the measured
point (:424-431) **[V]**. `build_engine_params` is the single Qt-free builder both GUI and
CLI share, per CLAUDE.md.

---

## Part 2 — What SAM3 does instead, and the divergence table

### 2.1 SAM3's contract and builder

- **The 1008 resize happens at datapoint construction, not in the builder.** Tiles are written
  to disk at native crop size; `sam3_lora/datapoints.py:22` (`RES = PREDICTOR_IMGSZ`) and
  `:165-174` resize each tile to `RES x RES` with `PILImage.BILINEAR` and scale the polygons
  by `[RES/w, RES/h]` in the same step. **This is a non-aspect-preserving stretch, not a
  letterbox**, so partial edge tiles are anisotropically distorted — a fact that matters for
  multi-scale (more scales means more edge tiles) and is not currently reported. **[V]**
- **Contract lives in `training/contracts.py`, NOT in `sam3_lora/`.** Tiling block at
  `training/contracts.py:248-254`, verbatim: comment `# Tiling, mirroring the SAHI sliced-training
  knobs.`, then `geometry_mode: str = "auto_object"`, `object_tile_fraction: float = 0.055`,
  `slice_width: int = 0`, `slice_height: int = 0`, `tile_overlap: float = 0.25`,
  `keep_empty_tiles: bool = True` **[V]**. (The eval spec cites `:250` for the fraction; the
  block reads the same.)
- `dataset_build.py:373-380`: **one** `tile_size_for_mode(...)` call with `imgsz=_SAM3_IMGSZ`,
  where `_SAM3_IMGSZ = 1008` is documented as "SAM3's native training resolution ... used
  only as the `imgsz` fallback for auto_model / auto_object geometry modes" (:94-97) **[V]**.
- Reference body px: measured with the **same** `measure_reference_body_px` imported from the
  YOLO builder (`dataset_build.py:46`, used at :336) — but aggregated differently. SAM3 takes
  a SQL **median over per-frame medians** (`:353-369`, even counts averaging the two middle
  rows), while YOLO pools every object's major axis across the whole corpus and takes one
  global median (`sliced_dataset.py:219, 263-267`). Different estimator over a different
  population; they coincide only for a perfectly uniform corpus. **[V]**
- `_tile_frame` (:239-278): `plan_tiles((h, w), tile_w, tile_h, overlap, overlap)`; clipped
  instances with `retained_frac < MIN_RETAINED_AREA_FRAC` are **kept as `is_crowd=1`**, not
  dropped (:270-275), and the tile is downgraded from exhaustive downstream
  (`datapoints.select_output_objects`, referenced `datapoints.py:57`, `dataloader.py:121`)
  **[V]**. `MIN_RETAINED_AREA_FRAC = 0.25` at `dataset_build.py:92`, with a long rationale at
  :63-91 explaining it was lowered from 0.5 and that downgrading REMOVES precision pressure
  from seam-adjacent tiles **[V]**.
- **No validation.** `Sam3LoraParams` is `@dataclass(slots=True)` with no `__post_init__`
  (`contracts.py:201`); the only external checks are `detectkit/config/training.py:710-713`
  (`tile_overlap` in `[0,1)`, `object_tile_fraction > 0`). **`geometry_mode` is never checked
  against `{auto_object, auto_model, custom}` anywhere**, and an unknown string falls silently
  through `tile_size_for_mode`'s final branch to a 1008x1008 tile. YOLO's inference-side
  clamp of `slice_width/height` to 8192 (`core/inference/config.py:653-654`) has no training-
  side counterpart. **[V]**
- **The SAM3 role's `imgsz` is parsed, validated and threaded into hyperparams
  (`detectkit/config/training.py:291-292`, `jobs/training.py:295-298`) but tiling never reads
  it** — `dataset_build.py:375` hardcodes `_SAM3_IMGSZ`. A plan setting `imgsz: 640` on a SAM3
  role silently gets 1008-px geometry. **[V]**
- Manifest writes `"tile_px": [w, h]` and `"reference_body_px"` (:542-543, :607-608) plus
  `"min_retained_area_frac"` (:550, :607) **[V]**.
- Publish: `publish_worker.py:262-273` stamps `train_tile_px` (from manifest `tile_px`),
  `reference_body_px`, `object_tile_fraction`, `imgsz = PREDICTOR_IMGSZ` into
  `<artifact>.sam3_meta.json` (`publish_worker.py:176`, `publish.py:282`) **[V]**.
  `publish.py:374-392` collapses a `[w,h]` pair to a scalar, refusing a non-square pair **[V]**.
- **Three partial-construction sites reintroduce the silent default**, none logging:
  `detectkit/config/training.py:509-556` (YAML plan coerces a field only `if name in
  sam3_values`), `sam3_lora/cli.py:88-89` (`Sam3LoraParams(**sam3_data)` from `spec.json`),
  `detectkit/jobs/dataset_preparation_sidecar.py:209`. The GUI is total
  (`sam3_training_panel.py:507-543` passes every field) but its spin box is *seeded* from
  `Sam3LoraParams()` (:492, :578), so an untouched widget ships 0.055 as if chosen. **[V]**

### 2.2 Two corrections to the record

1. **`train_tile_px` has NO reader anywhere in `src/`.** `grep -rn train_tile_px src/ tests/`
   returns only the writer (`publish_worker.py:267`), a comment (`publish.py:378`) and three
   test assertions **[V]**. The eval spec §11d's phrase "since TrackerKit reads
   `train_tile_px` back to prefill SAHI" describes an intent, not current behaviour. **The
   SAM3 stamping loop is open at both ends: nothing validates it in, nothing reads it out.**
2. Only `core/inference/semantic/sam3.py:104-140` and `semantic/checkpoints.py:268` read the
   `.sam3_meta.json` sidecar at all, and they use it for a checkpoint-integrity guard, not
   geometry **[V]**.

### 2.3 Divergence table — every tiling knob

| Knob | YOLO (`sliced_dataset.py` / `detectkit/config/training.py`) | SAM3 (`training/contracts.py` / `sam3_lora/dataset_build.py`) | Same meaning? | Same default? |
|---|---|---|---|---|
| `geometry_mode` | yes, `auto_object` default (`training.py` slicing block) | yes, `"auto_object"` `contracts.py:249` | **yes** — same `tile_size_for_mode` | **yes** |
| `object_tile_fraction` | yes, but **overridden** by `target_sizes` in `auto_object` (`sliced_dataset.py:167-184`); DetectKit stores `target_sizes` and derives `target/imgsz` (`training.py:252`) | yes, **authoritative**, `0.055` `contracts.py:250` | same formula, **different authority** | **NO** — YOLO effective = `median([200,300,400])/imgsz`; SAM3 = 0.055 |
| `imgsz` (fraction denominator) | model imgsz, user-set per role (`jobs/training.py:214` `target_sizes_for(request.imgsz_for(role))`) | fixed `_SAM3_IMGSZ = 1008` (:97) | different constant, same role | **NO** |
| `slice_width` / `slice_height` | yes, `0` = fall back to imgsz | yes, `0`, `contracts.py:251-252` | **yes** | **yes** |
| overlap | `overlap = 0.2` (`SliceBuildParams`, `sliced_dataset.py:102`) | `tile_overlap = 0.25` `contracts.py:253` | same semantic | **NAME DIVERGES**; SAM3 0.25 vs inference `DEFAULT_OVERLAP=0.5` (`tiling.py:39`) and direct-cal `OVERLAP_STEPS (0.1,0.2,0.3)` |
| **`target_sizes` (multi-scale)** | **yes**, `[200.0, 300.0, 400.0]` | **NO — absent entirely** | n/a | n/a |
| **`full_frame_mix`** | **yes**, `True` | **NO — absent entirely** | n/a | n/a |
| empty / negative tiles | `negative_tile_fraction = 0.15` (a *sampling rate*) | `keep_empty_tiles: bool = True` (a *boolean*) | **NO** — rate vs flag | **NO** (0.15 vs keep-all) |
| seam-fragment policy | `min_area_ratio = 0.1`, fragment **DROPPED** (`sliced_dataset.py:135-136`) | `MIN_RETAINED_AREA_FRAC = 0.25`, fragment **KEPT as `iscrowd=1`** + tile exhaustiveness downgraded (`dataset_build.py:270-275`) | **NO — opposite policies** | **NO** (0.1 vs 0.25) |
| fragment accounting | none | `downgraded_tiles`, `fragment_only_tiles`, `fragment_annotations` in `SplitCounts` (:51-63) | SAM3-only | — |
| `reference_body_px` estimator | global median over ALL object majors pooled corpus-wide (`sliced_dataset.py:219, 263-267`) | median over PER-FRAME medians (`dataset_build.py:353-369`) | **NO** | — |
| stamped geometry key | `slice_geometry{...}` -> `.slice_meta.json` v2 + registry (`model_publish.py:874-942`) | flat `train_tile_px`/`reference_body_px`/`object_tile_fraction` -> `.sam3_meta.json` (`publish_worker.py:267-269`) | **NO — two sidecar formats** | — |
| read back for prefill | **yes** (`slice_meta_to_panel_values` -> detection panel -> `build_engine_params`) | **no reader** (§2.2) | **NO** | — |
| calibration profiles on the artifact | yes, `profiles[]` + `primary_profile_id` | **none** | **NO** | — |
| role `imgsz` honoured by tiling | yes, per-role (`jobs/training.py:214`) | **no** — parsed then ignored, `_SAM3_IMGSZ` wins (`dataset_build.py:375`) | **NO** | — |
| `geometry_mode` value validated | inference side clamps/validates (`slice_meta.py:380-381`, `config.py:653-654`) | **never validated**; unknown -> silent 1008x1008 | **NO** | — |
| guard against baseline-geometry drift | none, but drift is at least *visible* via the stamped set | none — the 0.055 default entered silently (eval spec §10) | — | — |

**Divergence count: 14 of the 17 knob rows disagree** — the two sides differ in existence,
meaning, default, or direction. The exceptions are two clean matches (`geometry_mode` as a
field; `slice_width`/`slice_height`) and one row where both sides are equally unguarded
(baseline-geometry drift). The underlying `plan_tiles`/`tile_size_for_mode` call is shared and
is not counted as a knob.

### 2.4 Where the current code assumes ant-like scale

- `sliced_dataset.py:105` `target_sizes = [200.0, 300.0, 400.0]` — absolute apparent pixels,
  hardcoded, duplicated at `detectkit/config/training.py:169` and `detectkit/gui/models.py:181`
  **[V]**. Three copies of one un-calibrated constant.
- `detectkit/config/training.py:252` divides by a **literal `640.0`** rather than the actual
  imgsz **[V]** — a second, silently different denominator from `target_sizes_for(imgsz)` at
  :254.
- `contracts.py:250` `object_tile_fraction = 0.055` — the constant that caused the incident **[V]**.
  Note there are **three** different `object_tile_fraction` defaults in the tree: 0.055 (SAM3,
  `contracts.py:250`), 0.15 (YOLO training, `sliced_dataset.py:99`), 0.15 (inference,
  `core/inference/config.py:80`). **[V]**
- `tiling.py:27-33` `SEMANTIC_TILE_FRACTION_SEED = 0.05`, self-documented as ungrounded **[V]**.
- `inference_settings.py:200` falls back to a literal `96` body px when `target_sizes` is empty **[V]**.
- `dataset_build.py:92` `MIN_RETAINED_AREA_FRAC = 0.25`, justified by ant-corpus reasoning **[V]**.
- `_SAM3_IMGSZ = 1008` is a genuine model constant, not a species assumption **[V]**.

---

## Part 3 — The unification design

### 3.1 Principle: unify the CONTRACT and the POLICY, not the planner

The planner is already shared and its semantics are frozen against inference
(`slice_geometry.py:122-124`). **Nothing in this design changes `tile_size_for_mode` or
`plan_tiles`.** What gets unified is the layer above.

### 3.2 New shared module: `utils/tiling_plan.py` (Qt-free, imports only `slice_geometry`)

One dataclass and one resolver, consumed by both builders. Proposed shape (names, not code):

- `ScaleSpec` — one requested scale. Exactly one of: `target_apparent_px: float` (the
  apparent object size the model should see, in *its own* input resolution), or
  `object_tile_fraction: float`, or `custom_wh: tuple[int,int]`, or the sentinel
  `FULL_FRAME`. Mirrors the existing `TILE_FRACTION_GRID` convention where `None` = full
  frame (`tiling.py:37-38`) — reuse that sentinel meaning rather than a new one.
- `TilingContract` — `geometry_mode`, `scales: tuple[ScaleSpec, ...]`, `overlap`,
  `reference_body_px`, `model_imgsz`, `empty_tile_policy`, `fragment_policy`,
  `max_tiles_per_frame`. **`overlap` is the single canonical name**; `tile_overlap` becomes an
  alias read at contract-load time for backwards compatibility.
- `resolve_scales(contract) -> list[TileSize | FULL_FRAME]` — one generalisation of
  `sliced_dataset._tile_sizes_for_params` (:159-193). Both builders call this. Dedupe and
  clamping stay exactly where they are today.
- `ReferenceBodyEstimator` — one estimator, resolving the median-vs-mean divergence (§2.3).
  **Decision for the user, see §3.7 D1.**

Genuinely model-specific and staying out: SAM3's `_SAM3_IMGSZ = 1008` (a model input
constant, passed IN as `model_imgsz`), YOLO's per-role `imgsz`, mask-vs-OBB target encoding,
COCO-vs-YOLO-txt serialisation, `iscrowd` and exhaustiveness flags, negative-prompt pools.

### 3.3 Multi-scale for SAM3

Port `_tile_sizes_for_params` semantics via `resolve_scales`. Concretely, SAM3's
`dataset_build.py:373-380` becomes a loop over resolved scales, `_tile_frame` is called once
per scale, and the tile id/filename gains the scale token the YOLO builder already uses
(`_t{w}x{h}_`, `sliced_dataset.py:237`) so per-scale provenance survives into the COCO image
records. Full frame is one more entry in the scale list, emitted with the existing
`plan_tiles(..., full_frame=True)` path.

**Cross-scale composition of the seam policy.** This is the subtle part and must be stated:
`retained_frac` is computed against the instance's **full frame-space area**
(`dataset_build.py:266-274`), not against a per-scale quantity, so the *test* composes across
scales unchanged. What does NOT compose is the *rate*: a small tile size produces
proportionally more seam crossings, so a naive multi-scale build downgrades a much larger
share of tiles at the fine end than the coarse end, silently removing precision pressure
exactly where the eval spec says extras/frame is decided. Mitigations, in order of
preference: (i) report `downgraded_tiles` / `fragment_only_tiles` **per scale** in
`SplitCounts` so the cost is visible before training (the existing counters already exist for
precisely this reason, :51-63); (ii) treat the fragment floor as a per-scale contract value,
not a module constant; (iii) do NOT auto-tune it — it is a measured-per-project quantity.

**Per-scale instance balancing.** Today, neither builder balances (§1.2). Tile count grows
~quadratically as tile size shrinks, so an unweighted 3-scale build is dominated by the
finest scale. Proposal: a `per_scale_weight` policy in `TilingContract` with three named
modes — `none` (today's behaviour, the default so the change is opt-in),
`equal_tiles` (cap each scale at the min tile count across scales),
`equal_instances` (cap each scale at the min *positive-instance* count). Which is right is a
per-project measurement, not a constant; the build manifest must record the realised per-scale
counts either way.

### 3.4 Where the two calibrations fit

`build_candidate_grid` (`direct_calibration_grid.py:59-100`) is already **relative** to the
model's stamped training geometry (multiplicative `FRACTION_STEPS`), which makes it
species-agnostic by construction and is the right precedent. **[V]**
`semantic/calibration.py` already sweeps a tile-fraction grid *including a full-frame arm*
against the user's own labels with a recall-first objective. **[V]**

**Honest assessment: the existing machinery fits the SERVING side and only partly fits the
TRAINING side.** Both calibrators sweep an *already-trained* model over geometries — one
inference pass per geometry. Choosing a *training* scale set that way would require one
training run per candidate set, which is hours-to-days per arm and is not viable as a
pre-run wizard. So:

- **Reuse directly (fits):** choosing the serving geometry / primary profile for a
  multi-scale model, and choosing SAM3's *inference* tiling. No change needed beyond giving
  SAM3 artifacts a `.slice_meta.json`-shaped profile surface (§3.5).
- **Reuse the *inputs*, not the loop (partial fit):** the training scale set should be
  derived from the **label-corpus body-size distribution** the builder already measures
  (`measure_reference_body_px` / `object_major_axes_px`, `sliced_dataset.py:42-60`) — e.g.
  scales that bracket the observed size distribution's spread — plus the project's frame
  size. This is a *measurement at build time on the user's own corpus*, which satisfies
  "calibrate before run" without a training-per-candidate sweep, and it is what replaces the
  hardcoded `[200, 300, 400]`.
- **Does not fit, say so plainly:** there is no cheap way to know which training scale set
  yields the best model without training. The defensible substitute is (a) derive candidates
  from the corpus, (b) train ONE multi-scale model, (c) let the *existing* serving
  calibration pick the operating geometry per project. The eval spec's 2x2 (§11c) is
  precisely the evidence that a single-scale model cannot survive a serving-geometry change;
  a multi-scale model plus serving calibration is the structural answer.

### 3.5 THE STAMPING PROBLEM

**The answer already exists in shipped code and should be reused rather than re-invented.**
`slice_meta.py:287-317` already handles a multi-scale artifact: it stamps the whole
`target_sizes` list as training provenance and derives the `__training__` prefill as
`median(target_sizes) / imgsz` **[V]**. The v2 schema's split — `training_geometry` (what was
trained, provenance, possibly plural) vs `profiles[] + primary_profile_id` (what to serve,
measured, singular) — is exactly the distinction a multi-scale model needs.

Concrete contract:

1. **What is stamped.** The full scale set, losslessly: `training_geometry.target_sizes`
   (or the richer `scales[]` from `TilingContract`), `full_frame_mix`, `reference_body_px`,
   `overlap`, `imgsz`, `geometry_mode`, and the per-scale realised tile/instance counts.
   Plus, new: `scale_range_px = [min, max]` apparent size actually trained, which is the
   number a user and a guard can both reason about. **A multi-scale model must NOT stamp a
   single `object_tile_fraction` as if it were the trained geometry** — it may stamp the
   derived median as a clearly-named `prefill_object_tile_fraction` only.
2. **What the panel prefills.** Unchanged ladder, unchanged precedence: an explicit primary
   calibration profile if one exists; else the derived median-of-set training value with
   `resolution="training"` (already implemented, `slice_meta.py:321-382`). The panel should
   additionally *display* the trained scale range, because a user prefilling the median of a
   3-scale set deserves to know the model also saw 1.5x and 0.5x of it. For a multi-scale
   model the honest UI statement is "this model was trained at N scales spanning A-B px; the
   prefill is the median — calibrate to pick the best".
3. **What the CLI does.** Unchanged: `--sahi-profile` resolves against the same sidecar,
   hard-errors on an unresolvable explicit name (`cli_config.py:143-170`), and
   `_slice_profile_overlay` supplies the values (`engine_params.py:363-436, 968-986`). One
   addition: when a config names no profile and the model is multi-scale, log the trained
   scale range and the fact that the median is being used, at WARNING — because that is
   exactly the class of silent default that caused the 0.055 incident.
4. **SAM3 specifically.** `train_tile_px` (scalar, write-only, `publish_worker.py:267`) is
   not extensible to a set and has no reader. Either SAM3 artifacts adopt the v2
   `.slice_meta.json` surface, or `.sam3_meta.json` gains an embedded v2-shaped
   `training_geometry` block. **User decision D2 (§3.7).**
5. **The guard the incident asks for.** A run that names a comparison baseline must read the
   baseline artifact's stamped geometry and refuse-or-warn on mismatch. This is cheap, it is
   the eval spec's own recommendation (§10 "Root cause"), and it is independent of everything
   else here — it can ship first.

### 3.6 Calibration-before-run for the scale set

Replace the three hardcoded copies of `[200, 300, 400]` with a resolver that, at dataset-build
time, reads the corpus's own measured major-axis distribution
(`object_major_axes_px`, already computed) and the project frame size, and proposes a scale
set bracketing it. The proposal must be **shown and editable** in the DetectKit slice panel
(`detectkit/gui/panels/slice_settings_widget.py:480-511` already round-trips
`target_sizes` + `full_frame_mix` **[V]**), and recorded in the manifest with the
distribution it was derived from. No number in this document is proposed as a default: the
*rule* is shared, the *values* are per-project.

### 3.7 Decisions this forces on one side — for the user, not assumed

- **D1 — reference-body estimator.** Global median (YOLO) vs mean-of-per-frame-medians
  (SAM3). Unifying changes one side's stamped `reference_body_px`, hence its tile size, hence
  its trained geometry. Non-cosmetic. Recommend global median (more robust, and it is what
  the *inference* side's calibration shape-prior is fitted against), but this is the user's call.
- **D2 — sidecar unification.** SAM3 adopting `.slice_meta.json` gives it profiles,
  `--sahi-profile`, and the whole shipped handoff for free, but changes an artifact format
  with a live integrity guard (`semantic/sam3.py:104-140` *refuses to serve* a checkpoint
  whose sidecar is missing or unparseable **[V]** — a migration must not trip that).
- **D3 — seam-fragment policy.** YOLO drops sub-floor fragments; SAM3 keeps them as
  `iscrowd`. Both are deliberate and documented. Unifying the *floor value* is easy;
  unifying the *direction* would change one trainer's targets. Recommend: share the knob,
  keep both behaviours selectable, do NOT force convergence.
- **D4 — empty-tile policy.** Boolean (SAM3) vs sampling rate (YOLO). A rate generalises the
  boolean (`1.0` == keep all), so unify on the rate — but that changes SAM3's contract field
  name and type.
- **D5 — overlap default.** 0.25 (SAM3 training) vs 0.5 (semantic inference) vs 0.2 (v2
  prefill default, `slice_meta.py:305`) vs the 0.1/0.2/0.3 sweep. Four values in the tree.
  Unification means picking one *prefill*; the real value should come from calibration.
- **D5b — validate `geometry_mode` and honour the role `imgsz`.** Both are currently silent
  no-ops on the SAM3 side (§2.1). Fixing them is strictly better behaviour but *changes what
  an existing plan does*, so it is a decision, not a cleanup.
- **D6 — multi-scale opt-in vs default-on for SAM3.** Default-on changes every future SAM3
  run's dataset; opt-in leaves the incident class alive by default.

---

## Part 4 — Ranked plan, risks, and the full-frame question

### 4.1 Build order

1. **Geometry-drift guard + provenance logging (small, independent, highest value/cost).**
   Read a named comparison baseline's stamped geometry at plan/preflight time and warn-or-
   refuse on mismatch; log the effective geometry and its *source* (explicit / profile /
   contract default) at the start of every dataset build. Directly prevents the incident that
   produced the whole 2026-09-06 confound. Requires no unification.
2. **Fix the sidecar-drop gaps that would silently discard any stamping work**
   (`model_publish.py:874-879`, and the TrackerKit import copy noted as F4 in the merged
   audit). Cheap; otherwise everything below can vanish at publish time.
3. **`utils/tiling_plan.py` contract + `resolve_scales`, with YOLO refactored onto it and
   byte-identical dataset output as the gate.** No behaviour change; pure consolidation.
4. **SAM3 multi-scale via `resolve_scales`**, opt-in (D6), with per-scale `SplitCounts`
   reporting and manifest-recorded realised counts.
5. **Stamping: multi-scale-aware `training_geometry` for both, SAM3 onto the v2 surface (D2).**
6. **Corpus-derived scale-set proposal replacing `[200,300,400]`** in all three copies.
7. **Per-scale balancing modes** — defer; needs measurement to choose, and `none` is a valid
   shipping default.
8. **Deferred entirely:** any attempt to calibrate the *training* scale set by training
   multiple arms. Not viable as a wizard (§3.4).

### 4.2 Top risks

- **R1 — a shared contract silently changes trained geometry.** `tile_size_for_mode`
  semantics are frozen against inference (`slice_geometry.py:122-124`); D1/D4/D5 each shift a
  default that feeds it. A "refactor" that moves a default by one clamp reproduces exactly the
  0.055 incident at a larger blast radius. Mitigation: every consolidation step gated on
  byte-identical built datasets under the old parameters, and defaults preserved per-side
  until explicitly changed by the user.
- **R2 — the multi-scale seam/fragment rate is not scale-invariant** (§3.3). More scales at
  the fine end means a much larger fraction of downgraded (SAM3) or dropped (YOLO) instances,
  which moves precision — the exact metric the programme optimises — with no signal in the
  loss. Mitigation: per-scale counters, surfaced before training starts.
- **R3 — stamping ambiguity leaks into serving.** If a multi-scale model stamps a single
  collapsed fraction into a field consumers read as "the trained geometry", every downstream
  guard and prefill will confidently assert a geometry the model was never trained at. The
  median-collapse must be a *named prefill*, never the provenance. Compounded by R-adjacent:
  the SAM3 sidecar guard refuses to serve on a malformed sidecar, so a botched migration is a
  hard outage, not a degradation.
- **R3b — anisotropic edge-tile stretch scales with the number of scales.**
  `datapoints.py:165-174` stretches every tile to 1008x1008 without preserving aspect, and
  partial edge tiles are not square. Adding scales multiplies the edge-tile population, so a
  multi-scale SAM3 dataset contains proportionally more distorted supervision than a
  single-scale one — with no counter reporting it today. Mitigation: count and report
  non-square tiles per scale alongside the fragment counters.
- (R4, lesser) Three duplicated copies of `target_sizes` defaults plus the literal `640.0`
  denominator at `training.py:252` mean a "single" change is really four.

### 4.3 Will full frames actually help?

**Unknown, and this design refuses to assert it.** The arithmetic is real: at
`_SAM3_IMGSZ = 1008`, a 4512-px frame presents a 97-px animal at ~21 px, and the eval spec's
own 2x2 shows a 1.82x scale shift dominating every model difference measured. That argues a
full-frame arm is *at best* a weak-supervision signal and *at worst* teaches the model to
respond to blobs it can never resolve at serving time. But it is one corpus, one animal size,
one frame size — the user's own standing principle disqualifies generalising from it. Note
also that YOLO already ships `full_frame_mix = True` **[V]**, so on that side the question is
whether to keep an existing default, not whether to add one; and the semantic inference grid
already carries a full-frame arm precisely because "on a rig where animals are already large
at native resolution ... tiling HURTS" (`tiling.py:35-38`) **[V]**.

**What would need measuring to know.** A controlled ablation, per project, holding
everything else fixed: same corpus, same seed, same step budget, arms = {tiled scale set} vs
{tiled scale set + full frames}, evaluated on the **same held-out corpus rendered at each
serving geometry** (the 2x2 discipline the eval spec established), with the pre-registered
paired extras/frame statistic plus recall. The decisive covariate is
`apparent_px = reference_body_px * model_imgsz / frame_long_edge`; the honest hypothesis is
that full frames help only when that quantity stays above some resolvability floor, and
**that floor is the thing to measure, not to guess**. Until measured, full-frame mixing
should be an explicit, per-project, calibration-informed switch that the manifest records —
never an unstated default on either side.
