# Unified tiling & calibration geometry: four paths, one contract

*(Originally scoped as "unified SAHI training geometry". Widened twice at the user's direction:
first to four paths, then to the OBJECTIVE those paths optimise and the evidence a training run
leaves behind — Parts 5 and 6.)*

*(First widening:
"We have a parallel sahi calibration inside semantic escalation. We should just unify all
these paths to have minimal divergences." The document now covers four paths, not two.)*

**Status:** design proposal, pending review. No implementation.
**Repo:** `/Users/neurorishika/Projects/Rockefeller/Kronauer/multi-animal-tracker` @ `main` (`097408af`), read-only audit.

> ## AMENDED 2026-09-06 (second pass) — the YOLO baseline MOVED under this document
>
> The audit above was written against `097408af`. Five commits have landed on `main` since,
> and three of them change the very numbers Part 2 tabulated. **The unification's shared
> baseline is the NEW defaults, not the ones the audit captured.** Every affected row below
> is amended in place, keeping the historical value visible (`old -> new`, SHA) because the
> old numbers are load-bearing for the 2026-09-06 incident narrative.
>
> | Commit | What it changed |
> |---|---|
> | `4de070f3` feat(detectkit): use safer SAHI training defaults | YOLO `object_tile_fraction` 0.15 -> 0.10; `min_area_ratio` 0.1 -> 0.25; `target_sizes` (200,300,400) -> (32,64,96,128); **new** `target_size_fractions = (0.05,0.10,0.15,0.20)` — scale is now expressed RELATIVE to model input, with a legacy-compat branch |
> | `306738ec` feat(detectkit): balance multi-scale SAHI loss | new `training/ultralytics_scale_balance.py`; `balance_multiscale_loss = True` + `_power = 0.5`, default-ON; scale-grouped batch sampler + inverse-frequency loss weighting; stamped into the build manifest |
> | `011a34b2` refactor(semantic): `resolve_tile_px` delegates to `tile_size_for_mode` | **closes F-D / §2.5 row 1** — the 5th copy of the tile-size formula is gone |
> | `b577ea6c` + `e3058ecb` feat/fix(core): shared train/serve geometry-drift guard | **closes F-E / §2.5 row 24** — the guard left the Qt dialog into `core/inference/geometry_drift.py`, reaches both builders and the serving overlay, and effective geometry + its `GeometrySource` is now logged at every build and run |
>
> **Net effect on the counts:** §2.3 goes from *14 of 17* to **17 of 20** rows disagreeing
> (one row becomes a clean match, three rows are new, one row's content gets worse);
> §2.5 goes from *23 of 24* to **22 of 25** (rows 1 and 24 resolve to SAME, rows 5 and 11
> change value, one row is added). **The unification got smaller in two places and larger in
> three — net, there is more to unify, not less.** Full re-derivation in §2.3, §2.5 and the
> new §2.8.
>
> **What did NOT change and must not be read as changed:** SAM3's
> `training/contracts.py:264 object_tile_fraction = 0.055` and the inference-side
> `core/inference/config.py:80 object_tile_fraction = 0.15` are untouched. The tree therefore
> now holds **three mutually distinct** values (0.055 / 0.10 / 0.15) where §2.4 recorded two
> distinct values across three sites. That row got *worse*, not better. New decision **D16**.
>
> **Do not re-open D7/D8/D9** (§3.10) — they are resolved. New decisions raised by this
> amendment are numbered **D16-D19** (D13-D15 are taken by §6.6).

**The four paths.**

| # | Path | Module | Role |
|---|---|---|---|
| **P1** | YOLO SAHI sliced TRAINING | `training/sliced_dataset.py` | builds tiles to train on — **already multi-scale** |
| **P2** | SAM3 LoRA TRAINING | `training/sam3_lora/dataset_build.py` | builds tiles to train on — **single scale** |
| **P3** | DetectKit DIRECT calibration | `core/inference/direct_calibration{,_grid,_sweep}.py`, `detectkit/gui/dialogs/direct_calibration_*.py` | sweeps serving geometry for YOLO detect/obb/segment |
| **P4** | DetectKit SEMANTIC ESCALATION calibration | `core/inference/semantic/{calibration,shape_prior,tiling}.py`, `detectkit/gui/dialogs/semantic_escalation_dialog.py` | sweeps serving geometry for SAM3 |

Downstream of P3 and P4: the TrackerKit SAHI inference profiles (`core/inference/slice_meta.py`
v2, `trackerkit/engine_params.py`) that consume a chosen operating point.

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

**F-D. There are FOUR paths and FIVE copies of the tile-size formula.** The `plan_tiles`
*grid* is shared by all four **[V]**, but the *sizer* is not: `semantic/tiling.py:123-139`
(`resolve_tile_px`) reimplements the `auto_object` branch of `tile_size_for_mode` inline —
`int(max(64, min(4096, round(reference_body_px / frac))))`, numerically identical today, and
with **no `geometry_mode` concept at all** (no `custom`, no `auto_model`) **[V]**. A change to
`slice_geometry.py:113-137` therefore silently fails to reach P4.

> **SHIPPED `011a34b2` — F-D's fifth copy is gone.** `resolve_tile_px` now delegates to
> `tile_size_for_mode` with `geometry_mode` pinned internally to `auto_object`, its
> `None`/no-tiling and NaN guards kept local, pinned byte-identical by the characterization
> test committed in `a153e86a` (1572 assertions) **[V]**. Four copies remain, all of them the
> same shared helper. §2.5 row 1 becomes **SAME**. The *residual* divergence is narrower and
> should still be recorded: P4 still exposes **no `geometry_mode` concept** to its caller
> (§2.5 row 3 stands), and `resolve_tile_px`'s docstring now states as policy that it
> "deliberately never reads `SliceTrainingSettings.object_tile_fraction`: the sliced-training
> optimum and the SAM3 optimum differ by ~3x" **[V]** — a committed claim that one fraction
> cannot serve both, which is exactly the question D16 puts to the user.

**F-E. A train/serve geometry-drift guard ALREADY EXISTS — in exactly one of the four paths,
and it is trapped inside a Qt dialog.** `detectkit/gui/dialogs/semantic_escalation_dialog.py:382-406`
reads the SAM3 sidecar's `reference_body_px` and `object_tile_fraction`, prefills when the
project has none, and raises a modal `QMessageBox.warning` ("Body Size Mismatch ... verify this
is intentional") when they disagree — deliberately warn, never refuse **[V]**. This is exactly
the guard §4.1 recommends building, already written, already reasoned about, in a place no
headless run and no other path can reach. **It is the single best argument in this document
that the guard should be built once in core rather than four times.**

> **SHIPPED `b577ea6c` + `e3058ecb` — F-E is closed, and it closed the way this section
> argued.** The guard is now `core/inference/geometry_drift.py` (Qt-free, pure, typed
> verdict); the dialog re-points at it with wording, prefill semantics and the 1e-6 float
> tolerance preserved verbatim; both dataset builders and the serving-side profile overlay
> call it; and `log_effective_geometry` now prints the effective geometry **and its
> `GeometrySource`** (explicit / calibration profile / corpus-derived / contract default) at
> every build and every run **[V]**. Warn-never-refuse everywhere, which is the posture D11
> recommended — **D11 is therefore substantially settled by implementation, not by decision**;
> what remains open in D11 is only the narrow "explicitly-named comparison baseline" case,
> which the guard supports (`comparison_baseline` is threaded from the plan JSON) but still
> only warns on. `e3058ecb` also records the failure mode worth remembering: the guard was
> *inert* because `train_tile_px` is a `[w, h]` LIST and the reader coerced with `float()`,
> and the test hid it by stamping a scalar — "it asserted the shape the code assumed rather
> than the shape the artifact has, which is the same failure class as the incident this guard
> exists to prevent" **[V]**.

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

- `SliceBuildParams` (:95-109) **[V, re-verified post-amendment]**, verbatim defaults:
  `geometry_mode="auto_object"`, `imgsz=640`, `object_tile_fraction=0.10` (was `0.15`,
  `4de070f3`), `slice_width=0`, `slice_height=0`, `overlap=0.2`, `min_area_ratio=0.25`
  (was `0.1`, `4de070f3`), `negative_tile_fraction=0.15`,
  `target_sizes=[32.0, 64.0, 96.0, 128.0]` (was `[200.0, 300.0, 400.0]`, `4de070f3`),
  `full_frame_mix=True`, `reference_body_px=0.0`, **new** `balance_multiscale_loss=True`,
  `balance_multiscale_loss_power=0.5` (`306738ec`).
  **The builder is still PIXEL-FED.** Fractions are a DetectKit-config concept only:
  `detectkit/jobs/training.py:220` resolves them with
  `slicing.target_sizes_for(request.imgsz_for(role))` before constructing `SliceBuildParams`
  **[V]**. That layering matters for §3.5 — see the amendment there.
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

**~~There is no per-scale instance balancing.~~ SUPERSEDED by `306738ec` — there now is, on
the YOLO side, and it is default-ON.** The observation that motivated the claim still holds
(a small tile size contributes quadratically more tiles than a large one), but the tree now
answers it. `training/ultralytics_scale_balance.py` **[V, read in full]**:

- `scale_group_for_path(path)` — groups by the emitted stem: `_full` -> `"full"`,
  `_t{W}x{H}_{n}` -> `"tile:WxH"` (regex `_t(\d+)x(\d+)_\d+$`), anything else -> `"other"`.
- `scale_group_weights(paths, power)` — `(mean_count / group_count) ** power` over tile
  groups only; `full` and `other` are pinned to weight `1.0`, so the configured full-frame
  mix is preserved rather than re-weighted. `power` is clamped to `[0,1]`; `0.5` =
  square-root balancing (the default), `1.0` = exact group balance.
- `ScaleGroupedBatchSampler` — yields **scale-homogeneous** batches, every index exactly once
  per epoch, deterministic per `seed + epoch`. This is a *sampling-strategy* change as much
  as a loss change, and the audit's brief did not anticipate it.
- Installation is by monkeypatch from `training/ultralytics_entrypoint.py:main` — it patches
  `YOLODataset.__getitem__`/`collate_fn`, `DetectionTrainer.get_dataloader`, and
  `loss` on `DetectionModel`/`OBBModel`/`SegmentationModel`, gated by sniffing `data=` out of
  `argv` and reading `slice_geometry.multiscale_loss_balance` from the sliced dataset's
  `manifest.json`.
- **No tile is removed or replaced** — the epoch's data exposure is unchanged; only the loss
  weight and the batch composition move.
- **DDP silently opts out**: `rank != -1` or `WORLD_SIZE > 1` returns the unmodified loader
  with a LOGGER warning **[V]**. See new risk R7.

Surfaced in the GUI as "Balance multi-scale training loss" + "Balance strength", visible only
in `auto_object` mode (`slice_settings_widget.py`), in the plan schema as
`balance_multiscale_loss{,_power}` (`detectkit/config/training.py`), and stamped into the
build manifest as `slice_geometry.multiscale_loss_balance{enabled,power}`
(`sliced_dataset._slice_geometry_manifest`) **[V]**.

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

> **CORRECTION #1 IS NOW STALE — `e3058ecb` gave `train_tile_px` a reader.** The shared
> geometry-drift guard reads it, and reads it *in the shape the publisher actually writes*:
> a scalar **or** a `(w, h)` pair, compared element-wise, never collapsed to its width
> **[V]**. Keep the original finding in the record — it was true, and the fact that the first
> reader written against it was inert for exactly the shape reason above is the lesson. What
> remains true: nothing **validates** `train_tile_px` on the way IN, and the guard warns
> rather than refuses, so the loop is now closed at one end only.

### 2.3 Divergence table — every tiling knob

| Knob | YOLO (`sliced_dataset.py` / `detectkit/config/training.py`) | SAM3 (`training/contracts.py` / `sam3_lora/dataset_build.py`) | Same meaning? | Same default? |
|---|---|---|---|---|
| `geometry_mode` | yes, `auto_object` default (`training.py` slicing block) | yes, `"auto_object"` `contracts.py:249` | **yes** — same `tile_size_for_mode` | **yes** |
| `object_tile_fraction` | yes, but **overridden** by `target_sizes` in `auto_object` (`sliced_dataset.py:167-184`); DetectKit stores fractions and resolves them per role imgsz (`training.py target_sizes_for`). Default **0.15 -> 0.10** (`4de070f3`) | yes, **authoritative**, `0.055` `contracts.py:264` (UNCHANGED) | same formula, **different authority** | **NO, and now 3 distinct values in the tree** — YOLO train 0.10, SAM3 0.055, inference `config.py:80` 0.15. YOLO *effective* in `auto_object` = `median(0.05,0.10,0.15,0.20) = 0.125`. See **D16** |
| `imgsz` (fraction denominator) | model imgsz, user-set per role (`jobs/training.py:214` `target_sizes_for(request.imgsz_for(role))`) | fixed `_SAM3_IMGSZ = 1008` (:97) | different constant, same role | **NO** |
| `slice_width` / `slice_height` | yes, `0` = fall back to imgsz | yes, `0`, `contracts.py:251-252` | **yes** | **yes** |
| overlap | `overlap = 0.2` (`SliceBuildParams`, `sliced_dataset.py:102`) | `tile_overlap = 0.25` `contracts.py:253` | same semantic | **NAME DIVERGES**; SAM3 0.25 vs inference `DEFAULT_OVERLAP=0.5` (`tiling.py:39`) and direct-cal `OVERLAP_STEPS (0.1,0.2,0.3)` |
| **`target_sizes` (multi-scale)** | **yes**, `[200.0, 300.0, 400.0]` **-> `[32.0, 64.0, 96.0, 128.0]`** (`4de070f3`), now demoted to a legacy-compat field | **NO — absent entirely** | n/a | n/a |
| **`target_size_fractions` (NEW, `4de070f3`)** | **yes**, `(0.05, 0.10, 0.15, 0.20)` — scales expressed **relative to model input**, resolved per role imgsz; `target_fractions()` falls back to `target_sizes / 640.0` only when fractions are empty, and `from_dict` forces fractions to `()` when a legacy plan omits the key, so an explicit legacy pixel setting is preserved rather than masked | **NO — absent entirely** | n/a | n/a |
| **multi-scale loss balancing (NEW, `306738ec`)** | **yes, default-ON**: `balance_multiscale_loss=True`, `power=0.5`; scale-homogeneous batches + inverse-frequency loss weight; DDP opts out | **NO — absent entirely** (and SAM3 has no multi-scale to balance yet) | n/a | n/a |
| **emitted-stem scale token** | `_t{W}x{H}_{n}` / `_full` (`sliced_dataset.py:237, 255`) — was provenance only, **now load-bearing**: `scale_group_for_path` parses it to group batches and weight the loss | tile ids exist but carry no scale token (single scale) | **NO** | — |
| **`full_frame_mix`** | **yes**, `True` | **NO — absent entirely** | n/a | n/a |
| empty / negative tiles | `negative_tile_fraction = 0.15` (a *sampling rate*) | `keep_empty_tiles: bool = True` (a *boolean*) | **NO** — rate vs flag | **NO** (0.15 vs keep-all) |
| seam-fragment policy | `min_area_ratio = 0.1` **-> `0.25`** (`4de070f3`), fragment **DROPPED** (`sliced_dataset.py:146-148`) | `MIN_RETAINED_AREA_FRAC = 0.25`, fragment **KEPT as `iscrowd=1`** + tile exhaustiveness downgraded (`dataset_build.py:282`) | **NO — opposite policies, but the MEASUREMENT now agrees** (see verification note below) | **YES, 0.25 both sides** (was 0.1 vs 0.25) |
| fragment accounting | none | `downgraded_tiles`, `fragment_only_tiles`, `fragment_annotations` in `SplitCounts` (:51-63) | SAM3-only | — |
| `reference_body_px` estimator | global median over ALL object majors pooled corpus-wide (`sliced_dataset.py:219, 263-267`) | median over PER-FRAME medians (`dataset_build.py:353-369`) | **NO** | — |
| stamped geometry key | `slice_geometry{...}` -> `.slice_meta.json` v2 + registry (`model_publish.py:874-942`) | flat `train_tile_px`/`reference_body_px`/`object_tile_fraction` -> `.sam3_meta.json` (`publish_worker.py:267-269`) | **NO — two sidecar formats** | — |
| read back for prefill | **yes** (`slice_meta_to_panel_values` -> detection panel -> `build_engine_params`) | **no reader** (§2.2) | **NO** | — |
| calibration profiles on the artifact | yes, `profiles[]` + `primary_profile_id` | **none** | **NO** | — |
| role `imgsz` honoured by tiling | yes, per-role (`jobs/training.py:214`) | **no** — parsed then ignored, `_SAM3_IMGSZ` wins (`dataset_build.py:375`) | **NO** | — |
| `geometry_mode` value validated | inference side clamps/validates (`slice_meta.py:380-381`, `config.py:653-654`) | **never validated**; unknown -> silent 1008x1008 | **NO** | — |
| guard against baseline-geometry drift | none, but drift is at least *visible* via the stamped set | none — the 0.055 default entered silently (eval spec §10) | — | — |

*(This table is the TRAINER pair only, P1 vs P2. The four-path table is §2.5.)*

**VERIFIED before recording agreement — do the two `0.25`s mean the same thing?**
Two identical numbers with different definitions would be worse than an honest divergence, so
this was checked line by line rather than assumed **[V]**:

| | YOLO `_tile_one_image` (`sliced_dataset.py:137-148`) | SAM3 `_tile_frame` (`dataset_build.py:271-282`) |
|---|---|---|
| numerator | `polygon_area(clip_polygon_to_tile(poly_px, tile))` | `polygon_area(clip_polygon_to_tile(poly_px, tile))` |
| denominator | `polygon_area(poly_px)` — the instance's **full frame-space** area | `polygon_area(poly_px)` — the instance's **full frame-space** area |
| clipping | shared `clip_polygon_to_tile` (Sutherland-Hodgman, `slice_geometry.py`) | the **same** shared helper |
| tile rect | integer-clamped `(xi0, yi0, xi1, yi1)` against frame bounds | the **same** integer clamp |
| degenerate guard | `full_area <= 1e-6` -> skip | `full_area <= 1e-6` -> skip |
| test | `ratio < min_area_ratio` -> **`continue`** (instance dropped) | `retained_frac < MIN_RETAINED_AREA_FRAC` -> **`is_crowd = True`** (instance kept, tile downgraded) |

**Verdict: same numerator, same denominator, same clipping, same degenerate guard, same
comparison — the QUANTITY and the THRESHOLD now genuinely agree.** Only the *consequence*
differs, and it differs deliberately on both sides. So this row moves from "different number,
different policy" to "**same number, opposite policy**", which is a real improvement: the
shared floor is now a single value that a unified `TilingContract` can carry, and **D3**
(share the knob, keep both behaviours selectable) is *strengthened*, not resolved — do not
read the matching constants as permission to unify the direction. Note also that
`MIN_RETAINED_AREA_FRAC` remains a module constant on the SAM3 side while YOLO's is a
per-build parameter; the values coincide today by convergence, not by construction, and
nothing in the tree keeps them in step. That asymmetry is itself a divergence the unified
contract must remove.

**Divergence count, P1 vs P2 (amended): 17 of the 20 knob rows disagree** (was 14 of 17 at
`097408af`; the table gained 3 rows). The arithmetic, stated so it can be checked:

- **Improved but still counted (1):** seam-fragment. The *default* converged (0.1 vs 0.25 ->
  0.25 both sides) and the *measurement* is verified identical above — but the **policy** is
  still drop-vs-downgrade, so the row still disagrees. Honest bookkeeping: this is the one
  row where the headline number now matches and the behaviour still does not.
- **Newly a clean match (1):** the **drift-guard** row, previously "both sides equally
  unguarded" (an exception, not a divergence), is now "both sides guarded by the same shared
  `core/inference/geometry_drift.py`" (`b577ea6c`). It moves from exception to match; the
  disagree count is unaffected.
- **New rows (3), all divergences:** `target_size_fractions`, multi-scale loss balancing, and
  the emitted-stem scale token — all three exist on the YOLO side only.
- **Widened (1):** `object_tile_fraction` was two distinct values across three sites; it is
  now three distinct values (0.055 / 0.10 / 0.15). Same row, worse content.

Clean matches remain three: `geometry_mode` as a field, `slice_width`/`slice_height`, and now
the drift guard. The underlying
`plan_tiles`/`tile_size_for_mode` call is shared and is not counted as a knob.

### 2.4 Where the current code assumes ant-like scale

- ~~`sliced_dataset.py:105` `target_sizes = [200.0, 300.0, 400.0]`~~ **AMENDED `4de070f3`.**
  The absolute-pixel triplet is gone as the *primary* expression: the new primary is
  `target_size_fractions = (0.05, 0.10, 0.15, 0.20)`, **relative to model input**, and
  `target_sizes` survives only as `(32, 64, 96, 128)` — those same fractions evaluated at
  imgsz 640 — for legacy plans. **This is a genuine improvement in cross-species generality
  and is assessed as a design input in the new §2.8.** But it does NOT remove the constant:
  the ladder `(0.05, 0.10, 0.15, 0.20)` is still four hardcoded, un-calibrated numbers, now
  duplicated as *fraction+pixel pairs* at `detectkit/config/training.py` (`SliceTrainingConfig`),
  `detectkit/gui/models.py` (`SliceTrainingSettings`), `training/sliced_dataset.py`
  (`SliceBuildParams`, pixels only) and `slice_settings_widget.py`
  (`_TileLayoutPreview._target_fractions`) **[V]**. §3.6's derive-the-set-from-the-corpus
  position is therefore **unchanged and still the destination**; the new defaults are safer
  placeholders, not a calibration.
- `detectkit/config/training.py` `target_fractions()` divides by a **literal `640.0`**
  **[V]** — this is no longer a bug. Post-`4de070f3` it is the *documented legacy
  interpretation* ("older projects stored pixel targets with an implicit 640px model input"),
  correctly reached only when `target_size_fractions` is empty. What remains is duplication:
  the identical fallback is implemented twice, in `SliceTrainingConfig.target_fractions` and
  `SliceTrainingSettings.target_fractions` **[V]**. Folded into R4.
- `training/contracts.py:264` `object_tile_fraction = 0.055` — the constant that caused the
  incident, **UNCHANGED by `4de070f3`** **[V]**. The tree now holds **three mutually
  distinct** defaults: 0.055 (SAM3 training, `contracts.py:264`), **0.10** (YOLO training,
  `sliced_dataset.py:110`, was 0.15), 0.15 (inference, `core/inference/config.py:80`). The
  YOLO change therefore moved the SAM3/YOLO gap from 2.7x to **1.8x** without closing it, and
  simultaneously opened a new 1.5x gap between YOLO training and YOLO inference defaults.
  **This is the exact quantity behind the 2026-09-06 confound. Decision D16.**
- `tiling.py:27-33` `SEMANTIC_TILE_FRACTION_SEED = 0.05`, self-documented as ungrounded **[V]**.
- `inference_settings.py:200` falls back to a literal `96` body px when `target_sizes` is empty **[V]**.
- `dataset_build.py:100` `MIN_RETAINED_AREA_FRAC = 0.25`, justified by ant-corpus reasoning **[V]**.
  **Amended note:** `4de070f3` moved YOLO's `min_area_ratio` to the same `0.25`. Convergence
  on a shared value is what the unification wants — but the value that both paths now share
  is the one whose *only* written justification is ant-corpus reasoning
  (`dataset_build.py:63-91`). A per-project quantity has been propagated to a second path
  rather than derived. Not a regression, and not a blocker; it is a widened exposure, and the
  unified contract must carry this floor as a **per-build, per-project measurable**, never as
  a module constant on either side (**D18**).
- `_SAM3_IMGSZ = 1008` is a genuine model constant, not a species assumption **[V]**.

### 2.5 The four-path divergence table

`—` = the knob does not exist on that path. Verdict `SAME` requires name, unit, meaning AND
default to agree; `MEANING` = same concept, divergent name/unit/default; `DIVERGE` = the paths
genuinely disagree about behaviour or the knob is missing where it is needed.

| # | Knob | P1 YOLO train | P2 SAM3 train | P3 direct cal | P4 semantic cal | Verdict |
|---|---|---|---|---|---|---|
| 1 | tile-size formula | `tile_size_for_mode` | `tile_size_for_mode` | `tile_size_for_mode` via `SLICE_*` params | ~~reimplemented~~ **delegates** to `tile_size_for_mode` (`011a34b2`) | ~~DIVERGE~~ **SAME** — resolved |
| 2 | tile GRID planner | `plan_tiles` | `plan_tiles` | `plan_tiles` | `plan_tiles` (`tiling.py:198`) | **SAME** |
| 3 | `geometry_mode` | yes, `auto_object` | yes, `auto_object`, **unvalidated** | yes, carried through candidates (`grid.py:70-72`) | **absent** — fraction-only, no custom/auto_model | **DIVERGE** |
| 4 | tile fraction — name | `object_tile_fraction` (overridden by `target_sizes`) | `object_tile_fraction` | `object_tile_fraction` | `tile_fraction` | **MEANING** |
| 5 | tile fraction — default | **`0.10`** (`sliced_dataset.py:110`, was 0.15, `4de070f3`); effective in `auto_object` = `median(target_size_fractions) = 0.125` | `0.055` (`contracts.py:264`, unchanged) | swept: `FRACTION_STEPS (0.75, 1.0, 1.5)` **x the model's stamped fraction** | seed `0.05` (`tiling.py`), swept `(0.03, 0.05, 0.10, None)` | **DIVERGE** — still 4 unrelated numbers; the P1/P2 gap narrowed 2.7x -> 1.8x by coincidence, not by decision (**D16**) |
| 6 | fraction denominator | role `imgsz` (`jobs/training.py:214`), and a stray literal `640.0` (`training.py:252`) | `_SAM3_IMGSZ = 1008`, role `imgsz` ignored | model `imgsz` via config | none — fraction applies to `reference_body_px` directly | **DIVERGE** |
| 7 | **multi-scale scale set** | **yes** (`target_sizes`) | **no** | **no** — one geometry per candidate row | **no** — one fraction per point | **DIVERGE** |
| 8 | **full-frame arm** | `full_frame_mix = True` | **absent** | `enabled=False` candidate, always first (`grid.py:72-83`) | `None` sentinel in `TILE_FRACTION_GRID` (`tiling.py:37`) | **MEANING** — 3 different encodings of one idea |
| 9 | overlap — name/default | `overlap = 0.2` | `tile_overlap = 0.25` | swept `OVERLAP_STEPS (0.1, 0.2, 0.3)` | `DEFAULT_OVERLAP = 0.5` (`tiling.py:39`) | **DIVERGE** |
| 10 | `reference_body_px` estimator | global median, all majors pooled | median of per-frame medians | read from the model's stamped geometry (`grid.py:71`) | project value, typed/prefilled, mismatch-warned (`dialog:382-406`) | **DIVERGE** |
| 11 | seam / fragment policy | drop below `min_area_ratio = 0.1` **-> `0.25`** (`4de070f3`) | keep as `iscrowd`, floor `0.25`, downgrade tile | n/a (post-merge frame-space scoring) | **drop by `seam_margin_px`, default 4** (`tiling.py:40`) | **DIVERGE** — drop vs keep vs margin-drop |
| 12 | cross-tile merge | n/a | n/a | `merge_policy/metric/threshold/backend`, swept + stamped | `merge_iou = 0.5` + `DEFAULT_CONTAINMENT_OVERLAP = 0.80` (`tiling.py:41-47`) | **DIVERGE** |
| 13 | confidence | n/a | n/a | swept, stamped into the profile | `CONFIDENCE_GRID` 0.05..0.95 step 0.05 (`calibration.py:72`) | **MEANING** |
| 14 | empty / negative tiles | `negative_tile_fraction = 0.15` (rate) | `keep_empty_tiles = True` (flag) | n/a | n/a | **DIVERGE** |
| 15 | tile-count ceiling | `MAX_TILES_PER_FRAME = 4096` | same | **`DEFAULT_MAX_TOTAL_TILES = 20000`** budget, a different quantity (`grid.py:20`) | `MAX_TILES_PER_FRAME` only | **MEANING** |
| 16 | matcher admissibility | n/a | n/a | **hard IoU >= 0.5** (`direct_calibration.py:78, 115`) | area band + `match_quality >= 0.1` + containment; **IoU route deleted** | **DIVERGE — see §2.6** |
| 17 | matched-instance floor | n/a | n/a | `MIN_MATCHED_INSTANCES = 60` | `MIN_MATCHED_INSTANCES = 20` | **DIVERGE** (same name, 3x apart) |
| 18 | localization floor | n/a | n/a | `MIN_LOCALIZATION = 0.5` (mean IoU) | `MIN_MEAN_QUALITY = 0.35` (shape-aware quality) | **DIVERGE** |
| 19 | recommendation objective | n/a | n/a | **F1-balanced Pareto** (`RECOMMENDATION_RULE`, `direct_calibration.py:186-191`) | **recall-first lexicographic on tile cost**, `MIN_RECALL = 0.90`; F1 explicitly rejected | **DIVERGE — contradictory** |
| 20 | class awareness | per-class labels | single prompt class | class-aware matching (`direct_calibration.py:113`) | single class, prompt-derived | **MEANING** |
| 21 | shape / size prior | none | none | **none** | `fit_area_band`, `LOW_MULTIPLIER 0.3` / `HIGH_MULTIPLIER 2.5` (`shape_prior.py:44-45`) | **DIVERGE** |
| 22 | output of the path | dataset manifest `slice_geometry` | manifest `tile_px` -> `.sam3_meta.json` (no reader) | `.slice_meta.json` v2 profile, consumed by GUI + CLI | `semantic_escalation_settings` + `semantic_calibration` in DetectKit **project JSON only** | **DIVERGE** |
| 23 | operating point reaches a headless run | via publish + registry | no | **yes** (`--sahi-profile`, `engine_params.py:968-986`) | **no** — project-state only, never a model sidecar | **DIVERGE** |
| 24 | train/serve drift guard | ~~none~~ **calls the shared guard** | ~~none~~ **calls the shared guard** | ~~none~~ **serving overlay calls it** | extracted to core, dialog re-points at it | ~~DIVERGE~~ **SAME** — resolved by `b577ea6c`/`e3058ecb`; one Qt-free `core/inference/geometry_drift.py`, warn-never-refuse, plus `GeometrySource` logging at every build and run |
| **25** | **multi-scale loss balancing + the `_t{W}x{H}_{n}` scale token** | **yes, default-ON** (`306738ec`); the emitted stem is parsed by `scale_group_for_path` to group batches and weight the loss | no (single scale, no token) | n/a (serving) | n/a (serving) | **DIVERGE** — and the filename convention is now a **cross-path contract**, not just provenance (see §3.3) |

~~**Four-path divergence count: 18 DIVERGE + 5 MEANING = 23 of the 24 rows disagree at some
level.** Exactly one row is clean across all four paths.~~ *(Original count, `097408af`.)*

**AMENDED four-path divergence count: 17 DIVERGE + 5 MEANING = 22 of the 25 rows disagree.**
Rows 1 and 24 became **SAME** (`011a34b2`; `b577ea6c`/`e3058ecb`), row 25 was added and
diverges, and rows 5 and 11 changed value without changing verdict. **Three** rows are now
clean across all four paths: the `plan_tiles` grid call (row 2), the tile-size formula
(row 1 — the fifth copy is gone), and the train/serve drift guard (row 24). The
trainer-only count in §2.3 is now 17 of 20. **The shape of the finding is unchanged:** the
four paths now agree about *how to compute a tile* and about *how to notice geometry drift*,
and still disagree about essentially everything that decides *which* geometry to compute.
Two of the three closures came from this document's own Step-1 build order, which is the
intended outcome, not a reason to relax the remaining 22.

### 2.6 The two calibration harnesses, mapped against each other

Both solve the identical problem — *sweep tile geometry and confidence, score against the
user's own labels, recommend one operating point* — for different detector families. They
share **no code**: not the matcher, not the point record, not the objective, not the
persistence format.

| | P3 direct (`direct_calibration.py`) | P4 semantic (`semantic/calibration.py`) |
|---|---|---|
| swept axes | fraction x overlap x confidence x merge threshold | tile fraction (outer, one inference pass each) x confidence (inner, offline) |
| geometry candidates | **relative** to the model's stamped training geometry (`grid.py:67-72`) | **absolute** grid `(0.03, 0.05, 0.10, None)` (`tiling.py:37`) |
| matcher | greedy descending IoU, **hard gate IoU >= 0.5**, class-aware, task-reduced polygon (`_as_task_polygon`) | greedy descending `match_quality`; admissibility = area band AND `min_quality` AND containment of `representative_point` |
| size/shape prior | **none** | `fit_area_band` fitted to the user's labels |
| objective | fastest point within `F1_TOLERANCE = 0.01` of best F1, on the (misses, extras, seconds) Pareto frontier | among points clearing `MIN_RECALL = 0.90`, fewest `tiles_per_frame`, tie-break highest confidence |
| refusal behaviour | drops failed/undersampled points, still recommends | **explicit typed refusals** with user-facing remedies (no point cleared recall / mistargeted / insufficient data) |
| reports duplicates | **yes** (`duplicate`, to detect bad cross-tile merges) | no |
| task-shape awareness | **yes** (`detect` reduces both sides to AABB before IoU) | no — always polygon |
| persisted to | model sidecar `.slice_meta.json` profile -> GUI + CLI | DetectKit project JSON only |

**What each has that the other lacks** — and these are the concrete unification wins:
P3 uniquely has task-aware polygon reduction, duplicate accounting, merge-policy sweeping,
relative (species-agnostic) candidate generation, and a headless consumer. P4 uniquely has a
fitted shape prior, graded (non-binary) match quality, typed refusals with remedies, and the
train/serve drift guard.

**Is their scoring even comparable? No.** P3's `recall` counts a prediction as a match only at
IoU >= 0.5 against a same-class label; P4's counts it when a representative point is contained
and a graded quality score clears 0.1. A single model scored by both would post different
recalls on identical predictions. **No number from one harness may be compared to a number
from the other**, and today nothing in the UI says so.

### 2.7 Does the DIRECT path share the centroid defect? — NO, but it has the INVERSE defect

**Answer: no.** `b9e92bc7` fixed a *containment test on the vertex mean* — a point that lies
outside 15.9% of real ant outlines, vetoing near-perfect masks (`calibration.py:248-272`,
recall 0.867 -> 0.988 on identical predictions) **[V]**. `direct_calibration.py` has **no
centroid, no containment test and no representative point anywhere** **[V, verified by reading
the whole matcher, :74-142]**. That specific bug cannot exist there.

**But the direct path fails the same underlying phenomenon by the opposite mechanism, and this
is a live production concern.** Its sole admissibility criterion is a hard `IoU >= 0.5`
(`:78, :115`) — precisely the gate the semantic path measured and then **deleted**, on the
stated grounds that "SAM3 masks trace legs and antennae at ~1.7x the labelled body-core area"
so IoU "is still not a hard gate ... it enters the quality score" (`calibration.py:17-22`)
**[V]**. The consequence: a prediction whose silhouette is correct but whose *extent
convention* differs from the labels' scores **below 0.5, is counted as a MISS and an EXTRA
simultaneously**, double-penalising the operating point.

**Who is exposed.** `_as_task_polygon` keeps the full polygon for `task="segment"` and
`task="obb"` (`:56-71`) **[V]**, and DetectKit runs direct segment models. A segment model
whose masks trace appendages, or whose labels are body-core boxes, is scored by P3 under
exactly the convention P4 abandoned — and because `recommend_balanced` optimises F1
(`:186-191`), a systematically depressed recall pushes the recommendation toward
higher-confidence, fewer-detection operating points. **A silent recommendation bias, not a
crash.** Severity is unmeasured and must not be asserted: it depends entirely on how a given
project's labels and model agree about extent, which is per-project and per-species.

**Two further asymmetries worth flagging in the same breath.** (a) P3's objective is F1;
`semantic/calibration.py:9-13` rejects F1 with a measurement — "The F1-optimal threshold missed
4.7 animals/frame where a recall-first one missed 1.0" **[V]**. The two harnesses therefore
give *contradictory* advice about what a good operating point is, and the argument against F1
was never carried across. (b) P3 has no size/shape prior, so the mistargeting failure P4's
`fit_area_band` exists to catch — an arena-sized blob or a leg-sized fragment earning recall
credit — is uncaught on the direct path. `MIN_LOCALIZATION = 0.5` partially substitutes, but
it is a mean-IoU floor, not a size gate.

**Recommended framing for the user:** these are three separate decisions (matcher
admissibility, objective, shape prior), each of which changes recommended operating points on
one side. None should be changed silently as part of a refactor. See D7-D9.

### 2.8 RELATIVE vs ABSOLUTE scale expression — a design position, not a survey

*(New section, added by the 2026-09-06 amendment. `4de070f3` made this concrete on one path;
the unification has to decide it for all four.)*

**The change.** YOLO training now expresses its scale set as `target_size_fractions`
(fractions of the active model input) rather than `target_sizes` (absolute apparent pixels).
`SliceTrainingConfig.target_sizes_for(imgsz)` multiplies by the role's actual imgsz, so
changing model input size no longer requires the user to re-derive pixel targets **[V]**.

**Position: YES — the unified contract should express every SCALE knob relatively, and it
should be explicit about WHICH denominator each one is relative to.** This is the same
principle §3.4 already credits `direct_calibration_grid.py` for (multiplicative
`FRACTION_STEPS` against the model's stamped geometry, "species-agnostic by construction"),
and it is what the user's standing rule demands: *"we are not making this app just for one
video... Measurements are a single datapoint, not trustworthy."* An absolute pixel default is
a claim about one rig; a fraction is a claim about a model's input budget, which is a
property of the model.

**But there are TWO different relative forms in the tree, and conflating them would be the
next incident.** They must be named separately in `TilingContract`:

| Form | Formula | Denominator | Invariant to | NOT invariant to |
|---|---|---|---|---|
| `target_size_fraction` (P1, new) | `apparent_px = frac * imgsz` | **model input** | changing imgsz | frame size, animal size |
| `object_tile_fraction` (P1/P2/P3/P4) | `tile_px = reference_body_px / frac` | **the measured animal** | frame size, animal size, imgsz | the animal-to-tile ratio it asserts |

**Does `object_tile_fraction` already have the property? Yes — and more of it.** Because its
denominator is a *measured* `reference_body_px`, it is already invariant to both frame size
and species size: a 5 mm ant and a 40 mm mouse at the same fraction each get a tile in which
the animal spans the same share of the tile. That is strictly stronger than
`target_size_fraction`, which is invariant only to imgsz. **The two are related but not
redundant:** `object_tile_fraction` says "how big is the animal inside its tile"; the tile is
then resized to imgsz, so `target_size_fraction ~= object_tile_fraction` *only when tiles are
resized to imgsz without letterboxing*. In the YOLO builder they are two spellings of one
number (`_tile_sizes_for_params` converts `target/imgsz` into a fraction and calls
`tile_size_for_mode` **[V]**). Recording both in the contract without stating which is
authoritative would reproduce the "different authority" row of §2.3 at the contract layer.
**Recommended: `object_tile_fraction` is the canonical scale unit** (it is the one all four
paths already speak, and the only one grounded in a measurement of the user's own corpus);
`target_size_fraction` is a *user-facing spelling* of it, resolved at plan-load time.

**Is the relative form SUFFICIENT for cross-species generality? No — and the spec should say
so plainly rather than bank the win.** Three residual assumptions survive `4de070f3`:

1. **The ladder is still four hardcoded numbers.** `(0.05, 0.10, 0.15, 0.20)` asserts that
   useful supervision lives between "animal is 1/20th of the model input" and "animal is
   1/5th". That span was not derived from any corpus. It is *safer* than `(200, 300, 400)` px
   — those, at imgsz 640, asserted animals occupying 31-62% of the model input, which is a
   very large animal — but safer is not calibrated. §3.6 (derive the set from the corpus's
   own measured major-axis distribution, show it, make it editable, record what it was
   derived from) is **unchanged and still the destination**.
2. **`reference_body_px` must exist and be trustworthy.** The relative form's whole
   generality rests on one measured scalar, and the two builders still measure it with
   **different estimators** (§2.3, D1). Relative expression makes the estimator choice *more*
   consequential, not less: everything now scales off it.
3. **A single scalar assumes a unimodal body-size distribution.** Ants in one arena are
   near-uniform; a brood-plus-adult corpus, a larval time series, or a two-species assay is
   not. The multi-scale set is the partial answer, but nothing today *checks* whether the
   corpus's major-axis distribution is unimodal, and nothing reports its spread. **Cheap and
   worth doing: record the measured distribution's quantiles in the manifest, not just its
   median.** That is a measurement, not a default, so it is admissible.

**What breaks if the contract goes fully relative — the honest cost list.**

- **Stamping (§3.5).** The build manifest currently records `target_sizes` in **resolved
  pixels** plus `imgsz`, and `slice_meta._training_values` recovers a fraction as
  `median(target_sizes) / imgsz` **[V]**. So provenance is *recoverable today* — but only
  because `imgsz` happens to be stamped alongside. If a future manifest ever drops `imgsz`,
  every stamped scale becomes uninterpretable. **The sidecar should record fractions AND the
  imgsz they were resolved at AND the resolved pixels** — all three, because each answers a
  different question (what the user asked for / what it meant / what was actually built).
- **Profiles.** `.slice_meta.json` profiles store a serving `object_tile_fraction`, which is
  already relative-to-body. No break.
- **Legacy plans.** Handled, and handled well: `from_dict` forces `target_size_fractions` to
  `()` when a legacy plan omits the key, so an explicit legacy pixel setting is preserved
  rather than silently overwritten by the new relative defaults **[V]**. The cost is a
  permanent two-field contract with a precedence rule, duplicated in two `target_fractions()`
  implementations (R4).
- **The `target_sizes` legacy field.** It cannot simply be deleted: it is the only carrier for
  pre-`4de070f3` projects, and `slice_meta._training_values` reads it off already-published
  sidecars. **Recommended: keep it read-only-on-load, never write it from the UI, and have
  the unified contract expose exactly one settable scale field.** Deleting it is a separate,
  later, migration-gated decision.
- **The 640.0 literal.** It becomes *correct* under this reading (it is the imgsz the legacy
  pixels implicitly meant) and must be documented as a legacy constant, never reused as a
  denominator for anything new.

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
  **AMENDED (§2.8):** the `target_apparent_px` alternative should be spelled
  `target_size_fraction` (relative to `model_imgsz`), matching what `4de070f3` shipped, with
  absolute pixels accepted only as a legacy input. The two relative forms have **different
  denominators** — `object_tile_fraction` is relative to the measured body,
  `target_size_fraction` to the model input — so `ScaleSpec` must name them distinctly and
  `TilingContract` must declare `object_tile_fraction` the canonical unit.
- `TilingContract` — `geometry_mode`, `scales: tuple[ScaleSpec, ...]`, `overlap`,
  `reference_body_px`, `model_imgsz`, `empty_tile_policy`, `fragment_policy`,
  `max_tiles_per_frame`. **`overlap` is the single canonical name**; `tile_overlap` becomes an
  alias read at contract-load time for backwards compatibility.
- `resolve_scales(contract) -> list[TileSize | FULL_FRAME]` — one generalisation of
  `sliced_dataset._tile_sizes_for_params` (:159-193). Both builders call this. Dedupe and
  clamping stay exactly where they are today.
- `ReferenceBodyEstimator` — one estimator, resolving the median-vs-mean divergence (§2.3).
  **Decision for the user, see §3.7 D1.**

`TilingContract` serves all four paths: P1/P2 consume it to emit tiles, P3/P4 consume it to
*enumerate candidates* (a calibration candidate becomes one `TilingContract` with a single
scale). This is the property `direct_calibration_grid.py:1-7` already insists on for P3 —
"Candidates carry `SLICE_*` PARAMS, never a hand-built `SliceConfig` ... routing through the
shared params mapping is what makes a measured point expressible as TrackerKit settings"
**[V]**. Generalising that rule to all four is the whole unification in one sentence: **every
path expresses geometry in the same vocabulary, so a point measured anywhere is expressible
everywhere.**

`semantic/tiling.py:123-139` must be deleted in favour of `tile_size_for_mode` (row 1 of §2.5).
That is a pure de-duplication today — the numbers agree — but it is what makes P4 inherit
`geometry_mode` and any future change.

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

**Per-scale instance balancing — AMENDED `306738ec`, the proposal below is SUPERSEDED on the
YOLO side.**

> ~~Proposal: a `per_scale_weight` policy in `TilingContract` with three named modes —
> `none` (the default, so the change is opt-in), `equal_tiles` (cap each scale at the min
> tile count), `equal_instances` (cap each scale at the min positive-instance count).~~
>
> **What actually shipped is better than what this section proposed, and differs in three
> ways that the unification must inherit rather than re-decide:**
> 1. **No data reduction.** The shipped strategy *keeps every tile* and re-weights the loss
>    inverse-frequency by scale group. The proposal above **capped** — i.e. discarded real
>    supervision to equalise counts. Re-weighting dominates capping: same balance, no thrown-
>    away labels, and the fine scale keeps its coverage of crowded regions.
> 2. **Default-ON, not opt-in.** `balance_multiscale_loss = True`. This contradicts the
>    proposal's "default so the change is opt-in" and it interacts with **D6** (multi-scale
>    opt-in vs default-on for SAM3): the YOLO side has now set a precedent for default-on
>    behaviour change. **D17.**
> 3. **It is also a SAMPLING change.** `ScaleGroupedBatchSampler` makes every batch
>    scale-homogeneous. That is not just a loss knob — it changes batch statistics
>    (BatchNorm, mosaic/mixup interaction, gradient noise) and it is invisible in the name
>    "loss balance". The unified contract should name it for what it is: a
>    *scale-grouped sampling + inverse-frequency weighting* policy.
>
> The proposal's one surviving requirement stands and is not yet implemented: **the build
> manifest must record the REALISED per-scale tile and positive-instance counts.** Today the
> manifest records the balance *settings* (`multiscale_loss_balance{enabled,power}`) but not
> the counts they were computed over **[V]** — so a reader cannot tell how skewed the build
> actually was, which is the number that decides whether `power=0.5` was enough.

**Can SAM3 reuse it? Split the module in two, and the answer is yes for one half.**

| Layer | Reusable by SAM3? | Why |
|---|---|---|
| `scale_group_for_path`, `scale_group_weights`, `ScaleGroupedBatchSampler` | **YES** — framework-agnostic. They operate on a list of paths and a list of indices; nothing imports torch or Ultralytics at definition time (only `_grouped_loader` does) | pure Python over stems and index lists |
| `_grouped_loader`, `install_sahi_multiscale_loss_balance` | **NO** | monkeypatches `YOLODataset`/`DetectionTrainer`/`DetectionModel.loss`, builds an `InfiniteDataLoader`, and *sniffs `data=` out of `sys.argv`* to find the manifest. All three are Ultralytics-shaped; SAM3 trains through its own sidecar loop |

**Recommendation:** promote the three pure functions into the shared layer (they belong
next to `utils/slice_geometry.py`, not inside a file named `ultralytics_*`), and leave the
installer where it is as one *adapter*. SAM3 would then write a second, small adapter over
its own dataloader. The pure/adapter split is the same shape as `slice_geometry` (shared
planner) vs the two builders — the pattern this whole document argues for.

**The filename convention is now a CROSS-PATH CONTRACT, and that is a real coupling.**
`_t{W}x{H}_{n}` / `_full` was provenance when §1.2 recorded it; `scale_group_for_path` has
made it a parsed interface. Consequences that must be written down before SAM3 adopts it:

- If SAM3 must emit `_t{W}x{H}_{n}` to reuse the grouping, then **a filename format is a
  training-behaviour dependency across two trainers**. It is recorded as §2.5 row 25.
- It is a **silent** interface: a stem that fails the regex falls through to group `"other"`
  at weight `1.0` — no warning, no count, and the balancing quietly does nothing for those
  images (new risk **R8**). SAM3's COCO records key on image ids, not stems, so this failure
  is *more* likely there, not less.
- **Preferred alternative for the unified design: carry the scale group as DATA, not as a
  parsed filename** — a `scale_group` field in the manifest / COCO image record, with
  `scale_group_for_path` retained only as the legacy fallback for datasets already built.
  That removes the coupling instead of spreading it. **D19.**

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

> **AMENDED `4de070f3` — does 3.5 still hold now that the user configures FRACTIONS?
> Yes, structurally; but the sidecar must record fractions, not only pixels.**
>
> The `training_geometry` (plural provenance) vs `profiles[]` (singular measured serving
> point) split is untouched by the new defaults and remains correct. What changed is where
> the authoritative number lives. Verified layering **[V]**: the user sets fractions in
> `SliceTrainingConfig.target_size_fractions`; `detectkit/jobs/training.py:220` resolves them
> to **pixels** via `target_sizes_for(imgsz_for(role))`; `SliceBuildParams` is pixel-fed;
> `_slice_geometry_manifest` stamps `target_sizes` (pixels) **and** `imgsz`; and
> `slice_meta._training_values` recovers `median(target_sizes) / imgsz`.
>
> So the round-trip is **lossless today by luck of composition, not by contract** — it works
> only because `imgsz` is stamped next to the pixels. Three corrections to item 1 below:
> **(a)** stamp `target_size_fractions` explicitly, as the thing the user actually chose;
> **(b)** keep the resolved `target_sizes` pixels AND `imgsz`, because "what was actually
> built" is a different question from "what was asked for" and a guard needs both;
> **(c)** never let a reader re-derive a fraction from pixels when an explicit fraction is
> present — that re-derivation is exactly the kind of implicit denominator that put a literal
> `640.0` in the tree. Also stamp `multiscale_loss_balance{enabled,power}` and the realised
> per-scale counts (§3.3): a model trained with scale-homogeneous batches and re-weighted
> loss is a materially different artifact and its sidecar should say so.

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

**AMENDED `4de070f3`: still required, and only half-addressed.** The hardcoded set is now
`(0.05, 0.10, 0.15, 0.20)` expressed relatively instead of `[200, 300, 400]` expressed
absolutely — a better *unit*, the same *un-calibrated status*, and one more entry. §2.8
item 1 spells this out. The corpus-derived resolver below is unchanged in intent; it should
now propose **fractions**, and it should be seeded from the corpus's measured major-axis
**distribution quantiles**, not only its median.

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

### 3.8 A shared calibration core (P3 + P4)

Not "merge the two harnesses" — they legitimately differ in what they run (a YOLO executor vs a
SAM3 labeler) and in what they can prompt. What should be shared is everything between the
predictions and the recommendation:

- **One matcher module** with the admissibility rule as an injected *policy*, not a hardcoded
  gate — so `hard_iou(0.5)` and `containment + area_band + graded_quality` are two named
  policies over one greedy one-to-one core. Both current matchers are already
  greedy-descending-score one-to-one; only the score and the gate differ.
- **One point record and one frontier/Pareto utility.** `DirectCalibrationPoint` and
  `CalibrationPoint` describe the same thing.
- **One objective module** exposing named rules (`f1_balanced`, `recall_first_cheapest`) with
  the reasoning attached, so choosing between them is a visible per-project decision rather
  than an accident of which detector family you happen to be calibrating.
- **One refusal vocabulary.** P4's typed refusals with remedies are strictly better UX than
  P3's silent drop; porting them costs nothing behaviourally.
- **One persistence target.** P4's result should land on the model artifact as a
  `.slice_meta.json`-shaped profile (row 22/23 of §2.5), not only in DetectKit project JSON —
  that is what gives SAM3 the `--sahi-profile` headless path P3 already has.

The shape prior (`shape_prior.py`) becomes available to P3 as an opt-in policy rather than
being SAM3-only. Whether to turn it on for direct calibration is D9.

### 3.9 THE COUPLING DIRECTION — a position, not a survey

**Position: TRAINING emits the scale set; CALIBRATION selects within and validates against it.
The arrow runs training -> calibration, never calibration -> training.**

The reasoning, in order of weight:

1. **Cost asymmetry is decisive.** A calibration candidate costs one inference pass; a training
   candidate costs a full training run (hours to days — the 2026-09-06 run was ten epochs
   overnight). Any design in which calibration *chooses* a training geometry implies training
   once per candidate, which is not a pre-run wizard. The direction that fits the cost
   structure is the only one that can ship.
2. **A calibration profile is measured on ONE model. A training scale set precedes every
   model.** Letting a profile fitted to model *v1* dictate the geometry of model *v2* bakes v1's
   idiosyncrasies into the corpus — and, worse, makes the two circularly coupled: v2 trained at
   v1's best geometry will calibrate to that geometry, confirming it. That is a feedback loop,
   not evidence.
3. **The eval spec's 2x2 says the fix belongs in training, not serving.** §11c: the model effect
   flips sign between geometries; each model wins decisively at its own. A single-scale model is
   *brittle* to serving geometry, and no amount of serving calibration repairs brittleness — it
   only finds the one geometry that works. **Multi-scale training makes the model robust; then
   calibration picks the cheapest point on a flat-ish surface rather than the only point on a
   sharp peak.** That is the structural argument for doing both, in that order.
4. **What training should read instead of a profile:** the *corpus*, not a prior model. The
   builder already measures the label body-size distribution (`object_major_axes_px`); the scale
   set should bracket that distribution and the project's frame size (§3.6). This is
   "calibrate before run" honoured with a measurement on the user's own data, at the only point
   where it is affordable.

**What breaks if they disagree, and what to do about it.** They will disagree — the whole point
of multi-scale is that serving may prefer a scale near an edge of the trained set, or outside
it. Three cases, and the design's answer to each:

- **Chosen scale INSIDE the trained set** — the intended case. Nothing to do; record which
  trained scale the profile sits nearest.
- **Chosen scale BETWEEN trained scales** — acceptable and expected; that is what multi-scale
  buys. Record it as interpolating.
- **Chosen scale OUTSIDE `scale_range_px`** — this is the 2026-09-06 confound, detected instead
  of silent. **Warn loudly, never refuse** (following the precedent already set at
  `semantic_escalation_dialog.py:382-406`, which deliberately warns because "a deliberate
  re-scale is legitimate"). The warning must state both numbers and the trained range; a
  deliberate re-scale is a legitimate user choice, an accidental one is the incident.

The corollary is a hard rule worth stating on its own: **a calibration recommendation is
evidence about serving, never an input to a dataset build.** If a project's calibrations keep
landing outside the trained range, the correct response is to *retrain with a scale set that
covers where calibration keeps going* — a human decision, informed by the recorded history,
made once, not an automatic loop.

### 3.10 Additional decisions this forces — for the user

> ## RESOLVED 2026-09-06 — D7, D8, D9 decided by the user. D12 decided by measurement.
>
> All three were put to the user with the evidence and their recommendations were
> accepted. **These are no longer open.** Implementers must follow them and must
> not re-litigate them; any future change needs new evidence, not a new opinion.
>
> **D7 — direct-path matcher: ADOPT CONTAINMENT, DROP THE IoU GATE.**
> The direct path unifies onto the semantic path's inside-guaranteed
> `representative_point` + containment matcher (merged `b9e92bc7`). Rationale: a hard
> `IoU >= 0.5` as the SOLE criterion counts a correct silhouette with a different
> extent convention as a miss AND an extra simultaneously, because masks trace legs
> and antennae at ~1.7x labelled body-core area. One matcher means "recall" is finally
> the same quantity in both harnesses.
> **Required:** a before/after gate on the same predictions, in the shape of the one
> that validated the semantic fix (recall 0.867 -> 0.988, identical predictions, only
> the scorer changing). **Accepted cost:** existing YOLO/OBB calibration
> recommendations WILL move, and stored profiles become rule-relative — see R6, which
> is why the recommendation rule must be versioned into profiles before this lands.
>
> **D8 — one objective: RECALL-FIRST TO RECOMMEND, AP TO COMPARE. F1 IS RETIRED.**
> Calibration recommends on recall-first with quality floors (the semantic path's
> approach). Model-vs-model comparison uses AP with paired frame-bootstrap CIs.
> Cross-project reporting uses extras/frame at a target recall, quoted with
> labels-per-frame density. F1 is retired as an OPTIMISATION TARGET (it may still be
> reported).
> Rationale, two independent lines agreeing: `semantic/calibration.py:9-13` already
> rejected F1 with its own measurement ("the F1-optimal threshold missed 4.7
> animals/frame where a recall-first one missed 1.0"); and the 2026-09-06 reliability
> study ranked F1 LAST for stability — paired effect size **0.91, 0/6 CIs surviving
> Bonferroni**, versus **AP 2.92**. The second finding arrives from insensitivity
> rather than recall-hostility, so the two arguments are genuinely independent.
> **Binding limitation:** AP is NOT comparable across corpora (measured 0.96 on
> 1766-px tiles vs 0.61-0.69 on 971-px, same models). Never quote AP across corpora;
> that is what the extras/frame reporting metric is for.
>
> **D9 — shape prior: ADD IT TO THE DIRECT PATH.**
> Port `fit_area_band` so both harnesses reject mistargeted detections identically.
> Without it the direct path can score a blob spanning two animals as a success. Share
> the code rather than duplicating it — this document exists because the tile-size
> formula reached five copies.
>
> **Sequencing note.** D7 and D9 both change what direct calibration MEASURES, and D8
> changes what it OPTIMISES over those measurements. Land them together behind one
> before/after gate, not as three separate silent shifts, or the resulting change in
> recommendations will be unattributable — the exact failure the 2026-09-06 geometry
> confound demonstrated.


> ## RESOLVED 2026-09-07 — D10 and D11 decided and implemented.
>
> Branch `feat/d10-d11-calibration-persistence`. No longer open; the D10 and
> D11 bullets further down are pre-ruling analysis only.
>
> **D10 — a semantic calibration result is persisted on the MODEL SIDECAR.**
> Shipped as `core/inference/semantic/calibration_record.py`: one named block
> `serving_calibration` on `<checkpoint>.sam3_meta.json`, reusing the
> `scale_grouped_batching` shape convention (a single named dict of plain
> JSON fields) rather than inventing a new top-level key family. The writer
> is strictly ADDITIVE read-modify-write and refuses (returns False, never
> raises, never truncates) on a missing/corrupt/non-object sidecar — because
> `semantic/sam3.py` REFUSES TO SERVE on a malformed sidecar, so a partial
> rewrite would be a hard outage for every published model. The BACK-COMPAT
> READER ships in the SAME COMMIT: `serving_calibration()` treats an absent
> block as "no claim", and `geometry_drift.stamped_tile_px_set` continues to
> read the scalar `train_tile_px: 971` that both already-published
> checkpoints carry. **Nothing on disk is backfilled or migrated** — new
> persistence applies going forward. A project whose calibration lives only
> in DetectKit project JSON keeps working and is reported through
> `resolve_serving_calibration` as `CalibrationOrigin.PROJECT_LEGACY`
> (surfaced in the dialog's status line), never silently copied onto a
> sidecar. `preview_artifact` is deliberately NOT carried onto the sidecar: a
> project-relative path would dangle on any other machine. Scope difference,
> recorded on purpose: the sidecar block is PER MODEL and the project copy is
> PER PROJECT, so two projects calibrating one model are last-write-wins on
> the sidecar while each keeps its own project record.
>
> **D11 — warn everywhere; refuse only against an explicitly named comparison
> baseline.** The guard is `core/inference/geometry_drift.py` (already Qt-free
> and already called from four headless sites); D11 adds
> `GeometryDriftRefusal` + `enforce_drift_verdicts(...,
> comparison_baseline=...)`. With no baseline named the behaviour is
> unchanged: warn, never refuse, on every surface. With a baseline named the
> run refuses on **MISMATCH and UNREADABLE**, and never on NO_STAMPED.
> *That is the one judgment call here:* naming a baseline is a request that
> the comparison be guaranteed, and UNREADABLE means the guard cannot verify
> the very thing the run named — the 2026-09-06 shape exactly — while
> NO_STAMPED must not refuse or every comparison against an older unstamped
> artifact breaks. This also gives the deliberate NO_STAMPED/UNREADABLE
> distinction observable teeth instead of collapsing it.
> **Reachability, stated plainly.** The two call sites that actually name a
> baseline (`training/sliced_dataset.py:255`,
> `training/sam3_lora/dataset_build.py:604`) were under another agent's file
> lock (D18) when this landed, so the refuse arm ships in core WITHOUT its
> natural caller: swapping their `log_drift_verdicts` for
> `enforce_drift_verdicts(..., comparison_baseline=...)` is a two-line
> follow-up. Separately, `comparison_baseline` exists in
> `detectkit/config/training.py` and flows to those builders, but **no
> DetectKit training GUI widget sets it**, so a GUI-launched build cannot
> reach the refuse arm today either. Both gaps are deferred items, not
> silent omissions.

> **SUPERSEDED for D7, D8 and D9 — read the RESOLVED block above instead.**
> The three bullets below are the PRE-RULING analysis, kept for provenance only.
> Their recommendations were NOT what shipped: the D8 bullet in particular
> recommends defaulting each path to its current rule "so nothing changes",
> and the opposite was decided and implemented (recall-first everywhere, F1
> retired as a target). Implemented on branch `feat/direct-scoring-unification`
> with a measured before/after gate at
> `tests/data/direct_calibration_golden/ATTRIBUTION.md`.
> D10 and D11 below remain genuinely OPEN.

- **D7 — direct-path matcher admissibility.** Adopt the semantic path's graded/containment
  policy for P3 (fixes the inverse defect of §2.7) or keep the hard IoU gate? Adopting it will
  change recommended profiles for existing projects. **Recommend adopting for `segment`, where
  the extent-convention mismatch is real, and offering it as a policy for `obb`/`detect`;
  either way, do not change it silently.**
- **D8 — one objective or two?** F1-balanced (P3) vs recall-first-cheapest (P4) are
  contradictory, and P4 has a measurement against F1. Unifying means one side's recommendations
  move. **Recommend: expose both as named rules, default each path to its current rule so
  nothing changes on adoption, and surface the choice with its rationale.**
- **D9 — shape prior for the direct path.** Off today. Turning it on catches mistargeting P3
  cannot currently see, but changes scores.
- **D10 — where a semantic calibration result is persisted.** Moving it from DetectKit project
  JSON onto the model sidecar is what unlocks headless SAM3 serving parity, and interacts with
  D2 (sidecar format) and with `semantic/sam3.py`'s refuse-on-malformed-sidecar guard.
- **D11 — the drift guard's severity and location.** Today: GUI-only, modal, warn-never-refuse
  (P4 only). Shared and headless, it must pick a non-modal channel and a severity. **Recommend
  warn-never-refuse everywhere, matching the existing precedent** — except for the one case the
  eval spec identifies, a run that explicitly names a comparison baseline, where refusing is
  defensible.

### 3.11 Decisions raised by the 2026-09-06 default changes — for the user

*(New. D13-D15 are taken by §6.6, so these start at D16. Nothing here re-opens D7/D8/D9.)*

**D16 — should SAM3 adopt `object_tile_fraction = 0.10`?**

*The situation.* YOLO training moved 0.15 -> 0.10 (`4de070f3`). SAM3's
`training/contracts.py:264` is still **0.055**. This is the exact quantity behind the
2026-09-06 confound: two SAM3 checkpoints each won at their own tile geometry and the model
effect **flipped sign** — **+3.25 extras/frame at 971 px** versus **-2.81 at 1766 px**, both
CIs excluding zero. The gap narrowed from 2.7x to 1.8x, by a change made on the other path,
without anyone deciding that SAM3's number should move.

*Position: **do not adopt 0.10 by spec fiat, and do not adopt it as a matched constant at
all.*** Three reasons, in order of force:

1. **The user's standing rule forbids it.** 0.10 is not a measured SAM3 optimum; it is a
   YOLO-side safety adjustment. Copying it across would substitute *numerical agreement* for
   *evidence* — and a shared wrong number is harder to detect than two honestly different
   ones, because the drift guard would then report agreement.
2. **The tree already contains a committed claim that they should differ.**
   `resolve_tile_px`'s docstring, written on the semantic path, states it "deliberately never
   reads `SliceTrainingSettings.object_tile_fraction`: the sliced-training optimum and the
   SAM3 optimum differ by ~3x, so one persisted fraction cannot serve both" **[V]**. That
   claim is itself under-evidenced, but it is a documented position and adopting 0.10 would
   silently contradict it. **Whichever way this goes, that docstring must be updated in the
   same change — two contradictory committed positions is the worst outcome.**
3. **The structural fix may moot the constant entirely.** §3.4's answer to the confound is
   *train multi-scale, then let serving calibration choose the operating geometry per
   project*. A model trained across a scale span does not need its single training fraction
   to match anyone's; that is the whole point. Spending a measurement campaign on aligning
   two scalars, when the design intends to replace both with a span plus a calibrated serving
   point, is effort against the grain of this document.

*What HAS already improved, and should be credited so this is not read as "nothing changed":*
the shipped drift guard plus `GeometrySource` logging (`b577ea6c`/`e3058ecb`) mean a build at
0.055 compared against a checkpoint served at 0.10 now **warns, and prints where each number
came from**. The 2026-09-06 incident's *silence* is fixed. The *divergence* is not.

*What evidence would settle it.* A paired ablation, pre-registered, on a corpus that is not
the ant corpus if one is available: arms = `{0.055, 0.10}` training fraction **x**
`{their own serving geometry, the other's serving geometry}` — the 2x2 discipline the eval
spec established, since a 1x2 is exactly what produced the sign flip. Statistic:
**extras/frame at a target recall**, paired frame-bootstrap CIs, quoted with labels-per-frame
density (per D8; AP is not comparable across the two tile geometries — measured 0.96 at
1766 px vs 0.61-0.69 at 971 px on the same models). Decision rule fixed in advance: adopt a
shared value only if the interaction term's CI excludes a practically-relevant effect, i.e.
only if the fraction genuinely does not interact with serving geometry. **Interim posture:
leave 0.055, make it loud (already done), and prioritise multi-scale SAM3 over aligning the
scalar.**

**D17 — is default-ON acceptable for a training-behaviour change?**
`306738ec` shipped `balance_multiscale_loss = True` (and scale-homogeneous batches) as the
default, so every future YOLO sliced build trains differently from every past one. §3.3
had proposed the opposite (opt-in). The precedent now cuts against **D6**'s framing for
SAM3 multi-scale. The user should confirm: (a) is default-ON the house rule for
strictly-better training changes, and (b) if so, does the sidecar stamp make old and new
artifacts distinguishable after the fact? (It does for the settings; not yet for the
realised counts — §3.3.)

**D18 — where does the fragment floor live?** YOLO's `min_area_ratio` is a per-build
parameter; SAM3's `MIN_RETAINED_AREA_FRAC` is a module constant. They now hold the same value
(0.25) and the same verified meaning (§2.3), but nothing keeps them in step and only one is
per-project. Recommend: the unified `TilingContract` carries **one** per-build floor, both
builders read it, and the SAM3 module constant becomes that field's default. This changes no
behaviour today and prevents the two from silently drifting apart again. (D3 — drop vs
downgrade — stays open and separate.)

> **RESOLVED (fix/d18-unified-fragment-floor).** No standalone `TilingContract` dataclass
> exists in the tree, so the "one per-build floor" lives where the code's existing pattern
> already puts shared tiling defaults: a new `DEFAULT_MIN_AREA_RATIO = 0.25` constant in
> `utils/slice_geometry.py` — the module both builders already import for tile planning
> (`plan_tiles`, `resolve_scales`, etc.), Qt-free and dependency-light. Each builder keeps
> its own per-build dataclass field (`sliced_dataset.SliceBuildParams.min_area_ratio`,
> `training/contracts.py Sam3LoraParams.min_area_ratio`, mirroring how `object_tile_fraction`,
> `tile_overlap`, etc. are already duplicated per-role rather than merged into one struct),
> but both fields now default from the one shared constant instead of each hard-coding
> `0.25` independently. `dataset_build.py`'s `MIN_RETAINED_AREA_FRAC` module constant
> becomes `DEFAULT_MIN_AREA_RATIO` (same value) and the builder's tiling functions
> (`_tile_frame`, `_scaled_frame_jobs`) now read `params.min_area_ratio` instead of the
> module global, so a per-project override actually reaches the measurement. The drop-vs-
> downgrade policy split (D3) is untouched: YOLO's `_tile_one_image` still drops a sub-floor
> instance; SAM3's `_tile_frame` still keeps it and flags `is_crowd`. Guarded by
> `tests/test_d18_fragment_floor_unification.py` (characterization, not fail-first — D18 is
> explicitly behaviour-neutral): one test pins both dataclasses' field defaults to the same
> upstream constant, one proves the policy divergence survives. Existing
> `tests/test_sam3_dataset_build.py` and `tests/test_sliced_dataset.py` pass unmodified.

**D19 — scale group as data or as a filename?** `scale_group_for_path` parses
`_t{W}x{H}_{n}` / `_full`. Reusing it for SAM3 makes a filename convention a cross-trainer
contract with a silent failure mode (R8). Recommend carrying `scale_group` explicitly in the
manifest / COCO image record, with stem-parsing kept only as the legacy fallback. Cheap now,
expensive after a second consumer exists.

> **RESOLVED — already satisfied by the multi-scale SAM3 port, merged `50cb5b94`.**
> Verified directly in this tree: `dataset_build.py` writes `record["scale_group"] =
> scale_group` onto the COCO image record for every multi-scale emission, and
> `dataloader.py`'s `_load_dataset` reads it straight back as data —
> `scale_group=str(image_meta.get("scale_group", "") or ""),` — with no filename parsing on
> that path. `scale_group_for_path` (in `training/scale_balance.py`, re-exported by
> `training/ultralytics_scale_balance.py`) is imported only by the Ultralytics/YOLO
> scale-balance path and by tests (`test_ultralytics_scale_balance.py`,
> `test_scale_balance_shared.py`, `test_sam3_multiscale_build.py` — the last uses it only to
> cross-check the emitted filename token against the `scale_group` DATA field, not as SAM3's
> source of truth). That is exactly the "legacy fallback" surface the recommendation
> anticipated: a real residual, confined to the YOLO/Ultralytics side, that a stem-parse
> failure there cannot corrupt because SAM3 never round-trips through it.

---

## Part 4 — Ranked plan, risks, and the full-frame question

### 4.1 Build order

**Step 0 — EVIDENCE PRESERVATION, ahead of everything else (Part 6).** Per-epoch validation
series recorded unconditionally, and a retention *budget* replacing `KEEP_EPOCH_CHECKPOINTS = 3`.
This is promoted above the drift guard for one reason, and it is not that it is more important:
**it is the only item whose cost of delay is irreversible.** Every training run that happens
before it lands destroys evidence that cannot be recovered — the 2026-09-06 run's `epoch_001`
and `epoch_002` are already gone, unmeasured. The drift guard prevents a *future* mistake and
loses nothing by waiting a week; a pruned checkpoint is gone. Step 0 also has no design
dependency on D12 (§5.4) — recording is unconditional — so it can proceed while the metric
study runs, and it is a precondition for that study ever being repeatable. It changes no
behaviour, blocks nothing, and is measured in hours.

1. **[SHIPPED 2026-09-06 — `b577ea6c`, fixed by `e3058ecb`]** **ONE shared, Qt-free
   geometry-drift guard + provenance logging — built once in core, not
   four times.** Delivered as `core/inference/geometry_drift.py` with (a)-(d) all in place;
   see the SHIPPED note under F-E for what landed and for the inert-guard defect
   `e3058ecb` fixed. Revised in light of all four paths: the guard already exists, correct and
   well-reasoned, at `detectkit/gui/dialogs/semantic_escalation_dialog.py:382-406` — but it is
   inside a Qt dialog, so it cannot serve P1, P2, P3 or any headless run (§0 F-E). The step is
   therefore *extract, then apply four times*, not *write four guards*:
   (a) lift the compare-stamped-vs-effective logic into `core/` (near `slice_meta.py`, which
   already owns the read side) as a pure function returning a typed verdict, no Qt;
   (b) re-point the existing dialog at it, preserving today's warn-never-refuse behaviour
   verbatim so P4's user-visible behaviour does not move;
   (c) call it at SAM3 and YOLO dataset-build time — including against a named comparison
   baseline's sidecar, which is the exact miss that produced the 2026-09-06 confound;
   (d) call it on the serving side where `--sahi-profile` resolves.
   Alongside it, log the effective geometry AND its *source* (explicit / profile / corpus-
   derived / contract default) at the start of every build and every run — the 0.055 incident
   was undetectable precisely because the source was never printed.
   This is still the cheapest item, it is now also the one that most directly demonstrates the
   unification, and it requires no contract change.
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

1b. **[SHIPPED 2026-09-06 — `011a34b2`, characterized first by `a153e86a`]** **Delete the
   fifth tile-size formula** (`semantic/tiling.py` -> `tile_size_for_mode`).
   Numerically inert today, and it is the precondition for P4 ever inheriting a shared change.

**Steps 6 and 7 are amended by the 2026-09-06 default changes:**
- **Step 6** is *partially* done: `4de070f3` replaced the absolute `[200,300,400]` with a
  relative `(0.05,0.10,0.15,0.20)` in all copies, which fixes the *unit* but not the
  *derivation*. The corpus-derived proposal (§3.6, now proposing fractions and seeded from
  distribution quantiles) is still outstanding, and R4 grew.
- **Step 7 is no longer deferrable and no longer says `none` is the shipping default.**
  `306738ec` shipped scale-grouped sampling + inverse-frequency loss weighting, default-ON,
  for YOLO. The step becomes: *promote the three pure functions out of
  `training/ultralytics_scale_balance.py` into the shared layer, stamp the realised per-scale
  counts, and decide D17/D19 before SAM3 grows a second adapter.*

Steps 5-8 renumber unchanged. Two additions to the tail:

9. **Unified objective (D12)** — blocked on the metric-reliability study, not on engineering.
   When it resolves, it lands as a shared objective module (§3.8) with the chosen rule as the
   default and the current rules retained as named alternatives.
10. **Per-epoch detection-quality stopping (D13)** — blocked on 9, and correctly deferred: an
    inference pass per epoch bought to compute an unreliable signal is a net loss.

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
  **AMENDED `4de070f3`: this risk INCREASED on the YOLO side.** Raising `min_area_ratio`
  0.1 -> 0.25 means 2.5x more of the seam-crossing instance area is now required for an
  instance to survive, so proportionally more instances are dropped — and the drop rate is
  highest at the finest scale, which is also the scale that produces the most tiles. Moving
  to a 4-entry scale ladder with a finer minimum compounds it in the same direction. The
  change is defensible on its own terms (do not train on severely clipped animals) and the
  runbook says so; but the per-scale counters this risk asks for are **still not
  implemented** and are now more necessary than when the risk was written.
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
- **R5 — unifying the calibration harnesses moves recommended operating points.** D7/D8/D9 each
  change what gets recommended for projects already calibrated. A user who re-opens a saved
  calibration after the change may see a different recommendation over identical evidence.
  Mitigation: keep per-path defaults on adoption, version the recommendation rule into the
  stored profile, and display which rule produced a stored point.
- **R6 — resolving the objective (D12) invalidates stored calibration profiles' provenance.**
  Every `.slice_meta.json` profile in existence was chosen under a rule that may not be the
  chosen one. They remain valid *settings* but their "measured best" claim becomes rule-relative.
  Mitigation: version the recommendation rule into the profile record (already recommended in
  §5.5) BEFORE D12 resolves, so old profiles are identifiable rather than silently re-interpreted.
  **AMENDED — the ordering this mitigation depends on has EXPIRED.** §3.10 records D8 as
  decided (recall-first to recommend, AP to compare, F1 retired) and D12 as decided by
  measurement. Every stored profile in existence therefore *already* predates the chosen
  rule, and "version it in before D12 resolves" is no longer available. Re-stated mitigation:
  **rule-versioning must land in the same change as D7/D9**, and un-versioned legacy profiles
  must be labelled `rule: unknown (pre-2026-09-06)` rather than back-filled with an assumed
  rule — back-filling would assert provenance that does not exist. The risk itself is not
  reduced by the amendment; only its remedy changed shape.
- **R4 — AMENDED, and it got WORSE, not better.** ~~Three duplicated copies of `target_sizes`
  plus the literal `640.0`.~~ `4de070f3` turned one duplicated field into a duplicated
  *pair*: `target_size_fractions` **and** `target_sizes` now co-exist at
  `detectkit/config/training.py`, `detectkit/gui/models.py` and (pixels only)
  `training/sliced_dataset.py`, with a fourth fraction default in
  `slice_settings_widget.py::_TileLayoutPreview` **[V]**. The precedence rule
  ("fractions win; fall back to `target_sizes / 640.0`") is implemented **twice**, in two
  `target_fractions()` methods that are near-identical but not identical — the GUI copy
  additionally filters fractions to `(0, 1]` before deciding whether any exist, the config
  copy validates instead **[V]**. So a "single" change is now really five, and two of the
  five can disagree about whether a malformed fraction list counts as present. Mitigation
  unchanged in kind: one shared resolver in the unified contract, both call sites deleted.
  The `640.0` literal is no longer a defect (it is the correct legacy denominator) but is
  itself duplicated.
- **R7 (new) — the same manifest trains differently under DDP.**
  `install_sahi_multiscale_loss_balance` returns the unmodified loader when `rank != -1` or
  `WORLD_SIZE > 1`, logging a warning **[V]**. The decision is defensible and documented
  (a single-process sampler cannot partition scale groups across ranks without changing epoch
  exposure), but the consequence is that **a build manifest with
  `multiscale_loss_balance.enabled = true` produces a balanced model single-GPU and an
  unbalanced one under DDP, with nothing distinguishing the two artifacts afterwards.** For a
  programme whose whole failure mode is unattributable model differences, that is a
  first-class hazard. Mitigation: stamp the *realised* balance state (not the requested one)
  into the sidecar, and treat a DDP run of a balance-enabled plan as a distinct arm in any
  comparison.
- **R8 (new) — the scale-group regex fails silently.** `scale_group_for_path` returns
  `"other"` (weight 1.0, its own batch group) for any stem that does not end in
  `_t{W}x{H}_{n}` or `_full`. Nothing counts or reports how many images landed in `other`
  **[V]**. A renamed file, an augmentation that rewrites stems, a future builder that changes
  the token, or SAM3 adopting the convention imperfectly all degrade balancing to a no-op
  without a single log line. Mitigation: log the per-group census at install time and warn
  when `other` is non-empty — and prefer D19 (carry the group as data).

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

**AMENDED `306738ec` — the full-frame arm's effective weight is no longer what
`full_frame_mix` implies.** `scale_group_weights` pins `full` (and `other`) to weight `1.0`
while *up-weighting* every under-represented tile group **[V]**. The stated intent is
benign — "full-frame examples remain at weight one, preserving their configured mix" — but
the mix that is preserved is the *count*, not the *loss share*: because tile groups can be
weighted above 1.0, the full-frame arm's share of total gradient falls relative to an
unbalanced run, by an amount that depends on the realised per-scale counts and on `power`.
Anyone running the ablation this section specifies must therefore hold
`balance_multiscale_loss` and `power` fixed across arms, and record them, or the full-frame
comparison is confounded by the balancing policy. One more argument for stamping the
realised per-scale counts (§3.3).

---

## Part 5 — ONE OBJECTIVE, USED UNIFORMLY (requirement)

### 5.1 The requirement, and why this divergence is the worst one

**Requirement A: one objective function serves all three consumers — calibration
recommendation, checkpoint selection, and model comparison (within a project, across projects,
and across species).**

Today there are three, and they are not merely different, they are *contradictory*:

| Consumer | Objective today | Where |
|---|---|---|
| Direct (YOLO) calibration | **maximise F1**, then fastest within `F1_TOLERANCE = 0.01` on the (misses, extras, seconds) Pareto frontier | `direct_calibration.py:186-191, 247-283` **[V]** |
| Semantic (SAM3) calibration | **recall-first**: among points clearing `MIN_RECALL = 0.90`, fewest `tiles_per_frame`; **F1 explicitly rejected with a measurement** — "The F1-optimal threshold missed 4.7 animals/frame where a recall-first one missed 1.0" | `semantic/calibration.py:9-13, 606-656` **[V]** |
| SAM3 checkpoint selection | **none — always the last epoch**; validation loss is computed once, after `adapters.pt` is written, and is documented "for reporting ONLY. Never influences checkpoint selection" | `sam3_lora/cli.py:762-775, 778-796` **[V]** |
| Model comparison (the eval harness) | paired extras/frame at a target recall, plus AP, reported side by side | `tools/sam3_parity/compare_models.py` (per the 2026-09-06 eval spec §11) |

**Why this is worse than any tiling divergence.** Every divergence in Part 2 is a difference in
*what runs*; those can be reconciled by sharing code, and a shared contract makes them go away.
A divergence in the objective is a difference in **what "better" means**, and sharing code
cannot reconcile it — two harnesses that disagree about the goal will keep giving opposite
advice over identical evidence, correctly, forever. Concretely, today: P3 would recommend an
operating point that P4's rule rejects as under-recalling, and P4 would recommend one P3's rule
rejects as sub-optimal F1; both would be "right". And because their *matchers* also differ
(§2.6), the recall each reports is not even the same quantity — **no number produced by one
harness is comparable to a number produced by the other**, which silently invalidates every
cross-path comparison a user might reasonably make.

The same logic extends to checkpoint selection and to cross-species comparison. If the metric
that picks a checkpoint is not the metric that picks an operating point, a run can be tuned to
win on one and lose on the other. If the metric is not comparable across corpora, "model A is
better than model B" is not a statement that survives changing the evaluation set — which is
exactly the trap the 2026-09-06 evaluation fell into.

### 5.2 The two criteria a unified objective must satisfy

1. **Reproducible separation.** It must distinguish genuinely different models/checkpoints by
   more than run-to-run noise — i.e. the gap between arms must exceed a bootstrapped confidence
   interval on the metric itself. A metric that cannot separate is not an objective, it is a
   number.
2. **Comparability across corpora.** The same model evaluated on two renderings of the same
   underlying frames must produce comparable values, otherwise the metric measures the corpus
   as much as the model.

**Per-tile AP demonstrably fails criterion 2.** The 2026-09-06 eval measured both models at
**~0.96 on the 1766-px tile corpus and 0.61-0.69 on the 971-px corpus** — the same models, the
same underlying 16 frames, differing only in how they were cut up. The eval spec states the
mechanism plainly: the larger-tile corpus is an easier scoring regime (fewer, larger tiles; 147
vs 576) and "the 0.96 figures must never be compared against the 0.61-0.69 ones". Any tile-level
metric inherits this, because tiling is a free parameter of the *evaluation*, not a property of
the model. **This is the single strongest constraint the unified objective must satisfy, and it
argues that whatever is chosen must be computed on MERGED FULL FRAMES, not on tiles.**

### 5.3 Candidates, stated without a winner

| Candidate | Separation | Cross-corpus comparability | Notes |
|---|---|---|---|
| AP / area under PR (per tile) | good in practice | **FAILS** (§5.2) | disqualified as the *unified* objective on comparability alone; may survive as a within-corpus diagnostic |
| AP on merged full frames | unmeasured | plausible — removes the tiling free parameter | the natural repair of the above; cost is a merge pass per evaluation |
| Extras-per-frame at a target recall, on merged full frames | this is the pre-registered statistic the eval already uses, with paired CIs | plausible, and it is per-frame by construction | matches the semantic path's recall-first philosophy; needs a target recall, which is a **per-project** choice, not a constant |
| F1 at the best threshold | good | plausible | but `semantic/calibration.py:9-13` has a measurement against it as an *operating-point* rule; using it for comparison while rejecting it for selection would be its own incoherence |
| Recall at a fixed extras budget | the dual of the above | plausible | arguably the most operator-legible framing ("how many animals do I find, for a fixed amount of clicking") |

**No number in this table is proposed as a default, and no target recall or extras budget is
proposed at all** — both are per-project quantities that depend on animal density, label
quality and how much proofreading the operator will tolerate. What the unified objective fixes
is the *shape* of the question, not its parameters.

### 5.4 The decision is OPEN, and keyed to a measurement in flight

**D12 — which objective.** Deliberately unresolved here. A metric-reliability study is running
that bootstraps the AP confidence interval and ranks the candidates by stability; the choice
should be made on its output, not on this document's reasoning. *(Note: no file exists at
`.superpowers/sdd/sam3-metric-reliability.md` in this tree as of `097408af` — the study is
running outside the repo or has not yet landed. Whoever resolves D12 should attach its result
here.)*

**The live possibility that must not be designed away: the whole checkpoint ladder may be
within noise.** The 2026-09-06 ladder spread was AP 0.5985-0.6379 across four arms, and the
eval spec already flags that the ordering is *non-monotonic* in training duration and is "at
least as consistent with substantial epoch-to-epoch variance in the adapter" as with any real
effect. If the bootstrap shows the CI on AP is wider than that spread, then:

- **Checkpoint selection is unmotivated** — there is nothing to select on, and last-epoch is as
  defensible as anything else. The correct response is to say so and stop, not to invent a rule.
- **The objective question is still answered, for calibration.** Calibration compares operating
  points of ONE model on ONE corpus, which is a paired within-model comparison with far more
  favourable noise properties than cross-model comparison. A metric too noisy to rank
  checkpoints can still be perfectly adequate to rank tile fractions.

That asymmetry should be stated in whatever ships: **one objective FUNCTION, but the evidence
needed to act on it differs by consumer**, and the honest position may be "unified for
calibration, and explicitly declined for checkpoint selection until a study says otherwise".

### 5.5 What this means for D8

D8 (§3.10) asked whether to keep two recommendation rules or unify. **Requirement A resolves the
direction: unify.** What it does not resolve is *onto what* — that is D12. The interim posture
stands (expose both as named rules, default each path to its current rule so adoption changes
nothing, record the rule into the stored profile), but it is now explicitly a **bridge**, not a
destination.

---

## Part 6 — RECORD PER-EPOCH VALIDATION, UNCONDITIONALLY (requirement)

### 6.1 The requirement

**Requirement B: every training run records a per-epoch validation series, and retains every
epoch checkpoint, INDEPENDENT of whether anything ever selects on them.**

This is an evidence requirement, not a selection mechanism, and it must not be argued for or
against on selection grounds.

### 6.2 What exists today

- `_evaluate_and_write` runs **once**, after the final epoch, strictly after `adapters.pt` is
  written, and its docstring states it is "for reporting ONLY. Never influences checkpoint
  selection" (`sam3_lora/cli.py:762-775, 778-796`) **[V]**. **The series does not exist.**
- `KEEP_EPOCH_CHECKPOINTS = 3` (`cli.py:267`); `prune_epoch_checkpoints` deletes all but the
  newest three after every epoch write (`cli.py:270-287, 289-298`) **[V]**.

### 6.3 Why the absence is itself the bug

On the 2026-09-06 run, training loss stopped improving after epoch 2 and ten epochs bought
nothing measurable over three — **and nothing in the artifacts showed it while the run was
happening or after it finished.** A single terminal validation number cannot distinguish "the
model converged" from "the model stalled at epoch 2 and burned eight epochs of GPU time". The
series is the only artifact that can, and it is the cheapest possible diagnostic: it is a
number per epoch.

Worse, the retention cap **has already destroyed evidence**: `epoch_001` and `epoch_002` were
pruned before they could be evaluated, on the very run whose most informative result turned out
to be that `epoch_003` was the best of the four surviving arms. The best checkpoint of the run
may have been deleted, unmeasured, by a constant.

### 6.4 Retention — and the real constraint, stated fairly

Adapters are ~45.8 MB, so retaining all ten epochs of that run costs ~460 MB — negligible
against a 3.4 GB published artifact. **But the existing docstring's rationale is not silly and
must not be dismissed:** it says "epoch counts are user-supplied and disk exhaustion mid-run
would destroy the very artifact this feature exists to preserve" (`cli.py:273-276`) **[V]**.
That is a real failure mode for a 200-epoch run on a full disk.

So the fix is **not "delete the cap" but "replace a count with a budget"**: retain every epoch
subject to a disk-space budget, with the budget expressed against measured free space and the
measured adapter size (both knowable at run start, neither a corpus-derived constant), and log
loudly whenever the budget forces a prune. A count of 3 is a constant that silently destroys
evidence; a budget that refuses to fill the disk and *says so* preserves both properties. If
the budget cannot hold every epoch, prune by a stated policy (e.g. thin the middle, keep first
and last) rather than a sliding window that always destroys the early epochs — which is
precisely the failure that occurred.

### 6.5 Early stopping — framed correctly

**Early stopping is industry-standard practice and this document does not argue against it.**
The 2026-09-06 finding that validation loss anti-correlates with held-out AP is an argument
against **the signal**, not **the technique**.

The standard-practice answer is already visible in a dependency this repo ships: **Ultralytics
YOLO stops on `fitness`, a weighted mAP — a detection-quality metric — not on validation loss.**
That is the correct shape for SAM3 too: a per-epoch *detection-quality* evaluation, not a
per-epoch loss.

**Its cost, stated honestly:** an inference + merge + match pass per epoch. On the 2026-09-06
validation geometry that is 147 tiles per epoch. Whether that cost is worth paying depends
entirely on D12 — if the metric cannot separate checkpoints reliably (§5.4), then paying an
inference pass per epoch to compute an unreliable stopping signal is worse than not paying it.
**So: record the series unconditionally (Requirement B, cheap, no decision needed); decide
whether to STOP on it only after the reliability study (D13).**

**The historical irony, preserved deliberately.** The original rationale for last-epoch
selection claimed a val-loss/AP anti-correlation and was dismissed *because its fold had
train == valid*. On a genuinely disjoint split, the anti-correlation was measured again. The
dismissal was methodologically correct at the time and the conclusion still came back. That is
worth keeping in the record, both as evidence and as a caution: a correct procedural objection
to a finding is not a refutation of the finding.

### 6.6 Decisions

- **D13 — stop on a per-epoch detection metric?** Keyed to D12. Not before.
- **D14 — retention budget policy** (§6.4): budget basis, and the thinning policy when the
  budget binds. Affects every trainer, not just SAM3 — **P1/YOLO retention should be checked
  for the same class of constant** before this is called done.
- **D15 — does Requirement B apply to YOLO training too?** It should, for the same reasons.
  Ultralytics writes its own per-epoch `results.csv`, so the gap may already be closed on that
  side; **unverified in this audit** and worth confirming rather than assuming.
