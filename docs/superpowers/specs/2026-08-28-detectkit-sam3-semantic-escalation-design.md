# DetectKit SAM3 Semantic Escalation — Design

**Date:** 2026-08-28 (revised 2026-08-29 after adversarial review)
**Status:** Draft, pending review
**Branch:** `feat/sam3-semantic-escalation` (from `main` @ 04da82fa)

## Summary

Add **Semantic escalation** to DetectKit: the user types a noun phrase ("ant"), SAM3
segments every matching instance across a source's frames, and the result is staged for
review. Where the source already has labelled frames, a **calibration** pass measures
SAM3 against those labels and recommends a confidence threshold, so the operating point
is fitted to the user's data rather than inherited from ours.

The two escalations are complementary and both are kept. SAM2 escalation *converts*
geometry the user already has — OBB/AABB boxes become segmentation masks, one output
instance per input box — and cannot invent a label that isn't there. SAM3 semantic
escalation *finds* instances from a prompt, including animals missing from the existing
labels entirely. Accelerating the box-to-mask transition and recovering missed animals
are different jobs, so they stay different actions.

## Scope

**In scope**

- A `SemanticLabeler` seam and a SAM3 backend behind it.
- Tiled inference, reusing `utils/slice_geometry.py` for the grid.
- Calibration against the source's existing labelled frames, at any geometry level.
- A dialog: prompt, tiling controls, calibrate, single-tile preview.
- Staging + review, promoting to a **new sibling source** (see *Promotion*).
- Renaming the SAM2 action so the two escalations are tellable apart.

**Out of scope**

- Image exemplars as prompts (measured worse than text — see *Evidence*).
- SAM3 fine-tuning (possible via the official repo; better motivated once a project has
  a reviewed frame set, and untestable from the frames this feature starts with).
- Automatic prompt search (wording measured near-irrelevant next to tiling).
- Multi-class prompting. One run = one prompt = class `0`.
- Any change to TrackerKit, `core/inference` detection stages, or cache keys. The
  TrackerKit equivalence matrix is **not** a gate for this work.
- Any change to SAM2 escalation's behaviour. It is renamed, not touched.
- Any change to the X-AnyLabeling round trip (`dataset_panel.py:_prepare_xal_stage`).

## Architecture

### Promotion: a sibling source, not an in-place overwrite

SAM2's `accept_pending_escalation` (`jobs/sam2_escalation.py:299`) does
`rmtree(source/labels)` then `copytree(staged/labels, source/labels)` and sets
`source.level`. That is correct for SAM2, whose staged labels are a lossless upgrade of
the *same* instances.

It is wrong for SAM3, and this is the review finding that most shapes the design.
SAM3's staged labels are a different instance set, a different geometry convention
(masks trace legs and antennae; tracking labels bound the body core — median 1.7x area),
and all class `0`. Reusing SAM2's accept would silently delete a user's curated OBB
labels — exactly the harm this spec cites as its reason not to merge the two
conventions.

Therefore semantic escalation stages like SAM2 but **promotes differently**: accepting
writes a **new sibling source** (`<name>-sam3-<prompt-slug>`, level `polygon`,
`reviewed=False`) and leaves the original source untouched. The user then keeps, merges,
or deletes it with the tools they already have. Concretely, semantic escalation gets its
own `accept_pending_semantic_escalation` in `jobs/semantic_escalation.py`; SAM2's accept
path is not modified, and the two are dispatched on `PendingEscalation.primer_kind`.

Consequence: **`run_semantic_escalation` must not inherit SAM2's
`level != "polygon"` filter** (`sam2_escalation.py:177-181`). Running a prompt against a
polygon-level source to find animals the polygons missed is a primary use case, and
under SAM2's filter it is silently dropped without even a `skipped` entry.

### Staging

Reuse the SAM2 staging mechanics: write into
`artifacts/pending_escalations/<dirname>/`, record a `PendingEscalation` on the source,
and extend `review_escalations_dialog.py` with a second producer.

`PendingEscalation` needs generalising — it currently hardcodes `sam2_variant`
(`gui/models.py:14-20`). Add `primer_kind` ("sam2" | "sam3"), `primer_variant`,
`primer_prompt`, and `primer_params`, with `from_dict` back-filling `primer_kind="sam2"`
and `primer_variant` from a legacy `sam2_variant` key so existing projects load with no
migration. (`to_dict`/`from_dict` are hand-written with `d.get` defaults at
`models.py:22-39`, and `DetectKitProject.load`'s coercion loop never touches nested
source dicts, so this is safe.)

Two staging details SAM2 gets right that semantic escalation must adjust:

- The staging dirname hash is `sha1(str(src_root) + variant)` (`:198-201`). **The prompt
  must enter the hash**, or two prompts on one source collide and the
  replaced-pending cleanup at `:212-213` no-ops.
- `remove_staged_escalation_dir(staged_root)` is called unconditionally before writing
  (`:215`). Resumability (below) needs this to become conditional on a
  fingerprint match.

### The seam

```
core/inference/semantic/
    base.py          # SemanticLabeler protocol + SemanticInstance
    checkpoints.py   # SAM3 catalog + HF download + availability probe
    sam3.py          # Sam3SemanticLabeler
    tiling.py        # seam-drop + cross-tile merge (grid comes from slice_geometry)
    calibration.py   # sweep confidence against labelled frames
```

```python
@dataclass(frozen=True)
class SemanticInstance:
    polygon_px: np.ndarray   # (P, 2) float32, in the coordinate space of the
                             # image passed to label_image -- tile-local under
                             # tiled inference. tiling.py offsets to frame space.
    confidence: float

class SemanticLabeler(Protocol):
    @property
    def name(self) -> str: ...
    def label_image(self, image_bgr, prompt, *, confidence_threshold,
                    max_instances=0) -> list[SemanticInstance]: ...
                    # max_instances=0 means unlimited.
```

`Sam3SemanticLabeler` wraps `SAM3SemanticPredictor` from `ultralytics.models.sam`
(verified present in the installed 8.4.34). Checkpoints follow `sam2/checkpoints.py`:
pinned catalog, `hf_hub_download` into `get_models_dir() / "sam3"`, offline error path.
The catalog entry pins the HF repo id **and filename**, as `SAM2_VARIANTS` does.

`sam3.pt` is 3.45 GB and is **not** in ultralytics' `GITHUB_ASSETS_NAMES` (verified), so
it comes from the public `facebook/sam3` HF repo. Ultralytics AutoUpdate pip-installs
`clip` and `ftfy` on first use, which is unacceptable for an offline or shared install:
both are declared in a `sam3` extra, and the probe must fail loudly rather than let
AutoUpdate run.

**The availability probe is new code, not the SAM2 pattern.** `available_variants()`
(`sam2/checkpoints.py:48-49`) just returns `list(SAM2_VARIANTS.keys())` — a static dict
— so the tools-panel guard at `panels/tools_panel.py:172-181` never checks anything, and
its "install the SAM2 checkpoints" tooltip is already misleading. Reusing it would give
exactly the silent 3.45 GB download this spec forbids. `semantic/checkpoints.py`
provides `probe_availability() -> tuple[bool, str]` checking, in order:
`importlib.util.find_spec("ultralytics")` and the `SAM3SemanticPredictor` symbol,
`find_spec("clip")` and `find_spec("ftfy")`, and `checkpoint_path(variant).exists()` —
returning a reason string the tooltip shows. A missing checkpoint disables the button
with "download it from the dialog", not a click-time surprise. Fixing the SAM2 tooltip
is out of scope but should be noted in the PR.

Device selection reuses the existing cuda -> mps -> cpu picker rather than duplicating
it: move it from `sam2/executor.py:13-18` to `core/inference/torch_device.py`, leaving
`resolve_sam2_device` as a thin alias. Note this **does** break two existing tests:
`tests/test_sam2_executor.py:9-18` monkeypatches `executor.TORCH_CUDA_AVAILABLE` and
would no longer affect the aliased function. Those tests are repointed at the new module
in the same commit — the spec's earlier claim that `sam2/` "keeps working unchanged" was
wrong.

Similarly, `mask_to_contour` moves to `core/inference/masks.py` — and **so does
`clip_mask_to_polygon`**, its only module-mate (`sam2/masks.py` is 48 lines, two
functions). Import sites: `jobs/sam2_escalation.py:17` and `tests/test_sam2_masks.py:3`,
both of which import both names. Move the module wholesale; don't split it.

### Job result and prompt failure

```python
@dataclass
class SemanticEscalationResult:
    staged: list[str] = field(default_factory=list)
    labelled: int = 0          # instances staged
    empty_images: int = 0      # frames where the model returned nothing
    degenerate: int = 0        # contours with P < 3, dropped not fatal
    tile_px: int | None = None # resolved tile size, None = full frame
    skipped: list[tuple[str, str]] = field(default_factory=list)
```

`empty_images` is load-bearing, not a statistic. The dominant failure mode of this
feature is a noun phrase the model does not match, and that failure is silent: the run
completes, stages nothing, and looks like success. **A run whose `empty_images` is a
majority of the frames processed must be reported as a prompt failure**, with the
completion dialog saying so and suggesting the prompt be retried in the preview. Zero
instances staged is never a green result.

`degenerate` counts contours with fewer than 3 points. These are dropped and counted
rather than passed to `write_label_file`, whose `_polygon_points` raises on them
(`data/al/labels.py:45-50`, uncaught in `run_escalation`) and would otherwise abort a
multi-hour run over one bad contour.

`skipped` carries `(source_name, reason)` pairs, mirroring `EscalationResult.skipped`
(`jobs/sam2_escalation.py:106-114`) — primarily sources that already hold a pending
escalation when `overwrite` was not requested.

### Tiling

SAM3 letterboxes its input to 1008 px. An animal occupying 2% of a high-resolution frame
is ~18 px to the model, and detection collapses. Tiling is what makes the feature work —
and on a rig where animals are already large at native resolution, tiling would instead
*hurt*, which is why the tile size is derived from object scale rather than fixed.

**The grid comes from `utils/slice_geometry.py`, not from new code.** Verified: it
provides `tile_size_for_mode(geometry_mode="auto_object", ...)` computing
`reference_body_px / object_tile_fraction` (`:104-128`), `plan_tiles` returning
frame-space `(x0, y0, x1, y1)` tiles with overlap and last-tile-flush-to-edge (`:42-44`),
and `MAX_TILES_PER_FRAME = 4096` (`:20`).

**Semantic escalation gets its own fraction, not `SliceTrainingSettings`'s.** Sharing
`reference_body_px` is right; sharing the *fraction* is not.
`SliceTrainingSettings.object_tile_fraction` defaults to `0.15` (`gui/models.py:119`),
which at the measured `reference_body_px = 80` yields a 533 px tile — essentially the
tile-752 configuration Evidence row 2 shows costs 5.7x for no gain. The two consumers'
optima differ ~3x, so a single persisted value cannot serve both.

**The fraction is calibrated, not asserted.** An earlier draft of this spec derived a
fraction of 0.05 from a claim that SAM3 needs an object to reach ~50 px at its 1008 px
input. That derivation was circular: the only tile size ever measured good was 1504 px at
`reference_body_px = 80`, and `80 * 1008 / 1504 = 53.6`. The "50 px" was the measurement
rewritten in different units, and the 1504 it then "predicted" was its own input. Only two
tile sizes were ever tested, on one dataset, and no object-size-at-model-input sweep was
run. There is no independent grounding for that constant and the spec does not assert one.

What *is* defensible is the scale-invariance: tile size should track object size rather
than be a fixed pixel count, so a fraction transfers across resolutions better than a raw
tile size would. The specific value does not. Therefore the fraction becomes a
**calibrated parameter alongside confidence** (see Calibration below), with
`SEMANTIC_TILE_FRACTION_SEED = 0.05` used only as the dialog prefill when the user skips
calibration — labelled in the UI as a starting guess from one dataset, not a
recommendation.

`reference_body_px` resolves from the first available of: the DetectKit project setting
(`gui/models.py:120`, adoption helper `populate_measured_reference` at `:175`), the
median longest-side of the source's existing labels, or the user. If none resolve,
tiling falls back to full-frame and the dialog says why rather than guessing.

New in `tiling.py`, because `slice_geometry` does not cover it:

- **Seam drop.** Discard a detection whose polygon touches within `seam_margin_px` of a
  tile edge that is not a frame edge. With overlap, every object is interior to some
  tile, so the fragment is recovered from its neighbour instead of merged badly.
- **Cross-tile merge.** Greedy NMS over survivors by `polygon_iou`.

**`polygon_iou` is specified, not just named.** It does not exist yet (verified), and
`utils/rotated_iou.py:pairwise_obb_overlap` cannot be reused: it is a 4-corner
Sutherland-Hodgman convex clip, and SAM3 contours are arbitrary non-convex polygons with
variable P, for which convex clipping silently returns garbage. `polygon_iou`
rasterizes both polygons with `cv2.fillPoly` onto a shared bounding-box grid and counts
pixels — the boring, correct choice given the polygons came from masks in the first
place. It lives beside `mask_to_contour` in `core/inference/masks.py`. (Related but not
reusable: `slice_geometry.polygon_area` `:186`, `clip_polygon_to_tile` `:215` —
convex-only.)

### Calibration

Run the labeler over the source's already-labelled frames and report the frontier of
**missed vs. to-delete per frame** — not F1. The two errors are not equally expensive:
deleting a spurious polygon is one click, a missed animal must be found by eye. On
measured data the F1-optimal threshold missed 4.7 animals per frame where a recall-first
threshold missed 1.0.

**Calibration fits two parameters, not one: tile fraction and confidence.** Confidence is
swept offline from one cached inference pass; tile fraction cannot be, because changing it
changes the tiles. So the grid is an outer loop over fractions (one inference pass each,
over the labelled frames only) and an inner offline sweep over confidence. This is the
mechanism that removes the ungrounded constant from the design: the operating point is fit
to the user's own frames on both axes, which is the same principle already applied to
confidence.

```python
SEMANTIC_TILE_FRACTION_SEED = 0.05          # dialog prefill only, ungrounded
TILE_FRACTION_GRID = (0.03, 0.05, 0.10, None)   # None = full frame, no tiling
CONFIDENCE_GRID = ...                        # as before

@dataclass(frozen=True)
class CalibrationPoint:
    tile_fraction: float | None
    tile_px: int | None                     # None when full-frame
    tiles_per_frame: int
    seconds_per_frame: float
    confidence: float
    missed_per_frame: float
    extra_per_frame: float
    recall: float
    n_matched: int

def calibrate(labeler, frames: Sequence[tuple[Path, list[LabelRecord]]],
              prompt: str, *, reference_body_px: float | None,
              frame_wh: tuple[int, int],
              tile_fractions=TILE_FRACTION_GRID, grid=CONFIDENCE_GRID,
              progress: Callable[[int, str], None] | None = None,
              should_stop: Callable[[], bool] | None = None) -> list[CalibrationPoint]
```

`calibrate` returns the full 2-D frontier — one point per (fraction, confidence) pair.
Tile px per fraction is `reference_body_px / fraction` under the same 64-4096 clamps
`tile_size_for_mode` applies, and both calibration and the real run go through the one
`resolve_tile_px` helper, so they tile identically. `None` is in the grid on purpose: on a rig where animals are already large at
native resolution, tiling *hurts*, and the grid must be able to say so rather than
assuming tiling is always right. Fractions are skipped (not failed) when
`reference_body_px` is unknown, when the resulting tile exceeds the frame — full-frame
already covers that — or when `plan_tiles` raises on the tile ceiling.

**Recommendation is lexicographic and stated in the UI.** Among points clearing
`MIN_RECALL`, take the fewest `tiles_per_frame` (inference cost over the whole project is
roughly linear in it, and a full run is hours); break ties by the highest confidence
(fewest polygons to delete). If no point clears the floor, recommend nothing, show the
frontier, and say so. The user can pick any point off the table; the recommendation is a
default, not a gate.

**Cost is part of the output, not a footnote.** Each fraction's measured
`seconds_per_frame` on the user's own frames is shown next to its error rates, and scaled
to the project's frame count as an estimated run time. This replaces the archived
dev-machine numbers the Cost section forbids quoting: the only timing a user ever sees is
one measured on their data, on their hardware.

`progress(pct, msg)` matches the existing convention (`sam2_escalation.py:158`).
`calibrate` takes image paths and `LabelRecord`s, not an `OBBSource` — Core must not
import an app-layer type. (Core->Data is established and fine: 8 existing
`from hydra_suite.data` imports under `core/`.)

**Any geometry level.** Choosing an operating point needs instance *counts*, not masks,
so polygon, OBB and AABB labels all work. The DetectKit-side adapter reuses
`detectkit/gui/utils.py:220 parse_obb_label`, which already handles 5-field AABB,
9-field quad and odd-count polygon lines and returns pixel polygons. Do **not** reuse
`sam2_prompts.read_boxes_from_label` — it accepts only 4- and 8-value lines and silently
`continue`s on polygon lines (`jobs/sam2_prompts.py:49-60`) — and do not write a third
reader.

**Matching is one-to-one on centroid distance, gated by containment.** SAM3's masks run
~1.7x the labelled body-core area, so IoU penalises correct detections for a purely
conventional reason. Centroid matching alone is not enough either: in a dense cluster a
blob's centroid can land inside a neighbour's label, and two predictions can claim one
label. So the assignment is greedy nearest-centroid with each label and each prediction
used at most once.

**One inference pass per tile fraction; confidence swept offline within each — with the
merge redone per threshold.** This is why the grid is 2-D-outer/1-D-inner rather than a
flat product: tile geometry is baked into the candidates, confidence is not. The cache
holds **pre-merge, per-tile candidates**, not
merged survivors: seam-drop and NMS are survivor-dependent, so post-filtering an already
merged set does not reproduce a run at that threshold. Each swept threshold re-runs
seam-drop + NMS over the cached candidates, which is pure geometry and cheap.
`max_instances` truncates before the sweep and so is fixed for a cache; the dialog says
so.

**Honest limits, surfaced in the UI.** With a handful of correlated frames this fits a
threshold on very little data. Three mechanisms, not five:

- Refuse to recommend below a minimum matched-instance count; show the frontier and say
  the data is insufficient. The minimum applies to the *recommended* point's own matched
  count, so a fraction that barely detects anything cannot win by accident.
- Report per-frame ranges, not just means.
- The dialog asks the user to confirm the labelled frames are **exhaustively** labelled;
  a partially labelled frame counts every real-but-unlabelled animal as a false positive
  and biases the threshold upward.

Documented in the dialog's help text but not mechanised: frames labelled during active
learning were selected *because they were hard*, so the fitted threshold may be
conservative for the easy majority. (Leave-one-frame-out agreement was considered and
cut — it is statistical theatre on 3 correlated frames.)

### The candidate cache and a re-thresholdable staged result

The same pre-merge candidate cache the sweep uses is written into the staging directory
as `candidates.json` alongside `labels/`. It exists so a 30-hour run is not a one-shot
commitment to one threshold.

- It is keyed by image relative path and holds per-tile candidate polygons + scores.
- The review dialog gains a **"Re-threshold"** action: pick a new confidence, re-run
  seam-drop + NMS + label writing over the cache, no inference. Seconds, not hours.
- It is **consumed at accept and never copied out.** Promotion writes only
  `images/`/`labels/`/`classes.txt` into the new sibling source, so the cache cannot go
  stale against user edits and cannot reach the X-AnyLabeling round trip
  (`dataset_panel.py:_prepare_xal_stage:637-657` copies only those three, so a sidecar
  would never survive it anyway — this is why the earlier "per-instance confidence
  sidecar shipped with the labels" design is dropped).

Re-thresholding is the only consumer. There is no per-instance confidence UI, no
sorted-by-score review list, no polygon-level editor — `ReviewEscalationsDialog` is a
per-source accept/reject checklist (105 lines) and `detectkit/gui/canvas.py` is
view-only, so any of those would be a separate feature.

### Cancellation and resumability

Neither exists in the pattern being copied, so both are specified here.

- `BaseWorker` (`widgets/workers.py`) has no cancel support. Use the existing DetectKit
  precedent instead: a `_cancel` flag on the worker (`main_window.py:136-139`) wired via
  `progress.canceled.connect(worker.cancel)` (`:1735`).
- The SAM2 progress dialog is `QProgressDialog(msg, None, ...)` — second argument `None`
  means **no cancel button** — and application-modal (`main_window.py:1851-1857`). For a
  multi-hour run that is unacceptable: the semantic dialog gets a real cancel button.
- `should_stop: Callable[[], bool]` threads worker -> `run_semantic_escalation` ->
  per-tile check. A cancelled run leaves the partial staging directory in place and
  reports how far it got.
- **Resumability** requires making `remove_staged_escalation_dir` conditional. The
  staging directory carries a `run.json` fingerprint (prompt, variant, tile geometry,
  threshold, source content hash). On re-run: matching fingerprint -> skip images
  already present in `candidates.json` and continue; mismatched -> wipe and start over.
  Without this change, `run_escalation:215` wipes unconditionally and `overwrite=True`
  — the flag a resumed run must pass — is exactly the wiping path.

## GUI

### Tools panel

`_build_escalation_group` (`panels/tools_panel.py:156`) gains a second action and both
get names that say what they do. SAM2's behaviour is unchanged — a label and signal
rename only:

- Group title: `"SAM2 Escalation"` -> `"Escalation"`.
- `"Escalate to segment (SAM2)"` -> **`"Geometry escalation (SAM2): boxes to masks"`**;
  signal `escalate_sam2_requested` -> `escalate_geometry_requested` (`:103`, `:170`,
  and the connection at `main_window.py:741`).
- New **`"Semantic escalation (SAM3): prompt to masks..."`** ->
  `semantic_escalation_requested`.
- Hint text distinguishes them: geometry escalation converts existing labels and cannot
  add a missing animal; semantic escalation can, and needs no labels to start.

The new button's enablement comes from `probe_availability()`, not from the vacuous
SAM2 guard.

### Dialog

`dialogs/semantic_escalation_dialog.py` (`BaseDialog`), beside `escalate_sam2_dialog.py`:

- Backend and variant combos, prompt line edit. The prompt is the user's to author.
- Confidence, max instances, tiling group (mode, tile fraction, resolved tile size shown
  read-only with its `reference_body_px` provenance, overlap, seam margin). The tile
  fraction prefills to `SEMANTIC_TILE_FRACTION_SEED` with the label *"starting guess from
  one dataset — calibrate to fit your own"*; the spec does not let the UI present it as
  a tuned value.
- **Calibrate**, enabled whenever the selected sources contain a labelled frame at any
  geometry level, presented as the recommended next step. With no labelled frames the
  dialog invites the user to label a few first, and lets them proceed anyway.
- Calibration results open a results view showing the 2-D frontier: one row per
  (tile fraction, confidence) with missed/frame, extra/frame, recall, matched count,
  tiles/frame and measured s/frame plus a projected run time for the selected sources.
  The recommended row is preselected with its rule stated in one line; choosing any row
  writes both the fraction and the confidence back into the dialog.
- **Preview runs one tile**, not the frame — a full-frame preview would show near-zero
  detections and teach the user the feature is broken.
- Projected total runtime shown before the run starts, computed from a measured
  per-tile time on this machine (see *Cost* — the archived numbers are not fit to quote).

### Handlers

`_on_escalate_to_segment_sam2` (`main_window.py:1769-1943`), the new semantic handler,
and `_on_review_escalations` (`:1985-2010`) move into `gui/escalation_actions.py`;
`main_window.py` keeps signal connections only. Honest accounting: this removes ~205 of
2152 lines and does **not** bring `main_window.py` near CLAUDE.md's thin-coordinator
target. It is done because the new handler would otherwise add a third escalation flow
to an already-oversized file, not because it satisfies the rule.

Also required, and easy to miss: `ReviewEscalationsDialog` hardcodes SAM2 in its intro
text (`:45-49`) and item text (`f"({pending.sam2_variant}, staged ...)"`, `:60-61`).
And `self._escalation_worker` (`main_window.py:706`) is a single slot guarding
"a run is already in progress" (`:1778`) — the two escalation kinds share it
deliberately, since both are heavyweight GPU jobs.

## Cost

**The archived timings do not reconcile, and must be re-measured before any of them is
shown to a user.** On an M3 Max at 4512x4512: the 3-frame runs at tile 1504 / overlap
0.2 (16 tiles) measured **22 s/frame**, while the 78-frame full run at overlap 0.5
(~36 tiles) measured **107 s/frame** and took 140 minutes. Tile count explains ~2.25x of
a 4.9x gap; the remainder is unexplained (thermal, contention, per-image I/O). Likewise
full-frame inference appears as both 1.3 s (warm, inference only) and 3.3 s (per frame,
including a 4512² decode).

What survives the discrepancy and drives the design: this is a **batch operation at tens
of seconds per frame, not an interactive one** — a 1000-frame source is many hours on
MPS. Hence resumability, real cancellation, and documentation pointing large runs at the
CUDA box. The dialog's runtime projection must come from a timed preview tile on the
user's machine, never from these numbers.

Calibration now supplies that measurement for free: it already runs the labeler over the
labelled frames at each tile fraction, so `CalibrationPoint.seconds_per_frame` is a real
timing on the user's hardware and data. When the user skips calibration, the dialog times
a single tile before projecting. Neither path quotes the archived numbers.

Calibration's own cost is `len(TILE_FRACTION_GRID)` passes over the labelled frames only —
typically 3-5 frames, so single-digit minutes at the measured tens of seconds per frame,
against a full run of hours. Progress is reported per fraction and the whole thing is
cancellable.

## Evidence

Measured 2026-08-28 on one ant AL round: 78 frames at 4512x4512, 24 animals per frame,
median longest side 80 px, three frames labelled. **One dataset; these justify the
mechanisms, and none of them is a default.**

| Configuration | F1@0.5 | s/frame |
|---|---|---|
| Full frame | 0.075 | 3.3 |
| Tile 1504, overlap 0.2 | 0.719 | 22 |
| Tile 752, overlap 0.2 | 0.684 | 125 |
| Tile 1504 + cross-frame exemplars | 0.503 | 22 |

1. **Scale dominates.** Full-frame to tiled moved recall 0.08 to 0.75. Prompt wording
   moved best F1 only 0.688-0.720 — hence tiling is designed in and wording is the
   user's to explore.
2. **Smaller tiles are not better.** Tile 752 cost 5.7x tile 1504 for no gain. Only two
   tile sizes were tested, which is why the tile rule is derived from SAM3's input
   resolution rather than fitted to this curve.
3. **Seam handling is worth ~40% of false positives** at held recall.
4. **F1 is the wrong objective** — see Calibration.
5. **Exemplars measured worse at every threshold**, damaging ranking rather than
   calibration. Caveat: neither ultralytics nor HF transformers exposes cross-frame
   exemplars (box prompts are same-image), so this measured a hand-rolled prompt
   injection. It is evidence the available APIs don't support this usefully, not that
   SAM3 exemplars are worthless.
6. **Masks include appendages**, median 1.7x labelled area. A naive correction — scaling
   the min-area rect to 62% — failed badly (F1 0.013) because it shortens the long axis
   too. No automatic conversion is specified; review is where the convention is settled.
7. **Residual error concentrates in dense clusters.** Isolated animals were found
   reliably; background hallucinations were rare.

**Methodological caveats.** The three labelled frames come from one video at nearby
timestamps and share the same 24 individuals, so they are closer to n=1 than n=3. The
IoU-0.3 matching threshold was adopted *after* observing it improved scores; the
convention-gap rationale is sound but the choice was post-hoc. Neither is a reason to
distrust the ordering of the results — the tiling effect is ~10x and survives any
threshold — but both are reasons not to trust the specific numbers.

## Testing

Unit tests with a fake `SemanticLabeler` (no weights):

- Seam drop: interior-seam detection dropped, frame-edge detection kept.
- Cross-tile merge: one object in two overlapping tiles yields one instance; tile-local
  polygons are offset to frame space.
- Tile resolution: `reference_body_px` provenance chain; semantic fraction independent
  of `SliceTrainingSettings.object_tile_fraction`; full-frame fallback when unresolved;
  `MAX_TILES_PER_FRAME` respected.
- Tile-fraction grid: `None` yields one full-frame pass; a fraction whose tile exceeds
  the frame is skipped, not duplicated as full-frame; a fraction breaching the tile
  ceiling is skipped with a reason, not raised to the caller; `reference_body_px=None`
  leaves full-frame as the only surviving point.
- Recommendation ordering: among points clearing `MIN_RECALL`, fewest
  `tiles_per_frame` wins; ties broken by highest confidence; no point clearing the floor
  yields no recommendation with the frontier still returned.
- `polygon_iou`: disjoint, identical, partial overlap, **non-convex** (the case
  `pairwise_obb_overlap` gets wrong), degenerate.
- Calibration matching: one-to-one under a dense cluster where a blob centroid falls in
  a neighbour's label; polygon, OBB and AABB labels all match; refusal below the minimum
  matched-instance count.
- Calibration frontier monotone in confidence on canned candidates, **with merge redone
  per threshold** (a test that post-filters a merged set must fail).
- Degenerate contours (`P < 3`) are dropped and counted, never passed to
  `write_label_file`.
- Staging round-trip: `PendingEscalation` with `primer_kind="sam3"`; a legacy dict
  carrying only `sam2_variant` loads as `primer_kind="sam2"`.
- Staging dirname differs for two prompts on one source.
- Promotion: accepting a semantic escalation creates a sibling source and leaves the
  original's labels byte-identical.
- Resume: matching `run.json` fingerprint skips completed images; mismatched wipes.
- Cancellation: `should_stop` honoured between tiles; partial staging survives.
- Moved modules: `tests/test_sam2_masks.py` and `tests/test_sam2_executor.py` repointed
  and passing.

**Not a gate:** the TrackerKit equivalence matrix.

**Before implementation:** confirm SAM3 on CUDA (mehek, RTX 6000 Ada 48 GB). MPS is
already confirmed.
