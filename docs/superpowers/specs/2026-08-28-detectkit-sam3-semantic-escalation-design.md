# DetectKit SAM3 Semantic Escalation — Design

**Date:** 2026-08-28
**Status:** Draft, pending review
**Branch:** `feat/sam3-semantic-escalation` (from `main` @ 04da82fa)
**Supersedes:** `2026-08-17-detectkit-semantic-escalation-design.md`, which was written
against the August codebase and has since gone stale in its load-bearing sections.

## Summary

Add **Semantic escalation** to DetectKit: the user types a noun phrase ("ant"), SAM3
segments every matching instance across a source's frames, and the result is staged for
review exactly like SAM2 escalation already is. Where the source already has labelled
frames, a **calibration** pass measures SAM3 against those labels and recommends a
confidence threshold, so the operating point is fitted to the user's data rather than
inherited from ours.

Unlike SAM2 escalation, which promotes existing boxes to polygons and therefore needs
labels to start, this produces labels from nothing.

## Scope

**In scope**

- A `SemanticLabeler` seam and a SAM3 backend behind it.
- Tiled inference, reusing `utils/slice_geometry.py`.
- Calibration against the source's existing labelled frames.
- A dialog: prompt, tiling controls, calibrate, single-tile preview.
- Staging into the existing `PendingEscalation` review flow.

**Out of scope**

- Image exemplars as prompts (measured worse than text — see *Evidence*).
- SAM3 fine-tuning (possible via the official repo; better motivated once a project has
  a reviewed frame set, and untestable from the frames this feature starts with).
- Automatic prompt search (wording measured near-irrelevant next to tiling).
- Multi-class prompting. One run = one prompt = class `0`.
- Any change to TrackerKit, `core/inference` detection stages, or cache keys. The
  TrackerKit equivalence matrix is **not** a gate for this work.

## Architecture

### Output: stage, don't derive

`run_semantic_escalation` mirrors `sam2_escalation.run_escalation` (`jobs/sam2_escalation.py:153`):
write into `artifacts/pending_escalations/`, record a `PendingEscalation` on the source,
and let `accept_pending_escalation` / `reject_pending_escalation` (`:299`, `:373`)
promote or discard it. `review_escalations_dialog.py` gains a second producer.

This is chosen over writing a sibling source for two reasons the measurements force:
the output is deletion-heavy (below), so it must not reach a training source unreviewed;
and SAM3 masks trace legs and antennae while tracking labels bound the body core, so
merging the two conventions into one class would degrade YOLO training.

`PendingEscalation` needs generalising — it currently hardcodes `sam2_variant`
(`gui/models.py:14-20`). Add `primer_kind` ("sam2" | "sam3"), `primer_variant`,
`primer_prompt`, and `primer_params` (the resolved operating point), with `from_dict`
back-filling `primer_kind="sam2"` from a legacy `sam2_variant` so existing projects load.

### The seam

```
core/inference/semantic/
    base.py          # SemanticLabeler protocol + SemanticInstance
    checkpoints.py   # SAM3 catalog + HF download
    sam3.py          # Sam3SemanticLabeler
    tiling.py        # seam-drop + cross-tile merge (grid comes from slice_geometry)
    calibration.py   # sweep confidence against labelled frames
```

```python
@dataclass(frozen=True)
class SemanticInstance:
    polygon_px: np.ndarray   # (P, 2) float32, frame pixel space
    confidence: float

class SemanticLabeler(Protocol):
    @property
    def name(self) -> str: ...
    def label_image(self, image_bgr, prompt, *, confidence_threshold,
                    max_instances=0) -> list[SemanticInstance]: ...
```

`Sam3SemanticLabeler` wraps `SAM3SemanticPredictor` from `ultralytics.models.sam`
(present in the installed 8.4.34). Checkpoints follow `sam2/checkpoints.py`: pinned
catalog, `hf_hub_download` into `get_models_dir() / "sam3"`, offline error path,
`available_variants()` as the GUI availability probe.

Two facts the catalog must encode: `sam3.pt` is 3.45 GB and is **not** in ultralytics'
`GITHUB_ASSETS_NAMES`, so it comes from the public `facebook/sam3` HF repo; and
ultralytics AutoUpdate pip-installs `clip` and `ftfy` on first run, which is
unacceptable for an offline or shared install — both must be declared in a `sam3` extra,
and the probe must fail loudly rather than trigger AutoUpdate.

### Tiling

SAM3 letterboxes its input to 1008 px. An animal occupying 2% of a high-resolution frame
is ~18 px to the model, and detection collapses. Tiling is what makes the feature work —
and on a rig where animals are already large at native resolution, tiling would instead
*hurt*, which is why the tile size is derived from object scale rather than fixed.

**The grid comes from `utils/slice_geometry.py`, not from new code.** That module exists
precisely so training, inference, and preview tile identically, and it already provides
what is needed: `tile_size_for_mode(geometry_mode="auto_object", ...)` computes
`reference_body_px / object_tile_fraction` (`:104-128`), `plan_tiles` lays out the
overlap grid with last-tile-flush-to-edge, and `MAX_TILES_PER_FRAME = 4096` (`:20`)
caps pathological configurations. Semantic escalation adopts `object_tile_fraction` as
its knob; it does **not** introduce a second object-size convention.

`reference_body_px` resolves from the first available of: the DetectKit project setting
(`gui/models.py:120`, which already has an adoption helper at `:178`), the median
longest-side of the source's existing labels, or the user. If none resolve, tiling falls
back to full-frame and the dialog says why rather than guessing.

New in `tiling.py`, because `slice_geometry` does not cover it:

- **Seam drop.** Discard a detection whose polygon touches within `seam_margin_px` of a
  tile edge that is not a frame edge. With overlap, every object is interior to some
  tile, so the fragment is recovered from its neighbour instead of merged badly.
- **Cross-tile merge.** `polygon_iou` NMS over survivors. `polygon_iou` does not exist
  yet; add it beside `mask_to_contour`, which moves from `sam2/masks.py` to
  `core/inference/masks.py` (sole current import site is `jobs/sam2_escalation.py`).

### Calibration

Run the labeler over the source's already-labelled frames and report the frontier of
**missed vs. to-delete per frame** — not F1. The two errors are not equally expensive:
deleting a spurious polygon is one click, a missed animal must be found by eye. On
measured data the F1-optimal threshold missed 4.7 animals per frame where a recall-first
threshold missed 1.0.

```python
@dataclass(frozen=True)
class CalibrationPoint:
    confidence: float
    missed_per_frame: float
    extra_per_frame: float
    recall: float
    n_matched: int

def calibrate(labeler, frames: Sequence[tuple[Path, list[LabelRecord]]],
              prompt: str, *, tile_px, grid, progress=None) -> list[CalibrationPoint]
```

`calibrate` takes image paths and `LabelRecord`s, not an `OBBSource` — Core must not
import an app-layer type (CLAUDE.md dependency direction). The DetectKit job does the
adaptation.

**Any geometry level.** Choosing an operating point needs instance *counts*, not masks,
so polygon, OBB and AABB labels all work; matching happens at the level the labels
provide. Note that the existing `sam2_prompts.read_boxes_from_label` cannot be reused —
it accepts only 4- and 8-value lines and silently `continue`s on polygon lines
(`jobs/sam2_prompts.py:56-60`); calibration must use a level-aware reader.

**Matching is one-to-one on centroid distance, gated by containment.** SAM3's masks run
~1.7x the labelled body-core area, so IoU penalises correct detections for a purely
conventional reason. Centroid matching alone is not enough either: in a dense cluster a
blob's centroid can land inside a neighbour's label, and two predictions can claim one
label. So the assignment is greedy nearest-centroid with each label and each prediction
used at most once.

**Run inference once.** The confidence axis is swept offline over cached per-instance
scores; only tile size, if varied, costs a rerun. This is what keeps calibration to
minutes rather than hours.

**Honest limits, surfaced in the UI rather than buried here.** With a handful of
correlated frames this fits a threshold on very little data:

- Refuse to recommend below a minimum matched-instance count; show the frontier and say
  the data is insufficient.
- Report per-frame ranges, not just means.
- Leave-one-frame-out: recommend only if the per-frame choices agree; otherwise show the
  spread.
- Calibration assumes labelled frames are **exhaustively** labelled — a partially
  labelled frame counts every real-but-unlabelled animal as a false positive and biases
  the threshold upward. The dialog must ask the user to confirm this.
- Frames labelled during active learning were selected *because they were hard*, so the
  fitted threshold may be conservative for the easy majority.

### Persist confidence

`data/al/labels.py:_format_line` writes class and coordinates only (`:58-72`), so
per-instance confidence is lost at write time. Staged output writes a sidecar carrying
each instance's score.

Without it, the ~12 spurious polygons per frame at the recommended threshold arrive with
no sort order, and moving the threshold after seeing results means re-running the whole
source. With it, review sorts by confidence ascending, bulk-deletes a tail, and the
threshold becomes a review-time slider over a single inference run. This is the
difference between the calibration frontier being a blind one-shot commitment and a
reversible knob.

## GUI

`dialogs/semantic_escalation_dialog.py` (`BaseDialog`), beside the existing
`escalate_sam2_dialog.py`:

- Backend and variant combos, prompt line edit. The prompt is the user's to author and
  vary.
- Confidence, max instances, tiling group (mode, resolved tile size shown read-only with
  its `reference_body_px` provenance, overlap, seam margin).
- **Calibrate**, enabled whenever the selected sources contain a labelled frame at any
  geometry level, and presented as the recommended next step. With no labelled frames the
  dialog invites the user to label a few first, and lets them proceed anyway.
- **Preview runs one tile**, not the frame — a full-frame preview would show near-zero
  detections and teach the user the feature is broken. One tile at the resolved geometry
  is ~4 s; a tiled full frame is ~107 s and is not a preview.
- Projected total runtime shown before the run starts.

Handlers go in a new `gui/escalation_actions.py` alongside the existing SAM2 and
review-escalations handlers; `main_window.py` (2152 lines) keeps signal connections only.

## Cost

Measured on an M3 Max at 4512x4512: 1.3 s per full-frame inference warm, 24 s cold;
**107 s/frame** at the recommended tiled configuration; a 78-frame source took **140
minutes**. A 1000-frame source is ~30 h on MPS. This is a batch operation, not an
interactive one.

Therefore: per-image resumability so a crash or cancel does not restart from zero,
cancellation honoured between tiles, and documentation pointing large runs at the CUDA
box.

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
   tile sizes were tested, so the tile rule is deliberately inherited from
   `slice_geometry` rather than fitted here.
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
- Cross-tile merge: one object in two overlapping tiles yields one instance.
- Tile resolution: `reference_body_px` provenance chain; full-frame fallback when
  unresolved; `MAX_TILES_PER_FRAME` respected.
- Calibration matching: one-to-one under a dense cluster where a blob centroid falls in
  a neighbour's label; polygon, OBB and AABB labels all match; refusal below the minimum
  matched-instance count.
- Calibration frontier monotone in confidence on canned detections.
- `polygon_iou`: disjoint, identical, partial, degenerate.
- Staging round-trip: `PendingEscalation` with `primer_kind="sam3"`, and a legacy dict
  carrying only `sam2_variant`.
- Degenerate contours (`P < 3`) are dropped and counted, never passed to
  `write_label_file`, which refuses them (`data/al/labels.py:46-52`).

**Not a gate:** the TrackerKit equivalence matrix.

**Before implementation:** confirm SAM3 on CUDA (mehek, RTX 6000 Ada 48 GB). MPS is
already confirmed.
