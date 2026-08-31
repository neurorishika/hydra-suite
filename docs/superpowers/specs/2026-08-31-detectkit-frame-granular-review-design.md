# DetectKit Frame-Granular Review — Design

**Status:** pending implementation plan
**Date:** 2026-08-31
**Scope:** `detectkit/jobs/`, `detectkit/gui/`, `data/al/`, new `data/al/merge.py`

## Problem

Review in DetectKit is per-source and all-or-nothing. A run either lands
entirely or not at all, and the two producers land it in incompatible ways:

- **SAM2** overwrites the source's `labels/` wholesale —
  `shutil.rmtree(source_labels)` then `copytree` (`jobs/sam2_escalation.py:351`).
- **SAM3** builds a *new sibling source*, hardlinking the images and
  leaving the original untouched (`jobs/semantic_escalation.py:1157-1211`).

Neither matches how the work is actually judged. The reported symptom is
the SAM3 one: accepting a reviewed run produces a whole new source, which
is wrong when the run was a pass over labels that already exist. The
deeper problem is granularity — the review dialog offers one checkbox per
source (`gui/dialogs/review_escalations_dialog.py:66-77`), so a run that is
right on 130 frames and wrong on 10 has no expressible outcome.

Three further gaps fall out of the same shape:

1. **No merge primitive exists.** Every writer truncates:
   `write_label_file` opens `"w"` (`data/al/labels.py:73`), accept does
   `rmtree`+`copytree`, and the X-AnyLabeling sync-back does `rmtree`+copy.
   "Add these instances to what is already there" is not expressible.
2. **Model predictions cannot be written to labels at all.**
   `_dataset_predictions` is an in-memory dict, cleared on source switch
   (`gui/main_window.py:747,2079`); nothing writes it to disk. Inference and
   escalation are the same kind of proposal to a user, and are completely
   different objects in the code.
3. **Review state is per-source only.** Nothing per-frame exists in the
   project JSON. The only per-frame key that exists is implicit: staged
   labels mirror the image's path under `images/`, relied on by
   `_origin_image_for` and `find_staged_label_for_image`.

## Decisions taken

Recorded because each closes off an alternative a reader will otherwise
re-litigate:

- **Merge rule** — keep existing labels verbatim; add a staged instance
  only when it does not overlap one already present.
- **Level conflict** — the staged level wins; accepting polygons into an
  OBB source promotes the source. (This is already what SAM2's in-place
  accept does: it assigns `source.level = pending.target_level`.)
- **Sibling sources** — deleted, not demoted. SAM3 accepts into the source
  it ran on.
- **Predictions** — in scope. One review path for all three producers.

## Non-goals

- **Per-instance accept/reject.** Instance-level correction is
  X-AnyLabeling's job; the round trip already exists
  (`gui/panels/dataset_panel.py:637,952`). This design deliberately stops at
  frame granularity, which is why no instance identity is introduced.
- Changing how any producer *generates* its proposals.
- Changing the X-AnyLabeling round trip.

## Design

### 1. `StagedReview` — one concept replacing three

`PendingEscalation` generalises (`gui/models.py:18-37`):

```python
@dataclass
class StagedReview:
    staged_path: str
    target_level: str
    producer: str            # "sam2" | "sam3" | "inference"
    producer_variant: str    # model or checkpoint identity
    prompt: str              # SAM3's noun phrase; "" otherwise
    params: dict
    created_at: str
```

`producer` replaces `primer_kind`, which is currently load-bearing for
accept dispatch (`review_escalations_dialog.py:107`). After this change it
is load-bearing for *nothing* — it is provenance only, because all three
producers accept identically. That is the point of the refactor.

The staging layout SAM2 and SAM3 already share becomes the **contract**:

```
<staged_root>/
    labels/          # mirrors the source's images/ tree, one .txt per frame
    classes.txt
    run.json         # producer, params, fingerprint
    decisions.json   # NEW — per-frame outcome
    labels_before/   # NEW — snapshot for revert, written on first accept
```

A producer's only job is to fill `labels/` + `classes.txt` + `run.json`.
Everything downstream is producer-agnostic.

The `OBBSource` field is renamed `pending_escalation` → `staged_review`
to match, since it no longer holds only escalations.

**Backwards compatibility:** a project holding a pending SAM2 or SAM3
escalation staged under the old scheme reviews correctly without migration
— the directory layout is unchanged, and `from_dict` accepts the old
`pending_escalation` key and the old `primer_kind`/`primer_variant`/
`primer_prompt` names, mapping them onto the new ones. `to_dict` writes
only the new names; a project saved by this version is not readable by an
older one, which is consistent with how `runtime_tier` migration was
handled and is worth stating rather than discovering.

### 2. `merge_records` — the missing primitive

New `data/al/merge.py`:

```python
class MergeMode(Enum):
    OVERWRITE = auto()   # staged replaces existing for this frame
    ADD_NEW = auto()     # existing kept; non-overlapping staged appended

def merge_records(
    existing: Sequence[LabelRecord],
    staged: Sequence[LabelRecord],
    *,
    mode: MergeMode,
    iou_threshold: float,
    level: GeometryLevel,
) -> list[LabelRecord]: ...
```

`ADD_NEW` compares each staged record against every existing one and drops
it when `IoU >= iou_threshold`. Existing records are never modified,
reordered, or dropped — a merge can only add. That invariant is what makes
the operation safe to apply immediately (§4) and is worth asserting in
tests rather than trusting.

**IoU dependency.** The polygon IoU this needs lives in
`core/inference/masks.py:51`. `utils/rotated_iou.py` is a convex quad clip
and is silently wrong for the non-convex contours SAM3 produces — the
existing docstring says so. Rather than have `data/` import from
`core/inference`, move `polygon_iou` to `utils/` and have both call sites
use it. This is a move, not a rewrite; the function's rasterisation
behaviour (including its 4x supersampling correction) must not change, so
the existing `tests/test_semantic_masks.py` must pass untouched against the
moved function.

### 3. Level promotion on accept

If the staged level is **above** the source's, the source is promoted and
its existing labels lifted: an OBB quad becomes a 4-point polygon.
`_polygon_points` (`data/al/labels.py`) already duplicates the last vertex
when given exactly 4 points, precisely so a promoted quad never reads back
as an OBB — that machinery exists and is reused, not reinvented.

If the staged level is **below** the source's, staged records are derived
down with the existing `derive_down` (`data/al/escalation.py:103`).

Promotion is a property of the source, so the first accept that promotes
sets `source.level` and the rest of the review proceeds at the new level.

### 4. Frame-granular accept, applied immediately

Four operations, all keyed by the frame's relative path:

| Operation | Effect on the source |
|---|---|
| Accept frame (overwrite) | label file replaced by the staged one |
| Accept frame (add new) | `merge_records(..., ADD_NEW)` written back |
| Reject frame | nothing; the staged label is marked rejected |
| Accept all / Reject all | the same, over every frame not yet decided |

Applied **immediately** rather than accumulated into a pending set: the
result appears on the ground-truth layer as the user works, which is the
entire point of reviewing on the frame. The cost is that "undo" cannot be
"discard pending decisions", so it is provided explicitly:

- On the **first** accept of a review, `labels/` is snapshotted to
  `<staged_root>/labels_before/`.
- **Revert this review** restores that snapshot and clears
  `decisions.json`.

Text files, so the snapshot is cheap even for a large source. This is
chosen over per-frame inverse operations because `ADD_NEW` is not
invertible without recording exactly which records were appended, and a
snapshot is both simpler and harder to get wrong.

`decisions.json` maps relative image path → `"accepted_overwrite" |
"accepted_add_new" | "rejected"`. It lives in the **staging dir, not the
project JSON**: a 10k-frame source would otherwise add 10k entries to every
project save, and the staging directory is already the object whose
lifetime matches the review's.

A review is **complete** when every staged frame has a decision; completing
it removes the staging dir and clears `source.staged_review`, exactly as
accept/reject does today.

### 5. Inference as a producer

`_DetectKitDatasetInferenceWorker` (`gui/main_window.py:127`) already runs
across every image in the active source. This adds a stager that writes its
output into the staging contract — `write_label_file` per frame at the
model's level, `classes.txt` from the project, `run.json` recording the
model path, confidence and device.

The existing in-memory `_dataset_predictions` preview path is unaffected;
staging is a separate, explicit action ("Stage predictions for review"), so
running inference to look at it does not create reviewable state.

### 6. UI

A **review bar** above the canvas, visible only when the current source has
a staged review: the four operations, a "next undecided frame" control, and
a progress counter (`23/140 decided`). The staged proposals are already
drawn by the escalation overlay layer merged on 2026-08-31; accepting a
frame refreshes both that layer and the ground-truth layer so the change is
visible where it happened.

The per-source checkbox dialog (`review_escalations_dialog.py`) is retired.
Its one irreplaceable feature — SAM3 re-thresholding across sources — moves
to the review bar as a per-review action, since `rethreshold_staged` already
operates on one source's staging dir.

## What gets deleted

- `accept_pending_semantic_escalation` and `_unique_source_name`
  (`jobs/semantic_escalation.py:1091,1111`) — the sibling-source path.
- The `primer_kind` dispatch in `review_escalations_dialog._apply_checked`.
- `ReviewEscalationsDialog` itself.

`derived_from` and `original_path` on `OBBSource` are **retained**: they are
read by the bundle exporter (`gui/project.py:356-362`), independently of
escalation.

## Testing

1. **Merge rule** — overwrite vs add-new; the IoU boundary in both
   directions; empty existing; empty staged; and the invariant that
   `ADD_NEW` never modifies, reorders or drops an existing record.
2. **`polygon_iou` move** — `tests/test_semantic_masks.py` passes untouched
   against the relocated function.
3. **Level promotion** — an OBB source accepting polygons reads back as
   polygon with no coordinate drift, and a promoted quad does not read back
   as an OBB.
4. **Revert** — accept a mix of frames in both modes, revert, and assert the
   source's labels are byte-identical to before the review.
5. **Producer-agnosticism** — the accept path exercised once per producer
   against a fake stager, asserting identical outcomes for identical staged
   content. This is the test that would fail if `producer` ever became
   load-bearing again.
6. **Backwards compatibility** — a project JSON carrying an old
   `pending_escalation` (both kinds) loads and reviews.

No equivalence gate applies: nothing here is on the tracking pipeline path.

## Phasing

Each phase is independently mergeable and leaves the app working:

1. **`merge_records` + the `polygon_iou` move.** Pure library work, no UI.
2. **`StagedReview` model + `decisions.json` + revert snapshot**, with the
   existing per-source dialog still driving it (accept-all under the hood).
3. **Frame-granular accept + the review bar**; retire the dialog.
4. **Inference producer.**
5. **Delete the sibling-source path.** Last, so the new path is proven
   before the old one goes.

## Relationship to the overlay registry spec

`2026-08-31-detectkit-overlay-layer-registry-design.md` is deliberately
sequenced **after** this work. Its stated triggers are a fourth overlay
layer or the first per-instance interaction; this design produces neither —
per-instance work is out of scope, and folding predictions into staged
reviews likely *reduces* the layer count, since the prediction layer and the
escalation layer become the same object read from the same directory.
Building the registry first would mean cutting an abstraction over a layer
inventory that is about to change. Expect to amend that spec once this
lands.
