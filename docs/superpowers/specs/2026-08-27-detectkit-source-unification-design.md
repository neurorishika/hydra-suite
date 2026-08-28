# DetectKit Source Unification — Design Spec

> **Status:** APPROVED, ready for planning.
> **Decided:** 2026-08-27.
> **Scope:** Part A only (source data model + import + escalation). Parts B (multi-level
> canvas visualization) and C (clear-labels actions) are design notes at the end of this
> document — deliberately lighter-weight, to be turned into their own specs once Part A's
> actual shape is known. Per the user: "As soon as A is implemented B+C will be implemented."

## Goal

One registered `OBBSource` per logical dataset in a DetectKit project, regardless of how
many geometry levels (AABB / OBB / polygon) exist or could be derived for it. No more
`<name>_obb` / `<name>_aabb` / `<name>_obb_seg` siblings representing the same underlying
images as separate, independently-browsable "sources."

## Motivation (verified on main + this session)

- **The same image set currently lands in a project as up to 3 separate sources.** An
  active-learning round exported as `obb` (authoritative) + `aabb` (derived), then SAM2-escalated
  to a segmentation source, produces three dropdown entries — `20260827_175321_obb`,
  `20260827_175321_aabb`, `20260827_175321_obb_seg` — for what is, to the user, one dataset.
  Screenshot confirmed (user-reported).
- **This is not new with AL rounds; SAM2 escalation already did it.** `jobs/sam2_escalation.py`
  has always registered a new `<name>_seg` sibling rather than upgrading the source it escalated.
  A same-session fix to AL-round import (registering `aabb` alongside `obb`, to avoid silently
  dropping data) made the underlying problem worse, not better — it added a second flat
  sibling on top of an existing one. That fix is being **reverted** by this spec's import
  behavior (see "Relationship to prior work" below), not built upon.
- **Training already treats a source's lower geometry levels as derived, not registered.**
  `training/dataset_builders.py::prepare_role_dataset` builds AABB/segment role-datasets
  on the fly from the single merged OBB dataset at build time
  (`derive_detect_dataset_from_obb`, `derive_segment_dataset_from_source`). It never reads a
  separately-registered aabb source. **The bug is that import and escalation don't follow
  the pattern training already uses** — they persist and register siblings instead of
  deriving on demand. This spec brings import and escalation in line with the existing
  pattern rather than inventing a new one.
- **Side effect of the flat-sibling pattern:** `training_dialog.py::merged_level_and_blocker`
  takes the *minimum* geometry level across **all** `project.sources`, unfiltered by
  `reviewed`/`derived_from`. A round's own derived `aabb` sibling — 100% reconstructible from
  its `obb` sibling sitting right next to it — silently drags the *entire project's* training-role
  gating down to AABB, disabling OBB/segment training roles project-wide. Verified by reading
  `al_worker.py` (already registers `obb`+`aabb` per round today, independent of this session's
  work) and `training_dialog.py` (the ungated `min()`). Eliminating registered siblings
  eliminates this bug as a side effect — no separate fix needed.

## Decisions (locked during brainstorming)

1. **Escalation upgrades the same source in place, behind an accept/reject review step.**
   SAM2 escalation writes to a staging area; the source keeps its old level and `reviewed`
   state until the user reviews and accepts, at which point the staged result becomes the
   source's canonical data. Reject discards the staged copy; the source is untouched. No new
   source ever appears in the list for an escalation.
2. **No automatic migration of existing projects.** A project that already has flat sibling
   entries (like the user's 3-source example) keeps them as-is. The user manually removes
   duplicates and re-adds the round to get the new unified behavior. New imports and new
   escalations always get the unified treatment from the moment this ships. (Automatic or
   semi-automatic consolidation of pre-existing flat entries is an explicitly out-of-scope
   follow-up, not part of this spec.)
3. **Naming: the registered source's display name is the bare originally-selected folder's
   name, with no level suffix** — e.g. `20260827_175321`, not `20260827_175321_obb`.

## Architecture

No new module. This is a behavior change in four existing files, plus one small addition to
the project model.

```
src/hydra_suite/detectkit/gui/source_import.py
    inspect_detectkit_source()      # UNCHANGED: single-redirect AL-round detection
                                     # (stale-path fallback + refuse-if-authoritative-missing,
                                     # both already fixed this session, both KEPT)
    inspect_al_round()              # REMOVED (added this session, only multi-root consumer)
    materialize_al_round()          # REMOVED (same)
    ALRoundRoot / MaterializedALRoundRoot   # REMOVED (same)

src/hydra_suite/detectkit/gui/dialogs/source_manager.py
    _add_source()                   # REVERTS to single inspect/materialize/append;
                                     # display name = Path(selected_path).name (see Naming)
    _add_al_round_sources()         # REMOVED (added this session, no longer called)

src/hydra_suite/detectkit/jobs/al_worker.py
    run_active_learning()           # registers ONE OBBSource for the round's native/
                                     # authoritative level, not one per manifest root

src/hydra_suite/detectkit/jobs/sam2_escalation.py
    run_escalation()                # writes staged result + returns pending-escalation
                                     # descriptors; NO LONGER appends OBBSource directly

src/hydra_suite/detectkit/gui/dialogs/escalate_sam2_dialog.py (or a new sibling dialog)
    + new Accept/Reject review step for pending escalations

src/hydra_suite/detectkit/gui/models.py
    OBBSource.pending_escalation: PendingEscalation | None = None   # NEW field
    PendingEscalation dataclass (to_dict/from_dict)                 # NEW
```

`data/al/export.py` (the manifest/multi-root writer) is **unchanged**. It stays a useful
external-interchange format — exporting `obb` + `aabb` sibling folders is still valuable for
handing data to a collaborator or another tool. Only DetectKit's own *import/registration*
behavior changes: it reads the manifest to find the native/authoritative root and registers
only that.

## Import: collapse to one source per round

`inspect_detectkit_source(round_dir)` already resolves an AL-round container to its
authoritative root's `DetectKitSourceInspection` (this session's fix, kept as-is — including
the stale-manifest-path fallback to `round_dir/<level>` and the refusal to proceed when the
authoritative root itself can't be resolved, both verified via Fable's adversarial review).

`source_manager.py::_add_source` goes back to the single-source shape it had before this
session's multi-root expansion:

```
inspection = inspect_detectkit_source(selected_path)          # may redirect to round/<level>
selection  = confirm_detectkit_source_addition(self, str(inspection.dataset_root), inspection)
...
materialized = materialize_detectkit_source(selected_path, project_dir, ...)
name = Path(selected_path).name        # <- NEW: name from the ORIGINAL pick, not the
                                        #    resolved level-subfolder ("obb"), so an AL-round
                                        #    pick is named "20260827_175321", never "obb".
project.sources.append(OBBSource(path=..., name=name, ...))
```

For a plain (non-AL-round) pick, `selected_path` and the resolved dataset root are the same
directory, so this is a no-op behavior change there — the naming fix is specific to the
AL-round-redirect case.

`al_worker.py::run_active_learning` gets the equivalent change: instead of looping over every
`manifest["roots"]` entry and appending one `OBBSource` per level (current behavior, with
`derived_from` linking siblings), it registers **one** `OBBSource` for the round's
`native_level` root, named `al_round_<timestamp>` (no level suffix). The round's other
exported levels stay on disk (written by `export_al_dataset` as before) but are not
registered as project sources.

Nothing else needs to change to make this safe: `merged_level_and_blocker`, `_collect_sources`,
and the training-role builders all already key off the merged OBB dataset / per-source
`level` field, not off a specific set of registered siblings.

## Escalation: staged accept/reject, upgrading the same source

**New state — `models.py`:**

```python
@dataclass
class PendingEscalation:
    staged_path: str        # artifacts/pending_escalations/<source>-<variant>-<hash>/
    target_level: str       # GeometryLevel.label the staged result would become
    sam2_variant: str
    created_at: str         # ISO timestamp

    def to_dict(self) -> dict: ...
    @staticmethod
    def from_dict(d: dict) -> "PendingEscalation": ...

@dataclass
class OBBSource:
    ...
    pending_escalation: PendingEscalation | None = None   # NEW
```

**`sam2_escalation.py::run_escalation`** no longer appends a new `OBBSource` to
`req.project.sources`. For each source escalated, it writes the SAM2 result to
`project_dir/artifacts/pending_escalations/<source.name>-<variant>-<hash>/` (same
bundle-owned-path pattern as `artifacts/imported_sources/`, via `ensure_bundle_subdirectory`)
and sets that source's `pending_escalation` field. It returns enough information (source
name, staged path, counts) for the caller to show a review UI — it does not silently accept.
The `<hash>` component follows `source_import.py::_standardized_source_dir`'s existing
pattern: `sha1(str(resolved_source_path) + variant).hexdigest()[:10]`, so re-escalating the
same source with the same variant reuses/overwrites one staging directory rather than
accumulating stale ones.

**Review dialog** (extends `EscalateSam2Dialog` or a new small sibling dialog): lists sources
with a pending escalation, Accept / Reject per source (batch-friendly, since escalation is
already a multi-select flow).

- **Accept:** the source's canonical `labels/` + `classes.txt` are overwritten from the staged
  copy (same swap-and-validate pattern `dataset_panel.py::_sync_xal_stage_back` already uses
  for the X-AnyLabeling round-trip). `level` becomes the staged `target_level`. `reviewed`
  resets to `False` — same meaning as today ("machine output, not yet human-confirmed"), just
  attached to the one existing source instead of spawned onto a new sibling. `sam2_variant` is
  recorded on the source. Staging folder is removed; `pending_escalation` is cleared.
- **Reject:** staging folder is deleted; `pending_escalation` is cleared; source is untouched.
- A pending escalation that's never reviewed survives project close/reopen (it's on disk under
  `artifacts/` and referenced from the saved project file), so the user can come back to it.

`reviewed`'s lifecycle (what sets it back to `True`) is **unchanged by this spec** — there is
no existing "mark as reviewed" action in the codebase today, and adding one is out of scope
here.

## Relationship to prior work (this session)

This session's uncommitted diff added `inspect_al_round`/`materialize_al_round`/
`_add_al_round_sources` to register every AL-round sibling as its own source, in direct
response to a bug report ("cannot import an AL round at all"). That fix was correct as far as
it went — it stopped an import from failing — but the "register every level as a sibling"
shape is exactly what this spec replaces. The net result of implementing this spec is **less
code than what's currently uncommitted**: the single-redirect path (`inspect_detectkit_source`
resolving to the authoritative root) plus its two adversarially-reviewed bug fixes (stale
manifest paths, refuse-if-authoritative-missing) are kept; the multi-root registration
machinery built on top of it is deleted before ever being committed.

## Testing

- **Pure-function level (no Qt):** `materialize_detectkit_source`/`inspect_detectkit_source`
  AL-round redirect behavior (already covered by this session's tests, kept); a new test that
  `_add_source`'s naming uses the original selected path, not the resolved subfolder;
  `al_worker.py` registering exactly one source per round with the round's `native_level`;
  the new `PendingEscalation` to_dict/from_dict round-trip; a staged-escalation
  accept/reject pure-function pair (write staging → accept promotes labels+level+reviewed,
  or reject removes staging and leaves the source untouched).
- **Dialog-level:** `source_manager.py` add-source test asserts exactly one source registered
  for an AL-round pick, named after the round folder. A new review-dialog test drives
  accept and reject paths and asserts project-file state (`pending_escalation` set/cleared,
  `level`/`reviewed` updated only on accept).
- Follows existing patterns in `tests/test_detectkit_source_import.py`,
  `tests/test_detectkit_source_manager_dialog.py`, `tests/test_detectkit_sam2_escalation_wiring.py`.
  Run in the `hydra-mps` conda env. No CUDA/MPS equivalence gate needed — this doesn't touch
  the tracking/inference pipeline.

## Out of scope for this spec

- Automatic or semi-automatic consolidation of a project's existing flat sibling sources
  (decision: manual cleanup only, see "Decisions" above).
- Renaming a registered source after the fact (not requested).
- Canvas rendering of multiple geometry levels (Part B, design note below).
- Clear-labels destructive actions (Part C, design note below).
- Changing `reviewed`'s lifecycle beyond "escalation-accept resets it to False" (no existing
  "mark reviewed" action to hook into).

---

## Design note: Part B — multi-level canvas visualization

Not a committed spec; direction only, to be written up properly once Part A's actual shape
(especially the derivation helpers it ends up needing, if any) is known.

Because Part A means each source has exactly **one** canonical (highest-detail) geometry on
disk, "show all available data types" means: for the image currently displayed, derive and
render every level at or below the source's native level (e.g. a polygon-native source also
shows its derived OBB outline and derived AABB outline, each visually distinguished — color
and/or line style) using the same downward-derivation primitives training already relies on
(`_points_to_min_area_rect` for polygon→OBB-ish, a plain bbox for →AABB). Likely worth
extracting those into one small shared pure-function module so canvas and training call the
same derivation code instead of two independent implementations drifting apart.

`canvas.py` already has the right shape for this — a GT layer and a Pred layer, each with
per-class visibility (`_show_gt`/`_show_pred`/`_visible_class_ids`) built from parallel
item-lists (`_gt_obb_items`, `_pred_obb_items`). Part B likely generalizes that from two
hardcoded layers to an ordered list of per-level layers with their own visibility toggles,
reusing the same drawing primitives (`_draw_detections`).

Before designing new rendering, Part B's plan should include a concrete investigation task:
the user's screenshot of an "aabb" (or escalated) source showing only colored dots with no
box outlines is either an existing, unrelated rendering bug (e.g. a polygon/multi-point label
not being connected into a closed outline) or a red herring caused by viewing an already-stale
flat sibling from before this fix. Root-cause it against Part A's actual output before
building on top of it.

## Design note: Part C — clear-labels actions

Not a committed spec; direction only. Three scopes, each behind a confirmation dialog with an
explicit warning and affected-item count, following the existing `_delete_selected_images`
confirm-dialog in `dataset_panel.py` as the template (same wording style, same
Yes/Cancel-defaults-to-Cancel pattern):

1. **Right-click "Clear labels from frame..."** on the image list (works over the current
   selection, single or multi) — empties the matching label file(s) for the current source,
   leaves the image itself in place. Same context-menu mechanism as the existing
   "Delete image..." action.
2. **"Remove all labels from source"** button under Images — clears every label file for the
   currently selected source (images untouched). One strong confirmation naming the source
   and the image count.
3. **"Remove ALL labels from all sources"** button under Images — same, iterated over every
   registered source in the project. Strongest confirmation given the project-wide blast
   radius (likely a stronger gate than a single Yes/Cancel — e.g. typing the project name, or
   a second confirmation — to be decided when this is turned into a real spec).

All three should share one small, Qt-free helper (`clear_labels_for_source(source_path,
image_paths=None)`) that the per-frame action calls with a narrowing filter and the
per-source/per-project actions call unfiltered — one tested implementation, three UI entry
points.
