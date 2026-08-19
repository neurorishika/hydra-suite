# TrackerKit User/Debug Mode — Design

**Date:** 2026-08-07 (identity-column names refreshed 2026-08-11)
**Status:** Shipped — merged to main (655ec778).
**Scope:** TrackerKit (MAT) tracking pipeline output + GUI

> **Context update (2026-08-11):** the Identity Overhaul (Phases 3-7, merged to `main`
> @`d6fc4f4c`) landed *after* this design was approved and **changed the identity CSV
> vocabulary this spec depends on**. The single, clobbered `IdentityAssigned*` block was
> replaced by three provenance-explicit families — `IdentityEvidence*` (raw per-detection
> evidence summary), `IdentityRealtime*` (the online decoder decision), `IdentityFinal*`
> (the resolved identity, with `IdentityFinalSource ∈ realtime|offline|tag`) — plus a
> retained, now-actually-written `UniqueIdentityKey` (sorted `source=value` tokens). The
> shared column vocabulary lives in `core/individual/identity/columns.py`, and the two CSV
> header builders were already consolidated into `build_tracking_csv_header`. This spec's
> column names, counts, the User-mode `identity` derivation, and the "byte-identical to
> today" baseline have been updated below to that post-overhaul reality. The *design*
> (one Debug Mode toggle; User-mode clean CSV; three writers → one) is unchanged, and it is
> now *easier*: `IdentityFinalLabel` is the honest single resolved identity, so the clean
> column no longer needs a Smoothed→Assigned fallback chain to route around clobbering.

## Problem

TrackerKit's terminal output and its GUI have grown cluttered in a way that hurts the
end user:

- **Output is not clean** — and post-overhaul it is *more* cluttered, which strengthens
  the case. The raw tracking CSVs now carry the 8 core columns
  (`TrackID, TrajectoryID, Index, X, Y, Theta, FrameID, State`), three confidence columns
  (`DetectionConfidence, AssignmentConfidence, PositionUncertainty`), `DetectionID`, and —
  when identity ran — the **three provenance families (~19 identity columns)**:
  `IdentityRealtime{ID,Label,Confidence,Margin,Entropy,Committed,SlotLock}`,
  `IdentityEvidence{TopLabel,Confidence,Sources,ConflictFlag}`,
  `IdentityFinal{Label,ID,Confidence,Source,FragmentScore,SmoothedLabel,SmoothedConfidence,ConflictResolved}`,
  and `UniqueIdentityKey`, plus four AprilTag columns. (Pre-overhaul this was ~10
  `IdentityAssigned*` columns; the honest provenance split traded a clobbered single block
  for more, but explicit, columns.) Heading is `Theta` in radians, there is no time column,
  and three overlapping id-like columns (`TrackID`/`TrajectoryID`/`Index`) confuse users.
- **The GUI scatters ~25 checkboxes across 6 panels** controlling verbosity,
  diagnostics, extra files, and overlays. There is no single notion of "give me the
  clean result" vs "give me everything for debugging."
- **The terminal-export code is itself duplicated/messy.** The final-CSV writer
  `session._save_trajectories_to_csv` has a **verbatim twin** in `media_export.py`, and a
  third representation is produced by `rich_export.write_rich_export_csv`
  (`_with_individual.csv`).

## Goal

Introduce a **single `Debug Mode` toggle** (one checkbox in the UI) that cleanly
separates two experiences:

- **Debug OFF (User mode, default):** the user gets **only the final deliverable** — one
  clean, well-named, well-structured trajectory CSV, plus the annotated video if they
  enabled it. No intermediate CSVs, no diagnostic columns, no diagnostic videos, no
  verbose logs/profiling.
- **Debug ON:** everything is retained — intermediate CSVs, all quality-metric/identity
  diagnostic columns, confidence maps, profiling JSON, verbose logs, and diagnostic
  overlays — so a developer can systematically trace the full system.

The toggle **removes** the scattered diagnostic checkboxes, replacing them with this one
control.

## Non-goals

- No change to the tracking math or the live streaming write path (`worker.py`). (Note:
  the identity overhaul already renamed the identity columns the worker's row-writer emits,
  via the shared `columns.py` vocabulary — that migration is done and gated; this spec adds
  no further change to that path.)
- No change to Debug-mode output schema — it stays byte-identical to **the current
  (post-identity-overhaul) output** so the equivalence gate is unaffected. "Today's output"
  throughout this spec means the merged `main` @`d6fc4f4c` schema (the three identity
  families), not the pre-overhaul `IdentityAssigned*` schema.
- No change to deliberate export workflows (crop dataset, oriented per-track videos,
  active-learning dataset) — those keep their own trigger buttons and run in either mode.
- Caches (`.npz`, `.inference_cache_*`, `_caches/`) are already hidden/internal and are
  out of scope.

## Design decisions (locked)

1. **One checkbox, binary, no granular control.** Either you get no debug content (User
   mode) or you get all debug content for all elements (Debug mode). Default **OFF**.
2. **User-mode deliverable = one consolidated clean CSV** per video, plus the annotated
   video if enabled. Intermediate forward/backward CSVs are cleaned up.
3. **Debug-mode output ≡ today's output** (all files, all columns, current names).
4. **Export workflows stay independent** of the mode (own buttons).
5. **Video appearance controls stay** (markers, ID labels, orientation arrows, trails,
   pose) — legitimate user flexibility. Only *diagnostic* overlays are purged.
6. **Fix the output at the source, do not bolt on a finalizer.** The cleanup lives inside
   the terminal-export code that already owns "present final results," and that code is
   *consolidated* (three writers → one) rather than added to.

## Approach

Chosen: **consolidate the terminal export into one mode-aware writer, and make User mode
a branch of it — not a layer on top.**

Rejected alternatives:
- *Add a post-processing "finalizer" projection.* Rejected: adds a fourth output
  representation and more complexity to an already-complex system.
- *Rename columns in the live `worker.py` CSV writer.* Rejected: touches the hot
  tracking-write path and risks the byte-identical equivalence gate for no benefit.
- *Two saved config presets.* Rejected: doesn't handle renaming/consolidation/cleanup and
  is fragile.

### Component 1 — `debug_mode` config field

- Add `debug_mode: bool` (default `False`) to the TrackerKit config schema, with
  load/save round-tripping in `orchestrators/config.py`.
- At parameter-build time (`build_engine_params` / `get_parameters_dict`), `debug_mode`
  **deterministically sets** the granular diagnostic flags. Those keys remain internal
  (engine unchanged); the UI no longer exposes them:

  | Granular flag (internal) | Debug OFF | Debug ON |
  |---|---|---|
  | `save_confidence_metrics` | false | true |
  | `enable_profiling` | false | true |
  | `debug_logging` | false | true |
  | `export_confidence_density_video` | false | true |
  | `cleanup_temp_files` | true | false |
  | `show_kalman_uncertainty` | false | true |
  | `show_foreground_mask` | false | true |
  | `show_background_model` | false | true |
  | `show_yolo_obb` | false | true |

  (`enable_confidence_density_map` remains driven by its existing logic; only its
  diagnostic *video* export is mode-gated.)

### Component 2 — UI consolidation

- Add a single global **`Debug Mode`** toggle (toolbar/menu action), persisted via
  `debug_mode`.
- **Remove** the diagnostic checkboxes from the panels: debug logging + performance
  profiling (Setup→Debug), save-metrics (Setup→System), Kalman-uncertainty overlay
  (Setup→Preview Overlays), foreground-mask / background-model / YOLO-OBB overlays
  (Detection), density-diagnostic-video (Tracking), auto-cleanup-temp-files
  (Postprocess).
- **Keep** functional controls (detection method, backward tracking, interpolation /
  relinking, use-cached-detections, headless/realtime, "Export trajectory video" on/off),
  video-appearance controls (show markers / ID labels / orientation arrows / trails /
  pose), and independent export workflows (crop dataset, oriented per-track videos,
  active-learning dataset + their 6 review-metric flags).

Disposition summary:

| Bucket | Controls | Disposition |
|---|---|---|
| Debug-gated, removed from UI | debug logging, profiling, save-metrics, Kalman/fg/bg/YOLO-OBB overlays, density diagnostic video, auto-cleanup-temp-files | Deleted from panels; driven by `debug_mode`. |
| Functional, stays | detection method, backward tracking, interpolation/relinking, use-cached-detections, headless/realtime, export-trajectory-video on/off | Untouched. |
| Video appearance, stays | show markers / ID labels / orientation / trails / pose | Untouched (user flexibility). |
| Export workflows, stays | crop dataset, oriented per-track videos, active-learning + 6 metric flags | Own buttons; run in either mode. |

### Component 3 — one mode-aware terminal writer (the core work)

Collapse three overlapping terminal writers into **one** function,
`write_final_trajectories(...)`:

- `session._save_trajectories_to_csv`
- its **verbatim twin** in `media_export.py`
- `rich_export.write_rich_export_csv`

This is a net reduction in code and removes the existing duplication.

The single writer branches on `debug_mode`:

- **Debug branch** → emits **exactly today's** schema and files (final CSV +
  `_with_individual.csv`, all columns, current names). No behavioral change → equivalence
  gate untouched.
- **User branch** → emits **one** consolidated `<video>_tracks.csv` with the clean
  schema below, and does not retain intermediate/final debug CSVs.

The live streaming forward/backward CSVs in `worker.py` are **not touched** — they are
functionally-required intermediates for the bidirectional merge. In User mode they are
cleaned up via the existing cleanup behavior (now forced on when `debug_mode` is off).

**Build on the identity-overhaul groundwork (2026-08-11).** Two consolidations this
component needed are already done: (a) the CSV header is a single source of truth —
`headless_tracking.build_tracking_csv_header` routes through
`core/individual/identity/columns.py` and the GUI orchestrator now calls it (the old
inline header twin was removed); and (b) the identity column names are `columns.py`
constants (`C.FINAL_LABEL`, `C.REALTIME_*`, `C.EVIDENCE_*`, …). The User-mode clean writer
should **map from those constants**, not from string literals, so it can't drift from the
emitted schema. The remaining duplication to collapse is the *final-trajectory* writers
(`session._save_trajectories_to_csv`, its `media_export.py` twin, and
`rich_export.write_rich_export_csv`) — `rich_export`/`media_export` were touched by the
overhaul (overlay priority now `[UniqueIdentityKey, IdentityFinalLabel,
IdentityFinalSmoothedLabel]`) but the three-writer duplication itself remains.

## User-mode CSV schema — `<video>_tracks.csv`

Column naming: `lower_snake_case`. Columns are **conditional** — identity columns appear
only if an identity/tag method ran; pose columns appear only if pose ran. A pure-tracking
run yields the 8 core columns (`id, frame, time_s, x, y, heading_deg, state,
detection_confidence`).

| Clean column | Source | Notes |
|---|---|---|
| `id` | `TrajectoryID` | stable per-animal id |
| `frame` | `FrameID` | integer |
| `time_s` | `frame / fps` | **new**, seconds |
| `x` | `X` | pixels |
| `y` | `Y` | pixels |
| `heading_deg` | `degrees(Theta)`, normalized to `[0, 360)` | **converted** from radians |
| `state` | `State` | `active` / `occluded` / `interpolated` / `lost` |
| `detection_confidence` | `DetectionConfidence` | **always** — the primary downstream filter signal |
| `identity` | the resolved identity: `IdentityFinalLabel` (fall back to `IdentityFinalSmoothedLabel` only if Final is empty) | **only if** identity/tags computed. `IdentityFinalLabel` is now the single honest resolved label — no clobbering — so no Assigned/Smoothed routing chain is needed. |
| `identity_confidence` | `IdentityFinalConfidence` | **only if** identity/tags computed |
| `identity_source` | `IdentityFinalSource` (`realtime` \| `offline` \| `tag`) | **optional, only if** identity ran — a genuinely useful provenance signal for the user (which stage decided the label), not a tracer internal. Include it in User mode. |
| `<kpt>_x`, `<kpt>_y`, `<kpt>_conf` | `PoseKpt_<kpt>_X` / `_Y` / `_Conf` | **only if** pose computed; one triple per keypoint name |

**Confidence policy:** User mode includes the *curated, actionable* confidences people
filter on downstream — `detection_confidence` (always), `identity_confidence`
(= `IdentityFinalConfidence`, if identity ran), and per-keypoint `<kpt>_conf` (if pose
ran). These are per-row **data-quality signals**, not tracer diagnostics, so they belong in
the clean output. Pure tracer internals remain Debug-only: `PositionUncertainty`,
`AssignmentConfidence`, the entire `IdentityRealtime*` family
(`Margin`/`Entropy`/`Committed`/`ID`/`Confidence`/`SlotLock`), the entire
`IdentityEvidence*` family (`TopLabel`/`Confidence`/`Sources`/`ConflictFlag`), and the
non-headline `IdentityFinal*` columns (`FragmentScore`/`SmoothedConfidence`/
`ConflictResolved`/`ID`). Of the identity output, User mode surfaces only the resolved
`IdentityFinalLabel` / `IdentityFinalConfidence` / `IdentityFinalSource`.

**Row policy:** keep **all rows** for all animals across all frames, including
`occluded` / `lost` / `interpolated` frames, with the `state` column indicating provenance
(`lost` rows may have NaN `x`/`y`). This is the most transparent option for the user.

**Dropped in User mode (Debug-only):** `TrackID`, `Index`, `DetectionID`,
`AssignmentConfidence`, `PositionUncertainty`, the `IdentityRealtime*` family
(`ID`, `Label`, `Confidence`, `Margin`, `Entropy`, `Committed`, `SlotLock`), the
`IdentityEvidence*` family (`TopLabel`, `Confidence`, `Sources`, `ConflictFlag`), the
non-headline `IdentityFinal*` columns (`ID`, `FragmentScore`, `SmoothedLabel`,
`SmoothedConfidence`, `ConflictResolved`), pose-summary columns (`PoseMeanConf`,
`PoseValidFraction`, `PoseNumValid`, `PoseNumKeypoints`), heading diagnostics
(`HeadingMethod`, `HeadingIsDirected`, `HeadTailAngleRad`, `HeadTailClassifierConf`), CNN
per-class probability columns, `UniqueIdentityKey` (a technical `source=value` provenance
token — Debug-only; the friendly resolved `IdentityFinalLabel` is what User mode shows),
and AprilTag `Hamming`. (`DetectionConfidence`,
`IdentityFinalLabel`/`IdentityFinalConfidence`/`IdentityFinalSource`, and `PoseKpt_*_Conf`
are **kept** — see the curated confidence policy above.)

## Testing

- **User-mode golden:** commit a golden for the consolidated schema — column set, renames,
  radian→degree conversion, `time_s` derivation, and conditional pose/identity columns —
  including the curated confidences, across three fixtures: `fly_obb` (pure tracking —
  core columns + `detection_confidence`), `ant_cnn_identity` (adds `identity` +
  `identity_confidence`), and `ant_pose_headtail` (adds `<kpt>_x/_y/_conf`).
- **Debug-mode equivalence:** assert Debug-mode output is byte-identical to the current
  post-overhaul `main` (@`d6fc4f4c`) via the existing equivalence harness
  (`tools/equivalence/run_matrix.sh`); debug schema is unchanged by *this* work, so
  current-vs-Debug-branch stays at the determinism floor on **MPS and CUDA**. (The
  attribution baseline is post-overhaul `main`, NOT the pre-overhaul identity schema — the
  identity column rename already landed and is separately gated.)
- **Config round-trip:** `debug_mode` load/save + the derived-flag mapping; update the
  `get_parameters_dict` characterization golden accordingly.
- **UI:** the removed checkboxes no longer exist; the `Debug Mode` toggle persists across
  save/load.
- Update the stale `CSVWriterThread` docstring (`csv_writer.py:21-28`) to reflect the real
  emitted schema.

## Key files

- `src/hydra_suite/trackerkit/config/` (schema) + `orchestrators/config.py` (round-trip,
  param build)
- `src/hydra_suite/trackerkit/gui/panels/{setup,detection,tracking,postprocess}_panel.py`
  (checkbox removal) + `gui/main_window.py` (Debug Mode toggle)
- `src/hydra_suite/core/tracking/session.py`,
  `src/hydra_suite/core/post/{rich_export,export}.py`,
  `src/hydra_suite/core/post/media_export.py` (terminal-writer consolidation)
- `src/hydra_suite/trackerkit/headless_tracking.py` (`build_tracking_csv_header` — already
  the single header source, routing through `columns.py`)
- `src/hydra_suite/core/individual/identity/columns.py` (identity column-name constants —
  the User-mode writer maps identity columns from `C.FINAL_LABEL`/`C.FINAL_CONFIDENCE`/
  `C.FINAL_SOURCE` etc., not string literals)
- `src/hydra_suite/data/csv_writer.py` (docstring)
- `tests/` (new user-mode golden; equivalence + characterization updates. The
  `ant_cnn_identity` fixture now emits the three identity families — the golden's `identity`
  column comes from `IdentityFinalLabel`, `identity_confidence` from
  `IdentityFinalConfidence`, `identity_source` from `IdentityFinalSource`.)
