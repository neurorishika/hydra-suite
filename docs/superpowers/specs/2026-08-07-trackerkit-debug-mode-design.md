# TrackerKit User/Debug Mode — Design

**Date:** 2026-08-07
**Status:** Approved (design), pending implementation plan
**Scope:** TrackerKit (MAT) tracking pipeline output + GUI

## Problem

TrackerKit's terminal output and its GUI have grown cluttered in a way that hurts the
end user:

- **Output is not clean.** The raw tracking CSVs carry up to **25 columns**
  (`TrackID, TrajectoryID, Index, X, Y, Theta, FrameID, State`, three confidence
  columns, ten `Identity*` columns including `IdentityPosteriorMargin`,
  `IdentityEntropy`, `IdentityEvidenceSources`, `IdentityConflictFlag`,
  `IdentitySlotLockLabel`, plus four AprilTag columns). Heading is `Theta` in radians,
  there is no time column, and three overlapping id-like columns
  (`TrackID`/`TrajectoryID`/`Index`) confuse users.
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

- No change to the tracking math or the live streaming write path (`worker.py`).
- No change to Debug-mode output schema — it stays byte-identical to today's output so
  the equivalence gate is unaffected.
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

## User-mode CSV schema — `<video>_tracks.csv`

Column naming: `lower_snake_case`. Columns are **conditional** — identity columns appear
only if an identity/tag method ran; pose columns appear only if pose ran. A pure-tracking
run yields just the 7 core columns.

| Clean column | Source | Notes |
|---|---|---|
| `id` | `TrajectoryID` | stable per-animal id |
| `frame` | `FrameID` | integer |
| `time_s` | `frame / fps` | **new**, seconds |
| `x` | `X` | pixels |
| `y` | `Y` | pixels |
| `heading_deg` | `degrees(Theta)`, normalized to `[0, 360)` | **converted** from radians |
| `state` | `State` | `active` / `occluded` / `interpolated` / `lost` |
| `identity` | best stable label: `UniqueIdentityKey` → `IdentitySmoothedLabel` → `IdentityAssignedLabel`, or AprilTag id | **only if** identity/tags computed |
| `<kpt>_x`, `<kpt>_y` | `PoseKpt_<kpt>_X` / `_Y` | **only if** pose computed; one pair per keypoint name |

**Row policy:** keep **all rows** for all animals across all frames, including
`occluded` / `lost` / `interpolated` frames, with the `state` column indicating provenance
(`lost` rows may have NaN `x`/`y`). This is the most transparent option for the user.

**Dropped in User mode (Debug-only):** `TrackID`, `Index`, `DetectionID`,
`DetectionConfidence`, `AssignmentConfidence`, `PositionUncertainty`, all `Identity*`
diagnostics (`IdentityPosteriorMargin`, `IdentityEntropy`, `IdentityEvidenceSources`,
`IdentityConflictFlag`, `IdentitySlotLockLabel`, `IdentityCommitted`), `PoseKpt_*_Conf`
and pose-summary columns (`PoseMeanConf`, `PoseValidFraction`, `PoseNumValid`,
`PoseNumKeypoints`), heading diagnostics (`HeadingMethod`, `HeadingIsDirected`,
`HeadTailAngleRad`, `HeadTailClassifierConf`), CNN per-class probability columns, and
AprilTag `Conf` / `Hamming`.

## Testing

- **User-mode golden:** commit a golden for the consolidated schema — column set, renames,
  radian→degree conversion, `time_s` derivation, and conditional pose/identity columns —
  across three fixtures: `fly_obb` (pure tracking, 7 core columns), `ant_cnn_identity`
  (adds `identity`), and `ant_pose_headtail` (adds pose keypoint columns).
- **Debug-mode equivalence:** assert Debug-mode output is byte-identical to today via the
  existing equivalence harness (`tools/equivalence/run_matrix.sh`); debug schema is
  unchanged, so legacy-vs-current stays at the determinism floor on **MPS and CUDA**.
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
- `src/hydra_suite/trackerkit/headless_tracking.py` (`build_tracking_csv_header`)
- `src/hydra_suite/data/csv_writer.py` (docstring)
- `tests/` (new user-mode golden; equivalence + characterization updates)
