# DetectKit SAHI Calibration Profiles — Design

**Date:** 2026-09-01
**Status:** Shipped — merged to main
**Depends on:** shipped SAHI sliced inference and DetectKit sliced-dataset training; the existing DetectKit SAM3 semantic-escalation calibration flow.

## Summary

After training a SAHI-capable direct detector in DetectKit, let the user measure TrackerKit's actual sliced-inference operating points against labelled frames, inspect their trade-offs, and explicitly save one or more named settings profiles with the model.

One weights artifact remains **one model**. Profiles such as `Balanced`, `High recall`, and `Fast scan` are operating modes of that artifact—not copied checkpoints or separate registry entries. A selected profile is stored with the artifact and TrackerKit applies it when the model is selected, while retaining editable controls and a visible `Custom` state.

This follows SAM3 escalation calibration's interaction contract: measure a frontier on the user's own labelled frames and hardware, show visual and numerical evidence, highlight an optional recommendation, but never silently choose or overwrite the user's operating point.

## Scope

### In scope

- Human-in-the-loop SAHI calibration for newly trained, imported, or registered direct OBB, axis-aligned detection, and segmentation models.
- A bounded sweep of full-frame and SAHI geometries, confidence thresholds, and cross-tile merge settings on labelled full-resolution frames.
- A measured calibration frontier with ground-truth/prediction overlays.
- Creation, naming, selection, renaming, and removal of multiple profiles attached to one model artifact, including a user-selected primary profile.
- Versioned model sidecar metadata, TrackerKit profile selection/session persistence, and compatibility with today's flat slice-geometry sidecars.

### Out of scope

- Changing model weights, re-training, or automatically changing sliced-dataset training geometry.
- A global, developer-machine-optimal configuration or automatic full-tracking optimization.
- Saving/replacing a profile merely because a candidate has the highest score.
- Multiple weight copies or model-registry entries for profiles from one model.
- A generic cross-kit preset system; these profiles apply only to direct-detector TrackerKit inference.

## Existing foundation

Sliced DetectKit training already writes a `<model>.slice_meta.json` sidecar during publication and records matching geometry in the model registry. TrackerKit reads it on model selection and pre-fills SAHI mode, geometry, overlap, object fraction, and the model-internal measured body size. It deliberately leaves the tracking project's independent `REFERENCE_BODY_SIZE` alone.

This feature extends that handoff. Calibration must run the same direct-model executor, tile planner, cross-tile merger, and configuration mapping used by TrackerKit. A DetectKit-only approximation is not acceptable: it could select settings that behave differently during real tracking.

## Product workflow

### Train, calibrate, register

After a direct-model training run produces a checkpoint, DetectKit presents a registration review with two explicit paths:

1. **Register with training geometry** preserves today's fast path. The artifact is registered without calibrated profiles, using the geometry created with the sliced dataset when present.
2. **Calibrate for TrackerKit…** opens the workflow below. The user may save several profiles, select a primary one, and then register the one artifact.

Calibration is also available from Run History and from any registered direct model. It can therefore be repeated when labels improve, a collaborator needs a different operating mode, or an imported model needs measurement. Adding a profile later updates metadata beside the same artifact; it never creates another model entry.

Calibration is optional. Existing training, importing, and registration continue to work when it is skipped.

### Choose evidence

The calibration wizard lets the user choose labelled sources and/or their training splits. It defaults to the validation split, explaining that tuning on gradient-update frames makes results optimistic. Where video/session provenance is available, the default must keep neighbouring frames from one recording together.

The user may select another representative labelled set, but must affirm that it is **exhaustively labelled**. A genuine animal missing from ground truth looks like a false positive and biases the result toward overly strict settings—the same safeguard used by SAM3 calibration.

Before inference, show selected frame count, instance count, image-size range, and any sampling cap. The system may display thin-sample evidence but must not offer a recommendation below a documented minimum number of matched instances. Cancellation must never overwrite a prior complete calibration.

### Measure candidates

The first version uses a transparent, bounded grid, not an opaque optimizer. Initial candidates are:

- Full-frame inference (the no-SAHI baseline).
- The sliced-training geometry when it exists.
- Nearby `auto_object` target scales / tile fractions around that geometry.
- A small overlap set for each geometry.
- A user-added `custom` width/height geometry when needed.

The interface states the exact candidate list and estimated tile work before running. Existing tile/frame limits cap the grid. A broader sweep requires explicit user confirmation, rather than creating an accidental multi-hour job.

For every fixed tile plan, run inference once at a low candidate-confidence floor and retain raw per-tile predictions. Re-score the following alternatives offline, without another model call:

- Detector confidence threshold.
- Merge threshold.
- Merge policy/metric where the production merger supports them.

Changing geometry or overlap changes source predictions and requires another inference pass. Re-thresholding/re-merging must use the same production merge semantics, not filter already-final TrackerKit results. The temporary candidate cache is fingerprinted by checkpoint, task, image list, resolved tile plan, executor/image size, and inference parameters; incomplete or incompatible caches are never reused.

### Inspect and choose

DetectKit shows a SAM3-style results frontier. Each measured row reports:

- Geometry mode, resolved tile size, overlap, confidence, merge settings, and whether it is full-frame.
- Tiles/frame and measured seconds/frame on this machine.
- Projected duration for the chosen source set.
- Matched, missed, extra, and duplicate detections per frame.
- Precision, recall, F1, localization quality, and the frames/instances supporting the score.

Selecting a row updates an explorable original-image overlay: ground truth and that row's post-merge predictions can be independently shown/hidden and the user can page through representative frames. Switching rows must not run the model again.

DetectKit can highlight a `Balanced` recommendation, but it is explanatory rather than authoritative. Its proposed rule is to exclude failed/undersampled points, retain a Pareto frontier of misses, extras, and time, then select the lowest-time point within a small error tolerance of the best F1 and with acceptable localization. The rule is printed in the UI.

The user can instead choose a high-recall row that leaves more cleanup, a clean-detection row that misses more animals, or a faster low-tile row. They may save any measured row, including several rows from the same run. Suggested names are `Balanced`, `High recall`, and `Fast scan`; names are unique/editable and can have a short purpose note. The user marks at most one profile **Primary**. Closing the dialog without confirmation saves nothing.

## Detector evaluation semantics

Score full-frame, post-merge predictions against labels in their native frame coordinates. Never score the generated training tiles: the question is which TrackerKit configuration works on original acquisition images.

Matching is one-to-one and class-aware. It uses task-correct geometry:

- OBB: rotated box/polygon overlap with production-compatible geometry.
- Detect: axis-aligned box overlap.
- Segment: mask/polygon overlap.

A documented IoU/quality floor defines a valid match. The report separates misses from extras and identifies duplicate predictions that contend for one object. This prevents a configuration from looking high-recall simply because it emits several overlapping copies.

The calibration service returns measurements, not a hidden scientific objective. Humans choose based on recall, precision, speed, or a balanced compromise. Timings are explicitly measurements on this data and runtime tier, not portable performance guarantees.

## Data model and persistence

### One artifact, multiple profiles

The sidecar changes from today's flat training-geometry payload to a versioned document. The colocated sidecar remains the portable source consumed with the weights; the registry stores a summary/reference for inventory only.

```json
{
  "schema_version": 2,
  "training_geometry": {
    "geometry_mode": "auto_object",
    "imgsz": 640,
    "object_tile_fraction": 0.46875,
    "overlap": 0.2,
    "reference_body_px": 560.0,
    "target_sizes": [200.0, 300.0, 400.0]
  },
  "primary_profile_id": "balanced-8f3c",
  "profiles": [
    {
      "id": "balanced-8f3c",
      "name": "Balanced",
      "note": "Default for routine tracking",
      "settings": {
        "enabled": true,
        "geometry_mode": "auto_object",
        "slice_width": 0,
        "slice_height": 0,
        "overlap": 0.2,
        "object_tile_fraction": 0.46875,
        "trained_body_px": 560.0,
        "confidence_threshold": 0.35,
        "merge_policy": "greedy_nmm",
        "merge_metric": "ios",
        "merge_threshold": 0.5,
        "merge_backend": "cv2"
      },
      "measurement": {
        "created_at": "2026-09-01T14:00:00Z",
        "checkpoint_fingerprint": "sha256:…",
        "task": "obb",
        "label_set_fingerprint": "sha256:…",
        "split": "val",
        "frames": 80,
        "instances": 640,
        "runtime": "mps",
        "seconds_per_frame": 0.42,
        "precision": 0.94,
        "recall": 0.91,
        "f1": 0.92,
        "localization_quality": 0.81
      }
    }
  ]
}
```

`settings` contains every TrackerKit direct-detector inference value the profile claims to set. Its parser validates/clamps the same way `SliceConfig` does; unknown future keys round-trip without altering active behaviour. A profile never writes `REFERENCE_BODY_SIZE`; `trained_body_px` maps to the existing `SLICE_TRAINED_BODY_PX` mechanism.

The project stores the complete measured frontier and visual evidence as calibration history. The portable sidecar retains the selected profiles and compact provenance only; large raw candidate caches and rendered previews stay in the DetectKit project artifact directory.

### Compatibility and mutation rules

- A legacy flat `.slice_meta.json` remains valid and retains today's TrackerKit prefill behaviour.
- On the next DetectKit save, legacy fields become `training_geometry`; no calibration evidence/profile is invented.
- A v2 sidecar with no profiles behaves exactly like its training geometry.
- Missing/corrupt/stale/fingerprint-mismatched calibration data is non-fatal. TrackerKit falls back to training geometry or existing controls and explains the fallback.
- Removing a profile never removes weights, training geometry, other profiles, or historical evidence. Removing the primary profile requires choosing a replacement or explicitly clearing the designation.
- Sidecar updates are atomic. A failed update leaves the prior valid metadata intact and reports failure.

## TrackerKit behaviour

Choosing a direct model reads its sidecar and displays a **SAHI profile** selector:

- With only training geometry, show `Training geometry` and preserve current prefill.
- With profiles, list every saved name and visibly mark the primary one.
- A session with a valid saved profile id restores that profile rather than silently adopting a newly changed primary profile.
- Manual editing after applying a profile displays `Custom (based on <name>)`; it does not mutate the model sidecar.
- Reselecting a profile restores its complete settings.

TrackerKit stores the model path, selected profile id, and effective settings in its session configuration. This preserves reproducibility if another profile is later added. If a named profile is removed or incompatible, TrackerKit keeps valid saved effective settings when possible; otherwise it visibly falls back to the primary profile or training geometry. Switching models never carries settings from the old model into the next one.

Applying a profile is a user selection action. It can prefill and label controls but never makes hidden changes to a running analysis. Non-SAHI models and sidecars without profile metadata remain unaffected.

## Architecture

Separate pure calibration from Qt and registry mutation:

```
core/inference/
    direct_calibration.py       # candidate settings, scoring, matching, cache schema
    slice_meta.py               # versioned sidecar parsing/profile selection
    stages/slicing.py           # existing production tiling, reused
    stages/merge.py             # existing production merge seam, reused

detectkit/
    jobs/direct_calibration.py          # labelled-frame adapter + BaseWorker
    gui/dialogs/direct_calibration_*.py # wizard and results/profile controls
    gui/calibration_preview_store.py    # reuse/generalize SAM3 preview persistence
    gui/dialogs/training_dialog.py      # Review & Register entry point

trackerkit/
    gui/panels/detection_panel.py       # profile selector and Custom state
    gui/orchestrators/config.py         # apply/persist selected profile

training/model_publish.py               # initial v2 sidecar and registration
```

Core accepts paths, plain labels, resolved configurations, and an executor protocol; it imports no DetectKit/TrackerKit types. The DetectKit job adapts source labels, runs via `BaseWorker`, owns cancellation, and saves project-local evidence. TrackerKit only consumes sidecar profile data.

Before UI wiring, factor calibration inference through a reusable direct-executor seam. It must be parity-tested against TrackerKit direct inference for the same model, image, and effective configuration.

## Safeguards

- Disable calibration with a useful reason when the model, labels, runtime, or checkpoint is unavailable.
- Show tile-level progress, cancellation, and partial-result state. Partial work is inspectable but cannot replace complete calibration or become a profile.
- Every profile records checkpoint fingerprint. Replaced weights invalidate its evidence instead of silently applying stale settings.
- Candidate limits obey existing memory/tile protections. Failed candidates appear as failed rows with a reason; they are not silently omitted from recommendations.
- Effective profile settings participate in existing detection cache keys. Switching profiles cannot reuse detections made by another profile.
- Registry and sidecar profile summaries must agree after registration; any disagreement is a visible failure, never a second source of truth.

## Testing and acceptance

### Core tests

- Candidate-grid construction includes full-frame and training geometry, respects work limits, and gives a clear estimate.
- Identical fixed-geometry candidates invoke model inference once; offline confidence/merge re-scoring exactly matches a fresh production run at the same settings.
- Changed geometry/overlap cannot reuse a candidate cache.
- OBB/AABB/segment matching is one-to-one, class-aware, and counts misses/extras/duplicates correctly on synthetic crowded scenes.
- Calibration scores full-resolution source images, not derived training tiles.
- Version migration, validation, unknown-key round trip, primary selection, removal, and legacy-sidecar fallback are deterministic.

### DetectKit tests

- Exhaustive-label acknowledgement blocks the run; validation split is default when present.
- Cancellation/failed candidates preserve prior completed evidence.
- Results show all rows, flag partial work, render overlays without inference, and save only explicit user choices.
- One calibration can save several unique profiles, set/replace/clear primary, and never duplicate weights or registry entries.
- Review & Register supports both skipping calibration and calibrating first; an already registered model supports later calibration.

### TrackerKit/integration tests

- A v2 primary profile applies every claimed field except `REFERENCE_BODY_SIZE`.
- Selecting another profile restores it; a manual edit becomes `Custom` and never writes the sidecar.
- Saved sessions restore profile/effective config; removed/mismatched profiles produce an explicit safe fallback.
- Legacy sidecars retain existing prefill behaviour.
- End-to-end: register one direct model with two profiles, select each in TrackerKit, and prove `InferenceConfig`/detection cache keys differ exactly where the profiles differ.
- Calibration inference and production direct inference have parity for OBB, detect, and segment on fixed synthetic frames.

### Manual acceptance

On representative labelled high-resolution video, a user can compare full-frame with high-recall and fast SAHI rows, inspect their overlays and measured runtime, save both under one model, select either in TrackerKit, and reproduce its exact effective settings. No user choice is inferred or saved automatically.

## Rollout

1. Add v2 sidecar/profile schema and TrackerKit application/custom state while preserving flat sidecars.
2. Build core calibration and parity/scoring tests, initially for OBB.
3. Add DetectKit calibration UI, evidence cache, Review & Register, and multi-profile management.
4. Extend task-specific evaluation to direct detect and segment; run relevant MPS/CUDA verification for production-inference changes.

Label the first release **experimental calibration** until it has been exercised across several real datasets. Its measurements and explicit human choice are useful immediately, while the label avoids overstating generality from a small validation sample.
