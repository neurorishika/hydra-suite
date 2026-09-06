# Direct-path scoring unification — D7 + D8 + D9 behind one gate

**Spec:** `docs/superpowers/specs/2026-09-06-unified-sahi-training-geometry-design.md`
**Status:** pending implementation
**Branch:** `feat/direct-scoring-unification`

## Why one branch

The spec's §3.10 sequencing note is binding:

> D7 and D9 both change what direct calibration MEASURES, and D8 changes what it
> OPTIMISES over those measurements. Land them together behind one before/after
> gate, not as three separate silent shifts, or the resulting change in
> recommendations will be unattributable.

R6 (amended) adds a fourth, non-negotiable rider: **rule-versioning must land in the
same change**, and legacy profiles must be labelled `unknown (pre-2026-09-06)` rather
than back-filled.

## The three rulings (verbatim — these are DECIDED, do not re-litigate)

**D7 — direct-path matcher: ADOPT CONTAINMENT, DROP THE IoU GATE.** The direct path
unifies onto the semantic path's inside-guaranteed `representative_point` +
containment matcher. A hard `IoU >= 0.5` as the SOLE criterion counts a correct
silhouette with a different extent convention as a miss AND an extra simultaneously,
because masks trace legs and antennae at ~1.7x labelled body-core area.

**D8 — RECALL-FIRST TO RECOMMEND, AP TO COMPARE. F1 IS RETIRED AS AN OPTIMISATION
TARGET.** F1 **may still be reported** — it is retired as a target, not deleted as a
number. Model-vs-model comparison uses AP. AP is **not comparable across corpora**.

**D9 — shape prior: ADD IT TO THE DIRECT PATH.** Port `fit_area_band` so both
harnesses reject mistargeted detections identically. **Share the code, do not
duplicate it** — this document exists because the tile-size formula reached five
copies.

> WARNING for every implementer: the spec still contains STALE open-decision bullets
> for D7-D11 *below* the RESOLVED block (e.g. "default each path to its current rule
> so nothing changes"). Those are SUPERSEDED. The rulings above are the authority.

## Global Constraints

- Work in a worktree branched from local HEAD. Never push. Commit as the configured
  git user with **no** `Co-Authored-By: Claude` trailer.
- Never `git stash` bare, never `git clean -fdx`, never `rm -rf`.
- Tests: `PYTHONPATH=<worktree>/src python -m pytest ...`. Clear stale numba
  `__pycache__` before any run that touches jitted code.
- **This gate is ATTRIBUTION, not equivalence.** Direct-path scores WILL move by
  design. Only Task 1 (the shared-primitive extraction) is equivalence-bound: the
  semantic path's behaviour must be byte-identical across it.
- Dependency direction: `core/` must not import from any app layer. A module shared
  by the direct and semantic paths lives at `core/inference/`, not under
  `core/inference/semantic/`.
- No second implementation of AP. `core/inference/semantic/detection_metrics.py`
  already owns `average_precision` / `precision_recall_curve` / `OperatingPoint`.

## Current state (mapped 2026-09-06, file:line verified)

Direct path:
- `core/inference/direct_calibration.py:76` `match_frame()` — hard `iou >= 0.5` gate
  (`iou_threshold: float = 0.5` at `:80`), descending-IoU greedy one-to-one,
  class-aware. `detect` reduces both polygons to AABB quads via `_as_task_polygon`
  (`:59-73`) first.
- `:144` `score_frames()` -> `CalibrationScore(precision, recall, f1, mean_iou, ...)`.
- `:247` `recommend_balanced()` — eligibility by `MIN_MATCHED_INSTANCES=60` and
  `MIN_LOCALIZATION=0.5`, then Pareto over `(missed, extra, seconds_per_frame)`, then
  within `F1_TOLERANCE=0.01` of best F1, then fastest. Rule text is the module
  constant `RECOMMENDATION_RULE` (`:188`) and is **not persisted**.
- Persistence: `detectkit/jobs/direct_calibration.py:493` `save_direct_calibration()`
  -> `evidence_dir/direct_calibration.json.gz`, payload has `version`
  (`EVIDENCE_VERSION = 4`, `:323`) but **no rule field**. `:549`
  `load_direct_calibration()`. v1-3 are dropped on load with recorded reasons
  (`:306-322`).

Semantic path:
- `semantic/calibration.py:192` `representative_point(poly) -> np.ndarray` (area
  centroid -> pole of inaccessibility -> first vertex, each verified inside).
- `:130` `_contains(poly, point) -> bool`.
- `:248` `match_one_to_one(pred_polys, label_polys, *, area_band=None,
  min_quality=MIN_MATCH_QUALITY)` — finite guard, area band, containment either way,
  `match_quality >= 0.1`, ranked by quality with distance tie-break, greedy.
- `semantic/shape_prior.py:78` `fit_area_band(label_polys) -> AreaBand | None`;
  `AreaBand` dataclass at `:59`; `MIN_MATCH_QUALITY` at `:56`.
- `:606` `recommend(...)` — recall floor `MIN_RECALL=0.90`, quality floor
  `MIN_MEAN_QUALITY=0.35`, `MIN_MATCHED_INSTANCES=20`, then
  `min(tiles_per_frame, -confidence)`.

Shared today: only `utils/polygon_iou.py`.

---

## Task 1 — Promote the shared scoring primitives (equivalence-bound)

Move, do not copy:

- `semantic/shape_prior.py` -> `core/inference/shape_prior.py` (whole module:
  `AreaBand`, `fit_area_band`, `in_band`, `match_quality`, `MIN_MATCH_QUALITY`, and
  whatever else it exports).
- The matcher geometry out of `semantic/calibration.py` into a new
  `core/inference/match_geometry.py`: `representative_point`, `_contains` (export it
  as `contains`), the finite guards (`_is_finite`, `finite_label` or their current
  names), `_pole_of_inaccessibility`, and `match_one_to_one`.

`semantic/shape_prior.py` and `semantic/calibration.py` keep working by re-exporting
the moved names. Add tests asserting **object identity** (`is`), following the
pattern already used for `detection_metrics.py`, so a future copy-paste fails the
suite rather than silently forking.

Verification: the existing 30 tests in `tests/test_semantic_calibration.py` and the 3
in `tests/test_semantic_calibration_preview.py` pass **unmodified**. If a test needs
editing, the move was not behaviour-preserving — stop and report.

## Task 2 — Rule-versioning into the persisted profile (lands BEFORE the swap)

In `detectkit/jobs/direct_calibration.py`:

- Bump `EVIDENCE_VERSION` 4 -> 5.
- Add to the payload a `recommendation` block carrying at minimum a stable
  machine-readable rule id, a human rule description, and the date the rule took
  effect. The rule id must be a constant sourced from `direct_calibration.py`, not a
  literal at the save site.
- `load_direct_calibration()` must accept **v4** (do not drop it — v4 profiles are
  real and valid *settings*) and label it `rule: unknown (pre-2026-09-06)`.
  **Never back-fill an assumed rule** — R6 is explicit that back-filling asserts
  provenance that does not exist. v1-3 keep their existing drop behaviour.
- Anything that displays a stored recommendation must be able to show which rule
  produced it. If no such display exists, expose the field on the loaded outcome and
  note the absence in the report — do not build GUI.

Tests: v5 round-trip carries the rule; a v4 fixture loads with the unknown label and
is not back-filled; the rule id stamped on save equals the id of the rule the
recommender actually ran.

## Task 3 — Freeze the before-gate corpus

Build a committed characterization golden: a frozen prediction+label corpus and its
scores under the **current** rules (IoU>=0.5 matcher, no area band, F1-tolerance
recommender).

- Corpus lives under `tests/data/direct_calibration_golden/`. It must exercise the
  cases the rulings are about, at minimum: a correct silhouette whose extent
  convention inflates area ~1.7x (IoU<0.5 but contained — the D7 case); a blob
  spanning two animals (the D9 case); a dense cluster where an oversized prediction
  could steal a neighbour's label; degenerate/NaN geometry; and all three tasks
  (`detect`, `obb`, `segment`).
- Generate it deterministically from a committed seed/script so it is reproducible,
  and commit the resulting scores as the **before** side.
- The golden's own test asserts the before-scores at this commit.

Do NOT change any scoring behaviour in this task.

## Task 4 — The swap (D7 + D9 + D8)

- `match_frame()` adopts `match_one_to_one` from `core/inference/match_geometry.py`
  (D7): representative-point + containment, quality ranking, no hard IoU gate.
  Preserve class-awareness and the `detect`-task AABB reduction — those are
  orthogonal to the matcher and must survive.
- `score_frames()` / the sweep threads an `AreaBand` fitted once over the label set
  via the shared `fit_area_band` (D9). One band pooled over the sweep, matching the
  semantic path's `calibrate()` at `:435`.
- `recommend_balanced()` is replaced by a recall-first rule with quality floors (D8),
  mirroring `semantic/calibration.py:606`. Keep reporting `f1` on
  `CalibrationScore`; it must no longer appear in the selection objective.
  `RECOMMENDATION_RULE` and the rule id from Task 2 are updated together — a stale
  rule string stamped onto a new rule is exactly the provenance failure R6 names.
- AP for model-vs-model comparison comes from `semantic/detection_metrics.py`. Do not
  reimplement it.

The floor values are a judgement call: start from the semantic path's
(`MIN_RECALL=0.90`, `MIN_MEAN_QUALITY=0.35`) and state in the report what you chose
for `MIN_MATCHED_INSTANCES` and why (the two paths currently disagree, 60 vs 20).

## Task 5 — The after-gate and the attribution artifact

Re-score the **identical frozen corpus** from Task 3 under the new rules. Commit the
after-scores alongside the before-scores and a short written attribution: for each
metric that moved, which of D7 / D9 / D8 moved it. Score each ruling's contribution
separately where the code allows it (matcher-only, then +band, then +recommender) so
the attribution is measured rather than asserted.

The deliverable is a committed artifact a reader can check months later, in the shape
of the semantic fix's own gate (recall 0.867 -> 0.988, identical predictions, only the
scorer changing).

## Out of scope

Multi-scale SAM3, D16/D18/D19, the `ScaleGroupedBatchSampler` (R7), GUI surfaces, and
`comparison_baseline`. Named here so no implementer wanders into them.
