# Active Learning Detection Dataset Generation — Design

**Date:** 2026-05-03
**Status:** Approved for planning
**Scope:** Improve efficiency and utility of detection-dataset generation for active learning. Affects `data/dataset_generation.py`, the TrackerKit dataset path, and adds a DetectKit-side AL loop that uses the trained detector itself.

## Goals

1. Make AL frame selection less wasteful — avoid near-duplicate frames and frames the model already handles confidently.
2. Make AL frame selection more useful — surface frames where the *current detection model* is most likely to be wrong.
3. Avoid feature creep: no new feature extractors, no MC-Dropout, no prior-model disagreement, no multi-round state machine.
4. Share scoring/diversity primitives between TrackerKit and DetectKit instead of duplicating them.

## Non-goals

- No feature-space novelty / embedding-based "novelty vs labeled set" — handled upstream by FilterKit's existing perceptual-hash + histogram dedup.
- No changes to `export_dataset` annotation format on the tracker path.
- No new compute runtime work — reuses existing `compute_runtime` selection.
- No automatic multi-round state tracking — each AL run is independent; rounds are successive project sources.

## Architecture

```
                ┌────────────────────────────────────────┐
                │  hydra_suite/data/al/  (NEW SHARED)    │
                │  - frame_source.py  (Protocol + adapters) │
                │  - candidate_pool.py (FilterKit hook)     │
                │  - signals.py    (ALSignals + scorers)    │
                │  - acquisition.py (rank + select)         │
                └────────────────────────────────────────┘
                              ▲                ▲
   ┌──────────────────────┐   │                │   ┌────────────────────────────┐
   │ TrackerKit (existing)│   │                │   │ DetectKit (NEW)            │
   │ data/dataset_generation│ │                │   │ detectkit/jobs/al_worker.py│
   │ + dataset_worker.py  │───┘                └───│ detectkit/gui/dialogs/     │
   │ (refactored to use   │                        │   active_learning.py       │
   │  shared al/* core)   │                        │                            │
   └──────────────────────┘                        └────────────────────────────┘
```

**Layer rules:** `hydra_suite/data/al/` lives at the data layer. No GUI imports. Importable from CLI/scripts/tests. Both kits depend down into it. The tracker path keeps its tracker-specific signals (assignment cost, track loss, position uncertainty) as adapter logic that produces the same `ALSignals` dataclass.

## Components

### `data/al/frame_source.py`

Single Protocol plus three concrete adapters.

```python
@dataclass
class FrameRef:
    source_id: str
    frame_id: int
    path: str | None  # None for video-derived frames

class FrameSource(Protocol):
    def __iter__(self) -> Iterator[FrameRef]: ...
    def read(self, ref: FrameRef) -> np.ndarray: ...
    def length(self) -> int: ...
```

Adapters:

- `VideoFrameSource(video_path, stride=1)` — yields refs by frame index, `read` decodes via `cv2.VideoCapture`.
- `ImageFolderFrameSource(folder)` — yields refs by sorted file order, `read` via `cv2.imread`.
- `DetectKitProjectSource(project, only_unlabeled=True)` — yields refs from project sources, optionally restricted to images that have no label file or are flagged `pending_review`.

### `data/al/candidate_pool.py`

Thin wrapper around `filterkit.core.FilterKitCore`. Builds a diverse candidate pool from a `FrameSource` before any detector inference runs.

Knobs (via `CandidatePoolConfig` dataclass):

- `subsample_stride: int = 1` — keep every Nth frame.
- `dedup_method: Literal["phash", "ahash", "histogram", "none"] = "phash"`.
- `dedup_threshold: int = 8` — Hamming distance for hash methods, Bhattacharyya for histogram.
- `max_candidates: int | None = None` — hard cap.

Returns `list[FrameRef]`. Cheap — runs entirely on CPU, no model load.

### `data/al/signals.py`

```python
@dataclass
class ALSignals:
    frame_id: int
    n_detections: int
    mean_confidence: float
    margin: float                # weakest detection's gap above conf floor
    nms_instability: float       # set-IoU drift across NMS perturbations
    count_deviation: float       # |n - expected| / max(expected, 1)
    crowd_score: float           # max pairwise OBB IoU
    edge_score: float            # max box-edge proximity to frame border
    extras: dict[str, float]     # tracker-side: assignment_cost, track_loss, position_uncertainty
```

Scorer functions (pure, take detector output, return signal floats):

- `score_uncertainty(confidences) -> (mean_confidence, margin)` — average confidence and minimum margin above a fixed floor (0.5 default, configurable).
- `score_nms_instability(detector, frame, base_conf, base_iou) -> float` — re-run NMS at three (conf, iou) settings: `(base_conf, base_iou)`, `(base_conf*0.7, base_iou)`, `(base_conf, min(base_iou*1.3, 0.95))`. Compute pairwise set-IoU between detection sets (greedy match by box IoU > 0.5); return `1 - mean(set_IoU)`. ~3-5x inference cost vs single pass.
- `score_count_deviation(n, expected) -> float` — clipped to `[0, 1]`.
- `score_crowd(obb_corners) -> (crowd_score, edge_score)` — reuses `_polygon_overlap_ratio` from existing `dataset_generation.py`.

Each scorer is independently testable with synthetic inputs.

### `data/al/acquisition.py`

```python
@dataclass
class AcquisitionWeights:
    uncertainty: float = 0.40
    nms_instability: float = 0.20
    count: float = 0.20
    crowd: float = 0.15
    edge: float = 0.05
    # Tracker-only extras (used only when present in ALSignals.extras):
    assignment: float = 0.0
    track_loss: float = 0.0
    position_uncertainty: float = 0.0

PRESETS = {
    "balanced": AcquisitionWeights(
        uncertainty=0.40, nms_instability=0.20, count=0.20,
        crowd=0.15, edge=0.05,
    ),
    "uncertainty_heavy": AcquisitionWeights(
        uncertainty=0.55, nms_instability=0.25, count=0.10,
        crowd=0.05, edge=0.05,
    ),
    "exploration_heavy": AcquisitionWeights(
        uncertainty=0.25, nms_instability=0.15, count=0.15,
        crowd=0.30, edge=0.15,
    ),
    "tracker_default": AcquisitionWeights(
        uncertainty=0.30, nms_instability=0.0, count=0.20,
        crowd=0.15, edge=0.05, assignment=0.15,
        track_loss=0.10, position_uncertainty=0.05,
    ),
}
```

All weights normalized to sum to 1.0 at use; warn if user-supplied weights diverge by more than 1%.

`select(signals: list[ALSignals], weights, k, diversity_window=30, frame_hashes=None) -> list[FrameRef]`:

1. Min-max normalize each signal channel across the candidate pool.
2. Score = weighted sum.
3. Rank-based probabilistic sampling (keeps existing tracker behavior).
4. Diversity guard: enforce `diversity_window` in frame index; if `frame_hashes` provided, also enforce minimum perceptual-hash Hamming distance between picks.

### `detectkit/jobs/al_worker.py`

`BaseWorker` subclass. Pipeline:

1. Build `FrameSource` from user input.
2. `candidate_pool.build()` → diverse candidate refs.
3. Load DetectKit project's active model via `compute_runtime` (same artifact loader as `evaluation.py`).
4. Batched inference over candidates → `list[ALSignals]`.
5. Compute perceptual hashes for picked candidates (reuse FilterKit `_compute_phash`).
6. `acquisition.select(signals, weights, k, frame_hashes)` → top-K refs.
7. For each picked ref: write image + YOLO-OBB label file seeded with **full model predictions** (per user preference: easier to delete than draw).
8. Register a new project source named `al_round_<timestamp>` via the existing source-registration path. No new metadata flag introduced.

Uses cooperative cancellation via `_should_stop()` like other workers. Emits `progress_signal`, `finished_signal(source_id, n_picked)`, `error_signal`.

### `detectkit/gui/dialogs/active_learning.py`

Modal `BaseDialog`. Three sections:

1. **Input** — radio (Video / Image folder / Existing project source) + path picker; live label "≈ N candidates after FilterKit subsampling" updated on knob change.
2. **Acquisition** — model dropdown (defaults to project active model), preset dropdown (`balanced` / `uncertainty_heavy` / `exploration_heavy` / custom), expected-count spinner, budget (top-K) spinner. Collapsible "Advanced" expands per-weight sliders.
3. **Execution** — Run / Cancel buttons; progress bar + status; thumbnail grid of top-K with per-frame signal breakdown on hover; "Import to project" confirms the round.

### Tracker-side refactor

- `FrameQualityScorer` becomes a thin adapter that produces `ALSignals` (with tracker extras populated). Public methods `score_frame` and `get_worst_frames` retain signatures for API compatibility.
- `get_worst_frames` delegates to `acquisition.select` with the `tracker_default` preset.
- Hard-coded weights (currently sum to 1.3) are replaced. The current tracker UI's "Quality threshold" control becomes "Min selection score (0.0–1.0)" defaulting to 0.0 — under normalized scoring an unbounded threshold is meaningless. **This is a user-visible UI behavior change**; tooltip and migration note added.
- Annotation seeding on tracker path stays unchanged (tracker CSV).

## Data flow

```
DetectKit AL run:
  user input ─► FrameSource ─► FilterKit subsample ─► candidate FrameRefs
       │
       ├─► load model via compute_runtime
       │
       └─► batched inference ─► per-frame ALSignals ─► acquisition.select
                                                              │
                                                              ├─► pick top-K refs
                                                              │
                                                              └─► for each: write image + YOLO-OBB seeds (full predictions)
                                                                                    │
                                                                                    └─► new project source "al_round_<timestamp>"

TrackerKit AL run (existing entry, refactored internals):
  tracking loop ─► per-frame ALSignals (incl. tracker extras) ─► acquisition.select ─► export_dataset (unchanged)
```

## Error handling

| Failure | Behavior |
|---|---|
| No model selected / model load fails | Dialog blocks Run with inline message. Optional toggle "Run without model" runs a one-shot detector pass at low conf to seed signals (no NMS-instability). |
| Empty candidate pool after FilterKit dedup | Dialog surfaces "0 candidates after dedup; relax threshold or stride." No crash. |
| Existing AL round name collision | Suffix with timestamp (matches `_make_dataset_dir`). |
| GPU OOM during batched inference | Catch, halve batch, retry once; on second failure fall back to single-frame loop (mirrors `_detect_batch` fallback in existing `dataset_generation.py`). |
| User cancels mid-run | `_should_stop()` checked every batch; partial results discarded — no half-imported project source. |

## Testing

- **Unit:** each scorer in `signals.py` against synthetic detection inputs (covers entropy, margin, NMS-instability set-IoU, count, crowd/edge overlap).
- **Unit:** `acquisition.select` with controlled signal arrays — verify weight-preset behavior, diversity guard, top-K determinism with fixed numpy seed.
- **Unit:** `candidate_pool.build` over a synthetic image stream including known near-duplicates — verify dedup drops them.
- **Integration:** `al_worker` end-to-end against the existing 50-frame test fixture and a tiny YOLO checkpoint — assert top-K count, label files exist and are valid YOLO OBB, project source registered with `pending_review` flag.
- **Regression:** tracker-side `dataset_worker` against the existing fixture — verify refactored path selects the same frame set at the `tracker_default` preset, or document the diff if intentional and approved.
- **Smoke:** dialog construction and show without crashing (no headless rendering test required).

Benchmarks: not part of acceptance, but informational comparison of "FilterKit-pool size vs full-frame-set scoring" wall-clock on the test fixture.

## Migration notes

- `FrameQualityScorer.score_frame` and `get_worst_frames` keep their signatures — third-party scripts importing them keep working.
- `params["DATASET_CONF_THRESHOLD"]` semantics change from "average confidence below this triggers low-confidence flag" to "minimum normalized acquisition score for selection." Old configs continue to load; the new meaning is documented in the tracker dataset panel tooltip and in `docs/developer-guide/`.
- New module path: `hydra_suite.data.al`. No breaking changes elsewhere.

## Build sequence

1. **`data/al/` core** (frame_source, candidate_pool, signals, acquisition) with full unit tests.
2. **TrackerKit refactor** — `FrameQualityScorer` and `get_worst_frames` delegate to the new core. Run regression test.
3. **DetectKit AL worker** — non-GUI, exercised by integration test before any UI work.
4. **DetectKit AL dialog** — UI surface, smoke-tested.
5. **Tracker UI cleanup** — replace "Quality threshold" widget with "Min selection score" + tooltip.

Each step is independently mergeable.
