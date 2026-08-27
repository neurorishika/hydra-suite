# Identity Final-Output Consistency Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the final tracking CSV carry exactly one identity per trajectory, no NaN positions, and explicit values (or explicit denials) in every identity cell, and make the offline fragment solver faithful to the evidence it is given.

**Architecture:** (1) Vocabulary/record fixes in `identity/columns.py`, `identity/offline.py`, `postprocess_df.py`, `trajectory_writer.py`. (2) Solver rewrite of the assignment core in `identity/offline.py` (`_iterative_assign` + support construction), no changes to evidence loading/smoothing/PELT. (3) Position densification + trimming in `core/post/processing.py`, wired from `session.py`/`merge.py`. (4) Pipeline reorder: `apply_identity_postprocessing_to_df` split into `derive_identity_keys` + `resolve_identity`; `rich_export.relink_and_export_rich_csv` relinks first, resolves once, checks the invariant. (5) `media_export` label precedence. (6) Equivalence fixture + gate record + DEMO/ID acceptance.

**Tech Stack:** Python 3.11+, pandas, numpy, pytest. Conda env `hydra-mps` (`source ~/miniforge3/etc/profile.d/conda.sh 2>/dev/null || source ~/mambaforge/etc/profile.d/conda.sh; conda activate hydra-mps`); `export KMP_DUPLICATE_LIB_OK=TRUE` before importing `hydra_suite`. Run tests from the worktree root with `PYTHONPATH=<worktree>/src`.

**Spec:** `docs/superpowers/specs/2026-08-27-identity-final-consistency-design.md`

## Global Constraints

- Work only inside the worktree `.worktrees/identity-final` (branch `feat/identity-final-consistency`, from local `main` @ `f2d4ca36`). `PYTHONPATH=<worktree>/src` on every pytest/python invocation (an editable install of the main checkout would otherwise be imported).
- `IdentityFinalSource` vocabulary after Task 1: `offline | realtime | tag | nonidentifying | none`. The empty string is never written; readers treat `""`/NaN as `none`.
- The offline solver must never read `IDENTITY_DISPLAY_THRESHOLD` after Task 4. `substrate.solve_unique_assignment` is untouched (realtime path).
- No `IdentityRealtime*` column is ever written by post-processing (existing invariant; `tests/identity/test_provenance_no_clobber.py` guards it).
- Positions of detection rows (`State=active`) are never modified by this plan; only occluded/interpolated rows are added, filled, or dropped.
- Never import from `legacy/`. Run `make format` (black + isort) before each commit. Commit as the configured git user (no Co-Authored-By trailer).
- Existing tests whose assertions encode the *old* behaviour being replaced (temperature weighting, display-threshold gating of the solver/smoothed columns, `NONE == ""`, `UniqueIdentityKey`-first overlay) are updated in the task that changes the behaviour, with a one-line comment naming this plan; every other existing test must keep passing.

---

### Task 1: Explicit source vocabulary and boolean conflict flag

**Files:**
- Modify: `src/hydra_suite/core/individual/identity/columns.py:60`
- Modify: `src/hydra_suite/core/individual/postprocess_df.py:146`
- Modify: `src/hydra_suite/core/individual/identity/offline.py:1047-1086` (`_ensure_final_columns`)
- Modify: `src/hydra_suite/core/post/trajectory_writer.py` (`write_final_trajectories`, `project_user_tracks` block at ~150-186)
- Modify: `src/hydra_suite/core/post/identity_postprocess.py` (add `normalize_final_source_series`)
- Modify: `tests/identity/test_identity_columns.py:test_final_source_vocabulary`
- Test: `tests/identity/test_final_source_explicit.py`

**Interfaces:**
- Produces: `C.IdentityFinalSource.NONE == "none"`; `identity_postprocess.normalize_final_source_series(s: pd.Series) -> pd.Series` (NaN/`""` → `"none"`, others stripped); `C.FINAL_CONFLICT_RESOLVED` boolean everywhere in the written CSV.

- [ ] **Step 1: Write the failing tests**

```python
# tests/identity/test_final_source_explicit.py
import numpy as np
import pandas as pd

from hydra_suite.core.individual.identity import columns as C
from hydra_suite.core.individual.identity.offline import _ensure_final_columns
from hydra_suite.core.post.identity_postprocess import normalize_final_source_series
from hydra_suite.core.post.trajectory_writer import write_final_trajectories


def test_none_is_explicit_token():
    assert C.IdentityFinalSource.NONE == "none"


def test_normalize_final_source_series_maps_blank_and_nan_to_none():
    s = pd.Series([np.nan, "", "  ", "offline", " tag "])
    out = normalize_final_source_series(s)
    assert out.tolist() == ["none", "none", "none", "offline", "tag"]


def test_ensure_final_columns_creates_conflict_flag_false_only_when_absent():
    df = pd.DataFrame({"TrajectoryID": [0, 0], "FrameID": [1, 2]})
    out = _ensure_final_columns(df)
    assert out[C.FINAL_CONFLICT_RESOLVED].tolist() == [False, False]
    assert out[C.FINAL_SOURCE].tolist() == ["none", "none"]
    # existing merge-time True must survive
    df2 = pd.DataFrame(
        {"TrajectoryID": [0, 0], "FrameID": [1, 2], C.FINAL_CONFLICT_RESOLVED: [True, np.nan]}
    )
    out2 = _ensure_final_columns(df2)
    assert bool(out2[C.FINAL_CONFLICT_RESOLVED].iloc[0]) is True
    assert pd.isna(out2[C.FINAL_CONFLICT_RESOLVED].iloc[1])  # untouched; the writer fills NaN -> False


def test_written_csv_has_no_blank_source_and_boolean_conflict(tmp_path):
    final_csv = tmp_path / "clip_final.csv"
    df = pd.DataFrame(
        {
            "TrajectoryID": [0, 0, 1],
            "FrameID": [1, 2, 1],
            "X": [1.0, 2.0, 3.0],
            "Y": [1.0, 2.0, 3.0],
            "Theta": [0.0, 0.0, 0.0],
            "State": ["active"] * 3,
            "DetectionID": [1, 2, 3],
            C.FINAL_LABEL: ["ant_a", "ant_a", "unknown"],
            C.FINAL_ID: [1, 1, 0],
            C.FINAL_CONFIDENCE: [0.9, 0.9, 0.0],
            C.FINAL_SOURCE: ["offline", "offline", np.nan],
            C.FINAL_CONFLICT_RESOLVED: [True, np.nan, np.nan],
        }
    )
    out = write_final_trajectories(df, str(final_csv), debug_mode=True, fps=10.0)
    written = pd.read_csv(out)
    assert written[C.FINAL_SOURCE].tolist() == ["offline", "offline", "none"]
    assert written[C.FINAL_CONFLICT_RESOLVED].tolist() == [True, False, False]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src python -m pytest tests/identity/test_final_source_explicit.py -v`
Expected: FAIL (`NONE == ""`, `normalize_final_source_series` missing, conflict column NaN).

- [ ] **Step 3: Implement**

`columns.py:60`: `NONE = "none"` with docstring: *Explicit "no identity was resolved for this row" token. Never write `""`; readers normalise `""`/NaN to this value (`identity_postprocess.normalize_final_source_series`).*

`identity/columns.py` (new function, after the `IdentityFinalSource` class — it lives here, not in `core/post`, because `offline.py` needs it and `core/individual` must not import `core/post`):
```python
def normalize_final_source_series(source: "pd.Series") -> "pd.Series":
    """Map NaN / blank ``IdentityFinalSource`` cells (legacy CSVs, columns
    created before the solver ran) to the explicit ``IdentityFinalSource.NONE``
    token; strip whitespace from real tokens."""
    import pandas as pd  # local: columns.py is otherwise dependency-free

    token = source.astype(object).where(source.notna(), "").astype(str).str.strip()
    return token.where(token != "", IdentityFinalSource.NONE)
```
`identity_postprocess.py` re-exports it: `from hydra_suite.core.individual.identity.columns import normalize_final_source_series  # noqa: F401` (and lists it in `__all__` if one exists).

`postprocess_df.py:146`: replace `df[C.FINAL_SOURCE].fillna("").astype(str).str.strip()` with `C.normalize_final_source_series(df[C.FINAL_SOURCE])`.

`offline.py _ensure_final_columns`: after the `FINAL_FRAGMENT_SCORE` block add
```python
    if C.FINAL_CONFLICT_RESOLVED not in out.columns:
        out[C.FINAL_CONFLICT_RESOLVED] = False
```
and, after the `FINAL_SOURCE` create/coerce branches, `out[C.FINAL_SOURCE] = C.normalize_final_source_series(out[C.FINAL_SOURCE])`.

`trajectory_writer.write_final_trajectories`: at the top, before either branch:
```python
    rich_df = rich_df.copy()
    if C.FINAL_SOURCE in rich_df.columns:
        rich_df[C.FINAL_SOURCE] = C.normalize_final_source_series(rich_df[C.FINAL_SOURCE])
    if C.FINAL_CONFLICT_RESOLVED in rich_df.columns:
        rich_df[C.FINAL_CONFLICT_RESOLVED] = (
            rich_df[C.FINAL_CONFLICT_RESOLVED]
            .map(lambda v: bool(v) if pd.notna(v) and str(v).strip().lower() not in ("", "nan", "false", "0") else False)
            .astype(bool)
        )
```
Also in `project_user_tracks`, `out["identity_source"] = C.normalize_final_source_series(df[C.FINAL_SOURCE])`.

Update `tests/identity/test_identity_columns.py::test_final_source_vocabulary` to assert `NONE == "none"` (comment: `# 2026-08-27 identity-final-consistency: explicit denial token`).

- [ ] **Step 4: Run tests**

Run: `PYTHONPATH=src python -m pytest tests/identity/test_final_source_explicit.py tests/identity/test_identity_columns.py tests/identity/test_non_identifying_classes.py tests/identity/test_offline_evidence_breaker.py tests/test_core_identity_postprocess_df.py tests/test_rich_export_golden.py tests/test_rich_export_mode_aware.py -v`
Expected: PASS. If `test_rich_export_golden.py` compares a committed golden containing blank sources, regenerate the golden per that test's documented procedure and say so in the report.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "fix(identity): explicit IdentityFinalSource=none and boolean ConflictResolved in the written CSV"
```

---

### Task 2: Smoothed columns are a record, not a display

**Files:**
- Modify: `src/hydra_suite/core/individual/identity/offline.py:1259-1316` (`_annotate_smoothed_labels`)
- Modify: `src/hydra_suite/core/individual/identity/smoothing.py:298-340` (`smoothed_label_and_conf` gains `display_threshold: float | None`)
- Test: `tests/identity/test_offline_smoothed_record.py`

**Interfaces:**
- Produces: `_annotate_smoothed_labels(df, smoothed_by_traj, catalog, params)` writes, for every row of a trajectory with cache evidence, the argmax known label and its posterior; rows without evidence get `"unknown"`/`0.0`. `smoothed_label_and_conf(smoothed, catalog, display_threshold=None)` — `None` disables the gate (returns argmax + posterior always).

- [ ] **Step 1: Write the failing tests**

```python
# tests/identity/test_offline_smoothed_record.py
import numpy as np
import pandas as pd

from hydra_suite.core.individual.identity import columns as C
from hydra_suite.core.individual.identity.catalog import IdentityCatalog
from hydra_suite.core.individual.identity.offline import _annotate_smoothed_labels
from hydra_suite.core.individual.identity.smoothing import smoothed_label_and_conf


def _lp(p_a):
    p = np.array([1e-6, p_a, 1.0 - p_a - 1e-6])
    p = p / p.sum()
    return np.log(p)


def test_smoothed_label_and_conf_without_threshold_reports_argmax():
    cat = IdentityCatalog.from_labels(["ant_a", "ant_b"])
    out = smoothed_label_and_conf([_lp(0.55), _lp(0.05)], cat, display_threshold=None)
    assert out[0][0] == "ant_a" and abs(out[0][1] - 0.55) < 1e-3
    assert out[1][0] == "ant_b" and abs(out[1][1] - 0.95) < 1e-3


def test_annotate_writes_low_confidence_rows_and_unknown_for_no_evidence():
    cat = IdentityCatalog.from_labels(["ant_a", "ant_b"])
    df = pd.DataFrame(
        {"TrajectoryID": [7, 7, 7], "FrameID": [1, 2, 3], "DetectionID": [1, 2, np.nan]}
    )
    smoothed = {7: [(1, _lp(0.55)), (2, _lp(0.99))]}  # frame 3 has no evidence
    out = _annotate_smoothed_labels(df, smoothed, cat, {"IDENTITY_DISPLAY_THRESHOLD": 0.95})
    assert out[C.FINAL_SMOOTHED_LABEL].tolist() == ["ant_a", "ant_a", "unknown"]
    assert abs(out[C.FINAL_SMOOTHED_CONFIDENCE].iloc[0] - 0.55) < 1e-3
    assert out[C.FINAL_SMOOTHED_CONFIDENCE].iloc[2] == 0.0
```

- [ ] **Step 2: Run to verify failure** — `PYTHONPATH=src python -m pytest tests/identity/test_offline_smoothed_record.py -v` → FAIL (row 0 blank; row 2 `""`).

- [ ] **Step 3: Implement**

`smoothing.smoothed_label_and_conf`: signature `display_threshold: float | None`; inside the loop, `if display_threshold is not None and best_prob < display_threshold: results.append(("", 0.0))` else append `(label, best_prob)`. Keep the docstring's realtime note but state that `None` = ungated record.

`offline._annotate_smoothed_labels`: initialise the label column to `"unknown"` and confidence to `0.0` **for every row** (not `""`); call `smoothed_label_and_conf(log_probs, catalog, display_threshold=None)`; remove the `display_threshold = float(params.get("IDENTITY_DISPLAY_THRESHOLD", 0.6))` read. Update the docstring: *record of the cache-evidence forward-backward posterior; `unknown`/0.0 means no cache evidence joined this row (e.g. crop-pass rows with no DetectionID) — never a thresholded blank.*

- [ ] **Step 4: Run** — the new file plus `tests/identity/test_offline_smoothing.py tests/identity/test_honesty_fix.py tests/test_fragment_solver.py tests/test_rich_export_golden.py`. Any existing assertion expecting `""` under the threshold is updated with the plan-name comment.

- [ ] **Step 5: Commit** — `git commit -m "fix(identity): smoothed label/confidence are an ungated record; unknown/0.0 when no evidence joined"`

---

### Task 3: Evidence-faithful support (convex informative weights + support floor)

**Files:**
- Modify: `src/hydra_suite/core/individual/identity/offline.py` — add `_combined_support(frag_row, known_labels, params) -> dict[str, float]` and `FRAGMENT_MIN_SUPPORT` reading; use it in `_iterative_assign` (line ~542-552) and `solve_global_assignment` (line ~1170-1185).
- Modify: `src/hydra_suite/trackerkit/engine_params.py:1073-1100` — add `"FRAGMENT_MIN_SUPPORT": float(_cfg_get(cfg, "fragment_min_support", default=0.5))`.
- Modify: `src/hydra_suite/resources/configs/default.json` (add `"fragment_min_support": 0.5` next to `fragment_cnn_weight`) and `src/hydra_suite/trackerkit/config/schemas.py` if it enumerates identity keys (grep `fragment_cnn_weight`; mirror every site).
- Test: `tests/identity/test_offline_support.py`

**Interfaces:**
- Produces: `_combined_support(frag_row, known_labels, params)`; `params["FRAGMENT_MIN_SUPPORT"]`.

- [ ] **Step 1: Tests**

```python
# tests/identity/test_offline_support.py
import math

import pandas as pd

from hydra_suite.core.individual.identity.offline import _combined_support

K = [f"l{i}" for i in range(25)]


def _row(cnn_log=None, tag_log=None, online="unknown", conf=0.0):
    return pd.Series(
        {
            "CNNLogEvidence": cnn_log or {},
            "TagLogEvidence": tag_log or {},
            "OnlineLabel": online,
            "OnlineConfidence": conf,
        }
    )


def test_single_cnn_source_is_not_flattened_by_small_weight():
    cnn = {l: (math.log(0.99999) if l == "l0" else math.log(1e-5 / 24)) for l in K}
    sup = _combined_support(_row(cnn_log=cnn), K, {"FRAGMENT_CNN_WEIGHT": 0.1})
    assert sup["l0"] > 0.999


def test_uninformative_prior_does_not_count_as_a_source():
    cnn = {l: (math.log(0.99999) if l == "l0" else math.log(1e-5 / 24)) for l in K}
    sup = _combined_support(
        _row(cnn_log=cnn, online="not_in_catalog", conf=0.9),
        K,
        {"FRAGMENT_CNN_WEIGHT": 0.4, "ONLINE_PRIOR_WEIGHT": 0.25},
    )
    assert sup["l0"] > 0.999


def test_informative_prior_is_blended_convexly():
    cnn = {l: (math.log(0.6) if l == "l0" else math.log(0.4 / 24)) for l in K}
    sup_no = _combined_support(_row(cnn_log=cnn), K, {"FRAGMENT_CNN_WEIGHT": 1.0})
    sup_pr = _combined_support(
        _row(cnn_log=cnn, online="l1", conf=0.9),
        K,
        {"FRAGMENT_CNN_WEIGHT": 1.0, "ONLINE_PRIOR_WEIGHT": 1.0},
    )
    assert sup_pr["l1"] > sup_no["l1"] and sup_pr["l0"] < sup_no["l0"]
    assert abs(sum(sup_pr.values()) - 1.0) < 1e-9


def test_no_sources_gives_uniform():
    sup = _combined_support(_row(), K, {})
    assert all(abs(v - 1 / 25) < 1e-9 for v in sup.values())
```

- [ ] **Step 2: Run → FAIL** (`_combined_support` missing).

- [ ] **Step 3: Implement**

```python
def _combined_support(
    frag_row: pd.Series, known_labels: list[str], params: dict[str, Any]
) -> dict[str, float]:
    """Normalised per-label support from the *informative* sources of one
    fragment, blended convexly: ``Σ w_s·log_s / Σ w_s``. With one informative
    source this is that source's geometric-mean posterior itself -- the
    ``FRAGMENT_*_WEIGHT`` knobs are relative source weights, never a softmax
    temperature (the 2026-08-27 audit found ``cnn_w=0.1`` flattening 0.99999
    evidence to 0.2 support)."""
    cnn_w = float(params.get("FRAGMENT_CNN_WEIGHT", 0.40))
    tag_w = float(params.get("FRAGMENT_TAG_WEIGHT", 0.15))
    prior_w = float(params.get("ONLINE_PRIOR_WEIGHT", 0.25))
    cnn_log = frag_row.get("CNNLogEvidence") or {}
    tag_log = frag_row.get("TagLogEvidence") or {}
    online_lbl = str(frag_row.get("OnlineLabel", "unknown"))
    online_conf = float(frag_row.get("OnlineConfidence", 0.0))
    sources: list[tuple[float, dict[str, float]]] = []
    if cnn_log and cnn_w > 0:
        sources.append((cnn_w, cnn_log))
    if tag_log and tag_w > 0:
        sources.append((tag_w, tag_log))
    if prior_w > 0 and online_lbl in known_labels and np.isfinite(online_conf):
        sources.append((prior_w, _build_prior_log_scores(known_labels, online_lbl, online_conf)))
    if not sources:
        return _normalize_support_scores(known_labels, {})
    total_w = sum(w for w, _ in sources)
    combined = {
        label: sum(w * float(src.get(label, 0.0)) for w, src in sources) / total_w
        for label in known_labels
    }
    return _normalize_support_scores(known_labels, combined)
```
Replace both inline `combined_log = {...}` blocks (in `_iterative_assign` and `solve_global_assignment`) with `_combined_support(frag_row, known_labels, params)`. Add the `FRAGMENT_MIN_SUPPORT` engine param + default.json key + docstring line in `run_fragment_solver`'s params list (`FRAGMENT_MIN_SUPPORT float default 0.5 — a label is a candidate for a fragment only if its normalised support ≥ this; absolute posterior floor`). Task 4 consumes it.

- [ ] **Step 4: Run** — new file + `tests/test_fragment_solver.py tests/identity/`. Expect `test_solve_global_assignment_keeps_online_label_when_margin_too_small` may change meaning; if it fails, read it: if it asserts the *temperature* behaviour, update with the plan comment; if it asserts the margin gate, keep and fix the code.

- [ ] **Step 5: Commit** — `git commit -m "fix(identity): fragment support is a convex blend of informative sources, not a temperature; add FRAGMENT_MIN_SUPPORT"`

---

### Task 4: Mass-first seeding and exact-objective displacement

**Files:**
- Modify: `src/hydra_suite/core/individual/identity/offline.py:408-860` (`_base_assignment_via_substrate` deleted; `_support_to_slot_posterior` deleted if unused; `_iterative_assign` rewritten around a schedule object).
- Modify: `run_fragment_solver` docstring params list (add `FRAGMENT_MAX_BLOCKERS int default 4`; note `ASSIGNMENT_MARGIN_THRESHOLD` is floored at `1e-3`; remove `IDENTITY_DISPLAY_THRESHOLD` from the list).
- Modify: `src/hydra_suite/trackerkit/engine_params.py` — `"FRAGMENT_MAX_BLOCKERS": int(_cfg_get(cfg, "fragment_max_blockers", default=4))`.
- Test: `tests/identity/test_offline_assignment_mass.py`; update `tests/test_fragment_solver.py` where it imports `_base_assignment_via_substrate` (grep; none expected) or asserts substrate-gate behaviour.

**Interfaces:**
- Consumes: `_combined_support` (Task 3), `params["FRAGMENT_MIN_SUPPORT"]`.
- Produces: `_iterative_assign(frags, known_labels, params) -> dict[int, str | None]` (same signature); new module-level `_evidence_mass(duration, top_support) -> float`.

- [ ] **Step 1: Tests**

```python
# tests/identity/test_offline_assignment_mass.py
import math

import numpy as np
import pandas as pd

from hydra_suite.core.individual.identity.offline import _iterative_assign

LABELS = ["a", "b", "c"]


def _frag(tid, s, e, x0, y0, x1, y1, probs, stability=1.0):
    log = {l: math.log(max(probs.get(l, 1e-6), 1e-6)) for l in LABELS}
    return {
        "TrajectoryID": tid, "StartFrame": s, "EndFrame": e,
        "StartX": x0, "StartY": y0, "EndX": x1, "EndY": y1,
        "MeanCNNProbs": probs, "MeanTagProbs": {}, "CNNLogEvidence": log,
        "TagLogEvidence": {}, "Stability": stability,
        "OnlineLabel": "unknown", "OnlineConfidence": 0.0,
    }


PARAMS = {
    "FRAGMENT_CNN_WEIGHT": 0.4, "FRAGMENT_TAG_WEIGHT": 0.0, "ONLINE_PRIOR_WEIGHT": 0.0,
    "FRAGMENT_LENGTH_WEIGHT": 0.6, "MAX_VELOCITY_BREAK": 50.0, "MAX_BRIDGE_GAP_FRAMES": 30,
    "SPATIAL_NO_NEIGHBOR_SCORE": 0.3, "FRAGMENT_SPATIAL_VETO_THRESHOLD": 0.05,
    "ASSIGNMENT_MARGIN_THRESHOLD": 0.0, "FRAGMENT_TOP_K": 3, "FRAGMENT_MAX_PASSES": 10,
    "FRAGMENT_MIN_SUPPORT": 0.5, "FRAGMENT_MAX_BLOCKERS": 4,
}


def test_long_consistent_track_beats_short_fragments_for_its_label():
    long = _frag(0, 0, 700, 0, 0, 0, 700, {"a": 0.999, "b": 0.0005, "c": 0.0005})
    shorts = [
        _frag(i, 50 * i + 10, 50 * i + 20, 500, 50 * i, 500, 50 * i + 10, {"a": 0.6, "b": 0.3, "c": 0.1})
        for i in range(1, 5)
    ]
    frags = pd.DataFrame([long, *shorts])
    out = _iterative_assign(frags, LABELS, PARAMS)
    assert out[0] == "a"
    assert all(out[i] != "a" for i in range(1, 5))


def test_fragment_below_support_floor_stays_unknown():
    frags = pd.DataFrame([_frag(0, 0, 100, 0, 0, 0, 100, {"a": 0.3, "b": 0.3, "c": 0.4})])
    out = _iterative_assign(frags, LABELS, PARAMS)
    assert out[0] is None


def test_two_disjoint_fragments_may_share_a_label():
    f1 = _frag(0, 0, 100, 0, 0, 0, 100, {"a": 0.99, "b": 0.005, "c": 0.005})
    f2 = _frag(1, 110, 200, 0, 105, 0, 200, {"a": 0.99, "b": 0.005, "c": 0.005})
    out = _iterative_assign(pd.DataFrame([f1, f2]), LABELS, PARAMS)
    assert out == {0: "a", 1: "a"}


def test_overlapping_fragments_never_share_a_label():
    f1 = _frag(0, 0, 100, 0, 0, 0, 100, {"a": 0.99, "b": 0.005, "c": 0.005})
    f2 = _frag(1, 50, 150, 300, 0, 300, 150, {"a": 0.99, "b": 0.005, "c": 0.005})
    out = _iterative_assign(pd.DataFrame([f1, f2]), LABELS, PARAMS)
    assert not (out[0] == "a" and out[1] == "a")
    assert out[0] == "a"  # the longer/heavier one wins the tie on mass


def test_displacement_moves_multiple_blockers_when_it_raises_objective():
    long = _frag(0, 0, 700, 0, 0, 0, 700, {"a": 0.999, "b": 0.0005, "c": 0.0005})
    b1 = _frag(1, 100, 110, 400, 100, 400, 110, {"a": 0.55, "b": 0.45, "c": 0.0})
    b2 = _frag(2, 300, 310, 400, 300, 400, 310, {"a": 0.55, "c": 0.45, "b": 0.0})
    frags = pd.DataFrame([b1, b2, long])  # blockers first in index order on purpose
    out = _iterative_assign(frags, LABELS, PARAMS)
    assert out[2] == "a"
    assert out[0] != "a" and out[1] != "a"


def test_terminates_with_zero_margin_threshold():
    rng = np.random.default_rng(0)
    frags = []
    for i in range(40):
        s = int(rng.integers(0, 600)); e = s + int(rng.integers(5, 120))
        p = rng.dirichlet([1, 1, 1]); probs = dict(zip(LABELS, p))
        frags.append(_frag(i, s, e, float(rng.uniform(0, 500)), float(rng.uniform(0, 500)),
                           float(rng.uniform(0, 500)), float(rng.uniform(0, 500)), probs))
    out = _iterative_assign(pd.DataFrame(frags), LABELS, {**PARAMS, "ASSIGNMENT_MARGIN_THRESHOLD": 0.0})
    assert len(out) == 40
```

- [ ] **Step 2: Run → FAIL** (long track unknown / dict shape).

- [ ] **Step 3: Implement** — rewrite `_iterative_assign`:

```python
def _evidence_mass(duration: float, top_support: float) -> float:
    return float(duration) * float(top_support)


def _iterative_assign(frags, known_labels, params):
    """Mass-first seeding + doubt-ordered refinement with exact-objective
    multi-blocker displacement. Returns {frag_index: label_or_None}."""
    length_w = min(1.0, max(0.0, float(params.get("FRAGMENT_LENGTH_WEIGHT", 0.60))))
    max_vel = float(params.get("MAX_VELOCITY_BREAK", 50.0))
    max_bridge_gap = max(1, int(params.get("MAX_BRIDGE_GAP_FRAMES", 30)))
    no_neighbor_score = float(params.get("SPATIAL_NO_NEIGHBOR_SCORE", 0.3))
    spatial_veto = float(params.get("FRAGMENT_SPATIAL_VETO_THRESHOLD", 0.05))
    monotone_eps = max(1e-3, float(params.get("ASSIGNMENT_MARGIN_THRESHOLD", 0.10)))
    top_k = max(1, int(params.get("FRAGMENT_TOP_K", 3)))
    max_passes = max(1, int(params.get("FRAGMENT_MAX_PASSES", 10)))
    min_support = float(params.get("FRAGMENT_MIN_SUPPORT", 0.5))
    max_blockers = max(1, int(params.get("FRAGMENT_MAX_BLOCKERS", 4)))
    unknown_doubt_bonus = float(params.get("FRAGMENT_UNKNOWN_DOUBT_BONUS", 0.5))

    n = len(frags)
    if n == 0:
        return {}
    rows = [frags.iloc[i] for i in range(n)]
    durations = np.array([max(1, int(r["EndFrame"]) - int(r["StartFrame"]) + 1) for r in rows], float)
    log_max = math.log1p(durations.max())
    length_scales = np.log1p(durations) / log_max if log_max > 1e-9 else np.ones(n)
    length_factors = 1.0 - length_w * (1.0 - length_scales)
    supports = [_combined_support(r, known_labels, params) for r in rows]
    stabilities = np.array([float(r.get("Stability", 0.0)) for r in rows], float)
    segs = [_seg_from_row(r) for r in rows]
    candidates_of = [
        [l for l in sorted(known_labels, key=lambda l: -supports[i].get(l, 0.0))[:top_k]
         if supports[i].get(l, 0.0) >= min_support]
        for i in range(n)
    ]
    current: list[str | None] = [None] * n
    schedule: dict[str, list[int]] = {l: [] for l in known_labels}  # label -> frag indices

    def _overlaps(i, j):
        return int(rows[i]["StartFrame"]) <= int(rows[j]["EndFrame"]) and int(rows[j]["StartFrame"]) <= int(rows[i]["EndFrame"])

    def _blockers(i, label):
        return [j for j in schedule[label] if j != i and _overlaps(i, j)]

    def _score(i, label):
        """Score of fragment i under `label` given the CURRENT schedule (i excluded).
        Returns 0.0 when vetoed (collision or spatial veto)."""
        if label is None:
            return 0.0
        if _blockers(i, label):
            return 0.0
        sched = {label: [segs[j] for j in schedule[label] if j != i]}
        spatial_s, has_nb = _spatial_score_for_fragment(rows[i], label, sched, max_vel, no_neighbor_score, max_bridge_gap)
        if has_nb and spatial_s < spatial_veto:
            return 0.0
        ev = float(supports[i].get(label, 0.0))
        raw = ev * spatial_s if has_nb else ev
        return float(raw * length_factors[i])

    def _commit(i, label):
        cur = current[i]
        if cur is not None:
            schedule[cur].remove(i)
        if label is not None:
            schedule[label].append(i)
        current[i] = label

    def _objective(touched: set[int]) -> float:
        return sum(_score(j, current[j]) for j in touched)

    def _affected(label_a, label_b, extra):
        s = set(extra)
        for l in (label_a, label_b):
            if l is not None:
                s.update(schedule[l])
        return s

    def _best_alternative(j, exclude):
        best_s, best_l = 0.0, None
        for c in candidates_of[j]:
            if c == exclude:
                continue
            s = _score(j, c)
            if s > best_s:
                best_s, best_l = s, c
        return best_s, best_l

    def _try_displacement(i, c) -> bool:
        """Tentatively give `c` to i, re-home its blockers, accept iff the exact
        objective over every affected fragment rises by >= monotone_eps."""
        blockers = _blockers(i, c)
        if not blockers or len(blockers) > max_blockers:
            return False
        before_assign = {j: current[j] for j in blockers}
        before_assign[i] = current[i]
        touched = _affected(current[i], c, blockers)
        for j in blockers:
            touched.update(schedule[current[j]] if current[j] else [])
        j_before = _objective(touched)
        for j in blockers:
            _commit(j, None)
        _commit(i, c)
        new_labels = {}
        for j in blockers:
            _, alt = _best_alternative(j, c)
            new_labels[j] = alt
            _commit(j, alt)
            if alt is not None:
                touched.update(schedule[alt])
        j_after = _objective(touched)
        if j_after - j_before >= monotone_eps and _score(i, c) > 0.0:
            return True
        for j, lbl in new_labels.items():
            _commit(j, None)
        for j, lbl in before_assign.items():
            _commit(j, lbl)
        return False

    # --- 1. mass-first seeding ---
    order = sorted(range(n), key=lambda i: -_evidence_mass(durations[i], max(supports[i].values()) if supports[i] else 0.0))
    for i in order:
        for c in candidates_of[i]:
            if _score(i, c) > 0.0:
                _commit(i, c)
                break

    # --- 2. doubt-ordered refinement ---
    def _doubt(i):
        s_norm = 1.0 - stabilities[i]; l_norm = 1.0 - length_scales[i]
        if current[i] is None:
            return s_norm * l_norm + unknown_doubt_bonus
        sched = {current[i]: [segs[j] for j in schedule[current[i]] if j != i]}
        sp, has_nb = _spatial_score_for_fragment(rows[i], current[i], sched, max_vel, no_neighbor_score, max_bridge_gap)
        return s_norm * l_norm * (1.0 - (sp if has_nb else no_neighbor_score))

    for pass_idx in range(max_passes):
        flips = 0
        for i in sorted(range(n), key=lambda i: -_doubt(i)):
            cur = current[i]
            cur_s = _score(i, cur)
            best_l, best_s = cur, cur_s
            for c in candidates_of[i]:
                if c == cur:
                    continue
                s = _score(i, c)
                if s > 0.0 and s - cur_s >= monotone_eps and s > best_s:
                    best_l, best_s = c, s
            if best_l != cur:
                _commit(i, best_l); flips += 1; continue
            for c in candidates_of[i]:
                if c != cur and _score(i, c) == 0.0 and _try_displacement(i, c):
                    flips += 1; break
        log.debug("iterative fragment solver pass %d: %d flips", pass_idx + 1, flips)
        if flips == 0:
            break
    else:
        log.warning("Iterative fragment solver hit FRAGMENT_MAX_PASSES (%d) without convergence.", max_passes)

    # --- 3. unknown rescue by descending mass (may displace) ---
    for i in sorted((i for i in range(n) if current[i] is None),
                    key=lambda i: -_evidence_mass(durations[i], max(supports[i].values()) if supports[i] else 0.0)):
        placed = False
        for c in candidates_of[i]:
            if _score(i, c) > 0.0:
                _commit(i, c); placed = True; break
        if not placed:
            for c in candidates_of[i]:
                if _try_displacement(i, c):
                    break
    return {i: current[i] for i in range(n)}
```
Delete `_base_assignment_via_substrate` and `_support_to_slot_posterior` (grep for other users first; `tests/test_fragment_solver.py` imports neither). `solve_global_assignment` keeps computing `assigned_scores` but with `_combined_support`. Remove `display_threshold` from `_iterative_assign`. Every accepted move raises the exact objective over the affected set by ≥ `monotone_eps > 0`, and the objective is bounded by `n`, so the passes terminate; document this in the docstring.

- [ ] **Step 4: Run** — new file + `tests/test_fragment_solver.py tests/identity/ tests/test_core_identity_postprocess_df.py`. Update `test_fragment_solver.py` tests that encoded the old solver (e.g. any that expect a label assigned from support < 0.5) with the plan comment and a one-line reason each; report the list.

- [ ] **Step 5: Commit** — `git commit -m "fix(identity): mass-first seeding, support floor and exact-objective multi-blocker displacement in the fragment solver"`

---

### Task 5: Dense, position-complete trajectories

**Files:**
- Modify: `src/hydra_suite/core/post/processing.py` — add `densify_trajectory_frames(df) -> pd.DataFrame`, `trim_positionless_ends(df) -> pd.DataFrame`, `final_interpolation_max_gap(config, params) -> int`; change `interpolate_trajectories` to fill interior gaps beyond `max_gap` when `fill_all_interior=True` (new kwarg, default False) and log gap sizes > `max_gap`.
- Modify: `src/hydra_suite/core/tracking/session.py:229-250` (`_interpolate_and_scale`) and `:252-285` (`_merge`, `max_gap=` argument) to use `final_interpolation_max_gap` and `fill_all_interior=True`; `merge.py:164` passes the kwarg through (add `fill_all_interior: bool = False` param to `merge_trajectories`).
- Test: `tests/test_post_dense_trajectories.py`

**Interfaces:**
- Produces: `densify_trajectory_frames(df)` — per TrajectoryID reindex to `[min,max]` FrameID, new rows `State="occluded"`, `DetectionID` NaN, `DetectionConfidence=0.0`, other columns NaN, `arena_id` copied; `trim_positionless_ends(df)` — drops leading/trailing rows with NaN X or Y per trajectory (logs counts); `final_interpolation_max_gap(config, params) = max(round(interpolation_max_gap_seconds*FPS), MAX_OCCLUSION_GAP + 1)`; `interpolate_trajectories(..., fill_all_interior=False)`.

- [ ] **Step 1: Tests**

```python
# tests/test_post_dense_trajectories.py
import numpy as np
import pandas as pd

from hydra_suite.core.post.processing import (
    densify_trajectory_frames,
    final_interpolation_max_gap,
    interpolate_trajectories,
    trim_positionless_ends,
)


def _traj(frames, xs):
    return pd.DataFrame({"TrajectoryID": 0, "FrameID": frames, "X": xs, "Y": xs, "Theta": 0.0,
                         "State": ["active" if not np.isnan(x) else "occluded" for x in xs],
                         "DetectionID": [i if not np.isnan(x) else np.nan for i, x in enumerate(xs)],
                         "DetectionConfidence": [0.9 if not np.isnan(x) else np.nan for x in xs]})


def test_densify_inserts_missing_frames_as_occluded():
    df = _traj([1, 2, 5, 6], [1.0, 2.0, 5.0, 6.0])
    out = densify_trajectory_frames(df)
    assert out["FrameID"].tolist() == [1, 2, 3, 4, 5, 6]
    assert out.loc[out.FrameID == 3, "State"].iloc[0] == "occluded"
    assert np.isnan(out.loc[out.FrameID == 3, "DetectionID"].iloc[0])
    assert out.loc[out.FrameID == 3, "DetectionConfidence"].iloc[0] == 0.0


def test_interpolate_fills_gaps_longer_than_max_gap_when_fill_all_interior():
    frames = list(range(1, 12)); xs = [1.0] + [np.nan] * 9 + [11.0]
    df = _traj(frames, xs)
    capped = interpolate_trajectories(df, method="linear", max_gap=5)
    assert capped["X"].isna().sum() == 9
    full = interpolate_trajectories(df, method="linear", max_gap=5, fill_all_interior=True)
    assert full["X"].isna().sum() == 0
    assert abs(full.loc[full.FrameID == 6, "X"].iloc[0] - 6.0) < 1e-9


def test_trim_drops_leading_and_trailing_positionless_rows_only():
    df = _traj([1, 2, 3, 4, 5], [np.nan, np.nan, 3.0, np.nan, 5.0])
    out = trim_positionless_ends(df)
    assert out["FrameID"].tolist() == [3, 4, 5]


def test_final_interpolation_max_gap_never_below_user_knob():
    assert final_interpolation_max_gap({"interpolation_max_gap_seconds": 0.5}, {"FPS": 10, "MAX_OCCLUSION_GAP": 10}) == 11
    assert final_interpolation_max_gap({"interpolation_max_gap_seconds": 5.0}, {"FPS": 10, "MAX_OCCLUSION_GAP": 10}) == 50
```

- [ ] **Step 2: Run → FAIL.**

- [ ] **Step 3: Implement**

```python
def densify_trajectory_frames(df: pd.DataFrame) -> pd.DataFrame:
    """Reindex every trajectory to a dense FrameID range. Inserted rows are
    ``State="occluded"``, ``DetectionID`` NaN, ``DetectionConfidence`` 0.0."""
    if df is None or df.empty or "TrajectoryID" not in df.columns:
        return df
    parts = []
    inserted = 0
    for tid, g in df.groupby("TrajectoryID", sort=False):
        g = g.sort_values("FrameID").drop_duplicates("FrameID", keep="first")
        lo, hi = int(g["FrameID"].min()), int(g["FrameID"].max())
        if len(g) == hi - lo + 1:
            parts.append(g); continue
        full = g.set_index("FrameID").reindex(np.arange(lo, hi + 1)).reset_index()
        new = full["TrajectoryID"].isna()
        inserted += int(new.sum())
        full.loc[new, "TrajectoryID"] = tid
        if "State" in full.columns:
            full.loc[new, "State"] = "occluded"
        if "DetectionConfidence" in full.columns:
            full.loc[new, "DetectionConfidence"] = 0.0
        if "arena_id" in full.columns:
            vals = g["arena_id"].dropna().unique()
            if len(vals) == 1:
                full.loc[new, "arena_id"] = vals[0]
        full["TrajectoryID"] = full["TrajectoryID"].astype(g["TrajectoryID"].dtype)
        parts.append(full)
    if inserted:
        logger.info("densify_trajectory_frames: inserted %d occluded rows into frame gaps", inserted)
    return pd.concat(parts, ignore_index=True).sort_values(["TrajectoryID", "FrameID"], kind="stable").reset_index(drop=True)


def trim_positionless_ends(df: pd.DataFrame) -> pd.DataFrame:
    """Drop leading/trailing rows without a position (NaN X or Y) per trajectory:
    they carry no detection and cannot be interpolated."""
    if df is None or df.empty or "TrajectoryID" not in df.columns:
        return df
    keep = pd.Series(True, index=df.index)
    dropped = 0
    for _tid, g in df.groupby("TrajectoryID", sort=False):
        g = g.sort_values("FrameID")
        has = g["X"].notna() & g["Y"].notna()
        if not has.any():
            keep.loc[g.index] = False; dropped += len(g); continue
        first, last = has.idxmax(), has[::-1].idxmax()
        pos = g.index.get_indexer([first, last])
        mask = np.zeros(len(g), bool); mask[pos[0]: pos[1] + 1] = True
        keep.loc[g.index[~mask]] = False; dropped += int((~mask).sum())
    if dropped:
        logger.info("trim_positionless_ends: dropped %d leading/trailing position-less rows", dropped)
    return df.loc[keep].reset_index(drop=True)


def final_interpolation_max_gap(config, params) -> int:
    user_gap = max(1, round(float(config.get("interpolation_max_gap_seconds", 0.0)) * float(params["FPS"])))
    return int(max(user_gap, int(params.get("MAX_OCCLUSION_GAP", 30)) + 1))
```
`interpolate_trajectories`: add `fill_all_interior: bool = False`; pass to `_interpolate_column`/`_interpolate_angle` an `effective_max_gap = 10**9 if fill_all_interior else max_gap`, and when `fill_all_interior`, log at INFO each interior gap > `max_gap` (`"Trajectory %s: filled an interior gap of %d frames (> max_gap %d)"`). Then, in `interpolate_trajectories`, apply `trim_positionless_ends` at the end **only when** `fill_all_interior` (the pre-merge callers keep today's output byte-identical).

`session._interpolate_and_scale` and `_merge`: compute `max_gap = final_interpolation_max_gap(self.config, self.params)` when `interpolation_method != "none"`, pass `fill_all_interior=True`; `merge_trajectories` gains `fill_all_interior: bool = False` forwarded to `interpolate_trajectories`. When `interpolation_method == "none"` the final frame still gets `trim_positionless_ends` (positions are not fabricated, but position-less ends are dropped) — apply it in `_interpolate_and_scale`/`merge.py` right after the interpolation branch.

- [ ] **Step 4: Run** — new file + `tests/test_post_*.py tests/test_core_*.py tests/test_session_media_export_paths.py` and any test importing `interpolate_trajectories` (grep).

- [ ] **Step 5: Commit** — `git commit -m "fix(post): dense, position-complete final trajectories (fill all interior gaps, trim position-less ends)"`

---

### Task 6: Relink first, resolve identity once, invariant on the written frame

**Files:**
- Modify: `src/hydra_suite/core/individual/postprocess_df.py:210-700` — split `apply_identity_postprocessing_to_df` into `derive_identity_keys(df, params)` (the `_annotate_identity_summary_columns` + heads/`UniqueIdentityKey` tail) and `resolve_identity(df, params, identity_evidence_cache_path=None)` (solver → `_stamp_non_identifying_labels` → `_mirror_realtime_and_tag_into_final` → `fill_identity_nans_with_consensus` → `sort_trajectories_by_identity` → `derive_identity_keys`). Keep `apply_identity_postprocessing_to_df(df, params, identity_evidence_cache_path=None) = resolve_identity(derive_identity_keys(df, params), ...)` as the backwards-compatible composition (tests use it).
- Modify: `src/hydra_suite/core/post/identity_postprocess.py` — add `assert_one_identity_per_trajectory(df) -> list` and `collapse_to_majority_identity(df, offenders) -> pd.DataFrame`.
- Modify: `src/hydra_suite/core/post/rich_export.py:212-450` — `build_rich_export_dataframe(..., resolve: bool = True)`; `export_rich_csv(..., resolve: bool = True)`; `relink_and_export_rich_csv` = build(`resolve=False`) → `relink_trajectories_with_pose_by_arena` → `densify_trajectory_frames` → `interpolate_trajectories(fill_all_interior=True, max_gap=final gap)` (import from processing; `max_gap` from `params["FINAL_INTERPOLATION_MAX_GAP"]`, see below) → `resolve_identity` → invariant check/collapse → write both CSVs.
- Modify: `src/hydra_suite/core/tracking/session.py:321-346, 632-700` — `_export_rich(final_csv, resolve)`; `run_post_tracking` computes `postpass = should_run_interpolated_postpass(self.config) and not cb.should_stop()` once, calls `_export_rich(final_csv, resolve=not postpass)`; set `self.params["FINAL_INTERPOLATION_MAX_GAP"] = final_interpolation_max_gap(config, params)` in `__init__` (and `cli`/GUI param builders need nothing — it is derived).
- Test: `tests/test_rich_export_relink_then_resolve.py`, `tests/test_identity_invariant.py`

**Interfaces:**
- Consumes: Task 5's `densify_trajectory_frames`, `interpolate_trajectories(fill_all_interior=True)`, `final_interpolation_max_gap`.
- Produces: `postprocess_df.derive_identity_keys`, `postprocess_df.resolve_identity`; `identity_postprocess.assert_one_identity_per_trajectory(df) -> list[int]`; `identity_postprocess.collapse_to_majority_identity(df, offenders)`.

- [ ] **Step 1: Tests**

```python
# tests/test_identity_invariant.py
import pandas as pd

from hydra_suite.core.individual.identity import columns as C
from hydra_suite.core.post.identity_postprocess import (
    assert_one_identity_per_trajectory,
    collapse_to_majority_identity,
)


def _df():
    return pd.DataFrame(
        {
            "TrajectoryID": [0, 0, 0, 1, 1],
            "FrameID": [1, 2, 3, 1, 2],
            C.FINAL_LABEL: ["a", "a", "b", "c", "c"],
            C.FINAL_ID: [1, 1, 2, 3, 3],
            C.FINAL_SOURCE: ["offline", "offline", "offline", "offline", "offline"],
            C.FINAL_CONFIDENCE: [0.9, 0.9, 0.2, 0.8, 0.8],
        }
    )


def test_offenders_are_reported():
    assert assert_one_identity_per_trajectory(_df()) == [0]


def test_collapse_uses_majority_and_min_confidence():
    out = collapse_to_majority_identity(_df(), [0])
    t0 = out[out.TrajectoryID == 0]
    assert t0[C.FINAL_LABEL].unique().tolist() == ["a"]
    assert t0[C.FINAL_ID].unique().tolist() == [1]
    assert (t0[C.FINAL_CONFIDENCE] == 0.2).all()
    assert assert_one_identity_per_trajectory(out) == []
```

```python
# tests/test_rich_export_relink_then_resolve.py
"""relink_and_export_rich_csv must relink BEFORE resolving identity, resolve
exactly once, densify chains, and write a frame with one identity per track."""
import pandas as pd
import pytest

from hydra_suite.core.individual.identity import columns as C
from hydra_suite.core.post import rich_export


def test_relink_then_resolve_order_and_single_solve(tmp_path, monkeypatch):
    final = tmp_path / "clip_final.csv"
    base = pd.DataFrame(
        {
            "TrajectoryID": [0, 0, 1, 1],
            "FrameID": [1, 2, 5, 6],
            "X": [0.0, 1.0, 4.0, 5.0], "Y": [0.0, 0.0, 0.0, 0.0], "Theta": 0.0,
            "State": "active", "DetectionID": [1, 2, 3, 4],
        }
    )
    base.to_csv(final, index=False)
    calls = []

    def fake_build(final_csv_path, state, *, params, min_valid_conf, ignore_keypoints,
                   identity_evidence_cache_path=None, resolve=True):
        calls.append(("build", resolve))
        return pd.read_csv(final_csv_path)

    def fake_relink(df, params):
        calls.append(("relink", df["TrajectoryID"].nunique()))
        out = df.copy(); out["TrajectoryID"] = 0; return out

    def fake_resolve(df, params, identity_evidence_cache_path=None):
        calls.append(("resolve", df["FrameID"].tolist()))
        out = df.copy(); out[C.FINAL_LABEL] = "a"; out[C.FINAL_ID] = 1
        out[C.FINAL_SOURCE] = "offline"; out[C.FINAL_CONFIDENCE] = 0.9; return out

    monkeypatch.setattr(rich_export, "build_rich_export_dataframe", fake_build)
    import hydra_suite.core.post.processing as P
    monkeypatch.setattr(P, "relink_trajectories_with_pose_by_arena", fake_relink)
    import hydra_suite.core.individual.postprocess_df as PD
    monkeypatch.setattr(PD, "resolve_identity", fake_resolve)

    params = {"FINAL_INTERPOLATION_MAX_GAP": 11, "ENABLE_TRACKLET_RELINKING": True}
    out = rich_export.relink_and_export_rich_csv(str(final), state=None, params=params,
                                                 min_valid_conf=0.2, ignore_keypoints=None,
                                                 debug_mode=True, fps=10.0)
    assert calls[0] == ("build", False)
    assert calls[1][0] == "relink"
    assert calls[2][0] == "resolve" and calls[2][1] == [1, 2, 3, 4, 5, 6]  # densified before resolve
    assert sum(1 for c in calls if c[0] == "resolve") == 1
    written = pd.read_csv(out)
    assert written["FrameID"].tolist() == [1, 2, 3, 4, 5, 6]
    assert written["X"].isna().sum() == 0
    assert written[C.FINAL_LABEL].nunique() == 1
```
(The monkeypatch targets must match how `relink_and_export_rich_csv` imports them: import `relink_trajectories_with_pose_by_arena` and `resolve_identity` **inside** the function from their modules, as today's code does for relink, so patching the module attribute works.)

- [ ] **Step 2: Run → FAIL.**

- [ ] **Step 3: Implement** — per the Files list. `assert_one_identity_per_trajectory`: for each TrajectoryID, offenders where `nunique(dropna=False) > 1` on any of `FINAL_LABEL`, `FINAL_ID`, `FINAL_SOURCE` (missing columns → `[]`). `collapse_to_majority_identity`: majority label by row count (ties → first by FrameID order), its id/source from the first row carrying that label, confidence = min over the trajectory. In `relink_and_export_rich_csv` after `resolve_identity`: `offenders = assert_one_identity_per_trajectory(df); if offenders: logger.error("relink/resolve produced %d trajectories with more than one identity: %s -- collapsing to majority", len(offenders), offenders[:20]); df = collapse_to_majority_identity(df, offenders)`. Same check in `export_rich_csv` when `resolve`. Keep writing `relinked_base` (common base columns) to `final_csv_path` **from the resolved, densified frame** so `_final.csv` and `_with_individual.csv` agree on rows and ids.

- [ ] **Step 4: Run** — both new files + `tests/test_rich_export*.py tests/test_core_rich_export.py tests/test_core_identity_postprocess_df.py tests/identity/ tests/test_post_tracklet_relinking.py tests/test_session_media_export_paths.py tests/test_fragment_solver.py`.

- [ ] **Step 5: Commit** — `git commit -m "fix(post): relink before identity resolution, resolve once, enforce one identity per trajectory on the written frame"`

---

### Task 7: Rendered media labels by final identity

**Files:**
- Modify: `src/hydra_suite/core/post/media_export.py:136-186` — `identity_columns = [C.FINAL_LABEL, C.FINAL_SMOOTHED_LABEL, C.UNIQUE_IDENTITY_KEY]` in both builders; docstrings say why (audit S8).
- Modify: `tests/test_media_export.py:75-97` — rename to `test_overlay_priority_is_final_then_smoothed_then_unique_key`, assert the new order (row with `UniqueIdentityKey="apriltag=3"` and `IdentityFinalLabel="ant_a"` renders `ant_a`).
- Test: same file.

- [ ] **Step 1: Update/write test** (above; keep `test_color_key_array_prefers_identity_then_trajectory` as is — it has no Final column).
- [ ] **Step 2: Run → FAIL.**
- [ ] **Step 3: Implement** the reorder.
- [ ] **Step 4: Run** `tests/test_media_export.py`.
- [ ] **Step 5: Commit** — `git commit -m "fix(media): render and colour tracks by IdentityFinalLabel, not raw per-frame evidence"`

---

### Task 8: Equivalence fixture, gates, DEMO/ID acceptance

**Files:**
- Create: `tools/equivalence/fixtures/configs/ant_cnn_identity_relink.json` (copy of `ant_cnn_identity.json`; set `enable_tracklet_relinking: true`); register it wherever `run_matrix.sh`/`runner.py`/`manifest.json` enumerate clips (grep `ant_cnn_identity_marked` for the pattern — it reuses the `ant_cnn_identity` clip).
- Create: `tools/equivalence/notes/2026-08-27-identity-final-consistency-gate.md`
- Create: `scripts/audit_final_csv.py` — the acceptance checker: prints NaN X/Y/Theta count, leading/trailing occluded rows, missing interior frames, empty `IdentityFinalSource`, NaN `IdentityFinalConflictResolved`, trajectories with >1 label, labelled tracks whose label ≠ argmax of their mean smoothed posterior, unknown/labelled track counts, and for `--tracks 110 111 115` their labels. Exit code 1 if any of the first six is non-zero.

- [ ] **Step 1:** Write `scripts/audit_final_csv.py` (argparse: `csv`, `--tracks`), run it on the *shipped* `DEMO/ID/ONLINE/ant_tracking_final_with_individual.csv` and paste its output into the gate note as the "before" block (expected: 222 / 7 / 36 / 5683 / 12113 / 8).
- [ ] **Step 2:** Rerun post-processing for DEMO/ID/ONLINE with the worktree code, reusing caches (same recipe as the previous gate: `trackerkit track` CLI headless with `--config DEMO/ID/ONLINE/ant_config.json`, output to `/tmp/idfinal_online/` — copy the cache dir `.inference_cache_ant` and `ant_tracking_forward.csv`/`_backward.csv` there first; if the CLI insists on re-tracking, run it: it is ~90 s on MPS). Run the audit script on the result; paste as the "after" block. Acceptance = §4 of the spec. Record t111/t110/t115 labels, the labelled/unknown split, and the label-vs-evidence agreement.
- [ ] **Step 3:** MPS equivalence matrix vs `main` (`MAIN_SRC=<repo>/src` on `main` @ `f2d4ca36`, `WT_SRC=<worktree>/src`), all clips incl. the new fixture. Kill stale sleap/hydra processes first. Record per-clip: positions p99 on detection rows, row-count delta, identity-column divergence, and attribute each delta to §3.4 (interpolated/trimmed rows) or §3.1/3.3 (identity). `fly_obb`/`worm_bgsub` must be byte-identical.
- [ ] **Step 4:** Commit the fixture config + script + gate note: `git commit -m "test(equivalence): relink-enabled identity fixture; identity-final-consistency gate record"`.

---

### Task 9: CUDA gate on mehek + docs

**Files:**
- Modify: `tools/equivalence/notes/2026-08-27-identity-final-consistency-gate.md` (CUDA table)
- Modify: `docs/user-guide/identity*.md` (grep for the page describing `IdentityFinalSource`/the video overlay): document `none`, the smoothed columns as an ungated record, dense trajectories, and that the rendered video labels by `IdentityFinalLabel`.

- [ ] **Step 1:** Ship the worktree HEAD to mehek (git bundle recipe in memory `project_pose_cnn_batched_detection_slowdown`), run the matrix with `RUNTIME=cuda`, fold the table into the gate note (same attribution rules as Task 8 step 3).
- [ ] **Step 2:** Docs edits; `make docs-check`.
- [ ] **Step 3:** Commit — `git commit -m "docs(identity): explicit final-output vocabulary; CUDA gate results"`.
