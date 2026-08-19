# Identity Heads, Cross-Product Catalogs, and Non-Identifying Classes — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make non-identity classifiers stop polluting identity, make multiple identity classifiers compose as a cross-product, and let users declare classes non-identifying so untagged animals read as `notag_notag` without being forced into one exclusive slot.

**Architecture:** Three sequential slices. Slice 1 scopes every identity derivation to *identity heads* (`unique_identifier=True` classifiers). Slice 2 rebuilds the catalog as a cross-product over identity factor axes and generalizes the phase→global evidence remap so evidence survives the composite domain. Slice 3 excludes declared non-identifying classes from the catalog entirely — so no exclusivity mechanism can ever apply to them — and stamps a descriptive final label with `IdentityFinalID` pinned to the unknown slot.

**Tech Stack:** Python 3.11, NumPy, pandas, SciPy (Hungarian), PyQt (GUI only), pytest.

**Spec:** `docs/superpowers/specs/2026-08-18-identity-heads-and-non-identifying-classes-design.md`

## Global Constraints

- **Worktree isolation:** all work happens in a git worktree branched from local HEAD: `git worktree add .worktrees/identity-heads -b feat/identity-heads HEAD`. Never a fresh-from-origin worktree — local `main` is ahead of `origin/main`.
- **Core purity:** `core/individual/identity/*` and `core/post/*` must not import Qt or any app-layer package (trackerkit/posekit/classkit/refinekit/detectkit/filterkit/integrations). New Core modules: stdlib + numpy/pandas only.
- **Commit identity:** commit as the configured git user. Do **not** add a `Co-Authored-By: Claude` trailer.
- **Test command:** `python -m pytest tests/<file>.py -v`. Never run the whole `tests/` directory — `classkit` modal-dialog tests hang and SIGABRT. Batch per-file.
- **Worktree tests need** `PYTHONPATH=<worktree>/src` or they silently import the main editable `src`.
- **Formatting gate before every commit:** `make commit-prep` (black + isort), then `make lint-moderate`.
- **Byte-identity claim:** slices 1, 2, and 3-with-feature-off must be byte-identical on the equivalence matrix. Fixtures carry at most one identity classifier, so any diff means a bug.
- **Equivalence gate** (run after Slice 1, Slice 2, and Slice 3 — same baseline each time so each slice's effect is attributable):
  ```bash
  conda activate hydra-mps
  git worktree add --detach .worktrees/equiv-legacy legacy/main
  REPO=$PWD WT=$PWD MAIN_SRC=$PWD/.worktrees/equiv-legacy/src WT_SRC=$PWD/src \
    OUT=/tmp/equiv_identity RUNTIME=mps bash tools/equivalence/run_matrix.sh
  ```
  conda MUST be active or pose/SLEAP clips produce empty CSVs that falsely compare EQUIVALENT — verify `wc -l` > 1 on the CSVs. CUDA leg on `rutalab@mehek.taild08eb9.ts.net` with `hydra-cuda`.
- **Before any heavy run:** kill stale sleap/hydra processes. Never touch non-sleap/hydra processes.

---

## File Structure

**Created:**
- `src/hydra_suite/core/individual/identity/heads.py` — identity-head resolution and column scoping. Pure, stdlib-only.
- `tests/identity/test_identity_heads.py` — head/column scoping unit tests.
- `tests/identity/test_catalog_cross_product.py` — cross-product catalog resolution tests.
- `tests/identity/test_phase_catalog_remap.py` — phase→global remap tests (single-model equality + multi-model non-degeneracy).
- `tests/identity/test_non_identifying_classes.py` — exclusion, coexistence, and final-label stamping tests.

**Modified:**
- `src/hydra_suite/core/individual/identity/resolve.py` — owner of the identity domain: gains `identity_axes()`, `non_identifying_labels()`, cross-product assembly, exclusion filtering, size warning.
- `src/hydra_suite/core/individual/postprocess_df.py:67` — evidence summary columns scoped to identity heads; new non-identifying final-label stamp.
- `src/hydra_suite/core/post/identity_postprocess.py:182` — `derive_unique_identity_key_series` gains `identity_heads` / `non_identifying` filtering.
- `src/hydra_suite/core/tracking/worker.py:1927` — `_remap_source_log_probs_to_catalog` generalized to distribute over composites.
- `src/hydra_suite/trackerkit/gui/panels/identity_panel.py:995` — row config gains `factor_names` + `non_identifying_classes`; new dialog launcher.
- `src/hydra_suite/trackerkit/gui/dialogs/non_identifying_classes_dialog.py` — new `BaseDialog` subclass for marking classes.
- `src/hydra_suite/trackerkit/config/identity_schema.py:75` — `IdentityModelConfig.non_identifying_classes`.

**Not modified (deliberately):** `substrate.py`, `catalog.py`, `online.py`, `offline.py`. Slice 3 works by shrinking the domain, not by teaching the solvers about capacity. If a task tempts you to edit these, stop — the design is wrong or you are.

---

# SLICE 1 — Identity-head scoping

### Task 1: Identity-head resolution and column scoping

**Files:**
- Create: `src/hydra_suite/core/individual/identity/heads.py`
- Test: `tests/identity/test_identity_heads.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `identity_head_labels(cnn_classifiers: Sequence[Mapping]) -> tuple[str, ...]`
  - `identity_class_columns(columns: Iterable, head_labels: Sequence[str]) -> list[str]`
  - `HEADS_UNKNOWN: object` — sentinel meaning "config absent, use legacy all-columns behavior"
  - `resolve_identity_heads(params: Mapping) -> tuple[str, ...] | object` — returns `HEADS_UNKNOWN` when `params` has no `CNN_CLASSIFIERS` key at all.

- [ ] **Step 1: Write the failing test**

```python
# tests/identity/test_identity_heads.py
from hydra_suite.core.individual.identity.heads import (
    HEADS_UNKNOWN,
    identity_class_columns,
    identity_head_labels,
    resolve_identity_heads,
)


def test_only_unique_identifier_entries_are_identity_heads():
    cfgs = [
        {"label": "colortag", "unique_identifier": True},
        {"label": "behavior", "unique_identifier": False},
        {"label": "caste"},  # missing key == not an identity head
    ]
    assert identity_head_labels(cfgs) == ("colortag",)


def test_identity_class_columns_matches_flat_and_multifactor():
    columns = [
        "CNN_colortag_Class",
        "CNN_colortag_thorax_Class",
        "CNN_colortag_thorax_Conf",
        "CNN_behavior_Class",
        "TrajectoryID",
    ]
    got = identity_class_columns(columns, ("colortag",))
    assert got == ["CNN_colortag_Class", "CNN_colortag_thorax_Class"]


def test_identity_class_columns_handles_underscore_in_head_label():
    # "^CNN_(.+)_Class$" cannot tell "colour_tag" (flat) from "colour"+"tag"
    # (factor). Matching against known head labels can.
    columns = ["CNN_colour_tag_Class", "CNN_colour_tag_left_Class"]
    got = identity_class_columns(columns, ("colour_tag",))
    assert got == ["CNN_colour_tag_Class", "CNN_colour_tag_left_Class"]


def test_no_identity_heads_yields_no_columns():
    cfgs = [{"label": "behavior", "unique_identifier": False}]
    assert identity_head_labels(cfgs) == ()
    assert identity_class_columns(["CNN_behavior_Class"], ()) == []


def test_resolve_identity_heads_distinguishes_absent_from_empty():
    # Absent key -> legacy fallback sentinel; present-but-none -> empty tuple.
    assert resolve_identity_heads({}) is HEADS_UNKNOWN
    assert resolve_identity_heads({"CNN_CLASSIFIERS": []}) == ()
    assert (
        resolve_identity_heads(
            {"CNN_CLASSIFIERS": [{"label": "x", "unique_identifier": True}]}
        )
        == ("x",)
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/identity/test_identity_heads.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'hydra_suite.core.individual.identity.heads'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/hydra_suite/core/individual/identity/heads.py
"""Identity-head resolution: which CNN classifiers may influence identity.

A *identity head* is a ``CNN_CLASSIFIERS`` entry with
``unique_identifier=True``. Only identity heads may feed the identity
catalog, the identity evidence summary columns, or ``UniqueIdentityKey``.
Classifiers that are not identity heads (behavior, sex, caste) keep their
own ``CNN_<label>_*`` output columns and influence nothing about identity.

This module is Core: stdlib only, no numpy/pandas/Qt, no app-layer imports.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence


class _HeadsUnknown:
    """Sentinel: no classifier config available, use legacy all-columns behavior."""

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return "HEADS_UNKNOWN"


HEADS_UNKNOWN = _HeadsUnknown()


def identity_head_labels(cnn_classifiers: Sequence[Mapping]) -> tuple[str, ...]:
    """Labels of the classifiers that are identity providers, in config order."""
    labels: list[str] = []
    for cfg in cnn_classifiers or []:
        if not bool(cfg.get("unique_identifier", False)):
            continue
        label = str(cfg.get("label", "") or "").strip()
        if label and label not in labels:
            labels.append(label)
    return tuple(labels)


def resolve_identity_heads(params: Mapping):
    """Identity-head labels from engine params, or ``HEADS_UNKNOWN``.

    ``HEADS_UNKNOWN`` (no ``CNN_CLASSIFIERS`` key at all) means the caller is
    running over a dataframe with no engine config -- e.g. re-running
    post-processing on a bare CSV -- and must fall back to the legacy
    "every CNN column counts" behavior. A *present but empty* list, or a
    list with no identity provider in it, is a real answer: no CNN column
    feeds identity.
    """
    if "CNN_CLASSIFIERS" not in params:
        return HEADS_UNKNOWN
    return identity_head_labels(params.get("CNN_CLASSIFIERS") or [])


def identity_class_columns(
    columns: Iterable, head_labels: Sequence[str]
) -> list[str]:
    """The ``CNN_<head>[_<factor>]_Class`` columns belonging to identity heads.

    Matched against the *known* head labels rather than by regex capture:
    ``^CNN_(.+)_Class$`` cannot distinguish a head label containing "_"
    from a ``<label>_<factor>`` pair, and guessing wrong silently drops or
    admits the wrong classifier.
    """
    heads = [str(h) for h in head_labels if str(h).strip()]
    out: list[str] = []
    for col in columns:
        name = str(col)
        if not name.endswith("_Class"):
            continue
        for head in heads:
            if name == f"CNN_{head}_Class" or name.startswith(f"CNN_{head}_"):
                out.append(name)
                break
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/identity/test_identity_heads.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
make commit-prep && make lint-moderate
git add src/hydra_suite/core/individual/identity/heads.py tests/identity/test_identity_heads.py
git commit -m "feat(identity): identity-head resolution and column scoping helper"
```

---

### Task 2: Scope UniqueIdentityKey to identity heads

**Files:**
- Modify: `src/hydra_suite/core/post/identity_postprocess.py:182-213`
- Modify: `src/hydra_suite/core/individual/postprocess_df.py:352-360`
- Test: `tests/test_unique_identity_key_derivation.py` (append)

**Interfaces:**
- Consumes: `heads.identity_class_columns`, `heads.HEADS_UNKNOWN` (Task 1).
- Produces: `derive_unique_identity_key_series(df, identity_heads=None) -> pd.Series` — `None` keeps legacy all-columns behavior.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_unique_identity_key_derivation.py
import pandas as pd

from hydra_suite.core.post.identity_postprocess import (
    derive_unique_identity_key_series,
    identity_sources_conflict,
    parse_identity_key,
)


def _two_head_df():
    return pd.DataFrame(
        {
            "CNN_colortag_Class": ["red_blue", "red_blue"],
            "CNN_colortag_Conf": [0.8, 0.8],
            "CNN_behavior_Class": ["walking", "grooming"],
            "CNN_behavior_Conf": [0.98, 0.97],
        }
    )


def test_key_excludes_non_identity_heads():
    keys = derive_unique_identity_key_series(_two_head_df(), identity_heads=("colortag",))
    assert parse_identity_key(keys.iloc[0]) == {"cnn:colortag": "red_blue"}
    assert parse_identity_key(keys.iloc[1]) == {"cnn:colortag": "red_blue"}


def test_behavior_change_is_not_an_identity_conflict():
    keys = derive_unique_identity_key_series(_two_head_df(), identity_heads=("colortag",))
    lhs = parse_identity_key(keys.iloc[0])
    rhs = parse_identity_key(keys.iloc[1])
    assert not identity_sources_conflict(lhs, rhs)


def test_behavior_change_conflicts_under_legacy_unscoped_call():
    # Documents the bug this task fixes: unscoped, the behavior head makes two
    # fragments of the SAME animal look like an identity conflict.
    keys = derive_unique_identity_key_series(_two_head_df())
    lhs = parse_identity_key(keys.iloc[0])
    rhs = parse_identity_key(keys.iloc[1])
    assert identity_sources_conflict(lhs, rhs)


def test_empty_identity_heads_drops_all_cnn_sources():
    keys = derive_unique_identity_key_series(_two_head_df(), identity_heads=())
    assert keys.isna().all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_unique_identity_key_derivation.py -v`
Expected: FAIL — `TypeError: derive_unique_identity_key_series() got an unexpected keyword argument 'identity_heads'`

- [ ] **Step 3: Write minimal implementation**

In `src/hydra_suite/core/post/identity_postprocess.py`, replace the signature and the column-selection block of `derive_unique_identity_key_series`:

```python
def derive_unique_identity_key_series(
    df: pd.DataFrame, identity_heads=None
) -> pd.Series:
    """Re-derive the ``UniqueIdentityKey`` column from per-row evidence columns.

    Builds, per row, a ``source -> value`` dict from ``DetectedTagLabel``
    (preferred) / ``DetectedTagID`` for the ``apriltag`` source and the
    ``CNN_<head>_Class`` / ``CNN_<head>_<factor>_Class`` columns of the
    *identity heads* for the CNN sources, then serializes it with
    :func:`format_identity_key`. Rows with no evidence get ``np.nan``
    (never an empty string or a bare label).

    ``identity_heads`` is the tuple of classifier labels marked
    ``unique_identifier`` (see ``identity.heads.identity_head_labels``).
    Classifiers that are not identity heads -- behavior, sex, caste -- must
    never enter this key: it feeds the relink identity veto
    (``processing.py:_score_relink_candidate``), where a mere behavior change
    across an occlusion gap would otherwise read as an identity conflict and
    refuse a legitimate relink. ``None`` preserves the legacy
    every-CNN-column behavior for callers with no classifier config.
    """
    if df is None or df.empty:
        return pd.Series([], index=getattr(df, "index", None), dtype=object)

    if identity_heads is None:
        cnn_class_columns = [
            col for col in df.columns if _CNN_CLASS_COLUMN_RE.match(str(col))
        ]
    else:
        from hydra_suite.core.individual.identity.heads import identity_class_columns

        cnn_class_columns = identity_class_columns(df.columns, identity_heads)
```

The rest of the function body (`has_tag_label` onward) is unchanged.

In `src/hydra_suite/core/individual/postprocess_df.py`, change the call at the end of `apply_identity_postprocessing_to_df`:

```python
    try:
        from hydra_suite.core.individual.identity.heads import (
            HEADS_UNKNOWN,
            resolve_identity_heads,
        )
        from hydra_suite.core.post.identity_postprocess import (
            derive_unique_identity_key_series,
        )

        _heads = resolve_identity_heads(params)
        with_pose_df[C.UNIQUE_IDENTITY_KEY] = derive_unique_identity_key_series(
            with_pose_df,
            identity_heads=None if _heads is HEADS_UNKNOWN else _heads,
        )
    except Exception:
        logger.exception("UniqueIdentityKey derivation failed; column left unset.")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_unique_identity_key_derivation.py tests/test_identity_postprocess.py tests/test_core_identity_postprocess_df.py -v`
Expected: PASS, no regressions.

- [ ] **Step 5: Commit**

```bash
make commit-prep && make lint-moderate
git add src/hydra_suite/core/post/identity_postprocess.py src/hydra_suite/core/individual/postprocess_df.py tests/test_unique_identity_key_derivation.py
git commit -m "fix(identity): scope UniqueIdentityKey to identity heads

Non-identity classifiers (behavior, sex, caste) were bundled into the
identity key, so a behavior change across an occlusion gap registered as
an identity conflict and vetoed a legitimate relink."
```

---

### Task 3: Scope the IdentityEvidence* summary columns to identity heads

**Files:**
- Modify: `src/hydra_suite/core/individual/postprocess_df.py:57-80`
- Test: `tests/test_core_identity_postprocess_df.py` (append)

**Interfaces:**
- Consumes: `heads.resolve_identity_heads`, `heads.HEADS_UNKNOWN`, `heads.identity_class_columns` (Task 1).
- Produces: no new public surface — `_annotate_identity_summary_columns` is nested and stays nested.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_core_identity_postprocess_df.py
import pandas as pd

from hydra_suite.core.individual.identity import columns as C
from hydra_suite.core.individual.postprocess_df import apply_identity_postprocessing_to_df


def _df_with_confident_behavior_head():
    return pd.DataFrame(
        {
            "TrajectoryID": [0, 0],
            "FrameID": [0, 1],
            "CNN_colortag_Class": ["red_blue", "red_blue"],
            "CNN_colortag_Conf": [0.80, 0.80],
            "CNN_behavior_Class": ["walking", "walking"],
            "CNN_behavior_Conf": [0.98, 0.98],
        }
    )


_PARAMS = {
    "CNN_CLASSIFIERS": [
        {"label": "colortag", "unique_identifier": True},
        {"label": "behavior", "unique_identifier": False},
    ],
    "IDENTITY_POSTHOC_ENABLED": False,
    "ENABLE_IDENTITY_FRAGMENT_SOLVER": False,
}


def test_top_evidence_label_ignores_more_confident_non_identity_head():
    out = apply_identity_postprocessing_to_df(_df_with_confident_behavior_head(), _PARAMS)
    assert out[C.EVIDENCE_TOPLABEL].tolist() == ["red_blue", "red_blue"]
    assert out[C.EVIDENCE_CONFIDENCE].tolist() == [0.80, 0.80]


def test_non_identity_head_columns_are_still_exported():
    out = apply_identity_postprocessing_to_df(_df_with_confident_behavior_head(), _PARAMS)
    assert out["CNN_behavior_Class"].tolist() == ["walking", "walking"]


def test_absent_classifier_config_keeps_legacy_all_columns_behavior():
    # No CNN_CLASSIFIERS key at all -> legacy fallback, behavior head wins.
    out = apply_identity_postprocessing_to_df(
        _df_with_confident_behavior_head(),
        {"IDENTITY_POSTHOC_ENABLED": False, "ENABLE_IDENTITY_FRAGMENT_SOLVER": False},
    )
    assert out[C.EVIDENCE_TOPLABEL].tolist() == ["walking", "walking"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_core_identity_postprocess_df.py -v -k "evidence_label or still_exported or legacy_all_columns"`
Expected: FAIL on the first test — `EVIDENCE_TOPLABEL` is `["walking", "walking"]` because the behavior head's 0.98 beats 0.80.

- [ ] **Step 3: Write minimal implementation**

In `src/hydra_suite/core/individual/postprocess_df.py`, replace the column-selection block at the top of `_annotate_identity_summary_columns` (currently lines 66-73):

```python
        out = df.copy()
        from hydra_suite.core.individual.identity.heads import (
            HEADS_UNKNOWN,
            identity_class_columns,
            resolve_identity_heads,
        )

        # Only identity heads (`unique_identifier=True`) may feed the identity
        # evidence summary. A behavior/sex/caste classifier is output, not
        # identity: unscoped, its class wins IdentityEvidenceTopLabel whenever
        # it is more confident than the identity classifier.
        _heads = resolve_identity_heads(params)
        if _heads is HEADS_UNKNOWN:
            cnn_class_columns = [
                col
                for col in out.columns
                if str(col).startswith("CNN_") and str(col).endswith("_Class")
            ]
        else:
            cnn_class_columns = identity_class_columns(out.columns, _heads)
        cnn_conf_columns = {
            col: f"{str(col)[: -len('_Class')]}_Conf" for col in cnn_class_columns
        }
```

`params` is already in scope — `_annotate_identity_summary_columns` is nested inside `apply_identity_postprocessing_to_df(with_pose_df, params, ...)`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_core_identity_postprocess_df.py tests/identity/test_identity_columns.py tests/test_identity_conflict_resolution.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
make commit-prep && make lint-moderate
git add src/hydra_suite/core/individual/postprocess_df.py tests/test_core_identity_postprocess_df.py
git commit -m "fix(identity): scope IdentityEvidence* summary columns to identity heads"
```

- [ ] **Step 6: Run the Slice 1 equivalence gate**

Run the MPS matrix from Global Constraints. Expected: **EQUIVALENT on every clip** — fixtures carry at most one identity classifier, so Slice 1 must be a no-op for them. Verify CSV row counts > 1 before trusting any EQUIVALENT verdict. Record the result in the commit message of the next task or in `docs/superpowers/specs/done/` notes.

---

# SLICE 2 — Cross-product catalog across identity providers

### Task 4: Identity axes and cross-product catalog resolution

**Files:**
- Modify: `src/hydra_suite/core/individual/identity/resolve.py:47-100`
- Test: `tests/identity/test_catalog_cross_product.py`

**Interfaces:**
- Consumes: `heads.identity_head_labels` (Task 1).
- Produces:
  - `identity_axes(cnn_classifiers) -> list[IdentityAxis]` where
    `IdentityAxis = NamedTuple(model_label: str, factor_name: str, classes: tuple[str, ...])`, in model-config order then factor order.
  - `resolve_catalog_spec(cnn_classifiers, tag_identity_labels) -> IdentityCatalogSpec` — unchanged signature, now cross-product across models. `CatalogEntry.factors` pairs are `(f"{model_label}:{factor_name}", class_name)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/identity/test_catalog_cross_product.py
from hydra_suite.core.individual.identity.resolve import identity_axes, resolve_catalog_spec


def _thorax():
    return {
        "label": "thorax",
        "unique_identifier": True,
        "class_names_per_factor": [["red", "blue"]],
        "factor_names": ["dot"],
    }


def _abdomen():
    return {
        "label": "abdomen",
        "unique_identifier": True,
        "class_names_per_factor": [["square", "circle"]],
        "factor_names": ["shape"],
    }


def _behavior():
    return {
        "label": "behavior",
        "unique_identifier": False,
        "class_names_per_factor": [["walking", "grooming"]],
        "factor_names": ["state"],
    }


def test_axes_span_all_identity_models_in_config_order():
    axes = identity_axes([_thorax(), _abdomen(), _behavior()])
    assert [(a.model_label, a.factor_name, a.classes) for a in axes] == [
        ("thorax", "dot", ("red", "blue")),
        ("abdomen", "shape", ("square", "circle")),
    ]


def test_two_identity_models_cross_product_not_union():
    spec = resolve_catalog_spec([_thorax(), _abdomen()], [])
    assert spec.labels == ("red_square", "red_circle", "blue_square", "blue_circle")


def test_cross_product_entries_carry_qualified_factor_provenance():
    spec = resolve_catalog_spec([_thorax(), _abdomen()], [])
    assert spec.entries[0].factors == (
        ("thorax:dot", "red"),
        ("abdomen:shape", "square"),
    )


def test_non_identity_model_contributes_no_axis():
    spec = resolve_catalog_spec([_thorax(), _behavior()], [])
    assert spec.labels == ("red", "blue")


def test_single_multifactor_model_is_unchanged():
    # The pre-existing within-model product must be preserved exactly.
    cfg = {
        "label": "colortag",
        "unique_identifier": True,
        "class_names_per_factor": [["red", "blue"], ["big", "small"]],
        "factor_names": ["hue", "size"],
    }
    spec = resolve_catalog_spec([cfg], [])
    assert spec.labels == ("red_big", "red_small", "blue_big", "blue_small")


def test_missing_factor_names_fall_back_to_positional():
    cfg = {
        "label": "colortag",
        "unique_identifier": True,
        "class_names_per_factor": [["red", "blue"]],
    }
    spec = resolve_catalog_spec([cfg], [])
    assert spec.entries[0].factors == (("colortag:factor0", "red"),)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/identity/test_catalog_cross_product.py -v`
Expected: FAIL — `ImportError: cannot import name 'identity_axes'`; and once that import exists, `test_two_identity_models_cross_product_not_union` fails with `("red", "blue", "square", "circle")` (the union bug).

- [ ] **Step 3: Write minimal implementation**

In `src/hydra_suite/core/individual/identity/resolve.py`, add the axis model and rewrite the CNN half of `resolve_catalog_spec`:

```python
from typing import Mapping, NamedTuple, Sequence


class IdentityAxis(NamedTuple):
    """One factor axis of the identity domain.

    ``model_label`` is the owning classifier's label, ``factor_name`` its
    factor's name (positional ``factor<i>`` fallback when the config carries
    no ``factor_names``), and ``classes`` that factor's class vocabulary.
    """

    model_label: str
    factor_name: str
    classes: tuple[str, ...]

    @property
    def qualified_name(self) -> str:
        return f"{self.model_label}:{self.factor_name}"


def identity_axes(cnn_classifiers: Sequence[Mapping]) -> list[IdentityAxis]:
    """The identity domain's factor axes, in model-config then factor order.

    Every identity-providing model contributes each of its non-empty factors
    as one axis. The catalog is the cartesian product over ALL axes -- two
    complementary tag models (thorax colour + abdomen shape) describe one
    animal jointly, so their classes multiply rather than compete.

    Redundant voters (two models over the same class vocabulary) are not
    supported and would produce nonsensical ``ant1_ant1`` composites; see the
    Non-goals section of the design doc.
    """
    axes: list[IdentityAxis] = []
    for cfg in cnn_classifiers or []:
        if not bool(cfg.get("unique_identifier", False)):
            continue
        model_label = str(cfg.get("label", "") or "").strip() or "cnn"
        cnpf = list(cfg.get("class_names_per_factor") or [])
        if not cnpf:
            cnpf = _read_factors_from_model_file(str(cfg.get("model_path", "")))
        factor_names = list(cfg.get("factor_names") or [])
        non_empty_index = 0
        for i, factor_labels in enumerate(cnpf):
            classes = tuple(str(c) for c in (factor_labels or []) if c)
            if not classes:
                continue
            name = (
                str(factor_names[i])
                if i < len(factor_names) and str(factor_names[i]).strip()
                else f"factor{non_empty_index}"
            )
            non_empty_index += 1
            axes.append(IdentityAxis(model_label, name, classes))
    return axes
```

Then replace the `for cfg in cnn_classifiers or []:` loop body in `resolve_catalog_spec` (everything from that `for` down to `cnn_derived = set(seen)`) with:

```python
    axes = identity_axes(cnn_classifiers)
    if axes:
        for combo in itertools.product(*[a.classes for a in axes]):
            pairs = tuple(
                (axes[i].qualified_name, str(c)) for i, c in enumerate(combo) if c
            )
            display = "_".join(str(c) for c in combo if c)
            _add(display, pairs, "cnn")

    cnn_derived = set(seen)
```

The `_read_factors_from_model_file` helper, the `_add` closure, the tag-label loop, and the return are unchanged.

Note on the flat-label fallback: the old code had a `cfg.get("labels", [])` fallback when a model exposed no per-factor classes. `identity_axes` covers this by yielding no axis for such a model; if *no* identity model yields an axis, the catalog has no CNN entries and the tag-label branch (which filters on `cnn_derived`) admits tag labels as before. Preserve that behavior by leaving the tag loop untouched.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/identity/test_catalog_cross_product.py tests/identity/test_catalog_spec.py tests/test_tag_identity.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
make commit-prep && make lint-moderate
git add src/hydra_suite/core/individual/identity/resolve.py tests/identity/test_catalog_cross_product.py
git commit -m "feat(identity): cross-product catalog across identity providers

Two complementary tag models described one animal jointly but were unioned
into competing identities. They now contribute factor axes to one product."
```

---

### Task 5: Catalog-size warning

**Files:**
- Modify: `src/hydra_suite/core/individual/identity/resolve.py` (top of `resolve_catalog_spec`)
- Test: `tests/identity/test_catalog_cross_product.py` (append)

**Interfaces:**
- Consumes: `identity_axes` (Task 4).
- Produces: module-level `CATALOG_SIZE_WARN_THRESHOLD = 256`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/identity/test_catalog_cross_product.py
import logging


def _big_model(i):
    return {
        "label": f"m{i}",
        "unique_identifier": True,
        "class_names_per_factor": [[f"c{j}" for j in range(8)]],
        "factor_names": [f"f{i}"],
    }


def test_large_catalog_warns_and_names_axes(caplog):
    with caplog.at_level(logging.WARNING):
        spec = resolve_catalog_spec([_big_model(i) for i in range(4)], [])
    assert len(spec.entries) == 8**4
    assert any("m0:f0" in r.message % r.args for r in caplog.records)


def test_small_catalog_does_not_warn(caplog):
    with caplog.at_level(logging.WARNING):
        resolve_catalog_spec([_thorax(), _abdomen()], [])
    assert not caplog.records
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/identity/test_catalog_cross_product.py -v -k warns`
Expected: FAIL — no warning is emitted.

- [ ] **Step 3: Write minimal implementation**

In `resolve.py`, add a module logger and the check right after `axes = identity_axes(cnn_classifiers)`:

```python
import logging

logger = logging.getLogger(__name__)

CATALOG_SIZE_WARN_THRESHOLD = 256
"""Entry count above which the cross-product catalog is flagged as suspicious.

The Hungarian assignment cost matrix is N x (K + N) in the catalog size K, so
a runaway product is a real cost. This is a warning, not a cap: naming the
contributing axes is more useful than refusing to run.
"""
```

```python
    axes = identity_axes(cnn_classifiers)
    if axes:
        projected = 1
        for a in axes:
            projected *= max(1, len(a.classes))
        if projected > CATALOG_SIZE_WARN_THRESHOLD:
            logger.warning(
                "Identity catalog is the cross-product of %d axes and has %d "
                "entries (> %d). Contributing axes: %s. Check that every "
                "classifier marked 'unique identifier' really is one, and "
                "consider marking non-identifying classes.",
                len(axes),
                projected,
                CATALOG_SIZE_WARN_THRESHOLD,
                ", ".join(f"{a.qualified_name}({len(a.classes)})" for a in axes),
            )
        for combo in itertools.product(*[a.classes for a in axes]):
            ...
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/identity/test_catalog_cross_product.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
make commit-prep && make lint-moderate
git add src/hydra_suite/core/individual/identity/resolve.py tests/identity/test_catalog_cross_product.py
git commit -m "feat(identity): warn when the cross-product catalog grows past 256 entries"
```

---

### Task 6: Generalize the phase→global evidence remap

**Files:**
- Create: `src/hydra_suite/core/individual/identity/phase_remap.py`
- Modify: `src/hydra_suite/core/tracking/worker.py:1927-1953`
- Test: `tests/identity/test_phase_catalog_remap.py`

**Interfaces:**
- Consumes: `IdentityCatalog`, `IdentityCatalogSpec` (existing), `IdentityAxis` / `identity_axes` (Task 4).
- Produces:
  - `build_phase_label_map(spec, catalog, model_label) -> dict[str, list[int]]` — phase display label → global catalog indices.
  - `remap_phase_log_probs(log_probs, source_labels, catalog, phase_label_map) -> np.ndarray`

**Why this task is dangerous:** the current remap matches phase labels to global labels by exact string. Under a cross-product catalog, model A's phase label `red` matches nothing, `continue` fires for every label, and **all identity evidence is silently discarded with no error** — identity simply stops working. The non-degeneracy test below is the guard against shipping that.

- [ ] **Step 1: Write the failing test**

```python
# tests/identity/test_phase_catalog_remap.py
import numpy as np

from hydra_suite.core.individual.identity.catalog import IdentityCatalog
from hydra_suite.core.individual.identity.phase_remap import (
    build_phase_label_map,
    remap_phase_log_probs,
)
from hydra_suite.core.individual.identity.resolve import resolve_catalog_spec

THORAX = {
    "label": "thorax",
    "unique_identifier": True,
    "class_names_per_factor": [["red", "blue"]],
    "factor_names": ["dot"],
}
ABDOMEN = {
    "label": "abdomen",
    "unique_identifier": True,
    "class_names_per_factor": [["square", "circle"]],
    "factor_names": ["shape"],
}


def _legacy_remap(log_probs, source_labels, catalog):
    """The pre-change exact-match implementation, kept as the equality oracle."""
    arr = np.asarray(log_probs, dtype=np.float64)
    probs = np.exp(arr - np.max(arr))
    probs /= np.clip(probs.sum(), 1e-300, None)
    remapped = np.full(catalog.size, 1e-300, dtype=np.float64)
    for src_idx, label in enumerate(source_labels):
        if not catalog.contains(label):
            continue
        remapped[catalog.index_of(label)] += float(probs[src_idx])
    remapped /= np.clip(remapped.sum(), 1e-300, None)
    return np.log(np.clip(remapped, 1e-300, None))


def test_single_model_remap_matches_legacy_exactly():
    spec = resolve_catalog_spec([THORAX], [])
    catalog = IdentityCatalog.from_spec(spec)
    phase_labels = ("unknown", "red", "blue")
    log_probs = np.log(np.array([0.1, 0.7, 0.2]))
    pmap = build_phase_label_map(spec, catalog, "thorax")
    got = remap_phase_log_probs(log_probs, phase_labels, catalog, pmap)
    np.testing.assert_array_equal(got, _legacy_remap(log_probs, phase_labels, catalog))


def test_two_model_remap_is_not_degenerate():
    # The failure mode this test exists for: exact-match remapping drops every
    # label and leaves a flat/unknown posterior, so identity silently dies.
    spec = resolve_catalog_spec([THORAX, ABDOMEN], [])
    catalog = IdentityCatalog.from_spec(spec)
    pmap = build_phase_label_map(spec, catalog, "thorax")
    log_probs = np.log(np.array([0.05, 0.9, 0.05]))
    got = remap_phase_log_probs(log_probs, ("unknown", "red", "blue"), catalog, pmap)
    probs = np.exp(got)
    red_idxs = [catalog.index_of("red_square"), catalog.index_of("red_circle")]
    blue_idxs = [catalog.index_of("blue_square"), catalog.index_of("blue_circle")]
    assert probs[red_idxs].sum() > probs[blue_idxs].sum()
    assert probs[red_idxs].sum() > probs[catalog.unknown_index]


def test_two_models_fuse_to_the_correct_composite():
    spec = resolve_catalog_spec([THORAX, ABDOMEN], [])
    catalog = IdentityCatalog.from_spec(spec)
    thorax_lp = remap_phase_log_probs(
        np.log(np.array([0.05, 0.9, 0.05])),
        ("unknown", "red", "blue"),
        catalog,
        build_phase_label_map(spec, catalog, "thorax"),
    )
    abdomen_lp = remap_phase_log_probs(
        np.log(np.array([0.05, 0.05, 0.9])),
        ("unknown", "square", "circle"),
        catalog,
        build_phase_label_map(spec, catalog, "abdomen"),
    )
    fused = thorax_lp + abdomen_lp
    assert catalog.label_of(int(np.argmax(fused))) == "red_circle"


def test_phase_label_map_covers_every_phase_label():
    spec = resolve_catalog_spec([THORAX, ABDOMEN], [])
    catalog = IdentityCatalog.from_spec(spec)
    pmap = build_phase_label_map(spec, catalog, "thorax")
    assert sorted(pmap) == ["blue", "red"]
    assert len(pmap["red"]) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/identity/test_phase_catalog_remap.py -v`
Expected: FAIL — `ModuleNotFoundError: ...identity.phase_remap`

- [ ] **Step 3: Write minimal implementation**

```python
# src/hydra_suite/core/individual/identity/phase_remap.py
"""Phase-local evidence -> global catalog remapping.

Each CNN identity classifier builds its own *phase-local* cartesian catalog
from its own factors (``evidence_builder.build_phase_catalog_labels``). With
one identity model, that phase basis is identical to the global catalog and
the remap is a relabeling. With several identity models the global catalog is
the cross-product of all their axes, so a phase label such as ``"red"`` names
a *set* of global entries (``red_square``, ``red_circle``) rather than one.

This module owns that distribution. The semantics match
``substrate._factor_log_prob``'s composite branch -- assign each reachable
entry the phase probability, then renormalize -- lifted one level up from
within-model factors to across-model phases.

Core: numpy + stdlib only.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from hydra_suite.core.individual.identity.catalog import IdentityCatalog
from hydra_suite.core.individual.identity.spec import IdentityCatalogSpec


def build_phase_label_map(
    spec: IdentityCatalogSpec,
    catalog: IdentityCatalog,
    model_label: str,
) -> dict[str, list[int]]:
    """``phase_display_label -> [global catalog indices]`` for one model.

    A global entry is reachable from a phase label iff the entry's classes on
    *that model's* axes, joined in axis order with "_", equal the phase label
    -- the same join ``build_phase_catalog_labels`` used to name it.
    """
    prefix = f"{model_label}:"
    out: dict[str, list[int]] = {}
    for entry in spec.entries:
        own = [cls for (axis, cls) in entry.factors if axis.startswith(prefix)]
        if not own:
            continue
        phase_label = "_".join(own)
        try:
            idx = catalog.index_of(entry.display_label)
        except KeyError:
            continue
        out.setdefault(phase_label, []).append(idx)
    return out


def remap_phase_log_probs(
    log_probs: np.ndarray,
    source_labels: Sequence[str],
    catalog: IdentityCatalog,
    phase_label_map: dict[str, list[int]],
) -> np.ndarray:
    """Map one phase-basis log-prob vector into the global catalog basis.

    Unreachable entries keep the ``1e-300`` floor rather than a hard zero, so
    the single-model case is arithmetically identical to the exact-match
    implementation this replaced.
    """
    arr = np.asarray(log_probs, dtype=np.float64)
    probs = np.exp(arr - np.max(arr))
    probs /= np.clip(probs.sum(), 1e-300, None)

    remapped = np.full(catalog.size, 1e-300, dtype=np.float64)
    for src_idx, raw_label in enumerate(source_labels):
        label = str(raw_label)
        targets = phase_label_map.get(label)
        if targets is None:
            # "unknown" (and any label with no structured provenance, e.g. a
            # tag-sourced entry) still resolves by direct lookup.
            if not catalog.contains(label):
                continue
            targets = [catalog.index_of(label)]
        for idx in targets:
            remapped[idx] += float(probs[src_idx])

    remapped /= np.clip(remapped.sum(), 1e-300, None)
    return np.log(np.clip(remapped, 1e-300, None))
```

Then in `src/hydra_suite/core/tracking/worker.py`, replace the body of
`_remap_source_log_probs_to_catalog` (lines 1927-1953). The phase-label maps are
built once per run, keyed by classifier label, from the already-resolved catalog
spec — the local is `_catalog_spec`, assigned at `worker.py:1839` inside the same
`try` block that sets `_identity_catalog`. Initialize `_catalog_spec = None`
next to `_identity_catalog = None` at line 1825 so the reference below is safe
when catalog resolution raised:

```python
        _phase_label_maps: dict[str, dict] = {}
        if _identity_catalog is not None and _catalog_spec is not None:
            from hydra_suite.core.individual.identity.phase_remap import (
                build_phase_label_map,
            )

            for _cfg in p.get("CNN_CLASSIFIERS", []) or []:
                if not bool(_cfg.get("unique_identifier", False)):
                    continue
                _ml = str(_cfg.get("label", "") or "").strip()
                if _ml:
                    _phase_label_maps[_ml] = build_phase_label_map(
                        _catalog_spec, _identity_catalog, _ml
                    )

        def _remap_source_log_probs_to_catalog(
            log_probs: np.ndarray,
            source_labels: list[str] | tuple[str, ...] | None,
            source_name: str = "",
        ) -> np.ndarray:
            if _identity_catalog is None:
                return np.asarray(log_probs, dtype=np.float64)
            arr = np.asarray(log_probs, dtype=np.float64)
            if source_labels is None:
                if len(arr) == _identity_catalog.size:
                    out = arr.copy()
                    out -= np.logaddexp.reduce(out)
                    return out
                return _identity_catalog.known_uniform_log_prior()

            labels = tuple(str(label) for label in source_labels)
            if len(labels) != len(arr):
                return _identity_catalog.known_uniform_log_prior()

            from hydra_suite.core.individual.identity.phase_remap import (
                remap_phase_log_probs,
            )

            return remap_phase_log_probs(
                arr,
                labels,
                _identity_catalog,
                _phase_label_maps.get(str(source_name), {}),
            )
```

Update the single call site at `worker.py:3054` to pass the source name:

```python
                                    _mapped_lp = _remap_source_log_probs_to_catalog(
                                        _cached_ev.log_probs,
                                        _source_labels,
                                        _label,
                                    )
```

(`_label` is the CNN phase state's classifier label already in scope in that loop.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/identity/test_phase_catalog_remap.py tests/identity/test_evidence_phase_basis_parity.py tests/identity/test_evidence_builder_parity.py tests/identity/test_evidence_sidecar_consumption.py tests/test_identity_online.py -v`
Expected: PASS. The phase-basis parity test is the one that would catch a regression in the single-model path.

- [ ] **Step 5: Commit**

```bash
make commit-prep && make lint-moderate
git add src/hydra_suite/core/individual/identity/phase_remap.py src/hydra_suite/core/tracking/worker.py tests/identity/test_phase_catalog_remap.py
git commit -m "feat(identity): distribute phase evidence across composite catalog entries

Exact-label matching would drop every phase label against a cross-product
catalog, silently discarding all identity evidence."
```

- [ ] **Step 6: Run the Slice 2 equivalence gate**

Run the MPS matrix from Global Constraints, then the CUDA leg on mehek. Expected: **EQUIVALENT on every clip** (fixtures are single-identity-model). A diff here means the single-model remap path changed — investigate before proceeding to Slice 3.

---

# SLICE 3 — Non-identifying classes

### Task 7: Exclude declared non-identifying classes from the catalog

**Files:**
- Modify: `src/hydra_suite/core/individual/identity/resolve.py`
- Test: `tests/identity/test_non_identifying_classes.py`

**Interfaces:**
- Consumes: `identity_axes`, `IdentityAxis` (Task 4).
- Produces:
  - `is_non_identifying(combo: Sequence[str], axes: Sequence[IdentityAxis], marks_by_model: Mapping[str, Sequence[str]], display_label: str) -> bool`
  - `non_identifying_marks(cnn_classifiers) -> dict[str, tuple[str, ...]]` — `model_label -> declared marks`
  - `excluded_display_labels(cnn_classifiers) -> frozenset[str]` — every composite the marks exclude, used by Task 8's reporting.

- [ ] **Step 1: Write the failing test**

```python
# tests/identity/test_non_identifying_classes.py
from hydra_suite.core.individual.identity.resolve import (
    excluded_display_labels,
    resolve_catalog_spec,
)


def _tags(non_identifying=()):
    return {
        "label": "colortag",
        "unique_identifier": True,
        "class_names_per_factor": [["red", "notag"], ["blue", "notag"]],
        "factor_names": ["front", "back"],
        "non_identifying_classes": list(non_identifying),
    }


def test_no_marks_is_a_no_op():
    assert resolve_catalog_spec([_tags()], []).labels == (
        "red_blue",
        "red_notag",
        "notag_blue",
        "notag_notag",
    )


def test_bare_class_mark_excludes_every_containing_composite():
    spec = resolve_catalog_spec([_tags(["notag"])], [])
    assert spec.labels == ("red_blue",)


def test_axis_scoped_mark_excludes_only_that_axis():
    spec = resolve_catalog_spec([_tags(["front:notag"])], [])
    assert spec.labels == ("red_blue", "red_notag")


def test_whole_composite_mark_excludes_exactly_that_label():
    spec = resolve_catalog_spec([_tags(["notag_notag"])], [])
    assert spec.labels == ("red_blue", "red_notag", "notag_blue")


def test_excluded_display_labels_reports_what_was_dropped():
    assert excluded_display_labels([_tags(["notag_notag"])]) == frozenset(
        {"notag_notag"}
    )
    assert excluded_display_labels([_tags(["notag"])]) == frozenset(
        {"red_notag", "notag_blue", "notag_notag"}
    )


def test_all_excluded_yields_empty_spec_without_raising(caplog):
    import logging

    with caplog.at_level(logging.WARNING):
        spec = resolve_catalog_spec([_tags(["red", "notag", "blue"])], [])
    assert spec.entries == ()
    assert any("every identity" in (r.message % r.args).lower() for r in caplog.records)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/identity/test_non_identifying_classes.py -v`
Expected: FAIL — `ImportError: cannot import name 'excluded_display_labels'`; `test_bare_class_mark...` would fail anyway since marks are ignored.

- [ ] **Step 3: Write minimal implementation**

Add to `resolve.py`:

```python
def non_identifying_marks(
    cnn_classifiers: Sequence[Mapping],
) -> dict[str, tuple[str, ...]]:
    """``model_label -> declared non-identifying marks`` for identity models."""
    out: dict[str, tuple[str, ...]] = {}
    for cfg in cnn_classifiers or []:
        if not bool(cfg.get("unique_identifier", False)):
            continue
        label = str(cfg.get("label", "") or "").strip() or "cnn"
        marks = tuple(
            str(m).strip()
            for m in (cfg.get("non_identifying_classes") or [])
            if str(m).strip()
        )
        if marks:
            out[label] = marks
    return out


def is_non_identifying(
    combo: Sequence[str],
    axes: Sequence[IdentityAxis],
    marks_by_model: Mapping[str, Sequence[str]],
    display_label: str,
) -> bool:
    """True if this composite is excluded by any declared mark.

    Three accepted mark forms, all resolved here so the rest of the system
    only ever sees per-entry exclusion:

    - ``"notag"``       -- that class in any axis of the declaring model
    - ``"front:notag"`` -- that class in the named factor of that model
    - ``"notag_notag"`` -- that whole global composite display label
    """
    for model_label, marks in marks_by_model.items():
        for mark in marks:
            if mark == display_label:
                return True
            scoped_factor, _, scoped_class = mark.partition(":")
            for i, axis in enumerate(axes):
                if axis.model_label != model_label or i >= len(combo):
                    continue
                cls = str(combo[i])
                if scoped_class:
                    if axis.factor_name == scoped_factor and cls == scoped_class:
                        return True
                elif cls == mark:
                    return True
    return False


def excluded_display_labels(
    cnn_classifiers: Sequence[Mapping],
) -> frozenset[str]:
    """Every composite display label the declared marks exclude.

    The reporting layer uses this to recognize an observed composite as
    non-identifying without re-deriving the mark semantics.
    """
    axes = identity_axes(cnn_classifiers)
    marks = non_identifying_marks(cnn_classifiers)
    if not axes or not marks:
        return frozenset()
    out = set()
    for combo in itertools.product(*[a.classes for a in axes]):
        display = "_".join(str(c) for c in combo if c)
        if display and is_non_identifying(combo, axes, marks, display):
            out.add(display)
    return frozenset(out)
```

Then wire the filter into `resolve_catalog_spec`'s product loop:

```python
    axes = identity_axes(cnn_classifiers)
    marks_by_model = non_identifying_marks(cnn_classifiers)
    if axes:
        # ... size warning as in Task 5 ...
        for combo in itertools.product(*[a.classes for a in axes]):
            display = "_".join(str(c) for c in combo if c)
            if marks_by_model and is_non_identifying(
                combo, axes, marks_by_model, display
            ):
                # Excluded from the identity domain entirely: no Hungarian
                # column, no blocked_labels entry, no commit-blocking, no swap
                # candidacy, no offline collision veto. Any number of tracks
                # may carry this class, and none is ever merged onto another
                # because of it.
                continue
            pairs = tuple(
                (axes[i].qualified_name, str(c)) for i, c in enumerate(combo) if c
            )
            _add(display, pairs, "cnn")

        if marks_by_model and not seen:
            logger.warning(
                "Every identity in the catalog was excluded by the declared "
                "non-identifying classes (%s). Identity resolution will not "
                "run for this session.",
                ", ".join(
                    f"{m}: {', '.join(v)}" for m, v in sorted(marks_by_model.items())
                ),
            )
```

The empty spec is already handled downstream ("catalog_spec had no entries"); do **not** let `IdentityCatalog.from_labels`'s empty-list `ValueError` be reached.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/identity/test_non_identifying_classes.py tests/identity/test_catalog_cross_product.py tests/identity/test_catalog_spec.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
make commit-prep && make lint-moderate
git add src/hydra_suite/core/individual/identity/resolve.py tests/identity/test_non_identifying_classes.py
git commit -m "feat(identity): exclude declared non-identifying classes from the catalog

Untagged animals no longer compete for one exclusive notag_notag slot: the
label is not in the identity domain, so no exclusivity applies to it."
```

---

### Task 8: Report non-identifying composites as a descriptive final label

**Files:**
- Modify: `src/hydra_suite/core/individual/identity/columns.py` (add source constant)
- Modify: `src/hydra_suite/core/individual/identity/heads.py` (add axis-column resolution)
- Modify: `src/hydra_suite/core/individual/postprocess_df.py` (new stamp, called before `_mirror_realtime_and_tag_into_final`)
- Test: `tests/identity/test_non_identifying_classes.py` (append)

**Interfaces:**
- Consumes: `resolve.identity_axes`, `resolve.excluded_display_labels` (Tasks 4, 7); `heads` (Task 1).
- Produces:
  - `columns.IdentityFinalSource.NON_IDENTIFYING = "nonidentifying"`
  - `heads.identity_axis_columns(columns, cnn_classifiers) -> list[tuple[str, str]]` — ordered `(class_col, conf_col)` per identity axis, resolved against the columns actually present.
  - `postprocess_df._stamp_non_identifying_labels(df, params) -> pd.DataFrame`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/identity/test_non_identifying_classes.py
import numpy as np
import pandas as pd

from hydra_suite.core.individual.identity import columns as C
from hydra_suite.core.individual.postprocess_df import apply_identity_postprocessing_to_df

_PARAMS = {
    "CNN_CLASSIFIERS": [
        {
            "label": "colortag",
            "unique_identifier": True,
            "class_names_per_factor": [["red", "notag"], ["blue", "notag"]],
            "factor_names": ["front", "back"],
            "non_identifying_classes": ["notag_notag"],
        }
    ],
    "IDENTITY_POSTHOC_ENABLED": False,
    "ENABLE_IDENTITY_FRAGMENT_SOLVER": False,
}


def _three_untagged_tracks():
    rows = []
    for traj in (0, 1, 2):
        for frame in (0, 1):
            rows.append(
                {
                    "TrajectoryID": traj,
                    "FrameID": frame,
                    "CNN_colortag_front_Class": "notag",
                    "CNN_colortag_front_Conf": 0.9,
                    "CNN_colortag_back_Class": "notag",
                    "CNN_colortag_back_Conf": 0.7,
                }
            )
    return pd.DataFrame(rows)


def test_untagged_tracks_are_labelled_not_unknown():
    out = apply_identity_postprocessing_to_df(_three_untagged_tracks(), _PARAMS)
    assert set(out[C.FINAL_LABEL]) == {"notag_notag"}
    assert set(out[C.FINAL_SOURCE]) == {"nonidentifying"}


def test_untagged_tracks_keep_the_unknown_slot_id():
    out = apply_identity_postprocessing_to_df(_three_untagged_tracks(), _PARAMS)
    assert set(out[C.FINAL_ID]) == {0}


def test_untagged_tracks_are_never_merged():
    out = apply_identity_postprocessing_to_df(_three_untagged_tracks(), _PARAMS)
    assert out["TrajectoryID"].nunique() == 3


def test_confidence_is_the_weakest_axis():
    out = apply_identity_postprocessing_to_df(_three_untagged_tracks(), _PARAMS)
    assert np.allclose(out[C.FINAL_CONFIDENCE], 0.7)


def test_a_real_identity_is_not_overwritten():
    df = _three_untagged_tracks()
    df.loc[df["TrajectoryID"] == 0, "CNN_colortag_front_Class"] = "red"
    df.loc[df["TrajectoryID"] == 0, "CNN_colortag_back_Class"] = "blue"
    df[C.FINAL_LABEL] = ["red_blue"] * 2 + [np.nan] * 4
    df[C.FINAL_SOURCE] = ["offline"] * 2 + [""] * 4
    out = apply_identity_postprocessing_to_df(df, _PARAMS)
    traj0 = out[out["TrajectoryID"] == 0]
    assert set(traj0[C.FINAL_LABEL]) == {"red_blue"}
    assert set(traj0[C.FINAL_SOURCE]) == {"offline"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/identity/test_non_identifying_classes.py -v -k "untagged or weakest or not_overwritten"`
Expected: FAIL — `IdentityFinalLabel` is absent or `unknown`, never `notag_notag`.

- [ ] **Step 3: Write minimal implementation**

Add to `src/hydra_suite/core/individual/identity/columns.py`, inside `class IdentityFinalSource`:

```python
    NON_IDENTIFYING = "nonidentifying"
    """A declared non-identifying composite (e.g. an untagged animal).

    The label is descriptive only -- ``IdentityFinalID`` stays at the unknown
    slot (0), so nothing downstream can mistake it for a resolved identity.
    """
```

Add to `src/hydra_suite/core/individual/identity/heads.py`:

```python
def identity_axis_columns(columns, cnn_classifiers) -> list:
    """Ordered ``(class_col, conf_col)`` pairs, one per identity axis.

    Mirrors ``properties.export.build_cnn_output_columns``' naming: a
    single-factor model writes flat ``CNN_<label>_Class`` columns, a
    multi-factor model writes ``CNN_<label>_<factor>_Class``. Falls back to
    the flat name when the per-factor column is absent from `columns`, so a
    config/model factor-name mismatch degrades to "no axis" rather than a
    KeyError.
    """
    from hydra_suite.core.individual.identity.resolve import identity_axes

    present = {str(c) for c in columns}
    axes = identity_axes(cnn_classifiers)
    per_model = {}
    for axis in axes:
        per_model[axis.model_label] = per_model.get(axis.model_label, 0) + 1

    out = []
    for axis in axes:
        candidates = []
        if per_model[axis.model_label] > 1:
            candidates.append(f"CNN_{axis.model_label}_{axis.factor_name}")
        candidates.append(f"CNN_{axis.model_label}")
        for base in candidates:
            if f"{base}_Class" in present:
                out.append((f"{base}_Class", f"{base}_Conf"))
                break
    return out
```

Add to `src/hydra_suite/core/individual/postprocess_df.py`, at module level:

```python
def _stamp_non_identifying_labels(df, params):
    """Give trajectories whose observed composite is non-identifying a name.

    Excluded classes are absent from the identity catalog (by design -- that
    is what frees them from exclusivity), so those tracks resolve to unknown.
    "Unknown" is the wrong report: "the classifier saw no tag" and "the
    classifier could not tell" are different findings. This stamps the
    observed composite as the final label while pinning ``IdentityFinalID``
    to the unknown slot, so the label is descriptive and no consumer keying
    on a resolved identity slot can mistake it for one.

    Only trajectories with no already-resolved identity are touched.
    """
    from hydra_suite.core.individual.identity.heads import identity_axis_columns
    from hydra_suite.core.individual.identity.resolve import excluded_display_labels

    classifiers = params.get("CNN_CLASSIFIERS") or []
    excluded = excluded_display_labels(classifiers)
    if df is None or df.empty or not excluded or "TrajectoryID" not in df.columns:
        return df

    axis_cols = identity_axis_columns(df.columns, classifiers)
    if not axis_cols:
        return df

    out = df.copy()
    if C.FINAL_LABEL not in out.columns:
        out[C.FINAL_LABEL] = pd.Series(
            [np.nan] * len(out), index=out.index, dtype=object
        )
    else:
        out[C.FINAL_LABEL] = out[C.FINAL_LABEL].astype(object)
    if C.FINAL_SOURCE not in out.columns:
        out[C.FINAL_SOURCE] = pd.Series(
            [C.IdentityFinalSource.NONE] * len(out), index=out.index, dtype=object
        )
    if C.FINAL_ID not in out.columns:
        out[C.FINAL_ID] = np.nan
    if C.FINAL_CONFIDENCE not in out.columns:
        out[C.FINAL_CONFIDENCE] = np.nan

    class_cols = [c for c, _ in axis_cols]
    conf_cols = [c for _, c in axis_cols if c in out.columns]

    composite = out[class_cols[0]].astype(str)
    for col in class_cols[1:]:
        composite = composite + "_" + out[col].astype(str)

    unresolved = out[C.FINAL_LABEL].isna() | (
        out[C.FINAL_LABEL].astype(str).str.strip() == ""
    )

    for traj_id, group in out.groupby("TrajectoryID", sort=False):
        if not unresolved.loc[group.index].all():
            continue  # partially or fully resolved: leave it alone
        observed = composite.loc[group.index]
        observed = observed[observed.notna() & (observed != "")]
        if observed.empty:
            continue
        modal = str(observed.mode().iloc[0])
        if modal not in excluded:
            continue
        conf = 0.0
        if conf_cols:
            per_frame = out.loc[group.index, conf_cols].apply(
                pd.to_numeric, errors="coerce"
            )
            # Weakest axis per frame: a composite is only as trustworthy as
            # its least confident tag call.
            conf = float(per_frame.min(axis=1).mean())
        out.loc[group.index, C.FINAL_LABEL] = modal
        out.loc[group.index, C.FINAL_ID] = 0
        out.loc[group.index, C.FINAL_SOURCE] = C.IdentityFinalSource.NON_IDENTIFYING
        out.loc[group.index, C.FINAL_CONFIDENCE] = conf

    return out
```

Call it in `apply_identity_postprocessing_to_df`, immediately before the mirror step (currently line 344), so a stamped label is never overwritten by the realtime/tag mirror (which only fills empty labels):

```python
        with_pose_df = _stamp_non_identifying_labels(with_pose_df, params)
        with_pose_df = _mirror_realtime_and_tag_into_final(with_pose_df)
        with_pose_df = fill_identity_nans_with_consensus(with_pose_df)
        with_pose_df = sort_trajectories_by_identity(with_pose_df)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/identity/test_non_identifying_classes.py tests/test_core_identity_postprocess_df.py tests/identity/test_honesty_fix.py tests/identity/test_identity_columns.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
make commit-prep && make lint-moderate
git add src/hydra_suite/core/individual/identity/columns.py src/hydra_suite/core/individual/identity/heads.py src/hydra_suite/core/individual/postprocess_df.py tests/identity/test_non_identifying_classes.py
git commit -m "feat(identity): report non-identifying composites with IdentityFinalID=0

Untagged animals read as notag_notag rather than unknown, while the ID stays
the unknown slot so nothing mistakes a shared label for a resolved identity."
```

---

### Task 9: Keep non-identifying values out of the relink identity key

**Files:**
- Modify: `src/hydra_suite/core/post/identity_postprocess.py` (`derive_unique_identity_key_series`)
- Modify: `src/hydra_suite/core/individual/postprocess_df.py` (pass the excluded classes)
- Test: `tests/identity/test_non_identifying_classes.py` (append)

**Interfaces:**
- Consumes: `resolve.non_identifying_marks` (Task 7), the `identity_heads` parameter (Task 2).
- Produces: `derive_unique_identity_key_series(df, identity_heads=None, non_identifying_values=())`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/identity/test_non_identifying_classes.py
from hydra_suite.core.post.identity_postprocess import (
    derive_unique_identity_key_series,
    identity_sources_conflict,
    parse_identity_key,
)


def test_notag_is_not_evidence_of_agreement():
    df = pd.DataFrame(
        {
            "CNN_colortag_front_Class": ["notag", "notag"],
            "CNN_colortag_front_Conf": [0.9, 0.9],
            "CNN_colortag_back_Class": ["notag", "notag"],
            "CNN_colortag_back_Conf": [0.9, 0.9],
        }
    )
    keys = derive_unique_identity_key_series(
        df, identity_heads=("colortag",), non_identifying_values=("notag",)
    )
    # No evidence at all -> NaN, so two untagged fragments neither agree nor
    # conflict; the spatial gates alone decide whether they relink.
    assert keys.isna().all()


def test_real_class_survives_alongside_a_notag_axis():
    df = pd.DataFrame(
        {
            "CNN_colortag_front_Class": ["red"],
            "CNN_colortag_front_Conf": [0.9],
            "CNN_colortag_back_Class": ["notag"],
            "CNN_colortag_back_Conf": [0.9],
        }
    )
    keys = derive_unique_identity_key_series(
        df, identity_heads=("colortag",), non_identifying_values=("notag",)
    )
    assert parse_identity_key(keys.iloc[0]) == {"cnn:colortag:front": "red"}


def test_two_untagged_fragments_do_not_conflict():
    lhs = {"cnn:colortag:front": "notag"}
    rhs = {"cnn:colortag:front": "notag"}
    # Sanity: with the values present they'd count as agreement; the fix is
    # that they never reach the comparison at all.
    assert not identity_sources_conflict(lhs, rhs)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/identity/test_non_identifying_classes.py -v -k "notag_is_not_evidence or survives_alongside"`
Expected: FAIL — `TypeError: ... unexpected keyword argument 'non_identifying_values'`

- [ ] **Step 3: Write minimal implementation**

In `identity_postprocess.py`, extend `_cnn_identity_sources_for_row` and the public function:

```python
def _cnn_identity_sources_for_row(
    row: "pd.Series", cnn_class_columns: list, non_identifying_values=()
) -> dict:
```

and inside its loop, right after `value = _normalize_string(row.get(col))`:

```python
        if not value or value in non_identifying_values:
            # A non-identifying class carries no identity information. Left in,
            # `notag == notag` would count as AGREEMENT in
            # `_compare_identity_sources`' grouped tally and could out-vote a
            # genuine conflict on another axis.
            continue
```

(replacing the existing `if not value: continue`).

Then in `derive_unique_identity_key_series`:

```python
def derive_unique_identity_key_series(
    df: pd.DataFrame, identity_heads=None, non_identifying_values=()
) -> pd.Series:
```

and in `_row_key`:

```python
        sources.update(
            _cnn_identity_sources_for_row(
                row, cnn_class_columns, frozenset(non_identifying_values)
            )
        )
```

In `postprocess_df.py`, extend the call added in Task 2:

```python
        from hydra_suite.core.individual.identity.resolve import non_identifying_marks

        _marks = non_identifying_marks(params.get("CNN_CLASSIFIERS") or [])
        _bare_marks = frozenset(
            m.partition(":")[2] or m
            for marks in _marks.values()
            for m in marks
            if "_" not in m or ":" in m
        )
        with_pose_df[C.UNIQUE_IDENTITY_KEY] = derive_unique_identity_key_series(
            with_pose_df,
            identity_heads=None if _heads is HEADS_UNKNOWN else _heads,
            non_identifying_values=_bare_marks,
        )
```

(Whole-composite marks like `notag_notag` name a *joined* label, not a per-axis class value, so they are excluded from `_bare_marks` — per-axis filtering only applies to bare and axis-scoped marks.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/identity/test_non_identifying_classes.py tests/test_unique_identity_key_derivation.py tests/test_identity_conflict_resolution.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
make commit-prep && make lint-moderate
git add src/hydra_suite/core/post/identity_postprocess.py src/hydra_suite/core/individual/postprocess_df.py tests/identity/test_non_identifying_classes.py
git commit -m "fix(identity): drop non-identifying classes from the relink identity key"
```

---

### Task 10: GUI and config plumbing

**Files:**
- Create: `src/hydra_suite/trackerkit/gui/dialogs/non_identifying_classes_dialog.py`
- Modify: `src/hydra_suite/trackerkit/gui/panels/identity_panel.py:995-1050`
- Modify: `src/hydra_suite/trackerkit/config/identity_schema.py:75`
- Test: `tests/test_identity_panel_phase6.py` (append), `tests/identity/test_identity_config_schema.py` (append)

**Interfaces:**
- Consumes: nothing from Core.
- Produces: `CNNClassifierRow.to_config()` gains `"factor_names": list[str]` and `"non_identifying_classes": list[str]`; `IdentityModelConfig.non_identifying_classes: tuple[str, ...]`.

`CNN_CLASSIFIERS` engine params are the row config dicts passed verbatim (`engine_params.py:626-631`), so no `engine_params.py` change is needed — the new keys flow through automatically.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/identity/test_identity_config_schema.py
from hydra_suite.trackerkit.config.identity_schema import IdentityConfig, IdentityModelConfig


def test_non_identifying_classes_round_trip():
    cfg = IdentityConfig(
        models=[
            IdentityModelConfig(
                kind="cnn",
                name="colortag",
                unique_identifier=True,
                non_identifying_classes=("notag", "front:notag"),
            )
        ]
    )
    restored = IdentityConfig.from_dict(cfg.to_dict())
    assert restored.models[0].non_identifying_classes == ("notag", "front:notag")


def test_non_identifying_classes_defaults_empty():
    assert IdentityModelConfig().non_identifying_classes == ()
```

```python
# append to tests/test_identity_panel_phase6.py
# NOTE: this file already does `pytest.importorskip("PySide6")` and defines
# module-scoped `qapp` / `main_window` fixtures. Rows are created via
# `panel._add_cnn_classifier_row()` and removed via
# `panel._remove_cnn_classifier_row(row)` in a finally block — follow that
# pattern exactly; there is no per-row fixture.
def test_row_config_round_trips_non_identifying_classes(main_window):
    panel = main_window._identity_panel
    row = panel._add_cnn_classifier_row()
    other = panel._add_cnn_classifier_row()
    try:
        row.chk_unique_identifier.setChecked(True)
        row._non_identifying_classes = ["front:notag", "notag_notag"]
        cfg = row.to_config()
        assert cfg["non_identifying_classes"] == ["front:notag", "notag_notag"]

        other.load_from_config(cfg)
        assert other._non_identifying_classes == ["front:notag", "notag_notag"]
    finally:
        panel._remove_cnn_classifier_row(row)
        panel._remove_cnn_classifier_row(other)


def test_non_identifying_button_follows_unique_identifier(main_window):
    panel = main_window._identity_panel
    row = panel._add_cnn_classifier_row()
    try:
        row.chk_unique_identifier.setChecked(False)
        assert not row.btn_non_identifying.isEnabled()
        row.chk_unique_identifier.setChecked(True)
        assert row.btn_non_identifying.isEnabled()
    finally:
        panel._remove_cnn_classifier_row(row)
```

`to_config()` reads `meta` from the selected model; with no model selected it returns early. If the assertion above trips on that path, select a model first the way `test_cnn_row_exposes_calibration_affordance` handles the no-model case, or assert against `row._non_identifying_classes` round-tripping through `load_from_config` alone.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/identity/test_identity_config_schema.py tests/test_identity_panel_phase6.py -v -k non_identifying`
Expected: FAIL — `TypeError: IdentityModelConfig.__init__() got an unexpected keyword argument 'non_identifying_classes'`

- [ ] **Step 3: Write minimal implementation**

In `identity_schema.py`, add to `IdentityModelConfig`:

```python
    non_identifying_classes: tuple[str, ...] = ()
    """Classes/composites this model declares non-identifying.

    Forms: ``"notag"`` (that class in any of this model's axes),
    ``"front:notag"`` (that class in the named factor), ``"notag_notag"``
    (that whole composite display label). Excluded from the identity
    catalog entirely -- see the design doc.
    """
```

`from_dict` already does `IdentityModelConfig(**dict(m))`; ensure the tuple survives by normalizing in `__post_init__`:

```python
    def __post_init__(self) -> None:
        self.non_identifying_classes = tuple(
            str(c) for c in (self.non_identifying_classes or ())
        )
```

Create the dialog:

```python
# src/hydra_suite/trackerkit/gui/dialogs/non_identifying_classes_dialog.py
"""Mark a classifier's classes as non-identifying.

A non-identifying class (an untagged animal's ``notag``) names an animal
that carries no unique identity. Marked classes are excluded from the
identity catalog entirely, so any number of animals may carry them
simultaneously without competing for one slot.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QGroupBox,
    QLabel,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)

from hydra_suite.widgets.dialogs import BaseDialog


class NonIdentifyingClassesDialog(BaseDialog):
    """Per-factor class checkboxes plus a free-text composite field.

    ``BaseDialog.__init__(title, parent=None, ...)`` builds the Ok/Cancel
    button box itself; subclasses insert their UI above it with
    ``add_content(widget)``.
    """

    def __init__(self, parent, factor_names, class_names_per_factor, selected):
        super().__init__("Non-identifying classes", parent)
        self._checks: list[tuple[str, str, QCheckBox]] = []
        selected = list(selected or [])

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.addWidget(
            QLabel(
                "Classes that do not identify an individual (e.g. 'notag').\n"
                "Excluded from the identity catalog: any number of animals may\n"
                "carry them at once, and they are never merged or swapped."
            )
        )

        for idx, classes in enumerate(class_names_per_factor or []):
            if not classes:
                continue
            factor = (
                str(factor_names[idx])
                if idx < len(factor_names or [])
                else f"factor{idx}"
            )
            box = QGroupBox(factor)
            box_layout = QVBoxLayout()
            for cls in classes:
                cls = str(cls)
                chk = QCheckBox(cls)
                chk.setChecked(cls in selected or f"{factor}:{cls}" in selected)
                box_layout.addWidget(chk)
                self._checks.append((factor, cls, chk))
            box.setLayout(box_layout)
            layout.addWidget(box)

        layout.addWidget(QLabel("Whole composites (comma-separated, e.g. notag_notag):"))
        self._composites = QLineEdit(
            ", ".join(s for s in selected if "_" in s and ":" not in s)
        )
        layout.addWidget(self._composites)

        self.add_content(container)

    def selected_marks(self) -> list[str]:
        """Checked classes as ``factor:class``, plus any composite entries."""
        marks = [f"{f}:{c}" for f, c, chk in self._checks if chk.isChecked()]
        marks.extend(
            part.strip()
            for part in self._composites.text().split(",")
            if part.strip()
        )
        return marks
```

In `identity_panel.py`'s `CNNClassifierRow.__init__`, next to where
`chk_unique_identifier` is created (around line 832) and added to the form
(line 847):

```python
            self._non_identifying_classes: list[str] = []
            self.btn_non_identifying = QPushButton("Non-identifying classes…")
            self.btn_non_identifying.setToolTip(
                "Classes that do not identify an individual (e.g. 'notag').\n"
                "Excluded from the identity catalog: any number of animals may\n"
                "carry them at once without competing for one identity slot."
            )
            self.btn_non_identifying.setEnabled(
                self.chk_unique_identifier.isChecked()
            )
            self.btn_non_identifying.clicked.connect(self._edit_non_identifying)
            form.addRow("", self.btn_non_identifying)
            self.chk_unique_identifier.toggled.connect(
                self.btn_non_identifying.setEnabled
            )
```

and the handler as a method on the row:

```python
        def _edit_non_identifying(self) -> None:
            """Open the class-marking dialog for the currently selected model."""
            from hydra_suite.trackerkit.gui.dialogs.non_identifying_classes_dialog import (
                NonIdentifyingClassesDialog,
            )

            rel_path = self.combo_model.currentData()
            if not self._has_selected_model(rel_path):
                return
            meta = self._main_window._identity_panel._cnn_registry_entry(rel_path)
            dlg = NonIdentifyingClassesDialog(
                self,
                meta.get("factor_names") or [],
                meta.get("class_names_per_factor") or [],
                self._non_identifying_classes,
            )
            if dlg.exec():
                self._non_identifying_classes = dlg.selected_marks()
```

`QPushButton` is already imported in `identity_panel.py`; verify before adding it
to the import list. Then extend `to_config()`:

```python
                "unique_identifier": self.chk_unique_identifier.isChecked(),
                "factor_names": [str(f) for f in (meta.get("factor_names") or [])],
                "non_identifying_classes": list(self._non_identifying_classes),
```

and `load_from_config()`:

```python
            self._non_identifying_classes = [
                str(c) for c in (cfg.get("non_identifying_classes") or [])
            ]
```

Emitting `factor_names` also fixes the axis names: without it, `resolve.identity_axes` falls back to positional `factor0`/`factor1`, so the axis-scoped `front:notag` mark form would never match.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/identity/test_identity_config_schema.py tests/test_identity_panel_phase6.py tests/test_trackerkit_cli_config_identity_params.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
make commit-prep && make lint-moderate
git add src/hydra_suite/trackerkit/gui/dialogs/non_identifying_classes_dialog.py src/hydra_suite/trackerkit/gui/panels/identity_panel.py src/hydra_suite/trackerkit/config/identity_schema.py tests/identity/test_identity_config_schema.py tests/test_identity_panel_phase6.py
git commit -m "feat(trackerkit): mark classifier classes as non-identifying in the identity panel"
```

---

### Task 11: End-to-end coexistence proof, docs, and the final gate

**Files:**
- Test: `tests/identity/test_non_identifying_classes.py` (append)
- Modify: `docs/user-guide/` identity page (find it with `grep -rln "unique identifier" docs/`)
- Modify: `CHANGELOG.md` if the repo has one at root

- [ ] **Step 1: Write the failing end-to-end test**

```python
# append to tests/identity/test_non_identifying_classes.py
from hydra_suite.core.individual.identity.catalog import IdentityCatalog
from hydra_suite.core.individual.identity import substrate


def test_many_untagged_slots_coexist_in_one_hungarian_solve():
    """The whole point: N untagged animals must not compete for one slot.

    They are absent from the catalog, so every visible slot's posterior mass
    sits on unknown and the solver assigns none of them a known identity --
    rather than handing one of them 'notag_notag' and pushing the rest onto
    wrong real identities.
    """
    spec = resolve_catalog_spec([_tags(["notag"])], [])
    catalog = IdentityCatalog.from_spec(spec)
    assert catalog.labels == ("unknown", "red_blue")

    # Five untagged slots: all mass on unknown.
    posteriors = [np.array([0.95, 0.05]) for _ in range(5)]
    assignment = substrate.solve_unique_assignment(
        posteriors, catalog.num_known, display_threshold=0.6
    )
    assert assignment == [None] * 5


def test_untagged_slots_do_not_displace_a_real_identity():
    spec = resolve_catalog_spec([_tags(["notag"])], [])
    catalog = IdentityCatalog.from_spec(spec)
    posteriors = [
        np.array([0.05, 0.95]),  # a genuinely tagged animal
        np.array([0.95, 0.05]),  # untagged
        np.array([0.95, 0.05]),  # untagged
    ]
    assignment = substrate.solve_unique_assignment(
        posteriors, catalog.num_known, display_threshold=0.6
    )
    assert assignment == [1, None, None]
```

- [ ] **Step 2: Run test to verify it fails or passes**

Run: `python -m pytest tests/identity/test_non_identifying_classes.py -v -k coexist or displace`
Expected: PASS immediately (Task 7 already delivers this behavior). This test is the *regression lock* on the design's central claim — if it fails, the exclusion is not actually reaching the solver.

- [ ] **Step 3: Update the docs**

Find the identity documentation: `grep -rln "unique identifier\|unique_identifier" docs/`. Add a section covering:
- Only classifiers marked **Unique identifier** influence identity; all others are exported as columns and influence nothing.
- Multiple unique-identifier classifiers **combine** (thorax colour × abdomen shape = one composite identity), they do not compete.
- **Non-identifying classes**: what they are, the three mark forms, and the consequence — those animals are tracked normally and labelled (`notag_notag`) but receive no identity resolution, no identity-based fragment stitching across occlusions, and `IdentityFinalID` stays 0.

Add to the changelog (or `docs/superpowers/specs/done/` notes if there is no root `CHANGELOG.md`):
- **Breaking:** `UniqueIdentityKey` now contains identity heads only. Downstream parsers of that column see fewer sources.
- **Breaking:** configurations with two or more unique-identifier classifiers now produce a cross-product catalog instead of a union. Prior results for such configs were incorrect.

- [ ] **Step 4: Run the full identity test suite**

```bash
python -m pytest tests/identity/ -v
python -m pytest tests/test_identity_postprocess.py tests/test_core_identity_postprocess_df.py \
  tests/test_unique_identity_key_derivation.py tests/test_identity_conflict_resolution.py \
  tests/test_identity_online.py tests/test_postproc_identity_gating.py \
  tests/test_postproc_merge_stage_identity.py tests/test_tag_identity.py -v
```

Expected: PASS. Compare failures against the known pre-existing baseline (`main` carries ~24 pre-existing failures suite-wide, including 4 pose/SLEAP cutover tests that fail on `main` but skip in fresh worktrees) — use a delta gate, not an absolute one.

- [ ] **Step 5: Run the final equivalence gate on both platforms**

MPS (this box) and CUDA (mehek), per Global Constraints. Expected: **byte-identical on every clip** — the feature is opt-in and no fixture declares `non_identifying_classes`, so "off" must be a provable no-op. Verify CSV row counts > 1 before trusting any EQUIVALENT verdict.

- [ ] **Step 6: Commit**

```bash
make commit-prep && make lint-moderate && make docs-check
git add tests/identity/test_non_identifying_classes.py docs/
git commit -m "docs(identity): identity heads, composite catalogs, non-identifying classes

Includes the regression lock proving N untagged slots coexist in one
Hungarian solve without displacing a real identity."
```

---

## Post-implementation

Use `superpowers:finishing-a-development-branch` to decide integration. The three slices are separately revertible; if the equivalence gate flags a diff, bisect by slice — each has its own gate checkpoint (Tasks 3, 6, 11).
