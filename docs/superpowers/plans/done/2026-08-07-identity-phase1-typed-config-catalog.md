# Identity Overhaul — Phase 1: Typed Config + Persisted Catalog Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Introduce a typed `IdentityConfig` schema and a single, structured, round-trippable `IdentityCatalogSpec`, then route the tracking worker's catalog assembly and `build_engine_params`' identity keys through them — with **zero behavior change**.

**Architecture:** Today the identity domain is assembled inline inside the 4300-line tracking worker as a list of `"_"`-joined composite strings, and ~15 flat `IDENTITY_*` engine-params are emitted as loose `_cfg_get(...)` reads inside `build_engine_params`. This phase adds (a) a structured `IdentityCatalogSpec` that retains each composite's `(factor, class)` provenance and derives a backward-compatible display label, resolved once by a pure `resolve_catalog_spec(...)` function; and (b) a typed `IdentityConfig` dataclass from which `build_engine_params` derives the same flat keys. The worker consumes `IdentityCatalog.from_spec(...)`; the split-based *consumers* of the `"_"`-join encoding are intentionally left untouched (they are deleted in Phases 3/5). This is the foundation the evidence stage (Phase 3), substrate (Phase 4), and post-hoc smoother (Phase 5) build on.

**Tech Stack:** Python 3, `dataclasses`, NumPy, pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-07-22-identity-overhaul-consolidated-design.md` — Layer 0 (Config), Layer 1 (Catalog), Rollout "Phase 1".

## Global Constraints

- **Zero behavior change.** This is a refactor. The phase-end gate is the equivalence harness producing **byte-identical tracking positions** vs baseline on MPS **and** CUDA (identity columns are additive; positions must not move). See `tools/equivalence/README.md` and CLAUDE.md.
- **Isolation.** Do all work in a git worktree branched from local HEAD: `git worktree add .worktrees/identity-phase1 -b feat/identity-phase1 HEAD`. Local `main` is ahead of `origin/main`; never branch from origin.
- **Dependency direction.** `core/individual/*` (catalog, resolver) must **not** import from any app-layer package (`trackerkit`, etc.). The typed `IdentityConfig` lives in the app layer (`trackerkit/config/`) and may import from `core`, never the reverse.
- **Structured factor keys.** A composite identity is a tuple of `(factor_name, class_name)` pairs. Never use `"_".join(...)` / `label.split("_")` to *encode or decode* identity semantics. The `"_"`-join survives **only** as a derived, backward-compatible *display string* (so existing CSV labels and the runtime `IdentityCatalog.labels` are byte-identical to today).
- **Commit as the configured git user.** Do **not** add a `Co-Authored-By: Claude` trailer (see memory `feedback_git_commit_identity`).
- **Before commit:** `make format` then `make lint-moderate`. Kill stale `sleap`/`hydra` processes before any heavy run; never touch non-sleap/hydra processes.
- **Verification:** unit tests on `hydra-mps` (this box); the equivalence gate on `hydra-mps` here and `hydra-cuda` on mehek.

---

## File Structure

**Create:**
- `src/hydra_suite/core/individual/identity/spec.py` — `CatalogEntry`, `IdentityCatalogSpec` (structured, serializable). Lives in `core` because both the worker (app-adjacent core) and later the inference stage (core) consume it.
- `src/hydra_suite/core/individual/identity/resolve.py` — `resolve_catalog_spec(cnn_classifiers, tag_identity_labels) -> IdentityCatalogSpec`, the single catalog-resolution function (ports the worker's inline logic, structured).
- `src/hydra_suite/trackerkit/config/identity_schema.py` — the typed `IdentityConfig` and its sub-dataclasses (app layer).
- `tests/identity/test_catalog_spec.py`
- `tests/identity/test_resolve_catalog_spec.py`
- `tests/identity/test_identity_config_schema.py`

**Modify:**
- `src/hydra_suite/core/individual/identity/catalog.py` — add `IdentityCatalog.from_spec(...)` (small).
- `src/hydra_suite/core/individual/identity/__init__.py` — export the new names.
- `src/hydra_suite/core/tracking/worker.py:1844-1908` — replace the inline catalog assembly with `resolve_catalog_spec(...)` + `IdentityCatalog.from_spec(...)`.
- `src/hydra_suite/trackerkit/engine_params.py:604-635, 1132-1184` — derive the scalar `IDENTITY_*` keys from an `IdentityConfig`.
- `tests/test_get_parameters_dict_characterization.py` — extend the committed golden to lock the identity keys byte-identical.

**Explicitly NOT touched in Phase 1 (deferred, with reason):**
- `worker.py:3130-3197` (top-1 reconstruction + `split("_")` landmine) — the whole path is **deleted** in Phase 3 (evidence stage supplies true per-factor softmax). Touching it now is wasted work.
- `core/individual/identity/offline.py:81-107` (`split("_")` CSV reconstruction) and `core/individual/postprocess_df.py:74-107` (offline label build) — the offline path is repointed at the evidence cache in Phase 5. Unifying it onto the resolver now would shift offline behavior and break the no-change gate.
- `core/tracking/identity/evidence_emitter.py` — deleted in Phase 3.
- On-disk persistence of the spec into the saved config JSON — the spec is made *round-trippable* here; it is *persisted into the inference pass* in Phase 3 (which is where a before-inference catalog is actually needed).

---

## Interfaces (defined once, referenced by every task)

```python
# core/individual/identity/spec.py
@dataclass(frozen=True)
class CatalogEntry:
    display_label: str                    # backward-compatible "_"-join (or the bare label)
    factors: tuple[tuple[str, str], ...]  # ((factor_name, class_name), ...); () for a tag / flat label
    source: str                           # "cnn" | "tag"

@dataclass(frozen=True)
class IdentityCatalogSpec:
    entries: tuple[CatalogEntry, ...]     # ordered known identities; the unknown slot is implicit
    @property
    def labels(self) -> tuple[str, ...]: ...          # (e.display_label for e in entries)
    def to_dict(self) -> dict[str, object]: ...
    @classmethod
    def from_dict(cls, data: dict) -> "IdentityCatalogSpec": ...

# core/individual/identity/catalog.py  (added)
@staticmethod
def from_spec(spec: IdentityCatalogSpec) -> "IdentityCatalog": ...  # labels=(UNKNOWN, *spec.labels)

# core/individual/identity/resolve.py
def resolve_catalog_spec(
    cnn_classifiers: Sequence[Mapping[str, object]],
    tag_identity_labels: Sequence[object],
) -> IdentityCatalogSpec: ...

# trackerkit/config/identity_schema.py
@dataclass class SlotLockConfig
@dataclass class RealtimeIdentityConfig
@dataclass class PostHocIdentityConfig
@dataclass class RobustnessConfig
@dataclass class IdentityModelConfig
@dataclass class IdentityConfig:
    @classmethod
    def from_engine_config(cls, cfg, advanced, *, cfg_get) -> "IdentityConfig": ...
```

---

## Task 1: Structured `IdentityCatalogSpec` + `IdentityCatalog.from_spec`

**Files:**
- Create: `src/hydra_suite/core/individual/identity/spec.py`
- Modify: `src/hydra_suite/core/individual/identity/catalog.py` (add `from_spec`)
- Modify: `src/hydra_suite/core/individual/identity/__init__.py` (exports)
- Test: `tests/identity/test_catalog_spec.py`

**Interfaces:**
- Produces: `CatalogEntry`, `IdentityCatalogSpec` (with `.labels`, `.to_dict`, `.from_dict`), `IdentityCatalog.from_spec`. Consumed by Tasks 2, 3.
- Consumes: `IdentityCatalog` / `UNKNOWN_LABEL` from `catalog.py`.

- [ ] **Step 1: Write the failing test**

```python
# tests/identity/test_catalog_spec.py
import pytest
from hydra_suite.core.individual.identity.spec import CatalogEntry, IdentityCatalogSpec
from hydra_suite.core.individual.identity.catalog import IdentityCatalog, UNKNOWN_LABEL


def _spec():
    return IdentityCatalogSpec(entries=(
        CatalogEntry(display_label="red_big", factors=(("color", "red"), ("size", "big")), source="cnn"),
        CatalogEntry(display_label="blue_small", factors=(("color", "blue"), ("size", "small")), source="cnn"),
        CatalogEntry(display_label="ant7", factors=(), source="tag"),
    ))


def test_labels_are_display_labels_in_order():
    assert _spec().labels == ("red_big", "blue_small", "ant7")


def test_roundtrip_preserves_structure():
    spec = _spec()
    assert IdentityCatalogSpec.from_dict(spec.to_dict()) == spec


def test_underscore_in_class_name_survives_structurally():
    # A class name containing "_" would be mis-split by the legacy split("_") path.
    # The structured factors must round-trip exactly regardless of the display string.
    spec = IdentityCatalogSpec(entries=(
        CatalogEntry(display_label="dark_red_x_1", factors=(("color", "dark_red"), ("id", "x_1")), source="cnn"),
    ))
    restored = IdentityCatalogSpec.from_dict(spec.to_dict())
    assert restored.entries[0].factors == (("color", "dark_red"), ("id", "x_1"))


def test_from_spec_matches_from_labels():
    spec = _spec()
    cat = IdentityCatalog.from_spec(spec)
    assert cat.labels == (UNKNOWN_LABEL, "red_big", "blue_small", "ant7")
    assert cat.labels == IdentityCatalog.from_labels(list(spec.labels)).labels


def test_from_spec_empty_raises():
    with pytest.raises(ValueError):
        IdentityCatalog.from_spec(IdentityCatalogSpec(entries=()))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/identity/test_catalog_spec.py -v`
Expected: FAIL with `ModuleNotFoundError: hydra_suite.core.individual.identity.spec`.

- [ ] **Step 3: Write `spec.py`**

```python
# src/hydra_suite/core/individual/identity/spec.py
"""Structured, serializable identity catalog domain.

The persisted, ordered set of known identities for a run. Each entry retains
its ``(factor, class)`` provenance so composite identities never need to be
decoded by splitting a "_"-joined string (the legacy landmine). The joined
``display_label`` is kept only as a backward-compatible presentation string.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CatalogEntry:
    """One known identity and its structured factor provenance.

    Parameters
    ----------
    display_label:
        Backward-compatible presentation string. For a CNN composite this is
        ``"_".join(class for _, class in factors if class)``; for a tag/flat
        label it is the bare label.
    factors:
        Ordered ``(factor_name, class_name)`` pairs. Empty for a tag / flat
        single-label identity.
    source:
        ``"cnn"`` or ``"tag"``.
    """

    display_label: str
    factors: tuple[tuple[str, str], ...]
    source: str


@dataclass(frozen=True)
class IdentityCatalogSpec:
    """Ordered domain of known identities (the unknown slot is implicit at 0)."""

    entries: tuple[CatalogEntry, ...]

    @property
    def labels(self) -> tuple[str, ...]:
        """Display labels in registration order (excluding the unknown slot)."""
        return tuple(e.display_label for e in self.entries)

    def to_dict(self) -> dict[str, object]:
        return {
            "entries": [
                {
                    "display_label": e.display_label,
                    "factors": [list(pair) for pair in e.factors],
                    "source": e.source,
                }
                for e in self.entries
            ]
        }

    @classmethod
    def from_dict(cls, data: dict) -> "IdentityCatalogSpec":
        entries = tuple(
            CatalogEntry(
                display_label=str(d["display_label"]),
                factors=tuple((str(f[0]), str(f[1])) for f in d.get("factors", [])),
                source=str(d.get("source", "cnn")),
            )
            for d in data.get("entries", [])
        )
        return cls(entries=entries)
```

- [ ] **Step 4: Add `from_spec` to `catalog.py`**

Add to `IdentityCatalog` (after `from_labels`, `catalog.py:65`):

```python
    @staticmethod
    def from_spec(spec: "IdentityCatalogSpec") -> "IdentityCatalog":
        """Rebuild the frozen runtime catalog from a persisted spec.

        The runtime ``labels`` are byte-identical to ``from_labels`` applied to
        the spec's display labels, so every downstream consumer is unaffected.
        """
        return IdentityCatalog.from_labels(list(spec.labels))
```

Add the import at the top of `catalog.py` (under `TYPE_CHECKING` to avoid any import cycle):

```python
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from hydra_suite.core.individual.identity.spec import IdentityCatalogSpec
```

- [ ] **Step 5: Export from `identity/__init__.py`**

Append to `src/hydra_suite/core/individual/identity/__init__.py`:

```python
from hydra_suite.core.individual.identity.catalog import IdentityCatalog, UNKNOWN_LABEL
from hydra_suite.core.individual.identity.spec import CatalogEntry, IdentityCatalogSpec

__all__ = ["IdentityCatalog", "UNKNOWN_LABEL", "CatalogEntry", "IdentityCatalogSpec"]
```

- [ ] **Step 6: Run test to verify it passes**

Run: `python -m pytest tests/identity/test_catalog_spec.py -v`
Expected: PASS (5 tests).

- [ ] **Step 7: Format, lint, commit**

```bash
make format && make lint-moderate
git add src/hydra_suite/core/individual/identity/spec.py \
        src/hydra_suite/core/individual/identity/catalog.py \
        src/hydra_suite/core/individual/identity/__init__.py \
        tests/identity/test_catalog_spec.py
git commit -m "feat(identity): structured IdentityCatalogSpec + IdentityCatalog.from_spec"
```

---

## Task 2: `resolve_catalog_spec` — the single catalog resolver

Ports the worker's inline assembly (`worker.py:1844-1905`) into one pure, structured function. The **display label** for a composite is `"_".join(class for _, class in factors if class)` — identical to the current `"_".join(str(c) for c in _combo if c)` — so the resulting `IdentityCatalog.labels` are byte-identical. The difference is that the structured `factors` are now retained.

**Files:**
- Create: `src/hydra_suite/core/individual/identity/resolve.py`
- Test: `tests/identity/test_resolve_catalog_spec.py`

**Interfaces:**
- Consumes: `CatalogEntry`, `IdentityCatalogSpec` (Task 1).
- Produces: `resolve_catalog_spec(cnn_classifiers, tag_identity_labels) -> IdentityCatalogSpec`. Consumed by Task 3.

- [ ] **Step 1: Write the failing test** (characterization vs the legacy inline oracle)

```python
# tests/identity/test_resolve_catalog_spec.py
import itertools
from hydra_suite.core.individual.identity.resolve import resolve_catalog_spec


def _legacy_labels(cnn_classifiers, tag_labels):
    """Verbatim port of worker.py:1844-1905 — the oracle we must match."""
    known: list[str] = []
    for cfg in cnn_classifiers:
        if not bool(cfg.get("unique_identifier", False)):
            continue
        cnpf = cfg.get("class_names_per_factor") or []
        non_empty = [fl for fl in cnpf if fl]
        if len(non_empty) > 1:
            for combo in itertools.product(*non_empty):
                comp = "_".join(str(c) for c in combo if c)
                if comp and comp not in known:
                    known.append(comp)
        else:
            flat: list[str] = []
            for fl in non_empty:
                flat.extend([str(x) for x in fl if x])
            if not flat:
                flat = [str(x) for x in (cfg.get("labels", []) or []) if x]
            for lbl in flat:
                if lbl and lbl not in known:
                    known.append(lbl)
    cnn_derived = set(known)
    for lbl in tag_labels:
        s = str(lbl).strip() if lbl else ""
        if not s:
            continue
        if cnn_derived and s not in cnn_derived:
            continue
        if s not in known:
            known.append(s)
    return known


CASES = [
    # multi-factor composite
    ([{"unique_identifier": True,
       "class_names_per_factor": [["red", "blue"], ["big", "small"]]}], []),
    # single factor
    ([{"unique_identifier": True, "class_names_per_factor": [["a", "b", "c"]]}], []),
    # flat labels fallback
    ([{"unique_identifier": True, "labels": ["x", "y"]}], []),
    # non-unique classifier ignored
    ([{"unique_identifier": False, "class_names_per_factor": [["p", "q"]]}], []),
    # tag labels, filtered to CNN-derived set
    ([{"unique_identifier": True, "class_names_per_factor": [["red", "blue"]]}],
     ["red", "phaseA", "blue"]),
    # tag-only (no CNN): all tags accepted
    ([], ["ant1", "ant2", "", "ant1"]),
]


def test_labels_match_legacy_oracle():
    for cnn, tags in CASES:
        spec = resolve_catalog_spec(cnn, tags)
        assert list(spec.labels) == _legacy_labels(cnn, tags), (cnn, tags)


def test_structured_factors_captured_for_composite():
    spec = resolve_catalog_spec(
        [{"unique_identifier": True,
          "class_names_per_factor": [["red", "blue"], ["big", "small"]]}], [])
    first = next(e for e in spec.entries if e.display_label == "red_big")
    assert first.factors == (("factor0", "red"), ("factor1", "big"))
    assert first.source == "cnn"


def test_tag_entry_has_empty_factors():
    spec = resolve_catalog_spec([], ["ant1"])
    assert spec.entries[0].factors == ()
    assert spec.entries[0].source == "tag"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/identity/test_resolve_catalog_spec.py -v`
Expected: FAIL with `ModuleNotFoundError: ...identity.resolve`.

- [ ] **Step 3: Write `resolve.py`**

Port the worker logic, building structured `CatalogEntry` objects. Factor names come from the classifier's `factor_names` when present, else positional `factor{i}` (Phase 3 will thread real factor names through the evidence stage; the positional fallback keeps display labels and ordering identical today).

```python
# src/hydra_suite/core/individual/identity/resolve.py
"""The single identity-catalog resolver.

Given the configured CNN classifiers and tag labels, produce the ordered,
structured :class:`IdentityCatalogSpec`. This is the one place the identity
domain is assembled; the tracking worker consumes it via
``IdentityCatalog.from_spec``. Ported behavior-for-behavior from the former
inline assembly in ``core/tracking/worker.py`` (composite display labels are
"_"-joined exactly as before), with the addition of structured factor keys.
"""

from __future__ import annotations

import itertools
import json
import os
from typing import Mapping, Sequence

from hydra_suite.core.individual.identity.spec import CatalogEntry, IdentityCatalogSpec


def _read_factors_from_model_file(model_path: str) -> list[list[str]]:
    """Fallback: read per-factor class names from a model manifest JSON.

    Mirrors worker.py:1852-1872 — prefer ``class_names_per_factor``, then a
    flat ``class_names``, then ``factor_models[].class_names``.
    """
    try:
        if not model_path or not os.path.exists(model_path):
            return []
        with open(model_path) as fh:
            meta = json.load(fh)
    except Exception:
        return []
    cnpf = meta.get("class_names_per_factor") or []
    if cnpf:
        return cnpf
    flat = meta.get("class_names") or []
    if flat:
        return [flat]
    out: list[list[str]] = []
    for fe in meta.get("factor_models") or []:
        fl = fe.get("class_names") or []
        if fl:
            out.append(fl)
    return out


def resolve_catalog_spec(
    cnn_classifiers: Sequence[Mapping[str, object]],
    tag_identity_labels: Sequence[object],
) -> IdentityCatalogSpec:
    """Resolve the ordered identity domain from CNN + tag configuration."""
    entries: list[CatalogEntry] = []
    seen: set[str] = set()

    def _add(display: str, factors: tuple[tuple[str, str], ...], source: str) -> None:
        if display and display not in seen:
            seen.add(display)
            entries.append(CatalogEntry(display_label=display, factors=factors, source=source))

    for cfg in cnn_classifiers or []:
        if not bool(cfg.get("unique_identifier", False)):
            continue
        cnpf = list(cfg.get("class_names_per_factor") or [])
        if not cnpf:
            cnpf = _read_factors_from_model_file(str(cfg.get("model_path", "")))
        factor_names = list(cfg.get("factor_names") or [])
        non_empty = [fl for fl in cnpf if fl]

        if len(non_empty) > 1:
            for combo in itertools.product(*non_empty):
                pairs = tuple(
                    (factor_names[i] if i < len(factor_names) else f"factor{i}", str(c))
                    for i, c in enumerate(combo)
                    if c
                )
                display = "_".join(str(c) for c in combo if c)
                _add(display, pairs, "cnn")
        else:
            flat: list[str] = []
            for fl in non_empty:
                flat.extend([str(x) for x in fl if x])
            if not flat:
                flat = [str(x) for x in (cfg.get("labels", []) or []) if x]
            fname = factor_names[0] if factor_names else "factor0"
            for lbl in flat:
                _add(lbl, ((fname, lbl),), "cnn")

    cnn_derived = set(seen)
    for lbl in tag_identity_labels or []:
        s = str(lbl).strip() if lbl else ""
        if not s:
            continue
        if cnn_derived and s not in cnn_derived:
            continue
        _add(s, (), "tag")

    return IdentityCatalogSpec(entries=tuple(entries))
```

Note: when a tag label duplicates a CNN-derived display label, the legacy code skips it via `if s not in known` (the entry already exists, keeps its CNN provenance) — `_add`'s `seen` guard reproduces that exactly.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/identity/test_resolve_catalog_spec.py -v`
Expected: PASS (3 tests, all `CASES` matching the oracle).

- [ ] **Step 5: Format, lint, commit**

```bash
make format && make lint-moderate
git add src/hydra_suite/core/individual/identity/resolve.py \
        tests/identity/test_resolve_catalog_spec.py
git commit -m "feat(identity): single structured resolve_catalog_spec resolver"
```

---

## Task 3: Route the worker through `resolve_catalog_spec` + `from_spec`

Replace the inline assembly at `worker.py:1844-1908` with two calls. The runtime `_identity_catalog.labels` must be byte-identical, so the online decoder, uniqueness constraint, and CSV labels are unchanged.

**Files:**
- Modify: `src/hydra_suite/core/tracking/worker.py:1837-1920` (imports + assembly block)

**Interfaces:**
- Consumes: `resolve_catalog_spec` (Task 2), `IdentityCatalog.from_spec` (Task 1).
- Produces: no new interface; behavior-preserving edit.

- [ ] **Step 1: Replace the assembly block**

In `worker.py`, replace the body from the `import itertools as _itertools` / imports (`:1837-1842`) through the `if _known_labels_set:` construction (`:1907-1920`) with:

```python
                from hydra_suite.core.individual.identity.catalog import IdentityCatalog
                from hydra_suite.core.individual.identity.online import (
                    OnlineIdentityDecoder,
                )
                from hydra_suite.core.individual.identity.resolve import (
                    resolve_catalog_spec,
                )

                _catalog_spec = resolve_catalog_spec(
                    p.get("CNN_CLASSIFIERS", []),
                    p.get("TAG_IDENTITY_LABELS", []),
                )
                if _catalog_spec.entries:
                    _identity_catalog = IdentityCatalog.from_spec(_catalog_spec)
                    _identity_online_decoder = OnlineIdentityDecoder(
                        _identity_catalog, p
                    )
                    logger.info(
                        "Identity online decoder enabled: catalog size=%d labels=%s",
                        _identity_catalog.size,
                        _identity_catalog.labels,
                    )
                else:
                    logger.info(
                        "Identity online decoder: no known labels configured; decoder disabled."
                    )
```

Leave the surrounding `try/except` (`:1831`, `:1921-1926`), the gate at `:1830`, and the `_identity_online_decoder = None` / `_identity_catalog = None` initializers (`:1826-1828`) intact. The now-unused `import os`/`import json` inside this block are removed with the deleted code (they remain imported at module top for other uses — verify with a grep before deleting any module-level import).

- [ ] **Step 2: Verify no other code in the block references the deleted locals**

Run: `grep -n "_known_labels_set\|_cnpf_cfg\|_non_empty\|_cnn_derived\|_composite" src/hydra_suite/core/tracking/worker.py`
Expected: **only** hits inside the deleted range (now gone) or unrelated names. If any survive outside `:1844-1908`, stop and reconcile — they indicate a hidden dependency.

- [ ] **Step 3: Run the worker-adjacent test suite**

Run: `python -m pytest tests/test_get_parameters_dict_characterization.py tests/identity/ -v`
Expected: PASS. (The resolver parity from Task 2 is the unit-level guard; the equivalence gate below is the behavioral guard.)

- [ ] **Step 4: Equivalence smoke (fastest clip) — byte-identical positions**

Kill stale sleap/hydra first, then run the two fastest identity-free clips as a fast attribution check:

```bash
pkill -f 'sleap|hydra' 2>/dev/null; sleep 1
conda activate hydra-mps
git worktree add --detach .worktrees/equiv-legacy legacy/main 2>/dev/null || true
REPO=$PWD WT=$PWD MAIN_SRC=$PWD/.worktrees/equiv-legacy/src WT_SRC=$PWD/src \
  OUT=/tmp/equiv_p1 RUNTIME=mps \
  bash tools/equivalence/run_matrix.sh fly_obb worm_bgsub
```

Expected: EQUIVALENCE at/near the DETERMINISM floor (positions p99 ≈ 0, θ max ≈ 0, identical row counts) for both clips. **Verify CSV row counts > 1** before trusting an EQUIVALENT (empty CSVs falsely pass — see CLAUDE.md gotchas).

> The full identity-bearing clips (`ant_cnn_identity`, `emi_obb_identity`) run in the phase-end gate (after Task 5). This smoke only confirms the worker edit didn't disturb geometry.

- [ ] **Step 5: Format, lint, commit**

```bash
make format && make lint-moderate
git add src/hydra_suite/core/tracking/worker.py
git commit -m "refactor(identity): worker builds catalog via resolve_catalog_spec + from_spec"
```

---

## Task 4: Typed `IdentityConfig` schema

The typed carrier for all identity state. Phase 1 wires the fields that map to a currently-emitted flat key; fields reserved for later phases (calibration ref, robustness knobs, smoothing/changepoint toggles, the independent post-hoc `enabled`) are present with documented defaults so the type is stable across phases, but are **not** emitted yet.

**Files:**
- Create: `src/hydra_suite/trackerkit/config/identity_schema.py`
- Test: `tests/identity/test_identity_config_schema.py`

**Interfaces:**
- Consumes: nothing from earlier tasks (self-contained app-layer schema).
- Produces: `IdentityConfig.from_engine_config(cfg, advanced, *, cfg_get)` and the sub-dataclasses. Consumed by Task 5.

- [ ] **Step 1: Write the failing test**

```python
# tests/identity/test_identity_config_schema.py
from hydra_suite.trackerkit.config.identity_schema import IdentityConfig


def _get(cfg, key, default=None):
    return cfg.get(key, default)


def test_from_engine_config_maps_scalar_keys():
    cfg = {
        "enable_postprocessing": True,
        "enable_identity_in_tracking": True,
        "enable_identity_online_decoder": True,
        "identity_postprocess_mode": "Fragment Solver",
        "identity_weight": 0.7,
        "identity_commit_threshold": 0.9,
        "identity_display_threshold": 0.55,
        "identity_transition_epsilon": 0.03,
        "identity_unknown_prior": 0.04,
        "identity_rejoin_threshold": 0.6,
        "enable_identity_swap_correction": False,
        "identity_swap_min_frames": 10,
        "identity_disagree_min_run": 7,
        "identity_gates_trajectory_structure": False,
    }
    advanced = {
        "identity_swap_conf_margin": 0.25,
        "identity_rejoin_velocity_budget": 2.0,
        "identity_rejoin_dist_floor": 3.0,
    }
    ic = IdentityConfig.from_engine_config(cfg, advanced, cfg_get=_get)

    assert ic.realtime.enabled is True
    assert ic.realtime.bayesian_cost_enabled is True
    assert ic.realtime.association_weight == 0.7
    assert ic.realtime.commit_threshold == 0.9
    assert ic.realtime.display_threshold == 0.55
    assert ic.realtime.transition_epsilon == 0.03
    assert ic.realtime.unknown_prior == 0.04
    assert ic.realtime.rejoin_threshold == 0.6
    assert ic.realtime.swap_enabled is False
    assert ic.realtime.slot_lock.swap_min_frames == 10
    assert ic.realtime.slot_lock.swap_conf_margin == 0.25
    assert ic.realtime.slot_lock.rejoin_velocity_budget == 2.0
    assert ic.realtime.slot_lock.rejoin_dist_floor == 3.0
    assert ic.posthoc.postprocess_mode == "Fragment Solver"
    assert ic.posthoc.fragment_solver_enabled is True
    assert ic.posthoc.disagree_min_run == 7
    assert ic.posthoc.gates_trajectory_structure is False


def test_online_decoder_gated_by_master_switch():
    # bridge rule: online decoder ANDs with enable_identity_in_tracking.
    cfg = {"enable_identity_in_tracking": False, "enable_identity_online_decoder": True}
    ic = IdentityConfig.from_engine_config(cfg, {}, cfg_get=_get)
    assert ic.realtime.bayesian_cost_enabled is False


def test_postprocess_mode_falls_back_to_fragment_solver_when_key_absent():
    cfg = {"enable_postprocessing": True}
    ic = IdentityConfig.from_engine_config(cfg, {}, cfg_get=_get)
    assert ic.posthoc.postprocess_mode == "Fragment Solver"
    assert ic.posthoc.fragment_solver_enabled is True


def test_postprocess_mode_none_when_postprocessing_off():
    cfg = {"enable_postprocessing": False, "identity_postprocess_mode": "Fragment Solver"}
    ic = IdentityConfig.from_engine_config(cfg, {}, cfg_get=_get)
    assert ic.posthoc.postprocess_mode == "None"
    assert ic.posthoc.fragment_solver_enabled is False


def test_defaults_match_builder_defaults():
    ic = IdentityConfig.from_engine_config({}, {}, cfg_get=_get)
    assert ic.realtime.enabled is True
    assert ic.realtime.bayesian_cost_enabled is False
    assert ic.realtime.association_weight == 1.0
    assert ic.realtime.commit_threshold == 0.85
    assert ic.realtime.display_threshold == 0.6
    assert ic.realtime.transition_epsilon == 0.02
    assert ic.realtime.unknown_prior == 0.05
    assert ic.realtime.rejoin_threshold == 0.5
    assert ic.realtime.swap_enabled is True
    assert ic.realtime.slot_lock.swap_min_frames == 8
    assert ic.realtime.slot_lock.swap_conf_margin == 0.2
    assert ic.realtime.slot_lock.rejoin_velocity_budget == 1.5
    assert ic.realtime.slot_lock.rejoin_dist_floor is None
    assert ic.posthoc.disagree_min_run == 5
    assert ic.posthoc.gates_trajectory_structure is True


def test_roundtrip():
    ic = IdentityConfig.from_engine_config({}, {}, cfg_get=_get)
    assert IdentityConfig.from_dict(ic.to_dict()) == ic
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/identity/test_identity_config_schema.py -v`
Expected: FAIL with `ModuleNotFoundError: ...trackerkit.config.identity_schema`.

- [ ] **Step 3: Write `identity_schema.py`**

The defaults below are copied verbatim from `engine_params.py:1132-1178` and the gate logic from `engine_params.py:604-635`.

```python
# src/hydra_suite/trackerkit/config/identity_schema.py
"""Typed identity configuration for TrackerKit.

Single source of truth for identity state. ``build_engine_params`` derives the
flat ``IDENTITY_*`` engine params from this object (Phase 1); later phases
convert consumers to read it directly. Fields marked "reserved" are persisted
for round-trip stability but are not emitted into engine params until the phase
that wires them (calibration → Phase 2, robustness → Phase 3, smoothing /
changepoint / independent post-hoc toggle → Phases 5/6).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable


@dataclass
class SlotLockConfig:
    swap_min_frames: int = 8
    swap_conf_margin: float = 0.2
    rejoin_velocity_budget: float = 1.5
    rejoin_dist_floor: float | None = None


@dataclass
class RealtimeIdentityConfig:
    enabled: bool = True                 # master identity-in-tracking gate
    bayesian_cost_enabled: bool = False  # online decoder (ANDs with `enabled`)
    association_weight: float = 1.0
    rejoin_threshold: float = 0.5
    commit_threshold: float = 0.85
    display_threshold: float = 0.6
    transition_epsilon: float = 0.02
    unknown_prior: float = 0.05
    swap_enabled: bool = True
    slot_lock: SlotLockConfig = field(default_factory=SlotLockConfig)


@dataclass
class PostHocIdentityConfig:
    postprocess_mode: str = "Fragment Solver"
    fragment_solver_enabled: bool = False
    disagree_min_run: int = 5
    gates_trajectory_structure: bool = True
    # Reserved (Phases 5/6):
    enabled: bool = False                # independent of realtime; wired in Phase 6
    smoothing_enabled: bool = False
    changepoint_enabled: bool = False
    fragment_min_frames: int = 0
    ambiguity_margin: float = 0.0


@dataclass
class RobustnessConfig:
    # Reserved (Phase 3): no engine key emitted in Phase 1.
    per_frame_evidence_cap: float = 0.0
    prob_floor: float = 0.0
    source_weights: dict[str, float] = field(default_factory=dict)


@dataclass
class IdentityModelConfig:
    kind: str = "cnn"                    # "cnn" | "apriltag" | "color_tag"
    name: str = ""
    path: str | None = None
    unique_identifier: bool = False
    factors: tuple[str, ...] = ()
    # Reserved (Phase 2): fitted temperature + signature.
    calibration: dict[str, Any] | None = None


@dataclass
class IdentityConfig:
    enabled: bool = True                 # master identity classification on
    models: list[IdentityModelConfig] = field(default_factory=list)
    calibration_required: bool = False   # reserved (Phase 2 gate)
    realtime: RealtimeIdentityConfig = field(default_factory=RealtimeIdentityConfig)
    posthoc: PostHocIdentityConfig = field(default_factory=PostHocIdentityConfig)
    robustness: RobustnessConfig = field(default_factory=RobustnessConfig)

    @classmethod
    def from_engine_config(
        cls,
        cfg: Any,
        advanced: Any,
        *,
        cfg_get: Callable[..., Any],
    ) -> "IdentityConfig":
        """Build from the persisted snake_case config, reproducing the exact
        derivations in ``engine_params.py:604-635, 1132-1178``.

        ``cfg_get(cfg, key, default)`` is injected so this stays independent of
        ``engine_params`` internals; the caller passes the module's ``_cfg_get``.
        """
        enable_postprocessing = bool(cfg_get(cfg, "enable_postprocessing", True))
        enable_in_tracking = bool(cfg_get(cfg, "enable_identity_in_tracking", True))
        online = enable_in_tracking and bool(
            cfg_get(cfg, "enable_identity_online_decoder", False)
        )

        saved_mode = cfg_get(cfg, "identity_postprocess_mode", None)
        if saved_mode is None:
            saved_mode = "Fragment Solver"
        saved_mode = str(saved_mode)
        mode = saved_mode if enable_postprocessing else "None"
        fragment_solver = enable_postprocessing and saved_mode == "Fragment Solver"

        realtime = RealtimeIdentityConfig(
            enabled=enable_in_tracking,
            bayesian_cost_enabled=online,
            association_weight=float(cfg_get(cfg, "identity_weight", 1.0)),
            rejoin_threshold=float(cfg_get(cfg, "identity_rejoin_threshold", 0.5)),
            commit_threshold=float(cfg_get(cfg, "identity_commit_threshold", 0.85)),
            display_threshold=float(cfg_get(cfg, "identity_display_threshold", 0.6)),
            transition_epsilon=float(cfg_get(cfg, "identity_transition_epsilon", 0.02)),
            unknown_prior=float(cfg_get(cfg, "identity_unknown_prior", 0.05)),
            swap_enabled=bool(cfg_get(cfg, "enable_identity_swap_correction", True)),
            slot_lock=SlotLockConfig(
                swap_min_frames=int(cfg_get(cfg, "identity_swap_min_frames", 8)),
                swap_conf_margin=float(advanced.get("identity_swap_conf_margin", 0.2)),
                rejoin_velocity_budget=float(
                    advanced.get("identity_rejoin_velocity_budget", 1.5)
                ),
                rejoin_dist_floor=advanced.get("identity_rejoin_dist_floor", None),
            ),
        )
        posthoc = PostHocIdentityConfig(
            postprocess_mode=mode,
            fragment_solver_enabled=fragment_solver,
            disagree_min_run=int(cfg_get(cfg, "identity_disagree_min_run", 5)),
            gates_trajectory_structure=bool(
                cfg_get(cfg, "identity_gates_trajectory_structure", True)
            ),
        )
        return cls(realtime=realtime, posthoc=posthoc)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "IdentityConfig":
        d = dict(data)
        rt = dict(d.get("realtime", {}))
        sl = dict(rt.pop("slot_lock", {}) or {})
        realtime = RealtimeIdentityConfig(slot_lock=SlotLockConfig(**sl), **rt)
        posthoc = PostHocIdentityConfig(**dict(d.get("posthoc", {})))
        robustness = RobustnessConfig(**dict(d.get("robustness", {})))
        models = [IdentityModelConfig(**dict(m)) for m in d.get("models", [])]
        return cls(
            enabled=bool(d.get("enabled", True)),
            models=models,
            calibration_required=bool(d.get("calibration_required", False)),
            realtime=realtime,
            posthoc=posthoc,
            robustness=robustness,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/identity/test_identity_config_schema.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Format, lint, commit**

```bash
make format && make lint-moderate
git add src/hydra_suite/trackerkit/config/identity_schema.py \
        tests/identity/test_identity_config_schema.py
git commit -m "feat(identity): typed IdentityConfig schema + from_engine_config"
```

---

## Task 5: `build_engine_params` derives identity keys from `IdentityConfig`

Emit the ~15 scalar `IDENTITY_*` keys from an `IdentityConfig` built via `from_engine_config`, guarded by the committed characterization golden so the params dict stays byte-identical. The list/method keys (`CNN_CLASSIFIERS`, `USE_APRILTAGS`, `IDENTITY_METHOD`, `APRILTAG_*`, `COLOR_TAG_*`, `ENABLE_IDENTITY_ANALYSIS/…PIPELINE`) are unchanged.

**Files:**
- Modify: `src/hydra_suite/trackerkit/engine_params.py` (construct `identity_cfg`; swap RHS of the scalar identity keys; drop now-orphaned local derivations)
- Modify: `tests/test_get_parameters_dict_characterization.py` (lock the identity keys)

**Interfaces:**
- Consumes: `IdentityConfig.from_engine_config` (Task 4), the module-local `_cfg_get`.
- Produces: no new interface; byte-identical output.

- [ ] **Step 1: Capture the pre-change identity-keys golden**

Before editing `engine_params.py`, snapshot the current identity output for the golden fixtures so the change is provably inert. Inspect `tests/test_get_parameters_dict_characterization.py` for the fixture config(s) it already loads, then add a helper that extracts the identity subset:

```python
# tests/test_get_parameters_dict_characterization.py  (add near the existing fixtures)
IDENTITY_KEYS = [
    "IDENTITY_DISAGREE_MIN_RUN", "IDENTITY_GATES_TRAJECTORY_STRUCTURE",
    "ENABLE_IDENTITY_IN_TRACKING", "ENABLE_IDENTITY_ONLINE_DECODER",
    "IDENTITY_POSTPROCESS_MODE", "ENABLE_IDENTITY_FRAGMENT_SOLVER",
    "ASSOCIATION_IDENTITY_HINT_SCALE", "IDENTITY_COMMIT_THRESHOLD",
    "IDENTITY_DISPLAY_THRESHOLD", "IDENTITY_TRANSITION_EPSILON",
    "IDENTITY_UNKNOWN_PRIOR", "IDENTITY_REJOIN_THRESHOLD", "IDENTITY_SWAP_ENABLED",
    "IDENTITY_SWAP_MIN_FRAMES", "IDENTITY_SWAP_CONF_MARGIN",
    "IDENTITY_REJOIN_VELOCITY_BUDGET", "IDENTITY_REJOIN_DIST_FLOOR",
]

EXPECTED_IDENTITY = {
    # Filled in Step 2 from the pre-change run (committed baseline).
}


def test_identity_keys_byte_identical():
    for name, cfg in CHARACTERIZATION_CONFIGS.items():   # reuse the file's existing fixtures
        params = build_engine_params(cfg, runtime=_runtime_for(cfg), advanced_config={})
        got = {k: params[k] for k in IDENTITY_KEYS}
        assert got == EXPECTED_IDENTITY[name], name
```

(Adapt `CHARACTERIZATION_CONFIGS` / `_runtime_for` to the names the existing test file actually uses — read it first.)

- [ ] **Step 2: Fill `EXPECTED_IDENTITY` from the current build and confirm it passes on unchanged code**

Run a one-off to print the identity subset for each fixture, paste the values into `EXPECTED_IDENTITY`, then:

Run: `python -m pytest tests/test_get_parameters_dict_characterization.py::test_identity_keys_byte_identical -v`
Expected: PASS **against the un-modified `engine_params.py`** (this pins the baseline before refactoring).

- [ ] **Step 3: Refactor `engine_params.py` to emit from `IdentityConfig`**

Add the import at the top of `engine_params.py`:

```python
from hydra_suite.trackerkit.config.identity_schema import IdentityConfig
```

Immediately before the `params = {` literal (after the local derivations around `:717`), construct:

```python
    identity_cfg = IdentityConfig.from_engine_config(cfg, advanced, cfg_get=_cfg_get)
```

Replace the RHS of the scalar identity keys in the `params` literal (`:1132-1178`) with reads off `identity_cfg`:

```python
        "IDENTITY_DISAGREE_MIN_RUN": identity_cfg.posthoc.disagree_min_run,
        "IDENTITY_GATES_TRAJECTORY_STRUCTURE": identity_cfg.posthoc.gates_trajectory_structure,
        # ... ENABLE_IDENTITY_ANALYSIS / PIPELINE / IDENTITY_METHOD / USE_APRILTAGS /
        #     CNN_CLASSIFIERS / CNN_CLASSIFIER_WINDOW unchanged ...
        "ENABLE_IDENTITY_IN_TRACKING": identity_cfg.realtime.enabled,
        "ENABLE_IDENTITY_ONLINE_DECODER": identity_cfg.realtime.bayesian_cost_enabled,
        "IDENTITY_POSTPROCESS_MODE": identity_cfg.posthoc.postprocess_mode,
        "ENABLE_IDENTITY_FRAGMENT_SOLVER": identity_cfg.posthoc.fragment_solver_enabled,
        "ASSOCIATION_IDENTITY_HINT_SCALE": identity_cfg.realtime.association_weight,
        "IDENTITY_COMMIT_THRESHOLD": identity_cfg.realtime.commit_threshold,
        "IDENTITY_DISPLAY_THRESHOLD": identity_cfg.realtime.display_threshold,
        "IDENTITY_TRANSITION_EPSILON": identity_cfg.realtime.transition_epsilon,
        "IDENTITY_UNKNOWN_PRIOR": identity_cfg.realtime.unknown_prior,
        "IDENTITY_REJOIN_THRESHOLD": identity_cfg.realtime.rejoin_threshold,
        "IDENTITY_SWAP_ENABLED": identity_cfg.realtime.swap_enabled,
        "IDENTITY_SWAP_MIN_FRAMES": identity_cfg.realtime.slot_lock.swap_min_frames,
        "IDENTITY_SWAP_CONF_MARGIN": identity_cfg.realtime.slot_lock.swap_conf_margin,
        "IDENTITY_REJOIN_VELOCITY_BUDGET": identity_cfg.realtime.slot_lock.rejoin_velocity_budget,
        "IDENTITY_REJOIN_DIST_FLOOR": identity_cfg.realtime.slot_lock.rejoin_dist_floor,
```

Then delete the now-orphaned local derivations that fed *only* these keys — the block at `:604-635` for `enable_identity_in_tracking`, `enable_identity_online_decoder`, `saved_identity_postprocess_mode`, `identity_postprocess_mode`, `enable_identity_fragment_solver`. **Keep** `enable_postprocessing_flag` (`:604-606`) if any non-identity key still uses it.

- [ ] **Step 4: Verify no orphaned local is referenced elsewhere**

Run: `grep -n "enable_identity_in_tracking\|enable_identity_online_decoder\|saved_identity_postprocess_mode\|identity_postprocess_mode\|enable_identity_fragment_solver" src/hydra_suite/trackerkit/engine_params.py`
Expected: hits only inside `IdentityConfig.from_engine_config` (a different file) — none dangling in `engine_params.py`. If `enable_postprocessing_flag` lost all consumers, remove it too; if it retains non-identity consumers, keep it.

- [ ] **Step 5: Run the golden + equivalence-adjacent param tests**

Run: `python -m pytest tests/test_get_parameters_dict_characterization.py tests/test_gui_cli_param_equivalence.py tests/test_engine_params_extraction.py -v`
Expected: PASS, including `test_identity_keys_byte_identical` — proving the refactor is inert.

- [ ] **Step 6: Format, lint, commit**

```bash
make format && make lint-moderate
git add src/hydra_suite/trackerkit/engine_params.py \
        tests/test_get_parameters_dict_characterization.py
git commit -m "refactor(identity): build_engine_params derives IDENTITY_* keys from IdentityConfig"
```

---

## Phase-End Gate (run once, after Task 5)

- [ ] **Full identity + geometry equivalence — MPS (this box)**

```bash
pkill -f 'sleap|hydra' 2>/dev/null; sleep 1
conda activate hydra-mps
REPO=$PWD WT=$PWD MAIN_SRC=$PWD/.worktrees/equiv-legacy/src WT_SRC=$PWD/src \
  OUT=/tmp/equiv_p1_full RUNTIME=mps \
  bash tools/equivalence/run_matrix.sh
```
Expected: every clip EQUIVALENT at/near its DETERMINISM floor for both `_forward.csv` and `_tracking_final.csv`; identical row counts; known head/tail π-flip noise only. **Confirm CSV row counts > 1.**

- [ ] **Full equivalence — CUDA (mehek)** per CLAUDE.md's "CUDA box (mehek)" recipe against this branch's SHA.

- [ ] **Suite delta gate.** The base suite has ~24 pre-existing failures (memory `project_runtime_gen2_core_done`); use a before/after delta, not an absolute green. Run the identity + config tests green:
```bash
python -m pytest tests/identity/ tests/test_get_parameters_dict_characterization.py \
  tests/test_gui_cli_param_equivalence.py -v
```

- [ ] **Cleanup:** `git worktree remove --force .worktrees/equiv-legacy && git worktree prune`

---

## Self-Review (checked against the spec)

**Spec coverage (Phase 1 line + Layer 0/1):**
- "Introduce `IdentityConfig`" → Task 4. ✅
- "`IdentityCatalogSpec`" → Task 1. ✅
- "structured factor keys" → Tasks 1 (`CatalogEntry.factors`) + 2 (resolver captures them); the `split("_")` *consumers* are deferred to Phases 3/5 with explicit reasons in "File Structure". ✅
- "migrate `get_parameters_dict()` to derive from it" → Task 5 (the real builder is `build_engine_params`, which `get_parameters_dict` delegates to). ✅
- "No behavior change" → characterization golden (Task 5) + equivalence gate. ✅
- "Catalog resolved once, persisted" → resolved once by Task 2's single resolver, consumed by the worker (Task 3); **round-trippable** here (Task 1 `to_dict`/`from_dict`), *persisted into the inference pass* in Phase 3 (noted in File Structure). ✅
- Layer 0 dataclass shape → mirrored in Task 4 (wired fields active; reserved fields carried for type stability). ✅
- Layer 1 "structured factor keys replace `"_"`-joins; class names may contain any character" → Task 1 `test_underscore_in_class_name_survives_structurally`. ✅

**Placeholder scan:** every code step carries real code; no TBD/TODO. ✅

**Type consistency:** `IdentityCatalogSpec.entries` / `.labels` / `CatalogEntry.factors` used identically across Tasks 1–3; `IdentityConfig.realtime.*` / `.posthoc.*` names match between Task 4 definition and Task 5 consumption; `from_spec` signature identical in Tasks 1 and 3. ✅

**Deferred-with-reason (not gaps):** `worker.py:3176` split landmine and `offline.py:96` / `postprocess_df.py` CSV reconstruction are left in place — deleted in Phases 3/5 where their replacement (evidence stage / evidence-cache read) lands. Unifying them now would violate the no-change gate.
