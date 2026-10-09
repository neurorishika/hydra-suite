# SAHI Unification — Slices S1 + S2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the canonical SAHI tiling contract (`TilingSpec`, aliases, resolution) and the unified v3 tiling sidecar (writer + one reader for YOLO and SAM3), each slice merged to local `main` only after an independent adversarial review.

**Architecture:** S1 adds a pure, Qt-free `utils/tiling_spec.py` (contract, per-backend defaults, alias canonicalization, `(value, source)` resolution) with no callers. S2 adds `core/inference/tiling_meta.py` (v3 geometry builders + `read_tiling_meta`), bumps `.slice_meta.json` to schema 3 with `model_family`, switches the YOLO publish writer to v3, and makes SAM3 publish dual-write a `.slice_meta.json` beside its unchanged `.sam3_meta.json`. No inference behavior changes in either slice.

**Tech Stack:** Python 3, numpy, pytest. Conda env `hydra-mps` on this box.

**Spec:** `docs/superpowers/specs/2026-10-09-sahi-unification-design.md`

**Roadmap:** S3 (callers + F1–F4/F6/F8 + equivalence gates), S4 (shared widget + visual verification in both kits), S5 (docs) each get their own plan, written after the previous slice merges so they are grounded in the landed code. Every slice ends with an independent adversarial review gate.

## Global Constraints

- TrackerKit tracking output byte-identical: `SLICE_*` engine keys, `SliceConfig` fields, `_slice_config_hash`, and `tests/data/get_parameters_dict_golden/*.json` must not change.
- `REFERENCE_BODY_SIZE` is never written by any SAHI path.
- Read lenient, write strict: reading any legacy config/sidecar never raises on out-of-range values (clamp + warn); constructing a `TilingSpec` directly with invalid values raises `ValueError`.
- Writers emit canonical names only; legacy names are translated in exactly one place (`canonicalize`).
- `target_sizes` (legacy px) are read, never written; dividing them requires the imgsz they were expressed at (640 for legacy YOLO, `LEGACY_TARGET_SIZE_IMGSZ`).
- Layering: `utils/` imports nothing from `core`, `training`, or app layers; `core/inference/` imports no app layer.
- Commit as the configured git user, with no Co-Authored-By trailer.
- Each slice: worktree from local HEAD (`git worktree add .worktrees/<name> -b <branch> HEAD`), `make format` before commit, adversarial review gate before merge.
- Pin `PYTHONPATH=<worktree>/src` when running tests from a worktree (the env has an editable install of the main checkout).

## Spec deviations (decided while planning, from reading the code)

1. `TilingSpec` lives in a new `utils/tiling_spec.py`, not `utils/slice_geometry.py` (385 lines; the ~500-line rule). `slice_geometry.py` remains the grid module and is not modified.
2. SAM3 publish **dual-writes**: geometry stays in `.sam3_meta.json` (existing readers `sidecar_for` / `geometry_drift` keep working) and is also written to `.slice_meta.json` v3. Removing geometry from `.sam3_meta.json` is deferred to S3 once every reader goes through `read_tiling_meta`.
3. **YOLO v3 is strictly additive over v2** (adversarial review M3): every v2 key is kept verbatim — including `target_sizes` and the manifest's scalar `object_tile_fraction` — and canonical keys are added beside them. This keeps the v2 reader (`_training_values`), the baseline drift guard (`sliced_dataset.py:272`) and the DetectKit calibration grid byte-for-byte unchanged. Spec §4's "never writes `target_sizes`" is deferred to S3, when every reader goes through `read_tiling_meta`. SAM3 v3 (a new file, no legacy readers) is canonical-only and stamps only values present in the build (never defaults).
4. **One operating-scale rule: `np.median`** of the fraction set (adversarial review M2). It is what TrackerKit computes from `target_sizes` and what SAM3's `dataset_build` stamps as `prefill_object_tile_fraction` (`_median`); `_median_scale` (a set member) is used only for `prefill_tile_px` and is not a fraction rule. Stamped `prefill_object_tile_fraction` is always preferred over recomputation.
5. YOLO v3 does not stamp `tile_px_set`: the YOLO builder measures the reference body per frame, so no single set exists. SAM3 stamps its real set.
6. `prefill_object_tile_fraction` maps to the canonical *operating* scale, not to `object_tile_fractions` (spec §3.5 listed it as a fraction alias; it is a median, not a member of the set).
7. `TilingSpec.training_tile_sizes()` (fans out every scale) is named for training only; inference uses `operating_fraction` + `resolve_tile_size` — one scale, per the user's decision.
8. `canonicalize`'s `operating_fraction` matches `_training_values` for every shape a writer in this repo produces; it knowingly differs on hand-made shapes TrackerKit mishandles (bools as numbers, `imgsz` strings like `"640.5"`, prefill-only docs which TrackerKit ignores). Documented, not mirrored.

## Review Focus

1. An older TrackerKit (`slice_meta._training_values`) reading a v3 YOLO sidecar must return exactly (`==`, bit-exact floats) what it returns for the equivalent v2 sidecar — including multi-scale `target_sizes` medians. → Task 5.
2. Legacy `target_sizes` with no `imgsz` are divided by 640 for YOLO only; anything else raises rather than silently rescaling by 1008. → Task 2, Task 6.
3. Out-of-range or retired values in old files (overlap 0.95, fraction 0 or 1.5, `nmm`, unknown geometry mode, NaN) load with a warning and never raise; the same values passed straight to `TilingSpec(...)` raise. → Task 1, Task 2.
4. A SAM3 publish whose `.slice_meta.json` write fails still succeeds; an orphaned-attempt cleanup also removes the attempt's `.slice_meta.json`. → Task 8.
5. User calibration profiles in an existing sidecar survive a YOLO republish that upgrades v2 → v3, unchanged. → Task 7.
6. A publish never fails on a partial manifest (no `overlap`, no `imgsz` — shapes existing tests and older datasets carry); missing values are omitted, never invented. → Task 5, Task 8.

---

## Slice S1 — Canonical contract (`utils/tiling_spec.py`)

Setup (once, before Task 1):

```bash
cd /Users/neurorishika/Projects/Rockefeller/Kronauer/multi-animal-tracker
git worktree add .worktrees/sahi-s1 -b feat/sahi-unify-s1 HEAD
cd .worktrees/sahi-s1
conda activate hydra-mps
export PYTHONPATH=$PWD/src
python -c "import hydra_suite, sys; print(hydra_suite.__file__)"   # must print .worktrees/sahi-s1/src/...
```

### Task 1: `TilingSpec`, backend defaults, operating-scale rule

**Files:**
- Create: `src/hydra_suite/utils/tiling_spec.py`
- Test: `tests/test_tiling_spec.py`

**Interfaces:**
- Consumes: `hydra_suite.utils.slice_geometry.DEFAULT_MIN_AREA_RATIO`, `resolve_scales`, `tile_size_for_mode`.
- Produces:
  - `GEOMETRY_MODES`, `FRAGMENT_POLICIES`, `MERGE_POLICIES`, `MERGE_METRICS`, `OVERLAP_MAX = 0.9`, `OVERLAP_MARGIN = 0.05`, `DEFAULT_OVERLAP = 0.2`, `FRACTION_MIN = 0.01`, `FRACTION_MAX = 0.9`
  - `Backend = Literal["yolo_train", "yolo_infer", "sam3", "sam2"]`
  - `class Sourced(NamedTuple): value: Any; source: str`
  - `@dataclass(frozen=True) class BackendDefaults` and `BACKEND_DEFAULTS: dict[str, BackendDefaults]`
  - `operating_fraction(fractions) -> float | None` (np.median, clamped to [0.01, 0.9])
  - `@dataclass(frozen=True) class TilingSpec` with fields `enabled, geometry_mode, object_tile_fractions, reference_body_px, slice_width, slice_height, overlap, min_area_ratio, fragment_policy, merge_policy, merge_metric, merge_threshold`; methods `TilingSpec.defaults(backend)`, `.operating_fraction()`, `.training_tile_sizes(imgsz)`, `.to_mapping()`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_tiling_spec.py
import math

import pytest

from hydra_suite.utils.slice_geometry import resolve_scales
from hydra_suite.utils.tiling_spec import (
    BACKEND_DEFAULTS,
    TilingSpec,
    operating_fraction,
)


def test_default_spec_is_valid_and_disabled():
    spec = TilingSpec()
    assert spec.enabled is False
    assert spec.geometry_mode == "auto_model"
    assert spec.object_tile_fractions == ()
    assert spec.overlap is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"geometry_mode": "bogus"},
        {"object_tile_fractions": (0.0,)},
        {"object_tile_fractions": (1.5,)},
        {"object_tile_fractions": (math.nan,)},
        {"overlap": 0.95},
        {"overlap": -0.1},
        {"reference_body_px": -1.0},
        {"slice_width": -1},
        {"min_area_ratio": 1.5},
        {"fragment_policy": "keep"},
        {"merge_policy": "nmm"},
        {"merge_metric": "dice"},
        {"merge_threshold": 2.0},
    ],
)
def test_direct_construction_is_strict(kwargs):
    with pytest.raises(ValueError):
        TilingSpec(**kwargs)


def test_fractions_coerced_to_float_tuple():
    spec = TilingSpec(object_tile_fractions=[0.1, 0.2])
    assert spec.object_tile_fractions == (0.1, 0.2)
    assert all(isinstance(f, float) for f in spec.object_tile_fractions)


def test_backend_defaults_table():
    assert BACKEND_DEFAULTS["yolo_train"].object_tile_fractions == (0.05, 0.10, 0.15, 0.20)
    assert BACKEND_DEFAULTS["yolo_infer"].object_tile_fractions == (0.15,)
    assert BACKEND_DEFAULTS["yolo_infer"].geometry_mode == "auto_model"
    assert BACKEND_DEFAULTS["sam3"].object_tile_fractions == (0.055,)
    assert BACKEND_DEFAULTS["sam3"].fragment_policy == "crowd"
    assert BACKEND_DEFAULTS["sam3"].merge_metric == "polygon_iou"
    assert BACKEND_DEFAULTS["sam2"].object_tile_fractions == ()


def test_defaults_constructor_uses_table():
    spec = TilingSpec.defaults("sam3")
    assert spec.object_tile_fractions == (0.055,)
    assert spec.fragment_policy == "crowd"
    assert spec.enabled is False


def test_operating_fraction_is_np_median():
    # TrackerKit today: median(target_sizes)/imgsz -> 80/640 for [32,64,96,128];
    # SAM3 dataset_build stamps prefill_object_tile_fraction with the same rule.
    assert operating_fraction((0.05, 0.10, 0.15, 0.20)) == pytest.approx(0.125)
    assert operating_fraction((0.0275, 0.055)) == pytest.approx(0.04125)
    assert operating_fraction((0.055,)) == 0.055


def test_operating_fraction_clamps_and_handles_empty():
    assert operating_fraction(()) is None
    assert operating_fraction((0.005,)) == 0.01
    assert operating_fraction((0.95,)) == 0.9


def test_training_tile_sizes_delegates_to_resolve_scales():
    spec = TilingSpec(
        enabled=True,
        geometry_mode="auto_object",
        object_tile_fractions=(0.1, 0.2),
        reference_body_px=50.0,
    )
    expected = resolve_scales(
        geometry_mode="auto_object",
        imgsz=640,
        reference_body_px=50.0,
        fractions=(0.1, 0.2),
        object_tile_fraction=0.15000000000000002,
        slice_width=0,
        slice_height=0,
    )
    assert spec.training_tile_sizes(640) == expected == [(500, 500), (250, 250)]


def test_to_mapping_is_canonical_and_json_safe():
    mapping = TilingSpec(object_tile_fractions=(0.1,), overlap=0.2).to_mapping()
    assert mapping["object_tile_fractions"] == [0.1]
    assert mapping["overlap"] == 0.2
    assert set(mapping) == {
        "enabled", "geometry_mode", "object_tile_fractions", "reference_body_px",
        "slice_width", "slice_height", "overlap", "min_area_ratio",
        "fragment_policy", "merge_policy", "merge_metric", "merge_threshold",
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_tiling_spec.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'hydra_suite.utils.tiling_spec'`

- [ ] **Step 3: Implement**

```python
# src/hydra_suite/utils/tiling_spec.py
"""Canonical SAHI tiling contract shared by every kit.

Spec: docs/superpowers/specs/2026-10-09-sahi-unification-design.md (§3).
TrackerKit's vocabulary is canonical. This module is pure (numpy only) so
core, training, data and every kit can import it; ``slice_geometry`` stays
the grid module and is not modified.
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass, replace
from typing import Any, Literal, NamedTuple

import numpy as np

from .slice_geometry import DEFAULT_MIN_AREA_RATIO, resolve_scales

logger = logging.getLogger(__name__)

GEOMETRY_MODES = ("auto_model", "auto_object", "custom")
FRAGMENT_POLICIES = ("drop", "crowd", "mask")
# ``nmm`` is NOT here: it runs the same code path as ``greedy_nmm`` today
# (stages/merge.py), so ``canonicalize`` reads it as ``greedy_nmm``.
MERGE_POLICIES = ("nms", "greedy_nmm")
MERGE_METRICS = ("iou", "ios", "polygon_iou")
OVERLAP_MAX = 0.9  # same ceiling SliceConfig/_slice_config_from_params clamp to
# Derived overlap = max(fraction) + margin. 0.05 reproduces TrackerKit's 0.2
# default at its 0.15 default fraction. Overlap px >= body px is what puts
# every animal whole inside at least one tile (overlap*tile >= frac*tile).
OVERLAP_MARGIN = 0.05
DEFAULT_OVERLAP = 0.2
FRACTION_MIN = 0.01  # tile_size_for_mode's clamp
FRACTION_MAX = 0.9

Backend = Literal["yolo_train", "yolo_infer", "sam3", "sam2"]

_SPEC_FIELDS = (
    "enabled",
    "geometry_mode",
    "object_tile_fractions",
    "reference_body_px",
    "slice_width",
    "slice_height",
    "overlap",
    "min_area_ratio",
    "fragment_policy",
    "merge_policy",
    "merge_metric",
    "merge_threshold",
)


class Sourced(NamedTuple):
    """A resolved value plus where it came from (shown as a UI badge)."""

    value: Any
    source: str  # user | override | profile | stamped | dataset | derived | default


@dataclass(frozen=True)
class BackendDefaults:
    object_tile_fractions: tuple[float, ...]
    geometry_mode: str
    fragment_policy: str
    merge_policy: str
    merge_metric: str
    merge_threshold: float
    min_area_ratio: float


BACKEND_DEFAULTS: dict[str, BackendDefaults] = {
    # Multi-scale robustness set used by headless SliceTrainingConfig today.
    "yolo_train": BackendDefaults(
        (0.05, 0.10, 0.15, 0.20), "auto_object", "drop",
        "greedy_nmm", "ios", 0.5, DEFAULT_MIN_AREA_RATIO,
    ),
    # SliceConfig defaults (TrackerKit is canonical; must not move).
    "yolo_infer": BackendDefaults(
        (0.15,), "auto_model", "drop",
        "greedy_nmm", "ios", 0.5, DEFAULT_MIN_AREA_RATIO,
    ),
    # Sam3LoraParams.object_tile_fraction; SAM3 merge is polygon-IoU NMS after
    # a containment gate (semantic/tiling.py), fragments become is_crowd.
    "sam3": BackendDefaults(
        (0.055,), "auto_object", "crowd",
        "nms", "polygon_iou", 0.5, DEFAULT_MIN_AREA_RATIO,
    ),
    # Stock SAM2: full frame until calibrated (2026-10-03 spec: fractions do
    # not transfer between models). SAM2 owner tiles do not merge.
    "sam2": BackendDefaults(
        (), "auto_object", "drop",
        "nms", "iou", 0.5, DEFAULT_MIN_AREA_RATIO,
    ),
}


def _clamp_fraction(value: float) -> float:
    return max(FRACTION_MIN, min(FRACTION_MAX, float(value)))


def operating_fraction(fractions) -> float | None:
    """The ONE inference scale for a fraction set: np.median, clamped like the planner.

    The same rule TrackerKit applies to median(target_sizes)/imgsz and SAM3's
    dataset_build stamps as prefill_object_tile_fraction.
    """
    values = [float(f) for f in fractions]
    if not values:
        return None
    return _clamp_fraction(float(np.median(np.asarray(values))))


def _finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


@dataclass(frozen=True)
class TilingSpec:
    """One SAHI tiling configuration in canonical (TrackerKit) vocabulary."""

    enabled: bool = False
    geometry_mode: str = "auto_model"
    object_tile_fractions: tuple[float, ...] = ()
    reference_body_px: float = 0.0
    slice_width: int = 0
    slice_height: int = 0
    overlap: float | None = None  # None = unset -> resolve_overlap derives it
    min_area_ratio: float = DEFAULT_MIN_AREA_RATIO
    fragment_policy: str = "drop"
    merge_policy: str = "greedy_nmm"
    merge_metric: str = "ios"
    merge_threshold: float = 0.5

    def __post_init__(self) -> None:
        fractions = tuple(float(f) for f in self.object_tile_fractions)
        object.__setattr__(self, "object_tile_fractions", fractions)
        object.__setattr__(self, "enabled", bool(self.enabled))
        problems: list[str] = []
        if self.geometry_mode not in GEOMETRY_MODES:
            problems.append(f"geometry_mode {self.geometry_mode!r}")
        if any(not (math.isfinite(f) and 0.0 < f <= 1.0) for f in fractions):
            problems.append(f"object_tile_fractions {fractions!r} outside (0, 1]")
        body = _finite(self.reference_body_px)
        if body is None or body < 0:
            problems.append(f"reference_body_px {self.reference_body_px!r}")
        for name in ("slice_width", "slice_height"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= 8192:
                problems.append(f"{name} {value!r} outside [0, 8192]")
        if self.overlap is not None:
            overlap = _finite(self.overlap)
            if overlap is None or not 0.0 <= overlap <= OVERLAP_MAX:
                problems.append(f"overlap {self.overlap!r} outside [0, {OVERLAP_MAX}]")
        area = _finite(self.min_area_ratio)
        if area is None or not 0.0 <= area <= 1.0:
            problems.append(f"min_area_ratio {self.min_area_ratio!r}")
        if self.fragment_policy not in FRAGMENT_POLICIES:
            problems.append(f"fragment_policy {self.fragment_policy!r}")
        if self.merge_policy not in MERGE_POLICIES:
            problems.append(f"merge_policy {self.merge_policy!r}")
        if self.merge_metric not in MERGE_METRICS:
            problems.append(f"merge_metric {self.merge_metric!r}")
        threshold = _finite(self.merge_threshold)
        if threshold is None or not 0.0 <= threshold <= 1.0:
            problems.append(f"merge_threshold {self.merge_threshold!r}")
        if problems:
            raise ValueError("Invalid TilingSpec: " + "; ".join(problems))

    @classmethod
    def defaults(cls, backend: Backend) -> "TilingSpec":
        d = BACKEND_DEFAULTS[backend]
        return cls(
            geometry_mode=d.geometry_mode,
            object_tile_fractions=d.object_tile_fractions,
            min_area_ratio=d.min_area_ratio,
            fragment_policy=d.fragment_policy,
            merge_policy=d.merge_policy,
            merge_metric=d.merge_metric,
            merge_threshold=d.merge_threshold,
        )

    def operating_fraction(self) -> float | None:
        return operating_fraction(self.object_tile_fractions)

    def training_tile_sizes(self, imgsz: int) -> list[tuple[int, int]]:
        """TRAINING fan-out: one tile size per scale. Inference uses ONE scale
        (operating_fraction + resolve_tile_size), never this."""
        scalar = self.operating_fraction() or BACKEND_DEFAULTS["yolo_infer"].object_tile_fractions[0]
        return resolve_scales(
            geometry_mode=self.geometry_mode,
            imgsz=int(imgsz),
            reference_body_px=float(self.reference_body_px),
            fractions=self.object_tile_fractions,
            object_tile_fraction=scalar,
            slice_width=self.slice_width,
            slice_height=self.slice_height,
        )

    def to_mapping(self) -> dict[str, Any]:
        data = asdict(self)
        data["object_tile_fractions"] = list(self.object_tile_fractions)
        return data
```

Note: `test_tile_sizes_delegates_to_resolve_scales` passes `object_tile_fraction=0.15000000000000002` (np.median of 0.1, 0.2) — the scalar only matters when fan-out does not apply, so either value yields the same list here.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_tiling_spec.py -q`
Expected: PASS (all)

- [ ] **Step 5: Commit**

```bash
make format
git add src/hydra_suite/utils/tiling_spec.py tests/test_tiling_spec.py
git commit -m "feat(tiling): canonical TilingSpec contract and per-backend defaults"
```

### Task 2: `canonicalize` + `TilingSpec.from_mapping` (the one alias point)

**Files:**
- Modify: `src/hydra_suite/utils/tiling_spec.py` (append)
- Test: `tests/test_tiling_spec_canonicalize.py`

**Interfaces:**
- Consumes: Task 1 names.
- Produces:
  - `SLICE_ALIASES: dict[str, tuple[str, ...]]` (canonical → accepted keys, canonical first)
  - `canonicalize(mapping, *, legacy_px_imgsz: float | None = None) -> tuple[dict[str, Any], dict[str, Any]]` — returns `(canonical, extras)`. `canonical` holds only keys the input determined, using `_SPEC_FIELDS` names plus the extra key `"operating_fraction"` when the input stamped one (`prefill_object_tile_fraction`) or when it is derivable bit-exactly from `target_sizes` + `imgsz`. `extras` holds every unconsumed key unchanged (including `imgsz`).
  - `TilingSpec.from_mapping(mapping, *, backend=None, legacy_px_imgsz=None) -> TilingSpec` — dataclass defaults (or `defaults(backend)`) overlaid with `canonical`.
  - `TilingSpec.from_canonical(canonical, *, backend=None) -> TilingSpec` — same, from an existing `canonicalize` result.
  - `operating_fraction` in `canonical`, in order: `median(target_sizes)/imgsz` when both present (imgsz must parse to an int ≥ 1); else stamped `prefill_object_tile_fraction`; else the bare `object_tile_fraction` (clamped; unparseable → 0.15) when that key or `target_sizes` is present. Equals `_training_values` for every writer-produced shape (deviation 8).
  - Warnings for clamped/dropped legacy values are logged once per `(field, value)` per process (TrackerKit re-reads sidecars on every refresh).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_tiling_spec_canonicalize.py
import logging

import numpy as np
import pytest

from hydra_suite.utils.tiling_spec import TilingSpec, canonicalize


def test_trackerkit_profile_settings():
    canonical, extras = canonicalize(
        {
            "enabled": True, "geometry_mode": "auto_object", "overlap": 0.3,
            "object_tile_fraction": 0.12, "trained_body_px": 40.0,
            "slice_width": 0, "slice_height": 0, "confidence_threshold": 0.4,
            "merge_policy": "nms", "merge_metric": "iou", "merge_threshold": 0.6,
            "merge_backend": "cv2",
        }
    )
    assert canonical["object_tile_fractions"] == (0.12,)
    assert canonical["reference_body_px"] == 40.0
    assert canonical["overlap"] == 0.3
    assert extras == {"confidence_threshold": 0.4, "merge_backend": "cv2"}


def test_advanced_config_prefixed_keys():
    canonical, _ = canonicalize(
        {"slice_overlap": 0.25, "slice_object_tile_fraction": 0.2,
         "slice_trained_body_px": 33.0, "slice_merge_threshold": 0.4,
         "slice_width": 512}
    )
    assert canonical["overlap"] == 0.25
    assert canonical["object_tile_fractions"] == (0.2,)
    assert canonical["reference_body_px"] == 33.0
    assert canonical["merge_threshold"] == 0.4
    assert canonical["slice_width"] == 512  # slice_width is itself canonical


def test_target_sizes_with_imgsz_is_bit_exact_with_trackerkit():
    geometry = {"target_sizes": [32, 64, 96, 128], "imgsz": 640, "object_tile_fraction": 0.1}
    canonical, extras = canonicalize(geometry)
    assert canonical["object_tile_fractions"] == (0.05, 0.1, 0.15, 0.2)
    # core/inference/slice_meta._training_values computes exactly this:
    assert canonical["operating_fraction"] == max(0.01, min(0.9, float(np.median(np.asarray([32.0, 64.0, 96.0, 128.0]))) / 640))
    assert extras == {"imgsz": 640}
    assert "target_sizes" not in extras


def test_target_sizes_without_imgsz_needs_explicit_denominator():
    with pytest.raises(ValueError, match="imgsz"):
        canonicalize({"target_sizes": [64, 128]})
    canonical, _ = canonicalize({"target_sizes": [64, 128]}, legacy_px_imgsz=640.0)
    assert canonical["object_tile_fractions"] == (0.1, 0.2)
    # Without a stated imgsz TrackerKit ignores target_sizes for the operating
    # value and falls back to object_tile_fraction, else its 0.15 literal.
    assert canonical["operating_fraction"] == 0.15
    canonical, _ = canonicalize({"target_sizes": [64, 128], "object_tile_fraction": 0.12}, legacy_px_imgsz=640.0)
    assert canonical["operating_fraction"] == 0.12


def test_operating_mirrors_legacy_reader_on_bare_scalar_edge_cases():
    # slice_meta._training_values clamps 0 to 0.01 and maps unparseable to 0.15.
    assert canonicalize({"object_tile_fraction": 0})[0]["operating_fraction"] == 0.01
    assert canonicalize({"object_tile_fraction": "x"})[0]["operating_fraction"] == 0.15
    assert canonicalize({"object_tile_fraction": 2.0})[0]["operating_fraction"] == 0.9


def test_fraction_set_precedence():
    canonical, _ = canonicalize(
        {"object_tile_fractions": [0.03, 0.06], "target_size_fractions": [0.5],
         "target_sizes": [64], "object_tile_fraction": 0.2, "imgsz": 640}
    )
    assert canonical["object_tile_fractions"] == (0.03, 0.06)


def test_empty_fraction_set_falls_through():
    canonical, _ = canonicalize({"target_size_fractions": [], "object_tile_fraction": 0.1})
    assert canonical["object_tile_fractions"] == (0.1,)


def test_sam3_sidecar_multiscale():
    canonical, extras = canonicalize(
        {"object_tile_fractions": [0.0275, 0.055], "prefill_object_tile_fraction": 0.055,
         "train_tile_px_set": [[1940, 1940], [970, 970]], "reference_body_px": 53.4,
         "imgsz": 1008}
    )
    assert canonical["object_tile_fractions"] == (0.0275, 0.055)
    assert canonical["operating_fraction"] == 0.055
    assert "train_tile_px_set" in extras


def test_sam3_build_manifest_names():
    canonical, _ = canonicalize({"tile_overlap": 0.25, "min_retained_area_frac": 0.3})
    assert canonical["overlap"] == 0.25
    assert canonical["min_area_ratio"] == 0.3


def test_escalation_request_names():
    canonical, _ = canonicalize({"tile_fraction": 0.05, "overlap": 0.5, "merge_iou": 0.4})
    assert canonical["object_tile_fractions"] == (0.05,)
    assert canonical["merge_threshold"] == 0.4
    assert canonical["merge_metric"] == "polygon_iou"


@pytest.mark.parametrize("value", [None, 0, 0.0])
def test_escalation_full_frame_means_disabled(value):
    canonical, _ = canonicalize({"tile_fraction": value})
    assert canonical["enabled"] is False
    assert "object_tile_fractions" not in canonical


def test_explicit_merge_metric_beats_merge_iou_implication():
    canonical, _ = canonicalize({"merge_iou": 0.4, "merge_metric": "iou"})
    assert canonical["merge_metric"] == "iou"


def test_lenient_read_clamps_and_warns(caplog):
    caplog.set_level(logging.WARNING)
    canonical, _ = canonicalize(
        {"overlap": 0.95, "geometry_mode": "bogus", "merge_policy": "nmm",
         "object_tile_fraction": 1.5, "reference_body_px": float("nan"),
         "min_area_ratio": 2.0, "merge_threshold": -1}
    )
    assert canonical["overlap"] == 0.9
    assert "geometry_mode" not in canonical
    assert canonical["merge_policy"] == "greedy_nmm"
    assert "object_tile_fractions" not in canonical
    assert "reference_body_px" not in canonical
    assert canonical["min_area_ratio"] == 1.0
    assert canonical["merge_threshold"] == 0.0
    assert "overlap" in caplog.text and "geometry_mode" in caplog.text


def test_overlap_axes_disagreement_takes_width(caplog):
    caplog.set_level(logging.WARNING)
    canonical, _ = canonicalize({"overlap_width_ratio": 0.2, "overlap_height_ratio": 0.3})
    assert canonical["overlap"] == 0.2
    assert "overlap_height_ratio" in caplog.text


def test_from_mapping_lenient_never_raises_on_legacy_values():
    spec = TilingSpec.from_mapping({"overlap": 0.95, "merge_policy": "nmm"}, backend="yolo_infer")
    assert spec.overlap == 0.9
    assert spec.merge_policy == "greedy_nmm"
    assert spec.object_tile_fractions == (0.15,)  # backend default filled


def test_from_mapping_without_backend_does_not_invent_fractions():
    spec = TilingSpec.from_mapping({"geometry_mode": "auto_model"})
    assert spec.object_tile_fractions == ()


@pytest.mark.parametrize("imgsz", [0.5, float("inf"), float("nan"), "x", -640, True])
def test_unusable_imgsz_never_divides(imgsz):
    """Adversarial M4: no ZeroDivision/Overflow on hostile imgsz."""
    canonical, _ = canonicalize({"target_sizes": [64], "imgsz": imgsz}, legacy_px_imgsz=640.0)
    assert canonical["object_tile_fractions"] == (0.1,)


@pytest.mark.parametrize("raw,expected", [("false", False), ("0", False), ("true", True), (0, False), (1, True), (True, True)])
def test_enabled_parses_strings(raw, expected):
    assert canonicalize({"enabled": raw})[0]["enabled"] is expected


def test_repeated_legacy_warning_logged_once(caplog):
    caplog.set_level(logging.WARNING)
    for _ in range(5):
        canonicalize({"overlap": 0.97})
    assert caplog.text.count("overlap=0.97") == 1


def test_none_and_empty_mapping():
    assert canonicalize(None) == ({}, {})
    assert canonicalize({}) == ({}, {})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_tiling_spec_canonicalize.py -q`
Expected: FAIL — `ImportError: cannot import name 'canonicalize'`

- [ ] **Step 3: Implement (append to `utils/tiling_spec.py`)**

```python
# Canonical name -> accepted keys in precedence order (canonical first).
# The ONLY place legacy SAHI names are translated (spec §3.5).
SLICE_ALIASES: dict[str, tuple[str, ...]] = {
    "enabled": ("enabled", "slice_enabled"),
    "geometry_mode": ("geometry_mode", "slice_geometry_mode"),
    "reference_body_px": (
        "reference_body_px",
        "trained_body_px",
        "slice_trained_body_px",
        "measured_reference_body_px",
    ),
    "slice_width": ("slice_width",),
    "slice_height": ("slice_height",),
    "overlap": (
        "overlap",
        "slice_overlap",
        "tile_overlap",
        "overlap_width_ratio",
        "overlap_height_ratio",
    ),
    "min_area_ratio": ("min_area_ratio", "min_retained_area_frac"),
    "fragment_policy": ("fragment_policy",),
    "merge_policy": ("merge_policy", "slice_merge_policy"),
    "merge_metric": ("merge_metric", "slice_merge_metric"),
    "merge_threshold": ("merge_threshold", "slice_merge_threshold", "merge_iou"),
}
_FRACTION_SET_KEYS = ("object_tile_fractions", "target_size_fractions")
_FRACTION_SCALAR_KEYS = (
    "object_tile_fraction",
    "slice_object_tile_fraction",
    "prefill_object_tile_fraction",
    "tile_fraction",
)
_LEGACY_PX_KEY = "target_sizes"
# slice_meta._training_values' literal fallback; mirrored for bit-exact reads.
_LEGACY_READER_DEFAULT_FRACTION = 0.15


def _as_list(raw: Any) -> list[Any]:
    if raw is None or isinstance(raw, (str, bytes)):
        return []
    if isinstance(raw, (list, tuple)):
        return list(raw)
    return [raw]


def _fraction_list(raw: Any) -> list[float]:
    out: list[float] = []
    for item in _as_list(raw):
        value = _finite(item)
        if value is not None and 0.0 < value <= 1.0:
            out.append(value)
    return out


_WARNED: set[tuple[str, str]] = set()


def _warn_once(name: str, value: Any, message: str, *args: Any) -> None:
    """Legacy-value warnings fire once per (field, value) per process."""
    key = (name, repr(value))
    if key in _WARNED:
        return
    _WARNED.add(key)
    logger.warning(message, *args)


def _clamped(name: str, value: float, lo: float, hi: float) -> float:
    if value < lo or value > hi:
        clamped = max(lo, min(hi, value))
        _warn_once(name, value, "SAHI %s=%r is outside [%s, %s]; using %s", name, value, lo, hi, clamped)
        return clamped
    return value


def _parse_bool(raw: Any) -> bool:
    if isinstance(raw, str):
        return raw.strip().lower() not in ("", "0", "false", "no", "off")
    return bool(raw)


def _positive_int(raw: Any) -> int:
    """An image size usable as a denominator, else 0."""
    value = _finite(raw)
    if value is None or value < 1:
        return 0
    return min(8192, int(value))


def _normalize(name: str, raw: Any) -> Any:
    """Lenient per-field read. Returns _DROP when the value is unusable."""
    if name == "enabled":
        return _parse_bool(raw)
    if name in ("geometry_mode", "fragment_policy", "merge_metric", "merge_policy"):
        value = str(raw)
        if name == "merge_policy" and value == "nmm":
            return "greedy_nmm"
        allowed = {
            "geometry_mode": GEOMETRY_MODES,
            "fragment_policy": FRAGMENT_POLICIES,
            "merge_metric": MERGE_METRICS,
            "merge_policy": MERGE_POLICIES,
        }[name]
        if value not in allowed:
            _warn_once(name, value, "SAHI %s=%r is not one of %s; ignoring it", name, value, allowed)
            return _DROP
        return value
    number = _finite(raw)
    if number is None:
        _warn_once(name, raw, "SAHI %s=%r is not a finite number; ignoring it", name, raw)
        return _DROP
    if name in ("slice_width", "slice_height"):
        return int(_clamped(name, float(int(number)), 0, 8192))
    if name == "reference_body_px":
        if number < 0:
            _warn_once(name, number, "SAHI reference_body_px=%r is negative; ignoring it", number)
            return _DROP
        return number
    if name == "overlap":
        return _clamped(name, number, 0.0, OVERLAP_MAX)
    return _clamped(name, number, 0.0, 1.0)  # min_area_ratio, merge_threshold


_DROP = object()


def canonicalize(
    mapping: Any, *, legacy_px_imgsz: float | None = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Translate any SAHI mapping (config, project, plan, sidecar) to canonical keys.

    Read-lenient: out-of-range values are clamped or dropped with a warning,
    never raised. Returns ``(canonical, extras)``; ``extras`` keeps every key
    this function did not consume, unchanged.
    """
    src = dict(mapping or {})
    canonical: dict[str, Any] = {}
    consumed: set[str] = set()

    for name, aliases in SLICE_ALIASES.items():
        present = [key for key in aliases if key in src and src[key] is not None]
        consumed.update(key for key in aliases if key in src)
        if not present:
            continue
        chosen = present[0]
        value = _normalize(name, src[chosen])
        if value is _DROP:
            continue
        canonical[name] = value
        if name == "overlap" and {"overlap_width_ratio", "overlap_height_ratio"} <= set(present):
            if src["overlap_width_ratio"] != src["overlap_height_ratio"]:
                _warn_once(
                    "overlap_axes",
                    (src["overlap_width_ratio"], src["overlap_height_ratio"]),
                    "SAHI overlap_width_ratio=%r and overlap_height_ratio=%r differ; "
                    "the canonical single overlap uses the width ratio",
                    src["overlap_width_ratio"],
                    src["overlap_height_ratio"],
                )
        if chosen == "merge_iou" and not any(
            key in src for key in SLICE_ALIASES["merge_metric"]
        ):
            canonical["merge_metric"] = "polygon_iou"

    fractions: list[float] = []
    for key in _FRACTION_SET_KEYS:
        consumed.add(key)
        fractions = _fraction_list(src.get(key))
        if fractions:
            break

    consumed.add(_LEGACY_PX_KEY)
    raw_targets = [_finite(t) for t in _as_list(src.get(_LEGACY_PX_KEY))]
    raw_targets = [t for t in raw_targets if t is not None]
    stated_imgsz = _positive_int(src.get("imgsz"))
    if not fractions and any(t > 0 for t in raw_targets):
        denominator = stated_imgsz or _finite(legacy_px_imgsz)
        if not denominator or denominator <= 0:
            raise ValueError(
                "target_sizes are pixels at a model input size; the mapping has no "
                "imgsz, so pass legacy_px_imgsz (640 for legacy YOLO) explicitly"
            )
        fractions = [t / denominator for t in raw_targets if 0.0 < t / denominator <= 1.0]
    # operating_fraction mirrors core/inference/slice_meta._training_values
    # bit-for-bit (what TrackerKit serves today), then a stamped prefill.
    if raw_targets and stated_imgsz:
        canonical["operating_fraction"] = _clamp_fraction(
            float(np.median(np.asarray(raw_targets))) / stated_imgsz
        )

    for key in _FRACTION_SCALAR_KEYS:
        consumed.add(key)
    if not fractions:
        for key in _FRACTION_SCALAR_KEYS:
            if key not in src:
                continue
            if key == "tile_fraction":
                value = _finite(src[key])
                if value is None or value <= 0:
                    canonical.setdefault("enabled", False)
                    break
            got = _fraction_list(src[key])
            if got:
                fractions = got[:1]
                break
            if src[key] is not None:
                _warn_once(key, src[key], "SAHI %s=%r is outside (0, 1]; ignoring it", key, src[key])
    if fractions:
        canonical["object_tile_fractions"] = tuple(fractions)

    prefill = _finite(src.get("prefill_object_tile_fraction"))
    if "operating_fraction" not in canonical and prefill is not None and prefill > 0:
        canonical["operating_fraction"] = _clamp_fraction(prefill)
    if "operating_fraction" not in canonical and (
        "object_tile_fraction" in src or raw_targets
    ):
        bare = _finite(src.get("object_tile_fraction"))
        canonical["operating_fraction"] = (
            _LEGACY_READER_DEFAULT_FRACTION if bare is None else _clamp_fraction(bare)
        )

    extras = {key: value for key, value in src.items() if key not in consumed}
    return canonical, extras
```

Append to the `TilingSpec` class body:

```python
    @classmethod
    def from_mapping(
        cls,
        mapping: Any,
        *,
        backend: Backend | None = None,
        legacy_px_imgsz: float | None = None,
    ) -> "TilingSpec":
        """Read-lenient constructor: defaults (or a backend's) overlaid with ``canonicalize``."""
        canonical, _ = canonicalize(mapping, legacy_px_imgsz=legacy_px_imgsz)
        return cls.from_canonical(canonical, backend=backend)

    @classmethod
    def from_canonical(
        cls, canonical: dict[str, Any], *, backend: Backend | None = None
    ) -> "TilingSpec":
        """Build from ``canonicalize`` output (avoids canonicalizing twice)."""
        base = cls.defaults(backend) if backend else cls()
        return replace(base, **{k: v for k, v in canonical.items() if k in _SPEC_FIELDS})
```

Note: `from_mapping` references `canonicalize` defined later in the module; that is fine at call time. Keep `_DROP` defined before first call (module import completes before any call).

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_tiling_spec.py tests/test_tiling_spec_canonicalize.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
make format
git add src/hydra_suite/utils/tiling_spec.py tests/test_tiling_spec_canonicalize.py
git commit -m "feat(tiling): canonicalize legacy SAHI names in one alias map"
```

### Task 3: Resolution functions returning `(value, source)`

**Files:**
- Modify: `src/hydra_suite/utils/tiling_spec.py` (append)
- Test: `tests/test_tiling_resolution.py`

**Interfaces:**
- Consumes: Task 1–2 names; `slice_geometry.tile_size_for_mode`.
- Produces:
  - `resolve_reference_body_px(*, override=None, dataset_median=None, stamped=None, tracker_reference=None) -> Sourced` (float px; sources `override` / `dataset` / `stamped` / `user`; `Sourced(0.0, "default")` when nothing positive)
  - `resolve_operating_fraction(*, backend, profile=None, stamped_operating=None, stamped_fractions=()) -> Sourced` (float | None; sources `profile` / `stamped` / `default`)
  - `resolve_object_tile_fractions(*, backend, user=None, profile=None, stamped=()) -> Sourced` (tuple; sources `user` / `profile` / `stamped` / `default`) — the fraction SET for training (spec §3.3)
  - `resolve_overlap(*, override=None, saved=None, fractions=()) -> Sourced` (sources `override` / `user` / `derived` / `default`)
  - `resolve_tile_size(spec: TilingSpec, *, imgsz: int, fraction: float | None) -> Sourced` (`(w, h)`; source `user` in custom mode with an explicit size, else `derived`)

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_tiling_resolution.py
import pytest

from hydra_suite.utils.slice_geometry import tile_size_for_mode
from hydra_suite.utils.tiling_spec import (
    DEFAULT_OVERLAP,
    Sourced,
    TilingSpec,
    resolve_object_tile_fractions,
    resolve_operating_fraction,
    resolve_overlap,
    resolve_reference_body_px,
    resolve_tile_size,
)


def test_body_px_priority_chain():
    assert resolve_reference_body_px(override=50, dataset_median=40, stamped=30, tracker_reference=20) == Sourced(50.0, "override")
    assert resolve_reference_body_px(dataset_median=40, stamped=30, tracker_reference=20) == Sourced(40.0, "dataset")
    assert resolve_reference_body_px(stamped=30, tracker_reference=20) == Sourced(30.0, "stamped")
    assert resolve_reference_body_px(tracker_reference=20) == Sourced(20.0, "user")
    assert resolve_reference_body_px() == Sourced(0.0, "default")


@pytest.mark.parametrize("bad", [0, -5, float("nan"), "x", None, True])
def test_body_px_skips_unusable_values(bad):
    assert resolve_reference_body_px(override=bad, stamped=30) == Sourced(30.0, "stamped")


def test_operating_fraction_chain():
    assert resolve_operating_fraction(backend="yolo_infer", profile=0.12, stamped_operating=0.2) == Sourced(0.12, "profile")
    assert resolve_operating_fraction(backend="yolo_infer", stamped_operating=0.2) == Sourced(0.2, "stamped")
    assert resolve_operating_fraction(backend="yolo_infer", stamped_fractions=(0.05, 0.1, 0.15, 0.2)) == Sourced(pytest.approx(0.125), "stamped")
    assert resolve_operating_fraction(backend="sam3", stamped_fractions=(0.05, 0.1, 0.15, 0.2)) == Sourced(pytest.approx(0.125), "stamped")
    assert resolve_operating_fraction(backend="yolo_infer") == Sourced(0.15, "default")
    assert resolve_operating_fraction(backend="sam2") == Sourced(None, "default")


def test_fraction_set_chain():
    assert resolve_object_tile_fractions(backend="yolo_train", user=[0.1, 0.2], profile=(0.3,), stamped=(0.4,)) == Sourced((0.1, 0.2), "user")
    assert resolve_object_tile_fractions(backend="yolo_train", profile=(0.3,), stamped=(0.4,)) == Sourced((0.3,), "profile")
    assert resolve_object_tile_fractions(backend="yolo_train", stamped=(0.4,)) == Sourced((0.4,), "stamped")
    assert resolve_object_tile_fractions(backend="yolo_train") == Sourced((0.05, 0.10, 0.15, 0.20), "default")
    assert resolve_object_tile_fractions(backend="yolo_train", user=[], stamped=[0, 2.0]) == Sourced((0.05, 0.10, 0.15, 0.20), "default")
    assert resolve_object_tile_fractions(backend="sam2") == Sourced((), "default")


def test_overlap_chain():
    assert resolve_overlap(override=0.4, saved=0.3, fractions=(0.1,)) == Sourced(0.4, "override")
    assert resolve_overlap(saved=0.3, fractions=(0.1,)) == Sourced(0.3, "user")
    assert resolve_overlap(saved=0.2, fractions=(0.5,)) == Sourced(0.2, "user")  # saved is never re-derived
    assert resolve_overlap(fractions=(0.05, 0.15)) == Sourced(0.2, "derived")  # 0.15 + 0.05
    assert resolve_overlap(fractions=(0.88,)) == Sourced(0.9, "derived")  # ceiling
    assert resolve_overlap() == Sourced(DEFAULT_OVERLAP, "default")


def test_derived_overlap_reproduces_trackerkit_default_exactly():
    assert resolve_overlap(fractions=(0.15,)).value == 0.2


def test_derived_overlap_guarantees_whole_animal_in_some_tile():
    for frac in (0.03, 0.055, 0.1, 0.15, 0.3):
        overlap = resolve_overlap(fractions=(frac,)).value
        tile = 1000.0
        assert overlap * tile >= frac * tile


def test_tile_size_matches_planner():
    spec = TilingSpec(enabled=True, geometry_mode="auto_object", reference_body_px=60.0)
    expected = tile_size_for_mode(geometry_mode="auto_object", imgsz=640, reference_body_px=60.0,
                                  object_tile_fraction=0.15, slice_width=0, slice_height=0)
    assert resolve_tile_size(spec, imgsz=640, fraction=0.15) == Sourced(expected, "derived")


def test_tile_size_custom_is_user():
    spec = TilingSpec(geometry_mode="custom", slice_width=512, slice_height=384)
    assert resolve_tile_size(spec, imgsz=640, fraction=None) == Sourced((512, 384), "user")
    spec0 = TilingSpec(geometry_mode="custom")
    assert resolve_tile_size(spec0, imgsz=640, fraction=None) == Sourced((640, 640), "derived")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_tiling_resolution.py -q`
Expected: FAIL — `ImportError: cannot import name 'resolve_overlap'`

- [ ] **Step 3: Implement (append to `utils/tiling_spec.py`; add `tile_size_for_mode` to the `.slice_geometry` import)**

```python
def _positive(value: Any) -> float | None:
    parsed = _finite(value)
    return parsed if parsed is not None and parsed > 0 else None


def resolve_reference_body_px(
    *,
    override: Any = None,
    dataset_median: Any = None,
    stamped: Any = None,
    tracker_reference: Any = None,
) -> Sourced:
    """Body px that sizes tiles: override -> dataset -> stamped -> TrackerKit's own.

    ``tracker_reference`` is REFERENCE_BODY_SIZE x RESIZE_FACTOR, read only;
    this function never writes it.
    """
    for value, source in (
        (override, "override"),
        (dataset_median, "dataset"),
        (stamped, "stamped"),
        (tracker_reference, "user"),
    ):
        parsed = _positive(value)
        if parsed is not None:
            return Sourced(parsed, source)
    return Sourced(0.0, "default")


def resolve_operating_fraction(
    *,
    backend: Backend,
    profile: Any = None,
    stamped_operating: Any = None,
    stamped_fractions=(),
) -> Sourced:
    """The ONE inference scale: profile -> stamped -> backend default."""
    parsed = _positive(profile)
    if parsed is not None:
        return Sourced(_clamp_fraction(parsed), "profile")
    parsed = _positive(stamped_operating)
    if parsed is not None:
        return Sourced(_clamp_fraction(parsed), "stamped")
    stamped = operating_fraction(_fraction_list(stamped_fractions))
    if stamped is not None:
        return Sourced(stamped, "stamped")
    return Sourced(
        operating_fraction(BACKEND_DEFAULTS[backend].object_tile_fractions), "default"
    )


def resolve_object_tile_fractions(
    *, backend: Backend, user: Any = None, profile: Any = None, stamped: Any = ()
) -> Sourced:
    """The fraction SET (training): user -> profile -> stamped -> backend default."""
    for value, source in ((user, "user"), (profile, "profile"), (stamped, "stamped")):
        usable = _fraction_list(value)
        if usable:
            return Sourced(tuple(usable), source)
    return Sourced(BACKEND_DEFAULTS[backend].object_tile_fractions, "default")


def resolve_overlap(*, override: Any = None, saved: Any = None, fractions=()) -> Sourced:
    """Overlap: override -> saved -> derived from the largest scale -> default.

    A SAVED overlap is never re-derived, so existing configs (TrackerKit's
    persisted 0.2) keep their exact value.
    """
    for value, source in ((override, "override"), (saved, "user")):
        parsed = _finite(value)
        if parsed is not None and 0.0 <= parsed <= OVERLAP_MAX:
            return Sourced(parsed, source)
    usable = _fraction_list(fractions)
    if usable:
        return Sourced(min(OVERLAP_MAX, round(max(usable) + OVERLAP_MARGIN, 6)), "derived")
    return Sourced(DEFAULT_OVERLAP, "default")


def resolve_tile_size(spec: TilingSpec, *, imgsz: int, fraction: float | None) -> Sourced:
    """Tile (w, h) for one scale; editable only in custom mode with explicit sizes."""
    size = tile_size_for_mode(
        geometry_mode=spec.geometry_mode,
        imgsz=int(imgsz),
        reference_body_px=float(spec.reference_body_px),
        object_tile_fraction=float(fraction) if fraction is not None else BACKEND_DEFAULTS["yolo_infer"].object_tile_fractions[0],
        slice_width=spec.slice_width,
        slice_height=spec.slice_height,
    )
    explicit = spec.geometry_mode == "custom" and (spec.slice_width > 0 or spec.slice_height > 0)
    return Sourced(size, "user" if explicit else "derived")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_tiling_spec.py tests/test_tiling_spec_canonicalize.py tests/test_tiling_resolution.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
make format
git add src/hydra_suite/utils/tiling_spec.py tests/test_tiling_resolution.py
git commit -m "feat(tiling): sourced resolution for body px, scale, overlap, tile size"
```

### Task 4: S1 gate — regression, adversarial review, merge

- [ ] **Step 1: Regression suites (foreground)**

```bash
python -m pytest tests/test_tiling_spec.py tests/test_tiling_spec_canonicalize.py tests/test_tiling_resolution.py \
  tests/test_slice_geometry.py tests/test_slice_geometry_parity.py tests/test_inference_slicing.py \
  tests/test_core_import_is_light.py -q
make lint
```
Expected: all PASS; lint clean for the new module.

- [ ] **Step 2: Adversarial review (independent worker, different model)**

Dispatch `Agent` with `model: "fable"`, `subagent_type: "general-purpose"`, prompt:

> You are an adversarial reviewer. Find reasons NOT to merge branch `feat/sahi-unify-s1` (worktree `.worktrees/sahi-s1`, diff vs `main`). Spec: `docs/superpowers/specs/2026-10-09-sahi-unification-design.md`; plan: `docs/superpowers/plans/2026-10-09-sahi-unification-s1-s2.md`. EXECUTE code (`conda activate hydra-mps; export PYTHONPATH=$PWD/src`) to falsify claims; construct counterexamples. Targets: (1) `canonicalize` on every real legacy shape in the repo — grep writers of `.slice_meta.json`, `.sam3_meta.json`, `SliceTrainingSettings.to_dict`, `SliceTrainingConfig`, `Sam3LoraParams`, TrackerKit `slice_profile_settings` snapshot, `SemanticEscalationRequest`, SAM2 request — and report any key it drops, mis-maps, or raises on; (2) is `operating_fraction` in `canonicalize` bit-identical to `core/inference/slice_meta._training_values` for every input both accept (fuzz it); (3) does `operating_fraction` equal what `training/sam3_lora/dataset_build` stamps as `prefill_object_tile_fraction` (its `_median`) for random sets incl. ties/duplicates; (4) do the backend defaults match the live dataclass defaults they claim to mirror; (5) read-lenient/write-strict holds (no raise on any legacy value; strict constructor). Do NOT stash, reset, or edit files in the worktree; write scratch scripts under /tmp. Report findings ranked by severity with a reproducer each.

Run alongside it a normal whole-branch reviewer (`superpowers:requesting-code-review`).

- [ ] **Step 3: Fix wave**

For each confirmed finding: failing test first, fix, re-run Step 1. Re-dispatch the adversarial reviewer on the fix diff only if a fix touched more than the reported lines.

- [ ] **Step 4: Merge**

```bash
cd /Users/neurorishika/Projects/Rockefeller/Kronauer/multi-animal-tracker
git merge --no-ff feat/sahi-unify-s1 -m "Merge feat/sahi-unify-s1: canonical SAHI TilingSpec contract"
git worktree remove .worktrees/sahi-s1 && git branch -d feat/sahi-unify-s1
```

---

## Slice S2 — v3 tiling sidecar

> **Revised 2026-10-09 after the plan-stage adversarial review** (B1, M1–M5, m2–m4, m7 folded into Tasks 5–8 below). The S2 equivalence smoke only proves import-time inertness; writer behavior is covered by Task 7/8 tests.

Setup (after S1 is merged):

```bash
cd /Users/neurorishika/Projects/Rockefeller/Kronauer/multi-animal-tracker
git worktree add .worktrees/sahi-s2 -b feat/sahi-unify-s2 HEAD
cd .worktrees/sahi-s2 && conda activate hydra-mps && export PYTHONPATH=$PWD/src
```

### Task 5: v3 geometry builders + schema 3 + `model_family`

**Files:**
- Create: `src/hydra_suite/core/inference/tiling_meta.py`
- Modify: `src/hydra_suite/core/inference/slice_meta.py` — `SLICE_META_SCHEMA_VERSION = 3`; `training_geometry` strips envelope keys from flat docs; `normalized_slice_meta` carries `model_family`; `merge_training_geometry` gains `model_family`
- Test: `tests/test_tiling_meta_build.py`

**Interfaces:**
- Consumes (S1): `canonicalize`, `operating_fraction`; `utils.slice_geometry.LEGACY_TARGET_SIZE_IMGSZ`.
- Produces:
  - `MODEL_FAMILIES = ("yolo", "sam3")`
  - `training_geometry_from_yolo_manifest(slice_geometry: dict) -> dict` — **additive**: every input key kept verbatim; adds `object_tile_fractions` (only if non-empty), `prefill_object_tile_fraction` (only if derivable), `trained_body_px` (only if a body size is present), `fragment_policy="drop"` (only if absent). Never raises.
  - `training_geometry_from_sam3_manifest(build_manifest: dict, *, imgsz: int) -> dict` — canonical-only; stamps only values present in the input plus `fragment_policy="crowd"` and `imgsz`. Never invents `geometry_mode`/`overlap`/`min_area_ratio`. Bare `object_tile_fraction` only when single-scale. Never raises.
  - `slice_meta.merge_training_geometry(existing, training_geometry, *, model_family: str = "yolo") -> dict`
  - v3 document keys: `schema_version` (3), `model_family`, `training_geometry`, `primary_profile_id`, `profiles`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_tiling_meta_build.py
import pytest

from hydra_suite.core.inference.geometry_drift import stamped_object_tile_fraction
from hydra_suite.core.inference.slice_meta import (
    SLICE_META_SCHEMA_VERSION,
    _training_values,
    merge_training_geometry,
    normalized_slice_meta,
    training_geometry,
)
from hydra_suite.core.inference.tiling_meta import (
    training_geometry_from_sam3_manifest,
    training_geometry_from_yolo_manifest,
)


def _yolo_manifest(**over):
    base = {
        "geometry_mode": "auto_object", "imgsz": 640, "object_tile_fraction": 0.1,
        "slice_width": 0, "slice_height": 0, "overlap": 0.2, "min_area_ratio": 0.25,
        "negative_tile_fraction": 0.15, "target_sizes": [32, 64, 96, 128],
        "full_frame_mix": True, "reference_body_px": 41.5,
        "multiscale_loss_balance": {"enabled": True, "power": 0.5},
    }
    base.update(over)
    return base


YOLO_MANIFESTS = [
    _yolo_manifest(),
    _yolo_manifest(target_sizes=[33, 70, 101]),
    _yolo_manifest(target_sizes=[], object_tile_fraction=0.137),
    _yolo_manifest(target_sizes=[96], imgsz=1024),
    _yolo_manifest(geometry_mode="custom", slice_width=800, slice_height=600),
    _yolo_manifest(target_sizes=[4, 8], imgsz=640),
    {"geometry_mode": "auto_object", "target_sizes": [200.0, 300.0], "reference_body_px": 42.0},
    {"object_tile_fraction": 0, "overlap": 0.95},
    {},
]


@pytest.mark.parametrize("manifest", YOLO_MANIFESTS)
def test_yolo_v3_is_additive(manifest):
    """Review Focus 1 + adversarial M3: every v2 consumer sees identical input."""
    v3 = training_geometry_from_yolo_manifest(manifest)
    assert {k: v3[k] for k in manifest} == manifest
    assert _training_values(v3) == _training_values(manifest)
    assert stamped_object_tile_fraction(v3) == stamped_object_tile_fraction(manifest)


def test_yolo_v3_added_keys():
    v3 = training_geometry_from_yolo_manifest(_yolo_manifest())
    assert v3["object_tile_fractions"] == [0.05, 0.1, 0.15, 0.2]
    assert v3["prefill_object_tile_fraction"] == pytest.approx(0.125)
    assert v3["trained_body_px"] == 41.5
    assert v3["fragment_policy"] == "drop"
    assert "tile_px_set" not in v3  # YOLO measures body per frame: no single set


def test_yolo_v3_never_invents():
    """Adversarial B1/m2: an empty manifest gains no geometry."""
    v3 = training_geometry_from_yolo_manifest({})
    assert v3 == {"fragment_policy": "drop"}


def test_yolo_v3_does_not_mutate_input():
    manifest = _yolo_manifest()
    snapshot = dict(manifest)
    training_geometry_from_yolo_manifest(manifest)
    assert manifest == snapshot


SAM3_MULTI = {
    "geometry_mode": "auto_object", "tile_px_set": [[1940, 1940], [970, 970]],
    "object_tile_fractions": [0.0275, 0.055], "prefill_object_tile_fraction": 0.04125,
    "prefill_tile_px": [970, 970], "full_frame_mix": False, "scale_range_px": [970, 1940],
    "reference_body_px": 53.4, "tile_overlap": 0.25, "min_retained_area_frac": 0.3,
    "fragment_counts": {"x": 1}, "scale_counts": {"tile:970x970": 10},
}


def test_sam3_multiscale_shape():
    v3 = training_geometry_from_sam3_manifest(SAM3_MULTI, imgsz=1008)
    assert v3["object_tile_fractions"] == [0.0275, 0.055]
    assert v3["prefill_object_tile_fraction"] == 0.04125
    assert "object_tile_fraction" not in v3  # SAM3 multi-scale convention
    assert v3["tile_px_set"] == [[1940, 1940], [970, 970]]
    assert v3["overlap"] == 0.25
    assert v3["min_area_ratio"] == 0.3
    assert v3["geometry_mode"] == "auto_object"
    assert v3["trained_body_px"] == v3["reference_body_px"] == 53.4
    assert v3["fragment_policy"] == "crowd"
    assert v3["imgsz"] == 1008
    assert v3["full_frame_mix"] is False and v3["scale_range_px"] == [970, 1940]
    for legacy in ("tile_overlap", "min_retained_area_frac", "prefill_tile_px", "fragment_counts", "scale_counts"):
        assert legacy not in v3


def test_sam3_single_scale_keeps_bare_scalar():
    v3 = training_geometry_from_sam3_manifest(
        {"tile_px": [971, 971], "object_tile_fraction": 0.055, "reference_body_px": 53.4},
        imgsz=1008,
    )
    assert v3["object_tile_fraction"] == v3["prefill_object_tile_fraction"] == 0.055
    assert v3["tile_px_set"] == [[971, 971]]


def test_sam3_never_invents():
    """Adversarial B1: absent mode/overlap/min-area stay absent."""
    v3 = training_geometry_from_sam3_manifest({"tile_px": 971}, imgsz=1008)
    assert v3 == {"fragment_policy": "crowd", "imgsz": 1008, "tile_px_set": [[971, 971]]}


@pytest.mark.parametrize(
    "hostile",
    [{"tile_px_set": 971}, {"tile_px_set": [["a", "b"]]}, {"tile_px_set": [[float("nan"), 1]]},
     {"tile_px_set": [[float("inf"), 1]]}, {"tile_px": True}, {"object_tile_fractions": "x"}],
)
def test_sam3_builder_never_raises(hostile):
    training_geometry_from_sam3_manifest(hostile, imgsz=1008)


def test_schema_v3_and_family():
    assert SLICE_META_SCHEMA_VERSION == 3
    doc = merge_training_geometry(None, {"overlap": 0.2}, model_family="sam3")
    assert doc["schema_version"] == 3 and doc["model_family"] == "sam3"
    assert normalized_slice_meta(doc)["model_family"] == "sam3"
    assert normalized_slice_meta({"overlap": 0.2})["model_family"] == "yolo"  # v1/v2 were YOLO-only


def test_flat_doc_geometry_excludes_envelope():
    assert training_geometry({"overlap": 0.2, "schema_version": 1, "model_family": "yolo"}) == {"overlap": 0.2}


def test_merge_preserves_profiles():
    existing = {
        "schema_version": 2, "training_geometry": {"overlap": 0.2}, "primary_profile_id": "p-1",
        "profiles": [{"id": "p-1", "name": "Balanced", "note": "", "settings": {"overlap": 0.3}, "measurement": {"f1": 0.9}}],
    }
    doc = merge_training_geometry(existing, {"overlap": 0.25}, model_family="yolo")
    assert doc["profiles"] == existing["profiles"]
    assert doc["primary_profile_id"] == "p-1"
    assert doc["training_geometry"] == {"overlap": 0.25}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_tiling_meta_build.py -q`
Expected: FAIL — `ModuleNotFoundError: ...tiling_meta`

- [ ] **Step 3: Implement**

`core/inference/slice_meta.py`:

```python
SLICE_META_SCHEMA_VERSION = 3
_ENVELOPE_KEYS = ("schema_version", "model_family", "primary_profile_id", "profiles")
```

```python
def training_geometry(meta: dict[str, Any]) -> dict[str, Any]:
    """Return nested training geometry or a legacy flat payload, without mutation."""
    nested = meta.get("training_geometry")
    if isinstance(nested, dict):
        return dict(nested)
    return {k: v for k, v in meta.items() if k not in _ENVELOPE_KEYS}
```

```python
def normalized_slice_meta(meta: dict[str, Any]) -> dict[str, Any]:
    """Promote legacy metadata to the current document shape without inventing profiles.

    v1/v2 documents were written only for YOLO direct detectors, so an absent
    ``model_family`` means ``"yolo"``.
    """
    return {
        "schema_version": SLICE_META_SCHEMA_VERSION,
        "model_family": str(meta.get("model_family") or "yolo"),
        "training_geometry": training_geometry(meta),
        "primary_profile_id": str(meta.get("primary_profile_id", "") or ""),
        "profiles": available_slice_profiles(meta),
    }
```

```python
def merge_training_geometry(
    existing: dict[str, Any] | None,
    training_geometry: dict[str, Any],
    *,
    model_family: str = "yolo",
) -> dict[str, Any]:
    """Replace training geometry while preserving user-approved profiles.

    Publishing must never destroy calibration a user did before registering.
    """
    result = normalized_slice_meta(existing or {})
    result["training_geometry"] = dict(training_geometry)
    result["model_family"] = str(model_family)
    return result
```

`core/inference/tiling_meta.py`:

```python
"""Unified v3 tiling sidecar: geometry builders and one reader for every family.

Spec: docs/superpowers/specs/2026-10-09-sahi-unification-design.md (§4).
The file is still ``<model>.<ext>.slice_meta.json`` (slice_meta.sidecar_path).
YOLO v3 geometry is ADDITIVE over v2 (every v2 key verbatim) so the v2
reader, the baseline drift guard and the calibration grid are unchanged
until S3 moves them onto ``read_tiling_meta``. SAM3 v3 is a new file and is
canonical-only. Builders stamp only what the input states -- never defaults.
"""

from __future__ import annotations

import math
from typing import Any

from hydra_suite.utils.slice_geometry import LEGACY_TARGET_SIZE_IMGSZ
from hydra_suite.utils.tiling_spec import canonicalize, operating_fraction

MODEL_FAMILIES = ("yolo", "sam3")
# SAM3 build manifests carry build bookkeeping; only these survive into the stamp.
_SAM3_EXTRAS = ("full_frame_mix", "scale_range_px", "keep_empty_tiles")
_SAM3_PRESENT_ONLY = (
    "geometry_mode",
    "reference_body_px",
    "slice_width",
    "slice_height",
    "overlap",
    "min_area_ratio",
)


def _safe_canonicalize(mapping: Any, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """Publishing must never fail on a partial or odd manifest."""
    try:
        return canonicalize(mapping, **kwargs)
    except Exception:
        return {}, {}


def _tile_pairs(raw: Any) -> list[tuple[int, int]]:
    """Parse a tile size / tile-size set; skip anything not a positive finite size."""
    items = raw if isinstance(raw, (list, tuple)) else [raw]
    if len(items) == 2 and all(
        isinstance(v, (int, float)) and not isinstance(v, bool) for v in items
    ):
        items = [items]  # a bare [w, h] pair
    pairs: list[tuple[int, int]] = []
    for item in items:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            w, h = item
        else:
            w = h = item
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in (w, h)):
            continue
        if not all(math.isfinite(float(v)) and float(v) >= 1 for v in (w, h)):
            continue
        pairs.append((int(w), int(h)))
    return pairs


def training_geometry_from_yolo_manifest(slice_geometry: dict[str, Any]) -> dict[str, Any]:
    """v3 block for a YOLO sliced build: the v2 manifest verbatim + canonical keys."""
    geometry = dict(slice_geometry or {})
    # Legacy YOLO pixel target_sizes without imgsz were expressed at 640.
    canonical, _ = _safe_canonicalize(
        geometry, legacy_px_imgsz=LEGACY_TARGET_SIZE_IMGSZ
    )
    fractions = list(canonical.get("object_tile_fractions") or ())
    if fractions:
        geometry.setdefault("object_tile_fractions", fractions)
    operating = canonical.get("operating_fraction")
    if operating is not None:
        geometry.setdefault("prefill_object_tile_fraction", float(operating))
    if "reference_body_px" in canonical:
        geometry.setdefault("trained_body_px", float(canonical["reference_body_px"]))
    geometry.setdefault("fragment_policy", "drop")
    return geometry


def training_geometry_from_sam3_manifest(
    build_manifest: dict[str, Any], *, imgsz: int
) -> dict[str, Any]:
    """v3 block for a SAM3 tile build: canonical names, present values only."""
    canonical, extras = _safe_canonicalize(build_manifest)
    geometry: dict[str, Any] = {"fragment_policy": "crowd", "imgsz": int(imgsz)}
    for key in _SAM3_PRESENT_ONLY:
        if key in canonical:
            geometry[key] = canonical[key]
    if "reference_body_px" in canonical:
        geometry["trained_body_px"] = float(canonical["reference_body_px"])
    fractions = list(canonical.get("object_tile_fractions") or ())
    if fractions:
        geometry["object_tile_fractions"] = fractions
    operating = canonical.get("operating_fraction")
    if operating is None:
        operating = operating_fraction(fractions)
    if operating is not None:
        geometry["prefill_object_tile_fraction"] = float(operating)
        # A median under a measurement's name reads as "the" training tile
        # size downstream; multi-scale stamps it only as the named prefill.
        if len(fractions) <= 1:
            geometry["object_tile_fraction"] = float(operating)
    raw_set = (build_manifest or {}).get("tile_px_set")
    if not raw_set and (build_manifest or {}).get("tile_px") is not None:
        raw_set = build_manifest["tile_px"]
    tiles = _tile_pairs(raw_set) if raw_set is not None else []
    if tiles:
        geometry["tile_px_set"] = [[w, h] for w, h in tiles]
    geometry.update({k: extras[k] for k in _SAM3_EXTRAS if k in extras})
    return geometry
```

Note `_tile_pairs` input shapes: `[[1940, 1940], [970, 970]]` → two pairs; `[971, 971]` (a single-scale manifest's `tile_px`) → one pair; `971` → one pair; anything non-numeric/non-finite/<1 is skipped.

- [ ] **Step 4: Run tests, then the existing slice_meta suites**

Run: `python -m pytest tests/test_tiling_meta_build.py tests/test_slice_meta_read.py tests/test_slice_profile_resolution.py tests/test_slice_profile_mutations.py tests/test_engine_params_slice_profile.py tests/test_trackerkit_slice_meta_prefill.py tests/test_detectkit_direct_calibration_ui.py -q`
Expected: PASS. If an existing test asserts `schema_version == 2` on a normalized/written document, change that literal to `SLICE_META_SCHEMA_VERSION`; if it asserts exact equality of a normalized document, add `"model_family": "yolo"`. Nothing else may change.

- [ ] **Step 5: Commit**

```bash
make format
git add src/hydra_suite/core/inference/tiling_meta.py src/hydra_suite/core/inference/slice_meta.py tests/
git commit -m "feat(tiling): v3 slice_meta with model_family; additive YOLO and canonical SAM3 geometry"
```

### Task 6: `read_tiling_meta` — one reader for v1/v2/v3 and legacy `.sam3_meta.json`

**Files:**
- Modify: `src/hydra_suite/core/inference/tiling_meta.py` (append)
- Test: `tests/test_tiling_meta_read.py`

**Interfaces:**
- Consumes: Task 5; `slice_meta.read_slice_meta`, `training_geometry`, `available_slice_profiles`, `merge_training_geometry`; `geometry_drift.stamped_tile_px_set`; S1 `TilingSpec.from_canonical`.
- Produces:
  - `sam3_meta_path(model_path) -> Path` (`<artifact>.sam3_meta.json`)
  - `@dataclass(frozen=True) class TilingMeta: model_family: str; source: str; training: TilingSpec | None; imgsz: int; tile_px_set: tuple[tuple[int, int], ...]; operating_fraction: float | None; extras: dict; primary_profile_id: str; profiles: tuple[dict, ...]`
  - `read_tiling_meta(model_path) -> TilingMeta | None` — never raises. `source` ∈ {`"slice_meta"`, `"sam3_meta"`}. `training.fragment_policy` defaults to the family's (`drop`/`crowd`) when unstamped. `operating_fraction` = canonical operating value (stamped prefill or legacy target_sizes median), else np.median of the fractions; clamped [0.01, 0.9].

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_tiling_meta_read.py
import json

import pytest

from hydra_suite.core.inference.slice_meta import (
    _training_values,
    merge_training_geometry,
    sidecar_path,
    write_slice_meta,
)
from hydra_suite.core.inference.tiling_meta import (
    read_tiling_meta,
    sam3_meta_path,
    training_geometry_from_sam3_manifest,
    training_geometry_from_yolo_manifest,
)


def _write(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")


@pytest.fixture
def model(tmp_path):
    p = tmp_path / "det.pt"
    p.write_bytes(b"x")
    return p


V1_YOLO = {
    "geometry_mode": "auto_object", "imgsz": 640, "object_tile_fraction": 0.1,
    "overlap": 0.2, "target_sizes": [32, 64, 96, 128], "reference_body_px": 40.0,
    "slice_width": 0, "slice_height": 0,
}


def test_absent_returns_none(model):
    assert read_tiling_meta(model) is None


def test_corrupt_returns_none(model):
    sidecar_path(model).write_text("{not json", encoding="utf-8")
    assert read_tiling_meta(model) is None


def test_v1_flat_yolo(model):
    _write(sidecar_path(model), V1_YOLO)
    meta = read_tiling_meta(model)
    assert meta.model_family == "yolo" and meta.source == "slice_meta"
    assert meta.training.object_tile_fractions == (0.05, 0.1, 0.15, 0.2)
    assert meta.training.reference_body_px == 40.0
    assert meta.training.fragment_policy == "drop"
    assert meta.operating_fraction == _training_values(V1_YOLO)["object_tile_fraction"]
    assert meta.imgsz == 640


def test_v1_target_sizes_without_imgsz(model):
    """Review Focus 2: 640 anchor for YOLO; operating matches TrackerKit."""
    doc = {k: v for k, v in V1_YOLO.items() if k != "imgsz"}
    _write(sidecar_path(model), doc)
    meta = read_tiling_meta(model)
    assert meta.training.object_tile_fractions == (0.05, 0.1, 0.15, 0.2)
    assert meta.operating_fraction == _training_values(doc)["object_tile_fraction"] == 0.1


def test_v2_profiles_preserved(model):
    profile = {"id": "bal-1", "name": "Balanced", "note": "", "settings": {"overlap": 0.3}, "measurement": {}}
    _write(sidecar_path(model), {"schema_version": 2, "training_geometry": V1_YOLO,
                                 "primary_profile_id": "bal-1", "profiles": [profile]})
    meta = read_tiling_meta(model)
    assert meta.primary_profile_id == "bal-1"
    assert meta.profiles == (profile,)


def test_v2_profiles_only_no_geometry(model):
    profile = {"id": "a-1", "name": "A", "note": "", "settings": {}, "measurement": {}}
    _write(sidecar_path(model), {"schema_version": 2, "training_geometry": {}, "profiles": [profile]})
    meta = read_tiling_meta(model)
    assert meta.training is None
    assert meta.profiles == (profile,)


def test_v3_yolo_round_trip(model):
    geometry = training_geometry_from_yolo_manifest(dict(V1_YOLO, target_sizes=[33, 70, 101]))
    write_slice_meta(model, merge_training_geometry(None, geometry, model_family="yolo"))
    meta = read_tiling_meta(model)
    assert meta.operating_fraction == geometry["prefill_object_tile_fraction"]
    assert list(meta.training.object_tile_fractions) == geometry["object_tile_fractions"]


def test_v3_sam3_round_trip(model):
    geometry = training_geometry_from_sam3_manifest(
        {"tile_px_set": [[1940, 1940], [970, 970]], "object_tile_fractions": [0.0275, 0.055],
         "prefill_object_tile_fraction": 0.04125, "reference_body_px": 53.4,
         "tile_overlap": 0.25, "geometry_mode": "auto_object"},
        imgsz=1008,
    )
    write_slice_meta(model, merge_training_geometry(None, geometry, model_family="sam3"))
    meta = read_tiling_meta(model)
    assert meta.model_family == "sam3"
    assert meta.tile_px_set == ((1940, 1940), (970, 970))
    assert meta.operating_fraction == 0.04125
    assert meta.training.fragment_policy == "crowd"
    assert meta.training.overlap == 0.25


def test_legacy_sam3_meta_single_scale(model):
    _write(sam3_meta_path(model), {"base_variant": "sam3", "prompt": "ant", "train_tile_px": 971,
                                   "object_tile_fraction": 0.055, "reference_body_px": 53.4, "imgsz": 1008})
    meta = read_tiling_meta(model)
    assert meta.model_family == "sam3" and meta.source == "sam3_meta"
    assert meta.tile_px_set == ((971, 971),)
    assert meta.training.object_tile_fractions == (0.055,)
    assert meta.operating_fraction == 0.055
    assert meta.training.fragment_policy == "crowd"
    assert "prompt" not in meta.extras


def test_legacy_sam3_meta_multiscale(model):
    _write(sam3_meta_path(model), {"train_tile_px_set": [[1940, 1940], [970, 970]],
                                   "object_tile_fractions": [0.0275, 0.055],
                                   "prefill_object_tile_fraction": 0.04125,
                                   "reference_body_px": 53.4, "imgsz": 1008})
    meta = read_tiling_meta(model)
    assert meta.tile_px_set == ((1940, 1940), (970, 970))
    assert meta.operating_fraction == 0.04125


def test_slice_meta_geometry_wins_over_sam3_meta(model):
    _write(sam3_meta_path(model), {"train_tile_px": 500, "object_tile_fraction": 0.1, "imgsz": 1008})
    geometry = training_geometry_from_sam3_manifest(
        {"tile_px": [971, 971], "object_tile_fraction": 0.055, "reference_body_px": 53.4}, imgsz=1008)
    write_slice_meta(model, merge_training_geometry(None, geometry, model_family="sam3"))
    meta = read_tiling_meta(model)
    assert meta.source == "slice_meta"
    assert meta.tile_px_set == ((971, 971),)


def test_lenient_on_bad_values(model):
    _write(sidecar_path(model), {"overlap": 1.5, "geometry_mode": "weird", "object_tile_fraction": 0})
    meta = read_tiling_meta(model)
    assert meta.training.overlap == 0.9
    assert meta.training.geometry_mode == "auto_model"


@pytest.mark.parametrize(
    "doc",
    [
        {"tile_px_set": 971}, {"tile_px_set": [["a", "b"]]}, {"tile_px_set": [[float("nan"), 1]]},
        {"tile_px_set": [[float("inf"), 1]]}, {"imgsz": float("inf")}, {"imgsz": float("nan")},
        {"imgsz": 0.5, "target_sizes": [64]}, {"train_tile_px": 971, "imgsz": float("nan")},
        {"training_geometry": [1, 2]}, {"profiles": {"a": 1}}, {"model_family": 7, "overlap": 0.2},
        {"object_tile_fractions": {"a": 1}}, {"reference_body_px": "1e999"},
    ],
)
def test_hostile_documents_never_raise(model, doc):
    """Adversarial M4."""
    sidecar_path(model).write_text(json.dumps(doc, allow_nan=True), encoding="utf-8")
    read_tiling_meta(model)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_tiling_meta_read.py -q`
Expected: FAIL — `ImportError: cannot import name 'read_tiling_meta'`

- [ ] **Step 3: Implement (append to `tiling_meta.py`; extend imports)**

Imports to add at the top of `tiling_meta.py`:

```python
import json
import logging
from dataclasses import dataclass, field, replace
from pathlib import Path

from hydra_suite.utils.tiling_spec import FRACTION_MAX, FRACTION_MIN, TilingSpec

from .geometry_drift import stamped_tile_px_set
from .slice_meta import available_slice_profiles, read_slice_meta, training_geometry

logger = logging.getLogger(__name__)
_FAMILY_FRAGMENT = {"yolo": "drop", "sam3": "crowd"}
_SAM3_META_EXTRAS = ("full_frame_mix", "scale_range_px", "scale_grouped_batching", "augmentation")
_GEOMETRY_ONLY_KEYS = ("imgsz", "tile_px_set", "train_tile_px", "train_tile_px_set", "prefill_train_tile_px")
```

Then:

```python
def sam3_meta_path(model_path: str | Path) -> Path:
    """``<artifact>.sam3_meta.json`` (append-style, as publish_worker writes it)."""
    path = Path(model_path)
    return path.with_name(path.name + ".sam3_meta.json")


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _imgsz(raw: Any) -> int:
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return 0
    return int(raw) if math.isfinite(float(raw)) and raw >= 1 else 0


@dataclass(frozen=True)
class TilingMeta:
    model_family: str
    source: str  # "slice_meta" | "sam3_meta"
    training: TilingSpec | None
    imgsz: int
    tile_px_set: tuple[tuple[int, int], ...]
    operating_fraction: float | None
    extras: dict[str, Any] = field(default_factory=dict)
    primary_profile_id: str = ""
    profiles: tuple[dict[str, Any], ...] = ()


def read_tiling_meta(model_path: str | Path) -> TilingMeta | None:
    """The ONE reader for a model's SAHI geometry and calibration profiles. Never raises."""
    try:
        return _read_tiling_meta(model_path)
    except Exception:
        logger.warning("Unreadable SAHI metadata beside %s", model_path, exc_info=True)
        return None


def _read_tiling_meta(model_path: str | Path) -> TilingMeta | None:
    slice_doc = read_slice_meta(model_path) or {}
    geometry = training_geometry(slice_doc) if slice_doc else {}
    family = slice_doc.get("model_family") or "yolo"
    source = "slice_meta"
    if not geometry:
        sam3_doc = _read_json(sam3_meta_path(model_path))
        if sam3_doc:
            geometry, family, source = sam3_doc, "sam3", "sam3_meta"
    if not geometry and not slice_doc:
        return None
    if family not in MODEL_FAMILIES:
        logger.warning("Unknown SAHI model_family %r beside %s; reading as yolo", family, model_path)
        family = "yolo"

    training: TilingSpec | None = None
    operating: float | None = None
    extras: dict[str, Any] = {}
    tiles: tuple[tuple[int, int], ...] = ()
    imgsz = 0
    if geometry:
        legacy = LEGACY_TARGET_SIZE_IMGSZ if family == "yolo" else None
        canonical, extras = _safe_canonicalize(geometry, legacy_px_imgsz=legacy)
        try:
            training = TilingSpec.from_canonical(canonical)
        except Exception:
            logger.warning("Invalid SAHI geometry beside %s", model_path, exc_info=True)
            training = None
        if training is not None:
            training = replace(training, enabled=True)
            if "fragment_policy" not in canonical:
                training = replace(training, fragment_policy=_FAMILY_FRAGMENT[family])
            operating = canonical.get("operating_fraction")
            if operating is None:
                operating = training.operating_fraction()
            else:
                operating = max(FRACTION_MIN, min(FRACTION_MAX, float(operating)))
        imgsz = _imgsz(geometry.get("imgsz"))
        if geometry.get("tile_px_set") is not None:
            tiles = tuple(_tile_pairs(geometry["tile_px_set"]))
        else:
            try:
                stamped = stamped_tile_px_set(geometry) or ()
            except Exception:
                stamped = ()
            tiles = tuple(_tile_pairs([list(pair) for pair in stamped]))
        if source == "sam3_meta":
            extras = {k: extras[k] for k in _SAM3_META_EXTRAS if k in extras}
        for key in _GEOMETRY_ONLY_KEYS:
            extras.pop(key, None)

    return TilingMeta(
        model_family=family,
        source=source,
        training=training,
        imgsz=imgsz,
        tile_px_set=tiles,
        operating_fraction=operating,
        extras=extras,
        primary_profile_id=str(slice_doc.get("primary_profile_id", "") or ""),
        profiles=tuple(available_slice_profiles(slice_doc)),
    )
```

Note: an invalid stamped `geometry_mode` reads as the `TilingSpec` default `auto_model`, unlike `_training_values` (`auto_object`). No caller uses `read_tiling_meta` in S2; S3 decides the TrackerKit prefill default. Say so in the commit message.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_tiling_meta_read.py tests/test_tiling_meta_build.py tests/test_geometry_drift_guard.py tests/test_geometry_drift_severity.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
make format
git add src/hydra_suite/core/inference/tiling_meta.py tests/test_tiling_meta_read.py
git commit -m "feat(tiling): read_tiling_meta, one never-raising reader for v1/v2/v3 and legacy sam3_meta

An invalid stamped geometry_mode reads as the TilingSpec default
(auto_model), unlike _training_values (auto_object); no caller uses
read_tiling_meta yet, S3 decides the prefill default."
```

### Task 7: YOLO publish writes v3

**Files:**
- Modify: `src/hydra_suite/training/model_publish.py` (the `slice_geometry` branch, ~874-895)
- Modify: `tests/test_model_publish_slice_geometry.py` (the 3 `test_all_direct_detector_roles_publish_slice_geometry[*]` cases assert exact equality with the raw manifest)
- Test: `tests/test_tiling_meta_publish_yolo.py`

**Interfaces:**
- Consumes: `training_geometry_from_yolo_manifest`, `merge_training_geometry(..., model_family="yolo")`, `read_tiling_meta`.
- Produces: published `<dst>.pt.slice_meta.json` is schema 3, `model_family="yolo"`, additive v3 `training_geometry`. Registry `metadata["slice_geometry"]` stays the raw manifest.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_tiling_meta_publish_yolo.py
import json
from pathlib import Path

import hydra_suite.training.model_publish as mp
from hydra_suite.core.inference.slice_meta import _training_values, sidecar_path, write_slice_meta
from hydra_suite.core.inference.tiling_meta import read_tiling_meta
from hydra_suite.training.contracts import TrainingRole

MANIFEST = {
    "geometry_mode": "auto_object", "imgsz": 640, "object_tile_fraction": 0.1,
    "slice_width": 0, "slice_height": 0, "overlap": 0.2, "min_area_ratio": 0.25,
    "negative_tile_fraction": 0.15, "target_sizes": [32, 64, 96, 128],
    "full_frame_mix": True, "reference_body_px": 41.5,
    "multiscale_loss_balance": {"enabled": True, "power": 0.5},
}


def _publish_direct_model(tmp_path, monkeypatch, *, slice_geometry, source_sidecar=None):
    monkeypatch.setattr(mp, "get_models_root", lambda: tmp_path)
    src = tmp_path / "weights.pt"
    src.write_bytes(b"fake-weights")
    if source_sidecar is not None:
        write_slice_meta(src, source_sidecar)
    _key, stored = mp.publish_trained_model(
        role=TrainingRole.OBB_DIRECT, artifact_path=str(src), size="s", species="ant",
        model_info="sliced", trained_from_run_id="r1", dataset_fingerprint="fp",
        base_model="yolo26s-obb.pt", slice_geometry=slice_geometry,
    )
    return Path(stored)


def test_publish_writes_additive_v3(tmp_path, monkeypatch):
    dst = _publish_direct_model(tmp_path, monkeypatch, slice_geometry=MANIFEST)
    doc = json.loads(sidecar_path(dst).read_text())
    assert doc["schema_version"] == 3 and doc["model_family"] == "yolo"
    geometry = doc["training_geometry"]
    assert {k: geometry[k] for k in MANIFEST} == MANIFEST
    assert _training_values(geometry) == _training_values(MANIFEST)
    meta = read_tiling_meta(dst)
    assert meta.training.object_tile_fractions == (0.05, 0.1, 0.15, 0.2)
    reg = mp.load_model_registry()
    assert any(entry.get("slice_geometry") == MANIFEST for entry in reg["entries"].values())


def test_republish_upgrades_v2_and_keeps_profiles(tmp_path, monkeypatch):
    """Review Focus 5."""
    profile = {"id": "bal-1", "name": "Balanced", "note": "n",
               "settings": {"overlap": 0.3, "object_tile_fraction": 0.12},
               "measurement": {"f1": 0.91, "checkpoint_fingerprint": "sha256:abc"}}
    dst = _publish_direct_model(
        tmp_path, monkeypatch, slice_geometry=MANIFEST,
        source_sidecar={"schema_version": 2, "training_geometry": MANIFEST,
                        "primary_profile_id": "bal-1", "profiles": [profile]},
    )
    doc = json.loads(sidecar_path(dst).read_text())
    assert doc["schema_version"] == 3
    assert doc["profiles"] == [profile]
    assert doc["primary_profile_id"] == "bal-1"
```

`test_slice_geometry_written_as_sidecar_and_registry` in `tests/test_model_publish_slice_geometry.py` is the reference for the monkeypatch/publish pattern; if the registry/models root plumbing differs from this helper, mirror that test.

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_tiling_meta_publish_yolo.py -q`
Expected: FAIL — `KeyError: 'fragment_policy'`-style or `model_family` mismatch until the writer switches (the publish path still writes the raw manifest under `model_family` default `"yolo"`; the test that must fail is the one asserting `object_tile_fractions` via `read_tiling_meta` only if Task 6's reader cannot derive them — if Step 2 unexpectedly passes, add `assert "object_tile_fractions" in geometry` to `test_publish_writes_additive_v3` and re-run: it must fail).

- [ ] **Step 3: Implement** — in `model_publish.py`, import `training_geometry_from_yolo_manifest` from `hydra_suite.core.inference.tiling_meta` and change the merge:

```python
        source_meta = read_slice_meta(src)
        merged_slice_meta = merge_training_geometry(
            source_meta,
            training_geometry_from_yolo_manifest(dict(slice_geometry)),
            model_family="yolo",
        )
```

Leave `metadata["slice_geometry"] = dict(slice_geometry)` unchanged.

Retarget the 3 exact-equality assertions in `test_all_direct_detector_roles_publish_slice_geometry` from `training_geometry == manifest` to `{k: training_geometry[k] for k in manifest} == manifest` (additive v3). Change nothing else in that file.

- [ ] **Step 4: Run new + existing publish suites**

Run: `python -m pytest tests/test_tiling_meta_publish_yolo.py tests/test_model_publish_slice_geometry.py tests/test_service_publish_slice_geometry.py tests/test_trackerkit_slice_meta_prefill.py tests/test_engine_params_slice_profile.py tests/test_gui_cli_profile_parity.py tests/test_trackerkit_cli_sahi_profile.py tests/test_sliced_dataset_reference.py tests/test_geometry_drift_guard.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
make format
git add src/hydra_suite/training/model_publish.py tests/test_tiling_meta_publish_yolo.py tests/test_model_publish_slice_geometry.py
git commit -m "feat(tiling): YOLO publish stamps additive v3 training geometry"
```

### Task 8: SAM3 publish dual-writes `.slice_meta.json` (+ payload, cleanup)

**Files:**
- Modify: `src/hydra_suite/training/sam3_lora/publish.py` — `_request_payload` (~377-483) forwards `geometry_mode`, `tile_overlap`, `min_retained_area_frac`; `_cleanup_attempt` (~316-319) removes the attempt's tiling sidecar
- Modify: `src/hydra_suite/training/sam3_lora/publish_worker.py` — `_write_tiling_sidecar`, called after `_promote_staged_pair(...)` in `publish_sam3_artifact`
- Test: `tests/test_sam3_publish_tiling_sidecar.py`

**Interfaces:**
- Consumes: `training_geometry_from_sam3_manifest`, `slice_meta.merge_training_geometry`, `read_slice_meta`, `write_slice_meta`, `PREDICTOR_IMGSZ` (already in `publish_worker`).
- Produces:
  - `publish._request_payload(...)["build_manifest"]` additionally carries `geometry_mode` (str ∈ `auto_model|auto_object|custom`), `tile_overlap` (finite, [0, 1)), `min_retained_area_frac` (finite, [0, 1]) when the build manifest has them; invalid → `ValueError` like the other fields.
  - `publish_worker._write_tiling_sidecar(artifact_path: Path, build_manifest: dict) -> Path | None` — read-merges any existing `.slice_meta.json` (profiles kept), never raises.
  - `publish._cleanup_attempt` also deletes `<artifact>.slice_meta.json` and `<artifact>.slice_meta.json.tmp` when it deletes an owned final pair. The path is built locally (`artifact_path.with_name(artifact_path.name + ".slice_meta.json")`) — `publish.py` must not import `hydra_suite.core.inference` (its `__init__` loads torch).

- [ ] **Step 1: Read `_request_payload`'s signature and call site** (`publish.py:361` and `:585`) to build a valid call in the test. Then write the failing tests:

```python
# tests/test_sam3_publish_tiling_sidecar.py
import json
import logging

import pytest

from hydra_suite.core.inference.slice_meta import sidecar_path, write_slice_meta
from hydra_suite.core.inference.tiling_meta import read_tiling_meta
from hydra_suite.training.sam3_lora import publish, publish_worker

FULL_MANIFEST = {
    "type": "sam3_coco_tiles", "geometry_mode": "auto_object",
    "tile_px_set": [[1940, 1940], [970, 970]], "object_tile_fractions": [0.0275, 0.055],
    "prefill_object_tile_fraction": 0.04125, "prefill_tile_px": [970, 970],
    "reference_body_px": 53.4, "tile_overlap": 0.25, "min_retained_area_frac": 0.3,
    "full_frame_mix": False, "scale_range_px": [970, 1940],
}


def _child_manifest(full):
    """What the publish child really receives (adversarial M1)."""
    payload = publish._request_payload(**_request_kwargs(full))  # build kwargs per Step 1
    return payload["build_manifest"]


def test_child_manifest_carries_tiling_fields():
    child = _child_manifest(FULL_MANIFEST)
    assert child["geometry_mode"] == "auto_object"
    assert child["tile_overlap"] == 0.25
    assert child["min_retained_area_frac"] == 0.3


@pytest.mark.parametrize(
    "field,bad",
    [("geometry_mode", "weird"), ("tile_overlap", 1.0), ("tile_overlap", float("nan")),
     ("min_retained_area_frac", 1.5), ("tile_overlap", True)],
)
def test_child_manifest_rejects_invalid_tiling_fields(field, bad):
    with pytest.raises(ValueError):
        _child_manifest(dict(FULL_MANIFEST, **{field: bad}))


def test_tiling_sidecar_from_child_manifest(tmp_path):
    artifact = tmp_path / "sam3-run.pt"
    artifact.write_bytes(b"x")
    written = publish_worker._write_tiling_sidecar(artifact, _child_manifest(FULL_MANIFEST))
    assert written == sidecar_path(artifact)
    meta = read_tiling_meta(artifact)
    assert meta.model_family == "sam3" and meta.source == "slice_meta"
    assert meta.training.geometry_mode == "auto_object"
    assert meta.training.overlap == 0.25
    assert meta.training.min_area_ratio == 0.3
    assert meta.tile_px_set == ((1940, 1940), (970, 970))
    assert meta.imgsz == publish_worker.PREDICTOR_IMGSZ


def test_tiling_sidecar_keeps_existing_profiles(tmp_path):
    """Adversarial m3."""
    artifact = tmp_path / "sam3-run.pt"
    artifact.write_bytes(b"x")
    profile = {"id": "p-1", "name": "Calibrated", "note": "", "settings": {"object_tile_fraction": 0.05}, "measurement": {}}
    write_slice_meta(artifact, {"schema_version": 3, "model_family": "sam3", "training_geometry": {},
                                "primary_profile_id": "p-1", "profiles": [profile]})
    publish_worker._write_tiling_sidecar(artifact, _child_manifest(FULL_MANIFEST))
    meta = read_tiling_meta(artifact)
    assert meta.profiles == (profile,) and meta.primary_profile_id == "p-1"


def test_write_failure_is_non_fatal(tmp_path, caplog, monkeypatch):
    """Review Focus 4."""
    caplog.set_level(logging.WARNING)
    artifact = tmp_path / "sam3-run.pt"
    artifact.write_bytes(b"x")

    def boom(*_a, **_k):
        raise OSError("disk full")

    monkeypatch.setattr(publish_worker, "write_slice_meta", boom)
    assert publish_worker._write_tiling_sidecar(artifact, FULL_MANIFEST) is None
    assert "tiling sidecar" in caplog.text


def test_cleanup_removes_owned_tiling_sidecar(tmp_path):
    artifact = tmp_path / "sam3-run.pt"
    sam3_sidecar = tmp_path / "sam3-run.pt.sam3_meta.json"
    artifact.write_bytes(b"x")
    sam3_sidecar.write_text(json.dumps({"publish_attempt_id": "a" * 32}))
    tiling = sidecar_path(artifact)
    tiling.write_text("{}")
    tiling.with_name(tiling.name + ".tmp").write_text("{}")
    publish._cleanup_attempt(artifact_path=artifact, sidecar_path=sam3_sidecar,
                             control_dir=None, attempt_id="a" * 32)
    assert not artifact.exists() and not sam3_sidecar.exists()
    assert not tiling.exists() and not tiling.with_name(tiling.name + ".tmp").exists()


def test_cleanup_keeps_unowned_tiling_sidecar(tmp_path):
    artifact = tmp_path / "sam3-run.pt"
    sam3_sidecar = tmp_path / "sam3-run.pt.sam3_meta.json"
    artifact.write_bytes(b"x")
    sam3_sidecar.write_text(json.dumps({"publish_attempt_id": "b" * 32}))
    tiling = sidecar_path(artifact)
    tiling.write_text("{}")
    publish._cleanup_attempt(artifact_path=artifact, sidecar_path=sam3_sidecar,
                             control_dir=None, attempt_id="a" * 32)
    assert tiling.exists()
```

Define `_request_kwargs(full)` in the test module from what Step 1 shows `_request_payload` needs (run id, params, paths, `build_manifest=full`, …), using the same minimal values the existing `tests/test_sam3_publish*.py` tests use for it (grep them for `_request_payload`).

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_sam3_publish_tiling_sidecar.py -q`
Expected: FAIL — `KeyError: 'geometry_mode'` (payload) and `AttributeError: ... '_write_tiling_sidecar'`.

- [ ] **Step 3: Implement**

`publish.py`, in `_request_payload` just before `return {` (after the `geometry_fields` loop):

```python
    # Tiling settings the child stamps into the v3 .slice_meta.json. Without
    # them the child would record a geometry the build never used.
    mode = build_manifest.get("geometry_mode")
    if mode is not None:
        if mode not in ("auto_model", "auto_object", "custom"):
            raise ValueError(f"SAM3 publish geometry 'geometry_mode' is invalid: {mode!r}")
        geometry["geometry_mode"] = mode
    for field, upper_inclusive in (("tile_overlap", False), ("min_retained_area_frac", True)):
        if field not in build_manifest:
            continue
        value = build_manifest[field]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
            or (float(value) > 1.0 if upper_inclusive else float(value) >= 1.0)
        ):
            raise ValueError(f"SAM3 publish geometry {field!r} is out of range: {value!r}")
        geometry[field] = float(value)
```

`publish.py` `_cleanup_attempt`, inside `if owned_final_pair:`:

```python
        # Built locally: importing core.inference here would load torch in the
        # parent (core/inference/__init__ pulls the runner).
        tiling_sidecar = artifact_path.with_name(artifact_path.name + ".slice_meta.json")
        attempt(lambda: tiling_sidecar.unlink(missing_ok=True))
        attempt(
            lambda: tiling_sidecar.with_name(tiling_sidecar.name + ".tmp").unlink(
                missing_ok=True
            )
        )
```

`publish_worker.py` imports:

```python
from hydra_suite.core.inference.slice_meta import (
    merge_training_geometry,
    read_slice_meta,
    write_slice_meta,
)
from hydra_suite.core.inference.tiling_meta import training_geometry_from_sam3_manifest
```

New function near `_write_sidecar`:

```python
def _write_tiling_sidecar(artifact_path: Path, build_manifest: dict[str, Any]) -> Path | None:
    """Dual-write the canonical v3 ``.slice_meta.json`` beside a promoted artifact.

    Non-fatal by design: geometry is also in ``.sam3_meta.json`` and
    ``read_tiling_meta`` falls back to it. Read-merges an existing document so
    calibration profiles saved beside the artifact survive.
    """
    try:
        geometry = training_geometry_from_sam3_manifest(
            build_manifest, imgsz=PREDICTOR_IMGSZ
        )
        merged = merge_training_geometry(
            read_slice_meta(artifact_path), geometry, model_family="sam3"
        )
        return write_slice_meta(artifact_path, merged)
    except Exception:
        logger.warning(
            "sam3 publish: could not write the tiling sidecar for %s; "
            "readers fall back to .sam3_meta.json",
            artifact_path,
            exc_info=True,
        )
        return None
```

In `publish_sam3_artifact`, immediately after `_promote_staged_pair(...)` and before `return artifact_path, sidecar_path`:

```python
        _write_tiling_sidecar(artifact_path, build_manifest)
```

- [ ] **Step 4: Run new + existing SAM3 publish suites**

Run: `python -m pytest tests/test_sam3_publish_tiling_sidecar.py tests/test_sam3_publish.py tests/test_sam3_publish_sidecar.py tests/test_sam3_publish_atomic.py tests/test_sam3_publish_lifecycle.py tests/test_sam3_service_publish.py tests/test_sam3_multiscale_stamp.py tests/test_core_import_is_light.py -q`
Expected: PASS, except `test_sam3_publish_sidecar.py::test_importing_parent_publish_module_does_not_import_torch`, which already fails on `main` (verify it fails identically on `main` before accepting; it must not change). If a lifecycle test asserts the exact file set in the models dir, add the `.slice_meta.json` name — nothing else.

- [ ] **Step 5: Commit**

```bash
make format
git add src/hydra_suite/training/sam3_lora/publish.py src/hydra_suite/training/sam3_lora/publish_worker.py tests/test_sam3_publish_tiling_sidecar.py
git commit -m "feat(tiling): SAM3 publish forwards tiling fields and dual-writes the v3 sidecar"
```

### Task 9: S2 gate — regression, adversarial review, merge

- [ ] **Step 1: Regression (foreground; batch if the full suite is slow)**

```bash
python -m pytest tests/test_tiling_*.py tests/test_slice_*.py tests/test_sam3_publish*.py tests/test_sam3_multiscale_*.py \
  tests/test_model_publish_slice_geometry.py tests/test_service_publish_slice_geometry.py \
  tests/test_engine_params_slice_profile.py tests/test_gui_cli_profile_parity.py tests/test_trackerkit_cli_sahi_profile.py \
  tests/test_trackerkit_slice_meta_prefill.py tests/test_geometry_drift_guard.py tests/test_direct_calibration_grid.py \
  tests/test_sam3_gui_cli_training_parity.py tests/test_shared_scoring_primitives_identity.py tests/test_core_import_is_light.py -q
make lint
```
Expected: all PASS. Compare failure sets against `main` at the branch base for anything else that fails (memory: "pre-existing" is tested against the branch base).

- [ ] **Step 2: Adversarial review (independent worker, different model)**

Dispatch `Agent` with `model: "fable"`, `subagent_type: "general-purpose"`, prompt:

> You are an adversarial reviewer. Find reasons NOT to merge `feat/sahi-unify-s2` (worktree `.worktrees/sahi-s2`, diff vs `main`). Spec §4 and plan Tasks 5–8: `docs/superpowers/specs/2026-10-09-sahi-unification-design.md`, `docs/superpowers/plans/2026-10-09-sahi-unification-s1-s2.md`. EXECUTE code (`conda activate hydra-mps; export PYTHONPATH=$PWD/src`). Try to break: (1) old-reader invariance — fuzz YOLO manifests and assert `slice_meta._training_values(v3) == _training_values(v2)` bit-exactly, and run TrackerKit's `engine_params` profile overlay + `cli_config.apply_sahi_profile_override` against v3 sidecars with profiles; (2) every consumer of `.slice_meta.json` / `.sam3_meta.json` in `src/` (grep `read_slice_meta`, `training_geometry(`, `sidecar_for`, `stamped_*`) still gets the same values for a model published by the new code; (3) SAM3 publish: crash/kill between artifact promote and tiling-sidecar write, orphan cleanup with/without ownership, re-publish of the same run id, registry entries; (4) `read_tiling_meta` on malformed/partial/hostile JSON (lists where dicts expected, huge numbers, NaN, nested envelopes) must not raise; (5) profiles never lost or reordered on v2→v3. Do NOT stash/reset/edit the worktree; scratch under /tmp. Rank findings by severity with a reproducer each.

Run alongside it a normal whole-branch reviewer (`superpowers:requesting-code-review`).

- [ ] **Step 3: Fix wave** — failing test first per confirmed finding; re-run Step 1.

- [ ] **Step 4: Equivalence smoke (behavior must not move)** — S2 changes no inference path, so a 2-clip MPS smoke suffices here (full MPS + CUDA matrix is the S3 gate):

```bash
cd /Users/neurorishika/Projects/Rockefeller/Kronauer/multi-animal-tracker
pkill -u "$USER" -f "hydra|sleap" || true   # only stale sleap/hydra processes
REPO=$PWD WT=$PWD/.worktrees/sahi-s2 MAIN_SRC=$PWD/src WT_SRC=$PWD/.worktrees/sahi-s2/src \
  OUT=/tmp/equiv_sahi_s2 RUNTIME=mps bash tools/equivalence/run_matrix.sh fly_obb worm_bgsub
```
Expected: EQUIVALENT at the determinism floor for both clips, row counts > 1 (`wc -l` the CSVs).

- [ ] **Step 5: Merge**

```bash
git merge --no-ff feat/sahi-unify-s2 -m "Merge feat/sahi-unify-s2: v3 tiling sidecar and unified reader"
git worktree remove .worktrees/sahi-s2 && git branch -d feat/sahi-unify-s2
```

Then write the S3 plan (callers + F1–F4/F6/F8 + full MPS/CUDA equivalence gate), grounded in the merged code.
