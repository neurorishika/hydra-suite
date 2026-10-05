# SAM3 LoRA Augmentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Online, train-split-only, YOLO-parallel augmentation for SAM3 LoRA training. Image and polygons are transformed together, the run is configured through the existing `AugmentationProfile` nested on `Sam3LoraParams`, the settings are exposed in the DetectKit SAM3 tab, and the realised settings are stamped into the run dir and published sidecar.

**Architecture:** A pure module `training/sam3_lora/augment.py` owns the validation, op selection, per-tile RNG and `augment_tile`. The dataloader takes an optional `augmenter` callable that only the training loop passes, so validation, the probe and the detection-quality paths are untouched. Config, GUI and stamp plumbing reuse the existing SAM3 patterns (`scale_grouping` stamp, `_sam3_to_json`, panel `params()`/`set_params()`).

**Tech Stack:** Python 3.13, numpy, OpenCV, PySide6, pytest. The sidecar env (`hydra-sam3`) imports `hydra_suite.training.*`, so `augment.py` must import only numpy at module scope; cv2 and torch are imported lazily.

**Spec:** `docs/superpowers/specs/2026-10-05-sam3-lora-augmentation-design.md`

## Global Constraints

- Contract default `Sam3LoraParams().augmentation` = `AugmentationProfile(enabled=False)`, so old plans train exactly as before.
- GUI fresh-session default = `recommended_sam3_augmentation()`: enabled, fliplr 0.5, flipud 0.5, rot90 0.5, brightness 0.2, contrast 0.2, saturation 0.2, all else 0/False.
- Ranges: probabilities (`fliplr`, `flipud`, `rot90`, `decode_color_sim`, `resample_sim`) in [0, 1]; `rotate` in [0, 180]; `hue` in [0, 0.5]; `brightness`/`contrast`/`saturation` in [0, 1]. Unsupported for SAM3: `canonical_aug=True`, non-empty `label_expansion`, non-empty `args`.
- `rotate` border fill is `BORDER_CONSTANT` (114, 114, 114). Never reflect.
- Fragment floor is `params.min_area_ratio`, never the module constant.
- Per-tile RNG: `np.random.default_rng([epoch_seed % 2**32, image_id % 2**32, 0x5A3A06])`. Never the global `random` module.
- The augmenter is passed ONLY by the training loop. `collate_batches(val…)`, the autobatch probe and `detection_quality` stay augmenter-free.
- Stamp file name `hydra_sam3_augmentation.json`, written on both arms.
- Tests run in the `hydra-mps` env from the worktree. First run `python -c "import hydra_suite; print(hydra_suite.__file__)"` and confirm it prints a path under `.worktrees/sam3-aug/src`. If not, prefix commands with `PYTHONPATH=$PWD/src`.
- Commit as the configured git user. Formatting: `black` + `isort` on touched files before each commit.

## Review Focus

1. **A non-square edge tile under rot90**: w/h swap, polygons must still line up. Pinned in Task 2 (`test_rot90_non_square_tile_masks_align`).
2. **An old plan JSON / saved UI state with no `augmentation` key**: the CLI must train unaugmented, and the GUI must keep the panel's recommended default. Pinned in Task 1 (contract default), Task 5 (CLI load), Task 6 (dialog load).
3. **A hand-edited `spec.json` with an out-of-range value reaching the child**: the child must refuse loudly, not train. Pinned in Task 4 (`test_run_training_rejects_invalid_augmentation`).
4. **An instance rotated completely out of the tile**: it must be dropped, never kept as a zero-area object. Pinned in Task 2 (`test_rotate_drops_instance_fully_outside`).
5. **A grayscale-looking BGR tile with a colour concept**: the channel order must survive photometric ops. Pinned in Task 2 (`test_brightness_preserves_bgr_channel_order`).

---

### Task 1: Contract fields (`rot90`, nested `augmentation`) + reflective guards

**Files:**
- Modify: `src/hydra_suite/training/contracts.py` (`AugmentationProfile` ~L347, `Sam3LoraParams` ~L210)
- Modify: `tests/test_sam3_gui_cli_training_parity.py` (`_REFERENCE_KWARGS`), `tests/test_sam3_train_probe_phase.py`, `tests/test_sam3_slice_settings_shared.py`, `tests/test_geometry_drift_guard.py` — only as far as they fail
- Test: `tests/test_sam3_augmentation_contract.py` (create)

**Interfaces:**
- Produces: `AugmentationProfile.rot90: float = 0.0`; `Sam3LoraParams.augmentation: AugmentationProfile` (default `AugmentationProfile(enabled=False)`); `Sam3LoraParams.__post_init__` coerces a `dict` to `AugmentationProfile`.

- [ ] **Step 1: Write the failing test** `tests/test_sam3_augmentation_contract.py`:

```python
from dataclasses import asdict

from hydra_suite.training.contracts import AugmentationProfile, Sam3LoraParams


def test_rot90_field_defaults_off():
    assert AugmentationProfile().rot90 == 0.0


def test_sam3_params_default_augmentation_is_disabled():
    p = Sam3LoraParams()
    assert isinstance(p.augmentation, AugmentationProfile)
    assert p.augmentation.enabled is False


def test_default_instances_do_not_share_profile():
    a, b = Sam3LoraParams(), Sam3LoraParams()
    a.augmentation.fliplr = 0.9
    assert b.augmentation.fliplr == 0.0


def test_dict_round_trip_coerces_nested_profile():
    p = Sam3LoraParams(
        prompt="ant",
        augmentation=AugmentationProfile(enabled=True, fliplr=0.5, rot90=0.25),
    )
    q = Sam3LoraParams(**asdict(p))
    assert isinstance(q.augmentation, AugmentationProfile)
    assert q.augmentation == p.augmentation


def test_dict_without_augmentation_key_gets_disabled_profile():
    data = asdict(Sam3LoraParams(prompt="ant"))
    data.pop("augmentation")
    assert Sam3LoraParams(**data).augmentation.enabled is False
```

- [ ] **Step 2: Run it.** `python -m pytest tests/test_sam3_augmentation_contract.py -q`. Expected: FAIL (`rot90` / `augmentation` missing).

- [ ] **Step 3: Implement.** In `AugmentationProfile`, after `contrast`:

```python
    # P(rotate 90 degrees, CW/CCW 50/50). Label-exact for top-down views;
    # with fliplr/flipud covers all 8 dihedral symmetries. Consumed by SAM3
    # LoRA training only today (YOLO/classify ignore it, as they do `rotate`).
    rot90: float = 0.0
```

In `Sam3LoraParams`, add the last field and a `__post_init__`:

```python
    # Train-time tile augmentation (YOLO-parallel vocabulary; see
    # sam3_lora/augment.py for SAM3 semantics). Disabled by default so a plan
    # written before this field existed trains exactly as it did; the GUI's
    # fresh-session default is `augment.recommended_sam3_augmentation()`.
    augmentation: "AugmentationProfile" = field(
        default_factory=lambda: AugmentationProfile(enabled=False)
    )

    def __post_init__(self) -> None:
        # Every `Sam3LoraParams(**json_dict)` site (sidecar child, publish
        # CLI, dialog load, dataset-prep sidecar) hands the nested profile
        # over as a plain dict.
        if isinstance(self.augmentation, dict):
            self.augmentation = AugmentationProfile(**self.augmentation)
```

`AugmentationProfile` is defined later in the module. That is fine: the lambda and `__post_init__` resolve it at call time, and the annotation is a string under `from __future__ import annotations`.

- [ ] **Step 4: Run** the new test, then the reflective guards: `python -m pytest tests/test_sam3_augmentation_contract.py tests/test_sam3_gui_cli_training_parity.py tests/test_sam3_train_probe_phase.py tests/test_sam3_slice_settings_shared.py tests/test_geometry_drift_guard.py tests/test_augmentation_profile_canonical_copies.py tests/test_training_augmentation.py tests/test_sam3_contracts.py -q`. Fix only the guards that fail, by adding `augmentation` to their reference sets. In `_REFERENCE_KWARGS`, use a non-default value: `AugmentationProfile(enabled=True, fliplr=0.3, flipud=0.4, rot90=0.6, brightness=0.1)`. If the parity GUI driver cannot yet emit it (the panel has no controls until Task 6), mark ONLY that field's comparison `xfail(strict=True, reason="panel controls land in Task 6")`, and remove the xfail in Task 6.

- [ ] **Step 5: Commit** `feat(sam3): nested AugmentationProfile on Sam3LoraParams + rot90 field`.

---

### Task 2: Pure augmentation module

**Files:**
- Create: `src/hydra_suite/training/sam3_lora/augment.py`
- Test: `tests/test_sam3_augment.py`

**Interfaces:**
- Consumes: `AugmentationProfile` (Task 1); `hydra_suite.utils.slice_geometry.clip_polygon_to_tile(poly_px, (x0, y0, x1, y1)) -> ndarray | None`; `hydra_suite.training.augmentation.simulate_decode_color/simulate_resample(rgb_u8, prob, rng)`.
- Produces:
  - `Instance = tuple[np.ndarray, bool]`
  - `recommended_sam3_augmentation() -> AugmentationProfile`
  - `validate_sam3_augmentation(profile) -> list[str]`
  - `active_ops(profile) -> dict[str, float | bool]`; `is_active(profile) -> bool`
  - `tile_rng(epoch_seed: int, image_id: int) -> np.random.Generator`
  - `augment_tile(tile_bgr, instances, profile, rng, *, min_area_ratio: float) -> tuple[np.ndarray, list[Instance]]`
  - `make_tile_augmenter(profile, *, epoch_seed: int, min_area_ratio: float) -> Callable[[np.ndarray, list[Instance], int], tuple[np.ndarray, list[Instance]]] | None`
  - `AUGMENTATION_STAMP_FILENAME = "hydra_sam3_augmentation.json"`; `write_sam3_augmentation_stamp(run_dir, profile) -> Path | None`; `read_sam3_augmentation_stamp(run_dir) -> dict | None`

- [ ] **Step 1: Write failing tests** `tests/test_sam3_augment.py`:

```python
import json

import cv2
import numpy as np
import pytest

from hydra_suite.training.contracts import AugmentationProfile
from hydra_suite.training.sam3_lora import augment as aug


def _mask(poly, shape):
    m = np.zeros(shape[:2], np.uint8)
    # edge-convention polygon -> raster: shift by -0.5 to pixel-index space
    cv2.fillPoly(m, [np.round((poly - 0.5) * 16).astype(np.int32)], 1, shift=4)
    return m


def _tile(h=40, w=64):
    img = np.zeros((h, w, 3), np.uint8)
    poly = np.array([[5, 4], [20, 6], [18, 25], [6, 20]], np.float32)
    cv2.fillPoly(img, [np.round((poly - 0.5) * 16).astype(np.int32)], (0, 0, 255), shift=4)
    return img, poly


def _iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return inter / union


@pytest.mark.parametrize(
    "profile",
    [
        AugmentationProfile(enabled=True, fliplr=1.0),
        AugmentationProfile(enabled=True, flipud=1.0),
        AugmentationProfile(enabled=True, rot90=1.0),
        AugmentationProfile(enabled=True, fliplr=1.0, flipud=1.0, rot90=1.0),
    ],
)
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_geometric_ops_keep_image_and_polygon_aligned(profile, seed):
    img, poly = _tile()
    out, inst = aug.augment_tile(
        img, [(poly, False)], profile, np.random.default_rng(seed), min_area_ratio=0.1
    )
    painted = out[..., 2] > 127
    assert _iou(painted, _mask(inst[0][0], out.shape).astype(bool)) > 0.95


def test_rot90_non_square_tile_masks_align():
    img, poly = _tile(h=40, w=64)
    out, inst = aug.augment_tile(
        img, [(poly, False)], AugmentationProfile(enabled=True, rot90=1.0),
        np.random.default_rng(0), min_area_ratio=0.1,
    )
    assert out.shape[:2] == (64, 40)
    assert _iou(out[..., 2] > 127, _mask(inst[0][0], out.shape).astype(bool)) > 0.95


def test_rotate_uses_constant_fill_not_reflection():
    img = np.full((50, 50, 3), 255, np.uint8)
    out, _ = aug._rotate(img, [], 45.0, 0.1)
    assert (out[0, 0] == (114, 114, 114)).all()


def test_rotate_keeps_alignment_for_interior_instance():
    img, poly = _tile(64, 64)
    poly = poly + 18
    img = np.zeros((64, 64, 3), np.uint8)
    cv2.fillPoly(img, [np.round((poly - 0.5) * 16).astype(np.int32)], (0, 0, 255), shift=4)
    out, inst = aug.augment_tile(
        img, [(poly, False)], AugmentationProfile(enabled=True, rotate=30.0),
        np.random.default_rng(1), min_area_ratio=0.1,
    )
    assert _iou(out[..., 2] > 127, _mask(inst[0][0], out.shape).astype(bool)) > 0.9
    assert inst[0][1] is False


def test_rotate_marks_sub_floor_fragment_crowd_and_keeps_existing_crowd():
    img = np.zeros((40, 40, 3), np.uint8)
    corner = np.array([[0, 0], [6, 0], [6, 6], [0, 6]], np.float32)
    centre = np.array([[15, 15], [25, 15], [25, 25], [15, 25]], np.float32)
    out, inst = aug._rotate(img, [(corner, False), (centre, True)], 45.0, 0.9)
    assert len(inst) == 2
    assert all(flag for _, flag in inst)  # fragment -> crowd; crowd stays crowd
    _, inst_lo = aug._rotate(img, [(centre, False)], 45.0, 0.9)
    assert inst_lo[0][1] is False  # fully retained interior instance stays positive


def test_rotate_drops_instance_fully_outside():
    img = np.zeros((40, 40, 3), np.uint8)
    corner = np.array([[0, 0], [1.5, 0], [1.5, 1.5], [0, 1.5]], np.float32)
    _, inst = aug._rotate(img, [(corner, False)], 45.0, 0.1)
    assert inst == []


def test_photometric_ops_leave_polygons_bit_identical():
    img, poly = _tile()
    profile = AugmentationProfile(
        enabled=True, brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1,
        decode_color_sim=1.0, resample_sim=0.0,
    )
    _, inst = aug.augment_tile(img, [(poly, False)], profile, np.random.default_rng(0), min_area_ratio=0.1)
    np.testing.assert_array_equal(inst[0][0], poly)


def test_brightness_preserves_bgr_channel_order():
    img = np.zeros((8, 8, 3), np.uint8)
    img[..., 2] = 200  # pure red in BGR
    out, _ = aug.augment_tile(
        img, [], AugmentationProfile(enabled=True, brightness=0.5),
        np.random.default_rng(0), min_area_ratio=0.1,
    )
    assert out[..., 0].max() == 0 and out[..., 1].max() == 0 and out[..., 2].min() > 0


def test_same_seed_same_output_different_epoch_differs():
    img, poly = _tile()
    p = aug.recommended_sam3_augmentation()
    a = aug.make_tile_augmenter(p, epoch_seed=7, min_area_ratio=0.1)
    b = aug.make_tile_augmenter(p, epoch_seed=7, min_area_ratio=0.1)
    outs_a = [a(img, [(poly, False)], i)[0] for i in range(8)]
    outs_b = [b(img, [(poly, False)], i)[0] for i in reversed(range(8))][::-1]
    for x, y in zip(outs_a, outs_b):
        np.testing.assert_array_equal(x, y)  # order-independent
    c = aug.make_tile_augmenter(p, epoch_seed=8, min_area_ratio=0.1)
    assert any(not np.array_equal(c(img, [(poly, False)], i)[0], outs_a[i]) for i in range(8))


@pytest.mark.parametrize(
    "profile",
    [
        AugmentationProfile(enabled=False, fliplr=1.0),
        AugmentationProfile(enabled=True),
        AugmentationProfile(enabled=False),
    ],
)
def test_inactive_profiles_yield_no_augmenter(profile):
    assert aug.make_tile_augmenter(profile, epoch_seed=0, min_area_ratio=0.1) is None
    assert aug.is_active(profile) is False


def test_recommended_profile_values():
    p = aug.recommended_sam3_augmentation()
    assert (p.enabled, p.fliplr, p.flipud, p.rot90) == (True, 0.5, 0.5, 0.5)
    assert (p.brightness, p.contrast, p.saturation) == (0.2, 0.2, 0.2)
    assert (p.hue, p.rotate, p.decode_color_sim, p.resample_sim, p.monochrome) == (0.0, 0.0, 0.0, 0.0, False)
    assert aug.recommended_sam3_augmentation() is not p


@pytest.mark.parametrize(
    "kwargs, fragment",
    [
        ({"fliplr": 1.5}, "fliplr"),
        ({"rot90": -0.1}, "rot90"),
        ({"rotate": 181.0}, "rotate"),
        ({"hue": 0.6}, "hue"),
        ({"brightness": float("nan")}, "brightness"),
        ({"canonical_aug": True}, "canonical_aug"),
        ({"args": {"mosaic": 1.0}}, "args"),
        ({"label_expansion": {"fliplr": {"a": "b"}}}, "label_expansion"),
    ],
)
def test_validation_rejects(kwargs, fragment):
    errors = aug.validate_sam3_augmentation(AugmentationProfile(enabled=True, **kwargs))
    assert any(fragment in e for e in errors)


def test_validation_accepts_recommended_and_disabled():
    assert aug.validate_sam3_augmentation(aug.recommended_sam3_augmentation()) == []
    assert aug.validate_sam3_augmentation(AugmentationProfile(enabled=False)) == []


def test_stamp_both_arms(tmp_path):
    aug.write_sam3_augmentation_stamp(tmp_path / "on", aug.recommended_sam3_augmentation())
    on = aug.read_sam3_augmentation_stamp(tmp_path / "on")
    assert on["applied"]["augmentation"] is True
    assert on["applied"]["ops"]["fliplr"] == 0.5
    aug.write_sam3_augmentation_stamp(tmp_path / "off", AugmentationProfile(enabled=False))
    off = aug.read_sam3_augmentation_stamp(tmp_path / "off")
    assert off["applied"] == {"augmentation": False, "ops": {}, "reason": "disabled"}
    aug.write_sam3_augmentation_stamp(tmp_path / "zero", AugmentationProfile(enabled=True))
    assert aug.read_sam3_augmentation_stamp(tmp_path / "zero")["applied"]["reason"] == "no active ops"
    assert aug.read_sam3_augmentation_stamp(tmp_path / "missing") is None
```

- [ ] **Step 2: Run.** `python -m pytest tests/test_sam3_augment.py -q`. Expected: FAIL (module missing).

- [ ] **Step 3: Implement** `src/hydra_suite/training/sam3_lora/augment.py`:

```python
"""Train-time tile augmentation for SAM3 LoRA finetuning.

Pure and Qt-free; numpy only at module scope (cv2/torch lazily) because the
`hydra-sam3` sidecar imports this. Uses the YOLO-side `AugmentationProfile`
vocabulary -- see docs/superpowers/specs/2026-10-05-sam3-lora-augmentation-design.md
for the SAM3 semantics of each field.

Image and polygons are transformed TOGETHER in tile-pixel, edge-convention
coordinates (the convention `datapoints._scale_polygons_to_res` assumes),
before the RES resize and Meta's NormalizeAPI. Only the training loop builds
an augmenter; validation, the autobatch probe and detection-quality never do.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import numpy as np

from hydra_suite.training.contracts import AugmentationProfile

logger = logging.getLogger(__name__)

Instance = tuple[np.ndarray, bool]
TileAugmenter = Callable[[np.ndarray, list, int], tuple[np.ndarray, list]]

AUGMENTATION_STAMP_FILENAME = "hydra_sam3_augmentation.json"
# Decorrelates per-tile draws from the epoch-shuffle RNG seeded by the same
# epoch seed.
_RNG_SALT = 0x5A3A06
# Ultralytics' letterbox gray. A constant fill, never BORDER_REFLECT: a
# reflected border pastes UNLABELED mirrored animals into an exhaustive query.
_FILL_BGR = (114, 114, 114)

_PROBABILITY_FIELDS = ("fliplr", "flipud", "rot90", "decode_color_sim", "resample_sim")
_UNIT_FIELDS = ("brightness", "contrast", "saturation")
_NUMERIC_OPS = _PROBABILITY_FIELDS + _UNIT_FIELDS + ("rotate", "hue")


def recommended_sam3_augmentation() -> AugmentationProfile:
    """The GUI's fresh-session default: label-exact geometry + mild photometrics.

    `hue` stays 0 because colour can BE the concept (painted tags); `rotate`
    stays 0 because it needs border fill and fragment re-flagging.
    """
    return AugmentationProfile(
        enabled=True,
        fliplr=0.5,
        flipud=0.5,
        rot90=0.5,
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
    )


def validate_sam3_augmentation(profile: AugmentationProfile) -> list[str]:
    """Range + supported-field errors; empty when valid."""
    errors: list[str] = []

    def _check(name: str, lo: float, hi: float) -> None:
        try:
            value = float(getattr(profile, name))
        except (TypeError, ValueError):
            errors.append(f"augmentation.{name} must be a number")
            return
        if not (lo <= value <= hi):  # also rejects NaN
            errors.append(f"augmentation.{name} must be in [{lo:g}, {hi:g}], got {value!r}")

    for name in _PROBABILITY_FIELDS + _UNIT_FIELDS:
        _check(name, 0.0, 1.0)
    _check("rotate", 0.0, 180.0)
    _check("hue", 0.0, 0.5)
    if profile.canonical_aug:
        errors.append("augmentation.canonical_aug is not supported for SAM3 training")
    if profile.label_expansion:
        errors.append("augmentation.label_expansion is not supported for SAM3 training")
    if profile.args:
        errors.append(
            "augmentation.args (Ultralytics passthrough) is not supported for SAM3 training"
        )
    return errors


def active_ops(profile: AugmentationProfile | None) -> dict[str, Any]:
    """The ops a profile actually applies (empty when disabled)."""
    if profile is None or not profile.enabled:
        return {}
    ops: dict[str, Any] = {
        name: float(getattr(profile, name))
        for name in _NUMERIC_OPS
        if float(getattr(profile, name)) > 0.0
    }
    if profile.monochrome:
        ops["monochrome"] = True
    return ops


def is_active(profile: AugmentationProfile | None) -> bool:
    return bool(active_ops(profile))


def tile_rng(epoch_seed: int, image_id: int) -> np.random.Generator:
    """Per-(epoch, tile) generator: independent of shuffle/grouping order."""
    return np.random.default_rng(
        [int(epoch_seed) % 2**32, int(image_id) % 2**32, _RNG_SALT]
    )


def _polygon_area(poly: np.ndarray) -> float:
    x = poly[:, 0].astype(np.float64)
    y = poly[:, 1].astype(np.float64)
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _map(instances: list[Instance], fn) -> list[Instance]:
    return [(fn(poly).astype(np.float32), crowd) for poly, crowd in instances]


def _rotate(
    img: np.ndarray, instances: list[Instance], angle: float, min_area_ratio: float
) -> tuple[np.ndarray, list[Instance]]:
    import cv2

    from hydra_suite.utils.slice_geometry import clip_polygon_to_tile

    h, w = img.shape[:2]
    # cv2 maps pixel INDICES; rotating about the index centre equals rotating
    # about (w/2, h/2) in edge coordinates. Polygons: edge -> index -> edge.
    matrix = cv2.getRotationMatrix2D(((w - 1) / 2.0, (h - 1) / 2.0), angle, 1.0)
    img = cv2.warpAffine(
        img,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=_FILL_BGR,
    )
    out: list[Instance] = []
    for poly, crowd in instances:
        idx = poly.astype(np.float64) - 0.5
        rotated = idx @ matrix[:, :2].T + matrix[:, 2] + 0.5
        before = _polygon_area(rotated)
        clipped = clip_polygon_to_tile(rotated, (0, 0, w, h))
        if clipped is None or before <= 0.0:
            continue  # rotated out of the image: genuinely absent now
        retained = _polygon_area(clipped) / before
        if retained <= 0.0:
            continue
        # Reference area is the TILE polygon (dataset_build's full-frame area
        # is unavailable here). Sub-floor -> is_crowd, which
        # `datapoints.select_output_objects` excludes AND uses to mark the
        # positive query non-exhaustive. Already-crowd stays crowd.
        out.append(
            (
                np.asarray(clipped, dtype=np.float32),
                bool(crowd or retained < float(min_area_ratio)),
            )
        )
    return img, out


def _photometric(rgb: np.ndarray, p: AugmentationProfile, rng) -> np.ndarray:
    """Same semantics as `runner._apply_tiny_augmentation`, explicit RNG."""
    import cv2

    if p.brightness > 0:
        factor = rng.uniform(max(0.0, 1.0 - p.brightness), 1.0 + p.brightness)
        rgb = np.clip(rgb.astype(np.float32) * factor, 0, 255).astype(np.uint8)
    if p.contrast > 0:
        factor = rng.uniform(max(0.0, 1.0 - p.contrast), 1.0 + p.contrast)
        mean = rgb.mean(axis=(0, 1), keepdims=True)
        rgb = np.clip((rgb.astype(np.float32) - mean) * factor + mean, 0, 255).astype(
            np.uint8
        )
    if p.saturation > 0 or p.hue > 0:
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
        if p.saturation > 0:
            factor = rng.uniform(max(0.0, 1.0 - p.saturation), 1.0 + p.saturation)
            hsv[..., 1] = np.clip(hsv[..., 1] * factor, 0, 255)
        if p.hue > 0:
            hsv[..., 0] = (hsv[..., 0] + rng.uniform(-p.hue, p.hue) * 179.0) % 180.0
        rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    if p.monochrome:
        rgb = cv2.cvtColor(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
    if p.decode_color_sim > 0 or p.resample_sim > 0:
        from hydra_suite.training.augmentation import (
            simulate_decode_color,
            simulate_resample,
        )

        if p.decode_color_sim > 0:
            rgb = simulate_decode_color(rgb, float(p.decode_color_sim), rng)
        if p.resample_sim > 0:
            rgb = simulate_resample(rgb, float(p.resample_sim), rng)
    return rgb


_PHOTOMETRIC = ("brightness", "contrast", "saturation", "hue", "decode_color_sim", "resample_sim")


def augment_tile(
    tile_bgr: np.ndarray,
    instances: list[Instance],
    profile: AugmentationProfile,
    rng: np.random.Generator,
    *,
    min_area_ratio: float,
) -> tuple[np.ndarray, list[Instance]]:
    """Apply `profile` to one BGR tile and its edge-convention polygons."""
    import cv2

    img = tile_bgr
    polys: list[Instance] = [
        (np.asarray(poly, dtype=np.float32).reshape(-1, 2), bool(crowd))
        for poly, crowd in instances
    ]
    h, w = img.shape[:2]
    if profile.flipud > 0 and rng.random() < profile.flipud:
        img = cv2.flip(img, 0)
        polys = _map(polys, lambda p: np.column_stack([p[:, 0], h - p[:, 1]]))
    if profile.fliplr > 0 and rng.random() < profile.fliplr:
        img = cv2.flip(img, 1)
        polys = _map(polys, lambda p: np.column_stack([w - p[:, 0], p[:, 1]]))
    if profile.rot90 > 0 and rng.random() < profile.rot90:
        h, w = img.shape[:2]
        if rng.random() < 0.5:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            polys = _map(polys, lambda p: np.column_stack([h - p[:, 1], p[:, 0]]))
        else:
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            polys = _map(polys, lambda p: np.column_stack([p[:, 1], w - p[:, 0]]))
    if profile.rotate > 0:
        angle = float(rng.uniform(-profile.rotate, profile.rotate))
        if angle != 0.0 and math.isfinite(angle):
            img, polys = _rotate(img, polys, angle, min_area_ratio)
    if profile.monochrome or any(getattr(profile, n) > 0 for n in _PHOTOMETRIC):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rgb = _photometric(rgb, profile, rng)
        img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return np.ascontiguousarray(img), polys


def make_tile_augmenter(
    profile: AugmentationProfile | None, *, epoch_seed: int, min_area_ratio: float
) -> TileAugmenter | None:
    """`None` for a disabled/no-op profile, so that arm is literally today's path."""
    if not is_active(profile):
        return None

    def _augment(tile_bgr, instances, image_id):
        return augment_tile(
            tile_bgr,
            instances,
            profile,
            tile_rng(epoch_seed, image_id),
            min_area_ratio=min_area_ratio,
        )

    return _augment


def write_sam3_augmentation_stamp(
    run_dir: str | Path, profile: AugmentationProfile | None
) -> Path | None:
    """Requested-vs-applied record, written on BOTH arms (scale-grouping shape)."""
    profile = profile or AugmentationProfile(enabled=False)
    ops = active_ops(profile)
    reason = "" if ops else ("disabled" if not profile.enabled else "no active ops")
    payload = {
        "requested": asdict(profile),
        "applied": {"augmentation": bool(ops), "ops": ops, "reason": reason},
    }
    try:
        directory = Path(run_dir)
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / AUGMENTATION_STAMP_FILENAME
        target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", "utf-8")
        return target
    except OSError:
        logger.warning(
            "Failed to write SAM3 augmentation stamp to %s; the run continues, "
            "but its published sidecar will not record what augmentation ran.",
            Path(run_dir) / AUGMENTATION_STAMP_FILENAME,
        )
        return None


def read_sam3_augmentation_stamp(run_dir: str | Path) -> dict[str, Any] | None:
    try:
        data = json.loads(
            (Path(run_dir) / AUGMENTATION_STAMP_FILENAME).read_text(encoding="utf-8")
        )
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None
```

- [ ] **Step 4: Run** `python -m pytest tests/test_sam3_augment.py -q`. Expected: PASS. If a geometric alignment test fails, fix the transform, never the threshold. Also check the module imports with cv2 blocked: `python -c "import sys; sys.modules['cv2']=None; import hydra_suite.training.sam3_lora.augment"` must succeed.

- [ ] **Step 5: Commit** `feat(sam3): pure tile augmentation module`.

---

### Task 3: Dataloader threading (train arms only) + disabled-path identity

**Files:**
- Modify: `src/hydra_suite/training/sam3_lora/dataloader.py` (`load_datapoints`, `collate_batches`, `collate_epoch_batches`, `_grouped_epoch_batches`)
- Test: `tests/test_sam3_dataloader.py` (append)

**Interfaces:**
- Consumes: `TileAugmenter` (Task 2).
- Produces: `load_datapoints(descriptor, transform, augmenter=None)`, `collate_batches(descriptors, batch_size, augmenter=None)`, `collate_epoch_batches(descriptors, batch_size, *, seed, group_by_scale=False, augmenter=None)`.

- [ ] **Step 1: Append failing tests** to `tests/test_sam3_dataloader.py`:

```python
def _descs(n=5, group=""):
    return [
        dl.TileDescriptor(
            image_id=i,
            image_path=f"t{i}.png",
            positive_prompt="ant",
            negative_prompts=(),
            instances=(dl.InstanceDescriptor(polygon=((1, 1), (5, 1), (5, 5)), is_crowd=False),),
            width=16,
            height=16,
            scale_group=group,
        )
        for i in range(n)
    ]


def _capture(monkeypatch):
    seen = []

    def fake_imread(path):
        img = np.zeros((16, 16, 3), np.uint8)
        img[2:6, 2:6, 2] = 255
        return img

    def fake_build(tile_bgr, prompt, instances, negatives, transform):
        seen.append((tile_bgr.copy(), [(p.copy(), c) for p, c in instances]))
        return [len(seen)]

    monkeypatch.setattr(dl.cv2, "imread", fake_imread)
    monkeypatch.setattr(dl, "_default_transform", lambda: object())
    monkeypatch.setattr(dl, "build_shared_query_datapoints", fake_build)
    monkeypatch.setattr(dl, "collate_datapoints", lambda values: list(values))
    return seen


@pytest.mark.parametrize("group_by_scale", [False, True])
def test_disabled_augmentation_is_byte_identical(monkeypatch, group_by_scale):
    from hydra_suite.training.contracts import AugmentationProfile, Sam3LoraParams
    from hydra_suite.training.sam3_lora.augment import make_tile_augmenter

    for profile in (Sam3LoraParams().augmentation, AugmentationProfile(enabled=True)):
        seen_none = _capture(monkeypatch)
        list(dl.collate_epoch_batches(_descs(group="a"), 2, seed=3, group_by_scale=group_by_scale))
        seen_off = _capture(monkeypatch)
        augmenter = make_tile_augmenter(profile, epoch_seed=3, min_area_ratio=0.1)
        list(
            dl.collate_epoch_batches(
                _descs(group="a"), 2, seed=3, group_by_scale=group_by_scale, augmenter=augmenter
            )
        )
        assert len(seen_none) == len(seen_off)
        for (img_a, inst_a), (img_b, inst_b) in zip(seen_none, seen_off):
            assert img_a.tobytes() == img_b.tobytes()
            for (pa, ca), (pb, cb) in zip(inst_a, inst_b):
                assert pa.tobytes() == pb.tobytes() and ca == cb


@pytest.mark.parametrize("group_by_scale", [False, True])
def test_epoch_batches_apply_augmenter_with_image_id(monkeypatch, group_by_scale):
    seen = _capture(monkeypatch)
    calls = []

    def augmenter(tile, instances, image_id):
        calls.append(image_id)
        return tile[:, ::-1].copy(), instances

    list(dl.collate_epoch_batches(_descs(group="g"), 2, seed=0, group_by_scale=group_by_scale, augmenter=augmenter))
    assert sorted(calls) == list(range(5))
    assert all(img[2:6, 10:14, 2].all() for img, _ in seen)


def test_collate_batches_default_has_no_augmenter(monkeypatch):
    import inspect

    assert inspect.signature(dl.collate_batches).parameters["augmenter"].default is None
```

- [ ] **Step 2: Run.** `python -m pytest tests/test_sam3_dataloader.py -q`. Expected: the new tests FAIL (`augmenter` kwarg unknown).

- [ ] **Step 3: Implement.** In `load_datapoints`, add `augmenter: Any = None`. After building `instances`, add:

```python
    if augmenter is not None:
        # Train arm only (see `collate_epoch_batches`); image and polygons
        # move together, before the RES resize inside build_tile_datapoint.
        tile_bgr, instances = augmenter(tile_bgr, instances, descriptor.image_id)
```

In `collate_batches(descriptors, batch_size, augmenter: Any = None)`, replace the `load_datapoints(descriptor, transform)` call with:

```python
        loaded = (
            load_datapoints(descriptor, transform)
            if augmenter is None
            else load_datapoints(descriptor, transform, augmenter)
        )
        for datapoint in loaded:
```

The two-arity call keeps existing tests that monkeypatch `load_datapoints` with a two-argument lambda valid, and keeps the off arm literally unchanged. Add `augmenter: Any = None` to `collate_epoch_batches` and `_grouped_epoch_batches`, and forward it to every `collate_batches` call in both arms. Add one docstring sentence to `collate_epoch_batches`: "``augmenter`` is passed only by the training loop; validation, the autobatch probe and detection-quality call `collate_batches` without one."

- [ ] **Step 4: Run** `python -m pytest tests/test_sam3_dataloader.py tests/test_sam3_detection_quality.py tests/test_sam3_streaming_memory.py -q`. Expected: PASS. Then `grep -n "collate_batches(\|collate_epoch_batches(" src/hydra_suite/training/sam3_lora/*.py` and confirm that only the training loop will pass `augmenter=` (wired in Task 4).

- [ ] **Step 5: Commit** `feat(sam3): thread optional tile augmenter through train batching`.

---

### Task 4: Training-loop wiring, child-side validation, stamp + publish sidecar

**Files:**
- Modify: `src/hydra_suite/training/sam3_lora/cli.py` (imports ~L100; after `params = spec.sam3_params` ~L1270; epoch loop ~L1399)
- Modify: `src/hydra_suite/training/sam3_lora/publish_worker.py` (`_scale_metadata` merge ~L114)
- Test: `tests/test_sam3_cli.py` and `tests/test_sam3_publish_sidecar.py` (append; reuse their existing fake-training/publish harnesses — read them first)

**Interfaces:**
- Consumes: `validate_sam3_augmentation`, `make_tile_augmenter`, `write_sam3_augmentation_stamp`, `read_sam3_augmentation_stamp`, `active_ops` (Task 2); `collate_epoch_batches(..., augmenter=)` (Task 3).
- Produces: the sidecar metadata key `"augmentation": {"requested": {...}, "applied": {...}}`, present only when the stamp exists.

- [ ] **Step 1: Write failing tests.** Find the harness in `tests/test_sam3_cli.py` that drives the training function with a fake model/loss (grep `collate_epoch_batches` / `monkeypatch.setattr(sam3_cli`). Add:
  - `test_run_training_passes_augmenter_per_epoch`: monkeypatch `sam3_cli.collate_epoch_batches` with a recorder that returns `iter(())` (or the harness's fake batches). Run with `sam3_params.augmentation = recommended_sam3_augmentation()` and `epochs=2`. Assert that every call received a callable `augmenter`, and that the two epochs' augmenters are distinct objects. Assert that `hydra_sam3_augmentation.json` exists in the run dir with `applied.augmentation is True`.
  - `test_run_training_default_params_pass_no_augmenter`: same with the default params. Assert `augmenter is None` in every call and the stamp says `reason == "disabled"`.
  - `test_run_training_rejects_invalid_augmentation`: `augmentation=AugmentationProfile(enabled=True, fliplr=2.0)`. Assert `RuntimeError` matching `"augmentation.fliplr"` is raised before any `collate_epoch_batches` call.
  - Validation stays clean: assert the validation `collate_batches` call never receives `augmenter` (record its kwargs).

  In `tests/test_sam3_publish_sidecar.py` add `test_sidecar_carries_realised_augmentation_stamp`: write a stamp into the run dir with `write_sam3_augmentation_stamp(run_dir, recommended_sam3_augmentation())` and assert `_scale_metadata(manifest, run_dir)["augmentation"]["applied"]["augmentation"] is True`. Also assert that a run dir without the stamp has no `"augmentation"` key.

- [ ] **Step 2: Run them.** Expected: FAIL.

- [ ] **Step 3: Implement.** In `cli.py`, import from `.augment` (`active_ops`, `make_tile_augmenter`, `validate_sam3_augmentation`, `write_sam3_augmentation_stamp`). Immediately after `params = spec.sam3_params` in the training entry (~L1270), add:

```python
    augmentation = params.augmentation
    augmentation_errors = validate_sam3_augmentation(augmentation)
    if augmentation_errors:
        # A hand-edited spec.json can bypass plan validation; refuse rather
        # than train on a silently clamped or ignored setting.
        raise RuntimeError(
            "Invalid SAM3 augmentation settings: " + "; ".join(augmentation_errors)
        )
```

Next to `write_sam3_scale_grouping_stamp(...)`, add:

```python
    write_sam3_augmentation_stamp(run_dir_path, augmentation)
    emit_log(
        "augmentation "
        + (
            "ON: " + ", ".join(f"{k}={v}" for k, v in active_ops(augmentation).items())
            if active_ops(augmentation)
            else "OFF"
        )
    )
```

In the epoch loop:

```python
        epoch_batches = collate_epoch_batches(
            train_descriptors,
            batch_size,
            seed=spec.seed + epoch,
            group_by_scale=group_by_scale,
            # Fresh per epoch: draws are keyed on (epoch seed, image_id), so
            # each epoch sees new augmentations, reproducibly. None when the
            # profile is disabled/no-op -> literally the unaugmented path.
            augmenter=make_tile_augmenter(
                augmentation,
                epoch_seed=spec.seed + epoch,
                min_area_ratio=float(params.min_area_ratio),
            ),
        )
```

If `emit_log` is not in scope at that point, use whatever logging call the surrounding scale-grouping block uses. In `publish_worker.py`, add the function below and merge it next to `**_scale_grouping_metadata(run_dir)`:

```python
def _augmentation_metadata(run_dir: "Path | None") -> dict[str, Any]:
    """The REALISED augmentation block (from the run-dir stamp), or {}."""
    if run_dir is None:
        return {}
    stamp = read_sam3_augmentation_stamp(run_dir)
    if stamp is None:
        return {}
    return {
        "augmentation": {
            "requested": dict(stamp.get("requested") or {}),
            "applied": dict(stamp.get("applied") or {}),
        }
    }
```

- [ ] **Step 4: Run** `python -m pytest tests/test_sam3_cli.py tests/test_sam3_publish_sidecar.py tests/test_sam3_publish.py tests/test_sam3_train.py tests/test_sam3_early_stopping.py tests/test_sam3_train_probe_phase.py -q`. Expected: PASS.

- [ ] **Step 5: Commit** `feat(sam3): wire augmentation into the training loop + realised stamp`.

---

### Task 5: Plan config parsing + validation (CLI/JSON)

**Files:**
- Modify: `src/hydra_suite/detectkit/config/training.py` (augmentation parse ~L448-485; `sam3` parse ~L525-580; `validate()` ~L690)
- Test: `tests/test_sam3_augmentation_plan_config.py` (create). Use `tests/test_sam3_gui_cli_training_parity.py::_cli_plan_payload` as the template for a minimal valid plan dict.

**Interfaces:**
- Consumes: `validate_sam3_augmentation` (Task 2), `AugmentationProfile.rot90` (Task 1).
- Produces: `_parse_augmentation_profile(values: dict, label: str) -> AugmentationProfile` (module-private), used for both `training.augmentation` and `sam3.augmentation`.

- [ ] **Step 1: Write failing tests**:
  - A plan with `sam3.augmentation = {"enabled": true, "fliplr": 0.5, "rot90": 0.5, "brightness": 0.2}` loads (`load_training_plan`). `plan.sam3_params.augmentation` is an `AugmentationProfile` with those values, and `plan.to_dict()` → JSON → `from_dict` round-trips to an equal profile.
  - A plan with no `sam3.augmentation` loads with `augmentation.enabled is False`.
  - `sam3.augmentation.fliplr = 2` → `plan.validate()` raises `TrainingPlanError` mentioning `sam3.augmentation.fliplr`. The same for `args: {"mosaic": 1}` and `canonical_aug: true`.
  - `sam3.augmentation.fliplr = "yes"` → `TrainingPlanError` (type) at load. An unknown key `sam3.augmentation.mosaic` → `TrainingPlanError` at load.
  - `training.augmentation.rot90 = 0.5` loads into the plan-level profile.

- [ ] **Step 2: Run.** Expected: FAIL.

- [ ] **Step 3: Implement.** Extract the existing `training.augmentation` typing block into:

```python
def _parse_augmentation_profile(raw: object, label: str) -> AugmentationProfile:
    values = _require_mapping(raw, label)
    for name in ("enabled", "canonical_aug", "monochrome"):
        if name in values:
            values[name] = _require_bool(values[name], f"{label}.{name}")
    if "canonical_aug_copies" in values:
        values["canonical_aug_copies"] = _require_int(
            values["canonical_aug_copies"], f"{label}.canonical_aug_copies"
        )
    for name in (
        "flipud", "fliplr", "rot90", "rotate", "hue", "saturation",
        "brightness", "contrast", "decode_color_sim", "resample_sim",
    ):
        if name in values:
            values[name] = _require_number(values[name], f"{label}.{name}")
    for name in ("args", "label_expansion"):
        if name in values:
            values[name] = _require_mapping(values[name], f"{label}.{name}")
    return _construct_dataclass(AugmentationProfile, values, label)
```

Call it for `training.augmentation` (same behaviour as before, now with `rot90`). In the `sam3` block, add `if "augmentation" in sam3_values: sam3_values["augmentation"] = _parse_augmentation_profile(sam3_values["augmentation"], "sam3.augmentation")`. Check that `_construct_dataclass` rejects unknown keys; if it does not, reject them explicitly in the helper. In `validate()`, inside the SAM3 branch after the existing checks:

```python
            from hydra_suite.training.sam3_lora.augment import (
                validate_sam3_augmentation,
            )

            augmentation_errors = validate_sam3_augmentation(
                self.sam3_params.augmentation
            )
            if augmentation_errors:
                raise TrainingPlanError(
                    "; ".join(f"sam3.{error}" for error in augmentation_errors)
                )
```

Confirm that `_sam3_to_json` emits `augmentation` as a dict (`asdict` recurses) and that `_validate_training_plan_json_shape` tolerates the nesting depth; adjust its depth limit only if a test proves it rejects the payload.

- [ ] **Step 4: Run** the new file plus `python -m pytest tests/test_sam3_gui_cli_training_parity.py tests/test_sam3_params_threading.py tests/test_sam3_role_plumbing.py -q -k "not gui"` (or the full files if they don't need a display). Expected: PASS.

- [ ] **Step 5: Commit** `feat(detectkit): sam3.augmentation plan config + validation`.

---

### Task 6: SAM3 panel controls, dialog persistence, GUI/CLI parity, docs

**Files:**
- Modify: `src/hydra_suite/detectkit/gui/panels/sam3_training_panel.py` (`_build_ui`, `params()`, `set_params()`)
- Modify: `src/hydra_suite/detectkit/gui/dialogs/training_dialog.py` (persisted-state load ~L3459-3470)
- Modify: `tests/test_sam3_gui_cli_training_parity.py` (remove the Task 1 xfail; add the documented divergence)
- Modify: `docs/user-guide/detectkit.md` (SAM3 section)
- Test: `tests/test_sam3_training_panel.py` (append)

**Interfaces:**
- Consumes: `recommended_sam3_augmentation()` (Task 2), `Sam3LoraParams.augmentation` (Task 1).
- Produces: panel attributes `aug_group` (checkable `QGroupBox`), `aug_fliplr`, `aug_flipud`, `aug_rot90`, `aug_rotate`, `aug_brightness`, `aug_contrast`, `aug_saturation`, `aug_hue`, `aug_decode_color_sim`, `aug_resample_sim` (`QDoubleSpinBox`), `aug_monochrome` (`QCheckBox`).

- [ ] **Step 1: Write failing tests** in `tests/test_sam3_training_panel.py`, following its existing panel fixture:
  - A fresh panel: `panel.params().augmentation == recommended_sam3_augmentation()`.
  - Round-trip: `panel.set_params(Sam3LoraParams(prompt="ant", augmentation=AugmentationProfile(enabled=True, fliplr=0.1, flipud=0.2, rot90=0.3, rotate=10.0, brightness=0.4, contrast=0.5, saturation=0.6, hue=0.05, decode_color_sim=0.7, resample_sim=0.8, monochrome=True)))`, then `panel.params().augmentation` equals that profile.
  - `set_params` with `enabled=False` unchecks the group, and `params().augmentation.enabled is False`.
  - Spin ranges: `aug_rotate.maximum() == 180`, `aug_hue.maximum() == 0.5`, probability spins `maximum() == 1`.
  - Dialog persisted-state load (use the harness in `tests/test_sam3_dialog_wiring.py`): a saved `sam3` dict WITHOUT `augmentation` leaves the panel at the recommended profile. A saved dict WITH an `augmentation` dict applies it.
  In the parity test: remove the xfail. Add `test_augmentation_default_is_a_documented_intentional_divergence`, asserting that a fresh GUI panel emits `recommended_sam3_augmentation()` while a CLI plan without `sam3.augmentation` yields `enabled=False`. Explain in its docstring why (old plans must train unchanged), mirroring the `auto_import` test.

- [ ] **Step 2: Run.** Expected: FAIL.

- [ ] **Step 3: Implement.** In `_build_ui`, add after the "Optimisation" group (match the file's existing group/layout and tooltip idiom):

```python
        self.aug_group = QGroupBox("Augmentation")
        self.aug_group.setCheckable(True)
        self.aug_group.setToolTip(
            "Train-time augmentation of SAM3 tiles (training split only; "
            "validation is never augmented). Image and polygons are "
            "transformed together. Set flips to 0 for chiral concepts."
        )
        aug_form = QFormLayout(self.aug_group)

        def _aug_spin(maximum: float, step: float, decimals: int = 2) -> QDoubleSpinBox:
            spin = QDoubleSpinBox()
            spin.setRange(0.0, maximum)
            spin.setSingleStep(step)
            spin.setDecimals(decimals)
            return spin

        self.aug_fliplr = _aug_spin(1.0, 0.05)
        self.aug_flipud = _aug_spin(1.0, 0.05)
        self.aug_rot90 = _aug_spin(1.0, 0.05)
        self.aug_rotate = _aug_spin(180.0, 1.0, 1)
        self.aug_brightness = _aug_spin(1.0, 0.05)
        self.aug_contrast = _aug_spin(1.0, 0.05)
        self.aug_saturation = _aug_spin(1.0, 0.05)
        self.aug_hue = _aug_spin(0.5, 0.005, 3)
        self.aug_decode_color_sim = _aug_spin(1.0, 0.05)
        self.aug_resample_sim = _aug_spin(1.0, 0.05)
        self.aug_monochrome = QCheckBox("Monochrome")
        for label, widget, tip in (
            ("Flip left-right (p)", self.aug_fliplr, "Probability of a horizontal flip."),
            ("Flip up-down (p)", self.aug_flipud, "Probability of a vertical flip."),
            ("Rotate 90° (p)", self.aug_rot90, "Probability of a 90° rotation (direction random). Label-exact for top-down views."),
            ("Rotate ± (deg)", self.aug_rotate, "Maximum small-angle rotation. Borders are filled gray; instances clipped below the fragment floor become non-exhaustive."),
            ("Brightness ±", self.aug_brightness, "Multiplicative brightness jitter."),
            ("Contrast ±", self.aug_contrast, "Contrast jitter about the image mean."),
            ("Saturation ±", self.aug_saturation, "HSV saturation jitter."),
            ("Hue ±", self.aug_hue, "HSV hue shift (fraction of the hue circle). Keep 0 when colour is the concept."),
            ("Decode-colour sim (p)", self.aug_decode_color_sim, "Probability of re-simulating video decode colour conversion."),
            ("Resample sim (p)", self.aug_resample_sim, "Probability of an alternate-resampler sub-pixel warp."),
        ):
            widget.setToolTip(tip)
            aug_form.addRow(label, widget)
        self.aug_monochrome.setToolTip("Convert tiles to grayscale.")
        aug_form.addRow(self.aug_monochrome)
```

Add `self.aug_group` to the panel layout the same way the neighbouring groups are added, and import `QFormLayout`/`QDoubleSpinBox`/`QCheckBox` if they are not already imported. The constructor's existing `self.set_params(Sam3LoraParams())` (~L612) would apply the contract's OFF default. Change it to `self.set_params(Sam3LoraParams(augmentation=recommended_sam3_augmentation()))`.

In `params()`, add:

```python
            augmentation=AugmentationProfile(
                enabled=self.aug_group.isChecked(),
                fliplr=self.aug_fliplr.value(),
                flipud=self.aug_flipud.value(),
                rot90=self.aug_rot90.value(),
                rotate=self.aug_rotate.value(),
                brightness=self.aug_brightness.value(),
                contrast=self.aug_contrast.value(),
                saturation=self.aug_saturation.value(),
                hue=self.aug_hue.value(),
                decode_color_sim=self.aug_decode_color_sim.value(),
                resample_sim=self.aug_resample_sim.value(),
                monochrome=self.aug_monochrome.isChecked(),
            ),
```

In `set_params()`, add the mirror (`a = p.augmentation`; `self.aug_group.setChecked(a.enabled)`; set each spin and the checkbox). In `training_dialog.py` persisted-state load, before `self.sam3_panel.set_params(Sam3LoraParams(**values))`, add:

```python
            if "augmentation" not in values:
                # State saved before SAM3 augmentation existed: keep the
                # panel's (recommended) profile rather than the contract's OFF.
                values["augmentation"] = self.sam3_panel.params().augmentation
```

Check that the save path (~L3393, `asdict(self.sam3_panel.params())`) writes the nested dict to JSON (`asdict` recurses). In `docs/user-guide/detectkit.md`, SAM3 section, add a short "Augmentation" subsection: the table of controls, the fresh defaults, "training split only", the flip caveat for chiral concepts, `hue` kept 0 for colour concepts, the CLI key `sam3.augmentation` (default disabled when omitted), and the `hydra_sam3_augmentation.json` stamp.

- [ ] **Step 4: Run** `QT_QPA_PLATFORM=offscreen python -m pytest tests/test_sam3_training_panel.py tests/test_sam3_dialog_wiring.py tests/test_sam3_gui_cli_training_parity.py tests/test_sam3_slice_settings_shared.py -q` and `make docs-build` (or `mkdocs build --strict`). Expected: PASS.

- [ ] **Step 5: Commit** `feat(detectkit): SAM3 augmentation controls + parity + docs`.

---

### Task 7: Visual check + full suite + CUDA smoke on courtship

**Files:**
- Create: `tools/sam3_augmentation_preview.py` (dev tool: renders an augmented-tile grid)
- Test: none new (verification task)

- [ ] **Step 1: Write the preview tool.** CLI: `python tools/sam3_augmentation_preview.py --dataset <sam3 derived dataset dir> --out <png> [--n 6] [--epochs 3] [--seed 0]`. It calls `dataloader.build_descriptors(dataset, Sam3LoraParams(prompt="x", num_negatives=0), "train")`, takes the first `n` descriptors, decodes each with `cv2.imread`, and for each epoch `e` applies `make_tile_augmenter(recommended_sam3_augmentation() with rotate=15, epoch_seed=seed+e, min_area_ratio=0.1)`. It draws the polygons (green, crowd in red) with `cv2.polylines` on edge→index-shifted coords, and tiles an `n × (epochs+1)` grid (column 0 unaugmented) to `--out`.

- [ ] **Step 2: Render on a local SAM3 dataset or fixture.** Find a built SAM3 dataset (`find ~ -name build_manifest.json -path '*sam3*' 2>/dev/null | head`). If none exists locally, build a tiny synthetic COCO split in `/tmp` (3 images with filled polygons). Read the PNG with the Read tool and confirm that the polygons sit exactly on the animals in every augmented cell and that the borders are gray. Save it to `/tmp/sam3_aug_preview.png`.

- [ ] **Step 3: Full local suite.** Run every SAM3 + augmentation + guard file: `QT_QPA_PLATFORM=offscreen python -m pytest tests/test_sam3_*.py tests/test_augmentation_*.py tests/test_training_augmentation.py tests/test_geometry_drift_guard.py tests/test_gui_cli_profile_parity.py tests/test_semantic_sam3_overrides.py -q`, then `make pytest`. Compare any failures against the branch base `dba69a55` (run the same failing test there) before calling them pre-existing.

- [ ] **Step 4: CUDA smoke on courtship.** `ssh rutalab@courtship.taild08eb9.ts.net`. Steps:
  1. `source ~/anaconda3/etc/profile.d/conda.sh`.
  2. Find how the `hydra-sam3` env imports `hydra_suite`: `conda run -n hydra-sam3 python -c "import hydra_suite;print(hydra_suite.__file__)"`. Sync the branch source to that location (rsync the worktree `src/`, or git fetch a pushed-to-courtship ref). Do NOT push to origin.
  3. Check `nvidia-smi`. Kill only stale sleap/hydra processes, by PID.
  4. Copy `~/detectkit-training/improved_ant_detection/sam3_plan.json` to `sam3_aug_smoke_plan.json` with: `sam3.epochs=1`, `sam3.augmentation=` the recommended profile plus `rotate: 15`, a fresh workspace dir, and `publish.auto_import=false`. If 1 epoch of the full dataset takes more than about 2 h, cap it by pointing at a subset source.
  5. Run `detectkit train --config ... > smoke.log 2>&1` under `nohup`.
  6. Verify: the log shows `augmentation ON: ...` and a finite loss each step; there is no OOM; `hydra_sam3_augmentation.json` exists with `applied.augmentation=true`; `batch_resolution.json` shows the measured batch, and the peak reserved VRAM (from the log or `nvidia-smi` sampling) is within the admission budget.
  7. Record the numbers in the plan's completion notes.

- [ ] **Step 5: Commit** the tool, `chore(sam3): augmentation preview tool`.

---

### Task 8: Adversarial review, merge, docs lifecycle

- [ ] **Step 1:** Run an adversarial whole-branch review with a different model (e.g. `fable`) via the Agent tool: diff `dba69a55..HEAD`, spec and plan attached, asked to find real bugs (coordinate conventions, RNG determinism, probe/val leakage, config round-trip, GUI default drift). Fix the confirmed findings with tests.
- [ ] **Step 2:** Run `make format` on the touched files, `make lint-moderate`, and re-run the Task 7 Step 3 suite.
- [ ] **Step 3:** In the same commit as the merge (on the branch, before merging), `git mv` the spec into `docs/superpowers/specs/done/` and the plan into `docs/superpowers/plans/done/`, and stamp the spec's Status line as `Shipped — merged to main (<sha>)`.
- [ ] **Step 4:** In the main checkout, run `git merge --no-ff feat/sam3-lora-augmentation`. Do not push. Remove the worktree after the merge.
