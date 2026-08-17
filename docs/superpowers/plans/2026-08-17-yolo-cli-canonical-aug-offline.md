# YOLO-CLI CanonicalAug (Offline Epoch-Multiplied Prefit) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give YOLO-classify training the `CanonicalAug` robustness signal it currently misses, by baking K per-image augmented variants into the on-disk prefit dataset — the only injection point available when training shells out to the `yolo` CLI.

**Architecture:** `_prefit_yolo_classify_dataset` (`training/runner.py`) already rewrites every source crop through the Layer-2 letterbox (`CanonicalFitTransform`) into a fresh per-run `prefit_dataset/` directory. We extend it: when the augmentation profile opts in, in addition to the one clean copy it writes `canonical_aug_copies` extra copies, each = `CanonicalFitTransform(CanonicalAug(crop))` — i.e. `CanonicalAug` applied to the **canonical crop before** the letterbox, matching the exact order the non-YOLO `TinyDataset` path uses. Randomness comes from one seeded `CanonicalAug(seed=spec.seed)` drawn sequentially, so the whole prefit is reproducible. When the profile does not opt in, the function is byte-identical to today (clean copy only).

**Tech Stack:** Python, numpy, OpenCV (`cv2`), torch (via `CanonicalAug`), Ultralytics `yolo` CLI (unchanged — it still consumes an ImageFolder), pytest.

## Global Constraints

- **Isolation:** do all work in a git worktree branched from local HEAD (`git worktree add .worktrees/yolo-canon-aug -b feat/yolo-cli-canonical-aug HEAD`) — never fresh-from-origin; local `main` is ahead of `origin/main`.
- **Off-by-default byte-identity:** when `augmentation_profile.canonical_aug` is `False` (the default), `_prefit_yolo_classify_dataset` MUST produce exactly the same files, filenames, and bytes as before this change. This is the load-bearing regression guard.
- **Order invariant:** `CanonicalAug` runs on the **source canonical crop**, then `CanonicalFitTransform` letterboxes the result — never the reverse. `CanonicalAug` is training-data-generation-only.
- **Reproducibility:** the prefit output is a deterministic function of (source files in sorted order, `spec.seed`, `canonical_aug`, `canonical_aug_copies`). No use of global `numpy.random`/`random`; only the instance `CanonicalAug` RNG.
- **Idempotency preserved:** the existing "return early if `dest_dir` exists" guard stays. Each training run uses a fresh `run_dir/prefit_dataset`, so K changes take effect per run without stale reuse.
- **Scope:** inference paths are untouched, so no MPS/CUDA equivalence-harness rerun is required. The gate is the training-path unit tests (`make pytest` subset) on `hydra-mps`. YOLO-classify remains a known-lossy family (vendor `Resize+CenterCrop`); this plan does not change that, it only adds training-time robustness.
- **Commit as the configured git user** (no `Co-Authored-By: Claude` trailer).

---

### Task 1: Add `canonical_aug_copies` knob to `AugmentationProfile`

**Files:**
- Modify: `src/hydra_suite/training/contracts.py:143` (the `AugmentationProfile` dataclass, right after `canonical_aug`)
- Test: `tests/test_augmentation_profile_canonical_copies.py` (create)

**Interfaces:**
- Consumes: nothing new.
- Produces: `AugmentationProfile.canonical_aug_copies: int` (default `3`) — number of **extra augmented** copies written per source image when `canonical_aug` is on. The always-written clean copy is not counted by this field, so total copies per image = `1 + canonical_aug_copies` when on, `1` when off.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_augmentation_profile_canonical_copies.py
"""canonical_aug_copies knob for the YOLO-classify offline prefit multiplier."""

from dataclasses import asdict

from hydra_suite.training.contracts import AugmentationProfile


def test_canonical_aug_copies_default_is_three():
    assert AugmentationProfile().canonical_aug_copies == 3


def test_canonical_aug_copies_is_overridable():
    assert AugmentationProfile(canonical_aug_copies=5).canonical_aug_copies == 5


def test_canonical_aug_copies_serializes_in_asdict():
    # TrainingRunSpec.to_dict() uses dataclasses.asdict; the new field must
    # round-trip so persisted run specs carry the knob.
    d = asdict(AugmentationProfile(canonical_aug_copies=4))
    assert d["canonical_aug_copies"] == 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_augmentation_profile_canonical_copies.py -v`
Expected: FAIL — `AugmentationProfile` has no attribute / unexpected keyword `canonical_aug_copies`.

- [ ] **Step 3: Add the field**

In `src/hydra_suite/training/contracts.py`, immediately after the `canonical_aug` field and its comment block (line ~145), add:

```python
    canonical_aug_copies: int = 3  # extra augmented copies per image in the
    # YOLO-classify offline prefit when canonical_aug is on (0 => clean only).
    # The clean copy is always written; total = 1 + canonical_aug_copies.
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_augmentation_profile_canonical_copies.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/training/contracts.py tests/test_augmentation_profile_canonical_copies.py
git commit -m "feat(training): add canonical_aug_copies knob to AugmentationProfile"
```

---

### Task 2: Multiply the prefit — write clean + K augmented copies

**Files:**
- Modify: `src/hydra_suite/training/runner.py:114-149` (`_prefit_yolo_classify_dataset`)
- Test: `tests/test_prefit_canonical_aug_multiplier.py` (create)

**Interfaces:**
- Consumes: `AugmentationProfile` (Task 1), `CanonicalAug` (`training/canonical_aug.py`), `CanonicalFitTransform` + `cv2_bgr_loader` (`training/canonical_transform.py`).
- Produces: new signature
  `_prefit_yolo_classify_dataset(dataset_dir: Path, imgsz: int, dest_dir: Path, *, profile: "AugmentationProfile | None" = None, seed: int = 42) -> Path`.
  When `profile` is `None`, or `profile.enabled` is False, or `profile.canonical_aug` is False → clean copy only (unchanged behavior). Otherwise also writes `profile.canonical_aug_copies` augmented copies per image, named `f"{stem}.aug{k}{suffix}"` for `k` in `1..copies`.

**Design notes for the implementer (do not skip):**
- Keep the existing early-return: `if dest_dir.exists(): return dest_dir`.
- Keep the clean copy exactly as today: `cv2.imwrite(str(out_cls_dir / img_path.name), transform(img))` where `transform = CanonicalFitTransform((imgsz, imgsz))`. Do not rename or reorder it — the off-path must stay byte-identical.
- Build **one** `CanonicalAug(seed=seed)` instance for the whole prefit (before the file loop), and draw from it sequentially. Do not reconstruct it per image (that would re-seed and repeat draws). Construct it only when augmenting.
- Augmented copy: `aug_img = canon_aug(img)` (on the loaded BGR crop, i.e. **before** `transform`), then `cv2.imwrite(str(out_cls_dir / f"{img_path.stem}.aug{k}{img_path.suffix}"), transform(aug_img))`.
- Log the multiplication once when active: `logger.info("YOLO-classify prefit: canonical_aug on, writing 1 clean + %d augmented copies/image (seed=%d)", copies, seed)` using the module `logger` at `runner.py:22`. When off, log nothing new.
- `copies = int(profile.canonical_aug_copies)` guarded to `max(0, copies)`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_prefit_canonical_aug_multiplier.py
"""Offline epoch-multiplied prefit for YOLO-classify CanonicalAug."""

import cv2
import numpy as np
import pytest

from hydra_suite.training.contracts import AugmentationProfile
from hydra_suite.training.runner import _prefit_yolo_classify_dataset


def _make_src(root, n_classes=2, per_class=2, hw=(40, 90)):
    """Non-square crops so we can prove aug ran before the square letterbox."""
    h, w = hw
    for c in range(n_classes):
        d = root / "train" / f"cls{c}"
        d.mkdir(parents=True)
        for i in range(per_class):
            img = (np.arange(h * w * 3, dtype=np.uint8).reshape(h, w, 3) + c * 7 + i) % 255
            cv2.imwrite(str(d / f"img{i}.png"), img.astype(np.uint8))
    return root


def test_off_is_byte_identical_to_clean_prefit(tmp_path):
    src = _make_src(tmp_path / "src")
    dest_a = tmp_path / "out_none"
    dest_b = tmp_path / "out_off"
    # profile=None and an explicitly-off profile must both yield clean-only.
    _prefit_yolo_classify_dataset(src, 64, dest_a, profile=None, seed=42)
    _prefit_yolo_classify_dataset(
        src, 64, dest_b,
        profile=AugmentationProfile(canonical_aug=False, canonical_aug_copies=3),
        seed=42,
    )
    files_a = sorted(p.name for p in (dest_a / "train" / "cls0").iterdir())
    files_b = sorted(p.name for p in (dest_b / "train" / "cls0").iterdir())
    assert files_a == files_b == ["img0.png", "img1.png"]  # no .aug* files
    a = cv2.imread(str(dest_a / "train" / "cls0" / "img0.png"))
    b = cv2.imread(str(dest_b / "train" / "cls0" / "img0.png"))
    np.testing.assert_array_equal(a, b)


def test_on_writes_clean_plus_k_augmented(tmp_path):
    src = _make_src(tmp_path / "src")
    dest = tmp_path / "out_on"
    _prefit_yolo_classify_dataset(
        src, 64, dest,
        profile=AugmentationProfile(canonical_aug=True, canonical_aug_copies=3),
        seed=42,
    )
    names = sorted(p.name for p in (dest / "train" / "cls0").iterdir())
    # per source image: 1 clean + 3 augmented
    assert names == [
        "img0.aug1.png", "img0.aug2.png", "img0.aug3.png", "img0.png",
        "img1.aug1.png", "img1.aug2.png", "img1.aug3.png", "img1.png",
    ]
    # every output is the square model input (letterbox ran on all copies)
    for n in names:
        out = cv2.imread(str(dest / "train" / "cls0" / n))
        assert out.shape[:2] == (64, 64)
    # augmented differs from clean (aug had an effect)
    clean = cv2.imread(str(dest / "train" / "cls0" / "img0.png"))
    aug1 = cv2.imread(str(dest / "train" / "cls0" / "img0.aug1.png"))
    assert not np.array_equal(clean, aug1)


def test_prefit_is_reproducible_for_fixed_seed(tmp_path):
    src = _make_src(tmp_path / "src")
    d1 = tmp_path / "r1"
    d2 = tmp_path / "r2"
    prof = AugmentationProfile(canonical_aug=True, canonical_aug_copies=2)
    _prefit_yolo_classify_dataset(src, 64, d1, profile=prof, seed=7)
    _prefit_yolo_classify_dataset(src, 64, d2, profile=prof, seed=7)
    a = cv2.imread(str(d1 / "train" / "cls0" / "img0.aug1.png"))
    b = cv2.imread(str(d2 / "train" / "cls0" / "img0.aug1.png"))
    np.testing.assert_array_equal(a, b)


def test_aug_receives_prefit_crop_not_letterboxed(tmp_path, monkeypatch):
    """The aug must see the raw non-square crop (before Layer-2), proving order."""
    src = _make_src(tmp_path / "src", n_classes=1, per_class=1, hw=(40, 90))
    seen_shapes = []

    import hydra_suite.training.canonical_aug as canon_mod

    class _SpyAug:
        def __init__(self, *a, **k):
            pass

        def __call__(self, img):
            seen_shapes.append(img.shape[:2])
            return img  # passthrough

    monkeypatch.setattr(canon_mod, "CanonicalAug", _SpyAug)
    _prefit_yolo_classify_dataset(
        src, 64, tmp_path / "out",
        profile=AugmentationProfile(canonical_aug=True, canonical_aug_copies=1),
        seed=1,
    )
    assert seen_shapes and all(s == (40, 90) for s in seen_shapes)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_prefit_canonical_aug_multiplier.py -v`
Expected: FAIL — `_prefit_yolo_classify_dataset` has no `profile`/`seed` keyword.

- [ ] **Step 3: Implement the multiplier**

Replace the body of `_prefit_yolo_classify_dataset` (`runner.py:114-149`) with the augmenting version. Update the signature and docstring; keep the early-return and the clean-copy write unchanged. Sketch:

```python
def _prefit_yolo_classify_dataset(
    dataset_dir: Path,
    imgsz: int,
    dest_dir: Path,
    *,
    profile: "AugmentationProfile | None" = None,
    seed: int = 42,
) -> Path:
    """Pre-fit a YOLO-classify ImageFolder dataset onto a square canvas.

    ... (keep the existing explanation) ...

    When ``profile.canonical_aug`` is on, additionally writes
    ``profile.canonical_aug_copies`` augmented variants per image
    (``CanonicalAug`` applied to the canonical crop *before* the Layer-2
    letterbox) so the CLI-trained model sees the Moderate robustness signal
    the Python-hooked trainers get per-epoch. Off by default -> clean only,
    byte-identical to the prior behaviour. Idempotent (skipped when
    ``dest_dir`` exists).
    """
    import cv2

    from .canonical_transform import CanonicalFitTransform, cv2_bgr_loader

    if dest_dir.exists():
        return dest_dir

    copies = 0
    canon_aug = None
    if (
        profile is not None
        and getattr(profile, "enabled", False)
        and getattr(profile, "canonical_aug", False)
    ):
        copies = max(0, int(getattr(profile, "canonical_aug_copies", 0)))
        if copies > 0:
            from .canonical_aug import CanonicalAug

            canon_aug = CanonicalAug(seed=seed)
            logger.info(
                "YOLO-classify prefit: canonical_aug on, writing 1 clean + %d "
                "augmented copies/image (seed=%d)",
                copies,
                seed,
            )

    transform = CanonicalFitTransform((int(imgsz), int(imgsz)))
    for split_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
        for cls_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            out_cls_dir = dest_dir / split_dir.name / cls_dir.name
            out_cls_dir.mkdir(parents=True, exist_ok=True)
            for img_path in sorted(cls_dir.iterdir()):
                if not img_path.is_file():
                    continue
                try:
                    img = cv2_bgr_loader(img_path)
                except Exception:
                    continue
                cv2.imwrite(str(out_cls_dir / img_path.name), transform(img))
                for k in range(1, copies + 1):
                    aug_img = canon_aug(img)
                    cv2.imwrite(
                        str(out_cls_dir / f"{img_path.stem}.aug{k}{img_path.suffix}"),
                        transform(aug_img),
                    )
    return dest_dir
```

Add the `AugmentationProfile` import for the type hint if not already imported at module top (it is a string annotation above, so a runtime import is not required; only add one if the module already imports contracts types at top-level — otherwise leave the quoted annotation).

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_prefit_canonical_aug_multiplier.py tests/test_yolo_classify_canonical_fit.py -v`
Expected: PASS (new suite green; the existing YOLO-classify test still green).

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/training/runner.py tests/test_prefit_canonical_aug_multiplier.py
git commit -m "feat(training): bake K CanonicalAug copies into YOLO-classify prefit"
```

---

### Task 3: Wire the profile + seed from the run spec into the prefit call

**Files:**
- Modify: `src/hydra_suite/training/runner.py:159-167` (the `classify` branch of `build_ultralytics_command`)
- Test: `tests/test_build_ultralytics_command_canonical_aug.py` (create)

**Interfaces:**
- Consumes: `spec.augmentation_profile` and `spec.seed` (both on `TrainingRunSpec`), plus Task 2's keyword-only params.
- Produces: no signature change to `build_ultralytics_command`; it now passes `profile=spec.augmentation_profile, seed=spec.seed` into `_prefit_yolo_classify_dataset`. `classify_scale_override = 0.0` is unchanged (augmented copies are also pre-fitted to square, so RandomResizedCrop stays a no-op).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_build_ultralytics_command_canonical_aug.py
"""build_ultralytics_command threads the aug profile into the classify prefit."""

import cv2
import numpy as np

from hydra_suite.training import runner as R
from hydra_suite.training.contracts import AugmentationProfile


def _seed_classify_dataset(root):
    d = root / "train" / "clsA"
    d.mkdir(parents=True)
    cv2.imwrite(str(d / "a.png"), np.full((30, 50, 3), 120, np.uint8))
    return root


def test_classify_prefit_receives_profile_and_seed(tmp_path, monkeypatch):
    ds = _seed_classify_dataset(tmp_path / "ds")
    captured = {}

    def _fake_prefit(dataset_dir, imgsz, dest_dir, *, profile=None, seed=42):
        captured["profile"] = profile
        captured["seed"] = seed
        (dest_dir).mkdir(parents=True, exist_ok=True)
        return dest_dir

    monkeypatch.setattr(R, "_prefit_yolo_classify_dataset", _fake_prefit)

    prof = AugmentationProfile(canonical_aug=True, canonical_aug_copies=2)
    spec = R.TrainingRunSpec(
        role=R.TrainingRole.CLASSIFY_FLAT_YOLO,  # a YOLO classify role -> task "classify"
        source_datasets=[],
        derived_dataset_dir=str(ds),
        base_model="yolov8n-cls.pt",
        hyperparams=R.TrainingHyperParams(imgsz=64, epochs=1, batch=1),
        seed=99,
        augmentation_profile=prof,
    )
    R.build_ultralytics_command(spec, tmp_path / "run")
    assert captured["seed"] == 99
    assert captured["profile"] is prof
    assert captured["profile"].canonical_aug is True
```

> Implementer note: confirm the exact classify `TrainingRole` enum member and the required `TrainingHyperParams`/`TrainingRunSpec` fields from `contracts.py`; adjust the constructor kwargs to whatever the current dataclasses require (the brief's values are illustrative for a classify-task spec). If constructing a full spec is heavy, an equally valid version of this test builds a minimal spec via `object.__new__`/`SimpleNamespace` with only `.role`, `.derived_dataset_dir`, `.hyperparams.imgsz`, `.augmentation_profile`, `.seed`, `.base_model` and whatever `_resolve_ultralytics_data_arg`/`_ultralytics_task_for_role` read — pick the lighter one that still exercises the real `build_ultralytics_command` classify branch.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_build_ultralytics_command_canonical_aug.py -v`
Expected: FAIL — captured `profile` is `None` / `seed` is default `42` (profile not threaded yet).

- [ ] **Step 3: Thread the params**

In `build_ultralytics_command`, change the prefit call (currently `runner.py:162-166`) to:

```python
        _prefit_yolo_classify_dataset(
            Path(spec.derived_dataset_dir).expanduser(),
            int(spec.hyperparams.imgsz),
            prefit_dir,
            profile=spec.augmentation_profile,
            seed=int(spec.seed),
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_build_ultralytics_command_canonical_aug.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/training/runner.py tests/test_build_ultralytics_command_canonical_aug.py
git commit -m "feat(training): thread aug profile + seed into YOLO-classify prefit"
```

---

### Task 4 (optional): Expose the toggle in the YOLO training dialog

Only do this task if the reviewer/operator wants the feature reachable from the GUI in this branch; otherwise `canonical_aug`/`canonical_aug_copies` remain programmatic-only (the same state as every other kit today — `canonical_aug` is not currently surfaced in any GUI), and this task is deferred as a follow-up.

**Files:**
- Modify: `src/hydra_suite/trackerkit/gui/dialogs/train_yolo_dialog.py:1446-1449` (the `AugmentationProfile(...)` construction) plus the surrounding augmentation group to add a checkbox + a copies spinbox.
- Test: none required if this is purely a widget-wiring change; if a config schema/serialization path is touched, add a targeted round-trip test.

**Interfaces:**
- Consumes: `AugmentationProfile(canonical_aug=..., canonical_aug_copies=...)`.
- Produces: dialog-driven `augmentation_profile` now carries the two fields.

- [ ] **Step 1: Add a `QCheckBox` ("CanonicalAug (robustness)") and a `QSpinBox` (copies, range 0–8, default 3) into the existing augmentation group box** near the other augmentation controls. Follow the existing widget/naming conventions in the file (`self.chk_*`, `self.spin_*`).

- [ ] **Step 2: Pass them into the profile**

```python
                augmentation_profile=AugmentationProfile(
                    enabled=self.aug_group.isChecked(),
                    args=aug_args,
                    canonical_aug=self.chk_canonical_aug.isChecked(),
                    canonical_aug_copies=self.spin_canonical_copies.value(),
                ),
```

- [ ] **Step 3: Manual smoke** — launch `trackerkit`, open the YOLO training dialog, confirm the checkbox/spinbox render and gate correctly (spinbox disabled when the checkbox is off is a nice-to-have, not required).

- [ ] **Step 4: Commit**

```bash
git add src/hydra_suite/trackerkit/gui/dialogs/train_yolo_dialog.py
git commit -m "feat(trackerkit): expose CanonicalAug toggle in YOLO training dialog"
```

---

## Verification (whole branch, before merge)

- [ ] `python -m pytest tests/test_augmentation_profile_canonical_copies.py tests/test_prefit_canonical_aug_multiplier.py tests/test_build_ultralytics_command_canonical_aug.py tests/test_yolo_classify_canonical_fit.py tests/test_canonical_aug.py -v` on `hydra-mps` — all green.
- [ ] `make format-check` and `make lint-moderate` clean on the changed files.
- [ ] Confirm the **off-path byte-identity** guard (`test_off_is_byte_identical_to_clean_prefit`) passes — this is the regression contract for existing YOLO-classify runs.
- [ ] No inference path changed → **no MPS/CUDA equivalence-harness rerun required.** (Note this explicitly in the merge summary so the reviewer doesn't ask for it.)
- [ ] Operational follow-up (user-owned, not a merge blocker): when actually training a YOLO-classify model with the toggle on, regenerate the dataset (old prefit dirs are per-run and not reused) and sanity-check accuracy on a held-out split vs the clean-only baseline.

## Non-Goals

- No change to YOLO-detect / YOLO-pose (they carry keypoint labels; `CanonicalAug`'s sub-pixel warp would desync labels — a separate design).
- No move off the `yolo` CLI subprocess (that is Option C — the Python-API/custom-dataset route, deliberately not taken here).
- No change to inference, the Layer-2 contract, or the deterministic canonicalization used at inference.
