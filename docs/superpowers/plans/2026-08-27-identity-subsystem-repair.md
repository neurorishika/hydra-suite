# Identity Subsystem Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make TrackerKit identity usable: preprocess classifiers the way they were trained, stop the fragment solver shredding trajectories, and let evidence express ignorance.

**Architecture:** (1) an artifact-level `fit_policy` honoured by one shared Layer-2 function used by every classifier consumer, stamped by training; (2) head-first classifier crops fed from the head/tail stage; (3) fragment solver: raw-signal PELT, no dropped rows, self-owned re-merge, evidence-quality breaker; (4) live `unknown` prior with an evidence-cache schema bump.

**Tech Stack:** Python 3.11, numpy, pandas, torch (`F.interpolate`), ruptures, pytest. Env `hydra-mps` (MPS gate here), `hydra-cuda` on mehek.

**Spec:** `docs/superpowers/specs/2026-08-27-identity-subsystem-repair-design.md`

## Global Constraints

- Work in a worktree branched from local HEAD: `git worktree add .worktrees/identity-repair -b fix/identity-subsystem-repair HEAD`; run tests with `PYTHONPATH=$PWD/src` from inside the worktree.
- Commit as the configured git user (no Claude co-author trailer).
- `letterbox` Layer-2 output must stay **byte-identical** to today's `apply_fit` / `letterbox_fit`.
- Undirected crops (no head/tail confidence) must stay byte-identical to today's `extract_classifier_crops`.
- No new GUI knobs. Constants live next to the code that uses them.
- Before any equivalence run: kill stale `sleap`/`hydra` processes only; never touch other processes.
- Run `make format` before each commit.

---

### Task 1: `fit_policy` on `ClassifierMetadata`

**Files:**
- Modify: `src/hydra_suite/core/individual/classification/backend.py:36-80` (dataclass), torch loader `parse_metadata` (~L214-250), multihead loader `parse_metadata` (~L358-440), yolo loaders (~L271-350)
- Test: `tests/identity/test_fit_policy_metadata.py`

**Interfaces:**
- Produces: `ClassifierMetadata.fit_policy: str` ∈ `{"letterbox","squash","native"}`; module constant `FIT_POLICIES = ("letterbox", "squash", "native")`; helper `resolve_fit_policy(raw: object, source_path: str, *, native: bool = False) -> str`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/identity/test_fit_policy_metadata.py
import json, logging
import numpy as np, pytest, torch

from hydra_suite.core.individual.classification import backend as B


def _tiny_ckpt(tmp_path, **extra):
    ckpt = {
        "schema_version": 2, "arch": "tinyclassifier", "factor_names": ["flat"],
        "class_names_per_factor": [["a", "b"]], "class_names": ["a", "b"],
        "input_size": [32, 32], "num_classes": 2, "monochrome": False,
        "model_state_dict": {}, **extra,
    }
    p = tmp_path / "m.pth"; torch.save(ckpt, p); return str(p)


def test_torch_ckpt_without_fit_policy_defaults_to_squash_with_warning(tmp_path, caplog):
    path = _tiny_ckpt(tmp_path)
    with caplog.at_level(logging.WARNING):
        meta = B._select_loader(path).parse_metadata(path)
    assert meta.fit_policy == "squash"
    assert "fit_policy" in caplog.text and "squash" in caplog.text


def test_torch_ckpt_with_fit_policy_letterbox(tmp_path, caplog):
    path = _tiny_ckpt(tmp_path, fit_policy="letterbox")
    with caplog.at_level(logging.WARNING):
        meta = B._select_loader(path).parse_metadata(path)
    assert meta.fit_policy == "letterbox"
    assert "fit_policy" not in caplog.text


def test_invalid_fit_policy_raises(tmp_path):
    path = _tiny_ckpt(tmp_path, fit_policy="stretchy")
    with pytest.raises(B.ClassifierFormatError):
        B._select_loader(path).parse_metadata(path)


def test_multihead_manifest_fit_policy(tmp_path):
    a = _tiny_ckpt(tmp_path, fit_policy="letterbox")
    man = tmp_path / "bundle.multihead.json"
    man.write_text(json.dumps({
        "schema_version": 2, "kind": "classifier_multihead_bundle",
        "factor_names": ["flat", "flat_1"],
        "factor_models": [{"factor": "flat", "path": "m.pth", "class_names": ["a", "b"]},
                          {"factor": "flat_1", "path": "m.pth", "class_names": ["a", "b"]}],
        "input_size": [32, 32], "monochrome": False,
    }))
    meta = B._select_loader(str(man)).parse_metadata(str(man))
    assert meta.fit_policy == "squash"          # absent on manifest → legacy squash
    man.write_text(json.dumps({**json.loads(man.read_text()), "fit_policy": "letterbox"}))
    assert B._select_loader(str(man)).parse_metadata(str(man)).fit_policy == "letterbox"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/identity/test_fit_policy_metadata.py -v`
Expected: FAIL — `ClassifierMetadata` has no attribute `fit_policy`.

- [ ] **Step 3: Implement**

In `backend.py`, next to the dataclass:

```python
FIT_POLICIES: tuple[str, ...] = ("letterbox", "squash", "native")
_LEGACY_FIT_POLICY = "squash"


def resolve_fit_policy(raw: object, source_path: str, *, native: bool = False) -> str:
    """Return the Layer-2 fit policy an artifact was trained under.

    ``native`` (YOLO classifiers) always returns "native". A missing key
    means the artifact predates fit-policy stamping (training before
    commit 3a2163ac, 2026-08-05, used an anisotropic ``Resize((sz, sz))``)
    → "squash", logged loudly so the drift can never be silent again.
    """
    if native:
        return "native"
    if raw is None or raw == "":
        logger.warning(
            "%s: artifact carries no fit_policy; assuming legacy 'squash' "
            "preprocessing (pre-2026-08-05 training). Re-publish the model or run "
            "scripts/stamp_fit_policy.py to stamp it explicitly.",
            source_path,
        )
        return _LEGACY_FIT_POLICY
    policy = str(raw).strip().lower()
    if policy not in FIT_POLICIES:
        raise ClassifierFormatError(
            f"{source_path!r}: fit_policy must be one of {FIT_POLICIES}, got {raw!r}"
        )
    return policy
```

Add the field to `ClassifierMetadata` (after `monochrome`): `fit_policy: str = "letterbox"` with a docstring line `fit_policy: Layer-2 preprocessing the model was trained with ("letterbox" | "squash" | "native").`

In each `parse_metadata`:
- torch checkpoint loader: `fit_policy=resolve_fit_policy(ckpt.get("fit_policy"), path)`
- multihead manifest loader: `fit_policy=resolve_fit_policy(data.get("fit_policy"), path)`
- yolo (flat + multihead bundle) loaders: `fit_policy=resolve_fit_policy(None, path, native=True)`

- [ ] **Step 4: Run tests**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/identity/test_fit_policy_metadata.py tests/test_classifier_backend*.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/core/individual/classification/backend.py tests/identity/test_fit_policy_metadata.py
git commit -m "feat(classifier): carry fit_policy on ClassifierMetadata; legacy artifacts resolve to squash with a loud warning"
```

---

### Task 2: Policy-driven Layer 2 shared by every classifier consumer

**Files:**
- Modify: `src/hydra_suite/core/canonicalization/resample.py` (add `squash_fit`, `fit_batch_for_model`), `src/hydra_suite/core/canonicalization/fit.py` (add `fit_crops_for_model`; fix stale module docstring), `src/hydra_suite/core/inference/stages/cnn.py:70-215`, `src/hydra_suite/core/inference/stages/headtail.py:130-135` and its batch function, `src/hydra_suite/core/inference/stages/crops.py:347-362` (`apply_fit_batch` → accept policy)
- Test: `tests/test_canonical_fit_policy.py`

**Interfaces:**
- Consumes: `ClassifierMetadata.fit_policy` (Task 1).
- Produces:
  - `resample.squash_fit(crop_chw: torch.Tensor, model_wh: tuple[int,int]) -> torch.Tensor` (antialiased bilinear resize, no paste)
  - `resample.fit_batch_for_model(crops_chw: torch.Tensor, model_wh, policy: str) -> torch.Tensor`
  - `fit.fit_crops_for_model(crops: list[np.ndarray], model_hw: tuple[int,int], policy: str) -> list[np.ndarray]` (HWC uint8 in/out; `"letterbox"` byte-identical to `apply_fit(c, fit_to_model_input(...))`)

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_canonical_fit_policy.py
import numpy as np, torch, pytest
from hydra_suite.core.canonicalization.fit import apply_fit, fit_to_model_input, fit_crops_for_model
from hydra_suite.core.canonicalization.resample import squash_fit, fit_batch_for_model, letterbox_fit


def _crop(h=66, w=148, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)


def test_letterbox_policy_is_byte_identical_to_apply_fit():
    c = _crop()
    ref = apply_fit(c, fit_to_model_input((148, 66), (128, 128)))
    out = fit_crops_for_model([c], (128, 128), "letterbox")[0]
    assert np.array_equal(out, ref)


def test_squash_policy_fills_canvas_no_black_bars():
    c = _crop()
    c[:] = 200
    out = fit_crops_for_model([c], (128, 128), "squash")[0]
    assert out.shape == (128, 128, 3)
    assert out.min() >= 195          # no zero rows anywhere


def test_squash_matches_torch_antialiased_bilinear():
    c = _crop()
    chw = torch.from_numpy(c).permute(2, 0, 1).float()
    ref = (torch.nn.functional.interpolate(chw[None], size=(128, 128), mode="bilinear",
           align_corners=False, antialias=True)[0].round().clamp(0, 255).to(torch.uint8)
           .permute(1, 2, 0).numpy())
    out = fit_crops_for_model([c], (128, 128), "squash")[0]
    assert np.array_equal(out, ref)


def test_torch_batch_policy_dispatch():
    x = torch.rand(4, 3, 66, 148)
    assert torch.equal(fit_batch_for_model(x, (128, 128), "letterbox"), letterbox_fit(x, (128, 128)))
    assert fit_batch_for_model(x, (128, 128), "squash").shape == (4, 3, 128, 128)
    assert torch.equal(fit_batch_for_model(x, (128, 128), "squash"), squash_fit(x, (128, 128)))


def test_unknown_policy_raises():
    with pytest.raises(ValueError):
        fit_crops_for_model([_crop()], (128, 128), "stretchy")
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/test_canonical_fit_policy.py -v`
Expected: FAIL with ImportError on `fit_crops_for_model`.

- [ ] **Step 3: Implement**

`resample.py` (after `letterbox_fit`):

```python
def squash_fit(crop_chw: torch.Tensor, model_wh: tuple) -> torch.Tensor:
    """Anisotropic antialiased-bilinear resize straight to ``model_wh`` (no paste).

    Layer 2 for artifacts trained with torchvision ``Resize((sz, sz))`` on PIL
    images (every classifier published before 2026-08-05). PIL's Resize is
    antialiased; ``F.interpolate(antialias=True)`` is the closest torch match.
    """
    single = crop_chw.dim() == 3
    x = crop_chw.unsqueeze(0) if single else crop_chw
    mw, mh = int(model_wh[0]), int(model_wh[1])
    with torch.inference_mode():
        out = F.interpolate(x, size=(mh, mw), mode="bilinear", align_corners=False, antialias=True)
    return out.squeeze(0) if single else out


def fit_batch_for_model(crops_chw: torch.Tensor, model_wh: tuple, policy: str) -> torch.Tensor:
    """Dispatch Layer 2 by ``policy`` (see ``ClassifierMetadata.fit_policy``)."""
    if policy == "letterbox":
        return letterbox_fit(crops_chw, model_wh)
    if policy == "squash":
        return squash_fit(crops_chw, model_wh)
    raise ValueError(f"unsupported fit_policy for tensor Layer 2: {policy!r}")
```

`fit.py` (after `apply_fit`); also rewrite the module docstring's "one resampler (INTER_AREA down, INTER_LINEAR up)" sentence to "one resampler: antialiased bilinear via the torch seam (`resample.letterbox_fit` / `resample.squash_fit`)":

```python
def fit_crops_for_model(
    crops: list[np.ndarray], model_hw: tuple[int, int], policy: str
) -> list[np.ndarray]:
    """Layer 2 for a list of HWC uint8 canonical crops, by the model's fit policy.

    "letterbox" is byte-identical to ``apply_fit(c, fit_to_model_input(...))``;
    "squash" is the legacy anisotropic resize; "native" returns crops untouched
    (the backend applies its own transform, e.g. ultralytics).
    """
    if not crops:
        return []
    in_h, in_w = int(model_hw[0]), int(model_hw[1])
    if policy == "native":
        return list(crops)
    if policy == "letterbox":
        fit = fit_to_model_input((crops[0].shape[1], crops[0].shape[0]), (in_w, in_h))
        return [apply_fit(c, fit) for c in crops]
    if policy == "squash":
        from hydra_suite.core.canonicalization.resample import squash_fit

        out = []
        for c in crops:
            arr = np.asarray(c)
            if arr.dtype != np.uint8:
                raise TypeError(f"Layer 2 requires uint8 input, got {arr.dtype}")
            chw = torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1).float()
            out.append(
                squash_fit(chw, (in_w, in_h)).round().clamp_(0, 255).to(torch.uint8)
                .permute(1, 2, 0).contiguous().numpy()
            )
        return out
    raise ValueError(f"unsupported fit_policy: {policy!r}")
```

`stages/crops.py` `apply_fit_batch(crops, fit)` → add `apply_fit_batch_for_model(crops, model_hw, policy)` that maps `fit_crops_for_model` across the warp pool in chunks (keep the old function for other callers).

`stages/cnn.py`: in `run_cnn` replace the two fit lines with
```python
np_crops = fit_crops_for_model(canon_crops, model.input_size, model.backend.metadata.fit_policy)
```
and in `run_cnn_batch` replace `fit = fit_to_model_input(...)` + both branches: CUDA branch `fitted = fit_batch_for_model(batch.crops, (in_w, in_h), policy)`; CPU branch `np_crops = apply_fit_batch_for_model(batch.crops, model.input_size, policy)` where `policy = model.backend.metadata.fit_policy`.

`stages/headtail.py`: same two substitutions (L134-135 and the batch function), using the head/tail model's `metadata.fit_policy`.

- [ ] **Step 4: Run tests**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/test_canonical_fit_policy.py tests/test_canonical_fit.py tests/test_canonical_crop.py tests/identity/ -v -x`
Expected: PASS.

- [ ] **Step 5: Manual check on the DEMO model** (evidence the fix works before moving on)

```bash
PYTHONPATH=$PWD/src KMP_DUPLICATE_LIB_OK=TRUE python - <<'EOF'
import cv2, numpy as np
from hydra_suite.runtime.resolver import ResolvedBackend
from hydra_suite.core.individual.classification.backend import ClassifierBackend
from hydra_suite.core.canonicalization.geometry import CanonicalGeometry
from hydra_suite.core.canonicalization.fit import fit_crops_for_model
from hydra_suite.core.inference.result import OBBResult
from hydra_suite.core.inference.stages.crops import extract_classifier_crops_batch_np
D="/Users/neurorishika/Projects/Rockefeller/Ruta/Presentation/DEMO/ID/OFFLINE/"
geo=CanonicalGeometry.from_reference(75.19,2.25,1.3)
z=np.load(D+".inference_cache_ant/detection.npz"); ev=np.load(D+".inference_cache_ant/detection_identity_evidence_batch_f63ae50236485442.npz")
f=9301; m=z["frame_indices"]==f; keep=np.isin(z["detection_ids"][m], ev[f"f{f}__colortag_det_ids"])
obb=OBBResult(frame_idx=f,centroids=z["centroids"][m][keep],angles=z["angles"][m][keep],sizes=z["sizes"][m][keep],shapes=z["shapes"][m][keep],confidences=z["confidences"][m][keep],corners=z["corners"][m][keep],detection_ids=z["detection_ids"][m][keep])
cap=cv2.VideoCapture(D+"ant.mp4"); cap.set(1,f); _,fr=cap.read()
be=ClassifierBackend("/Users/neurorishika/Library/Application Support/hydra-suite/models/classification/identity/20260429-105036_classifier_multihead_obiroi_colortag.multihead.json", ResolvedBackend("torch","cpu",False))
crops=fit_crops_for_model(extract_classifier_crops_batch_np([fr],[obb],geo).crops,(128,128),be.metadata.fit_policy)
mx=[min(max(p) for p in per) for per in be.predict_batch(crops)]
print("policy",be.metadata.fit_policy,"median per-head min max-prob",np.median(mx))
EOF
```
Expected: `policy squash`, median ≥ 0.8 (was ≈0.14 under letterbox).

- [ ] **Step 6: Commit**

```bash
git add src/hydra_suite/core/canonicalization tests/test_canonical_fit_policy.py src/hydra_suite/core/inference/stages
git commit -m "fix(inference): Layer-2 fit follows the artifact's fit_policy for identity CNN and head/tail stages"
```

---

### Task 3: Training stamps `fit_policy`; stamping script for existing artifacts

**Files:**
- Modify: `src/hydra_suite/training/runner.py` (checkpoint dicts at ~L765 and ~L828; multihead manifest writer), `src/hydra_suite/training/model_publish.py` (manifest passthrough if it rewrites JSON)
- Create: `scripts/stamp_fit_policy.py`
- Test: `tests/test_training_fit_policy_stamp.py`

**Interfaces:**
- Produces: checkpoint key `"fit_policy": "letterbox"`; multihead manifest key `"fit_policy": "letterbox"`; CLI `python scripts/stamp_fit_policy.py <artifact.pth|artifact.multihead.json> --policy letterbox|squash`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_training_fit_policy_stamp.py
import json, subprocess, sys, torch
from hydra_suite.training import runner as R


def test_checkpoint_dict_carries_letterbox_fit_policy():
    d = R.build_checkpoint_dict(arch="tinyclassifier", factor_names=["flat"],
                                class_names_per_factor=[["a", "b"]], input_size=(32, 32),
                                monochrome=False, state_dict={}, best_val_acc=0.5, history={})
    assert d["fit_policy"] == "letterbox"


def test_stamp_script_torch_and_manifest(tmp_path):
    p = tmp_path / "m.pth"; torch.save({"schema_version": 2, "arch": "tinyclassifier"}, p)
    man = tmp_path / "b.multihead.json"; man.write_text(json.dumps({"schema_version": 2}))
    for target in (p, man):
        subprocess.run([sys.executable, "scripts/stamp_fit_policy.py", str(target), "--policy", "squash"], check=True)
    assert torch.load(p, weights_only=False)["fit_policy"] == "squash"
    assert json.loads(man.read_text())["fit_policy"] == "squash"
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/test_training_fit_policy_stamp.py -v`
Expected: FAIL (`build_checkpoint_dict` missing / script missing).

- [ ] **Step 3: Implement**

In `runner.py`, factor the two literal checkpoint dicts (L765, L828) into one builder and add the key:

```python
FIT_POLICY_TRAINED = "letterbox"   # CanonicalFitTransform (training/canonical_transform.py)


def build_checkpoint_dict(*, arch, factor_names, class_names_per_factor, input_size,
                          monochrome, state_dict, best_val_acc, history, **extra) -> dict:
    return {
        "schema_version": 2,
        "arch": arch,
        "factor_names": list(factor_names),
        "class_names_per_factor": [list(c) for c in class_names_per_factor],
        "input_size": [int(input_size[0]), int(input_size[1])],
        "num_classes": sum(len(c) for c in class_names_per_factor),
        "monochrome": bool(monochrome),
        "model_state_dict": state_dict,
        "best_val_acc": float(best_val_acc),
        "history": history,
        "fit_policy": FIT_POLICY_TRAINED,
        **extra,
    }
```
Replace both literal dicts with calls (pass the existing extra keys — `trainable_layers`, `backbone_lr_scale`, `class_names`, `fine_tune_method`, `layerwise_lr_decay`, `gradual_unfreeze_interval`, `ignore_label_name` — through `**extra`). The multihead manifest JSON is assembled in `src/hydra_suite/training/model_publish.py` (`_TRACKERKIT_MULTIHEAD_KIND`, L122 and the dict that uses it): add `"fit_policy": FIT_POLICY_TRAINED` (import it from `runner`, or move the constant to `training/canonical_transform.py` and import from there in both).

`scripts/stamp_fit_policy.py`:

```python
"""Stamp an existing classifier artifact with the Layer-2 fit policy it was trained under.

Usage: python scripts/stamp_fit_policy.py <model.pth | bundle.multihead.json> --policy letterbox|squash
Models trained before 2026-08-05 (commit 3a2163ac) used torchvision Resize((sz,sz)) → squash.
Models trained after that with the CanonicalFitTransform → letterbox.
"""
import argparse, json, sys
from pathlib import Path

POLICIES = ("letterbox", "squash")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("artifact"); ap.add_argument("--policy", required=True, choices=POLICIES)
    a = ap.parse_args(argv); p = Path(a.artifact)
    if p.suffix == ".json":
        d = json.loads(p.read_text()); d["fit_policy"] = a.policy
        p.write_text(json.dumps(d, indent=2))
        for fm in d.get("factor_models", []):
            sub = p.parent / fm["path"]
            if sub.exists():
                main([str(sub), "--policy", a.policy])
    else:
        import torch
        ck = torch.load(p, map_location="cpu", weights_only=False)  # checkpoints hold dict/list metadata; trusted local artifacts only
        if not isinstance(ck, dict):
            print(f"{p}: not a dict checkpoint", file=sys.stderr); return 2
        ck["fit_policy"] = a.policy; torch.save(ck, p)
    print(f"stamped {p} fit_policy={a.policy}"); return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/test_training_fit_policy_stamp.py tests/test_training*.py -v -x`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/training scripts/stamp_fit_policy.py tests/test_training_fit_policy_stamp.py
git commit -m "feat(training): stamp fit_policy=letterbox on published classifiers; add stamp_fit_policy.py for existing artifacts"
```

---

### Task 4: Head-first classifier crops from the head/tail stage

**Files:**
- Modify: `src/hydra_suite/core/inference/stages/crops.py:185-345` (`extract_classifier_crops`, `extract_classifier_crops_batch_np`), `src/hydra_suite/core/inference/stages/cnn.py` (`run_cnn_batch` signature), `src/hydra_suite/core/inference/pipeline.py:380-392`
- Test: `tests/test_classifier_crop_orientation.py`

**Interfaces:**
- Consumes: `HeadTailResult.heading_hints (D,) radians`, `.directed_mask (D,) uint8` (`core/inference/result.py:107`); `resolve_directed_angle` (`core/individual/geometry.py:203`).
- Produces: `extract_classifier_crops(frame, obb_result, geometry, heading_hints=None, directed_mask=None)`; `extract_classifier_crops_batch_np(frames, obb_results, geometry, headtail_by_frame: dict[int, HeadTailResult] | None = None)`; `run_cnn_batch(..., headtail_by_frame=None)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_classifier_crop_orientation.py
import numpy as np
from hydra_suite.core.canonicalization.geometry import CanonicalGeometry
from hydra_suite.core.inference.result import OBBResult
from hydra_suite.core.inference.stages.crops import extract_classifier_crops


def _frame_with_marker():
    fr = np.full((400, 400, 3), 128, np.uint8)
    fr[195:205, 260:300] = 255        # bright marker on the +x side of centre (200,200)
    return fr


def _obb(cx=200, cy=200, w=120, h=50):
    c = np.array([[cx - w/2, cy - h/2], [cx + w/2, cy - h/2], [cx + w/2, cy + h/2], [cx - w/2, cy + h/2]])
    return OBBResult(frame_idx=0, centroids=np.array([[cx, cy]]), angles=np.zeros(1), sizes=np.ones(1),
                     shapes=np.ones((1, 2)), confidences=np.ones(1), corners=c[None], detection_ids=np.array([1]))


def test_undirected_is_unchanged_and_directed_flips():
    geo = CanonicalGeometry.from_reference(100.0, 2.0, 1.3)
    fr, obb = _frame_with_marker(), _obb()
    base = extract_classifier_crops(fr, obb, geo)[0]
    same = extract_classifier_crops(fr, obb, geo, heading_hints=np.array([np.pi]), directed_mask=np.array([0], np.uint8))[0]
    assert np.array_equal(base, same)                       # undirected → byte-identical
    flipped = extract_classifier_crops(fr, obb, geo, heading_hints=np.array([np.pi]), directed_mask=np.array([1], np.uint8))[0]
    assert np.array_equal(flipped, np.ascontiguousarray(base[::-1, ::-1]))   # head points -x → rotate 180°
    keep = extract_classifier_crops(fr, obb, geo, heading_hints=np.array([0.0]), directed_mask=np.array([1], np.uint8))[0]
    assert np.array_equal(keep, base)                        # head already +x → unchanged
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/test_classifier_crop_orientation.py -v`
Expected: FAIL — unexpected keyword `heading_hints`.

- [ ] **Step 3: Implement**

In `crops.py` add a helper and use it in both classifier entry points:

```python
def _directed_align(m_align: np.ndarray, theta: float, hint, directed, geometry: CanonicalGeometry) -> np.ndarray:
    """Rotate a canonical affine by π about the canvas centre when the head/tail
    stage says the head points opposite to the OBB major axis (+x after align)."""
    if not directed or hint is None or not np.isfinite(hint):
        return m_align
    from hydra_suite.core.individual.geometry import resolve_directed_angle

    angle, is_dir, _ = resolve_directed_angle(float(theta), float(hint), True)
    if not is_dir:
        return m_align
    d = (angle - theta + np.pi) % (2 * np.pi) - np.pi
    if abs(d) < np.pi / 2:
        return m_align
    w, h = geometry.canvas_w, geometry.canvas_h
    flip = np.array([[-1.0, 0.0, w], [0.0, -1.0, h]])          # (x,y) -> (w-x, h-y)
    m3 = np.vstack([m_align, [0.0, 0.0, 1.0]])
    return (np.vstack([flip, [0.0, 0.0, 1.0]]) @ m3)[:2]
```
`extract_classifier_crops(frame, obb_result, geometry, heading_hints=None, directed_mask=None)`: inside the affine loop, after `canonical_affine`, call `m_align = _directed_align(m_align, _theta, heading_hints[i] if heading_hints is not None else None, bool(directed_mask[i]) if directed_mask is not None else False, geometry)`. Note: a flipped affine maps the canvas through `(w-x, h-y)`; for even canvas sizes this is an exact 180° pixel rotation, which is what the test asserts.

`extract_classifier_crops_batch_np(frames, obb_results, geometry, headtail_by_frame=None)`: per frame look up `ht = headtail_by_frame.get(obb.frame_idx)` and pass `ht.heading_hints`, `ht.directed_mask` when present.

`run_cnn_batch(..., headtail_by_frame=None)`: thread into `extract_classifier_crops_batch_np` (CPU branch) and into `extract_canonical_crops_batch` (CUDA branch) — give `extract_canonical_crops_batch` the same optional argument, applying `_directed_align` in its affine loop.

`pipeline.py:382`: `run_cnn_batch(nonempty_frames, nonempty_obbs, mdl, cfg_cnn, self.runtime, geometry, headtail_by_frame=headtail)`.

- [ ] **Step 4: Run tests**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/test_classifier_crop_orientation.py tests/test_canonical_crop*.py tests/identity/test_evidence_stage_runner.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/core/inference tests/test_classifier_crop_orientation.py
git commit -m "feat(inference): orient identity classifier crops head-first from the head/tail stage"
```

---

### Task 5: Live `unknown` prior in catalog evidence + evidence schema bump

**Files:**
- Modify: `src/hydra_suite/core/individual/identity/substrate.py:291-` (`map_cnn_to_catalog`), `src/hydra_suite/core/individual/identity/evidence_builder.py` (constructor + `_build_log_probs_from_posteriors`), `src/hydra_suite/core/inference/runner.py:361` (`_build_identity_evidence_stage` passes the prior), `src/hydra_suite/core/individual/identity/cache.py` (`evidence_schema_version` 2 → 3; reject older on load)
- Test: `tests/identity/test_unknown_prior_live.py`

**Interfaces:**
- Produces: `map_cnn_to_catalog(..., unknown_prior: float = 0.0)`; `EvidenceBuilder(..., unknown_prior: float = 0.0)`; `cache._SCHEMA_VERSION = 3` (module constant at `cache.py:61`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/identity/test_unknown_prior_live.py
import numpy as np
from hydra_suite.core.individual.identity import substrate
from hydra_suite.core.individual.identity.catalog import IdentityCatalog


def _mapped(unknown_prior):
    catalog = IdentityCatalog.from_labels(["a", "b"])
    log_p, _ = substrate.map_cnn_to_catalog(
        [np.array([0.6, 0.4])], class_labels_per_factor=[["a", "b"]],
        factor_class_to_catalog={}, is_composite=False, catalog_size=3, catalog=catalog,
        unknown_prior=unknown_prior)
    return np.exp(log_p)


def test_default_zero_prior_is_backward_compatible():
    p = _mapped(0.0)
    assert p[0] < 1e-5


def test_unknown_prior_gets_exactly_that_mass():
    p = _mapped(0.05)
    assert np.isclose(p[0], 0.05) and np.isclose(p[1:].sum(), 0.95)
    assert np.isclose(p[1] / p[2], 0.6 / 0.4)


def test_cache_schema_version_is_3(tmp_path):
    from hydra_suite.core.individual.identity import cache
    assert cache._SCHEMA_VERSION == 3
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/identity/test_unknown_prior_live.py -v`
Expected: FAIL (unexpected kwarg `unknown_prior`).

- [ ] **Step 3: Implement**

`substrate.map_cnn_to_catalog` — add parameter `unknown_prior: float = 0.0`; after the fused vector is normalised (before the optional `prob_floor` block):

```python
    if unknown_prior > 0.0:
        probs = np.exp(fused)
        known = probs[1:]
        known_sum = float(known.sum())
        if known_sum > 0:
            probs[1:] = known * ((1.0 - unknown_prior) / known_sum)
        probs[0] = unknown_prior
        fused = np.log(np.clip(probs, 1e-300, None))
```
`EvidenceBuilder.__init__(..., unknown_prior: float = 0.0)` stores it and passes `unknown_prior=self._unknown_prior` in `_build_log_probs_from_posteriors`. In `runner._build_identity_evidence_stage`, pass `unknown_prior=float(config.identity_unknown_prior)` — add `identity_unknown_prior: float = 0.05` to `InferenceConfig` (`core/inference/config.py`) and populate it from `IDENTITY_UNKNOWN_PRIOR` where `InferenceConfig` is built from engine params (search `InferenceConfig(` in `trackerkit/engine_params.py` / `core/tracking/session.py`).

`cache.py:61`: `_SCHEMA_VERSION = 3` (already written as `evidence_schema_version` at L201; update the docstring at L17); in `_load`, if stored version `!= SCHEMA_VERSION` log `INFO "identity evidence sidecar schema %d != %d; will be rebuilt"` and treat the cache as empty. `core/inference/identity_evidence_key.py:17`: add parameter `unknown_prior: float = 0.0` to `identity_evidence_cache_key(catalog_spec, per_factor_temps, base_signature, unknown_prior=0.0)` and put `"unknown_prior": round(float(unknown_prior), 6)` into `payload`; pass it from the one caller (grep `identity_evidence_cache_key(`).

- [ ] **Step 4: Run tests**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/identity -v -x`
Expected: PASS (update `test_evidence_builder_parity.py` golden only if it asserts on the schema number; the math with `unknown_prior=0.0` is unchanged).

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/core/individual/identity src/hydra_suite/core/inference tests/identity/test_unknown_prior_live.py
git commit -m "fix(identity): honour identity_unknown_prior in catalog evidence; bump evidence sidecar schema to 3"
```

---

### Task 6: PELT on the raw signal (no per-trajectory z-scoring)

**Files:**
- Modify: `src/hydra_suite/core/individual/identity/offline.py:121-216`
- Test: `tests/identity/test_offline_changepoint.py` (extend)

- [ ] **Step 1: Write the failing tests** (append to the existing file, reusing its helpers)

```python
def test_constant_posterior_yields_no_changepoints():
    """Regression: per-trajectory z-scoring turned float noise on a constant
    posterior into unit-variance signal and PELT split it 5-22 times."""
    rng = np.random.default_rng(0)
    seq = []
    for f in range(400):
        lp = np.full(4, -20.0); lp[2] = 0.0
        lp += rng.normal(0, 1e-4, 4)          # float-noise jitter
        seq.append((f, lp - np.logaddexp.reduce(lp)))
    out = detect_identity_changepoints({7: seq}, _catalog4(), {"PELT_MODEL": "l2", "CHANGEPOINT_PENALTY": 3.0})
    assert out == {}


def test_single_clean_switch_yields_one_changepoint():
    seq = []
    for f in range(400):
        lp = np.full(4, -20.0); lp[1 if f < 200 else 3] = 0.0
        seq.append((f, lp - np.logaddexp.reduce(lp)))
    out = detect_identity_changepoints({7: seq}, _catalog4(), {"PELT_MODEL": "l2", "CHANGEPOINT_PENALTY": 3.0})
    assert out == {7: [199]}
```
(`_catalog4()` = `IdentityCatalog.from_labels(["a","b","c"])`; add it next to the file's existing fixture helper.)

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/identity/test_offline_changepoint.py -v`
Expected: `test_constant_posterior_yields_no_changepoints` FAILS (several splits).

- [ ] **Step 3: Implement**

In `detect_identity_changepoints` delete the block

```python
        # Z-score per column to suppress magnitude drift.
        # Skipped for l1 which is already median-based and scale-insensitive.
        if pelt_model != "l1":
            col_std = signal.std(axis=0)
            col_std[col_std < 1e-8] = 1.0
            signal = (signal - signal.mean(axis=0)) / col_std
```
and replace with

```python
        # The signal is a probability simplex slice in [0, 1]; the penalty is in
        # those units. Per-trajectory z-scoring made the penalty's units
        # trajectory-dependent and inflated float noise on constant posteriors
        # into unit-variance "signal" (660 splits on 128 tracks, 2026-08-27).
```
Update the docstring ("Z-scoring is skipped for l1…" → remove) and the module docstring's `CHANGEPOINT_PENALTY` line: `float default 3.0 — in probability units (raw, un-normalised signal)`.

- [ ] **Step 4: Run tests**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/identity/test_offline_changepoint.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/core/individual/identity/offline.py tests/identity/test_offline_changepoint.py
git commit -m "fix(identity): run PELT on the raw posterior signal; drop per-trajectory z-scoring that split float noise"
```

---

### Task 7: Never drop rows on split; solver re-merges its own cuts

**Files:**
- Modify: `src/hydra_suite/core/individual/identity/offline.py:219-281` (`split_trajectories_at_changepoints`), `~1036-1210` (`solve_global_assignment` — add re-merge after label write-back), `run_fragment_solver`
- Test: `tests/identity/test_offline_split_merge.py`

**Interfaces:**
- Produces: `merge_same_label_neighbours(df: pd.DataFrame, label_col: str = C.FINAL_LABEL) -> pd.DataFrame` (pure; reads `OriginalTrajectoryID`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/identity/test_offline_split_merge.py
import numpy as np, pandas as pd
from hydra_suite.core.individual.identity import columns as C
from hydra_suite.core.individual.identity.offline import split_trajectories_at_changepoints, merge_same_label_neighbours


def _traj(tid, frames):
    return pd.DataFrame({"TrajectoryID": tid, "FrameID": frames, "X": 0.0, "Y": 0.0})


def test_short_remnant_is_merged_not_dropped():
    df = _traj(1, range(0, 100))
    out = split_trajectories_at_changepoints(df, {1: [97]}, {"MIN_FRAGMENT_FRAMES": 5})
    assert len(out) == 100                       # no rows lost
    assert out["TrajectoryID"].nunique() == 1    # 2-frame remnant folded back


def test_leading_short_remnant_merges_forward():
    df = _traj(1, range(0, 100))
    out = split_trajectories_at_changepoints(df, {1: [2, 60]}, {"MIN_FRAGMENT_FRAMES": 5})
    assert len(out) == 100 and out["TrajectoryID"].nunique() == 2


def test_merge_same_label_neighbours_undoes_needless_cut():
    df = pd.concat([_traj(10, range(0, 50)), _traj(11, range(50, 100)), _traj(12, range(100, 150))])
    df["OriginalTrajectoryID"] = 1
    df[C.FINAL_LABEL] = np.where(df["TrajectoryID"] == 12, "b", "a")
    out = merge_same_label_neighbours(df)
    assert out["TrajectoryID"].nunique() == 2
    assert out.loc[out.FrameID < 100, "TrajectoryID"].nunique() == 1


def test_merge_respects_different_originals():
    df = pd.concat([_traj(10, range(0, 50)), _traj(11, range(50, 100))])
    df["OriginalTrajectoryID"] = [1] * 50 + [2] * 50
    df[C.FINAL_LABEL] = "a"
    assert merge_same_label_neighbours(df)["TrajectoryID"].nunique() == 2
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/identity/test_offline_split_merge.py -v`
Expected: FAIL (ImportError on `merge_same_label_neighbours`; remnant test loses rows).

- [ ] **Step 3: Implement**

In `split_trajectories_at_changepoints`, replace the segment loop:

```python
        segments: list[pd.DataFrame] = []
        for start_f, end_f in boundaries:
            seg = grp[(grp["FrameID"] >= start_f) & (grp["FrameID"] <= end_f)]
            if seg.empty:
                continue
            if len(seg) < min_frames and segments:
                segments[-1] = pd.concat([segments[-1], seg])       # fold into previous
            elif len(seg) < min_frames:
                segments.append(seg)                                # leading remnant: fold into next
                continue
            elif segments and len(segments[-1]) < min_frames:
                segments[-1] = pd.concat([segments[-1], seg])
            else:
                segments.append(seg)
        for seg in segments:
            seg = seg.copy()
            seg["TrajectoryID"] = next_id
            seg["OriginalTrajectoryID"] = traj_id
            next_id += 1
            parts.append(seg)
```
Update the docstring: "Sub-segments shorter than MIN_FRAGMENT_FRAMES rows are merged into their neighbour (never dropped)."

Add after `_annotate_smoothed_labels`:

```python
def merge_same_label_neighbours(df: pd.DataFrame, label_col: str = C.FINAL_LABEL) -> pd.DataFrame:
    """Undo solver cuts that changed nothing: consecutive fragments of the same
    ``OriginalTrajectoryID`` whose final labels agree (unknown == unknown too)
    are re-joined under the earlier fragment's TrajectoryID. Relink cannot do
    this (it rejects gap == 0), so the solver owns it."""
    if "OriginalTrajectoryID" not in df.columns or label_col not in df.columns:
        return df
    out = df.copy()
    spans = (
        out.groupby("TrajectoryID")
        .agg(orig=("OriginalTrajectoryID", "first"), start=("FrameID", "min"),
             end=("FrameID", "max"), label=(label_col, "first"))
        .sort_values(["orig", "start"])
    )
    remap: dict = {}
    prev_tid = prev_orig = prev_label = None
    prev_end = None
    for tid, r in spans.iterrows():
        same = (prev_tid is not None and r.orig == prev_orig and r.start == prev_end + 1
                and str(r.label) == str(prev_label))
        if same:
            remap[tid] = remap.get(prev_tid, prev_tid)
        else:
            prev_tid, prev_orig, prev_label = tid, r.orig, r.label
        prev_end = r.end
    if remap:
        out["TrajectoryID"] = out["TrajectoryID"].map(lambda t: remap.get(t, t))
    return out
```
At the end of `run_fragment_solver`, wrap the return: `solved = solve_global_assignment(...)`; `merged = merge_same_label_neighbours(solved)`; log `"fragment_solver: re-merged %d → %d trajectories after assignment"`; return `merged`.

- [ ] **Step 4: Run tests**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/identity/test_offline_split_merge.py tests/identity/test_offline_*.py tests/identity/test_honesty_fix.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/core/individual/identity/offline.py tests/identity/test_offline_split_merge.py
git commit -m "fix(identity): never drop short split remnants; re-merge same-label neighbour fragments after assignment"
```

---

### Task 8: Evidence-quality circuit breaker

**Files:**
- Modify: `src/hydra_suite/core/individual/identity/offline.py` (`run_fragment_solver`, before the PELT block)
- Test: `tests/identity/test_offline_evidence_breaker.py`

**Interfaces:**
- Produces: `assess_evidence_quality(smoothed_by_traj, raw_evidence, catalog) -> EvidenceQuality(conf_frac: float, diversity: float, n_frames: int, ok: bool)`; constants `EVIDENCE_MIN_CONF_FRAC = 0.10`, `EVIDENCE_MIN_DIVERSITY = 0.30`, `EVIDENCE_CONF_LEVEL = 0.5`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/identity/test_offline_evidence_breaker.py
import numpy as np
from hydra_suite.core.individual.identity.catalog import IdentityCatalog
from hydra_suite.core.individual.identity.offline import assess_evidence_quality


def _raw(n_traj, n_frames, maxp, same_label=False, seed=0):
    rng = np.random.default_rng(seed); cat = 26; out = {}
    for t in range(n_traj):
        seq = []
        for f in range(n_frames):
            p = np.full(cat, (1 - maxp) / (cat - 1)); p[0] = 1e-12
            k = 5 if same_label else 1 + (t % (cat - 1))
            p[k] = maxp; p /= p.sum(); seq.append((f, np.log(p)))
        out[t] = seq
    return out


def test_diffuse_same_label_evidence_trips_breaker():
    cat = IdentityCatalog.from_labels([f"l{i}" for i in range(25)])
    raw = _raw(20, 100, 0.14, same_label=True)
    q = assess_evidence_quality(raw, cat)
    assert q.conf_frac == 0.0 and q.diversity < 0.3 and not q.ok


def test_confident_diverse_evidence_passes():
    cat = IdentityCatalog.from_labels([f"l{i}" for i in range(25)])
    q = assess_evidence_quality(_raw(20, 100, 0.9), cat)
    assert q.conf_frac > 0.9 and q.diversity > 0.5 and q.ok
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/identity/test_offline_evidence_breaker.py -v`
Expected: FAIL (ImportError).

- [ ] **Step 3: Implement** (in `offline.py`, above `run_fragment_solver`)

```python
EVIDENCE_CONF_LEVEL = 0.5       # a detection "knows" its label at this posterior
EVIDENCE_MIN_CONF_FRAC = 0.10   # <10% confident detections → source is uninformative
EVIDENCE_MIN_DIVERSITY = 0.30   # distinct labels/frame vs achievable → collapsed source


@dataclass(frozen=True)
class EvidenceQuality:
    conf_frac: float
    diversity: float
    n_frames: int
    ok: bool


def assess_evidence_quality(raw_evidence: dict, catalog: IdentityCatalog) -> EvidenceQuality:
    """Cheap, source-level sanity check on RAW (unsmoothed) per-frame evidence.

    conf_frac: fraction of (frame, detection) rows whose max KNOWN posterior
        >= EVIDENCE_CONF_LEVEL.
    diversity: mean over frames of distinct argmax labels / min(#known labels,
        #detections in that frame).
    Both were ~0 / 0.3 on the 2026-08-27 failure (a mis-preprocessed classifier)
    and >0.8 / >0.7 with correct preprocessing.
    """
    per_frame: dict[int, list[int]] = {}
    conf = 0; total = 0
    for seq in raw_evidence.values():
        for frame_id, lp in seq:
            p = np.exp(lp - np.logaddexp.reduce(lp))
            known = p[1:]
            total += 1
            if known.max() >= EVIDENCE_CONF_LEVEL:
                conf += 1
            per_frame.setdefault(int(frame_id), []).append(int(known.argmax()))
    if total == 0:
        return EvidenceQuality(0.0, 0.0, 0, False)
    n_known = max(1, len(catalog.labels) - 1)
    div = float(np.mean([len(set(v)) / min(n_known, len(v)) for v in per_frame.values()]))
    cf = conf / total
    return EvidenceQuality(cf, div, len(per_frame), cf >= EVIDENCE_MIN_CONF_FRAC and div >= EVIDENCE_MIN_DIVERSITY)
```
In `run_fragment_solver`, right after `raw_evidence` is loaded (non-empty): `quality = assess_evidence_quality(raw_evidence, catalog)`; if `not quality.ok`: `log.error("fragment_solver: identity evidence is uninformative (confident=%.1f%% of detections, diversity=%.2f over %d frames) — refusing to split or assign identities. Check the classifier's fit_policy / preprocessing.", ...)`, set `smoothed_by_traj = None` before the PELT block (so PELT is skipped, `solve_global_assignment` gets no evidence and writes `unknown`/`NONE`), but still call `_annotate_smoothed_labels` with the smoothed posteriors for inspection. Add `from dataclasses import dataclass` at the top.

- [ ] **Step 4: Run tests**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/identity -v -x`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/core/individual/identity/offline.py tests/identity/test_offline_evidence_breaker.py
git commit -m "feat(identity): refuse to restructure trajectories on uninformative classifier evidence"
```

---

### Task 9: End-to-end acceptance on DEMO/ID + equivalence gates

**Files:**
- Create: `tools/equivalence/notes/2026-08-27-identity-repair-gate.md` (results record)
- Test: manual (below)

- [ ] **Step 1: Stamp the DEMO models and re-run OFFLINE post-processing**

```bash
M="/Users/neurorishika/Library/Application Support/hydra-suite/models/classification"
python scripts/stamp_fit_policy.py "$M/identity/20260429-105036_classifier_multihead_obiroi_colortag.multihead.json" --policy squash
python scripts/stamp_fit_policy.py "$M/orientation/20260429-104937_efficientnet_b0_obiroi_train1.pth" --policy squash
```
Copy `DEMO/ID/OFFLINE` to `DEMO/ID/OFFLINE_v2` (video + config only; delete `.inference_cache_ant/cnn_colortag.npz`, `headtail.npz`, and the evidence sidecar so they are rebuilt; keep `detection.npz`). Run the headless tracker on it:
`PYTHONPATH=$PWD/src python -m hydra_suite.trackerkit.cli --help` then run it with `DEMO/ID/OFFLINE_v2/ant_config.json` (the Qt-free runner in `trackerkit/cli.py` → `run_headless_tracking_session`).

- [ ] **Step 2: Measure acceptance**

```bash
PYTHONPATH=$PWD/src python - <<'EOF'
import pandas as pd
d="/Users/neurorishika/Projects/Rockefeller/Ruta/Presentation/DEMO/ID/OFFLINE_v2/"
f=pd.read_csv(d+"ant_tracking_final_with_individual.csv")
g=f.groupby("TrajectoryID").FrameID.count()
print("trajectories",len(g),"median len",g.median())
print("distinct smoothed labels/frame", f.groupby("FrameID").IdentityFinalSmoothedLabel.nunique().mean())
print("final source", f.IdentityFinalSource.value_counts(dropna=False).to_dict())
for tid in (30,36,58):
    m=f.TrajectoryID==tid
    print(tid, f.loc[m,"IdentityFinalLabel"].value_counts().head(2).to_dict())
EOF
grep -E "PELT found|re-merged|uninformative" DEMO/ID/OFFLINE_v2/ant_logs/*.log
```
Expected: trajectories ≤ 150 (OFF had 117); ≥ 15 distinct smoothed labels/frame; `PELT found` ≤ ~120; no `uninformative` line; tracks 30/36/58 → `blue_blue`, `orange_yellow`, `orange_green` or `green_orange`. If the breaker trips, the stamp did not take effect — check the WARNING/`fit_policy` line in the log before touching anything else.

- [ ] **Step 3: Equivalence matrix, MPS then CUDA** (per CLAUDE.md "The fast path"; baseline = local `main` at `efca3d71`, not `legacy/main`)

```bash
git worktree add --detach .worktrees/equiv-base efca3d71
REPO=$PWD WT=$PWD MAIN_SRC=$PWD/.worktrees/equiv-base/src WT_SRC=$PWD/src OUT=/tmp/equiv_idrepair RUNTIME=mps bash tools/equivalence/run_matrix.sh
```
Expected: `fly_obb`, `worm_bgsub`, `ant_obb_sequential`, `ant_pose_headtail`, `ant_obb_sleap` — byte-identical (head/tail model in fixtures: if its checkpoint is unstamped it now runs under `squash` → `ant_pose_headtail` θ may change; record the exact delta and whether the fixture model's training date is pre-2026-08-05). `ant_cnn_identity`, `emi_obb_identity` — identity columns differ by design; positions must be identical unless the fixture config has `identity_weight > 0` (state which). Repeat on mehek with `RUNTIME=cuda`. Write every number into `tools/equivalence/notes/2026-08-27-identity-repair-gate.md`.

- [ ] **Step 4: Commit the gate record**

```bash
git add tools/equivalence/notes/2026-08-27-identity-repair-gate.md
git commit -m "docs(equivalence): record identity-repair gate results (MPS + CUDA)"
```

---

### Task 10: Docs and merge hygiene

**Files:**
- Modify: `docs/developer-guide/runtime-integration.md` (new subsection "Classifier fit policy"), `docs/user-guide/` identity page (note: legacy models auto-resolve to squash; how to stamp), `CLAUDE.md` (one line under Extension Points: "New classifier artifacts must carry `fit_policy`")
- On merge: `git mv` the spec and this plan into `docs/superpowers/specs/done/` and `docs/superpowers/plans/done/`, set `**Status:** Shipped — merged to main (<sha>)`.

- [ ] **Step 1: Write the docs** — three paragraphs: what `fit_policy` is, the legacy rule (absent → squash + warning, `scripts/stamp_fit_policy.py`), and the identity post-processing behaviour change (raw PELT units, remnant merge, re-merge, evidence breaker with its two thresholds and the ERROR line to look for).
- [ ] **Step 2: `make format && make lint && make docs-check`** — Expected: clean.
- [ ] **Step 3: Commit**

```bash
git add docs CLAUDE.md
git commit -m "docs: classifier fit_policy and identity post-processing repair"
```
- [ ] **Step 4: Adversarial review before merge** (memory: a different model, refute-mode) then merge `--no-ff` into local `main`, move docs to `done/` in the merge commit, remove the worktree.

## Follow-ups (out of scope, tracked)
- ONLINE vs OFFLINE differ in 14 fragment partitions downstream of the solver (identity conflict resolution / renumbering) — isolate separately.
- `identity_gates_trajectory_structure` only acts when `IdentityRealtimeCommitted` rows exist; document or make it act on offline labels too.
- Retrain the colour-tag and head/tail models through the current ClassKit path (letterbox, stamped, head-oriented dataset) once this ships; compare against the squash-resolved legacy models.
- Equivalence fixture with a stamped classifier and known tag colours (the identity path has no byte-level coverage).
