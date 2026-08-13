# Identity Overhaul — Phase 2: Calibration Workflow (ClassKit Integration) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fit temperature-scaling calibration at the tail of CNN classifier training (per factor for multihead), bake the temperature + a model-weight signature + ECE into the model artifact, consume the stored temperature automatically at tracking time, gate the Bayesian identity decoders on calibration when required, and expose calibration status/recalibration in the UI.

**Architecture:** Model softmax confidences are overconfident; feeding them into a log-space Bayesian identity decoder compounds into false certainty. Temperature scaling (Guo et al. 2017) — one scalar T per output head, fit on a held-out validation split by minimizing NLL — is the honesty fix. Every primitive already half-exists: a `TemperatureScaling` LBFGS fitter (in ClassKit), a persisted `<dataset>/val/<class>/…` split, a `CalibrationModel` that *applies* a temperature at inference, a `recommended_confidence_threshold` artifact-metadata pattern to mirror, and a `CNNConfig.calibration_temperature` consumption seam. This phase connects them: it adds calibration math to **Core** (so the Training layer can call it without an illegal upward import into ClassKit), fits at training tail, stores in all three artifact forms, surfaces on `ClassifierMetadata`, and makes `CNNConfig.calibration_temperature` fall back to the artifact instead of a hardcoded `1.0`.

**Tech Stack:** Python 3, PyTorch (LBFGS), NumPy, pytest, PySide6/Qt (ClassKit + TrackerKit dialogs).

**Spec:** `docs/superpowers/specs/2026-07-22-identity-overhaul-consolidated-design.md` — Layer 1 (Calibration as artifact), "Calibration Lifecycle (ClassKit integration)", Rollout "Phase 2". Builds on Phase 1 (`IdentityConfig.calibration_required` and `IdentityModelConfig.calibration` fields already reserved in `trackerkit/config/identity_schema.py`).

## Global Constraints

- **Dependency direction (CLAUDE.md), load-bearing here.** Core/Runtime/Data/**Training**/Utils must NOT import from any app-layer package (`classkit`, `trackerkit`, …). ClassKit's existing `TemperatureScaling` (`classkit/core/train/calibrate.py`) is in an app layer, so `training/runner.py` (Training layer) cannot import it. **The new calibration math lives in Core** (`core/individual/identity/calibration.py`), importable by Training and by app layers. ClassKit's embedding-trainer `TemperatureScaling` is left untouched (a future cleanup may delegate it to the Core function; out of scope here).
- **Refit must not invalidate the CNN cache.** The CNN cache stores RAW pre-calibration probabilities and excludes temperature from its key (`core/inference/cache/keys.py:230-238`, `core/inference/stages/cnn.py:73,81-83`). Do NOT add temperature to any cache key. Recalibration rewrites only artifact metadata.
- **No tracking-behavior change unless a model carries a stored temperature.** A model with no stored calibration keeps `calibration_temperature == 1.0` (today's behavior). The equivalence gate (positions byte-identical, identity off) must still pass on MPS + CUDA. Calibration only changes identity *confidences* for models that have been calibrated — assert this explicitly.
- **Per-factor storage, flat consumption in Phase 2.** Fit and STORE one temperature per factor (a `list[float]`) for multihead models. The current consume path (`CNNConfig.calibration_temperature`, a scalar → `evidence.py:_calibrate`) applies a single temperature; Phase 2 wires the fallback for the **flat/single-factor** case. Per-factor *application* over the true per-factor softmax is Phase 3's evidence stage (which consumes the stored per-factor list). Storing the list now is forward-compatible; do not try to collapse a multihead list into the scalar path.
- **Isolation.** Work in a git worktree branched from the Phase-1-merged local HEAD: `git worktree add .worktrees/identity-phase2 -b feat/identity-phase2 HEAD`. Never branch from origin.
- **Commit as the configured git user.** No `Co-Authored-By: Claude` trailer.
- **Before commit:** `make format` then `make lint` (the moderate flake8 gate; `make lint-moderate` does not exist). Revert unrelated isort/black drift. Kill stale `sleap`/`hydra` before heavy runs.
- **Verification:** unit tests on `hydra-mps`; a real end-to-end train→calibrate→load check on `hydra-mps`; equivalence gate on MPS + CUDA (mehek).

---

## File Structure

**Create:**
- `tests/identity/test_calibration_math.py` — ECE, weight-signature, fit-from-logits unit tests.
- `tests/identity/test_calibration_artifact_roundtrip.py` — checkpoint/sidecar/manifest store→parse round-trip.
- `tests/identity/test_calibration_consume_gate.py` — CNNConfig fallback + mandatory-calibration gate.
- `src/hydra_suite/classkit/gui/dialogs/recalibrate_dialog.py` — ClassKit Recalibrate dialog (BaseDialog).

**Modify:**
- `src/hydra_suite/core/individual/identity/calibration.py` — add `fit_temperature`, `expected_calibration_error`, `model_weight_signature` (Core home for calibration math).
- `src/hydra_suite/training/calibration_fit.py` (Create) — `fit_calibration_from_val(...)`: run a val loader, collect (per-factor) logits, fit, compute ECE before/after. Training layer (imports Core).
- `src/hydra_suite/training/runner.py` — call the fit at the tiny + torchvision + multihead training tails; thread temps/signature/ECE into the save calls.
- `src/hydra_suite/training/torchvision_model.py` — accept + write `calibration_temperature`, `calibration_signature`, `calibration_ece` in the checkpoint dict.
- `src/hydra_suite/training/model_publish.py` — write the same three into the `.v2meta.json` sidecar and `.multihead.json` manifest.
- `src/hydra_suite/core/individual/classification/backend.py` — add `calibration_temperature`, `calibration_signature`, `calibration_ece` to `ClassifierMetadata` + normalizer + the 4 parse sites.
- `src/hydra_suite/core/inference/config.py` — `build_inference_config_from_params`: `CNNConfig.calibration_temperature` falls back to `ClassifierMetadata`; add the mandatory-calibration gate.
- `src/hydra_suite/trackerkit/engine_params.py` — emit `IDENTITY_CALIBRATION_REQUIRED` from `IdentityConfig.calibration_required`.
- `src/hydra_suite/trackerkit/gui/dialogs/cnn_identity_import_dialog.py` — calibration status row.
- `src/hydra_suite/classkit/gui/main_window.py` — Recalibrate action + `_on_training_success` ECE log line.

**Explicitly deferred (with reason):**
- **Robustness knobs** (`per_frame_evidence_cap`, `prob_floor`, `source_weights` — Phase-1-reserved `RobustnessConfig`): the spec bundles them under Phase 2, but they are *applied* only by Phase 3's `IdentityEvidenceStage`. Wiring a knob nothing reads is dead config (YAGNI). Phase 3 populates + applies them. Phase 2 leaves the reserved fields as-is.
- **Per-factor calibration application** in the decoder: Phase 3 (evidence stage over true per-factor softmax).
- **ClassKit `TemperatureScaling` de-duplication**: leave the embedding-trainer copy; optional later cleanup.

---

## Interfaces (defined once, referenced by every task)

```python
# core/individual/identity/calibration.py  (added — Core, no app imports)
def fit_temperature(logits: np.ndarray, labels: np.ndarray, max_iter: int = 50) -> float:
    """LBFGS/NLL temperature (Guo et al.); clamps to [0.1, 10.0]. logits (N,K), labels (N,)."""

def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Standard ECE over `n_bins` equal-width confidence bins. probs (N,K), labels (N,)."""

def model_weight_signature(state_dict: "collections.OrderedDict") -> str:
    """Deterministic sha1 over sorted (name, tensor-bytes) — identifies the trained weights."""

# training/calibration_fit.py  (Create — Training layer)
@dataclass
class CalibrationResult:
    temperatures: list[float]     # one per factor (len 1 for flat)
    signature: str                # model_weight_signature of the fitted weights
    ece_before: list[float]       # per factor
    ece_after: list[float]        # per factor

def fit_calibration_from_val(
    model, val_loader, device: str, *, split_logits=None, num_factors: int = 1,
) -> CalibrationResult:
    """Run val_loader through model (eval, no grad); collect logits (flat) or per-factor
    logits via `split_logits(logits)->list[Tensor]`; per factor fit_temperature + ECE
    before/after. Labels: (N,) flat, (N,num_factors) multihead."""

# core/individual/classification/backend.py  ClassifierMetadata (added fields)
calibration_temperature: tuple[float, ...] | None   # per-factor; None = uncalibrated
calibration_signature: str | None
calibration_ece: tuple[float, ...] | None            # ece_after per factor

# core/individual/classification/backend.py  (added helper)
def calibration_status(meta: ClassifierMetadata, current_signature: str | None) -> str:
    """'calibrated' | 'stale' | 'uncalibrated'. 'stale' iff a temperature is stored but its
    signature != current_signature."""
```

Artifact keys added to every form (checkpoint dict, `.v2meta.json`, `.multihead.json`):
`calibration_temperature` (list[float]), `calibration_signature` (str), `calibration_ece` (list[float]).

---

## Task 1: Calibration math in Core (ECE, weight-signature, fit)

**Files:**
- Modify: `src/hydra_suite/core/individual/identity/calibration.py`
- Test: `tests/identity/test_calibration_math.py`

**Interfaces:** Produces `fit_temperature`, `expected_calibration_error`, `model_weight_signature` (consumed by Tasks 2, 5, 7).

- [ ] **Step 1: Write failing tests**

```python
# tests/identity/test_calibration_math.py
import numpy as np
from collections import OrderedDict
import torch
from hydra_suite.core.individual.identity.calibration import (
    fit_temperature, expected_calibration_error, model_weight_signature,
)


def _overconfident_logits(n=2000, k=4, seed=0):
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, k, size=n)
    logits = rng.normal(0, 1, size=(n, k))
    # make the true class win but inflate magnitude → overconfident
    logits[np.arange(n), labels] += 2.5
    logits *= 3.0
    return logits.astype(np.float64), labels.astype(np.int64)


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def test_fit_temperature_reduces_ece_on_overconfident_set():
    logits, labels = _overconfident_logits()
    ece_before = expected_calibration_error(_softmax(logits), labels)
    t = fit_temperature(logits, labels)
    assert t > 1.0  # overconfident ⇒ temperature > 1 softens
    ece_after = expected_calibration_error(_softmax(logits / t), labels)
    assert ece_after < ece_before


def test_fit_temperature_clamped_range():
    logits, labels = _overconfident_logits()
    t = fit_temperature(logits, labels)
    assert 0.1 <= t <= 10.0


def test_ece_zero_for_perfectly_calibrated():
    # one-hot-ish perfectly-confident-and-correct ⇒ ECE ~ 0
    n, k = 500, 3
    labels = np.tile(np.arange(k), n // k + 1)[:n]
    probs = np.full((n, k), 1e-6)
    probs[np.arange(n), labels] = 1.0
    probs /= probs.sum(1, keepdims=True)
    assert expected_calibration_error(probs, labels) < 1e-3


def test_weight_signature_deterministic_and_sensitive():
    sd1 = OrderedDict({"w": torch.ones(3, 3), "b": torch.zeros(3)})
    sd2 = OrderedDict({"w": torch.ones(3, 3), "b": torch.zeros(3)})
    sd3 = OrderedDict({"w": torch.ones(3, 3), "b": torch.ones(3)})
    assert model_weight_signature(sd1) == model_weight_signature(sd2)
    assert model_weight_signature(sd1) != model_weight_signature(sd3)
    assert isinstance(model_weight_signature(sd1), str) and len(model_weight_signature(sd1)) >= 8
```

- [ ] **Step 2: Run — expect ImportError**

Run: `python -m pytest tests/identity/test_calibration_math.py -v`
Expected: FAIL (functions not defined).

- [ ] **Step 3: Implement in `calibration.py`**

Append to `src/hydra_suite/core/individual/identity/calibration.py` (mirrors ClassKit's `TemperatureScaling.fit` LBFGS/NLL, but pure-functional and in Core):

```python
import hashlib
import numpy as np


def fit_temperature(logits: np.ndarray, labels: np.ndarray, max_iter: int = 50) -> float:
    """Fit a single temperature by NLL minimization (Guo et al. 2017), clamped [0.1, 10.0]."""
    import torch
    import torch.nn.functional as F

    z = torch.as_tensor(np.asarray(logits), dtype=torch.float32)
    y = torch.as_tensor(np.asarray(labels), dtype=torch.long)
    if z.ndim != 2 or z.shape[0] == 0:
        return 1.0
    t = torch.nn.Parameter(torch.ones(1))
    opt = torch.optim.LBFGS([t], lr=0.01, max_iter=max_iter)

    def _closure():
        opt.zero_grad()
        loss = F.cross_entropy(z / t.clamp_min(1e-3), y)
        loss.backward()
        return loss

    opt.step(_closure)
    return float(np.clip(t.item(), 0.1, 10.0))


def expected_calibration_error(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 15
) -> float:
    """Equal-width-bin ECE over predicted-class confidence."""
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels)
    if probs.ndim != 2 or probs.shape[0] == 0:
        return 0.0
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = probs.shape[0]
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if not m.any():
            continue
        ece += (m.sum() / n) * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def model_weight_signature(state_dict) -> str:
    """Deterministic sha1 over sorted (name, contiguous tensor bytes)."""
    h = hashlib.sha1()
    for name in sorted(state_dict.keys()):
        t = state_dict[name]
        h.update(name.encode("utf-8"))
        arr = t.detach().cpu().contiguous().numpy()
        h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()
```

- [ ] **Step 4: Run — expect PASS**

Run: `python -m pytest tests/identity/test_calibration_math.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Format, lint, commit**

```bash
make format && make lint
git add src/hydra_suite/core/individual/identity/calibration.py tests/identity/test_calibration_math.py
git commit -m "feat(identity): calibration math in Core (ECE, weight-signature, temperature fit)"
```

---

## Task 2: `fit_calibration_from_val` (Training layer)

Runs a validation loader through the trained model and produces per-factor temperatures + ECE + weight signature. Pure w.r.t. Qt; unit-tested with a synthetic model + loader.

**Files:**
- Create: `src/hydra_suite/training/calibration_fit.py`
- Test: `tests/identity/test_calibration_math.py` (append) or a new `test_calibration_fit.py`

**Interfaces:** Consumes Task 1. Produces `CalibrationResult`, `fit_calibration_from_val` (consumed by Task 3, 4).

- [ ] **Step 1: Write the failing test**

```python
# tests/identity/test_calibration_fit.py
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from hydra_suite.training.calibration_fit import fit_calibration_from_val, CalibrationResult


class _FlatNet(torch.nn.Module):
    def __init__(self, k=4):
        super().__init__()
        self.lin = torch.nn.Linear(4, k)

    def forward(self, x):
        return self.lin(x) * 3.0  # inflate ⇒ overconfident


def _flat_loader(n=800, k=4, seed=1):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 4, generator=g)
    y = torch.randint(0, k, (n,), generator=g)
    return DataLoader(TensorDataset(x, y), batch_size=64)


def test_flat_returns_single_temperature_and_ece_drop():
    model = _FlatNet()
    res = fit_calibration_from_val(model, _flat_loader(), "cpu", num_factors=1)
    assert isinstance(res, CalibrationResult)
    assert len(res.temperatures) == 1
    assert len(res.ece_before) == 1 and len(res.ece_after) == 1
    assert res.ece_after[0] <= res.ece_before[0] + 1e-6
    assert isinstance(res.signature, str) and res.signature


def test_multihead_returns_per_factor_temperatures():
    # 2 factors of width 3 and 2; labels (N,2)
    class _MH(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(4, 5)  # 3 + 2 concat

        def forward(self, x):
            return self.lin(x) * 3.0

    g = torch.Generator().manual_seed(2)
    x = torch.randn(400, 4, generator=g)
    y = torch.stack([torch.randint(0, 3, (400,), generator=g),
                     torch.randint(0, 2, (400,), generator=g)], dim=1)
    loader = DataLoader(TensorDataset(x, y), batch_size=50)

    def split(logits):  # (N,5) -> [(N,3),(N,2)]
        return [logits[:, :3], logits[:, 3:]]

    res = fit_calibration_from_val(_MH(), loader, "cpu", split_logits=split, num_factors=2)
    assert len(res.temperatures) == 2
    assert len(res.ece_before) == 2 and len(res.ece_after) == 2
```

- [ ] **Step 2: Run — expect ImportError**

Run: `python -m pytest tests/identity/test_calibration_fit.py -v` → FAIL.

- [ ] **Step 3: Implement `calibration_fit.py`**

```python
# src/hydra_suite/training/calibration_fit.py
"""Fit temperature-scaling calibration from a validation loader (Training layer).

Imports the calibration math from Core (allowed: Training -> Core). Produces one
temperature per factor plus ECE before/after and a model-weight signature, for
persistence into the model artifact.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from hydra_suite.core.individual.identity.calibration import (
    expected_calibration_error,
    fit_temperature,
    model_weight_signature,
)


@dataclass
class CalibrationResult:
    temperatures: list[float]
    signature: str
    ece_before: list[float]
    ece_after: list[float]


def _softmax_np(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


@torch.no_grad()
def _collect(model, val_loader, device, split_logits, num_factors):
    model.eval()
    per_factor_logits: list[list[np.ndarray]] = [[] for _ in range(num_factors)]
    per_factor_labels: list[list[np.ndarray]] = [[] for _ in range(num_factors)]
    for batch in val_loader:
        xs, ys = batch[0].to(device), batch[1]
        out = model(xs)
        parts = split_logits(out) if split_logits is not None else [out]
        for k in range(num_factors):
            per_factor_logits[k].append(parts[k].detach().cpu().numpy())
            yk = ys if ys.ndim == 1 else ys[:, k]
            per_factor_labels[k].append(yk.detach().cpu().numpy())
    return per_factor_logits, per_factor_labels


def fit_calibration_from_val(
    model, val_loader, device: str, *, split_logits=None, num_factors: int = 1
) -> CalibrationResult:
    logits_by_f, labels_by_f = _collect(
        model, val_loader, device, split_logits, num_factors
    )
    temps, ece_b, ece_a = [], [], []
    for k in range(num_factors):
        logits = np.concatenate(logits_by_f[k], axis=0)
        labels = np.concatenate(labels_by_f[k], axis=0)
        ece_b.append(expected_calibration_error(_softmax_np(logits), labels))
        t = fit_temperature(logits, labels)
        temps.append(t)
        ece_a.append(expected_calibration_error(_softmax_np(logits / t), labels))
    sig = model_weight_signature(model.state_dict())
    return CalibrationResult(
        temperatures=temps, signature=sig, ece_before=ece_b, ece_after=ece_a
    )
```

- [ ] **Step 4: Run — expect PASS**

Run: `python -m pytest tests/identity/test_calibration_fit.py -v` → PASS (2 tests).

- [ ] **Step 5: Format, lint, commit**

```bash
make format && make lint
git add src/hydra_suite/training/calibration_fit.py tests/identity/test_calibration_fit.py
git commit -m "feat(training): fit_calibration_from_val — per-factor temperature + ECE from val loader"
```

---

## Task 3: Store calibration in the checkpoint dicts (both save functions)

Add three artifact keys to the tiny and torchvision checkpoint writers. This task only *plumbs the keys through the save functions*; the actual fit calls are Task 4 (so this stays a small, independently-reviewable schema change).

**Files:**
- Modify: `src/hydra_suite/training/torchvision_model.py:389-427` (`save_torchvision_checkpoint`)
- Modify: `src/hydra_suite/training/runner.py:645-689` (`_save_tiny_checkpoint`)
- Test: `tests/identity/test_calibration_artifact_roundtrip.py` (checkpoint half)

**Interfaces:** Produces checkpoint dicts carrying `calibration_temperature`/`calibration_signature`/`calibration_ece`. Consumed by Task 5 (parse) and Task 4 (fill).

- [ ] **Step 1: Write the failing test** (checkpoint round-trip — save then torch.load, assert keys present)

```python
# tests/identity/test_calibration_artifact_roundtrip.py
import torch
from hydra_suite.training.torchvision_model import save_torchvision_checkpoint


def test_torchvision_checkpoint_carries_calibration(tmp_path):
    model = torch.nn.Linear(4, 3)
    p = tmp_path / "m.pth"
    save_torchvision_checkpoint(
        path=str(p), model=model, backbone="resnet18",
        factor_names=["flat"], class_names_per_factor=[["a", "b", "c"]],
        input_size=(64, 64), num_classes=3, monochrome=False, best_val_acc=0.9,
        history=[], trainable_layers=0, backbone_lr_scale=1.0,
        calibration_temperature=[1.7], calibration_signature="abc123",
        calibration_ece=[0.04],
    )
    ckpt = torch.load(str(p), map_location="cpu", weights_only=False)
    assert ckpt["calibration_temperature"] == [1.7]
    assert ckpt["calibration_signature"] == "abc123"
    assert ckpt["calibration_ece"] == [0.04]
```

(Adapt the `save_torchvision_checkpoint` kwargs to its real signature — READ `torchvision_model.py:345` first; the names above must match.)

- [ ] **Step 2: Run — expect FAIL** (unexpected-kwarg TypeError or missing keys).

- [ ] **Step 3: Add the params + dict keys**

In `save_torchvision_checkpoint` (`torchvision_model.py:345`) add three optional params defaulting to `None`, and inside the `ckpt` dict (`:389-402`) add:

```python
        "calibration_temperature": (
            list(calibration_temperature) if calibration_temperature is not None else None
        ),
        "calibration_signature": calibration_signature,
        "calibration_ece": (
            list(calibration_ece) if calibration_ece is not None else None
        ),
```

Do the same in `_save_tiny_checkpoint` (`runner.py:645`, dict at `:672-687`) — add the three kwargs (default `None`) and the same three dict entries.

- [ ] **Step 4: Run — expect PASS**

Run: `python -m pytest tests/identity/test_calibration_artifact_roundtrip.py::test_torchvision_checkpoint_carries_calibration -v` → PASS.

- [ ] **Step 5: Format, lint, commit**

```bash
make format && make lint
git add src/hydra_suite/training/torchvision_model.py src/hydra_suite/training/runner.py \
        tests/identity/test_calibration_artifact_roundtrip.py
git commit -m "feat(training): checkpoint dicts carry calibration_temperature/signature/ece"
```

---

## Task 4: Fit calibration at the training tails and pass into the save calls

Wire `fit_calibration_from_val` into the three training branches, threading the result into the Task-3 save params. This is the behavioral core.

**Files:**
- Modify: `src/hydra_suite/training/runner.py` — tiny (`~:922` before `_save_tiny_checkpoint` at `:956`); torchvision (`_train_custom_classify`, post-`_run_torchvision_training_loop`); multihead (`_train_multihead_shared_classify`, `:1677`, using `_split_logits_per_factor` at `:1869` and saving at `:1941/1964`).
- Test: an end-to-end real-training smoke (small, `@pytest.mark.slow` acceptable) + the equivalence-neutrality assertion.

**Interfaces:** Consumes Task 2, Task 3. No new interface.

- [ ] **Step 1: Write a focused end-to-end test** (small real train → assert artifact carries a plausible temperature + ece_after ≤ ece_before)

```python
# tests/identity/test_calibration_end_to_end.py  (mark slow; small dataset via tmp_path ImageFolder)
# Build a tiny 2-class ImageFolder train/ + val/ (a few solid-color images per class),
# call run_training for the tiny arch, load the produced checkpoint, and assert:
#   ckpt["calibration_temperature"] is a list of one float in [0.1, 10.0]
#   ckpt["calibration_signature"] is a non-empty str
#   ckpt["calibration_ece"] is a list of one float
# READ runner.run_training's real signature + the tiny arch role string before writing this.
```

(If a full `run_training` smoke is too heavy for CI, instead unit-test the tiny branch's new fit-and-save block by extracting it into a helper `_calibrate_and_pack(model, val_loader, device, num_factors, split) -> dict` that returns the three kwargs, and test that helper directly with a synthetic model/loader. Prefer the helper — it keeps the seam testable without a full training run.)

- [ ] **Step 2: Run — expect FAIL.**

- [ ] **Step 3: Implement the fit calls**

Introduce one shared helper near the training branches:

```python
# runner.py (module level, near the classify trainers)
def _calibrate_and_pack(model, val_loader, device, *, num_factors=1, split_logits=None):
    """Fit calibration on the best model + val loader; return save kwargs (or empty)."""
    if val_loader is None:
        return {}
    from hydra_suite.training.calibration_fit import fit_calibration_from_val

    res = fit_calibration_from_val(
        model, val_loader, device, split_logits=split_logits, num_factors=num_factors
    )
    logger.info(
        "Calibration fit: T=%s  ECE %s -> %s",
        [round(t, 3) for t in res.temperatures],
        [round(e, 4) for e in res.ece_before],
        [round(e, 4) for e in res.ece_after],
    )
    return {
        "calibration_temperature": res.temperatures,
        "calibration_signature": res.signature,
        "calibration_ece": res.ece_after,
    }
```

- **Tiny** (`runner.py:~922`): after the loop returns `best_state`, load it into `model` (`model.load_state_dict(best_state)`), then `cal = _calibrate_and_pack(model, val_loader, device, num_factors=1)` and pass `**cal` into `_save_tiny_checkpoint(...)` at `:956`.
- **Torchvision flat** (`_train_custom_classify`, after `_run_torchvision_training_loop`): reload the best checkpoint into `model` (the loop saved it at `best_ckpt_path`), `cal = _calibrate_and_pack(model, val_loader, device, num_factors=1)`, then re-save via `save_torchvision_checkpoint(..., **cal)` OR patch the existing best checkpoint dict with the three keys and re-`torch.save`. Prefer re-invoking `save_torchvision_checkpoint` with the same args + `**cal` to keep one writer.
- **Multihead shared** (`_train_multihead_shared_classify`, `:1677`): `cal = _calibrate_and_pack(model, val_loader, device, num_factors=len(factor_names), split_logits=_split_logits_per_factor)`; pass `**cal` into the `save_torchvision_checkpoint(...)` calls at `:1941`/`:1964`.

Threading `val_loader` into scope: tiny already has it (`:890`); torchvision/multihead build it in the loop — return it from the loop or rebuild it from `<dataset>/val` at the seam (rebuild is cheap and side-effect free).

- [ ] **Step 4: Run the helper test + a quick real tiny-train smoke on hydra-mps** → PASS; log shows `ECE x -> y` with y ≤ x.

- [ ] **Step 5: Format, lint, commit**

```bash
make format && make lint
git add src/hydra_suite/training/runner.py tests/identity/test_calibration_end_to_end.py
git commit -m "feat(training): fit temperature calibration at CNN training tails (tiny/torchvision/multihead)"
```

---

## Task 5: Propagate to sidecar/manifest + surface on `ClassifierMetadata`

**Files:**
- Modify: `src/hydra_suite/training/model_publish.py` — `.v2meta.json` (`:766/782-785`, payload `_normalize_classifier_meta:212-231`) and `.multihead.json` (`write_classifier_multihead_manifest:168-179`).
- Modify: `src/hydra_suite/core/individual/classification/backend.py` — `ClassifierMetadata` fields (`:35-68`), normalizer (near `:71`), 4 parse sites (`:182, :240, :272, :358`), and add `calibration_status(meta, current_signature)`.
- Test: `tests/identity/test_calibration_artifact_roundtrip.py` (sidecar + manifest + metadata halves).

**Interfaces:** Consumes Task 3's checkpoint keys. Produces `ClassifierMetadata.calibration_*` + `calibration_status` (consumed by Tasks 6, 7, 8).

- [ ] **Step 1: Write failing tests** — for each artifact form, write the three keys, parse via `ClassifierBackend(path).metadata`, assert `meta.calibration_temperature == (…)`, `meta.calibration_signature == …`; plus `calibration_status`:

```python
def test_calibration_status_transitions():
    from hydra_suite.core.individual.classification.backend import (
        ClassifierMetadata, calibration_status,
    )
    base = dict(arch="resnet18", input_size=(64, 64), is_multihead=False,
                factor_names=["flat"], class_names_per_factor=[["a", "b"]],
                monochrome=False, recommended_confidence_threshold=None, source_path="x")
    uncal = ClassifierMetadata(**base, calibration_temperature=None,
                               calibration_signature=None, calibration_ece=None)
    cal = ClassifierMetadata(**base, calibration_temperature=(1.5,),
                             calibration_signature="sigA", calibration_ece=(0.03,))
    assert calibration_status(uncal, "sigA") == "uncalibrated"
    assert calibration_status(cal, "sigA") == "calibrated"
    assert calibration_status(cal, "sigB") == "stale"
```

- [ ] **Step 2: Run — expect FAIL** (TypeError: unexpected kwarg / missing attr).

- [ ] **Step 3: Implement**

`ClassifierMetadata` — add three frozen fields after `recommended_confidence_threshold` (`:67`):

```python
    calibration_temperature: tuple[float, ...] | None = None
    calibration_signature: str | None = None
    calibration_ece: tuple[float, ...] | None = None
```

Add a normalizer beside `_normalize_recommended_confidence_threshold` (`:71`):

```python
def _normalize_calibration_temperature(raw) -> tuple[float, ...] | None:
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        raw = [raw]
    try:
        vals = tuple(float(x) for x in raw)
    except (TypeError, ValueError):
        return None
    return vals or None
```

At each of the 4 `ClassifierMetadata(...)` constructions (`:182, :240, :272, :358`) add (source is the checkpoint dict `ckpt`, the sidecar `sidecar`, or the manifest `data` per site):

```python
    calibration_temperature=_normalize_calibration_temperature(
        <source>.get("calibration_temperature")
    ),
    calibration_signature=(<source>.get("calibration_signature") or None),
    calibration_ece=_normalize_calibration_temperature(
        <source>.get("calibration_ece")
    ),
```

Add the status helper (module level):

```python
def calibration_status(meta: "ClassifierMetadata", current_signature: str | None) -> str:
    if not meta.calibration_temperature:
        return "uncalibrated"
    if current_signature is not None and meta.calibration_signature != current_signature:
        return "stale"
    return "calibrated"
```

`model_publish.py` — in `_normalize_classifier_meta` (`:212-231`) and `write_classifier_multihead_manifest` payload (`:168-179`) and the `.v2meta.json` write (`:782-785`), carry the three keys through from the incoming meta (mirror exactly how `recommended_confidence_threshold` is threaded at `:221-228`).

- [ ] **Step 4: Run — expect PASS** (all round-trip + status tests).

- [ ] **Step 5: Format, lint, commit**

```bash
make format && make lint
git add src/hydra_suite/core/individual/classification/backend.py \
        src/hydra_suite/training/model_publish.py \
        tests/identity/test_calibration_artifact_roundtrip.py
git commit -m "feat(identity): surface calibration_temperature/signature/ece on ClassifierMetadata + sidecar/manifest"
```

---

## Task 6: Consume — `CNNConfig.calibration_temperature` falls back to the artifact

**Files:**
- Modify: `src/hydra_suite/core/inference/config.py:858-873` (`build_inference_config_from_params`).
- Test: `tests/identity/test_calibration_consume_gate.py` (fallback half).

**Interfaces:** Consumes Task 5 (`ClassifierMetadata.calibration_temperature`). No new interface.

- [ ] **Step 1: Write the failing test** — a params dict with a CNN classifier pointing at a calibrated flat artifact (write one via Task 3's save) and NO `calibration_temperature` in params ⇒ `build_inference_config_from_params(params)` yields a `CNNConfig` whose `calibration_temperature == stored T`. And: an explicit params `calibration_temperature` OVERRIDES the artifact (params win). And: an uncalibrated artifact ⇒ `1.0`.

- [ ] **Step 2: Run — expect FAIL** (still hardcoded `1.0`).

- [ ] **Step 3: Implement the fallback**

At `config.py:867-872`, replace the hardcoded fallback chain so that when params omit both `calibration_temperature` and `temperature`, it reads the artifact:

```python
        calibration_temperature=_resolve_cnn_temperature(cnn_cfg_dict, cnn_model_path),
```

with a module helper:

```python
def _resolve_cnn_temperature(cnn_cfg_dict: dict, model_path: str) -> float:
    explicit = cnn_cfg_dict.get("calibration_temperature", cnn_cfg_dict.get("temperature"))
    if explicit is not None:
        return float(explicit)
    # Fall back to the stored per-factor temperature; flat consume uses factor 0.
    try:
        from hydra_suite.core.individual.classification.backend import ClassifierBackend
        from hydra_suite.runtime.resolver import ResolvedBackend

        backend = ClassifierBackend(model_path, ResolvedBackend("torch", "cpu", False))
        try:
            temps = backend.metadata.calibration_temperature
        finally:
            backend.close()
        if temps:
            return float(temps[0])
    except Exception:
        pass
    return 1.0
```

(Confirm the cheapest way to read metadata without loading weights — if `ClassifierBackend` construction loads the model, prefer a lighter metadata-only parse; READ `backend.py` for a metadata-only entry point and use it. The behavior — stored T when present, else 1.0 — is what the test pins.)

- [ ] **Step 4: Run — expect PASS.**

- [ ] **Step 5: Format, lint, commit**

```bash
make format && make lint
git add src/hydra_suite/core/inference/config.py tests/identity/test_calibration_consume_gate.py
git commit -m "feat(inference): CNNConfig.calibration_temperature falls back to artifact metadata"
```

---

## Task 7: Mandatory-calibration gate

When `IdentityConfig.calibration_required` is set, any `unique_identifier` CNN model lacking a matching-signature calibration must make the Bayesian decoders refuse to run with a clear error naming the recalibrate action; a user override downgrades to a warning.

**Files:**
- Modify: `src/hydra_suite/trackerkit/engine_params.py` — emit `IDENTITY_CALIBRATION_REQUIRED` from `identity_cfg.calibration_required` (and extend the golden — `tests/test_get_parameters_dict_characterization.py::test_identity_keys_byte_identical` — with the new key; capture its baseline value first).
- Modify: `src/hydra_suite/core/inference/config.py` — in `build_inference_config_from_params`, after building CNN phases, run the gate.
- Test: `tests/identity/test_calibration_consume_gate.py` (gate half).

**Interfaces:** Consumes Task 5 (`calibration_status`), Task 6.

- [ ] **Step 1: Write failing tests** — params with `IDENTITY_CALIBRATION_REQUIRED=True`, a `unique_identifier` CNN classifier pointing at an UNCALIBRATED artifact ⇒ `build_inference_config_from_params` raises a clear error mentioning "recalibrate". Same but calibrated & matching signature ⇒ no raise. `calibration_required=False` ⇒ never raises. Override flag ⇒ warns (caplog), no raise.

- [ ] **Step 2: Run — expect FAIL.**

- [ ] **Step 3: Implement the gate** — a helper `_gate_calibration(params, cnn_phases)` that, when `params.get("IDENTITY_CALIBRATION_REQUIRED")` and the classifier is `unique_identifier`, computes the artifact's `current_signature` (via `model_weight_signature` over the loaded state_dict, or a stored self-signature) and calls `calibration_status(meta, current_signature)`; on `"uncalibrated"`/`"stale"` raise `CalibrationRequiredError` (add to `core/individual/classification/errors.py`) unless an override key is set, in which case `logger.warning(...)`. Emit `IDENTITY_CALIBRATION_REQUIRED` in `engine_params.py` from `identity_cfg.calibration_required`.

- [ ] **Step 4: Run — expect PASS** (gate raises/warns/passes across the four cases; golden updated).

- [ ] **Step 5: Format, lint, commit**

```bash
make format && make lint
git add src/hydra_suite/core/inference/config.py src/hydra_suite/core/individual/classification/errors.py \
        src/hydra_suite/trackerkit/engine_params.py tests/identity/test_calibration_consume_gate.py \
        tests/test_get_parameters_dict_characterization.py
git commit -m "feat(identity): mandatory-calibration gate on unique_identifier models"
```

---

## Task 8: TrackerKit CNN-import dialog — calibration status row

**Files:**
- Modify: `src/hydra_suite/trackerkit/gui/dialogs/cnn_identity_import_dialog.py` — `describe_cnn_identity_candidate` (`:33-42`) returns calibration fields; add a `layout.addRow("Calibration:", QLabel(...))` after the threshold row (`:73`).
- Test: a focused unit test of `describe_cnn_identity_candidate` (no Qt exec) asserting the calibration summary keys for a calibrated vs uncalibrated artifact.

**Interfaces:** Consumes Task 5.

- [ ] **Step 1: Failing test** — `describe_cnn_identity_candidate(path)` for a calibrated artifact returns `{"calibration_status": "calibrated", "calibration_temperature": (…)}`; uncalibrated returns `"uncalibrated"`.

- [ ] **Step 2: Run — FAIL.**

- [ ] **Step 3: Implement** — in `describe_cnn_identity_candidate`, after reading `meta`, compute `sig = model_weight_signature(...)` for the model (or read a stored self-signature) and `status = calibration_status(meta, sig)`; add `"calibration_status": status`, `"calibration_temperature": meta.calibration_temperature` to the dict. In `__init__`, after `:73`:

```python
        cal = summary.get("calibration_status", "uncalibrated")
        temps = summary.get("calibration_temperature")
        cal_text = {
            "calibrated": f"calibrated (T={', '.join(f'{t:.2f}' for t in temps)})" if temps else "calibrated",
            "stale": "stale — recalibrate (weights changed since calibration)",
            "uncalibrated": "not calibrated",
        }.get(cal, "not calibrated")
        layout.addRow("Calibration:", QLabel(cal_text))
```

- [ ] **Step 4: Run — PASS.**

- [ ] **Step 5: Format, lint, commit**

```bash
make format && make lint
git add src/hydra_suite/trackerkit/gui/dialogs/cnn_identity_import_dialog.py tests/identity/test_import_dialog_calibration.py
git commit -m "feat(trackerkit): CNN-import dialog shows calibration status"
```

---

## Task 9: ClassKit — Recalibrate action + ECE-in-report

**Files:**
- Create: `src/hydra_suite/classkit/gui/dialogs/recalibrate_dialog.py` (BaseDialog subclass).
- Modify: `src/hydra_suite/classkit/gui/main_window.py` — `setup_menus` (`:346`, `&Compute` menu `:402`) add `recalibrate_action` → `self.recalibrate_model`; `_on_training_success` (`:7988`) append an ECE log line.
- Test: a headless test of the recalibrate *logic* (locate `<dataset>/val`, refit, rewrite artifact metadata) extracted into a pure helper `recalibrate_artifact(model_path, val_dir) -> CalibrationResult`, tested without Qt.

**Interfaces:** Consumes Tasks 2, 5. The Qt wiring is thin; the tested substance is `recalibrate_artifact`.

- [ ] **Step 1: Failing test** — build a small artifact + a tmp `val/` ImageFolder; `recalibrate_artifact(path, val_dir)` refits, rewrites the checkpoint's `calibration_temperature`/`signature`/`ece`, and returns a `CalibrationResult`; re-parsing the artifact shows `calibration_status(meta, sig) == "calibrated"`.

- [ ] **Step 2: Run — FAIL.**

- [ ] **Step 3: Implement**
  - `recalibrate_artifact(model_path, val_dir)` (put in `training/calibration_fit.py`): load the model + build a val loader over `val_dir` (reuse the training loader builder), call `fit_calibration_from_val`, then rewrite the artifact metadata in place (torch.load → update the three keys → torch.save; for `.v2meta.json`/`.multihead.json` rewrite the JSON). No weight change.
  - `recalibrate_dialog.py`: a `BaseDialog` that lets the user confirm the model + the val source (default `classkit_export_dir(self.project_path)/val`; if absent, prompt to pick a labeled dataset), runs `recalibrate_artifact` in a worker (mirror the local `QRunnable`+`TaskSignals` pattern used by `ClassKitTrainingWorker`, or `BaseWorker`), and reports the resulting ECE.
  - `main_window.setup_menus`: add the action after `:426` (mirror `train_action`); gate `setEnabled(self.project_path is not None)`.
  - `_on_training_success` (`:7988`): if `results` carry `calibration_ece`/`_before`, `dialog.append_log(f"Calibration ECE {before} → {after}")`.

- [ ] **Step 4: Run — PASS** (the `recalibrate_artifact` test). Manually smoke the ClassKit menu action on hydra-mps once.

- [ ] **Step 5: Format, lint, commit**

```bash
make format && make lint
git add src/hydra_suite/classkit/gui/dialogs/recalibrate_dialog.py \
        src/hydra_suite/classkit/gui/main_window.py \
        src/hydra_suite/training/calibration_fit.py \
        tests/identity/test_recalibrate.py
git commit -m "feat(classkit): Recalibrate action + ECE reported after training"
```

---

## Phase-End Gate

- [ ] **Calibration-neutrality equivalence (MPS + CUDA).** With NO model carrying a stored temperature (existing fixtures are uncalibrated), the full matrix must stay byte-identical to the Phase-1-merged baseline — calibration is inert until a model is calibrated. Run per CLAUDE.md, **main-HEAD baseline** (not legacy): `MAIN_SRC=$PWD/src WT_SRC=$PWD/.worktrees/identity-phase2/src`. Verify CSV row counts > 1.
- [ ] **Calibrated-model behavior check.** Calibrate one fixture identity model (via Task 9's recalibrate), re-run its clip, and confirm identity *confidence* columns change while *positions* stay byte-identical (identity is additive to geometry). This proves the temperature actually flows.
- [ ] **Suite delta gate:** `python -m pytest tests/identity/ tests/test_get_parameters_dict_characterization.py -v` green (delta vs the ~24 pre-existing base failures).
- [ ] **Real train→calibrate→load loop on hydra-mps:** train a tiny 2-class model, confirm the artifact carries a temperature and ECE dropped, load it in TrackerKit import dialog, see "calibrated (T=…)".

---

## Self-Review (against the spec)

**Spec coverage (Calibration Lifecycle + Phase 2 line):**
- "Fit at the tail end of CNN training … per factor" → Tasks 2, 4. ✅
- "Store T + weight-hash signature in the artifact metadata (checkpoint / .v2meta.json / .multihead.json) and surface via ClassifierMetadata" → Tasks 3, 5. ✅
- "make CNNConfig.calibration_temperature fall back to artifact metadata" → Task 6. ✅
- "Mandatory-calibration gate" → Task 7. ✅
- "Recalibrate action" + "calibration status in the CNN import dialog" + "Training report surfaces ECE before/after" → Tasks 8, 9. ✅
- "Report ECE before/after" → Tasks 2 (compute), 9 (surface). ✅ (ClassKit's `TemperatureScaling` computes no ECE — Task 1 adds it in Core.)
- **Deviation from spec (documented):** the spec said "no new core calibration module is required — reuse `classkit/core/train/calibrate.py`." That fitter is in an app layer and the CNN training path is in the Training layer, so reusing it would violate the dependency direction. The plan instead adds the math to **Core** (`core/individual/identity/calibration.py`). Functionally identical fit; correct layering. ClassKit's copy is untouched.
- **Deferred (documented):** runtime robustness knobs (applied by Phase 3's evidence stage); per-factor calibration *application* (Phase 3); ClassKit fitter de-dup.

**Placeholder scan:** the novel logic (calibration math, fit-from-val) carries full code; wiring/UI tasks carry exact anchors + the specific edits, with a `READ <file>:<line> first` note where a real signature must be matched rather than guessed. No TBD/"handle errors".

**Type consistency:** `calibration_temperature` is a per-factor `list[float]` at write time and a `tuple[float, ...] | None` on `ClassifierMetadata`; the flat consume path (Task 6) takes `temps[0]`. `CalibrationResult` fields (`temperatures`/`signature`/`ece_before`/`ece_after`) are used identically across Tasks 2, 4, 9. `calibration_status(meta, current_signature)` signature identical in Tasks 5, 7, 8.

**Open risk to flag at execution:** computing `current_signature` for the gate/import-status means hashing the model weights, which requires loading the state_dict (cost). If that proves too heavy for the import dialog, store a self-signature *inside* the artifact at training time (hash computed once) and compare the stored self-signature to `calibration_signature` — same staleness semantics without re-hashing at load. Prefer the stored self-signature approach if load cost matters; decide during Task 5.
