# DetectKit SAM3 LoRA Finetuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `SEMANTIC_SAM3` training role that finetunes SAM3 on a DetectKit source's polygon labels and publishes a merged checkpoint the existing ultralytics escalation path loads unchanged.

**Architecture:** LoRA adapters are trained against Meta's `sam3` package (using Meta's own loss and Hungarian matcher), then merged into the base weights and written in Meta's `detector.`-prefixed key layout. Inference stays entirely on ultralytics — no second backend, no training dependencies in the runtime install.

**Tech Stack:** PyTorch, Meta `sam3` (training only), ultralytics `SAM3SemanticPredictor` (inference), COCO polygon datasets, PyQt6 (DetectKit GUI).

**Spec:** `docs/superpowers/specs/2026-08-31-detectkit-sam3-finetune-design.md`

## Global Constraints

- **Dependency direction:** `training/` and `core/inference/semantic/` must never import from an app layer (`detectkit/`, `trackerkit/`, ...). DetectKit imports them.
- **No god objects:** a new class or file over ~500 lines is doing too much. `detectkit/gui/dialogs/training_dialog.py` is already 2676 lines — add nothing to it beyond delegation.
- **Qt-free trainer:** everything under `training/sam3_lora/` must import no Qt.
- **Meta `sam3` is training-only** and lazily imported. A missing training dependency disables the action with a reason; it never raises at click time and never triggers an ultralytics AutoUpdate pip install.
- **SAM3 architecture input size is 1008** (`ultralytics/models/sam/build_sam3.py:38,308`). Training and inference must both use it.
- **Checkpoint selection is `last`, never `best`** — `best` is selected on validation loss, and validation loss was empirically anti-correlated with held-out AP in the spike.
- **Published artifacts go to `get_models_dir()/"sam3_finetuned"/`**, never `get_models_dir()/"sam3"/`, which is the stock download cache (`core/inference/semantic/checkpoints.py:91`).
- **Every task ends green:** run the named tests, then commit.

---

## Task 1: Pin `imgsz` in the SAM3 predictor overrides

The escalation path has always run SAM3 at ultralytics' default 640 (rounded to
644 for the stride-14 backbone) while the architecture is built at 1008. This
task changes stock escalation behaviour, so it ships alone, with its own
measurement, or a later quality change gets misattributed to the adapters.

**Files:**
- Modify: `src/hydra_suite/core/inference/semantic/sam3.py:31-54`
- Test: `tests/test_semantic_sam3_overrides.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `PREDICTOR_IMGSZ: int = 1008`, and `predictor_overrides(...)` now returns a dict containing `"imgsz": 1008`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_semantic_sam3_overrides.py
"""imgsz must be PINNED, not inherited from ultralytics' default cfg."""

from hydra_suite.core.inference.semantic.sam3 import (
    PREDICTOR_IMGSZ,
    predictor_overrides,
)


def test_overrides_pin_imgsz_to_the_architecture_size():
    # build_sam3.py builds SAM3 at img_size=1008; ultralytics' default cfg is
    # 640 -> 644. Inheriting it silently runs the model off-size.
    assert PREDICTOR_IMGSZ == 1008
    ov = predictor_overrides("/tmp/fake.pt", "cpu")
    assert ov["imgsz"] == 1008


def test_overrides_still_pin_conf_and_iou():
    ov = predictor_overrides("/tmp/fake.pt", "cpu", confidence_floor=0.05)
    assert ov["conf"] == 0.05
    assert ov["iou"] == 0.7
    assert ov["model"] == "/tmp/fake.pt"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_semantic_sam3_overrides.py -v`
Expected: FAIL with `ImportError: cannot import name 'PREDICTOR_IMGSZ'`

- [ ] **Step 3: Write minimal implementation**

In `src/hydra_suite/core/inference/semantic/sam3.py`, beside the existing
`PREDICTOR_NMS_IOU` constant, add:

```python
# Pinned for the same reason as PREDICTOR_NMS_IOU. ultralytics' default cfg
# imgsz is 640 -- rounded up to 644 for the stride-14 backbone -- but
# build_sam3.py builds the SAM3 architecture at img_size=1008 and
# BasePredictor calls model.set_imgsz(self.imgsz). Inheriting the default
# therefore runs a 1008-native model at 644 with no warning. It also makes
# train/serve scale disagree for any finetuned checkpoint.
PREDICTOR_IMGSZ = 1008
```

and add `"imgsz": PREDICTOR_IMGSZ,` to the dict returned by
`predictor_overrides`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_semantic_sam3_overrides.py -v`
Expected: 2 passed

- [ ] **Step 5: Measure the behaviour change on stock SAM3**

This is a real behaviour change and must be quantified, not assumed. On a
machine with the SAM3 weights:

```bash
cd scratch/sam3_lora_spike   # from branch spike/sam3-lora
PYTHONPATH=$PWD/../../src python show_on_frames.py --n 3 --conf 0.4 --imgsz 644
PYTHONPATH=$PWD/../../src python show_on_frames.py --n 3 --conf 0.4 --imgsz 1008
```

Record both instance counts in the commit message. The spike measured
23/31/31 at 644 versus 23/33/31 at 1008, with confidences rising from
~0.88-0.90 to ~0.91-0.93.

- [ ] **Step 6: Commit**

```bash
git add tests/test_semantic_sam3_overrides.py src/hydra_suite/core/inference/semantic/sam3.py
git commit -m "fix(semantic): pin SAM3 predictor imgsz to the architecture's 1008

ultralytics' default cfg imgsz (640 -> 644) was being inherited while
build_sam3 builds SAM3 at img_size=1008, so every semantic escalation ran the
model off-size. Measured on three unlabelled frames at conf>=0.4: 23/31/31
instances at 644 vs 23/33/31 at 1008, confidence ~0.89 -> ~0.92."
```

---

## Task 2: Add predictor params to the staging-directory hash

Task 1 changes results while every cached staging directory still looks valid.
`staged_dirname_for` hashes only `(source path, variant, prompt)`, so a run at a
different `imgsz` or confidence floor silently reuses stale candidates.

**Files:**
- Modify: `src/hydra_suite/detectkit/jobs/semantic_escalation.py:164-176`
- Test: `tests/test_semantic_staging_hash.py`

**Interfaces:**
- Consumes: `PREDICTOR_IMGSZ` from Task 1.
- Produces: `staged_dirname_for(src, variant, prompt, *, imgsz=PREDICTOR_IMGSZ)` — the new keyword is optional so existing call sites keep working.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_semantic_staging_hash.py
"""Predictor geometry must enter the staging hash, or stale candidate caches
are silently reused after an imgsz change."""

from types import SimpleNamespace

from hydra_suite.detectkit.jobs.semantic_escalation import staged_dirname_for


def _src(tmp_path):
    return SimpleNamespace(path=str(tmp_path), name="src")


def test_imgsz_changes_the_staging_dirname(tmp_path):
    a = staged_dirname_for(_src(tmp_path), "sam3", "ant", imgsz=644)
    b = staged_dirname_for(_src(tmp_path), "sam3", "ant", imgsz=1008)
    assert a != b


def test_prompt_and_variant_still_change_it(tmp_path):
    base = staged_dirname_for(_src(tmp_path), "sam3", "ant")
    assert staged_dirname_for(_src(tmp_path), "sam3", "beetle") != base
    assert staged_dirname_for(_src(tmp_path), "sam3-x", "ant") != base


def test_same_inputs_are_stable(tmp_path):
    a = staged_dirname_for(_src(tmp_path), "sam3", "ant", imgsz=1008)
    b = staged_dirname_for(_src(tmp_path), "sam3", "ant", imgsz=1008)
    assert a == b
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_semantic_staging_hash.py -v`
Expected: FAIL with `TypeError: staged_dirname_for() got an unexpected keyword argument 'imgsz'`

- [ ] **Step 3: Write minimal implementation**

```python
def staged_dirname_for(
    src: OBBSource,
    variant: str,
    prompt: str,
    *,
    imgsz: int = PREDICTOR_IMGSZ,
) -> str:
    """The staging directory NAME a (source, variant, prompt, imgsz) run targets.

    Shared with the GUI so it can tell a RESUME of the same run (same
    directory) from a REPLACE of a different one (different directory)
    without duplicating the hashing rule.
    """
    # DEPARTURE 2: the PROMPT enters the hash. Without it two prompts on
    # one source collide and the replaced-pending cleanup no-ops.
    # DEPARTURE 3: IMGSZ enters the hash. Candidates collected at one input
    # size are not interchangeable with another's, and nothing else would
    # invalidate them -- a silently stale cache reads as a successful reuse.
    content_hash = sha1(
        (str(Path(src.path).resolve()) + variant + prompt + f"|imgsz={int(imgsz)}").encode("utf-8")
    ).hexdigest()[:10]
    return f"{src.name}-sam3-{prompt_slug(prompt)}-{content_hash}"
```

Add the import: `from hydra_suite.core.inference.semantic.sam3 import PREDICTOR_IMGSZ`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_semantic_staging_hash.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add tests/test_semantic_staging_hash.py src/hydra_suite/detectkit/jobs/semantic_escalation.py
git commit -m "fix(detectkit): put predictor imgsz in the SAM3 staging hash

Candidates collected at one input size are not interchangeable with another's.
Without imgsz in the key, the imgsz fix reuses stale staging dirs and the
reuse looks successful."
```

---

## Task 3: LoRA seam — inject, extract, merge

Pure tensor code with no SAM3 import, so it is unit-testable on a toy module
and needs no GPU or licence-gated weights.

**Files:**
- Create: `src/hydra_suite/training/sam3_lora/__init__.py`
- Create: `src/hydra_suite/training/sam3_lora/lora.py`
- Test: `tests/test_sam3_lora_seam.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `LoraConfig(rank: int, alpha: int, dropout: float, target_suffixes: tuple[str, ...])`
  - `inject_adapters(model: nn.Module, cfg: LoraConfig) -> int` — wraps matching `nn.Linear` modules, returns how many were wrapped.
  - `adapter_state_dict(model: nn.Module) -> dict[str, torch.Tensor]` — only `lora_A`/`lora_B` tensors, keyed by the wrapped module's dotted path.
  - `merge_adapters(base: dict[str, torch.Tensor], adapters: dict[str, torch.Tensor], cfg: LoraConfig, *, prefix: str = "detector.") -> dict[str, torch.Tensor]` — folds `W + (B @ A) * alpha / rank` into `f"{prefix}{path}.weight"`. Raises `KeyError` if an adapter resolves to no base key.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sam3_lora_seam.py
"""LoRA inject/merge round-trip, provable without SAM3 or a GPU."""

import pytest
import torch
from torch import nn

from hydra_suite.training.sam3_lora.lora import (
    LoraConfig,
    adapter_state_dict,
    inject_adapters,
    merge_adapters,
)


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(8, 8, bias=False)
        self.other = nn.Linear(8, 8, bias=False)


def _cfg(rank=2, alpha=4):
    return LoraConfig(rank=rank, alpha=alpha, dropout=0.0,
                      target_suffixes=("qkv",))


def test_inject_only_wraps_targeted_suffixes():
    m = Toy()
    assert inject_adapters(m, _cfg()) == 1
    assert "qkv" in " ".join(adapter_state_dict(m).keys())
    assert "other" not in " ".join(adapter_state_dict(m).keys())


def test_zero_initialised_adapter_merges_as_a_no_op():
    m = Toy()
    inject_adapters(m, _cfg())
    base = {"detector.qkv.weight": torch.randn(8, 8)}
    merged = merge_adapters(base, adapter_state_dict(m), _cfg())
    # lora_B is zero-initialised, so an untrained adapter must change nothing.
    assert torch.equal(merged["detector.qkv.weight"], base["detector.qkv.weight"])


def test_merge_applies_the_scaled_low_rank_delta():
    m = Toy()
    cfg = _cfg(rank=2, alpha=4)
    inject_adapters(m, cfg)
    sd = adapter_state_dict(m)
    a_key = next(k for k in sd if k.endswith("lora_A"))
    b_key = next(k for k in sd if k.endswith("lora_B"))
    sd[b_key] = torch.randn_like(sd[b_key])
    w = torch.randn(8, 8)
    merged = merge_adapters({"detector.qkv.weight": w}, sd, cfg)
    expected = w + (sd[b_key] @ sd[a_key]) * (cfg.alpha / cfg.rank)
    assert torch.allclose(merged["detector.qkv.weight"], expected, atol=1e-6)


def test_unresolved_adapter_is_a_hard_error():
    m = Toy()
    inject_adapters(m, _cfg())
    # A silent skip is indistinguishable from a successful merge and yields a
    # checkpoint that differs from base in bytes but not in behaviour.
    with pytest.raises(KeyError):
        merge_adapters({"detector.somethingelse.weight": torch.randn(8, 8)},
                       adapter_state_dict(m), _cfg())


def test_non_targeted_base_keys_pass_through_untouched():
    m = Toy()
    inject_adapters(m, _cfg())
    base = {"detector.qkv.weight": torch.randn(8, 8),
            "detector.buffer": torch.randn(3)}
    merged = merge_adapters(base, adapter_state_dict(m), _cfg())
    assert torch.equal(merged["detector.buffer"], base["detector.buffer"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_sam3_lora_seam.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'hydra_suite.training.sam3_lora'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/hydra_suite/training/sam3_lora/__init__.py
"""SAM3 LoRA finetuning (Qt-free). Heavy imports are lazy; see availability.py."""
```

```python
# src/hydra_suite/training/sam3_lora/lora.py
"""Low-rank adapters: inject, extract, merge.

Deliberately free of any SAM3 import so the whole seam is testable on a toy
nn.Module without a GPU or the licence-gated checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class LoraConfig:
    rank: int
    alpha: int
    dropout: float
    target_suffixes: tuple[str, ...]

    @property
    def scaling(self) -> float:
        return float(self.alpha) / float(self.rank)


class LoraLinear(nn.Module):
    """Frozen base Linear plus a trainable rank-r branch."""

    def __init__(self, base: nn.Linear, cfg: LoraConfig) -> None:
        super().__init__()
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        self.lora_A = nn.Parameter(torch.zeros(cfg.rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, cfg.rank))
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        # lora_B stays zero: an untrained adapter must be an exact no-op.
        self.dropout = nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity()
        self.scaling = cfg.scaling

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
        return self.base(x) + delta * self.scaling


def inject_adapters(model: nn.Module, cfg: LoraConfig) -> int:
    """Wrap every nn.Linear whose dotted path ends in a target suffix."""
    targets = [
        (name, mod)
        for name, mod in model.named_modules()
        if isinstance(mod, nn.Linear)
        and name.split(".")[-1] in cfg.target_suffixes
    ]
    for name, mod in targets:
        *parent_path, attr = name.split(".")
        parent = model
        for part in parent_path:
            parent = getattr(parent, part)
        setattr(parent, attr, LoraLinear(mod, cfg))
    return len(targets)


def adapter_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for name, mod in model.named_modules():
        if isinstance(mod, LoraLinear):
            out[f"{name}.lora_A"] = mod.lora_A.detach().cpu()
            out[f"{name}.lora_B"] = mod.lora_B.detach().cpu()
    return out


def merge_adapters(
    base: dict[str, torch.Tensor],
    adapters: dict[str, torch.Tensor],
    cfg: LoraConfig,
    *,
    prefix: str = "detector.",
) -> dict[str, torch.Tensor]:
    """Fold every adapter into the base state dict, in the base's key layout.

    Adapters are trained against Meta's un-prefixed model; the published
    checkpoint is `detector.`-prefixed. An adapter that resolves to no base key
    is a HARD ERROR: skipping it silently produces a checkpoint that differs
    from base in bytes but not in behaviour, which is indistinguishable from a
    successful merge.
    """
    merged = {k: v.clone() for k, v in base.items()}
    paths = sorted({k.rsplit(".", 1)[0] for k in adapters})
    for path in paths:
        key = f"{prefix}{path}.weight"
        if key not in merged:
            raise KeyError(
                f"adapter {path!r} resolves to {key!r}, which is not in the "
                f"base checkpoint; refusing a partial merge"
            )
        a = adapters[f"{path}.lora_A"]
        b = adapters[f"{path}.lora_B"]
        delta = (b @ a) * cfg.scaling
        merged[key] = merged[key] + delta.to(merged[key].dtype)
    return merged
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_sam3_lora_seam.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/training/sam3_lora/ tests/test_sam3_lora_seam.py
git commit -m "feat(training): LoRA inject/extract/merge seam for SAM3

No SAM3 import, so the whole seam is testable on a toy module. Unresolved
adapters are a hard error -- a silent skip yields a checkpoint that differs in
bytes but not behaviour."
```

---

## Task 4: Availability probe

Mirrors `core/inference/semantic/checkpoints.py:probe_availability`: checks
the training dependencies and the base checkpoint without importing
ultralytics, importing `sam3`, or touching the network.

**Files:**
- Create: `src/hydra_suite/training/sam3_lora/availability.py`
- Test: `tests/test_sam3_lora_availability.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `Sam3TrainingAvailability(usable: bool, reason: str)` and `probe_sam3_training_availability(cache_dir: Path | None = None) -> Sam3TrainingAvailability`. Module constant `TRAINING_PACKAGES = ("sam3", "torch", "torchmetrics", "scipy", "einops")`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sam3_lora_availability.py
"""The probe must explain WHY it is unusable, and never import or download."""

import sys

from hydra_suite.training.sam3_lora import availability as av


def test_missing_package_names_itself_and_the_install(monkeypatch):
    monkeypatch.setattr(av, "_find_spec", lambda n: None if n == "torchmetrics" else object())
    monkeypatch.setattr(av, "_checkpoint_present", lambda cache_dir=None: True)
    got = av.probe_sam3_training_availability()
    assert not got.usable
    assert "torchmetrics" in got.reason


def test_missing_checkpoint_is_reported_not_downloaded(monkeypatch):
    monkeypatch.setattr(av, "_find_spec", lambda n: object())
    monkeypatch.setattr(av, "_checkpoint_present", lambda cache_dir=None: False)
    got = av.probe_sam3_training_availability()
    assert not got.usable
    assert "checkpoint" in got.reason.lower()


def test_all_present_is_usable(monkeypatch):
    monkeypatch.setattr(av, "_find_spec", lambda n: object())
    monkeypatch.setattr(av, "_checkpoint_present", lambda cache_dir=None: True)
    assert av.probe_sam3_training_availability().usable


def test_probe_does_not_import_sam3(monkeypatch):
    monkeypatch.setattr(av, "_find_spec", lambda n: object())
    monkeypatch.setattr(av, "_checkpoint_present", lambda cache_dir=None: True)
    sys.modules.pop("sam3", None)
    av.probe_sam3_training_availability()
    assert "sam3" not in sys.modules
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_sam3_lora_availability.py -v`
Expected: FAIL with `ModuleNotFoundError: ...sam3_lora.availability`

- [ ] **Step 3: Write minimal implementation**

```python
# src/hydra_suite/training/sam3_lora/availability.py
"""Structured availability for SAM3 LoRA training.

Same discipline as core/inference/semantic/checkpoints.py: never import the
heavy packages, never download. The GUI disables the action with `reason`
rather than failing at click time.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TRAINING_PACKAGES = ("sam3", "torch", "torchmetrics", "scipy", "einops")

INSTALL_HINTS = {
    "sam3": "pip install git+https://github.com/facebookresearch/sam3.git",
}
DEFAULT_INSTALL_HINT = "pip install 'hydra-suite[sam3-train]'"


@dataclass(frozen=True)
class Sam3TrainingAvailability:
    usable: bool
    reason: str = ""


def _find_spec(name: str) -> Any:  # seam for tests
    return importlib.util.find_spec(name)


def _checkpoint_present(cache_dir: Path | None = None) -> bool:  # seam for tests
    from hydra_suite.core.inference.semantic.checkpoints import checkpoint_path

    return checkpoint_path("sam3", cache_dir).exists()


def probe_sam3_training_availability(
    cache_dir: Path | None = None,
) -> Sam3TrainingAvailability:
    for pkg in TRAINING_PACKAGES:
        if _find_spec(pkg) is None:
            hint = INSTALL_HINTS.get(pkg, DEFAULT_INSTALL_HINT)
            return Sam3TrainingAvailability(
                False, f"Python package {pkg!r} is missing. Install it with: {hint}"
            )
    if not _checkpoint_present(cache_dir):
        return Sam3TrainingAvailability(
            False,
            "The SAM3 base checkpoint has not been downloaded yet. Run a "
            "semantic escalation once to fetch it, or accept the licence at "
            "https://huggingface.co/facebook/sam3 and run `hf auth login`.",
        )
    return Sam3TrainingAvailability(True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_sam3_lora_availability.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/training/sam3_lora/availability.py tests/test_sam3_lora_availability.py
git commit -m "feat(training): SAM3 training availability probe

Never imports sam3 or ultralytics and never downloads, so the GUI can disable
the action with a reason instead of failing at click time."
```

---

## Task 5: Contracts — the role and its params

**Files:**
- Modify: `src/hydra_suite/training/contracts.py` (add to `TrainingRole`; add `Sam3LoraParams`; add the `sam3_params` field to `TrainingRunSpec`)
- Test: `tests/test_sam3_contracts.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `TrainingRole.SEMANTIC_SAM3` (value `"semantic_sam3"`), `Sam3LoraParams`, and `TrainingRunSpec.sam3_params: Sam3LoraParams | None = None`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sam3_contracts.py
from hydra_suite.training.contracts import (
    Sam3LoraParams,
    TrainingRole,
    TrainingRunSpec,
    TrainingHyperParams,
    SourceDataset,
)


def test_role_exists_with_the_expected_value():
    assert TrainingRole.SEMANTIC_SAM3.value == "semantic_sam3"


def test_defaults_match_the_measured_spike():
    p = Sam3LoraParams(prompt="ant")
    assert p.rank == 16 and p.alpha == 32
    # 10, not 40: AP plateaus by epoch ~9 (sd 0.040 thereafter).
    assert p.epochs == 10
    # batch 1: batch 2 OOMed at 1008px on a 47 GB card.
    assert p.batch == 1 and p.grad_accum == 8
    # Adapting the text encoder risks eroding prompt discrimination.
    assert p.adapt_text_encoder is False
    assert p.negative_prompts == []


def test_spec_round_trips_sam3_params():
    spec = TrainingRunSpec(
        role=TrainingRole.SEMANTIC_SAM3,
        source_datasets=[SourceDataset(path="/tmp/x", level="polygon")],
        derived_dataset_dir="/tmp/d",
        base_model="sam3",
        hyperparams=TrainingHyperParams(),
        sam3_params=Sam3LoraParams(prompt="ant with color patch"),
    )
    d = spec.to_dict()
    assert d["role"] == "semantic_sam3"
    assert d["sam3_params"]["prompt"] == "ant with color patch"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_sam3_contracts.py -v`
Expected: FAIL with `ImportError: cannot import name 'Sam3LoraParams'`

- [ ] **Step 3: Write minimal implementation**

Add to the `TrainingRole` enum, after the ClassKit roles:

```python
    # DetectKit promptable-concept segmentation
    SEMANTIC_SAM3 = "semantic_sam3"
```

Add the dataclass beside `CustomCNNParams`:

```python
@dataclass(slots=True)
class Sam3LoraParams:
    """SAM3 LoRA finetuning hyperparameters.

    Defaults are measured on the spike (see the design doc's Why section), not
    chosen by taste; where a value is a judgement call the comment says so.
    """

    prompt: str = ""  # required; there is no defensible default concept
    negative_prompts: list[str] = field(default_factory=list)
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.1
    lr: float = 5e-5
    # AP75 plateaus by epoch ~9 (mean .642, sd .040); 40 buys nothing.
    epochs: int = 10
    # batch 2 OOMs at 1008 px on a 47 GB card; effective batch is batch*accum.
    batch: int = 1
    grad_accum: int = 8
    mixed_precision: str = "bf16"
    num_negatives: int = 3
    # Which submodules receive adapters. The text encoder is False as a
    # precaution against eroding prompt discrimination -- untested; the spike
    # froze it in every configuration.
    adapt_vision_encoder: bool = True
    adapt_text_encoder: bool = False
    adapt_geometry_encoder: bool = True
    adapt_detr_encoder: bool = True
    adapt_detr_decoder: bool = True
    adapt_mask_decoder: bool = True
    # Tiling, mirroring the SAHI sliced-training knobs.
    geometry_mode: str = "auto_object"  # auto_object | auto_model | custom
    object_tile_fraction: float = 0.055
    slice_width: int = 0  # custom mode only; 0 => fall back to imgsz
    slice_height: int = 0  # custom mode only
    tile_overlap: float = 0.25
    keep_empty_tiles: bool = True
```

Add to `TrainingRunSpec`, beside `custom_params`:

```python
    sam3_params: Sam3LoraParams | None = None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_sam3_contracts.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/training/contracts.py tests/test_sam3_contracts.py
git commit -m "feat(training): SEMANTIC_SAM3 role and Sam3LoraParams"
```

---

## Task 6: Unblock the role in the existing pipeline

Six sites raise or misroute for an unknown role. This task makes the role
*reachable*; the builder and trainer arrive next. Verified breakages, from the
design's table.

**Files:**
- Modify: `src/hydra_suite/training/dataset_builders.py:1001-1016` (`_ROLE_MIN_LEVEL`)
- Modify: `src/hydra_suite/training/validation.py:430-463` (`validate_role_dataset`)
- Modify: `src/hydra_suite/training/runner.py:2261-2303` (`run_training` dispatch)
- Test: `tests/test_sam3_role_plumbing.py`

**Interfaces:**
- Consumes: `TrainingRole.SEMANTIC_SAM3` from Task 5.
- Produces: `validate_coco_dataset(dataset_dir) -> ValidationReport`, exported from `training/validation.py`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sam3_role_plumbing.py
"""The role must be reachable: no raise, no ultralytics fall-through."""

import json

import pytest

from hydra_suite.training.contracts import TrainingRole
from hydra_suite.training.dataset_builders import role_min_level
from hydra_suite.training.geometry_levels import GeometryLevel
from hydra_suite.training.validation import validate_role_dataset


def _coco(tmp_path, n_images=2, n_anns=3):
    for split in ("train", "valid"):
        d = tmp_path / split
        d.mkdir(parents=True)
        (d / "_annotations.coco.json").write_text(json.dumps({
            "images": [{"id": i, "file_name": f"{i}.jpg",
                        "width": 8, "height": 8} for i in range(n_images)],
            "annotations": [{"id": j, "image_id": 0, "category_id": 1,
                             "segmentation": [[0, 0, 1, 0, 1, 1]],
                             "area": 1.0, "bbox": [0, 0, 1, 1],
                             "iscrowd": 0} for j in range(n_anns)],
            "categories": [{"id": 1, "name": "ant"}],
        }))
    return tmp_path


def test_role_has_a_geometry_level():
    assert role_min_level(TrainingRole.SEMANTIC_SAM3) is GeometryLevel.POLYGON


def test_validate_accepts_a_coco_layout(tmp_path):
    # validate_role_dataset used to call inspect_obb_or_detect_dataset
    # unconditionally, which RAISES on a COCO layout.
    report = validate_role_dataset(_coco(tmp_path), TrainingRole.SEMANTIC_SAM3)
    assert report.valid


def test_validate_rejects_an_empty_coco_dataset(tmp_path):
    # The old fall-through returned valid=True for unhandled roles, so a
    # forgotten validator would silently pass. It must actually inspect.
    bad = _coco(tmp_path, n_images=0, n_anns=0)
    report = validate_role_dataset(bad, TrainingRole.SEMANTIC_SAM3)
    assert not report.valid


def test_run_training_does_not_reach_the_ultralytics_builder(monkeypatch):
    from hydra_suite.training import runner

    called = {}
    monkeypatch.setattr(runner, "build_ultralytics_command",
                        lambda *a, **k: called.setdefault("yolo", True))
    monkeypatch.setattr(runner, "_train_sam3_lora",
                        lambda *a, **k: {"success": True})
    from hydra_suite.training.contracts import (
        Sam3LoraParams, SourceDataset, TrainingHyperParams, TrainingRunSpec)
    spec = TrainingRunSpec(
        role=TrainingRole.SEMANTIC_SAM3,
        source_datasets=[SourceDataset(path="/tmp/x", level="polygon")],
        derived_dataset_dir="/tmp/d", base_model="sam3",
        hyperparams=TrainingHyperParams(),
        sam3_params=Sam3LoraParams(prompt="ant"))
    out = runner.run_training(spec, "/tmp/run")
    assert out["success"]
    assert "yolo" not in called
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_sam3_role_plumbing.py -v`
Expected: FAIL — `role_min_level` raises `RuntimeError("Role has no geometry-level requirement")`

- [ ] **Step 3: Write minimal implementation**

In `dataset_builders.py`, add to `_ROLE_MIN_LEVEL`:

```python
    TrainingRole.SEMANTIC_SAM3: GeometryLevel.POLYGON,
```

In `validation.py`, add a COCO validator and branch to it **before** the
unconditional inspector:

```python
def validate_coco_dataset(
    dataset_dir: str | Path, *, min_train: int = 1, min_val: int = 0
) -> ValidationReport:
    """Validate a COCO instance-segmentation layout (train/valid/_annotations.coco.json)."""
    root = Path(dataset_dir)
    issues: list[ValidationIssue] = []
    stats: dict[str, Any] = {"root_dir": str(root)}
    for split, floor in (("train", min_train), ("valid", min_val)):
        ann = root / split / "_annotations.coco.json"
        if not ann.exists():
            if floor > 0:
                issues.append(ValidationIssue(
                    "error", "coco_missing_split",
                    f"missing {ann}", str(ann)))
            continue
        data = json.loads(ann.read_text())
        n_img = len(data.get("images", []))
        n_ann = len(data.get("annotations", []))
        stats[f"{split}_images"] = n_img
        stats[f"{split}_annotations"] = n_ann
        if n_img < floor or (floor > 0 and n_ann == 0):
            issues.append(ValidationIssue(
                "error", "coco_empty_split",
                f"{split}: {n_img} images, {n_ann} annotations", str(ann)))
    return ValidationReport(
        valid=not any(i.severity == "error" for i in issues),
        issues=issues, stats=stats)
```

and at the top of `validate_role_dataset`, before the inspector call:

```python
    # MUST precede inspect_obb_or_detect_dataset: that inspector RAISES on a
    # COCO layout, so branching after it is a crash, not a fall-through.
    if role is TrainingRole.SEMANTIC_SAM3:
        return validate_coco_dataset(
            dataset_dir, min_train=min_train, min_val=0)
```

In `runner.py`, before the `build_ultralytics_command` fall-through:

```python
    if spec.role is TrainingRole.SEMANTIC_SAM3:
        return _train_sam3_lora(
            spec, run_dir, log_cb=log_cb, progress_cb=progress_cb,
            should_cancel=should_cancel,
        )
```

and a temporary stub so this task is independently green (Task 8 replaces it):

```python
def _train_sam3_lora(spec, run_dir, *, log_cb=None, progress_cb=None,
                     should_cancel=None) -> dict:
    from hydra_suite.training.sam3_lora.train import train_sam3_lora

    return train_sam3_lora(spec, run_dir, log_cb=log_cb,
                           progress_cb=progress_cb, should_cancel=should_cancel)
```

Add `import json` to `validation.py` if absent.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_sam3_role_plumbing.py -v`
Expected: 4 passed

- [ ] **Step 5: Run the neighbouring suites for regressions**

Run: `python -m pytest tests/test_training_validation.py tests/test_dataset_builders.py -v`
Expected: no new failures versus `git stash`-ed baseline

- [ ] **Step 6: Commit**

```bash
git add src/hydra_suite/training/dataset_builders.py src/hydra_suite/training/validation.py src/hydra_suite/training/runner.py tests/test_sam3_role_plumbing.py
git commit -m "feat(training): make SEMANTIC_SAM3 reachable through the pipeline

validate_role_dataset called inspect_obb_or_detect_dataset unconditionally,
which raises on a COCO layout; role_min_level raised for unknown roles; and
run_training fell through to build_ultralytics_command."
```

---

## Task 7: COCO tile dataset builder

**Files:**
- Create: `src/hydra_suite/training/sam3_lora/dataset_build.py`
- Modify: `src/hydra_suite/training/dataset_builders.py` (`prepare_role_dataset` branch)
- Test: `tests/test_sam3_dataset_build.py`

**Interfaces:**
- Consumes: `Sam3LoraParams` (Task 5).
- Produces: `build_sam3_coco_dataset(source_dir, out_dir, params, *, seed=42, split=SplitConfig()) -> dict` writing `train/` and `valid/` COCO datasets, returning stats with keys `train_images`, `train_annotations`, `crowd_annotations`, `tile_px`, `negative_prompts`, `validation`.
- Produces: `resolve_negative_prompts(params, source_class_names) -> list[str]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sam3_dataset_build.py
"""Tiling, iscrowd boundary, frame-level split, negative-prompt resolution."""

import json

import cv2
import numpy as np
import pytest

from hydra_suite.training.contracts import Sam3LoraParams
from hydra_suite.training.sam3_lora.dataset_build import (
    build_sam3_coco_dataset,
    resolve_negative_prompts,
)

CURATED = ("background", "shadow", "debris")


def _source(tmp_path, n_frames=3, size=2048):
    img_dir = tmp_path / "images"
    lbl_dir = tmp_path / "labels"
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for i in range(n_frames):
        img = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
        cv2.imwrite(str(img_dir / f"f{i}.jpg"), img)
        # one polygon near the frame centre, normalised
        poly = np.array([[0.50, 0.50], [0.52, 0.50], [0.52, 0.52], [0.50, 0.52]])
        line = "0 " + " ".join(f"{v:.6f}" for v in poly.reshape(-1))
        (lbl_dir / f"f{i}.txt").write_text(line + "\n")
    (tmp_path / "classes.txt").write_text("ant\n")
    return tmp_path


def _params(**kw):
    base = dict(prompt="ant with color patch", geometry_mode="custom",
                slice_width=512, slice_height=512, tile_overlap=0.25)
    base.update(kw)
    return Sam3LoraParams(**base)


def test_category_name_is_the_prompt(tmp_path):
    out = tmp_path / "out"
    build_sam3_coco_dataset(_source(tmp_path / "src"), out, _params())
    data = json.loads((out / "train" / "_annotations.coco.json").read_text())
    assert data["categories"][0]["name"] == "ant with color patch"


def test_split_is_by_frame_not_by_tile(tmp_path):
    out = tmp_path / "out"
    build_sam3_coco_dataset(_source(tmp_path / "src", n_frames=3), out, _params())
    tr = {n["file_name"].split("_")[0]
          for n in json.loads((out / "train" / "_annotations.coco.json").read_text())["images"]}
    va = {n["file_name"].split("_")[0]
          for n in json.loads((out / "valid" / "_annotations.coco.json").read_text())["images"]}
    # Overlapping tiles from one frame share pixels; a tile-level split leaks.
    assert tr.isdisjoint(va)


def test_single_frame_source_trains_without_validation(tmp_path):
    out = tmp_path / "out"
    stats = build_sam3_coco_dataset(_source(tmp_path / "src", n_frames=1), out, _params())
    # Refusing would block the exact bootstrap case this feature serves.
    assert stats["train_images"] > 0
    assert stats["validation"] == "none"


def test_empty_tiles_are_kept_when_requested(tmp_path):
    out = tmp_path / "out"
    stats = build_sam3_coco_dataset(_source(tmp_path / "src"), out,
                                    _params(keep_empty_tiles=True))
    imgs = json.loads((out / "train" / "_annotations.coco.json").read_text())["images"]
    anns = json.loads((out / "train" / "_annotations.coco.json").read_text())["annotations"]
    with_ann = {a["image_id"] for a in anns}
    assert len(imgs) > len(with_ann)  # some tiles are pure background


def test_negative_prompts_prefer_explicit_then_classes_then_curated():
    p = _params(negative_prompts=["mite"])
    assert resolve_negative_prompts(p, ["ant", "beetle"]) == ["mite"]

    p2 = _params()
    assert resolve_negative_prompts(p2, ["ant", "beetle"]) == ["beetle"]

    p3 = _params()
    got = resolve_negative_prompts(p3, ["ant"])
    assert got and set(got).issubset(set(CURATED))


def test_curated_negatives_drop_word_overlap_with_the_prompt():
    p = Sam3LoraParams(prompt="ant on a shadow")
    got = resolve_negative_prompts(p, ["ant"])
    assert "shadow" not in got
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_sam3_dataset_build.py -v`
Expected: FAIL — `ModuleNotFoundError: ...sam3_lora.dataset_build`

- [ ] **Step 3: Write minimal implementation**

Create `src/hydra_suite/training/sam3_lora/dataset_build.py`. Use
`hydra_suite.utils.slice_geometry` for all geometry — `tile_size_for_mode`,
`plan_tiles`, `clip_polygon_to_tile`, `polygon_area`. Key rules, each of which
a test above pins:

- `category.name = params.prompt`.
- Split by **frame**, shuffled under `seed`; 1 frame => train-only with
  `stats["validation"] = "none"`; 2 frames => 1/1 with a logged warning.
- An instance retaining `< MIN_RETAINED_AREA_FRAC = 0.5` of its area after
  clipping is written with `iscrowd = 1`, not dropped.
- Empty tiles are written when `params.keep_empty_tiles`.
- `resolve_negative_prompts` returns, in order: explicit `params.negative_prompts`;
  else the source's other class names; else `CURATED_NEGATIVES = ("background",
  "shadow", "debris")` minus any entry sharing a word with the prompt.
- Write `build_manifest.json` recording `tile_px`, `reference_body_px`,
  `object_tile_fraction`, the resolved negative prompts, and the frame split.

Then in `dataset_builders.prepare_role_dataset`, branch before the OBB path:

```python
    if role is TrainingRole.SEMANTIC_SAM3:
        # Concept training is PER SOURCE, not on the merged OBB dataset.
        from hydra_suite.training.sam3_lora.dataset_build import (
            build_sam3_coco_dataset,
        )

        stats = build_sam3_coco_dataset(
            source_dir=merged_obb_dataset_dir,
            out_dir=role_output_root,
            params=sam3_params,
        )
        return DatasetBuildResult(dataset_dir=str(role_output_root), stats=stats)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_sam3_dataset_build.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/training/sam3_lora/dataset_build.py src/hydra_suite/training/dataset_builders.py tests/test_sam3_dataset_build.py
git commit -m "feat(training): COCO tile dataset builder for SAM3 finetuning

Reuses slice_geometry for all tiling. Splits by frame (overlapping tiles leak
pixels), marks seam-clipped instances iscrowd, keeps empty tiles, and resolves
negative prompts explicitly -- a single-category dataset has no other category
to sample them from."
```

---

## Task 8: The training loop

**Files:**
- Create: `src/hydra_suite/training/sam3_lora/train.py`
- Create: `src/hydra_suite/training/sam3_lora/preflight.py`
- Test: `tests/test_sam3_preflight.py`

**Interfaces:**
- Consumes: Tasks 3, 4, 5, 7.
- Produces: `train_sam3_lora(spec, run_dir, *, log_cb=None, progress_cb=None, should_cancel=None) -> dict` with keys `success`, `artifact_path` (the `adapters.pt`), `metrics_path`, `canceled`.
- Produces: `preflight(spec) -> list[str]` returning refusal reasons (empty means OK).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sam3_preflight.py
"""Preflight refuses before any weights load, and never imports ultralytics."""

import sys

import pytest

from hydra_suite.training.contracts import (
    Sam3LoraParams, SourceDataset, TrainingHyperParams, TrainingRole,
    TrainingRunSpec)
from hydra_suite.training.sam3_lora import preflight as pf


def _spec(**kw):
    p = Sam3LoraParams(prompt=kw.pop("prompt", "ant"))
    base = dict(
        role=TrainingRole.SEMANTIC_SAM3,
        source_datasets=[SourceDataset(path="/tmp/x", level="polygon")],
        derived_dataset_dir="/tmp/d", base_model="sam3",
        hyperparams=TrainingHyperParams(), sam3_params=p)
    base.update(kw)
    return TrainingRunSpec(**base)


def test_empty_prompt_is_refused(monkeypatch):
    monkeypatch.setattr(pf, "_cuda_free_gb", lambda: 48.0)
    monkeypatch.setattr(pf, "_instance_count", lambda d: 100)
    assert any("prompt" in r.lower() for r in pf.preflight(_spec(prompt="")))


def test_non_cuda_is_refused(monkeypatch):
    monkeypatch.setattr(pf, "_cuda_free_gb", lambda: None)
    monkeypatch.setattr(pf, "_instance_count", lambda d: 100)
    assert any("cuda" in r.lower() for r in pf.preflight(_spec()))


def test_vram_band_between_refuse_and_warn_is_refused(monkeypatch):
    # 29 GB was the measured requirement; a 24-29 GB card must NOT pass.
    monkeypatch.setattr(pf, "_cuda_free_gb", lambda: 26.0)
    monkeypatch.setattr(pf, "_instance_count", lambda d: 100)
    assert any("gb" in r.lower() for r in pf.preflight(_spec()))


def test_too_few_instances_is_refused(monkeypatch):
    monkeypatch.setattr(pf, "_cuda_free_gb", lambda: 48.0)
    monkeypatch.setattr(pf, "_instance_count", lambda d: 5)
    assert any("instance" in r.lower() for r in pf.preflight(_spec()))


def test_resume_from_is_refused(monkeypatch):
    monkeypatch.setattr(pf, "_cuda_free_gb", lambda: 48.0)
    monkeypatch.setattr(pf, "_instance_count", lambda d: 100)
    assert any("resume" in r.lower()
               for r in pf.preflight(_spec(resume_from="/tmp/last.pt")))


def test_healthy_spec_passes_and_imports_nothing_heavy(monkeypatch):
    monkeypatch.setattr(pf, "_cuda_free_gb", lambda: 48.0)
    monkeypatch.setattr(pf, "_instance_count", lambda d: 100)
    sys.modules.pop("ultralytics", None)
    assert pf.preflight(_spec()) == []
    assert "ultralytics" not in sys.modules
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_sam3_preflight.py -v`
Expected: FAIL — `ModuleNotFoundError: ...sam3_lora.preflight`

- [ ] **Step 3: Write minimal implementation**

`preflight.py` constants and rules:

```python
# Measured: ~29 GB at batch 1 on the spike; batch 2 OOMed on a 47 GB card.
# REFUSE must sit ABOVE the measured requirement, or the 24-29 GB band passes
# preflight and then OOMs -- the exact failure preflight exists to prevent.
REFUSE_BELOW_GB = 32.0
WARN_BELOW_GB = 40.0
MIN_TRAIN_INSTANCES = 20  # matches calibration.py's existing floor
```

`preflight` returns a list of reasons for: non-CUDA device, free VRAM below
`REFUSE_BELOW_GB`, empty prompt, `_instance_count(...) < MIN_TRAIN_INSTANCES`,
and non-empty `spec.resume_from` (optimiser state is not checkpointed).
`_cuda_free_gb` and `_instance_count` are module-level seams so tests can
monkeypatch them without a GPU.

`train.py` implements the loop: lazily import `sam3`, build the model, inject
adapters via Task 3, build the datapoints from the Task 7 COCO dirs, and use
Meta's own objective:

```python
from sam3.train.loss.sam3_loss import Sam3LossWrapper
from sam3.train.matcher import BinaryHungarianMatcherV2
```

AdamW, cosine schedule with warmup, grad accumulation, bf16 with a compute-
capability check falling back to fp32 with a logged notice, gradient clipping
at 1.0. Seed torch/numpy/python from `spec.seed`. Call `should_cancel()`
between optimiser steps and return `{"canceled": True}` promptly. Write
`adapters.pt` (via `adapter_state_dict`) and `val_stats.json` into `run_dir`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_sam3_preflight.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/training/sam3_lora/train.py src/hydra_suite/training/sam3_lora/preflight.py tests/test_sam3_preflight.py
git commit -m "feat(training): SAM3 LoRA training loop and preflight

Uses Meta's own Sam3LossWrapper and BinaryHungarianMatcherV2. Preflight
refuses below 32 GB free rather than 24, so the measured 29 GB requirement
cannot slip through into an OOM."
```

---

## Task 9: Publish — merge, sidecar, registry

**Files:**
- Create: `src/hydra_suite/training/sam3_lora/publish.py`
- Test: `tests/test_sam3_publish.py`

**Interfaces:**
- Consumes: `merge_adapters` (Task 3), `adapters.pt` (Task 8).
- Produces: `publish_sam3_model(run_id, adapters_path, base_checkpoint, build_manifest, params, *, models_root=None) -> tuple[str, str]` returning `(artifact_path, sidecar_path)`.
- Produces: `stripped_keys(state_dict) -> list[str]` reproducing ultralytics' transform.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sam3_publish.py
"""The merged artifact must load through ultralytics' own key transform."""

import json

import torch

from hydra_suite.training.contracts import Sam3LoraParams
from hydra_suite.training.sam3_lora.publish import (
    publish_sam3_model,
    stripped_keys,
)


def test_stripped_keys_reproduce_ultralytics_transform():
    sd = {"detector.a.weight": torch.zeros(1),
          "other.b.weight": torch.zeros(1),
          "x.detector.c": torch.zeros(1)}
    # build_sam3.py:357 filters on the SUBSTRING "detector", not a prefix.
    got = set(stripped_keys(sd))
    assert "a.weight" in got
    assert "other.b.weight" not in got
    assert "x.c" in got


def test_artifact_is_not_written_into_the_stock_cache(tmp_path):
    base = {"detector.qkv.weight": torch.randn(4, 4)}
    torch.save(base, tmp_path / "base.pt")
    torch.save({"qkv.lora_A": torch.zeros(2, 4), "qkv.lora_B": torch.zeros(4, 2)},
               tmp_path / "adapters.pt")
    art, side = publish_sam3_model(
        run_id="r1", adapters_path=tmp_path / "adapters.pt",
        base_checkpoint=tmp_path / "base.pt",
        build_manifest={"tile_px": 1007, "reference_body_px": 55.4},
        params=Sam3LoraParams(prompt="ant"), models_root=tmp_path / "models")
    # get_models_dir()/"sam3" is checkpoints.py's DOWNLOAD CACHE.
    assert "sam3_finetuned" in str(art)
    assert "/sam3/" not in str(art)


def test_sidecar_records_the_guard_fields(tmp_path):
    base = {"detector.qkv.weight": torch.randn(4, 4)}
    torch.save(base, tmp_path / "base.pt")
    torch.save({"qkv.lora_A": torch.randn(2, 4), "qkv.lora_B": torch.randn(4, 2)},
               tmp_path / "adapters.pt")
    _, side = publish_sam3_model(
        run_id="r1", adapters_path=tmp_path / "adapters.pt",
        base_checkpoint=tmp_path / "base.pt",
        build_manifest={"tile_px": 1007, "reference_body_px": 55.4},
        params=Sam3LoraParams(prompt="ant"), models_root=tmp_path / "models")
    meta = json.loads(open(side).read())
    assert meta["prompt"] == "ant"
    assert meta["train_tile_px"] == 1007
    assert meta["imgsz"] == 1008
    assert meta["stripped_keys"]
    assert meta["tuned_fingerprints"]  # sha256 of tensors the merge changed
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_sam3_publish.py -v`
Expected: FAIL — `ModuleNotFoundError: ...sam3_lora.publish`

- [ ] **Step 3: Write minimal implementation**

`publish_sam3_model` must:

1. Load the base state dict and the adapters; call `merge_adapters`.
2. **Carry across stock-only keys.** Meta's `build_sam3_image_model` does not
   instantiate every submodule the published checkpoint carries (observed:
   22 `vision_backbone.sam2_convs.*` tensors). Under `strict=False` those
   would stay at random init with no error. Copy them from base untouched.
3. Write to `models_root/"sam3_finetuned"/f"{run_id}.pt"`.
4. Write `<artifact>.sam3_meta.json` with `base_variant`, `prompt`,
   `train_tile_px`, `reference_body_px`, `object_tile_fraction`, `imgsz`
   (`PREDICTOR_IMGSZ`), `stripped_keys`, `tuned_fingerprints` (sha256 of the
   raw bytes of 2-3 tensors the merge changed), `source_fingerprint`,
   `label_quality_acknowledged`.
5. Register in `model_registry.json` with `task_family="semantic"` and
   `usage_role="semantic_sam3"` — a registry entry, never a directory scan.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_sam3_publish.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/training/sam3_lora/publish.py tests/test_sam3_publish.py
git commit -m "feat(training): publish merged SAM3 checkpoints

Writes to sam3_finetuned/, never the stock download cache. Carries across the
stock-only keys Meta's builder omits (observed: 22 sam2_convs tensors) which
would otherwise stay at random init under strict=False."
```

---

## Task 10: Escalation consumption and the silent-load guard

**Files:**
- Modify: `src/hydra_suite/core/inference/semantic/checkpoints.py`
- Modify: `src/hydra_suite/core/inference/semantic/sam3.py`
- Test: `tests/test_sam3_resolve_and_guard.py`

**Interfaces:**
- Consumes: the sidecar from Task 9.
- Produces: `resolve_checkpoint(key, cache_dir=None) -> Path`, `available_models() -> list[str]`, `probe_dependencies() -> Sam3Availability`, `probe_checkpoint(key, cache_dir=None) -> Sam3Availability`, and `Sam3SemanticLabeler.from_variant(..., checkpoint: Path | str | None = None)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sam3_resolve_and_guard.py
"""A published key must resolve, and a mis-loading checkpoint must raise."""

import json

import pytest
import torch

from hydra_suite.core.inference.semantic import checkpoints as ck
from hydra_suite.core.inference.semantic.sam3 import assert_checkpoint_loaded


def test_probe_dependencies_is_variant_independent(monkeypatch):
    # probe_availability rejected anything not in SAM3_VARIANTS, so every
    # published model read as "Unknown SAM3 variant" and stayed disabled.
    monkeypatch.setattr(ck, "_find_spec", lambda n: object())
    assert ck.probe_dependencies().usable


def test_available_models_includes_registry_entries(monkeypatch):
    monkeypatch.setattr(ck, "_registry_semantic_models",
                        lambda: ["run123"])
    got = ck.available_models()
    assert "sam3" in got and "run123" in got


def test_guard_raises_when_tuned_tensors_are_absent(tmp_path):
    # The failure this guard exists for: all keys present, but the model holds
    # BASE weights because ultralytics' load-time transform changed.
    meta = {"stripped_keys": ["a.weight"],
            "tuned_fingerprints": {"a.weight": "deadbeef"}}
    live = {"a.weight": torch.zeros(2, 2)}
    with pytest.raises(RuntimeError, match="a.weight"):
        assert_checkpoint_loaded(live, meta)


def test_guard_raises_on_a_missing_key(tmp_path):
    meta = {"stripped_keys": ["a.weight", "b.weight"], "tuned_fingerprints": {}}
    live = {"a.weight": torch.zeros(2, 2)}
    with pytest.raises(RuntimeError, match="b.weight"):
        assert_checkpoint_loaded(live, meta)


def test_guard_passes_when_fingerprints_match():
    import hashlib

    t = torch.randn(2, 2)
    fp = hashlib.sha256(t.numpy().tobytes()).hexdigest()
    meta = {"stripped_keys": ["a.weight"], "tuned_fingerprints": {"a.weight": fp}}
    assert_checkpoint_loaded({"a.weight": t}, meta) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_sam3_resolve_and_guard.py -v`
Expected: FAIL — `ImportError: cannot import name 'assert_checkpoint_loaded'`

- [ ] **Step 3: Write minimal implementation**

In `checkpoints.py`: split `probe_availability` into `probe_dependencies()`
(packages + ultralytics symbol, variant-independent) and
`probe_checkpoint(key)`; keep `probe_availability` as a thin composition so
existing callers keep working. Add `resolve_checkpoint(key)` returning a stock
variant's path or a registry-published artifact's path, and `available_models()`
returning stock variants plus registry entries with
`usage_role == "semantic_sam3"`.

In `sam3.py`: add `assert_checkpoint_loaded(live_state_dict, meta)` raising a
`RuntimeError` naming any `stripped_keys` entry absent from the live model, and
any `tuned_fingerprints` entry whose sha256 does not match. Add a `checkpoint:`
parameter to `from_variant`; when it is a published artifact, force eager
`setup_model()` and call the guard.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_sam3_resolve_and_guard.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/core/inference/semantic/ tests/test_sam3_resolve_and_guard.py
git commit -m "feat(semantic): resolve published SAM3 models + silent-load guard

Verifies TENSOR CONTENT, not just key names: if ultralytics' load-time
transform changes, the key namespace is unchanged and a key-only check passes
on exactly the failure it targets."
```

---

## Task 11: DetectKit training panel

**Files:**
- Create: `src/hydra_suite/detectkit/gui/panels/sam3_training_panel.py`
- Modify: `src/hydra_suite/detectkit/gui/dialogs/training_dialog.py:55` (`_SELECTION_ROLE_MAP`) and the role dispatch at `:1892-1945`
- Test: `tests/test_sam3_training_panel.py`

**Interfaces:**
- Consumes: `Sam3LoraParams` (Task 5), `probe_sam3_training_availability` (Task 4).
- Produces: `Sam3TrainingPanel(QWidget)` with `params() -> Sam3LoraParams`, `set_params(p)`, and `acknowledged() -> bool`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sam3_training_panel.py
"""The panel owns the knobs, the acknowledgement, and the disabled-with-reason."""

import pytest

pytest.importorskip("PyQt6")

from hydra_suite.detectkit.gui.panels.sam3_training_panel import Sam3TrainingPanel


def test_params_round_trip(qtbot):
    p = Sam3TrainingPanel()
    qtbot.addWidget(p)
    got = p.params()
    assert got.epochs == 10 and got.rank == 16


def test_training_is_blocked_until_labels_are_acknowledged(qtbot):
    p = Sam3TrainingPanel()
    qtbot.addWidget(p)
    # Provenance does not survive a review, so the user must confirm the
    # labels are good before SAM3 learns them.
    assert p.acknowledged() is False


def test_unavailable_backend_disables_with_a_reason(qtbot, monkeypatch):
    import hydra_suite.detectkit.gui.panels.sam3_training_panel as mod

    monkeypatch.setattr(
        mod, "probe_sam3_training_availability",
        lambda: mod.Sam3TrainingAvailability(False, "package 'sam3' is missing"))
    p = Sam3TrainingPanel()
    qtbot.addWidget(p)
    assert not p.isEnabled() or "sam3" in p.unavailable_reason()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_sam3_training_panel.py -v`
Expected: FAIL — `ModuleNotFoundError: ...panels.sam3_training_panel`

- [ ] **Step 3: Write minimal implementation**

Build `Sam3TrainingPanel` as a self-contained `QWidget`: prompt field, negative
prompts, LoRA group (rank/alpha/dropout), optimisation group
(lr/epochs/batch/accum/precision), the six `adapt_*` checkboxes, the tiling
group, a CUDA-host notice, and a mandatory "I have verified these labels"
checkbox. `training_dialog.py` gains only a `_SELECTION_ROLE_MAP` entry and a
dispatch branch that delegates to the panel — **no inline widgets**; that file
is already 2676 lines.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_sam3_training_panel.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/detectkit/gui/panels/sam3_training_panel.py src/hydra_suite/detectkit/gui/dialogs/training_dialog.py tests/test_sam3_training_panel.py
git commit -m "feat(detectkit): SAM3 training panel

A separate panel rather than inline widgets: training_dialog.py is already
2676 lines, 5x the project's 500-line guidance."
```

---

## Task 12: Acceptance gates on the CUDA box

Not a unit test — the manual gate. Run on mehek with real weights.

**Files:**
- Modify: `scratch/sam3_lora_spike/arm_sam3_native.py` (add the presence term)
- Modify: `scratch/sam3_lora_spike/arm_sam3.py` (accept `--checkpoint`)
- Create: `docs/superpowers/plans/notes/2026-08-31-sam3-acceptance-results.md`

**Interfaces:**
- Consumes: everything above.
- Produces: a results note recording each gate's numbers.

- [ ] **Step 1: Fix the harness scorer to match both stacks**

Meta's own postprocessor (`sam3/eval/postprocessors.py:101-104`) multiplies by
presence; the spike harness did not, which made it the outlier rather than
ultralytics. Add:

```python
scores = out["pred_logits"].sigmoid()[0].max(-1).values
presence = out.get("presence_logit_dec")
if presence is not None:
    scores = scores * presence.sigmoid().reshape(-1)[0]
scores = scores.float().cpu().numpy()
```

Note in the commit that `presence` is a per-image scalar, so within-tile
ranking — and therefore per-frame AP — is unchanged; what moves is cross-tile
merge ordering and absolute thresholds.

- [ ] **Step 2: Gate 1 — stack parity**

Run the merged checkpoint through **ultralytics** and through **native sam3**
on one held-out frame. Pass condition: AP50 and AP75 each within **0.05
absolute**.

- [ ] **Step 3: Gate 2 — beats the tuned baseline**

Through the ultralytics path, on **every** held-out frame, AP75 must exceed the
tuned YOLO-seg arm (0.255 mean, 0.220 on f008078).

- [ ] **Step 4: Gate 3 — scale round-trip**

Train at `train_tile_px` X, escalate at X: AP75 within the same 0.05 band.
Then run a deliberate scale mismatch as a **diagnostic only** — record what it
does; it is not a blocker.

- [ ] **Step 5: Record and commit**

```bash
git add docs/superpowers/plans/notes/2026-08-31-sam3-acceptance-results.md scratch/sam3_lora_spike/
git commit -m "test(sam3): acceptance gate results on CUDA"
```

---

## Task 13: Documentation

**Files:**
- Modify: `docs/user-guide/detectkit.md` (or the nearest existing DetectKit page)
- Modify: `pyproject.toml` (add the `sam3-train` extra)
- Modify: `CLAUDE.md` (one line under DetectKit noting the CUDA-only role)

- [ ] **Step 1: Add the extra**

```toml
sam3-train = ["torchmetrics", "scipy", "einops", "decord"]
```

`sam3` itself is a git dependency and cannot go in a published extra (the same
constraint `checkpoints.py` already documents for `clip`), so the user guide
names it explicitly.

- [ ] **Step 2: Write the user-guide section**

Cover: what the role does, the CUDA-only ~32 GB requirement, the label-quality
warning and why provenance cannot be filtered, that `epochs=10` is measured,
and that a published model is selectable in the semantic escalation dialog and
prefills prompt and tile fraction.

- [ ] **Step 3: Verify docs build**

Run: `make docs-check`
Expected: clean build

- [ ] **Step 4: Commit**

```bash
git add docs/ pyproject.toml CLAUDE.md
git commit -m "docs(detectkit): SAM3 finetuning role"
```
